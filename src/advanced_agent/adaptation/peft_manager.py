"""
HuggingFace PEFT LoRA アダプタプール管理システム
PEFT get_peft_model による複数アダプタ管理とメモリ効率化
"""

import asyncio
import time
import os
import json
import hashlib
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

try:
    import torch
    from peft import (
        PeftModel, PeftConfig, LoraConfig, AdaLoraConfig, IA3Config,
        get_peft_model, TaskType, PeftType
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import HfApi, Repository
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    PeftModel = None
    LoraConfig = None

from ..core.config import get_config
from ..core.logger import get_logger
from ..monitoring.system_monitor import SystemMonitor


class AdapterType(Enum):
    """アダプタタイプ"""
    LORA = "lora"
    ADALORA = "adalora"
    IA3 = "ia3"
    PROMPT_TUNING = "prompt_tuning"
    PREFIX_TUNING = "prefix_tuning"


class AdapterStatus(Enum):
    """アダプタ状態"""
    INACTIVE = "inactive"
    LOADING = "loading"
    ACTIVE = "active"
    TRAINING = "training"
    SAVING = "saving"
    ERROR = "error"


@dataclass
class AdapterConfig:
    """アダプタ設定"""
    name: str
    adapter_type: AdapterType
    task_type: str = "CAUSAL_LM"
    
    # LoRA設定
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: Optional[List[str]] = None
    
    # AdaLoRA設定
    target_r: int = 8
    init_r: int = 12
    tinit: int = 0
    tfinal: int = 1000
    deltaT: int = 10
    
    # IA3設定
    feedforward_modules: Optional[List[str]] = None
    
    # 共通設定
    bias: str = "none"
    fan_in_fan_out: bool = False
    enable_lora: Optional[List[str]] = None
    
    # メタデータ
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterInfo:
    """アダプタ情報"""
    config: AdapterConfig
    status: AdapterStatus
    model_path: Optional[str] = None
    hub_model_id: Optional[str] = None
    local_path: Optional[str] = None
    
    # パフォーマンス情報
    parameter_count: int = 0
    memory_usage_mb: float = 0.0
    load_time: float = 0.0
    
    # 品質情報
    performance_score: float = 0.0
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    # 使用統計
    usage_count: int = 0
    last_used: Optional[datetime] = None
    
    # エラー情報
    error_message: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterPoolStats:
    """アダプタプール統計"""
    total_adapters: int = 0
    active_adapters: int = 0
    inactive_adapters: int = 0
    error_adapters: int = 0
    
    total_parameters: int = 0
    total_memory_mb: float = 0.0
    
    average_performance_score: float = 0.0
    best_adapter: Optional[str] = None
    most_used_adapter: Optional[str] = None
    
    last_updated: datetime = field(default_factory=datetime.now)


class PEFTAdapterPool:
    """HuggingFace PEFT アダプタプール管理"""
    
    def __init__(
        self, 
        base_model_name: str,
        pool_directory: Optional[str] = None,
        system_monitor: Optional[SystemMonitor] = None
    ):
        self.base_model_name = base_model_name
        self.pool_directory = Path(pool_directory or "./adapter_pool")
        self.system_monitor = system_monitor
        
        self.config = get_config()
        self.logger = get_logger()
        
        # アダプタプール
        self.adapters: Dict[str, AdapterInfo] = {}
        self.active_adapters: Dict[str, PeftModel] = {}
        
        # ベースモデル
        self.base_model = None
        self.tokenizer = None
        
        # 設定
        self.max_active_adapters = 3
        self.memory_threshold_mb = 2048  # 2GB
        self.auto_cleanup = True
        
        # 統計
        self.pool_stats = AdapterPoolStats()
        
        # HuggingFace Hub設定
        self.hf_api = None
        if PEFT_AVAILABLE:
            try:
                self.hf_api = HfApi()
            except Exception:
                pass
        
        # ディレクトリ作成
        self.pool_directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.log_startup(
            component="peft_adapter_pool",
            version="1.0.0",
            config_summary={
                "base_model": base_model_name,
                "pool_directory": str(self.pool_directory),
                "peft_available": PEFT_AVAILABLE,
                "max_active_adapters": self.max_active_adapters
            }
        )
    
    async def initialize(self) -> bool:
        """アダプタプール初期化"""
        try:
            if not PEFT_AVAILABLE:
                self.logger.log_alert(
                    alert_type="peft_unavailable",
                    severity="WARNING",
                    message="PEFT not available, adapter functionality limited"
                )
                return False
            
            # ベースモデルロード
            self.logger.log_performance_metric(
                metric_name="base_model_loading_start",
                value=1,
                unit="model",
                component="peft_adapter_pool"
            )
            
            self.base_model = await asyncio.to_thread(
                AutoModelForCausalLM.from_pretrained,
                self.base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            self.tokenizer = await asyncio.to_thread(
                AutoTokenizer.from_pretrained,
                self.base_model_name
            )
            
            # 既存アダプタ読み込み
            await self._load_existing_adapters()
            
            # 統計更新
            self._update_pool_stats()
            
            self.logger.log_startup(
                component="peft_pool_initialized",
                version="1.0.0",
                config_summary={
                    "base_model_loaded": True,
                    "existing_adapters": len(self.adapters),
                    "pool_stats": self.pool_stats.__dict__
                }
            )
            
            return True
            
        except Exception as e:
            self.logger.log_alert(
                alert_type="peft_pool_initialization_failed",
                severity="ERROR",
                message=f"PEFT adapter pool initialization failed: {e}"
            )
            return False
    
    async def _load_existing_adapters(self) -> None:
        """既存アダプタ読み込み"""
        try:
            for adapter_dir in self.pool_directory.iterdir():
                if adapter_dir.is_dir():
                    config_file = adapter_dir / "adapter_config.json"
                    if config_file.exists():
                        await self._load_adapter_from_directory(adapter_dir)
        
        except Exception as e:
            self.logger.log_alert(
                alert_type="existing_adapters_load_failed",
                severity="WARNING",
                message=f"Failed to load existing adapters: {e}"
            )
    
    async def _load_adapter_from_directory(self, adapter_dir: Path) -> None:
        """ディレクトリからアダプタ読み込み"""
        try:
            config_file = adapter_dir / "adapter_info.json"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    adapter_data = json.load(f)
                
                # AdapterConfig復元
                config_data = adapter_data["config"]
                adapter_config = AdapterConfig(
                    name=config_data["name"],
                    adapter_type=AdapterType(config_data["adapter_type"]),
                    **{k: v for k, v in config_data.items() if k not in ["name", "adapter_type"]}
                )
                
                # AdapterInfo復元
                adapter_info = AdapterInfo(
                    config=adapter_config,
                    status=AdapterStatus.INACTIVE,
                    local_path=str(adapter_dir),
                    **{k: v for k, v in adapter_data.items() if k != "config"}
                )
                
                self.adapters[adapter_config.name] = adapter_info
                
        except Exception as e:
            self.logger.log_alert(
                alert_type="adapter_load_failed",
                severity="WARNING",
                message=f"Failed to load adapter from {adapter_dir}: {e}"
            )
    
    def create_adapter_config(
        self,
        name: str,
        adapter_type: AdapterType = AdapterType.LORA,
        **kwargs
    ) -> AdapterConfig:
        """アダプタ設定作成"""
        
        # デフォルト target_modules 設定
        if "target_modules" not in kwargs and adapter_type == AdapterType.LORA:
            # 一般的なLLMのLinear層をターゲット
            kwargs["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        config = AdapterConfig(
            name=name,
            adapter_type=adapter_type,
            **kwargs
        )
        
        return config
    
    def create_peft_config(self, adapter_config: AdapterConfig):
        """PEFT設定作成"""
        if not PEFT_AVAILABLE:
            raise RuntimeError("PEFT not available")
        
        task_type = getattr(TaskType, adapter_config.task_type, TaskType.CAUSAL_LM)
        
        if adapter_config.adapter_type == AdapterType.LORA:
            return LoraConfig(
                task_type=task_type,
                r=adapter_config.r,
                lora_alpha=adapter_config.lora_alpha,
                lora_dropout=adapter_config.lora_dropout,
                target_modules=adapter_config.target_modules,
                bias=adapter_config.bias,
                fan_in_fan_out=adapter_config.fan_in_fan_out
            )
        
        elif adapter_config.adapter_type == AdapterType.ADALORA:
            return AdaLoraConfig(
                task_type=task_type,
                r=adapter_config.r,
                lora_alpha=adapter_config.lora_alpha,
                lora_dropout=adapter_config.lora_dropout,
                target_modules=adapter_config.target_modules,
                target_r=adapter_config.target_r,
                init_r=adapter_config.init_r,
                tinit=adapter_config.tinit,
                tfinal=adapter_config.tfinal,
                deltaT=adapter_config.deltaT
            )
        
        elif adapter_config.adapter_type == AdapterType.IA3:
            return IA3Config(
                task_type=task_type,
                target_modules=adapter_config.target_modules,
                feedforward_modules=adapter_config.feedforward_modules
            )
        
        else:
            raise ValueError(f"Unsupported adapter type: {adapter_config.adapter_type}")
    
    async def create_adapter(
        self,
        adapter_config: AdapterConfig,
        auto_activate: bool = True
    ) -> AdapterInfo:
        """アダプタ作成"""
        
        if adapter_config.name in self.adapters:
            raise ValueError(f"Adapter '{adapter_config.name}' already exists")
        
        start_time = time.time()
        
        try:
            # PEFT設定作成
            peft_config = self.create_peft_config(adapter_config)
            
            # PEFTモデル作成
            peft_model = await asyncio.to_thread(
                get_peft_model,
                self.base_model,
                peft_config
            )
            
            # パラメータ数計算
            trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
            
            # アダプタ情報作成
            adapter_info = AdapterInfo(
                config=adapter_config,
                status=AdapterStatus.ACTIVE if auto_activate else AdapterStatus.INACTIVE,
                parameter_count=trainable_params,
                load_time=time.time() - start_time,
                memory_usage_mb=self._estimate_model_memory(peft_model)
            )
            
            # プールに追加
            self.adapters[adapter_config.name] = adapter_info
            
            if auto_activate:
                self.active_adapters[adapter_config.name] = peft_model
                
                # メモリ管理
                await self._manage_memory()
            
            # ローカル保存
            await self._save_adapter_locally(adapter_config.name, peft_model)
            
            # 統計更新
            self._update_pool_stats()
            
            self.logger.log_performance_metric(
                metric_name="adapter_created",
                value=trainable_params,
                unit="parameters",
                component="peft_adapter_pool"
            )
            
            return adapter_info
            
        except Exception as e:
            self.logger.log_alert(
                alert_type="adapter_creation_failed",
                severity="ERROR",
                message=f"Failed to create adapter '{adapter_config.name}': {e}"
            )
            raise
    
    async def load_adapter(
        self,
        adapter_name: str,
        source: Optional[str] = None
    ) -> AdapterInfo:
        """アダプタ読み込み"""
        
        if adapter_name in self.active_adapters:
            # 既にアクティブ
            adapter_info = self.adapters[adapter_name]
            adapter_info.usage_count += 1
            adapter_info.last_used = datetime.now()
            return adapter_info
        
        start_time = time.time()
        
        try:
            adapter_info = self.adapters.get(adapter_name)
            
            if adapter_info is None:
                # 新しいアダプタ
                if source is None:
                    raise ValueError(f"Adapter '{adapter_name}' not found and no source provided")
                
                # HuggingFace Hubまたはローカルパスから読み込み
                adapter_info = await self._load_adapter_from_source(adapter_name, source)
            
            # アダプタ状態更新
            adapter_info.status = AdapterStatus.LOADING
            
            # PEFTモデル読み込み
            if adapter_info.local_path:
                peft_model = await asyncio.to_thread(
                    PeftModel.from_pretrained,
                    self.base_model,
                    adapter_info.local_path
                )
            elif adapter_info.hub_model_id:
                peft_model = await asyncio.to_thread(
                    PeftModel.from_pretrained,
                    self.base_model,
                    adapter_info.hub_model_id
                )
            else:
                raise ValueError(f"No valid source for adapter '{adapter_name}'")
            
            # アクティブプールに追加
            self.active_adapters[adapter_name] = peft_model
            
            # アダプタ情報更新
            adapter_info.status = AdapterStatus.ACTIVE
            adapter_info.load_time = time.time() - start_time
            adapter_info.usage_count += 1
            adapter_info.last_used = datetime.now()
            adapter_info.memory_usage_mb = self._estimate_model_memory(peft_model)
            
            # メモリ管理
            await self._manage_memory()
            
            # 統計更新
            self._update_pool_stats()
            
            self.logger.log_performance_metric(
                metric_name="adapter_loaded",
                value=adapter_info.load_time,
                unit="seconds",
                component="peft_adapter_pool"
            )
            
            return adapter_info
            
        except Exception as e:
            if adapter_name in self.adapters:
                self.adapters[adapter_name].status = AdapterStatus.ERROR
                self.adapters[adapter_name].error_message = str(e)
            
            self.logger.log_alert(
                alert_type="adapter_load_failed",
                severity="ERROR",
                message=f"Failed to load adapter '{adapter_name}': {e}"
            )
            raise
    
    async def _load_adapter_from_source(
        self,
        adapter_name: str,
        source: str
    ) -> AdapterInfo:
        """ソースからアダプタ読み込み"""
        
        # HuggingFace Hub ID判定
        if "/" in source and not os.path.exists(source):
            # HuggingFace Hub
            hub_model_id = source
            local_path = None
            
            # 設定取得
            peft_config = await asyncio.to_thread(
                PeftConfig.from_pretrained,
                hub_model_id
            )
            
        else:
            # ローカルパス
            local_path = source
            hub_model_id = None
            
            # 設定取得
            peft_config = await asyncio.to_thread(
                PeftConfig.from_pretrained,
                local_path
            )
        
        # AdapterConfig作成
        adapter_config = AdapterConfig(
            name=adapter_name,
            adapter_type=AdapterType.LORA,  # TODO: 設定から判定
            r=getattr(peft_config, 'r', 16),
            lora_alpha=getattr(peft_config, 'lora_alpha', 32),
            lora_dropout=getattr(peft_config, 'lora_dropout', 0.1),
            target_modules=getattr(peft_config, 'target_modules', None)
        )
        
        # AdapterInfo作成
        adapter_info = AdapterInfo(
            config=adapter_config,
            status=AdapterStatus.INACTIVE,
            hub_model_id=hub_model_id,
            local_path=local_path
        )
        
        # プールに追加
        self.adapters[adapter_name] = adapter_info
        
        return adapter_info
    
    async def unload_adapter(self, adapter_name: str) -> None:
        """アダプタアンロード"""
        
        if adapter_name not in self.active_adapters:
            return
        
        try:
            # アクティブプールから削除
            del self.active_adapters[adapter_name]
            
            # 状態更新
            if adapter_name in self.adapters:
                self.adapters[adapter_name].status = AdapterStatus.INACTIVE
            
            # メモリクリーンアップ
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 統計更新
            self._update_pool_stats()
            
            self.logger.log_performance_metric(
                metric_name="adapter_unloaded",
                value=1,
                unit="adapter",
                component="peft_adapter_pool"
            )
            
        except Exception as e:
            self.logger.log_alert(
                alert_type="adapter_unload_failed",
                severity="WARNING",
                message=f"Failed to unload adapter '{adapter_name}': {e}"
            )
    
    async def swap_adapter(
        self,
        current_adapter: str,
        new_adapter: str
    ) -> AdapterInfo:
        """アダプタスワップ"""
        
        # 現在のアダプタをアンロード
        await self.unload_adapter(current_adapter)
        
        # 新しいアダプタをロード
        return await self.load_adapter(new_adapter)
    
    async def _manage_memory(self) -> None:
        """メモリ管理"""
        
        if not self.auto_cleanup:
            return
        
        # 現在のメモリ使用量計算
        total_memory = sum(
            info.memory_usage_mb 
            for name, info in self.adapters.items() 
            if name in self.active_adapters
        )
        
        # メモリ制限チェック
        if total_memory > self.memory_threshold_mb or len(self.active_adapters) > self.max_active_adapters:
            # 使用頻度の低いアダプタをアンロード
            candidates = [
                (name, info) 
                for name, info in self.adapters.items() 
                if name in self.active_adapters
            ]
            
            # 最後に使用された時間でソート
            candidates.sort(key=lambda x: x[1].last_used or datetime.min)
            
            # 古いものから削除
            while (total_memory > self.memory_threshold_mb or 
                   len(self.active_adapters) > self.max_active_adapters) and candidates:
                
                adapter_name, adapter_info = candidates.pop(0)
                await self.unload_adapter(adapter_name)
                total_memory -= adapter_info.memory_usage_mb
    
    async def _save_adapter_locally(
        self,
        adapter_name: str,
        peft_model: PeftModel
    ) -> None:
        """アダプタローカル保存"""
        
        try:
            adapter_dir = self.pool_directory / adapter_name
            adapter_dir.mkdir(exist_ok=True)
            
            # PEFTモデル保存
            await asyncio.to_thread(
                peft_model.save_pretrained,
                str(adapter_dir)
            )
            
            # アダプタ情報保存
            adapter_info = self.adapters[adapter_name]
            adapter_info.local_path = str(adapter_dir)
            
            info_file = adapter_dir / "adapter_info.json"
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "config": {
                        "name": adapter_info.config.name,
                        "adapter_type": adapter_info.config.adapter_type.value,
                        "r": adapter_info.config.r,
                        "lora_alpha": adapter_info.config.lora_alpha,
                        "lora_dropout": adapter_info.config.lora_dropout,
                        "target_modules": adapter_info.config.target_modules,
                        "description": adapter_info.config.description,
                        "tags": adapter_info.config.tags,
                        "created_at": adapter_info.config.created_at.isoformat()
                    },
                    "parameter_count": adapter_info.parameter_count,
                    "performance_score": adapter_info.performance_score,
                    "usage_count": adapter_info.usage_count,
                    "last_used": adapter_info.last_used.isoformat() if adapter_info.last_used else None
                }, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            self.logger.log_alert(
                alert_type="adapter_save_failed",
                severity="WARNING",
                message=f"Failed to save adapter '{adapter_name}' locally: {e}"
            )
    
    def _estimate_model_memory(self, model) -> float:
        """モデルメモリ使用量推定（MB）"""
        try:
            param_count = sum(p.numel() for p in model.parameters())
            # パラメータあたり2バイト（float16）と仮定
            return param_count * 2 / (1024**2)
        except Exception:
            return 0.0
    
    def _update_pool_stats(self) -> None:
        """プール統計更新"""
        
        self.pool_stats.total_adapters = len(self.adapters)
        self.pool_stats.active_adapters = len(self.active_adapters)
        self.pool_stats.inactive_adapters = len([
            info for info in self.adapters.values() 
            if info.status == AdapterStatus.INACTIVE
        ])
        self.pool_stats.error_adapters = len([
            info for info in self.adapters.values() 
            if info.status == AdapterStatus.ERROR
        ])
        
        self.pool_stats.total_parameters = sum(
            info.parameter_count for info in self.adapters.values()
        )
        self.pool_stats.total_memory_mb = sum(
            info.memory_usage_mb 
            for name, info in self.adapters.items() 
            if name in self.active_adapters
        )
        
        # パフォーマンススコア平均
        scores = [info.performance_score for info in self.adapters.values() if info.performance_score > 0]
        if scores:
            self.pool_stats.average_performance_score = sum(scores) / len(scores)
            
            # 最高性能アダプタ
            best_adapter = max(
                self.adapters.items(),
                key=lambda x: x[1].performance_score
            )
            self.pool_stats.best_adapter = best_adapter[0]
        
        # 最多使用アダプタ
        if self.adapters:
            most_used = max(
                self.adapters.items(),
                key=lambda x: x[1].usage_count
            )
            self.pool_stats.most_used_adapter = most_used[0]
        
        self.pool_stats.last_updated = datetime.now()
    
    def get_adapter_info(self, adapter_name: str) -> Optional[AdapterInfo]:
        """アダプタ情報取得"""
        return self.adapters.get(adapter_name)
    
    def list_adapters(
        self,
        status_filter: Optional[AdapterStatus] = None,
        adapter_type_filter: Optional[AdapterType] = None
    ) -> List[AdapterInfo]:
        """アダプタ一覧取得"""
        
        adapters = list(self.adapters.values())
        
        if status_filter:
            adapters = [info for info in adapters if info.status == status_filter]
        
        if adapter_type_filter:
            adapters = [info for info in adapters if info.config.adapter_type == adapter_type_filter]
        
        return adapters
    
    def get_pool_stats(self) -> AdapterPoolStats:
        """プール統計取得"""
        self._update_pool_stats()
        return self.pool_stats
    
    async def cleanup_inactive_adapters(self) -> int:
        """非アクティブアダプタクリーンアップ"""
        
        cleaned_count = 0
        
        for adapter_name, adapter_info in list(self.adapters.items()):
            if (adapter_info.status == AdapterStatus.INACTIVE and 
                adapter_info.usage_count == 0):
                
                try:
                    # ローカルファイル削除
                    if adapter_info.local_path and os.path.exists(adapter_info.local_path):
                        import shutil
                        shutil.rmtree(adapter_info.local_path)
                    
                    # プールから削除
                    del self.adapters[adapter_name]
                    cleaned_count += 1
                    
                except Exception as e:
                    self.logger.log_alert(
                        alert_type="adapter_cleanup_failed",
                        severity="WARNING",
                        message=f"Failed to cleanup adapter '{adapter_name}': {e}"
                    )
        
        # 統計更新
        self._update_pool_stats()
        
        return cleaned_count
    
    async def shutdown(self) -> None:
        """アダプタプール終了"""
        
        # 全アクティブアダプタをアンロード
        for adapter_name in list(self.active_adapters.keys()):
            await self.unload_adapter(adapter_name)
        
        # メモリクリーンアップ
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        final_stats = self.get_pool_stats()
        
        self.logger.log_shutdown(
            component="peft_adapter_pool",
            uptime_seconds=0,  # TODO: 実際の稼働時間計算
            final_stats=final_stats.__dict__
        )


# 便利関数
async def create_peft_adapter_pool(
    base_model_name: str,
    pool_directory: Optional[str] = None,
    system_monitor: Optional[SystemMonitor] = None
) -> PEFTAdapterPool:
    """PEFT アダプタプール作成・初期化"""
    
    pool = PEFTAdapterPool(base_model_name, pool_directory, system_monitor)
    
    if await pool.initialize():
        return pool
    else:
        raise RuntimeError("Failed to initialize PEFT adapter pool")


# 使用例
async def main():
    """テスト用メイン関数"""
    try:
        # アダプタプール作成
        pool = await create_peft_adapter_pool("microsoft/DialoGPT-medium")
        
        print("=== PEFT Adapter Pool Test ===")
        
        # アダプタ設定作成
        adapter_config = pool.create_adapter_config(
            name="test_lora",
            adapter_type=AdapterType.LORA,
            r=16,
            lora_alpha=32,
            description="Test LoRA adapter"
        )
        
        print(f"Created adapter config: {adapter_config.name}")
        
        # アダプタ作成
        adapter_info = await pool.create_adapter(adapter_config)
        print(f"Created adapter: {adapter_info.config.name}")
        print(f"Parameters: {adapter_info.parameter_count}")
        print(f"Memory: {adapter_info.memory_usage_mb:.2f} MB")
        
        # プール統計
        stats = pool.get_pool_stats()
        print(f"Pool stats: {stats}")
        
        # アダプタ一覧
        adapters = pool.list_adapters()
        print(f"Total adapters: {len(adapters)}")
        
        await pool.shutdown()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())