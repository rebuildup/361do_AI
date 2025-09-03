"""
HuggingFace BitsAndBytes 動的量子化システム
BitsAndBytesConfig による自動量子化設定と動的量子化管理
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import os
import psutil
import torch

try:
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
    from accelerate import Accelerator, init_empty_weights
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    BitsAndBytesConfig = None

from ..core.config import get_config
from ..core.logger import get_logger
from ..monitoring.system_monitor import SystemMonitor


class QuantizationLevel(Enum):
    """量子化レベル"""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    NF4 = "nf4"
    FP4 = "fp4"


class QuantizationStrategy(Enum):
    """量子化戦略"""
    STATIC = "static"
    DYNAMIC = "dynamic"
    ADAPTIVE = "adaptive"
    MEMORY_AWARE = "memory_aware"


@dataclass
class QuantizationConfig:
    """量子化設定"""
    level: QuantizationLevel
    strategy: QuantizationStrategy
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_use_double_quant: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    llm_int8_threshold: float = 6.0
    llm_int8_skip_modules: Optional[List[str]] = None
    llm_int8_enable_fp32_cpu_offload: bool = False
    memory_threshold_mb: int = 4096  # 4GB
    auto_adjust: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantizationResult:
    """量子化結果"""
    config_used: QuantizationConfig
    original_memory_mb: float
    quantized_memory_mb: float
    memory_reduction_ratio: float
    quantization_time: float
    model_size_mb: float
    performance_impact: float = 0.0
    quality_score: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BitsAndBytesManager:
    """HuggingFace BitsAndBytes 動的量子化マネージャー"""
    
    def __init__(self, system_monitor: Optional[SystemMonitor] = None):
        self.config = get_config()
        self.logger = get_logger()
        self.system_monitor = system_monitor
        
        # 量子化履歴
        self.quantization_history: List[QuantizationResult] = []
        
        # 現在の量子化設定
        self.current_config: Optional[QuantizationConfig] = None
        
        # パフォーマンス統計
        self.performance_stats = {
            "total_quantizations": 0,
            "successful_quantizations": 0,
            "failed_quantizations": 0,
            "average_memory_reduction": 0.0,
            "average_quantization_time": 0.0
        }
        
        # デフォルト設定
        self.default_configs = self._create_default_configs()
        
        self.logger.log_startup(
            component="bitsandbytes_manager",
            version="1.0.0",
            config_summary={
                "bitsandbytes_available": BITSANDBYTES_AVAILABLE,
                "cuda_available": torch.cuda.is_available(),
                "default_configs": len(self.default_configs)
            }
        )
    
    def _create_default_configs(self) -> Dict[str, QuantizationConfig]:
        """デフォルト量子化設定作成"""
        configs = {}
        
        # INT8量子化設定
        configs["int8_basic"] = QuantizationConfig(
            level=QuantizationLevel.INT8,
            strategy=QuantizationStrategy.STATIC,
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            memory_threshold_mb=6144  # 6GB
        )
        
        configs["int8_cpu_offload"] = QuantizationConfig(
            level=QuantizationLevel.INT8,
            strategy=QuantizationStrategy.MEMORY_AWARE,
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_threshold=6.0,
            memory_threshold_mb=4096  # 4GB
        )
        
        # INT4量子化設定
        configs["int4_nf4"] = QuantizationConfig(
            level=QuantizationLevel.NF4,
            strategy=QuantizationStrategy.STATIC,
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            memory_threshold_mb=3072  # 3GB
        )
        
        configs["int4_fp4"] = QuantizationConfig(
            level=QuantizationLevel.FP4,
            strategy=QuantizationStrategy.STATIC,
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_compute_dtype="float16",
            memory_threshold_mb=2048  # 2GB
        )
        
        # 動的量子化設定
        configs["dynamic_adaptive"] = QuantizationConfig(
            level=QuantizationLevel.INT8,
            strategy=QuantizationStrategy.ADAPTIVE,
            load_in_8bit=True,
            auto_adjust=True,
            memory_threshold_mb=4096
        )
        
        # メモリ制約対応設定
        configs["memory_constrained"] = QuantizationConfig(
            level=QuantizationLevel.NF4,
            strategy=QuantizationStrategy.MEMORY_AWARE,
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            llm_int8_enable_fp32_cpu_offload=True,
            memory_threshold_mb=2048,
            auto_adjust=True
        )
        
        return configs
    
    def check_availability(self) -> Dict[str, Any]:
        """BitsAndBytes利用可能性チェック"""
        availability = {
            "bitsandbytes_installed": BITSANDBYTES_AVAILABLE,
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "system_memory_gb": psutil.virtual_memory().total / (1024**3),
            "gpu_memory_gb": 0.0,
            "recommendations": []
        }
        
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                availability["gpu_memory_gb"] = gpu_memory
                
                if gpu_memory < 4:
                    availability["recommendations"].append("GPU memory < 4GB: Use INT4 quantization")
                elif gpu_memory < 8:
                    availability["recommendations"].append("GPU memory < 8GB: Use INT8 quantization")
                else:
                    availability["recommendations"].append("GPU memory >= 8GB: Full precision or light quantization")
                    
            except Exception as e:
                availability["gpu_error"] = str(e)
        
        if not BITSANDBYTES_AVAILABLE:
            availability["recommendations"].append("Install bitsandbytes: pip install bitsandbytes")
        
        return availability
    
    def get_optimal_config(
        self, 
        model_size_mb: Optional[float] = None,
        available_memory_mb: Optional[float] = None,
        target_memory_mb: Optional[float] = None
    ) -> QuantizationConfig:
        """最適量子化設定取得"""
        
        # システム情報取得
        if available_memory_mb is None:
            if torch.cuda.is_available():
                available_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            else:
                available_memory_mb = psutil.virtual_memory().available / (1024**2)
        
        if target_memory_mb is None:
            target_memory_mb = available_memory_mb * 0.8  # 80%を目標
        
        # モデルサイズ推定
        if model_size_mb is None:
            model_size_mb = 7000  # 7Bモデルの推定サイズ
        
        # 設定選択ロジック
        if target_memory_mb < 2048:  # 2GB未満
            config_name = "memory_constrained"
        elif target_memory_mb < 3072:  # 3GB未満
            config_name = "int4_nf4"
        elif target_memory_mb < 4096:  # 4GB未満
            config_name = "int8_cpu_offload"
        elif target_memory_mb < 6144:  # 6GB未満
            config_name = "int8_basic"
        else:
            config_name = "dynamic_adaptive"
        
        config = self.default_configs[config_name].copy() if config_name in self.default_configs else self.default_configs["int8_basic"]
        
        # 動的調整
        if config.auto_adjust:
            config.memory_threshold_mb = int(target_memory_mb)
            
            # メモリ制約に応じた調整
            if target_memory_mb < model_size_mb * 0.3:  # モデルサイズの30%未満
                config.level = QuantizationLevel.NF4
                config.load_in_4bit = True
                config.load_in_8bit = False
                config.bnb_4bit_use_double_quant = True
            elif target_memory_mb < model_size_mb * 0.5:  # モデルサイズの50%未満
                config.level = QuantizationLevel.INT8
                config.load_in_8bit = True
                config.load_in_4bit = False
        
        self.logger.log_performance_metric(
            metric_name="optimal_config_selected",
            value=target_memory_mb,
            unit="mb",
            component="bitsandbytes_manager"
        )
        
        return config
    
    def create_bnb_config(self, quant_config: QuantizationConfig) -> Optional[BitsAndBytesConfig]:
        """BitsAndBytesConfig作成"""
        if not BITSANDBYTES_AVAILABLE:
            self.logger.log_alert(
                alert_type="bitsandbytes_unavailable",
                severity="WARNING",
                message="BitsAndBytes not available, skipping quantization"
            )
            return None
        
        try:
            config_kwargs = {}
            
            # 基本設定
            if quant_config.load_in_8bit:
                config_kwargs["load_in_8bit"] = True
                config_kwargs["llm_int8_threshold"] = quant_config.llm_int8_threshold
                
                if quant_config.llm_int8_skip_modules:
                    config_kwargs["llm_int8_skip_modules"] = quant_config.llm_int8_skip_modules
                
                if quant_config.llm_int8_enable_fp32_cpu_offload:
                    config_kwargs["llm_int8_enable_fp32_cpu_offload"] = True
            
            if quant_config.load_in_4bit:
                config_kwargs["load_in_4bit"] = True
                config_kwargs["bnb_4bit_use_double_quant"] = quant_config.bnb_4bit_use_double_quant
                config_kwargs["bnb_4bit_quant_type"] = quant_config.bnb_4bit_quant_type
                
                # データ型設定
                if quant_config.bnb_4bit_compute_dtype == "float16":
                    config_kwargs["bnb_4bit_compute_dtype"] = torch.float16
                elif quant_config.bnb_4bit_compute_dtype == "bfloat16":
                    config_kwargs["bnb_4bit_compute_dtype"] = torch.bfloat16
                else:
                    config_kwargs["bnb_4bit_compute_dtype"] = torch.float16
            
            bnb_config = BitsAndBytesConfig(**config_kwargs)
            
            self.logger.log_performance_metric(
                metric_name="bnb_config_created",
                value=1,
                unit="config",
                component="bitsandbytes_manager"
            )
            
            return bnb_config
            
        except Exception as e:
            self.logger.log_alert(
                alert_type="bnb_config_creation_failed",
                severity="ERROR",
                message=f"Failed to create BitsAndBytesConfig: {e}"
            )
            return None
    
    async def quantize_model(
        self,
        model_name_or_path: str,
        quant_config: Optional[QuantizationConfig] = None,
        **model_kwargs
    ) -> Tuple[Any, QuantizationResult]:
        """モデル量子化実行"""
        
        start_time = time.time()
        
        # 量子化前メモリ測定
        initial_memory = self._get_memory_usage()
        
        # 設定準備
        if quant_config is None:
            quant_config = self.get_optimal_config()
        
        self.current_config = quant_config
        
        try:
            # BitsAndBytesConfig作成
            bnb_config = self.create_bnb_config(quant_config)
            
            if bnb_config is None:
                # 量子化なしでモデルロード
                model = await asyncio.to_thread(
                    AutoModelForCausalLM.from_pretrained,
                    model_name_or_path,
                    **model_kwargs
                )
                
                result = QuantizationResult(
                    config_used=quant_config,
                    original_memory_mb=initial_memory,
                    quantized_memory_mb=initial_memory,
                    memory_reduction_ratio=0.0,
                    quantization_time=time.time() - start_time,
                    model_size_mb=self._estimate_model_size(model),
                    success=True,
                    error_message="Quantization skipped (BitsAndBytes unavailable)"
                )
                
            else:
                # 量子化モデルロード
                model = await asyncio.to_thread(
                    AutoModelForCausalLM.from_pretrained,
                    model_name_or_path,
                    quantization_config=bnb_config,
                    **model_kwargs
                )
                
                # 量子化後メモリ測定
                quantized_memory = self._get_memory_usage()
                
                # 結果作成
                result = QuantizationResult(
                    config_used=quant_config,
                    original_memory_mb=initial_memory,
                    quantized_memory_mb=quantized_memory,
                    memory_reduction_ratio=(initial_memory - quantized_memory) / initial_memory if initial_memory > 0 else 0.0,
                    quantization_time=time.time() - start_time,
                    model_size_mb=self._estimate_model_size(model),
                    success=True
                )
            
            # 統計更新
            self.performance_stats["total_quantizations"] += 1
            self.performance_stats["successful_quantizations"] += 1
            self._update_performance_stats(result)
            
            # 履歴に追加
            self.quantization_history.append(result)
            
            self.logger.log_performance_metric(
                metric_name="quantization_success",
                value=result.quantization_time,
                unit="seconds",
                component="bitsandbytes_manager"
            )
            
            return model, result
            
        except Exception as e:
            # エラー結果作成
            error_result = QuantizationResult(
                config_used=quant_config,
                original_memory_mb=initial_memory,
                quantized_memory_mb=initial_memory,
                memory_reduction_ratio=0.0,
                quantization_time=time.time() - start_time,
                model_size_mb=0.0,
                success=False,
                error_message=str(e)
            )
            
            # 統計更新
            self.performance_stats["total_quantizations"] += 1
            self.performance_stats["failed_quantizations"] += 1
            
            # 履歴に追加
            self.quantization_history.append(error_result)
            
            self.logger.log_alert(
                alert_type="quantization_failed",
                severity="ERROR",
                message=f"Model quantization failed: {e}"
            )
            
            raise RuntimeError(f"Model quantization failed: {e}")
    
    def _get_memory_usage(self) -> float:
        """メモリ使用量取得（MB）"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**2)
        else:
            process = psutil.Process()
            return process.memory_info().rss / (1024**2)
    
    def _estimate_model_size(self, model) -> float:
        """モデルサイズ推定（MB）"""
        try:
            param_count = sum(p.numel() for p in model.parameters())
            # パラメータあたり4バイト（float32）と仮定
            return param_count * 4 / (1024**2)
        except Exception:
            return 0.0
    
    def _update_performance_stats(self, result: QuantizationResult) -> None:
        """パフォーマンス統計更新"""
        if result.success:
            # 平均メモリ削減率更新
            successful_count = self.performance_stats["successful_quantizations"]
            current_avg_reduction = self.performance_stats["average_memory_reduction"]
            
            new_avg_reduction = ((current_avg_reduction * (successful_count - 1)) + result.memory_reduction_ratio) / successful_count
            self.performance_stats["average_memory_reduction"] = new_avg_reduction
            
            # 平均量子化時間更新
            current_avg_time = self.performance_stats["average_quantization_time"]
            new_avg_time = ((current_avg_time * (successful_count - 1)) + result.quantization_time) / successful_count
            self.performance_stats["average_quantization_time"] = new_avg_time
    
    async def benchmark_quantization(
        self,
        model_name_or_path: str,
        configs_to_test: Optional[List[str]] = None
    ) -> Dict[str, QuantizationResult]:
        """量子化ベンチマーク実行"""
        
        if configs_to_test is None:
            configs_to_test = ["int8_basic", "int4_nf4", "memory_constrained"]
        
        results = {}
        
        for config_name in configs_to_test:
            if config_name not in self.default_configs:
                continue
            
            try:
                self.logger.log_performance_metric(
                    metric_name="benchmark_config_start",
                    value=len(config_name),
                    unit="chars",
                    component="bitsandbytes_manager"
                )
                
                config = self.default_configs[config_name]
                model, result = await self.quantize_model(model_name_or_path, config)
                
                results[config_name] = result
                
                # メモリクリーンアップ
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                self.logger.log_alert(
                    alert_type="benchmark_config_failed",
                    severity="WARNING",
                    message=f"Benchmark failed for config {config_name}: {e}"
                )
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計取得"""
        stats = self.performance_stats.copy()
        
        # 成功率計算
        if stats["total_quantizations"] > 0:
            stats["success_rate"] = stats["successful_quantizations"] / stats["total_quantizations"]
        else:
            stats["success_rate"] = 0.0
        
        # 最近の結果統計
        recent_results = self.quantization_history[-5:]  # 最新5件
        if recent_results:
            successful_recent = [r for r in recent_results if r.success]
            if successful_recent:
                stats["recent_average_reduction"] = sum(r.memory_reduction_ratio for r in successful_recent) / len(successful_recent)
                stats["recent_average_time"] = sum(r.quantization_time for r in successful_recent) / len(successful_recent)
        
        return stats
    
    def get_quantization_history(
        self,
        limit: Optional[int] = None,
        success_only: bool = False
    ) -> List[QuantizationResult]:
        """量子化履歴取得"""
        history = self.quantization_history
        
        if success_only:
            history = [r for r in history if r.success]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def get_config_recommendations(
        self,
        target_memory_mb: Optional[float] = None,
        model_size_mb: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """設定推奨事項取得"""
        recommendations = []
        
        # システム情報取得
        availability = self.check_availability()
        
        if not availability["bitsandbytes_installed"]:
            recommendations.append({
                "priority": "high",
                "type": "installation",
                "message": "Install bitsandbytes for quantization support",
                "command": "pip install bitsandbytes"
            })
        
        if target_memory_mb is None:
            target_memory_mb = availability["gpu_memory_gb"] * 1024 * 0.8  # 80%
        
        # メモリベース推奨
        if target_memory_mb < 2048:
            recommendations.append({
                "priority": "high",
                "type": "config",
                "message": "Use aggressive 4-bit quantization for memory constraints",
                "config": "memory_constrained"
            })
        elif target_memory_mb < 4096:
            recommendations.append({
                "priority": "medium",
                "type": "config",
                "message": "Use 4-bit NF4 quantization for balanced performance",
                "config": "int4_nf4"
            })
        else:
            recommendations.append({
                "priority": "low",
                "type": "config",
                "message": "Use 8-bit quantization for good quality",
                "config": "int8_basic"
            })
        
        # パフォーマンス履歴ベース推奨
        if self.quantization_history:
            successful_results = [r for r in self.quantization_history if r.success]
            if successful_results:
                best_result = max(successful_results, key=lambda r: r.memory_reduction_ratio)
                recommendations.append({
                    "priority": "info",
                    "type": "history",
                    "message": f"Best previous result: {best_result.memory_reduction_ratio:.1%} memory reduction",
                    "config": best_result.config_used.level.value
                })
        
        return recommendations
    
    async def cleanup(self) -> None:
        """リソースクリーンアップ"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.log_alert(
            alert_type="quantization_cleanup",
            severity="INFO",
            message="Quantization resources cleaned up"
        )
    
    async def shutdown(self) -> None:
        """マネージャー終了"""
        await self.cleanup()
        
        final_stats = self.get_performance_stats()
        
        self.logger.log_shutdown(
            component="bitsandbytes_manager",
            uptime_seconds=0,  # TODO: 実際の稼働時間計算
            final_stats=final_stats
        )


# 便利関数
async def create_bitsandbytes_manager(system_monitor: Optional[SystemMonitor] = None) -> BitsAndBytesManager:
    """BitsAndBytes マネージャー作成"""
    return BitsAndBytesManager(system_monitor)


def get_quantization_recommendations(
    available_memory_gb: float,
    model_size_gb: Optional[float] = None
) -> Dict[str, Any]:
    """量子化推奨事項取得（スタンドアロン関数）"""
    recommendations = {
        "recommended_config": "int8_basic",
        "reasoning": "",
        "alternatives": [],
        "warnings": []
    }
    
    target_memory_mb = available_memory_gb * 1024 * 0.8  # 80%使用を目標
    
    if available_memory_gb < 2:
        recommendations["recommended_config"] = "memory_constrained"
        recommendations["reasoning"] = "Very limited memory requires aggressive 4-bit quantization"
        recommendations["warnings"].append("Performance may be significantly impacted")
    elif available_memory_gb < 4:
        recommendations["recommended_config"] = "int4_nf4"
        recommendations["reasoning"] = "Limited memory benefits from 4-bit quantization"
        recommendations["alternatives"] = ["memory_constrained"]
    elif available_memory_gb < 6:
        recommendations["recommended_config"] = "int8_basic"
        recommendations["reasoning"] = "Moderate memory allows 8-bit quantization with good quality"
        recommendations["alternatives"] = ["int4_nf4", "int8_cpu_offload"]
    else:
        recommendations["recommended_config"] = "dynamic_adaptive"
        recommendations["reasoning"] = "Sufficient memory allows adaptive quantization"
        recommendations["alternatives"] = ["int8_basic", "int4_nf4"]
    
    return recommendations


# 使用例
async def main():
    """テスト用メイン関数"""
    try:
        # BitsAndBytes マネージャー作成
        manager = await create_bitsandbytes_manager()
        
        print("=== BitsAndBytes Manager Test ===")
        
        # 利用可能性チェック
        availability = manager.check_availability()
        print(f"Availability: {availability}")
        
        # 推奨設定取得
        recommendations = manager.get_config_recommendations(target_memory_mb=4096)
        print(f"Recommendations: {recommendations}")
        
        # 最適設定取得
        optimal_config = manager.get_optimal_config(
            model_size_mb=7000,
            available_memory_mb=6144,
            target_memory_mb=4096
        )
        print(f"Optimal Config: {optimal_config}")
        
        # BitsAndBytesConfig作成テスト
        bnb_config = manager.create_bnb_config(optimal_config)
        print(f"BnB Config Created: {bnb_config is not None}")
        
        # パフォーマンス統計
        stats = manager.get_performance_stats()
        print(f"Performance Stats: {stats}")
        
        await manager.shutdown()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())