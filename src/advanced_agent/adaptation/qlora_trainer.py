"""
PEFT + BitsAndBytes QLoRA システム統合
BitsAndBytesConfig + LoraConfig による QLoRA 統合と4GB制限学習パイプライン
"""

import asyncio
import time
import os
import json
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl
    )
    from datasets import Dataset as HFDataset
    import bitsandbytes as bnb
    QLORA_AVAILABLE = True
except ImportError:
    QLORA_AVAILABLE = False
    Trainer = None
    TrainingArguments = None
    # 型参照エラー回避用のダミー型
    class HFDataset:  # type: ignore
        @staticmethod
        def from_dict(data):
            return data

from ..core.config import get_config
from ..core.logger import get_logger
from ..monitoring.system_monitor import SystemMonitor
from ..quantization.bitsandbytes_manager import BitsAndBytesManager, QuantizationConfig, QuantizationLevel
from .peft_manager import PEFTAdapterPool, AdapterConfig, AdapterType


class TrainingPhase(Enum):
    """学習フェーズ"""
    INITIALIZATION = "initialization"
    PREPARATION = "preparation"
    TRAINING = "training"
    EVALUATION = "evaluation"
    SAVING = "saving"
    COMPLETED = "completed"
    ERROR = "error"


class QLoRAOptimization(Enum):
    """QLoRA最適化レベル"""
    MEMORY_OPTIMIZED = "memory_optimized"  # 最大メモリ効率
    BALANCED = "balanced"                  # バランス型
    PERFORMANCE_OPTIMIZED = "performance_optimized"  # 性能重視


@dataclass
class QLoRAConfig:
    """QLoRA設定"""
    # LoRA設定
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    
    # 量子化設定
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    
    # 学習設定
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    max_steps: int = -1
    
    # メモリ最適化設定
    gradient_checkpointing: bool = True
    dataloader_pin_memory: bool = False
    remove_unused_columns: bool = False
    
    # 最適化設定
    optimization_level: QLoRAOptimization = QLoRAOptimization.BALANCED
    memory_limit_gb: float = 4.0
    
    # 出力設定
    output_dir: str = "./qlora_output"
    save_steps: int = 500
    logging_steps: int = 10
    
    # その他
    seed: int = 42
    fp16: bool = False
    bf16: bool = True
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingMetrics:
    """学習メトリクス"""
    epoch: float = 0.0
    step: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    
    # メモリ使用量
    gpu_memory_mb: float = 0.0
    cpu_memory_mb: float = 0.0
    
    # 時間情報
    training_time: float = 0.0
    samples_per_second: float = 0.0
    
    # 品質指標
    perplexity: Optional[float] = None
    eval_loss: Optional[float] = None
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class QLoRATrainingResult:
    """QLoRA学習結果"""
    config: QLoRAConfig
    adapter_name: str
    
    # 学習統計
    total_epochs: int = 0
    total_steps: int = 0
    final_loss: float = 0.0
    best_eval_loss: Optional[float] = None
    
    # 時間統計
    training_time: float = 0.0
    average_step_time: float = 0.0
    
    # メモリ統計
    peak_memory_mb: float = 0.0
    average_memory_mb: float = 0.0
    
    # モデル情報
    trainable_parameters: int = 0
    total_parameters: int = 0
    trainable_ratio: float = 0.0
    
    # 品質情報
    final_perplexity: Optional[float] = None
    
    # 結果情報
    success: bool = True
    error_message: Optional[str] = None
    
    # メトリクス履歴
    metrics_history: List[TrainingMetrics] = field(default_factory=list)
    
    # ファイルパス
    model_path: Optional[str] = None
    checkpoint_paths: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class QLoRATrainingCallback(TrainerCallback):
    """QLoRA学習専用コールバック"""
    
    def __init__(self, qlora_trainer: 'QLoRATrainer'):
        self.qlora_trainer = qlora_trainer
        self.logger = get_logger()
        self.start_time = None
        self.step_start_time = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        """学習開始時"""
        self.start_time = time.time()
        self.qlora_trainer.current_phase = TrainingPhase.TRAINING
        
        self.logger.log_performance_metric(
            metric_name="qlora_training_started",
            value=state.max_steps,
            unit="steps",
            component="qlora_trainer"
        )
    
    def on_step_begin(self, args, state, control, **kwargs):
        """ステップ開始時"""
        self.step_start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        """ステップ終了時"""
        if self.step_start_time:
            step_time = time.time() - self.step_start_time
            
            # メモリ使用量取得
            gpu_memory = 0.0
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024**2)
            
            # メトリクス記録
            metrics = TrainingMetrics(
                epoch=state.epoch,
                step=state.global_step,
                loss=state.log_history[-1].get('train_loss', 0.0) if state.log_history else 0.0,
                learning_rate=state.log_history[-1].get('learning_rate', 0.0) if state.log_history else 0.0,
                gpu_memory_mb=gpu_memory,
                training_time=time.time() - self.start_time if self.start_time else 0.0,
                samples_per_second=1.0 / step_time if step_time > 0 else 0.0
            )
            
            self.qlora_trainer.training_metrics.append(metrics)
            
            # メモリ統計更新
            if gpu_memory > self.qlora_trainer.peak_memory_mb:
                self.qlora_trainer.peak_memory_mb = gpu_memory
    
    def on_evaluate(self, args, state, control, **kwargs):
        """評価時"""
        self.qlora_trainer.current_phase = TrainingPhase.EVALUATION
        
        # 評価メトリクス記録
        if state.log_history:
            latest_log = state.log_history[-1]
            eval_loss = latest_log.get('eval_loss')
            
            if eval_loss is not None:
                if (self.qlora_trainer.best_eval_loss is None or 
                    eval_loss < self.qlora_trainer.best_eval_loss):
                    self.qlora_trainer.best_eval_loss = eval_loss
    
    def on_save(self, args, state, control, **kwargs):
        """保存時"""
        self.qlora_trainer.current_phase = TrainingPhase.SAVING
        
        self.logger.log_performance_metric(
            metric_name="qlora_checkpoint_saved",
            value=state.global_step,
            unit="step",
            component="qlora_trainer"
        )
    
    def on_train_end(self, args, state, control, **kwargs):
        """学習終了時"""
        self.qlora_trainer.current_phase = TrainingPhase.COMPLETED
        
        total_time = time.time() - self.start_time if self.start_time else 0.0
        
        self.logger.log_performance_metric(
            metric_name="qlora_training_completed",
            value=total_time,
            unit="seconds",
            component="qlora_trainer"
        )


class QLoRATrainer:
    """QLoRA学習システム"""
    
    def __init__(
        self,
        base_model_name: str,
        bnb_manager: Optional[BitsAndBytesManager] = None,
        peft_pool: Optional[PEFTAdapterPool] = None,
        system_monitor: Optional[SystemMonitor] = None
    ):
        self.base_model_name = base_model_name
        self.bnb_manager = bnb_manager
        self.peft_pool = peft_pool
        self.system_monitor = system_monitor
        
        self.config = get_config()
        self.logger = get_logger()
        
        # 学習状態
        self.current_phase = TrainingPhase.INITIALIZATION
        self.training_metrics: List[TrainingMetrics] = []
        self.peak_memory_mb = 0.0
        self.best_eval_loss: Optional[float] = None
        
        # モデルとトークナイザー
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # 学習履歴
        self.training_history: List[QLoRATrainingResult] = []
        
        self.logger.log_startup(
            component="qlora_trainer",
            version="1.0.0",
            config_summary={
                "base_model": base_model_name,
                "qlora_available": QLORA_AVAILABLE,
                "cuda_available": torch.cuda.is_available() if QLORA_AVAILABLE else False
            }
        )
    
    async def initialize(self) -> bool:
        """QLoRA学習システム初期化"""
        try:
            if not QLORA_AVAILABLE:
                self.logger.log_alert(
                    alert_type="qlora_unavailable",
                    severity="WARNING",
                    message="QLoRA dependencies not available"
                )
                return False
            
            self.current_phase = TrainingPhase.INITIALIZATION
            
            self.logger.log_startup(
                component="qlora_trainer_initialized",
                version="1.0.0",
                config_summary={
                    "initialization_complete": True
                }
            )
            
            return True
            
        except Exception as e:
            self.logger.log_alert(
                alert_type="qlora_initialization_failed",
                severity="ERROR",
                message=f"QLoRA trainer initialization failed: {e}"
            )
            return False
    
    def create_qlora_config(
        self,
        optimization_level: QLoRAOptimization = QLoRAOptimization.BALANCED,
        memory_limit_gb: float = 4.0,
        **kwargs
    ) -> QLoRAConfig:
        """QLoRA設定作成"""
        
        # 最適化レベルに応じた設定調整
        if optimization_level == QLoRAOptimization.MEMORY_OPTIMIZED:
            defaults = {
                "lora_r": 32,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 8,
                "gradient_checkpointing": True,
                "dataloader_pin_memory": False,
                "bf16": True,
                "fp16": False
            }
        elif optimization_level == QLoRAOptimization.PERFORMANCE_OPTIMIZED:
            defaults = {
                "lora_r": 128,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 2,
                "gradient_checkpointing": False,
                "dataloader_pin_memory": True,
                "bf16": False,
                "fp16": True
            }
        else:  # BALANCED
            defaults = {
                "lora_r": 64,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "gradient_checkpointing": True,
                "dataloader_pin_memory": False,
                "bf16": True,
                "fp16": False
            }
        
        # メモリ制限に応じた調整
        if memory_limit_gb < 3.0:
            defaults.update({
                "lora_r": min(defaults["lora_r"], 32),
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": max(defaults["gradient_accumulation_steps"], 8)
            })
        
        # ユーザー指定値で上書き
        defaults.update(kwargs)
        defaults["optimization_level"] = optimization_level
        defaults["memory_limit_gb"] = memory_limit_gb
        
        return QLoRAConfig(**defaults)
    
    def create_quantization_config(self, qlora_config: QLoRAConfig) -> BitsAndBytesConfig:
        """量子化設定作成"""
        
        compute_dtype = torch.bfloat16 if qlora_config.bf16 else torch.float16
        
        return BitsAndBytesConfig(
            load_in_4bit=qlora_config.load_in_4bit,
            bnb_4bit_use_double_quant=qlora_config.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=qlora_config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype
        )
    
    def create_lora_config(self, qlora_config: QLoRAConfig) -> LoraConfig:
        """LoRA設定作成"""
        
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=qlora_config.lora_r,
            lora_alpha=qlora_config.lora_alpha,
            lora_dropout=qlora_config.lora_dropout,
            target_modules=qlora_config.target_modules,
            bias="none"
        )
    
    async def prepare_model(self, qlora_config: QLoRAConfig) -> Tuple[PeftModel, Any]:
        """モデル準備"""
        
        self.current_phase = TrainingPhase.PREPARATION
        
        try:
            # 量子化設定
            bnb_config = self.create_quantization_config(qlora_config)
            
            # ベースモデル読み込み
            self.logger.log_performance_metric(
                metric_name="qlora_model_loading_start",
                value=1,
                unit="model",
                component="qlora_trainer"
            )
            
            base_model = await asyncio.to_thread(
                AutoModelForCausalLM.from_pretrained,
                self.base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            # トークナイザー読み込み
            self.tokenizer = await asyncio.to_thread(
                AutoTokenizer.from_pretrained,
                self.base_model_name,
                trust_remote_code=True
            )
            
            # パディングトークン設定
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # LoRA設定
            lora_config = self.create_lora_config(qlora_config)
            
            # PEFTモデル作成
            self.model = await asyncio.to_thread(
                get_peft_model,
                base_model,
                lora_config
            )
            
            # 学習可能パラメータ情報
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            
            self.logger.log_performance_metric(
                metric_name="qlora_model_prepared",
                value=trainable_params,
                unit="parameters",
                component="qlora_trainer"
            )
            
            return self.model, {
                "trainable_parameters": trainable_params,
                "total_parameters": total_params,
                "trainable_ratio": trainable_params / total_params if total_params > 0 else 0.0
            }
            
        except Exception as e:
            self.logger.log_alert(
                alert_type="qlora_model_preparation_failed",
                severity="ERROR",
                message=f"QLoRA model preparation failed: {e}"
            )
            raise
    
    def create_training_arguments(self, qlora_config: QLoRAConfig) -> TrainingArguments:
        """学習引数作成"""
        
        return TrainingArguments(
            output_dir=qlora_config.output_dir,
            num_train_epochs=qlora_config.num_train_epochs,
            per_device_train_batch_size=qlora_config.per_device_train_batch_size,
            gradient_accumulation_steps=qlora_config.gradient_accumulation_steps,
            learning_rate=qlora_config.learning_rate,
            warmup_steps=qlora_config.warmup_steps,
            max_steps=qlora_config.max_steps,
            
            # 最適化設定
            gradient_checkpointing=qlora_config.gradient_checkpointing,
            dataloader_pin_memory=qlora_config.dataloader_pin_memory,
            remove_unused_columns=qlora_config.remove_unused_columns,
            
            # 精度設定
            fp16=qlora_config.fp16,
            bf16=qlora_config.bf16,
            
            # 保存・ログ設定
            save_steps=qlora_config.save_steps,
            logging_steps=qlora_config.logging_steps,
            save_total_limit=3,
            
            # その他
            seed=qlora_config.seed,
            report_to=None,  # wandb等の外部ログ無効化
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False
        )
    
    async def train(
        self,
        train_dataset: Union[Dataset, HFDataset],
        qlora_config: QLoRAConfig,
        eval_dataset: Optional[Union[Dataset, HFDataset]] = None,
        adapter_name: Optional[str] = None
    ) -> QLoRATrainingResult:
        """QLoRA学習実行"""
        
        if adapter_name is None:
            adapter_name = f"qlora_{int(time.time())}"
        
        start_time = time.time()
        
        try:
            # モデル準備
            model, model_info = await self.prepare_model(qlora_config)
            
            # 学習引数作成
            training_args = self.create_training_arguments(qlora_config)
            
            # コールバック作成
            callback = QLoRATrainingCallback(self)
            
            # Trainer作成
            self.trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                callbacks=[callback]
            )
            
            # 学習実行
            self.current_phase = TrainingPhase.TRAINING
            
            train_result = await asyncio.to_thread(self.trainer.train)
            
            # 結果作成
            training_time = time.time() - start_time
            
            result = QLoRATrainingResult(
                config=qlora_config,
                adapter_name=adapter_name,
                total_epochs=qlora_config.num_train_epochs,
                total_steps=train_result.global_step,
                final_loss=train_result.training_loss,
                best_eval_loss=self.best_eval_loss,
                training_time=training_time,
                average_step_time=training_time / train_result.global_step if train_result.global_step > 0 else 0.0,
                peak_memory_mb=self.peak_memory_mb,
                average_memory_mb=sum(m.gpu_memory_mb for m in self.training_metrics) / len(self.training_metrics) if self.training_metrics else 0.0,
                trainable_parameters=model_info["trainable_parameters"],
                total_parameters=model_info["total_parameters"],
                trainable_ratio=model_info["trainable_ratio"],
                metrics_history=self.training_metrics.copy(),
                model_path=qlora_config.output_dir,
                success=True
            )
            
            # アダプタ保存
            await self._save_adapter(adapter_name, qlora_config.output_dir)
            
            # 履歴に追加
            self.training_history.append(result)
            
            self.logger.log_performance_metric(
                metric_name="qlora_training_success",
                value=training_time,
                unit="seconds",
                component="qlora_trainer"
            )
            
            return result
            
        except Exception as e:
            # エラー結果作成
            error_result = QLoRATrainingResult(
                config=qlora_config,
                adapter_name=adapter_name,
                training_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
            
            self.training_history.append(error_result)
            
            self.logger.log_alert(
                alert_type="qlora_training_failed",
                severity="ERROR",
                message=f"QLoRA training failed: {e}"
            )
            
            raise
    
    async def _save_adapter(self, adapter_name: str, output_dir: str) -> None:
        """アダプタ保存"""
        
        try:
            if self.model and self.peft_pool:
                # PEFTプールに追加
                adapter_config = AdapterConfig(
                    name=adapter_name,
                    adapter_type=AdapterType.LORA,
                    description=f"QLoRA trained adapter from {self.base_model_name}"
                )
                
                # アダプタ情報をプールに登録
                # 注意: 実際のモデルは既に保存されているので、パス情報のみ登録
                from .peft_manager import AdapterInfo, AdapterStatus
                
                adapter_info = AdapterInfo(
                    config=adapter_config,
                    status=AdapterStatus.INACTIVE,
                    local_path=output_dir,
                    parameter_count=self.training_history[-1].trainable_parameters if self.training_history else 0
                )
                
                self.peft_pool.adapters[adapter_name] = adapter_info
                
        except Exception as e:
            self.logger.log_alert(
                alert_type="adapter_save_failed",
                severity="WARNING",
                message=f"Failed to save adapter to pool: {e}"
            )
    
    def get_training_stats(self) -> Dict[str, Any]:
        """学習統計取得"""
        
        if not self.training_history:
            return {"total_trainings": 0}
        
        successful_trainings = [r for r in self.training_history if r.success]
        
        stats = {
            "total_trainings": len(self.training_history),
            "successful_trainings": len(successful_trainings),
            "failed_trainings": len(self.training_history) - len(successful_trainings),
            "success_rate": len(successful_trainings) / len(self.training_history) if self.training_history else 0.0
        }
        
        if successful_trainings:
            stats.update({
                "average_training_time": sum(r.training_time for r in successful_trainings) / len(successful_trainings),
                "average_final_loss": sum(r.final_loss for r in successful_trainings) / len(successful_trainings),
                "average_peak_memory": sum(r.peak_memory_mb for r in successful_trainings) / len(successful_trainings),
                "average_trainable_ratio": sum(r.trainable_ratio for r in successful_trainings) / len(successful_trainings)
            })
        
        return stats
    
    def get_training_history(
        self,
        limit: Optional[int] = None,
        success_only: bool = False
    ) -> List[QLoRATrainingResult]:
        """学習履歴取得"""
        
        history = self.training_history
        
        if success_only:
            history = [r for r in history if r.success]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    async def cleanup(self) -> None:
        """リソースクリーンアップ"""
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.log_alert(
            alert_type="qlora_cleanup",
            severity="INFO",
            message="QLoRA trainer resources cleaned up"
        )
    
    async def shutdown(self) -> None:
        """QLoRA学習システム終了"""
        
        await self.cleanup()
        
        final_stats = self.get_training_stats()
        
        self.logger.log_shutdown(
            component="qlora_trainer",
            uptime_seconds=0,  # TODO: 実際の稼働時間計算
            final_stats=final_stats
        )


# 便利関数
async def create_qlora_trainer(
    base_model_name: str,
    bnb_manager: Optional[BitsAndBytesManager] = None,
    peft_pool: Optional[PEFTAdapterPool] = None,
    system_monitor: Optional[SystemMonitor] = None
) -> QLoRATrainer:
    """QLoRA学習システム作成・初期化"""
    
    trainer = QLoRATrainer(base_model_name, bnb_manager, peft_pool, system_monitor)
    
    if await trainer.initialize():
        return trainer
    else:
        raise RuntimeError("Failed to initialize QLoRA trainer")


def create_simple_dataset(texts: List[str], tokenizer, max_length: int = 512) -> HFDataset:
    """簡単なデータセット作成"""
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
    
    dataset = HFDataset.from_dict({"text": texts})
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    return tokenized_dataset


# 使用例
async def main():
    """テスト用メイン関数"""
    try:
        # QLoRA学習システム作成
        trainer = await create_qlora_trainer("microsoft/DialoGPT-small")
        
        print("=== QLoRA Trainer Test ===")
        
        # QLoRA設定作成
        qlora_config = trainer.create_qlora_config(
            optimization_level=QLoRAOptimization.MEMORY_OPTIMIZED,
            memory_limit_gb=4.0,
            num_train_epochs=1,
            max_steps=10  # テスト用に短縮
        )
        
        print(f"QLoRA Config: {qlora_config}")
        
        # 簡単なデータセット作成
        sample_texts = [
            "Hello, how are you?",
            "I'm fine, thank you.",
            "What's the weather like?",
            "It's sunny today."
        ]
        
        # モデル準備（学習なしでテスト）
        model, model_info = await trainer.prepare_model(qlora_config)
        
        print(f"Model Info: {model_info}")
        print(f"Trainable ratio: {model_info['trainable_ratio']:.4f}")
        
        # 統計取得
        stats = trainer.get_training_stats()
        print(f"Training Stats: {stats}")
        
        await trainer.shutdown()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())