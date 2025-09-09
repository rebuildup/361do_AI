"""
Quantization Optimization
量子化最適化システム

モデルの量子化によるメモリ使用量削減と推論速度向上を提供します。
RTX 4050 6GB VRAM環境での効率的な量子化を実装します。
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic
import numpy as np

logger = logging.getLogger(__name__)


class QuantizationType(Enum):
    """量子化タイプ"""
    DYNAMIC = "dynamic"      # 動的量子化
    STATIC = "static"        # 静的量子化
    QAT = "qat"             # 量子化認識訓練
    INT8 = "int8"           # INT8量子化
    INT4 = "int4"           # INT4量子化
    FP16 = "fp16"           # FP16半精度


class QuantizationStrategy(Enum):
    """量子化戦略"""
    MEMORY_OPTIMIZED = "memory_optimized"    # メモリ最適化
    SPEED_OPTIMIZED = "speed_optimized"      # 速度最適化
    BALANCED = "balanced"                    # バランス型
    RTX4050_OPTIMIZED = "rtx4050_optimized"  # RTX 4050専用


@dataclass
class QuantizationConfig:
    """量子化設定"""
    quantization_type: QuantizationType = QuantizationType.DYNAMIC
    strategy: QuantizationStrategy = QuantizationStrategy.RTX4050_OPTIMIZED
    target_precision: str = "int8"
    calibration_samples: int = 100
    enable_fusion: bool = True
    preserve_accuracy_threshold: float = 0.95
    memory_reduction_target: float = 0.5  # 50%削減目標
    speed_improvement_target: float = 1.5  # 1.5倍高速化目標


@dataclass
class QuantizationMetrics:
    """量子化メトリクス"""
    original_model_size_mb: float = 0.0
    quantized_model_size_mb: float = 0.0
    memory_reduction_percent: float = 0.0
    compression_ratio: float = 0.0
    inference_time_original_ms: float = 0.0
    inference_time_quantized_ms: float = 0.0
    speed_improvement_factor: float = 0.0
    accuracy_original: float = 0.0
    accuracy_quantized: float = 0.0
    accuracy_drop_percent: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class QuantizationOptimizer:
    """量子化最適化システム"""
    
    def __init__(self, config: Optional[QuantizationConfig] = None):
        """
        初期化
        
        Args:
            config: 量子化設定
        """
        self.config = config or QuantizationConfig()
        self.quantization_history: List[QuantizationMetrics] = []
        self.quantized_models: Dict[str, Any] = {}
        
        logger.info(f"量子化最適化システム初期化完了 - タイプ: {self.config.quantization_type.value}")
    
    async def quantize_model(
        self,
        model: nn.Module,
        model_name: str,
        calibration_data: Optional[List[torch.Tensor]] = None,
        test_data: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Dict[str, Any]:
        """
        モデル量子化
        
        Args:
            model: 量子化対象モデル
            model_name: モデル名
            calibration_data: キャリブレーションデータ
            test_data: テストデータ
            
        Returns:
            量子化結果
        """
        try:
            logger.info(f"モデル量子化開始: {model_name}")
            
            # 元モデルのメトリクス取得
            original_metrics = await self._get_model_metrics(model, test_data)
            
            # 量子化実行
            if self.config.quantization_type == QuantizationType.DYNAMIC:
                quantized_model = await self._dynamic_quantization(model)
            elif self.config.quantization_type == QuantizationType.STATIC:
                quantized_model = await self._static_quantization(model, calibration_data)
            elif self.config.quantization_type == QuantizationType.INT8:
                quantized_model = await self._int8_quantization(model)
            elif self.config.quantization_type == QuantizationType.FP16:
                quantized_model = await self._fp16_quantization(model)
            else:
                raise ValueError(f"未対応の量子化タイプ: {self.config.quantization_type}")
            
            # 量子化後モデルのメトリクス取得
            quantized_metrics = await self._get_model_metrics(quantized_model, test_data)
            
            # 量子化メトリクス計算
            quantization_metrics = self._calculate_quantization_metrics(
                original_metrics, quantized_metrics
            )
            
            # 履歴に追加
            self.quantization_history.append(quantization_metrics)
            self.quantized_models[model_name] = quantized_model
            
            result = {
                "model_name": model_name,
                "quantization_successful": True,
                "quantized_model": quantized_model,
                "metrics": quantization_metrics,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"モデル量子化完了: {model_name} - メモリ削減: {quantization_metrics.memory_reduction_percent:.1f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"モデル量子化エラー: {e}")
            return {
                "model_name": model_name,
                "quantization_successful": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def optimize_for_rtx4050(
        self,
        model: nn.Module,
        model_name: str,
        test_data: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Dict[str, Any]:
        """
        RTX 4050専用最適化
        
        Args:
            model: 最適化対象モデル
            model_name: モデル名
            test_data: テストデータ
            
        Returns:
            最適化結果
        """
        try:
            logger.info(f"RTX 4050専用最適化開始: {model_name}")
            
            optimization_results = []
            
            # 複数の量子化戦略を試行
            strategies = [
                QuantizationType.DYNAMIC,
                QuantizationType.INT8,
                QuantizationType.FP16
            ]
            
            best_result = None
            best_score = 0.0
            
            for strategy in strategies:
                # 一時的に量子化タイプを変更
                original_type = self.config.quantization_type
                self.config.quantization_type = strategy
                
                try:
                    result = await self.quantize_model(model, f"{model_name}_{strategy.value}", test_data=test_data)
                    
                    if result["quantization_successful"]:
                        metrics = result["metrics"]
                        
                        # RTX 4050用スコア計算（メモリ削減と速度向上のバランス）
                        memory_score = metrics.memory_reduction_percent / 100.0
                        speed_score = min(metrics.speed_improvement_factor / 2.0, 1.0)  # 2倍を上限とする
                        accuracy_score = max(0, 1.0 - metrics.accuracy_drop_percent / 100.0)
                        
                        # 重み付きスコア（メモリ: 40%, 速度: 30%, 精度: 30%）
                        total_score = (memory_score * 0.4 + speed_score * 0.3 + accuracy_score * 0.3)
                        
                        if total_score > best_score:
                            best_score = total_score
                            best_result = result
                        
                        optimization_results.append({
                            "strategy": strategy.value,
                            "score": total_score,
                            "metrics": metrics
                        })
                
                except Exception as e:
                    logger.warning(f"戦略 {strategy.value} の最適化失敗: {e}")
                
                finally:
                    # 元の設定に戻す
                    self.config.quantization_type = original_type
            
            if best_result:
                logger.info(f"RTX 4050最適化完了: {model_name} - 最適戦略: {best_result['model_name']}")
                return {
                    "optimization_successful": True,
                    "best_result": best_result,
                    "all_results": optimization_results,
                    "best_score": best_score,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "optimization_successful": False,
                    "error": "すべての最適化戦略が失敗しました",
                    "all_results": optimization_results,
                    "timestamp": datetime.now().isoformat()
                }
            
        except Exception as e:
            logger.error(f"RTX 4050最適化エラー: {e}")
            return {
                "optimization_successful": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _dynamic_quantization(self, model: nn.Module) -> nn.Module:
        """動的量子化"""
        try:
            # 動的量子化実行
            quantized_model = quantize_dynamic(
                model,
                {nn.Linear, nn.LSTM, nn.GRU},
                dtype=torch.qint8
            )
            
            return quantized_model
            
        except Exception as e:
            logger.error(f"動的量子化エラー: {e}")
            raise
    
    async def _static_quantization(
        self,
        model: nn.Module,
        calibration_data: Optional[List[torch.Tensor]] = None
    ) -> nn.Module:
        """静的量子化"""
        try:
            # 現行のPyTorch安定版では本実装は未対応のため未実装扱い
            raise NotImplementedError("静的量子化は未サポートです")
            
        except Exception as e:
            logger.error(f"静的量子化エラー: {e}")
            raise
    
    async def _int8_quantization(self, model: nn.Module) -> nn.Module:
        """INT8量子化"""
        try:
            # INT8量子化（簡略化実装）
            model.eval()
            
            # 実際の実装では、より詳細なINT8量子化処理が必要
            # ここでは動的量子化を使用
            quantized_model = quantize_dynamic(
                model,
                {nn.Linear, nn.Conv2d, nn.LSTM, nn.GRU},
                dtype=torch.qint8
            )
            
            return quantized_model
            
        except Exception as e:
            logger.error(f"INT8量子化エラー: {e}")
            raise
    
    async def _fp16_quantization(self, model: nn.Module) -> nn.Module:
        """FP16量子化"""
        try:
            # FP16半精度変換
            if torch.cuda.is_available():
                model = model.half()
            else:
                # CPU環境ではFP16をサポートしていない場合がある
                logger.warning("CPU環境ではFP16量子化をスキップします")
                return model
            
            return model
            
        except Exception as e:
            logger.error(f"FP16量子化エラー: {e}")
            raise
    
    async def _get_model_metrics(
        self,
        model: nn.Module,
        test_data: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Dict[str, Any]:
        """モデルメトリクス取得"""
        try:
            # モデルサイズ計算
            model_size_mb = self._calculate_model_size(model)
            
            # 推論時間測定
            inference_time_ms = 0.0
            if test_data:
                inference_time_ms = await self._measure_inference_time(model, test_data[:10])
            
            # 精度測定（簡略化）
            accuracy = 0.0
            if test_data:
                accuracy = await self._measure_accuracy(model, test_data[:50])
            
            return {
                "model_size_mb": model_size_mb,
                "inference_time_ms": inference_time_ms,
                "accuracy": accuracy
            }
            
        except Exception as e:
            logger.error(f"モデルメトリクス取得エラー: {e}")
            return {
                "model_size_mb": 0.0,
                "inference_time_ms": 0.0,
                "accuracy": 0.0
            }
    
    def _calculate_model_size(self, model: nn.Module) -> float:
        """モデルサイズ計算（MB）"""
        try:
            # パラメータ数を計算
            total_params = sum(p.numel() for p in model.parameters())
            
            # データ型に基づくサイズ計算
            if hasattr(model, 'dtype'):
                if model.dtype == torch.float32:
                    bytes_per_param = 4
                elif model.dtype == torch.float16:
                    bytes_per_param = 2
                elif model.dtype == torch.qint8:
                    bytes_per_param = 1
                else:
                    bytes_per_param = 4  # デフォルト
            else:
                bytes_per_param = 4  # デフォルト
            
            total_size_bytes = total_params * bytes_per_param
            total_size_mb = total_size_bytes / (1024 * 1024)
            
            return total_size_mb
            
        except Exception as e:
            logger.error(f"モデルサイズ計算エラー: {e}")
            return 0.0
    
    async def _measure_inference_time(
        self,
        model: nn.Module,
        test_data: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> float:
        """推論時間測定"""
        try:
            model.eval()
            
            total_time = 0.0
            num_samples = len(test_data)
            
            with torch.no_grad():
                for input_data, _ in test_data:
                    start_time = time.time()
                    _ = model(input_data)
                    end_time = time.time()
                    
                    total_time += (end_time - start_time) * 1000  # ms
            
            avg_time = total_time / num_samples if num_samples > 0 else 0.0
            return avg_time
            
        except Exception as e:
            logger.error(f"推論時間測定エラー: {e}")
            return 0.0
    
    async def _measure_accuracy(
        self,
        model: nn.Module,
        test_data: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> float:
        """精度測定"""
        try:
            model.eval()
            
            correct = 0
            total = 0
            
            with torch.no_grad():
                for input_data, target in test_data:
                    output = model(input_data)
                    
                    # 分類タスクの場合
                    if len(output.shape) > 1 and output.shape[1] > 1:
                        predicted = torch.argmax(output, dim=1)
                        correct += (predicted == target).sum().item()
                        total += target.size(0)
                    else:
                        # 回帰タスクの場合（簡略化）
                        correct += 1
                        total += 1
            
            accuracy = correct / total if total > 0 else 0.0
            return accuracy
            
        except Exception as e:
            logger.error(f"精度測定エラー: {e}")
            return 0.0
    
    def _calculate_quantization_metrics(
        self,
        original_metrics: Dict[str, Any],
        quantized_metrics: Dict[str, Any]
    ) -> QuantizationMetrics:
        """量子化メトリクス計算"""
        try:
            original_size = original_metrics["model_size_mb"]
            quantized_size = quantized_metrics["model_size_mb"]
            
            memory_reduction_percent = 0.0
            compression_ratio = 0.0
            
            if original_size > 0:
                memory_reduction_percent = ((original_size - quantized_size) / original_size) * 100
                compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
            
            original_time = original_metrics["inference_time_ms"]
            quantized_time = quantized_metrics["inference_time_ms"]
            
            speed_improvement_factor = 0.0
            if quantized_time > 0:
                speed_improvement_factor = original_time / quantized_time
            
            original_accuracy = original_metrics["accuracy"]
            quantized_accuracy = quantized_metrics["accuracy"]
            
            accuracy_drop_percent = 0.0
            if original_accuracy > 0:
                accuracy_drop_percent = ((original_accuracy - quantized_accuracy) / original_accuracy) * 100
            
            return QuantizationMetrics(
                original_model_size_mb=original_size,
                quantized_model_size_mb=quantized_size,
                memory_reduction_percent=memory_reduction_percent,
                compression_ratio=compression_ratio,
                inference_time_original_ms=original_time,
                inference_time_quantized_ms=quantized_time,
                speed_improvement_factor=speed_improvement_factor,
                accuracy_original=original_accuracy,
                accuracy_quantized=quantized_accuracy,
                accuracy_drop_percent=accuracy_drop_percent
            )
            
        except Exception as e:
            logger.error(f"量子化メトリクス計算エラー: {e}")
            return QuantizationMetrics()
    
    def get_quantization_history(self) -> List[QuantizationMetrics]:
        """量子化履歴取得"""
        return self.quantization_history.copy()
    
    def get_quantized_models(self) -> Dict[str, Any]:
        """量子化済みモデル取得"""
        return self.quantized_models.copy()
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """最適化サマリー取得"""
        if not self.quantization_history:
            return {"message": "量子化履歴がありません"}
        
        total_models = len(self.quantization_history)
        avg_memory_reduction = sum(m.memory_reduction_percent for m in self.quantization_history) / total_models
        avg_speed_improvement = sum(m.speed_improvement_factor for m in self.quantization_history) / total_models
        avg_accuracy_drop = sum(m.accuracy_drop_percent for m in self.quantization_history) / total_models
        
        return {
            "total_models_quantized": total_models,
            "average_memory_reduction_percent": avg_memory_reduction,
            "average_speed_improvement_factor": avg_speed_improvement,
            "average_accuracy_drop_percent": avg_accuracy_drop,
            "quantized_model_names": list(self.quantized_models.keys())
        }


# 使用例
async def main():
    """使用例"""
    # 設定
    config = QuantizationConfig(
        quantization_type=QuantizationType.DYNAMIC,
        strategy=QuantizationStrategy.RTX4050_OPTIMIZED
    )
    
    # 量子化最適化システム初期化
    optimizer = QuantizationOptimizer(config)
    
    # サンプルモデル（簡略化）
    class SampleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(100, 50)
            self.linear2 = nn.Linear(50, 10)
        
        def forward(self, x):
            x = torch.relu(self.linear1(x))
            x = self.linear2(x)
            return x
    
    model = SampleModel()
    
    # テストデータ生成
    test_data = [
        (torch.randn(1, 100), torch.randint(0, 10, (1,)))
        for _ in range(100)
    ]
    
    try:
        # モデル量子化
        result = await optimizer.quantize_model(model, "sample_model", test_data=test_data)
        
        if result["quantization_successful"]:
            metrics = result["metrics"]
            print(f"量子化成功:")
            print(f"  メモリ削減: {metrics.memory_reduction_percent:.1f}%")
            print(f"  速度向上: {metrics.speed_improvement_factor:.2f}倍")
            print(f"  精度低下: {metrics.accuracy_drop_percent:.1f}%")
        
        # RTX 4050専用最適化
        rtx_result = await optimizer.optimize_for_rtx4050(model, "sample_model_rtx", test_data=test_data)
        
        if rtx_result["optimization_successful"]:
            print(f"RTX 4050最適化成功 - スコア: {rtx_result['best_score']:.3f}")
        
        # サマリー表示
        summary = optimizer.get_optimization_summary()
        print(f"最適化サマリー: {summary}")
        
    except Exception as e:
        print(f"エラー: {e}")


if __name__ == "__main__":
    asyncio.run(main())
