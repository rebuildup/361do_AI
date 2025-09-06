"""
Inference Speed Optimization
推論速度最適化システム

推論処理の高速化とスループット向上を提供します。
RTX 4050 6GB VRAM環境での効率的な推論最適化を実装します。
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import torch
import torch.nn as nn
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """最適化戦略"""
    SPEED_FIRST = "speed_first"        # 速度優先
    THROUGHPUT_FIRST = "throughput_first"  # スループット優先
    BALANCED = "balanced"              # バランス型
    RTX4050_OPTIMIZED = "rtx4050_optimized"  # RTX 4050専用


class BatchStrategy(Enum):
    """バッチ戦略"""
    DYNAMIC_BATCHING = "dynamic_batching"    # 動的バッチング
    FIXED_BATCHING = "fixed_batching"        # 固定バッチング
    ADAPTIVE_BATCHING = "adaptive_batching"  # 適応的バッチング


@dataclass
class InferenceOptimizationConfig:
    """推論最適化設定"""
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.RTX4050_OPTIMIZED
    batch_strategy: BatchStrategy = BatchStrategy.ADAPTIVE_BATCHING
    max_batch_size: int = 32
    min_batch_size: int = 1
    target_latency_ms: float = 100.0
    target_throughput_qps: float = 100.0
    enable_model_compilation: bool = True
    enable_tensor_parallelism: bool = False
    enable_pipeline_parallelism: bool = False
    enable_mixed_precision: bool = True
    enable_caching: bool = True
    cache_size: int = 1000
    enable_prefetching: bool = True
    prefetch_buffer_size: int = 10


@dataclass
class InferenceMetrics:
    """推論メトリクス"""
    inference_time_ms: float = 0.0
    throughput_qps: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    batch_size: int = 1
    memory_usage_mb: float = 0.0
    gpu_utilization_percent: float = 0.0
    cpu_utilization_percent: float = 0.0
    cache_hit_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BatchRequest:
    """バッチリクエスト"""
    request_id: str
    input_data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 0  # 優先度（高いほど優先）


class InferenceOptimizer:
    """推論速度最適化システム"""
    
    def __init__(self, config: Optional[InferenceOptimizationConfig] = None):
        """
        初期化
        
        Args:
            config: 推論最適化設定
        """
        self.config = config or InferenceOptimizationConfig()
        self.inference_history: List[InferenceMetrics] = []
        self.optimized_models: Dict[str, Any] = {}
        self.cache: Dict[str, Any] = {}
        self.batch_queue: Queue = Queue()
        self.is_processing = False
        self.processing_thread: Optional[threading.Thread] = None
        
        # パフォーマンス統計
        self.performance_stats = {
            "total_inferences": 0,
            "total_batches": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_inference_time_ms": 0.0,
            "avg_throughput_qps": 0.0,
            "total_processing_time_ms": 0.0
        }
        
        # バッチ処理用のスレッドプール
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"推論速度最適化システム初期化完了 - 戦略: {self.config.optimization_strategy.value}")
    
    async def optimize_model(
        self,
        model: nn.Module,
        model_name: str,
        sample_input: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        モデル最適化
        
        Args:
            model: 最適化対象モデル
            model_name: モデル名
            sample_input: サンプル入力（コンパイル用）
            
        Returns:
            最適化結果
        """
        try:
            logger.info(f"モデル最適化開始: {model_name}")
            
            optimized_model = model
            
            # 戦略に基づく最適化実行
            if self.config.optimization_strategy == OptimizationStrategy.RTX4050_OPTIMIZED:
                optimized_model = await self._rtx4050_optimization(model, sample_input)
            elif self.config.optimization_strategy == OptimizationStrategy.SPEED_FIRST:
                optimized_model = await self._speed_first_optimization(model, sample_input)
            elif self.config.optimization_strategy == OptimizationStrategy.THROUGHPUT_FIRST:
                optimized_model = await self._throughput_first_optimization(model, sample_input)
            else:  # BALANCED
                optimized_model = await self._balanced_optimization(model, sample_input)
            
            # 最適化済みモデルを保存
            self.optimized_models[model_name] = optimized_model
            
            result = {
                "model_name": model_name,
                "optimization_successful": True,
                "optimized_model": optimized_model,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"モデル最適化完了: {model_name}")
            
            return result
            
        except Exception as e:
            logger.error(f"モデル最適化エラー: {e}")
            return {
                "model_name": model_name,
                "optimization_successful": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def batch_inference(
        self,
        model: nn.Module,
        batch_requests: List[BatchRequest],
        model_name: str = "default"
    ) -> Dict[str, Any]:
        """
        バッチ推論実行
        
        Args:
            model: 推論モデル
            batch_requests: バッチリクエスト
            model_name: モデル名
            
        Returns:
            バッチ推論結果
        """
        try:
            if not batch_requests:
                return {"batch_results": [], "metrics": InferenceMetrics()}
            
            logger.info(f"バッチ推論開始: {len(batch_requests)}個のリクエスト")
            
            start_time = time.time()
            
            # バッチサイズの決定
            optimal_batch_size = await self._determine_optimal_batch_size(model, batch_requests)
            
            # バッチに分割
            batches = self._create_batches(batch_requests, optimal_batch_size)
            
            # バッチ推論実行
            batch_results = []
            for batch in batches:
                batch_result = await self._execute_batch_inference(model, batch, model_name)
                batch_results.append(batch_result)
            
            end_time = time.time()
            total_time_ms = (end_time - start_time) * 1000
            
            # メトリクス計算
            metrics = await self._calculate_batch_metrics(batch_results, total_time_ms, len(batch_requests))
            
            # 履歴に追加
            self.inference_history.append(metrics)
            self.performance_stats["total_inferences"] += len(batch_requests)
            self.performance_stats["total_batches"] += len(batches)
            
            result = {
                "batch_results": batch_results,
                "metrics": metrics,
                "total_requests": len(batch_requests),
                "total_batches": len(batches),
                "optimal_batch_size": optimal_batch_size,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"バッチ推論完了: {len(batch_requests)}個のリクエスト, {total_time_ms:.1f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"バッチ推論エラー: {e}")
            return {
                "batch_results": [],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def single_inference(
        self,
        model: nn.Module,
        input_data: Any,
        model_name: str = "default"
    ) -> Dict[str, Any]:
        """
        単一推論実行
        
        Args:
            model: 推論モデル
            input_data: 入力データ
            model_name: モデル名
            
        Returns:
            推論結果
        """
        try:
            # キャッシュチェック
            cache_key = self._generate_cache_key(input_data, model_name)
            if self.config.enable_caching and cache_key in self.cache:
                self.performance_stats["cache_hits"] += 1
                return {
                    "result": self.cache[cache_key],
                    "from_cache": True,
                    "inference_time_ms": 0.0,
                    "timestamp": datetime.now().isoformat()
                }
            
            self.performance_stats["cache_misses"] += 1
            
            # 推論実行
            start_time = time.time()
            
            with torch.no_grad():
                if isinstance(input_data, torch.Tensor):
                    if torch.cuda.is_available():
                        input_data = input_data.cuda()
                    result = model(input_data)
                    if torch.cuda.is_available():
                        result = result.cpu()
                else:
                    result = model(input_data)
            
            end_time = time.time()
            inference_time_ms = (end_time - start_time) * 1000
            
            # キャッシュに保存
            if self.config.enable_caching:
                self._update_cache(cache_key, result)
            
            # 統計更新
            self.performance_stats["total_inferences"] += 1
            self.performance_stats["total_processing_time_ms"] += inference_time_ms
            
            return {
                "result": result,
                "from_cache": False,
                "inference_time_ms": inference_time_ms,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"単一推論エラー: {e}")
            return {
                "result": None,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def start_batch_processing(self):
        """バッチ処理開始"""
        if self.is_processing:
            logger.warning("バッチ処理は既に開始されています")
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._batch_processing_loop)
        self.processing_thread.start()
        
        logger.info("バッチ処理開始")
    
    async def stop_batch_processing(self):
        """バッチ処理停止"""
        if not self.is_processing:
            return
        
        self.is_processing = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        logger.info("バッチ処理停止")
    
    async def _rtx4050_optimization(self, model: nn.Module, sample_input: Optional[torch.Tensor]) -> nn.Module:
        """RTX 4050専用最適化"""
        try:
            optimized_model = model
            
            # モデルを評価モードに設定
            optimized_model.eval()
            
            # 混合精度の有効化
            if self.config.enable_mixed_precision and torch.cuda.is_available():
                optimized_model = optimized_model.half()
            
            # モデルコンパイル（PyTorch 2.0+）
            if self.config.enable_model_compilation and hasattr(torch, 'compile'):
                try:
                    optimized_model = torch.compile(optimized_model)
                except Exception as e:
                    logger.warning(f"モデルコンパイル失敗: {e}")
            
            # CUDA最適化
            if torch.cuda.is_available():
                optimized_model = optimized_model.cuda()
                
                # メモリ最適化
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            return optimized_model
            
        except Exception as e:
            logger.error(f"RTX 4050最適化エラー: {e}")
            return model
    
    async def _speed_first_optimization(self, model: nn.Module, sample_input: Optional[torch.Tensor]) -> nn.Module:
        """速度優先最適化"""
        try:
            optimized_model = model.eval()
            
            # 積極的な最適化
            if torch.cuda.is_available():
                optimized_model = optimized_model.cuda()
                
                # 混合精度
                if self.config.enable_mixed_precision:
                    optimized_model = optimized_model.half()
                
                # モデルコンパイル
                if self.config.enable_model_compilation and hasattr(torch, 'compile'):
                    try:
                        optimized_model = torch.compile(optimized_model, mode="max-autotune")
                    except Exception as e:
                        logger.warning(f"モデルコンパイル失敗: {e}")
            
            return optimized_model
            
        except Exception as e:
            logger.error(f"速度優先最適化エラー: {e}")
            return model
    
    async def _throughput_first_optimization(self, model: nn.Module, sample_input: Optional[torch.Tensor]) -> nn.Module:
        """スループット優先最適化"""
        try:
            optimized_model = model.eval()
            
            # スループット最適化
            if torch.cuda.is_available():
                optimized_model = optimized_model.cuda()
                
                # バッチ処理最適化
                if self.config.enable_model_compilation and hasattr(torch, 'compile'):
                    try:
                        optimized_model = torch.compile(optimized_model, mode="reduce-overhead")
                    except Exception as e:
                        logger.warning(f"モデルコンパイル失敗: {e}")
            
            return optimized_model
            
        except Exception as e:
            logger.error(f"スループット優先最適化エラー: {e}")
            return model
    
    async def _balanced_optimization(self, model: nn.Module, sample_input: Optional[torch.Tensor]) -> nn.Module:
        """バランス型最適化"""
        try:
            optimized_model = model.eval()
            
            # バランス型最適化
            if torch.cuda.is_available():
                optimized_model = optimized_model.cuda()
                
                # 適度な最適化
                if self.config.enable_mixed_precision:
                    optimized_model = optimized_model.half()
                
                if self.config.enable_model_compilation and hasattr(torch, 'compile'):
                    try:
                        optimized_model = torch.compile(optimized_model, mode="default")
                    except Exception as e:
                        logger.warning(f"モデルコンパイル失敗: {e}")
            
            return optimized_model
            
        except Exception as e:
            logger.error(f"バランス型最適化エラー: {e}")
            return model
    
    async def _determine_optimal_batch_size(
        self,
        model: nn.Module,
        batch_requests: List[BatchRequest]
    ) -> int:
        """最適バッチサイズ決定"""
        try:
            # リクエスト数に基づく決定
            request_count = len(batch_requests)
            
            if self.config.batch_strategy == BatchStrategy.FIXED_BATCHING:
                return min(self.config.max_batch_size, request_count)
            
            elif self.config.batch_strategy == BatchStrategy.DYNAMIC_BATCHING:
                # 動的バッチサイズ決定
                if request_count <= 4:
                    return 1
                elif request_count <= 16:
                    return 4
                elif request_count <= 64:
                    return 16
                else:
                    return min(32, request_count)
            
            else:  # ADAPTIVE_BATCHING
                # 適応的バッチサイズ決定
                # メモリ使用量とリクエスト数を考慮
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    if gpu_memory < 4 * 1024 * 1024 * 1024:  # 4GB未満
                        return min(8, request_count)
                    elif gpu_memory < 8 * 1024 * 1024 * 1024:  # 8GB未満
                        return min(16, request_count)
                    else:
                        return min(32, request_count)
                else:
                    return min(8, request_count)
            
        except Exception as e:
            logger.error(f"最適バッチサイズ決定エラー: {e}")
            return min(self.config.max_batch_size, len(batch_requests))
    
    def _create_batches(self, batch_requests: List[BatchRequest], batch_size: int) -> List[List[BatchRequest]]:
        """バッチ作成"""
        batches = []
        for i in range(0, len(batch_requests), batch_size):
            batch = batch_requests[i:i + batch_size]
            batches.append(batch)
        return batches
    
    async def _execute_batch_inference(
        self,
        model: nn.Module,
        batch: List[BatchRequest],
        model_name: str
    ) -> Dict[str, Any]:
        """バッチ推論実行"""
        try:
            start_time = time.time()
            
            # バッチデータの準備
            batch_inputs = []
            for request in batch:
                batch_inputs.append(request.input_data)
            
            # バッチ推論実行
            with torch.no_grad():
                if isinstance(batch_inputs[0], torch.Tensor):
                    # テンソルの場合
                    batch_tensor = torch.stack(batch_inputs)
                    if torch.cuda.is_available():
                        batch_tensor = batch_tensor.cuda()
                    batch_output = model(batch_tensor)
                    if torch.cuda.is_available():
                        batch_output = batch_output.cpu()
                    
                    # 結果を分割
                    results = [batch_output[i] for i in range(len(batch))]
                else:
                    # その他の場合
                    results = []
                    for input_data in batch_inputs:
                        result = model(input_data)
                        results.append(result)
            
            end_time = time.time()
            inference_time_ms = (end_time - start_time) * 1000
            
            # 結果をリクエストとマッピング
            batch_results = []
            for i, request in enumerate(batch):
                batch_results.append({
                    "request_id": request.request_id,
                    "result": results[i],
                    "inference_time_ms": inference_time_ms / len(batch)
                })
            
            return {
                "batch_results": batch_results,
                "total_inference_time_ms": inference_time_ms,
                "batch_size": len(batch)
            }
            
        except Exception as e:
            logger.error(f"バッチ推論実行エラー: {e}")
            return {
                "batch_results": [],
                "error": str(e),
                "batch_size": len(batch)
            }
    
    async def _calculate_batch_metrics(
        self,
        batch_results: List[Dict[str, Any]],
        total_time_ms: float,
        total_requests: int
    ) -> InferenceMetrics:
        """バッチメトリクス計算"""
        try:
            # 推論時間の統計
            inference_times = []
            for batch_result in batch_results:
                if "total_inference_time_ms" in batch_result:
                    inference_times.append(batch_result["total_inference_time_ms"])
            
            if inference_times:
                avg_inference_time = sum(inference_times) / len(inference_times)
                latency_p50 = np.percentile(inference_times, 50)
                latency_p95 = np.percentile(inference_times, 95)
                latency_p99 = np.percentile(inference_times, 99)
            else:
                avg_inference_time = total_time_ms / total_requests
                latency_p50 = latency_p95 = latency_p99 = avg_inference_time
            
            # スループット計算
            throughput_qps = (total_requests / total_time_ms) * 1000 if total_time_ms > 0 else 0
            
            # メモリ使用量（簡略化）
            memory_usage_mb = 0.0
            if torch.cuda.is_available():
                memory_usage_mb = torch.cuda.memory_allocated() / 1024 / 1024
            
            # キャッシュヒット率
            cache_hit_rate = 0.0
            if self.performance_stats["total_inferences"] > 0:
                cache_hit_rate = self.performance_stats["cache_hits"] / self.performance_stats["total_inferences"]
            
            return InferenceMetrics(
                inference_time_ms=avg_inference_time,
                throughput_qps=throughput_qps,
                latency_p50_ms=latency_p50,
                latency_p95_ms=latency_p95,
                latency_p99_ms=latency_p99,
                batch_size=total_requests,
                memory_usage_mb=memory_usage_mb,
                cache_hit_rate=cache_hit_rate
            )
            
        except Exception as e:
            logger.error(f"バッチメトリクス計算エラー: {e}")
            return InferenceMetrics()
    
    def _generate_cache_key(self, input_data: Any, model_name: str) -> str:
        """キャッシュキー生成"""
        try:
            if isinstance(input_data, torch.Tensor):
                # テンソルの場合、ハッシュ値を計算
                tensor_hash = hash(input_data.data_ptr())
                return f"{model_name}_{tensor_hash}"
            else:
                # その他の場合、文字列表現のハッシュ
                data_str = str(input_data)
                return f"{model_name}_{hash(data_str)}"
        except Exception as e:
            logger.error(f"キャッシュキー生成エラー: {e}")
            return f"{model_name}_{int(time.time())}"
    
    def _update_cache(self, cache_key: str, result: Any):
        """キャッシュ更新"""
        try:
            # キャッシュサイズ制限
            if len(self.cache) >= self.config.cache_size:
                # 古いエントリを削除
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[cache_key] = result
            
        except Exception as e:
            logger.error(f"キャッシュ更新エラー: {e}")
    
    def _batch_processing_loop(self):
        """バッチ処理ループ"""
        while self.is_processing:
            try:
                # バッチキューからリクエストを取得
                batch_requests = []
                timeout = 0.1  # 100ms
                
                try:
                    # 最初のリクエストを取得
                    first_request = self.batch_queue.get(timeout=timeout)
                    batch_requests.append(first_request)
                    
                    # 追加のリクエストを取得（タイムアウト内で）
                    while len(batch_requests) < self.config.max_batch_size:
                        try:
                            request = self.batch_queue.get(timeout=0.01)  # 10ms
                            batch_requests.append(request)
                        except Empty:
                            break
                
                except Empty:
                    continue
                
                # バッチ処理実行（非同期で実行）
                if batch_requests:
                    asyncio.run(self._process_batch_requests(batch_requests))
                
            except Exception as e:
                logger.error(f"バッチ処理ループエラー: {e}")
                time.sleep(0.1)
    
    async def _process_batch_requests(self, batch_requests: List[BatchRequest]):
        """バッチリクエスト処理"""
        try:
            # デフォルトモデルで処理（実際の実装では、適切なモデルを選択）
            if self.optimized_models:
                model_name = list(self.optimized_models.keys())[0]
                model = self.optimized_models[model_name]
                
                result = await self.batch_inference(model, batch_requests, model_name)
                logger.info(f"バッチ処理完了: {len(batch_requests)}個のリクエスト")
            
        except Exception as e:
            logger.error(f"バッチリクエスト処理エラー: {e}")
    
    def get_inference_history(self) -> List[InferenceMetrics]:
        """推論履歴取得"""
        return self.inference_history.copy()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計取得"""
        stats = self.performance_stats.copy()
        
        # 平均値の計算
        if stats["total_inferences"] > 0:
            stats["avg_inference_time_ms"] = stats["total_processing_time_ms"] / stats["total_inferences"]
            stats["avg_throughput_qps"] = (stats["total_inferences"] / stats["total_processing_time_ms"]) * 1000 if stats["total_processing_time_ms"] > 0 else 0
        
        return stats
    
    def get_optimized_models(self) -> Dict[str, Any]:
        """最適化済みモデル取得"""
        return self.optimized_models.copy()
    
    async def cleanup(self):
        """リソースクリーンアップ"""
        await self.stop_batch_processing()
        self.thread_pool.shutdown(wait=True)
        logger.info("推論速度最適化システムクリーンアップ完了")


# 使用例
async def main():
    """使用例"""
    # 設定
    config = InferenceOptimizationConfig(
        optimization_strategy=OptimizationStrategy.RTX4050_OPTIMIZED,
        batch_strategy=BatchStrategy.ADAPTIVE_BATCHING,
        max_batch_size=16,
        enable_model_compilation=True,
        enable_mixed_precision=True,
        enable_caching=True
    )
    
    # 推論最適化システム初期化
    optimizer = InferenceOptimizer(config)
    
    # サンプルモデル
    class SampleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(100, 50)
        
        def forward(self, x):
            return self.linear(x)
    
    model = SampleModel()
    sample_input = torch.randn(1, 100)
    
    try:
        # モデル最適化
        optimization_result = await optimizer.optimize_model(model, "sample_model", sample_input)
        print(f"最適化結果: {optimization_result['optimization_successful']}")
        
        # 単一推論
        single_result = await optimizer.single_inference(model, sample_input, "sample_model")
        print(f"単一推論時間: {single_result['inference_time_ms']:.2f}ms")
        
        # バッチ推論
        batch_requests = [
            BatchRequest(f"req_{i}", torch.randn(1, 100))
            for i in range(10)
        ]
        
        batch_result = await optimizer.batch_inference(model, batch_requests, "sample_model")
        print(f"バッチ推論: {len(batch_result['batch_results'])}個のバッチ")
        
        # 統計表示
        stats = optimizer.get_performance_stats()
        print(f"パフォーマンス統計: {stats}")
        
    finally:
        await optimizer.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
