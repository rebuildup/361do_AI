"""
Batch Processing Optimization
バッチ処理最適化システム

大量のデータを効率的にバッチ処理するための最適化システムです。
RTX 4050 6GB VRAM環境での効率的なバッチ処理を実装します。
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable, Union, Iterator
import torch
import torch.nn as nn
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from queue import Queue, Empty, PriorityQueue
import psutil

logger = logging.getLogger(__name__)


class BatchProcessingStrategy(Enum):
    """バッチ処理戦略"""
    SEQUENTIAL = "sequential"          # 逐次処理
    PARALLEL = "parallel"              # 並列処理
    PIPELINE = "pipeline"              # パイプライン処理
    ADAPTIVE = "adaptive"              # 適応的処理
    RTX4050_OPTIMIZED = "rtx4050_optimized"  # RTX 4050専用


class BatchSizeStrategy(Enum):
    """バッチサイズ戦略"""
    FIXED = "fixed"                    # 固定サイズ
    DYNAMIC = "dynamic"                # 動的サイズ
    ADAPTIVE = "adaptive"              # 適応的サイズ
    MEMORY_BASED = "memory_based"      # メモリベース


@dataclass
class BatchProcessingConfig:
    """バッチ処理設定"""
    strategy: BatchProcessingStrategy = BatchProcessingStrategy.RTX4050_OPTIMIZED
    batch_size_strategy: BatchSizeStrategy = BatchSizeStrategy.ADAPTIVE
    max_batch_size: int = 32
    min_batch_size: int = 1
    target_batch_size: int = 16
    max_workers: int = 4
    enable_gpu_acceleration: bool = True
    enable_memory_optimization: bool = True
    enable_prefetching: bool = True
    prefetch_buffer_size: int = 10
    enable_priority_queuing: bool = True
    max_queue_size: int = 1000
    processing_timeout_seconds: int = 300
    memory_threshold_percent: float = 80.0


@dataclass
class BatchItem:
    """バッチアイテム"""
    item_id: str
    data: Any
    priority: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """バッチ結果"""
    batch_id: str
    items: List[BatchItem]
    results: List[Any]
    processing_time_ms: float
    memory_usage_mb: float
    success_count: int
    error_count: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BatchProcessingMetrics:
    """バッチ処理メトリクス"""
    total_items_processed: int = 0
    total_batches_processed: int = 0
    total_processing_time_ms: float = 0.0
    average_batch_size: float = 0.0
    average_processing_time_ms: float = 0.0
    throughput_items_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_utilization_percent: float = 0.0
    cpu_utilization_percent: float = 0.0
    error_rate_percent: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class BatchProcessor:
    """バッチ処理最適化システム"""
    
    def __init__(self, config: Optional[BatchProcessingConfig] = None):
        """
        初期化
        
        Args:
            config: バッチ処理設定
        """
        self.config = config or BatchProcessingConfig()
        self.processing_queue: PriorityQueue = PriorityQueue(maxsize=self.config.max_queue_size)
        self.result_queue: Queue = Queue()
        self.is_processing = False
        self.processing_threads: List[threading.Thread] = []
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # メトリクス
        self.metrics = BatchProcessingMetrics()
        self.batch_results: List[BatchResult] = []
        
        # 統計
        self.processing_stats = {
            "total_items": 0,
            "total_batches": 0,
            "successful_items": 0,
            "failed_items": 0,
            "total_processing_time_ms": 0.0,
            "peak_memory_usage_mb": 0.0,
            "average_batch_size": 0.0
        }
        
        logger.info(f"バッチ処理最適化システム初期化完了 - 戦略: {self.config.strategy.value}")
    
    async def process_batch(
        self,
        items: List[BatchItem],
        processing_function: Callable[[List[Any]], List[Any]],
        batch_id: Optional[str] = None
    ) -> BatchResult:
        """
        バッチ処理実行
        
        Args:
            items: 処理対象アイテム
            processing_function: 処理関数
            batch_id: バッチID
            
        Returns:
            バッチ処理結果
        """
        try:
            if not items:
                return BatchResult(
                    batch_id=batch_id or "empty",
                    items=[],
                    results=[],
                    processing_time_ms=0.0,
                    memory_usage_mb=0.0,
                    success_count=0,
                    error_count=0
                )
            
            logger.info(f"バッチ処理開始: {len(items)}個のアイテム")
            
            start_time = time.time()
            
            # バッチサイズの決定
            optimal_batch_size = await self._determine_optimal_batch_size(items)
            
            # バッチに分割
            batches = self._create_batches(items, optimal_batch_size)
            
            # バッチ処理実行
            batch_results = []
            for i, batch in enumerate(batches):
                batch_result = await self._process_single_batch(
                    batch, processing_function, f"{batch_id}_batch_{i}"
                )
                batch_results.append(batch_result)
            
            end_time = time.time()
            total_processing_time = (end_time - start_time) * 1000
            
            # 結果を統合
            all_results = []
            all_items = []
            success_count = 0
            error_count = 0
            
            for batch_result in batch_results:
                all_results.extend(batch_result.results)
                all_items.extend(batch_result.items)
                success_count += batch_result.success_count
                error_count += batch_result.error_count
            
            # 最終結果作成
            final_result = BatchResult(
                batch_id=batch_id or f"batch_{int(time.time())}",
                items=all_items,
                results=all_results,
                processing_time_ms=total_processing_time,
                memory_usage_mb=self._get_current_memory_usage(),
                success_count=success_count,
                error_count=error_count
            )
            
            # メトリクス更新
            self._update_metrics(final_result)
            
            logger.info(f"バッチ処理完了: {len(items)}個のアイテム, {total_processing_time:.1f}ms")
            
            return final_result
            
        except Exception as e:
            logger.error(f"バッチ処理エラー: {e}")
            return BatchResult(
                batch_id=batch_id or "error",
                items=items,
                results=[],
                processing_time_ms=0.0,
                memory_usage_mb=0.0,
                success_count=0,
                error_count=len(items)
            )
    
    async def process_stream(
        self,
        data_stream: Iterator[Any],
        processing_function: Callable[[List[Any]], List[Any]],
        batch_size: Optional[int] = None
    ) -> Iterator[BatchResult]:
        """
        ストリーム処理
        
        Args:
            data_stream: データストリーム
            processing_function: 処理関数
            batch_size: バッチサイズ
            
        Yields:
            バッチ処理結果
        """
        try:
            logger.info("ストリーム処理開始")
            
            current_batch = []
            batch_id_counter = 0
            
            for item_data in data_stream:
                # アイテム作成
                item = BatchItem(
                    item_id=f"stream_item_{batch_id_counter}",
                    data=item_data
                )
                current_batch.append(item)
                
                # バッチサイズに達したら処理
                target_size = batch_size or await self._determine_optimal_batch_size(current_batch)
                if len(current_batch) >= target_size:
                    batch_result = await self.process_batch(
                        current_batch, processing_function, f"stream_batch_{batch_id_counter}"
                    )
                    yield batch_result
                    
                    current_batch = []
                    batch_id_counter += 1
            
            # 残りのアイテムを処理
            if current_batch:
                batch_result = await self.process_batch(
                    current_batch, processing_function, f"stream_batch_{batch_id_counter}"
                )
                yield batch_result
            
            logger.info("ストリーム処理完了")
            
        except Exception as e:
            logger.error(f"ストリーム処理エラー: {e}")
    
    async def start_background_processing(self):
        """バックグラウンド処理開始"""
        if self.is_processing:
            logger.warning("バックグラウンド処理は既に開始されています")
            return
        
        self.is_processing = True
        
        # 処理スレッドを開始
        for i in range(self.config.max_workers):
            thread = threading.Thread(
                target=self._background_processing_loop,
                name=f"BatchProcessor-{i}"
            )
            thread.start()
            self.processing_threads.append(thread)
        
        logger.info(f"バックグラウンド処理開始: {self.config.max_workers}個のスレッド")
    
    async def stop_background_processing(self):
        """バックグラウンド処理停止"""
        if not self.is_processing:
            return
        
        self.is_processing = False
        
        # スレッドの停止を待機
        for thread in self.processing_threads:
            thread.join(timeout=5.0)
        
        self.processing_threads.clear()
        
        logger.info("バックグラウンド処理停止")
    
    def add_to_queue(self, item: BatchItem):
        """キューにアイテム追加"""
        try:
            if self.config.enable_priority_queuing:
                # 優先度付きキュー
                priority = -item.priority  # 高い優先度を先に処理
                self.processing_queue.put((priority, item))
            else:
                self.processing_queue.put(item)
            
            self.processing_stats["total_items"] += 1
            
        except Exception as e:
            logger.error(f"キュー追加エラー: {e}")
    
    def get_queue_size(self) -> int:
        """キューサイズ取得"""
        return self.processing_queue.qsize()
    
    async def _determine_optimal_batch_size(self, items: List[BatchItem]) -> int:
        """最適バッチサイズ決定"""
        try:
            if not items:
                return self.config.min_batch_size
            
            item_count = len(items)
            
            if self.config.batch_size_strategy == BatchSizeStrategy.FIXED:
                return min(self.config.target_batch_size, item_count)
            
            elif self.config.batch_size_strategy == BatchSizeStrategy.DYNAMIC:
                # 動的バッチサイズ決定
                if item_count <= 4:
                    return 1
                elif item_count <= 16:
                    return 4
                elif item_count <= 64:
                    return 16
                else:
                    return min(32, item_count)
            
            elif self.config.batch_size_strategy == BatchSizeStrategy.MEMORY_BASED:
                # メモリベースのバッチサイズ決定
                memory_usage = self._get_current_memory_usage()
                memory_percent = (memory_usage / psutil.virtual_memory().total) * 100
                
                if memory_percent > self.config.memory_threshold_percent:
                    return self.config.min_batch_size
                elif memory_percent > self.config.memory_threshold_percent * 0.7:
                    return min(8, item_count)
                else:
                    return min(self.config.max_batch_size, item_count)
            
            else:  # ADAPTIVE
                # 適応的バッチサイズ決定
                # 過去のパフォーマンスに基づく決定
                if self.batch_results:
                    recent_results = self.batch_results[-10:]  # 最近の10個の結果
                    avg_batch_size = sum(r.items.__len__() for r in recent_results) / len(recent_results)
                    avg_processing_time = sum(r.processing_time_ms for r in recent_results) / len(recent_results)
                    
                    # 処理時間が短い場合はバッチサイズを増やす
                    if avg_processing_time < 100:  # 100ms未満
                        return min(int(avg_batch_size * 1.5), self.config.max_batch_size, item_count)
                    else:
                        return min(int(avg_batch_size * 0.8), self.config.max_batch_size, item_count)
                else:
                    return min(self.config.target_batch_size, item_count)
            
        except Exception as e:
            logger.error(f"最適バッチサイズ決定エラー: {e}")
            return min(self.config.target_batch_size, len(items))
    
    def _create_batches(self, items: List[BatchItem], batch_size: int) -> List[List[BatchItem]]:
        """バッチ作成"""
        batches = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batches.append(batch)
        return batches
    
    async def _process_single_batch(
        self,
        batch: List[BatchItem],
        processing_function: Callable[[List[Any]], List[Any]],
        batch_id: str
    ) -> BatchResult:
        """単一バッチ処理"""
        try:
            start_time = time.time()
            
            # データの準備
            batch_data = [item.data for item in batch]
            
            # 処理実行
            if self.config.strategy == BatchProcessingStrategy.PARALLEL:
                # 並列処理
                results = await self._parallel_processing(batch_data, processing_function)
            elif self.config.strategy == BatchProcessingStrategy.PIPELINE:
                # パイプライン処理
                results = await self._pipeline_processing(batch_data, processing_function)
            else:
                # 逐次処理
                results = processing_function(batch_data)
            
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000
            
            # 結果の検証
            success_count = len([r for r in results if r is not None])
            error_count = len(batch) - success_count
            
            result = BatchResult(
                batch_id=batch_id,
                items=batch,
                results=results,
                processing_time_ms=processing_time,
                memory_usage_mb=self._get_current_memory_usage(),
                success_count=success_count,
                error_count=error_count
            )
            
            return result
            
        except Exception as e:
            logger.error(f"単一バッチ処理エラー: {e}")
            return BatchResult(
                batch_id=batch_id,
                items=batch,
                results=[],
                processing_time_ms=0.0,
                memory_usage_mb=0.0,
                success_count=0,
                error_count=len(batch)
            )
    
    async def _parallel_processing(
        self,
        batch_data: List[Any],
        processing_function: Callable[[List[Any]], List[Any]]
    ) -> List[Any]:
        """並列処理"""
        try:
            # データを分割
            chunk_size = max(1, len(batch_data) // self.config.max_workers)
            chunks = [batch_data[i:i + chunk_size] for i in range(0, len(batch_data), chunk_size)]
            
            # 並列実行
            loop = asyncio.get_event_loop()
            tasks = []
            for chunk in chunks:
                task = loop.run_in_executor(self.thread_pool, processing_function, chunk)
                tasks.append(task)
            
            # 結果を待機
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 結果を統合
            results = []
            for chunk_result in chunk_results:
                if isinstance(chunk_result, Exception):
                    logger.error(f"並列処理エラー: {chunk_result}")
                    results.extend([None] * len(chunk_result))
                else:
                    results.extend(chunk_result)
            
            return results
            
        except Exception as e:
            logger.error(f"並列処理エラー: {e}")
            return [None] * len(batch_data)
    
    async def _pipeline_processing(
        self,
        batch_data: List[Any],
        processing_function: Callable[[List[Any]], List[Any]]
    ) -> List[Any]:
        """パイプライン処理"""
        try:
            # パイプライン処理（簡略化実装）
            # 実際の実装では、より複雑なパイプライン処理が必要
            
            results = []
            for i in range(0, len(batch_data), self.config.max_workers):
                chunk = batch_data[i:i + self.config.max_workers]
                chunk_results = processing_function(chunk)
                results.extend(chunk_results)
            
            return results
            
        except Exception as e:
            logger.error(f"パイプライン処理エラー: {e}")
            return [None] * len(batch_data)
    
    def _get_current_memory_usage(self) -> float:
        """現在のメモリ使用量取得（MB）"""
        try:
            memory_info = psutil.virtual_memory()
            return memory_info.used / 1024 / 1024
        except Exception as e:
            logger.error(f"メモリ使用量取得エラー: {e}")
            return 0.0
    
    def _update_metrics(self, batch_result: BatchResult):
        """メトリクス更新"""
        try:
            self.batch_results.append(batch_result)
            
            # 統計更新
            self.processing_stats["total_batches"] += 1
            self.processing_stats["successful_items"] += batch_result.success_count
            self.processing_stats["failed_items"] += batch_result.error_count
            self.processing_stats["total_processing_time_ms"] += batch_result.processing_time_ms
            
            # ピークメモリ使用量更新
            if batch_result.memory_usage_mb > self.processing_stats["peak_memory_usage_mb"]:
                self.processing_stats["peak_memory_usage_mb"] = batch_result.memory_usage_mb
            
            # 平均バッチサイズ更新
            total_items = self.processing_stats["successful_items"] + self.processing_stats["failed_items"]
            if total_items > 0:
                self.processing_stats["average_batch_size"] = total_items / self.processing_stats["total_batches"]
            
            # メトリクス更新
            self.metrics.total_items_processed = total_items
            self.metrics.total_batches_processed = self.processing_stats["total_batches"]
            self.metrics.total_processing_time_ms = self.processing_stats["total_processing_time_ms"]
            self.metrics.average_batch_size = self.processing_stats["average_batch_size"]
            self.metrics.memory_usage_mb = batch_result.memory_usage_mb
            
            if self.metrics.total_processing_time_ms > 0:
                self.metrics.throughput_items_per_second = (total_items / self.metrics.total_processing_time_ms) * 1000
            
            if total_items > 0:
                self.metrics.error_rate_percent = (self.processing_stats["failed_items"] / total_items) * 100
            
        except Exception as e:
            logger.error(f"メトリクス更新エラー: {e}")
    
    def _background_processing_loop(self):
        """バックグラウンド処理ループ"""
        while self.is_processing:
            try:
                # キューからアイテムを取得
                try:
                    if self.config.enable_priority_queuing:
                        priority, item = self.processing_queue.get(timeout=1.0)
                    else:
                        item = self.processing_queue.get(timeout=1.0)
                    
                    # バッチ処理（簡略化）
                    # 実際の実装では、適切なバッチ処理関数を呼び出す
                    logger.debug(f"バックグラウンド処理: {item.item_id}")
                    
                except Empty:
                    continue
                
            except Exception as e:
                logger.error(f"バックグラウンド処理ループエラー: {e}")
                time.sleep(0.1)
    
    def get_processing_metrics(self) -> BatchProcessingMetrics:
        """処理メトリクス取得"""
        return self.metrics
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """処理統計取得"""
        return self.processing_stats.copy()
    
    def get_batch_results(self) -> List[BatchResult]:
        """バッチ結果取得"""
        return self.batch_results.copy()
    
    async def cleanup(self):
        """リソースクリーンアップ"""
        await self.stop_background_processing()
        self.thread_pool.shutdown(wait=True)
        logger.info("バッチ処理最適化システムクリーンアップ完了")


# 使用例
async def main():
    """使用例"""
    # 設定
    config = BatchProcessingConfig(
        strategy=BatchProcessingStrategy.RTX4050_OPTIMIZED,
        batch_size_strategy=BatchSizeStrategy.ADAPTIVE,
        max_batch_size=16,
        max_workers=4,
        enable_gpu_acceleration=True,
        enable_memory_optimization=True
    )
    
    # バッチ処理システム初期化
    processor = BatchProcessor(config)
    
    # サンプル処理関数
    def sample_processing_function(data_list):
        """サンプル処理関数"""
        results = []
        for data in data_list:
            # 簡単な処理（実際の実装では、より複雑な処理）
            if isinstance(data, (int, float)):
                result = data * 2
            else:
                result = str(data).upper()
            results.append(result)
        return results
    
    try:
        # サンプルデータ
        sample_items = [
            BatchItem(f"item_{i}", i, priority=i % 3)
            for i in range(100)
        ]
        
        # バッチ処理実行
        result = await processor.process_batch(
            sample_items, sample_processing_function, "sample_batch"
        )
        
        print(f"バッチ処理結果:")
        print(f"  処理アイテム数: {len(result.items)}")
        print(f"  成功数: {result.success_count}")
        print(f"  失敗数: {result.error_count}")
        print(f"  処理時間: {result.processing_time_ms:.1f}ms")
        print(f"  メモリ使用量: {result.memory_usage_mb:.1f}MB")
        
        # ストリーム処理
        def data_stream():
            for i in range(50):
                yield i
        
        print("\nストリーム処理開始...")
        async for stream_result in processor.process_stream(data_stream(), sample_processing_function):
            print(f"ストリームバッチ: {len(stream_result.items)}個のアイテム")
        
        # 統計表示
        metrics = processor.get_processing_metrics()
        stats = processor.get_processing_stats()
        
        print(f"\n処理メトリクス:")
        print(f"  総処理アイテム数: {metrics.total_items_processed}")
        print(f"  総バッチ数: {metrics.total_batches_processed}")
        print(f"  平均バッチサイズ: {metrics.average_batch_size:.1f}")
        print(f"  スループット: {metrics.throughput_items_per_second:.1f} items/sec")
        print(f"  エラー率: {metrics.error_rate_percent:.1f}%")
        
    finally:
        await processor.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
