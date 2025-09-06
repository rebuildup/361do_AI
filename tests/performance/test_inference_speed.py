"""
Inference Speed Optimization Tests
推論速度最適化テスト
"""

import pytest
import asyncio
import time
import torch
import torch.nn as nn
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

from src.advanced_agent.optimization.inference_optimizer import (
    InferenceOptimizer, InferenceOptimizationConfig, OptimizationStrategy,
    BatchStrategy, BatchRequest, InferenceMetrics
)
from src.advanced_agent.optimization.batch_processor import (
    BatchProcessor, BatchProcessingConfig, BatchProcessingStrategy,
    BatchSizeStrategy, BatchItem, BatchResult
)

logger = logging.getLogger(__name__)


class TestInferenceOptimizer:
    """推論速度最適化システムのテスト"""
    
    @pytest.fixture
    def inference_config(self):
        """テスト用推論最適化設定"""
        return InferenceOptimizationConfig(
            optimization_strategy=OptimizationStrategy.RTX4050_OPTIMIZED,
            batch_strategy=BatchStrategy.ADAPTIVE_BATCHING,
            max_batch_size=16,
            min_batch_size=1,
            target_latency_ms=100.0,
            target_throughput_qps=100.0,
            enable_model_compilation=True,
            enable_mixed_precision=True,
            enable_caching=True,
            cache_size=100
        )
    
    @pytest.fixture
    def inference_optimizer(self, inference_config):
        """テスト用推論最適化システム"""
        return InferenceOptimizer(inference_config)
    
    @pytest.fixture
    def sample_model(self):
        """テスト用サンプルモデル"""
        class SampleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(100, 50)
                self.linear2 = nn.Linear(50, 10)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.linear1(x))
                x = self.linear2(x)
                return x
        
        return SampleModel()
    
    @pytest.fixture
    def sample_input(self):
        """テスト用サンプル入力"""
        return torch.randn(1, 100)
    
    def test_inference_optimizer_initialization(self, inference_optimizer):
        """推論最適化システムの初期化テスト"""
        assert inference_optimizer is not None
        assert inference_optimizer.config is not None
        assert inference_optimizer.config.optimization_strategy == OptimizationStrategy.RTX4050_OPTIMIZED
        assert len(inference_optimizer.inference_history) == 0
        assert len(inference_optimizer.optimized_models) == 0
        assert inference_optimizer.is_processing is False
    
    @pytest.mark.asyncio
    async def test_model_optimization(self, inference_optimizer, sample_model, sample_input):
        """モデル最適化テスト"""
        result = await inference_optimizer.optimize_model(
            sample_model, "test_model", sample_input
        )
        
        assert result["optimization_successful"] is True
        assert "optimized_model" in result
        assert "timestamp" in result
        
        # 最適化済みモデルが保存されていることを確認
        assert "test_model" in inference_optimizer.optimized_models
    
    @pytest.mark.asyncio
    async def test_single_inference(self, inference_optimizer, sample_model, sample_input):
        """単一推論テスト"""
        result = await inference_optimizer.single_inference(
            sample_model, sample_input, "test_model"
        )
        
        assert "result" in result
        assert "inference_time_ms" in result
        assert "timestamp" in result
        assert result["inference_time_ms"] >= 0
        
        # 統計が更新されていることを確認
        stats = inference_optimizer.get_performance_stats()
        assert stats["total_inferences"] > 0
    
    @pytest.mark.asyncio
    async def test_batch_inference(self, inference_optimizer, sample_model):
        """バッチ推論テスト"""
        # バッチリクエスト作成
        batch_requests = [
            BatchRequest(f"req_{i}", torch.randn(1, 100))
            for i in range(10)
        ]
        
        result = await inference_optimizer.batch_inference(
            sample_model, batch_requests, "test_model"
        )
        
        assert "batch_results" in result
        assert "metrics" in result
        assert "total_requests" in result
        assert "total_batches" in result
        assert result["total_requests"] == 10
        
        # メトリクス確認
        metrics = result["metrics"]
        assert isinstance(metrics, InferenceMetrics)
        assert metrics.inference_time_ms >= 0
        assert metrics.throughput_qps >= 0
    
    @pytest.mark.asyncio
    async def test_optimization_strategies(self, sample_model, sample_input):
        """最適化戦略テスト"""
        strategies = [
            OptimizationStrategy.SPEED_FIRST,
            OptimizationStrategy.THROUGHPUT_FIRST,
            OptimizationStrategy.BALANCED,
            OptimizationStrategy.RTX4050_OPTIMIZED
        ]
        
        for strategy in strategies:
            config = InferenceOptimizationConfig(optimization_strategy=strategy)
            optimizer = InferenceOptimizer(config)
            
            result = await optimizer.optimize_model(sample_model, f"test_model_{strategy.value}", sample_input)
            assert result["optimization_successful"] is True
            
            await optimizer.cleanup()
    
    @pytest.mark.asyncio
    async def test_caching_functionality(self, inference_optimizer, sample_model, sample_input):
        """キャッシュ機能テスト"""
        # 最初の推論（キャッシュミス）
        result1 = await inference_optimizer.single_inference(
            sample_model, sample_input, "test_model"
        )
        assert result1["from_cache"] is False
        
        # 2回目の推論（キャッシュヒット）
        result2 = await inference_optimizer.single_inference(
            sample_model, sample_input, "test_model"
        )
        assert result2["from_cache"] is True
        assert result2["inference_time_ms"] == 0.0
        
        # 統計確認
        stats = inference_optimizer.get_performance_stats()
        assert stats["cache_hits"] > 0
        assert stats["cache_misses"] > 0
    
    @pytest.mark.asyncio
    async def test_batch_processing_control(self, inference_optimizer):
        """バッチ処理制御テスト"""
        # バッチ処理開始
        await inference_optimizer.start_batch_processing()
        assert inference_optimizer.is_processing is True
        
        # 少し待機
        await asyncio.sleep(0.1)
        
        # バッチ処理停止
        await inference_optimizer.stop_batch_processing()
        assert inference_optimizer.is_processing is False
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, inference_optimizer, sample_model):
        """パフォーマンスメトリクステスト"""
        # 複数回の推論実行
        for i in range(5):
            input_data = torch.randn(1, 100)
            await inference_optimizer.single_inference(sample_model, input_data, "test_model")
        
        # メトリクス確認
        history = inference_optimizer.get_inference_history()
        assert len(history) > 0
        
        stats = inference_optimizer.get_performance_stats()
        assert stats["total_inferences"] >= 5
        assert stats["avg_inference_time_ms"] >= 0
    
    @pytest.mark.asyncio
    async def test_inference_optimizer_cleanup(self, inference_optimizer):
        """推論最適化システムのクリーンアップテスト"""
        # バッチ処理開始
        await inference_optimizer.start_batch_processing()
        assert inference_optimizer.is_processing is True
        
        # クリーンアップ
        await inference_optimizer.cleanup()
        assert inference_optimizer.is_processing is False


class TestBatchProcessor:
    """バッチ処理最適化システムのテスト"""
    
    @pytest.fixture
    def batch_config(self):
        """テスト用バッチ処理設定"""
        return BatchProcessingConfig(
            strategy=BatchProcessingStrategy.RTX4050_OPTIMIZED,
            batch_size_strategy=BatchSizeStrategy.ADAPTIVE,
            max_batch_size=16,
            min_batch_size=1,
            target_batch_size=8,
            max_workers=4,
            enable_gpu_acceleration=True,
            enable_memory_optimization=True,
            enable_prefetching=True,
            enable_priority_queuing=True,
            max_queue_size=100
        )
    
    @pytest.fixture
    def batch_processor(self, batch_config):
        """テスト用バッチ処理システム"""
        return BatchProcessor(batch_config)
    
    @pytest.fixture
    def sample_processing_function(self):
        """テスト用処理関数"""
        def processing_function(data_list):
            results = []
            for data in data_list:
                if isinstance(data, (int, float)):
                    result = data * 2
                else:
                    result = str(data).upper()
                results.append(result)
            return results
        return processing_function
    
    @pytest.fixture
    def sample_batch_items(self):
        """テスト用バッチアイテム"""
        return [
            BatchItem(f"item_{i}", i, priority=i % 3)
            for i in range(20)
        ]
    
    def test_batch_processor_initialization(self, batch_processor):
        """バッチ処理システムの初期化テスト"""
        assert batch_processor is not None
        assert batch_processor.config is not None
        assert batch_processor.config.strategy == BatchProcessingStrategy.RTX4050_OPTIMIZED
        assert batch_processor.is_processing is False
        assert len(batch_processor.batch_results) == 0
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, batch_processor, sample_batch_items, sample_processing_function):
        """バッチ処理テスト"""
        result = await batch_processor.process_batch(
            sample_batch_items, sample_processing_function, "test_batch"
        )
        
        assert result.batch_id == "test_batch"
        assert len(result.items) == 20
        assert len(result.results) == 20
        assert result.processing_time_ms >= 0
        assert result.success_count >= 0
        assert result.error_count >= 0
        
        # バッチ結果が保存されていることを確認
        assert len(batch_processor.batch_results) > 0
    
    @pytest.mark.asyncio
    async def test_batch_size_strategies(self, sample_batch_items, sample_processing_function):
        """バッチサイズ戦略テスト"""
        strategies = [
            BatchSizeStrategy.FIXED,
            BatchSizeStrategy.DYNAMIC,
            BatchSizeStrategy.ADAPTIVE,
            BatchSizeStrategy.MEMORY_BASED
        ]
        
        for strategy in strategies:
            config = BatchProcessingConfig(batch_size_strategy=strategy)
            processor = BatchProcessor(config)
            
            result = await processor.process_batch(
                sample_batch_items[:10], sample_processing_function, f"test_{strategy.value}"
            )
            
            assert result.batch_id == f"test_{strategy.value}"
            assert len(result.items) == 10
            
            await processor.cleanup()
    
    @pytest.mark.asyncio
    async def test_processing_strategies(self, sample_batch_items, sample_processing_function):
        """処理戦略テスト"""
        strategies = [
            BatchProcessingStrategy.SEQUENTIAL,
            BatchProcessingStrategy.PARALLEL,
            BatchProcessingStrategy.PIPELINE,
            BatchProcessingStrategy.ADAPTIVE,
            BatchProcessingStrategy.RTX4050_OPTIMIZED
        ]
        
        for strategy in strategies:
            config = BatchProcessingConfig(strategy=strategy)
            processor = BatchProcessor(config)
            
            result = await processor.process_batch(
                sample_batch_items[:5], sample_processing_function, f"test_{strategy.value}"
            )
            
            assert result.batch_id == f"test_{strategy.value}"
            assert len(result.items) == 5
            
            await processor.cleanup()
    
    @pytest.mark.asyncio
    async def test_stream_processing(self, batch_processor, sample_processing_function):
        """ストリーム処理テスト"""
        def data_stream():
            for i in range(15):
                yield i
        
        results = []
        async for result in batch_processor.process_stream(data_stream(), sample_processing_function):
            results.append(result)
            assert result.batch_id.startswith("stream_batch_")
            assert len(result.items) > 0
        
        assert len(results) > 0
    
    @pytest.mark.asyncio
    async def test_background_processing(self, batch_processor):
        """バックグラウンド処理テスト"""
        # バックグラウンド処理開始
        await batch_processor.start_background_processing()
        assert batch_processor.is_processing is True
        
        # キューにアイテム追加
        for i in range(5):
            item = BatchItem(f"bg_item_{i}", i)
            batch_processor.add_to_queue(item)
        
        # キューサイズ確認
        assert batch_processor.get_queue_size() == 5
        
        # 少し待機
        await asyncio.sleep(0.5)
        
        # バックグラウンド処理停止
        await batch_processor.stop_background_processing()
        assert batch_processor.is_processing is False
    
    @pytest.mark.asyncio
    async def test_priority_queuing(self, batch_processor):
        """優先度付きキューイングテスト"""
        # 異なる優先度のアイテムを追加
        high_priority_item = BatchItem("high_priority", "high", priority=10)
        low_priority_item = BatchItem("low_priority", "low", priority=1)
        medium_priority_item = BatchItem("medium_priority", "medium", priority=5)
        
        batch_processor.add_to_queue(low_priority_item)
        batch_processor.add_to_queue(high_priority_item)
        batch_processor.add_to_queue(medium_priority_item)
        
        # キューサイズ確認
        assert batch_processor.get_queue_size() == 3
    
    def test_processing_metrics(self, batch_processor):
        """処理メトリクステスト"""
        metrics = batch_processor.get_processing_metrics()
        assert isinstance(metrics, BatchProcessingMetrics)
        assert metrics.total_items_processed == 0  # 初期状態
        
        stats = batch_processor.get_processing_stats()
        assert isinstance(stats, dict)
        assert "total_items" in stats
        assert "total_batches" in stats
    
    @pytest.mark.asyncio
    async def test_batch_processor_cleanup(self, batch_processor):
        """バッチ処理システムのクリーンアップテスト"""
        # バックグラウンド処理開始
        await batch_processor.start_background_processing()
        assert batch_processor.is_processing is True
        
        # クリーンアップ
        await batch_processor.cleanup()
        assert batch_processor.is_processing is False


class TestInferenceSpeedIntegration:
    """推論速度最適化統合テスト"""
    
    @pytest.mark.asyncio
    async def test_inference_and_batch_integration(self):
        """推論最適化とバッチ処理の統合テスト"""
        # 推論最適化システム
        inference_config = InferenceOptimizationConfig(
            optimization_strategy=OptimizationStrategy.RTX4050_OPTIMIZED
        )
        inference_optimizer = InferenceOptimizer(inference_config)
        
        # バッチ処理システム
        batch_config = BatchProcessingConfig(
            strategy=BatchProcessingStrategy.RTX4050_OPTIMIZED
        )
        batch_processor = BatchProcessor(batch_config)
        
        # サンプルモデル
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 50)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        try:
            # モデル最適化
            optimization_result = await inference_optimizer.optimize_model(
                model, "integration_test_model"
            )
            assert optimization_result["optimization_successful"] is True
            
            # バッチアイテム作成
            batch_items = [
                BatchItem(f"item_{i}", torch.randn(1, 100))
                for i in range(10)
            ]
            
            # バッチ処理関数
            def model_processing_function(data_list):
                results = []
                for data in data_list:
                    with torch.no_grad():
                        result = model(data)
                    results.append(result)
                return results
            
            # バッチ処理実行
            batch_result = await batch_processor.process_batch(
                batch_items, model_processing_function, "integration_batch"
            )
            
            assert batch_result.batch_id == "integration_batch"
            assert len(batch_result.items) == 10
            assert batch_result.success_count >= 0
            
        finally:
            # クリーンアップ
            await inference_optimizer.cleanup()
            await batch_processor.cleanup()
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """負荷下でのパフォーマンステスト"""
        config = InferenceOptimizationConfig(
            optimization_strategy=OptimizationStrategy.SPEED_FIRST,
            max_batch_size=32
        )
        optimizer = InferenceOptimizer(config)
        
        # サンプルモデル
        model = nn.Linear(100, 50)
        
        try:
            # モデル最適化
            await optimizer.optimize_model(model, "load_test_model")
            
            # 大量の推論実行
            start_time = time.time()
            
            for i in range(50):
                input_data = torch.randn(1, 100)
                result = await optimizer.single_inference(model, input_data, "load_test_model")
                assert result["inference_time_ms"] >= 0
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # パフォーマンス確認
            stats = optimizer.get_performance_stats()
            assert stats["total_inferences"] == 50
            assert total_time < 10.0  # 10秒以内に完了
            
        finally:
            await optimizer.cleanup()


if __name__ == "__main__":
    # 直接実行時のテスト
    async def main():
        print("⚡ 推論速度最適化テスト実行中...")
        
        # 推論最適化システムテスト
        inference_config = InferenceOptimizationConfig()
        inference_optimizer = InferenceOptimizer(inference_config)
        
        # サンプルモデル
        model = nn.Linear(100, 50)
        sample_input = torch.randn(1, 100)
        
        print("モデル最適化テスト...")
        optimization_result = await inference_optimizer.optimize_model(model, "test_model", sample_input)
        print(f"最適化結果: {optimization_result['optimization_successful']}")
        
        print("単一推論テスト...")
        single_result = await inference_optimizer.single_inference(model, sample_input, "test_model")
        print(f"推論時間: {single_result['inference_time_ms']:.2f}ms")
        
        print("バッチ推論テスト...")
        batch_requests = [BatchRequest(f"req_{i}", torch.randn(1, 100)) for i in range(5)]
        batch_result = await inference_optimizer.batch_inference(model, batch_requests, "test_model")
        print(f"バッチ結果: {len(batch_result['batch_results'])}個のバッチ")
        
        print("統計情報...")
        stats = inference_optimizer.get_performance_stats()
        print(f"推論統計: {stats}")
        
        await inference_optimizer.cleanup()
        print("✅ 推論速度最適化テスト完了")
    
    asyncio.run(main())
