"""
Memory Optimization Tests
メモリ最適化テスト
"""

import pytest
import asyncio
import time
import psutil
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

from src.advanced_agent.optimization.memory_manager import (
    MemoryOptimizationManager, MemoryOptimizationConfig, OptimizationStrategy,
    MemoryType, MemoryMetrics
)
from src.advanced_agent.optimization.quantization import (
    QuantizationOptimizer, QuantizationConfig, QuantizationType,
    QuantizationStrategy, QuantizationMetrics
)

logger = logging.getLogger(__name__)


class TestMemoryOptimizationManager:
    """メモリ最適化管理システムのテスト"""
    
    @pytest.fixture
    def memory_config(self):
        """テスト用メモリ最適化設定"""
        return MemoryOptimizationConfig(
            max_gpu_memory_usage_percent=80.0,
            max_system_memory_usage_percent=75.0,
            cache_cleanup_threshold_mb=500.0,
            model_offload_threshold_mb=1000.0,
            garbage_collection_interval_seconds=10,
            memory_monitoring_interval_seconds=2,
            optimization_strategy=OptimizationStrategy.RTX4050_OPTIMIZED,
            enable_automatic_cleanup=True,
            enable_model_offloading=True,
            enable_cache_optimization=True
        )
    
    @pytest.fixture
    def memory_manager(self, memory_config):
        """テスト用メモリ最適化管理システム"""
        return MemoryOptimizationManager(memory_config)
    
    @pytest.mark.asyncio
    async def test_memory_manager_initialization(self, memory_manager):
        """メモリ最適化管理システムの初期化テスト"""
        assert memory_manager is not None
        assert memory_manager.config is not None
        assert memory_manager.config.optimization_strategy == OptimizationStrategy.RTX4050_OPTIMIZED
        assert memory_manager.is_monitoring is False
        assert len(memory_manager.memory_metrics) == 0
    
    @pytest.mark.asyncio
    async def test_memory_metrics_collection(self, memory_manager):
        """メモリメトリクス収集テスト"""
        metrics = await memory_manager.get_memory_metrics()
        
        assert isinstance(metrics, MemoryMetrics)
        assert metrics.gpu_memory_used_mb >= 0
        assert metrics.gpu_memory_total_mb >= 0
        assert metrics.system_memory_used_mb >= 0
        assert metrics.system_memory_total_mb >= 0
        assert metrics.timestamp is not None
        
        # 履歴に追加されていることを確認
        assert len(memory_manager.memory_metrics) == 1
    
    @pytest.mark.asyncio
    async def test_memory_monitoring(self, memory_manager):
        """メモリ監視テスト"""
        # 監視開始
        await memory_manager.start_monitoring()
        assert memory_manager.is_monitoring is True
        
        # 少し待機してメトリクスが収集されることを確認
        await asyncio.sleep(3)
        
        # 監視停止
        await memory_manager.stop_monitoring()
        assert memory_manager.is_monitoring is False
        
        # メトリクスが収集されていることを確認
        assert len(memory_manager.memory_metrics) > 0
    
    @pytest.mark.asyncio
    async def test_memory_optimization(self, memory_manager):
        """メモリ最適化テスト"""
        # 強制的に最適化を実行
        result = await memory_manager.optimize_memory(force=True)
        
        assert result["optimization_performed"] is True
        assert "timestamp" in result
        assert "initial_metrics" in result
        assert "final_metrics" in result
        assert "optimizations" in result
        assert "memory_freed" in result
    
    @pytest.mark.asyncio
    async def test_cache_cleanup(self, memory_manager):
        """キャッシュクリーンアップテスト"""
        result = await memory_manager.cleanup_cache()
        
        assert result["cache_cleanup_performed"] is True
        assert "cache_freed_mb" in result
        assert "timestamp" in result
        
        # 統計が更新されていることを確認
        stats = memory_manager.get_optimization_stats()
        assert stats["cache_cleanup_count"] > 0
        assert stats["garbage_collection_count"] > 0
    
    @pytest.mark.asyncio
    async def test_model_offloading(self, memory_manager):
        """モデルオフロードテスト"""
        result = await memory_manager.offload_models()
        
        assert result["model_offload_performed"] is True
        assert "memory_freed_mb" in result
        assert "timestamp" in result
        
        # 統計が更新されていることを確認
        stats = memory_manager.get_optimization_stats()
        assert stats["offload_count"] > 0
    
    @pytest.mark.asyncio
    async def test_optimization_strategies(self):
        """最適化戦略テスト"""
        strategies = [
            OptimizationStrategy.AGGRESSIVE,
            OptimizationStrategy.BALANCED,
            OptimizationStrategy.CONSERVATIVE,
            OptimizationStrategy.RTX4050_OPTIMIZED
        ]
        
        for strategy in strategies:
            config = MemoryOptimizationConfig(optimization_strategy=strategy)
            manager = MemoryOptimizationManager(config)
            
            result = await manager.optimize_memory(force=True)
            assert result["optimization_performed"] is True
            
            await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_optimization_callbacks(self, memory_manager):
        """最適化コールバックテスト"""
        callback_results = []
        
        async def test_callback(metrics: MemoryMetrics):
            callback_results.append(metrics)
        
        memory_manager.add_optimization_callback(test_callback)
        
        # 監視開始
        await memory_manager.start_monitoring()
        
        # 少し待機
        await asyncio.sleep(3)
        
        # 監視停止
        await memory_manager.stop_monitoring()
        
        # コールバックが呼ばれていることを確認
        assert len(callback_results) > 0
    
    @pytest.mark.asyncio
    async def test_optimization_stats(self, memory_manager):
        """最適化統計テスト"""
        # 初期統計
        initial_stats = memory_manager.get_optimization_stats()
        assert initial_stats["cleanup_count"] == 0
        assert initial_stats["total_memory_freed_mb"] == 0.0
        
        # 最適化実行
        await memory_manager.optimize_memory(force=True)
        
        # 統計更新確認
        updated_stats = memory_manager.get_optimization_stats()
        assert updated_stats["cleanup_count"] > 0
    
    @pytest.mark.asyncio
    async def test_memory_manager_cleanup(self, memory_manager):
        """メモリ最適化管理システムのクリーンアップテスト"""
        # 監視開始
        await memory_manager.start_monitoring()
        assert memory_manager.is_monitoring is True
        
        # クリーンアップ
        await memory_manager.cleanup()
        assert memory_manager.is_monitoring is False


class TestQuantizationOptimizer:
    """量子化最適化システムのテスト"""
    
    @pytest.fixture
    def quantization_config(self):
        """テスト用量子化設定"""
        return QuantizationConfig(
            quantization_type=QuantizationType.DYNAMIC,
            strategy=QuantizationStrategy.RTX4050_OPTIMIZED,
            target_precision="int8",
            calibration_samples=10,
            enable_fusion=True,
            preserve_accuracy_threshold=0.9,
            memory_reduction_target=0.4,
            speed_improvement_target=1.3
        )
    
    @pytest.fixture
    def quantization_optimizer(self, quantization_config):
        """テスト用量子化最適化システム"""
        return QuantizationOptimizer(quantization_config)
    
    @pytest.fixture
    def sample_model(self):
        """テスト用サンプルモデル"""
        import torch.nn as nn
        
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
    def sample_test_data(self):
        """テスト用サンプルデータ"""
        import torch
        
        return [
            (torch.randn(1, 100), torch.randint(0, 10, (1,)))
            for _ in range(50)
        ]
    
    def test_quantization_optimizer_initialization(self, quantization_optimizer):
        """量子化最適化システムの初期化テスト"""
        assert quantization_optimizer is not None
        assert quantization_optimizer.config is not None
        assert quantization_optimizer.config.quantization_type == QuantizationType.DYNAMIC
        assert len(quantization_optimizer.quantization_history) == 0
        assert len(quantization_optimizer.quantized_models) == 0
    
    @pytest.mark.asyncio
    async def test_dynamic_quantization(self, quantization_optimizer, sample_model, sample_test_data):
        """動的量子化テスト"""
        result = await quantization_optimizer.quantize_model(
            sample_model, "test_model", test_data=sample_test_data
        )
        
        assert result["quantization_successful"] is True
        assert "quantized_model" in result
        assert "metrics" in result
        assert "timestamp" in result
        
        metrics = result["metrics"]
        assert isinstance(metrics, QuantizationMetrics)
        assert metrics.original_model_size_mb > 0
        assert metrics.quantized_model_size_mb >= 0
    
    @pytest.mark.asyncio
    async def test_quantization_types(self, sample_model, sample_test_data):
        """量子化タイプテスト"""
        quantization_types = [
            QuantizationType.DYNAMIC,
            QuantizationType.INT8,
            QuantizationType.FP16
        ]
        
        for qtype in quantization_types:
            config = QuantizationConfig(quantization_type=qtype)
            optimizer = QuantizationOptimizer(config)
            
            result = await optimizer.quantize_model(
                sample_model, f"test_model_{qtype.value}", test_data=sample_test_data
            )
            
            # 量子化が成功するか、エラーが適切に処理されることを確認
            assert "quantization_successful" in result
            assert "timestamp" in result
    
    @pytest.mark.asyncio
    async def test_rtx4050_optimization(self, quantization_optimizer, sample_model, sample_test_data):
        """RTX 4050専用最適化テスト"""
        result = await quantization_optimizer.optimize_for_rtx4050(
            sample_model, "test_model_rtx", test_data=sample_test_data
        )
        
        assert "optimization_successful" in result
        assert "all_results" in result
        assert "timestamp" in result
        
        if result["optimization_successful"]:
            assert "best_result" in result
            assert "best_score" in result
            assert result["best_score"] > 0
    
    @pytest.mark.asyncio
    async def test_model_metrics_calculation(self, quantization_optimizer, sample_model, sample_test_data):
        """モデルメトリクス計算テスト"""
        metrics = await quantization_optimizer._get_model_metrics(sample_model, sample_test_data)
        
        assert "model_size_mb" in metrics
        assert "inference_time_ms" in metrics
        assert "accuracy" in metrics
        
        assert metrics["model_size_mb"] > 0
        assert metrics["inference_time_ms"] >= 0
        assert 0 <= metrics["accuracy"] <= 1
    
    def test_quantization_history(self, quantization_optimizer):
        """量子化履歴テスト"""
        history = quantization_optimizer.get_quantization_history()
        assert isinstance(history, list)
        assert len(history) == 0  # 初期状態
    
    def test_quantized_models(self, quantization_optimizer):
        """量子化済みモデルテスト"""
        models = quantization_optimizer.get_quantized_models()
        assert isinstance(models, dict)
        assert len(models) == 0  # 初期状態
    
    def test_optimization_summary(self, quantization_optimizer):
        """最適化サマリーテスト"""
        summary = quantization_optimizer.get_optimization_summary()
        assert isinstance(summary, dict)
        assert "message" in summary  # 初期状態では履歴がない


class TestMemoryOptimizationIntegration:
    """メモリ最適化統合テスト"""
    
    @pytest.mark.asyncio
    async def test_memory_and_quantization_integration(self):
        """メモリ最適化と量子化の統合テスト"""
        # メモリ最適化管理システム
        memory_config = MemoryOptimizationConfig(
            optimization_strategy=OptimizationStrategy.RTX4050_OPTIMIZED
        )
        memory_manager = MemoryOptimizationManager(memory_config)
        
        # 量子化最適化システム
        quantization_config = QuantizationConfig(
            strategy=QuantizationStrategy.RTX4050_OPTIMIZED
        )
        quantization_optimizer = QuantizationOptimizer(quantization_config)
        
        try:
            # メモリ監視開始
            await memory_manager.start_monitoring()
            
            # 初期メモリ状態取得
            initial_metrics = await memory_manager.get_memory_metrics()
            
            # サンプルモデルとデータ
            import torch.nn as nn
            import torch
            
            class TestModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(100, 50)
                
                def forward(self, x):
                    return self.linear(x)
            
            model = TestModel()
            test_data = [(torch.randn(1, 100), torch.randn(1, 50)) for _ in range(20)]
            
            # 量子化実行
            quantization_result = await quantization_optimizer.quantize_model(
                model, "integration_test_model", test_data=test_data
            )
            
            # メモリ最適化実行
            memory_result = await memory_manager.optimize_memory(force=True)
            
            # 最終メモリ状態取得
            final_metrics = await memory_manager.get_memory_metrics()
            
            # 結果検証
            assert quantization_result["quantization_successful"] is True
            assert memory_result["optimization_performed"] is True
            
            # メモリ使用量の変化を確認
            memory_change = final_metrics.system_memory_used_mb - initial_metrics.system_memory_used_mb
            assert isinstance(memory_change, float)
            
        finally:
            # クリーンアップ
            await memory_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_performance_under_memory_pressure(self):
        """メモリ圧迫下でのパフォーマンステスト"""
        memory_config = MemoryOptimizationConfig(
            max_gpu_memory_usage_percent=70.0,  # 低い閾値
            max_system_memory_usage_percent=70.0,
            optimization_strategy=OptimizationStrategy.AGGRESSIVE
        )
        memory_manager = MemoryOptimizationManager(memory_config)
        
        try:
            # 監視開始
            await memory_manager.start_monitoring()
            
            # 複数回の最適化実行
            for i in range(5):
                result = await memory_manager.optimize_memory(force=True)
                assert result["optimization_performed"] is True
                
                # 統計確認
                stats = memory_manager.get_optimization_stats()
                assert stats["cleanup_count"] > 0
                
                await asyncio.sleep(1)
            
        finally:
            await memory_manager.cleanup()


if __name__ == "__main__":
    # 直接実行時のテスト
    async def main():
        print("🧠 メモリ最適化テスト実行中...")
        
        # メモリ最適化管理システムテスト
        memory_config = MemoryOptimizationConfig()
        memory_manager = MemoryOptimizationManager(memory_config)
        
        print("メモリメトリクス取得テスト...")
        metrics = await memory_manager.get_memory_metrics()
        print(f"GPU メモリ: {metrics.gpu_memory_used_mb:.1f}MB / {metrics.gpu_memory_total_mb:.1f}MB")
        print(f"システムメモリ: {metrics.system_memory_used_mb:.1f}MB / {metrics.system_memory_total_mb:.1f}MB")
        
        print("メモリ最適化テスト...")
        result = await memory_manager.optimize_memory(force=True)
        print(f"最適化結果: {result['optimization_performed']}")
        
        print("統計情報...")
        stats = memory_manager.get_optimization_stats()
        print(f"最適化統計: {stats}")
        
        await memory_manager.cleanup()
        print("✅ メモリ最適化テスト完了")
    
    asyncio.run(main())
