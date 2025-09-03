"""
Auto Optimizer のユニットテスト

要件: 4.2, 4.4, 4.5
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from src.advanced_agent.optimization.auto_optimizer import (
    AutoOptimizer, OptimizationConfig, OptimizationResult,
    OptimizationStrategy, ResourceAllocation
)
from src.advanced_agent.monitoring.system_monitor import (
    SystemMonitor, SystemMetrics, CPUMetrics, MemoryMetrics, GPUMetrics
)


class TestAutoOptimizer:
    """AutoOptimizer のテストクラス"""
    
    @pytest.fixture
    def system_monitor(self):
        """テスト用システム監視"""
        return Mock(spec=SystemMonitor)
    
    @pytest.fixture
    def optimization_config(self):
        """テスト用最適化設定"""
        return OptimizationConfig(
            strategy=OptimizationStrategy.RTX4050_OPTIMIZED,
            optimization_interval_seconds=5.0,
            max_vram_usage_percent=85.0
        )
    
    @pytest.fixture
    def optimizer(self, system_monitor, optimization_config):
        """テスト用最適化器"""
        return AutoOptimizer(system_monitor, optimization_config)
    
    @pytest.fixture
    def sample_metrics(self):
        """サンプルシステムメトリクス"""
        cpu_metrics = CPUMetrics(75.0, 2400.0, 65.0, 8, 16, [1.2, 1.5, 1.8], 1000000, 500000)
        memory_metrics = MemoryMetrics(32.0, 8.0, 24.0, 75.0, 8.0, 1.0, 12.5, 4.0, 2.0)
        gpu_metrics = GPUMetrics(0, "RTX 4050", 80.0, 6144.0, 5120.0, 1024.0, 83.3, 72.0, 120.0, 150.0, 65.0, 1800.0, 7000.0)
        
        return SystemMetrics(
            cpu=cpu_metrics,
            memory=memory_metrics,
            gpu=gpu_metrics,
            disk_usage={"C:": 60.0},
            network_io={"bytes_sent": 1000000, "bytes_recv": 2000000},
            process_count=150,
            boot_time=datetime.now() - timedelta(hours=5),
            uptime_seconds=18000.0
        )
    
    def test_init(self, optimizer, optimization_config):
        """初期化テスト"""
        assert optimizer.config == optimization_config
        assert optimizer.config.strategy == OptimizationStrategy.RTX4050_OPTIMIZED
        assert optimizer.is_optimizing is False
        assert len(optimizer.optimization_history) == 0
        assert optimizer.optimization_count == 0
    
    def test_rtx4050_setup(self, system_monitor):
        """RTX 4050 設定テスト"""
        config = OptimizationConfig(strategy=OptimizationStrategy.RTX4050_OPTIMIZED)
        optimizer = AutoOptimizer(system_monitor, config)
        
        assert optimizer.config.max_vram_usage_percent == 85.0
        assert optimizer.config.target_temperature_celsius == 75.0
        assert optimizer.config.batch_size_range == (1, 8)
        assert optimizer.current_parameters["quantization_level"] == "4bit"
    
    def test_current_parameters(self, optimizer):
        """現在のパラメータテスト"""
        params = optimizer.get_current_parameters()
        
        assert "learning_rate" in params
        assert "batch_size" in params
        assert "quantization_level" in params
        assert "gradient_checkpointing" in params
        assert "mixed_precision" in params
    
    def test_set_parameters(self, optimizer):
        """パラメータ設定テスト"""
        new_params = {
            "learning_rate": 2e-4,
            "batch_size": 8,
            "quantization_level": "8bit"
        }
        
        optimizer.set_parameters(new_params)
        
        current_params = optimizer.get_current_parameters()
        assert current_params["learning_rate"] == 2e-4
        assert current_params["batch_size"] == 8
        assert current_params["quantization_level"] == "8bit"
    
    def test_determine_optimization_actions(self, optimizer, sample_metrics):
        """最適化アクション決定テスト"""
        # GPU メモリ使用率が高い場合
        high_vram_metrics = sample_metrics
        high_vram_metrics.gpu.memory_percent = 90.0
        
        actions = optimizer._determine_optimization_actions(high_vram_metrics)
        
        # 量子化レベル調整が含まれることを確認
        assert "quantization_level" in actions or "batch_size" in actions
    
    def test_rtx4050_specific_optimizations(self, optimizer, sample_metrics):
        """RTX 4050 固有最適化テスト"""
        # VRAM使用量が5GB超過
        high_vram_metrics = sample_metrics
        high_vram_metrics.gpu.memory_used_mb = 5200.0
        
        actions = optimizer._rtx4050_specific_optimizations(high_vram_metrics)
        
        assert "quantization_level" in actions
        assert actions["quantization_level"] == "3bit"
        assert "batch_size" in actions
        assert actions["batch_size"] == 1
    
    def test_calculate_performance_score(self, optimizer, sample_metrics):
        """性能スコア計算テスト"""
        score = optimizer._calculate_performance_score(sample_metrics)
        
        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)
    
    def test_calculate_resource_savings(self, optimizer):
        """リソース節約計算テスト"""
        actions = {
            "quantization_level": "4bit",
            "batch_size": 2,
            "gradient_checkpointing": True
        }
        
        savings = optimizer._calculate_resource_savings(actions)
        
        assert "vram_mb" in savings
        assert "system_memory_mb" in savings
        assert "power_watts" in savings
        assert "temperature_celsius" in savings
        
        # 量子化による VRAM 節約
        assert savings["vram_mb"] > 0
    
    def test_generate_recommendations(self, optimizer, sample_metrics):
        """推奨事項生成テスト"""
        actions = {
            "quantization_level": "4bit",
            "enable_cpu_offload": True
        }
        
        recommendations = optimizer._generate_recommendations(sample_metrics, actions)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # RTX 4050 固有推奨事項の確認
        rtx4050_recommendations = [r for r in recommendations if "RTX 4050" in r or "6GB" in r]
        if sample_metrics.gpu and sample_metrics.gpu.memory_percent > 80:
            assert len(rtx4050_recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_measure_optimization_effect(self, optimizer, sample_metrics):
        """最適化効果測定テスト"""
        actions = {
            "quantization_level": "4bit",
            "batch_size": 2,
            "mixed_precision": True
        }
        
        improvement = await optimizer._measure_optimization_effect(sample_metrics, actions)
        
        assert 0.0 <= improvement <= 50.0  # 最大50%改善
        assert isinstance(improvement, float)
    
    def test_calculate_action_similarity(self, optimizer):
        """アクション類似度計算テスト"""
        actions1 = {"quantization_level": "4bit", "batch_size": 4}
        actions2 = {"quantization_level": "4bit", "batch_size": 2}
        actions3 = {"learning_rate": 1e-4, "mixed_precision": True}
        
        # 部分的に類似
        similarity1 = optimizer._calculate_action_similarity(actions1, actions2)
        assert 0.0 < similarity1 < 1.0
        
        # 全く異なる
        similarity2 = optimizer._calculate_action_similarity(actions1, actions3)
        assert similarity2 == 0.0
        
        # 完全に同じ
        similarity3 = optimizer._calculate_action_similarity(actions1, actions1)
        assert similarity3 == 1.0
    
    @pytest.mark.asyncio
    async def test_manual_optimization(self, optimizer, sample_metrics):
        """手動最適化テスト"""
        # システムメトリクスをモック
        optimizer.system_monitor.get_latest_metrics.return_value = sample_metrics
        
        result = await optimizer.manual_optimization("vram_usage")
        
        assert isinstance(result, OptimizationResult)
        assert result.optimization_id.startswith("manual_")
        assert result.strategy_used == optimizer.config.strategy
    
    @pytest.mark.asyncio
    async def test_optimization_lifecycle(self, optimizer, sample_metrics):
        """最適化ライフサイクルテスト"""
        # システムメトリクスをモック
        optimizer.system_monitor.get_latest_metrics.return_value = sample_metrics
        
        assert optimizer.is_optimizing is False
        
        # 最適化開始
        await optimizer.start_optimization()
        assert optimizer.is_optimizing is True
        assert optimizer.optimization_task is not None
        
        # 少し待機
        await asyncio.sleep(0.1)
        
        # 最適化停止
        await optimizer.stop_optimization()
        assert optimizer.is_optimizing is False
    
    def test_optimization_stats(self, optimizer):
        """最適化統計テスト"""
        # ダミー履歴追加
        result1 = OptimizationResult(
            optimization_id="test1",
            strategy_used=OptimizationStrategy.RTX4050_OPTIMIZED,
            parameters_adjusted={"batch_size": {"old": 4, "new": 2}},
            performance_improvement=15.0,
            resource_savings={"vram_mb": 1024},
            execution_time=1.5,
            success=True
        )
        
        result2 = OptimizationResult(
            optimization_id="test2",
            strategy_used=OptimizationStrategy.RTX4050_OPTIMIZED,
            parameters_adjusted={},
            performance_improvement=0.0,
            resource_savings={},
            execution_time=0.5,
            success=False,
            error_message="Test error"
        )
        
        optimizer.optimization_history.extend([result1, result2])
        
        stats = optimizer.get_optimization_stats()
        
        assert stats["total_optimizations"] == 2
        assert stats["successful_optimizations"] == 1
        assert stats["success_rate"] == 0.5
        assert stats["average_improvement"] == 15.0
        assert stats["current_strategy"] == "rtx4050_optimized"
    
    def test_optimization_history(self, optimizer):
        """最適化履歴テスト"""
        # 履歴が空であることを確認
        assert len(optimizer.get_optimization_history()) == 0
        
        # ダミー結果追加
        result = OptimizationResult(
            optimization_id="test",
            strategy_used=OptimizationStrategy.RTX4050_OPTIMIZED,
            parameters_adjusted={},
            performance_improvement=10.0,
            resource_savings={},
            execution_time=1.0,
            success=True
        )
        
        optimizer.optimization_history.append(result)
        
        # 履歴確認
        history = optimizer.get_optimization_history()
        assert len(history) == 1
        assert history[0] == result
        
        # 制限付き履歴取得
        limited_history = optimizer.get_optimization_history(limit=1)
        assert len(limited_history) == 1
    
    @pytest.mark.asyncio
    async def test_cleanup(self, optimizer):
        """クリーンアップテスト"""
        # 最適化開始
        optimizer.system_monitor.get_latest_metrics.return_value = Mock()
        await optimizer.start_optimization()
        assert optimizer.is_optimizing is True
        
        # クリーンアップ
        await optimizer.cleanup()
        assert optimizer.is_optimizing is False


class TestOptimizationConfig:
    """OptimizationConfig のテストクラス"""
    
    def test_default_config(self):
        """デフォルト設定テスト"""
        config = OptimizationConfig()
        
        assert config.strategy == OptimizationStrategy.RTX4050_OPTIMIZED
        assert config.resource_allocation == ResourceAllocation.AUTO
        assert config.max_vram_usage_percent == 85.0
        assert config.target_temperature_celsius == 75.0
        assert config.auto_adjust_parameters is True
    
    def test_custom_config(self):
        """カスタム設定テスト"""
        config = OptimizationConfig(
            strategy=OptimizationStrategy.AGGRESSIVE,
            max_vram_usage_percent=90.0,
            optimization_interval_seconds=60.0
        )
        
        assert config.strategy == OptimizationStrategy.AGGRESSIVE
        assert config.max_vram_usage_percent == 90.0
        assert config.optimization_interval_seconds == 60.0


class TestOptimizationResult:
    """OptimizationResult のテストクラス"""
    
    def test_successful_result(self):
        """成功結果テスト"""
        result = OptimizationResult(
            optimization_id="test_success",
            strategy_used=OptimizationStrategy.RTX4050_OPTIMIZED,
            parameters_adjusted={"batch_size": {"old": 4, "new": 2}},
            performance_improvement=20.0,
            resource_savings={"vram_mb": 1024, "power_watts": 15},
            execution_time=2.5,
            success=True,
            recommendations=["Use 4bit quantization", "Enable CPU offload"]
        )
        
        assert result.success is True
        assert result.performance_improvement == 20.0
        assert result.resource_savings["vram_mb"] == 1024
        assert len(result.recommendations) == 2
        assert result.error_message is None
    
    def test_failed_result(self):
        """失敗結果テスト"""
        result = OptimizationResult(
            optimization_id="test_failure",
            strategy_used=OptimizationStrategy.BALANCED,
            parameters_adjusted={},
            performance_improvement=0.0,
            resource_savings={},
            execution_time=0.1,
            success=False,
            error_message="Optimization failed due to insufficient resources"
        )
        
        assert result.success is False
        assert result.performance_improvement == 0.0
        assert result.error_message is not None
        assert len(result.resource_savings) == 0


@pytest.mark.integration
class TestAutoOptimizerIntegration:
    """統合テスト"""
    
    @pytest.mark.asyncio
    async def test_full_optimization_cycle(self):
        """完全最適化サイクルテスト"""
        # モックシステム監視
        system_monitor = Mock(spec=SystemMonitor)
        
        # サンプルメトリクス
        cpu_metrics = CPUMetrics(50.0, 2400.0, 60.0, 8, 16, [1.0, 1.2, 1.5], 1000000, 500000)
        memory_metrics = MemoryMetrics(32.0, 16.0, 16.0, 50.0, 8.0, 0.5, 6.25, 4.0, 2.0)
        gpu_metrics = GPUMetrics(0, "RTX 4050", 60.0, 6144.0, 3072.0, 3072.0, 50.0, 65.0, 100.0, 150.0, 50.0, 1600.0, 6500.0)
        
        metrics = SystemMetrics(
            cpu=cpu_metrics,
            memory=memory_metrics,
            gpu=gpu_metrics,
            disk_usage={"C:": 50.0},
            network_io={"bytes_sent": 500000, "bytes_recv": 1000000},
            process_count=100,
            boot_time=datetime.now() - timedelta(hours=2),
            uptime_seconds=7200.0
        )
        
        system_monitor.get_latest_metrics.return_value = metrics
        
        # 最適化器初期化
        config = OptimizationConfig(
            strategy=OptimizationStrategy.RTX4050_OPTIMIZED,
            optimization_interval_seconds=0.1
        )
        optimizer = AutoOptimizer(system_monitor, config)
        
        try:
            # 手動最適化実行
            result = await optimizer.manual_optimization("vram_usage")
            
            assert isinstance(result, OptimizationResult)
            assert result.strategy_used == OptimizationStrategy.RTX4050_OPTIMIZED
            
            # 統計確認
            stats = optimizer.get_optimization_stats()
            assert stats["current_strategy"] == "rtx4050_optimized"
            
        finally:
            await optimizer.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])