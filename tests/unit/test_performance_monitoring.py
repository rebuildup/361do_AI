"""
Performance Monitoring System Tests
パフォーマンス監視システムのテスト
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from src.advanced_agent.monitoring.performance_analyzer import (
    PerformanceAnalyzer, BottleneckAnalysis, BottleneckType, 
    PerformanceLevel, PerformanceReport, TrendAnalysis
)
from src.advanced_agent.monitoring.system_monitor import (
    SystemMonitor, SystemMetrics, CPUMetrics, MemoryMetrics, GPUMetrics
)


class TestPerformanceAnalyzer:
    """パフォーマンス分析システムのテスト"""
    
    @pytest.fixture
    def system_monitor(self):
        """テスト用システム監視器"""
        return SystemMonitor(
            collection_interval=0.1,
            enable_gpu_monitoring=False  # テスト環境ではGPU無効
        )
    
    @pytest.fixture
    def analyzer(self, system_monitor):
        """テスト用パフォーマンス分析器"""
        return PerformanceAnalyzer(
            system_monitor=system_monitor,
            analysis_window=300,  # 5分間
            rtx4050_optimized=True
        )
    
    @pytest.fixture
    def sample_system_metrics(self):
        """サンプルシステムメトリクス"""
        cpu_metrics = CPUMetrics(
            usage_percent=45.5,
            frequency_mhz=2400.0,
            temperature_celsius=65.0,
            core_count=8,
            thread_count=16,
            load_average=[1.2, 1.5, 1.8],
            context_switches=1000000,
            interrupts=500000
        )
        
        memory_metrics = MemoryMetrics(
            total_gb=32.0,
            available_gb=16.0,
            used_gb=16.0,
            usage_percent=50.0,
            swap_total_gb=8.0,
            swap_used_gb=1.0,
            swap_percent=12.5,
            cached_gb=4.0,
            buffers_gb=2.0
        )
        
        gpu_metrics = GPUMetrics(
            gpu_id=0,
            name="RTX 4050",
            utilization_percent=75.0,
            memory_total_mb=6144.0,
            memory_used_mb=4608.0,
            memory_free_mb=1536.0,
            memory_percent=75.0,
            temperature_celsius=70.0,
            power_draw_watts=85.0,
            power_limit_watts=115.0,
            fan_speed_percent=60.0,
            clock_graphics_mhz=2100.0,
            clock_memory_mhz=7000.0
        )
        
        return SystemMetrics(
            cpu=cpu_metrics,
            memory=memory_metrics,
            gpu=gpu_metrics,
            disk_usage={"C:": 75.0, "D:": 45.0},
            network_io={"bytes_sent": 1000000, "bytes_recv": 2000000},
            process_count=150,
            boot_time=datetime.now() - timedelta(hours=2),
            uptime_seconds=7200.0
        )
    
    def test_performance_analyzer_initialization(self, analyzer):
        """パフォーマンス分析器の初期化テスト"""
        assert analyzer.rtx4050_optimized is True
        assert analyzer.analysis_window == 300  # 5分間 = 300秒
        assert analyzer.thresholds is not None
        assert "cpu_usage" in analyzer.thresholds
        assert "gpu_memory_usage" in analyzer.thresholds
    
    def test_analyze_current_performance_basic(self, analyzer, sample_system_metrics):
        """基本的なパフォーマンス分析テスト"""
        # システム監視器にメトリクスを追加
        analyzer.system_monitor._add_to_history(sample_system_metrics)
        
        # パフォーマンス分析実行
        report = analyzer.analyze_current_performance()
        
        assert isinstance(report, PerformanceReport)
        assert report.timestamp is not None
    
    def test_analyze_bottlenecks_high_cpu(self, analyzer):
        """高CPU使用率のボトルネック分析テスト"""
        # 高CPU使用率のメトリクス
        high_cpu_metrics = SystemMetrics(
            cpu=CPUMetrics(
                usage_percent=95.0,  # 高使用率
                frequency_mhz=2400.0,
                temperature_celsius=85.0,
                core_count=8,
                thread_count=16,
                load_average=[8.0, 8.5, 9.0],
                context_switches=2000000,
                interrupts=1000000
            ),
            memory=MemoryMetrics(
                total_gb=32.0,
                available_gb=16.0,
                used_gb=16.0,
                usage_percent=50.0,
                swap_total_gb=8.0,
                swap_used_gb=1.0,
                swap_percent=12.5,
                cached_gb=4.0,
                buffers_gb=2.0
            ),
            gpu=None,
            disk_usage={},
            network_io={},
            process_count=200,
            boot_time=datetime.now(),
            uptime_seconds=3600.0
        )
        
        bottlenecks = analyzer.analyze_bottlenecks(high_cpu_metrics, [])
        
        # CPUボトルネックが検出されるはず
        cpu_bottlenecks = [b for b in bottlenecks if b.bottleneck_type == BottleneckType.CPU]
        assert len(cpu_bottlenecks) > 0
        assert cpu_bottlenecks[0].severity > 0.5
    
    def test_analyze_bottlenecks_high_gpu_memory(self, analyzer):
        """高GPUメモリ使用率のボトルネック分析テスト"""
        # 高GPUメモリ使用率のメトリクス
        high_gpu_memory_metrics = SystemMetrics(
            cpu=CPUMetrics(
                usage_percent=45.0,
                frequency_mhz=2400.0,
                temperature_celsius=65.0,
                core_count=8,
                thread_count=16,
                load_average=[1.2, 1.5, 1.8],
                context_switches=1000000,
                interrupts=500000
            ),
            memory=MemoryMetrics(
                total_gb=32.0,
                available_gb=16.0,
                used_gb=16.0,
                usage_percent=50.0,
                swap_total_gb=8.0,
                swap_used_gb=1.0,
                swap_percent=12.5,
                cached_gb=4.0,
                buffers_gb=2.0
            ),
            gpu=GPUMetrics(
                gpu_id=0,
                name="RTX 4050",
                utilization_percent=90.0,
                memory_total_mb=6144.0,
                memory_used_mb=5800.0,  # 高メモリ使用率
                memory_free_mb=344.0,
                memory_percent=94.5,  # 危険レベル
                temperature_celsius=80.0,
                power_draw_watts=110.0,
                power_limit_watts=115.0,
                fan_speed_percent=80.0,
                clock_graphics_mhz=2100.0,
                clock_memory_mhz=7000.0
            ),
            disk_usage={},
            network_io={},
            process_count=150,
            boot_time=datetime.now(),
            uptime_seconds=3600.0
        )
        
        bottlenecks = analyzer.analyze_bottlenecks(high_gpu_memory_metrics, [])
        
        # GPUメモリボトルネックが検出されるはず
        gpu_memory_bottlenecks = [b for b in bottlenecks if b.bottleneck_type == BottleneckType.GPU_MEMORY]
        assert len(gpu_memory_bottlenecks) > 0
        assert gpu_memory_bottlenecks[0].severity > 0.8
    
    def test_calculate_performance_score(self, analyzer, sample_system_metrics):
        """パフォーマンススコア計算テスト"""
        score = analyzer.calculate_performance_score(sample_system_metrics, [])
        
        assert 0.0 <= score <= 1.0
        # 正常な状態では高いスコアが期待される
        assert score > 0.5
    
    def test_get_performance_level(self, analyzer):
        """パフォーマンスレベル判定テスト"""
        # 優秀なスコア
        excellent_level = analyzer.get_performance_level(0.9)
        assert excellent_level == PerformanceLevel.EXCELLENT
        
        # 良好なスコア
        good_level = analyzer.get_performance_level(0.7)
        assert good_level == PerformanceLevel.GOOD
        
        # 普通のスコア
        fair_level = analyzer.get_performance_level(0.5)
        assert fair_level == PerformanceLevel.FAIR
        
        # 悪いスコア
        poor_level = analyzer.get_performance_level(0.3)
        assert poor_level == PerformanceLevel.POOR
        
        # 危険なスコア
        critical_level = analyzer.get_performance_level(0.1)
        assert critical_level == PerformanceLevel.CRITICAL
    
    def test_analyze_efficiency(self, analyzer, sample_system_metrics):
        """効率性分析テスト"""
        history = [sample_system_metrics] * 5  # 履歴データ
        efficiency = analyzer.analyze_efficiency(sample_system_metrics, history)
        
        assert isinstance(efficiency, dict)
        assert "cpu_efficiency" in efficiency
        assert "memory_efficiency" in efficiency
        assert "gpu_efficiency" in efficiency
        assert "overall_efficiency" in efficiency
        
        # 効率性スコアは0-1の範囲
        for key, value in efficiency.items():
            assert 0.0 <= value <= 1.0
    
    def test_get_optimization_recommendations(self, analyzer, sample_system_metrics):
        """最適化推奨事項取得テスト"""
        recommendations = analyzer.get_optimization_recommendations(sample_system_metrics, [])
        
        assert isinstance(recommendations, list)
        # 正常な状態では推奨事項は少ないはず
        assert len(recommendations) <= 3
    
    def test_get_rtx4050_recommendations(self, analyzer):
        """RTX 4050最適化推奨事項テスト"""
        # RTX 4050向けの高負荷メトリクス
        rtx4050_metrics = SystemMetrics(
            cpu=CPUMetrics(
                usage_percent=60.0,
                frequency_mhz=2400.0,
                temperature_celsius=70.0,
                core_count=8,
                thread_count=16,
                load_average=[2.0, 2.5, 3.0],
                context_switches=1500000,
                interrupts=750000
            ),
            memory=MemoryMetrics(
                total_gb=32.0,
                available_gb=12.0,
                used_gb=20.0,
                usage_percent=62.5,
                swap_total_gb=8.0,
                swap_used_gb=2.0,
                swap_percent=25.0,
                cached_gb=6.0,
                buffers_gb=3.0
            ),
            gpu=GPUMetrics(
                gpu_id=0,
                name="RTX 4050",
                utilization_percent=95.0,
                memory_total_mb=6144.0,
                memory_used_mb=5500.0,
                memory_free_mb=644.0,
                memory_percent=89.5,
                temperature_celsius=75.0,
                power_draw_watts=105.0,
                power_limit_watts=115.0,
                fan_speed_percent=70.0,
                clock_graphics_mhz=2100.0,
                clock_memory_mhz=7000.0
            ),
            disk_usage={"C:": 80.0},
            network_io={"bytes_sent": 2000000, "bytes_recv": 4000000},
            process_count=200,
            boot_time=datetime.now(),
            uptime_seconds=3600.0
        )
        
        recommendations = analyzer.get_optimization_recommendations(rtx4050_metrics, [])
        
        # RTX 4050向けの推奨事項が含まれるはず
        rtx4050_recommendations = [r for r in recommendations if "RTX 4050" in r or "GPU" in r]
        assert len(rtx4050_recommendations) > 0
    
    def test_generate_performance_report(self, analyzer, sample_system_metrics):
        """パフォーマンスレポート生成テスト"""
        history = [sample_system_metrics] * 10
        report = analyzer.generate_performance_report(sample_system_metrics, history)
        
        assert isinstance(report, PerformanceReport)
        assert report.timestamp is not None
        assert report.performance_score is not None
        assert report.performance_level is not None
        assert report.bottlenecks is not None
        assert report.efficiency_metrics is not None
        assert report.recommendations is not None
    
    def test_analyze_performance_trend(self, analyzer):
        """パフォーマンストレンド分析テスト"""
        # 時間経過とともに性能が悪化するシナリオ
        base_time = datetime.now()
        metrics_history = []
        
        for i in range(10):
            cpu_usage = 50.0 + i * 5.0  # 50%から95%まで上昇
            memory_usage = 40.0 + i * 3.0  # 40%から67%まで上昇
            
            metrics = SystemMetrics(
                cpu=CPUMetrics(
                    usage_percent=cpu_usage,
                    frequency_mhz=2400.0,
                    temperature_celsius=65.0 + i * 2.0,
                    core_count=8,
                    thread_count=16,
                    load_average=[1.0 + i * 0.5, 1.2 + i * 0.5, 1.5 + i * 0.5],
                    context_switches=1000000 + i * 100000,
                    interrupts=500000 + i * 50000
                ),
                memory=MemoryMetrics(
                    total_gb=32.0,
                    available_gb=20.0 - i * 1.0,
                    used_gb=12.0 + i * 1.0,
                    usage_percent=memory_usage,
                    swap_total_gb=8.0,
                    swap_used_gb=1.0 + i * 0.2,
                    swap_percent=12.5 + i * 2.5,
                    cached_gb=4.0,
                    buffers_gb=2.0
                ),
                gpu=None,
                disk_usage={},
                network_io={},
                process_count=150 + i * 5,
                boot_time=base_time - timedelta(hours=2),
                uptime_seconds=7200.0 + i * 60.0
            )
            metrics_history.append(metrics)
        
        trend = analyzer.analyze_performance_trend(metrics_history)
        
        assert isinstance(trend, TrendAnalysis)
        assert trend.trend_direction in ["improving", "stable", "declining"]
        assert trend.confidence_score >= 0.0
        assert trend.confidence_score <= 1.0
        assert len(trend.key_metrics) > 0


class TestBottleneckAnalysis:
    """ボトルネック分析結果のテスト"""
    
    def test_bottleneck_analysis_initialization(self):
        """ボトルネック分析結果の初期化テスト"""
        analysis = BottleneckAnalysis(
            bottleneck_type=BottleneckType.CPU,
            severity=0.8,
            description="High CPU usage detected",
            current_value=85.0,
            threshold_value=80.0,
            impact_score=0.7,
            recommendations=["Reduce CPU load", "Optimize processes"]
        )
        
        assert analysis.bottleneck_type == BottleneckType.CPU
        assert analysis.severity == 0.8
        assert analysis.description == "High CPU usage detected"
        assert analysis.current_value == 85.0
        assert analysis.threshold_value == 80.0
        assert analysis.impact_score == 0.7
        assert len(analysis.recommendations) == 2


class TestPerformanceReport:
    """パフォーマンスレポートのテスト"""
    
    def test_performance_report_initialization(self):
        """パフォーマンスレポートの初期化テスト"""
        report = PerformanceReport(
            timestamp=datetime.now(),
            performance_score=0.75,
            performance_level=PerformanceLevel.GOOD,
            bottlenecks=[],
            efficiency_metrics={"cpu_efficiency": 0.8, "memory_efficiency": 0.7},
            recommendations=["Optimize memory usage"],
            trend_analysis=None
        )
        
        assert report.performance_score == 0.75
        assert report.performance_level == PerformanceLevel.GOOD
        assert len(report.bottlenecks) == 0
        assert "cpu_efficiency" in report.efficiency_metrics
        assert len(report.recommendations) == 1


if __name__ == "__main__":
    pytest.main([__file__])
