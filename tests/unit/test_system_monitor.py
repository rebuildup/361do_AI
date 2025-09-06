"""
System Monitor のユニットテスト

要件: 4.1, 4.2, 4.5
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from src.advanced_agent.monitoring.system_monitor import (
    SystemMonitor, SystemMetrics, CPUMetrics, MemoryMetrics, 
    GPUMetrics, AlertConfig, AlertLevel
)


class TestSystemMonitor:
    """SystemMonitor のテストクラス"""
    
    @pytest.fixture
    def monitor(self):
        """テスト用システム監視"""
        return SystemMonitor(
            collection_interval=0.1,
            enable_gpu_monitoring=False  # テスト環境ではGPU無効
        )
    
    @pytest.fixture
    def sample_cpu_metrics(self):
        """サンプル CPU メトリクス"""
        return CPUMetrics(
            usage_percent=45.5,
            frequency_mhz=2400.0,
            temperature_celsius=65.0,
            core_count=8,
            thread_count=16,
            load_average=[1.2, 1.5, 1.8],
            context_switches=1000000,
            interrupts=500000
        )
    
    @pytest.fixture
    def sample_memory_metrics(self):
        """サンプルメモリメトリクス"""
        return MemoryMetrics(
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
    
    @pytest.fixture
    def sample_gpu_metrics(self):
        """サンプル GPU メトリクス"""
        return GPUMetrics(
            gpu_id=0,
            name="RTX 4050",
            utilization_percent=75.0,
            memory_total_mb=6144.0,
            memory_used_mb=4096.0,
            memory_free_mb=2048.0,
            memory_percent=66.7,
            temperature_celsius=70.0,
            power_draw_watts=120.0,
            power_limit_watts=150.0,
            fan_speed_percent=60.0,
            clock_graphics_mhz=1800.0,
            clock_memory_mhz=7000.0
        )
    
    def test_init(self, monitor):
        """初期化テスト"""
        assert monitor.collection_interval == 0.1
        assert isinstance(monitor.alert_config, AlertConfig)
        assert monitor.enable_gpu_monitoring is False
        assert monitor.is_monitoring is False
        assert len(monitor.metrics_history) == 0
    
    def test_alert_config(self):
        """アラート設定テスト"""
        config = AlertConfig(
            cpu_threshold=90.0,
            memory_threshold=85.0,
            gpu_memory_threshold=95.0,
            enabled=True
        )
        
        assert config.cpu_threshold == 90.0
        assert config.memory_threshold == 85.0
        assert config.gpu_memory_threshold == 95.0
        assert config.enabled is True
    
    @patch('src.advanced_agent.monitoring.system_monitor.psutil.cpu_percent')
    @patch('src.advanced_agent.monitoring.system_monitor.psutil.cpu_freq')
    @patch('src.advanced_agent.monitoring.system_monitor.psutil.cpu_count')
    @patch('src.advanced_agent.monitoring.system_monitor.psutil.cpu_stats')
    def test_get_cpu_metrics(self, mock_cpu_stats, mock_cpu_count, 
                           mock_cpu_freq, mock_cpu_percent, monitor):
        """CPU メトリクス取得テスト"""
        # モック設定
        mock_cpu_percent.return_value = 45.5
        mock_cpu_freq.return_value = Mock(current=2400.0)
        mock_cpu_count.side_effect = [8, 16]  # physical, logical
        mock_cpu_stats.return_value = Mock(ctx_switches=1000000, interrupts=500000)
        
        metrics = monitor.get_cpu_metrics()
        
        assert isinstance(metrics, CPUMetrics)
        assert metrics.usage_percent == 45.5
        assert metrics.frequency_mhz == 2400.0
        assert metrics.core_count == 8
        assert metrics.thread_count == 16
        assert metrics.context_switches == 1000000
        assert metrics.interrupts == 500000
    
    @patch('src.advanced_agent.monitoring.system_monitor.psutil.virtual_memory')
    @patch('src.advanced_agent.monitoring.system_monitor.psutil.swap_memory')
    def test_get_memory_metrics(self, mock_swap_memory, mock_virtual_memory, monitor):
        """メモリメトリクス取得テスト"""
        # モック設定
        mock_virtual_memory.return_value = Mock(
            total=32 * 1024**3,
            available=16 * 1024**3,
            used=16 * 1024**3,
            percent=50.0,
            cached=4 * 1024**3,
            buffers=2 * 1024**3
        )
        mock_swap_memory.return_value = Mock(
            total=8 * 1024**3,
            used=1 * 1024**3,
            percent=12.5
        )
        
        metrics = monitor.get_memory_metrics()
        
        assert isinstance(metrics, MemoryMetrics)
        assert metrics.total_gb == 32.0
        assert metrics.available_gb == 16.0
        assert metrics.used_gb == 16.0
        assert metrics.usage_percent == 50.0
        assert metrics.swap_percent == 12.5
    
    @patch('src.advanced_agent.monitoring.system_monitor.psutil.disk_partitions')
    @patch('src.advanced_agent.monitoring.system_monitor.psutil.disk_usage')
    def test_get_disk_metrics(self, mock_disk_usage, mock_disk_partitions, monitor):
        """ディスクメトリクス取得テスト"""
        # モック設定
        mock_disk_partitions.return_value = [
            Mock(device='C:', mountpoint='C:'),
            Mock(device='D:', mountpoint='D:')
        ]
        mock_disk_usage.side_effect = [
            Mock(total=1000, used=500),  # C: 50%
            Mock(total=2000, used=1600)  # D: 80%
        ]
        
        disk_metrics = monitor.get_disk_metrics()
        
        assert isinstance(disk_metrics, dict)
        assert disk_metrics['C:'] == 50.0
        assert disk_metrics['D:'] == 80.0
    
    @patch('src.advanced_agent.monitoring.system_monitor.psutil.net_io_counters')
    def test_get_network_metrics(self, mock_net_io_counters, monitor):
        """ネットワークメトリクス取得テスト"""
        # モック設定
        mock_net_io_counters.return_value = Mock(
            bytes_sent=1000000,
            bytes_recv=2000000,
            packets_sent=5000,
            packets_recv=8000
        )
        
        network_metrics = monitor.get_network_metrics()
        
        assert isinstance(network_metrics, dict)
        assert network_metrics['bytes_sent'] == 1000000
        assert network_metrics['bytes_recv'] == 2000000
        assert network_metrics['packets_sent'] == 5000
        assert network_metrics['packets_recv'] == 8000
    
    def test_check_alerts(self, monitor, sample_cpu_metrics, sample_memory_metrics):
        """アラートチェックテスト"""
        # 高使用率のメトリクス作成
        high_cpu = CPUMetrics(
            usage_percent=95.0,  # 閾値超過
            frequency_mhz=2400.0,
            temperature_celsius=65.0,
            core_count=8,
            thread_count=16,
            load_average=[1.2, 1.5, 1.8],
            context_switches=1000000,
            interrupts=500000
        )
        
        high_memory = MemoryMetrics(
            total_gb=32.0,
            available_gb=2.0,
            used_gb=30.0,
            usage_percent=95.0,  # 閾値超過
            swap_total_gb=8.0,
            swap_used_gb=1.0,
            swap_percent=12.5,
            cached_gb=4.0,
            buffers_gb=2.0
        )
        
        system_metrics = SystemMetrics(
            cpu=high_cpu,
            memory=high_memory,
            gpu=None,
            disk_usage={},
            network_io={},
            process_count=100,
            boot_time=datetime.now(),
            uptime_seconds=3600.0
        )
        
        alerts = monitor.check_alerts(system_metrics)
        
        assert len(alerts) >= 2  # CPU と メモリのアラート
        alert_types = [alert[0] for alert in alerts]
        assert AlertLevel.WARNING in alert_types
    
    def test_alert_callbacks(self, monitor):
        """アラートコールバックテスト"""
        callback_called = []
        
        def test_callback(level, message, data):
            callback_called.append((level, message, data))
        
        # コールバック追加
        monitor.add_alert_callback(test_callback)
        
        # アラート発火
        alerts = [(AlertLevel.WARNING, "Test alert", {"test": "data"})]
        monitor._trigger_alerts(alerts)
        
        assert len(callback_called) == 1
        assert callback_called[0][0] == AlertLevel.WARNING
        assert callback_called[0][1] == "Test alert"
        assert callback_called[0][2] == {"test": "data"}
        
        # コールバック削除
        monitor.remove_alert_callback(test_callback)
        monitor._trigger_alerts(alerts)
        
        # 新しいアラートは呼ばれない
        assert len(callback_called) == 1
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, monitor):
        """監視ライフサイクルテスト"""
        assert monitor.is_monitoring is False
        
        # 監視開始
        await monitor.start_monitoring()
        assert monitor.is_monitoring is True
        assert monitor.monitor_task is not None
        
        # 少し待機
        await asyncio.sleep(0.2)
        
        # 監視停止
        await monitor.stop_monitoring()
        assert monitor.is_monitoring is False
    
    def test_metrics_history(self, monitor, sample_cpu_metrics, sample_memory_metrics):
        """メトリクス履歴テスト"""
        # 履歴が空であることを確認
        assert len(monitor.get_metrics_history()) == 0
        assert monitor.get_latest_metrics() is None
        
        # メトリクス追加（手動）
        system_metrics = SystemMetrics(
            cpu=sample_cpu_metrics,
            memory=sample_memory_metrics,
            gpu=None,
            disk_usage={},
            network_io={},
            process_count=100,
            boot_time=datetime.now(),
            uptime_seconds=3600.0
        )
        
        monitor.metrics_history.append(system_metrics)
        
        # 履歴確認
        history = monitor.get_metrics_history()
        assert len(history) == 1
        assert history[0] == system_metrics
        
        latest = monitor.get_latest_metrics()
        assert latest == system_metrics
        
        # 履歴制限テスト
        monitor.max_history_size = 2
        for i in range(3):
            monitor._add_to_history(system_metrics)
        
        # 最大サイズを超えないことを確認
        assert len(monitor.metrics_history) <= monitor.max_history_size
        
        # 履歴クリア
        monitor.clear_history()
        assert len(monitor.get_metrics_history()) == 0
    
    def test_gpu_availability(self, monitor):
        """GPU 利用可能性テスト"""
        # GPU 無効の場合
        assert monitor.is_gpu_available() is False
        assert monitor.get_gpu_count() == 0
        assert monitor.get_gpu_metrics(0) is None
    
    @pytest.mark.asyncio
    async def test_cleanup(self, monitor):
        """クリーンアップテスト"""
        # 監視開始
        await monitor.start_monitoring()
        assert monitor.is_monitoring is True
        
        # クリーンアップ
        await monitor.cleanup()
        assert monitor.is_monitoring is False


class TestCPUMetrics:
    """CPUMetrics のテストクラス"""
    
    def test_init(self):
        """初期化テスト"""
        metrics = CPUMetrics(
            usage_percent=50.0,
            frequency_mhz=2400.0,
            temperature_celsius=65.0,
            core_count=8,
            thread_count=16,
            load_average=[1.0, 1.2, 1.5],
            context_switches=1000000,
            interrupts=500000
        )
        
        assert metrics.usage_percent == 50.0
        assert metrics.frequency_mhz == 2400.0
        assert metrics.temperature_celsius == 65.0
        assert metrics.core_count == 8
        assert metrics.thread_count == 16
        assert metrics.load_average == [1.0, 1.2, 1.5]
        assert metrics.context_switches == 1000000
        assert metrics.interrupts == 500000
        assert isinstance(metrics.timestamp, datetime)


class TestMemoryMetrics:
    """MemoryMetrics のテストクラス"""
    
    def test_init(self):
        """初期化テスト"""
        metrics = MemoryMetrics(
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
        
        assert metrics.total_gb == 32.0
        assert metrics.available_gb == 16.0
        assert metrics.used_gb == 16.0
        assert metrics.usage_percent == 50.0
        assert metrics.swap_total_gb == 8.0
        assert metrics.swap_used_gb == 1.0
        assert metrics.swap_percent == 12.5
        assert metrics.cached_gb == 4.0
        assert metrics.buffers_gb == 2.0
        assert isinstance(metrics.timestamp, datetime)


class TestGPUMetrics:
    """GPUMetrics のテストクラス"""
    
    def test_init(self):
        """初期化テスト"""
        metrics = GPUMetrics(
            gpu_id=0,
            name="RTX 4050",
            utilization_percent=75.0,
            memory_total_mb=6144.0,
            memory_used_mb=4096.0,
            memory_free_mb=2048.0,
            memory_percent=66.7,
            temperature_celsius=70.0,
            power_draw_watts=120.0,
            power_limit_watts=150.0,
            fan_speed_percent=60.0,
            clock_graphics_mhz=1800.0,
            clock_memory_mhz=7000.0
        )
        
        assert metrics.gpu_id == 0
        assert metrics.name == "RTX 4050"
        assert metrics.utilization_percent == 75.0
        assert metrics.memory_total_mb == 6144.0
        assert metrics.memory_used_mb == 4096.0
        assert metrics.memory_free_mb == 2048.0
        assert metrics.memory_percent == 66.7
        assert metrics.temperature_celsius == 70.0
        assert metrics.power_draw_watts == 120.0
        assert metrics.power_limit_watts == 150.0
        assert metrics.fan_speed_percent == 60.0
        assert metrics.clock_graphics_mhz == 1800.0
        assert metrics.clock_memory_mhz == 7000.0
        assert isinstance(metrics.timestamp, datetime)


class TestSystemMetrics:
    """SystemMetrics のテストクラス"""
    
    def test_init(self):
        """初期化テスト"""
        cpu_metrics = CPUMetrics(50.0, 2400.0, 65.0, 8, 16, [1.0], 1000000, 500000)
        memory_metrics = MemoryMetrics(32.0, 16.0, 16.0, 50.0, 8.0, 1.0, 12.5, 4.0, 2.0)
        
        system_metrics = SystemMetrics(
            cpu=cpu_metrics,
            memory=memory_metrics,
            gpu=None,
            disk_usage={"C:": 50.0},
            network_io={"bytes_sent": 1000},
            process_count=100,
            boot_time=datetime.now(),
            uptime_seconds=3600.0
        )
        
        assert system_metrics.cpu == cpu_metrics
        assert system_metrics.memory == memory_metrics
        assert system_metrics.gpu is None
        assert system_metrics.disk_usage == {"C:": 50.0}
        assert system_metrics.network_io == {"bytes_sent": 1000}
        assert system_metrics.process_count == 100
        assert system_metrics.uptime_seconds == 3600.0
        assert isinstance(system_metrics.timestamp, datetime)


@pytest.mark.integration
class TestSystemMonitorIntegration:
    """統合テスト"""
    
    @pytest.mark.asyncio
    async def test_full_monitoring_cycle(self):
        """完全監視サイクルテスト"""
        monitor = SystemMonitor(
            collection_interval=0.1,
            enable_gpu_monitoring=False
        )
        
        try:
            # 監視開始
            await monitor.start_monitoring()
            
            # データ収集のため少し待機
            await asyncio.sleep(0.3)
            
            # メトリクス確認
            latest = monitor.get_latest_metrics()
            assert latest is not None
            assert isinstance(latest, SystemMetrics)
            
            # 履歴確認
            history = monitor.get_metrics_history()
            assert len(history) > 0
            
        finally:
            await monitor.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])