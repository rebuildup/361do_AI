"""
Prometheus Metrics Collector

Prometheus Client による統合メトリクス収集システム
RTX 4050 6GB VRAM環境でのメトリクス収集とエクスポートを提供します。

要件: 4.1, 4.2, 4.5
"""

import asyncio
import time
from typing import Dict, Optional, List
import logging
from datetime import datetime

# オプショナル依存関係
try:
    from prometheus_client import (
        CollectorRegistry, Gauge, Counter, Histogram, Info,
        generate_latest, CONTENT_TYPE_LATEST, start_http_server
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # モック クラス
    class CollectorRegistry:
        pass
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, value): pass
        def labels(self, **kwargs): return self
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, value=1): pass
        def labels(self, **kwargs): return self
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, value): pass
        def labels(self, **kwargs): return self
    class Info:
        def __init__(self, *args, **kwargs): pass
        def info(self, data): pass

from .system_monitor import SystemMonitor, SystemMetrics, AlertLevel

logger = logging.getLogger(__name__)


class PrometheusMetricsCollector:
    """Prometheus メトリクス収集クラス"""
    
    def __init__(self,
                 system_monitor: SystemMonitor,
                 registry: Optional[CollectorRegistry] = None,
                 namespace: str = "advanced_agent"):
        """
        初期化
        
        Args:
            system_monitor: システム監視インスタンス
            registry: Prometheus レジストリ
            namespace: メトリクス名前空間
        """
        self.system_monitor = system_monitor
        self.namespace = namespace
        self.registry = registry or CollectorRegistry()
        
        # メトリクス定義
        self._define_metrics()
        
        # 収集状態
        self.is_collecting = False
        self.collection_task: Optional[asyncio.Task] = None
        self.http_server_port: Optional[int] = None
        
        # 統計
        self.collection_count = 0
        self.last_collection_time: Optional[datetime] = None
        self.collection_errors = 0
        
    def _define_metrics(self):
        """Prometheus メトリクス定義"""
        # CPU メトリクス
        self.cpu_usage = Gauge(
            f'{self.namespace}_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.cpu_frequency = Gauge(
            f'{self.namespace}_cpu_frequency_mhz',
            'CPU frequency in MHz',
            registry=self.registry
        )
        
        self.cpu_temperature = Gauge(
            f'{self.namespace}_cpu_temperature_celsius',
            'CPU temperature in Celsius',
            registry=self.registry
        )
        
        self.cpu_load_average = Gauge(
            f'{self.namespace}_cpu_load_average',
            'CPU load average',
            ['period'],
            registry=self.registry
        )
        
        # メモリメトリクス
        self.memory_usage = Gauge(
            f'{self.namespace}_memory_usage_percent',
            'Memory usage percentage',
            registry=self.registry
        )
        
        self.memory_total = Gauge(
            f'{self.namespace}_memory_total_gb',
            'Total memory in GB',
            registry=self.registry
        )
        
        self.memory_available = Gauge(
            f'{self.namespace}_memory_available_gb',
            'Available memory in GB',
            registry=self.registry
        )
        
        self.memory_used = Gauge(
            f'{self.namespace}_memory_used_gb',
            'Used memory in GB',
            registry=self.registry
        )
        
        self.swap_usage = Gauge(
            f'{self.namespace}_swap_usage_percent',
            'Swap usage percentage',
            registry=self.registry
        )
        
        # GPU メトリクス
        self.gpu_utilization = Gauge(
            f'{self.namespace}_gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id', 'gpu_name'],
            registry=self.registry
        )
        
        self.gpu_memory_usage = Gauge(
            f'{self.namespace}_gpu_memory_usage_percent',
            'GPU memory usage percentage',
            ['gpu_id', 'gpu_name'],
            registry=self.registry
        )
        
        self.gpu_memory_used = Gauge(
            f'{self.namespace}_gpu_memory_used_mb',
            'GPU memory used in MB',
            ['gpu_id', 'gpu_name'],
            registry=self.registry
        )
        
        self.gpu_memory_total = Gauge(
            f'{self.namespace}_gpu_memory_total_mb',
            'GPU memory total in MB',
            ['gpu_id', 'gpu_name'],
            registry=self.registry
        )
        
        self.gpu_temperature = Gauge(
            f'{self.namespace}_gpu_temperature_celsius',
            'GPU temperature in Celsius',
            ['gpu_id', 'gpu_name'],
            registry=self.registry
        )
        
        self.gpu_power_draw = Gauge(
            f'{self.namespace}_gpu_power_draw_watts',
            'GPU power draw in Watts',
            ['gpu_id', 'gpu_name'],
            registry=self.registry
        )
        
        self.gpu_fan_speed = Gauge(
            f'{self.namespace}_gpu_fan_speed_percent',
            'GPU fan speed percentage',
            ['gpu_id', 'gpu_name'],
            registry=self.registry
        )
        
        # ディスクメトリクス
        self.disk_usage = Gauge(
            f'{self.namespace}_disk_usage_percent',
            'Disk usage percentage',
            ['device'],
            registry=self.registry
        )
        
        # ネットワークメトリクス
        self.network_bytes_sent = Counter(
            f'{self.namespace}_network_bytes_sent_total',
            'Total bytes sent',
            registry=self.registry
        )
        
        self.network_bytes_received = Counter(
            f'{self.namespace}_network_bytes_received_total',
            'Total bytes received',
            registry=self.registry
        )
        
        # システムメトリクス
        self.process_count = Gauge(
            f'{self.namespace}_process_count',
            'Number of running processes',
            registry=self.registry
        )
        
        self.uptime_seconds = Gauge(
            f'{self.namespace}_uptime_seconds',
            'System uptime in seconds',
            registry=self.registry
        )
        
        # アラートメトリクス
        self.alert_count = Counter(
            f'{self.namespace}_alerts_total',
            'Total number of alerts',
            ['level', 'metric'],
            registry=self.registry
        )
        
        # 収集統計
        self.collection_duration = Histogram(
            f'{self.namespace}_collection_duration_seconds',
            'Time spent collecting metrics',
            registry=self.registry
        )
        
        self.collection_errors_total = Counter(
            f'{self.namespace}_collection_errors_total',
            'Total collection errors',
            registry=self.registry
        )
        
        # システム情報
        self.system_info = Info(
            f'{self.namespace}_system_info',
            'System information',
            registry=self.registry
        )
        
    def update_metrics(self, metrics: SystemMetrics):
        """メトリクス更新"""
        try:
            start_time = time.time()
            
            # CPU メトリクス
            self.cpu_usage.set(metrics.cpu.usage_percent)
            self.cpu_frequency.set(metrics.cpu.frequency_mhz)
            
            if metrics.cpu.temperature_celsius is not None:
                self.cpu_temperature.set(metrics.cpu.temperature_celsius)
            
            # 負荷平均
            load_periods = ['1min', '5min', '15min']
            for i, period in enumerate(load_periods):
                if i < len(metrics.cpu.load_average):
                    self.cpu_load_average.labels(period=period).set(metrics.cpu.load_average[i])
            
            # メモリメトリクス
            self.memory_usage.set(metrics.memory.usage_percent)
            self.memory_total.set(metrics.memory.total_gb)
            self.memory_available.set(metrics.memory.available_gb)
            self.memory_used.set(metrics.memory.used_gb)
            self.swap_usage.set(metrics.memory.swap_percent)
            
            # GPU メトリクス
            if metrics.gpu:
                gpu_labels = {
                    'gpu_id': str(metrics.gpu.gpu_id),
                    'gpu_name': metrics.gpu.name
                }
                
                self.gpu_utilization.labels(**gpu_labels).set(metrics.gpu.utilization_percent)
                self.gpu_memory_usage.labels(**gpu_labels).set(metrics.gpu.memory_percent)
                self.gpu_memory_used.labels(**gpu_labels).set(metrics.gpu.memory_used_mb)
                self.gpu_memory_total.labels(**gpu_labels).set(metrics.gpu.memory_total_mb)
                self.gpu_temperature.labels(**gpu_labels).set(metrics.gpu.temperature_celsius)
                self.gpu_power_draw.labels(**gpu_labels).set(metrics.gpu.power_draw_watts)
                self.gpu_fan_speed.labels(**gpu_labels).set(metrics.gpu.fan_speed_percent)
            
            # ディスクメトリクス
            for device, usage in metrics.disk_usage.items():
                self.disk_usage.labels(device=device).set(usage)
            
            # ネットワークメトリクス（累積値として更新）
            # 注意: Counterは単調増加のみなので、差分を計算する必要がある
            # ここでは簡略化して現在値を設定
            
            # システムメトリクス
            self.process_count.set(metrics.process_count)
            self.uptime_seconds.set(metrics.uptime_seconds)
            
            # 収集時間記録
            collection_time = time.time() - start_time
            self.collection_duration.observe(collection_time)
            
            # 統計更新
            self.collection_count += 1
            self.last_collection_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Failed to update Prometheus metrics: {e}")
            self.collection_errors += 1
            self.collection_errors_total.inc()
    
    def record_alert(self, level: AlertLevel, metric: str):
        """アラート記録"""
        try:
            self.alert_count.labels(level=level.value, metric=metric).inc()
        except Exception as e:
            logger.error(f"Failed to record alert metric: {e}")
    
    def set_system_info(self, info: Dict[str, str]):
        """システム情報設定"""
        try:
            self.system_info.info(info)
        except Exception as e:
            logger.error(f"Failed to set system info: {e}")
    
    async def start_collection(self, interval: float = 5.0):
        """メトリクス収集開始"""
        if self.is_collecting:
            logger.warning("Metrics collection already started")
            return
        
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available, collection disabled")
            return
        
        self.is_collecting = True
        self.collection_task = asyncio.create_task(self._collection_loop(interval))
        logger.info(f"Prometheus metrics collection started (interval: {interval}s)")
    
    async def stop_collection(self):
        """メトリクス収集停止"""
        if not self.is_collecting:
            return
        
        self.is_collecting = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Prometheus metrics collection stopped")
    
    async def _collection_loop(self, interval: float):
        """メトリクス収集ループ"""
        try:
            while self.is_collecting:
                try:
                    # システムメトリクス取得
                    metrics = self.system_monitor.get_system_metrics()
                    
                    # Prometheus メトリクス更新
                    self.update_metrics(metrics)
                    
                except Exception as e:
                    logger.error(f"Metrics collection error: {e}")
                    self.collection_errors += 1
                    self.collection_errors_total.inc()
                
                # 次の収集まで待機
                await asyncio.sleep(interval)
                
        except asyncio.CancelledError:
            logger.info("Metrics collection loop cancelled")
        except Exception as e:
            logger.error(f"Metrics collection loop error: {e}")
            self.is_collecting = False
    
    def start_http_server(self, port: int = 8000):
        """HTTP メトリクスサーバー開始"""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available, HTTP server disabled")
            return False
        
        try:
            start_http_server(port, registry=self.registry)
            self.http_server_port = port
            logger.info(f"Prometheus HTTP server started on port {port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start HTTP server: {e}")
            return False
    
    def get_metrics_text(self) -> str:
        """メトリクステキスト取得"""
        if not PROMETHEUS_AVAILABLE:
            return "# Prometheus client not available\n"
        
        try:
            return generate_latest(self.registry).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to generate metrics text: {e}")
            return f"# Error generating metrics: {e}\n"
    
    def get_collection_stats(self) -> Dict[str, any]:
        """収集統計取得"""
        return {
            "is_collecting": self.is_collecting,
            "collection_count": self.collection_count,
            "last_collection_time": self.last_collection_time,
            "collection_errors": self.collection_errors,
            "http_server_port": self.http_server_port,
            "prometheus_available": PROMETHEUS_AVAILABLE
        }
    
    async def cleanup(self):
        """リソースクリーンアップ"""
        await self.stop_collection()
        logger.info("Prometheus collector cleanup completed")


# アラートハンドラー統合
class PrometheusAlertHandler:
    """Prometheus アラートハンドラー"""
    
    def __init__(self, collector: PrometheusMetricsCollector):
        self.collector = collector
    
    def handle_alert(self, level: AlertLevel, message: str, data: Dict[str, any]):
        """アラート処理"""
        # メトリック名を抽出
        metric_name = data.get('metric', 'unknown')
        
        # Prometheus にアラート記録
        self.collector.record_alert(level, metric_name)
        
        # ログ出力
        logger.info(f"Alert recorded in Prometheus: {level.value} - {metric_name}")


# 使用例とテスト用のヘルパー関数
async def demo_prometheus_collection():
    """デモ用 Prometheus 収集実行"""
    from .system_monitor import SystemMonitor
    
    # システム監視初期化
    system_monitor = SystemMonitor(collection_interval=1.0)
    
    # Prometheus 収集器初期化
    collector = PrometheusMetricsCollector(system_monitor)
    
    # アラートハンドラー設定
    alert_handler = PrometheusAlertHandler(collector)
    system_monitor.add_alert_callback(alert_handler.handle_alert)
    
    try:
        print("=== Prometheus Metrics Collection Demo ===")
        
        # システム情報設定
        import platform
        system_info = {
            "hostname": platform.node(),
            "os": platform.system(),
            "architecture": platform.machine(),
            "python_version": platform.python_version()
        }
        collector.set_system_info(system_info)
        
        # HTTP サーバー開始
        if collector.start_http_server(8000):
            print("📊 Prometheus HTTP server started on http://localhost:8000/metrics")
        
        # メトリクス収集開始
        await collector.start_collection(interval=2.0)
        await system_monitor.start_monitoring()
        
        print("🔄 Collecting metrics for 15 seconds...")
        await asyncio.sleep(15)
        
        # 統計表示
        stats = collector.get_collection_stats()
        print(f"\n📈 Collection Statistics:")
        print(f"  Collections: {stats['collection_count']}")
        print(f"  Errors: {stats['collection_errors']}")
        print(f"  Last Collection: {stats['last_collection_time']}")
        
        # メトリクステキスト表示（一部）
        metrics_text = collector.get_metrics_text()
        lines = metrics_text.split('\n')
        print(f"\n📋 Sample Metrics (first 10 lines):")
        for line in lines[:10]:
            if line and not line.startswith('#'):
                print(f"  {line}")
        
    finally:
        await collector.cleanup()
        await system_monitor.cleanup()


if __name__ == "__main__":
    asyncio.run(demo_prometheus_collection())