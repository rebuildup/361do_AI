"""
System Monitor Integration

PSUtil + NVIDIA-ML による統合システム監視
RTX 4050 6GB VRAM環境でのリアルタイム性能監視を提供します。

要件: 4.1, 4.2, 4.5
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
import logging
import psutil

# オプショナル依存関係
try:
    import pynvml
    NVIDIA_ML_AVAILABLE = True
except ImportError:
    pynvml = None
    NVIDIA_ML_AVAILABLE = False

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """アラートレベル"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class CPUMetrics:
    """CPU メトリクス"""
    usage_percent: float
    frequency_mhz: float
    temperature_celsius: Optional[float]
    core_count: int
    thread_count: int
    load_average: List[float]
    context_switches: int
    interrupts: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MemoryMetrics:
    """メモリメトリクス"""
    total_gb: float
    available_gb: float
    used_gb: float
    usage_percent: float
    swap_total_gb: float
    swap_used_gb: float
    swap_percent: float
    cached_gb: float
    buffers_gb: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class GPUMetrics:
    """GPU メトリクス"""
    gpu_id: int
    name: str
    utilization_percent: float
    memory_total_mb: float
    memory_used_mb: float
    memory_free_mb: float
    memory_percent: float
    temperature_celsius: float
    power_draw_watts: float
    power_limit_watts: float
    fan_speed_percent: float
    clock_graphics_mhz: float
    clock_memory_mhz: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SystemMetrics:
    """システム全体メトリクス"""
    cpu: CPUMetrics
    memory: MemoryMetrics
    gpu: Optional[GPUMetrics]
    disk_usage: Dict[str, float]
    network_io: Dict[str, int]
    process_count: int
    boot_time: datetime
    uptime_seconds: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AlertConfig:
    """アラート設定"""
    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    gpu_memory_threshold: float = 90.0
    gpu_temperature_threshold: float = 80.0
    disk_threshold: float = 90.0
    enabled: bool = True


class SystemMonitor:
    """統合システム監視クラス"""
    
    def __init__(self,
                 collection_interval: float = 1.0,
                 alert_config: Optional[AlertConfig] = None,
                 enable_gpu_monitoring: bool = True):
        """
        初期化
        
        Args:
            collection_interval: メトリクス収集間隔（秒）
            alert_config: アラート設定
            enable_gpu_monitoring: GPU監視を有効にするか
        """
        self.collection_interval = collection_interval
        self.alert_config = alert_config or AlertConfig()
        self.enable_gpu_monitoring = enable_gpu_monitoring and NVIDIA_ML_AVAILABLE
        
        # GPU初期化
        self.gpu_initialized = False
        self.gpu_count = 0
        self.gpu_handles = []
        
        # メトリクス履歴
        self.metrics_history: List[SystemMetrics] = []
        self.max_history_size = 1000
        
        # アラートコールバック
        self.alert_callbacks: List[Callable[[AlertLevel, str, Dict[str, Any]], None]] = []
        
        # 監視状態
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        # 初期化
        self._initialize_gpu()
        
    def _initialize_gpu(self):
        """GPU監視初期化"""
        if not self.enable_gpu_monitoring:
            logger.info("GPU monitoring disabled")
            return
            
        if not NVIDIA_ML_AVAILABLE:
            logger.warning("NVIDIA-ML not available, GPU monitoring disabled")
            self.enable_gpu_monitoring = False
            return
            
        try:
            pynvml.nvmlInit()
            self.gpu_count = pynvml.nvmlDeviceGetCount()
            
            # GPU ハンドル取得
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                self.gpu_handles.append(handle)
            
            self.gpu_initialized = True
            logger.info(f"GPU monitoring initialized for {self.gpu_count} GPU(s)")
            
        except Exception as e:
            logger.error(f"Failed to initialize GPU monitoring: {e}")
            self.enable_gpu_monitoring = False
    
    def get_cpu_metrics(self) -> CPUMetrics:
        """CPU メトリクス取得"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # CPU周波数
            cpu_freq = psutil.cpu_freq()
            frequency = cpu_freq.current if cpu_freq else 0.0
            
            # CPU温度（利用可能な場合）
            temperature = None
            try:
                temps = psutil.sensors_temperatures()
                if 'coretemp' in temps:
                    temperature = temps['coretemp'][0].current
                elif 'cpu_thermal' in temps:
                    temperature = temps['cpu_thermal'][0].current
            except (AttributeError, KeyError, IndexError):
                pass
            
            # CPU情報
            cpu_count = psutil.cpu_count(logical=False)
            thread_count = psutil.cpu_count(logical=True)
            
            # 負荷平均
            try:
                load_avg = list(psutil.getloadavg())
            except AttributeError:
                # Windows では利用不可
                load_avg = [0.0, 0.0, 0.0]
            
            # システム統計
            cpu_stats = psutil.cpu_stats()
            
            return CPUMetrics(
                usage_percent=cpu_percent,
                frequency_mhz=frequency,
                temperature_celsius=temperature,
                core_count=cpu_count,
                thread_count=thread_count,
                load_average=load_avg,
                context_switches=cpu_stats.ctx_switches,
                interrupts=cpu_stats.interrupts
            )
            
        except Exception as e:
            logger.error(f"Failed to get CPU metrics: {e}")
            return CPUMetrics(0.0, 0.0, None, 0, 0, [0.0, 0.0, 0.0], 0, 0)
    
    def get_memory_metrics(self) -> MemoryMetrics:
        """メモリメトリクス取得"""
        try:
            # 仮想メモリ
            virtual_mem = psutil.virtual_memory()
            
            # スワップメモリ
            swap_mem = psutil.swap_memory()
            
            return MemoryMetrics(
                total_gb=virtual_mem.total / (1024**3),
                available_gb=virtual_mem.available / (1024**3),
                used_gb=virtual_mem.used / (1024**3),
                usage_percent=virtual_mem.percent,
                swap_total_gb=swap_mem.total / (1024**3),
                swap_used_gb=swap_mem.used / (1024**3),
                swap_percent=swap_mem.percent,
                cached_gb=getattr(virtual_mem, 'cached', 0) / (1024**3),
                buffers_gb=getattr(virtual_mem, 'buffers', 0) / (1024**3)
            )
            
        except Exception as e:
            logger.error(f"Failed to get memory metrics: {e}")
            return MemoryMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def get_gpu_metrics(self, gpu_id: int = 0) -> Optional[GPUMetrics]:
        """GPU メトリクス取得"""
        if not self.gpu_initialized or gpu_id >= len(self.gpu_handles):
            return None
            
        try:
            handle = self.gpu_handles[gpu_id]
            
            # GPU名
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            
            # GPU使用率
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # メモリ情報
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # 温度
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # 電力情報
            try:
                power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0
            except pynvml.NVMLError:
                power_draw = 0.0
                power_limit = 0.0
            
            # ファン速度
            try:
                fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
            except pynvml.NVMLError:
                fan_speed = 0.0
            
            # クロック速度
            try:
                graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            except pynvml.NVMLError:
                graphics_clock = 0.0
                memory_clock = 0.0
            
            return GPUMetrics(
                gpu_id=gpu_id,
                name=name,
                utilization_percent=utilization.gpu,
                memory_total_mb=memory_info.total / (1024**2),
                memory_used_mb=memory_info.used / (1024**2),
                memory_free_mb=memory_info.free / (1024**2),
                memory_percent=(memory_info.used / memory_info.total) * 100,
                temperature_celsius=temperature,
                power_draw_watts=power_draw,
                power_limit_watts=power_limit,
                fan_speed_percent=fan_speed,
                clock_graphics_mhz=graphics_clock,
                clock_memory_mhz=memory_clock
            )
            
        except Exception as e:
            logger.error(f"Failed to get GPU metrics for GPU {gpu_id}: {e}")
            return None
    
    def get_disk_metrics(self) -> Dict[str, float]:
        """ディスク使用率取得"""
        disk_usage = {}
        try:
            partitions = psutil.disk_partitions()
            for partition in partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_usage[partition.device] = (usage.used / usage.total) * 100
                except (PermissionError, FileNotFoundError):
                    continue
        except Exception as e:
            logger.error(f"Failed to get disk metrics: {e}")
        
        return disk_usage
    
    def get_network_metrics(self) -> Dict[str, int]:
        """ネットワーク I/O 取得"""
        try:
            net_io = psutil.net_io_counters()
            return {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            }
        except Exception as e:
            logger.error(f"Failed to get network metrics: {e}")
            return {"bytes_sent": 0, "bytes_recv": 0, "packets_sent": 0, "packets_recv": 0}
    
    def get_system_metrics(self) -> SystemMetrics:
        """システム全体メトリクス取得"""
        try:
            # 各コンポーネントのメトリクス取得
            cpu_metrics = self.get_cpu_metrics()
            memory_metrics = self.get_memory_metrics()
            gpu_metrics = self.get_gpu_metrics(0) if self.enable_gpu_monitoring else None
            disk_metrics = self.get_disk_metrics()
            network_metrics = self.get_network_metrics()
            
            # システム情報
            process_count = len(psutil.pids())
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            uptime = time.time() - psutil.boot_time()
            
            metrics = SystemMetrics(
                cpu=cpu_metrics,
                memory=memory_metrics,
                gpu=gpu_metrics,
                disk_usage=disk_metrics,
                network_io=network_metrics,
                process_count=process_count,
                boot_time=boot_time,
                uptime_seconds=uptime
            )
            
            # 履歴に追加（サイズ制限付き）
            self._add_to_history(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            # エラー時はデフォルト値を返す
            return SystemMetrics(
                cpu=CPUMetrics(0.0, 0.0, None, 0, 0, [0.0, 0.0, 0.0], 0, 0),
                memory=MemoryMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                gpu=None,
                disk_usage={},
                network_io={},
                process_count=0,
                boot_time=datetime.now(),
                uptime_seconds=0.0
            )
    
    def check_alerts(self, metrics: SystemMetrics) -> List[tuple]:
        """アラートチェック"""
        alerts = []
        
        if not self.alert_config.enabled:
            return alerts
        
        # CPU使用率チェック
        if metrics.cpu.usage_percent > self.alert_config.cpu_threshold:
            alerts.append((
                AlertLevel.WARNING,
                f"High CPU usage: {metrics.cpu.usage_percent:.1f}%",
                {"metric": "cpu_usage", "value": metrics.cpu.usage_percent}
            ))
        
        # メモリ使用率チェック
        if metrics.memory.usage_percent > self.alert_config.memory_threshold:
            alerts.append((
                AlertLevel.WARNING,
                f"High memory usage: {metrics.memory.usage_percent:.1f}%",
                {"metric": "memory_usage", "value": metrics.memory.usage_percent}
            ))
        
        # GPU メモリチェック
        if metrics.gpu and metrics.gpu.memory_percent > self.alert_config.gpu_memory_threshold:
            alerts.append((
                AlertLevel.CRITICAL,
                f"High GPU memory usage: {metrics.gpu.memory_percent:.1f}%",
                {"metric": "gpu_memory", "value": metrics.gpu.memory_percent}
            ))
        
        # GPU 温度チェック
        if metrics.gpu and metrics.gpu.temperature_celsius > self.alert_config.gpu_temperature_threshold:
            alerts.append((
                AlertLevel.WARNING,
                f"High GPU temperature: {metrics.gpu.temperature_celsius:.1f}°C",
                {"metric": "gpu_temperature", "value": metrics.gpu.temperature_celsius}
            ))
        
        # ディスク使用率チェック
        for device, usage in metrics.disk_usage.items():
            if usage > self.alert_config.disk_threshold:
                alerts.append((
                    AlertLevel.WARNING,
                    f"High disk usage on {device}: {usage:.1f}%",
                    {"metric": "disk_usage", "device": device, "value": usage}
                ))
        
        return alerts
    
    def add_alert_callback(self, callback: Callable[[AlertLevel, str, Dict[str, Any]], None]):
        """アラートコールバック追加"""
        self.alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable[[AlertLevel, str, Dict[str, Any]], None]):
        """アラートコールバック削除"""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
    
    def _trigger_alerts(self, alerts: List[tuple]):
        """アラート発火"""
        for level, message, data in alerts:
            logger.log(
                logging.WARNING if level == AlertLevel.WARNING else logging.ERROR,
                f"ALERT [{level.value.upper()}]: {message}"
            )
            
            # コールバック実行
            for callback in self.alert_callbacks:
                try:
                    callback(level, message, data)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
    
    async def start_monitoring(self):
        """監視開始"""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"System monitoring started (interval: {self.collection_interval}s)")
    
    async def stop_monitoring(self):
        """監視停止"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("System monitoring stopped")
    
    async def _monitoring_loop(self):
        """監視ループ"""
        try:
            while self.is_monitoring:
                # メトリクス収集
                metrics = self.get_system_metrics()
                
                # 履歴に追加
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history.pop(0)
                
                # アラートチェック
                alerts = self.check_alerts(metrics)
                if alerts:
                    self._trigger_alerts(alerts)
                
                # 次の収集まで待機
                await asyncio.sleep(self.collection_interval)
                
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Monitoring loop error: {e}")
            self.is_monitoring = False
    
    def get_metrics_history(self, limit: Optional[int] = None) -> List[SystemMetrics]:
        """メトリクス履歴取得"""
        if limit is None:
            return self.metrics_history.copy()
        return self.metrics_history[-limit:].copy()
    
    def get_latest_metrics(self) -> Optional[SystemMetrics]:
        """最新メトリクス取得"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """システム統計情報取得（SelfLearningAgent用）"""
        metrics = self.get_system_metrics()
        
        stats = {
            "cpu": {
                "usage_percent": metrics.cpu.usage_percent,
                "core_count": metrics.cpu.core_count,
                "frequency_mhz": metrics.cpu.frequency_mhz
            },
            "memory": {
                "usage_percent": metrics.memory.usage_percent,
                "total_gb": metrics.memory.total_gb,
                "used_gb": metrics.memory.used_gb,
                "available_gb": metrics.memory.available_gb
            },
            "gpu": None,
            "disk": {
                "usage_percent": metrics.disk_usage.get("usage_percent", 0.0),
                "total_gb": metrics.disk_usage.get("total_gb", 0.0),
                "used_gb": metrics.disk_usage.get("used_gb", 0.0),
                "free_gb": metrics.disk_usage.get("free_gb", 0.0)
            },
            "process_count": metrics.process_count,
            "uptime_seconds": metrics.uptime_seconds,
            "timestamp": metrics.timestamp.isoformat()
        }
        
        if metrics.gpu:
            stats["gpu"] = {
                "utilization_percent": metrics.gpu.utilization_percent,
                "memory_percent": metrics.gpu.memory_percent,
                "memory_used_mb": metrics.gpu.memory_used_mb,
                "memory_total_mb": metrics.gpu.memory_total_mb,
                "temperature_celsius": metrics.gpu.temperature_celsius,
                "power_usage_watts": metrics.gpu.power_usage_watts
            }
        
        return stats
    
    def clear_history(self):
        """履歴クリア"""
        self.metrics_history.clear()
        logger.info("Metrics history cleared")
    
    def _add_to_history(self, metrics: SystemMetrics):
        """履歴に追加（サイズ制限付き）"""
        self.metrics_history.append(metrics)
        
        # 最大サイズを超えた場合は古いものを削除
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
    
    def get_gpu_count(self) -> int:
        """GPU数取得"""
        return self.gpu_count
    
    def is_gpu_available(self) -> bool:
        """GPU利用可能性チェック"""
        return self.gpu_initialized
    
    async def cleanup(self):
        """リソースクリーンアップ"""
        await self.stop_monitoring()
        
        if self.gpu_initialized:
            try:
                pynvml.nvmlShutdown()
                logger.info("NVIDIA-ML shutdown completed")
            except Exception as e:
                logger.error(f"NVIDIA-ML shutdown error: {e}")


# 使用例とテスト用のヘルパー関数
async def demo_system_monitoring():
    """デモ用システム監視実行"""
    monitor = SystemMonitor(collection_interval=2.0)
    
    # アラートコールバック設定
    def alert_handler(level: AlertLevel, message: str, data: Dict[str, Any]):
        print(f"🚨 ALERT [{level.value.upper()}]: {message}")
        print(f"   Data: {data}")
    
    monitor.add_alert_callback(alert_handler)
    
    try:
        print("=== System Monitoring Demo ===")
        
        # 現在のメトリクス表示
        print("\n📊 Current System Metrics:")
        metrics = monitor.get_system_metrics()
        
        print(f"CPU Usage: {metrics.cpu.usage_percent:.1f}%")
        print(f"Memory Usage: {metrics.memory.usage_percent:.1f}% ({metrics.memory.used_gb:.1f}GB / {metrics.memory.total_gb:.1f}GB)")
        
        if metrics.gpu:
            print(f"GPU Usage: {metrics.gpu.utilization_percent:.1f}%")
            print(f"GPU Memory: {metrics.gpu.memory_percent:.1f}% ({metrics.gpu.memory_used_mb:.0f}MB / {metrics.gpu.memory_total_mb:.0f}MB)")
            print(f"GPU Temperature: {metrics.gpu.temperature_celsius:.1f}°C")
        else:
            print("GPU: Not available")
        
        print(f"Process Count: {metrics.process_count}")
        print(f"Uptime: {metrics.uptime_seconds / 3600:.1f} hours")
        
        # 短時間の監視実行
        print(f"\n🔄 Starting monitoring for 10 seconds...")
        await monitor.start_monitoring()
        await asyncio.sleep(10)
        await monitor.stop_monitoring()
        
        # 履歴表示
        history = monitor.get_metrics_history(limit=5)
        print(f"\n📈 Recent Metrics History ({len(history)} samples):")
        for i, h in enumerate(history):
            print(f"  {i+1}. CPU: {h.cpu.usage_percent:.1f}%, Memory: {h.memory.usage_percent:.1f}%")
        
    finally:
        await monitor.cleanup()


if __name__ == "__main__":
    asyncio.run(demo_system_monitoring())