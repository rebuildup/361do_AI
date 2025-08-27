"""
Performance Monitor
Codex互換エージェントのパフォーマンス監視システム
"""

import time
import asyncio
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime, timedelta
from loguru import logger


@dataclass
class RequestMetrics:
    """リクエストメトリクス"""
    request_id: str
    start_time: float
    end_time: Optional[float] = None
    request_type: str = "unknown"  # "complete", "chat", "stream"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 0
    success: bool = True
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    
    @property
    def duration(self) -> float:
        """リクエスト処理時間を取得"""
        if self.end_time is None:
            return 0.0  # まだ完了していない場合は0を返す
        return max(0.0, self.end_time - self.start_time)
    
    @property
    def tokens_per_second(self) -> float:
        """トークン/秒を計算"""
        duration = self.duration
        if duration <= 0:
            return 0.0
        return self.completion_tokens / duration


@dataclass
class SystemMetrics:
    """システムメトリクス"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    active_requests: int
    total_requests: int
    error_rate: float
    avg_response_time: float
    tokens_per_second: float


class PerformanceMonitor:
    """
    パフォーマンス監視クラス
    リクエスト処理時間、システムリソース、エラー率などを監視
    """
    
    def __init__(self, max_history: int = 1000, monitoring_interval: float = 5.0):
        self.max_history = max_history
        self.monitoring_interval = monitoring_interval
        
        # メトリクス保存
        self.request_metrics: deque[RequestMetrics] = deque(maxlen=max_history)
        self.system_metrics: deque[SystemMetrics] = deque(maxlen=max_history)
        
        # アクティブリクエスト追跡
        self.active_requests: Dict[str, RequestMetrics] = {}
        
        # 統計情報
        self.total_requests = 0
        self.total_errors = 0
        
        # 監視スレッド
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._lock = threading.Lock()
        
        # アラート設定
        self.alert_thresholds = {
            'response_time_ms': 10000,  # 10秒
            'error_rate_percent': 10.0,  # 10%
            'memory_percent': 80.0,     # 80%
            'cpu_percent': 80.0         # 80%
        }
        self.alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
    
    def start_monitoring(self):
        """監視開始"""
        if self._monitoring_thread is not None:
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """監視停止"""
        if self._monitoring_thread is None:
            return
        
        self._stop_monitoring.set()
        self._monitoring_thread.join(timeout=5.0)
        self._monitoring_thread = None
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """監視ループ"""
        while not self._stop_monitoring.wait(self.monitoring_interval):
            try:
                self._collect_system_metrics()
                self._check_alerts()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def _collect_system_metrics(self):
        """システムメトリクス収集"""
        try:
            # CPU・メモリ使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # リクエスト統計
            with self._lock:
                active_count = len(self.active_requests)
                
                # 過去1分間の統計
                recent_metrics = [
                    m for m in self.request_metrics
                    if time.time() - m.start_time < 60
                ]
                
                error_count = sum(1 for m in recent_metrics if not m.success)
                error_rate = (error_count / len(recent_metrics) * 100) if recent_metrics else 0.0
                
                avg_response_time = (
                    sum(m.duration for m in recent_metrics if m.end_time) / len(recent_metrics)
                    if recent_metrics else 0.0
                )
                
                tokens_per_second = (
                    sum(m.tokens_per_second for m in recent_metrics if m.end_time) / len(recent_metrics)
                    if recent_metrics else 0.0
                )
            
            # システムメトリクス保存
            system_metric = SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / 1024 / 1024,
                active_requests=active_count,
                total_requests=self.total_requests,
                error_rate=error_rate,
                avg_response_time=avg_response_time,
                tokens_per_second=tokens_per_second
            )
            
            with self._lock:
                self.system_metrics.append(system_metric)
        
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def _check_alerts(self):
        """アラートチェック"""
        if not self.system_metrics:
            return
        
        latest = self.system_metrics[-1]
        alerts = []
        
        # レスポンス時間アラート
        if latest.avg_response_time * 1000 > self.alert_thresholds['response_time_ms']:
            alerts.append({
                'type': 'high_response_time',
                'value': latest.avg_response_time * 1000,
                'threshold': self.alert_thresholds['response_time_ms'],
                'message': f"High response time: {latest.avg_response_time*1000:.1f}ms"
            })
        
        # エラー率アラート
        if latest.error_rate > self.alert_thresholds['error_rate_percent']:
            alerts.append({
                'type': 'high_error_rate',
                'value': latest.error_rate,
                'threshold': self.alert_thresholds['error_rate_percent'],
                'message': f"High error rate: {latest.error_rate:.1f}%"
            })
        
        # メモリ使用率アラート
        if latest.memory_percent > self.alert_thresholds['memory_percent']:
            alerts.append({
                'type': 'high_memory_usage',
                'value': latest.memory_percent,
                'threshold': self.alert_thresholds['memory_percent'],
                'message': f"High memory usage: {latest.memory_percent:.1f}%"
            })
        
        # CPU使用率アラート
        if latest.cpu_percent > self.alert_thresholds['cpu_percent']:
            alerts.append({
                'type': 'high_cpu_usage',
                'value': latest.cpu_percent,
                'threshold': self.alert_thresholds['cpu_percent'],
                'message': f"High CPU usage: {latest.cpu_percent:.1f}%"
            })
        
        # アラート通知
        for alert in alerts:
            self._trigger_alert(alert['type'], alert)
    
    def _trigger_alert(self, alert_type: str, alert_data: Dict[str, Any]):
        """アラート発火"""
        logger.warning(f"Performance alert: {alert_data['message']}")
        
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, alert_data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def start_request(self, request_id: str, request_type: str, **kwargs) -> RequestMetrics:
        """リクエスト開始記録"""
        metric = RequestMetrics(
            request_id=request_id,
            start_time=time.time(),
            request_type=request_type,
            model=kwargs.get('model', ''),
            temperature=kwargs.get('temperature', 0.0),
            max_tokens=kwargs.get('max_tokens', 0)
        )
        
        with self._lock:
            self.active_requests[request_id] = metric
            self.total_requests += 1
        
        return metric
    
    def end_request(
        self,
        request_id: str,
        success: bool = True,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        **kwargs
    ):
        """リクエスト終了記録"""
        with self._lock:
            if request_id not in self.active_requests:
                logger.warning(f"Request {request_id} not found in active requests")
                return
            
            metric = self.active_requests.pop(request_id)
            metric.end_time = time.time()
            metric.success = success
            metric.error_type = error_type
            metric.error_message = error_message
            
            # トークン情報更新
            metric.prompt_tokens = kwargs.get('prompt_tokens', 0)
            metric.completion_tokens = kwargs.get('completion_tokens', 0)
            metric.total_tokens = kwargs.get('total_tokens', 0)
            
            self.request_metrics.append(metric)
            
            if not success:
                self.total_errors += 1
    
    def get_performance_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """パフォーマンス要約取得"""
        cutoff_time = time.time() - (time_window_minutes * 60)
        
        with self._lock:
            # 指定時間内のメトリクス
            recent_requests = [
                m for m in self.request_metrics
                if m.start_time >= cutoff_time and m.end_time is not None
            ]
            
            recent_system = [
                m for m in self.system_metrics
                if m.timestamp >= cutoff_time
            ]
        
        if not recent_requests:
            return {
                'time_window_minutes': time_window_minutes,
                'total_requests': 0,
                'error_rate': 0.0,
                'avg_response_time_ms': 0.0,
                'p95_response_time_ms': 0.0,
                'avg_tokens_per_second': 0.0,
                'system_metrics': {}
            }
        
        # リクエスト統計
        durations = [m.duration * 1000 for m in recent_requests]  # ms
        durations.sort()
        
        successful_requests = [m for m in recent_requests if m.success]
        error_count = len(recent_requests) - len(successful_requests)
        
        # システム統計
        system_summary = {}
        if recent_system:
            system_summary = {
                'avg_cpu_percent': sum(m.cpu_percent for m in recent_system) / len(recent_system),
                'avg_memory_percent': sum(m.memory_percent for m in recent_system) / len(recent_system),
                'avg_memory_used_mb': sum(m.memory_used_mb for m in recent_system) / len(recent_system),
                'max_active_requests': max(m.active_requests for m in recent_system)
            }
        
        return {
            'time_window_minutes': time_window_minutes,
            'total_requests': len(recent_requests),
            'successful_requests': len(successful_requests),
            'error_count': error_count,
            'error_rate': (error_count / len(recent_requests)) * 100,
            'avg_response_time_ms': sum(durations) / len(durations),
            'p95_response_time_ms': durations[int(len(durations) * 0.95)] if durations else 0,
            'p99_response_time_ms': durations[int(len(durations) * 0.99)] if durations else 0,
            'avg_tokens_per_second': sum(m.tokens_per_second for m in successful_requests) / len(successful_requests) if successful_requests else 0,
            'total_tokens_processed': sum(m.total_tokens for m in successful_requests),
            'system_metrics': system_summary,
            'active_requests': len(self.active_requests)
        }
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """アラートコールバック追加"""
        self.alert_callbacks.append(callback)
    
    def set_alert_threshold(self, metric: str, value: float):
        """アラート閾値設定"""
        if metric in self.alert_thresholds:
            self.alert_thresholds[metric] = value
            logger.info(f"Alert threshold updated: {metric} = {value}")
    
    def get_request_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """リクエスト履歴取得"""
        with self._lock:
            recent_requests = list(self.request_metrics)[-limit:]
        
        return [
            {
                'request_id': m.request_id,
                'start_time': m.start_time,
                'duration_ms': m.duration * 1000,
                'request_type': m.request_type,
                'model': m.model,
                'total_tokens': m.total_tokens,
                'tokens_per_second': m.tokens_per_second,
                'success': m.success,
                'error_type': m.error_type
            }
            for m in recent_requests
        ]