"""
Prometheus + Grafana 異常検出・復旧システム

Prometheus の既存パターン検出による エラー分類を統合し、
Grafana アラートによる段階的復旧戦略実行を実装
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import aiohttp
import numpy as np
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram
from prometheus_client.parser import text_string_to_metric_families

from .system_monitor import SystemMonitor
# from .prometheus_collector import PrometheusCollector

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """異常タイプ"""
    GPU_MEMORY_HIGH = "gpu_memory_high"
    GPU_TEMPERATURE_HIGH = "gpu_temperature_high"
    CPU_USAGE_HIGH = "cpu_usage_high"
    INFERENCE_SLOW = "inference_slow"
    MODEL_ERROR = "model_error"
    MEMORY_LEAK = "memory_leak"
    SYSTEM_OVERLOAD = "system_overload"


class SeverityLevel(Enum):
    """重要度レベル"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyPattern:
    """異常パターン"""
    pattern_id: str
    name: str
    description: str
    anomaly_type: AnomalyType
    severity: SeverityLevel
    detection_query: str  # Prometheus クエリ
    threshold: float
    duration_minutes: int = 5
    recovery_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectedAnomaly:
    """検出された異常"""
    anomaly_id: str
    pattern: AnomalyPattern
    detected_at: datetime
    current_value: float
    threshold: float
    severity: SeverityLevel
    affected_components: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class RecoveryAction:
    """復旧アクション"""
    action_id: str
    name: str
    description: str
    action_type: str  # "system", "model", "resource"
    priority: int  # 1=highest, 5=lowest
    estimated_duration: int  # seconds
    prerequisites: List[str] = field(default_factory=list)
    rollback_possible: bool = True


class PrometheusAnomalyDetector:
    """Prometheus ベース異常検出システム"""
    
    def __init__(self,
                 prometheus_url: str = "http://localhost:9090",
                 grafana_url: str = "http://localhost:3000",
                 grafana_api_key: Optional[str] = None):
        self.prometheus_url = prometheus_url
        self.grafana_url = grafana_url
        self.grafana_api_key = grafana_api_key
        
        # 監視システム統合
        self.system_monitor = SystemMonitor()
        # self.prometheus_collector = PrometheusCollector()
        
        # 異常パターン定義
        self.anomaly_patterns = self._initialize_patterns()
        
        # 検出された異常の履歴
        self.detected_anomalies: Dict[str, DetectedAnomaly] = {}
        self.active_anomalies: Dict[str, DetectedAnomaly] = {}
        
        # 復旧アクション定義
        self.recovery_actions = self._initialize_recovery_actions()
        
        # メトリクス
        self.registry = CollectorRegistry()
        self.anomaly_counter = Counter(
            'anomalies_detected_total',
            'Total number of anomalies detected',
            ['anomaly_type', 'severity'],
            registry=self.registry
        )
        self.recovery_counter = Counter(
            'recovery_actions_executed_total',
            'Total number of recovery actions executed',
            ['action_type', 'success'],
            registry=self.registry
        )
        self.detection_duration = Histogram(
            'anomaly_detection_duration_seconds',
            'Time spent on anomaly detection',
            registry=self.registry
        )
    
    def _initialize_patterns(self) -> Dict[str, AnomalyPattern]:
        """異常パターンの初期化"""
        patterns = {}
        
        # GPU メモリ高使用率
        patterns["gpu_memory_high"] = AnomalyPattern(
            pattern_id="gpu_memory_high",
            name="GPU Memory High Usage",
            description="GPU メモリ使用率が異常に高い",
            anomaly_type=AnomalyType.GPU_MEMORY_HIGH,
            severity=SeverityLevel.HIGH,
            detection_query='gpu_memory_usage_percent > 90',
            threshold=90.0,
            duration_minutes=2,
            recovery_actions=[
                "increase_quantization",
                "offload_to_cpu",
                "clear_cache"
            ]
        )
        
        # GPU 温度異常
        patterns["gpu_temperature_high"] = AnomalyPattern(
            pattern_id="gpu_temperature_high",
            name="GPU Temperature High",
            description="GPU 温度が危険レベル",
            anomaly_type=AnomalyType.GPU_TEMPERATURE_HIGH,
            severity=SeverityLevel.CRITICAL,
            detection_query='gpu_temperature_celsius > 85',
            threshold=85.0,
            duration_minutes=1,
            recovery_actions=[
                "reduce_gpu_load",
                "emergency_cooling",
                "system_throttle"
            ]
        )
        
        # 推論速度低下
        patterns["inference_slow"] = AnomalyPattern(
            pattern_id="inference_slow",
            name="Inference Performance Degradation",
            description="推論速度が異常に遅い",
            anomaly_type=AnomalyType.INFERENCE_SLOW,
            severity=SeverityLevel.MEDIUM,
            detection_query='inference_duration_seconds > 10',
            threshold=10.0,
            duration_minutes=3,
            recovery_actions=[
                "optimize_model_params",
                "restart_inference_engine",
                "check_resource_contention"
            ]
        )
        
        # メモリリーク検出
        patterns["memory_leak"] = AnomalyPattern(
            pattern_id="memory_leak",
            name="Memory Leak Detection",
            description="メモリ使用量が継続的に増加",
            anomaly_type=AnomalyType.MEMORY_LEAK,
            severity=SeverityLevel.HIGH,
            detection_query='increase(system_memory_used_bytes[30m]) > 1073741824',  # 1GB increase
            threshold=1073741824,  # 1GB
            duration_minutes=30,
            recovery_actions=[
                "garbage_collection",
                "restart_components",
                "memory_profiling"
            ]
        )
        
        return patterns
    
    def _initialize_recovery_actions(self) -> Dict[str, RecoveryAction]:
        """復旧アクションの初期化"""
        actions = {}
        
        # 量子化レベル上昇
        actions["increase_quantization"] = RecoveryAction(
            action_id="increase_quantization",
            name="Increase Model Quantization",
            description="モデルの量子化レベルを上げてメモリ使用量を削減",
            action_type="model",
            priority=1,
            estimated_duration=30,
            rollback_possible=True
        )
        
        # CPU オフロード
        actions["offload_to_cpu"] = RecoveryAction(
            action_id="offload_to_cpu",
            name="Offload Processing to CPU",
            description="処理の一部を CPU にオフロード",
            action_type="resource",
            priority=2,
            estimated_duration=10,
            rollback_possible=True
        )
        
        # キャッシュクリア
        actions["clear_cache"] = RecoveryAction(
            action_id="clear_cache",
            name="Clear GPU Cache",
            description="GPU キャッシュをクリアしてメモリを解放",
            action_type="system",
            priority=1,
            estimated_duration=5,
            rollback_possible=False
        )
        
        # GPU 負荷削減
        actions["reduce_gpu_load"] = RecoveryAction(
            action_id="reduce_gpu_load",
            name="Reduce GPU Load",
            description="GPU 負荷を削減して温度を下げる",
            action_type="resource",
            priority=1,
            estimated_duration=15,
            rollback_possible=True
        )
        
        # システムスロットリング
        actions["system_throttle"] = RecoveryAction(
            action_id="system_throttle",
            name="System Throttling",
            description="システム全体の処理速度を制限",
            action_type="system",
            priority=3,
            estimated_duration=60,
            rollback_possible=True
        )
        
        return actions
    
    async def detect_anomalies(self) -> List[DetectedAnomaly]:
        """異常検出の実行"""
        with self.detection_duration.time():
            detected = []
            
            for pattern_id, pattern in self.anomaly_patterns.items():
                try:
                    anomaly = await self._check_pattern(pattern)
                    if anomaly:
                        detected.append(anomaly)
                        
                        # メトリクス更新
                        self.anomaly_counter.labels(
                            anomaly_type=pattern.anomaly_type.value,
                            severity=pattern.severity.value
                        ).inc()
                        
                        logger.warning(f"異常検出: {anomaly.pattern.name} - 値: {anomaly.current_value}")
                        
                except Exception as e:
                    logger.error(f"パターン {pattern_id} の検出中にエラー: {e}")
            
            return detected
    
    async def _check_pattern(self, pattern: AnomalyPattern) -> Optional[DetectedAnomaly]:
        """個別パターンの検証"""
        try:
            # Prometheus クエリ実行
            current_value = await self._execute_prometheus_query(pattern.detection_query)
            
            if current_value is None:
                return None
            
            # 閾値チェック
            if current_value > pattern.threshold:
                # 既存の異常をチェック
                existing_anomaly = self.active_anomalies.get(pattern.pattern_id)
                
                if existing_anomaly:
                    # 既存異常の更新
                    existing_anomaly.current_value = current_value
                    return existing_anomaly
                else:
                    # 新しい異常の作成
                    anomaly = DetectedAnomaly(
                        anomaly_id=f"{pattern.pattern_id}_{datetime.now().isoformat()}",
                        pattern=pattern,
                        detected_at=datetime.now(),
                        current_value=current_value,
                        threshold=pattern.threshold,
                        severity=pattern.severity,
                        context=await self._gather_context(pattern)
                    )
                    
                    self.active_anomalies[pattern.pattern_id] = anomaly
                    self.detected_anomalies[anomaly.anomaly_id] = anomaly
                    
                    return anomaly
            else:
                # 異常が解決された場合
                if pattern.pattern_id in self.active_anomalies:
                    resolved_anomaly = self.active_anomalies.pop(pattern.pattern_id)
                    resolved_anomaly.resolved = True
                    resolved_anomaly.resolved_at = datetime.now()
                    logger.info(f"異常解決: {resolved_anomaly.pattern.name}")
        
        except Exception as e:
            logger.error(f"パターン検証エラー {pattern.pattern_id}: {e}")
        
        return None
    
    async def _execute_prometheus_query(self, query: str) -> Optional[float]:
        """Prometheus クエリの実行"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.prometheus_url}/api/v1/query"
                params = {"query": query}
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data["status"] == "success" and data["data"]["result"]:
                            # 最初の結果の値を取得
                            result = data["data"]["result"][0]
                            return float(result["value"][1])
                    
                    logger.warning(f"Prometheus クエリ失敗: {query} - Status: {response.status}")
                    return None
        
        except Exception as e:
            logger.error(f"Prometheus クエリエラー: {e}")
            return None
    
    async def _gather_context(self, pattern: AnomalyPattern) -> Dict[str, Any]:
        """異常検出時のコンテキスト収集"""
        context = {}
        
        try:
            # システム統計取得
            system_stats = await self.system_monitor.get_system_stats()
            context["system_stats"] = system_stats
            
            # GPU 統計取得
            gpu_stats = await self.system_monitor.get_gpu_stats()
            context["gpu_stats"] = gpu_stats
            
            # 関連メトリクス取得
            related_metrics = await self._get_related_metrics(pattern.anomaly_type)
            context["related_metrics"] = related_metrics
            
        except Exception as e:
            logger.error(f"コンテキスト収集エラー: {e}")
            context["error"] = str(e)
        
        return context
    
    async def _get_related_metrics(self, anomaly_type: AnomalyType) -> Dict[str, float]:
        """関連メトリクスの取得"""
        metrics = {}
        
        try:
            if anomaly_type == AnomalyType.GPU_MEMORY_HIGH:
                queries = {
                    "gpu_utilization": "gpu_utilization_percent",
                    "model_memory": "model_memory_usage_bytes",
                    "cache_memory": "gpu_cache_memory_bytes"
                }
            elif anomaly_type == AnomalyType.INFERENCE_SLOW:
                queries = {
                    "queue_length": "inference_queue_length",
                    "active_requests": "active_inference_requests",
                    "cpu_usage": "cpu_usage_percent"
                }
            else:
                return metrics
            
            for metric_name, query in queries.items():
                value = await self._execute_prometheus_query(query)
                if value is not None:
                    metrics[metric_name] = value
        
        except Exception as e:
            logger.error(f"関連メトリクス取得エラー: {e}")
        
        return metrics
    
    async def start_monitoring(self, interval_seconds: int = 30):
        """継続的監視の開始"""
        logger.info(f"異常監視開始 - 間隔: {interval_seconds}秒")
        
        while True:
            try:
                # 異常検出実行
                detected_anomalies = await self.detect_anomalies()
                
                # 新しい異常があれば通知
                for anomaly in detected_anomalies:
                    if not anomaly.resolved:
                        await self._send_alert(anomaly)
                
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"監視ループエラー: {e}")
                await asyncio.sleep(interval_seconds)
    
    async def _send_alert(self, anomaly: DetectedAnomaly):
        """Grafana アラート送信"""
        try:
            if not self.grafana_api_key:
                logger.warning("Grafana API キーが設定されていません")
                return
            
            alert_data = {
                "title": f"異常検出: {anomaly.pattern.name}",
                "message": f"値: {anomaly.current_value}, 閾値: {anomaly.threshold}",
                "severity": anomaly.severity.value,
                "anomaly_type": anomaly.pattern.anomaly_type.value,
                "timestamp": anomaly.detected_at.isoformat(),
                "context": anomaly.context
            }
            
            headers = {
                "Authorization": f"Bearer {self.grafana_api_key}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.grafana_url}/api/alerts"
                async with session.post(url, json=alert_data, headers=headers) as response:
                    if response.status == 200:
                        logger.info(f"Grafana アラート送信成功: {anomaly.anomaly_id}")
                    else:
                        logger.warning(f"Grafana アラート送信失敗: {response.status}")
        
        except Exception as e:
            logger.error(f"アラート送信エラー: {e}")
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """異常検出サマリー"""
        return {
            "total_patterns": len(self.anomaly_patterns),
            "active_anomalies": len(self.active_anomalies),
            "total_detected": len(self.detected_anomalies),
            "anomaly_types": {
                anomaly_type.value: sum(
                    1 for anomaly in self.detected_anomalies.values()
                    if anomaly.pattern.anomaly_type == anomaly_type
                )
                for anomaly_type in AnomalyType
            },
            "severity_distribution": {
                severity.value: sum(
                    1 for anomaly in self.detected_anomalies.values()
                    if anomaly.severity == severity
                )
                for severity in SeverityLevel
            }
        }