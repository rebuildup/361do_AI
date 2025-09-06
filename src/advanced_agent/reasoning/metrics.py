"""
Evaluation metrics for reasoning quality
推論品質評価指標
"""

import time
import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

from .quality_evaluator import QualityEvaluation, QualityDimension


class MetricType(Enum):
    """メトリクスタイプ"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class MetricCategory(Enum):
    """メトリクスカテゴリ"""
    PERFORMANCE = "performance"
    QUALITY = "quality"
    USAGE = "usage"
    ERROR = "error"
    SYSTEM = "system"


@dataclass
class MetricPoint:
    """メトリクスデータポイント"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    category: MetricCategory = MetricCategory.SYSTEM
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class MetricSummary:
    """メトリクスサマリー"""
    name: str
    count: int
    sum: float
    min: float
    max: float
    mean: float
    median: float
    std_dev: float
    percentiles: Dict[str, float] = field(default_factory=dict)
    time_range: Tuple[datetime, datetime] = (datetime.min, datetime.max)


class MetricsCollector:
    """メトリクス収集器"""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics: List[MetricPoint] = []
        self.metric_configs: Dict[str, Dict[str, Any]] = {}
        
        # メトリクス設定初期化
        self._initialize_metric_configs()
        
        # 統計情報
        self.stats = {
            "total_metrics": 0,
            "metrics_by_category": {},
            "metrics_by_type": {},
            "last_cleanup": datetime.now()
        }
    
    def _initialize_metric_configs(self):
        """メトリクス設定初期化"""
        self.metric_configs = {
            # パフォーマンスメトリクス
            "reasoning_time": {
                "type": MetricType.HISTOGRAM,
                "category": MetricCategory.PERFORMANCE,
                "description": "推論処理時間（秒）",
                "unit": "seconds",
                "buckets": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
            },
            "evaluation_time": {
                "type": MetricType.HISTOGRAM,
                "category": MetricCategory.PERFORMANCE,
                "description": "品質評価時間（秒）",
                "unit": "seconds",
                "buckets": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
            },
            "response_length": {
                "type": MetricType.HISTOGRAM,
                "category": MetricCategory.PERFORMANCE,
                "description": "レスポンス長（文字数）",
                "unit": "characters",
                "buckets": [50, 100, 200, 500, 1000, 2000, 5000]
            },
            "step_count": {
                "type": MetricType.HISTOGRAM,
                "category": MetricCategory.PERFORMANCE,
                "description": "推論ステップ数",
                "unit": "steps",
                "buckets": [1, 2, 3, 5, 8, 10, 15, 20]
            },
            
            # 品質メトリクス
            "overall_quality": {
                "type": MetricType.GAUGE,
                "category": MetricCategory.QUALITY,
                "description": "総合品質スコア",
                "unit": "score",
                "range": (0.0, 1.0)
            },
            "accuracy_score": {
                "type": MetricType.GAUGE,
                "category": MetricCategory.QUALITY,
                "description": "正確性スコア",
                "unit": "score",
                "range": (0.0, 1.0)
            },
            "completeness_score": {
                "type": MetricType.GAUGE,
                "category": MetricCategory.QUALITY,
                "description": "完全性スコア",
                "unit": "score",
                "range": (0.0, 1.0)
            },
            "clarity_score": {
                "type": MetricType.GAUGE,
                "category": MetricCategory.QUALITY,
                "description": "明確性スコア",
                "unit": "score",
                "range": (0.0, 1.0)
            },
            "logical_consistency_score": {
                "type": MetricType.GAUGE,
                "category": MetricCategory.QUALITY,
                "description": "論理的一貫性スコア",
                "unit": "score",
                "range": (0.0, 1.0)
            },
            "usefulness_score": {
                "type": MetricType.GAUGE,
                "category": MetricCategory.QUALITY,
                "description": "有用性スコア",
                "unit": "score",
                "range": (0.0, 1.0)
            },
            "efficiency_score": {
                "type": MetricType.GAUGE,
                "category": MetricCategory.QUALITY,
                "description": "効率性スコア",
                "unit": "score",
                "range": (0.0, 1.0)
            },
            "creativity_score": {
                "type": MetricType.GAUGE,
                "category": MetricCategory.QUALITY,
                "description": "創造性スコア",
                "unit": "score",
                "range": (0.0, 1.0)
            },
            "safety_score": {
                "type": MetricType.GAUGE,
                "category": MetricCategory.QUALITY,
                "description": "安全性スコア",
                "unit": "score",
                "range": (0.0, 1.0)
            },
            "confidence_score": {
                "type": MetricType.GAUGE,
                "category": MetricCategory.QUALITY,
                "description": "信頼度スコア",
                "unit": "score",
                "range": (0.0, 1.0)
            },
            
            # 使用メトリクス
            "total_requests": {
                "type": MetricType.COUNTER,
                "category": MetricCategory.USAGE,
                "description": "総リクエスト数",
                "unit": "requests"
            },
            "successful_requests": {
                "type": MetricType.COUNTER,
                "category": MetricCategory.USAGE,
                "description": "成功リクエスト数",
                "unit": "requests"
            },
            "failed_requests": {
                "type": MetricType.COUNTER,
                "category": MetricCategory.USAGE,
                "description": "失敗リクエスト数",
                "unit": "requests"
            },
            "requests_per_minute": {
                "type": MetricType.GAUGE,
                "category": MetricCategory.USAGE,
                "description": "1分あたりのリクエスト数",
                "unit": "requests/minute"
            },
            
            # エラーメトリクス
            "error_rate": {
                "type": MetricType.GAUGE,
                "category": MetricCategory.ERROR,
                "description": "エラー率",
                "unit": "percentage",
                "range": (0.0, 100.0)
            },
            "timeout_errors": {
                "type": MetricType.COUNTER,
                "category": MetricCategory.ERROR,
                "description": "タイムアウトエラー数",
                "unit": "errors"
            },
            "validation_errors": {
                "type": MetricType.COUNTER,
                "category": MetricCategory.ERROR,
                "description": "バリデーションエラー数",
                "unit": "errors"
            },
            "system_errors": {
                "type": MetricType.COUNTER,
                "category": MetricCategory.ERROR,
                "description": "システムエラー数",
                "unit": "errors"
            },
            
            # システムメトリクス
            "memory_usage": {
                "type": MetricType.GAUGE,
                "category": MetricCategory.SYSTEM,
                "description": "メモリ使用量",
                "unit": "MB"
            },
            "cpu_usage": {
                "type": MetricType.GAUGE,
                "category": MetricCategory.SYSTEM,
                "description": "CPU使用率",
                "unit": "percentage",
                "range": (0.0, 100.0)
            },
            "active_sessions": {
                "type": MetricType.GAUGE,
                "category": MetricCategory.SYSTEM,
                "description": "アクティブセッション数",
                "unit": "sessions"
            }
        }
    
    def record_metric(self, 
                     name: str, 
                     value: float, 
                     labels: Optional[Dict[str, str]] = None,
                     timestamp: Optional[datetime] = None):
        """メトリクス記録"""
        if timestamp is None:
            timestamp = datetime.now()
        
        if labels is None:
            labels = {}
        
        # メトリクス設定取得
        config = self.metric_configs.get(name, {})
        category = config.get("category", MetricCategory.SYSTEM)
        metric_type = config.get("type", MetricType.GAUGE)
        
        # 値の検証
        if not self._validate_metric_value(name, value):
            return False
        
        # メトリクスポイント作成
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=timestamp,
            labels=labels,
            category=category,
            metric_type=metric_type
        )
        
        # メトリクス追加
        self.metrics.append(point)
        
        # 統計更新
        self._update_stats(name, category, metric_type)
        
        # 古いメトリクス削除
        self._cleanup_old_metrics()
        
        return True
    
    def _validate_metric_value(self, name: str, value: float) -> bool:
        """メトリクス値の検証"""
        config = self.metric_configs.get(name, {})
        
        # 範囲チェック
        if "range" in config:
            min_val, max_val = config["range"]
            if not (min_val <= value <= max_val):
                return False
        
        # 型チェック
        if not isinstance(value, (int, float)):
            return False
        
        return True
    
    def _update_stats(self, name: str, category: MetricCategory, metric_type: MetricType):
        """統計更新"""
        self.stats["total_metrics"] += 1
        
        # カテゴリ別統計
        if category not in self.stats["metrics_by_category"]:
            self.stats["metrics_by_category"][category] = 0
        self.stats["metrics_by_category"][category] += 1
        
        # タイプ別統計
        if metric_type not in self.stats["metrics_by_type"]:
            self.stats["metrics_by_type"][metric_type] = 0
        self.stats["metrics_by_type"][metric_type] += 1
    
    def _cleanup_old_metrics(self):
        """古いメトリクス削除"""
        now = datetime.now()
        cutoff_time = now - timedelta(hours=self.retention_hours)
        
        # 古いメトリクスを削除
        self.metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
        
        # クリーンアップ時間更新
        self.stats["last_cleanup"] = now
    
    def record_quality_evaluation(self, evaluation: QualityEvaluation, labels: Optional[Dict[str, str]] = None):
        """品質評価メトリクス記録"""
        if labels is None:
            labels = {}
        
        timestamp = datetime.now()
        
        # 総合品質スコア
        self.record_metric("overall_quality", evaluation.overall_score, labels, timestamp)
        
        # 各次元のスコア
        for dimension, score in evaluation.dimension_scores.items():
            metric_name = f"{dimension.value}_score"
            self.record_metric(metric_name, score.score, labels, timestamp)
        
        # 評価時間
        self.record_metric("evaluation_time", evaluation.evaluation_time, labels, timestamp)
        
        # メタデータから追加メトリクス
        if "response_length" in evaluation.metadata:
            self.record_metric("response_length", evaluation.metadata["response_length"], labels, timestamp)
        
        if "step_count" in evaluation.metadata:
            self.record_metric("step_count", evaluation.metadata["step_count"], labels, timestamp)
        
        if "processing_time" in evaluation.metadata:
            self.record_metric("reasoning_time", evaluation.metadata["processing_time"], labels, timestamp)
        
        if "confidence" in evaluation.metadata:
            self.record_metric("confidence_score", evaluation.metadata["confidence"], labels, timestamp)
    
    def record_request_metrics(self, 
                              success: bool, 
                              processing_time: float,
                              labels: Optional[Dict[str, str]] = None):
        """リクエストメトリクス記録"""
        if labels is None:
            labels = {}
        
        timestamp = datetime.now()
        
        # リクエスト数
        self.record_metric("total_requests", 1, labels, timestamp)
        
        if success:
            self.record_metric("successful_requests", 1, labels, timestamp)
        else:
            self.record_metric("failed_requests", 1, labels, timestamp)
        
        # 処理時間
        self.record_metric("reasoning_time", processing_time, labels, timestamp)
        
        # エラー率計算
        self._update_error_rate()
    
    def _update_error_rate(self):
        """エラー率更新"""
        # 直近1時間のデータを取得
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        
        recent_metrics = [m for m in self.metrics if m.timestamp > one_hour_ago]
        
        total_requests = sum(1 for m in recent_metrics if m.name == "total_requests")
        failed_requests = sum(1 for m in recent_metrics if m.name == "failed_requests")
        
        if total_requests > 0:
            error_rate = (failed_requests / total_requests) * 100
            self.record_metric("error_rate", error_rate)
    
    def get_metric_summary(self, 
                          name: str, 
                          time_range: Optional[Tuple[datetime, datetime]] = None,
                          labels_filter: Optional[Dict[str, str]] = None) -> Optional[MetricSummary]:
        """メトリクスサマリー取得"""
        # フィルタリング
        filtered_metrics = self._filter_metrics(name, time_range, labels_filter)
        
        if not filtered_metrics:
            return None
        
        values = [m.value for m in filtered_metrics]
        timestamps = [m.timestamp for m in filtered_metrics]
        
        return MetricSummary(
            name=name,
            count=len(values),
            sum=sum(values),
            min=min(values),
            max=max(values),
            mean=statistics.mean(values),
            median=statistics.median(values),
            std_dev=statistics.stdev(values) if len(values) > 1 else 0.0,
            percentiles=self._calculate_percentiles(values),
            time_range=(min(timestamps), max(timestamps))
        )
    
    def _filter_metrics(self, 
                       name: str, 
                       time_range: Optional[Tuple[datetime, datetime]] = None,
                       labels_filter: Optional[Dict[str, str]] = None) -> List[MetricPoint]:
        """メトリクスフィルタリング"""
        filtered = [m for m in self.metrics if m.name == name]
        
        # 時間範囲フィルタ
        if time_range:
            start_time, end_time = time_range
            filtered = [m for m in filtered if start_time <= m.timestamp <= end_time]
        
        # ラベルフィルタ
        if labels_filter:
            filtered = [m for m in filtered if all(m.labels.get(k) == v for k, v in labels_filter.items())]
        
        return filtered
    
    def _calculate_percentiles(self, values: List[float]) -> Dict[str, float]:
        """パーセンタイル計算"""
        if not values:
            return {}
        
        sorted_values = sorted(values)
        percentiles = {}
        
        for p in [50, 90, 95, 99]:
            index = int((p / 100) * (len(sorted_values) - 1))
            percentiles[f"p{p}"] = sorted_values[index]
        
        return percentiles
    
    def get_metrics_by_category(self, category: MetricCategory) -> List[MetricPoint]:
        """カテゴリ別メトリクス取得"""
        return [m for m in self.metrics if m.category == category]
    
    def get_metrics_by_type(self, metric_type: MetricType) -> List[MetricPoint]:
        """タイプ別メトリクス取得"""
        return [m for m in self.metrics if m.metric_type == metric_type]
    
    def get_latest_metric(self, name: str, labels_filter: Optional[Dict[str, str]] = None) -> Optional[MetricPoint]:
        """最新メトリクス取得"""
        filtered = self._filter_metrics(name, labels_filter=labels_filter)
        if not filtered:
            return None
        
        return max(filtered, key=lambda m: m.timestamp)
    
    def get_metric_trend(self, 
                        name: str, 
                        time_range: Optional[Tuple[datetime, datetime]] = None,
                        labels_filter: Optional[Dict[str, str]] = None,
                        interval_minutes: int = 5) -> List[Tuple[datetime, float]]:
        """メトリクストレンド取得"""
        filtered = self._filter_metrics(name, time_range, labels_filter)
        
        if not filtered:
            return []
        
        # 時間間隔でグループ化
        interval = timedelta(minutes=interval_minutes)
        groups = {}
        
        for metric in filtered:
            # 時間を間隔に丸める
            rounded_time = metric.timestamp.replace(
                minute=(metric.timestamp.minute // interval_minutes) * interval_minutes,
                second=0,
                microsecond=0
            )
            
            if rounded_time not in groups:
                groups[rounded_time] = []
            groups[rounded_time].append(metric.value)
        
        # 各間隔の平均値を計算
        trend = []
        for time_point in sorted(groups.keys()):
            avg_value = statistics.mean(groups[time_point])
            trend.append((time_point, avg_value))
        
        return trend
    
    def export_metrics(self, 
                      format: str = "json",
                      time_range: Optional[Tuple[datetime, datetime]] = None) -> str:
        """メトリクスエクスポート"""
        if time_range:
            filtered_metrics = [m for m in self.metrics if time_range[0] <= m.timestamp <= time_range[1]]
        else:
            filtered_metrics = self.metrics
        
        if format.lower() == "json":
            data = {
                "metrics": [
                    {
                        "name": m.name,
                        "value": m.value,
                        "timestamp": m.timestamp.isoformat(),
                        "labels": m.labels,
                        "category": m.category.value,
                        "type": m.metric_type.value
                    }
                    for m in filtered_metrics
                ],
                "export_time": datetime.now().isoformat(),
                "total_count": len(filtered_metrics)
            }
            return json.dumps(data, indent=2, ensure_ascii=False)
        
        elif format.lower() == "prometheus":
            # Prometheus形式でのエクスポート
            lines = []
            for metric in filtered_metrics:
                labels_str = ",".join([f'{k}="{v}"' for k, v in metric.labels.items()])
                if labels_str:
                    labels_str = "{" + labels_str + "}"
                
                line = f"{metric.name}{labels_str} {metric.value} {int(metric.timestamp.timestamp() * 1000)}"
                lines.append(line)
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_collector_stats(self) -> Dict[str, Any]:
        """コレクター統計取得"""
        return {
            "total_metrics": len(self.metrics),
            "retention_hours": self.retention_hours,
            "stats": self.stats,
            "metric_configs_count": len(self.metric_configs),
            "oldest_metric": min(m.timestamp for m in self.metrics) if self.metrics else None,
            "newest_metric": max(m.timestamp for m in self.metrics) if self.metrics else None
        }
    
    def reset_metrics(self):
        """メトリクスリセット"""
        self.metrics.clear()
        self.stats = {
            "total_metrics": 0,
            "metrics_by_category": {},
            "metrics_by_type": {},
            "last_cleanup": datetime.now()
        }


# グローバルメトリクスコレクター
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """メトリクスコレクター取得（シングルトン）"""
    global _metrics_collector
    
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    
    return _metrics_collector


def record_metric(name: str, value: float, labels: Optional[Dict[str, str]] = None):
    """メトリクス記録（便利関数）"""
    collector = get_metrics_collector()
    return collector.record_metric(name, value, labels)


def record_quality_evaluation(evaluation: QualityEvaluation, labels: Optional[Dict[str, str]] = None):
    """品質評価メトリクス記録（便利関数）"""
    collector = get_metrics_collector()
    return collector.record_quality_evaluation(evaluation, labels)


def record_request_metrics(success: bool, processing_time: float, labels: Optional[Dict[str, str]] = None):
    """リクエストメトリクス記録（便利関数）"""
    collector = get_metrics_collector()
    return collector.record_request_metrics(success, processing_time, labels)


def get_metric_summary(name: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> Optional[MetricSummary]:
    """メトリクスサマリー取得（便利関数）"""
    collector = get_metrics_collector()
    return collector.get_metric_summary(name, time_range)


def get_metric_trend(name: str, time_range: Optional[Tuple[datetime, datetime]] = None, interval_minutes: int = 5) -> List[Tuple[datetime, float]]:
    """メトリクストレンド取得（便利関数）"""
    collector = get_metrics_collector()
    return collector.get_metric_trend(name, time_range, interval_minutes=interval_minutes)


# 使用例
if __name__ == "__main__":
    # メトリクスコレクター作成
    collector = MetricsCollector()
    
    # サンプルメトリクス記録
    print("=== Recording Sample Metrics ===")
    
    # 品質評価メトリクス
    from .quality_evaluator import QualityEvaluation, QualityScore, QualityDimension
    
    sample_evaluation = QualityEvaluation(
        overall_score=0.85,
        dimension_scores={
            QualityDimension.ACCURACY: QualityScore(QualityDimension.ACCURACY, 0.9, 0.8, "High accuracy"),
            QualityDimension.COMPLETENESS: QualityScore(QualityDimension.COMPLETENESS, 0.8, 0.7, "Good completeness"),
            QualityDimension.CLARITY: QualityScore(QualityDimension.CLARITY, 0.7, 0.6, "Moderate clarity")
        },
        strengths=["High accuracy", "Good structure"],
        weaknesses=["Could be clearer"],
        recommendations=["Improve clarity"],
        evaluation_time=0.5
    )
    
    collector.record_quality_evaluation(sample_evaluation, {"model": "qwen2:7b-instruct", "session": "test_123"})
    
    # リクエストメトリクス
    collector.record_request_metrics(True, 2.5, {"model": "qwen2:7b-instruct"})
    collector.record_request_metrics(False, 1.0, {"model": "qwen2:7b-instruct"})
    
    # システムメトリクス
    collector.record_metric("memory_usage", 512.5, {"component": "reasoning_engine"})
    collector.record_metric("cpu_usage", 45.2, {"component": "reasoning_engine"})
    
    print("Metrics recorded successfully")
    
    # メトリクスサマリー取得
    print("\n=== Metric Summaries ===")
    
    quality_summary = collector.get_metric_summary("overall_quality")
    if quality_summary:
        print(f"Overall Quality: {quality_summary.mean:.3f} (min: {quality_summary.min:.3f}, max: {quality_summary.max:.3f})")
    
    reasoning_time_summary = collector.get_metric_summary("reasoning_time")
    if reasoning_time_summary:
        print(f"Reasoning Time: {reasoning_time_summary.mean:.3f}s (min: {reasoning_time_summary.min:.3f}s, max: {reasoning_time_summary.max:.3f}s)")
    
    # メトリクストレンド取得
    print("\n=== Metric Trends ===")
    
    quality_trend = collector.get_metric_trend("overall_quality")
    print(f"Quality trend points: {len(quality_trend)}")
    
    # 統計情報
    print("\n=== Collector Statistics ===")
    stats = collector.get_collector_stats()
    print(f"Total metrics: {stats['total_metrics']}")
    print(f"Metrics by category: {stats['stats']['metrics_by_category']}")
    print(f"Metrics by type: {stats['stats']['metrics_by_type']}")
    
    # エクスポート
    print("\n=== Export Metrics ===")
    exported = collector.export_metrics("json")
    print(f"Exported {len(exported)} characters of metrics data")
