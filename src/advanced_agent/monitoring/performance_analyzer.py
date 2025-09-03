"""
Performance Analyzer and Bottleneck Detection

システム性能分析とボトルネック検出機能
RTX 4050 6GB VRAM環境での最適化提案を提供します。

要件: 4.1, 4.2, 4.5
"""

import asyncio
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import logging

from .system_monitor import SystemMonitor, SystemMetrics, AlertLevel

logger = logging.getLogger(__name__)


class BottleneckType(Enum):
    """ボトルネックタイプ"""
    CPU = "cpu"
    MEMORY = "memory"
    GPU_COMPUTE = "gpu_compute"
    GPU_MEMORY = "gpu_memory"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    THERMAL = "thermal"


class PerformanceLevel(Enum):
    """パフォーマンスレベル"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class BottleneckAnalysis:
    """ボトルネック分析結果"""
    bottleneck_type: BottleneckType
    severity: float  # 0.0-1.0
    description: str
    current_value: float
    threshold_value: float
    impact_score: float
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceReport:
    """パフォーマンスレポート"""
    overall_score: float  # 0.0-1.0
    performance_level: PerformanceLevel
    bottlenecks: List[BottleneckAnalysis]
    resource_utilization: Dict[str, float]
    efficiency_metrics: Dict[str, float]
    optimization_suggestions: List[str]
    analysis_period: timedelta
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TrendAnalysis:
    """トレンド分析"""
    metric_name: str
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_strength: float  # 0.0-1.0
    average_value: float
    min_value: float
    max_value: float
    variance: float
    prediction: Optional[float] = None


class PerformanceAnalyzer:
    """パフォーマンス分析クラス"""
    
    def __init__(self,
                 system_monitor: SystemMonitor,
                 analysis_window: int = 300,  # 5分間
                 rtx4050_optimized: bool = True):
        """
        初期化
        
        Args:
            system_monitor: システム監視インスタンス
            analysis_window: 分析ウィンドウ（秒）
            rtx4050_optimized: RTX 4050 最適化を有効にするか
        """
        self.system_monitor = system_monitor
        self.analysis_window = analysis_window
        self.rtx4050_optimized = rtx4050_optimized
        
        # RTX 4050 固有の閾値設定
        if rtx4050_optimized:
            self.thresholds = {
                "cpu_usage": 75.0,
                "memory_usage": 80.0,
                "gpu_memory_usage": 85.0,  # 6GB の 85% = 5.1GB
                "gpu_utilization": 90.0,
                "gpu_temperature": 75.0,  # RTX 4050 の安全温度
                "disk_usage": 85.0,
                "thermal_throttling": 80.0
            }
        else:
            self.thresholds = {
                "cpu_usage": 80.0,
                "memory_usage": 85.0,
                "gpu_memory_usage": 90.0,
                "gpu_utilization": 95.0,
                "gpu_temperature": 80.0,
                "disk_usage": 90.0,
                "thermal_throttling": 85.0
            }
        
        # 分析履歴
        self.analysis_history: List[PerformanceReport] = []
        self.max_history_size = 100
        
        # 統計
        self.analysis_count = 0
        self.last_analysis_time: Optional[datetime] = None
    
    def analyze_current_performance(self) -> PerformanceReport:
        """現在のパフォーマンス分析"""
        try:
            # 最新メトリクス取得
            latest_metrics = self.system_monitor.get_latest_metrics()
            if not latest_metrics:
                raise ValueError("No metrics available for analysis")
            
            # 履歴メトリクス取得（分析ウィンドウ内）
            history = self._get_analysis_window_metrics()
            
            # ボトルネック検出
            bottlenecks = self._detect_bottlenecks(latest_metrics, history)
            
            # リソース使用率計算
            resource_utilization = self._calculate_resource_utilization(latest_metrics)
            
            # 効率性メトリクス計算
            efficiency_metrics = self._calculate_efficiency_metrics(latest_metrics, history)
            
            # 総合スコア計算
            overall_score = self._calculate_overall_score(latest_metrics, bottlenecks)
            
            # パフォーマンスレベル判定
            performance_level = self._determine_performance_level(overall_score)
            
            # 最適化提案生成
            optimization_suggestions = self._generate_optimization_suggestions(
                bottlenecks, latest_metrics, history
            )
            
            report = PerformanceReport(
                overall_score=overall_score,
                performance_level=performance_level,
                bottlenecks=bottlenecks,
                resource_utilization=resource_utilization,
                efficiency_metrics=efficiency_metrics,
                optimization_suggestions=optimization_suggestions,
                analysis_period=timedelta(seconds=self.analysis_window)
            )
            
            # 履歴に追加
            self.analysis_history.append(report)
            if len(self.analysis_history) > self.max_history_size:
                self.analysis_history.pop(0)
            
            # 統計更新
            self.analysis_count += 1
            self.last_analysis_time = datetime.now()
            
            return report
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return self._create_error_report(str(e))
    
    def _get_analysis_window_metrics(self) -> List[SystemMetrics]:
        """分析ウィンドウ内のメトリクス取得"""
        all_metrics = self.system_monitor.get_metrics_history()
        if not all_metrics:
            return []
        
        # 分析ウィンドウ内のメトリクスをフィルタ
        cutoff_time = datetime.now() - timedelta(seconds=self.analysis_window)
        return [m for m in all_metrics if m.timestamp >= cutoff_time]
    
    def _detect_bottlenecks(self,
                          current_metrics: SystemMetrics,
                          history: List[SystemMetrics]) -> List[BottleneckAnalysis]:
        """ボトルネック検出"""
        bottlenecks = []
        
        # CPU ボトルネック
        if current_metrics.cpu.usage_percent > self.thresholds["cpu_usage"]:
            severity = min(1.0, current_metrics.cpu.usage_percent / 100.0)
            bottlenecks.append(BottleneckAnalysis(
                bottleneck_type=BottleneckType.CPU,
                severity=severity,
                description=f"High CPU usage: {current_metrics.cpu.usage_percent:.1f}%",
                current_value=current_metrics.cpu.usage_percent,
                threshold_value=self.thresholds["cpu_usage"],
                impact_score=self._calculate_cpu_impact(current_metrics, history),
                recommendations=self._get_cpu_recommendations(current_metrics)
            ))
        
        # メモリボトルネック
        if current_metrics.memory.usage_percent > self.thresholds["memory_usage"]:
            severity = min(1.0, current_metrics.memory.usage_percent / 100.0)
            bottlenecks.append(BottleneckAnalysis(
                bottleneck_type=BottleneckType.MEMORY,
                severity=severity,
                description=f"High memory usage: {current_metrics.memory.usage_percent:.1f}%",
                current_value=current_metrics.memory.usage_percent,
                threshold_value=self.thresholds["memory_usage"],
                impact_score=self._calculate_memory_impact(current_metrics, history),
                recommendations=self._get_memory_recommendations(current_metrics)
            ))
        
        # GPU ボトルネック
        if current_metrics.gpu:
            # GPU メモリ
            if current_metrics.gpu.memory_percent > self.thresholds["gpu_memory_usage"]:
                severity = min(1.0, current_metrics.gpu.memory_percent / 100.0)
                bottlenecks.append(BottleneckAnalysis(
                    bottleneck_type=BottleneckType.GPU_MEMORY,
                    severity=severity,
                    description=f"High GPU memory usage: {current_metrics.gpu.memory_percent:.1f}%",
                    current_value=current_metrics.gpu.memory_percent,
                    threshold_value=self.thresholds["gpu_memory_usage"],
                    impact_score=self._calculate_gpu_memory_impact(current_metrics, history),
                    recommendations=self._get_gpu_memory_recommendations(current_metrics)
                ))
            
            # GPU 使用率
            if current_metrics.gpu.utilization_percent > self.thresholds["gpu_utilization"]:
                severity = min(1.0, current_metrics.gpu.utilization_percent / 100.0)
                bottlenecks.append(BottleneckAnalysis(
                    bottleneck_type=BottleneckType.GPU_COMPUTE,
                    severity=severity,
                    description=f"High GPU utilization: {current_metrics.gpu.utilization_percent:.1f}%",
                    current_value=current_metrics.gpu.utilization_percent,
                    threshold_value=self.thresholds["gpu_utilization"],
                    impact_score=self._calculate_gpu_compute_impact(current_metrics, history),
                    recommendations=self._get_gpu_compute_recommendations(current_metrics)
                ))
            
            # GPU 温度
            if current_metrics.gpu.temperature_celsius > self.thresholds["gpu_temperature"]:
                severity = min(1.0, current_metrics.gpu.temperature_celsius / 100.0)
                bottlenecks.append(BottleneckAnalysis(
                    bottleneck_type=BottleneckType.THERMAL,
                    severity=severity,
                    description=f"High GPU temperature: {current_metrics.gpu.temperature_celsius:.1f}°C",
                    current_value=current_metrics.gpu.temperature_celsius,
                    threshold_value=self.thresholds["gpu_temperature"],
                    impact_score=self._calculate_thermal_impact(current_metrics, history),
                    recommendations=self._get_thermal_recommendations(current_metrics)
                ))
        
        # ディスクボトルネック
        for device, usage in current_metrics.disk_usage.items():
            if usage > self.thresholds["disk_usage"]:
                severity = min(1.0, usage / 100.0)
                bottlenecks.append(BottleneckAnalysis(
                    bottleneck_type=BottleneckType.DISK_IO,
                    severity=severity,
                    description=f"High disk usage on {device}: {usage:.1f}%",
                    current_value=usage,
                    threshold_value=self.thresholds["disk_usage"],
                    impact_score=0.6,  # 固定値
                    recommendations=self._get_disk_recommendations(device, usage)
                ))
        
        return sorted(bottlenecks, key=lambda x: x.severity, reverse=True)
    
    def _calculate_resource_utilization(self, metrics: SystemMetrics) -> Dict[str, float]:
        """リソース使用率計算"""
        utilization = {
            "cpu": metrics.cpu.usage_percent,
            "memory": metrics.memory.usage_percent,
            "swap": metrics.memory.swap_percent
        }
        
        if metrics.gpu:
            utilization.update({
                "gpu_compute": metrics.gpu.utilization_percent,
                "gpu_memory": metrics.gpu.memory_percent
            })
        
        return utilization
    
    def _calculate_efficiency_metrics(self,
                                    current_metrics: SystemMetrics,
                                    history: List[SystemMetrics]) -> Dict[str, float]:
        """効率性メトリクス計算"""
        efficiency = {}
        
        if not history:
            return {"data_insufficient": 0.0}
        
        # CPU 効率性
        cpu_usages = [m.cpu.usage_percent for m in history]
        cpu_variance = statistics.variance(cpu_usages) if len(cpu_usages) > 1 else 0.0
        efficiency["cpu_stability"] = max(0.0, 1.0 - (cpu_variance / 100.0))
        
        # メモリ効率性
        memory_usages = [m.memory.usage_percent for m in history]
        memory_variance = statistics.variance(memory_usages) if len(memory_usages) > 1 else 0.0
        efficiency["memory_stability"] = max(0.0, 1.0 - (memory_variance / 100.0))
        
        # GPU 効率性
        if current_metrics.gpu and all(m.gpu for m in history):
            gpu_usages = [m.gpu.utilization_percent for m in history if m.gpu]
            if gpu_usages:
                gpu_variance = statistics.variance(gpu_usages) if len(gpu_usages) > 1 else 0.0
                efficiency["gpu_stability"] = max(0.0, 1.0 - (gpu_variance / 100.0))
                
                # GPU メモリ効率性
                gpu_mem_usages = [m.gpu.memory_percent for m in history if m.gpu]
                if gpu_mem_usages:
                    avg_gpu_mem = statistics.mean(gpu_mem_usages)
                    efficiency["gpu_memory_efficiency"] = min(1.0, avg_gpu_mem / 85.0)  # 85%を最適とする
        
        # 全体効率性
        if efficiency:
            efficiency["overall_efficiency"] = statistics.mean(efficiency.values())
        
        return efficiency
    
    def _calculate_overall_score(self,
                               metrics: SystemMetrics,
                               bottlenecks: List[BottleneckAnalysis]) -> float:
        """総合スコア計算"""
        base_score = 1.0
        
        # ボトルネックによる減点
        for bottleneck in bottlenecks:
            penalty = bottleneck.severity * bottleneck.impact_score * 0.3
            base_score -= penalty
        
        # リソース使用率による調整
        resource_penalty = 0.0
        if metrics.cpu.usage_percent > 90:
            resource_penalty += 0.1
        if metrics.memory.usage_percent > 90:
            resource_penalty += 0.1
        if metrics.gpu and metrics.gpu.memory_percent > 90:
            resource_penalty += 0.2  # GPU メモリは重要
        
        base_score -= resource_penalty
        
        return max(0.0, min(1.0, base_score))
    
    def _determine_performance_level(self, score: float) -> PerformanceLevel:
        """パフォーマンスレベル判定"""
        if score >= 0.9:
            return PerformanceLevel.EXCELLENT
        elif score >= 0.7:
            return PerformanceLevel.GOOD
        elif score >= 0.5:
            return PerformanceLevel.FAIR
        elif score >= 0.3:
            return PerformanceLevel.POOR
        else:
            return PerformanceLevel.CRITICAL
    
    def _generate_optimization_suggestions(self,
                                         bottlenecks: List[BottleneckAnalysis],
                                         current_metrics: SystemMetrics,
                                         history: List[SystemMetrics]) -> List[str]:
        """最適化提案生成"""
        suggestions = []
        
        # ボトルネック別提案
        for bottleneck in bottlenecks:
            suggestions.extend(bottleneck.recommendations)
        
        # RTX 4050 固有の提案
        if self.rtx4050_optimized and current_metrics.gpu:
            if current_metrics.gpu.memory_percent > 80:
                suggestions.append("RTX 4050の6GB VRAM制限を考慮し、モデルの量子化レベルを上げることを検討してください")
            
            if current_metrics.gpu.temperature_celsius > 70:
                suggestions.append("RTX 4050の温度管理のため、GPU負荷を一時的にCPUにオフロードすることを検討してください")
        
        # 一般的な提案
        if current_metrics.memory.usage_percent > 85:
            suggestions.append("メモリ使用量が高いため、不要なプロセスを終了することを検討してください")
        
        if current_metrics.cpu.usage_percent > 85:
            suggestions.append("CPU使用率が高いため、並列処理の最適化を検討してください")
        
        return list(set(suggestions))  # 重複除去
    
    # 影響度計算メソッド
    def _calculate_cpu_impact(self, metrics: SystemMetrics, history: List[SystemMetrics]) -> float:
        """CPU影響度計算"""
        base_impact = 0.7
        if metrics.cpu.usage_percent > 95:
            base_impact = 0.9
        return base_impact
    
    def _calculate_memory_impact(self, metrics: SystemMetrics, history: List[SystemMetrics]) -> float:
        """メモリ影響度計算"""
        base_impact = 0.8
        if metrics.memory.swap_percent > 50:
            base_impact = 0.95  # スワップ使用は深刻
        return base_impact
    
    def _calculate_gpu_memory_impact(self, metrics: SystemMetrics, history: List[SystemMetrics]) -> float:
        """GPU メモリ影響度計算"""
        # RTX 4050 では GPU メモリが最重要
        return 0.95 if self.rtx4050_optimized else 0.8
    
    def _calculate_gpu_compute_impact(self, metrics: SystemMetrics, history: List[SystemMetrics]) -> float:
        """GPU 計算影響度計算"""
        return 0.7
    
    def _calculate_thermal_impact(self, metrics: SystemMetrics, history: List[SystemMetrics]) -> float:
        """熱影響度計算"""
        if metrics.gpu and metrics.gpu.temperature_celsius > 85:
            return 0.9  # 高温は深刻
        return 0.6
    
    # 推奨事項生成メソッド
    def _get_cpu_recommendations(self, metrics: SystemMetrics) -> List[str]:
        """CPU推奨事項"""
        recommendations = []
        if metrics.cpu.usage_percent > 90:
            recommendations.extend([
                "CPU集約的なタスクを分散または延期する",
                "並列処理の最適化を検討する",
                "不要なバックグラウンドプロセスを終了する"
            ])
        return recommendations
    
    def _get_memory_recommendations(self, metrics: SystemMetrics) -> List[str]:
        """メモリ推奨事項"""
        recommendations = []
        if metrics.memory.usage_percent > 85:
            recommendations.extend([
                "メモリ使用量の多いプロセスを特定し最適化する",
                "不要なアプリケーションを終了する",
                "メモリリークの可能性を調査する"
            ])
        if metrics.memory.swap_percent > 20:
            recommendations.append("スワップ使用量が多いため、物理メモリの増設を検討する")
        return recommendations
    
    def _get_gpu_memory_recommendations(self, metrics: SystemMetrics) -> List[str]:
        """GPU メモリ推奨事項"""
        recommendations = []
        if self.rtx4050_optimized:
            recommendations.extend([
                "モデルの量子化レベルを上げる（8bit→4bit→3bit）",
                "バッチサイズを削減する",
                "不要なGPUメモリを解放する",
                "CPU オフロードを活用する"
            ])
        else:
            recommendations.extend([
                "GPU メモリ使用量を最適化する",
                "モデルサイズを削減する",
                "バッチ処理を最適化する"
            ])
        return recommendations
    
    def _get_gpu_compute_recommendations(self, metrics: SystemMetrics) -> List[str]:
        """GPU 計算推奨事項"""
        return [
            "GPU 負荷を分散する",
            "計算効率を最適化する",
            "不要な GPU 処理を削減する"
        ]
    
    def _get_thermal_recommendations(self, metrics: SystemMetrics) -> List[str]:
        """熱管理推奨事項"""
        recommendations = []
        if metrics.gpu and metrics.gpu.temperature_celsius > 80:
            recommendations.extend([
                "GPU 負荷を一時的に削減する",
                "冷却システムを確認する",
                "処理を CPU にオフロードする",
                "ファン速度を上げる"
            ])
        return recommendations
    
    def _get_disk_recommendations(self, device: str, usage: float) -> List[str]:
        """ディスク推奨事項"""
        return [
            f"{device} の不要なファイルを削除する",
            "ディスク容量を増やす",
            "一時ファイルをクリーンアップする"
        ]
    
    def _create_error_report(self, error_message: str) -> PerformanceReport:
        """エラーレポート作成"""
        return PerformanceReport(
            overall_score=0.0,
            performance_level=PerformanceLevel.CRITICAL,
            bottlenecks=[],
            resource_utilization={},
            efficiency_metrics={},
            optimization_suggestions=[f"Analysis error: {error_message}"],
            analysis_period=timedelta(seconds=0)
        )
    
    def analyze_trends(self, metric_name: str, lookback_hours: int = 1) -> Optional[TrendAnalysis]:
        """トレンド分析"""
        try:
            # 指定期間のメトリクス取得
            cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
            history = [m for m in self.system_monitor.get_metrics_history() 
                      if m.timestamp >= cutoff_time]
            
            if len(history) < 3:
                return None
            
            # メトリクス値抽出
            values = []
            for metrics in history:
                if metric_name == "cpu_usage":
                    values.append(metrics.cpu.usage_percent)
                elif metric_name == "memory_usage":
                    values.append(metrics.memory.usage_percent)
                elif metric_name == "gpu_memory_usage" and metrics.gpu:
                    values.append(metrics.gpu.memory_percent)
                elif metric_name == "gpu_utilization" and metrics.gpu:
                    values.append(metrics.gpu.utilization_percent)
            
            if not values:
                return None
            
            # 統計計算
            avg_value = statistics.mean(values)
            min_value = min(values)
            max_value = max(values)
            variance = statistics.variance(values) if len(values) > 1 else 0.0
            
            # トレンド方向判定
            if len(values) >= 2:
                recent_avg = statistics.mean(values[-len(values)//3:])
                early_avg = statistics.mean(values[:len(values)//3])
                
                if recent_avg > early_avg * 1.1:
                    trend_direction = "increasing"
                    trend_strength = min(1.0, (recent_avg - early_avg) / early_avg)
                elif recent_avg < early_avg * 0.9:
                    trend_direction = "decreasing"
                    trend_strength = min(1.0, (early_avg - recent_avg) / early_avg)
                else:
                    trend_direction = "stable"
                    trend_strength = max(0.0, 1.0 - variance / 100.0)
            else:
                trend_direction = "stable"
                trend_strength = 0.5
            
            return TrendAnalysis(
                metric_name=metric_name,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                average_value=avg_value,
                min_value=min_value,
                max_value=max_value,
                variance=variance
            )
            
        except Exception as e:
            logger.error(f"Trend analysis failed for {metric_name}: {e}")
            return None
    
    def get_analysis_history(self, limit: Optional[int] = None) -> List[PerformanceReport]:
        """分析履歴取得"""
        if limit is None:
            return self.analysis_history.copy()
        return self.analysis_history[-limit:].copy()
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """分析統計取得"""
        return {
            "analysis_count": self.analysis_count,
            "last_analysis_time": self.last_analysis_time,
            "history_size": len(self.analysis_history),
            "rtx4050_optimized": self.rtx4050_optimized,
            "analysis_window_seconds": self.analysis_window
        }


# 使用例とテスト用のヘルパー関数
async def demo_performance_analysis():
    """デモ用パフォーマンス分析実行"""
    from .system_monitor import SystemMonitor
    
    # システム監視初期化
    system_monitor = SystemMonitor(collection_interval=1.0)
    
    # パフォーマンス分析器初期化
    analyzer = PerformanceAnalyzer(
        system_monitor,
        analysis_window=60,  # 1分間
        rtx4050_optimized=True
    )
    
    try:
        print("=== Performance Analysis Demo ===")
        
        # 監視開始
        await system_monitor.start_monitoring()
        
        # データ収集のため少し待機
        print("🔄 Collecting baseline data for 30 seconds...")
        await asyncio.sleep(30)
        
        # パフォーマンス分析実行
        print("\n📊 Analyzing current performance...")
        report = analyzer.analyze_current_performance()
        
        print(f"\n📈 Performance Report:")
        print(f"  Overall Score: {report.overall_score:.2f}")
        print(f"  Performance Level: {report.performance_level.value}")
        print(f"  Analysis Period: {report.analysis_period}")
        
        print(f"\n🔍 Resource Utilization:")
        for resource, usage in report.resource_utilization.items():
            print(f"  {resource}: {usage:.1f}%")
        
        print(f"\n⚡ Efficiency Metrics:")
        for metric, value in report.efficiency_metrics.items():
            print(f"  {metric}: {value:.2f}")
        
        if report.bottlenecks:
            print(f"\n🚨 Detected Bottlenecks ({len(report.bottlenecks)}):")
            for i, bottleneck in enumerate(report.bottlenecks[:3], 1):
                print(f"  {i}. {bottleneck.bottleneck_type.value}: {bottleneck.description}")
                print(f"     Severity: {bottleneck.severity:.2f}, Impact: {bottleneck.impact_score:.2f}")
        else:
            print("\n✅ No significant bottlenecks detected")
        
        if report.optimization_suggestions:
            print(f"\n💡 Optimization Suggestions:")
            for i, suggestion in enumerate(report.optimization_suggestions[:5], 1):
                print(f"  {i}. {suggestion}")
        
        # トレンド分析
        print(f"\n📈 Trend Analysis:")
        for metric in ["cpu_usage", "memory_usage", "gpu_memory_usage"]:
            trend = analyzer.analyze_trends(metric, lookback_hours=1)
            if trend:
                print(f"  {metric}: {trend.trend_direction} (strength: {trend.trend_strength:.2f})")
        
        # 統計表示
        stats = analyzer.get_analysis_stats()
        print(f"\n📊 Analysis Statistics:")
        print(f"  Total Analyses: {stats['analysis_count']}")
        print(f"  RTX 4050 Optimized: {stats['rtx4050_optimized']}")
        print(f"  Analysis Window: {stats['analysis_window_seconds']}s")
        
    finally:
        await system_monitor.cleanup()


if __name__ == "__main__":
    asyncio.run(demo_performance_analysis())