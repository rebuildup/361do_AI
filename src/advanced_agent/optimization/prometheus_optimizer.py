"""
Prometheus Metrics Based Optimizer

Prometheus メトリクスによる 動的リソース配分システム
RTX 4050 6GB VRAM環境での最適化学習を提供します。

要件: 4.2, 4.4, 4.5
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Tuple
import logging
import statistics

from ..monitoring.prometheus_collector import PrometheusMetricsCollector
from ..monitoring.system_monitor import SystemMonitor, SystemMetrics
from .auto_optimizer import OptimizationResult, OptimizationStrategy

logger = logging.getLogger(__name__)


class OptimizationRule(Enum):
    """最適化ルール"""
    VRAM_THRESHOLD = "vram_threshold"
    TEMPERATURE_THRESHOLD = "temperature_threshold"
    CPU_THRESHOLD = "cpu_threshold"
    MEMORY_THRESHOLD = "memory_threshold"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_IMBALANCE = "resource_imbalance"


@dataclass
class MetricRule:
    """メトリクスルール"""
    rule_type: OptimizationRule
    metric_name: str
    threshold_value: float
    comparison: str  # "gt", "lt", "eq"
    action: Dict[str, Any]
    priority: int = 1
    enabled: bool = True
    cooldown_seconds: float = 60.0
    last_triggered: Optional[datetime] = None


@dataclass
class OptimizationLearning:
    """最適化学習データ"""
    pattern_id: str
    conditions: Dict[str, float]
    actions_taken: Dict[str, Any]
    effectiveness_score: float
    usage_count: int = 0
    last_used: Optional[datetime] = None
    success_rate: float = 0.0


class MetricBasedOptimizer:
    """メトリクスベース最適化器"""
    
    def __init__(self,
                 prometheus_collector: PrometheusMetricsCollector,
                 auto_optimizer: 'AutoOptimizer'):
        """
        初期化
        
        Args:
            prometheus_collector: Prometheus メトリクス収集器
            auto_optimizer: 自動最適化器
        """
        self.prometheus_collector = prometheus_collector
        self.auto_optimizer = auto_optimizer
        
        # 最適化ルール
        self.optimization_rules: List[MetricRule] = []
        self._setup_default_rules()
        
        # 学習データ
        self.learning_patterns: Dict[str, OptimizationLearning] = {}
        self.max_patterns = 100
        
        # 実行状態
        self.is_running = False
        self.optimization_task: Optional[asyncio.Task] = None
        
        # 統計
        self.rules_triggered = 0
        self.optimizations_applied = 0
        self.learning_updates = 0
    
    def _setup_default_rules(self):
        """デフォルトルール設定"""
        # RTX 4050 VRAM ルール
        self.optimization_rules.extend([
            MetricRule(
                rule_type=OptimizationRule.VRAM_THRESHOLD,
                metric_name="gpu_memory_usage_percent",
                threshold_value=85.0,
                comparison="gt",
                action={"quantization_level": "4bit", "batch_size": 2},
                priority=1,
                cooldown_seconds=30.0
            ),
            MetricRule(
                rule_type=OptimizationRule.VRAM_THRESHOLD,
                metric_name="gpu_memory_usage_percent",
                threshold_value=90.0,
                comparison="gt",
                action={"quantization_level": "3bit", "batch_size": 1, "enable_cpu_offload": True},
                priority=0,  # 最高優先度
                cooldown_seconds=15.0
            ),
            MetricRule(
                rule_type=OptimizationRule.TEMPERATURE_THRESHOLD,
                metric_name="gpu_temperature_celsius",
                threshold_value=75.0,
                comparison="gt",
                action={"enable_cpu_offload": True, "reduce_clock_speed": True},
                priority=1,
                cooldown_seconds=45.0
            ),
            MetricRule(
                rule_type=OptimizationRule.CPU_THRESHOLD,
                metric_name="cpu_usage_percent",
                threshold_value=85.0,
                comparison="gt",
                action={"prefer_gpu_processing": True, "parallel_processing": False},
                priority=2,
                cooldown_seconds=60.0
            ),
            MetricRule(
                rule_type=OptimizationRule.MEMORY_THRESHOLD,
                metric_name="memory_usage_percent",
                threshold_value=85.0,
                comparison="gt",
                action={"gradient_checkpointing": True, "mixed_precision": True},
                priority=2,
                cooldown_seconds=60.0
            )
        ])
    
    async def start_monitoring(self, check_interval: float = 10.0):
        """監視開始"""
        if self.is_running:
            logger.warning("Metric-based optimization already running")
            return
        
        self.is_running = True
        self.optimization_task = asyncio.create_task(self._monitoring_loop(check_interval))
        logger.info(f"Metric-based optimization started (interval: {check_interval}s)")
    
    async def stop_monitoring(self):
        """監視停止"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.optimization_task:
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Metric-based optimization stopped")
    
    async def _monitoring_loop(self, check_interval: float):
        """監視ループ"""
        try:
            while self.is_running:
                try:
                    # メトリクス取得
                    current_metrics = await self._get_current_metrics()
                    
                    if current_metrics:
                        # ルール評価
                        triggered_rules = self._evaluate_rules(current_metrics)
                        
                        if triggered_rules:
                            # 最適化実行
                            await self._execute_optimizations(triggered_rules, current_metrics)
                        
                        # 学習パターン更新
                        await self._update_learning_patterns(current_metrics)
                
                except Exception as e:
                    logger.error(f"Monitoring loop error: {e}")
                
                await asyncio.sleep(check_interval)
                
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Monitoring loop error: {e}")
            self.is_running = False
    
    async def _get_current_metrics(self) -> Optional[Dict[str, float]]:
        """現在のメトリクス取得"""
        try:
            # システムメトリクス取得
            system_metrics = self.auto_optimizer.system_monitor.get_latest_metrics()
            if not system_metrics:
                return None
            
            # メトリクス辞書作成
            metrics = {
                "cpu_usage_percent": system_metrics.cpu.usage_percent,
                "memory_usage_percent": system_metrics.memory.usage_percent,
                "process_count": float(system_metrics.process_count)
            }
            
            if system_metrics.gpu:
                metrics.update({
                    "gpu_utilization_percent": system_metrics.gpu.utilization_percent,
                    "gpu_memory_usage_percent": system_metrics.gpu.memory_percent,
                    "gpu_temperature_celsius": system_metrics.gpu.temperature_celsius,
                    "gpu_power_draw_watts": system_metrics.gpu.power_draw_watts
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Metrics retrieval failed: {e}")
            return None
    
    def _evaluate_rules(self, metrics: Dict[str, float]) -> List[MetricRule]:
        """ルール評価"""
        triggered_rules = []
        
        try:
            current_time = datetime.now()
            
            for rule in self.optimization_rules:
                if not rule.enabled:
                    continue
                
                # クールダウンチェック
                if (rule.last_triggered and 
                    (current_time - rule.last_triggered).total_seconds() < rule.cooldown_seconds):
                    continue
                
                # メトリクス値取得
                metric_value = metrics.get(rule.metric_name)
                if metric_value is None:
                    continue
                
                # 条件評価
                condition_met = False
                if rule.comparison == "gt":
                    condition_met = metric_value > rule.threshold_value
                elif rule.comparison == "lt":
                    condition_met = metric_value < rule.threshold_value
                elif rule.comparison == "eq":
                    condition_met = abs(metric_value - rule.threshold_value) < 0.1
                
                if condition_met:
                    rule.last_triggered = current_time
                    triggered_rules.append(rule)
                    self.rules_triggered += 1
                    
                    logger.info(f"Rule triggered: {rule.rule_type.value} "
                              f"({rule.metric_name}: {metric_value} {rule.comparison} {rule.threshold_value})")
            
            # 優先度でソート
            triggered_rules.sort(key=lambda r: r.priority)
            
            return triggered_rules
            
        except Exception as e:
            logger.error(f"Rule evaluation failed: {e}")
            return []
    
    async def _execute_optimizations(self, 
                                   triggered_rules: List[MetricRule],
                                   current_metrics: Dict[str, float]):
        """最適化実行"""
        try:
            # 学習パターンから最適なアクションを予測
            predicted_actions = await self._predict_optimal_actions(current_metrics)
            
            # ルールベースアクションと学習ベースアクションを統合
            combined_actions = {}
            
            # ルールベースアクション
            for rule in triggered_rules[:3]:  # 上位3つのルールのみ
                combined_actions.update(rule.action)
            
            # 学習ベースアクション（重複しない場合のみ）
            for key, value in predicted_actions.items():
                if key not in combined_actions:
                    combined_actions[key] = value
            
            if combined_actions:
                # パラメータ設定
                self.auto_optimizer.set_parameters(combined_actions)
                self.optimizations_applied += 1
                
                logger.info(f"Applied optimizations: {combined_actions}")
                
                # 効果測定のため少し待機
                await asyncio.sleep(5.0)
                
                # 効果測定
                effectiveness = await self._measure_effectiveness(
                    current_metrics, combined_actions
                )
                
                # 学習データ更新
                await self._record_optimization_result(
                    current_metrics, combined_actions, effectiveness
                )
        
        except Exception as e:
            logger.error(f"Optimization execution failed: {e}")
    
    async def _predict_optimal_actions(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """学習パターンから最適アクション予測"""
        try:
            if not self.learning_patterns:
                return {}
            
            # 現在の状況に最も類似するパターンを検索
            best_pattern = None
            best_similarity = 0.0
            
            for pattern in self.learning_patterns.values():
                similarity = self._calculate_pattern_similarity(metrics, pattern.conditions)
                
                # 効果性と成功率を考慮
                weighted_score = similarity * pattern.effectiveness_score * pattern.success_rate
                
                if weighted_score > best_similarity:
                    best_similarity = weighted_score
                    best_pattern = pattern
            
            if best_pattern and best_similarity > 0.7:  # 70%以上の類似度
                best_pattern.usage_count += 1
                best_pattern.last_used = datetime.now()
                
                logger.info(f"Using learned pattern: {best_pattern.pattern_id} "
                          f"(similarity: {best_similarity:.2f})")
                
                return best_pattern.actions_taken.copy()
            
            return {}
            
        except Exception as e:
            logger.error(f"Action prediction failed: {e}")
            return {}
    
    def _calculate_pattern_similarity(self, 
                                    current_metrics: Dict[str, float],
                                    pattern_conditions: Dict[str, float]) -> float:
        """パターン類似度計算"""
        try:
            if not pattern_conditions:
                return 0.0
            
            similarities = []
            
            for metric_name, pattern_value in pattern_conditions.items():
                current_value = current_metrics.get(metric_name)
                if current_value is None:
                    continue
                
                # 相対的類似度計算
                max_val = max(abs(current_value), abs(pattern_value), 1.0)
                diff = abs(current_value - pattern_value)
                similarity = max(0.0, 1.0 - (diff / max_val))
                similarities.append(similarity)
            
            return statistics.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    async def _measure_effectiveness(self,
                                   before_metrics: Dict[str, float],
                                   actions: Dict[str, Any]) -> float:
        """最適化効果測定"""
        try:
            # 最適化後のメトリクス取得
            await asyncio.sleep(10.0)  # 効果が現れるまで待機
            after_metrics = await self._get_current_metrics()
            
            if not after_metrics:
                return 0.0
            
            # 改善度計算
            improvements = []
            
            # VRAM使用率改善
            if "gpu_memory_usage_percent" in before_metrics and "gpu_memory_usage_percent" in after_metrics:
                before_vram = before_metrics["gpu_memory_usage_percent"]
                after_vram = after_metrics["gpu_memory_usage_percent"]
                if before_vram > 80:  # 高使用率の場合のみ評価
                    improvement = (before_vram - after_vram) / before_vram
                    improvements.append(improvement)
            
            # 温度改善
            if "gpu_temperature_celsius" in before_metrics and "gpu_temperature_celsius" in after_metrics:
                before_temp = before_metrics["gpu_temperature_celsius"]
                after_temp = after_metrics["gpu_temperature_celsius"]
                if before_temp > 70:  # 高温の場合のみ評価
                    improvement = (before_temp - after_temp) / before_temp
                    improvements.append(improvement)
            
            # CPU使用率改善
            before_cpu = before_metrics.get("cpu_usage_percent", 0)
            after_cpu = after_metrics.get("cpu_usage_percent", 0)
            if before_cpu > 80:
                improvement = (before_cpu - after_cpu) / before_cpu
                improvements.append(improvement)
            
            # 平均改善度
            if improvements:
                effectiveness = statistics.mean(improvements)
                return max(0.0, min(1.0, effectiveness))  # 0-1に正規化
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Effectiveness measurement failed: {e}")
            return 0.0
    
    async def _record_optimization_result(self,
                                        conditions: Dict[str, float],
                                        actions: Dict[str, Any],
                                        effectiveness: float):
        """最適化結果記録"""
        try:
            # パターンID生成
            pattern_id = self._generate_pattern_id(conditions, actions)
            
            if pattern_id in self.learning_patterns:
                # 既存パターン更新
                pattern = self.learning_patterns[pattern_id]
                
                # 成功率更新
                total_uses = pattern.usage_count + 1
                success_count = pattern.success_rate * pattern.usage_count
                if effectiveness > 0.1:  # 10%以上の改善を成功とする
                    success_count += 1
                
                pattern.success_rate = success_count / total_uses
                pattern.effectiveness_score = (
                    pattern.effectiveness_score * 0.8 + effectiveness * 0.2
                )  # 指数移動平均
                
            else:
                # 新規パターン作成
                if len(self.learning_patterns) >= self.max_patterns:
                    # 最も効果の低いパターンを削除
                    worst_pattern_id = min(
                        self.learning_patterns.keys(),
                        key=lambda k: self.learning_patterns[k].effectiveness_score
                    )
                    del self.learning_patterns[worst_pattern_id]
                
                self.learning_patterns[pattern_id] = OptimizationLearning(
                    pattern_id=pattern_id,
                    conditions=conditions.copy(),
                    actions_taken=actions.copy(),
                    effectiveness_score=effectiveness,
                    success_rate=1.0 if effectiveness > 0.1 else 0.0
                )
            
            self.learning_updates += 1
            
            logger.info(f"Recorded optimization result: {pattern_id} "
                      f"(effectiveness: {effectiveness:.2f})")
            
        except Exception as e:
            logger.error(f"Result recording failed: {e}")
    
    def _generate_pattern_id(self, 
                           conditions: Dict[str, float],
                           actions: Dict[str, Any]) -> str:
        """パターンID生成"""
        try:
            # 条件を丸めて正規化
            normalized_conditions = {}
            for key, value in conditions.items():
                if key.endswith("_percent"):
                    normalized_conditions[key] = round(value / 10) * 10  # 10%単位
                elif key.endswith("_celsius"):
                    normalized_conditions[key] = round(value / 5) * 5   # 5度単位
                else:
                    normalized_conditions[key] = round(value)
            
            # アクションキーをソート
            sorted_actions = sorted(actions.keys())
            
            # ID生成
            conditions_str = "_".join(f"{k}:{v}" for k, v in sorted(normalized_conditions.items()))
            actions_str = "_".join(sorted_actions)
            
            return f"{conditions_str}|{actions_str}"
            
        except Exception as e:
            logger.error(f"Pattern ID generation failed: {e}")
            return f"pattern_{int(time.time())}"
    
    async def _update_learning_patterns(self, current_metrics: Dict[str, float]):
        """学習パターン更新"""
        try:
            # 使用頻度の低いパターンの効果性を減衰
            current_time = datetime.now()
            
            for pattern in self.learning_patterns.values():
                if pattern.last_used:
                    days_since_use = (current_time - pattern.last_used).days
                    if days_since_use > 7:  # 1週間以上未使用
                        decay_factor = 0.95 ** days_since_use
                        pattern.effectiveness_score *= decay_factor
            
            # 効果性の低いパターンを削除
            patterns_to_remove = [
                pattern_id for pattern_id, pattern in self.learning_patterns.items()
                if pattern.effectiveness_score < 0.1 and pattern.usage_count < 3
            ]
            
            for pattern_id in patterns_to_remove:
                del self.learning_patterns[pattern_id]
                logger.info(f"Removed ineffective pattern: {pattern_id}")
            
        except Exception as e:
            logger.error(f"Learning pattern update failed: {e}")
    
    def add_custom_rule(self, rule: MetricRule):
        """カスタムルール追加"""
        try:
            self.optimization_rules.append(rule)
            logger.info(f"Added custom rule: {rule.rule_type.value}")
        except Exception as e:
            logger.error(f"Custom rule addition failed: {e}")
    
    def remove_rule(self, rule_type: OptimizationRule):
        """ルール削除"""
        try:
            self.optimization_rules = [
                rule for rule in self.optimization_rules 
                if rule.rule_type != rule_type
            ]
            logger.info(f"Removed rules of type: {rule_type.value}")
        except Exception as e:
            logger.error(f"Rule removal failed: {e}")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """学習統計取得"""
        try:
            if not self.learning_patterns:
                return {"patterns": 0, "average_effectiveness": 0.0}
            
            effectiveness_scores = [p.effectiveness_score for p in self.learning_patterns.values()]
            success_rates = [p.success_rate for p in self.learning_patterns.values()]
            
            return {
                "total_patterns": len(self.learning_patterns),
                "average_effectiveness": statistics.mean(effectiveness_scores),
                "average_success_rate": statistics.mean(success_rates),
                "rules_triggered": self.rules_triggered,
                "optimizations_applied": self.optimizations_applied,
                "learning_updates": self.learning_updates,
                "is_running": self.is_running
            }
            
        except Exception as e:
            logger.error(f"Learning stats calculation failed: {e}")
            return {"error": str(e)}
    
    def get_top_patterns(self, limit: int = 5) -> List[Dict[str, Any]]:
        """上位パターン取得"""
        try:
            sorted_patterns = sorted(
                self.learning_patterns.values(),
                key=lambda p: p.effectiveness_score * p.success_rate,
                reverse=True
            )
            
            return [
                {
                    "pattern_id": p.pattern_id,
                    "effectiveness_score": p.effectiveness_score,
                    "success_rate": p.success_rate,
                    "usage_count": p.usage_count,
                    "conditions": p.conditions,
                    "actions": p.actions_taken
                }
                for p in sorted_patterns[:limit]
            ]
            
        except Exception as e:
            logger.error(f"Top patterns retrieval failed: {e}")
            return []
    
    async def cleanup(self):
        """リソースクリーンアップ"""
        await self.stop_monitoring()
        logger.info("Prometheus optimizer cleanup completed")


class PrometheusOptimizer:
    """Prometheus 統合最適化システム"""
    
    def __init__(self,
                 prometheus_collector: PrometheusMetricsCollector,
                 system_monitor: SystemMonitor):
        """
        初期化
        
        Args:
            prometheus_collector: Prometheus メトリクス収集器
            system_monitor: システム監視
        """
        self.prometheus_collector = prometheus_collector
        self.system_monitor = system_monitor
        
        # 自動最適化器（遅延インポート）
        from .auto_optimizer import AutoOptimizer
        self.auto_optimizer = AutoOptimizer(system_monitor)
        
        # メトリクスベース最適化器
        self.metric_optimizer = MetricBasedOptimizer(
            prometheus_collector, self.auto_optimizer
        )
    
    async def start_integrated_optimization(self):
        """統合最適化開始"""
        try:
            # 各コンポーネント開始
            await self.prometheus_collector.start_collection(interval=5.0)
            await self.system_monitor.start_monitoring()
            await self.auto_optimizer.start_optimization()
            await self.metric_optimizer.start_monitoring(check_interval=15.0)
            
            logger.info("Integrated optimization system started")
            
        except Exception as e:
            logger.error(f"Integrated optimization start failed: {e}")
    
    async def stop_integrated_optimization(self):
        """統合最適化停止"""
        try:
            await self.metric_optimizer.stop_monitoring()
            await self.auto_optimizer.stop_optimization()
            await self.system_monitor.stop_monitoring()
            await self.prometheus_collector.stop_collection()
            
            logger.info("Integrated optimization system stopped")
            
        except Exception as e:
            logger.error(f"Integrated optimization stop failed: {e}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """総合統計取得"""
        try:
            return {
                "auto_optimizer": self.auto_optimizer.get_optimization_stats(),
                "metric_optimizer": self.metric_optimizer.get_learning_stats(),
                "prometheus_collector": self.prometheus_collector.get_collection_stats(),
                "top_learning_patterns": self.metric_optimizer.get_top_patterns(3)
            }
        except Exception as e:
            logger.error(f"Comprehensive stats failed: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """リソースクリーンアップ"""
        await self.stop_integrated_optimization()
        await self.metric_optimizer.cleanup()
        await self.auto_optimizer.cleanup()
        logger.info("Prometheus optimizer cleanup completed")


# 使用例とテスト用のヘルパー関数
async def demo_prometheus_optimization():
    """デモ用 Prometheus 最適化実行"""
    from ..monitoring.system_monitor import SystemMonitor
    from ..monitoring.prometheus_collector import PrometheusMetricsCollector
    
    # システム監視初期化
    system_monitor = SystemMonitor(collection_interval=2.0)
    
    # Prometheus 収集器初期化
    prometheus_collector = PrometheusMetricsCollector(system_monitor)
    
    # 統合最適化システム初期化
    optimizer = PrometheusOptimizer(prometheus_collector, system_monitor)
    
    try:
        print("=== Prometheus Optimization Demo ===")
        
        # 統合最適化開始
        await optimizer.start_integrated_optimization()
        
        print("🔄 Running integrated optimization for 60 seconds...")
        await asyncio.sleep(60)
        
        # 統計表示
        stats = optimizer.get_comprehensive_stats()
        
        print(f"\n📊 Comprehensive Statistics:")
        
        if "auto_optimizer" in stats:
            auto_stats = stats["auto_optimizer"]
            print(f"  Auto Optimizer:")
            print(f"    Total Optimizations: {auto_stats.get('total_optimizations', 0)}")
            print(f"    Success Rate: {auto_stats.get('success_rate', 0):.1%}")
            print(f"    Average Improvement: {auto_stats.get('average_improvement', 0):.2f}%")
        
        if "metric_optimizer" in stats:
            metric_stats = stats["metric_optimizer"]
            print(f"  Metric Optimizer:")
            print(f"    Learning Patterns: {metric_stats.get('total_patterns', 0)}")
            print(f"    Rules Triggered: {metric_stats.get('rules_triggered', 0)}")
            print(f"    Optimizations Applied: {metric_stats.get('optimizations_applied', 0)}")
        
        if "top_learning_patterns" in stats:
            patterns = stats["top_learning_patterns"]
            print(f"  Top Learning Patterns:")
            for i, pattern in enumerate(patterns, 1):
                print(f"    {i}. Effectiveness: {pattern['effectiveness_score']:.2f}, "
                      f"Success Rate: {pattern['success_rate']:.1%}")
        
    finally:
        await optimizer.cleanup()


if __name__ == "__main__":
    asyncio.run(demo_prometheus_optimization())