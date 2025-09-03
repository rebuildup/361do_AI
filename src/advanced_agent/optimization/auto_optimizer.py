"""
Auto Optimizer Integration

HuggingFace の既存最適化機能による パラメータ自動調整システム
RTX 4050 6GB VRAM環境での動的最適化を提供します。

要件: 4.2, 4.4, 4.5
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Union
import logging
import json

# オプショナル依存関係
try:
    from transformers import (
        AutoConfig, AutoTokenizer, AutoModelForCausalLM,
        TrainingArguments, Trainer, BitsAndBytesConfig
    )
    from accelerate import Accelerator
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # モッククラス
    class AutoConfig:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()
    class BitsAndBytesConfig:
        def __init__(self, *args, **kwargs):
            pass

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    class LoraConfig:
        def __init__(self, *args, **kwargs):
            pass

from ..monitoring.system_monitor import SystemMonitor, SystemMetrics

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """最適化戦略"""
    CONSERVATIVE = "conservative"  # 安全重視
    BALANCED = "balanced"         # バランス重視
    AGGRESSIVE = "aggressive"     # 性能重視
    RTX4050_OPTIMIZED = "rtx4050_optimized"  # RTX 4050 特化


class ResourceAllocation(Enum):
    """リソース配分戦略"""
    GPU_PRIORITY = "gpu_priority"
    CPU_PRIORITY = "cpu_priority"
    HYBRID = "hybrid"
    AUTO = "auto"


@dataclass
class OptimizationConfig:
    """最適化設定"""
    strategy: OptimizationStrategy = OptimizationStrategy.RTX4050_OPTIMIZED
    resource_allocation: ResourceAllocation = ResourceAllocation.AUTO
    max_vram_usage_percent: float = 85.0
    target_temperature_celsius: float = 75.0
    optimization_interval_seconds: float = 30.0
    learning_rate_range: tuple = (1e-5, 1e-3)
    batch_size_range: tuple = (1, 16)
    quantization_levels: List[str] = field(default_factory=lambda: ["8bit", "4bit", "3bit"])
    enable_gradient_checkpointing: bool = True
    enable_mixed_precision: bool = True
    auto_adjust_parameters: bool = True


@dataclass
class OptimizationResult:
    """最適化結果"""
    optimization_id: str
    strategy_used: OptimizationStrategy
    parameters_adjusted: Dict[str, Any]
    performance_improvement: float
    resource_savings: Dict[str, float]
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class AutoOptimizer:
    """HuggingFace 自動最適化システム"""
    
    def __init__(self,
                 system_monitor: SystemMonitor,
                 config: Optional[OptimizationConfig] = None):
        """
        初期化
        
        Args:
            system_monitor: システム監視インスタンス
            config: 最適化設定
        """
        self.system_monitor = system_monitor
        self.config = config or OptimizationConfig()
        
        # 最適化履歴
        self.optimization_history: List[OptimizationResult] = []
        self.max_history_size = 100
        
        # 現在の設定
        self.current_parameters = {
            "learning_rate": 1e-4,
            "batch_size": 4,
            "quantization_level": "4bit",
            "gradient_checkpointing": True,
            "mixed_precision": True
        }
        
        # 最適化状態
        self.is_optimizing = False
        self.optimization_task: Optional[asyncio.Task] = None
        
        # 学習データ
        self.performance_data: List[Dict[str, float]] = []
        self.optimization_count = 0
        
        # RTX 4050 固有設定
        if self.config.strategy == OptimizationStrategy.RTX4050_OPTIMIZED:
            self._setup_rtx4050_optimizations()
    
    def _setup_rtx4050_optimizations(self):
        """RTX 4050 固有最適化設定"""
        # VRAM制限を考慮した設定
        self.config.max_vram_usage_percent = 85.0
        self.config.target_temperature_celsius = 75.0
        self.config.batch_size_range = (1, 8)  # 小さなバッチサイズ
        
        # 量子化を積極的に使用
        self.current_parameters["quantization_level"] = "4bit"
        self.current_parameters["gradient_checkpointing"] = True
        
        logger.info("RTX 4050 optimizations configured")
    
    async def start_optimization(self):
        """自動最適化開始"""
        if self.is_optimizing:
            logger.warning("Optimization already running")
            return
        
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, optimization disabled")
            return
        
        self.is_optimizing = True
        self.optimization_task = asyncio.create_task(self._optimization_loop())
        logger.info(f"Auto optimization started (interval: {self.config.optimization_interval_seconds}s)")
    
    async def stop_optimization(self):
        """自動最適化停止"""
        if not self.is_optimizing:
            return
        
        self.is_optimizing = False
        if self.optimization_task:
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Auto optimization stopped")
    
    async def _optimization_loop(self):
        """最適化ループ"""
        try:
            while self.is_optimizing:
                try:
                    # システムメトリクス取得
                    metrics = self.system_monitor.get_latest_metrics()
                    if metrics:
                        # 最適化実行
                        result = await self._perform_optimization(metrics)
                        
                        # 結果を履歴に追加
                        self.optimization_history.append(result)
                        if len(self.optimization_history) > self.max_history_size:
                            self.optimization_history.pop(0)
                        
                        # 学習データ更新
                        self._update_performance_data(metrics, result)
                        
                        if result.success:
                            logger.info(f"Optimization completed: {result.performance_improvement:.2f}% improvement")
                        else:
                            logger.warning(f"Optimization failed: {result.error_message}")
                
                except Exception as e:
                    logger.error(f"Optimization loop error: {e}")
                
                # 次の最適化まで待機
                await asyncio.sleep(self.config.optimization_interval_seconds)
                
        except asyncio.CancelledError:
            logger.info("Optimization loop cancelled")
        except Exception as e:
            logger.error(f"Optimization loop error: {e}")
            self.is_optimizing = False
    
    async def _perform_optimization(self, metrics: SystemMetrics) -> OptimizationResult:
        """最適化実行"""
        start_time = time.time()
        optimization_id = f"opt_{int(time.time())}"
        
        try:
            # 現在の性能ベースライン
            baseline_performance = self._calculate_performance_score(metrics)
            
            # 最適化戦略決定
            optimization_actions = self._determine_optimization_actions(metrics)
            
            # パラメータ調整実行
            adjusted_parameters = {}
            for action, new_value in optimization_actions.items():
                old_value = self.current_parameters.get(action)
                self.current_parameters[action] = new_value
                adjusted_parameters[action] = {"old": old_value, "new": new_value}
            
            # 最適化効果測定（シミュレーション）
            performance_improvement = await self._measure_optimization_effect(
                metrics, optimization_actions
            )
            
            # リソース節約計算
            resource_savings = self._calculate_resource_savings(optimization_actions)
            
            # 推奨事項生成
            recommendations = self._generate_recommendations(metrics, optimization_actions)
            
            execution_time = time.time() - start_time
            
            return OptimizationResult(
                optimization_id=optimization_id,
                strategy_used=self.config.strategy,
                parameters_adjusted=adjusted_parameters,
                performance_improvement=performance_improvement,
                resource_savings=resource_savings,
                execution_time=execution_time,
                success=True,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return OptimizationResult(
                optimization_id=optimization_id,
                strategy_used=self.config.strategy,
                parameters_adjusted={},
                performance_improvement=0.0,
                resource_savings={},
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _determine_optimization_actions(self, metrics: SystemMetrics) -> Dict[str, Any]:
        """最適化アクション決定"""
        actions = {}
        
        # GPU メモリ最適化
        if metrics.gpu and metrics.gpu.memory_percent > self.config.max_vram_usage_percent:
            # 量子化レベル上昇
            current_quant = self.current_parameters.get("quantization_level", "4bit")
            if current_quant == "8bit":
                actions["quantization_level"] = "4bit"
            elif current_quant == "4bit":
                actions["quantization_level"] = "3bit"
            
            # バッチサイズ削減
            current_batch = self.current_parameters.get("batch_size", 4)
            if current_batch > self.config.batch_size_range[0]:
                actions["batch_size"] = max(1, current_batch // 2)
        
        # GPU 温度最適化
        if metrics.gpu and metrics.gpu.temperature_celsius > self.config.target_temperature_celsius:
            # 処理負荷削減
            actions["enable_cpu_offload"] = True
            
            # 学習率調整（収束を早める）
            current_lr = self.current_parameters.get("learning_rate", 1e-4)
            if current_lr < self.config.learning_rate_range[1]:
                actions["learning_rate"] = min(self.config.learning_rate_range[1], current_lr * 1.2)
        
        # CPU 使用率最適化
        if metrics.cpu.usage_percent > 85:
            # GPU により多くの処理を移譲
            actions["prefer_gpu_processing"] = True
        
        # メモリ使用率最適化
        if metrics.memory.usage_percent > 85:
            # グラディエントチェックポイント有効化
            actions["gradient_checkpointing"] = True
            
            # 混合精度有効化
            actions["mixed_precision"] = True
        
        # RTX 4050 固有最適化
        if self.config.strategy == OptimizationStrategy.RTX4050_OPTIMIZED:
            actions.update(self._rtx4050_specific_optimizations(metrics))
        
        return actions
    
    def _rtx4050_specific_optimizations(self, metrics: SystemMetrics) -> Dict[str, Any]:
        """RTX 4050 固有最適化"""
        actions = {}
        
        if metrics.gpu:
            # 6GB VRAM の効率的使用
            vram_usage_mb = metrics.gpu.memory_used_mb
            
            if vram_usage_mb > 5000:  # 5GB 超過時
                actions.update({
                    "quantization_level": "3bit",
                    "batch_size": 1,
                    "gradient_accumulation_steps": 4,
                    "enable_cpu_offload": True
                })
            elif vram_usage_mb > 4000:  # 4GB 超過時
                actions.update({
                    "quantization_level": "4bit",
                    "batch_size": 2,
                    "gradient_accumulation_steps": 2
                })
            
            # 温度管理
            if metrics.gpu.temperature_celsius > 70:
                actions.update({
                    "reduce_clock_speed": True,
                    "enable_thermal_throttling": True
                })
        
        return actions
    
    async def _measure_optimization_effect(self,
                                         metrics: SystemMetrics,
                                         actions: Dict[str, Any]) -> float:
        """最適化効果測定（シミュレーション）"""
        try:
            # 基本性能スコア
            base_score = self._calculate_performance_score(metrics)
            
            # アクション別効果予測
            improvement = 0.0
            
            for action, value in actions.items():
                if action == "quantization_level":
                    if value == "4bit":
                        improvement += 0.15  # 15% VRAM節約
                    elif value == "3bit":
                        improvement += 0.25  # 25% VRAM節約
                
                elif action == "batch_size" and isinstance(value, int):
                    current_batch = self.current_parameters.get("batch_size", 4)
                    if value < current_batch:
                        improvement += 0.1  # 10% メモリ節約
                
                elif action == "gradient_checkpointing" and value:
                    improvement += 0.08  # 8% メモリ節約
                
                elif action == "mixed_precision" and value:
                    improvement += 0.12  # 12% 速度向上
                
                elif action == "enable_cpu_offload" and value:
                    improvement += 0.05  # 5% GPU負荷軽減
            
            # 履歴データからの学習効果
            if self.performance_data:
                historical_improvement = self._predict_from_history(actions)
                improvement = (improvement + historical_improvement) / 2
            
            return min(improvement * 100, 50.0)  # 最大50%改善
            
        except Exception as e:
            logger.error(f"Effect measurement failed: {e}")
            return 0.0
    
    def _calculate_performance_score(self, metrics: SystemMetrics) -> float:
        """性能スコア計算"""
        try:
            # 各リソースの効率性を計算
            cpu_efficiency = max(0, 1.0 - (metrics.cpu.usage_percent / 100.0))
            memory_efficiency = max(0, 1.0 - (metrics.memory.usage_percent / 100.0))
            
            gpu_efficiency = 1.0
            if metrics.gpu:
                gpu_memory_efficiency = max(0, 1.0 - (metrics.gpu.memory_percent / 100.0))
                gpu_temp_efficiency = max(0, 1.0 - (metrics.gpu.temperature_celsius / 100.0))
                gpu_efficiency = (gpu_memory_efficiency + gpu_temp_efficiency) / 2
            
            # 重み付き平均
            weights = {"cpu": 0.3, "memory": 0.3, "gpu": 0.4}
            total_score = (
                cpu_efficiency * weights["cpu"] +
                memory_efficiency * weights["memory"] +
                gpu_efficiency * weights["gpu"]
            )
            
            return total_score
            
        except Exception as e:
            logger.error(f"Performance score calculation failed: {e}")
            return 0.5
    
    def _calculate_resource_savings(self, actions: Dict[str, Any]) -> Dict[str, float]:
        """リソース節約計算"""
        savings = {
            "vram_mb": 0.0,
            "system_memory_mb": 0.0,
            "power_watts": 0.0,
            "temperature_celsius": 0.0
        }
        
        try:
            for action, value in actions.items():
                if action == "quantization_level":
                    if value == "4bit":
                        savings["vram_mb"] += 1024  # 1GB節約
                    elif value == "3bit":
                        savings["vram_mb"] += 1536  # 1.5GB節約
                
                elif action == "batch_size" and isinstance(value, int):
                    current_batch = self.current_parameters.get("batch_size", 4)
                    if value < current_batch:
                        savings["vram_mb"] += (current_batch - value) * 256
                
                elif action == "gradient_checkpointing" and value:
                    savings["system_memory_mb"] += 512
                
                elif action == "enable_cpu_offload" and value:
                    savings["power_watts"] += 20
                    savings["temperature_celsius"] += 5
            
            return savings
            
        except Exception as e:
            logger.error(f"Resource savings calculation failed: {e}")
            return savings
    
    def _generate_recommendations(self,
                                metrics: SystemMetrics,
                                actions: Dict[str, Any]) -> List[str]:
        """推奨事項生成"""
        recommendations = []
        
        try:
            # アクション別推奨事項
            for action, value in actions.items():
                if action == "quantization_level":
                    recommendations.append(f"量子化レベルを{value}に変更してVRAM使用量を削減")
                
                elif action == "batch_size":
                    recommendations.append(f"バッチサイズを{value}に調整してメモリ効率を向上")
                
                elif action == "enable_cpu_offload":
                    recommendations.append("CPU オフロードを有効化してGPU負荷を軽減")
                
                elif action == "gradient_checkpointing":
                    recommendations.append("グラディエントチェックポイントでメモリ使用量を最適化")
            
            # RTX 4050 固有推奨事項
            if self.config.strategy == OptimizationStrategy.RTX4050_OPTIMIZED:
                if metrics.gpu and metrics.gpu.memory_percent > 80:
                    recommendations.append("RTX 4050の6GB制限を考慮し、より積極的な量子化を検討")
                
                if metrics.gpu and metrics.gpu.temperature_celsius > 70:
                    recommendations.append("RTX 4050の温度管理のため、処理負荷の分散を検討")
            
            # 一般的な推奨事項
            if not actions:
                recommendations.append("現在の設定は最適です。継続的な監視を推奨")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendations generation failed: {e}")
            return ["推奨事項の生成に失敗しました"]
    
    def _predict_from_history(self, actions: Dict[str, Any]) -> float:
        """履歴データからの予測"""
        try:
            if len(self.performance_data) < 3:
                return 0.0
            
            # 類似のアクションパターンを検索
            similar_patterns = []
            for data in self.performance_data[-10:]:  # 最新10件
                similarity = self._calculate_action_similarity(actions, data.get("actions", {}))
                if similarity > 0.7:  # 70%以上の類似度
                    similar_patterns.append(data.get("improvement", 0.0))
            
            if similar_patterns:
                return sum(similar_patterns) / len(similar_patterns) / 100.0
            
            return 0.0
            
        except Exception as e:
            logger.error(f"History prediction failed: {e}")
            return 0.0
    
    def _calculate_action_similarity(self, actions1: Dict[str, Any], actions2: Dict[str, Any]) -> float:
        """アクション類似度計算"""
        try:
            if not actions1 or not actions2:
                return 0.0
            
            common_keys = set(actions1.keys()) & set(actions2.keys())
            if not common_keys:
                return 0.0
            
            similarity_sum = 0.0
            for key in common_keys:
                if actions1[key] == actions2[key]:
                    similarity_sum += 1.0
                elif isinstance(actions1[key], (int, float)) and isinstance(actions2[key], (int, float)):
                    # 数値の場合は相対的類似度
                    diff = abs(actions1[key] - actions2[key])
                    max_val = max(abs(actions1[key]), abs(actions2[key]), 1)
                    similarity_sum += max(0, 1.0 - diff / max_val)
            
            return similarity_sum / len(common_keys)
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    def _update_performance_data(self, metrics: SystemMetrics, result: OptimizationResult):
        """性能データ更新"""
        try:
            performance_entry = {
                "timestamp": datetime.now(),
                "actions": result.parameters_adjusted,
                "improvement": result.performance_improvement,
                "vram_usage": metrics.gpu.memory_percent if metrics.gpu else 0,
                "cpu_usage": metrics.cpu.usage_percent,
                "memory_usage": metrics.memory.usage_percent,
                "success": result.success
            }
            
            self.performance_data.append(performance_entry)
            
            # データサイズ制限
            if len(self.performance_data) > 100:
                self.performance_data.pop(0)
            
            self.optimization_count += 1
            
        except Exception as e:
            logger.error(f"Performance data update failed: {e}")
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """現在のパラメータ取得"""
        return self.current_parameters.copy()
    
    def set_parameters(self, parameters: Dict[str, Any]):
        """パラメータ設定"""
        try:
            for key, value in parameters.items():
                if key in self.current_parameters:
                    self.current_parameters[key] = value
                    logger.info(f"Parameter updated: {key} = {value}")
        except Exception as e:
            logger.error(f"Parameter setting failed: {e}")
    
    def get_optimization_history(self, limit: Optional[int] = None) -> List[OptimizationResult]:
        """最適化履歴取得"""
        if limit is None:
            return self.optimization_history.copy()
        return self.optimization_history[-limit:].copy()
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """最適化統計取得"""
        try:
            successful_optimizations = [r for r in self.optimization_history if r.success]
            
            stats = {
                "total_optimizations": len(self.optimization_history),
                "successful_optimizations": len(successful_optimizations),
                "success_rate": len(successful_optimizations) / len(self.optimization_history) if self.optimization_history else 0,
                "average_improvement": sum(r.performance_improvement for r in successful_optimizations) / len(successful_optimizations) if successful_optimizations else 0,
                "is_optimizing": self.is_optimizing,
                "current_strategy": self.config.strategy.value,
                "optimization_count": self.optimization_count
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Stats calculation failed: {e}")
            return {"error": str(e)}
    
    async def manual_optimization(self, target_metric: str = "vram_usage") -> OptimizationResult:
        """手動最適化実行"""
        try:
            metrics = self.system_monitor.get_latest_metrics()
            if not metrics:
                raise ValueError("No system metrics available")
            
            # ターゲットメトリクスに基づく最適化
            if target_metric == "vram_usage" and metrics.gpu:
                actions = {"quantization_level": "4bit", "batch_size": 2}
            elif target_metric == "temperature" and metrics.gpu:
                actions = {"enable_cpu_offload": True, "reduce_clock_speed": True}
            elif target_metric == "memory_usage":
                actions = {"gradient_checkpointing": True, "mixed_precision": True}
            else:
                actions = self._determine_optimization_actions(metrics)
            
            # 最適化実行
            return await self._perform_optimization(metrics)
            
        except Exception as e:
            logger.error(f"Manual optimization failed: {e}")
            return OptimizationResult(
                optimization_id=f"manual_{int(time.time())}",
                strategy_used=self.config.strategy,
                parameters_adjusted={},
                performance_improvement=0.0,
                resource_savings={},
                execution_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def cleanup(self):
        """リソースクリーンアップ"""
        await self.stop_optimization()
        logger.info("Auto optimizer cleanup completed")


# 使用例とテスト用のヘルパー関数
async def demo_auto_optimization():
    """デモ用自動最適化実行"""
    from ..monitoring.system_monitor import SystemMonitor
    
    # システム監視初期化
    system_monitor = SystemMonitor(collection_interval=2.0)
    
    # 自動最適化器初期化
    config = OptimizationConfig(
        strategy=OptimizationStrategy.RTX4050_OPTIMIZED,
        optimization_interval_seconds=10.0
    )
    optimizer = AutoOptimizer(system_monitor, config)
    
    try:
        print("=== Auto Optimization Demo ===")
        
        # 現在のパラメータ表示
        print(f"\n🔧 Current Parameters:")
        for key, value in optimizer.get_current_parameters().items():
            print(f"  {key}: {value}")
        
        # 監視開始
        await system_monitor.start_monitoring()
        
        # 手動最適化実行
        print(f"\n⚡ Running manual optimization...")
        result = await optimizer.manual_optimization("vram_usage")
        
        print(f"\n📊 Optimization Result:")
        print(f"  Success: {result.success}")
        print(f"  Performance Improvement: {result.performance_improvement:.2f}%")
        print(f"  Execution Time: {result.execution_time:.2f}s")
        
        if result.parameters_adjusted:
            print(f"  Parameters Adjusted:")
            for param, changes in result.parameters_adjusted.items():
                print(f"    {param}: {changes['old']} → {changes['new']}")
        
        if result.resource_savings:
            print(f"  Resource Savings:")
            for resource, saving in result.resource_savings.items():
                if saving > 0:
                    print(f"    {resource}: {saving:.1f}")
        
        if result.recommendations:
            print(f"  Recommendations:")
            for rec in result.recommendations:
                print(f"    • {rec}")
        
        # 自動最適化開始
        print(f"\n🔄 Starting auto optimization for 30 seconds...")
        await optimizer.start_optimization()
        await asyncio.sleep(30)
        await optimizer.stop_optimization()
        
        # 統計表示
        stats = optimizer.get_optimization_stats()
        print(f"\n📈 Optimization Statistics:")
        print(f"  Total Optimizations: {stats['total_optimizations']}")
        print(f"  Success Rate: {stats['success_rate']:.1%}")
        print(f"  Average Improvement: {stats['average_improvement']:.2f}%")
        
    finally:
        await optimizer.cleanup()
        await system_monitor.cleanup()


if __name__ == "__main__":
    asyncio.run(demo_auto_optimization())