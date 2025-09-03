"""
Dynamic Resource Manager

動的リソース配分システム
RTX 4050 6GB VRAM環境での効率的リソース管理を提供します。

要件: 4.2, 4.4, 4.5
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import logging
import statistics

from ..monitoring.system_monitor import SystemMonitor, SystemMetrics

logger = logging.getLogger(__name__)


class AllocationStrategy(Enum):
    """配分戦略"""
    PERFORMANCE_FIRST = "performance_first"
    EFFICIENCY_FIRST = "efficiency_first"
    BALANCED = "balanced"
    RTX4050_OPTIMIZED = "rtx4050_optimized"


class ResourceType(Enum):
    """リソースタイプ"""
    GPU_MEMORY = "gpu_memory"
    GPU_COMPUTE = "gpu_compute"
    CPU_CORES = "cpu_cores"
    SYSTEM_MEMORY = "system_memory"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"


@dataclass
class ResourceConstraints:
    """リソース制約"""
    max_gpu_memory_mb: float = 5120.0  # RTX 4050: 5GB制限
    max_gpu_temperature: float = 75.0
    max_cpu_usage_percent: float = 85.0
    max_memory_usage_percent: float = 85.0
    min_available_memory_gb: float = 2.0
    max_power_draw_watts: float = 130.0


@dataclass
class ResourceAllocation:
    """リソース配分"""
    resource_type: ResourceType
    allocated_amount: float
    max_amount: float
    usage_percent: float
    priority: int
    constraints: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AllocationPlan:
    """配分計画"""
    plan_id: str
    strategy: AllocationStrategy
    allocations: List[ResourceAllocation]
    expected_performance: float
    resource_efficiency: float
    constraints_satisfied: bool
    execution_time: float
    timestamp: datetime = field(default_factory=datetime.now)


class DynamicResourceManager:
    """動的リソース管理システム"""
    
    def __init__(self,
                 system_monitor: SystemMonitor,
                 constraints: Optional[ResourceConstraints] = None,
                 strategy: AllocationStrategy = AllocationStrategy.RTX4050_OPTIMIZED):
        """
        初期化
        
        Args:
            system_monitor: システム監視インスタンス
            constraints: リソース制約
            strategy: 配分戦略
        """
        self.system_monitor = system_monitor
        self.constraints = constraints or ResourceConstraints()
        self.strategy = strategy
        
        # 現在の配分状態
        self.current_allocations: Dict[ResourceType, ResourceAllocation] = {}
        
        # 配分履歴
        self.allocation_history: List[AllocationPlan] = []
        self.max_history_size = 50
        
        # 管理状態
        self.is_managing = False
        self.management_task: Optional[asyncio.Task] = None
        
        # 統計
        self.reallocation_count = 0
        self.constraint_violations = 0
        self.performance_improvements = 0
        
        # RTX 4050 固有設定
        if strategy == AllocationStrategy.RTX4050_OPTIMIZED:
            self._setup_rtx4050_constraints()
    
    def _setup_rtx4050_constraints(self):
        """RTX 4050 固有制約設定"""
        self.constraints.max_gpu_memory_mb = 5120.0  # 6GB の 85%
        self.constraints.max_gpu_temperature = 75.0
        self.constraints.max_power_draw_watts = 130.0  # RTX 4050 TGP
        
        logger.info("RTX 4050 resource constraints configured")
    
    async def start_management(self, reallocation_interval: float = 20.0):
        """リソース管理開始"""
        if self.is_managing:
            logger.warning("Resource management already running")
            return
        
        self.is_managing = True
        self.management_task = asyncio.create_task(
            self._management_loop(reallocation_interval)
        )
        logger.info(f"Dynamic resource management started (interval: {reallocation_interval}s)")
    
    async def stop_management(self):
        """リソース管理停止"""
        if not self.is_managing:
            return
        
        self.is_managing = False
        if self.management_task:
            self.management_task.cancel()
            try:
                await self.management_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Dynamic resource management stopped")
    
    async def _management_loop(self, reallocation_interval: float):
        """管理ループ"""
        try:
            while self.is_managing:
                try:
                    # 現在のシステム状態取得
                    metrics = self.system_monitor.get_latest_metrics()
                    
                    if metrics:
                        # リソース配分計画作成
                        plan = await self._create_allocation_plan(metrics)
                        
                        # 配分実行
                        if plan and plan.constraints_satisfied:
                            await self._execute_allocation_plan(plan)
                        
                        # 制約違反チェック
                        violations = self._check_constraint_violations(metrics)
                        if violations:
                            await self._handle_constraint_violations(violations, metrics)
                
                except Exception as e:
                    logger.error(f"Management loop error: {e}")
                
                await asyncio.sleep(reallocation_interval)
                
        except asyncio.CancelledError:
            logger.info("Management loop cancelled")
        except Exception as e:
            logger.error(f"Management loop error: {e}")
            self.is_managing = False
    
    async def _create_allocation_plan(self, metrics: SystemMetrics) -> Optional[AllocationPlan]:
        """配分計画作成"""
        try:
            start_time = time.time()
            plan_id = f"plan_{int(time.time())}"
            
            # 現在のリソース使用状況分析
            resource_analysis = self._analyze_resource_usage(metrics)
            
            # 配分戦略に基づく最適配分計算
            allocations = await self._calculate_optimal_allocations(
                resource_analysis, metrics
            )
            
            # 制約チェック
            constraints_satisfied = self._validate_constraints(allocations, metrics)
            
            # 性能予測
            expected_performance = self._predict_performance(allocations, metrics)
            
            # 効率性計算
            resource_efficiency = self._calculate_efficiency(allocations, metrics)
            
            execution_time = time.time() - start_time
            
            plan = AllocationPlan(
                plan_id=plan_id,
                strategy=self.strategy,
                allocations=allocations,
                expected_performance=expected_performance,
                resource_efficiency=resource_efficiency,
                constraints_satisfied=constraints_satisfied,
                execution_time=execution_time
            )
            
            # 履歴に追加
            self.allocation_history.append(plan)
            if len(self.allocation_history) > self.max_history_size:
                self.allocation_history.pop(0)
            
            return plan
            
        except Exception as e:
            logger.error(f"Allocation plan creation failed: {e}")
            return None
    
    def _analyze_resource_usage(self, metrics: SystemMetrics) -> Dict[str, float]:
        """リソース使用状況分析"""
        analysis = {
            "cpu_utilization": metrics.cpu.usage_percent / 100.0,
            "memory_utilization": metrics.memory.usage_percent / 100.0,
            "memory_pressure": max(0, (metrics.memory.usage_percent - 70) / 30.0),
            "cpu_pressure": max(0, (metrics.cpu.usage_percent - 70) / 30.0)
        }
        
        if metrics.gpu:
            analysis.update({
                "gpu_compute_utilization": metrics.gpu.utilization_percent / 100.0,
                "gpu_memory_utilization": metrics.gpu.memory_percent / 100.0,
                "gpu_thermal_pressure": max(0, (metrics.gpu.temperature_celsius - 60) / 20.0),
                "gpu_power_pressure": max(0, (metrics.gpu.power_draw_watts - 100) / 50.0)
            })
        
        return analysis
    
    async def _calculate_optimal_allocations(self,
                                           analysis: Dict[str, float],
                                           metrics: SystemMetrics) -> List[ResourceAllocation]:
        """最適配分計算"""
        allocations = []
        
        try:
            # GPU メモリ配分
            if metrics.gpu:
                gpu_memory_allocation = self._calculate_gpu_memory_allocation(analysis, metrics)
                allocations.append(gpu_memory_allocation)
                
                # GPU 計算配分
                gpu_compute_allocation = self._calculate_gpu_compute_allocation(analysis, metrics)
                allocations.append(gpu_compute_allocation)
            
            # CPU 配分
            cpu_allocation = self._calculate_cpu_allocation(analysis, metrics)
            allocations.append(cpu_allocation)
            
            # システムメモリ配分
            memory_allocation = self._calculate_memory_allocation(analysis, metrics)
            allocations.append(memory_allocation)
            
            return allocations
            
        except Exception as e:
            logger.error(f"Optimal allocation calculation failed: {e}")
            return []
    
    def _calculate_gpu_memory_allocation(self,
                                       analysis: Dict[str, float],
                                       metrics: SystemMetrics) -> ResourceAllocation:
        """GPU メモリ配分計算"""
        current_usage = metrics.gpu.memory_used_mb
        total_memory = metrics.gpu.memory_total_mb
        
        # RTX 4050 固有の最適化
        if self.strategy == AllocationStrategy.RTX4050_OPTIMIZED:
            # 6GB制限を考慮した配分
            safe_limit = min(self.constraints.max_gpu_memory_mb, total_memory * 0.85)
            
            # 温度とメモリ使用率に基づく動的調整
            thermal_factor = max(0.7, 1.0 - analysis.get("gpu_thermal_pressure", 0) * 0.3)
            allocated_amount = safe_limit * thermal_factor
        else:
            allocated_amount = total_memory * 0.9  # 90%まで使用可能
        
        return ResourceAllocation(
            resource_type=ResourceType.GPU_MEMORY,
            allocated_amount=allocated_amount,
            max_amount=total_memory,
            usage_percent=(current_usage / allocated_amount) * 100,
            priority=1,  # 最高優先度
            constraints={
                "max_temperature": self.constraints.max_gpu_temperature,
                "thermal_pressure": analysis.get("gpu_thermal_pressure", 0)
            }
        )
    
    def _calculate_gpu_compute_allocation(self,
                                        analysis: Dict[str, float],
                                        metrics: SystemMetrics) -> ResourceAllocation:
        """GPU 計算配分計算"""
        current_utilization = metrics.gpu.utilization_percent
        
        # 温度制約に基づく計算配分
        thermal_pressure = analysis.get("gpu_thermal_pressure", 0)
        power_pressure = analysis.get("gpu_power_pressure", 0)
        
        # 制約に基づく配分調整
        constraint_factor = 1.0 - max(thermal_pressure, power_pressure) * 0.4
        allocated_utilization = 95.0 * constraint_factor  # 最大95%
        
        return ResourceAllocation(
            resource_type=ResourceType.GPU_COMPUTE,
            allocated_amount=allocated_utilization,
            max_amount=100.0,
            usage_percent=current_utilization,
            priority=2,
            constraints={
                "max_temperature": self.constraints.max_gpu_temperature,
                "max_power": self.constraints.max_power_draw_watts
            }
        )
    
    def _calculate_cpu_allocation(self,
                                analysis: Dict[str, float],
                                metrics: SystemMetrics) -> ResourceAllocation:
        """CPU 配分計算"""
        current_usage = metrics.cpu.usage_percent
        
        # GPU負荷に基づくCPU配分調整
        gpu_pressure = analysis.get("gpu_thermal_pressure", 0)
        
        if gpu_pressure > 0.5:  # GPU負荷が高い場合
            # CPUにより多くの処理を移譲
            allocated_usage = min(90.0, self.constraints.max_cpu_usage_percent)
        else:
            allocated_usage = self.constraints.max_cpu_usage_percent
        
        return ResourceAllocation(
            resource_type=ResourceType.CPU_CORES,
            allocated_amount=allocated_usage,
            max_amount=100.0,
            usage_percent=current_usage,
            priority=3,
            constraints={
                "max_usage": self.constraints.max_cpu_usage_percent
            }
        )
    
    def _calculate_memory_allocation(self,
                                   analysis: Dict[str, float],
                                   metrics: SystemMetrics) -> ResourceAllocation:
        """メモリ配分計算"""
        current_usage = metrics.memory.usage_percent
        available_gb = metrics.memory.available_gb
        
        # 最小利用可能メモリを確保
        reserved_gb = max(self.constraints.min_available_memory_gb, 1.0)
        max_usage_percent = ((metrics.memory.total_gb - reserved_gb) / metrics.memory.total_gb) * 100
        
        allocated_usage = min(max_usage_percent, self.constraints.max_memory_usage_percent)
        
        return ResourceAllocation(
            resource_type=ResourceType.SYSTEM_MEMORY,
            allocated_amount=allocated_usage,
            max_amount=100.0,
            usage_percent=current_usage,
            priority=2,
            constraints={
                "min_available_gb": self.constraints.min_available_memory_gb,
                "max_usage_percent": self.constraints.max_memory_usage_percent
            }
        )
    
    def _validate_constraints(self,
                            allocations: List[ResourceAllocation],
                            metrics: SystemMetrics) -> bool:
        """制約検証"""
        try:
            for allocation in allocations:
                if allocation.resource_type == ResourceType.GPU_MEMORY:
                    if allocation.allocated_amount > self.constraints.max_gpu_memory_mb:
                        return False
                
                elif allocation.resource_type == ResourceType.GPU_COMPUTE:
                    if metrics.gpu and metrics.gpu.temperature_celsius > self.constraints.max_gpu_temperature:
                        return False
                
                elif allocation.resource_type == ResourceType.CPU_CORES:
                    if allocation.usage_percent > self.constraints.max_cpu_usage_percent:
                        return False
                
                elif allocation.resource_type == ResourceType.SYSTEM_MEMORY:
                    if metrics.memory.available_gb < self.constraints.min_available_memory_gb:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Constraint validation failed: {e}")
            return False
    
    def _predict_performance(self,
                           allocations: List[ResourceAllocation],
                           metrics: SystemMetrics) -> float:
        """性能予測"""
        try:
            performance_factors = []
            
            for allocation in allocations:
                if allocation.resource_type == ResourceType.GPU_MEMORY:
                    # GPU メモリ効率
                    efficiency = 1.0 - (allocation.usage_percent / 100.0) ** 2
                    performance_factors.append(efficiency * 0.4)  # 40%の重み
                
                elif allocation.resource_type == ResourceType.GPU_COMPUTE:
                    # GPU 計算効率
                    efficiency = min(1.0, allocation.allocated_amount / 100.0)
                    performance_factors.append(efficiency * 0.3)  # 30%の重み
                
                elif allocation.resource_type == ResourceType.CPU_CORES:
                    # CPU 効率
                    efficiency = min(1.0, allocation.allocated_amount / 100.0)
                    performance_factors.append(efficiency * 0.2)  # 20%の重み
                
                elif allocation.resource_type == ResourceType.SYSTEM_MEMORY:
                    # メモリ効率
                    efficiency = 1.0 - max(0, (allocation.usage_percent - 70) / 30.0)
                    performance_factors.append(efficiency * 0.1)  # 10%の重み
            
            return sum(performance_factors) if performance_factors else 0.5
            
        except Exception as e:
            logger.error(f"Performance prediction failed: {e}")
            return 0.5
    
    def _calculate_efficiency(self,
                            allocations: List[ResourceAllocation],
                            metrics: SystemMetrics) -> float:
        """効率性計算"""
        try:
            efficiency_scores = []
            
            for allocation in allocations:
                # 使用率と配分量の比率
                if allocation.allocated_amount > 0:
                    utilization_ratio = allocation.usage_percent / (allocation.allocated_amount / allocation.max_amount * 100)
                    efficiency = min(1.0, utilization_ratio)
                    efficiency_scores.append(efficiency)
            
            return statistics.mean(efficiency_scores) if efficiency_scores else 0.5
            
        except Exception as e:
            logger.error(f"Efficiency calculation failed: {e}")
            return 0.5
    
    async def _execute_allocation_plan(self, plan: AllocationPlan):
        """配分計画実行"""
        try:
            # 現在の配分を更新
            for allocation in plan.allocations:
                self.current_allocations[allocation.resource_type] = allocation
            
            self.reallocation_count += 1
            
            # 性能改善があった場合
            if plan.expected_performance > 0.7:
                self.performance_improvements += 1
            
            logger.info(f"Executed allocation plan: {plan.plan_id} "
                      f"(performance: {plan.expected_performance:.2f}, "
                      f"efficiency: {plan.resource_efficiency:.2f})")
            
        except Exception as e:
            logger.error(f"Allocation plan execution failed: {e}")
    
    def _check_constraint_violations(self, metrics: SystemMetrics) -> List[str]:
        """制約違反チェック"""
        violations = []
        
        try:
            # GPU メモリ制約
            if metrics.gpu and metrics.gpu.memory_used_mb > self.constraints.max_gpu_memory_mb:
                violations.append(f"GPU memory exceeded: {metrics.gpu.memory_used_mb:.0f}MB > {self.constraints.max_gpu_memory_mb:.0f}MB")
            
            # GPU 温度制約
            if metrics.gpu and metrics.gpu.temperature_celsius > self.constraints.max_gpu_temperature:
                violations.append(f"GPU temperature exceeded: {metrics.gpu.temperature_celsius:.1f}°C > {self.constraints.max_gpu_temperature:.1f}°C")
            
            # CPU 使用率制約
            if metrics.cpu.usage_percent > self.constraints.max_cpu_usage_percent:
                violations.append(f"CPU usage exceeded: {metrics.cpu.usage_percent:.1f}% > {self.constraints.max_cpu_usage_percent:.1f}%")
            
            # メモリ制約
            if metrics.memory.available_gb < self.constraints.min_available_memory_gb:
                violations.append(f"Available memory too low: {metrics.memory.available_gb:.1f}GB < {self.constraints.min_available_memory_gb:.1f}GB")
            
            # 電力制約
            if metrics.gpu and metrics.gpu.power_draw_watts > self.constraints.max_power_draw_watts:
                violations.append(f"Power draw exceeded: {metrics.gpu.power_draw_watts:.1f}W > {self.constraints.max_power_draw_watts:.1f}W")
            
            if violations:
                self.constraint_violations += len(violations)
            
            return violations
            
        except Exception as e:
            logger.error(f"Constraint violation check failed: {e}")
            return []
    
    async def _handle_constraint_violations(self,
                                          violations: List[str],
                                          metrics: SystemMetrics):
        """制約違反処理"""
        try:
            logger.warning(f"Constraint violations detected: {len(violations)}")
            
            for violation in violations:
                logger.warning(f"  - {violation}")
            
            # 緊急配分調整
            emergency_allocations = await self._create_emergency_allocations(violations, metrics)
            
            if emergency_allocations:
                for allocation in emergency_allocations:
                    self.current_allocations[allocation.resource_type] = allocation
                
                logger.info("Emergency resource reallocation applied")
            
        except Exception as e:
            logger.error(f"Constraint violation handling failed: {e}")
    
    async def _create_emergency_allocations(self,
                                          violations: List[str],
                                          metrics: SystemMetrics) -> List[ResourceAllocation]:
        """緊急配分作成"""
        emergency_allocations = []
        
        try:
            for violation in violations:
                if "GPU memory exceeded" in violation:
                    # GPU メモリを大幅削減
                    emergency_allocations.append(ResourceAllocation(
                        resource_type=ResourceType.GPU_MEMORY,
                        allocated_amount=self.constraints.max_gpu_memory_mb * 0.8,
                        max_amount=metrics.gpu.memory_total_mb if metrics.gpu else 6144,
                        usage_percent=80.0,
                        priority=0  # 最高優先度
                    ))
                
                elif "GPU temperature exceeded" in violation:
                    # GPU 計算を制限
                    emergency_allocations.append(ResourceAllocation(
                        resource_type=ResourceType.GPU_COMPUTE,
                        allocated_amount=70.0,  # 70%に制限
                        max_amount=100.0,
                        usage_percent=70.0,
                        priority=0
                    ))
                
                elif "CPU usage exceeded" in violation:
                    # CPU 使用率を制限
                    emergency_allocations.append(ResourceAllocation(
                        resource_type=ResourceType.CPU_CORES,
                        allocated_amount=self.constraints.max_cpu_usage_percent * 0.9,
                        max_amount=100.0,
                        usage_percent=self.constraints.max_cpu_usage_percent * 0.9,
                        priority=1
                    ))
            
            return emergency_allocations
            
        except Exception as e:
            logger.error(f"Emergency allocation creation failed: {e}")
            return []
    
    def get_current_allocations(self) -> Dict[ResourceType, ResourceAllocation]:
        """現在の配分取得"""
        return self.current_allocations.copy()
    
    def get_allocation_history(self, limit: Optional[int] = None) -> List[AllocationPlan]:
        """配分履歴取得"""
        if limit is None:
            return self.allocation_history.copy()
        return self.allocation_history[-limit:].copy()
    
    def get_management_stats(self) -> Dict[str, Any]:
        """管理統計取得"""
        try:
            recent_plans = self.allocation_history[-10:] if self.allocation_history else []
            
            stats = {
                "is_managing": self.is_managing,
                "reallocation_count": self.reallocation_count,
                "constraint_violations": self.constraint_violations,
                "performance_improvements": self.performance_improvements,
                "total_plans": len(self.allocation_history),
                "strategy": self.strategy.value
            }
            
            if recent_plans:
                avg_performance = statistics.mean(p.expected_performance for p in recent_plans)
                avg_efficiency = statistics.mean(p.resource_efficiency for p in recent_plans)
                
                stats.update({
                    "average_performance": avg_performance,
                    "average_efficiency": avg_efficiency,
                    "constraints_satisfaction_rate": sum(1 for p in recent_plans if p.constraints_satisfied) / len(recent_plans)
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Management stats calculation failed: {e}")
            return {"error": str(e)}
    
    def update_constraints(self, new_constraints: ResourceConstraints):
        """制約更新"""
        try:
            self.constraints = new_constraints
            logger.info("Resource constraints updated")
        except Exception as e:
            logger.error(f"Constraint update failed: {e}")
    
    def set_strategy(self, new_strategy: AllocationStrategy):
        """戦略変更"""
        try:
            self.strategy = new_strategy
            
            if new_strategy == AllocationStrategy.RTX4050_OPTIMIZED:
                self._setup_rtx4050_constraints()
            
            logger.info(f"Allocation strategy changed to: {new_strategy.value}")
        except Exception as e:
            logger.error(f"Strategy change failed: {e}")
    
    async def cleanup(self):
        """リソースクリーンアップ"""
        await self.stop_management()
        logger.info("Dynamic resource manager cleanup completed")


# 使用例とテスト用のヘルパー関数
async def demo_resource_management():
    """デモ用リソース管理実行"""
    from ..monitoring.system_monitor import SystemMonitor
    
    # システム監視初期化
    system_monitor = SystemMonitor(collection_interval=2.0)
    
    # リソース管理器初期化
    constraints = ResourceConstraints(
        max_gpu_memory_mb=5120.0,  # RTX 4050: 5GB制限
        max_gpu_temperature=75.0,
        max_cpu_usage_percent=85.0
    )
    
    resource_manager = DynamicResourceManager(
        system_monitor,
        constraints,
        AllocationStrategy.RTX4050_OPTIMIZED
    )
    
    try:
        print("=== Dynamic Resource Management Demo ===")
        
        # 監視開始
        await system_monitor.start_monitoring()
        await resource_manager.start_management(reallocation_interval=15.0)
        
        print("🔄 Running resource management for 45 seconds...")
        await asyncio.sleep(45)
        
        # 現在の配分表示
        allocations = resource_manager.get_current_allocations()
        print(f"\n📊 Current Resource Allocations:")
        for resource_type, allocation in allocations.items():
            print(f"  {resource_type.value}:")
            print(f"    Allocated: {allocation.allocated_amount:.1f}")
            print(f"    Usage: {allocation.usage_percent:.1f}%")
            print(f"    Priority: {allocation.priority}")
        
        # 統計表示
        stats = resource_manager.get_management_stats()
        print(f"\n📈 Management Statistics:")
        print(f"  Reallocations: {stats['reallocation_count']}")
        print(f"  Constraint Violations: {stats['constraint_violations']}")
        print(f"  Performance Improvements: {stats['performance_improvements']}")
        print(f"  Strategy: {stats['strategy']}")
        
        if "average_performance" in stats:
            print(f"  Average Performance: {stats['average_performance']:.2f}")
            print(f"  Average Efficiency: {stats['average_efficiency']:.2f}")
        
    finally:
        await resource_manager.cleanup()
        await system_monitor.cleanup()


if __name__ == "__main__":
    asyncio.run(demo_resource_management())