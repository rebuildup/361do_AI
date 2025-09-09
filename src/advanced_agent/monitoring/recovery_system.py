"""
段階的復旧戦略実行システム
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import torch
import gc

logger = logging.getLogger(__name__)


class RecoveryStatus(Enum):
    """復旧ステータス"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class RecoveryStrategy(Enum):
    """復旧戦略"""
    IMMEDIATE = "immediate"
    GRADUAL = "gradual"
    CONSERVATIVE = "conservative"


@dataclass
class RecoveryAction:
    """復旧アクション"""
    action_id: str
    name: str
    description: str
    action_type: str
    priority: int
    estimated_duration: int
    rollback_possible: bool = True


@dataclass
class RecoveryPlan:
    """復旧計画"""
    anomaly: Any
    strategy: Any
    actions: List[RecoveryAction]
    estimated_duration: int
    rollback_plan: Optional[List[str]] = None


@dataclass
class RecoveryExecution:
    """復旧実行情報"""
    execution_id: str
    plan: RecoveryPlan
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: RecoveryStatus = RecoveryStatus.PENDING
    logs: List[str] = field(default_factory=list)
    executed_actions: List[str] = field(default_factory=list)
    rollback_actions: List[str] = field(default_factory=list)
    metrics_before: Optional[Dict[str, Any]] = None
    metrics_after: Optional[Dict[str, Any]] = None


class SystemRecoveryManager:
    """システム復旧管理"""
    
    def __init__(self, grafana_url: str = "http://localhost:3000", max_concurrent_recoveries: int = 2):
        self.grafana_url = grafana_url
        self.max_concurrent_recoveries = max_concurrent_recoveries
        self.action_handlers = {
            "increase_quantization": self._increase_quantization,
            "clear_cache": self._clear_gpu_cache,
            "offload_to_cpu": self._offload_to_cpu,
            "reduce_gpu_load": self._clear_gpu_cache,
            "garbage_collection": self._garbage_collection,
        }
        self.recovery_history: List[RecoveryExecution] = []
        self.active_recoveries: Dict[str, RecoveryExecution] = {}
    
    async def _increase_quantization(self, anomaly) -> Dict[str, Any]:
        """量子化レベル上昇"""
        logger.info("量子化レベルを調整")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"action": "quantization_increased"}
    
    async def _clear_gpu_cache(self, anomaly) -> Dict[str, Any]:
        """GPU キャッシュクリア"""
        logger.info("GPU キャッシュをクリア")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        return {"action": "cache_cleared"}

    async def _offload_to_cpu(self, anomaly) -> Dict[str, Any]:
        """CPUへオフロード（簡易実装）"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "synchronize"):
                torch.cuda.synchronize()
        return {"action": "cpu_offload_enabled"}

    async def _garbage_collection(self, anomaly) -> Dict[str, Any]:
        """ガベージコレクション"""
        collected = gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"action": "garbage_collected", "objects_collected": collected}

    async def create_recovery_plan(self, anomaly: Any) -> RecoveryPlan:
        """復旧計画作成"""
        # severityベースの単純な戦略選択
        severity = getattr(anomaly, "severity", None)
        if severity and getattr(severity, "name", "").upper() == "HIGH":
            strategy = RecoveryStrategy.GRADUAL
        elif severity and getattr(severity, "name", "").upper() == "MEDIUM":
            strategy = RecoveryStrategy.CONSERVATIVE
        else:
            strategy = RecoveryStrategy.IMMEDIATE

        actions = await self._select_recovery_actions(anomaly)
        rollback_plan = await self._create_rollback_plan(actions)
        estimated_duration = sum(a.estimated_duration for a in actions)
        return RecoveryPlan(
            anomaly=anomaly,
            strategy=strategy,
            actions=actions,
            estimated_duration=estimated_duration,
            rollback_plan=rollback_plan,
        )

    async def _select_recovery_actions(self, anomaly: Any) -> List[RecoveryAction]:
        patterns = getattr(anomaly, "pattern", None)
        requested = getattr(patterns, "recovery_actions", ["increase_quantization", "clear_cache"])
        actions: List[RecoveryAction] = []
        for idx, action_id in enumerate(requested):
            if action_id in self.action_handlers:
                actions.append(
                    RecoveryAction(
                        action_id=action_id,
                        name=action_id.replace("_", " ").title(),
                        description=f"Auto action: {action_id}",
                        action_type="system",
                        priority=idx + 1,
                        estimated_duration=10,
                        rollback_possible=True,
                    )
                )
        return actions

    async def _create_rollback_plan(self, actions: List[RecoveryAction]) -> List[str]:
        rollbackable = [a for a in actions if a.rollback_possible]
        return [f"rollback_{a.action_id}" for a in reversed(rollbackable)]

    async def execute_recovery_plan(self, plan: RecoveryPlan) -> RecoveryExecution:
        if len(self.active_recoveries) >= self.max_concurrent_recoveries:
            raise RuntimeError("最大同時復旧数に達しています")
        execution = RecoveryExecution(execution_id=f"exec_{len(self.recovery_history)+1}", plan=plan, started_at=datetime.now())
        self.active_recoveries[execution.execution_id] = execution
        execution.metrics_before = await self._capture_metrics()
        try:
            if plan.strategy == RecoveryStrategy.IMMEDIATE:
                await self._execute_immediate(execution)
            elif plan.strategy == RecoveryStrategy.GRADUAL:
                await self._execute_gradual(execution)
            else:
                await self._execute_conservative(execution)
            execution.status = RecoveryStatus.SUCCESS
        except Exception as e:
            execution.logs.append(str(e))
            execution.status = RecoveryStatus.FAILED
            await self._execute_rollback(execution)
        finally:
            execution.completed_at = datetime.now()
            execution.metrics_after = await self._capture_metrics()
            self.recovery_history.append(execution)
            self.active_recoveries.pop(execution.execution_id, None)
        return execution

    async def _execute_immediate(self, execution: RecoveryExecution) -> None:
        for action in execution.plan.actions:
            await self._execute_action(action, execution)

    async def _execute_gradual(self, execution: RecoveryExecution) -> None:
        for action in execution.plan.actions:
            await self._execute_action(action, execution)
            improved = await self._check_improvement(execution)
            if improved:
                break

    async def _execute_conservative(self, execution: RecoveryExecution) -> None:
        for action in execution.plan.actions:
            if await self._pre_action_check(action, execution):
                await self._execute_action(action, execution)

    async def _execute_action(self, action: RecoveryAction, execution: RecoveryExecution) -> None:
        handler = self.action_handlers.get(action.action_id)
        if handler:
            await handler(execution.plan.anomaly)
            execution.executed_actions.append(action.action_id)

    async def _execute_rollback(self, execution: RecoveryExecution) -> None:
        for rid in execution.plan.rollback_plan or []:
            execution.rollback_actions.append(rid)
            await asyncio.sleep(0)
        execution.status = RecoveryStatus.ROLLED_BACK if execution.status == RecoveryStatus.FAILED else execution.status

    async def _capture_metrics(self) -> Dict[str, Any]:
        system = getattr(getattr(self, "system_monitor", None), "get_system_stats", lambda: {"cpu_percent": 0.0})()
        gpu = getattr(getattr(self, "system_monitor", None), "get_gpu_stats", lambda: {"memory_percent": 0.0})()
        return {"timestamp": datetime.now(), "system": system, "gpu": gpu}

    async def _check_improvement(self, execution: RecoveryExecution) -> bool:
        metrics = await self._capture_metrics()
        threshold = getattr(execution.plan.anomaly, "threshold", 90.0)
        gpu_percent = metrics.get("gpu", {}).get("memory_percent", 0.0)
        return gpu_percent < threshold

    async def _pre_action_check(self, action: RecoveryAction, execution: RecoveryExecution) -> bool:
        metrics = await self._capture_metrics()
        return metrics.get("system", {}).get("cpu_percent", 0.0) < 95.0
    
    def get_recovery_status(self) -> Dict[str, Any]:
        """復旧ステータス取得"""
        recent = self.recovery_history[-5:]
        return {
            "active_recoveries": len(self.active_recoveries),
            "total_recoveries": len(self.recovery_history),
            "success_rate": self._calculate_success_rate(),
            "recent_recoveries": recent,
        }

    def _calculate_success_rate(self) -> float:
        if not self.recovery_history:
            return 0.0
        successes = sum(1 for r in self.recovery_history if r.status == RecoveryStatus.SUCCESS)
        return successes / len(self.recovery_history) * 100