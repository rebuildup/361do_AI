"""
Advanced Agent Optimization System

HuggingFace + Prometheus による自動最適化システム
RTX 4050 6GB VRAM環境での動的リソース配分と最適化学習を提供します。

要件: 4.2, 4.4, 4.5
"""

from .auto_optimizer import (
    AutoOptimizer, OptimizationConfig, OptimizationResult,
    OptimizationStrategy
)
# 循環インポートを避けるため、必要時にインポート
try:
    from .prometheus_optimizer import (
        PrometheusOptimizer, MetricBasedOptimizer, OptimizationRule,
        MetricRule, OptimizationLearning
    )
except ImportError:
    # 依存関係が不足している場合はスキップ
    PrometheusOptimizer = None
    MetricBasedOptimizer = None
    OptimizationRule = None
    MetricRule = None
    OptimizationLearning = None
from .resource_manager import (
    DynamicResourceManager, ResourceConstraints, AllocationStrategy,
    ResourceAllocation, AllocationPlan, ResourceType
)

__all__ = [
    "AutoOptimizer",
    "OptimizationConfig",
    "OptimizationResult", 
    "OptimizationStrategy",
    "PrometheusOptimizer",
    "MetricBasedOptimizer",
    "OptimizationRule",
    "MetricRule",
    "OptimizationLearning",
    "DynamicResourceManager",
    "ResourceConstraints",
    "AllocationStrategy",
    "ResourceAllocation",
    "AllocationPlan",
    "ResourceType"
]