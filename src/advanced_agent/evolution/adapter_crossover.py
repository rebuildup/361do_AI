"""
PEFT + AutoGen アダプタ交配システム統合
PEFT の既存マージ機能による LoRA アダプタ重み交配とAutoGenエージェント による交配確率・変異率動的調整
"""

import asyncio
import time
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

try:
    import autogen
    from autogen import AssistantAgent, GroupChat, GroupChatManager
    from peft import PeftModel, LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    CROSSOVER_AVAILABLE = True
except ImportError:
    CROSSOVER_AVAILABLE = False
    PeftModel = None

from ..core.config import get_config
from ..core.logger import get_logger
from ..monitoring.system_monitor import SystemMonitor
from ..adaptation.peft_manager import PEFTAdapterPool, AdapterInfo, AdapterConfig, AdapterType
from ..adaptation.adapter_evaluator import AdapterEvaluator, EvaluationResult
from .evolutionary_system import Individual, EvolutionConfig


class CrossoverStrategy(Enum):
    """交配戦略"""
    WEIGHT_AVERAGING = "weight_averaging"
    SELECTIVE_MERGE = "selective_merge"
    LAYER_WISE_MERGE = "layer_wise_merge"
    ATTENTION_MERGE = "attention_merge"


class MergeMethod(Enum):
    """マージ方法"""
    LINEAR_INTERPOLATION = "linear_interpolation"
    WEIGHTED_AVERAGE = "weighted_average"
    TASK_VECTOR = "task_vector"
    TIES_MERGE = "ties_merge"


@dataclass
class CrossoverConfig:
    """交配設定"""
    strategy: CrossoverStrategy = CrossoverStrategy.WEIGHT_AVERAGING
    merge_method: MergeMethod = MergeMethod.WEIGHTED_AVERAGE
    
    # 重み設定
    parent1_weight: float = 0.5
    parent2_weight: float = 0.5
    
    # 選択的マージ設定
    selection_threshold: float = 0.1
    layer_selection_ratio: float = 0.5
    
    # 動的調整設定
    adaptive_weights: bool = True
    performance_based_weighting: bool = True
    
    # AutoGen設定
    use_autogen_coordination: bool = True
    negotiation_rounds: int = 3
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossoverResult:
    """交配結果"""
    parent1_name: str
    parent2_name: str
    offspring_name: str
    
    # 交配情報
    strategy_used: CrossoverStrategy
    merge_method_used: MergeMethod
    final_weights: Dict[str, float]
    
    # 品質情報
    parent1_fitness: float = 0.0
    parent2_fitness: float = 0.0
    offspring_fitness: float = 0.0
    
    # 統計
    crossover_time: float = 0.0
    parameter_count: int = 0
    merge_success_rate: float = 0.0
    
    # AutoGen情報
    negotiation_log: List[str] = field(default_factory=list)
    consensus_reached: bool = False
    
    # 結果
    success: bool = True
    error_message: Optional[str] = None
    
    timestamp: datetime = field(default_factory=datetime.now)


class CrossoverNegotiatorAgent(AssistantAgent if CROSSOVER_AVAILABLE else object):
    """交配交渉エージェント（AutoGen AssistantAgent拡張）"""
    
    def __init__(self, name: str, individual: Individual, llm_config: Dict[str, Any]):
        if not CROSSOVER_AVAILABLE:
            super().__init__()
            self.name = name
            self.individual = individual
            return
        
        super().__init__(
            name=name,
            llm_config=llm_config,
            system_message=f"""
あなたは進化的学習システムにおけるアダプタ交配の交渉エージェントです。

個体情報:
- 名前: {individual.adapter_name}
- 適応度: {individual.fitness:.4f}
- 世代: {individual.generation}
- 遺伝子: {individual.genes}

あなたの役割:
1. 他のエージェントとの交配交渉
2. 最適な交配重みの提案
3. 交配戦略の合意形成

交渉時の指針:
- 自身の強みを活かせる交配を提案
- 相手の特徴を考慮した建設的な提案
- 全体の性能向上を目指した協調的態度
- 具体的な数値（重み、閾値等）を含む提案

回答は簡潔で具体的にしてください。
"""
        )
        self.individual = individual
    
    def propose_crossover_weights(self, partner_individual: Individual) -> Dict[str, float]:
        """交配重み提案"""
        
        # 適応度に基づく重み提案
        total_fitness = self.individual.fitness + partner_individual.fitness
        
        if total_fitness > 0:
            self_weight = self.individual.fitness / total_fitness
            partner_weight = partner_individual.fitness / total_fitness
        else:
            self_weight = 0.5
            partner_weight = 0.5
        
        return {
            "self_weight": self_weight,
            "partner_weight": partner_weight,
            "confidence": min(self.individual.fitness, partner_individual.fitness)
        }
    
    def evaluate_crossover_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """交配提案評価"""
        
        evaluation = {
            "acceptance": False,
            "confidence": 0.0,
            "suggestions": []
        }
        
        # 提案された重みを評価
        proposed_self_weight = proposal.get("self_weight", 0.5)
        
        # 自身の適応度が高い場合、より高い重みを期待
        expected_weight = max(0.3, self.individual.fitness)
        
        if proposed_self_weight >= expected_weight * 0.8:
            evaluation["acceptance"] = True
            evaluation["confidence"] = min(1.0, proposed_self_weight / expected_weight)
        else:
            evaluation["suggestions"].append(f"自身の重みを{expected_weight:.2f}以上に調整することを提案します")
        
        return evaluation


class AdapterCrossoverSystem:
    """PEFT + AutoGen アダプタ交配システム"""
    
    def __init__(
        self,
        peft_pool: PEFTAdapterPool,
        evaluator: AdapterEvaluator,
        system_monitor: Optional[SystemMonitor] = None
    ):
        self.peft_pool = peft_pool
        self.evaluator = evaluator
        self.system_monitor = system_monitor
        
        self.config = get_config()
        self.logger = get_logger()
        
        # 交配履歴
        self.crossover_history: List[CrossoverResult] = []
        
        # AutoGenエージェント
        self.negotiator_agents: Dict[str, CrossoverNegotiatorAgent] = {}
        
        # 統計
        self.crossover_stats = {
            "total_crossovers": 0,
            "successful_crossovers": 0,
            "failed_crossovers": 0,
            "average_improvement": 0.0,
            "best_offspring_fitness": 0.0
        }
        
        self.logger.log_startup(
            component="adapter_crossover_system",
            version="1.0.0",
            config_summary={
                "crossover_available": CROSSOVER_AVAILABLE,
                "peft_pool_adapters": len(self.peft_pool.adapters)
            }
        )
    
    async def initialize(self) -> bool:
        """交配システム初期化"""
        try:
            if not CROSSOVER_AVAILABLE:
                self.logger.log_alert(
                    alert_type="crossover_unavailable",
                    severity="WARNING",
                    message="Crossover dependencies not available"
                )
                return False
            
            self.logger.log_startup(
                component="crossover_system_initialized",
                version="1.0.0",
                config_summary={
                    "initialization_complete": True
                }
            )
            
            return True
            
        except Exception as e:
            self.logger.log_alert(
                alert_type="crossover_initialization_failed",
                severity="ERROR",
                message=f"Crossover system initialization failed: {e}"
            )
            return False    
    async def crossover_adapters(
        self,
        parent1: Individual,
        parent2: Individual,
        crossover_config: CrossoverConfig
    ) -> CrossoverResult:
        """アダプタ交配実行"""
        
        start_time = time.time()
        
        try:
            # AutoGen交渉（有効な場合）
            negotiation_result = None
            if crossover_config.use_autogen_coordination and CROSSOVER_AVAILABLE:
                negotiation_result = await self._negotiate_crossover(
                    parent1, parent2, crossover_config
                )
            
            # 交配重み決定
            final_weights = self._determine_crossover_weights(
                parent1, parent2, crossover_config, negotiation_result
            )
            
            # アダプタ重み交配実行
            offspring_name = f"cross_{parent1.adapter_name}_{parent2.adapter_name}_{int(time.time() % 10000)}"
            
            offspring_adapter = await self._perform_weight_crossover(
                parent1.adapter_name,
                parent2.adapter_name,
                offspring_name,
                final_weights,
                crossover_config
            )
            
            # 結果評価
            offspring_fitness = await self._evaluate_offspring(offspring_name)
            
            # 結果作成
            result = CrossoverResult(
                parent1_name=parent1.adapter_name,
                parent2_name=parent2.adapter_name,
                offspring_name=offspring_name,
                strategy_used=crossover_config.strategy,
                merge_method_used=crossover_config.merge_method,
                final_weights=final_weights,
                parent1_fitness=parent1.fitness,
                parent2_fitness=parent2.fitness,
                offspring_fitness=offspring_fitness,
                crossover_time=time.time() - start_time,
                parameter_count=self._count_parameters(offspring_adapter),
                negotiation_log=negotiation_result.get("log", []) if negotiation_result else [],
                consensus_reached=negotiation_result.get("consensus", False) if negotiation_result else False,
                success=True
            )
            
            # 統計更新
            self.crossover_stats["total_crossovers"] += 1
            self.crossover_stats["successful_crossovers"] += 1
            self._update_crossover_stats(result)
            
            # 履歴に追加
            self.crossover_history.append(result)
            
            self.logger.log_performance_metric(
                metric_name="adapter_crossover_completed",
                value=result.crossover_time,
                unit="seconds",
                component="adapter_crossover_system"
            )
            
            return result
            
        except Exception as e:
            # エラー結果作成
            error_result = CrossoverResult(
                parent1_name=parent1.adapter_name,
                parent2_name=parent2.adapter_name,
                offspring_name="",
                strategy_used=crossover_config.strategy,
                merge_method_used=crossover_config.merge_method,
                final_weights={},
                crossover_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
            
            # 統計更新
            self.crossover_stats["total_crossovers"] += 1
            self.crossover_stats["failed_crossovers"] += 1
            
            # 履歴に追加
            self.crossover_history.append(error_result)
            
            self.logger.log_alert(
                alert_type="adapter_crossover_failed",
                severity="ERROR",
                message=f"Adapter crossover failed: {e}"
            )
            
            raise
    
    async def _negotiate_crossover(
        self,
        parent1: Individual,
        parent2: Individual,
        crossover_config: CrossoverConfig
    ) -> Dict[str, Any]:
        """AutoGen交配交渉"""
        
        if not CROSSOVER_AVAILABLE:
            return {"consensus": False, "log": ["AutoGen not available"]}
        
        try:
            # LLM設定
            llm_config = {
                "config_list": [
                    {
                        "model": "gpt-3.5-turbo",
                        "api_key": "dummy_key"  # 実際のAPIキーに変更
                    }
                ],
                "temperature": 0.7
            }
            
            # 交渉エージェント作成
            agent1 = CrossoverNegotiatorAgent(
                name=f"negotiator_{parent1.adapter_name}",
                individual=parent1,
                llm_config=llm_config
            )
            
            agent2 = CrossoverNegotiatorAgent(
                name=f"negotiator_{parent2.adapter_name}",
                individual=parent2,
                llm_config=llm_config
            )
            
            # 交渉ログ
            negotiation_log = []
            
            # 初期提案
            proposal1 = agent1.propose_crossover_weights(parent2)
            proposal2 = agent2.propose_crossover_weights(parent1)
            
            negotiation_log.append(f"Agent1 proposal: {proposal1}")
            negotiation_log.append(f"Agent2 proposal: {proposal2}")
            
            # 提案評価
            eval1 = agent1.evaluate_crossover_proposal(proposal2)
            eval2 = agent2.evaluate_crossover_proposal(proposal1)
            
            negotiation_log.append(f"Agent1 evaluation: {eval1}")
            negotiation_log.append(f"Agent2 evaluation: {eval2}")
            
            # 合意判定
            consensus = eval1["acceptance"] and eval2["acceptance"]
            
            # 最終重み決定
            if consensus:
                final_weights = {
                    "parent1_weight": (proposal1["self_weight"] + proposal2["partner_weight"]) / 2,
                    "parent2_weight": (proposal1["partner_weight"] + proposal2["self_weight"]) / 2
                }
            else:
                # 適応度ベースのフォールバック
                total_fitness = parent1.fitness + parent2.fitness
                if total_fitness > 0:
                    final_weights = {
                        "parent1_weight": parent1.fitness / total_fitness,
                        "parent2_weight": parent2.fitness / total_fitness
                    }
                else:
                    final_weights = {"parent1_weight": 0.5, "parent2_weight": 0.5}
            
            return {
                "consensus": consensus,
                "final_weights": final_weights,
                "log": negotiation_log
            }
            
        except Exception as e:
            self.logger.log_alert(
                alert_type="crossover_negotiation_failed",
                severity="WARNING",
                message=f"Crossover negotiation failed: {e}"
            )
            
            return {
                "consensus": False,
                "log": [f"Negotiation error: {e}"]
            }
    
    def _determine_crossover_weights(
        self,
        parent1: Individual,
        parent2: Individual,
        crossover_config: CrossoverConfig,
        negotiation_result: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """交配重み決定"""
        
        # AutoGen交渉結果を優先
        if negotiation_result and negotiation_result.get("consensus"):
            return negotiation_result["final_weights"]
        
        # 性能ベース重み付け
        if crossover_config.performance_based_weighting:
            total_fitness = parent1.fitness + parent2.fitness
            
            if total_fitness > 0:
                parent1_weight = parent1.fitness / total_fitness
                parent2_weight = parent2.fitness / total_fitness
            else:
                parent1_weight = 0.5
                parent2_weight = 0.5
        else:
            # 設定値使用
            parent1_weight = crossover_config.parent1_weight
            parent2_weight = crossover_config.parent2_weight
        
        # 正規化
        total_weight = parent1_weight + parent2_weight
        if total_weight > 0:
            parent1_weight /= total_weight
            parent2_weight /= total_weight
        
        return {
            "parent1_weight": parent1_weight,
            "parent2_weight": parent2_weight
        }
    
    async def _perform_weight_crossover(
        self,
        parent1_name: str,
        parent2_name: str,
        offspring_name: str,
        weights: Dict[str, float],
        crossover_config: CrossoverConfig
    ) -> PeftModel:
        """重み交配実行"""
        
        # 親アダプタロード
        await self.peft_pool.load_adapter(parent1_name)
        await self.peft_pool.load_adapter(parent2_name)
        
        parent1_model = self.peft_pool.active_adapters[parent1_name]
        parent2_model = self.peft_pool.active_adapters[parent2_name]
        
        # 交配戦略に応じた処理
        if crossover_config.strategy == CrossoverStrategy.WEIGHT_AVERAGING:
            offspring_model = await self._weight_averaging_crossover(
                parent1_model, parent2_model, weights, crossover_config
            )
        elif crossover_config.strategy == CrossoverStrategy.SELECTIVE_MERGE:
            offspring_model = await self._selective_merge_crossover(
                parent1_model, parent2_model, weights, crossover_config
            )
        elif crossover_config.strategy == CrossoverStrategy.LAYER_WISE_MERGE:
            offspring_model = await self._layer_wise_merge_crossover(
                parent1_model, parent2_model, weights, crossover_config
            )
        else:
            offspring_model = await self._weight_averaging_crossover(
                parent1_model, parent2_model, weights, crossover_config
            )
        
        # 子アダプタをプールに追加
        offspring_config = AdapterConfig(
            name=offspring_name,
            adapter_type=AdapterType.LORA,
            description=f"Crossover of {parent1_name} and {parent2_name}"
        )
        
        # アダプタ情報作成
        from ..adaptation.peft_manager import AdapterInfo, AdapterStatus
        
        offspring_info = AdapterInfo(
            config=offspring_config,
            status=AdapterStatus.ACTIVE,
            parameter_count=self._count_parameters(offspring_model)
        )
        
        self.peft_pool.adapters[offspring_name] = offspring_info
        self.peft_pool.active_adapters[offspring_name] = offspring_model
        
        return offspring_model
    
    async def _weight_averaging_crossover(
        self,
        parent1_model: PeftModel,
        parent2_model: PeftModel,
        weights: Dict[str, float],
        crossover_config: CrossoverConfig
    ) -> PeftModel:
        """重み平均交配"""
        
        # ベースモデル取得
        base_model = parent1_model.base_model
        
        # 新しいPEFTモデル作成
        peft_config = parent1_model.peft_config
        offspring_model = get_peft_model(base_model, peft_config)
        
        # 重み平均計算
        w1 = weights["parent1_weight"]
        w2 = weights["parent2_weight"]
        
        # アダプタ重みを平均
        with torch.no_grad():
            for name, param in offspring_model.named_parameters():
                if "lora" in name.lower():
                    param1 = dict(parent1_model.named_parameters())[name]
                    param2 = dict(parent2_model.named_parameters())[name]
                    
                    # 重み付き平均
                    param.data = w1 * param1.data + w2 * param2.data
        
        return offspring_model
    
    async def _selective_merge_crossover(
        self,
        parent1_model: PeftModel,
        parent2_model: PeftModel,
        weights: Dict[str, float],
        crossover_config: CrossoverConfig
    ) -> PeftModel:
        """選択的マージ交配"""
        
        # ベースモデル取得
        base_model = parent1_model.base_model
        peft_config = parent1_model.peft_config
        offspring_model = get_peft_model(base_model, peft_config)
        
        # 選択的マージ
        threshold = crossover_config.selection_threshold
        
        with torch.no_grad():
            for name, param in offspring_model.named_parameters():
                if "lora" in name.lower():
                    param1 = dict(parent1_model.named_parameters())[name]
                    param2 = dict(parent2_model.named_parameters())[name]
                    
                    # パラメータの重要度計算（L2ノルム）
                    importance1 = torch.norm(param1.data)
                    importance2 = torch.norm(param2.data)
                    
                    # より重要なパラメータを選択
                    if importance1 > importance2 + threshold:
                        param.data = param1.data
                    elif importance2 > importance1 + threshold:
                        param.data = param2.data
                    else:
                        # 重要度が近い場合は重み付き平均
                        w1 = weights["parent1_weight"]
                        w2 = weights["parent2_weight"]
                        param.data = w1 * param1.data + w2 * param2.data
        
        return offspring_model
    
    async def _layer_wise_merge_crossover(
        self,
        parent1_model: PeftModel,
        parent2_model: PeftModel,
        weights: Dict[str, float],
        crossover_config: CrossoverConfig
    ) -> PeftModel:
        """層別マージ交配"""
        
        # ベースモデル取得
        base_model = parent1_model.base_model
        peft_config = parent1_model.peft_config
        offspring_model = get_peft_model(base_model, peft_config)
        
        # 層別選択
        layer_names = [name for name, _ in offspring_model.named_parameters() if "lora" in name.lower()]
        selection_ratio = crossover_config.layer_selection_ratio
        
        # ランダムに層を選択
        import random
        selected_layers = random.sample(layer_names, int(len(layer_names) * selection_ratio))
        
        with torch.no_grad():
            for name, param in offspring_model.named_parameters():
                if "lora" in name.lower():
                    param1 = dict(parent1_model.named_parameters())[name]
                    param2 = dict(parent2_model.named_parameters())[name]
                    
                    if name in selected_layers:
                        # 選択された層はparent1から
                        param.data = param1.data
                    else:
                        # その他の層はparent2から
                        param.data = param2.data
        
        return offspring_model
    
    def _count_parameters(self, model: PeftModel) -> int:
        """パラメータ数カウント"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    async def _evaluate_offspring(self, offspring_name: str) -> float:
        """子個体評価"""
        
        try:
            from ..adaptation.adapter_evaluator import EvaluationConfig, EvaluationTask, EvaluationMetric
            
            eval_config = EvaluationConfig(
                task=EvaluationTask.ACCURACY,
                metrics=[EvaluationMetric.ACCURACY],
                max_samples=20  # 高速評価
            )
            
            result = await self.evaluator.evaluate_adapter(offspring_name, eval_config)
            
            return result.metrics.get("accuracy", 0.0)
            
        except Exception as e:
            self.logger.log_alert(
                alert_type="offspring_evaluation_failed",
                severity="WARNING",
                message=f"Failed to evaluate offspring {offspring_name}: {e}"
            )
            return 0.0
    
    def _update_crossover_stats(self, result: CrossoverResult) -> None:
        """交配統計更新"""
        
        if result.success:
            # 改善度計算
            parent_avg_fitness = (result.parent1_fitness + result.parent2_fitness) / 2
            improvement = result.offspring_fitness - parent_avg_fitness
            
            # 平均改善度更新
            successful_count = self.crossover_stats["successful_crossovers"]
            current_avg = self.crossover_stats["average_improvement"]
            
            new_avg = ((current_avg * (successful_count - 1)) + improvement) / successful_count
            self.crossover_stats["average_improvement"] = new_avg
            
            # 最高適応度更新
            if result.offspring_fitness > self.crossover_stats["best_offspring_fitness"]:
                self.crossover_stats["best_offspring_fitness"] = result.offspring_fitness
    
    def get_crossover_stats(self) -> Dict[str, Any]:
        """交配統計取得"""
        stats = self.crossover_stats.copy()
        
        # 成功率計算
        if stats["total_crossovers"] > 0:
            stats["success_rate"] = stats["successful_crossovers"] / stats["total_crossovers"]
        else:
            stats["success_rate"] = 0.0
        
        # 最近の交配統計
        recent_crossovers = self.crossover_history[-5:]  # 最新5件
        if recent_crossovers:
            successful_recent = [c for c in recent_crossovers if c.success]
            if successful_recent:
                stats["recent_average_time"] = sum(c.crossover_time for c in successful_recent) / len(successful_recent)
                stats["recent_average_improvement"] = sum(
                    c.offspring_fitness - (c.parent1_fitness + c.parent2_fitness) / 2
                    for c in successful_recent
                ) / len(successful_recent)
        
        return stats
    
    def get_crossover_history(
        self,
        limit: Optional[int] = None,
        success_only: bool = False
    ) -> List[CrossoverResult]:
        """交配履歴取得"""
        
        history = self.crossover_history
        
        if success_only:
            history = [c for c in history if c.success]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    async def shutdown(self) -> None:
        """交配システム終了"""
        
        final_stats = self.get_crossover_stats()
        
        self.logger.log_shutdown(
            component="adapter_crossover_system",
            uptime_seconds=0,  # TODO: 実際の稼働時間計算
            final_stats=final_stats
        )


# 便利関数
async def create_adapter_crossover_system(
    peft_pool: PEFTAdapterPool,
    evaluator: AdapterEvaluator,
    system_monitor: Optional[SystemMonitor] = None
) -> AdapterCrossoverSystem:
    """アダプタ交配システム作成・初期化"""
    
    system = AdapterCrossoverSystem(peft_pool, evaluator, system_monitor)
    
    if await system.initialize():
        return system
    else:
        raise RuntimeError("Failed to initialize adapter crossover system")


# 使用例
async def main():
    """テスト用メイン関数"""
    try:
        from ..adaptation.peft_manager import create_peft_adapter_pool
        from ..adaptation.adapter_evaluator import create_adapter_evaluator
        
        # 依存関係作成
        peft_pool = await create_peft_adapter_pool("microsoft/DialoGPT-small")
        evaluator = await create_adapter_evaluator(peft_pool)
        
        # 交配システム作成
        crossover_system = await create_adapter_crossover_system(peft_pool, evaluator)
        
        print("=== Adapter Crossover System Test ===")
        
        # 交配設定作成
        crossover_config = CrossoverConfig(
            strategy=CrossoverStrategy.WEIGHT_AVERAGING,
            merge_method=MergeMethod.WEIGHTED_AVERAGE,
            use_autogen_coordination=False  # テスト用に無効化
        )
        
        print(f"Crossover Config: {crossover_config}")
        
        # 統計取得
        stats = crossover_system.get_crossover_stats()
        print(f"Crossover Stats: {stats}")
        
        await crossover_system.shutdown()
        await evaluator.shutdown()
        await peft_pool.shutdown()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())