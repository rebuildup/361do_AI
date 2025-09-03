"""
AutoGen 進化的エージェント群統合システム
AutoGen AssistantAgent による モデル交配・自然選択と世代管理・性能追跡
"""

import asyncio
import time
import random
import json
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod

try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    AssistantAgent = None
    GroupChat = None

from ..core.config import get_config
from ..core.logger import get_logger
from ..monitoring.system_monitor import SystemMonitor
from ..adaptation.peft_manager import PEFTAdapterPool, AdapterInfo, AdapterConfig, AdapterType
from ..adaptation.adapter_evaluator import AdapterEvaluator, EvaluationResult, EvaluationTask


class EvolutionStrategy(Enum):
    """進化戦略"""
    GENETIC_ALGORITHM = "genetic_algorithm"
    EVOLUTIONARY_STRATEGY = "evolutionary_strategy"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    PARTICLE_SWARM = "particle_swarm"


class SelectionMethod(Enum):
    """選択方法"""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK_BASED = "rank_based"
    ELITIST = "elitist"


class CrossoverMethod(Enum):
    """交配方法"""
    UNIFORM = "uniform"
    SINGLE_POINT = "single_point"
    MULTI_POINT = "multi_point"
    ARITHMETIC = "arithmetic"


class MutationMethod(Enum):
    """変異方法"""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    POLYNOMIAL = "polynomial"
    ADAPTIVE = "adaptive"


@dataclass
class EvolutionConfig:
    """進化設定"""
    # 基本設定
    population_size: int = 20
    max_generations: int = 50
    elite_size: int = 2
    
    # 進化パラメータ
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    selection_pressure: float = 2.0
    
    # 戦略設定
    evolution_strategy: EvolutionStrategy = EvolutionStrategy.GENETIC_ALGORITHM
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    crossover_method: CrossoverMethod = CrossoverMethod.UNIFORM
    mutation_method: MutationMethod = MutationMethod.GAUSSIAN
    
    # 評価設定
    fitness_function: str = "weighted_performance"
    evaluation_tasks: List[EvaluationTask] = field(default_factory=lambda: [EvaluationTask.ACCURACY])
    
    # 収束設定
    convergence_threshold: float = 0.001
    stagnation_limit: int = 10
    
    # その他
    seed: int = 42
    parallel_evaluation: bool = True
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Individual:
    """個体（アダプタ）"""
    adapter_name: str
    adapter_config: AdapterConfig
    
    # 遺伝子（パラメータ）
    genes: Dict[str, Any] = field(default_factory=dict)
    
    # 適応度
    fitness: float = 0.0
    fitness_components: Dict[str, float] = field(default_factory=dict)
    
    # 世代情報
    generation: int = 0
    parent_names: List[str] = field(default_factory=list)
    
    # 評価結果
    evaluation_results: List[EvaluationResult] = field(default_factory=list)
    
    # メタデータ
    birth_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Generation:
    """世代"""
    generation_number: int
    individuals: List[Individual]
    
    # 統計
    best_fitness: float = 0.0
    average_fitness: float = 0.0
    worst_fitness: float = 0.0
    fitness_std: float = 0.0
    
    # 多様性指標
    diversity_score: float = 0.0
    
    # 時間情報
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    evaluation_time: float = 0.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvolutionResult:
    """進化結果"""
    config: EvolutionConfig
    
    # 世代履歴
    generations: List[Generation] = field(default_factory=list)
    
    # 最終結果
    best_individual: Optional[Individual] = None
    final_population: List[Individual] = field(default_factory=list)
    
    # 統計
    total_generations: int = 0
    total_evaluations: int = 0
    convergence_achieved: bool = False
    
    # 時間統計
    total_time: float = 0.0
    average_generation_time: float = 0.0
    
    # 成功情報
    success: bool = True
    error_message: Optional[str] = None
    
    timestamp: datetime = field(default_factory=datetime.now)


class FitnessFunction(ABC):
    """適応度関数の抽象基底クラス"""
    
    @abstractmethod
    def calculate(self, individual: Individual, evaluation_results: List[EvaluationResult]) -> float:
        """適応度計算"""
        pass


class WeightedPerformanceFitness(FitnessFunction):
    """重み付き性能適応度関数"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "accuracy": 0.4,
            "f1": 0.3,
            "efficiency": 0.2,
            "diversity": 0.1
        }
    
    def calculate(self, individual: Individual, evaluation_results: List[EvaluationResult]) -> float:
        """重み付き適応度計算"""
        if not evaluation_results:
            return 0.0
        
        fitness_components = {}
        
        # 性能メトリクス
        for result in evaluation_results:
            for metric_name, value in result.metrics.items():
                if metric_name in self.weights:
                    fitness_components[metric_name] = value
        
        # 効率性（処理時間の逆数）
        if evaluation_results:
            avg_time = sum(r.evaluation_time for r in evaluation_results) / len(evaluation_results)
            fitness_components["efficiency"] = 1.0 / (1.0 + avg_time)
        
        # 多様性（パラメータの分散）
        if individual.genes:
            param_values = [v for v in individual.genes.values() if isinstance(v, (int, float))]
            if param_values:
                fitness_components["diversity"] = np.std(param_values) / (np.mean(param_values) + 1e-8)
        
        # 重み付き合計
        weighted_fitness = sum(
            self.weights.get(component, 0.0) * value
            for component, value in fitness_components.items()
        )
        
        individual.fitness_components = fitness_components
        return weighted_fitness


class EvolutionaryAgent(AssistantAgent if AUTOGEN_AVAILABLE else object):
    """進化的エージェント（AutoGen AssistantAgent拡張）"""
    
    def __init__(self, name: str, individual: Individual, llm_config: Dict[str, Any]):
        if not AUTOGEN_AVAILABLE:
            super().__init__()
            self.name = name
            self.individual = individual
            return
        
        super().__init__(
            name=name,
            llm_config=llm_config,
            system_message=f"""
あなたは進化的学習システムの一部として動作するAIエージェントです。
個体名: {individual.adapter_name}
世代: {individual.generation}
現在の適応度: {individual.fitness:.4f}

あなたの役割:
1. 他のエージェントとの協調による問題解決
2. 自身の性能向上のための戦略提案
3. 進化プロセスへの貢献

常に建設的で協力的な態度を保ち、集団全体の性能向上を目指してください。
"""
        )
        self.individual = individual
    
    def get_fitness_report(self) -> str:
        """適応度レポート取得"""
        components = self.individual.fitness_components
        report = f"個体 {self.individual.adapter_name} の適応度レポート:\n"
        report += f"総合適応度: {self.individual.fitness:.4f}\n"
        
        if components:
            report += "構成要素:\n"
            for component, value in components.items():
                report += f"  - {component}: {value:.4f}\n"
        
        return report
    
    def suggest_improvement(self) -> str:
        """改善提案"""
        components = self.individual.fitness_components
        
        if not components:
            return "評価データが不足しているため、まず性能評価を実行することを提案します。"
        
        # 最も低いスコアの要素を特定
        min_component = min(components.items(), key=lambda x: x[1])
        
        suggestions = {
            "accuracy": "精度向上のため、より多様な学習データでの追加学習を提案します。",
            "f1": "F1スコア向上のため、クラス不均衡に対応した学習戦略を提案します。",
            "efficiency": "効率性向上のため、モデル量子化や軽量化を提案します。",
            "diversity": "多様性向上のため、パラメータ範囲の拡大を提案します。"
        }
        
        return suggestions.get(min_component[0], "継続的な最適化を提案します。")


class EvolutionarySystem:
    """AutoGen 進化的エージェントシステム"""
    
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
        
        # 進化状態
        self.current_generation = 0
        self.population: List[Individual] = []
        self.generation_history: List[Generation] = []
        
        # AutoGenエージェント
        self.agents: Dict[str, EvolutionaryAgent] = {}
        self.group_chat: Optional[GroupChat] = None
        self.chat_manager: Optional[GroupChatManager] = None
        
        # 適応度関数
        self.fitness_functions: Dict[str, FitnessFunction] = {
            "weighted_performance": WeightedPerformanceFitness()
        }
        
        # 統計
        self.evolution_stats = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "best_fitness_achieved": 0.0,
            "total_evaluations": 0
        }
        
        self.logger.log_startup(
            component="evolutionary_system",
            version="1.0.0",
            config_summary={
                "autogen_available": AUTOGEN_AVAILABLE,
                "peft_pool_adapters": len(self.peft_pool.adapters),
                "fitness_functions": len(self.fitness_functions)
            }
        )
    
    async def initialize(self) -> bool:
        """進化システム初期化"""
        try:
            if not AUTOGEN_AVAILABLE:
                self.logger.log_alert(
                    alert_type="autogen_unavailable",
                    severity="WARNING",
                    message="AutoGen not available, using simplified evolution"
                )
                return True  # AutoGenなしでも基本機能は動作
            
            self.logger.log_startup(
                component="evolutionary_system_initialized",
                version="1.0.0",
                config_summary={
                    "initialization_complete": True
                }
            )
            
            return True
            
        except Exception as e:
            self.logger.log_alert(
                alert_type="evolution_initialization_failed",
                severity="ERROR",
                message=f"Evolutionary system initialization failed: {e}"
            )
            return False
    
    def create_initial_population(
        self,
        evolution_config: EvolutionConfig,
        base_adapter_config: AdapterConfig
    ) -> List[Individual]:
        """初期個体群作成"""
        
        population = []
        
        for i in range(evolution_config.population_size):
            # 遺伝子（パラメータ）生成
            genes = self._generate_random_genes(base_adapter_config)
            
            # アダプタ設定作成
            adapter_config = self._genes_to_adapter_config(genes, base_adapter_config)
            adapter_config.name = f"evo_gen0_ind{i}"
            
            # 個体作成
            individual = Individual(
                adapter_name=adapter_config.name,
                adapter_config=adapter_config,
                genes=genes,
                generation=0
            )
            
            population.append(individual)
        
        return population
    
    def _generate_random_genes(self, base_config: AdapterConfig) -> Dict[str, Any]:
        """ランダム遺伝子生成"""
        
        genes = {}
        
        # LoRAパラメータの範囲
        if base_config.adapter_type == AdapterType.LORA:
            genes["r"] = random.randint(8, 128)
            genes["lora_alpha"] = random.randint(8, 64)
            genes["lora_dropout"] = random.uniform(0.05, 0.3)
        
        # 学習パラメータ
        genes["learning_rate"] = random.uniform(1e-5, 1e-3)
        genes["batch_size"] = random.choice([1, 2, 4, 8])
        genes["warmup_steps"] = random.randint(50, 500)
        
        return genes
    
    def _genes_to_adapter_config(
        self,
        genes: Dict[str, Any],
        base_config: AdapterConfig
    ) -> AdapterConfig:
        """遺伝子からアダプタ設定作成"""
        
        # ベース設定をコピー
        new_config = AdapterConfig(
            name=base_config.name,
            adapter_type=base_config.adapter_type,
            task_type=base_config.task_type,
            target_modules=base_config.target_modules.copy() if base_config.target_modules else None,
            description=base_config.description,
            tags=base_config.tags.copy()
        )
        
        # 遺伝子から設定更新
        if "r" in genes:
            new_config.r = genes["r"]
        if "lora_alpha" in genes:
            new_config.lora_alpha = genes["lora_alpha"]
        if "lora_dropout" in genes:
            new_config.lora_dropout = genes["lora_dropout"]
        
        return new_config
    
    async def evaluate_population(
        self,
        population: List[Individual],
        evolution_config: EvolutionConfig
    ) -> None:
        """個体群評価"""
        
        fitness_function = self.fitness_functions[evolution_config.fitness_function]
        
        if evolution_config.parallel_evaluation:
            # 並列評価
            tasks = []
            for individual in population:
                task = self._evaluate_individual(individual, evolution_config, fitness_function)
                tasks.append(task)
            
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # 逐次評価
            for individual in population:
                await self._evaluate_individual(individual, evolution_config, fitness_function)
    
    async def _evaluate_individual(
        self,
        individual: Individual,
        evolution_config: EvolutionConfig,
        fitness_function: FitnessFunction
    ) -> None:
        """個体評価"""
        
        try:
            # アダプタが存在しない場合は作成
            if individual.adapter_name not in self.peft_pool.adapters:
                await self.peft_pool.create_adapter(individual.adapter_config)
            
            # 各タスクで評価
            evaluation_results = []
            
            for task in evolution_config.evaluation_tasks:
                from ..adaptation.adapter_evaluator import EvaluationConfig, EvaluationMetric
                
                eval_config = EvaluationConfig(
                    task=task,
                    metrics=[EvaluationMetric.ACCURACY, EvaluationMetric.F1_SCORE],
                    max_samples=50  # 高速評価のため制限
                )
                
                result = await self.evaluator.evaluate_adapter(
                    individual.adapter_name,
                    eval_config
                )
                
                evaluation_results.append(result)
            
            # 適応度計算
            individual.evaluation_results = evaluation_results
            individual.fitness = fitness_function.calculate(individual, evaluation_results)
            
            self.evolution_stats["total_evaluations"] += len(evaluation_results)
            
        except Exception as e:
            self.logger.log_alert(
                alert_type="individual_evaluation_failed",
                severity="WARNING",
                message=f"Failed to evaluate individual {individual.adapter_name}: {e}"
            )
            individual.fitness = 0.0
    
    def select_parents(
        self,
        population: List[Individual],
        selection_method: SelectionMethod,
        num_parents: int
    ) -> List[Individual]:
        """親選択"""
        
        if selection_method == SelectionMethod.TOURNAMENT:
            return self._tournament_selection(population, num_parents)
        elif selection_method == SelectionMethod.ROULETTE:
            return self._roulette_selection(population, num_parents)
        elif selection_method == SelectionMethod.RANK_BASED:
            return self._rank_based_selection(population, num_parents)
        elif selection_method == SelectionMethod.ELITIST:
            return self._elitist_selection(population, num_parents)
        else:
            return self._tournament_selection(population, num_parents)
    
    def _tournament_selection(self, population: List[Individual], num_parents: int) -> List[Individual]:
        """トーナメント選択"""
        parents = []
        tournament_size = max(2, len(population) // 4)
        
        for _ in range(num_parents):
            tournament = random.sample(population, tournament_size)
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner)
        
        return parents
    
    def _roulette_selection(self, population: List[Individual], num_parents: int) -> List[Individual]:
        """ルーレット選択"""
        # 適応度を正規化
        fitnesses = [ind.fitness for ind in population]
        min_fitness = min(fitnesses)
        
        # 負の適応度を調整
        if min_fitness < 0:
            fitnesses = [f - min_fitness + 1e-8 for f in fitnesses]
        
        total_fitness = sum(fitnesses)
        if total_fitness == 0:
            return random.sample(population, num_parents)
        
        probabilities = [f / total_fitness for f in fitnesses]
        
        parents = []
        for _ in range(num_parents):
            r = random.random()
            cumulative = 0
            for i, prob in enumerate(probabilities):
                cumulative += prob
                if r <= cumulative:
                    parents.append(population[i])
                    break
        
        return parents
    
    def _rank_based_selection(self, population: List[Individual], num_parents: int) -> List[Individual]:
        """ランクベース選択"""
        sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
        
        # ランクに基づく確率計算
        ranks = list(range(len(population), 0, -1))
        total_rank = sum(ranks)
        probabilities = [r / total_rank for r in ranks]
        
        parents = []
        for _ in range(num_parents):
            r = random.random()
            cumulative = 0
            for i, prob in enumerate(probabilities):
                cumulative += prob
                if r <= cumulative:
                    parents.append(sorted_population[i])
                    break
        
        return parents
    
    def _elitist_selection(self, population: List[Individual], num_parents: int) -> List[Individual]:
        """エリート選択"""
        sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
        return sorted_population[:num_parents]
    
    def crossover(
        self,
        parent1: Individual,
        parent2: Individual,
        crossover_method: CrossoverMethod,
        generation: int
    ) -> Tuple[Individual, Individual]:
        """交配"""
        
        if crossover_method == CrossoverMethod.UNIFORM:
            return self._uniform_crossover(parent1, parent2, generation)
        elif crossover_method == CrossoverMethod.SINGLE_POINT:
            return self._single_point_crossover(parent1, parent2, generation)
        elif crossover_method == CrossoverMethod.ARITHMETIC:
            return self._arithmetic_crossover(parent1, parent2, generation)
        else:
            return self._uniform_crossover(parent1, parent2, generation)
    
    def _uniform_crossover(
        self,
        parent1: Individual,
        parent2: Individual,
        generation: int
    ) -> Tuple[Individual, Individual]:
        """一様交配"""
        
        # 子の遺伝子作成
        child1_genes = {}
        child2_genes = {}
        
        all_genes = set(parent1.genes.keys()) | set(parent2.genes.keys())
        
        for gene in all_genes:
            if random.random() < 0.5:
                child1_genes[gene] = parent1.genes.get(gene, parent2.genes.get(gene))
                child2_genes[gene] = parent2.genes.get(gene, parent1.genes.get(gene))
            else:
                child1_genes[gene] = parent2.genes.get(gene, parent1.genes.get(gene))
                child2_genes[gene] = parent1.genes.get(gene, parent2.genes.get(gene))
        
        # 子個体作成
        child1 = self._create_child(child1_genes, [parent1.adapter_name, parent2.adapter_name], generation)
        child2 = self._create_child(child2_genes, [parent1.adapter_name, parent2.adapter_name], generation)
        
        return child1, child2
    
    def _single_point_crossover(
        self,
        parent1: Individual,
        parent2: Individual,
        generation: int
    ) -> Tuple[Individual, Individual]:
        """一点交配"""
        
        genes1 = list(parent1.genes.items())
        genes2 = list(parent2.genes.items())
        
        if len(genes1) <= 1:
            return self._uniform_crossover(parent1, parent2, generation)
        
        crossover_point = random.randint(1, len(genes1) - 1)
        
        child1_genes = dict(genes1[:crossover_point] + genes2[crossover_point:])
        child2_genes = dict(genes2[:crossover_point] + genes1[crossover_point:])
        
        child1 = self._create_child(child1_genes, [parent1.adapter_name, parent2.adapter_name], generation)
        child2 = self._create_child(child2_genes, [parent1.adapter_name, parent2.adapter_name], generation)
        
        return child1, child2
    
    def _arithmetic_crossover(
        self,
        parent1: Individual,
        parent2: Individual,
        generation: int
    ) -> Tuple[Individual, Individual]:
        """算術交配"""
        
        alpha = random.uniform(0.3, 0.7)
        
        child1_genes = {}
        child2_genes = {}
        
        all_genes = set(parent1.genes.keys()) | set(parent2.genes.keys())
        
        for gene in all_genes:
            val1 = parent1.genes.get(gene, 0)
            val2 = parent2.genes.get(gene, 0)
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                child1_genes[gene] = alpha * val1 + (1 - alpha) * val2
                child2_genes[gene] = (1 - alpha) * val1 + alpha * val2
                
                # 整数型の場合は丸める
                if isinstance(val1, int) and isinstance(val2, int):
                    child1_genes[gene] = int(round(child1_genes[gene]))
                    child2_genes[gene] = int(round(child2_genes[gene]))
            else:
                # 数値以外はランダム選択
                if random.random() < 0.5:
                    child1_genes[gene] = val1
                    child2_genes[gene] = val2
                else:
                    child1_genes[gene] = val2
                    child2_genes[gene] = val1
        
        child1 = self._create_child(child1_genes, [parent1.adapter_name, parent2.adapter_name], generation)
        child2 = self._create_child(child2_genes, [parent1.adapter_name, parent2.adapter_name], generation)
        
        return child1, child2
    
    def _create_child(
        self,
        genes: Dict[str, Any],
        parent_names: List[str],
        generation: int
    ) -> Individual:
        """子個体作成"""
        
        # 遺伝子を制約内に調整
        genes = self._constrain_genes(genes)
        
        # アダプタ設定作成
        base_config = AdapterConfig(
            name=f"evo_gen{generation}_child_{int(time.time() * 1000) % 10000}",
            adapter_type=AdapterType.LORA
        )
        
        adapter_config = self._genes_to_adapter_config(genes, base_config)
        
        # 個体作成
        individual = Individual(
            adapter_name=adapter_config.name,
            adapter_config=adapter_config,
            genes=genes,
            generation=generation,
            parent_names=parent_names
        )
        
        return individual
    
    def _constrain_genes(self, genes: Dict[str, Any]) -> Dict[str, Any]:
        """遺伝子制約適用"""
        
        constrained = genes.copy()
        
        # LoRAパラメータ制約
        if "r" in constrained:
            constrained["r"] = max(1, min(256, int(constrained["r"])))
        
        if "lora_alpha" in constrained:
            constrained["lora_alpha"] = max(1, min(128, int(constrained["lora_alpha"])))
        
        if "lora_dropout" in constrained:
            constrained["lora_dropout"] = max(0.0, min(0.5, float(constrained["lora_dropout"])))
        
        # 学習パラメータ制約
        if "learning_rate" in constrained:
            constrained["learning_rate"] = max(1e-6, min(1e-2, float(constrained["learning_rate"])))
        
        if "batch_size" in constrained:
            valid_sizes = [1, 2, 4, 8, 16]
            constrained["batch_size"] = min(valid_sizes, key=lambda x: abs(x - constrained["batch_size"]))
        
        return constrained
    
    def mutate(
        self,
        individual: Individual,
        mutation_method: MutationMethod,
        mutation_rate: float
    ) -> Individual:
        """変異"""
        
        if random.random() > mutation_rate:
            return individual
        
        if mutation_method == MutationMethod.GAUSSIAN:
            return self._gaussian_mutation(individual)
        elif mutation_method == MutationMethod.UNIFORM:
            return self._uniform_mutation(individual)
        elif mutation_method == MutationMethod.ADAPTIVE:
            return self._adaptive_mutation(individual)
        else:
            return self._gaussian_mutation(individual)
    
    def _gaussian_mutation(self, individual: Individual) -> Individual:
        """ガウス変異"""
        
        mutated_genes = individual.genes.copy()
        
        for gene, value in mutated_genes.items():
            if isinstance(value, (int, float)):
                if random.random() < 0.1:  # 10%の確率で変異
                    if isinstance(value, int):
                        mutation = int(random.gauss(0, max(1, abs(value) * 0.1)))
                        mutated_genes[gene] = value + mutation
                    else:
                        mutation = random.gauss(0, abs(value) * 0.1)
                        mutated_genes[gene] = value + mutation
        
        # 制約適用
        mutated_genes = self._constrain_genes(mutated_genes)
        
        # 新個体作成
        mutated_individual = Individual(
            adapter_name=f"mut_{individual.adapter_name}_{int(time.time() * 1000) % 1000}",
            adapter_config=self._genes_to_adapter_config(mutated_genes, individual.adapter_config),
            genes=mutated_genes,
            generation=individual.generation,
            parent_names=[individual.adapter_name]
        )
        
        return mutated_individual
    
    def _uniform_mutation(self, individual: Individual) -> Individual:
        """一様変異"""
        
        mutated_genes = individual.genes.copy()
        
        for gene in mutated_genes:
            if random.random() < 0.1:  # 10%の確率で変異
                # 新しいランダム値を生成
                if gene == "r":
                    mutated_genes[gene] = random.randint(8, 128)
                elif gene == "lora_alpha":
                    mutated_genes[gene] = random.randint(8, 64)
                elif gene == "lora_dropout":
                    mutated_genes[gene] = random.uniform(0.05, 0.3)
                elif gene == "learning_rate":
                    mutated_genes[gene] = random.uniform(1e-5, 1e-3)
                elif gene == "batch_size":
                    mutated_genes[gene] = random.choice([1, 2, 4, 8])
        
        # 新個体作成
        mutated_individual = Individual(
            adapter_name=f"mut_{individual.adapter_name}_{int(time.time() * 1000) % 1000}",
            adapter_config=self._genes_to_adapter_config(mutated_genes, individual.adapter_config),
            genes=mutated_genes,
            generation=individual.generation,
            parent_names=[individual.adapter_name]
        )
        
        return mutated_individual
    
    def _adaptive_mutation(self, individual: Individual) -> Individual:
        """適応的変異"""
        
        # 適応度に基づいて変異率を調整
        fitness_factor = max(0.1, 1.0 - individual.fitness)
        
        mutated_genes = individual.genes.copy()
        
        for gene, value in mutated_genes.items():
            if isinstance(value, (int, float)) and random.random() < fitness_factor * 0.2:
                if isinstance(value, int):
                    mutation_strength = max(1, int(abs(value) * fitness_factor * 0.2))
                    mutation = random.randint(-mutation_strength, mutation_strength)
                    mutated_genes[gene] = value + mutation
                else:
                    mutation_strength = abs(value) * fitness_factor * 0.2
                    mutation = random.uniform(-mutation_strength, mutation_strength)
                    mutated_genes[gene] = value + mutation
        
        # 制約適用
        mutated_genes = self._constrain_genes(mutated_genes)
        
        # 新個体作成
        mutated_individual = Individual(
            adapter_name=f"mut_{individual.adapter_name}_{int(time.time() * 1000) % 1000}",
            adapter_config=self._genes_to_adapter_config(mutated_genes, individual.adapter_config),
            genes=mutated_genes,
            generation=individual.generation,
            parent_names=[individual.adapter_name]
        )
        
        return mutated_individual
    
    async def create_autogen_agents(self, population: List[Individual]) -> None:
        """AutoGenエージェント作成"""
        
        if not AUTOGEN_AVAILABLE:
            return
        
        # LLM設定
        llm_config = {
            "config_list": [
                {
                    "model": "gpt-3.5-turbo",  # 実際の設定に応じて変更
                    "api_key": "dummy_key"  # 実際のAPIキーに変更
                }
            ],
            "temperature": 0.7
        }
        
        # エージェント作成
        self.agents = {}
        for individual in population[:5]:  # 最大5エージェント
            agent = EvolutionaryAgent(
                name=f"agent_{individual.adapter_name}",
                individual=individual,
                llm_config=llm_config
            )
            self.agents[individual.adapter_name] = agent
        
        # グループチャット設定
        if self.agents:
            agent_list = list(self.agents.values())
            
            # ユーザープロキシエージェント
            user_proxy = UserProxyAgent(
                name="evolution_coordinator",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=3,
                code_execution_config=False
            )
            
            agent_list.append(user_proxy)
            
            # グループチャット作成
            self.group_chat = GroupChat(
                agents=agent_list,
                messages=[],
                max_round=10
            )
            
            self.chat_manager = GroupChatManager(
                groupchat=self.group_chat,
                llm_config=llm_config
            )
    
    async def run_evolution(
        self,
        evolution_config: EvolutionConfig,
        base_adapter_config: AdapterConfig
    ) -> EvolutionResult:
        """進化実行"""
        
        start_time = time.time()
        
        try:
            # 初期個体群作成
            self.population = self.create_initial_population(evolution_config, base_adapter_config)
            self.current_generation = 0
            
            # 進化ループ
            stagnation_count = 0
            best_fitness_history = []
            
            for generation in range(evolution_config.max_generations):
                generation_start = time.time()
                
                self.current_generation = generation
                
                # 個体群評価
                await self.evaluate_population(self.population, evolution_config)
                
                # 世代統計計算
                fitnesses = [ind.fitness for ind in self.population]
                generation_stats = Generation(
                    generation_number=generation,
                    individuals=self.population.copy(),
                    best_fitness=max(fitnesses),
                    average_fitness=np.mean(fitnesses),
                    worst_fitness=min(fitnesses),
                    fitness_std=np.std(fitnesses),
                    evaluation_time=time.time() - generation_start
                )
                
                self.generation_history.append(generation_stats)
                
                # 収束判定
                best_fitness_history.append(generation_stats.best_fitness)
                
                if len(best_fitness_history) > evolution_config.stagnation_limit:
                    recent_improvement = (
                        best_fitness_history[-1] - 
                        best_fitness_history[-evolution_config.stagnation_limit]
                    )
                    
                    if recent_improvement < evolution_config.convergence_threshold:
                        stagnation_count += 1
                        if stagnation_count >= evolution_config.stagnation_limit:
                            break
                    else:
                        stagnation_count = 0
                
                # 最終世代でない場合は次世代作成
                if generation < evolution_config.max_generations - 1:
                    self.population = await self._create_next_generation(
                        self.population, evolution_config, generation + 1
                    )
                
                # AutoGenエージェント更新
                if generation % 5 == 0:  # 5世代ごと
                    await self.create_autogen_agents(self.population)
                
                self.logger.log_performance_metric(
                    metric_name="evolution_generation_completed",
                    value=generation_stats.best_fitness,
                    unit="fitness",
                    component="evolutionary_system"
                )
            
            # 結果作成
            best_individual = max(self.population, key=lambda x: x.fitness)
            
            result = EvolutionResult(
                config=evolution_config,
                generations=self.generation_history,
                best_individual=best_individual,
                final_population=self.population,
                total_generations=len(self.generation_history),
                total_evaluations=self.evolution_stats["total_evaluations"],
                convergence_achieved=stagnation_count >= evolution_config.stagnation_limit,
                total_time=time.time() - start_time,
                average_generation_time=sum(g.evaluation_time for g in self.generation_history) / len(self.generation_history),
                success=True
            )
            
            # 統計更新
            self.evolution_stats["total_runs"] += 1
            self.evolution_stats["successful_runs"] += 1
            
            if best_individual.fitness > self.evolution_stats["best_fitness_achieved"]:
                self.evolution_stats["best_fitness_achieved"] = best_individual.fitness
            
            return result
            
        except Exception as e:
            # エラー結果作成
            error_result = EvolutionResult(
                config=evolution_config,
                generations=self.generation_history,
                total_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
            
            # 統計更新
            self.evolution_stats["total_runs"] += 1
            self.evolution_stats["failed_runs"] += 1
            
            self.logger.log_alert(
                alert_type="evolution_failed",
                severity="ERROR",
                message=f"Evolution failed: {e}"
            )
            
            raise
    
    async def _create_next_generation(
        self,
        current_population: List[Individual],
        evolution_config: EvolutionConfig,
        generation: int
    ) -> List[Individual]:
        """次世代作成"""
        
        next_generation = []
        
        # エリート保存
        elite_individuals = sorted(current_population, key=lambda x: x.fitness, reverse=True)[:evolution_config.elite_size]
        next_generation.extend(elite_individuals)
        
        # 残りを交配・変異で生成
        while len(next_generation) < evolution_config.population_size:
            # 親選択
            parents = self.select_parents(current_population, evolution_config.selection_method, 2)
            
            if len(parents) >= 2:
                # 交配
                if random.random() < evolution_config.crossover_rate:
                    child1, child2 = self.crossover(parents[0], parents[1], evolution_config.crossover_method, generation)
                else:
                    child1, child2 = parents[0], parents[1]
                
                # 変異
                child1 = self.mutate(child1, evolution_config.mutation_method, evolution_config.mutation_rate)
                child2 = self.mutate(child2, evolution_config.mutation_method, evolution_config.mutation_rate)
                
                next_generation.extend([child1, child2])
        
        # 個体数調整
        return next_generation[:evolution_config.population_size]
    
    def get_evolution_stats(self) -> Dict[str, Any]:
        """進化統計取得"""
        stats = self.evolution_stats.copy()
        
        # 成功率計算
        if stats["total_runs"] > 0:
            stats["success_rate"] = stats["successful_runs"] / stats["total_runs"]
        else:
            stats["success_rate"] = 0.0
        
        # 現在の世代情報
        if self.generation_history:
            latest_generation = self.generation_history[-1]
            stats["current_generation"] = latest_generation.generation_number
            stats["current_best_fitness"] = latest_generation.best_fitness
            stats["current_average_fitness"] = latest_generation.average_fitness
        
        return stats
    
    async def shutdown(self) -> None:
        """進化システム終了"""
        
        final_stats = self.get_evolution_stats()
        
        self.logger.log_shutdown(
            component="evolutionary_system",
            uptime_seconds=0,  # TODO: 実際の稼働時間計算
            final_stats=final_stats
        )


# 便利関数
async def create_evolutionary_system(
    peft_pool: PEFTAdapterPool,
    evaluator: AdapterEvaluator,
    system_monitor: Optional[SystemMonitor] = None
) -> EvolutionarySystem:
    """進化システム作成・初期化"""
    
    system = EvolutionarySystem(peft_pool, evaluator, system_monitor)
    
    if await system.initialize():
        return system
    else:
        raise RuntimeError("Failed to initialize evolutionary system")


# 使用例
async def main():
    """テスト用メイン関数"""
    try:
        from ..adaptation.peft_manager import create_peft_adapter_pool
        from ..adaptation.adapter_evaluator import create_adapter_evaluator
        
        # 依存関係作成
        peft_pool = await create_peft_adapter_pool("microsoft/DialoGPT-small")
        evaluator = await create_adapter_evaluator(peft_pool)
        
        # 進化システム作成
        evolution_system = await create_evolutionary_system(peft_pool, evaluator)
        
        print("=== Evolutionary System Test ===")
        
        # 進化設定作成
        evolution_config = EvolutionConfig(
            population_size=10,
            max_generations=5,
            elite_size=2
        )
        
        # ベースアダプタ設定
        base_config = AdapterConfig(
            name="base_adapter",
            adapter_type=AdapterType.LORA,
            r=16,
            lora_alpha=32
        )
        
        print(f"Evolution Config: {evolution_config}")
        
        # 統計取得
        stats = evolution_system.get_evolution_stats()
        print(f"Evolution Stats: {stats}")
        
        await evolution_system.shutdown()
        await evaluator.shutdown()
        await peft_pool.shutdown()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())