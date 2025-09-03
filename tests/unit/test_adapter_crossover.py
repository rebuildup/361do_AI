"""
Adapter Crossover System のテスト
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
import torch

from src.advanced_agent.evolution.adapter_crossover import (
    AdapterCrossoverSystem, CrossoverConfig, CrossoverResult,
    CrossoverStrategy, MergeMethod, CrossoverNegotiatorAgent,
    create_adapter_crossover_system
)
from src.advanced_agent.adaptation.peft_manager import PEFTAdapterPool, AdapterConfig, AdapterType
from src.advanced_agent.adaptation.adapter_evaluator import AdapterEvaluator, EvaluationResult
from src.advanced_agent.evolution.evolutionary_system import Individual
from src.advanced_agent.monitoring.system_monitor import SystemMonitor


class TestAdapterCrossoverSystem:
    """AdapterCrossoverSystem クラスのテスト"""
    
    @pytest.fixture
    def mock_peft_pool(self):
        """モックPEFTプール"""
        pool = Mock(spec=PEFTAdapterPool)
        pool.adapters = {}
        pool.active_adapters = {}
        pool.load_adapter = AsyncMock()
        
        # モックPEFTモデル
        mock_model = Mock()
        mock_model.base_model = Mock()
        mock_model.peft_config = Mock()
        mock_model.named_parameters = Mock(return_value=[
            ("lora_A", Mock(data=torch.randn(10, 10))),
            ("lora_B", Mock(data=torch.randn(10, 10)))
        ])
        mock_model.parameters = Mock(return_value=[
            Mock(numel=Mock(return_value=100), requires_grad=True)
        ])
        
        pool.active_adapters = {
            "parent1": mock_model,
            "parent2": mock_model
        }
        
        return pool
    
    @pytest.fixture
    def mock_evaluator(self):
        """モック評価器"""
        evaluator = Mock(spec=AdapterEvaluator)
        
        mock_result = EvaluationResult(
            adapter_name="test_adapter",
            task=Mock(),
            config=Mock(),
            metrics={"accuracy": 0.85},
            success=True
        )
        evaluator.evaluate_adapter = AsyncMock(return_value=mock_result)
        
        return evaluator
    
    @pytest.fixture
    def mock_system_monitor(self):
        """モックシステムモニター"""
        return Mock(spec=SystemMonitor)
    
    @pytest.fixture
    def crossover_system(self, mock_peft_pool, mock_evaluator, mock_system_monitor):
        """AdapterCrossoverSystem インスタンス"""
        with patch('src.advanced_agent.evolution.adapter_crossover.get_config') as mock_get_config, \
             patch('src.advanced_agent.evolution.adapter_crossover.CROSSOVER_AVAILABLE', True):
            from src.advanced_agent.core.config import AdvancedAgentConfig
            mock_get_config.return_value = AdvancedAgentConfig()
            
            return AdapterCrossoverSystem(mock_peft_pool, mock_evaluator, mock_system_monitor)
    
    def test_init(self, crossover_system, mock_peft_pool, mock_evaluator, mock_system_monitor):
        """初期化テスト"""
        assert crossover_system.peft_pool == mock_peft_pool
        assert crossover_system.evaluator == mock_evaluator
        assert crossover_system.system_monitor == mock_system_monitor
        assert len(crossover_system.crossover_history) == 0
        assert len(crossover_system.negotiator_agents) == 0
        assert crossover_system.crossover_stats["total_crossovers"] == 0
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, crossover_system):
        """初期化成功テスト"""
        result = await crossover_system.initialize()
        
        assert result is True
    
    @patch('src.advanced_agent.evolution.adapter_crossover.CROSSOVER_AVAILABLE', False)
    @pytest.mark.asyncio
    async def test_initialize_unavailable(self, crossover_system):
        """Crossover利用不可時の初期化テスト"""
        result = await crossover_system.initialize()
        
        assert result is False
    
    def test_determine_crossover_weights_performance_based(self, crossover_system):
        """性能ベース交配重み決定テスト"""
        parent1 = Individual(
            adapter_name="parent1",
            adapter_config=AdapterConfig(name="parent1", adapter_type=AdapterType.LORA),
            fitness=0.8
        )
        
        parent2 = Individual(
            adapter_name="parent2",
            adapter_config=AdapterConfig(name="parent2", adapter_type=AdapterType.LORA),
            fitness=0.6
        )
        
        crossover_config = CrossoverConfig(performance_based_weighting=True)
        
        weights = crossover_system._determine_crossover_weights(
            parent1, parent2, crossover_config, None
        )
        
        # 適応度に比例した重み
        expected_parent1_weight = 0.8 / (0.8 + 0.6)  # 0.571...
        expected_parent2_weight = 0.6 / (0.8 + 0.6)  # 0.428...
        
        assert abs(weights["parent1_weight"] - expected_parent1_weight) < 0.01
        assert abs(weights["parent2_weight"] - expected_parent2_weight) < 0.01
    
    def test_determine_crossover_weights_config_based(self, crossover_system):
        """設定ベース交配重み決定テスト"""
        parent1 = Individual(
            adapter_name="parent1",
            adapter_config=AdapterConfig(name="parent1", adapter_type=AdapterType.LORA),
            fitness=0.8
        )
        
        parent2 = Individual(
            adapter_name="parent2",
            adapter_config=AdapterConfig(name="parent2", adapter_type=AdapterType.LORA),
            fitness=0.6
        )
        
        crossover_config = CrossoverConfig(
            performance_based_weighting=False,
            parent1_weight=0.7,
            parent2_weight=0.3
        )
        
        weights = crossover_system._determine_crossover_weights(
            parent1, parent2, crossover_config, None
        )
        
        # 正規化された設定値
        assert abs(weights["parent1_weight"] - 0.7) < 0.01
        assert abs(weights["parent2_weight"] - 0.3) < 0.01
    
    def test_determine_crossover_weights_negotiation_priority(self, crossover_system):
        """交渉結果優先テスト"""
        parent1 = Individual(
            adapter_name="parent1",
            adapter_config=AdapterConfig(name="parent1", adapter_type=AdapterType.LORA),
            fitness=0.8
        )
        
        parent2 = Individual(
            adapter_name="parent2",
            adapter_config=AdapterConfig(name="parent2", adapter_type=AdapterType.LORA),
            fitness=0.6
        )
        
        crossover_config = CrossoverConfig(performance_based_weighting=True)
        
        negotiation_result = {
            "consensus": True,
            "final_weights": {
                "parent1_weight": 0.9,
                "parent2_weight": 0.1
            }
        }
        
        weights = crossover_system._determine_crossover_weights(
            parent1, parent2, crossover_config, negotiation_result
        )
        
        # 交渉結果が優先される
        assert weights["parent1_weight"] == 0.9
        assert weights["parent2_weight"] == 0.1
    
    def test_count_parameters(self, crossover_system):
        """パラメータ数カウントテスト"""
        mock_model = Mock()
        mock_param1 = Mock(numel=Mock(return_value=100), requires_grad=True)
        mock_param2 = Mock(numel=Mock(return_value=200), requires_grad=True)
        mock_param3 = Mock(numel=Mock(return_value=50), requires_grad=False)  # 学習対象外
        
        mock_model.parameters.return_value = [mock_param1, mock_param2, mock_param3]
        
        count = crossover_system._count_parameters(mock_model)
        
        assert count == 300  # 100 + 200 (学習対象外は除外)
    
    @pytest.mark.asyncio
    async def test_evaluate_offspring(self, crossover_system):
        """子個体評価テスト"""
        offspring_name = "test_offspring"
        
        fitness = await crossover_system._evaluate_offspring(offspring_name)
        
        assert fitness == 0.85  # モック評価器の返り値
        crossover_system.evaluator.evaluate_adapter.assert_called_once()
    
    def test_update_crossover_stats(self, crossover_system):
        """交配統計更新テスト"""
        # 初期状態
        assert crossover_system.crossover_stats["successful_crossovers"] == 0
        assert crossover_system.crossover_stats["average_improvement"] == 0.0
        
        # 最初の結果
        result1 = CrossoverResult(
            parent1_name="parent1",
            parent2_name="parent2",
            offspring_name="child1",
            strategy_used=CrossoverStrategy.WEIGHT_AVERAGING,
            merge_method_used=MergeMethod.WEIGHTED_AVERAGE,
            final_weights={},
            parent1_fitness=0.7,
            parent2_fitness=0.6,
            offspring_fitness=0.8,  # 改善: 0.8 - 0.65 = 0.15
            success=True
        )
        
        crossover_system.crossover_stats["successful_crossovers"] = 1
        crossover_system._update_crossover_stats(result1)
        
        assert crossover_system.crossover_stats["average_improvement"] == 0.15
        assert crossover_system.crossover_stats["best_offspring_fitness"] == 0.8
        
        # 2番目の結果
        result2 = CrossoverResult(
            parent1_name="parent3",
            parent2_name="parent4",
            offspring_name="child2",
            strategy_used=CrossoverStrategy.WEIGHT_AVERAGING,
            merge_method_used=MergeMethod.WEIGHTED_AVERAGE,
            final_weights={},
            parent1_fitness=0.5,
            parent2_fitness=0.4,
            offspring_fitness=0.6,  # 改善: 0.6 - 0.45 = 0.15
            success=True
        )
        
        crossover_system.crossover_stats["successful_crossovers"] = 2
        crossover_system._update_crossover_stats(result2)
        
        # 平均改善度確認
        assert crossover_system.crossover_stats["average_improvement"] == 0.15  # (0.15 + 0.15) / 2
        assert crossover_system.crossover_stats["best_offspring_fitness"] == 0.8  # 最高値維持
    
    def test_get_crossover_stats_empty(self, crossover_system):
        """空の交配統計取得テスト"""
        stats = crossover_system.get_crossover_stats()
        
        assert stats["total_crossovers"] == 0
        assert stats["successful_crossovers"] == 0
        assert stats["failed_crossovers"] == 0
        assert stats["success_rate"] == 0.0
    
    def test_get_crossover_stats_with_history(self, crossover_system):
        """履歴ありの交配統計取得テスト"""
        # テスト履歴追加
        results = [
            CrossoverResult(
                parent1_name="p1", parent2_name="p2", offspring_name="c1",
                strategy_used=CrossoverStrategy.WEIGHT_AVERAGING,
                merge_method_used=MergeMethod.WEIGHTED_AVERAGE,
                final_weights={}, crossover_time=5.0,
                parent1_fitness=0.7, parent2_fitness=0.6, offspring_fitness=0.8,
                success=True
            ),
            CrossoverResult(
                parent1_name="p3", parent2_name="p4", offspring_name="c2",
                strategy_used=CrossoverStrategy.WEIGHT_AVERAGING,
                merge_method_used=MergeMethod.WEIGHTED_AVERAGE,
                final_weights={}, crossover_time=7.0,
                parent1_fitness=0.5, parent2_fitness=0.4, offspring_fitness=0.6,
                success=True
            ),
            CrossoverResult(
                parent1_name="p5", parent2_name="p6", offspring_name="",
                strategy_used=CrossoverStrategy.WEIGHT_AVERAGING,
                merge_method_used=MergeMethod.WEIGHTED_AVERAGE,
                final_weights={}, success=False
            )
        ]
        
        crossover_system.crossover_history.extend(results)
        crossover_system.crossover_stats.update({
            "total_crossovers": 3,
            "successful_crossovers": 2,
            "failed_crossovers": 1,
            "average_improvement": 0.15
        })
        
        stats = crossover_system.get_crossover_stats()
        
        assert stats["total_crossovers"] == 3
        assert stats["successful_crossovers"] == 2
        assert stats["failed_crossovers"] == 1
        assert stats["success_rate"] == 2/3
        assert "recent_average_time" in stats
        assert "recent_average_improvement" in stats
    
    def test_get_crossover_history_no_filter(self, crossover_system):
        """フィルタなし交配履歴取得テスト"""
        # テスト履歴追加
        results = [
            CrossoverResult(
                parent1_name="p1", parent2_name="p2", offspring_name="c1",
                strategy_used=CrossoverStrategy.WEIGHT_AVERAGING,
                merge_method_used=MergeMethod.WEIGHTED_AVERAGE,
                final_weights={}, success=True
            ),
            CrossoverResult(
                parent1_name="p3", parent2_name="p4", offspring_name="",
                strategy_used=CrossoverStrategy.WEIGHT_AVERAGING,
                merge_method_used=MergeMethod.WEIGHTED_AVERAGE,
                final_weights={}, success=False
            )
        ]
        crossover_system.crossover_history.extend(results)
        
        history = crossover_system.get_crossover_history()
        
        assert len(history) == 2
    
    def test_get_crossover_history_success_filter(self, crossover_system):
        """成功フィルタ交配履歴取得テスト"""
        # テスト履歴追加
        results = [
            CrossoverResult(
                parent1_name="p1", parent2_name="p2", offspring_name="c1",
                strategy_used=CrossoverStrategy.WEIGHT_AVERAGING,
                merge_method_used=MergeMethod.WEIGHTED_AVERAGE,
                final_weights={}, success=True
            ),
            CrossoverResult(
                parent1_name="p3", parent2_name="p4", offspring_name="",
                strategy_used=CrossoverStrategy.WEIGHT_AVERAGING,
                merge_method_used=MergeMethod.WEIGHTED_AVERAGE,
                final_weights={}, success=False
            )
        ]
        crossover_system.crossover_history.extend(results)
        
        history = crossover_system.get_crossover_history(success_only=True)
        
        assert len(history) == 1
        assert history[0].success is True
    
    def test_get_crossover_history_limit(self, crossover_system):
        """件数制限交配履歴取得テスト"""
        # テスト履歴追加
        results = [
            CrossoverResult(
                parent1_name=f"p{i}", parent2_name=f"p{i+1}", offspring_name=f"c{i}",
                strategy_used=CrossoverStrategy.WEIGHT_AVERAGING,
                merge_method_used=MergeMethod.WEIGHTED_AVERAGE,
                final_weights={}, success=True
            )
            for i in range(5)
        ]
        crossover_system.crossover_history.extend(results)
        
        history = crossover_system.get_crossover_history(limit=3)
        
        assert len(history) == 3


class TestCrossoverNegotiatorAgent:
    """CrossoverNegotiatorAgent クラスのテスト"""
    
    @pytest.fixture
    def test_individual(self):
        """テスト個体"""
        return Individual(
            adapter_name="test_agent",
            adapter_config=AdapterConfig(name="test_agent", adapter_type=AdapterType.LORA),
            fitness=0.75,
            genes={"r": 32, "lora_alpha": 16}
        )
    
    @pytest.fixture
    def negotiator_agent(self, test_individual):
        """CrossoverNegotiatorAgent インスタンス"""
        with patch('src.advanced_agent.evolution.adapter_crossover.CROSSOVER_AVAILABLE', True):
            llm_config = {"config_list": [{"model": "test", "api_key": "test"}]}
            return CrossoverNegotiatorAgent("test_negotiator", test_individual, llm_config)
    
    def test_propose_crossover_weights(self, negotiator_agent, test_individual):
        """交配重み提案テスト"""
        partner = Individual(
            adapter_name="partner",
            adapter_config=AdapterConfig(name="partner", adapter_type=AdapterType.LORA),
            fitness=0.25
        )
        
        proposal = negotiator_agent.propose_crossover_weights(partner)
        
        assert "self_weight" in proposal
        assert "partner_weight" in proposal
        assert "confidence" in proposal
        
        # 適応度に基づく重み
        total_fitness = 0.75 + 0.25
        expected_self_weight = 0.75 / total_fitness
        expected_partner_weight = 0.25 / total_fitness
        
        assert abs(proposal["self_weight"] - expected_self_weight) < 0.01
        assert abs(proposal["partner_weight"] - expected_partner_weight) < 0.01
        assert proposal["confidence"] == min(0.75, 0.25)  # 0.25
    
    def test_evaluate_crossover_proposal_accept(self, negotiator_agent):
        """交配提案評価（受諾）テスト"""
        # 高い重みの提案
        proposal = {"self_weight": 0.8, "partner_weight": 0.2}
        
        evaluation = negotiator_agent.evaluate_crossover_proposal(proposal)
        
        assert evaluation["acceptance"] is True
        assert evaluation["confidence"] > 0.0
        assert len(evaluation["suggestions"]) == 0
    
    def test_evaluate_crossover_proposal_reject(self, negotiator_agent):
        """交配提案評価（拒否）テスト"""
        # 低い重みの提案
        proposal = {"self_weight": 0.2, "partner_weight": 0.8}
        
        evaluation = negotiator_agent.evaluate_crossover_proposal(proposal)
        
        assert evaluation["acceptance"] is False
        assert len(evaluation["suggestions"]) > 0


class TestCrossoverDataClasses:
    """交配データクラスのテスト"""
    
    def test_crossover_config(self):
        """CrossoverConfig テスト"""
        config = CrossoverConfig(
            strategy=CrossoverStrategy.SELECTIVE_MERGE,
            merge_method=MergeMethod.TASK_VECTOR,
            parent1_weight=0.6,
            parent2_weight=0.4,
            adaptive_weights=True,
            use_autogen_coordination=False
        )
        
        assert config.strategy == CrossoverStrategy.SELECTIVE_MERGE
        assert config.merge_method == MergeMethod.TASK_VECTOR
        assert config.parent1_weight == 0.6
        assert config.parent2_weight == 0.4
        assert config.adaptive_weights is True
        assert config.use_autogen_coordination is False
    
    def test_crossover_result(self):
        """CrossoverResult テスト"""
        result = CrossoverResult(
            parent1_name="parent1",
            parent2_name="parent2",
            offspring_name="child1",
            strategy_used=CrossoverStrategy.WEIGHT_AVERAGING,
            merge_method_used=MergeMethod.WEIGHTED_AVERAGE,
            final_weights={"parent1_weight": 0.7, "parent2_weight": 0.3},
            parent1_fitness=0.8,
            parent2_fitness=0.6,
            offspring_fitness=0.85,
            crossover_time=15.0,
            parameter_count=1000000,
            consensus_reached=True,
            success=True
        )
        
        assert result.parent1_name == "parent1"
        assert result.parent2_name == "parent2"
        assert result.offspring_name == "child1"
        assert result.strategy_used == CrossoverStrategy.WEIGHT_AVERAGING
        assert result.merge_method_used == MergeMethod.WEIGHTED_AVERAGE
        assert result.final_weights == {"parent1_weight": 0.7, "parent2_weight": 0.3}
        assert result.parent1_fitness == 0.8
        assert result.parent2_fitness == 0.6
        assert result.offspring_fitness == 0.85
        assert result.crossover_time == 15.0
        assert result.parameter_count == 1000000
        assert result.consensus_reached is True
        assert result.success is True
        assert isinstance(result.timestamp, datetime)


class TestCrossoverEnums:
    """交配列挙型のテスト"""
    
    def test_crossover_strategy_enum(self):
        """CrossoverStrategy 列挙型テスト"""
        assert CrossoverStrategy.WEIGHT_AVERAGING.value == "weight_averaging"
        assert CrossoverStrategy.SELECTIVE_MERGE.value == "selective_merge"
        assert CrossoverStrategy.LAYER_WISE_MERGE.value == "layer_wise_merge"
        assert CrossoverStrategy.ATTENTION_MERGE.value == "attention_merge"
    
    def test_merge_method_enum(self):
        """MergeMethod 列挙型テスト"""
        assert MergeMethod.LINEAR_INTERPOLATION.value == "linear_interpolation"
        assert MergeMethod.WEIGHTED_AVERAGE.value == "weighted_average"
        assert MergeMethod.TASK_VECTOR.value == "task_vector"
        assert MergeMethod.TIES_MERGE.value == "ties_merge"


class TestCreateAdapterCrossoverSystem:
    """create_adapter_crossover_system 関数のテスト"""
    
    @pytest.mark.asyncio
    async def test_create_success(self):
        """作成成功テスト"""
        mock_pool = Mock(spec=PEFTAdapterPool)
        mock_evaluator = Mock(spec=AdapterEvaluator)
        
        with patch('src.advanced_agent.evolution.adapter_crossover.AdapterCrossoverSystem') as MockSystem:
            mock_system = Mock()
            mock_system.initialize = AsyncMock(return_value=True)
            MockSystem.return_value = mock_system
            
            result = await create_adapter_crossover_system(mock_pool, mock_evaluator)
            
            assert result == mock_system
            MockSystem.assert_called_once_with(mock_pool, mock_evaluator, None)
            mock_system.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_failure(self):
        """作成失敗テスト"""
        mock_pool = Mock(spec=PEFTAdapterPool)
        mock_evaluator = Mock(spec=AdapterEvaluator)
        
        with patch('src.advanced_agent.evolution.adapter_crossover.AdapterCrossoverSystem') as MockSystem:
            mock_system = Mock()
            mock_system.initialize = AsyncMock(return_value=False)
            MockSystem.return_value = mock_system
            
            with pytest.raises(RuntimeError, match="Failed to initialize adapter crossover system"):
                await create_adapter_crossover_system(mock_pool, mock_evaluator)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])