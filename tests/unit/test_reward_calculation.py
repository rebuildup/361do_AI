"""
Tests for Reward Calculation System
報酬計算システムのテスト
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.advanced_agent.reward.reward_calculator import RewardCalculator, RewardMetrics
from src.advanced_agent.reward.engagement_analyzer import EngagementAnalyzer, EngagementMetrics
from src.advanced_agent.reward.rl_agent import RLAgent, RLState, RLAction


class TestRewardMetrics:
    """報酬メトリクスのテスト"""
    
    def test_reward_metrics_initialization(self):
        """報酬メトリクスの初期化テスト"""
        metrics = RewardMetrics()
        assert metrics.user_engagement == 0.0
        assert metrics.response_quality == 0.0
        assert metrics.total_reward == 0.0
        assert metrics.timestamp is not None
        assert metrics.interaction_id != ""
    
    def test_reward_metrics_to_dict(self):
        """報酬メトリクスの辞書変換テスト"""
        metrics = RewardMetrics(
            user_engagement=0.8,
            response_quality=0.9,
            total_reward=0.85
        )
        
        data = metrics.to_dict()
        assert data["user_engagement"] == 0.8
        assert data["response_quality"] == 0.9
        assert data["total_reward"] == 0.85
        assert "timestamp" in data
        assert "interaction_id" in data
    
    def test_reward_metrics_from_dict(self):
        """辞書からの報酬メトリクス作成テスト"""
        data = {
            "user_engagement": 0.7,
            "response_quality": 0.8,
            "total_reward": 0.75,
            "timestamp": datetime.now().isoformat(),
            "interaction_id": "test_id"
        }
        
        metrics = RewardMetrics.from_dict(data)
        assert metrics.user_engagement == 0.7
        assert metrics.response_quality == 0.8
        assert metrics.total_reward == 0.75
        assert metrics.interaction_id == "test_id"


class TestRewardCalculator:
    """報酬計算システムのテスト"""
    
    def test_reward_calculator_initialization(self):
        """報酬計算システムの初期化テスト"""
        calculator = RewardCalculator()
        assert calculator.weights is not None
        assert len(calculator.reward_history) == 0
        assert "user_engagement" in calculator.weights
        assert "response_quality" in calculator.weights
    
    @pytest.mark.asyncio
    async def test_calculate_reward_basic(self):
        """基本的な報酬計算テスト"""
        calculator = RewardCalculator()
        
        user_input = "こんにちは、元気ですか？"
        agent_response = "こんにちは！元気です、ありがとうございます。あなたはいかがですか？"
        context = {
            "response_time": 2.5,
            "interaction_duration": 10.0,
            "session_length": 5
        }
        
        metrics = await calculator.calculate_reward(user_input, agent_response, context)
        
        assert isinstance(metrics, RewardMetrics)
        assert 0.0 <= metrics.total_reward <= 1.0
        assert metrics.session_id == ""
        assert len(calculator.reward_history) == 1
    
    @pytest.mark.asyncio
    async def test_calculate_user_engagement(self):
        """ユーザー関与度計算テスト"""
        calculator = RewardCalculator()
        
        # 長い質問
        long_question = "これは非常に長い質問です。" * 20
        context = {"session_length": 10}
        
        engagement = await calculator._calculate_user_engagement(long_question, "", context)
        assert 0.0 <= engagement <= 1.0
        
        # 感情的な質問
        emotional_question = "すごい！ありがとうございます！"
        engagement = await calculator._calculate_user_engagement(emotional_question, "", context)
        assert engagement > 0.0
    
    @pytest.mark.asyncio
    async def test_calculate_response_quality_fallback(self):
        """回答品質計算のフォールバックテスト"""
        calculator = RewardCalculator()
        
        # 長い回答
        long_response = "これは詳細な回答です。" * 50
        quality = calculator._fallback_quality_assessment(long_response)
        assert 0.0 <= quality <= 1.0
        
        # 構造化された回答
        structured_response = "1. 最初のポイント\n2. 二番目のポイント\n3. 三番目のポイント"
        quality = calculator._fallback_quality_assessment(structured_response)
        assert quality > 0.0
    
    def test_identify_task_type(self):
        """タスクタイプ判定テスト"""
        calculator = RewardCalculator()
        
        # 質問
        question = "なぜ空は青いのですか？"
        task_type = calculator._identify_task_type(question)
        assert task_type == "question"
        
        # リクエスト
        request = "コードを書いてください"
        task_type = calculator._identify_task_type(request)
        assert task_type == "request"
        
        # 会話
        conversation = "今日はいい天気ですね"
        task_type = calculator._identify_task_type(conversation)
        assert task_type == "conversation"
    
    def test_calculate_total_reward(self):
        """総合報酬計算テスト"""
        calculator = RewardCalculator()
        
        metrics = RewardMetrics(
            user_engagement=0.8,
            response_quality=0.9,
            task_completion=0.7,
            creativity_score=0.6,
            helpfulness_score=0.8
        )
        
        total_reward = calculator._calculate_total_reward(metrics)
        assert 0.0 <= total_reward <= 1.0
    
    def test_get_reward_statistics(self):
        """報酬統計取得テスト"""
        calculator = RewardCalculator()
        
        # 空の履歴
        stats = calculator.get_reward_statistics()
        assert stats["total_interactions"] == 0
        
        # 履歴を追加
        calculator.reward_history.append(RewardMetrics(total_reward=0.8))
        calculator.reward_history.append(RewardMetrics(total_reward=0.9))
        
        stats = calculator.get_reward_statistics()
        assert stats["total_interactions"] == 2
        assert "average_reward" in stats
        assert "max_reward" in stats
        assert "min_reward" in stats


class TestEngagementMetrics:
    """関与度メトリクスのテスト"""
    
    def test_engagement_metrics_initialization(self):
        """関与度メトリクスの初期化テスト"""
        metrics = EngagementMetrics()
        assert metrics.interaction_frequency == 0.0
        assert metrics.overall_engagement == 0.0
        assert metrics.timestamp is not None
    
    def test_engagement_metrics_to_dict(self):
        """関与度メトリクスの辞書変換テスト"""
        metrics = EngagementMetrics(
            interaction_frequency=0.7,
            overall_engagement=0.8
        )
        
        data = metrics.to_dict()
        assert data["interaction_frequency"] == 0.7
        assert data["overall_engagement"] == 0.8
        assert "timestamp" in data


class TestEngagementAnalyzer:
    """関与度分析システムのテスト"""
    
    def test_engagement_analyzer_initialization(self):
        """関与度分析システムの初期化テスト"""
        analyzer = EngagementAnalyzer()
        assert analyzer.weights is not None
        assert len(analyzer.engagement_history) == 0
        assert "interaction_frequency" in analyzer.weights
    
    @pytest.mark.asyncio
    async def test_analyze_engagement_basic(self):
        """基本的な関与度分析テスト"""
        analyzer = EngagementAnalyzer()
        
        user_input = "こんにちは、元気ですか？"
        context = {
            "interaction_count": 5,
            "session_duration_minutes": 10,
            "response_time_seconds": 2.0
        }
        
        metrics = await analyzer.analyze_engagement(user_input, context)
        
        assert isinstance(metrics, EngagementMetrics)
        assert 0.0 <= metrics.overall_engagement <= 1.0
        assert len(analyzer.engagement_history) == 1
    
    @pytest.mark.asyncio
    async def test_calculate_message_length(self):
        """メッセージ長計算テスト"""
        analyzer = EngagementAnalyzer()
        
        # 長いメッセージ
        long_message = "これは非常に長いメッセージです。" * 20
        length_score = await analyzer._calculate_message_length(long_message)
        assert 0.0 <= length_score <= 1.0
        
        # 短いメッセージ
        short_message = "はい"
        length_score = await analyzer._calculate_message_length(short_message)
        assert 0.0 <= length_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_calculate_emotional_intensity(self):
        """感情的強度計算テスト"""
        analyzer = EngagementAnalyzer()
        
        # 感情的なメッセージ
        emotional_message = "すごい！ありがとうございます！"
        intensity = await analyzer._calculate_emotional_intensity(emotional_message)
        assert 0.0 <= intensity <= 1.0
        
        # 中性的なメッセージ
        neutral_message = "今日はいい天気ですね"
        intensity = await analyzer._calculate_emotional_intensity(neutral_message)
        assert 0.0 <= intensity <= 1.0
    
    def test_get_engagement_statistics(self):
        """関与度統計取得テスト"""
        analyzer = EngagementAnalyzer()
        
        # 空の履歴
        stats = analyzer.get_engagement_statistics()
        assert stats["total_sessions"] == 0
        
        # 履歴を追加
        analyzer.engagement_history.append(EngagementMetrics(overall_engagement=0.8))
        analyzer.engagement_history.append(EngagementMetrics(overall_engagement=0.9))
        
        stats = analyzer.get_engagement_statistics()
        assert stats["total_sessions"] == 2
        assert "average_engagement" in stats
        assert "max_engagement" in stats
        assert "min_engagement" in stats


class TestRLState:
    """強化学習状態のテスト"""
    
    def test_rl_state_initialization(self):
        """強化学習状態の初期化テスト"""
        state = RLState()
        assert state.user_engagement == 0.0
        assert state.session_context == {}
        assert state.conversation_history == []
        assert state.timestamp is not None
        assert state.state_id != ""
    
    def test_rl_state_to_dict(self):
        """強化学習状態の辞書変換テスト"""
        state = RLState(
            user_engagement=0.8,
            current_prompt="test prompt"
        )
        
        data = state.to_dict()
        assert data["user_engagement"] == 0.8
        assert data["current_prompt"] == "test prompt"
        assert "timestamp" in data
        assert "state_id" in data


class TestRLAction:
    """強化学習行動のテスト"""
    
    def test_rl_action_initialization(self):
        """強化学習行動の初期化テスト"""
        action = RLAction()
        assert action.action_type == ""
        assert action.parameters == {}
        assert action.timestamp is not None
        assert action.action_id != ""
    
    def test_rl_action_to_dict(self):
        """強化学習行動の辞書変換テスト"""
        action = RLAction(
            action_type="test_action",
            parameters={"param1": "value1"}
        )
        
        data = action.to_dict()
        assert data["action_type"] == "test_action"
        assert data["parameters"]["param1"] == "value1"
        assert "timestamp" in data
        assert "action_id" in data


class TestRLAgent:
    """強化学習エージェントのテスト"""
    
    def test_rl_agent_initialization(self):
        """強化学習エージェントの初期化テスト"""
        agent = RLAgent()
        assert agent.learning_rate == 0.01
        assert agent.discount_factor == 0.95
        assert agent.epsilon == 0.1
        assert len(agent.action_space) > 0
        assert len(agent.q_table) == 0
    
    @pytest.mark.asyncio
    async def test_select_action(self):
        """行動選択テスト"""
        agent = RLAgent()
        state = RLState(user_engagement=0.8)
        
        action = await agent.select_action(state)
        
        assert isinstance(action, RLAction)
        assert action.action_type in agent.action_space
        assert action.parameters is not None
    
    def test_get_state_key(self):
        """状態キー生成テスト"""
        agent = RLAgent()
        
        # 高関与度状態
        high_state = RLState(user_engagement=0.8, response_quality=0.9, learning_progress=0.8)
        key = agent._get_state_key(high_state)
        assert "high" in key
        
        # 低関与度状態
        low_state = RLState(user_engagement=0.2, response_quality=0.3, learning_progress=0.2)
        key = agent._get_state_key(low_state)
        assert "low" in key
    
    def test_get_action_parameters(self):
        """行動パラメータ取得テスト"""
        agent = RLAgent()
        state = RLState()
        
        for action_type in agent.action_space:
            parameters = agent._get_action_parameters(action_type, state)
            assert isinstance(parameters, dict)
    
    @pytest.mark.asyncio
    async def test_update_q_value(self):
        """Q値更新テスト"""
        agent = RLAgent()
        
        state = RLState(user_engagement=0.8)
        action = RLAction(action_type="use_simple_prompt")
        reward = 0.8
        next_state = RLState(user_engagement=0.9)
        
        await agent.update_q_value(state, action, reward, next_state)
        
        state_key = agent._get_state_key(state)
        assert state_key in agent.q_table
        assert action.action_type in agent.q_table[state_key]
        assert len(agent.experience_buffer) == 1
    
    def test_estimate_reward(self):
        """報酬推定テスト"""
        agent = RLAgent()
        
        # 高関与度状態
        high_state = RLState(user_engagement=0.8)
        action = RLAction(action_type="use_detailed_prompt")
        reward = agent._estimate_reward(high_state, action)
        assert 0.0 <= reward <= 1.0
        
        # 低関与度状態
        low_state = RLState(user_engagement=0.2)
        action = RLAction(action_type="ask_clarification")
        reward = agent._estimate_reward(low_state, action)
        assert 0.0 <= reward <= 1.0
    
    def test_get_learning_statistics(self):
        """学習統計取得テスト"""
        agent = RLAgent()
        
        stats = agent.get_learning_statistics()
        assert "total_learning_episodes" in stats
        assert "current_epsilon" in stats
        assert "q_table_size" in stats
    
    def test_reset_learning(self):
        """学習リセットテスト"""
        agent = RLAgent()
        
        # データを追加
        agent.q_table["test_state"] = {"test_action": 0.5}
        agent.experience_buffer.append((RLState(), RLAction(), 0.8, RLState()))
        agent.learning_history.append({"test": "data"})
        
        # リセット
        agent.reset_learning()
        
        assert len(agent.q_table) == 0
        assert len(agent.experience_buffer) == 0
        assert len(agent.learning_history) == 0
        assert agent.epsilon == 0.1


if __name__ == "__main__":
    pytest.main([__file__])

