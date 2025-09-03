"""
Chain-of-Thought エンジンのテスト
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.advanced_agent.reasoning.chain_of_thought import (
    ChainOfThoughtEngine, ChainOfThoughtSession, ThoughtStep, 
    ThoughtState, ChainOfThoughtCallbackHandler, create_chain_of_thought_engine
)
from src.advanced_agent.inference.ollama_client import OllamaClient
from src.advanced_agent.inference.tools import ToolManager
from src.advanced_agent.core.config import AdvancedAgentConfig


class TestChainOfThoughtEngine:
    """ChainOfThoughtEngine クラスのテスト"""
    
    @pytest.fixture
    def mock_ollama_client(self):
        """モックOllamaクライアント"""
        client = Mock(spec=OllamaClient)
        client.primary_model = "deepseek-r1:7b"
        
        # モックLLM
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value="Test reasoning response")
        client.primary_llm = mock_llm
        
        return client
    
    @pytest.fixture
    def mock_tool_manager(self):
        """モックツールマネージャー"""
        manager = Mock(spec=ToolManager)
        
        # モックツール
        mock_tool = Mock()
        mock_tool.name = "test_reasoning_tool"
        mock_tool.description = "Test reasoning tool"
        
        manager.get_tool.return_value = mock_tool
        return manager
    
    @pytest.fixture
    def cot_engine(self, mock_ollama_client, mock_tool_manager):
        """ChainOfThoughtEngine インスタンス"""
        with patch('src.advanced_agent.reasoning.chain_of_thought.get_config') as mock_get_config:
            mock_get_config.return_value = AdvancedAgentConfig()
            return ChainOfThoughtEngine(mock_ollama_client, mock_tool_manager)
    
    def test_init(self, cot_engine, mock_ollama_client, mock_tool_manager):
        """初期化テスト"""
        assert cot_engine.ollama_client == mock_ollama_client
        assert cot_engine.tool_manager == mock_tool_manager
        assert cot_engine.max_iterations == 8
        assert cot_engine.max_execution_time == 180
        assert cot_engine.memory is not None
        assert len(cot_engine.active_sessions) == 0
        assert len(cot_engine.session_history) == 0
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, cot_engine):
        """初期化成功テスト"""
        with patch('src.advanced_agent.reasoning.chain_of_thought.create_react_agent') as mock_create, \
             patch('src.advanced_agent.reasoning.chain_of_thought.AgentExecutor') as mock_executor:
            
            mock_create.return_value = Mock()
            mock_executor.return_value = Mock()
            
            result = await cot_engine.initialize()
            
            assert result is True
            assert cot_engine.agent is not None
            assert cot_engine.agent_executor is not None
    
    @pytest.mark.asyncio
    async def test_initialize_failure(self, cot_engine):
        """初期化失敗テスト"""
        cot_engine.ollama_client.primary_llm = None
        
        result = await cot_engine.initialize()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_create_cot_tools(self, cot_engine):
        """Chain-of-Thought専用ツール作成テスト"""
        cot_tools = await cot_engine._create_cot_tools()
        
        assert len(cot_tools) == 3
        
        tool_names = [tool.name for tool in cot_tools]
        assert "thought_organizer" in tool_names
        assert "logic_validator" in tool_names
        assert "confidence_evaluator" in tool_names
        
        # 各ツールの動作テスト
        organizer = next(tool for tool in cot_tools if tool.name == "thought_organizer")
        result = organizer._run("まず問題を分析します。次に解決策を検討します。最後に結論を出します。")
        assert "整理された思考" in result
        
        validator = next(tool for tool in cot_tools if tool.name == "logic_validator")
        result = validator._run("なぜなら根拠があるからです。したがって結論は正しいです。")
        assert "✅ 問題なし" in result
        
        evaluator = next(tool for tool in cot_tools if tool.name == "confidence_evaluator")
        result = evaluator._run("詳細な分析結果として、データに基づいて95%の確率で正しいと推測されます。")
        assert "信頼度評価" in result
    
    def test_create_cot_prompt(self, cot_engine):
        """Chain-of-Thought プロンプトテンプレート作成テスト"""
        prompt_template = cot_engine._create_cot_prompt()
        
        assert prompt_template is not None
        assert "段階的思考" in prompt_template.template
        assert "Thought:" in prompt_template.template
        assert "Action:" in prompt_template.template
        assert "Observation:" in prompt_template.template
        assert "Final Answer:" in prompt_template.template
        
        # 必要な入力変数が含まれているか確認
        required_vars = ["tools", "input", "agent_scratchpad", "chat_history"]
        for var in required_vars:
            assert var in prompt_template.input_variables
    
    def test_get_reasoning_chain(self, cot_engine):
        """推論チェーン取得テスト"""
        session = ChainOfThoughtSession(
            session_id="test_session",
            initial_query="Test query"
        )
        
        session.reasoning_chain = [
            "Step 1: Initial analysis",
            "Step 2: Detailed consideration", 
            "Step 3: Final conclusion"
        ]
        
        chain = cot_engine.get_reasoning_chain(session)
        
        assert len(chain) == 3
        assert chain[0] == "Step 1: Initial analysis"
        assert chain[1] == "Step 2: Detailed consideration"
        assert chain[2] == "Step 3: Final conclusion"
        
        # 元のチェーンが変更されないことを確認
        chain.append("Modified")
        assert len(session.reasoning_chain) == 3
    
    def test_get_thought_analysis(self, cot_engine):
        """思考分析取得テスト"""
        session = ChainOfThoughtSession(
            session_id="test_session",
            initial_query="Test query",
            total_processing_time=5.0,
            confidence_score=0.8
        )
        
        # テストステップ追加
        steps = [
            ThoughtStep(
                step_number=1,
                state=ThoughtState.ANALYZING,
                thought_content="First thought",
                reasoning="First reasoning",
                confidence_score=0.7,
                processing_time=1.0
            ),
            ThoughtStep(
                step_number=2,
                state=ThoughtState.ACTING,
                thought_content="Second thought",
                action_taken="test_action",
                observation="Test observation",
                reflection="Test reflection",
                confidence_score=0.9,
                processing_time=2.0
            )
        ]
        
        session.thought_steps.extend(steps)
        
        analysis = cot_engine.get_thought_analysis(session)
        
        assert analysis["total_steps"] == 2
        assert analysis["processing_time"] == 5.0
        assert analysis["confidence_score"] == 0.8
        assert "analyzing" in analysis["thought_states"]
        assert "acting" in analysis["thought_states"]
        assert len(analysis["step_details"]) == 2
        
        # ステップ詳細確認
        step1_detail = analysis["step_details"][0]
        assert step1_detail["step"] == 1
        assert step1_detail["state"] == "analyzing"
        assert step1_detail["has_reasoning"] is True
        assert step1_detail["has_action"] is False
        assert step1_detail["confidence"] == 0.7
        
        step2_detail = analysis["step_details"][1]
        assert step2_detail["step"] == 2
        assert step2_detail["state"] == "acting"
        assert step2_detail["has_action"] is True
        assert step2_detail["has_observation"] is True
        assert step2_detail["has_reflection"] is True
    
    def test_compare_sessions(self, cot_engine):
        """セッション比較テスト"""
        session1 = ChainOfThoughtSession(
            session_id="session1",
            initial_query="Query 1",
            total_processing_time=3.0,
            confidence_score=0.8
        )
        session1.thought_steps = [Mock(), Mock(), Mock()]  # 3ステップ
        session1.reasoning_chain = ["Step 1", "Step 2", "Step 3"]
        
        session2 = ChainOfThoughtSession(
            session_id="session2", 
            initial_query="Query 2",
            total_processing_time=5.0,
            confidence_score=0.6
        )
        session2.thought_steps = [Mock(), Mock()]  # 2ステップ
        session2.reasoning_chain = ["Step 1", "Step 2"]
        
        comparison = cot_engine.compare_sessions(session1, session2)
        
        assert comparison["session1_id"] == "session1"
        assert comparison["session2_id"] == "session2"
        assert comparison["step_count_diff"] == 1  # 3 - 2
        assert comparison["processing_time_diff"] == -2.0  # 3.0 - 5.0
        assert comparison["confidence_diff"] == 0.2  # 0.8 - 0.6
        assert comparison["reasoning_complexity"]["session1"] == 3
        assert comparison["reasoning_complexity"]["session2"] == 2
    
    @pytest.mark.asyncio
    async def test_get_session_summary_active(self, cot_engine):
        """アクティブセッションサマリー取得テスト"""
        session = ChainOfThoughtSession(
            session_id="active_session",
            initial_query="Active query",
            final_conclusion="Active conclusion",
            total_processing_time=2.5,
            confidence_score=0.75
        )
        session.thought_steps = [Mock(), Mock()]
        session.reasoning_chain = ["Step 1", "Step 2"]
        session.end_time = datetime.now()
        
        cot_engine.active_sessions["active_session"] = session
        
        summary = await cot_engine.get_session_summary("active_session")
        
        assert summary is not None
        assert summary["session_id"] == "active_session"
        assert summary["initial_query"] == "Active query"
        assert summary["final_conclusion"] == "Active conclusion"
        assert summary["total_steps"] == 2
        assert summary["processing_time"] == 2.5
        assert summary["confidence_score"] == 0.75
        assert summary["reasoning_chain_length"] == 2
    
    @pytest.mark.asyncio
    async def test_get_session_summary_history(self, cot_engine):
        """履歴セッションサマリー取得テスト"""
        session = ChainOfThoughtSession(
            session_id="history_session",
            initial_query="History query"
        )
        
        cot_engine.session_history.append(session)
        
        summary = await cot_engine.get_session_summary("history_session")
        
        assert summary is not None
        assert summary["session_id"] == "history_session"
    
    @pytest.mark.asyncio
    async def test_get_session_summary_not_found(self, cot_engine):
        """存在しないセッションサマリー取得テスト"""
        summary = await cot_engine.get_session_summary("nonexistent_session")
        
        assert summary is None


class TestChainOfThoughtCallbackHandler:
    """ChainOfThoughtCallbackHandler クラスのテスト"""
    
    @pytest.fixture
    def test_session(self):
        """テストセッション"""
        return ChainOfThoughtSession(
            session_id="test_callback_session",
            initial_query="Test callback query"
        )
    
    @pytest.fixture
    def callback_handler(self, test_session):
        """コールバックハンドラー"""
        return ChainOfThoughtCallbackHandler(test_session)
    
    def test_init(self, callback_handler, test_session):
        """初期化テスト"""
        assert callback_handler.session == test_session
        assert callback_handler.current_step == 0
        assert callback_handler.step_start_time is None
        assert callback_handler.thought_pattern is not None
        assert callback_handler.action_pattern is not None
        assert callback_handler.observation_pattern is not None
    
    def test_extract_thought(self, callback_handler):
        """思考内容抽出テスト"""
        log_text = """Thought: この問題を解決するために、まず状況を分析する必要があります。
Action: analyze_situation
Action Input: {"situation": "complex problem"}"""
        
        thought = callback_handler._extract_thought(log_text)
        assert "この問題を解決するために" in thought
        assert "まず状況を分析する必要があります" in thought
    
    def test_extract_reasoning(self, callback_handler):
        """推論内容抽出テスト"""
        log_text = """Thought: 問題を分析します
複数の要因が関係しています
根本原因を特定する必要があります
Action: root_cause_analysis"""
        
        reasoning = callback_handler._extract_reasoning(log_text)
        assert "複数の要因が関係しています" in reasoning
        assert "根本原因を特定する必要があります" in reasoning
    
    def test_generate_reflection(self, callback_handler):
        """反省内容生成テスト"""
        # 成功ケース
        success_step = ThoughtStep(
            step_number=1,
            state=ThoughtState.OBSERVING,
            thought_content="Test thought",
            observation="詳細な分析結果が得られました。データは正確で信頼性が高いです。"
        )
        
        reflection = callback_handler._generate_reflection(success_step)
        assert "詳細な情報を取得できました" in reflection
        
        # エラーケース
        error_step = ThoughtStep(
            step_number=2,
            state=ThoughtState.OBSERVING,
            thought_content="Test thought",
            observation="Error: Failed to process the request"
        )
        
        error_reflection = callback_handler._generate_reflection(error_step)
        assert "期待した結果を得られませんでした" in error_reflection
    
    def test_calculate_step_confidence(self, callback_handler):
        """ステップ信頼度計算テスト"""
        # 高品質ステップ
        high_quality_step = ThoughtStep(
            step_number=1,
            state=ThoughtState.COMPLETED,
            thought_content="これは詳細で論理的な思考内容です。複数の観点から分析しています。",
            reasoning="根拠に基づいた推論です",
            observation="成功的な結果が得られました。データは信頼性が高く、期待通りの成果です。"
        )
        
        confidence = callback_handler._calculate_step_confidence(high_quality_step)
        assert confidence > 0.7
        
        # 低品質ステップ
        low_quality_step = ThoughtStep(
            step_number=2,
            state=ThoughtState.ERROR,
            thought_content="短い思考",
            observation="error occurred"
        )
        
        low_confidence = callback_handler._calculate_step_confidence(low_quality_step)
        assert low_confidence < 0.7
    
    def test_calculate_overall_confidence(self, callback_handler):
        """全体信頼度計算テスト"""
        # 複数のステップを追加
        steps = [
            ThoughtStep(step_number=1, state=ThoughtState.COMPLETED, thought_content="Good", confidence_score=0.8),
            ThoughtStep(step_number=2, state=ThoughtState.COMPLETED, thought_content="Better", confidence_score=0.9),
            ThoughtStep(step_number=3, state=ThoughtState.COMPLETED, thought_content="Best", confidence_score=0.7)
        ]
        
        callback_handler.session.thought_steps.extend(steps)
        
        overall_confidence = callback_handler._calculate_overall_confidence()
        
        # 平均は0.8だが、ステップ数による調整があるため少し低くなる
        assert 0.7 <= overall_confidence <= 0.85
    
    def test_on_agent_action(self, callback_handler):
        """エージェントアクション開始時テスト"""
        # モックアクション
        mock_action = Mock()
        mock_action.log = "Thought: 問題を分析する必要があります\nAction: analyze"
        mock_action.tool = "analyze_tool"
        mock_action.tool_input = {"input": "test_data"}
        
        callback_handler.on_agent_action(mock_action)
        
        assert callback_handler.current_step == 1
        assert len(callback_handler.session.thought_steps) == 1
        assert len(callback_handler.session.reasoning_chain) == 1
        
        step = callback_handler.session.thought_steps[0]
        assert step.step_number == 1
        assert step.state == ThoughtState.ACTING
        assert step.action_taken == "analyze_tool"
        assert step.action_input == {"input": "test_data"}
    
    def test_on_tool_end(self, callback_handler):
        """ツール終了時テスト"""
        # 事前にステップ追加
        callback_handler.step_start_time = time.time() - 1  # 1秒前
        
        step = ThoughtStep(
            step_number=1,
            state=ThoughtState.ACTING,
            thought_content="Test thought"
        )
        callback_handler.session.thought_steps.append(step)
        
        callback_handler.on_tool_end("Tool execution completed successfully")
        
        assert callback_handler.session.thought_steps[0].observation == "Tool execution completed successfully"
        assert callback_handler.session.thought_steps[0].state == ThoughtState.OBSERVING
        assert callback_handler.session.thought_steps[0].reflection is not None
        assert callback_handler.session.thought_steps[0].confidence_score > 0
        assert callback_handler.session.thought_steps[0].processing_time > 0


class TestChainOfThoughtDataClasses:
    """Chain-of-Thought データクラスのテスト"""
    
    def test_thought_step(self):
        """ThoughtStep テスト"""
        step = ThoughtStep(
            step_number=1,
            state=ThoughtState.ANALYZING,
            thought_content="Test thought",
            reasoning="Test reasoning",
            action_plan="Test plan",
            confidence_score=0.8
        )
        
        assert step.step_number == 1
        assert step.state == ThoughtState.ANALYZING
        assert step.thought_content == "Test thought"
        assert step.reasoning == "Test reasoning"
        assert step.action_plan == "Test plan"
        assert step.confidence_score == 0.8
        assert isinstance(step.timestamp, datetime)
    
    def test_chain_of_thought_session(self):
        """ChainOfThoughtSession テスト"""
        session = ChainOfThoughtSession(
            session_id="test_session",
            initial_query="Test query",
            final_conclusion="Test conclusion",
            confidence_score=0.85
        )
        
        assert session.session_id == "test_session"
        assert session.initial_query == "Test query"
        assert session.final_conclusion == "Test conclusion"
        assert session.confidence_score == 0.85
        assert len(session.thought_steps) == 0
        assert len(session.reasoning_chain) == 0
        assert isinstance(session.start_time, datetime)


class TestCreateChainOfThoughtEngine:
    """create_chain_of_thought_engine 関数のテスト"""
    
    @pytest.mark.asyncio
    async def test_create_success(self):
        """作成成功テスト"""
        mock_client = Mock(spec=OllamaClient)
        mock_manager = Mock(spec=ToolManager)
        
        with patch('src.advanced_agent.reasoning.chain_of_thought.ChainOfThoughtEngine') as MockEngine:
            mock_engine = Mock()
            mock_engine.initialize = AsyncMock(return_value=True)
            MockEngine.return_value = mock_engine
            
            result = await create_chain_of_thought_engine(mock_client, mock_manager)
            
            assert result == mock_engine
            mock_engine.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_failure(self):
        """作成失敗テスト"""
        mock_client = Mock(spec=OllamaClient)
        mock_manager = Mock(spec=ToolManager)
        
        with patch('src.advanced_agent.reasoning.chain_of_thought.ChainOfThoughtEngine') as MockEngine:
            mock_engine = Mock()
            mock_engine.initialize = AsyncMock(return_value=False)
            MockEngine.return_value = mock_engine
            
            with pytest.raises(RuntimeError, match="Failed to initialize Chain-of-Thought engine"):
                await create_chain_of_thought_engine(mock_client, mock_manager)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])