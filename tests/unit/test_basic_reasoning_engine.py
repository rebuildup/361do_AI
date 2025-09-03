"""
基本推論エンジンのテスト
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.advanced_agent.reasoning.basic_engine import (
    BasicReasoningEngine, ReasoningRequest, ReasoningResponse, 
    ReasoningState, ReasoningCallbackHandler, create_basic_reasoning_engine
)
from src.advanced_agent.inference.ollama_client import OllamaClient
from src.advanced_agent.core.config import AdvancedAgentConfig


class TestBasicReasoningEngine:
    """BasicReasoningEngine クラスのテスト"""
    
    @pytest.fixture
    def mock_config(self):
        """モック設定"""
        config = AdvancedAgentConfig()
        config.models.primary = "deepseek-r1:7b"
        return config
    
    @pytest.fixture
    def mock_ollama_client(self):
        """モックOllamaクライアント"""
        client = Mock(spec=OllamaClient)
        client.primary_model = "deepseek-r1:7b"
        client.fallback_models = ["llama2:7b"]
        
        # モックLLM
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value="Test response")
        client.primary_llm = mock_llm
        
        return client
    
    @pytest.fixture
    def reasoning_engine(self, mock_ollama_client):
        """BasicReasoningEngine インスタンス"""
        with patch('src.advanced_agent.reasoning.basic_engine.get_config') as mock_get_config:
            mock_get_config.return_value = AdvancedAgentConfig()
            return BasicReasoningEngine(mock_ollama_client)
    
    def test_init(self, reasoning_engine, mock_ollama_client):
        """初期化テスト"""
        assert reasoning_engine.ollama_client == mock_ollama_client
        assert len(reasoning_engine.prompt_templates) == 0
        assert len(reasoning_engine.chat_templates) == 0
        assert len(reasoning_engine.reasoning_history) == 0
        assert reasoning_engine.performance_stats["total_requests"] == 0
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, reasoning_engine):
        """初期化成功テスト"""
        result = await reasoning_engine.initialize()
        
        assert result is True
        assert len(reasoning_engine.prompt_templates) > 0
        assert len(reasoning_engine.chat_templates) > 0
        
        # デフォルトテンプレートの確認
        expected_templates = ["basic_qa", "analysis", "code_analysis", "problem_solving", "summarization"]
        for template_name in expected_templates:
            assert template_name in reasoning_engine.prompt_templates
    
    @pytest.mark.asyncio
    async def test_initialize_failure(self, reasoning_engine):
        """初期化失敗テスト"""
        reasoning_engine.ollama_client.primary_llm = None
        
        result = await reasoning_engine.initialize()
        
        assert result is False
    
    def test_register_template(self, reasoning_engine):
        """テンプレート登録テスト"""
        from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
        
        # プロンプトテンプレート登録
        prompt_template = PromptTemplate(
            input_variables=["test_var"],
            template="Test template: {test_var}"
        )
        
        reasoning_engine.register_template("test_prompt", prompt_template)
        assert "test_prompt" in reasoning_engine.prompt_templates
        
        # チャットテンプレート登録
        chat_template = ChatPromptTemplate.from_messages([
            ("system", "Test system message"),
            ("human", "{input}")
        ])
        
        reasoning_engine.register_template("test_chat", chat_template)
        assert "test_chat" in reasoning_engine.chat_templates
    
    def test_get_template(self, reasoning_engine):
        """テンプレート取得テスト"""
        from langchain_core.prompts import PromptTemplate
        
        # テンプレート登録
        template = PromptTemplate(
            input_variables=["test"],
            template="Test: {test}"
        )
        reasoning_engine.register_template("test_template", template)
        
        # 取得テスト
        retrieved = reasoning_engine.get_template("test_template")
        assert retrieved == template
        
        # 存在しないテンプレート
        not_found = reasoning_engine.get_template("nonexistent")
        assert not_found is None
    
    def test_list_templates(self, reasoning_engine):
        """テンプレート一覧テスト"""
        from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
        
        # テンプレート追加
        prompt_template = PromptTemplate(input_variables=["test"], template="Test: {test}")
        chat_template = ChatPromptTemplate.from_messages([("human", "{input}")])
        
        reasoning_engine.register_template("test_prompt", prompt_template)
        reasoning_engine.register_template("test_chat", chat_template)
        
        templates = reasoning_engine.list_templates()
        
        assert "prompt_templates" in templates
        assert "chat_templates" in templates
        assert "test_prompt" in templates["prompt_templates"]
        assert "test_chat" in templates["chat_templates"]
    
    @pytest.mark.asyncio
    async def test_reason_basic(self, reasoning_engine):
        """基本推論テスト"""
        # 初期化
        await reasoning_engine.initialize()
        
        # モックLLMの設定
        reasoning_engine.ollama_client.primary_llm.invoke = Mock(return_value="Test response")
        
        # 推論実行
        response = await reasoning_engine.reason("Test prompt")
        
        assert response.state == ReasoningState.COMPLETED
        assert response.response_text == "Test response"
        assert response.processing_time > 0
        assert response.model_used == "deepseek-r1:7b"
        
        # 統計確認
        assert reasoning_engine.performance_stats["total_requests"] == 1
        assert reasoning_engine.performance_stats["successful_requests"] == 1
        assert len(reasoning_engine.reasoning_history) == 1
    
    @pytest.mark.asyncio
    async def test_reason_with_template(self, reasoning_engine):
        """テンプレート使用推論テスト"""
        # 初期化
        await reasoning_engine.initialize()
        
        # モックLLMの設定
        reasoning_engine.ollama_client.primary_llm.invoke = Mock(return_value="Analysis result")
        
        # テンプレート使用推論
        response = await reasoning_engine.reason(
            prompt="Test content",
            template_name="analysis",
            template_variables={"content": "Test content", "analysis_type": "品質分析"}
        )
        
        assert response.state == ReasoningState.COMPLETED
        assert response.template_used == "analysis"
        assert "Test content" in reasoning_engine.ollama_client.primary_llm.invoke.call_args[0][0]
    
    @pytest.mark.asyncio
    async def test_reason_error_handling(self, reasoning_engine):
        """推論エラーハンドリングテスト"""
        # 初期化
        await reasoning_engine.initialize()
        
        # エラーを発生させる
        reasoning_engine.ollama_client.primary_llm.invoke = Mock(side_effect=Exception("Test error"))
        
        # 推論実行
        response = await reasoning_engine.reason("Test prompt")
        
        assert response.state == ReasoningState.ERROR
        assert response.error_message == "Test error"
        assert reasoning_engine.performance_stats["failed_requests"] == 1
    
    @pytest.mark.asyncio
    async def test_batch_reason(self, reasoning_engine):
        """バッチ推論テスト"""
        # 初期化
        await reasoning_engine.initialize()
        
        # モックLLMの設定
        reasoning_engine.ollama_client.primary_llm.invoke = Mock(return_value="Batch response")
        
        # バッチリクエスト
        requests = [
            {"prompt": "Question 1"},
            {"prompt": "Question 2", "template_name": "basic_qa"},
            {"prompt": "Question 3"}
        ]
        
        responses = await reasoning_engine.batch_reason(requests)
        
        assert len(responses) == 3
        assert all(r.state == ReasoningState.COMPLETED for r in responses)
        assert all(r.response_text == "Batch response" for r in responses)
    
    @pytest.mark.asyncio
    async def test_batch_reason_with_errors(self, reasoning_engine):
        """バッチ推論エラーテスト"""
        # 初期化
        await reasoning_engine.initialize()
        
        # 一部でエラーを発生させる
        def mock_invoke(prompt):
            if "error" in prompt:
                raise Exception("Batch error")
            return "Success response"
        
        reasoning_engine.ollama_client.primary_llm.invoke = Mock(side_effect=mock_invoke)
        
        # バッチリクエスト
        requests = [
            {"prompt": "Normal question"},
            {"prompt": "This will cause error"},
            {"prompt": "Another normal question"}
        ]
        
        responses = await reasoning_engine.batch_reason(requests)
        
        assert len(responses) == 3
        assert responses[0].state == ReasoningState.COMPLETED
        assert responses[1].state == ReasoningState.ERROR
        assert responses[2].state == ReasoningState.COMPLETED
    
    def test_get_performance_stats(self, reasoning_engine):
        """パフォーマンス統計取得テスト"""
        # 初期統計
        stats = reasoning_engine.get_performance_stats()
        assert stats["total_requests"] == 0
        assert stats["success_rate"] == 0.0
        
        # 履歴追加
        successful_response = ReasoningResponse(
            request_id="test1",
            response_text="Success",
            processing_time=1.0,
            state=ReasoningState.COMPLETED
        )
        
        error_response = ReasoningResponse(
            request_id="test2",
            response_text="",
            processing_time=0.0,
            state=ReasoningState.ERROR
        )
        
        reasoning_engine.reasoning_history.extend([successful_response, error_response])
        reasoning_engine.performance_stats["total_requests"] = 2
        reasoning_engine.performance_stats["successful_requests"] = 1
        reasoning_engine.performance_stats["failed_requests"] = 1
        
        stats = reasoning_engine.get_performance_stats()
        assert stats["total_requests"] == 2
        assert stats["success_rate"] == 0.5
    
    def test_get_reasoning_history(self, reasoning_engine):
        """推論履歴取得テスト"""
        # テスト履歴作成
        responses = [
            ReasoningResponse(
                request_id=f"test{i}",
                response_text=f"Response {i}",
                processing_time=1.0,
                state=ReasoningState.COMPLETED if i % 2 == 0 else ReasoningState.ERROR
            )
            for i in range(5)
        ]
        
        reasoning_engine.reasoning_history.extend(responses)
        
        # 全履歴取得
        all_history = reasoning_engine.get_reasoning_history()
        assert len(all_history) == 5
        
        # 件数制限
        limited_history = reasoning_engine.get_reasoning_history(limit=3)
        assert len(limited_history) == 3
        
        # 状態フィルタ
        completed_history = reasoning_engine.get_reasoning_history(
            state_filter=ReasoningState.COMPLETED
        )
        assert len(completed_history) == 3  # 0, 2, 4番目
        assert all(r.state == ReasoningState.COMPLETED for r in completed_history)
    
    @pytest.mark.asyncio
    async def test_clear_history(self, reasoning_engine):
        """履歴クリアテスト"""
        # 履歴追加
        response = ReasoningResponse(
            request_id="test",
            response_text="Test",
            processing_time=1.0
        )
        reasoning_engine.reasoning_history.append(response)
        
        assert len(reasoning_engine.reasoning_history) == 1
        
        # クリア
        await reasoning_engine.clear_history()
        
        assert len(reasoning_engine.reasoning_history) == 0


class TestReasoningCallbackHandler:
    """ReasoningCallbackHandler クラスのテスト"""
    
    @pytest.fixture
    def callback_handler(self):
        """コールバックハンドラー"""
        return ReasoningCallbackHandler("test_request")
    
    def test_init(self, callback_handler):
        """初期化テスト"""
        assert callback_handler.request_id == "test_request"
        assert callback_handler.start_time is None
        assert callback_handler.token_count == 0
        assert isinstance(callback_handler.processing_metrics, dict)
    
    def test_on_llm_start(self, callback_handler):
        """LLM開始時テスト"""
        serialized = {"name": "test_llm"}
        prompts = ["Test prompt"]
        
        callback_handler.on_llm_start(serialized, prompts)
        
        assert callback_handler.start_time is not None
    
    def test_on_llm_end(self, callback_handler):
        """LLM終了時テスト"""
        from langchain.schema import Generation
        from langchain_core.outputs import LLMResult
        
        # 開始時間設定
        callback_handler.start_time = time.time() - 1  # 1秒前
        
        # モックレスポンス
        generation = Generation(text="Test response text")
        llm_result = LLMResult(generations=[[generation]])
        
        callback_handler.on_llm_end(llm_result)
        
        assert "processing_time" in callback_handler.processing_metrics
        assert callback_handler.processing_metrics["processing_time"] > 0
        assert callback_handler.token_count > 0
    
    def test_on_llm_error(self, callback_handler):
        """LLMエラー時テスト"""
        test_error = Exception("Test error")
        
        # エラーハンドリング（ログ出力のみなので例外が発生しないことを確認）
        try:
            callback_handler.on_llm_error(test_error)
        except Exception as e:
            pytest.fail(f"on_llm_error should not raise exception: {e}")


class TestReasoningDataClasses:
    """推論データクラスのテスト"""
    
    def test_reasoning_request(self):
        """ReasoningRequest テスト"""
        request = ReasoningRequest(
            request_id="test_req",
            prompt="Test prompt",
            template_name="test_template",
            template_variables={"var1": "value1"},
            max_tokens=100,
            temperature=0.7
        )
        
        assert request.request_id == "test_req"
        assert request.prompt == "Test prompt"
        assert request.template_name == "test_template"
        assert request.template_variables == {"var1": "value1"}
        assert request.max_tokens == 100
        assert request.temperature == 0.7
        assert isinstance(request.timestamp, datetime)
    
    def test_reasoning_response(self):
        """ReasoningResponse テスト"""
        response = ReasoningResponse(
            request_id="test_resp",
            response_text="Test response",
            processing_time=1.5,
            token_count=50,
            model_used="test_model",
            template_used="test_template",
            state=ReasoningState.COMPLETED
        )
        
        assert response.request_id == "test_resp"
        assert response.response_text == "Test response"
        assert response.processing_time == 1.5
        assert response.token_count == 50
        assert response.model_used == "test_model"
        assert response.template_used == "test_template"
        assert response.state == ReasoningState.COMPLETED
        assert response.error_message is None
        assert isinstance(response.timestamp, datetime)


class TestCreateBasicReasoningEngine:
    """create_basic_reasoning_engine 関数のテスト"""
    
    @pytest.mark.asyncio
    async def test_create_success(self):
        """作成成功テスト"""
        mock_client = Mock(spec=OllamaClient)
        mock_client.primary_llm = Mock()
        
        with patch('src.advanced_agent.reasoning.basic_engine.BasicReasoningEngine') as MockEngine:
            mock_engine = Mock()
            mock_engine.initialize = AsyncMock(return_value=True)
            MockEngine.return_value = mock_engine
            
            result = await create_basic_reasoning_engine(mock_client)
            
            assert result == mock_engine
            mock_engine.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_failure(self):
        """作成失敗テスト"""
        mock_client = Mock(spec=OllamaClient)
        
        with patch('src.advanced_agent.reasoning.basic_engine.BasicReasoningEngine') as MockEngine:
            mock_engine = Mock()
            mock_engine.initialize = AsyncMock(return_value=False)
            MockEngine.return_value = mock_engine
            
            with pytest.raises(RuntimeError, match="Failed to initialize basic reasoning engine"):
                await create_basic_reasoning_engine(mock_client)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])