"""
基本推論エンジンのテスト
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.advanced_agent.reasoning.base_engine import (
    BasicReasoningEngine, MemoryAwareReasoningEngine, ReasoningRequest,
    ReasoningContext, ReasoningResult, create_reasoning_engine, quick_reasoning
)
from src.advanced_agent.reasoning.callbacks import PerformanceCallbackHandler
from src.advanced_agent.inference.ollama_client import OllamaClient


class TestBasicReasoningEngine:
    """BasicReasoningEngine クラスのテスト"""
    
    @pytest.fixture
    def mock_ollama_client(self):
        """モック Ollama クライアント"""
        client = Mock(spec=OllamaClient)
        client.generate = AsyncMock()
        return client
    
    @pytest.fixture
    def reasoning_engine(self, mock_ollama_client):
        """BasicReasoningEngine インスタンス"""
        return BasicReasoningEngine(mock_ollama_client)
    
    def test_init(self, reasoning_engine, mock_ollama_client):
        """初期化テスト"""
        assert reasoning_engine.ollama_client == mock_ollama_client
        assert reasoning_engine.config is not None
        assert reasoning_engine.logger is not None
        assert len(reasoning_engine.prompt_templates) > 0
    
    def test_get_prompt_template(self, reasoning_engine):
        """プロンプトテンプレート取得テスト"""
        # 存在するテンプレート
        general_template = reasoning_engine.get_prompt_template("general")
        assert general_template is not None
        assert "question" in general_template.input_variables
        
        # 存在しないテンプレート（デフォルトを返す）
        unknown_template = reasoning_engine.get_prompt_template("unknown_type")
        assert unknown_template is not None
        assert unknown_template == reasoning_engine.prompt_templates["general"]
    
    def test_build_context_string(self, reasoning_engine):
        """コンテキスト文字列構築テスト"""
        context = ReasoningContext(
            session_id="test_session",
            system_context="テストシステム",
            domain_context="テストドメイン",
            conversation_history=[
                {"role": "user", "content": "こんにちは"},
                {"role": "assistant", "content": "こんにちは！"}
            ],
            metadata={"topic": "greeting", "priority": "low"}
        )
        
        context_str = reasoning_engine._build_context_string(context)
        
        assert "システム情報: テストシステム" in context_str
        assert "ドメイン情報: テストドメイン" in context_str
        assert "最近の会話:" in context_str
        assert "user: こんにちは" in context_str
        assert "assistant: こんにちは！" in context_str
        assert "topic" in context_str
    
    def test_build_context_string_empty(self, reasoning_engine):
        """空のコンテキスト文字列構築テスト"""
        context = ReasoningContext(session_id="empty_session")
        context_str = reasoning_engine._build_context_string(context)
        
        assert "特別なコンテキストはありません" in context_str
    
    def test_build_constraints_string(self, reasoning_engine):
        """制約条件文字列構築テスト"""
        constraints = ["制約1", "制約2", "制約3"]
        constraints_str = reasoning_engine._build_constraints_string(constraints)
        
        assert "1. 制約1" in constraints_str
        assert "2. 制約2" in constraints_str
        assert "3. 制約3" in constraints_str
    
    def test_build_constraints_string_empty(self, reasoning_engine):
        """空の制約条件文字列構築テスト"""
        constraints_str = reasoning_engine._build_constraints_string([])
        assert "特別な制約はありません" in constraints_str
    
    def test_calculate_confidence(self, reasoning_engine):
        """信頼度計算テスト"""
        # 通常のレスポンス
        normal_response = "これは通常の回答です。明確で詳細な説明を提供します。"
        confidence = reasoning_engine._calculate_confidence(normal_response)
        assert 0.0 <= confidence <= 1.0
        
        # 短いレスポンス
        short_response = "短い回答"
        short_confidence = reasoning_engine._calculate_confidence(short_response)
        assert short_confidence < confidence
        
        # 不確実性を含むレスポンス
        uncertain_response = "これはわからないですが、おそらく正しいかもしれません。"
        uncertain_confidence = reasoning_engine._calculate_confidence(uncertain_response)
        assert uncertain_confidence < confidence
        
        # 確実性を含むレスポンス
        certain_response = "これは確実に正しい事実として知られています。明確に証明されています。"
        certain_confidence = reasoning_engine._calculate_confidence(certain_response)
        assert certain_confidence >= confidence
    
    @pytest.mark.asyncio
    async def test_get_llm_with_callbacks(self, reasoning_engine):
        """コールバック付きLLM取得テスト"""
        callback = Mock()
        llm = await reasoning_engine._get_llm_with_callbacks([callback])
        
        assert llm is not None
        assert callback in llm.callbacks
        assert llm.model == reasoning_engine.config.models.primary
        assert llm.base_url == reasoning_engine.config.models.ollama_base_url
    
    @pytest.mark.asyncio
    async def test_reason_success(self, reasoning_engine):
        """推論成功テスト"""
        # LLMChain のモック
        with patch('src.advanced_agent.reasoning.base_engine.LLMChain') as mock_chain_class:
            mock_chain = Mock()
            mock_chain.arun = AsyncMock(return_value="これはテスト回答です。")
            mock_chain_class.return_value = mock_chain
            
            # 推論リクエスト
            context = ReasoningContext(session_id="test_session")
            request = ReasoningRequest(
                prompt="テスト質問",
                context=context,
                reasoning_type="general"
            )
            
            result = await reasoning_engine.reason(request)
            
            assert isinstance(result, ReasoningResult)
            assert result.final_answer == "これはテスト回答です。"
            assert result.processing_time > 0
            assert result.confidence_score > 0
            assert result.context_used == context
    
    @pytest.mark.asyncio
    async def test_reason_error(self, reasoning_engine):
        """推論エラーテスト"""
        # LLMChain でエラーが発生する場合
        with patch('src.advanced_agent.reasoning.base_engine.LLMChain') as mock_chain_class:
            mock_chain = Mock()
            mock_chain.arun = AsyncMock(side_effect=Exception("Test error"))
            mock_chain_class.return_value = mock_chain
            
            context = ReasoningContext(session_id="error_test_session")
            request = ReasoningRequest(
                prompt="エラーテスト質問",
                context=context
            )
            
            result = await reasoning_engine.reason(request)
            
            assert isinstance(result, ReasoningResult)
            assert "エラーが発生しました" in result.final_answer
            assert result.confidence_score == 0.0
            assert "error" in result.metadata


class TestMemoryAwareReasoningEngine:
    """MemoryAwareReasoningEngine クラスのテスト"""
    
    @pytest.fixture
    def mock_ollama_client(self):
        """モック Ollama クライアント"""
        client = Mock(spec=OllamaClient)
        return client
    
    @pytest.fixture
    def memory_engine(self, mock_ollama_client):
        """MemoryAwareReasoningEngine インスタンス"""
        return MemoryAwareReasoningEngine(mock_ollama_client)
    
    def test_init(self, memory_engine, mock_ollama_client):
        """初期化テスト"""
        assert memory_engine.ollama_client == mock_ollama_client
        assert memory_engine.memory_enabled is False  # 初期状態では無効
    
    @pytest.mark.asyncio
    async def test_reason_without_memory(self, memory_engine):
        """記憶なし推論テスト"""
        # 基本推論エンジンの動作をモック
        with patch.object(BasicReasoningEngine, 'reason') as mock_reason:
            mock_result = ReasoningResult(
                request_id="test",
                final_answer="テスト回答",
                processing_time=1.0
            )
            mock_reason.return_value = mock_result
            
            context = ReasoningContext(session_id="memory_test")
            request = ReasoningRequest(
                prompt="記憶テスト質問",
                context=context,
                use_memory=True  # 記憶使用を要求するが、無効なので無視される
            )
            
            result = await memory_engine.reason(request)
            
            assert result == mock_result
            mock_reason.assert_called_once()


class TestReasoningFunctions:
    """推論関数のテスト"""
    
    @pytest.mark.asyncio
    async def test_create_reasoning_engine_basic(self):
        """基本推論エンジン作成テスト"""
        mock_client = Mock(spec=OllamaClient)
        
        engine = await create_reasoning_engine(mock_client, "basic")
        
        assert isinstance(engine, BasicReasoningEngine)
        assert engine.ollama_client == mock_client
    
    @pytest.mark.asyncio
    async def test_create_reasoning_engine_memory_aware(self):
        """記憶対応推論エンジン作成テスト"""
        mock_client = Mock(spec=OllamaClient)
        
        engine = await create_reasoning_engine(mock_client, "memory_aware")
        
        assert isinstance(engine, MemoryAwareReasoningEngine)
        assert engine.ollama_client == mock_client
    
    @pytest.mark.asyncio
    async def test_create_reasoning_engine_invalid_type(self):
        """無効なエンジンタイプテスト"""
        mock_client = Mock(spec=OllamaClient)
        
        with pytest.raises(ValueError, match="Unknown engine type"):
            await create_reasoning_engine(mock_client, "invalid_type")
    
    @pytest.mark.asyncio
    async def test_quick_reasoning(self):
        """簡易推論テスト"""
        with patch('src.advanced_agent.reasoning.base_engine.create_ollama_client') as mock_create_client, \
             patch('src.advanced_agent.reasoning.base_engine.create_reasoning_engine') as mock_create_engine:
            
            # モッククライアント
            mock_client = Mock(spec=OllamaClient)
            mock_create_client.return_value = mock_client
            
            # モック推論エンジン
            mock_engine = Mock()
            mock_result = ReasoningResult(
                request_id="quick_test",
                final_answer="簡易推論の回答",
                processing_time=0.5
            )
            mock_engine.reason = AsyncMock(return_value=mock_result)
            mock_create_engine.return_value = mock_engine
            
            # 簡易推論実行
            answer = await quick_reasoning("テスト質問", "general")
            
            assert answer == "簡易推論の回答"
            mock_create_client.assert_called_once()
            mock_create_engine.assert_called_once_with(mock_client)
            mock_engine.reason.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])