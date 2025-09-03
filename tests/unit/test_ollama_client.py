"""
Ollama クライアントのテスト
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.advanced_agent.inference.ollama_client import (
    OllamaClient, InferenceRequest, InferenceResponse, ModelInfo, ModelStatus
)
from src.advanced_agent.core.config import AdvancedAgentConfig


class TestOllamaClient:
    """OllamaClient クラスのテスト"""
    
    @pytest.fixture
    def mock_config(self):
        """モック設定"""
        config = AdvancedAgentConfig()
        config.models.ollama_base_url = "http://localhost:11434"
        config.models.primary = "deepseek-r1:7b"
        config.models.fallback = "qwen2.5:7b-instruct-q4_k_m"
        config.models.emergency = "qwen2:1.5b-instruct-q4_k_m"
        return config
    
    @pytest.fixture
    def mock_ollama_response(self):
        """モックOllamaレスポンス"""
        return {
            'models': [
                {
                    'name': 'deepseek-r1:7b',
                    'size': 4 * 1024**3,  # 4GB
                    'details': {'parameter_size': '7B'}
                },
                {
                    'name': 'qwen2.5:7b-instruct-q4_k_m',
                    'size': 3 * 1024**3,  # 3GB
                    'details': {'parameter_size': '7B'}
                }
            ]
        }
    
    @pytest.fixture
    def ollama_client(self, mock_config):
        """OllamaClient インスタンス"""
        with patch('src.advanced_agent.inference.ollama_client.get_config', return_value=mock_config):
            return OllamaClient()
    
    def test_init(self, ollama_client, mock_config):
        """初期化テスト"""
        assert ollama_client.base_url == mock_config.models.ollama_base_url
        assert ollama_client.primary_model == mock_config.models.primary
        assert ollama_client.fallback_model == mock_config.models.fallback
        assert ollama_client.emergency_model == mock_config.models.emergency
        assert isinstance(ollama_client.model_cache, dict)
    
    @pytest.mark.asyncio
    async def test_check_server_connection_success(self, ollama_client, mock_ollama_response):
        """サーバー接続成功テスト"""
        with patch.object(ollama_client.ollama_client, 'list', return_value=mock_ollama_response):
            result = await ollama_client._check_server_connection()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_check_server_connection_failure(self, ollama_client):
        """サーバー接続失敗テスト"""
        with patch.object(ollama_client.ollama_client, 'list', side_effect=Exception("Connection failed")):
            result = await ollama_client._check_server_connection()
            assert result is False
    
    @pytest.mark.asyncio
    async def test_refresh_model_list(self, ollama_client, mock_ollama_response):
        """モデルリスト更新テスト"""
        with patch.object(ollama_client.ollama_client, 'list', return_value=mock_ollama_response):
            await ollama_client._refresh_model_list()
            
            assert len(ollama_client.model_cache) == 2
            assert 'deepseek-r1:7b' in ollama_client.model_cache
            assert 'qwen2.5:7b-instruct-q4_k_m' in ollama_client.model_cache
            
            # モデル情報確認
            deepseek_model = ollama_client.model_cache['deepseek-r1:7b']
            assert deepseek_model.name == 'deepseek-r1:7b'
            assert deepseek_model.size_gb == 4.0
            assert deepseek_model.status == ModelStatus.AVAILABLE
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, ollama_client, mock_ollama_response):
        """初期化成功テスト"""
        with patch.object(ollama_client.ollama_client, 'list', return_value=mock_ollama_response):
            result = await ollama_client.initialize()
            
            assert result is True
            assert len(ollama_client.model_cache) > 0
            assert ollama_client.primary_llm is not None
            assert ollama_client.fallback_llm is not None
    
    @pytest.mark.asyncio
    async def test_initialize_failure(self, ollama_client):
        """初期化失敗テスト"""
        with patch.object(ollama_client.ollama_client, 'list', side_effect=Exception("Connection failed")):
            result = await ollama_client.initialize()
            assert result is False
    
    @pytest.mark.asyncio
    async def test_build_prompt(self, ollama_client):
        """プロンプト構築テスト"""
        request = InferenceRequest(
            prompt="What is AI?",
            system_message="You are a helpful assistant.",
            context=["AI is artificial intelligence", "Machine learning is a subset of AI"]
        )
        
        prompt = await ollama_client._build_prompt(request)
        
        assert "System: You are a helpful assistant." in prompt
        assert "Context:" in prompt
        assert "- AI is artificial intelligence" in prompt
        assert "- Machine learning is a subset of AI" in prompt
        assert "Human: What is AI?" in prompt
        assert "Assistant:" in prompt
    
    @pytest.mark.asyncio
    async def test_generate_success(self, ollama_client, mock_ollama_response):
        """推論成功テスト"""
        # 初期化
        with patch.object(ollama_client.ollama_client, 'list', return_value=mock_ollama_response):
            await ollama_client.initialize()
        
        # モックLLMレスポンス
        mock_response = "AI stands for Artificial Intelligence."
        
        with patch.object(ollama_client.primary_llm, 'invoke', return_value=mock_response):
            request = InferenceRequest(prompt="What is AI?")
            response = await ollama_client.generate(request)
            
            assert isinstance(response, InferenceResponse)
            assert response.content == mock_response
            assert response.model_used == ollama_client.primary_model
            assert response.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_generate_with_fallback(self, ollama_client, mock_ollama_response):
        """フォールバック推論テスト"""
        # 初期化
        with patch.object(ollama_client.ollama_client, 'list', return_value=mock_ollama_response):
            await ollama_client.initialize()
        
        # プライマリモデルでエラー、フォールバックで成功
        fallback_response = "AI is artificial intelligence (from fallback)."
        
        with patch.object(ollama_client.primary_llm, 'invoke', side_effect=Exception("Primary model failed")), \
             patch.object(ollama_client.fallback_llm, 'invoke', return_value=fallback_response):
            
            request = InferenceRequest(prompt="What is AI?")
            response = await ollama_client.generate(request)
            
            assert response.content == fallback_response
            assert response.model_used == ollama_client.fallback_model
            assert response.metadata.get("fallback_used") is True
    
    @pytest.mark.asyncio
    async def test_chat(self, ollama_client, mock_ollama_response):
        """チャット形式テスト"""
        # 初期化
        with patch.object(ollama_client.ollama_client, 'list', return_value=mock_ollama_response):
            await ollama_client.initialize()
        
        mock_response = "Hello! I'm doing well, thank you for asking."
        
        with patch.object(ollama_client.primary_llm, 'invoke', return_value=mock_response):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm fine, thank you!"},
                {"role": "user", "content": "What can you help me with?"}
            ]
            
            response = await ollama_client.chat(messages)
            
            assert isinstance(response, InferenceResponse)
            assert response.content == mock_response
    
    @pytest.mark.asyncio
    async def test_get_model_info(self, ollama_client, mock_ollama_response):
        """モデル情報取得テスト"""
        with patch.object(ollama_client.ollama_client, 'list', return_value=mock_ollama_response):
            await ollama_client._refresh_model_list()
        
        model_info = await ollama_client.get_model_info('deepseek-r1:7b')
        
        assert model_info is not None
        assert model_info.name == 'deepseek-r1:7b'
        assert model_info.size_gb == 4.0
        assert model_info.status == ModelStatus.AVAILABLE
        
        # 存在しないモデル
        non_existent = await ollama_client.get_model_info('non-existent-model')
        assert non_existent is None
    
    @pytest.mark.asyncio
    async def test_list_models(self, ollama_client, mock_ollama_response):
        """モデル一覧テスト"""
        with patch.object(ollama_client.ollama_client, 'list', return_value=mock_ollama_response):
            models = await ollama_client.list_models()
            
            assert len(models) == 2
            assert all(isinstance(model, ModelInfo) for model in models)
            
            model_names = [model.name for model in models]
            assert 'deepseek-r1:7b' in model_names
            assert 'qwen2.5:7b-instruct-q4_k_m' in model_names
    
    @pytest.mark.asyncio
    async def test_pull_model_success(self, ollama_client):
        """モデルダウンロード成功テスト"""
        with patch.object(ollama_client.ollama_client, 'pull', return_value=None), \
             patch.object(ollama_client, '_refresh_model_list', return_value=None):
            
            result = await ollama_client.pull_model('new-model:latest')
            assert result is True
    
    @pytest.mark.asyncio
    async def test_pull_model_failure(self, ollama_client):
        """モデルダウンロード失敗テスト"""
        with patch.object(ollama_client.ollama_client, 'pull', side_effect=Exception("Download failed")):
            result = await ollama_client.pull_model('invalid-model')
            assert result is False
    
    @pytest.mark.asyncio
    async def test_delete_model_success(self, ollama_client):
        """モデル削除成功テスト"""
        # モデルをキャッシュに追加
        ollama_client.model_cache['test-model'] = ModelInfo(
            name='test-model',
            size_gb=1.0,
            status=ModelStatus.AVAILABLE
        )
        
        with patch.object(ollama_client.ollama_client, 'delete', return_value=None):
            result = await ollama_client.delete_model('test-model')
            
            assert result is True
            assert 'test-model' not in ollama_client.model_cache
    
    @pytest.mark.asyncio
    async def test_delete_model_failure(self, ollama_client):
        """モデル削除失敗テスト"""
        with patch.object(ollama_client.ollama_client, 'delete', side_effect=Exception("Delete failed")):
            result = await ollama_client.delete_model('test-model')
            assert result is False
    
    @pytest.mark.asyncio
    async def test_health_check(self, ollama_client, mock_ollama_response):
        """ヘルスチェックテスト"""
        with patch.object(ollama_client.ollama_client, 'list', return_value=mock_ollama_response), \
             patch.object(ollama_client, 'generate', return_value=InferenceResponse(
                 content="Hello",
                 model_used="deepseek-r1:7b",
                 processing_time=0.5
             )):
            
            health = await ollama_client.health_check()
            
            assert health["server_connected"] is True
            assert health["models_available"] == 2
            assert health["primary_model_available"] is True
            assert health["fallback_model_available"] is True
            assert health["inference_test"] == "passed"
            assert "test_response_time" in health
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, ollama_client):
        """ヘルスチェック失敗テスト"""
        with patch.object(ollama_client, '_check_server_connection', return_value=False):
            health = await ollama_client.health_check()
            
            assert health["server_connected"] is False
            assert health["models_available"] == 0
            assert health["primary_model_available"] is False
            assert health["fallback_model_available"] is False


class TestInferenceRequest:
    """InferenceRequest クラスのテスト"""
    
    def test_default_values(self):
        """デフォルト値テスト"""
        request = InferenceRequest(prompt="Test prompt")
        
        assert request.prompt == "Test prompt"
        assert request.model_name is None
        assert request.temperature == 0.1
        assert request.max_tokens is None
        assert request.stream is False
        assert request.system_message is None
        assert request.context is None
        assert isinstance(request.metadata, dict)
    
    def test_custom_values(self):
        """カスタム値テスト"""
        request = InferenceRequest(
            prompt="Custom prompt",
            model_name="custom-model",
            temperature=0.5,
            max_tokens=100,
            stream=True,
            system_message="Custom system message",
            context=["context1", "context2"],
            metadata={"key": "value"}
        )
        
        assert request.prompt == "Custom prompt"
        assert request.model_name == "custom-model"
        assert request.temperature == 0.5
        assert request.max_tokens == 100
        assert request.stream is True
        assert request.system_message == "Custom system message"
        assert request.context == ["context1", "context2"]
        assert request.metadata == {"key": "value"}


class TestInferenceResponse:
    """InferenceResponse クラスのテスト"""
    
    def test_basic_response(self):
        """基本レスポンステスト"""
        response = InferenceResponse(
            content="Test response",
            model_used="test-model",
            processing_time=1.5
        )
        
        assert response.content == "Test response"
        assert response.model_used == "test-model"
        assert response.processing_time == 1.5
        assert response.token_count is None
        assert response.finish_reason is None
        assert isinstance(response.metadata, dict)
    
    def test_complete_response(self):
        """完全レスポンステスト"""
        response = InferenceResponse(
            content="Complete response",
            model_used="complete-model",
            processing_time=2.0,
            token_count=50,
            finish_reason="stop",
            metadata={"fallback_used": True}
        )
        
        assert response.content == "Complete response"
        assert response.model_used == "complete-model"
        assert response.processing_time == 2.0
        assert response.token_count == 50
        assert response.finish_reason == "stop"
        assert response.metadata["fallback_used"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])