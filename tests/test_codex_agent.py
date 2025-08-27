"""
Codex Agent Unit Tests
Codex互換エージェントの単体テスト
"""

import pytest
import asyncio
import os
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

# テスト用のパス設定
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from codex_agent.config import CodexConfig, ModelProviderInfo
from codex_agent.errors import CodexError, OllamaConnectionError, ValidationError
from codex_agent.compatibility_layer import CompatibilityLayer
from codex_agent.ollama_client import CodexOllamaClient
from codex_agent.agent_interface import CodexAgentInterface


class TestCodexConfig:
    """CodexConfig のテスト"""
    
    def test_default_config(self):
        """デフォルト設定のテスト"""
        config = CodexConfig()
        
        assert config.model == "qwen2:7b-instruct"
        assert config.ollama_base_url == "http://localhost:11434"
        assert config.model_provider_id == "ollama"
        assert config.ollama_timeout == 30
    
    def test_env_override(self):
        """環境変数による設定上書きのテスト"""
        with patch.dict(os.environ, {
            'OLLAMA_BASE_URL': 'http://test:11434',
            'OLLAMA_MODEL': 'test-model',
            'OLLAMA_TIMEOUT': '60'
        }):
            config = CodexConfig()
            assert config.ollama_base_url == "http://test:11434"
            assert config.model == "test-model"
            assert config.ollama_timeout == 60
    
    def test_model_provider_property(self):
        """モデルプロバイダープロパティのテスト"""
        config = CodexConfig()
        provider = config.model_provider
        
        assert isinstance(provider, ModelProviderInfo)
        assert provider.base_url == config.ollama_base_url
        assert provider.wire_api == "chat"
    
    def test_validation_success(self):
        """設定バリデーション成功のテスト"""
        config = CodexConfig()
        assert config.validate() is True
    
    def test_validation_failure(self):
        """設定バリデーション失敗のテスト"""
        config = CodexConfig()
        config.model = ""  # 空のモデル名
        
        with pytest.raises(ValueError, match="Model name is required"):
            config.validate()


class TestCompatibilityLayer:
    """CompatibilityLayer のテスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.layer = CompatibilityLayer()
    
    def test_codex_to_ollama_basic(self):
        """基本的なCodex→OLLAMA変換のテスト"""
        codex_request = {
            "prompt": "Hello, world!",
            "model": "test-model",
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        ollama_request = self.layer.translate_codex_to_ollama(codex_request)
        
        assert ollama_request["model"] == "test-model"
        assert ollama_request["prompt"] == "Hello, world!"
        assert ollama_request["options"]["num_predict"] == 100
        assert ollama_request["options"]["temperature"] == 0.7
    
    def test_ollama_to_codex_basic(self):
        """基本的なOLLAMA→Codex変換のテスト"""
        ollama_response = {
            "response": "Hello there!",
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5
        }
        
        original_request = {"model": "test-model"}
        
        codex_response = self.layer.translate_ollama_to_codex(
            ollama_response, original_request
        )
        
        assert codex_response["object"] == "text_completion"
        assert codex_response["model"] == "test-model"
        assert len(codex_response["choices"]) == 1
        assert codex_response["choices"][0]["text"] == "Hello there!"
        assert codex_response["usage"]["prompt_tokens"] == 10
        assert codex_response["usage"]["completion_tokens"] == 5
        assert codex_response["usage"]["total_tokens"] == 15
    
    def test_chat_request_conversion(self):
        """チャットリクエスト変換のテスト"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        prompt = self.layer.translate_chat_request(messages)
        
        expected_parts = [
            "System: You are a helpful assistant.",
            "User: Hello!",
            "Assistant: Hi there!",
            "User: How are you?",
            "Assistant:"
        ]
        
        assert prompt == "\n\n".join(expected_parts)
    
    def test_request_validation_success(self):
        """リクエストバリデーション成功のテスト"""
        valid_request = {
            "prompt": "Test prompt",
            "temperature": 0.5,
            "max_tokens": 100
        }
        
        assert self.layer.validate_request(valid_request) is True
    
    def test_request_validation_missing_prompt(self):
        """プロンプト欠如バリデーションのテスト"""
        invalid_request = {"temperature": 0.5}
        
        with pytest.raises(ValidationError, match="Missing required field: prompt"):
            self.layer.validate_request(invalid_request)
    
    def test_request_validation_invalid_temperature(self):
        """不正な温度バリデーションのテスト"""
        invalid_request = {
            "prompt": "Test",
            "temperature": 3.0  # 範囲外
        }
        
        with pytest.raises(ValidationError, match="Temperature must be between 0 and 2"):
            self.layer.validate_request(invalid_request)


class TestCodexErrors:
    """エラーハンドリングのテスト"""
    
    def test_codex_error_creation(self):
        """CodexError作成のテスト"""
        error = CodexError("Test error", details={"key": "value"})
        
        assert str(error) == "[unknown_error] Test error"
        assert error.details["key"] == "value"
    
    def test_ollama_connection_error(self):
        """OLLAMA接続エラーのテスト"""
        error = OllamaConnectionError()
        
        assert "No running Ollama server detected" in str(error)
        assert error.details["service"] == "ollama"
    
    def test_error_to_dict(self):
        """エラー辞書変換のテスト"""
        error = CodexError("Test", details={"test": True})
        error_dict = error.to_dict()
        
        assert error_dict["type"] == "unknown_error"
        assert error_dict["message"] == "Test"
        assert error_dict["details"]["test"] is True


@pytest.mark.asyncio
class TestCodexOllamaClient:
    """CodexOllamaClient のテスト（モック使用）"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.config = CodexConfig()
    
    @patch('codex_agent.ollama_client.CodexOllamaClient.health_check')
    async def test_health_check_success(self, mock_health_check):
        """ヘルスチェック成功のテスト"""
        mock_health_check.return_value = {"status": "ok", "url": "http://localhost:11434/api/tags"}
        
        client = CodexOllamaClient(self.config)
        result = await client.health_check()
        
        assert result["status"] == "ok"
        mock_health_check.assert_called_once()
    
    @patch('codex_agent.ollama_client.CodexOllamaClient.health_check')
    async def test_health_check_failure(self, mock_health_check):
        """ヘルスチェック失敗のテスト"""
        mock_health_check.side_effect = ConnectionError("No running Ollama server detected")
        
        client = CodexOllamaClient(self.config)
        
        with pytest.raises(ConnectionError, match="No running Ollama server detected"):
            await client.health_check()
    
    async def test_client_initialization(self):
        """クライアント初期化のテスト"""
        client = CodexOllamaClient(self.config)
        
        assert client.base_url == self.config.ollama_base_url
        assert client.config.model == self.config.model
        assert client.session is None  # 初期化前はNone


@pytest.mark.asyncio
class TestCodexAgentInterface:
    """CodexAgentInterface のテスト（モック使用）"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.config = CodexConfig()
    
    async def test_initialization_not_called(self):
        """初期化前の操作エラーのテスト"""
        agent = CodexAgentInterface(self.config)
        
        with pytest.raises(CodexError, match="Agent not initialized"):
            await agent.complete("test prompt")
    
    async def test_complete_success(self):
        """補完成功のテスト"""
        # モッククライアント設定
        mock_client = AsyncMock()
        
        # シンプルなレスポンスオブジェクト
        class MockResponse:
            def __init__(self):
                self.response = "Generated text"
                self.done = True
                self.prompt_eval_count = 10
                self.eval_count = 5
                self.__dict__ = {
                    "response": "Generated text",
                    "done": True,
                    "prompt_eval_count": 10,
                    "eval_count": 5
                }
        
        mock_client.generate.return_value = MockResponse()
        mock_client.get_model_info.return_value = {"model": "test-model"}
        
        agent = CodexAgentInterface(self.config)
        agent.ollama_client = mock_client
        agent._initialized = True
        
        result = await agent.complete("test prompt")
        
        assert result["choices"][0]["text"] == "Generated text"
        assert "response_time" in result
        assert "model_info" in result
    
    async def test_complete_error_handling(self):
        """補完エラーハンドリングのテスト"""
        mock_client = AsyncMock()
        mock_client.generate.side_effect = Exception("OLLAMA connection failed")
        
        agent = CodexAgentInterface(self.config)
        agent.ollama_client = mock_client
        agent._initialized = True
        
        result = await agent.complete("test prompt")
        
        assert "error" in result
        assert result["object"] == "error"
    
    async def test_chat_success(self):
        """チャット成功のテスト"""
        mock_client = AsyncMock()
        
        class MockResponse:
            def __init__(self):
                self.response = "Hello there!"
                self.done = True
                self.prompt_eval_count = 15
                self.eval_count = 8
                self.__dict__ = {
                    "response": "Hello there!",
                    "done": True,
                    "prompt_eval_count": 15,
                    "eval_count": 8
                }
        
        mock_client.generate.return_value = MockResponse()
        mock_client.get_model_info.return_value = {"model": "test-model"}
        
        agent = CodexAgentInterface(self.config)
        agent.ollama_client = mock_client
        agent._initialized = True
        
        messages = [{"role": "user", "content": "Hello!"}]
        result = await agent.chat(messages)
        
        assert result["choices"][0]["text"].strip() == "Hello there!"
        assert "session_id" in result
        assert "message_count" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])