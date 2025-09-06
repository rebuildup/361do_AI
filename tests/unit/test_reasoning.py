"""
Unit tests for reasoning module
推論モジュールの単体テスト
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.advanced_agent.reasoning.ollama_client import (
    OllamaClient, OllamaMessage, OllamaRequest, OllamaResponse, OllamaStreamChunk
)
from src.advanced_agent.reasoning.prompt_templates import (
    PromptTemplateManager, PromptTemplate, PromptType, 
    get_template_manager, format_prompt
)
from src.advanced_agent.config.settings import OllamaConfig


class TestOllamaClient:
    """Ollamaクライアントテスト"""
    
    @pytest.fixture
    def ollama_config(self):
        """Ollama設定"""
        return OllamaConfig(
            base_url="http://localhost:11434",
            model="qwen2:7b-instruct",
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_tokens=2048,
            timeout=30,
            retry_attempts=3,
            retry_delay=1.0
        )
    
    @pytest.fixture
    def mock_ollama_client(self, ollama_config):
        """モックOllamaクライアント"""
        client = OllamaClient(ollama_config)
        client._client = Mock()
        client._session = Mock()
        return client
    
    def test_ollama_message_creation(self):
        """Ollamaメッセージ作成テスト"""
        message = OllamaMessage(
            role="user",
            content="Hello, world!"
        )
        
        assert message.role == "user"
        assert message.content == "Hello, world!"
        assert isinstance(message.timestamp, datetime)
    
    def test_ollama_request_creation(self):
        """Ollamaリクエスト作成テスト"""
        messages = [
            OllamaMessage(role="user", content="Hello")
        ]
        
        request = OllamaRequest(
            model="qwen2:7b-instruct",
            messages=messages,
            temperature=0.7
        )
        
        assert request.model == "qwen2:7b-instruct"
        assert len(request.messages) == 1
        assert request.temperature == 0.7
        assert request.stream is False
    
    def test_ollama_response_creation(self):
        """Ollamaレスポンス作成テスト"""
        response = OllamaResponse(
            content="Hello, world!",
            model="qwen2:7b-instruct",
            created_at=datetime.now()
        )
        
        assert response.content == "Hello, world!"
        assert response.model == "qwen2:7b-instruct"
        assert response.done is True
    
    def test_ollama_stream_chunk_creation(self):
        """Ollamaストリームチャンク作成テスト"""
        chunk = OllamaStreamChunk(
            content="Hello",
            done=False
        )
        
        assert chunk.content == "Hello"
        assert chunk.done is False
    
    @pytest.mark.asyncio
    async def test_ollama_client_initialization(self, ollama_config):
        """Ollamaクライアント初期化テスト"""
        with patch('httpx.AsyncClient') as mock_httpx, \
             patch('aiohttp.ClientSession') as mock_aiohttp:
            
            mock_httpx.return_value = Mock()
            mock_aiohttp.return_value = Mock()
            
            client = OllamaClient(ollama_config)
            await client.initialize()
            
            assert client._client is not None
            assert client._session is not None
    
    @pytest.mark.asyncio
    async def test_generate_complete(self, mock_ollama_client):
        """完全生成テスト"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {"content": "Hello, world!"},
            "model": "qwen2:7b-instruct",
            "created_at": datetime.now().isoformat(),
            "done": True
        }
        mock_response.raise_for_status.return_value = None
        
        mock_ollama_client._client.post = AsyncMock(return_value=mock_response)
        
        request = OllamaRequest(
            model="qwen2:7b-instruct",
            messages=[OllamaMessage(role="user", content="Hello")]
        )
        
        response = await mock_ollama_client._generate_complete(request)
        
        assert isinstance(response, OllamaResponse)
        assert response.content == "Hello, world!"
        assert response.model == "qwen2:7b-instruct"
    
    @pytest.mark.asyncio
    async def test_list_models(self, mock_ollama_client):
        """モデル一覧取得テスト"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [
                {"name": "qwen2:7b-instruct"},
                {"name": "llama2:7b"}
            ]
        }
        mock_response.raise_for_status.return_value = None
        
        mock_ollama_client._client.get = AsyncMock(return_value=mock_response)
        
        models = await mock_ollama_client.list_models()
        
        assert len(models) == 2
        assert models[0]["name"] == "qwen2:7b-instruct"
        assert models[1]["name"] == "llama2:7b"
    
    @pytest.mark.asyncio
    async def test_health_check(self, mock_ollama_client):
        """ヘルスチェックテスト"""
        with patch.object(mock_ollama_client, 'generate') as mock_generate:
            mock_response = OllamaResponse(
                content="Hello",
                model="qwen2:7b-instruct",
                created_at=datetime.now()
            )
            mock_generate.return_value = mock_response
            
            health = await mock_ollama_client.health_check()
            
            assert health["status"] == "healthy"
            assert health["model"] == "qwen2:7b-instruct"
            assert "response_time" in health


class TestPromptTemplates:
    """プロンプトテンプレートテスト"""
    
    @pytest.fixture
    def template_manager(self):
        """テンプレートマネージャー"""
        return PromptTemplateManager()
    
    def test_template_creation(self):
        """テンプレート作成テスト"""
        template = PromptTemplate(
            name="test_template",
            template="Hello {name}!",
            prompt_type=PromptType.USER,
            variables=["name"],
            description="Test template"
        )
        
        assert template.name == "test_template"
        assert template.template == "Hello {name}!"
        assert template.prompt_type == PromptType.USER
        assert template.variables == ["name"]
        assert template.description == "Test template"
    
    def test_template_manager_initialization(self, template_manager):
        """テンプレートマネージャー初期化テスト"""
        assert len(template_manager.templates) > 0
        assert "system_base" in template_manager.templates
        assert "reasoning_cot" in template_manager.templates
        assert "math_reasoning" in template_manager.templates
    
    def test_get_template(self, template_manager):
        """テンプレート取得テスト"""
        template = template_manager.get_template("system_base")
        
        assert template is not None
        assert template.name == "system_base"
        assert template.prompt_type == PromptType.SYSTEM
    
    def test_get_template_not_found(self, template_manager):
        """存在しないテンプレート取得テスト"""
        template = template_manager.get_template("nonexistent")
        
        assert template is None
    
    def test_get_templates_by_type(self, template_manager):
        """タイプ別テンプレート取得テスト"""
        system_templates = template_manager.get_templates_by_type(PromptType.SYSTEM)
        reasoning_templates = template_manager.get_templates_by_type(PromptType.REASONING)
        
        assert len(system_templates) >= 1
        assert len(reasoning_templates) >= 3
        
        for template in system_templates:
            assert template.prompt_type == PromptType.SYSTEM
    
    def test_get_templates_by_category(self, template_manager):
        """カテゴリ別テンプレート取得テスト"""
        system_templates = template_manager.get_templates_by_category("system")
        reasoning_templates = template_manager.get_templates_by_category("reasoning")
        
        assert len(system_templates) >= 1
        assert len(reasoning_templates) >= 1
    
    def test_format_template(self, template_manager):
        """テンプレートフォーマットテスト"""
        formatted = template_manager.format_template(
            "system_base",
            session_id="test_123",
            learning_epoch=5,
            total_interactions=100,
            reward_score=0.85
        )
        
        assert "test_123" in formatted
        assert "5" in formatted
        assert "100" in formatted
        assert "0.85" in formatted
    
    def test_format_template_missing_variable(self, template_manager):
        """テンプレートフォーマット（変数不足）テスト"""
        with pytest.raises(ValueError, match="Missing variable"):
            template_manager.format_template(
                "system_base",
                session_id="test_123"
                # 他の必須変数が不足
            )
    
    def test_add_template(self, template_manager):
        """テンプレート追加テスト"""
        new_template = PromptTemplate(
            name="custom_template",
            template="Custom: {input}",
            prompt_type=PromptType.USER,
            variables=["input"],
            description="Custom template"
        )
        
        template_manager.add_template(new_template)
        
        retrieved = template_manager.get_template("custom_template")
        assert retrieved is not None
        assert retrieved.name == "custom_template"
    
    def test_remove_template(self, template_manager):
        """テンプレート削除テスト"""
        # カスタムテンプレート追加
        custom_template = PromptTemplate(
            name="to_remove",
            template="To be removed",
            prompt_type=PromptType.USER,
            variables=[],
            description="To be removed"
        )
        template_manager.add_template(custom_template)
        
        # 削除
        result = template_manager.remove_template("to_remove")
        assert result is True
        
        # 確認
        retrieved = template_manager.get_template("to_remove")
        assert retrieved is None
    
    def test_update_template(self, template_manager):
        """テンプレート更新テスト"""
        template_manager.update_template(
            "system_base",
            description="Updated description"
        )
        
        template = template_manager.get_template("system_base")
        assert template.description == "Updated description"
    
    def test_get_template_stats(self, template_manager):
        """テンプレート統計取得テスト"""
        stats = template_manager.get_template_stats()
        
        assert "total_templates" in stats
        assert "by_type" in stats
        assert "by_category" in stats
        assert "most_used" in stats
        assert "least_used" in stats
        
        assert stats["total_templates"] > 0
        assert len(stats["by_type"]) > 0
        assert len(stats["by_category"]) > 0
    
    def test_export_import_templates(self, template_manager):
        """テンプレートエクスポート・インポートテスト"""
        # エクスポート
        exported = template_manager.export_templates()
        
        assert isinstance(exported, dict)
        assert len(exported) > 0
        
        # 新しいマネージャー作成
        new_manager = PromptTemplateManager()
        new_manager.templates.clear()  # デフォルトテンプレートをクリア
        
        # インポート
        new_manager.import_templates(exported)
        
        # 確認
        assert len(new_manager.templates) == len(exported)
        for name in exported:
            assert name in new_manager.templates


class TestPromptTemplateConvenienceFunctions:
    """プロンプトテンプレート便利関数テスト"""
    
    def test_get_template_manager_singleton(self):
        """テンプレートマネージャーシングルトンテスト"""
        manager1 = get_template_manager()
        manager2 = get_template_manager()
        
        assert manager1 is manager2
    
    def test_format_prompt_function(self):
        """format_prompt関数テスト"""
        formatted = format_prompt(
            "system_base",
            session_id="test_123",
            learning_epoch=5,
            total_interactions=100,
            reward_score=0.85
        )
        
        assert "test_123" in formatted
        assert "5" in formatted
        assert "100" in formatted
        assert "0.85" in formatted
    
    def test_get_system_prompt_function(self):
        """get_system_prompt関数テスト"""
        prompt = get_system_prompt(
            session_id="test_123",
            learning_epoch=5,
            total_interactions=100,
            reward_score=0.85
        )
        
        assert "test_123" in prompt
        assert "5" in prompt
        assert "100" in prompt
        assert "0.85" in prompt
        assert "自己学習AIエージェント" in prompt
    
    def test_get_reasoning_prompt_function(self):
        """get_reasoning_prompt関数テスト"""
        prompt = get_reasoning_prompt("2+2は何ですか？")
        
        assert "2+2は何ですか？" in prompt
        assert "段階的に解決" in prompt
        assert "理解" in prompt
        assert "分析" in prompt
    
    def test_get_math_prompt_function(self):
        """get_math_prompt関数テスト"""
        prompt = get_math_prompt("x^2 + 5x + 6 = 0 を解いてください")
        
        assert "x^2 + 5x + 6 = 0 を解いてください" in prompt
        assert "数学問題" in prompt
        assert "計算" in prompt
    
    def test_get_logic_prompt_function(self):
        """get_logic_prompt関数テスト"""
        prompt = get_logic_prompt("AはBより背が高い。BはCより背が高い。最も背が高いのは？")
        
        assert "AはBより背が高い" in prompt
        assert "論理問題" in prompt
        assert "推論" in prompt
    
    def test_get_evaluation_prompt_function(self):
        """get_evaluation_prompt関数テスト"""
        prompt = get_evaluation_prompt("質問", "回答")
        
        assert "質問" in prompt
        assert "回答" in prompt
        assert "評価" in prompt
        assert "正確性" in prompt
    
    def test_get_learning_prompt_function(self):
        """get_learning_prompt関数テスト"""
        prompt = get_learning_prompt("質問", "回答", "評価")
        
        assert "質問" in prompt
        assert "回答" in prompt
        assert "評価" in prompt
        assert "学習" in prompt


class TestPromptTemplateIntegration:
    """プロンプトテンプレート統合テスト"""
    
    def test_template_usage_tracking(self):
        """テンプレート使用追跡テスト"""
        manager = PromptTemplateManager()
        
        # 初期使用回数
        template = manager.get_template("system_base")
        initial_count = template.metadata.get("usage_count", 0)
        
        # テンプレート使用
        manager.format_template(
            "system_base",
            session_id="test",
            learning_epoch=1,
            total_interactions=1,
            reward_score=0.5
        )
        
        # 使用回数確認
        updated_template = manager.get_template("system_base")
        updated_count = updated_template.metadata.get("usage_count", 0)
        
        assert updated_count == initial_count + 1
    
    def test_template_variable_validation(self):
        """テンプレート変数検証テスト"""
        manager = PromptTemplateManager()
        
        # 正しい変数
        try:
            manager.format_template(
                "system_base",
                session_id="test",
                learning_epoch=1,
                total_interactions=1,
                reward_score=0.5
            )
        except ValueError:
            pytest.fail("Valid variables should not raise ValueError")
        
        # 不足した変数
        with pytest.raises(ValueError):
            manager.format_template(
                "system_base",
                session_id="test"
                # 他の必須変数が不足
            )
        
        # 余分な変数（警告は出るがエラーにはならない）
        try:
            manager.format_template(
                "system_base",
                session_id="test",
                learning_epoch=1,
                total_interactions=1,
                reward_score=0.5,
                extra_variable="extra"
            )
        except ValueError:
            pytest.fail("Extra variables should not raise ValueError")
    
    def test_template_metadata_consistency(self):
        """テンプレートメタデータ一貫性テスト"""
        manager = PromptTemplateManager()
        
        for name, template in manager.templates.items():
            # 必須メタデータの存在確認
            assert "category" in template.metadata
            assert "priority" in template.metadata
            assert "usage_count" in template.metadata
            
            # メタデータ値の妥当性確認
            assert template.metadata["priority"] in ["high", "medium", "low"]
            assert isinstance(template.metadata["usage_count"], int)
            assert template.metadata["usage_count"] >= 0


if __name__ == "__main__":
    pytest.main([__file__])
