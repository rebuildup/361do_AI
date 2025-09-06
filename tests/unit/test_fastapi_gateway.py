"""
FastAPI Gateway のテスト
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient

from src.advanced_agent.interfaces.fastapi_gateway import FastAPIGateway
from src.advanced_agent.interfaces.api_models import (
    ChatCompletionRequest, ChatMessage, MessageRole,
    InferenceRequest, MemorySearchRequest, SessionRequest
)


class TestFastAPIGateway:
    """FastAPI Gateway のテスト"""
    
    @pytest.fixture
    def gateway(self):
        """テスト用ゲートウェイ"""
        return FastAPIGateway(
            title="Test API",
            version="1.0.0-test",
            enable_auth=False,  # テストでは認証無効
            cors_origins=["*"]
        )
    
    @pytest.fixture
    def client(self, gateway):
        """テスト用クライアント"""
        return TestClient(gateway.app)
    
    def test_health_check(self, client):
        """ヘルスチェックテスト"""
        response = client.get("/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] in ["healthy", "unhealthy"]
        assert "timestamp" in data
        assert "version" in data
        assert "system_info" in data
    
    def test_list_models(self, client):
        """モデル一覧テスト"""
        response = client.get("/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["object"] == "list"
        assert "data" in data
        assert len(data["data"]) > 0
        
        # 最初のモデル情報チェック
        model = data["data"][0]
        assert "id" in model
        assert "object" in model
        assert "created" in model
        assert "owned_by" in model
    
    def test_chat_completions(self, client):
        """チャット完了テスト"""
        request_data = {
            "model": "deepseek-r1:7b",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        response = client.post("/v1/chat/completions", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "id" in data
        assert data["object"] == "chat.completion"
        assert "created" in data
        assert data["model"] == "deepseek-r1:7b"
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "usage" in data
        
        # 選択肢チェック
        choice = data["choices"][0]
        assert choice["index"] == 0
        assert "message" in choice
        assert choice["message"]["role"] == "assistant"
        assert "content" in choice["message"]
    
    def test_chat_completions_streaming(self, client):
        """ストリーミングチャット完了テスト"""
        request_data = {
            "model": "deepseek-r1:7b",
            "messages": [
                {"role": "user", "content": "Count to 3"}
            ],
            "stream": True
        }
        
        response = client.post("/v1/chat/completions", json=request_data)
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        
        # ストリーミングデータの確認
        content = response.text
        assert "data:" in content
        assert "[DONE]" in content
    
    def test_inference_endpoint(self, client):
        """推論エンドポイントテスト"""
        request_data = {
            "prompt": "What is artificial intelligence?",
            "model": "deepseek-r1:7b",
            "temperature": 0.5,
            "use_cot": True
        }
        
        response = client.post("/v1/inference", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "id" in data
        assert "response" in data
        assert "processing_time" in data
        assert "memory_usage" in data
        assert "model_info" in data
        assert data["model_info"]["model"] == "deepseek-r1:7b"
    
    def test_memory_search(self, client):
        """記憶検索テスト"""
        request_data = {
            "query": "artificial intelligence",
            "max_results": 5,
            "similarity_threshold": 0.7
        }
        
        response = client.post("/v1/memory/search", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        assert "total_found" in data
        assert "search_time" in data
        assert isinstance(data["results"], list)
        assert isinstance(data["total_found"], int)
        assert isinstance(data["search_time"], float)
    
    def test_session_management(self, client):
        """セッション管理テスト"""
        # セッション作成
        create_data = {
            "user_id": "test_user",
            "session_name": "Test Session",
            "metadata": {"test": True}
        }
        
        response = client.post("/v1/sessions", json=create_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "session_id" in data
        assert data["user_id"] == "test_user"
        assert data["session_name"] == "Test Session"
        assert data["metadata"]["test"] is True
        
        session_id = data["session_id"]
        
        # セッション取得
        response = client.get(f"/v1/sessions/{session_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["session_id"] == session_id
        assert data["user_id"] == "test_user"
    
    def test_system_stats(self, client):
        """システム統計テスト"""
        request_data = {
            "include_gpu": True,
            "include_memory": True,
            "include_processes": False
        }
        
        response = client.post("/v1/system/stats", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "timestamp" in data
        assert "cpu" in data
        assert "memory" in data
        # GPU は環境によって異なる可能性があるため、存在チェックのみ
        
    def test_invalid_session(self, client):
        """無効なセッションテスト"""
        response = client.get("/v1/sessions/invalid-session-id")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
    
    def test_openapi_schema(self, gateway):
        """OpenAPI スキーマテスト"""
        schema = gateway.get_openapi_schema()
        
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        assert schema["info"]["title"] == "Test API"
        assert schema["info"]["version"] == "1.0.0-test"
    
    @pytest.mark.asyncio
    async def test_messages_to_prompt(self, gateway):
        """メッセージ→プロンプト変換テスト"""
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            ChatMessage(role=MessageRole.USER, content="Hello!"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!")
        ]
        
        prompt = gateway._messages_to_prompt(messages)
        
        assert "System: You are a helpful assistant." in prompt
        assert "User: Hello!" in prompt
        assert "Assistant: Hi there!" in prompt
    
    def test_cors_headers(self, client):
        """CORS ヘッダーテスト"""
        # Originヘッダーを付けてリクエスト
        response = client.get("/v1/health", headers={"Origin": "http://localhost:3000"})
        
        # CORS ヘッダーが設定されていることを確認
        assert "access-control-allow-origin" in response.headers
    
    def test_process_time_header(self, client):
        """処理時間ヘッダーテスト"""
        response = client.get("/v1/health")
        
        assert "x-process-time" in response.headers
        process_time = float(response.headers["x-process-time"])
        assert process_time >= 0


class TestFastAPIGatewayWithAuth:
    """認証有効時の FastAPI Gateway テスト"""
    
    @pytest.fixture
    def auth_gateway(self):
        """認証有効なテスト用ゲートウェイ"""
        return FastAPIGateway(
            title="Test API with Auth",
            version="1.0.0-test",
            enable_auth=True,
            api_key="test-api-key-12345",
            cors_origins=["*"]
        )
    
    @pytest.fixture
    def auth_client(self, auth_gateway):
        """認証有効なテスト用クライアント"""
        return TestClient(auth_gateway.app)
    
    def test_unauthorized_access(self, auth_client):
        """認証なしアクセステスト"""
        response = auth_client.get("/v1/models")
        
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data
    
    def test_invalid_api_key(self, auth_client):
        """無効な API キーテスト"""
        headers = {"Authorization": "Bearer invalid-key"}
        response = auth_client.get("/v1/models", headers=headers)
        
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data
    
    def test_valid_api_key(self, auth_client):
        """有効な API キーテスト"""
        headers = {"Authorization": "Bearer test-api-key-12345"}
        response = auth_client.get("/v1/models", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data


@pytest.mark.asyncio
async def test_gateway_lifecycle():
    """ゲートウェイライフサイクルテスト"""
    gateway = FastAPIGateway(enable_auth=False)
    
    # 起動処理テスト
    await gateway._startup()
    
    # コンポーネントが初期化されていることを確認
    assert gateway.app is not None
    assert gateway.active_sessions is not None
    assert gateway.enable_auth is False
    
    # 終了処理テスト
    await gateway._shutdown()


if __name__ == "__main__":
    pytest.main([__file__])