"""
Codex Agent Integration Tests
Codex互換エージェントの統合テスト
"""

import pytest
import asyncio
import time
import os
from pathlib import Path

# テスト用のパス設定
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from codex_agent.config import CodexConfig
from codex_agent.agent_interface import CodexAgentInterface
from codex_agent.ollama_client import CodexOllamaClient
from codex_agent.compatibility_layer import CompatibilityLayer


@pytest.mark.asyncio
@pytest.mark.integration
class TestCodexAgentIntegration:
    """Codex互換エージェントの統合テスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.config = CodexConfig()
        # テスト用の設定調整
        self.config.model_max_output_tokens = 100
        self.config.ollama_timeout = 10
    
    @pytest.mark.skipif(
        not os.getenv("INTEGRATION_TEST", "").lower() == "true",
        reason="Integration tests require INTEGRATION_TEST=true environment variable"
    )
    async def test_end_to_end_completion(self):
        """エンドツーエンドのコード補完テスト"""
        agent = CodexAgentInterface(self.config)
        
        try:
            # エージェント初期化
            await agent.initialize()
            
            # コード補完テスト
            start_time = time.time()
            result = await agent.complete(
                prompt="def hello_world():",
                max_tokens=50,
                temperature=0.1
            )
            response_time = time.time() - start_time
            
            # 結果検証
            assert "choices" in result
            assert len(result["choices"]) > 0
            assert "text" in result["choices"][0]
            assert result["choices"][0]["text"].strip() != ""
            
            # パフォーマンス検証
            assert response_time < 30.0  # 30秒以内
            assert "response_time" in result
            assert result["response_time"] > 0
            
            # メタデータ検証
            assert "model_info" in result
            assert "usage" in result
            
            print(f"✓ Code completion test passed in {response_time:.2f}s")
            print(f"  Generated: {result['choices'][0]['text'][:50]}...")
            
        finally:
            await agent.shutdown()
    
    @pytest.mark.skipif(
        not os.getenv("INTEGRATION_TEST", "").lower() == "true",
        reason="Integration tests require INTEGRATION_TEST=true environment variable"
    )
    async def test_end_to_end_chat(self):
        """エンドツーエンドのチャットテスト"""
        agent = CodexAgentInterface(self.config)
        
        try:
            # エージェント初期化
            await agent.initialize()
            
            # チャットテスト
            messages = [
                {"role": "user", "content": "Hello! How are you?"}
            ]
            
            start_time = time.time()
            result = await agent.chat(
                messages=messages,
                max_tokens=100,
                temperature=0.7
            )
            response_time = time.time() - start_time
            
            # 結果検証
            assert "choices" in result
            assert len(result["choices"]) > 0
            assert "text" in result["choices"][0]
            assert result["choices"][0]["text"].strip() != ""
            
            # セッション管理検証
            assert "session_id" in result
            assert "message_count" in result
            
            # パフォーマンス検証
            assert response_time < 30.0  # 30秒以内
            
            print(f"✓ Chat test passed in {response_time:.2f}s")
            print(f"  Response: {result['choices'][0]['text'][:50]}...")
            
        finally:
            await agent.shutdown()
    
    @pytest.mark.skipif(
        not os.getenv("INTEGRATION_TEST", "").lower() == "true",
        reason="Integration tests require INTEGRATION_TEST=true environment variable"
    )
    async def test_multiple_requests_performance(self):
        """複数リクエストのパフォーマンステスト"""
        agent = CodexAgentInterface(self.config)
        
        try:
            # エージェント初期化
            await agent.initialize()
            
            # 複数のリクエストを並行実行
            prompts = [
                "def add(a, b):",
                "class Calculator:",
                "import json",
                "for i in range(10):",
                "if __name__ == '__main__':"
            ]
            
            start_time = time.time()
            
            # 並行実行
            tasks = []
            for prompt in prompts:
                task = agent.complete(
                    prompt=prompt,
                    max_tokens=30,
                    temperature=0.1
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            # 結果検証
            successful_results = [r for r in results if not isinstance(r, Exception)]
            failed_results = [r for r in results if isinstance(r, Exception)]
            
            assert len(successful_results) >= 3  # 少なくとも3つは成功
            assert len(failed_results) <= 2  # 失敗は2つまで許容
            
            # パフォーマンス検証
            assert total_time < 60.0  # 1分以内
            avg_time = total_time / len(prompts)
            assert avg_time < 15.0  # 平均15秒以内
            
            print(f"✓ Performance test passed: {len(successful_results)}/{len(prompts)} successful")
            print(f"  Total time: {total_time:.2f}s, Average: {avg_time:.2f}s")
            
        finally:
            await agent.shutdown()
    
    @pytest.mark.skipif(
        not os.getenv("INTEGRATION_TEST", "").lower() == "true",
        reason="Integration tests require INTEGRATION_TEST=true environment variable"
    )
    async def test_error_recovery(self):
        """エラー回復テスト"""
        agent = CodexAgentInterface(self.config)
        
        try:
            # エージェント初期化
            await agent.initialize()
            
            # 不正なリクエストでエラーを発生させる
            result1 = await agent.complete(
                prompt="",  # 空のプロンプト
                max_tokens=10
            )
            
            # エラーレスポンスの確認
            assert "error" in result1 or "choices" in result1
            
            # 正常なリクエストで回復を確認
            result2 = await agent.complete(
                prompt="def test():",
                max_tokens=30
            )
            
            # 正常なレスポンスの確認
            assert "choices" in result2
            assert len(result2["choices"]) > 0
            
            print("✓ Error recovery test passed")
            
        finally:
            await agent.shutdown()


@pytest.mark.asyncio
@pytest.mark.integration
class TestOllamaClientIntegration:
    """OLLAMAクライアントの統合テスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.config = CodexConfig()
    
    @pytest.mark.skipif(
        not os.getenv("INTEGRATION_TEST", "").lower() == "true",
        reason="Integration tests require INTEGRATION_TEST=true environment variable"
    )
    async def test_ollama_connection(self):
        """OLLAMA接続テスト"""
        client = CodexOllamaClient(self.config)
        
        try:
            # クライアント初期化
            await client.initialize()
            
            # ヘルスチェック
            health_result = await client.health_check()
            assert health_result["status"] == "ok"
            
            # モデル一覧取得
            models = await client.fetch_models()
            assert len(models) > 0
            
            # 設定されたモデルが利用可能か確認
            model_names = [model.name for model in models]
            assert self.config.model in model_names
            
            print(f"✓ OLLAMA connection test passed")
            print(f"  Available models: {len(models)}")
            print(f"  Current model: {self.config.model}")
            
        finally:
            await client.close()
    
    @pytest.mark.skipif(
        not os.getenv("INTEGRATION_TEST", "").lower() == "true",
        reason="Integration tests require INTEGRATION_TEST=true environment variable"
    )
    async def test_ollama_generation(self):
        """OLLAMA生成テスト"""
        client = CodexOllamaClient(self.config)
        
        try:
            # クライアント初期化
            await client.initialize()
            
            # テキスト生成
            start_time = time.time()
            response = await client.generate(
                prompt="Hello, world!",
                max_tokens=50
            )
            generation_time = time.time() - start_time
            
            # レスポンス検証
            assert hasattr(response, 'response')
            assert hasattr(response, 'done')
            assert response.response.strip() != ""
            assert response.done is True
            
            # パフォーマンス検証
            assert generation_time < 30.0  # 30秒以内
            
            print(f"✓ OLLAMA generation test passed in {generation_time:.2f}s")
            print(f"  Generated: {response.response[:50]}...")
            
        finally:
            await client.close()


@pytest.mark.integration
class TestCompatibilityLayerIntegration:
    """互換性レイヤーの統合テスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.layer = CompatibilityLayer()
    
    def test_request_response_roundtrip(self):
        """リクエスト・レスポンス変換の往復テスト"""
        # Codexリクエスト
        codex_request = {
            "prompt": "def fibonacci(n):",
            "model": "qwen2:7b-instruct",
            "max_tokens": 100,
            "temperature": 0.5,
            "stop": ["\n\n"]
        }
        
        # Codex → OLLAMA変換
        ollama_request = self.layer.translate_codex_to_ollama(codex_request)
        
        # 変換結果検証
        assert ollama_request["model"] == "qwen2:7b-instruct"
        assert ollama_request["prompt"] == "def fibonacci(n):"
        assert ollama_request["options"]["num_predict"] == 100
        assert ollama_request["options"]["temperature"] == 0.5
        assert ollama_request["options"]["stop"] == ["\n\n"]
        
        # 模擬OLLAMAレスポンス
        ollama_response = {
            "response": "    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "done": True,
            "prompt_eval_count": 20,
            "eval_count": 25
        }
        
        # OLLAMA → Codex変換
        codex_response = self.layer.translate_ollama_to_codex(
            ollama_response, codex_request
        )
        
        # 変換結果検証
        assert codex_response["object"] == "text_completion"
        assert codex_response["model"] == "qwen2:7b-instruct"
        assert len(codex_response["choices"]) == 1
        assert codex_response["choices"][0]["text"] == ollama_response["response"]
        assert codex_response["usage"]["prompt_tokens"] == 20
        assert codex_response["usage"]["completion_tokens"] == 25
        assert codex_response["usage"]["total_tokens"] == 45
        
        print("✓ Request-response roundtrip test passed")
    
    def test_chat_message_conversion(self):
        """チャットメッセージ変換テスト"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "Tell me more about it."}
        ]
        
        prompt = self.layer.translate_chat_request(messages)
        
        expected_parts = [
            "System: You are a helpful assistant.",
            "User: What is Python?",
            "Assistant: Python is a programming language.",
            "User: Tell me more about it.",
            "Assistant:"
        ]
        
        assert prompt == "\n\n".join(expected_parts)
        print("✓ Chat message conversion test passed")


if __name__ == "__main__":
    # 統合テストの実行例
    print("Running Codex Agent Integration Tests...")
    print("Set INTEGRATION_TEST=true to run actual integration tests")
    pytest.main([__file__, "-v", "-m", "integration"])