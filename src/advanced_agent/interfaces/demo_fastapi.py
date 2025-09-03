"""
FastAPI Gateway デモ

OpenAI 互換 API サーバーのデモンストレーション
"""

import asyncio
import logging
import json
import aiohttp
from datetime import datetime
from typing import Dict, Any

from .fastapi_gateway import FastAPIGateway
from .api_models import (
    ChatCompletionRequest, ChatMessage, MessageRole,
    InferenceRequest, MemorySearchRequest, SessionRequest
)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FastAPIDemo:
    """FastAPI Gateway デモ"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_health_check(self) -> Dict[str, Any]:
        """ヘルスチェックテスト"""
        logger.info("🏥 ヘルスチェック実行中...")
        
        async with self.session.get(f"{self.base_url}/v1/health") as response:
            data = await response.json()
            
            if response.status == 200:
                logger.info(f"✅ ヘルスチェック成功 - ステータス: {data['status']}")
                logger.info(f"   バージョン: {data['version']}")
                logger.info(f"   タイムスタンプ: {data['timestamp']}")
            else:
                logger.error(f"❌ ヘルスチェック失敗 - ステータス: {response.status}")
            
            return data
    
    async def test_models_list(self) -> Dict[str, Any]:
        """モデル一覧テスト"""
        logger.info("📋 モデル一覧取得中...")
        
        async with self.session.get(f"{self.base_url}/v1/models") as response:
            data = await response.json()
            
            if response.status == 200:
                logger.info(f"✅ モデル一覧取得成功 - {len(data['data'])} モデル")
                for model in data['data']:
                    logger.info(f"   - {model['id']} (所有者: {model['owned_by']})")
            else:
                logger.error(f"❌ モデル一覧取得失敗 - ステータス: {response.status}")
            
            return data
    
    async def test_chat_completion(self) -> Dict[str, Any]:
        """チャット完了テスト"""
        logger.info("💬 チャット完了テスト実行中...")
        
        request_data = {
            "model": "deepseek-r1:7b",
            "messages": [
                {"role": "system", "content": "あなたは親切なAIアシスタントです。"},
                {"role": "user", "content": "人工知能について簡潔に説明してください。"}
            ],
            "temperature": 0.7,
            "max_tokens": 200
        }
        
        async with self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=request_data
        ) as response:
            data = await response.json()
            
            if response.status == 200:
                logger.info("✅ チャット完了成功")
                logger.info(f"   モデル: {data['model']}")
                logger.info(f"   レスポンス: {data['choices'][0]['message']['content'][:100]}...")
                logger.info(f"   使用トークン: {data['usage']['total_tokens']}")
            else:
                logger.error(f"❌ チャット完了失敗 - ステータス: {response.status}")
            
            return data
    
    async def test_streaming_chat(self) -> None:
        """ストリーミングチャットテスト"""
        logger.info("🌊 ストリーミングチャットテスト実行中...")
        
        request_data = {
            "model": "deepseek-r1:7b",
            "messages": [
                {"role": "user", "content": "1から5まで数えてください。"}
            ],
            "stream": True
        }
        
        async with self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=request_data
        ) as response:
            
            if response.status == 200:
                logger.info("✅ ストリーミング開始")
                
                full_response = ""
                async for line in response.content:
                    line_text = line.decode('utf-8').strip()
                    
                    if line_text.startswith('data: '):
                        data_text = line_text[6:]  # 'data: ' を除去
                        
                        if data_text == '[DONE]':
                            logger.info("✅ ストリーミング完了")
                            break
                        
                        try:
                            chunk_data = json.loads(data_text)
                            if 'choices' in chunk_data and chunk_data['choices']:
                                delta = chunk_data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    content = delta['content']
                                    full_response += content
                                    print(content, end='', flush=True)
                        except json.JSONDecodeError:
                            continue
                
                print()  # 改行
                logger.info(f"   完全なレスポンス: {full_response}")
            else:
                logger.error(f"❌ ストリーミング失敗 - ステータス: {response.status}")
    
    async def test_inference_endpoint(self) -> Dict[str, Any]:
        """推論エンドポイントテスト"""
        logger.info("🧠 推論エンドポイントテスト実行中...")
        
        request_data = {
            "prompt": "量子コンピューターの基本原理を説明してください。",
            "model": "deepseek-r1:7b",
            "temperature": 0.5,
            "use_cot": True
        }
        
        async with self.session.post(
            f"{self.base_url}/v1/inference",
            json=request_data
        ) as response:
            data = await response.json()
            
            if response.status == 200:
                logger.info("✅ 推論成功")
                logger.info(f"   処理時間: {data['processing_time']:.3f}秒")
                logger.info(f"   レスポンス: {data['response'][:100]}...")
                if data.get('confidence_score'):
                    logger.info(f"   信頼度: {data['confidence_score']:.3f}")
            else:
                logger.error(f"❌ 推論失敗 - ステータス: {response.status}")
            
            return data
    
    async def test_memory_search(self) -> Dict[str, Any]:
        """記憶検索テスト"""
        logger.info("🔍 記憶検索テスト実行中...")
        
        request_data = {
            "query": "人工知能 機械学習",
            "max_results": 5,
            "similarity_threshold": 0.7
        }
        
        async with self.session.post(
            f"{self.base_url}/v1/memory/search",
            json=request_data
        ) as response:
            data = await response.json()
            
            if response.status == 200:
                logger.info("✅ 記憶検索成功")
                logger.info(f"   検索時間: {data['search_time']:.3f}秒")
                logger.info(f"   見つかった結果: {data['total_found']}件")
                logger.info(f"   返された結果: {len(data['results'])}件")
            else:
                logger.error(f"❌ 記憶検索失敗 - ステータス: {response.status}")
            
            return data
    
    async def test_session_management(self) -> Dict[str, Any]:
        """セッション管理テスト"""
        logger.info("👤 セッション管理テスト実行中...")
        
        # セッション作成
        create_data = {
            "user_id": "demo_user",
            "session_name": "Demo Session",
            "metadata": {
                "demo": True,
                "created_by": "fastapi_demo",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        async with self.session.post(
            f"{self.base_url}/v1/sessions",
            json=create_data
        ) as response:
            session_data = await response.json()
            
            if response.status == 200:
                logger.info("✅ セッション作成成功")
                session_id = session_data['session_id']
                logger.info(f"   セッションID: {session_id}")
                logger.info(f"   セッション名: {session_data['session_name']}")
                
                # セッション取得テスト
                async with self.session.get(
                    f"{self.base_url}/v1/sessions/{session_id}"
                ) as get_response:
                    get_data = await get_response.json()
                    
                    if get_response.status == 200:
                        logger.info("✅ セッション取得成功")
                        logger.info(f"   ユーザーID: {get_data['user_id']}")
                        logger.info(f"   作成日時: {get_data['created_at']}")
                    else:
                        logger.error(f"❌ セッション取得失敗 - ステータス: {get_response.status}")
            else:
                logger.error(f"❌ セッション作成失敗 - ステータス: {response.status}")
            
            return session_data
    
    async def test_system_stats(self) -> Dict[str, Any]:
        """システム統計テスト"""
        logger.info("📊 システム統計テスト実行中...")
        
        request_data = {
            "include_gpu": True,
            "include_memory": True,
            "include_processes": False
        }
        
        async with self.session.post(
            f"{self.base_url}/v1/system/stats",
            json=request_data
        ) as response:
            data = await response.json()
            
            if response.status == 200:
                logger.info("✅ システム統計取得成功")
                logger.info(f"   CPU使用率: {data['cpu'].get('percent', 'N/A')}%")
                logger.info(f"   メモリ使用率: {data['memory'].get('percent', 'N/A')}%")
                if data.get('gpu'):
                    logger.info(f"   GPU情報: 利用可能")
                logger.info(f"   タイムスタンプ: {data['timestamp']}")
            else:
                logger.error(f"❌ システム統計取得失敗 - ステータス: {response.status}")
            
            return data
    
    async def run_all_tests(self):
        """全テスト実行"""
        logger.info("🚀 FastAPI Gateway 全機能テスト開始")
        logger.info("=" * 60)
        
        tests = [
            ("ヘルスチェック", self.test_health_check),
            ("モデル一覧", self.test_models_list),
            ("チャット完了", self.test_chat_completion),
            ("ストリーミングチャット", self.test_streaming_chat),
            ("推論エンドポイント", self.test_inference_endpoint),
            ("記憶検索", self.test_memory_search),
            ("セッション管理", self.test_session_management),
            ("システム統計", self.test_system_stats)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            try:
                logger.info(f"\n--- {test_name} ---")
                result = await test_func()
                results[test_name] = {"status": "success", "data": result}
                
            except Exception as e:
                logger.error(f"❌ {test_name} でエラー: {e}")
                results[test_name] = {"status": "error", "error": str(e)}
        
        # 結果サマリー
        logger.info("\n" + "=" * 60)
        logger.info("📋 テスト結果サマリー")
        logger.info("=" * 60)
        
        success_count = 0
        for test_name, result in results.items():
            status_icon = "✅" if result["status"] == "success" else "❌"
            logger.info(f"{status_icon} {test_name}: {result['status']}")
            if result["status"] == "success":
                success_count += 1
        
        logger.info(f"\n成功: {success_count}/{len(tests)} テスト")
        logger.info("🎉 テスト完了")
        
        return results


async def run_server_demo():
    """サーバーデモ実行"""
    logger.info("🚀 FastAPI Gateway サーバーデモ開始")
    
    # ゲートウェイ作成
    gateway = FastAPIGateway(
        title="Advanced AI Agent API - Demo",
        version="1.0.0-demo",
        description="デモ用 OpenAI 互換 AI エージェント API",
        enable_auth=False,  # デモでは認証無効
        cors_origins=["*"]
    )
    
    logger.info("サーバー設定:")
    logger.info(f"  - タイトル: {gateway.title}")
    logger.info(f"  - バージョン: {gateway.version}")
    logger.info(f"  - 認証: {'有効' if gateway.enable_auth else '無効'}")
    logger.info(f"  - CORS: {gateway.cors_origins}")
    
    # サーバー起動
    await gateway.start_server(
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


async def run_client_demo():
    """クライアントデモ実行"""
    logger.info("🧪 FastAPI Gateway クライアントデモ開始")
    
    async with FastAPIDemo() as demo:
        await demo.run_all_tests()


async def run_interactive_demo():
    """インタラクティブデモ"""
    print("🚀 FastAPI Gateway インタラクティブデモ")
    print("=" * 60)
    print("1. サーバー起動")
    print("2. クライアントテスト実行")
    print("3. 両方実行（別ターミナルでサーバー起動が必要）")
    
    choice = input("\n選択してください (1-3): ").strip()
    
    if choice == "1":
        await run_server_demo()
    elif choice == "2":
        await run_client_demo()
    elif choice == "3":
        print("\n注意: サーバーを別ターミナルで起動してください")
        print("コマンド: python -m src.advanced_agent.interfaces.demo_fastapi server")
        input("サーバー起動後、Enter を押してください...")
        await run_client_demo()
    else:
        print("無効な選択です")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "server":
            asyncio.run(run_server_demo())
        elif sys.argv[1] == "client":
            asyncio.run(run_client_demo())
        else:
            print("使用法: python demo_fastapi.py [server|client]")
    else:
        asyncio.run(run_interactive_demo())