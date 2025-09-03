"""
簡略化された FastAPI REST API

基本的な OpenAI 互換エンドポイントを提供
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

logger = logging.getLogger(__name__)


class SimpleAPIGateway:
    """簡略化された API ゲートウェイ"""
    
    def __init__(self, title: str = "Simple AI Agent API", version: str = "1.0.0"):
        self.title = title
        self.version = version
        
        # FastAPI アプリケーション作成
        self.app = FastAPI(
            title=self.title,
            version=self.version,
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # CORS 設定
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # ルート設定
        self._setup_routes()
        
        logger.info(f"Simple API Gateway 初期化完了")
    
    def _setup_routes(self):
        """ルート設定"""
        
        @self.app.get("/health")
        async def health_check():
            """ヘルスチェック"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": self.version
            }
        
        @self.app.get("/v1/models")
        async def list_models():
            """モデル一覧"""
            return {
                "object": "list",
                "data": [
                    {
                        "id": "deepseek-r1:7b",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "advanced-agent"
                    }
                ]
            }
        
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: dict):
            """チャット完了"""
            try:
                messages = request.get("messages", [])
                model = request.get("model", "deepseek-r1:7b")
                
                # 簡単なレスポンス生成
                last_message = messages[-1] if messages else {"content": ""}
                response_text = f"Mock response for: {last_message.get('content', '')[:50]}..."
                
                return {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                        "total_tokens": 30
                    }
                }
                
            except Exception as e:
                logger.error(f"チャット完了エラー: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def start_server(self, host: str = "0.0.0.0", port: int = 8000):
        """サーバー起動"""
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        
        logger.info(f"Simple API サーバー起動: http://{host}:{port}")
        logger.info(f"API ドキュメント: http://{host}:{port}/docs")
        
        await server.serve()


async def run_demo():
    """デモ実行"""
    print("🚀 Simple FastAPI Gateway デモ")
    
    gateway = SimpleAPIGateway()
    
    await gateway.start_server(host="0.0.0.0", port=8000)


if __name__ == "__main__":
    asyncio.run(run_demo())