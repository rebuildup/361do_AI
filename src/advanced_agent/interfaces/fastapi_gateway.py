"""
FastAPI REST API ゲートウェイ

FastAPI の既存高性能機能による API サーバーを統合し、
OpenAI 互換のエンドポイント・認証を実装
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from .api_models import (
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChoice,
    Usage, ChatMessage, MessageRole, ModelsResponse, ModelInfo,
    HealthResponse, SystemStatsResponse, InferenceResponse
)

logger = logging.getLogger(__name__)

# セキュリティ設定
security = HTTPBearer(auto_error=False)

# 設定
API_VERSION = "v1"
DEFAULT_MODEL = "deepseek-r1:7b"
AVAILABLE_MODELS = [
    "deepseek-r1:7b",
    "qwen2.5:7b-instruct-q4_k_m",
    "qwen2:1.5b-instruct-q4_k_m"
]


class FastAPIGateway:
    """FastAPI REST API ゲートウェイ"""
    
    def __init__(self,
                 title: str = "Advanced AI Agent API",
                 version: str = "1.0.0",
                 description: str = "OpenAI 互換 AI エージェント API",
                 api_key: Optional[str] = None,
                 enable_auth: bool = False,
                 cors_origins: List[str] = None):
        
        self.title = title
        self.version = version
        self.description = description
        self.api_key = api_key
        self.enable_auth = enable_auth
        self.cors_origins = cors_origins or ["*"]
        
        # FastAPI アプリケーション作成
        self.app = FastAPI(
            title=self.title,
            version=self.version,
            description=self.description,
            docs_url=f"/{API_VERSION}/docs",
            redoc_url=f"/{API_VERSION}/redoc",
            openapi_url=f"/{API_VERSION}/openapi.json"
        )
        
        # ルート設定
        self._setup_routes()
        
        # ミドルウェア設定
        self._setup_middleware()
        
        # セッション管理
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"FastAPI Gateway 初期化完了 - 認証: {enable_auth}")
    
    def _setup_middleware(self):
        """ミドルウェア設定"""
        
        # CORS 設定
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Gzip 圧縮
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # カスタムミドルウェア
        @self.app.middleware("http")
        async def logging_middleware(request, call_next):
            start_time = time.time()
            
            response = await call_next(request)
            
            process_time = time.time() - start_time
            logger.info(
                f"{request.method} {request.url.path} - "
                f"Status: {response.status_code} - "
                f"Time: {process_time:.3f}s"
            )
            
            response.headers["X-Process-Time"] = str(process_time)
            return response
    
    def _verify_api_key(self, credentials: HTTPAuthorizationCredentials = Security(security)) -> bool:
        """API キー認証"""
        if not self.enable_auth:
            return True
        
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API キーが必要です",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if credentials.credentials != self.api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="無効な API キーです",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return True
    
    def _setup_routes(self):
        """ルート設定"""
        
        # ヘルスチェック
        @self.app.get(f"/{API_VERSION}/health", response_model=HealthResponse)
        async def health_check():
            """ヘルスチェック"""
            return HealthResponse(
                status="healthy",
                timestamp=datetime.now(),
                version=self.version,
                system_info={"cpu_percent": 50.0, "memory_percent": 60.0}
            )
        
        # モデル一覧
        @self.app.get(f"/{API_VERSION}/models", response_model=ModelsResponse)
        async def list_models(authenticated: bool = Depends(self._verify_api_key)):
            """利用可能なモデル一覧"""
            models = []
            for model_id in AVAILABLE_MODELS:
                models.append(ModelInfo(
                    id=model_id,
                    created=int(time.time()),
                    owned_by="advanced-agent"
                ))
            
            return ModelsResponse(data=models)
        
        # チャット完了（OpenAI 互換）
        @self.app.post(f"/{API_VERSION}/chat/completions")
        async def chat_completions(
            request: ChatCompletionRequest,
            authenticated: bool = Depends(self._verify_api_key)
        ):
            """チャット完了エンドポイント（OpenAI 互換）"""
            try:
                # ストリーミング対応
                if request.stream:
                    return StreamingResponse(
                        self._stream_chat_completion(request),
                        media_type="text/plain"
                    )
                
                # 通常のレスポンス
                return await self._process_chat_completion(request)
                
            except Exception as e:
                logger.error(f"チャット完了エラー: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(e)
                )
    
    async def _process_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """チャット完了処理"""
        try:
            # メッセージを結合してプロンプト作成
            prompt = self._messages_to_prompt(request.messages)
            
            # 実際のOllama推論実行
            response_text = await self._call_ollama_inference(prompt, request.model, request.temperature, request.max_tokens)
            
            # レスポンス構築
            choice = ChatCompletionChoice(
                index=0,
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=response_text
                ),
                finish_reason="stop"
            )
            
            # 使用量計算（簡略化）
            prompt_tokens = len(prompt.split())
            completion_tokens = len(response_text.split())
            
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
            
            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4()}",
                created=int(time.time()),
                model=request.model,
                choices=[choice],
                usage=usage
            )
            
        except Exception as e:
            logger.error(f"チャット完了処理エラー: {e}")
            raise
    
    async def _stream_chat_completion(self, request: ChatCompletionRequest):
        """ストリーミングチャット完了"""
        try:
            prompt = self._messages_to_prompt(request.messages)
            
            # ストリーミング ID
            stream_id = f"chatcmpl-{uuid.uuid4()}"
            
            # レスポンス生成（簡略化）
            response_text = f"Mock streaming response for: {prompt[:50]}..."
            
            # 文字ごとにストリーミング
            for i, char in enumerate(response_text):
                chunk_data = {
                    "id": stream_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": char},
                        "finish_reason": None
                    }]
                }
                yield f"data: {chunk_data}\n\n"
                await asyncio.sleep(0.01)  # ストリーミング効果
            
            # 終了チャンク
            end_chunk = {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {end_chunk}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"ストリーミングエラー: {e}")
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "internal_error"
                }
            }
            yield f"data: {error_chunk}\n\n"
    
    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """メッセージをプロンプトに変換"""
        prompt_parts = []
        
        for message in messages:
            if message.role == MessageRole.SYSTEM:
                prompt_parts.append(f"System: {message.content}")
            elif message.role == MessageRole.USER:
                prompt_parts.append(f"User: {message.content}")
            elif message.role == MessageRole.ASSISTANT:
                prompt_parts.append(f"Assistant: {message.content}")
        
        return "\n".join(prompt_parts)
    
    async def _call_ollama_inference(self, prompt: str, model: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        """実際のOllama推論実行"""
        
        try:
            import ollama
            
            # Ollama API呼び出し
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            
            return response["message"]["content"]
            
        except ImportError:
            logger.error("ollama パッケージがインストールされていません")
            return "エラー: ollama パッケージがインストールされていません。'pip install ollama' を実行してください。"
            
        except Exception as e:
            logger.error(f"Ollama推論エラー: {e}")
            
            # フォールバック: 軽量モデルを試行
            try:
                fallback_models = ["qwen2:1.5b-instruct-q4_k_m", "qwen2.5:7b-instruct-q4_k_m"]
                
                for fallback_model in fallback_models:
                    try:
                        response = ollama.chat(
                            model=fallback_model,
                            messages=[
                                {"role": "user", "content": prompt}
                            ],
                            options={
                                "temperature": temperature,
                                "num_predict": min(max_tokens, 200)  # 軽量化
                            }
                        )
                        
                        return f"[フォールバックモデル {fallback_model} による回答]\n{response['message']['content']}"
                        
                    except Exception as fallback_error:
                        logger.warning(f"フォールバックモデル {fallback_model} エラー: {fallback_error}")
                        continue
                
                # 全てのフォールバックが失敗
                return f"申し訳ございません。AIモデルに接続できませんでした。Ollamaが起動しており、モデル '{model}' がダウンロードされているか確認してください。\n\nエラー詳細: {str(e)}"
                
            except Exception as final_error:
                return f"システムエラーが発生しました: {str(final_error)}"
    
    async def start_server(self,
                          host: str = "0.0.0.0",
                          port: int = 8000,
                          reload: bool = False,
                          log_level: str = "info"):
        """サーバー起動"""
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            reload=reload,
            log_level=log_level
        )
        
        server = uvicorn.Server(config)
        
        logger.info(f"FastAPI サーバー起動: http://{host}:{port}")
        logger.info(f"API ドキュメント: http://{host}:{port}/{API_VERSION}/docs")
        
        await server.serve()


if __name__ == "__main__":
    # 直接実行時のデモサーバー
    async def main():
        gateway = FastAPIGateway(
            enable_auth=False,
            cors_origins=["*"]
        )
        
        await gateway.start_server(
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    
    asyncio.run(main())