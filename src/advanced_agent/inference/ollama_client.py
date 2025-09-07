"""
LangChain + Ollama 統合クライアント
DeepSeek-R1 モデル通信とフォールバック機能
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import ollama
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from ..core.config import get_config
from ..core.logger import get_logger


class ModelStatus(Enum):
    """モデル状態"""
    AVAILABLE = "available"
    LOADING = "loading"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


@dataclass
class ModelInfo:
    """モデル情報"""
    name: str
    size_gb: float
    status: ModelStatus
    last_used: Optional[datetime] = None
    error_message: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceRequest:
    """推論リクエスト"""
    prompt: str
    model_name: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    stream: bool = False
    system_message: Optional[str] = None
    context: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'prompt': self.prompt,
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'stream': self.stream,
            'system_message': self.system_message,
            'context': self.context,
            'metadata': self.metadata
        }


@dataclass
class InferenceResponse:
    """推論レスポンス"""
    content: str
    model_used: str
    processing_time: float
    token_count: Optional[int] = None
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class OllamaCallbackHandler(BaseCallbackHandler):
    """Ollama 専用コールバックハンドラー"""
    
    def __init__(self):
        self.start_time = None
        self.tokens_generated = 0
        self.logger = get_logger()
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """LLM開始時"""
        self.start_time = time.time()
        self.tokens_generated = 0
        
        model_name = serialized.get("name", "unknown")
        prompt_length = len(prompts[0]) if prompts else 0
        
        self.logger.log_inference_start(
            model_name=model_name,
            prompt_length=prompt_length,
            context_length=prompt_length
        )
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """LLM終了時"""
        if self.start_time:
            processing_time = time.time() - self.start_time
            
            # レスポンス長計算
            response_length = 0
            if response.generations:
                for generation in response.generations:
                    for gen in generation:
                        response_length += len(gen.text)
            
            self.logger.log_inference_complete(
                model_name="ollama",
                response_length=response_length,
                processing_time=processing_time,
                memory_used_mb=0  # TODO: 実際のメモリ使用量取得
            )
    
    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        """LLMエラー時"""
        self.logger.log_inference_error(
            model_name="ollama",
            error=error,
            fallback_used=False
        )


class OllamaClient:
    """LangChain + Ollama 統合クライアント"""
    
    def __init__(self, base_url: Optional[str] = None):
        self.config = get_config()
        self.logger = get_logger()
        
        self.base_url = base_url or self.config.models.ollama_base_url
        self.primary_model = self.config.models.primary
        self.fallback_model = self.config.models.fallback
        self.emergency_model = self.config.models.emergency
        
        # Ollama クライアント初期化
        self.ollama_client = ollama.Client(host=self.base_url)
        
        # LangChain Ollama LLM 初期化
        self.primary_llm = None
        self.fallback_llm = None
        self.emergency_llm = None
        
        # モデル情報キャッシュ
        self.model_cache: Dict[str, ModelInfo] = {}
        
        # コールバックハンドラー
        self.callback_handler = OllamaCallbackHandler()
        
        self.logger.log_startup(
            component="ollama_client",
            version="1.0.0",
            config_summary={
                "base_url": self.base_url,
                "primary_model": self.primary_model,
                "fallback_model": self.fallback_model
            }
        )
    
    async def initialize(self) -> bool:
        """クライアント初期化"""
        try:
            # サーバー接続確認
            if not await self._check_server_connection():
                raise ConnectionError(f"Cannot connect to Ollama server at {self.base_url}")
            
            # 利用可能モデル取得
            await self._refresh_model_list()
            
            # LangChain LLM インスタンス作成
            await self._initialize_llms()
            
            self.logger.log_startup(
                component="ollama_client_initialized",
                version="1.0.0",
                config_summary={
                    "available_models": len(self.model_cache),
                    "primary_available": self.primary_model in self.model_cache,
                    "fallback_available": self.fallback_model in self.model_cache
                }
            )
            
            return True
            
        except Exception as e:
            self.logger.log_inference_error(
                model_name="initialization",
                error=e,
                fallback_used=False
            )
            return False
    
    async def _check_server_connection(self) -> bool:
        """サーバー接続確認"""
        try:
            # 非同期でOllamaサーバーに接続テスト
            models = await asyncio.to_thread(self.ollama_client.list)
            return True
        except Exception as e:
            self.logger.log_alert(
                alert_type="ollama_connection_failed",
                severity="ERROR",
                message=f"Failed to connect to Ollama server: {e}"
            )
            return False
    
    async def _refresh_model_list(self) -> None:
        """モデルリスト更新"""
        try:
            models_response = await asyncio.to_thread(self.ollama_client.list)
            
            self.model_cache.clear()
            
            for model in models_response.get('models', []):
                model_name = model['name']
                size_bytes = model.get('size', 0)
                size_gb = size_bytes / (1024**3) if size_bytes else 0
                
                self.model_cache[model_name] = ModelInfo(
                    name=model_name,
                    size_gb=size_gb,
                    status=ModelStatus.AVAILABLE,
                    parameters=model.get('details', {})
                )
            
            self.logger.log_system_stats({
                "available_models": len(self.model_cache),
                "total_size_gb": sum(model.size_gb for model in self.model_cache.values())
            })
            
        except Exception as e:
            self.logger.log_inference_error(
                model_name="model_list_refresh",
                error=e,
                fallback_used=False
            )
    
    async def _initialize_llms(self) -> None:
        """LangChain LLM インスタンス初期化"""
        try:
            # プライマリモデル
            if self.primary_model in self.model_cache:
                self.primary_llm = Ollama(
                    model=self.primary_model,
                    base_url=self.base_url,
                    temperature=0.1,
                    callbacks=[self.callback_handler]
                )
            
            # フォールバックモデル
            if self.fallback_model in self.model_cache:
                self.fallback_llm = Ollama(
                    model=self.fallback_model,
                    base_url=self.base_url,
                    temperature=0.1,
                    callbacks=[self.callback_handler]
                )
            
            # 緊急時モデル
            if self.emergency_model in self.model_cache:
                self.emergency_llm = Ollama(
                    model=self.emergency_model,
                    base_url=self.base_url,
                    temperature=0.1,
                    callbacks=[self.callback_handler]
                )
            
        except Exception as e:
            self.logger.log_inference_error(
                model_name="llm_initialization",
                error=e,
                fallback_used=False
            )
    
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """テキスト生成"""
        start_time = time.time()
        
        # モデル選択
        model_name = request.model_name or self.primary_model
        llm = await self._select_llm(model_name)
        
        if not llm:
            raise ValueError(f"No available LLM for model: {model_name}")
        
        try:
            # プロンプト構築
            prompt = await self._build_prompt(request)
            
            # 推論実行
            if request.stream:
                return await self._generate_stream(llm, prompt, model_name, start_time)
            else:
                return await self._generate_sync(llm, prompt, model_name, start_time)
                
        except Exception as e:
            # フォールバック試行
            return await self._try_fallback(request, e, start_time)
    
    async def _select_llm(self, model_name: str) -> Optional[Ollama]:
        """LLM選択"""
        if model_name == self.primary_model and self.primary_llm:
            return self.primary_llm
        elif model_name == self.fallback_model and self.fallback_llm:
            return self.fallback_llm
        elif model_name == self.emergency_model and self.emergency_llm:
            return self.emergency_llm
        else:
            # 動的にLLM作成
            if model_name in self.model_cache:
                return Ollama(
                    model=model_name,
                    base_url=self.base_url,
                    temperature=0.1,
                    callbacks=[self.callback_handler]
                )
        return None
    
    async def _build_prompt(self, request: InferenceRequest) -> str:
        """プロンプト構築"""
        prompt_parts = []
        
        # システムメッセージ
        if request.system_message:
            prompt_parts.append(f"System: {request.system_message}")
        
        # コンテキスト
        if request.context:
            prompt_parts.append("Context:")
            for ctx in request.context:
                prompt_parts.append(f"- {ctx}")
        
        # メインプロンプト
        prompt_parts.append(f"Human: {request.prompt}")
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
    
    async def _generate_sync(self, llm: Ollama, prompt: str, model_name: str, start_time: float) -> InferenceResponse:
        """同期生成"""
        try:
            response = await asyncio.to_thread(llm.invoke, prompt)
            
            processing_time = time.time() - start_time
            
            return InferenceResponse(
                content=response,
                model_used=model_name,
                processing_time=processing_time,
                token_count=len(response.split()),  # 簡易トークン数
                finish_reason="stop"
            )
            
        except Exception as e:
            raise e
    
    async def _generate_stream(self, llm: Ollama, prompt: str, model_name: str, start_time: float) -> InferenceResponse:
        """ストリーミング生成"""
        try:
            # LangChainのストリーミングは複雑なので、直接Ollamaクライアントを使用
            response_parts = []
            
            stream = await asyncio.to_thread(
                self.ollama_client.generate,
                model=model_name,
                prompt=prompt,
                stream=True
            )
            
            for chunk in stream:
                if 'response' in chunk:
                    response_parts.append(chunk['response'])
            
            full_response = ''.join(response_parts)
            processing_time = time.time() - start_time
            
            return InferenceResponse(
                content=full_response,
                model_used=model_name,
                processing_time=processing_time,
                token_count=len(full_response.split()),
                finish_reason="stop"
            )
            
        except Exception as e:
            raise e
    
    async def _try_fallback(self, request: InferenceRequest, original_error: Exception, start_time: float) -> InferenceResponse:
        """フォールバック試行"""
        self.logger.log_inference_error(
            model_name=request.model_name or self.primary_model,
            error=original_error,
            fallback_used=True
        )
        
        # フォールバック順序
        fallback_models = [self.fallback_model, self.emergency_model]
        
        for fallback_model in fallback_models:
            if fallback_model in self.model_cache:
                try:
                    fallback_request = InferenceRequest(
                        prompt=request.prompt,
                        model_name=fallback_model,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        stream=False,  # フォールバック時はストリーミング無効
                        system_message=request.system_message,
                        context=request.context
                    )
                    
                    llm = await self._select_llm(fallback_model)
                    if llm:
                        prompt = await self._build_prompt(fallback_request)
                        response = await self._generate_sync(llm, prompt, fallback_model, start_time)
                        
                        # フォールバック使用をメタデータに記録
                        response.metadata["fallback_used"] = True
                        response.metadata["original_error"] = str(original_error)
                        
                        return response
                        
                except Exception as fallback_error:
                    self.logger.log_inference_error(
                        model_name=fallback_model,
                        error=fallback_error,
                        fallback_used=True
                    )
                    continue
        
        # 全てのフォールバックが失敗
        raise Exception(f"All models failed. Original error: {original_error}")
    
    async def chat(self, messages: List[Dict[str, str]], model_name: Optional[str] = None) -> InferenceResponse:
        """チャット形式での対話"""
        # メッセージを単一プロンプトに変換
        prompt_parts = []
        system_message = None
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                system_message = content
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt = "\n\n".join(prompt_parts)
        
        request = InferenceRequest(
            prompt=prompt,
            model_name=model_name,
            system_message=system_message
        )
        
        return await self.generate(request)
    
    async def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """モデル情報取得"""
        return self.model_cache.get(model_name)
    
    async def list_models(self) -> List[ModelInfo]:
        """利用可能モデル一覧"""
        await self._refresh_model_list()
        return list(self.model_cache.values())
    
    async def pull_model(self, model_name: str) -> bool:
        """モデルダウンロード"""
        try:
            self.logger.log_alert(
                alert_type="model_download_start",
                severity="INFO",
                message=f"Starting download of model: {model_name}"
            )
            
            # モデルプル実行
            await asyncio.to_thread(self.ollama_client.pull, model_name)
            
            # モデルリスト更新
            await self._refresh_model_list()
            
            self.logger.log_alert(
                alert_type="model_download_complete",
                severity="INFO",
                message=f"Successfully downloaded model: {model_name}"
            )
            
            return True
            
        except Exception as e:
            self.logger.log_alert(
                alert_type="model_download_failed",
                severity="ERROR",
                message=f"Failed to download model {model_name}: {e}"
            )
            return False
    
    async def delete_model(self, model_name: str) -> bool:
        """モデル削除"""
        try:
            await asyncio.to_thread(self.ollama_client.delete, model_name)
            
            # キャッシュから削除
            if model_name in self.model_cache:
                del self.model_cache[model_name]
            
            self.logger.log_alert(
                alert_type="model_deleted",
                severity="INFO",
                message=f"Successfully deleted model: {model_name}"
            )
            
            return True
            
        except Exception as e:
            self.logger.log_alert(
                alert_type="model_delete_failed",
                severity="ERROR",
                message=f"Failed to delete model {model_name}: {e}"
            )
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """ヘルスチェック"""
        health_status = {
            "server_connected": False,
            "models_available": 0,
            "primary_model_available": False,
            "fallback_model_available": False,
            "last_check": datetime.now().isoformat()
        }
        
        try:
            # サーバー接続確認
            health_status["server_connected"] = await self._check_server_connection()
            
            if health_status["server_connected"]:
                # モデル情報更新
                await self._refresh_model_list()
                
                health_status["models_available"] = len(self.model_cache)
                health_status["primary_model_available"] = self.primary_model in self.model_cache
                health_status["fallback_model_available"] = self.fallback_model in self.model_cache
                
                # 簡単な推論テスト
                if health_status["primary_model_available"]:
                    test_request = InferenceRequest(
                        prompt="Hello",
                        model_name=self.primary_model
                    )
                    
                    try:
                        test_response = await self.generate(test_request)
                        health_status["inference_test"] = "passed"
                        health_status["test_response_time"] = test_response.processing_time
                    except Exception as e:
                        health_status["inference_test"] = "failed"
                        health_status["inference_error"] = str(e)
            
        except Exception as e:
            health_status["error"] = str(e)
        
        return health_status
    
    async def shutdown(self) -> None:
        """クライアント終了"""
        self.logger.log_shutdown(
            component="ollama_client",
            uptime_seconds=0,  # TODO: 実際の稼働時間計算
            final_stats={
                "models_cached": len(self.model_cache),
                "primary_model": self.primary_model,
                "fallback_model": self.fallback_model
            }
        )


# 便利関数
async def create_ollama_client(base_url: Optional[str] = None) -> OllamaClient:
    """Ollama クライアント作成・初期化"""
    client = OllamaClient(base_url)
    
    if await client.initialize():
        return client
    else:
        raise ConnectionError("Failed to initialize Ollama client")


async def quick_inference(prompt: str, model_name: Optional[str] = None) -> str:
    """簡易推論実行"""
    client = await create_ollama_client()
    
    request = InferenceRequest(prompt=prompt, model_name=model_name)
    response = await client.generate(request)
    
    return response.content


# 使用例
async def main():
    """テスト用メイン関数"""
    try:
        # クライアント作成
        client = await create_ollama_client()
        
        # ヘルスチェック
        health = await client.health_check()
        print("Health Check:", health)
        
        # モデル一覧
        models = await client.list_models()
        print(f"Available models: {len(models)}")
        for model in models:
            print(f"  - {model.name} ({model.size_gb:.1f}GB)")
        
        # 簡単な推論テスト
        if models:
            request = InferenceRequest(
                prompt="What is the capital of Japan?",
                system_message="You are a helpful assistant."
            )
            
            response = await client.generate(request)
            print(f"\nInference Test:")
            print(f"Model: {response.model_used}")
            print(f"Response: {response.content}")
            print(f"Time: {response.processing_time:.2f}s")
        
        # チャット形式テスト
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        chat_response = await client.chat(messages)
        print(f"\nChat Test:")
        print(f"Response: {chat_response.content}")
        
        await client.shutdown()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())