"""
Ollama client for self-learning AI agent
自己学習AIエージェント用Ollamaクライアント
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
import aiohttp
import httpx

from ..config import OllamaConfig

logger = logging.getLogger(__name__)


@dataclass
class OllamaMessage:
    """Ollamaメッセージ"""
    role: str  # "system", "user", "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OllamaRequest:
    """Ollamaリクエスト"""
    model: str
    messages: List[OllamaMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048
    stream: bool = False
    format: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OllamaResponse:
    """Ollamaレスポンス"""
    content: str
    model: str
    created_at: datetime
    done: bool = True
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OllamaStreamChunk:
    """Ollamaストリームチャンク"""
    content: str
    done: bool = False
    model: Optional[str] = None
    created_at: Optional[datetime] = None


class OllamaClient:
    """Ollamaクライアント"""
    
    def __init__(self, config: OllamaConfig):
        self.config = config
        self.base_url = config.base_url.rstrip('/')
        self.model = config.model
        self.timeout = config.timeout
        self.retry_attempts = config.retry_attempts
        self.retry_delay = config.retry_delay
        
        # HTTPクライアント
        self._client: Optional[httpx.AsyncClient] = None
        self._session: Optional[aiohttp.ClientSession] = None
        
        logger.info(f"Ollamaクライアント初期化: {self.base_url}, モデル: {self.model}")
    
    async def __aenter__(self):
        """非同期コンテキストマネージャー開始"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """非同期コンテキストマネージャー終了"""
        await self.close()
    
    async def initialize(self):
        """クライアント初期化"""
        try:
            # httpxクライアント初期化
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
            
            # aiohttpセッション初期化（ストリーミング用）
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
            
            # モデル存在確認
            await self._check_model_availability()
            
            logger.info("Ollamaクライアント初期化完了")
            
        except Exception as e:
            logger.error(f"Ollamaクライアント初期化エラー: {e}")
            raise
    
    async def close(self):
        """クライアント終了"""
        if self._client:
            await self._client.aclose()
        if self._session:
            await self._session.close()
        logger.info("Ollamaクライアント終了")
    
    async def _check_model_availability(self):
        """モデル存在確認"""
        try:
            models = await self.list_models()
            model_names = [model.get('name', '') for model in models]
            
            if self.model not in model_names:
                logger.warning(f"モデルが見つかりません: {self.model}")
                logger.info(f"利用可能なモデル: {model_names}")
                
                # 類似モデル検索
                similar_models = [name for name in model_names if self.model.split(':')[0] in name]
                if similar_models:
                    logger.info(f"類似モデル: {similar_models}")
            else:
                logger.info(f"モデル確認完了: {self.model}")
                
        except Exception as e:
            logger.warning(f"モデル確認エラー: {e}")
    
    async def generate(self, 
                      prompt: str,
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      top_p: Optional[float] = None,
                      top_k: Optional[int] = None,
                      max_tokens: Optional[int] = None,
                      stream: bool = False) -> Union[OllamaResponse, AsyncGenerator[OllamaStreamChunk, None]]:
        """
        テキスト生成
        
        Args:
            prompt: プロンプト
            system_prompt: システムプロンプト
            temperature: 温度パラメータ
            top_p: Top-pパラメータ
            top_k: Top-kパラメータ
            max_tokens: 最大トークン数
            stream: ストリーミングフラグ
            
        Returns:
            OllamaResponse or AsyncGenerator[OllamaStreamChunk]
        """
        messages = []
        
        if system_prompt:
            messages.append(OllamaMessage(role="system", content=system_prompt))
        
        messages.append(OllamaMessage(role="user", content=prompt))
        
        request = OllamaRequest(
            model=self.model,
            messages=messages,
            temperature=temperature or self.config.temperature,
            top_p=top_p or self.config.top_p,
            top_k=top_k or self.config.top_k,
            max_tokens=max_tokens or self.config.max_tokens,
            stream=stream
        )
        
        if stream:
            return self._generate_stream(request)
        else:
            return await self._generate_complete(request)
    
    async def chat(self,
                  messages: List[OllamaMessage],
                  temperature: Optional[float] = None,
                  top_p: Optional[float] = None,
                  top_k: Optional[int] = None,
                  max_tokens: Optional[int] = None,
                  stream: bool = False) -> Union[OllamaResponse, AsyncGenerator[OllamaStreamChunk, None]]:
        """
        チャット形式での生成
        
        Args:
            messages: メッセージリスト
            temperature: 温度パラメータ
            top_p: Top-pパラメータ
            top_k: Top-kパラメータ
            max_tokens: 最大トークン数
            stream: ストリーミングフラグ
            
        Returns:
            OllamaResponse or AsyncGenerator[OllamaStreamChunk]
        """
        request = OllamaRequest(
            model=self.model,
            messages=messages,
            temperature=temperature or self.config.temperature,
            top_p=top_p or self.config.top_p,
            top_k=top_k or self.config.top_k,
            max_tokens=max_tokens or self.config.max_tokens,
            stream=stream
        )
        
        if stream:
            return self._generate_stream(request)
        else:
            return await self._generate_complete(request)
    
    async def _generate_complete(self, request: OllamaRequest) -> OllamaResponse:
        """完全生成"""
        for attempt in range(self.retry_attempts):
            try:
                start_time = time.time()
                
                payload = {
                    "model": request.model,
                    "messages": [
                        {"role": msg.role, "content": msg.content}
                        for msg in request.messages
                    ],
                    "stream": False,
                    "options": {
                        "temperature": request.temperature,
                        "top_p": request.top_p,
                        "top_k": request.top_k,
                        "num_predict": request.max_tokens,
                        **request.options
                    }
                }
                
                if request.format:
                    payload["format"] = request.format
                
                response = await self._client.post(
                    f"{self.base_url}/api/chat",
                    json=payload
                )
                response.raise_for_status()
                
                data = response.json()
                processing_time = time.time() - start_time
                
                return OllamaResponse(
                    content=data.get("message", {}).get("content", ""),
                    model=data.get("model", request.model),
                    created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
                    done=data.get("done", True),
                    total_duration=data.get("total_duration"),
                    load_duration=data.get("load_duration"),
                    prompt_eval_count=data.get("prompt_eval_count"),
                    prompt_eval_duration=data.get("prompt_eval_duration"),
                    eval_count=data.get("eval_count"),
                    eval_duration=data.get("eval_duration"),
                    metadata={"processing_time": processing_time}
                )
                
            except Exception as e:
                logger.warning(f"生成試行 {attempt + 1}/{self.retry_attempts} 失敗: {e}")
                
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"生成失敗: {e}")
                    raise
    
    async def _generate_stream(self, request: OllamaRequest) -> AsyncGenerator[OllamaStreamChunk, None]:
        """ストリーミング生成"""
        for attempt in range(self.retry_attempts):
            try:
                payload = {
                    "model": request.model,
                    "messages": [
                        {"role": msg.role, "content": msg.content}
                        for msg in request.messages
                    ],
                    "stream": True,
                    "options": {
                        "temperature": request.temperature,
                        "top_p": request.top_p,
                        "top_k": request.top_k,
                        "num_predict": request.max_tokens,
                        **request.options
                    }
                }
                
                if request.format:
                    payload["format"] = request.format
                
                async with self._session.post(
                    f"{self.base_url}/api/chat",
                    json=payload
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        
                        if line:
                            try:
                                data = json.loads(line)
                                
                                yield OllamaStreamChunk(
                                    content=data.get("message", {}).get("content", ""),
                                    done=data.get("done", False),
                                    model=data.get("model"),
                                    created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())) if data.get("created_at") else None
                                )
                                
                                if data.get("done", False):
                                    break
                                    
                            except json.JSONDecodeError:
                                logger.warning(f"JSON解析エラー: {line}")
                                continue
                
                return
                
            except Exception as e:
                logger.warning(f"ストリーミング試行 {attempt + 1}/{self.retry_attempts} 失敗: {e}")
                
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"ストリーミング失敗: {e}")
                    raise
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """モデル一覧取得"""
        try:
            response = await self._client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            data = response.json()
            return data.get("models", [])
            
        except Exception as e:
            logger.error(f"モデル一覧取得エラー: {e}")
            raise
    
    async def pull_model(self, model_name: str) -> bool:
        """モデルプル"""
        try:
            payload = {"name": model_name}
            
            async with self._session.post(
                f"{self.base_url}/api/pull",
                json=payload
            ) as response:
                response.raise_for_status()
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line:
                        try:
                            data = json.loads(line)
                            if data.get("status") == "success":
                                logger.info(f"モデルプル完了: {model_name}")
                                return True
                        except json.JSONDecodeError:
                            continue
            
            return False
            
        except Exception as e:
            logger.error(f"モデルプルエラー: {e}")
            return False
    
    async def delete_model(self, model_name: str) -> bool:
        """モデル削除"""
        try:
            payload = {"name": model_name}
            
            response = await self._client.delete(
                f"{self.base_url}/api/delete",
                json=payload
            )
            response.raise_for_status()
            
            logger.info(f"モデル削除完了: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"モデル削除エラー: {e}")
            return False
    
    async def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """モデル情報取得"""
        try:
            model = model_name or self.model
            
            payload = {"name": model}
            
            response = await self._client.post(
                f"{self.base_url}/api/show",
                json=payload
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"モデル情報取得エラー: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """ヘルスチェック"""
        try:
            start_time = time.time()
            
            # 簡単な生成テスト
            response = await self.generate("Hello", max_tokens=10)
            
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "model": self.model,
                "base_url": self.base_url,
                "response_time": response_time,
                "response_length": len(response.content)
            }
            
        except Exception as e:
            logger.error(f"ヘルスチェックエラー: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": self.model,
                "base_url": self.base_url
            }


class OllamaClientManager:
    """Ollamaクライアント管理"""
    
    def __init__(self, config: OllamaConfig):
        self.config = config
        self._client: Optional[OllamaClient] = None
    
    async def get_client(self) -> OllamaClient:
        """クライアント取得"""
        if self._client is None:
            self._client = OllamaClient(self.config)
            await self._client.initialize()
        
        return self._client
    
    async def close(self):
        """クライアント終了"""
        if self._client:
            await self._client.close()
            self._client = None


# 便利関数
async def create_ollama_client(config: OllamaConfig) -> OllamaClient:
    """Ollamaクライアント作成"""
    client = OllamaClient(config)
    await client.initialize()
    return client


async def test_ollama_connection(config: OllamaConfig) -> Dict[str, Any]:
    """Ollama接続テスト"""
    try:
        async with OllamaClient(config) as client:
            return await client.health_check()
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
