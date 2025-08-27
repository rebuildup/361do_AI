"""
Codex Compatible OLLAMA Client
CodexのOllamaClientをベースにしたPython実装
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass
import aiohttp
from loguru import logger

from .config import CodexConfig, ModelProviderInfo


@dataclass
class OllamaModel:
    """OLLAMA モデル情報"""
    name: str
    size: Optional[int] = None
    modified_at: Optional[str] = None


@dataclass
class GenerateResponse:
    """生成レスポンス"""
    response: str
    done: bool
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None


class CodexOllamaClient:
    """
    Codex互換OLLAMAクライアント
    Rust版OllamaClientの機能をPythonで実装
    """
    
    def __init__(self, config: CodexConfig):
        self.config = config
        self.provider = config.model_provider
        self.base_url = self.provider.base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=self.provider.timeout)
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """非同期コンテキストマネージャー開始"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """非同期コンテキストマネージャー終了"""
        await self.close()
    
    async def initialize(self):
        """クライアント初期化 (Codex try_from_provider相当)"""
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        
        # サーバー接続確認
        await self.health_check()
        logger.info(f"OLLAMA client initialized: {self.base_url}")
    
    async def close(self):
        """クライアント終了処理"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def health_check(self) -> Dict[str, Any]:
        """
        ヘルスチェック (Codex probe_server相当)
        OLLAMAサーバーの接続確認
        """
        if not self.session:
            raise RuntimeError("Client not initialized")
        
        try:
            # OLLAMA APIのタグエンドポイントで接続確認
            url = f"{self.base_url}/api/tags"
            async with self.session.get(url) as response:
                if response.status == 200:
                    return {"status": "ok", "url": url}
                else:
                    raise aiohttp.ClientError(f"HTTP {response.status}")
        
        except Exception as e:
            error_msg = (
                "No running Ollama server detected. "
                "Start it with: `ollama serve` (after installing). "
                "Install instructions: https://github.com/ollama/ollama"
            )
            logger.error(f"OLLAMA health check failed: {e}")
            raise ConnectionError(error_msg) from e
    
    async def fetch_models(self) -> List[OllamaModel]:
        """
        利用可能なモデル一覧を取得 (Codex fetch_models相当)
        """
        if not self.session:
            raise RuntimeError("Client not initialized")
        
        try:
            url = f"{self.base_url}/api/tags"
            async with self.session.get(url) as response:
                if response.status != 200:
                    return []
                
                data = await response.json()
                models = []
                
                for model_data in data.get("models", []):
                    models.append(OllamaModel(
                        name=model_data.get("name", ""),
                        size=model_data.get("size"),
                        modified_at=model_data.get("modified_at")
                    ))
                
                return models
        
        except Exception as e:
            logger.error(f"Failed to fetch models: {e}")
            return []
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> GenerateResponse:
        """
        テキスト生成 (Codex generate相当)
        """
        if not self.session:
            raise RuntimeError("Client not initialized")
        
        model_name = model or self.config.model
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": stream,
            **kwargs
        }
        
        # max_tokensをnum_predictに変換 (OLLAMA形式)
        if "max_tokens" in payload:
            payload["options"] = payload.get("options", {})
            payload["options"]["num_predict"] = payload.pop("max_tokens")
        
        # GPU最適化設定を追加
        self._add_gpu_options(payload)
        
        try:
            url = f"{self.base_url}/api/generate"
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise aiohttp.ClientError(f"HTTP {response.status}: {error_text}")
                
                data = await response.json()
                
                return GenerateResponse(
                    response=data.get("response", ""),
                    done=data.get("done", True),
                    context=data.get("context"),
                    total_duration=data.get("total_duration"),
                    load_duration=data.get("load_duration"),
                    prompt_eval_count=data.get("prompt_eval_count"),
                    eval_count=data.get("eval_count")
                )
        
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[GenerateResponse, None]:
        """
        ストリーミング生成 (Codex streaming相当)
        """
        if not self.session:
            raise RuntimeError("Client not initialized")
        
        model_name = model or self.config.model
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": True,
            **kwargs
        }
        
        # max_tokensをnum_predictに変換
        if "max_tokens" in payload:
            payload["options"] = payload.get("options", {})
            payload["options"]["num_predict"] = payload.pop("max_tokens")
        
        # GPU最適化設定を追加
        self._add_gpu_options(payload)
        
        try:
            url = f"{self.base_url}/api/generate"
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise aiohttp.ClientError(f"HTTP {response.status}: {error_text}")
                
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            yield GenerateResponse(
                                response=data.get("response", ""),
                                done=data.get("done", False),
                                context=data.get("context"),
                                total_duration=data.get("total_duration"),
                                load_duration=data.get("load_duration"),
                                prompt_eval_count=data.get("prompt_eval_count"),
                                eval_count=data.get("eval_count")
                            )
                            
                            if data.get("done", False):
                                break
                        
                        except json.JSONDecodeError:
                            continue
        
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        チャット形式の会話 (Codex chat_completions相当)
        """
        if not self.session:
            raise RuntimeError("Client not initialized")
        
        model_name = model or self.config.model
        
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        
        # max_tokensをnum_predictに変換
        if "max_tokens" in payload:
            payload["options"] = payload.get("options", {})
            payload["options"]["num_predict"] = payload.pop("max_tokens")
        
        # GPU最適化設定を追加
        self._add_gpu_options(payload)
        
        try:
            url = f"{self.base_url}/api/chat"
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise aiohttp.ClientError(f"HTTP {response.status}: {error_text}")
                
                return await response.json()
        
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            raise
    
    async def pull_model(self, model: str) -> bool:
        """
        モデルのプル (Codex pull_with_reporter相当)
        """
        if not self.session:
            raise RuntimeError("Client not initialized")
        
        payload = {"model": model}
        
        try:
            url = f"{self.base_url}/api/pull"
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    return False
                
                # プル完了まで待機 (簡易実装)
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if data.get("status") == "success":
                                return True
                        except json.JSONDecodeError:
                            continue
                
                return False
        
        except Exception as e:
            logger.error(f"Model pull failed: {e}")
            return False
    
    def _add_gpu_options(self, payload: Dict[str, Any]):
        """GPU最適化オプションを追加"""
        if not hasattr(self.config, 'gpu_enabled') or not self.config.gpu_enabled:
            return
        
        options = payload.setdefault("options", {})
        
        # GPU層数の設定
        if hasattr(self.config, 'gpu_layers') and self.config.gpu_layers is not None:
            options["num_gpu"] = self.config.gpu_layers
        
        # 並列処理の最適化
        if hasattr(self.config, 'parallel_requests'):
            options["num_thread"] = min(self.config.parallel_requests, 8)
        
        # メモリ効率の最適化
        options["use_mmap"] = True
        options["use_mlock"] = True
        
        # バッチサイズの最適化
        options["batch_size"] = 512
    
    def get_model_info(self) -> Dict[str, Any]:
        """現在のモデル情報を取得"""
        info = {
            "model": self.config.model,
            "base_url": self.base_url,
            "provider": self.config.model_provider_id,
            "context_window": self.config.model_context_window,
            "max_output_tokens": self.config.model_max_output_tokens
        }
        
        # GPU情報を追加
        if hasattr(self.config, 'gpu_enabled'):
            info["gpu_enabled"] = self.config.gpu_enabled
            if self.config.gpu_enabled:
                info["gpu_memory_fraction"] = getattr(self.config, 'gpu_memory_fraction', 0.8)
                info["gpu_layers"] = getattr(self.config, 'gpu_layers', 'auto')
                info["parallel_requests"] = getattr(self.config, 'parallel_requests', 4)
        
        return info