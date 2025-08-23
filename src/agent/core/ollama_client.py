"""
OLLAMA Client
OLLAMAとの通信を管理するクライアント
"""

import asyncio
import json
from typing import Any, Dict, List, Optional

import aiohttp
from loguru import logger


class OllamaClient:
    """OLLAMAクライアント"""

    def __init__(self, config: Dict[str, Any]):
        self.base_url = config['base_url']
        self.model = config['model']
        # 非同期 HTTP セッションは初期化時に作成される
        self.session: Optional[aiohttp.ClientSession] = None
        # デフォルトタイムアウト (秒)
        self.timeout = aiohttp.ClientTimeout(total=300)  # 5分タイムアウト

    @property
    def session_active(self) -> aiohttp.ClientSession:
        """使用する際にセッションが初期化されていることを保証するプロパティ"""
        if self.session is None:
            raise RuntimeError("Client session is not initialized")
        return self.session

    async def initialize(self):
        """クライアント初期化"""
        logger.info(f"Initializing OLLAMA client: {self.base_url}")

        self.session = aiohttp.ClientSession(timeout=self.timeout)

        # 接続確認
        await self.health_check()

        # モデル確認
        await self._ensure_model_available()

        logger.info("OLLAMA client initialized successfully")

    async def close(self):
        """クライアント終了"""
        if self.session:
            await self.session.close()

    async def health_check(self) -> Dict[str, Any]:
        """ヘルスチェック"""
        try:
            async with self.session_active.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'status': 'healthy',
                        'models': [model['name'] for model in data.get('models', [])]
                    }
                else:
                    return {
                        'status': 'unhealthy',
                        'error': f"HTTP {response.status}"
                    }
        except Exception as e:
            logger.error(f"OLLAMA health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }

    async def _ensure_model_available(self):
        """モデルが利用可能か確認"""
        health_status = await self.health_check()

        if health_status['status'] == 'healthy':
            available_models = health_status.get('models', [])
            if self.model not in available_models:
                logger.warning(f"Model {self.model} not found. Available models: {available_models}")
                # 最初に利用可能なモデルを使用
                if available_models:
                    self.model = available_models[0]
                    logger.info(f"Using alternative model: {self.model}")
                else:
                    raise RuntimeError("No models available in OLLAMA")
        else:
            raise RuntimeError(f"OLLAMA is not healthy: {health_status.get('error')}")

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        stream: bool = False
    ) -> str:
        """テキスト生成"""

        # リクエストボディ構築
        request_data = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            }
        }

        if system_prompt:
            request_data["system"] = system_prompt

        try:
            async with self.session_active.post(
                f"{self.base_url}/api/generate",
                json=request_data
            ) as response:

                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"OLLAMA API error: {response.status} - {error_text}")

                if stream:
                    return await self._handle_stream_response(response)
                else:
                    data = await response.json()
                    return data.get('response', '').strip()

        except asyncio.TimeoutError:
            logger.error("OLLAMA request timed out")
            raise RuntimeError("Request timed out")
        except Exception as e:
            logger.error(f"OLLAMA generation failed: {e}")
            raise

    async def _handle_stream_response(self, response) -> str:
        """ストリーミングレスポンス処理"""
        full_response = ""

        async for line in response.content:
            line = line.decode('utf-8').strip()
            if line:
                try:
                    data = json.loads(line)
                    if 'response' in data:
                        full_response += data['response']
                    if data.get('done', False):
                        break
                except json.JSONDecodeError:
                    continue

        return full_response.strip()

    async def generate_response(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        stream: bool = False,
    ) -> str:
        """互換性ラッパー: 古いコードが generate_response を呼んでいる場合に対応します。
        既存の generate() を委譲して動作を保ちます。
        """
        return await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            stream=stream,
        )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stream: bool = False
    ) -> str:
        """チャット形式での生成"""

        request_data = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            }
        }

        try:
            async with self.session_active.post(
                f"{self.base_url}/api/chat",
                json=request_data
            ) as response:

                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"OLLAMA Chat API error: {response.status} - {error_text}")

                if stream:
                    return await self._handle_chat_stream_response(response)
                else:
                    data = await response.json()
                    return data.get('message', {}).get('content', '').strip()

        except asyncio.TimeoutError:
            logger.error("OLLAMA chat request timed out")
            raise RuntimeError("Chat request timed out")
        except Exception as e:
            logger.error(f"OLLAMA chat failed: {e}")
            raise

    async def _handle_chat_stream_response(self, response) -> str:
        """チャットストリーミングレスポンス処理"""
        full_response = ""

        async for line in response.content:
            line = line.decode('utf-8').strip()
            if line:
                try:
                    data = json.loads(line)
                    if 'message' in data and 'content' in data['message']:
                        full_response += data['message']['content']
                    if data.get('done', False):
                        break
                except json.JSONDecodeError:
                    continue

        return full_response.strip()

    async def embed(self, text: str) -> List[float]:
        """テキスト埋め込み生成"""
        request_data = {
            "model": self.model,
            "prompt": text
        }

        try:
            async with self.session_active.post(
                f"{self.base_url}/api/embeddings",
                json=request_data
            ) as response:

                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"OLLAMA Embeddings API error: {response.status} - {error_text}")

                data = await response.json()
                return data.get('embedding', [])

        except Exception as e:
            logger.error(f"OLLAMA embedding failed: {e}")
            raise

    async def get_model_info(self) -> Dict[str, Any]:
        """モデル情報取得"""
        request_data = {"name": self.model}

        try:
            async with self.session_active.post(
                f"{self.base_url}/api/show",
                json=request_data
            ) as response:

                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise RuntimeError(f"Model info API error: {response.status} - {error_text}")

        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            raise

    async def list_models(self) -> List[Dict[str, Any]]:
        """利用可能なモデル一覧取得"""
        try:
            async with self.session_active.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('models', [])
                else:
                    error_text = await response.text()
                    raise RuntimeError(f"List models API error: {response.status} - {error_text}")

        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise

    async def pull_model(self, model_name: str) -> bool:
        """モデルをプル"""
        request_data = {"name": model_name}

        try:
            async with self.session_active.post(
                f"{self.base_url}/api/pull",
                json=request_data
            ) as response:

                if response.status == 200:
                    # ストリーミングレスポンスを処理
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line:
                            try:
                                data = json.loads(line)
                                status = data.get('status', '')
                                logger.info(f"Pull status: {status}")

                                if data.get('error'):
                                    logger.error(f"Pull error: {data['error']}")
                                    return False

                                if 'success' in status.lower():
                                    return True
                            except json.JSONDecodeError:
                                continue

                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Pull model API error: {response.status} - {error_text}")
                    return False

        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False
