"""
Codex Agent Interface
CodexのエージェントインターフェースをOLLAMAで実装
"""

import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass
from loguru import logger

from .config import CodexConfig
from .ollama_client import CodexOllamaClient
from .compatibility_layer import CompatibilityLayer
from .errors import CodexError, handle_ollama_error, ErrorReporter
from .performance_monitor import PerformanceMonitor
from .request_pool import AdaptiveRequestPool


@dataclass
class ConversationContext:
    """会話コンテキスト (Codex ConversationHistory相当)"""
    messages: List[Dict[str, str]]
    session_id: str
    created_at: float
    
    def add_message(self, role: str, content: str):
        """メッセージを追加"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
    
    def get_recent_messages(self, limit: int = 10) -> List[Dict[str, str]]:
        """最新のメッセージを取得"""
        return self.messages[-limit:] if len(self.messages) > limit else self.messages


class CodexAgentInterface:
    """
    Codex互換エージェントインターフェース
    Rust版Codexの機能をPythonで実装
    """
    
    def __init__(self, config: CodexConfig):
        self.config = config
        self.ollama_client: Optional[CodexOllamaClient] = None
        self.compatibility_layer = CompatibilityLayer()
        self.error_reporter = ErrorReporter(verbose=True)
        self.conversations: Dict[str, ConversationContext] = {}
        self._initialized = False
        
        # パフォーマンス監視
        self.performance_monitor = PerformanceMonitor()
        self.performance_monitor.start_monitoring()
        
        # リクエストプール（並行処理最適化）
        self.request_pool = AdaptiveRequestPool(
            initial_concurrent=3,
            min_concurrent=1,
            max_concurrent=10,
            default_timeout=30.0
        )
    
    async def initialize(self):
        """エージェント初期化 (Codex spawn相当)"""
        try:
            self.ollama_client = CodexOllamaClient(self.config)
            await self.ollama_client.initialize()
            
            # リクエストプール開始
            await self.request_pool.start()
            
            self._initialized = True
            logger.info("Codex Agent Interface initialized successfully")
        
        except Exception as e:
            error = handle_ollama_error(e)
            logger.error(f"Failed to initialize agent: {error}")
            raise error
    
    async def shutdown(self):
        """エージェント終了処理"""
        if self.ollama_client:
            await self.ollama_client.close()
        
        # リクエストプール停止
        await self.request_pool.stop()
        
        # パフォーマンス監視停止
        self.performance_monitor.stop_monitoring()
        
        self._initialized = False
        logger.info("Codex Agent Interface shutdown complete")
    
    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        コード補完機能 (Codex complete相当)
        プロンプトベースのコード生成
        """
        if not self._initialized:
            raise CodexError("Agent not initialized")
        
        # パフォーマンス監視開始
        request_id = str(uuid.uuid4())
        self.performance_monitor.start_request(
            request_id=request_id,
            request_type="complete",
            model=model or self.config.model,
            max_tokens=max_tokens or self.config.model_max_output_tokens,
            temperature=temperature or 0.1
        )
        
        try:
            # リクエスト構築
            request = {
                "prompt": prompt,
                "model": model or self.config.model,
                "max_tokens": max_tokens or self.config.model_max_output_tokens,
                "temperature": temperature or 0.1,  # コード生成では低めの温度
                "stop": stop,
                "stream": stream
            }
            
            # バリデーション
            self.compatibility_layer.validate_request(request)
            
            # OLLAMA形式に変換
            ollama_request = self.compatibility_layer.translate_codex_to_ollama(request)
            
            # OLLAMA APIを呼び出し
            start_time = time.time()
            ollama_response = await self.ollama_client.generate(**ollama_request)
            response_time = time.time() - start_time
            
            # Codex形式に変換
            codex_response = self.compatibility_layer.translate_ollama_to_codex(
                ollama_response.__dict__, request
            )
            
            # メタデータ追加
            codex_response["response_time"] = response_time
            codex_response["model_info"] = self.ollama_client.get_model_info()
            
            # パフォーマンス監視終了（成功）
            usage = codex_response.get("usage", {})
            self.performance_monitor.end_request(
                request_id=request_id,
                success=True,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0)
            )
            
            logger.info(f"Code completion completed in {response_time:.2f}s")
            return codex_response
        
        except Exception as e:
            # パフォーマンス監視終了（エラー）
            self.performance_monitor.end_request(
                request_id=request_id,
                success=False,
                error_type=type(e).__name__,
                error_message=str(e)
            )
            
            error_msg = self.error_reporter.handle_exception(e)
            logger.error(f"Code completion failed: {error_msg}")
            return self.compatibility_layer.create_error_response(e)
    
    async def complete_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        ストリーミングコード補完
        """
        if not self._initialized:
            raise CodexError("Agent not initialized")
        
        try:
            # リクエスト構築
            request = {
                "prompt": prompt,
                "model": model or self.config.model,
                "max_tokens": max_tokens or self.config.model_max_output_tokens,
                "temperature": temperature or 0.1,
                "stop": stop
            }
            
            # バリデーション
            self.compatibility_layer.validate_request(request)
            
            # OLLAMA形式に変換
            ollama_request = self.compatibility_layer.translate_codex_to_ollama(request)
            
            # ストリーミング生成
            async for chunk in self.ollama_client.generate_stream(**ollama_request):
                # Codex形式に変換
                codex_chunk = self.compatibility_layer.translate_ollama_to_codex(
                    chunk.__dict__, request
                )
                yield codex_chunk
        
        except Exception as e:
            error_msg = self.error_reporter.handle_exception(e)
            logger.error(f"Streaming completion failed: {error_msg}")
            yield self.compatibility_layer.create_error_response(e)   
 
    async def chat(
        self,
        messages: List[Dict[str, str]],
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        チャット機能 (Codex chat相当)
        会話形式のやり取り処理
        """
        if not self._initialized:
            raise CodexError("Agent not initialized")
        
        # パフォーマンス監視開始
        request_id = str(uuid.uuid4())
        self.performance_monitor.start_request(
            request_id=request_id,
            request_type="chat",
            model=model or self.config.model,
            max_tokens=max_tokens or self.config.model_max_output_tokens,
            temperature=temperature or 0.7
        )
        
        try:
            # セッション管理
            if session_id is None:
                session_id = str(uuid.uuid4())
            
            # 会話コンテキストの取得または作成
            if session_id not in self.conversations:
                self.conversations[session_id] = ConversationContext(
                    messages=[],
                    session_id=session_id,
                    created_at=time.time()
                )
            
            context = self.conversations[session_id]
            
            # 新しいメッセージを追加
            if messages:
                for msg in messages:
                    context.add_message(msg["role"], msg["content"])
            
            # プロンプト形式に変換
            prompt = self.compatibility_layer.translate_chat_request(context.get_recent_messages())
            
            # 補完実行（内部でパフォーマンス監視される）
            response = await self.complete(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature or 0.7  # チャットでは少し高めの温度
            )
            
            # 応答をコンテキストに追加
            if "choices" in response and response["choices"]:
                assistant_response = response["choices"][0].get("text", "").strip()
                if assistant_response:
                    context.add_message("assistant", assistant_response)
            
            # セッション情報を追加
            response["session_id"] = session_id
            response["message_count"] = len(context.messages)
            
            # パフォーマンス監視終了（成功）
            usage = response.get("usage", {})
            self.performance_monitor.end_request(
                request_id=request_id,
                success=True,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0)
            )
            
            return response
        
        except Exception as e:
            # パフォーマンス監視終了（エラー）
            self.performance_monitor.end_request(
                request_id=request_id,
                success=False,
                error_type=type(e).__name__,
                error_message=str(e)
            )
            
            error_msg = self.error_reporter.handle_exception(e)
            logger.error(f"Chat failed: {error_msg}")
            return self.compatibility_layer.create_error_response(e)
    
    async def get_available_models(self) -> List[str]:
        """利用可能なモデル一覧を取得"""
        if not self._initialized:
            raise CodexError("Agent not initialized")
        
        try:
            models = await self.ollama_client.fetch_models()
            return [model.name for model in models]
        
        except Exception as e:
            logger.error(f"Failed to fetch models: {e}")
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """ヘルスチェック"""
        try:
            if not self._initialized:
                return {"status": "not_initialized", "healthy": False}
            
            # OLLAMA接続確認
            ollama_status = await self.ollama_client.health_check()
            
            # モデル確認
            models = await self.get_available_models()
            current_model_available = self.config.model in models
            
            # パフォーマンス要約取得
            performance_summary = self.performance_monitor.get_performance_summary(time_window_minutes=5)
            
            return {
                "status": "healthy" if current_model_available else "model_unavailable",
                "healthy": current_model_available,
                "ollama_status": ollama_status,
                "current_model": self.config.model,
                "available_models": models[:5],  # 最初の5つのみ
                "total_models": len(models),
                "active_conversations": len(self.conversations),
                "performance": performance_summary
            }
        
        except Exception as e:
            return {
                "status": "error",
                "healthy": False,
                "error": str(e)
            }
    
    def get_performance_metrics(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """パフォーマンスメトリクス取得"""
        return self.performance_monitor.get_performance_summary(time_window_minutes)
    
    def get_request_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """リクエスト履歴取得"""
        return self.performance_monitor.get_request_history(limit)
    
    def set_performance_alert_threshold(self, metric: str, value: float):
        """パフォーマンスアラート閾値設定"""
        self.performance_monitor.set_alert_threshold(metric, value)
    
    def add_performance_alert_callback(self, callback):
        """パフォーマンスアラートコールバック追加"""
        self.performance_monitor.add_alert_callback(callback)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """システム統計情報取得"""
        performance_metrics = self.performance_monitor.get_performance_summary(time_window_minutes=60)
        pool_stats = self.request_pool.get_stats()
        
        return {
            'agent_status': {
                'initialized': self._initialized,
                'active_conversations': len(self.conversations),
                'model': self.config.model,
                'ollama_url': self.config.ollama_base_url
            },
            'performance_metrics': performance_metrics,
            'request_pool_stats': pool_stats,
            'resource_usage': {
                'memory_percent': performance_metrics.get('system_metrics', {}).get('avg_memory_percent', 0),
                'cpu_percent': performance_metrics.get('system_metrics', {}).get('avg_cpu_percent', 0),
                'memory_used_mb': performance_metrics.get('system_metrics', {}).get('avg_memory_used_mb', 0)
            }
        }