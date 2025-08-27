"""
Codex-OLLAMA Compatibility Layer
CodexとOLLAMA間のAPI変換機能
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from loguru import logger

from .errors import CodexError, ValidationError


@dataclass
class CodexRequest:
    """Codex互換リクエスト形式"""
    prompt: str
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stop: Optional[List[str]] = None
    stream: bool = False


@dataclass
class CodexResponse:
    """Codex互換レスポンス形式"""
    id: str
    object: str = "text_completion"
    created: int = None
    model: str = ""
    choices: List[Dict[str, Any]] = None
    usage: Dict[str, int] = None
    
    def __post_init__(self):
        if self.created is None:
            self.created = int(time.time())
        if self.choices is None:
            self.choices = []
        if self.usage is None:
            self.usage = {}


class CompatibilityLayer:
    """
    CodexとOLLAMA間の互換性レイヤー
    API形式の変換とエラーハンドリングを提供
    """
    
    def __init__(self):
        self.request_id_counter = 0
    
    def translate_codex_to_ollama(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        CodexリクエストをOLLAMA形式に変換
        """
        try:
            # 基本的なバリデーション
            if "prompt" not in request:
                raise ValidationError("Missing required field: prompt")
            
            ollama_request = {
                "model": request.get("model", "qwen2:7b-instruct"),
                "prompt": request["prompt"],
                "stream": request.get("stream", False)
            }
            
            # オプションパラメータの変換
            options = {}
            
            # max_tokens → num_predict
            if "max_tokens" in request:
                options["num_predict"] = request["max_tokens"]
            
            # temperature
            if "temperature" in request:
                options["temperature"] = request["temperature"]
            
            # stop sequences
            if "stop" in request and request["stop"]:
                options["stop"] = request["stop"]
            
            if options:
                ollama_request["options"] = options
            
            logger.debug(f"Converted Codex request to OLLAMA: {ollama_request}")
            return ollama_request
        
        except Exception as e:
            logger.error(f"Request conversion failed: {e}")
            raise ValidationError(f"Failed to convert request: {str(e)}")
    
    def translate_ollama_to_codex(
        self, 
        ollama_response: Dict[str, Any], 
        original_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        OLLAMAレスポンスをCodex形式に変換
        """ 
        try:
            # レスポンスIDの生成
            response_id = f"cmpl-{uuid.uuid4().hex[:8]}"
            
            # 基本レスポンス構造
            codex_response = {
                "id": response_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": original_request.get("model", "qwen2:7b-instruct"),
                "choices": [],
                "usage": {}
            }
            
            # OLLAMAレスポンスからテキストを抽出
            response_text = ollama_response.get("response", "")
            
            # choices配列の構築
            choice = {
                "text": response_text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop" if ollama_response.get("done", True) else None
            }
            codex_response["choices"] = [choice]
            
            # usage情報の構築
            usage = {}
            if "prompt_eval_count" in ollama_response:
                usage["prompt_tokens"] = ollama_response["prompt_eval_count"]
            if "eval_count" in ollama_response:
                usage["completion_tokens"] = ollama_response["eval_count"]
            if usage:
                usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                codex_response["usage"] = usage
            
            logger.debug(f"Converted OLLAMA response to Codex: {codex_response}")
            return codex_response
        
        except Exception as e:
            logger.error(f"Response conversion failed: {e}")
            raise CodexError(f"Failed to convert response: {str(e)}")
    
    def translate_chat_request(self, messages: List[Dict[str, str]]) -> str:
        """
        チャットメッセージをプロンプト形式に変換
        """
        try:
            prompt_parts = []
            
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                
                if role == "system":
                    prompt_parts.append(f"System: {content}")
                elif role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")
            
            # 最後にAssistant:を追加して応答を促す
            prompt_parts.append("Assistant:")
            
            return "\n\n".join(prompt_parts)
        
        except Exception as e:
            logger.error(f"Chat request conversion failed: {e}")
            raise ValidationError(f"Failed to convert chat request: {str(e)}")
    
    def validate_request(self, request: Dict[str, Any]) -> bool:
        """
        リクエストの妥当性チェック
        """
        required_fields = ["prompt"]
        
        for field in required_fields:
            if field not in request:
                raise ValidationError(f"Missing required field: {field}")
        
        # プロンプトの長さチェック
        prompt = request["prompt"]
        if not isinstance(prompt, str) or len(prompt.strip()) == 0:
            raise ValidationError("Prompt must be a non-empty string")
        
        # パラメータの範囲チェック
        if "temperature" in request:
            temp = request["temperature"]
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                raise ValidationError("Temperature must be between 0 and 2")
        
        if "max_tokens" in request:
            max_tokens = request["max_tokens"]
            if not isinstance(max_tokens, int) or max_tokens <= 0:
                raise ValidationError("max_tokens must be a positive integer")
        
        return True
    
    def create_error_response(self, error: Exception, request_id: Optional[str] = None) -> Dict[str, Any]:
        """
        エラーレスポンスの生成
        """
        error_id = request_id or f"err-{uuid.uuid4().hex[:8]}"
        
        return {
            "id": error_id,
            "object": "error",
            "created": int(time.time()),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "code": getattr(error, 'error_code', 500)
            }
        }