"""
Codex Compatible Error Handling
CodexのエラーハンドリングシステムをPythonで実装
"""

from typing import Optional, Dict, Any
from enum import Enum


class CodexErrorType(Enum):
    """エラータイプ (Codex CodexErr相当)"""
    CONNECTION_ERROR = "connection_error"
    MODEL_ERROR = "model_error"
    CONFIGURATION_ERROR = "configuration_error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT_ERROR = "timeout_error"
    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    UNKNOWN_ERROR = "unknown_error"


class CodexError(Exception):
    """
    Codex基底エラークラス (Codex CodexErr相当)
    """
    
    def __init__(
        self,
        message: str,
        error_type: CodexErrorType = CodexErrorType.UNKNOWN_ERROR,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.details = details or {}
        self.original_error = original_error
    
    def to_dict(self) -> Dict[str, Any]:
        """エラー情報を辞書形式で取得"""
        return {
            "type": self.error_type.value,
            "message": self.message,
            "details": self.details,
            "original_error": str(self.original_error) if self.original_error else None
        }
    
    def __str__(self) -> str:
        return f"[{self.error_type.value}] {self.message}"


class OllamaConnectionError(CodexError):
    """OLLAMA接続エラー (Codex OLLAMA_CONNECTION_ERROR相当)"""
    
    def __init__(self, message: Optional[str] = None, original_error: Optional[Exception] = None):
        default_message = (
            "No running Ollama server detected. "
            "Start it with: `ollama serve` (after installing). "
            "Install instructions: https://github.com/ollama/ollama"
        )
        super().__init__(
            message or default_message,
            CodexErrorType.CONNECTION_ERROR,
            {"service": "ollama"},
            original_error
        )


class ModelNotFoundError(CodexError):
    """モデル未発見エラー"""
    
    def __init__(self, model_name: str, available_models: Optional[list] = None):
        message = f"Model '{model_name}' not found"
        if available_models:
            message += f". Available models: {', '.join(available_models)}"
        
        super().__init__(
            message,
            CodexErrorType.MODEL_ERROR,
            {"model": model_name, "available_models": available_models}
        )


class ConfigurationError(CodexError):
    """設定エラー"""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(
            message,
            CodexErrorType.CONFIGURATION_ERROR,
            {"config_key": config_key}
        )


class ValidationError(CodexError):
    """バリデーションエラー"""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        super().__init__(
            message,
            CodexErrorType.VALIDATION_ERROR,
            {"field": field, "value": value}
        )


class TimeoutError(CodexError):
    """タイムアウトエラー"""
    
    def __init__(self, operation: str, timeout_seconds: int):
        message = f"Operation '{operation}' timed out after {timeout_seconds} seconds"
        super().__init__(
            message,
            CodexErrorType.TIMEOUT_ERROR,
            {"operation": operation, "timeout": timeout_seconds}
        )


class RateLimitError(CodexError):
    """レート制限エラー"""
    
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(
            message,
            CodexErrorType.RATE_LIMIT_ERROR,
            {"retry_after": retry_after}
        )


def handle_ollama_error(error: Exception) -> CodexError:
    """
    OLLAMA関連エラーの変換 (Codex error handling相当)
    """
    import aiohttp
    
    if isinstance(error, aiohttp.ClientConnectorError):
        return OllamaConnectionError(original_error=error)
    
    elif isinstance(error, aiohttp.ClientTimeout):
        return TimeoutError("OLLAMA request", 30)
    
    elif isinstance(error, aiohttp.ClientResponseError):
        if error.status == 404:
            return ModelNotFoundError("unknown", [])
        elif error.status == 429:
            return RateLimitError("Too many requests to OLLAMA server")
        else:
            return CodexError(
                f"OLLAMA HTTP error: {error.status}",
                CodexErrorType.CONNECTION_ERROR,
                {"status_code": error.status},
                error
            )
    
    elif isinstance(error, ConnectionError):
        return OllamaConnectionError(str(error), error)
    
    else:
        return CodexError(
            f"Unexpected OLLAMA error: {str(error)}",
            CodexErrorType.UNKNOWN_ERROR,
            original_error=error
        )


def get_error_message_ui(error: CodexError) -> str:
    """
    ユーザー向けエラーメッセージの生成 (Codex get_error_message_ui相当)
    """
    if error.error_type == CodexErrorType.CONNECTION_ERROR:
        if "ollama" in error.details.get("service", ""):
            return (
                "❌ OLLAMA server connection failed\n"
                "💡 Make sure OLLAMA is running: `ollama serve`\n"
                "📖 Installation guide: https://github.com/ollama/ollama"
            )
        else:
            return f"❌ Connection error: {error.message}"
    
    elif error.error_type == CodexErrorType.MODEL_ERROR:
        if "not found" in error.message.lower():
            available = error.details.get("available_models", [])
            if available:
                return (
                    f"❌ Model not found: {error.details.get('model', 'unknown')}\n"
                    f"💡 Available models: {', '.join(available[:5])}"
                )
            else:
                return f"❌ {error.message}\n💡 Try: `ollama list` to see available models"
        else:
            return f"❌ Model error: {error.message}"
    
    elif error.error_type == CodexErrorType.CONFIGURATION_ERROR:
        return f"⚙️ Configuration error: {error.message}"
    
    elif error.error_type == CodexErrorType.TIMEOUT_ERROR:
        return f"⏱️ Timeout: {error.message}"
    
    elif error.error_type == CodexErrorType.RATE_LIMIT_ERROR:
        retry_after = error.details.get("retry_after")
        if retry_after:
            return f"🚦 Rate limited: {error.message} (retry after {retry_after}s)"
        else:
            return f"🚦 Rate limited: {error.message}"
    
    else:
        return f"❌ Error: {error.message}"


class ErrorReporter:
    """
    エラー報告システム (Codex error reporting相当)
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def report_error(self, error: CodexError) -> str:
        """エラーを報告し、ユーザー向けメッセージを返す"""
        from loguru import logger
        
        # ログに詳細を記録
        if self.verbose:
            logger.error(f"CodexError: {error.to_dict()}")
        else:
            logger.error(f"Error: {error.message}")
        
        # ユーザー向けメッセージを生成
        return get_error_message_ui(error)
    
    def handle_exception(self, exc: Exception) -> str:
        """例外をCodexErrorに変換して報告"""
        if isinstance(exc, CodexError):
            return self.report_error(exc)
        else:
            # 一般的な例外をCodexErrorに変換
            codex_error = handle_ollama_error(exc)
            return self.report_error(codex_error)