"""
API データモデル

OpenAI 互換 API のリクエスト・レスポンスモデル
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    """メッセージロール"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class ChatMessage(BaseModel):
    """チャットメッセージ"""
    role: MessageRole
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None


class ChatCompletionRequest(BaseModel):
    """チャット完了リクエスト（OpenAI 互換）"""
    model: str = Field(default="deepseek-r1:7b", description="使用するモデル")
    messages: List[ChatMessage] = Field(description="会話メッセージ")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = Field(default=False)
    user: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    """チャット完了選択肢"""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    """使用量統計"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """チャット完了レスポンス（OpenAI 互換）"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class StreamChoice(BaseModel):
    """ストリーム選択肢"""
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    """チャット完了ストリームレスポンス"""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]


class ModelInfo(BaseModel):
    """モデル情報"""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "advanced-agent"
    permission: List[Dict[str, Any]] = []


class ModelsResponse(BaseModel):
    """モデル一覧レスポンス"""
    object: str = "list"
    data: List[ModelInfo]


class HealthResponse(BaseModel):
    """ヘルスチェックレスポンス"""
    status: str
    timestamp: datetime
    version: str
    system_info: Dict[str, Any]


class ErrorResponse(BaseModel):
    """エラーレスポンス"""
    error: Dict[str, Any]


class SystemStatsRequest(BaseModel):
    """システム統計リクエスト"""
    include_gpu: bool = Field(default=True)
    include_memory: bool = Field(default=True)
    include_processes: bool = Field(default=False)


class SystemStatsResponse(BaseModel):
    """システム統計レスポンス"""
    timestamp: datetime
    cpu: Dict[str, Any]
    memory: Dict[str, Any]
    gpu: Optional[Dict[str, Any]] = None
    processes: Optional[List[Dict[str, Any]]] = None


class InferenceRequest(BaseModel):
    """推論リクエスト"""
    prompt: str
    model: Optional[str] = Field(default="deepseek-r1:7b")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    use_cot: Optional[bool] = Field(default=True, description="Chain-of-Thought を使用")
    session_id: Optional[str] = None


class InferenceResponse(BaseModel):
    """推論レスポンス"""
    id: str
    response: str
    reasoning_steps: Optional[List[Dict[str, Any]]] = None
    confidence_score: Optional[float] = None
    processing_time: float
    memory_usage: Dict[str, Any]
    model_info: Dict[str, Any]


class MemorySearchRequest(BaseModel):
    """記憶検索リクエスト"""
    query: str
    session_id: Optional[str] = None
    max_results: int = Field(default=5, ge=1, le=50)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class MemorySearchResponse(BaseModel):
    """記憶検索レスポンス"""
    results: List[Dict[str, Any]]
    total_found: int
    search_time: float


class SessionRequest(BaseModel):
    """セッション作成リクエスト"""
    user_id: Optional[str] = None
    session_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SessionResponse(BaseModel):
    """セッションレスポンス"""
    session_id: str
    created_at: datetime
    user_id: Optional[str] = None
    session_name: Optional[str] = None
    metadata: Dict[str, Any]


class ConfigUpdateRequest(BaseModel):
    """設定更新リクエスト"""
    config_section: str
    config_data: Dict[str, Any]


class ConfigResponse(BaseModel):
    """設定レスポンス"""
    success: bool
    message: str
    current_config: Dict[str, Any]