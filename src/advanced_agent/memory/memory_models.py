"""
Memory System Data Models

記憶システムのデータモデル定義
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional


@dataclass
class ConversationItem:
    """会話項目"""
    conversation_id: str
    session_id: str
    timestamp: datetime
    user_input: str
    agent_response: str
    context_used: List[str]
    importance_score: float
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryItem:
    """記憶項目"""
    memory_id: str
    content: str
    memory_type: str  # "conversation", "knowledge", "pattern", "preference"
    importance: float
    access_count: int
    last_accessed: datetime
    created_at: datetime
    embedding: Optional[List[float]] = None
    related_memories: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class SessionContext:
    """セッションコンテキスト"""
    session_id: str
    user_id: Optional[str]
    start_time: datetime
    last_activity: datetime
    conversation_count: int
    active_memories: List[str]
    preferences: Dict[str, Any] = field(default_factory=dict)
    summary: Optional[str] = None


@dataclass
class KnowledgeItem:
    """知識項目"""
    knowledge_id: str
    title: str
    content: str
    domain: str
    confidence: float
    source_conversations: List[str]
    validation_count: int
    last_updated: datetime
    embedding: List[float]
    relationships: Dict[str, float] = field(default_factory=dict)