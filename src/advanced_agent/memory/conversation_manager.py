"""
Conversation Manager

会話管理とセッション継続機能
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from .persistent_memory import LangChainPersistentMemory
from .memory_models import ConversationItem, SessionContext


class ConversationManager:
    """会話管理システム"""
    
    def __init__(self, 
                 memory_system: LangChainPersistentMemory,
                 session_timeout_hours: int = 24):
        self.memory_system = memory_system
        self.session_timeout_hours = session_timeout_hours
        self.active_sessions: Dict[str, SessionContext] = {}
    
    async def start_conversation(self, 
                               user_id: Optional[str] = None,
                               session_id: Optional[str] = None) -> str:
        """新しい会話セッションを開始"""
        
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # メモリシステムでセッション初期化
        await self.memory_system.initialize_session(session_id, user_id)
        
        # セッションコンテキスト作成
        session_context = SessionContext(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.now(),
            last_activity=datetime.now(),
            conversation_count=0,
            active_memories=[],
            preferences={},
            summary=None
        )
        
        self.active_sessions[session_id] = session_context
        
        return session_id
    
    async def continue_conversation(self, 
                                 session_id: str,
                                 user_input: str,
                                 agent_response: str,
                                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """既存セッションで会話を継続"""
        
        # セッション存在確認
        if session_id not in self.active_sessions:
            # セッション復元を試行
            await self._restore_session(session_id)
        
        # 会話を記憶システムに保存
        conversation_id = await self.memory_system.store_conversation(
            user_input=user_input,
            agent_response=agent_response,
            metadata=metadata
        )
        
        # セッションコンテキスト更新
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.last_activity = datetime.now()
            session.conversation_count += 1
        
        return conversation_id
    
    async def get_conversation_context(self, 
                                    session_id: str,
                                    query: Optional[str] = None,
                                    max_context: int = 10) -> Dict[str, Any]:
        """会話コンテキストの取得"""
        
        if query:
            # クエリベースの関連コンテキスト検索
            context = await self.memory_system.retrieve_relevant_context(
                query=query,
                session_id=session_id,
                max_results=max_context
            )
        else:
            # 現在セッションの最新コンテキスト
            context = await self.memory_system.retrieve_relevant_context(
                query="最新の会話",
                session_id=session_id,
                max_results=max_context
            )
        
        # セッション情報を追加
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            context["session_info"] = {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "start_time": session.start_time.isoformat(),
                "conversation_count": session.conversation_count,
                "preferences": session.preferences
            }
        
        return context
    
    async def _restore_session(self, session_id: str):
        """セッション復元"""
        
        # メモリシステムからセッション情報を復元
        await self.memory_system.initialize_session(session_id)
        
        # セッションコンテキストを再構築
        with self.memory_system.SessionLocal() as db:
            from .persistent_memory import SessionRecord
            
            session_record = db.query(SessionRecord).filter(
                SessionRecord.session_id == session_id
            ).first()
            
            if session_record:
                session_context = SessionContext(
                    session_id=session_id,
                    user_id=session_record.user_id,
                    start_time=session_record.start_time,
                    last_activity=session_record.last_activity,
                    conversation_count=session_record.conversation_count,
                    active_memories=[],
                    preferences=json.loads(session_record.preferences or "{}"),
                    summary=session_record.summary
                )
                
                self.active_sessions[session_id] = session_context
    
    async def end_session(self, session_id: str) -> bool:
        """セッション終了"""
        
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            
            # セッション要約生成
            if session.conversation_count > 0:
                context = await self.get_conversation_context(session_id)
                summary = context.get("session_summary", "")
                
                # データベースに要約保存
                with self.memory_system.SessionLocal() as db:
                    from .persistent_memory import SessionRecord
                    
                    session_record = db.query(SessionRecord).filter(
                        SessionRecord.session_id == session_id
                    ).first()
                    
                    if session_record:
                        session_record.summary = summary
                        session_record.preferences = json.dumps(session.preferences)
                        db.commit()
            
            # アクティブセッションから削除
            del self.active_sessions[session_id]
            return True
        
        return False
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """アクティブセッション一覧"""
        
        sessions = []
        for session_id, session in self.active_sessions.items():
            sessions.append({
                "session_id": session_id,
                "user_id": session.user_id,
                "start_time": session.start_time.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "conversation_count": session.conversation_count
            })
        
        return sessions
    
    async def cleanup_inactive_sessions(self) -> int:
        """非アクティブセッションのクリーンアップ"""
        
        current_time = datetime.now()
        timeout_threshold = current_time.timestamp() - (self.session_timeout_hours * 3600)
        
        inactive_sessions = []
        for session_id, session in self.active_sessions.items():
            if session.last_activity.timestamp() < timeout_threshold:
                inactive_sessions.append(session_id)
        
        # 非アクティブセッションを終了
        for session_id in inactive_sessions:
            await self.end_session(session_id)
        
        return len(inactive_sessions)
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """セッション統計情報"""
        
        total_conversations = sum(
            session.conversation_count 
            for session in self.active_sessions.values()
        )
        
        return {
            "active_sessions": len(self.active_sessions),
            "total_conversations": total_conversations,
            "memory_statistics": self.memory_system.get_memory_statistics()
        }