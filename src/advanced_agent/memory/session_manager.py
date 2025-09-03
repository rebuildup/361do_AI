"""
SQLAlchemy + LangChain Session Management System

SQLAlchemy ORM と LangChain を統合した高度なセッション管理システム
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from sqlalchemy import (
    create_engine, Column, String, DateTime, Float, Text, Integer, 
    Boolean, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage

from .persistent_memory import LangChainPersistentMemory
from .memory_models import SessionContext, ConversationItem

# SQLAlchemy Base
Base = declarative_base()


class UserRecord(Base):
    """ユーザー記録テーブル"""
    __tablename__ = "users"
    
    user_id = Column(String, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True)
    created_at = Column(DateTime, default=datetime.now)
    last_active = Column(DateTime, default=datetime.now)
    preferences = Column(Text)  # JSON文字列
    is_active = Column(Boolean, default=True)
    
    # リレーション
    sessions = relationship("SessionRecord", back_populates="user")
    
    __table_args__ = (
        Index('idx_user_active', 'is_active'),
        Index('idx_user_last_active', 'last_active'),
    )


class SessionRecord(Base):
    """セッション記録テーブル（拡張版）"""
    __tablename__ = "sessions"
    
    session_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey('users.user_id'), nullable=False)
    session_name = Column(String)
    start_time = Column(DateTime, default=datetime.now)
    last_activity = Column(DateTime, default=datetime.now)
    end_time = Column(DateTime)
    conversation_count = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    session_summary = Column(Text)
    session_metadata = Column(Text)  # JSON文字列
    is_active = Column(Boolean, default=True)
    session_type = Column(String, default="conversation")  # conversation, task, analysis
    
    # リレーション
    user = relationship("UserRecord", back_populates="sessions")
    conversations = relationship("ConversationRecord", back_populates="session")
    
    __table_args__ = (
        Index('idx_session_user_active', 'user_id', 'is_active'),
        Index('idx_session_last_activity', 'last_activity'),
        Index('idx_session_type', 'session_type'),
    )


class ConversationRecord(Base):
    """会話記録テーブル（拡張版）"""
    __tablename__ = "conversations"
    
    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey('sessions.session_id'), nullable=False)
    user_input = Column(Text)
    agent_response = Column(Text)
    timestamp = Column(DateTime, default=datetime.now)
    importance_score = Column(Float, default=0.5)
    response_time = Column(Float)  # 応答時間（秒）
    token_count = Column(Integer, default=0)
    conversation_metadata = Column(Text)  # JSON文字列
    is_deleted = Column(Boolean, default=False)
    
    # リレーション
    session = relationship("SessionRecord", back_populates="conversations")
    
    __table_args__ = (
        Index('idx_conversation_session', 'session_id'),
        Index('idx_conversation_timestamp', 'timestamp'),
        Index('idx_conversation_importance', 'importance_score'),
    )


class SessionStateRecord(Base):
    """セッション状態記録テーブル"""
    __tablename__ = "session_states"
    
    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey('sessions.session_id'), nullable=False)
    state_type = Column(String, nullable=False)  # memory, context, preferences
    state_data = Column(Text)  # JSON文字列
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    is_current = Column(Boolean, default=True)
    
    __table_args__ = (
        Index('idx_session_state', 'session_id', 'state_type'),
        UniqueConstraint('session_id', 'state_type', 'is_current', name='uq_current_state'),
    )


class SQLAlchemySessionManager:
    """SQLAlchemy + LangChain 統合セッション管理システム"""
    
    def __init__(self,
                 db_path: str = "data/session_management.db",
                 persistent_memory: Optional[LangChainPersistentMemory] = None):
        
        self.db_path = db_path
        self.persistent_memory = persistent_memory
        
        # SQLAlchemy セットアップ
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # アクティブセッション管理
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # セッション統計
        self.session_stats = {
            "total_sessions": 0,
            "active_sessions": 0,
            "total_users": 0,
            "restored_sessions": 0,
            "integrity_repairs": 0
        }
    
    async def create_user(self,
                         username: str,
                         email: Optional[str] = None,
                         preferences: Optional[Dict[str, Any]] = None) -> str:
        """新規ユーザー作成"""
        
        user_id = str(uuid.uuid4())
        
        with self.SessionLocal() as db:
            try:
                user = UserRecord(
                    user_id=user_id,
                    username=username,
                    email=email,
                    preferences=json.dumps(preferences or {}),
                    created_at=datetime.now(),
                    last_active=datetime.now()
                )
                
                db.add(user)
                db.commit()
                
                self.session_stats["total_users"] += 1
                
                return user_id
                
            except IntegrityError as e:
                db.rollback()
                raise ValueError(f"ユーザー作成に失敗しました: {e}")
    
    async def create_session(self,
                           user_id: str,
                           session_name: Optional[str] = None,
                           session_type: str = "conversation",
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """新規セッション作成"""
        
        session_id = str(uuid.uuid4())
        
        with self.SessionLocal() as db:
            try:
                # ユーザー存在確認
                user = db.query(UserRecord).filter(UserRecord.user_id == user_id).first()
                if not user:
                    raise ValueError(f"ユーザーが見つかりません: {user_id}")
                
                # セッション作成
                session = SessionRecord(
                    session_id=session_id,
                    user_id=user_id,
                    session_name=session_name or f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    session_type=session_type,
                    session_metadata=json.dumps(metadata or {}),
                    start_time=datetime.now(),
                    last_activity=datetime.now()
                )
                
                db.add(session)
                
                # ユーザーの最終アクティブ時間更新
                user.last_active = datetime.now()
                
                db.commit()
                
                # アクティブセッションに追加
                self.active_sessions[session_id] = {
                    "user_id": user_id,
                    "session_name": session.session_name,
                    "session_type": session_type,
                    "start_time": session.start_time,
                    "langchain_memory": ConversationBufferMemory(
                        return_messages=True,
                        memory_key="chat_history"
                    )
                }
                
                self.session_stats["total_sessions"] += 1
                self.session_stats["active_sessions"] += 1
                
                return session_id
                
            except SQLAlchemyError as e:
                db.rollback()
                raise ValueError(f"セッション作成に失敗しました: {e}")
    
    async def restore_session(self, session_id: str) -> Dict[str, Any]:
        """セッション完全復元"""
        
        with self.SessionLocal() as db:
            try:
                # セッション情報取得
                session = db.query(SessionRecord).filter(
                    SessionRecord.session_id == session_id,
                    SessionRecord.is_active == True
                ).first()
                
                if not session:
                    raise ValueError(f"セッションが見つかりません: {session_id}")
                
                # ユーザー情報取得
                user = db.query(UserRecord).filter(
                    UserRecord.user_id == session.user_id
                ).first()
                
                # 会話履歴取得
                conversations = db.query(ConversationRecord).filter(
                    ConversationRecord.session_id == session_id,
                    ConversationRecord.is_deleted == False
                ).order_by(ConversationRecord.timestamp).all()
                
                # セッション状態取得
                session_states = db.query(SessionStateRecord).filter(
                    SessionStateRecord.session_id == session_id,
                    SessionStateRecord.is_current == True
                ).all()
                
                # LangChain Memory 復元
                langchain_memory = ConversationBufferMemory(
                    return_messages=True,
                    memory_key="chat_history"
                )
                
                # 会話履歴を LangChain Memory に復元
                for conv in conversations:
                    langchain_memory.save_context(
                        {"input": conv.user_input},
                        {"output": conv.agent_response}
                    )
                
                # アクティブセッションに追加
                self.active_sessions[session_id] = {
                    "user_id": session.user_id,
                    "session_name": session.session_name,
                    "session_type": session.session_type,
                    "start_time": session.start_time,
                    "langchain_memory": langchain_memory,
                    "restored": True
                }
                
                # セッション活動時間更新
                session.last_activity = datetime.now()
                user.last_active = datetime.now()
                db.commit()
                
                self.session_stats["restored_sessions"] += 1
                self.session_stats["active_sessions"] += 1
                
                # 復元情報返却
                return {
                    "session_id": session_id,
                    "user_id": session.user_id,
                    "username": user.username,
                    "session_name": session.session_name,
                    "session_type": session.session_type,
                    "conversation_count": len(conversations),
                    "start_time": session.start_time.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
                    "restored_conversations": len(conversations),
                    "session_states": {
                        state.state_type: json.loads(state.state_data)
                        for state in session_states
                    }
                }
                
            except SQLAlchemyError as e:
                raise ValueError(f"セッション復元に失敗しました: {e}")
    
    async def save_conversation(self,
                              session_id: str,
                              user_input: str,
                              agent_response: str,
                              response_time: Optional[float] = None,
                              importance_score: Optional[float] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> str:
        """会話保存"""
        
        conversation_id = str(uuid.uuid4())
        
        with self.SessionLocal() as db:
            try:
                # セッション存在確認
                session = db.query(SessionRecord).filter(
                    SessionRecord.session_id == session_id
                ).first()
                
                if not session:
                    raise ValueError(f"セッションが見つかりません: {session_id}")
                
                # 会話記録作成
                conversation = ConversationRecord(
                    id=conversation_id,
                    session_id=session_id,
                    user_input=user_input,
                    agent_response=agent_response,
                    timestamp=datetime.now(),
                    importance_score=importance_score or 0.5,
                    response_time=response_time,
                    token_count=len(user_input.split()) + len(agent_response.split()),
                    conversation_metadata=json.dumps(metadata or {})
                )
                
                db.add(conversation)
                
                # セッション統計更新
                session.conversation_count += 1
                session.total_tokens += conversation.token_count
                session.last_activity = datetime.now()
                
                # ユーザー活動時間更新
                user = db.query(UserRecord).filter(
                    UserRecord.user_id == session.user_id
                ).first()
                if user:
                    user.last_active = datetime.now()
                
                db.commit()
                
                # アクティブセッションの LangChain Memory 更新
                if session_id in self.active_sessions:
                    langchain_memory = self.active_sessions[session_id]["langchain_memory"]
                    langchain_memory.save_context(
                        {"input": user_input},
                        {"output": agent_response}
                    )
                
                return conversation_id
                
            except SQLAlchemyError as e:
                db.rollback()
                raise ValueError(f"会話保存に失敗しました: {e}")
    
    async def save_session_state(self,
                               session_id: str,
                               state_type: str,
                               state_data: Dict[str, Any]) -> str:
        """セッション状態保存"""
        
        state_id = str(uuid.uuid4())
        
        with self.SessionLocal() as db:
            try:
                # 既存の現在状態を無効化
                db.query(SessionStateRecord).filter(
                    SessionStateRecord.session_id == session_id,
                    SessionStateRecord.state_type == state_type,
                    SessionStateRecord.is_current == True
                ).update({"is_current": False})
                
                # 新しい状態を保存
                session_state = SessionStateRecord(
                    id=state_id,
                    session_id=session_id,
                    state_type=state_type,
                    state_data=json.dumps(state_data),
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    is_current=True
                )
                
                db.add(session_state)
                db.commit()
                
                return state_id
                
            except SQLAlchemyError as e:
                db.rollback()
                raise ValueError(f"セッション状態保存に失敗しました: {e}")
    
    async def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """セッションコンテキスト取得"""
        
        if session_id not in self.active_sessions:
            # セッションが非アクティブの場合は復元を試行
            await self.restore_session(session_id)
        
        active_session = self.active_sessions.get(session_id)
        if not active_session:
            raise ValueError(f"セッションが見つかりません: {session_id}")
        
        # LangChain Memory からコンテキスト取得
        langchain_memory = active_session["langchain_memory"]
        messages = langchain_memory.chat_memory.messages
        
        # データベースから詳細情報取得
        with self.SessionLocal() as db:
            session = db.query(SessionRecord).filter(
                SessionRecord.session_id == session_id
            ).first()
            
            user = db.query(UserRecord).filter(
                UserRecord.user_id == session.user_id
            ).first()
            
            recent_conversations = db.query(ConversationRecord).filter(
                ConversationRecord.session_id == session_id,
                ConversationRecord.is_deleted == False
            ).order_by(ConversationRecord.timestamp.desc()).limit(10).all()
        
        return {
            "session_id": session_id,
            "user_info": {
                "user_id": user.user_id,
                "username": user.username,
                "preferences": json.loads(user.preferences or "{}")
            },
            "session_info": {
                "session_name": session.session_name,
                "session_type": session.session_type,
                "start_time": session.start_time.isoformat(),
                "conversation_count": session.conversation_count,
                "total_tokens": session.total_tokens
            },
            "langchain_context": {
                "messages": [
                    {
                        "type": msg.type,
                        "content": msg.content
                    }
                    for msg in messages
                ],
                "message_count": len(messages)
            },
            "recent_conversations": [
                {
                    "id": conv.id,
                    "user_input": conv.user_input,
                    "agent_response": conv.agent_response,
                    "timestamp": conv.timestamp.isoformat(),
                    "importance_score": conv.importance_score
                }
                for conv in recent_conversations
            ]
        }
    
    async def end_session(self, session_id: str) -> Dict[str, Any]:
        """セッション終了"""
        
        with self.SessionLocal() as db:
            try:
                session = db.query(SessionRecord).filter(
                    SessionRecord.session_id == session_id
                ).first()
                
                if not session:
                    raise ValueError(f"セッションが見つかりません: {session_id}")
                
                # セッション終了処理
                session.end_time = datetime.now()
                session.is_active = False
                
                # セッション要約生成（LangChain Memory から）
                if session_id in self.active_sessions:
                    langchain_memory = self.active_sessions[session_id]["langchain_memory"]
                    messages = langchain_memory.chat_memory.messages
                    
                    # 簡単な要約生成
                    summary = f"会話数: {len(messages)}, 開始: {session.start_time}, 終了: {session.end_time}"
                    session.session_summary = summary
                
                db.commit()
                
                # アクティブセッションから削除
                if session_id in self.active_sessions:
                    del self.active_sessions[session_id]
                    self.session_stats["active_sessions"] -= 1
                
                return {
                    "session_id": session_id,
                    "end_time": session.end_time.isoformat(),
                    "total_conversations": session.conversation_count,
                    "total_tokens": session.total_tokens,
                    "duration_minutes": (session.end_time - session.start_time).total_seconds() / 60
                }
                
            except SQLAlchemyError as e:
                db.rollback()
                raise ValueError(f"セッション終了に失敗しました: {e}")
    
    async def repair_session_integrity(self, session_id: str) -> Dict[str, Any]:
        """セッション整合性自動修復"""
        
        repair_results = {
            "session_id": session_id,
            "repairs_performed": [],
            "issues_found": [],
            "repair_success": True
        }
        
        with self.SessionLocal() as db:
            try:
                session = db.query(SessionRecord).filter(
                    SessionRecord.session_id == session_id
                ).first()
                
                if not session:
                    repair_results["issues_found"].append("セッションが存在しません")
                    repair_results["repair_success"] = False
                    return repair_results
                
                # 1. 会話数の整合性チェック
                actual_conv_count = db.query(ConversationRecord).filter(
                    ConversationRecord.session_id == session_id,
                    ConversationRecord.is_deleted == False
                ).count()
                
                if session.conversation_count != actual_conv_count:
                    repair_results["issues_found"].append(
                        f"会話数不整合: 記録={session.conversation_count}, 実際={actual_conv_count}"
                    )
                    session.conversation_count = actual_conv_count
                    repair_results["repairs_performed"].append("会話数を修正")
                
                # 2. トークン数の再計算
                conversations = db.query(ConversationRecord).filter(
                    ConversationRecord.session_id == session_id,
                    ConversationRecord.is_deleted == False
                ).all()
                
                total_tokens = sum(conv.token_count or 0 for conv in conversations)
                if session.total_tokens != total_tokens:
                    repair_results["issues_found"].append(
                        f"トークン数不整合: 記録={session.total_tokens}, 実際={total_tokens}"
                    )
                    session.total_tokens = total_tokens
                    repair_results["repairs_performed"].append("トークン数を修正")
                
                # 3. 最終活動時間の修正
                if conversations:
                    latest_conv = max(conversations, key=lambda x: x.timestamp)
                    if session.last_activity < latest_conv.timestamp:
                        repair_results["issues_found"].append("最終活動時間が古い")
                        session.last_activity = latest_conv.timestamp
                        repair_results["repairs_performed"].append("最終活動時間を修正")
                
                # 4. ユーザー関連の整合性チェック
                user = db.query(UserRecord).filter(
                    UserRecord.user_id == session.user_id
                ).first()
                
                if not user:
                    repair_results["issues_found"].append("関連ユーザーが存在しません")
                    repair_results["repair_success"] = False
                elif not user.is_active:
                    repair_results["issues_found"].append("関連ユーザーが非アクティブ")
                
                # 5. 重複状態の修復
                duplicate_states = db.query(SessionStateRecord).filter(
                    SessionStateRecord.session_id == session_id,
                    SessionStateRecord.is_current == True
                ).all()
                
                state_types = {}
                for state in duplicate_states:
                    if state.state_type in state_types:
                        # 重複発見 - 古い方を無効化
                        older_state = state_types[state.state_type]
                        if older_state.updated_at < state.updated_at:
                            older_state.is_current = False
                            repair_results["repairs_performed"].append(f"重複状態を修正: {state.state_type}")
                        else:
                            state.is_current = False
                            repair_results["repairs_performed"].append(f"重複状態を修正: {state.state_type}")
                    else:
                        state_types[state.state_type] = state
                
                db.commit()
                self.session_stats["integrity_repairs"] += 1
                
            except SQLAlchemyError as e:
                db.rollback()
                repair_results["repair_success"] = False
                repair_results["issues_found"].append(f"修復中にエラー: {e}")
        
        return repair_results
    
    def get_user_sessions(self, user_id: str, include_inactive: bool = False) -> List[Dict[str, Any]]:
        """ユーザーのセッション一覧取得"""
        
        with self.SessionLocal() as db:
            query = db.query(SessionRecord).filter(SessionRecord.user_id == user_id)
            
            if not include_inactive:
                query = query.filter(SessionRecord.is_active == True)
            
            sessions = query.order_by(SessionRecord.last_activity.desc()).all()
            
            return [
                {
                    "session_id": session.session_id,
                    "session_name": session.session_name,
                    "session_type": session.session_type,
                    "start_time": session.start_time.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
                    "conversation_count": session.conversation_count,
                    "is_active": session.is_active,
                    "is_currently_active": session.session_id in self.active_sessions
                }
                for session in sessions
            ]
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """システム統計情報"""
        
        with self.SessionLocal() as db:
            # データベース統計
            total_users = db.query(UserRecord).count()
            active_users = db.query(UserRecord).filter(UserRecord.is_active == True).count()
            total_sessions = db.query(SessionRecord).count()
            active_sessions_db = db.query(SessionRecord).filter(SessionRecord.is_active == True).count()
            total_conversations = db.query(ConversationRecord).filter(ConversationRecord.is_deleted == False).count()
            
            # 最近の活動
            recent_activity = db.query(SessionRecord).filter(
                SessionRecord.last_activity >= datetime.now() - timedelta(hours=24)
            ).count()
        
        return {
            "database_stats": {
                "total_users": total_users,
                "active_users": active_users,
                "total_sessions": total_sessions,
                "active_sessions_db": active_sessions_db,
                "total_conversations": total_conversations,
                "recent_activity_24h": recent_activity
            },
            "runtime_stats": self.session_stats,
            "active_sessions_memory": len(self.active_sessions),
            "active_session_ids": list(self.active_sessions.keys())
        }