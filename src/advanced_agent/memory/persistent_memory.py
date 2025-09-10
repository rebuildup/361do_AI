"""
LangChain + ChromaDB Persistent Memory System

LangChain Memory と ChromaDB を統合した永続的記憶システム
"""

import os
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.config import Settings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.schema import BaseMessage
from sqlalchemy import create_engine, Column, String, DateTime, Float, Text, Integer
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from .memory_models import ConversationItem, MemoryItem, SessionContext

# SQLAlchemy Base
Base = declarative_base()


class ConversationRecord(Base):
    """会話記録テーブル"""
    __tablename__ = "conversations"

    id = Column(String, primary_key=True)
    session_id = Column(String, index=True)
    user_input = Column(Text)
    agent_response = Column(Text)
    timestamp = Column(DateTime)
    importance_score = Column(Float)
    extra_metadata = Column(Text)  # JSON文字列として保存（metadataは予約語のため変更）


class MemoryRecord(Base):
    """記憶記録テーブル"""
    __tablename__ = "memories"

    id = Column(String, primary_key=True)
    content = Column(Text)
    memory_type = Column(String)
    importance = Column(Float)
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime)
    created_at = Column(DateTime)
    tags = Column(Text)  # JSON文字列として保存


class SessionRecord(Base):
    """セッション記録テーブル"""
    __tablename__ = "sessions"

    session_id = Column(String, primary_key=True)
    user_id = Column(String)
    start_time = Column(DateTime)
    last_activity = Column(DateTime)
    conversation_count = Column(Integer, default=0)
    summary = Column(Text)
    preferences = Column(Text)  # JSON文字列として保存


class LangChainPersistentMemory:
    """LangChain + ChromaDB による永続的記憶システム"""

    def __init__(self,
                 db_path: str = "data/agent_memory.db",
                 chroma_path: str = "data/chroma_db",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 summary_model: str = "qwen2:7b-instruct",
                 ollama_base_url: str = "http://localhost:11434"):
        
        self.db_path = db_path
        self.chroma_path = chroma_path
        
        # ディレクトリ作成
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(chroma_path).mkdir(parents=True, exist_ok=True)
        
        # 埋め込みモデル初期化
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}  # CPU使用でVRAM節約
        )
        
        # ChromaDB PersistentClient を明示的に使用（サーバーテナント接続を避ける）
        self._chroma_client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Chroma VectorStore（LangChain）初期化を PersistentClient 経由で行う
        self.vector_store = Chroma(
            client=self._chroma_client,
            collection_name="agent_memory",
            embedding_function=self.embeddings,
            persist_directory=chroma_path
        )
        
        # 要約用LLM（軽量モデル）
        self.summary_llm = OllamaLLM(
            model=summary_model,
            base_url=ollama_base_url,
            temperature=0.1
        )
        
        # LangChain Memory（要約機能付き）
        self.conversation_memory = ConversationSummaryBufferMemory(
            llm=self.summary_llm,
            max_token_limit=2000,
            return_messages=True,
            memory_key="chat_history"
        )
        
        # SQLAlchemy セットアップ
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # 現在のセッション
        self.current_session_id: Optional[str] = None

    async def initialize_session(self, 
                               session_id: Optional[str] = None,
                               user_id: Optional[str] = None) -> str:
        """セッション初期化"""
        
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        self.current_session_id = session_id
        
        # セッション記録の作成または更新
        with self.SessionLocal() as db:
            session_record = db.query(SessionRecord).filter(
                SessionRecord.session_id == session_id
            ).first()
            
            if session_record is None:
                # 新規セッション
                session_record = SessionRecord(
                    session_id=session_id,
                    user_id=user_id,
                    start_time=datetime.now(),
                    last_activity=datetime.now(),
                    conversation_count=0
                )
                db.add(session_record)
            else:
                # 既存セッション復元
                session_record.last_activity = datetime.now()
                await self._restore_session_context(session_id)
            
            db.commit()
        
        return session_id

    async def _restore_session_context(self, session_id: str):
        """セッションコンテキスト復元"""
        
        # 過去の会話履歴を LangChain Memory に復元
        with self.SessionLocal() as db:
            conversations = db.query(ConversationRecord).filter(
                ConversationRecord.session_id == session_id
            ).order_by(ConversationRecord.timestamp.desc()).limit(10).all()
            
            # 最新10件の会話を逆順で復元
            for conv in reversed(conversations):
                self.conversation_memory.save_context(
                    {"input": conv.user_input},
                    {"output": conv.agent_response}
                )

    async def store_conversation(self,
                               user_input: str,
                               agent_response: str,
                               metadata: Optional[Dict[str, Any]] = None) -> str:
        """会話の永続化保存"""
        
        if self.current_session_id is None:
            await self.initialize_session()
        
        # 1. LangChain Memory に追加
        self.conversation_memory.save_context(
            {"input": user_input},
            {"output": agent_response}
        )
        
        # 2. 重要度スコア計算
        importance_score = await self._calculate_importance(
            f"User: {user_input}\nAgent: {agent_response}"
        )
        
        # 3. ChromaDB に埋め込みベクトルとして保存
        conversation_text = f"User: {user_input}\nAgent: {agent_response}"
        doc_id = f"{self.current_session_id}_{datetime.now().timestamp()}"
        
        self.vector_store.add_texts(
            texts=[conversation_text],
            metadatas=[{
                "session_id": self.current_session_id,
                "timestamp": datetime.now().isoformat(),
                "type": "conversation",
                "importance": importance_score,
                **(metadata or {})
            }],
            ids=[doc_id]
        )
        
        # 4. SQLAlchemy による構造化データ保存
        with self.SessionLocal() as db:
            conversation = ConversationRecord(
                id=doc_id,
                session_id=self.current_session_id,
                user_input=user_input,
                agent_response=agent_response,
                timestamp=datetime.now(),
                importance_score=importance_score,
                extra_metadata=str(metadata or {})
            )
            db.add(conversation)
            
            # セッション統計更新
            session_record = db.query(SessionRecord).filter(
                SessionRecord.session_id == self.current_session_id
            ).first()
            if session_record:
                session_record.conversation_count += 1
                session_record.last_activity = datetime.now()
            
            db.commit()
        
        return doc_id

    async def retrieve_relevant_context(self,
                                      query: str,
                                      session_id: Optional[str] = None,
                                      max_results: int = 5) -> Dict[str, Any]:
        """関連コンテキストの検索"""
        
        # ChromaDB による類似度検索
        search_filter = {}
        if session_id:
            search_filter["session_id"] = session_id
        
        search_results = self.vector_store.similarity_search_with_score(
            query=query,
            k=max_results,
            filter=search_filter if search_filter else None
        )
        
        # LangChain Memory からの現在セッション情報
        current_context = self.conversation_memory.chat_memory.messages
        
        # 要約生成
        summary = ""
        if current_context:
            try:
                summary = self.conversation_memory.predict_new_summary(
                    messages=current_context,
                    existing_summary=""
                )
            except Exception:
                summary = "要約生成に失敗しました"
        
        return {
            "similar_conversations": [
                {
                    "content": doc.page_content,
                    "score": score,
                    "metadata": doc.metadata
                }
                for doc, score in search_results
            ],
            "current_session": [
                {
                    "type": msg.type,
                    "content": msg.content
                }
                for msg in current_context
            ],
            "session_summary": summary,
            "query": query
        }

    async def _calculate_importance(self, text: str) -> float:
        """重要度スコア計算"""
        
        # 簡単な重要度計算（文字数、キーワード、感情分析など）
        importance = 0.5  # ベーススコア
        
        # 文字数による重要度調整
        text_length = len(text)
        if text_length > 500:
            importance += 0.2
        elif text_length > 200:
            importance += 0.1
        
        # キーワードによる重要度調整
        important_keywords = [
            "重要", "問題", "エラー", "学習", "改善", "最適化",
            "設定", "設計", "実装", "テスト", "デバッグ"
        ]
        
        for keyword in important_keywords:
            if keyword in text:
                importance += 0.1
        
        # 0.0-1.0の範囲に正規化
        return min(1.0, max(0.0, importance))

    def get_memory_statistics(self) -> Dict[str, Any]:
        """記憶システムの統計情報"""
        
        # ChromaDB 統計
        try:
            collection = self.vector_store._collection
            chroma_count = collection.count()
        except Exception:
            chroma_count = 0
        
        # SQLAlchemy 統計
        with self.SessionLocal() as db:
            total_conversations = db.query(ConversationRecord).count()
            unique_sessions = db.query(SessionRecord).count()
            total_memories = db.query(MemoryRecord).count()
        
        return {
            "vector_store": {
                "total_documents": chroma_count,
                "collection_name": "agent_memory"
            },
            "structured_data": {
                "total_conversations": total_conversations,
                "unique_sessions": unique_sessions,
                "total_memories": total_memories
            },
            "memory_buffer": {
                "current_messages": len(self.conversation_memory.chat_memory.messages),
                "current_session": self.current_session_id
            }
        }

    async def cleanup_old_memories(self, 
                                 days_threshold: int = 30,
                                 importance_threshold: float = 0.3) -> int:
        """古い記憶の自動整理"""
        
        cutoff_date = datetime.now().timestamp() - (days_threshold * 24 * 3600)
        removed_count = 0
        
        with self.SessionLocal() as db:
            # 重要度が低く古い会話を削除
            old_conversations = db.query(ConversationRecord).filter(
                ConversationRecord.timestamp < datetime.fromtimestamp(cutoff_date),
                ConversationRecord.importance_score < importance_threshold
            ).all()
            
            for conv in old_conversations:
                # ChromaDBからも削除
                try:
                    self.vector_store.delete([conv.id])
                except Exception:
                    pass
                
                db.delete(conv)
                removed_count += 1
            
            db.commit()
        
        return removed_count

    async def get_conversation_history(self, 
                                     session_id: Optional[str] = None,
                                     limit: int = 50) -> List[Dict[str, Any]]:
        """会話履歴の取得"""
        try:
            target_session_id = session_id or self.current_session_id
            if not target_session_id:
                return []
            
            with self.SessionLocal() as db:
                conversations = db.query(ConversationRecord).filter(
                    ConversationRecord.session_id == target_session_id
                ).order_by(ConversationRecord.timestamp.desc()).limit(limit).all()
                
                history = []
                for conv in conversations:
                    history.append({
                        "id": conv.id,
                        "user_input": conv.user_input,
                        "agent_response": conv.agent_response,
                        "timestamp": conv.timestamp.isoformat() if conv.timestamp else None,
                        "importance_score": conv.importance_score,
                        "metadata": conv.extra_metadata
                    })
                
                return history
                
        except Exception as e:
            logging.error(f"会話履歴取得エラー: {e}")
            return []

    def close(self):
        """リソースのクリーンアップ"""
        if hasattr(self, 'vector_store'):
            try:
                self.vector_store.persist()
            except Exception:
                pass


class PersistentMemoryManager:
    """永続的記憶管理システム - Streamlit UI用ラッパー"""
    
    def __init__(self, 
                 db_path: str = "data/agent_memory.db",
                 chroma_path: str = "data/chroma_db",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 summary_model: str = "qwen2:7b-instruct",
                 ollama_base_url: str = "http://localhost:11434"):
        
        self.memory_system = LangChainPersistentMemory(
            db_path=db_path,
            chroma_path=chroma_path,
            embedding_model=embedding_model,
            summary_model=summary_model,
            ollama_base_url=ollama_base_url
        )
        
        # セッション管理
        self.current_session_id: Optional[str] = None
    
    async def initialize_session(self, 
                               session_id: Optional[str] = None,
                               user_id: Optional[str] = None) -> str:
        """セッション初期化"""
        return await self.memory_system.initialize_session(session_id, user_id)
    
    async def store_conversation(self,
                               user_input: str,
                               agent_response: str,
                               metadata: Optional[Dict[str, Any]] = None) -> str:
        """会話の永続化保存"""
        return await self.memory_system.store_conversation(user_input, agent_response, metadata)
    
    async def get_conversation_history(self, 
                                     session_id: Optional[str] = None,
                                     limit: int = 50) -> List[Dict[str, Any]]:
        """会話履歴の取得"""
        # Delegate to the underlying memory system
        return await self.memory_system.get_conversation_history(session_id, limit)
    
    async def search_memories(self, 
                            query: str,
                            max_results: int = 5,
                            similarity_threshold: float = 0.7,
                            session_id: Optional[str] = None) -> Dict[str, Any]:
        """記憶検索"""
        try:
            # 関連コンテキストの検索
            context = await self.memory_system.retrieve_relevant_context(
                query=query,
                session_id=session_id,
                max_results=max_results
            )
            
            # 結果を整形
            results = []
            for conv in context.get("similar_conversations", []):
                if conv.get("score", 0) <= similarity_threshold:
                    results.append({
                        "title": f"会話記録 ({conv.get('metadata', {}).get('timestamp', 'N/A')})",
                        "content": conv.get("content", ""),
                        "similarity": conv.get("score", 0),
                        "created_at": conv.get("metadata", {}).get("timestamp", ""),
                        "type": "conversation"
                    })
            
            return {
                "results": results,
                "total_found": len(results),
                "query": query,
                "session_summary": context.get("session_summary", "")
            }
            
        except Exception as e:
            logging.error(f"記憶検索エラー: {e}")
            return {
                "results": [],
                "total_found": 0,
                "error": str(e)
            }
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """記憶システムの統計情報"""
        return self.memory_system.get_memory_statistics()
    
    async def cleanup_old_memories(self, 
                                 days_threshold: int = 30,
                                 importance_threshold: float = 0.3) -> int:
        """古い記憶の自動整理"""
        return await self.memory_system.cleanup_old_memories(days_threshold, importance_threshold)
    
    def close(self):
        """リソースのクリーンアップ"""
        if hasattr(self, 'memory_system'):
            self.memory_system.close()

    async def cleanup(self) -> None:
        """非同期クリーンアップ互換"""
        self.close()