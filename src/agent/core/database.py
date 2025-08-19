"""
Database Management
データベース管理とスキーマ定義
"""

import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import aiosqlite
from loguru import logger
from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, JSON, String, Text, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class Conversation(Base):
    """会話テーブル"""
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False)
    user_input = Column(Text, nullable=False)
    agent_response = Column(Text, nullable=False)
    response_time = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    context_data = Column(JSON)
    user_feedback = Column(Integer, default=None)  # -1: 悪い, 0: 普通, 1: 良い
    feedback_comment = Column(Text, default=None)
    quality_score = Column(Float, default=None)
    improvement_applied = Column(Boolean, default=False)


class QualityMetric(Base):
    """品質指標テーブル"""
    __tablename__ = 'quality_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, nullable=False)
    relevance_score = Column(Float, nullable=False)
    accuracy_score = Column(Float, nullable=False)
    helpfulness_score = Column(Float, nullable=False)
    clarity_score = Column(Float, nullable=False)
    overall_score = Column(Float, nullable=False)
    evaluation_method = Column(String, nullable=False)  # 'auto', 'user', 'hybrid'
    timestamp = Column(DateTime, default=datetime.utcnow)


class KnowledgeBase(Base):
    """知識ベーステーブル"""
    __tablename__ = 'knowledge_base'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(String, nullable=False)  # 'web_design', 'general', 'technical'
    knowledge_type = Column(String, nullable=False)  # 'pattern', 'fact', 'procedure'
    content = Column(Text, nullable=False)
    confidence_score = Column(Float, nullable=False)
    source_conversations = Column(JSON)  # 元となった会話のID配列
    usage_count = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    active = Column(Boolean, default=True)


class PromptTemplate(Base):
    """プロンプトテンプレートテーブル"""
    __tablename__ = 'prompt_templates'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    template_name = Column(String, unique=True, nullable=False)
    template_content = Column(Text, nullable=False)
    category = Column(String, nullable=False)  # 'system', 'user', 'web_design'
    version = Column(Integer, nullable=False, default=1)
    performance_score = Column(Float, default=0.0)
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    active = Column(Boolean, default=True)


class LearningHistory(Base):
    """学習履歴テーブル"""
    __tablename__ = 'learning_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    learning_type = Column(String, nullable=False)  # 'prompt_update', 'knowledge_addition'
    description = Column(Text, nullable=False)
    before_state = Column(JSON)
    after_state = Column(JSON)
    performance_impact = Column(Float, default=None)
    timestamp = Column(DateTime, default=datetime.utcnow)


class DatabaseManager:
    """データベース管理クラス"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.SessionLocal = None
        self.connection = None
        
    async def initialize(self):
        """データベース初期化"""
        logger.info(f"Initializing database: {self.database_url}")
        
        try:
            # SQLAlchemyエンジン作成
            self.engine = create_engine(
                self.database_url,
                echo=False,
                pool_pre_ping=True
            )
            
            # セッションファクトリー作成
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            # テーブル作成
            Base.metadata.create_all(bind=self.engine)
            
            # 非同期接続も初期化
            if self.database_url.startswith("sqlite"):
                db_path = self.database_url.replace("sqlite:///", "")
                self.connection = await aiosqlite.connect(db_path)
            
            # 初期データ投入
            await self._insert_initial_data()
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def close(self):
        """データベース接続を閉じる"""
        if self.connection:
            await self.connection.close()
        
        if self.engine:
            self.engine.dispose()
    
    async def _insert_initial_data(self):
        """初期データ投入"""
        try:
            # 初期プロンプトテンプレート
            initial_prompts = [
                {
                    "template_name": "system_prompt",
                    "template_content": """あなたは自己学習型のAIエージェントです。
ユーザーとの会話を通じて継続的に学習し、より良い応答を提供することが目標です。
以下の原則に従って行動してください：
1. 正確で有用な情報を提供する
2. ユーザーの意図を正確に理解する
3. 分からないことは素直に認める
4. 学習した知識を適切に活用する""",
                    "category": "system",
                    "version": 1
                },
                {
                    "template_name": "web_design_system_prompt",
                    "template_content": """あなたはWebデザイン専門のAIエージェントです。
モダンで使いやすく、アクセシブルなWebサイトの設計とコード生成を行います。
以下の要素を考慮してください：
1. ユーザビリティとアクセシビリティ
2. レスポンシブデザイン
3. パフォーマンス最適化
4. 最新のデザイントレンド
5. SEO対応""",
                    "category": "web_design",
                    "version": 1
                }
            ]
            
            await self._insert_prompt_templates(initial_prompts)
            
        except Exception as e:
            logger.error(f"Initial data insertion failed: {e}")
    
    async def _insert_prompt_templates(self, templates: List[Dict]):
        """プロンプトテンプレート挿入"""
        for template in templates:
            await self.execute_query(
                """INSERT OR IGNORE INTO prompt_templates 
                   (template_name, template_content, category, version) 
                   VALUES (?, ?, ?, ?)""",
                (
                    template["template_name"],
                    template["template_content"],
                    template["category"],
                    template["version"]
                )
            )
    
    async def execute_query(self, query: str, params: tuple = None) -> Any:
        """クエリ実行"""
        try:
            async with self.connection.execute(query, params or ()) as cursor:
                if query.strip().upper().startswith('SELECT'):
                    return await cursor.fetchall()
                else:
                    await self.connection.commit()
                    return cursor.lastrowid
        except Exception as e:
            logger.error(f"Query execution failed: {query} - {e}")
            raise
    
    async def insert_conversation(
        self,
        session_id: str,
        user_input: str,
        agent_response: str,
        response_time: float,
        context_data: Dict = None
    ) -> int:
        """会話記録挿入"""
        query = """
        INSERT INTO conversations 
        (session_id, user_input, agent_response, response_time, context_data, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        
        params = (
            session_id,
            user_input,
            agent_response,
            response_time,
            json.dumps(context_data) if context_data else None,
            datetime.utcnow().isoformat()
        )
        
        return await self.execute_query(query, params)
    
    async def update_conversation_feedback(
        self,
        conversation_id: int,
        feedback_score: int,
        feedback_comment: str = None
    ):
        """会話フィードバック更新"""
        query = """
        UPDATE conversations 
        SET user_feedback = ?, feedback_comment = ?
        WHERE id = ?
        """
        
        params = (feedback_score, feedback_comment, conversation_id)
        await self.execute_query(query, params)
    
    async def insert_quality_metrics(
        self,
        conversation_id: int,
        relevance_score: float,
        accuracy_score: float,
        helpfulness_score: float,
        clarity_score: float,
        overall_score: float,
        evaluation_method: str = "auto"
    ):
        """品質指標挿入"""
        query = """
        INSERT INTO quality_metrics 
        (conversation_id, relevance_score, accuracy_score, helpfulness_score, 
         clarity_score, overall_score, evaluation_method, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            conversation_id,
            relevance_score,
            accuracy_score,
            helpfulness_score,
            clarity_score,
            overall_score,
            evaluation_method,
            datetime.utcnow().isoformat()
        )
        
        await self.execute_query(query, params)
    
    async def get_conversations_by_quality(
        self,
        min_score: float = None,
        max_score: float = None,
        limit: int = 100
    ) -> List[Dict]:
        """品質スコアで会話を取得"""
        conditions = []
        params = []
        
        if min_score is not None:
            conditions.append("c.quality_score >= ?")
            params.append(min_score)
        
        if max_score is not None:
            conditions.append("c.quality_score <= ?")
            params.append(max_score)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
        SELECT c.*, qm.overall_score as quality_score
        FROM conversations c
        LEFT JOIN quality_metrics qm ON c.id = qm.conversation_id
        WHERE {where_clause} AND c.quality_score IS NOT NULL
        ORDER BY c.timestamp DESC
        LIMIT ?
        """
        
        params.append(limit)
        rows = await self.execute_query(query, tuple(params))
        
        return [dict(row) for row in rows] if rows else []
    
    async def insert_knowledge(
        self,
        category: str,
        knowledge_type: str,
        content: str,
        confidence_score: float,
        source_conversations: List[int] = None
    ) -> int:
        """知識ベース挿入"""
        query = """
        INSERT INTO knowledge_base 
        (category, knowledge_type, content, confidence_score, source_conversations, 
         created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        now = datetime.utcnow().isoformat()
        params = (
            category,
            knowledge_type,
            content,
            confidence_score,
            json.dumps(source_conversations or []),
            now,
            now
        )
        
        return await self.execute_query(query, params)
    
    async def get_active_knowledge(
        self,
        category: str = None,
        knowledge_type: str = None
    ) -> List[Dict]:
        """アクティブな知識を取得"""
        conditions = ["active = 1"]
        params = []
        
        if category:
            conditions.append("category = ?")
            params.append(category)
        
        if knowledge_type:
            conditions.append("knowledge_type = ?")
            params.append(knowledge_type)
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
        SELECT * FROM knowledge_base
        WHERE {where_clause}
        ORDER BY confidence_score DESC, usage_count DESC
        """
        
        rows = await self.execute_query(query, tuple(params))
        return [dict(row) for row in rows] if rows else []
    
    async def get_prompt_template(self, template_name: str) -> Optional[Dict]:
        """プロンプトテンプレート取得"""
        query = """
        SELECT * FROM prompt_templates
        WHERE template_name = ? AND active = 1
        ORDER BY version DESC
        LIMIT 1
        """
        
        rows = await self.execute_query(query, (template_name,))
        return dict(rows[0]) if rows else None
    
    async def insert_learning_history(
        self,
        learning_type: str,
        description: str,
        before_state: Dict = None,
        after_state: Dict = None,
        performance_impact: float = None
    ) -> int:
        """学習履歴挿入"""
        query = """
        INSERT INTO learning_history 
        (learning_type, description, before_state, after_state, 
         performance_impact, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        
        params = (
            learning_type,
            description,
            json.dumps(before_state) if before_state else None,
            json.dumps(after_state) if after_state else None,
            performance_impact,
            datetime.utcnow().isoformat()
        )
        
        return await self.execute_query(query, params)
    
    async def get_performance_metrics(
        self,
        days: int = 30
    ) -> Dict:
        """パフォーマンス指標取得"""
        from_date = (datetime.utcnow().timestamp() - (days * 24 * 3600))
        
        query = """
        SELECT 
            COUNT(*) as total_conversations,
            AVG(quality_score) as avg_quality,
            AVG(response_time) as avg_response_time,
            COUNT(CASE WHEN user_feedback = 1 THEN 1 END) as positive_feedback,
            COUNT(CASE WHEN user_feedback = -1 THEN 1 END) as negative_feedback
        FROM conversations
        WHERE timestamp >= datetime(?, 'unixepoch')
        """
        
        rows = await self.execute_query(query, (from_date,))
        return dict(rows[0]) if rows else {}
