"""
Database Management
データベース管理とスキーマ定義
"""

import json
import sqlite3
from uuid import uuid4
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import aiosqlite
from loguru import logger
import os
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
    
    template_id = Column(String, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    template_content = Column(Text, nullable=False)
    description = Column(Text)
    category = Column(String, nullable=False)  # 'system', 'user', 'web_design', 'custom'
    version = Column(Integer, nullable=False, default=1)
    performance_score = Column(Float, default=0.0)
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    active = Column(Boolean, default=True)


class PromptOptimizationHistory(Base):
    """プロンプト最適化履歴テーブル"""
    __tablename__ = 'prompt_optimization_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    template_id = Column(String, nullable=False)
    original_content = Column(Text, nullable=False)
    optimized_content = Column(Text, nullable=False)
    improvement_score = Column(Float, nullable=False)
    optimization_reason = Column(Text)
    optimized_at = Column(DateTime, default=datetime.utcnow)
    applied = Column(Boolean, default=True)


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


class LearningData(Base):
    """学習データテーブル"""
    __tablename__ = 'learning_data'
    
    id = Column(String, primary_key=True)
    content = Column(Text, nullable=False)
    category = Column(String, nullable=False)
    quality_score = Column(Float, nullable=False)
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime, default=datetime.utcnow)
    tags = Column(Text)  # JSON形式で保存
    metadata_json = Column(Text)  # JSON形式で保存


class KnowledgeItem(Base):
    """知識アイテムテーブル"""
    __tablename__ = 'knowledge_items'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    fact = Column(Text, nullable=False)
    category = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    source_context = Column(Text)
    applicability = Column(Text)
    source_conversation_id = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


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
            # SQLite: 既存ファイルがありスキーマが古い場合はバックアップして再作成する
            if self.database_url.startswith("sqlite"):
                db_path = self.database_url.replace("sqlite:///", "")
                try:
                    if os.path.exists(db_path):
                        # 簡易的に prompt_templates テーブルのカラムを確認
                        conn = sqlite3.connect(db_path)
                        cursor = conn.cursor()
                        try:
                            cursor.execute("PRAGMA table_info('prompt_templates')")
                            cols = [row[1] for row in cursor.fetchall()]
                            # 必須カラムが揃っていない場合はバックアップ
                            required = {'template_id', 'name', 'template_content'}
                            if not required.issubset(set(cols)):
                                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                                backup_path = f"{db_path}.backup.{ts}"
                                logger.warning(f"Database schema mismatch detected, backing up '{db_path}' to '{backup_path}' and recreating database")
                                conn.close()
                                os.replace(db_path, backup_path)
                        finally:
                            try:
                                cursor.close()
                            except Exception:
                                pass
                            try:
                                conn.close()
                            except Exception:
                                pass
                except Exception as e:
                    logger.warning(f"Failed to inspect existing sqlite DB for schema: {e}")

                # Check for a UNIQUE constraint on prompt_templates.name and migrate if present
                try:
                    conn = sqlite3.connect(db_path)
                    cur = conn.cursor()
                    try:
                        # Get list of indices on prompt_templates
                        cur.execute("PRAGMA index_list('prompt_templates')")
                        indices = cur.fetchall()
                        need_migration = False
                        for idx in indices:
                            # PRAGMA index_list returns: seq, name, unique, origin, partial
                            idx_name = idx[1]
                            idx_unique = idx[2]
                            if idx_unique:
                                # check indexed columns
                                cur.execute(f"PRAGMA index_info('{idx_name}')")
                                cols = [r[2] for r in cur.fetchall()]
                                if 'name' in cols:
                                    need_migration = True
                                    logger.info(f"Detected UNIQUE index '{idx_name}' on prompt_templates(name) - will migrate")
                                    break

                        if need_migration:
                            # Backup before destructive migration
                            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                            backup_path = f"{db_path}.pre_migrate.backup.{ts}"
                            logger.info(f"Backing up DB before prompt_templates migration: {backup_path}")
                            conn.close()
                            os.replace(db_path, backup_path)

                            # Re-open a new connection to the original path (it does not exist now)
                            # We'll create a fresh DB and copy data via the backup
                            src_conn = sqlite3.connect(backup_path)
                            dst_conn = sqlite3.connect(db_path)
                            try:
                                src_cur = src_conn.cursor()
                                dst_cur = dst_conn.cursor()

                                # Create new prompt_templates table without UNIQUE constraint on name
                                dst_cur.execute('''
                                CREATE TABLE IF NOT EXISTS prompt_templates (
                                    template_id TEXT PRIMARY KEY,
                                    name TEXT NOT NULL,
                                    template_content TEXT NOT NULL,
                                    description TEXT,
                                    category TEXT NOT NULL,
                                    version INTEGER NOT NULL DEFAULT 1,
                                    performance_score REAL DEFAULT 0.0,
                                    usage_count INTEGER DEFAULT 0,
                                    created_at TEXT,
                                    updated_at TEXT,
                                    active INTEGER DEFAULT 1
                                )
                                ''')

                                # Copy existing data from backup if the table exists
                                src_cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='prompt_templates'")
                                if src_cur.fetchone():
                                    # Inspect existing columns and select available ones
                                    src_cur.execute("PRAGMA table_info('prompt_templates')")
                                    existing_cols = [r[1] for r in src_cur.fetchall()]

                                    expected_cols = ['template_id', 'name', 'template_content', 'description', 'category', 'version', 'performance_score', 'usage_count', 'created_at', 'updated_at', 'active']
                                    select_cols = [c for c in expected_cols if c in existing_cols]

                                    if select_cols:
                                        # Build a safe SELECT using only available columns
                                        src_cur.execute("SELECT " + ", ".join(select_cols) + " FROM prompt_templates")
                                        rows = src_cur.fetchall()

                                        insert_rows = []
                                        for row in rows:
                                            # Map fetched values to column names
                                            row_dict = dict(zip(select_cols, row))

                                            # Ensure template_id exists; generate if missing
                                            tmpl_id = row_dict.get('template_id')
                                            if not tmpl_id:
                                                # Prefer name-based id, otherwise random UUID
                                                name_val = row_dict.get('name')
                                                tmpl_id = f"{name_val}_migrated" if name_val else str(uuid4())

                                            name_val = row_dict.get('name') or tmpl_id
                                            template_content = row_dict.get('template_content') or ''
                                            description = row_dict.get('description')
                                            category = row_dict.get('category') or 'custom'
                                            version = row_dict.get('version') if 'version' in row_dict and row_dict.get('version') is not None else 1
                                            performance_score = row_dict.get('performance_score') if 'performance_score' in row_dict and row_dict.get('performance_score') is not None else 0.0
                                            usage_count = row_dict.get('usage_count') if 'usage_count' in row_dict and row_dict.get('usage_count') is not None else 0
                                            created_at = row_dict.get('created_at')
                                            updated_at = row_dict.get('updated_at')
                                            active = row_dict.get('active') if 'active' in row_dict and row_dict.get('active') is not None else 1

                                            insert_rows.append((tmpl_id, name_val, template_content, description, category, version, performance_score, usage_count, created_at, updated_at, active))

                                        if insert_rows:
                                            dst_cur.executemany('''
                                            INSERT OR REPLACE INTO prompt_templates
                                            (template_id, name, template_content, description, category, version, performance_score, usage_count, created_at, updated_at, active)
                                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                            ''', insert_rows)

                                dst_conn.commit()
                                logger.info("prompt_templates migration completed successfully")

                            finally:
                                try:
                                    src_conn.close()
                                except Exception:
                                    pass
                                try:
                                    dst_conn.close()
                                except Exception:
                                    pass
                    finally:
                        try:
                            cur.close()
                        except Exception:
                            pass
                        try:
                            conn.close()
                        except Exception:
                            pass

                except Exception as e:
                    logger.warning(f"Prompt templates migration check failed: {e}")

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
                # 行を辞書形式で扱えるように設定
                self.connection.row_factory = aiosqlite.Row
            
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
                    "template_id": "system_prompt_v1",
                    "name": "system_prompt",
                    "template_content": """あなたは自己学習型のAIエージェントです。
ユーザーとの会話を通じて継続的に学習し、より良い応答を提供することが目標です。
以下の原則に従って行動してください：
1. 正確で有用な情報を提供する
2. ユーザーの意図を正確に理解する
3. 分からないことは素直に認める
4. 学習した知識を適切に活用する
5. 継続的に改善を目指す""",
                    "description": "システム全体の基本プロンプト",
                    "category": "system",
                    "version": 1
                },
                {
                    "template_id": "web_design_system_prompt_v1",
                    "name": "web_design_system_prompt",
                    "template_content": """あなたはWebデザイン専門のAIエージェントです。
モダンで使いやすく、アクセシブルなWebサイトの設計とコード生成を行います。
以下の要素を考慮してください：
1. ユーザビリティとアクセシビリティ
2. レスポンシブデザイン
3. パフォーマンス最適化
4. 最新のデザイントレンド
5. SEO対応""",
                    "description": "Webデザイン専用のシステムプロンプト",
                    "category": "web_design",
                    "version": 1
                },
                {
                    "template_id": "conversation_prompt_v1",
                    "name": "conversation_prompt",
                    "template_content": """あなたは親切で丁寧なAIエージェントです。
ユーザーとの会話において以下の点に注意してください：
1. 常に丁寧で親切な口調で応答する
2. ユーザーの質問に正確に答える
3. 必要に応じて追加情報を提供する
4. 分からないことは素直に認める
5. ユーザーの学習をサポートする""",
                    "description": "会話用の基本プロンプト",
                    "category": "conversation",
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
                   (template_id, name, template_content, description, category, version) 
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    template["template_id"],
                    template["name"],
                    template["template_content"],
                    template.get("description", ""),
                    template["category"],
                    template["version"]
                )
            )
    
    async def execute_query(self, query: str, params: Optional[tuple] = None) -> Any:
        """クエリ実行"""
        try:
            if self.connection is None:
                raise RuntimeError("Database connection is not initialized")

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
        context_data: Optional[Dict] = None
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
        feedback_comment: Optional[str] = None
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
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
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
        source_conversations: Optional[List[int]] = None
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
        category: Optional[str] = None,
        knowledge_type: Optional[str] = None
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
        WHERE name = ? AND active = 1
        ORDER BY version DESC
        LIMIT 1
        """
        
        rows = await self.execute_query(query, (template_name,))
        return dict(rows[0]) if rows else None
    
    # プロンプト管理機能
    async def get_all_prompt_templates(self) -> List[Dict]:
        """全てのプロンプトテンプレートを取得"""
        query = """
        SELECT * FROM prompt_templates
        WHERE active = 1
        ORDER BY category, name, version DESC
        """
        
        rows = await self.execute_query(query)
        return [dict(row) for row in rows] if rows else []
    
    async def get_prompt_template_by_name(self, name: str) -> Optional[Dict]:
        """名前でプロンプトテンプレートを取得"""
        query = """
        SELECT * FROM prompt_templates
        WHERE name = ? AND active = 1
        ORDER BY version DESC
        LIMIT 1
        """
        
        rows = await self.execute_query(query, (name,))
        return dict(rows[0]) if rows else None

    async def get_prompt_template_any_version(self, name: str) -> Optional[Dict]:
        """名前でプロンプトテンプレートを取得（activeフラグに関係なく最新バージョンを返す）"""
        query = """
        SELECT * FROM prompt_templates
        WHERE name = ?
        ORDER BY version DESC
        LIMIT 1
        """

        rows = await self.execute_query(query, (name,))
        return dict(rows[0]) if rows else None
    
    async def insert_prompt_template(
        self,
        template_id: str,
        name: str,
        template_content: str,
        description: Optional[str] = None,
        category: str = "custom"
    ) -> int:
        """プロンプトテンプレート挿入"""
        query = """
        INSERT INTO prompt_templates 
        (template_id, name, template_content, description, category, version, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, 1, ?, ?)
        """
        
        now = datetime.utcnow().isoformat()
        params = (
            template_id,
            name,
            template_content,
            description or f"Custom prompt: {name}",
            category,
            now,
            now
        )
        
        return await self.execute_query(query, params)
    
    async def update_prompt_template(
        self,
        name: str,
        template_content: str,
        description: Optional[str] = None
    ) -> int:
        """プロンプトテンプレート更新"""
        # 既存のテンプレートを取得（activeでないものも含めて最新を探す）
        existing = await self.get_prompt_template_any_version(name)
        if not existing:
            return 0

        # 既存行を更新してバージョンをインクリメントする（UNIQUE制約を避けるため）
        new_version = existing['version'] + 1
        now = datetime.utcnow().isoformat()

        query = """
        UPDATE prompt_templates
        SET template_content = ?, description = ?, version = ?, updated_at = ?
        WHERE template_id = ?
        """

        params = (
            template_content,
            description or existing.get('description', ''),
            new_version,
            now,
            existing['template_id']
        )

        # 実行
        return await self.execute_query(query, params)
    
    async def delete_prompt_template(self, name: str) -> int:
        """プロンプトテンプレート削除（非アクティブ化）"""
        query = "UPDATE prompt_templates SET active = 0 WHERE name = ?"
        return await self.execute_query(query, (name,))
    
    async def insert_prompt_optimization_history(
        self,
        template_id: str,
        original_content: str,
        optimized_content: str,
        improvement_score: float,
        optimization_reason: Optional[str] = None
    ) -> int:
        """プロンプト最適化履歴を記録"""
        query = """
        INSERT INTO prompt_optimization_history 
        (template_id, original_content, optimized_content, improvement_score, 
         optimization_reason, optimized_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        
        params = (
            template_id,
            original_content,
            optimized_content,
            improvement_score,
            optimization_reason,
            datetime.utcnow().isoformat()
        )
        
        return await self.execute_query(query, params)
    
    async def insert_learning_history(
        self,
        learning_type: str,
        description: str,
        before_state: Optional[Dict] = None,
        after_state: Optional[Dict] = None,
        performance_impact: Optional[float] = None
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
    
    # 学習データ関連メソッド
    async def insert_learning_data(
        self,
        data_id: str,
        content: str,
        category: str,
        quality_score: float,
        tags: Optional[str] = None,
        metadata_json: Optional[str] = None
    ) -> int:
        """学習データ挿入"""
        query = """
        INSERT OR REPLACE INTO learning_data 
        (id, content, category, quality_score, tags, metadata_json, created_at, last_used)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        now = datetime.utcnow().isoformat()
        params = (
            data_id,
            content,
            category,
            quality_score,
            tags,
            metadata_json,
            now,
            now
        )
        
        return await self.execute_query(query, params)
    
    async def get_learning_data_by_quality(
        self,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        limit: int = 50
    ) -> List[Dict]:
        """品質スコアで学習データを取得"""
        conditions = []
        params = []
        
        if min_score is not None:
            conditions.append("quality_score >= ?")
            params.append(min_score)
        
        if max_score is not None:
            conditions.append("quality_score <= ?")
            params.append(max_score)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
        SELECT * FROM learning_data
        WHERE {where_clause}
        ORDER BY quality_score DESC, last_used DESC
        LIMIT ?
        """
        
        params.append(limit)
        rows = await self.execute_query(query, tuple(params))
        return [dict(row) for row in rows] if rows else []
    
    async def get_learning_data(
        self,
        category: Optional[str] = None,
        min_quality: Optional[float] = None,
        limit: int = 50
    ) -> List[Dict]:
        """カテゴリと品質スコアで学習データを取得"""
        conditions = []
        params = []
        
        if category is not None:
            conditions.append("category = ?")
            params.append(category)
        
        if min_quality is not None:
            conditions.append("quality_score >= ?")
            params.append(min_quality)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
        SELECT * FROM learning_data
        WHERE {where_clause}
        ORDER BY quality_score DESC, last_used DESC
        LIMIT ?
        """
        
        params.append(limit)
        rows = await self.execute_query(query, tuple(params))
        return [dict(row) for row in rows] if rows else []
    
    async def update_learning_data(
        self,
        data_id: str,
        content: Optional[str] = None,
        quality_score: Optional[float] = None
    ) -> int:
        """学習データ更新"""
        updates = []
        params = []
        
        if content is not None:
            updates.append("content = ?")
            params.append(content)
        
        if quality_score is not None:
            updates.append("quality_score = ?")
            params.append(quality_score)
        
        if not updates:
            return 0
        
        updates.append("last_used = ?")
        params.append(datetime.utcnow().isoformat())
        params.append(data_id)
        
        query = f"""
        UPDATE learning_data
        SET {', '.join(updates)}
        WHERE id = ?
        """
        
        return await self.execute_query(query, tuple(params))
    
    async def get_unused_learning_data(
        self,
        cutoff_date: datetime,
        limit: int = 50
    ) -> List[Dict]:
        """使用されていない学習データを取得"""
        query = """
        SELECT * FROM learning_data
        WHERE last_used < ?
        ORDER BY last_used ASC
        LIMIT ?
        """
        
        rows = await self.execute_query(query, (cutoff_date.isoformat(), limit))
        return [dict(row) for row in rows] if rows else []
    
    async def delete_learning_data(self, data_id: str) -> int:
        """学習データ削除"""
        query = "DELETE FROM learning_data WHERE id = ?"
        return await self.execute_query(query, (data_id,))
    
    async def insert_knowledge_item(
        self,
        fact: str,
        category: str,
        confidence: float,
        source_context: Optional[str] = None,
        applicability: Optional[str] = None,
        source_conversation_id: Optional[int] = None
    ) -> int:
        """知識アイテム挿入"""
        query = """
        INSERT INTO knowledge_items 
        (fact, category, confidence, source_context, applicability, 
         source_conversation_id, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        now = datetime.utcnow().isoformat()
        params = (
            fact,
            category,
            confidence,
            source_context,
            applicability,
            source_conversation_id,
            now,
            now
        )
        
        return await self.execute_query(query, params)
    
    async def get_duplicate_knowledge_items(self) -> List[List[Dict]]:
        """重複する知識アイテムを取得"""
        query = """
        SELECT fact, COUNT(*) as count, GROUP_CONCAT(id) as ids
        FROM knowledge_items
        GROUP BY fact
        HAVING COUNT(*) > 1
        """
        
        rows = await self.execute_query(query)
        duplicate_groups = []
        
        for row in rows:
            ids = [int(id_str) for id_str in row['ids'].split(',')]
            group_items = []
            for item_id in ids:
                item_query = "SELECT * FROM knowledge_items WHERE id = ?"
                item_rows = await self.execute_query(item_query, (item_id,))
                if item_rows:
                    group_items.append(dict(item_rows[0]))
            duplicate_groups.append(group_items)
        
        return duplicate_groups
    
    async def delete_knowledge_item(self, knowledge_id: int) -> int:
        """知識アイテム削除"""
        query = "DELETE FROM knowledge_items WHERE id = ?"
        return await self.execute_query(query, (knowledge_id,))
    
    async def update_knowledge_item(
        self,
        knowledge_id: int,
        fact: Optional[str] = None,
        confidence: Optional[float] = None
    ) -> int:
        """知識アイテム更新"""
        updates = []
        params = []
        
        if fact is not None:
            updates.append("fact = ?")
            params.append(fact)
        
        if confidence is not None:
            updates.append("confidence = ?")
            params.append(confidence)
        
        if not updates:
            return 0
        
        updates.append("updated_at = ?")
        params.append(datetime.utcnow().isoformat())
        params.append(knowledge_id)
        
        query = f"""
        UPDATE knowledge_items
        SET {', '.join(updates)}
        WHERE id = ?
        """
        
        return await self.execute_query(query, tuple(params))
    
    async def delete_low_confidence_knowledge(self, threshold: float = 0.3) -> int:
        """低信頼度の知識を削除"""
        query = "DELETE FROM knowledge_items WHERE confidence < ?"
        return await self.execute_query(query, (threshold,))
    
    async def get_learning_data_stats(self) -> Dict:
        """学習データの統計を取得"""
        query = """
        SELECT 
            COUNT(*) as total_items,
            AVG(quality_score) as avg_quality,
            COUNT(CASE WHEN quality_score >= 0.8 THEN 1 END) as high_quality_count,
            COUNT(CASE WHEN quality_score < 0.6 THEN 1 END) as low_quality_count
        FROM learning_data
        """
        
        rows = await self.execute_query(query)
        return dict(rows[0]) if rows else {}
    
    async def get_learning_statistics(self) -> Dict:
        """学習システムの統計情報を取得"""
        learning_stats = await self.get_learning_data_stats()
        knowledge_stats = await self.get_knowledge_base_stats()
        
        return {
            "total_learning_data": learning_stats.get("total_items", 0),
            "total_knowledge_items": knowledge_stats.get("total_items", 0),
            "average_quality_score": learning_stats.get("avg_quality", 0.0),
            "high_quality_count": learning_stats.get("high_quality_count", 0),
            "low_quality_count": learning_stats.get("low_quality_count", 0),
            "average_confidence": knowledge_stats.get("avg_confidence", 0.0),
            "high_confidence_count": knowledge_stats.get("high_confidence_count", 0)
        }
    
    async def get_knowledge_base_stats(self) -> Dict:
        """知識ベースの統計を取得"""
        query = """
        SELECT 
            COUNT(*) as total_items,
            AVG(confidence) as avg_confidence,
            COUNT(CASE WHEN confidence >= 0.8 THEN 1 END) as high_confidence_count,
            COUNT(CASE WHEN confidence < 0.5 THEN 1 END) as low_confidence_count
        FROM knowledge_items
        """
        
        rows = await self.execute_query(query)
        return dict(rows[0]) if rows else {}
    
    async def get_prompt_optimization_stats(self) -> Dict:
        """プロンプト最適化の統計を取得"""
        query = """
        SELECT 
            COUNT(*) as total_optimizations,
            AVG(improvement_score) as avg_improvement,
            COUNT(CASE WHEN improvement_score >= 0.2 THEN 1 END) as significant_improvements,
            COUNT(CASE WHEN applied = 1 THEN 1 END) as applied_count
        FROM prompt_optimization_history
        """
        
        rows = await self.execute_query(query)
        return dict(rows[0]) if rows else {}
