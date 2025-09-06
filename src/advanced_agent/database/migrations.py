"""
Database migration management for self-learning AI agent
自己学習AIエージェント用データベースマイグレーション管理
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from sqlalchemy import text, inspect
from sqlalchemy.exc import SQLAlchemyError

from .connection import DatabaseManager
from .models import Base

logger = logging.getLogger(__name__)


class MigrationManager:
    """マイグレーション管理クラス"""
    
    def __init__(self, db_manager: DatabaseManager, migrations_dir: str = "data/migrations"):
        """
        マイグレーションマネージャー初期化
        
        Args:
            db_manager: データベースマネージャー
            migrations_dir: マイグレーションファイルディレクトリ
        """
        self.db_manager = db_manager
        self.migrations_dir = Path(migrations_dir)
        self.migrations_dir.mkdir(parents=True, exist_ok=True)
        
        # マイグレーション履歴テーブル作成
        self.create_migrations_table()
    
    def create_migrations_table(self):
        """マイグレーション履歴テーブル作成"""
        try:
            with self.db_manager.get_session() as session:
                # マイグレーション履歴テーブル作成
                session.execute(text("""
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        version VARCHAR(50) NOT NULL UNIQUE,
                        description TEXT,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        checksum VARCHAR(64),
                        rollback_sql TEXT
                    )
                """))
                session.commit()
                logger.info("マイグレーション履歴テーブル作成完了")
        except Exception as e:
            logger.error(f"マイグレーション履歴テーブル作成エラー: {e}")
            raise
    
    def get_applied_migrations(self) -> List[str]:
        """適用済みマイグレーション取得"""
        try:
            with self.db_manager.get_session() as session:
                result = session.execute(text("SELECT version FROM schema_migrations ORDER BY applied_at"))
                return [row[0] for row in result.fetchall()]
        except Exception as e:
            logger.error(f"適用済みマイグレーション取得エラー: {e}")
            return []
    
    def get_pending_migrations(self) -> List[Dict[str, Any]]:
        """未適用マイグレーション取得"""
        applied = set(self.get_applied_migrations())
        pending = []
        
        # マイグレーションファイル検索
        for migration_file in sorted(self.migrations_dir.glob("*.json")):
            version = migration_file.stem
            if version not in applied:
                try:
                    with open(migration_file, 'r', encoding='utf-8') as f:
                        migration_data = json.load(f)
                    migration_data['file_path'] = str(migration_file)
                    pending.append(migration_data)
                except Exception as e:
                    logger.error(f"マイグレーションファイル読み込みエラー {migration_file}: {e}")
        
        return pending
    
    def apply_migration(self, migration: Dict[str, Any]) -> bool:
        """マイグレーション適用"""
        try:
            version = migration['version']
            description = migration.get('description', '')
            sql_commands = migration.get('sql', [])
            rollback_sql = migration.get('rollback_sql', [])
            checksum = migration.get('checksum', '')
            
            with self.db_manager.get_session() as session:
                # マイグレーション実行
                for sql in sql_commands:
                    if sql.strip():
                        session.execute(text(sql))
                
                # 履歴記録
                session.execute(text("""
                    INSERT INTO schema_migrations (version, description, checksum, rollback_sql)
                    VALUES (:version, :description, :checksum, :rollback_sql)
                """), {
                    'version': version,
                    'description': description,
                    'checksum': checksum,
                    'rollback_sql': json.dumps(rollback_sql)
                })
                
                session.commit()
                logger.info(f"マイグレーション適用完了: {version} - {description}")
                return True
                
        except Exception as e:
            logger.error(f"マイグレーション適用エラー {migration.get('version', 'unknown')}: {e}")
            return False
    
    def rollback_migration(self, version: str) -> bool:
        """マイグレーションロールバック"""
        try:
            with self.db_manager.get_session() as session:
                # ロールバックSQL取得
                result = session.execute(text("""
                    SELECT rollback_sql FROM schema_migrations WHERE version = :version
                """), {'version': version})
                
                row = result.fetchone()
                if not row:
                    logger.error(f"マイグレーション履歴が見つかりません: {version}")
                    return False
                
                rollback_sql = json.loads(row[0])
                
                # ロールバック実行
                for sql in rollback_sql:
                    if sql.strip():
                        session.execute(text(sql))
                
                # 履歴削除
                session.execute(text("""
                    DELETE FROM schema_migrations WHERE version = :version
                """), {'version': version})
                
                session.commit()
                logger.info(f"マイグレーションロールバック完了: {version}")
                return True
                
        except Exception as e:
            logger.error(f"マイグレーションロールバックエラー {version}: {e}")
            return False
    
    def migrate(self) -> Dict[str, Any]:
        """全マイグレーション実行"""
        result = {
            'success': True,
            'applied_migrations': [],
            'failed_migrations': [],
            'total_pending': 0,
            'total_applied': 0
        }
        
        try:
            pending = self.get_pending_migrations()
            result['total_pending'] = len(pending)
            
            for migration in pending:
                version = migration['version']
                if self.apply_migration(migration):
                    result['applied_migrations'].append(version)
                    result['total_applied'] += 1
                else:
                    result['failed_migrations'].append(version)
                    result['success'] = False
            
            logger.info(f"マイグレーション完了: {result['total_applied']}/{result['total_pending']} 適用")
            
        except Exception as e:
            logger.error(f"マイグレーション実行エラー: {e}")
            result['success'] = False
            result['error'] = str(e)
        
        return result
    
    def validate_schema(self) -> Dict[str, Any]:
        """スキーマ検証"""
        try:
            inspector = inspect(self.db_manager.engine)
            existing_tables = set(inspector.get_table_names())
            
            # 期待されるテーブル一覧
            expected_tables = {
                'agent_states', 'prompt_templates', 'tuning_data',
                'evolution_candidates', 'evolution_tuning_data', 'reward_signals',
                'interactions', 'learning_sessions', 'system_metrics', 'configurations',
                'schema_migrations'
            }
            
            missing_tables = expected_tables - existing_tables
            extra_tables = existing_tables - expected_tables
            
            return {
                'is_valid': len(missing_tables) == 0,
                'missing_tables': list(missing_tables),
                'extra_tables': list(extra_tables),
                'existing_tables': list(existing_tables),
                'expected_tables': list(expected_tables)
            }
            
        except Exception as e:
            logger.error(f"スキーマ検証エラー: {e}")
            return {
                'is_valid': False,
                'error': str(e)
            }
    
    def create_initial_migration(self) -> str:
        """初期マイグレーション作成"""
        migration_data = {
            'version': '001_initial_schema',
            'description': 'Initial database schema creation',
            'sql': [
                # テーブル作成SQL（models.pyから生成）
                """
                CREATE TABLE agent_states (
                    session_id VARCHAR(36) PRIMARY KEY,
                    user_id VARCHAR(255),
                    current_prompt_version VARCHAR(50) NOT NULL DEFAULT '1.0.0',
                    learning_epoch INTEGER NOT NULL DEFAULT 0,
                    total_interactions INTEGER NOT NULL DEFAULT 0,
                    reward_score FLOAT NOT NULL DEFAULT 0.0,
                    evolution_generation INTEGER NOT NULL DEFAULT 0,
                    last_activity TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    performance_metrics JSON,
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """,
                """
                CREATE TABLE prompt_templates (
                    version VARCHAR(50) PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata JSON,
                    performance_score FLOAT NOT NULL DEFAULT 0.0,
                    usage_count INTEGER NOT NULL DEFAULT 0,
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_modified TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """,
                """
                CREATE TABLE tuning_data (
                    id VARCHAR(36) PRIMARY KEY,
                    content TEXT NOT NULL,
                    data_type VARCHAR(50) NOT NULL,
                    quality_score FLOAT NOT NULL DEFAULT 0.0,
                    usage_count INTEGER NOT NULL DEFAULT 0,
                    tags JSON,
                    metadata JSON,
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """,
                """
                CREATE TABLE evolution_candidates (
                    id VARCHAR(36) PRIMARY KEY,
                    parent_ids JSON,
                    prompt_template_version VARCHAR(50) NOT NULL,
                    fitness_score FLOAT NOT NULL DEFAULT 0.0,
                    generation INTEGER NOT NULL DEFAULT 0,
                    evaluation_metrics JSON,
                    is_selected BOOLEAN NOT NULL DEFAULT 0,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    evaluated_at TIMESTAMP,
                    FOREIGN KEY (prompt_template_version) REFERENCES prompt_templates(version)
                )
                """,
                """
                CREATE TABLE evolution_tuning_data (
                    evolution_candidate_id VARCHAR(36),
                    tuning_data_id VARCHAR(36),
                    weight FLOAT NOT NULL DEFAULT 1.0,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (evolution_candidate_id, tuning_data_id),
                    FOREIGN KEY (evolution_candidate_id) REFERENCES evolution_candidates(id),
                    FOREIGN KEY (tuning_data_id) REFERENCES tuning_data(id)
                )
                """,
                """
                CREATE TABLE reward_signals (
                    id VARCHAR(36) PRIMARY KEY,
                    interaction_id VARCHAR(36) NOT NULL,
                    reward_type VARCHAR(50) NOT NULL,
                    value FLOAT NOT NULL,
                    context JSON,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (interaction_id) REFERENCES interactions(id)
                )
                """,
                """
                CREATE TABLE interactions (
                    id VARCHAR(36) PRIMARY KEY,
                    session_id VARCHAR(36) NOT NULL,
                    user_input TEXT NOT NULL,
                    agent_response TEXT NOT NULL,
                    reasoning_steps JSON,
                    tool_usage JSON,
                    processing_time FLOAT,
                    quality_score FLOAT,
                    metadata JSON,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES agent_states(session_id)
                )
                """,
                """
                CREATE TABLE learning_sessions (
                    id VARCHAR(36) PRIMARY KEY,
                    session_id VARCHAR(36) NOT NULL,
                    session_type VARCHAR(50) NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    parameters JSON,
                    results JSON,
                    metrics JSON,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES agent_states(session_id)
                )
                """,
                """
                CREATE TABLE system_metrics (
                    id VARCHAR(36) PRIMARY KEY,
                    metric_name VARCHAR(100) NOT NULL,
                    metric_value FLOAT NOT NULL,
                    metric_type VARCHAR(50) NOT NULL,
                    tags JSON,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(metric_name, timestamp)
                )
                """,
                """
                CREATE TABLE configurations (
                    id VARCHAR(36) PRIMARY KEY,
                    config_key VARCHAR(100) NOT NULL UNIQUE,
                    config_value TEXT NOT NULL,
                    config_type VARCHAR(50) NOT NULL,
                    description TEXT,
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            ],
            'rollback_sql': [
                "DROP TABLE IF EXISTS configurations",
                "DROP TABLE IF EXISTS system_metrics",
                "DROP TABLE IF EXISTS learning_sessions",
                "DROP TABLE IF EXISTS interactions",
                "DROP TABLE IF EXISTS reward_signals",
                "DROP TABLE IF EXISTS evolution_tuning_data",
                "DROP TABLE IF EXISTS evolution_candidates",
                "DROP TABLE IF EXISTS tuning_data",
                "DROP TABLE IF EXISTS prompt_templates",
                "DROP TABLE IF EXISTS agent_states"
            ],
            'checksum': 'initial_schema_v1'
        }
        
        # マイグレーションファイル保存
        migration_file = self.migrations_dir / f"{migration_data['version']}.json"
        with open(migration_file, 'w', encoding='utf-8') as f:
            json.dump(migration_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"初期マイグレーション作成完了: {migration_file}")
        return str(migration_file)
    
    def get_migration_status(self) -> Dict[str, Any]:
        """マイグレーション状況取得"""
        try:
            applied = self.get_applied_migrations()
            pending = self.get_pending_migrations()
            validation = self.validate_schema()
            
            return {
                'applied_migrations': applied,
                'pending_migrations': [m['version'] for m in pending],
                'total_applied': len(applied),
                'total_pending': len(pending),
                'schema_validation': validation,
                'migrations_dir': str(self.migrations_dir)
            }
        except Exception as e:
            logger.error(f"マイグレーション状況取得エラー: {e}")
            return {
                'error': str(e),
                'applied_migrations': [],
                'pending_migrations': [],
                'total_applied': 0,
                'total_pending': 0
            }


def initialize_database_with_migrations(db_path: str = "data/self_learning_agent.db") -> DatabaseManager:
    """マイグレーション付きデータベース初期化"""
    # データベースマネージャー初期化
    db_manager = DatabaseManager(db_path=db_path)
    
    # マイグレーションマネージャー初期化
    migration_manager = MigrationManager(db_manager)
    
    # 初期マイグレーション作成（存在しない場合）
    if not migration_manager.get_applied_migrations():
        migration_manager.create_initial_migration()
    
    # マイグレーション実行
    result = migration_manager.migrate()
    if not result['success']:
        logger.error(f"マイグレーション失敗: {result}")
        raise RuntimeError(f"マイグレーション失敗: {result}")
    
    logger.info("データベース初期化完了（マイグレーション適用済み）")
    return db_manager