"""
Database connection management for self-learning AI agent
自己学習AIエージェント用データベース接続管理
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import contextmanager
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from .models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """データベース管理クラス"""
    
    def __init__(self, 
                 db_path: str = "data/self_learning_agent.db",
                 echo: bool = False):
        """
        データベースマネージャー初期化
        
        Args:
            db_path: データベースファイルパス
            echo: SQLクエリのログ出力フラグ
        """
        self.db_path = db_path
        self.echo = echo
        
        # データベースディレクトリ作成
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # エンジン初期化
        self.engine: Optional[Engine] = None
        self.SessionLocal: Optional[sessionmaker] = None
        
        self._initialize_engine()
        self._setup_event_listeners()
    
    def _initialize_engine(self):
        """エンジン初期化"""
        try:
            # SQLite用の設定
            connect_args = {
                "check_same_thread": False,
                "timeout": 30
            }
            
            # エンジン作成
            self.engine = create_engine(
                f"sqlite:///{self.db_path}",
                echo=self.echo,
                connect_args=connect_args,
                poolclass=StaticPool,
                pool_pre_ping=True,
                pool_recycle=3600  # 1時間でコネクションリサイクル
            )
            
            # セッションファクトリ作成
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            logger.info(f"データベースエンジン初期化完了: {self.db_path}")
            
        except Exception as e:
            logger.error(f"データベースエンジン初期化エラー: {e}")
            raise
    
    def _setup_event_listeners(self):
        """イベントリスナー設定"""
        if not self.engine:
            return
        
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """SQLite設定"""
            cursor = dbapi_connection.cursor()
            # 外部キー制約有効化
            cursor.execute("PRAGMA foreign_keys=ON")
            # WALモード有効化（並行性向上）
            cursor.execute("PRAGMA journal_mode=WAL")
            # 同期設定（パフォーマンス向上）
            cursor.execute("PRAGMA synchronous=NORMAL")
            # キャッシュサイズ設定
            cursor.execute("PRAGMA cache_size=10000")
            # 一時ストレージ設定
            cursor.execute("PRAGMA temp_store=MEMORY")
            cursor.close()
        
        @event.listens_for(self.engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            """コネクションチェックアウト時の処理"""
            logger.debug("データベースコネクションチェックアウト")
        
        @event.listens_for(self.engine, "checkin")
        def receive_checkin(dbapi_connection, connection_record):
            """コネクションチェックイン時の処理"""
            logger.debug("データベースコネクションチェックイン")
    
    def create_tables(self):
        """テーブル作成"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("データベーステーブル作成完了")
        except Exception as e:
            logger.error(f"テーブル作成エラー: {e}")
            raise
    
    def drop_tables(self):
        """テーブル削除"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("データベーステーブル削除完了")
        except Exception as e:
            logger.error(f"テーブル削除エラー: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Session:
        """セッション取得（コンテキストマネージャー）"""
        if not self.SessionLocal:
            raise RuntimeError("データベースが初期化されていません")
        
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"データベースセッションエラー: {e}")
            raise
        finally:
            session.close()
    
    def get_session_direct(self) -> Session:
        """セッション直接取得"""
        if not self.SessionLocal:
            raise RuntimeError("データベースが初期化されていません")
        return self.SessionLocal()
    
    def execute_raw_sql(self, sql: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """生SQL実行"""
        try:
            with self.get_session() as session:
                result = session.execute(text(sql), params or {})
                return result.fetchall()
        except Exception as e:
            logger.error(f"生SQL実行エラー: {e}")
            raise
    
    def get_connection_info(self) -> Dict[str, Any]:
        """接続情報取得"""
        if not self.engine:
            return {"status": "not_initialized"}
        
        try:
            with self.engine.connect() as conn:
                # データベース情報取得
                result = conn.execute(text("SELECT sqlite_version()"))
                sqlite_version = result.fetchone()[0]
                
                # テーブル一覧取得
                result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
                tables = [row[0] for row in result.fetchall()]
                
                return {
                    "status": "connected",
                    "database_path": self.db_path,
                    "sqlite_version": sqlite_version,
                    "tables": tables
                }
        except Exception as e:
            logger.error(f"接続情報取得エラー: {e}")
            return {"status": "error", "error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """ヘルスチェック"""
        try:
            with self.get_session() as session:
                # 簡単なクエリ実行
                result = session.execute(text("SELECT 1"))
                result.fetchone()
                
                return {
                    "status": "healthy",
                    "database_path": self.db_path
                }
        except Exception as e:
            logger.error(f"ヘルスチェックエラー: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def backup_database(self, backup_path: str) -> bool:
        """データベースバックアップ"""
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"データベースバックアップ完了: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"バックアップエラー: {e}")
            return False
    
    def restore_database(self, backup_path: str) -> bool:
        """データベース復元"""
        try:
            import shutil
            if not os.path.exists(backup_path):
                logger.error(f"バックアップファイルが見つかりません: {backup_path}")
                return False
            
            # 現在のデータベースをバックアップ
            import time
            current_backup = f"{self.db_path}.backup.{int(time.time())}"
            if os.path.exists(self.db_path):
                shutil.copy2(self.db_path, current_backup)
            
            # バックアップから復元
            shutil.copy2(backup_path, self.db_path)
            logger.info(f"データベース復元完了: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"復元エラー: {e}")
            return False
    
    def optimize_database(self) -> bool:
        """データベース最適化"""
        try:
            with self.get_session() as session:
                # VACUUM実行
                session.execute(text("VACUUM"))
                # ANALYZE実行
                session.execute(text("ANALYZE"))
                logger.info("データベース最適化完了")
                return True
        except Exception as e:
            logger.error(f"最適化エラー: {e}")
            return False
    
    def close(self):
        """リソースクリーンアップ"""
        if self.engine:
            self.engine.dispose()
            logger.info("データベースエンジン終了")


# グローバルデータベースマネージャーインスタンス
_database_manager: Optional[DatabaseManager] = None


def get_database_manager(db_path: str = "data/self_learning_agent.db") -> DatabaseManager:
    """データベースマネージャー取得（シングルトン）"""
    global _database_manager
    
    if _database_manager is None:
        _database_manager = DatabaseManager(db_path=db_path)
        _database_manager.create_tables()
    
    return _database_manager


def initialize_database(db_path: str = "data/self_learning_agent.db", 
                       echo: bool = False) -> DatabaseManager:
    """データベース初期化"""
    global _database_manager
    
    if _database_manager is not None:
        _database_manager.close()
    
    _database_manager = DatabaseManager(db_path=db_path, echo=echo)
    _database_manager.create_tables()
    
    return _database_manager


def close_database():
    """データベース終了"""
    global _database_manager
    
    if _database_manager is not None:
        _database_manager.close()
        _database_manager = None
