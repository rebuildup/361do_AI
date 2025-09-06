"""
Unit tests for database module
データベースモジュールの単体テスト
"""

import pytest
import tempfile
import os
from datetime import datetime
from pathlib import Path

from src.advanced_agent.database.connection import DatabaseManager
from src.advanced_agent.database.migrations import MigrationManager
from src.advanced_agent.database.models import (
    AgentState,
    PromptTemplate,
    TuningData,
    EvolutionCandidate,
    RewardSignal,
    Interaction,
    LearningSession,
    SystemMetrics,
    Configuration
)


class TestDatabaseManager:
    """データベースマネージャーテスト"""
    
    @pytest.fixture
    def temp_db_path(self):
        """一時データベースパス"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def db_manager(self, temp_db_path):
        """データベースマネージャー"""
        manager = DatabaseManager(db_path=temp_db_path, echo=False)
        manager.create_tables()
        yield manager
        manager.close()
    
    def test_database_initialization(self, temp_db_path):
        """データベース初期化テスト"""
        manager = DatabaseManager(db_path=temp_db_path, echo=False)
        assert manager.engine is not None
        assert manager.SessionLocal is not None
        manager.close()
    
    def test_create_tables(self, db_manager):
        """テーブル作成テスト"""
        # テーブルが作成されていることを確認
        with db_manager.get_session() as session:
            from sqlalchemy import text
            result = session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result.fetchall()]
            
            expected_tables = [
                'agent_states', 'prompt_templates', 'tuning_data',
                'evolution_candidates', 'evolution_tuning_data', 'reward_signals',
                'interactions', 'learning_sessions', 'system_metrics', 'configurations'
            ]
            
            for table in expected_tables:
                assert table in tables
    
    def test_session_context_manager(self, db_manager):
        """セッションコンテキストマネージャーテスト"""
        with db_manager.get_session() as session:
            assert session is not None
            # セッションが正常に動作することを確認
            from sqlalchemy import text
            result = session.execute(text("SELECT 1"))
            assert result.fetchone()[0] == 1
    
    def test_connection_info(self, db_manager):
        """接続情報テスト"""
        info = db_manager.get_connection_info()
        assert info["status"] == "connected"
        assert "sqlite_version" in info
        assert "tables" in info
    
    def test_health_check(self, db_manager):
        """ヘルスチェックテスト"""
        health = db_manager.health_check()
        assert health["status"] == "healthy"
        assert "connection_pool" in health


class TestDatabaseModels:
    """データベースモデルテスト"""
    
    @pytest.fixture
    def temp_db_path(self):
        """一時データベースパス"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def db_manager(self, temp_db_path):
        """データベースマネージャー"""
        manager = DatabaseManager(db_path=temp_db_path, echo=False)
        manager.create_tables()
        yield manager
        manager.close()
    
    def test_agent_state_model(self, db_manager):
        """エージェント状態モデルテスト"""
        with db_manager.get_session() as session:
            agent_state = AgentState(
                user_id="test_user",
                current_prompt_version="1.0.0",
                learning_epoch=1,
                total_interactions=10,
                reward_score=0.8
            )
            
            session.add(agent_state)
            session.commit()
            
            # 取得テスト
            retrieved = session.query(AgentState).filter_by(user_id="test_user").first()
            assert retrieved is not None
            assert retrieved.current_prompt_version == "1.0.0"
            assert retrieved.learning_epoch == 1
            assert retrieved.total_interactions == 10
            assert retrieved.reward_score == 0.8
    
    def test_prompt_template_model(self, db_manager):
        """プロンプトテンプレートモデルテスト"""
        with db_manager.get_session() as session:
            template = PromptTemplate(
                version="1.0.0",
                content="Test prompt template",
                prompt_metadata={"type": "test"},
                performance_score=0.9,
                usage_count=5
            )
            
            session.add(template)
            session.commit()
            
            # 取得テスト
            retrieved = session.query(PromptTemplate).filter_by(version="1.0.0").first()
            assert retrieved is not None
            assert retrieved.content == "Test prompt template"
            assert retrieved.performance_score == 0.9
            assert retrieved.usage_count == 5
    
    def test_tuning_data_model(self, db_manager):
        """チューニングデータモデルテスト"""
        with db_manager.get_session() as session:
            tuning_data = TuningData(
                content="Test tuning data",
                data_type="conversation",
                quality_score=0.7,
                tags=["test", "conversation"]
            )
            
            session.add(tuning_data)
            session.commit()
            
            # 取得テスト
            retrieved = session.query(TuningData).filter_by(data_type="conversation").first()
            assert retrieved is not None
            assert retrieved.content == "Test tuning data"
            assert retrieved.quality_score == 0.7
            assert "test" in retrieved.tags
    
    def test_evolution_candidate_model(self, db_manager):
        """進化候補モデルテスト"""
        with db_manager.get_session() as session:
            # プロンプトテンプレート作成
            template = PromptTemplate(
                version="1.0.0",
                content="Test template",
                performance_score=0.8
            )
            session.add(template)
            session.flush()
            
            # 進化候補作成
            candidate = EvolutionCandidate(
                parent_ids=["parent1", "parent2"],
                prompt_template_version="1.0.0",
                fitness_score=0.9,
                generation=1
            )
            
            session.add(candidate)
            session.commit()
            
            # 取得テスト
            retrieved = session.query(EvolutionCandidate).filter_by(generation=1).first()
            assert retrieved is not None
            assert retrieved.fitness_score == 0.9
            assert "parent1" in retrieved.parent_ids
    
    def test_interaction_model(self, db_manager):
        """インタラクションモデルテスト"""
        with db_manager.get_session() as session:
            # エージェント状態作成
            agent_state = AgentState(user_id="test_user")
            session.add(agent_state)
            session.flush()
            
            # インタラクション作成
            interaction = Interaction(
                session_id=agent_state.session_id,
                user_input="Test input",
                agent_response="Test response",
                processing_time=1.5,
                quality_score=0.8
            )
            
            session.add(interaction)
            session.commit()
            
            # 取得テスト
            retrieved = session.query(Interaction).filter_by(session_id=agent_state.session_id).first()
            assert retrieved is not None
            assert retrieved.user_input == "Test input"
            assert retrieved.agent_response == "Test response"
            assert retrieved.processing_time == 1.5
    
    def test_reward_signal_model(self, db_manager):
        """報酬信号モデルテスト"""
        with db_manager.get_session() as session:
            # エージェント状態とインタラクション作成
            agent_state = AgentState(user_id="test_user")
            session.add(agent_state)
            session.flush()
            
            interaction = Interaction(
                session_id=agent_state.session_id,
                user_input="Test input",
                agent_response="Test response"
            )
            session.add(interaction)
            session.flush()
            
            # 報酬信号作成
            reward = RewardSignal(
                interaction_id=interaction.id,
                reward_type="user_engagement",
                value=0.7,
                context={"test": "context"}
            )
            
            session.add(reward)
            session.commit()
            
            # 取得テスト
            retrieved = session.query(RewardSignal).filter_by(reward_type="user_engagement").first()
            assert retrieved is not None
            assert retrieved.value == 0.7
            assert retrieved.context["test"] == "context"


class TestMigrationManager:
    """マイグレーションマネージャーテスト"""
    
    @pytest.fixture
    def temp_db_path(self):
        """一時データベースパス"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def temp_migrations_dir(self):
        """一時マイグレーションディレクトリ"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def db_manager(self, temp_db_path):
        """データベースマネージャー"""
        manager = DatabaseManager(db_path=temp_db_path, echo=False)
        yield manager
        manager.close()
    
    def test_create_migrations_table(self, db_manager):
        """マイグレーション履歴テーブル作成テスト"""
        migration_manager = MigrationManager(db_manager)
        migration_manager.create_migrations_table()
        
        with db_manager.get_session() as session:
            from sqlalchemy import text
            result = session.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'"))
            assert result.fetchone() is not None
    
    def test_get_applied_migrations(self, db_manager):
        """適用済みマイグレーション取得テスト"""
        migration_manager = MigrationManager(db_manager)
        migration_manager.create_migrations_table()
        
        applied = migration_manager.get_applied_migrations()
        assert isinstance(applied, list)
    
    def test_schema_validation(self, db_manager):
        """スキーマ検証テスト"""
        migration_manager = MigrationManager(db_manager)
        
        # テーブル作成前は無効
        validation = migration_manager.validate_schema()
        assert validation["is_valid"] is False
        
        # テーブル作成後は有効
        db_manager.create_tables()
        validation = migration_manager.validate_schema()
        assert validation["is_valid"] is True


class TestDatabaseIntegration:
    """データベース統合テスト"""
    
    @pytest.fixture
    def temp_db_path(self):
        """一時データベースパス"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def db_manager(self, temp_db_path):
        """データベースマネージャー"""
        manager = DatabaseManager(db_path=temp_db_path, echo=False)
        manager.create_tables()
        yield manager
        manager.close()
    
    def test_full_workflow(self, db_manager):
        """完全ワークフローテスト"""
        with db_manager.get_session() as session:
            # 1. エージェント状態作成
            agent_state = AgentState(
                user_id="test_user",
                current_prompt_version="1.0.0"
            )
            session.add(agent_state)
            session.flush()
            
            # 2. プロンプトテンプレート作成
            template = PromptTemplate(
                version="1.0.0",
                content="Test template",
                performance_score=0.8
            )
            session.add(template)
            session.flush()
            
            # 3. チューニングデータ作成
            tuning_data = TuningData(
                content="Test data",
                data_type="conversation",
                quality_score=0.7
            )
            session.add(tuning_data)
            session.flush()
            
            # 4. インタラクション作成
            interaction = Interaction(
                session_id=agent_state.session_id,
                user_input="Test input",
                agent_response="Test response"
            )
            session.add(interaction)
            session.flush()
            
            # 5. 報酬信号作成
            reward = RewardSignal(
                interaction_id=interaction.id,
                reward_type="user_engagement",
                value=0.6
            )
            session.add(reward)
            
            # 6. 進化候補作成
            candidate = EvolutionCandidate(
                prompt_template_version="1.0.0",
                fitness_score=0.9,
                generation=1
            )
            session.add(candidate)
            
            session.commit()
            
            # データ整合性確認
            assert agent_state.session_id is not None
            assert template.version == "1.0.0"
            assert tuning_data.data_type == "conversation"
            assert interaction.session_id == agent_state.session_id
            assert reward.interaction_id == interaction.id
            assert candidate.prompt_template_version == "1.0.0"
    
    def test_foreign_key_constraints(self, db_manager):
        """外部キー制約テスト"""
        with db_manager.get_session() as session:
            # 存在しない外部キーでエラーが発生することを確認
            with pytest.raises(Exception):
                interaction = Interaction(
                    session_id="non_existent_session",
                    user_input="Test input",
                    agent_response="Test response"
                )
                session.add(interaction)
                session.commit()
    
    def test_json_field_operations(self, db_manager):
        """JSONフィールド操作テスト"""
        with db_manager.get_session() as session:
            agent_state = AgentState(
                user_id="test_user",
                performance_metrics={
                    "accuracy": 0.9,
                    "response_time": 1.5,
                    "user_satisfaction": 0.8
                }
            )
            session.add(agent_state)
            session.commit()
            
            # 取得テスト
            retrieved = session.query(AgentState).filter_by(user_id="test_user").first()
            assert retrieved.performance_metrics["accuracy"] == 0.9
            assert retrieved.performance_metrics["response_time"] == 1.5
            assert retrieved.performance_metrics["user_satisfaction"] == 0.8


if __name__ == "__main__":
    pytest.main([__file__])
