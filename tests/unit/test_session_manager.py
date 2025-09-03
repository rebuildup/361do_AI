"""
Test SQLAlchemy + LangChain Session Management System

SQLAlchemy + LangChain 統合セッション管理システムのテスト
"""

import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from src.advanced_agent.memory.session_manager import SQLAlchemySessionManager
from src.advanced_agent.memory.persistent_memory import LangChainPersistentMemory


class TestSQLAlchemySessionManager:
    """SQLAlchemy セッション管理システムのテスト"""
    
    @pytest.fixture
    def temp_dir(self):
        """テスト用一時ディレクトリ"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def session_manager(self, temp_dir):
        """テスト用セッション管理システム"""
        db_path = Path(temp_dir) / "test_session.db"
        
        manager = SQLAlchemySessionManager(db_path=str(db_path))
        
        yield manager
    
    @pytest.mark.asyncio
    async def test_user_creation(self, session_manager):
        """ユーザー作成テスト"""
        
        # 新規ユーザー作成
        user_id = await session_manager.create_user(
            username="test_user",
            email="test@example.com",
            preferences={"theme": "dark", "language": "ja"}
        )
        
        assert user_id is not None
        assert len(user_id) > 0
        
        # 重複ユーザー名でエラーテスト
        with pytest.raises(ValueError):
            await session_manager.create_user(
                username="test_user",  # 同じユーザー名
                email="test2@example.com"
            )
    
    @pytest.mark.asyncio
    async def test_session_creation(self, session_manager):
        """セッション作成テスト"""
        
        # ユーザー作成
        user_id = await session_manager.create_user(
            username="session_test_user",
            email="session@example.com"
        )
        
        # セッション作成
        session_id = await session_manager.create_session(
            user_id=user_id,
            session_name="Test Session",
            session_type="conversation",
            metadata={"test": True}
        )
        
        assert session_id is not None
        assert session_id in session_manager.active_sessions
        
        # セッション情報確認
        active_session = session_manager.active_sessions[session_id]
        assert active_session["user_id"] == user_id
        assert active_session["session_name"] == "Test Session"
        assert "langchain_memory" in active_session
        
        # 存在しないユーザーでエラーテスト
        with pytest.raises(ValueError):
            await session_manager.create_session(
                user_id="nonexistent_user",
                session_name="Invalid Session"
            )
    
    @pytest.mark.asyncio
    async def test_conversation_saving(self, session_manager):
        """会話保存テスト"""
        
        # ユーザーとセッション作成
        user_id = await session_manager.create_user(username="conv_test_user")
        session_id = await session_manager.create_session(user_id=user_id)
        
        # 会話保存
        conversation_id = await session_manager.save_conversation(
            session_id=session_id,
            user_input="こんにちは",
            agent_response="こんにちは！何かお手伝いできることはありますか？",
            response_time=1.5,
            importance_score=0.7,
            metadata={"test_conversation": True}
        )
        
        assert conversation_id is not None
        
        # LangChain Memory に反映されているか確認
        active_session = session_manager.active_sessions[session_id]
        langchain_memory = active_session["langchain_memory"]
        messages = langchain_memory.chat_memory.messages
        
        assert len(messages) >= 2  # ユーザー入力 + エージェント応答
        
        # 存在しないセッションでエラーテスト
        with pytest.raises(ValueError):
            await session_manager.save_conversation(
                session_id="nonexistent_session",
                user_input="test",
                agent_response="test"
            )
    
    @pytest.mark.asyncio
    async def test_session_restoration(self, session_manager):
        """セッション復元テスト"""
        
        # ユーザーとセッション作成
        user_id = await session_manager.create_user(username="restore_test_user")
        session_id = await session_manager.create_session(user_id=user_id)
        
        # 会話を追加
        await session_manager.save_conversation(
            session_id=session_id,
            user_input="復元テスト1",
            agent_response="復元テスト応答1"
        )
        await session_manager.save_conversation(
            session_id=session_id,
            user_input="復元テスト2",
            agent_response="復元テスト応答2"
        )
        
        # セッションを非アクティブにする（復元テストのため）
        if session_id in session_manager.active_sessions:
            del session_manager.active_sessions[session_id]
        
        # セッション復元
        restore_info = await session_manager.restore_session(session_id)
        
        assert restore_info["session_id"] == session_id
        assert restore_info["user_id"] == user_id
        assert restore_info["restored_conversations"] == 2
        assert session_id in session_manager.active_sessions
        
        # 復元されたLangChain Memoryの確認
        active_session = session_manager.active_sessions[session_id]
        langchain_memory = active_session["langchain_memory"]
        messages = langchain_memory.chat_memory.messages
        
        assert len(messages) >= 4  # 2つの会話 = 4つのメッセージ
    
    @pytest.mark.asyncio
    async def test_session_context(self, session_manager):
        """セッションコンテキスト取得テスト"""
        
        # ユーザーとセッション作成
        user_id = await session_manager.create_user(
            username="context_test_user",
            preferences={"test_pref": "value"}
        )
        session_id = await session_manager.create_session(user_id=user_id)
        
        # 会話追加
        await session_manager.save_conversation(
            session_id=session_id,
            user_input="コンテキストテスト",
            agent_response="コンテキスト応答"
        )
        
        # コンテキスト取得
        context = await session_manager.get_session_context(session_id)
        
        assert context["session_id"] == session_id
        assert "user_info" in context
        assert "session_info" in context
        assert "langchain_context" in context
        assert "recent_conversations" in context
        
        # ユーザー情報確認
        user_info = context["user_info"]
        assert user_info["user_id"] == user_id
        assert user_info["username"] == "context_test_user"
        assert user_info["preferences"]["test_pref"] == "value"
        
        # LangChainコンテキスト確認
        langchain_context = context["langchain_context"]
        assert langchain_context["message_count"] >= 2
        assert len(langchain_context["messages"]) >= 2
    
    @pytest.mark.asyncio
    async def test_session_state_management(self, session_manager):
        """セッション状態管理テスト"""
        
        # ユーザーとセッション作成
        user_id = await session_manager.create_user(username="state_test_user")
        session_id = await session_manager.create_session(user_id=user_id)
        
        # セッション状態保存
        state_data = {
            "current_topic": "機械学習",
            "user_preferences": {"detail_level": "advanced"},
            "context_variables": {"last_model": "GPT-4"}
        }
        
        state_id = await session_manager.save_session_state(
            session_id=session_id,
            state_type="context",
            state_data=state_data
        )
        
        assert state_id is not None
        
        # 状態更新
        updated_state_data = {
            "current_topic": "深層学習",
            "user_preferences": {"detail_level": "expert"},
            "context_variables": {"last_model": "Claude"}
        }
        
        updated_state_id = await session_manager.save_session_state(
            session_id=session_id,
            state_type="context",
            state_data=updated_state_data
        )
        
        assert updated_state_id != state_id
        
        # セッション復元で状態確認
        restore_info = await session_manager.restore_session(session_id)
        session_states = restore_info["session_states"]
        
        assert "context" in session_states
        assert session_states["context"]["current_topic"] == "深層学習"
    
    @pytest.mark.asyncio
    async def test_session_ending(self, session_manager):
        """セッション終了テスト"""
        
        # ユーザーとセッション作成
        user_id = await session_manager.create_user(username="end_test_user")
        session_id = await session_manager.create_session(user_id=user_id)
        
        # 会話追加
        await session_manager.save_conversation(
            session_id=session_id,
            user_input="終了テスト",
            agent_response="終了テスト応答"
        )
        
        # セッション終了
        end_info = await session_manager.end_session(session_id)
        
        assert end_info["session_id"] == session_id
        assert "end_time" in end_info
        assert "total_conversations" in end_info
        assert "duration_minutes" in end_info
        
        # アクティブセッションから削除されているか確認
        assert session_id not in session_manager.active_sessions
    
    @pytest.mark.asyncio
    async def test_integrity_repair(self, session_manager):
        """整合性修復テスト"""
        
        # ユーザーとセッション作成
        user_id = await session_manager.create_user(username="repair_test_user")
        session_id = await session_manager.create_session(user_id=user_id)
        
        # 会話追加
        await session_manager.save_conversation(
            session_id=session_id,
            user_input="修復テスト1",
            agent_response="修復テスト応答1"
        )
        await session_manager.save_conversation(
            session_id=session_id,
            user_input="修復テスト2",
            agent_response="修復テスト応答2"
        )
        
        # 整合性修復実行
        repair_result = await session_manager.repair_session_integrity(session_id)
        
        assert repair_result["session_id"] == session_id
        assert "repairs_performed" in repair_result
        assert "issues_found" in repair_result
        assert "repair_success" in repair_result
        
        # 存在しないセッションの修復テスト
        repair_result_invalid = await session_manager.repair_session_integrity("nonexistent_session")
        assert repair_result_invalid["repair_success"] is False
        assert "セッションが存在しません" in repair_result_invalid["issues_found"]
    
    def test_user_sessions_listing(self, session_manager):
        """ユーザーセッション一覧テスト"""
        
        # 非同期関数を同期的に実行するためのヘルパー
        import asyncio
        
        async def setup_test_data():
            # ユーザー作成
            user_id = await session_manager.create_user(username="list_test_user")
            
            # 複数セッション作成
            session1_id = await session_manager.create_session(
                user_id=user_id,
                session_name="Session 1",
                session_type="conversation"
            )
            session2_id = await session_manager.create_session(
                user_id=user_id,
                session_name="Session 2",
                session_type="task"
            )
            
            return user_id, session1_id, session2_id
        
        user_id, session1_id, session2_id = asyncio.run(setup_test_data())
        
        # セッション一覧取得
        sessions = session_manager.get_user_sessions(user_id)
        
        assert len(sessions) == 2
        assert all(session["is_active"] for session in sessions)
        assert all(session["is_currently_active"] for session in sessions)
        
        # セッション名確認
        session_names = [session["session_name"] for session in sessions]
        assert "Session 1" in session_names
        assert "Session 2" in session_names
    
    def test_system_statistics(self, session_manager):
        """システム統計テスト"""
        
        # 非同期関数を同期的に実行するためのヘルパー
        import asyncio
        
        async def setup_test_data():
            # テストデータ作成
            user_id = await session_manager.create_user(username="stats_test_user")
            session_id = await session_manager.create_session(user_id=user_id)
            await session_manager.save_conversation(
                session_id=session_id,
                user_input="統計テスト",
                agent_response="統計テスト応答"
            )
        
        asyncio.run(setup_test_data())
        
        # システム統計取得
        stats = session_manager.get_system_statistics()
        
        assert "database_stats" in stats
        assert "runtime_stats" in stats
        assert "active_sessions_memory" in stats
        assert "active_session_ids" in stats
        
        # データベース統計確認
        db_stats = stats["database_stats"]
        assert db_stats["total_users"] >= 1
        assert db_stats["total_sessions"] >= 1
        assert db_stats["total_conversations"] >= 1
        
        # ランタイム統計確認
        runtime_stats = stats["runtime_stats"]
        assert "total_sessions" in runtime_stats
        assert "active_sessions" in runtime_stats


if __name__ == "__main__":
    pytest.main([__file__])