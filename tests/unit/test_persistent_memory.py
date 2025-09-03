"""
Test LangChain + ChromaDB Persistent Memory System

永続的記憶システムのテスト
"""

import pytest
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

from src.advanced_agent.memory.persistent_memory import LangChainPersistentMemory
from src.advanced_agent.memory.conversation_manager import ConversationManager


class TestLangChainPersistentMemory:
    """LangChain永続的記憶システムのテスト"""
    
    @pytest.fixture
    def temp_dir(self):
        """テスト用一時ディレクトリ"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def memory_system(self, temp_dir):
        """テスト用記憶システム"""
        db_path = Path(temp_dir) / "test_memory.db"
        chroma_path = Path(temp_dir) / "test_chroma"
        
        memory = LangChainPersistentMemory(
            db_path=str(db_path),
            chroma_path=str(chroma_path),
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        yield memory
        memory.close()
    
    @pytest.mark.asyncio
    async def test_session_initialization(self, memory_system):
        """セッション初期化テスト"""
        
        # 新規セッション作成
        session_id = await memory_system.initialize_session(user_id="test_user")
        
        assert session_id is not None
        assert memory_system.current_session_id == session_id
        
        # 既存セッション復元
        session_id2 = await memory_system.initialize_session(session_id=session_id)
        assert session_id2 == session_id
    
    @pytest.mark.asyncio
    async def test_conversation_storage(self, memory_system):
        """会話保存テスト"""
        
        # セッション初期化
        session_id = await memory_system.initialize_session()
        
        # 会話保存
        user_input = "こんにちは、テストです"
        agent_response = "こんにちは！テストを開始します。"
        
        conversation_id = await memory_system.store_conversation(
            user_input=user_input,
            agent_response=agent_response,
            metadata={"test": True}
        )
        
        assert conversation_id is not None
        assert session_id in conversation_id
        
        # 統計確認
        stats = memory_system.get_memory_statistics()
        assert stats["structured_data"]["total_conversations"] >= 1
    
    @pytest.mark.asyncio
    async def test_context_retrieval(self, memory_system):
        """コンテキスト検索テスト"""
        
        # セッション初期化と会話保存
        session_id = await memory_system.initialize_session()
        
        await memory_system.store_conversation(
            user_input="Pythonのテストについて教えて",
            agent_response="Pythonのテストにはpytestが推奨されます。"
        )
        
        await memory_system.store_conversation(
            user_input="機械学習のモデル評価方法は？",
            agent_response="機械学習では交差検証やホールドアウト法が使われます。"
        )
        
        # 関連コンテキスト検索
        context = await memory_system.retrieve_relevant_context(
            query="テスト",
            session_id=session_id,
            max_results=5
        )
        
        assert "similar_conversations" in context
        assert "current_session" in context
        assert "session_summary" in context
        assert len(context["similar_conversations"]) > 0
    
    @pytest.mark.asyncio
    async def test_importance_calculation(self, memory_system):
        """重要度計算テスト"""
        
        # 短いテキスト
        short_importance = await memory_system._calculate_importance("短いテスト")
        
        # 長いテキスト
        long_text = "これは非常に重要な長いテキストです。" * 20
        long_importance = await memory_system._calculate_importance(long_text)
        
        # キーワード含有テキスト
        keyword_text = "重要な問題が発生しました。エラーの改善が必要です。"
        keyword_importance = await memory_system._calculate_importance(keyword_text)
        
        assert 0.0 <= short_importance <= 1.0
        assert 0.0 <= long_importance <= 1.0
        assert 0.0 <= keyword_importance <= 1.0
        assert keyword_importance > short_importance  # キーワードで重要度上昇
    
    @pytest.mark.asyncio
    async def test_memory_cleanup(self, memory_system):
        """記憶整理テスト"""
        
        # セッション初期化
        await memory_system.initialize_session()
        
        # 低重要度の会話を保存
        await memory_system.store_conversation(
            user_input="test",
            agent_response="test response"
        )
        
        # 統計確認
        stats_before = memory_system.get_memory_statistics()
        
        # クリーンアップ実行（閾値を高く設定して削除されないようにする）
        removed_count = await memory_system.cleanup_old_memories(
            days_threshold=0,  # 即座に古いとみなす
            importance_threshold=0.9  # 高い閾値で削除されないようにする
        )
        
        stats_after = memory_system.get_memory_statistics()
        
        # 削除されていないことを確認（重要度が閾値以下のため）
        assert stats_after["structured_data"]["total_conversations"] == stats_before["structured_data"]["total_conversations"]


class TestConversationManager:
    """会話管理システムのテスト"""
    
    @pytest.fixture
    def temp_dir(self):
        """テスト用一時ディレクトリ"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def conversation_manager(self, temp_dir):
        """テスト用会話管理システム"""
        db_path = Path(temp_dir) / "test_memory.db"
        chroma_path = Path(temp_dir) / "test_chroma"
        
        memory_system = LangChainPersistentMemory(
            db_path=str(db_path),
            chroma_path=str(chroma_path)
        )
        
        manager = ConversationManager(
            memory_system=memory_system,
            session_timeout_hours=1
        )
        
        yield manager
        memory_system.close()
    
    @pytest.mark.asyncio
    async def test_conversation_flow(self, conversation_manager):
        """会話フローテスト"""
        
        # 会話開始
        session_id = await conversation_manager.start_conversation(user_id="test_user")
        assert session_id is not None
        
        # 会話継続
        conversation_id = await conversation_manager.continue_conversation(
            session_id=session_id,
            user_input="Hello",
            agent_response="Hi there!",
            metadata={"test": True}
        )
        assert conversation_id is not None
        
        # コンテキスト取得
        context = await conversation_manager.get_conversation_context(session_id)
        assert "session_info" in context
        assert context["session_info"]["session_id"] == session_id
        
        # セッション終了
        success = await conversation_manager.end_session(session_id)
        assert success is True
    
    @pytest.mark.asyncio
    async def test_session_restoration(self, conversation_manager):
        """セッション復元テスト"""
        
        # 会話開始と継続
        session_id = await conversation_manager.start_conversation()
        await conversation_manager.continue_conversation(
            session_id=session_id,
            user_input="Test message",
            agent_response="Test response"
        )
        
        # セッション終了
        await conversation_manager.end_session(session_id)
        
        # 新しい会話で同じセッションIDを使用（復元テスト）
        await conversation_manager.continue_conversation(
            session_id=session_id,
            user_input="Restored message",
            agent_response="Restored response"
        )
        
        # コンテキスト確認
        context = await conversation_manager.get_conversation_context(session_id)
        assert len(context["current_session"]) > 0
    
    def test_session_statistics(self, conversation_manager):
        """セッション統計テスト"""
        
        stats = conversation_manager.get_session_statistics()
        
        assert "active_sessions" in stats
        assert "total_conversations" in stats
        assert "memory_statistics" in stats
        assert isinstance(stats["active_sessions"], int)
        assert isinstance(stats["total_conversations"], int)


if __name__ == "__main__":
    pytest.main([__file__])