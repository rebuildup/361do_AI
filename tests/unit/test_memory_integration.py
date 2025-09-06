"""
Memory Integration System Tests

記憶統合システムのテスト
"""

import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from src.advanced_agent.memory.memory_integrator import MemoryIntegrator
from src.advanced_agent.memory.context_builder import ContextBuilder
from src.advanced_agent.memory.vector_store import ChromaVectorStore
from src.advanced_agent.memory.embedding_manager import EmbeddingManager
from src.advanced_agent.memory.importance_evaluator import ImportanceEvaluator
from src.advanced_agent.memory.memory_cleaner import MemoryCleaner


class TestMemoryIntegrator:
    """MemoryIntegrator のテスト"""
    
    @pytest.fixture
    def temp_dir(self):
        """一時ディレクトリを作成"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def vector_store(self, temp_dir):
        """ChromaVectorStore インスタンスを作成"""
        return ChromaVectorStore(
            persist_directory=temp_dir,
            collection_name="test_memory",
            device="cpu"
        )
    
    @pytest.fixture
    def embedding_manager(self, temp_dir):
        """EmbeddingManager インスタンスを作成"""
        return EmbeddingManager(device="cpu", cache_dir=temp_dir)
    
    @pytest.fixture
    def importance_evaluator(self, embedding_manager):
        """ImportanceEvaluator インスタンスを作成"""
        return ImportanceEvaluator(embedding_manager=embedding_manager)
    
    @pytest.fixture
    def memory_cleaner(self, importance_evaluator, embedding_manager):
        """MemoryCleaner インスタンスを作成"""
        return MemoryCleaner(
            importance_evaluator=importance_evaluator,
            embedding_manager=embedding_manager
        )
    
    @pytest.fixture
    def memory_integrator(self, vector_store, embedding_manager, importance_evaluator, memory_cleaner):
        """MemoryIntegrator インスタンスを作成"""
        return MemoryIntegrator(
            vector_store=vector_store,
            embedding_manager=embedding_manager,
            importance_evaluator=importance_evaluator,
            memory_cleaner=memory_cleaner,
            short_term_capacity=50
        )
    
    def test_initialization(self, memory_integrator):
        """初期化テスト"""
        assert memory_integrator.short_term_capacity == 50
        assert memory_integrator.consolidation_threshold == 0.7
        assert memory_integrator.consolidation_interval_hours == 24
        assert len(memory_integrator.short_term_memories) == 0
        assert "conversation" in memory_integrator.memory_types
        assert "fact" in memory_integrator.memory_types
    
    def test_add_memory(self, memory_integrator):
        """記憶追加テスト"""
        content = "This is a test memory"
        memory_id = memory_integrator.add_memory(
            content=content,
            memory_type="conversation",
            session_id="test_session"
        )
        
        assert memory_id is not None
        assert len(memory_integrator.short_term_memories) == 1
        
        # 追加された記憶を確認
        added_memory = memory_integrator.short_term_memories[0]
        assert added_memory["content"] == content
        assert added_memory["metadata"]["type"] == "conversation"
        assert added_memory["metadata"]["session_id"] == "test_session"
    
    def test_add_high_importance_memory(self, memory_integrator):
        """高重要度記憶追加テスト"""
        content = "This is a critical error that needs immediate attention"
        memory_id = memory_integrator.add_memory(
            content=content,
            memory_type="error",
            session_id="test_session"
        )
        
        assert memory_id is not None
        assert len(memory_integrator.short_term_memories) == 1
        
        # 高重要度の場合は長期記憶にも追加される
        added_memory = memory_integrator.short_term_memories[0]
        assert "long_term_id" in added_memory
    
    def test_consolidate_memories(self, memory_integrator):
        """記憶統合テスト"""
        # 複数の記憶を追加
        for i in range(5):
            memory_integrator.add_memory(
                content=f"Test memory {i}",
                memory_type="conversation",
                session_id="test_session"
            )
        
        # 統合実行
        consolidation_result = memory_integrator.consolidate_memories()
        
        assert "consolidated_count" in consolidation_result
        assert "skipped_count" in consolidation_result
        assert "error_count" in consolidation_result
        assert consolidation_result["error_count"] == 0
    
    def test_auto_consolidate(self, memory_integrator):
        """自動統合テスト"""
        # 記憶を追加
        memory_integrator.add_memory(
            content="Test memory for auto consolidation",
            memory_type="conversation"
        )
        
        # 自動統合実行
        result = memory_integrator.auto_consolidate()
        
        assert "message" in result or "consolidated_count" in result
    
    def test_search_memories(self, memory_integrator):
        """記憶検索テスト"""
        # テスト記憶を追加
        test_memories = [
            "I love programming in Python",
            "Python is a great language",
            "I enjoy machine learning",
            "Deep learning is fascinating"
        ]
        
        for memory in test_memories:
            memory_integrator.add_memory(
                content=memory,
                memory_type="conversation"
            )
        
        # 検索実行
        results = memory_integrator.search_memories(
            query="programming language",
            n_results=3
        )
        
        assert isinstance(results, list)
        assert len(results) <= 3
    
    def test_get_memory_context(self, memory_integrator):
        """記憶コンテキスト取得テスト"""
        # テスト記憶を追加
        memory_integrator.add_memory(
            content="Important programming concept",
            memory_type="fact",
            session_id="test_session"
        )
        
        # コンテキスト取得
        context = memory_integrator.get_memory_context(
            query="programming",
            context_window=3
        )
        
        assert "query" in context
        assert "relevant_memories" in context
        assert "context_summary" in context
        assert "memory_types" in context
        assert "temporal_distribution" in context
    
    def test_cleanup_expired_memories(self, memory_integrator):
        """期限切れ記憶クリーンアップテスト"""
        # 古い記憶を追加
        old_memory = {
            "id": "old_memory",
            "content": "Old memory",
            "metadata": {
                "type": "temporary",
                "retention_days": 1,
                "timestamp": datetime.now().isoformat()
            },
            "added_at": datetime.now() - timedelta(days=2)
        }
        memory_integrator.short_term_memories.append(old_memory)
        
        # クリーンアップ実行
        cleanup_result = memory_integrator.cleanup_expired_memories()
        
        assert "short_term_cleaned" in cleanup_result
        assert "total_cleaned" in cleanup_result
        assert cleanup_result["short_term_cleaned"] >= 0
    
    def test_get_memory_stats(self, memory_integrator):
        """記憶統計取得テスト"""
        # テスト記憶を追加
        memory_integrator.add_memory(
            content="Test memory for stats",
            memory_type="conversation"
        )
        
        # 統計取得
        stats = memory_integrator.get_memory_stats()
        
        assert "short_term" in stats
        assert "long_term" in stats
        assert "consolidation" in stats
        assert "type_distribution" in stats
        assert "total_memories" in stats
        assert stats["short_term"]["count"] >= 1
    
    def test_update_memory_type_config(self, memory_integrator):
        """記憶タイプ設定更新テスト"""
        new_config = {
            "priority": 0.9,
            "retention_days": 60
        }
        
        memory_integrator.update_memory_type_config("custom_type", new_config)
        
        assert "custom_type" in memory_integrator.memory_types
        assert memory_integrator.memory_types["custom_type"]["priority"] == 0.9
        assert memory_integrator.memory_types["custom_type"]["retention_days"] == 60
    
    def test_get_integration_config(self, memory_integrator):
        """統合設定取得テスト"""
        config = memory_integrator.get_integration_config()
        
        assert "short_term_capacity" in config
        assert "consolidation_threshold" in config
        assert "consolidation_interval_hours" in config
        assert "memory_types" in config
        assert "last_consolidation" in config


class TestContextBuilder:
    """ContextBuilder のテスト"""
    
    @pytest.fixture
    def temp_dir(self):
        """一時ディレクトリを作成"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def memory_integrator(self, temp_dir):
        """MemoryIntegrator インスタンスを作成"""
        vector_store = ChromaVectorStore(
            persist_directory=temp_dir,
            collection_name="test_memory",
            device="cpu"
        )
        embedding_manager = EmbeddingManager(device="cpu", cache_dir=temp_dir)
        importance_evaluator = ImportanceEvaluator(embedding_manager=embedding_manager)
        memory_cleaner = MemoryCleaner(
            importance_evaluator=importance_evaluator,
            embedding_manager=embedding_manager
        )
        
        return MemoryIntegrator(
            vector_store=vector_store,
            embedding_manager=embedding_manager,
            importance_evaluator=importance_evaluator,
            memory_cleaner=memory_cleaner
        )
    
    @pytest.fixture
    def context_builder(self, memory_integrator):
        """ContextBuilder インスタンスを作成"""
        return ContextBuilder(
            memory_integrator=memory_integrator,
            max_context_length=2000
        )
    
    def test_initialization(self, context_builder):
        """初期化テスト"""
        assert context_builder.max_context_length == 2000
        assert context_builder.context_window_size == 10
        assert "relevance" in context_builder.context_weights
        assert "recency" in context_builder.context_weights
        assert "importance" in context_builder.context_weights
    
    def test_build_context(self, context_builder):
        """コンテキスト構築テスト"""
        # テスト記憶を追加
        context_builder.memory_integrator.add_memory(
            content="Important programming concept",
            memory_type="fact",
            session_id="test_session"
        )
        
        # コンテキスト構築
        context = context_builder.build_context(
            query="programming",
            session_id="test_session",
            context_type="factual"
        )
        
        assert "query" in context
        assert "session_id" in context
        assert "context_type" in context
        assert "timestamp" in context
        assert "memories" in context
        assert "summary" in context
        assert "metadata" in context
    
    def test_classify_memories_by_type(self, context_builder):
        """記憶タイプ分類テスト"""
        test_memories = [
            {
                "content": "Conversation memory",
                "metadata": {"type": "conversation"}
            },
            {
                "content": "Factual information",
                "metadata": {"type": "fact"}
            },
            {
                "content": "User preference",
                "metadata": {"type": "preference"}
            }
        ]
        
        classified = context_builder._classify_memories_by_type(test_memories)
        
        assert "conversation" in classified
        assert "factual" in classified
        assert "preference" in classified
        assert len(classified["conversation"]) == 1
        assert len(classified["factual"]) == 1
        assert len(classified["preference"]) == 1
    
    def test_select_context_memories(self, context_builder):
        """コンテキスト記憶選択テスト"""
        classified_memories = {
            "conversation": [
                {
                    "content": "Conversation 1",
                    "metadata": {"type": "conversation", "importance_score": 0.8},
                    "similarity": 0.9
                },
                {
                    "content": "Conversation 2",
                    "metadata": {"type": "conversation", "importance_score": 0.6},
                    "similarity": 0.7
                }
            ],
            "factual": [
                {
                    "content": "Fact 1",
                    "metadata": {"type": "fact", "importance_score": 0.9},
                    "similarity": 0.8
                }
            ]
        }
        
        selected = context_builder._select_context_memories(classified_memories, "conversation")
        
        assert isinstance(selected, list)
        assert len(selected) > 0
    
    def test_remove_duplicates(self, context_builder):
        """重複除去テスト"""
        test_memories = [
            {"content": "Same content", "id": "1"},
            {"content": "Different content", "id": "2"},
            {"content": "Same content", "id": "3"}  # 重複
        ]
        
        unique = context_builder._remove_duplicates(test_memories)
        
        assert len(unique) == 2
        assert unique[0]["id"] == "1"
        assert unique[1]["id"] == "2"
    
    def test_calculate_context_scores(self, context_builder):
        """コンテキストスコア計算テスト"""
        test_memories = [
            {
                "content": "Test memory",
                "metadata": {
                    "type": "conversation",
                    "importance_score": 0.8,
                    "timestamp": datetime.now().isoformat()
                },
                "similarity": 0.9
            }
        ]
        
        scored = context_builder._calculate_context_scores(test_memories)
        
        assert len(scored) == 1
        assert "context_score" in scored[0]
        assert "score_breakdown" in scored[0]
        assert 0 <= scored[0]["context_score"] <= 1
    
    def test_calculate_recency_score(self, context_builder):
        """新しさスコア計算テスト"""
        # 現在時刻
        current_time = datetime.now()
        recent_timestamp = (current_time - timedelta(hours=1)).isoformat()
        old_timestamp = (current_time - timedelta(days=30)).isoformat()
        
        # 新しい記憶
        recent_score = context_builder._calculate_recency_score(recent_timestamp)
        assert recent_score > 0.5
        
        # 古い記憶
        old_score = context_builder._calculate_recency_score(old_timestamp)
        assert old_score < recent_score
        assert 0 <= old_score <= 1
    
    def test_calculate_diversity_score(self, context_builder):
        """多様性スコア計算テスト"""
        test_memory = {
            "content": "Unique memory",
            "metadata": {"type": "fact"}
        }
        
        all_memories = [
            test_memory,
            {"content": "Another fact", "metadata": {"type": "fact"}},
            {"content": "Conversation", "metadata": {"type": "conversation"}}
        ]
        
        diversity_score = context_builder._calculate_diversity_score(test_memory, all_memories)
        
        assert 0 <= diversity_score <= 1
    
    def test_calculate_coherence_score(self, context_builder):
        """一貫性スコア計算テスト"""
        # 完全なメタデータ
        complete_memory = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "type": "conversation",
                "importance_score": 0.8,
                "session_id": "test",
                "priority": 0.7
            }
        }
        
        complete_score = context_builder._calculate_coherence_score(complete_memory)
        assert complete_score > 0.5
        
        # 不完全なメタデータ
        incomplete_memory = {
            "metadata": {
                "type": "conversation"
            }
        }
        
        incomplete_score = context_builder._calculate_coherence_score(incomplete_memory)
        assert incomplete_score < complete_score
    
    def test_limit_context_length(self, context_builder):
        """コンテキスト長制限テスト"""
        test_memories = [
            {"content": "Short memory", "context_score": 0.9},
            {"content": "A" * 1000, "context_score": 0.8},  # 長い記憶
            {"content": "Another short memory", "context_score": 0.7}
        ]
        
        limited = context_builder._limit_context_length(test_memories)
        
        assert len(limited) <= len(test_memories)
        # 長さ制限内で収まることを確認
        total_length = sum(len(mem.get("content", "")) for mem in limited)
        assert total_length <= context_builder.max_context_length
    
    def test_build_conversation_context(self, context_builder):
        """会話コンテキスト構築テスト"""
        conversation_history = [
            {"user_input": "Hello", "agent_response": "Hi there!"},
            {"user_input": "How are you?", "agent_response": "I'm doing well!"}
        ]
        
        context = context_builder.build_conversation_context(
            current_query="What's your name?",
            session_id="test_session",
            conversation_history=conversation_history
        )
        
        assert "query" in context
        assert "session_id" in context
        assert "context_type" in context
        assert "memories" in context
        assert "conversation_turns" in context
        assert "summary" in context
    
    def test_update_context_weights(self, context_builder):
        """コンテキスト重み更新テスト"""
        new_weights = {
            "relevance": 0.5,
            "recency": 0.3,
            "importance": 0.2,
            "diversity": 0.0,
            "coherence": 0.0
        }
        
        context_builder.update_context_weights(new_weights)
        
        # 重みの合計が1.0になることを確認
        total_weight = sum(context_builder.context_weights.values())
        assert abs(total_weight - 1.0) < 0.001
    
    def test_get_context_config(self, context_builder):
        """コンテキスト設定取得テスト"""
        config = context_builder.get_context_config()
        
        assert "max_context_length" in config
        assert "context_window_size" in config
        assert "context_weights" in config
        assert "context_types" in config


if __name__ == "__main__":
    pytest.main([__file__])
