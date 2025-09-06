"""
Memory Importance Evaluation Tests

記憶重要度評価システムのテスト
"""

import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from src.advanced_agent.memory.importance_evaluator import ImportanceEvaluator
from src.advanced_agent.memory.memory_cleaner import MemoryCleaner
from src.advanced_agent.memory.embedding_manager import EmbeddingManager


class TestImportanceEvaluator:
    """ImportanceEvaluator のテスト"""
    
    @pytest.fixture
    def temp_dir(self):
        """一時ディレクトリを作成"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def embedding_manager(self, temp_dir):
        """EmbeddingManager インスタンスを作成"""
        return EmbeddingManager(device="cpu", cache_dir=temp_dir)
    
    @pytest.fixture
    def importance_evaluator(self, embedding_manager):
        """ImportanceEvaluator インスタンスを作成"""
        return ImportanceEvaluator(embedding_manager=embedding_manager)
    
    def test_initialization(self, importance_evaluator):
        """初期化テスト"""
        assert importance_evaluator.importance_threshold == 0.6
        assert importance_evaluator.decay_factor == 0.1
        assert "frequency" in importance_evaluator.weights
        assert "recency" in importance_evaluator.weights
        assert "uniqueness" in importance_evaluator.weights
        assert "context_richness" in importance_evaluator.weights
        assert "user_interaction" in importance_evaluator.weights
    
    def test_calculate_frequency_score(self, importance_evaluator):
        """頻度スコア計算テスト"""
        test_memories = [
            {"content": "I love programming"},
            {"content": "Programming is great"},
            {"content": "I enjoy cooking"},
            {"content": "Cooking is fun"}
        ]
        
        # プログラミング関連のテキスト
        content = "I love programming languages"
        frequency_score = importance_evaluator.calculate_frequency_score(content, test_memories)
        
        assert 0 <= frequency_score <= 1
        assert isinstance(frequency_score, float)
    
    def test_calculate_recency_score(self, importance_evaluator):
        """新しさスコア計算テスト"""
        # 現在時刻
        current_time = datetime.now()
        recent_timestamp = (current_time - timedelta(hours=1)).isoformat()
        old_timestamp = (current_time - timedelta(days=30)).isoformat()
        
        # 新しい記憶
        recent_score = importance_evaluator.calculate_recency_score(recent_timestamp, current_time)
        assert recent_score > 0.5
        
        # 古い記憶
        old_score = importance_evaluator.calculate_recency_score(old_timestamp, current_time)
        assert old_score < recent_score
        assert 0 <= old_score <= 1
    
    def test_calculate_uniqueness_score(self, importance_evaluator):
        """独自性スコア計算テスト"""
        test_memories = [
            {"content": "I love programming"},
            {"content": "Programming is great"},
            {"content": "I enjoy cooking"}
        ]
        
        # ユニークな内容
        unique_content = "I love quantum physics"
        uniqueness_score = importance_evaluator.calculate_uniqueness_score(unique_content, test_memories)
        
        assert 0 <= uniqueness_score <= 1
        assert isinstance(uniqueness_score, float)
    
    def test_calculate_context_richness_score(self, importance_evaluator):
        """文脈豊富さスコア計算テスト"""
        # 短いテキスト
        short_text = "Hello"
        short_score = importance_evaluator.calculate_context_richness_score(short_text)
        
        # 長いテキスト（重要キーワード含む）
        long_text = "This is an important error that needs to be remembered. It's a critical bug in the system that requires immediate attention."
        long_score = importance_evaluator.calculate_context_richness_score(long_text)
        
        assert 0 <= short_score <= 1
        assert 0 <= long_score <= 1
        assert long_score > short_score  # 長いテキストの方が高スコア
    
    def test_calculate_user_interaction_score(self, importance_evaluator):
        """ユーザーインタラクションスコア計算テスト"""
        # 高インタラクション
        high_interaction_metadata = {
            "session_count": 10,
            "reference_count": 5,
            "user_rating": 0.9,
            "modification_count": 3
        }
        high_score = importance_evaluator.calculate_user_interaction_score(high_interaction_metadata)
        
        # 低インタラクション
        low_interaction_metadata = {
            "session_count": 1,
            "reference_count": 0,
            "user_rating": 0.2,
            "modification_count": 0
        }
        low_score = importance_evaluator.calculate_user_interaction_score(low_interaction_metadata)
        
        assert 0 <= high_score <= 1
        assert 0 <= low_score <= 1
        assert high_score > low_score
    
    def test_calculate_importance_score(self, importance_evaluator):
        """総合重要度スコア計算テスト"""
        test_memory = {
            "content": "This is an important error that needs to be remembered",
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "session_count": 5,
                "reference_count": 3,
                "user_rating": 0.8
            }
        }
        
        test_memories = [test_memory]
        importance_score = importance_evaluator.calculate_importance_score(test_memory, test_memories)
        
        assert 0 <= importance_score <= 1
        assert isinstance(importance_score, float)
    
    def test_evaluate_memories(self, importance_evaluator):
        """記憶評価テスト"""
        test_memories = [
            {
                "id": "mem1",
                "content": "Important error message",
                "metadata": {"timestamp": datetime.now().isoformat()}
            },
            {
                "id": "mem2",
                "content": "Hello world",
                "metadata": {"timestamp": datetime.now().isoformat()}
            }
        ]
        
        evaluated = importance_evaluator.evaluate_memories(test_memories)
        
        assert len(evaluated) == len(test_memories)
        for memory in evaluated:
            assert "importance_score" in memory["metadata"]
            assert 0 <= memory["metadata"]["importance_score"] <= 1
    
    def test_filter_important_memories(self, importance_evaluator):
        """重要記憶フィルタリングテスト"""
        test_memories = [
            {
                "id": "mem1",
                "content": "Critical error that needs attention",
                "metadata": {"timestamp": datetime.now().isoformat()}
            },
            {
                "id": "mem2",
                "content": "Hello",
                "metadata": {"timestamp": datetime.now().isoformat()}
            }
        ]
        
        important_memories = importance_evaluator.filter_important_memories(test_memories, threshold=0.3)
        
        assert len(important_memories) <= len(test_memories)
        for memory in important_memories:
            assert memory["metadata"]["importance_score"] >= 0.3
    
    def test_get_memory_insights(self, importance_evaluator):
        """記憶洞察テスト"""
        test_memories = [
            {
                "id": "mem1",
                "content": "Important information",
                "metadata": {"timestamp": datetime.now().isoformat()}
            },
            {
                "id": "mem2",
                "content": "Less important info",
                "metadata": {"timestamp": datetime.now().isoformat()}
            }
        ]
        
        insights = importance_evaluator.get_memory_insights(test_memories)
        
        assert "total_memories" in insights
        assert "average_importance" in insights
        assert "max_importance" in insights
        assert "min_importance" in insights
        assert insights["total_memories"] == len(test_memories)
    
    def test_update_importance_weights(self, importance_evaluator):
        """重要度重み更新テスト"""
        new_weights = {
            "frequency": 0.4,
            "recency": 0.3,
            "uniqueness": 0.2,
            "context_richness": 0.05,
            "user_interaction": 0.05
        }
        
        importance_evaluator.update_importance_weights(new_weights)
        
        # 重みの合計が1.0になることを確認
        total_weight = sum(importance_evaluator.weights.values())
        assert abs(total_weight - 1.0) < 0.001
    
    def test_add_important_keyword(self, importance_evaluator):
        """重要キーワード追加テスト"""
        keyword = "urgent"
        weight = 0.9
        
        importance_evaluator.add_important_keyword(keyword, weight)
        
        assert keyword in importance_evaluator.important_keywords
        assert importance_evaluator.important_keywords[keyword] == weight


class TestMemoryCleaner:
    """MemoryCleaner のテスト"""
    
    @pytest.fixture
    def temp_dir(self):
        """一時ディレクトリを作成"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
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
            embedding_manager=embedding_manager,
            max_memories=100
        )
    
    def test_initialization(self, memory_cleaner):
        """初期化テスト"""
        assert memory_cleaner.max_memories == 100
        assert memory_cleaner.cleanup_threshold == 0.3
        assert "duplicate_threshold" in memory_cleaner.cleanup_rules
        assert "age_threshold_days" in memory_cleaner.cleanup_rules
    
    def test_identify_duplicates(self, memory_cleaner):
        """重複記憶特定テスト"""
        test_memories = [
            {
                "id": "mem1",
                "content": "I love programming"
            },
            {
                "id": "mem2",
                "content": "I love programming"  # 重複
            },
            {
                "id": "mem3",
                "content": "I enjoy cooking"
            }
        ]
        
        duplicates = memory_cleaner.identify_duplicates(test_memories)
        
        assert isinstance(duplicates, list)
        # 重複が見つかった場合、グループが返される
        if duplicates:
            assert len(duplicates[0]) >= 2
    
    def test_identify_old_memories(self, memory_cleaner):
        """古い記憶特定テスト"""
        old_date = (datetime.now() - timedelta(days=100)).isoformat()
        recent_date = (datetime.now() - timedelta(days=1)).isoformat()
        
        test_memories = [
            {
                "id": "mem1",
                "content": "Old memory",
                "metadata": {"timestamp": old_date}
            },
            {
                "id": "mem2",
                "content": "Recent memory",
                "metadata": {"timestamp": recent_date}
            }
        ]
        
        old_memories = memory_cleaner.identify_old_memories(test_memories)
        
        assert "mem1" in old_memories
        assert "mem2" not in old_memories
    
    def test_identify_low_importance_memories(self, memory_cleaner):
        """低重要度記憶特定テスト"""
        test_memories = [
            {
                "id": "mem1",
                "content": "Important error message",
                "metadata": {"timestamp": datetime.now().isoformat()}
            },
            {
                "id": "mem2",
                "content": "Hello",
                "metadata": {"timestamp": datetime.now().isoformat()}
            }
        ]
        
        low_importance = memory_cleaner.identify_low_importance_memories(test_memories)
        
        assert isinstance(low_importance, list)
        # 低重要度の記憶が特定される可能性がある
    
    def test_select_memories_to_keep(self, memory_cleaner):
        """保持記憶選択テスト"""
        test_memory_group = [
            {
                "id": "mem1",
                "content": "Important memory",
                "metadata": {"importance_score": 0.9}
            },
            {
                "id": "mem2",
                "content": "Less important memory",
                "metadata": {"importance_score": 0.3}
            },
            {
                "id": "mem3",
                "content": "Another important memory",
                "metadata": {"importance_score": 0.8}
            }
        ]
        
        keep_ids = memory_cleaner.select_memories_to_keep(test_memory_group)
        
        assert isinstance(keep_ids, list)
        assert len(keep_ids) <= len(test_memory_group)
        # 重要度の高い記憶が選択される
        if len(keep_ids) > 0:
            assert "mem1" in keep_ids  # 最高重要度
    
    def test_generate_cleanup_plan(self, memory_cleaner):
        """クリーンアップ計画生成テスト"""
        test_memories = [
            {
                "id": "mem1",
                "content": "Important memory",
                "metadata": {"timestamp": datetime.now().isoformat()}
            },
            {
                "id": "mem2",
                "content": "Old memory",
                "metadata": {"timestamp": (datetime.now() - timedelta(days=100)).isoformat()}
            }
        ]
        
        cleanup_plan = memory_cleaner.generate_cleanup_plan(test_memories)
        
        assert "total_memories" in cleanup_plan
        assert "memories_to_delete" in cleanup_plan
        assert "memories_to_keep" in cleanup_plan
        assert "cleanup_summary" in cleanup_plan
        assert cleanup_plan["total_memories"] == len(test_memories)
    
    def test_execute_cleanup(self, memory_cleaner):
        """クリーンアップ実行テスト"""
        test_memories = [
            {
                "id": "mem1",
                "content": "Keep this memory",
                "metadata": {"timestamp": datetime.now().isoformat()}
            },
            {
                "id": "mem2",
                "content": "Delete this old memory",
                "metadata": {"timestamp": (datetime.now() - timedelta(days=100)).isoformat()}
            }
        ]
        
        cleanup_result = memory_cleaner.execute_cleanup(test_memories)
        
        assert "original_count" in cleanup_result
        assert "cleaned_count" in cleanup_result
        assert "deleted_count" in cleanup_result
        assert "success" in cleanup_result
        assert cleanup_result["success"] is True
    
    def test_auto_cleanup(self, memory_cleaner):
        """自動クリーンアップテスト"""
        # 上限以下の記憶
        small_memory_list = [
            {
                "id": f"mem{i}",
                "content": f"Memory {i}",
                "metadata": {"timestamp": datetime.now().isoformat()}
            }
            for i in range(50)  # 上限100以下
        ]
        
        result = memory_cleaner.auto_cleanup(small_memory_list)
        
        assert result["success"] is True
        assert result["deleted_count"] == 0
        assert result["cleaned_count"] == len(small_memory_list)
    
    def test_update_cleanup_rules(self, memory_cleaner):
        """クリーンアップルール更新テスト"""
        new_rules = {
            "duplicate_threshold": 0.9,
            "age_threshold_days": 60
        }
        
        memory_cleaner.update_cleanup_rules(new_rules)
        
        assert memory_cleaner.cleanup_rules["duplicate_threshold"] == 0.9
        assert memory_cleaner.cleanup_rules["age_threshold_days"] == 60
    
    def test_get_cleanup_stats(self, memory_cleaner):
        """クリーンアップ統計テスト"""
        test_memories = [
            {
                "id": "mem1",
                "content": "Test memory",
                "metadata": {"timestamp": datetime.now().isoformat()}
            }
        ]
        
        stats = memory_cleaner.get_cleanup_stats(test_memories)
        
        assert "current_memory_count" in stats
        assert "max_memory_limit" in stats
        assert "cleanup_needed" in stats
        assert "cleanup_plan" in stats
        assert "cleanup_rules" in stats


if __name__ == "__main__":
    pytest.main([__file__])
