"""
Test ChromaDB + Sentence-Transformers Semantic Search System

ChromaDB + Sentence-Transformers 統合記憶検索システムのテスト
"""

import pytest
import tempfile
import shutil
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from src.advanced_agent.memory.persistent_memory import LangChainPersistentMemory
from src.advanced_agent.memory.semantic_search import (
    SentenceTransformersSearchEngine,
    ChromaDBSemanticMemory
)


class TestSentenceTransformersSearchEngine:
    """Sentence-Transformers 検索エンジンのテスト"""
    
    @pytest.fixture
    def search_engine(self):
        """テスト用検索エンジン"""
        return SentenceTransformersSearchEngine(device="cpu")
    
    def test_text_encoding(self, search_engine):
        """テキスト埋め込みテスト"""
        
        text = "これはテストテキストです"
        embedding = search_engine.encode_text(text)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1  # 1次元ベクトル
        assert embedding.shape[0] > 0  # 次元数が正の値
        
        # キャッシュテスト
        embedding2 = search_engine.encode_text(text, use_cache=True)
        np.testing.assert_array_equal(embedding, embedding2)
    
    def test_similarity_calculation(self, search_engine):
        """類似度計算テスト"""
        
        # 類似したテキスト
        text1 = "Pythonでの機械学習"
        text2 = "Pythonを使った機械学習"
        similarity_high = search_engine.calculate_similarity(text1, text2)
        
        # 異なるテキスト
        text3 = "今日の天気は晴れです"
        similarity_low = search_engine.calculate_similarity(text1, text3)
        
        assert 0.0 <= similarity_high <= 1.0
        assert 0.0 <= similarity_low <= 1.0
        assert similarity_high > similarity_low  # 類似テキストの方が高い類似度
    
    def test_importance_calculation(self, search_engine):
        """重要度計算テスト"""
        
        target_text = "重要なシステムエラーが発生しました"
        reference_texts = [
            "システムエラーの対処法",
            "重要な問題の解決方法",
            "今日の天気予報"
        ]
        
        importance = search_engine.calculate_importance_by_similarity(
            target_text, reference_texts
        )
        
        assert 0.0 <= importance <= 1.0
        
        # 参照テキストなしの場合
        importance_empty = search_engine.calculate_importance_by_similarity(
            target_text, []
        )
        assert importance_empty == 0.5
    
    def test_similar_memory_search(self, search_engine):
        """類似記憶検索テスト"""
        
        query = "Python 機械学習"
        memory_texts = [
            "Pythonでの機械学習について説明します",
            "JavaScriptの基本文法",
            "機械学習アルゴリズムの種類",
            "データベース設計の原則",
            "Pythonライブラリの使い方"
        ]
        memory_metadata = [{"id": i} for i in range(len(memory_texts))]
        
        results = search_engine.find_similar_memories(
            query=query,
            memory_texts=memory_texts,
            memory_metadata=memory_metadata,
            max_results=3
        )
        
        assert isinstance(results, list)
        assert len(results) <= 3
        
        # 結果が類似度順にソートされていることを確認
        if len(results) > 1:
            for i in range(1, len(results)):
                assert results[i-1]["similarity"] >= results[i]["similarity"]
    
    def test_memory_clustering(self, search_engine):
        """記憶クラスタリングテスト"""
        
        memory_texts = [
            "Pythonでの機械学習",
            "機械学習アルゴリズム",
            "JavaScriptの基本",
            "Web開発の手法",
            "データベース設計",
            "SQL クエリ最適化"
        ]
        
        clustering_result = search_engine.cluster_memories(
            memory_texts=memory_texts,
            n_clusters=3
        )
        
        assert "clusters" in clustering_result
        assert "representatives" in clustering_result
        assert "cluster_centers" in clustering_result
        
        # クラスタ数の確認
        assert len(clustering_result["clusters"]) <= 3
        assert len(clustering_result["representatives"]) == len(clustering_result["clusters"])


class TestChromaDBSemanticMemory:
    """ChromaDB 意味的記憶システムのテスト"""
    
    @pytest.fixture
    def temp_dir(self):
        """テスト用一時ディレクトリ"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def semantic_memory(self, temp_dir):
        """テスト用意味的記憶システム"""
        db_path = Path(temp_dir) / "test_memory.db"
        chroma_path = Path(temp_dir) / "test_chroma"
        
        # 永続化記憶システム
        persistent_memory = LangChainPersistentMemory(
            db_path=str(db_path),
            chroma_path=str(chroma_path)
        )
        
        # 検索エンジン
        search_engine = SentenceTransformersSearchEngine(device="cpu")
        
        # 意味的記憶システム
        semantic_memory = ChromaDBSemanticMemory(
            persistent_memory=persistent_memory,
            search_engine=search_engine
        )
        
        yield semantic_memory
        persistent_memory.close()
    
    @pytest.mark.asyncio
    async def test_enhanced_similarity_search(self, semantic_memory):
        """拡張類似度検索テスト"""
        
        # セッション初期化
        session_id = await semantic_memory.persistent_memory.initialize_session()
        
        # テスト用会話を追加
        conversations = [
            ("Pythonでの機械学習について", "scikit-learn, TensorFlow, PyTorchが人気です"),
            ("データベース設計の原則", "正規化とインデックスが重要です"),
            ("機械学習アルゴリズムの種類", "教師あり学習、教師なし学習、強化学習があります")
        ]
        
        for user_input, agent_response in conversations:
            await semantic_memory.persistent_memory.store_conversation(
                user_input=user_input,
                agent_response=agent_response
            )
        
        # 拡張検索実行
        search_result = await semantic_memory.enhanced_similarity_search(
            query="Python 機械学習",
            session_id=session_id,
            max_results=5
        )
        
        assert "query" in search_result
        assert "enhanced_results" in search_result
        assert "pattern_matches" in search_result
        assert "search_stats" in search_result
        
        # 検索結果の確認
        enhanced_results = search_result["enhanced_results"]
        assert isinstance(enhanced_results, list)
        
        # 結果に combined_score が含まれていることを確認
        for result in enhanced_results:
            assert "combined_score" in result
            assert "similarity" in result
            assert 0.0 <= result["combined_score"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_pattern_learning(self, semantic_memory):
        """パターン学習テスト"""
        
        query = "Python エラー解決"
        successful_results = [
            {
                "text": "ImportError の解決方法",
                "similarity": 0.8,
                "metadata": {"importance_score": 0.7}
            },
            {
                "text": "Python パッケージのインストール",
                "similarity": 0.75,
                "metadata": {"importance_score": 0.6}
            }
        ]
        
        # パターン学習
        pattern_name = await semantic_memory.learn_search_pattern(
            query=query,
            successful_results=successful_results,
            pattern_name="python_error_pattern"
        )
        
        assert pattern_name == "python_error_pattern"
        assert pattern_name in semantic_memory.learned_patterns
        
        # 学習されたパターンの確認
        pattern_data = semantic_memory.learned_patterns[pattern_name]
        assert pattern_data["query_pattern"] == query
        assert len(pattern_data["related_memories"]) == len(successful_results)
        assert "created_at" in pattern_data
        assert "success_count" in pattern_data
    
    @pytest.mark.asyncio
    async def test_session_clustering(self, semantic_memory):
        """セッションクラスタリングテスト"""
        
        # セッション初期化
        session_id = await semantic_memory.persistent_memory.initialize_session()
        
        # 多様な会話を追加
        conversations = [
            ("Python基礎", "変数と関数について"),
            ("Python応用", "クラスとモジュール"),
            ("JavaScript基礎", "DOM操作について"),
            ("JavaScript応用", "非同期処理"),
            ("データベース", "SQLの基本"),
            ("データベース設計", "正規化について")
        ]
        
        for user_input, agent_response in conversations:
            await semantic_memory.persistent_memory.store_conversation(
                user_input=user_input,
                agent_response=agent_response
            )
        
        # クラスタリング実行
        clustering_result = await semantic_memory.cluster_session_memories(
            session_id=session_id,
            n_clusters=3
        )
        
        assert "session_id" in clustering_result
        assert "clusters" in clustering_result
        assert "total_memories" in clustering_result
        assert "n_clusters" in clustering_result
        
        # クラスタの確認
        clusters = clustering_result["clusters"]
        assert len(clusters) <= 3
        
        for cluster_id, cluster_data in clusters.items():
            assert "memories" in cluster_data
            assert "representative" in cluster_data
            assert "size" in cluster_data
            assert cluster_data["size"] > 0
    
    @pytest.mark.asyncio
    async def test_memory_evolution(self, semantic_memory):
        """記憶進化検出テスト"""
        
        # セッション初期化
        session_id = await semantic_memory.persistent_memory.initialize_session()
        
        # 時系列で関連する会話を追加
        base_time = datetime.now()
        conversations_with_time = [
            (base_time - timedelta(days=5), "Python基礎学習開始", "変数について学びました"),
            (base_time - timedelta(days=3), "Python関数学習", "関数の定義と呼び出し"),
            (base_time - timedelta(days=1), "Pythonクラス学習", "オブジェクト指向プログラミング"),
            (base_time, "Python応用", "機械学習ライブラリの使用")
        ]
        
        for timestamp, user_input, agent_response in conversations_with_time:
            # タイムスタンプ付きメタデータで保存
            await semantic_memory.persistent_memory.store_conversation(
                user_input=user_input,
                agent_response=agent_response,
                metadata={"timestamp": timestamp.isoformat()}
            )
        
        # 進化パターン検出
        evolution_result = await semantic_memory.find_memory_evolution(
            topic="Python学習",
            session_id=session_id,
            time_window_days=10
        )
        
        assert "topic" in evolution_result
        assert "evolution_patterns" in evolution_result
        assert "summary" in evolution_result
        
        # 進化パターンの確認
        evolution_patterns = evolution_result["evolution_patterns"]
        assert isinstance(evolution_patterns, list)
        
        for pattern in evolution_patterns:
            assert "from_memory" in pattern
            assert "to_memory" in pattern
            assert "content_similarity" in pattern
            assert "time_difference_hours" in pattern
            assert "evolution_type" in pattern
    
    def test_search_statistics(self, semantic_memory):
        """検索統計テスト"""
        
        stats = semantic_memory.get_search_statistics()
        
        assert "search_performance" in stats
        assert "learned_patterns" in stats
        assert "model_info" in stats
        
        # 検索パフォーマンス統計
        search_perf = stats["search_performance"]
        assert "total_searches" in search_perf
        assert "successful_searches" in search_perf
        assert "success_rate" in search_perf
        
        # モデル情報
        model_info = stats["model_info"]
        assert "embedding_dimension" in model_info
        assert "similarity_threshold" in model_info
        assert "cache_size" in model_info


if __name__ == "__main__":
    pytest.main([__file__])