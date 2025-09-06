"""
Vector Memory System Tests

ベクトル記憶システムのテスト
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from src.advanced_agent.memory.vector_store import ChromaVectorStore
from src.advanced_agent.memory.embedding_manager import EmbeddingManager
from src.advanced_agent.memory.semantic_search import SentenceTransformersSearchEngine


class TestChromaVectorStore:
    """ChromaVectorStore のテスト"""
    
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
    
    def test_initialization(self, vector_store):
        """初期化テスト"""
        assert vector_store.collection_name == "test_memory"
        assert vector_store.embeddings is not None
        assert vector_store.collection is not None
    
    def test_add_memory(self, vector_store):
        """記憶追加テスト"""
        content = "This is a test memory"
        memory_id = vector_store.add_memory(
            content=content,
            memory_type="test",
            session_id="test_session",
            importance_score=0.8
        )
        
        assert memory_id is not None
        assert len(memory_id) > 0
    
    def test_search_similar(self, vector_store):
        """類似検索テスト"""
        # テストデータを追加
        test_memories = [
            "I love programming in Python",
            "Python is a great programming language",
            "I enjoy machine learning",
            "Deep learning is fascinating"
        ]
        
        for memory in test_memories:
            vector_store.add_memory(
                content=memory,
                memory_type="test",
                importance_score=0.7
            )
        
        # 類似検索実行
        results = vector_store.search_similar(
            query="programming language",
            n_results=2
        )
        
        assert len(results) <= 2
        if results:
            assert "content" in results[0]
            assert "similarity" in results[0]
            assert "metadata" in results[0]
    
    def test_get_memory_by_id(self, vector_store):
        """ID取得テスト"""
        content = "Test memory for ID retrieval"
        memory_id = vector_store.add_memory(
            content=content,
            memory_type="test"
        )
        
        retrieved = vector_store.get_memory_by_id(memory_id)
        
        assert retrieved is not None
        assert retrieved["content"] == content
        assert retrieved["id"] == memory_id
    
    def test_update_memory(self, vector_store):
        """記憶更新テスト"""
        content = "Original content"
        memory_id = vector_store.add_memory(
            content=content,
            memory_type="test"
        )
        
        # 更新実行
        success = vector_store.update_memory(
            memory_id=memory_id,
            content="Updated content",
            metadata={"updated": True}
        )
        
        assert success is True
        
        # 更新確認
        updated = vector_store.get_memory_by_id(memory_id)
        assert updated["content"] == "Updated content"
        assert updated["metadata"].get("updated") is True
    
    def test_delete_memory(self, vector_store):
        """記憶削除テスト"""
        content = "Memory to be deleted"
        memory_id = vector_store.add_memory(
            content=content,
            memory_type="test"
        )
        
        # 削除実行
        success = vector_store.delete_memory(memory_id)
        assert success is True
        
        # 削除確認
        deleted = vector_store.get_memory_by_id(memory_id)
        assert deleted is None
    
    def test_get_collection_stats(self, vector_store):
        """コレクション統計テスト"""
        # テストデータ追加
        vector_store.add_memory("Test 1", memory_type="type1")
        vector_store.add_memory("Test 2", memory_type="type2")
        
        stats = vector_store.get_collection_stats()
        
        assert "total_memories" in stats
        assert stats["total_memories"] >= 2
        assert "collection_name" in stats
    
    def test_backup_and_restore(self, vector_store):
        """バックアップ・復元テスト"""
        # テストデータ追加
        test_content = "Backup test memory"
        memory_id = vector_store.add_memory(
            content=test_content,
            memory_type="backup_test"
        )
        
        # バックアップ
        backup_path = Path(vector_store.persist_directory) / "backup.json"
        backup_success = vector_store.backup_collection(str(backup_path))
        assert backup_success is True
        assert backup_path.exists()
        
        # コレクションクリア
        clear_success = vector_store.clear_collection()
        assert clear_success is True
        
        # 復元
        restore_success = vector_store.restore_collection(str(backup_path))
        assert restore_success is True
        
        # 復元確認
        restored = vector_store.get_memory_by_id(memory_id)
        assert restored is not None
        assert restored["content"] == test_content


class TestEmbeddingManager:
    """EmbeddingManager のテスト"""
    
    @pytest.fixture
    def temp_dir(self):
        """一時ディレクトリを作成"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def embedding_manager(self, temp_dir):
        """EmbeddingManager インスタンスを作成"""
        return EmbeddingManager(
            device="cpu",
            cache_dir=temp_dir
        )
    
    def test_initialization(self, embedding_manager):
        """初期化テスト"""
        assert embedding_manager.model_name is not None
        assert embedding_manager.sentence_model is not None
        assert embedding_manager.device == "cpu"
    
    def test_encode_text(self, embedding_manager):
        """テキスト埋め込みテスト"""
        text = "This is a test text"
        embedding = embedding_manager.encode_text(text)
        
        assert embedding is not None
        assert len(embedding) > 0
        assert isinstance(embedding, type(embedding_manager.sentence_model.encode("test")))
    
    def test_encode_texts(self, embedding_manager):
        """複数テキスト埋め込みテスト"""
        texts = [
            "First text",
            "Second text",
            "Third text"
        ]
        
        embeddings = embedding_manager.encode_texts(texts)
        
        assert len(embeddings) == len(texts)
        for embedding in embeddings:
            assert embedding is not None
            assert len(embedding) > 0
    
    def test_calculate_similarity(self, embedding_manager):
        """類似度計算テスト"""
        text1 = "I love programming"
        text2 = "Programming is great"
        text3 = "I hate cooking"
        
        emb1 = embedding_manager.encode_text(text1)
        emb2 = embedding_manager.encode_text(text2)
        emb3 = embedding_manager.encode_text(text3)
        
        # 類似テキストの類似度は高い
        sim12 = embedding_manager.calculate_similarity(emb1, emb2)
        assert sim12 > 0.5
        
        # 非類似テキストの類似度は低い
        sim13 = embedding_manager.calculate_similarity(emb1, emb3)
        assert sim13 < sim12
    
    def test_find_most_similar(self, embedding_manager):
        """類似検索テスト"""
        query_text = "machine learning"
        candidate_texts = [
            "artificial intelligence",
            "deep learning",
            "cooking recipes",
            "neural networks"
        ]
        
        query_embedding = embedding_manager.encode_text(query_text)
        candidate_embeddings = embedding_manager.encode_texts(candidate_texts)
        
        similar_indices = embedding_manager.find_most_similar(
            query_embedding,
            candidate_embeddings,
            top_k=2
        )
        
        assert len(similar_indices) <= 2
        for idx, similarity in similar_indices:
            assert 0 <= idx < len(candidate_texts)
            assert 0 <= similarity <= 1
    
    def test_cluster_embeddings(self, embedding_manager):
        """埋め込みクラスタリングテスト"""
        texts = [
            "machine learning algorithms",
            "deep learning models",
            "cooking recipes",
            "baking techniques",
            "programming languages",
            "software development"
        ]
        
        embeddings = embedding_manager.encode_texts(texts)
        clusters = embedding_manager.cluster_embeddings(embeddings, n_clusters=3)
        
        assert "cluster_labels" in clusters
        assert "clusters" in clusters
        assert len(clusters["cluster_labels"]) == len(texts)
    
    def test_get_model_info(self, embedding_manager):
        """モデル情報取得テスト"""
        info = embedding_manager.get_model_info()
        
        assert "model_name" in info
        assert "device" in info
        assert "embedding_dimension" in info
        assert "cache_size" in info
        assert info["device"] == "cpu"
    
    def test_cache_functionality(self, embedding_manager):
        """キャッシュ機能テスト"""
        text = "Cache test text"
        
        # 初回埋め込み（キャッシュなし）
        embedding1 = embedding_manager.encode_text(text, use_cache=True)
        
        # 2回目埋め込み（キャッシュあり）
        embedding2 = embedding_manager.encode_text(text, use_cache=True)
        
        # 同じ結果になることを確認
        assert embedding_manager.calculate_similarity(embedding1, embedding2) > 0.99
        
        # キャッシュクリア
        embedding_manager.clear_cache()
        assert len(embedding_manager.embedding_cache) == 0


class TestSentenceTransformersSearchEngine:
    """SentenceTransformersSearchEngine のテスト"""
    
    @pytest.fixture
    def search_engine(self):
        """SentenceTransformersSearchEngine インスタンスを作成"""
        return SentenceTransformersSearchEngine(device="cpu")
    
    def test_initialization(self, search_engine):
        """初期化テスト"""
        assert search_engine.sentence_model is not None
        assert search_engine.device == "cpu"
        assert search_engine.similarity_threshold > 0
    
    def test_encode_text(self, search_engine):
        """テキスト埋め込みテスト"""
        text = "Test text for encoding"
        embedding = search_engine.encode_text(text)
        
        assert embedding is not None
        assert len(embedding) > 0
    
    def test_similarity_threshold(self, search_engine):
        """類似度閾値テスト"""
        # 類似度閾値の設定確認
        assert 0 < search_engine.similarity_threshold < 1
        assert 0 < search_engine.importance_threshold < 1


if __name__ == "__main__":
    pytest.main([__file__])
