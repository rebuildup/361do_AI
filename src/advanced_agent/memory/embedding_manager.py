"""
Embedding Manager for Vector Memory System

埋め込みベクトルの管理と最適化
"""

import os
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from .memory_models import MemoryItem


class EmbeddingManager:
    """埋め込みベクトル管理クラス"""
    
    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu",
                 cache_dir: str = "data/embedding_cache",
                 max_cache_size: int = 10000):
        
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        
        # ディレクトリ作成
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        # 埋め込みモデル初期化
        self.sentence_model = SentenceTransformer(
            model_name,
            device=self.device
        )
        
        # LangChain 埋め込み（互換性のため）
        self.langchain_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': self.device}
        )
        
        # 埋め込みキャッシュ
        self.embedding_cache = {}
        self.cache_file = Path(cache_dir) / "embedding_cache.json"
        
        # キャッシュ読み込み
        self._load_cache()
        
        self.logger = logging.getLogger(__name__)
    
    def _load_cache(self):
        """埋め込みキャッシュを読み込み"""
        
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    self.embedding_cache = cache_data.get('embeddings', {})
                    
                self.logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
        except Exception as e:
            self.logger.warning(f"Failed to load embedding cache: {e}")
            self.embedding_cache = {}
    
    def _save_cache(self):
        """埋め込みキャッシュを保存"""
        
        try:
            # キャッシュサイズ制限
            if len(self.embedding_cache) > self.max_cache_size:
                # 古いエントリを削除
                sorted_items = sorted(
                    self.embedding_cache.items(),
                    key=lambda x: x[1].get('timestamp', 0)
                )
                items_to_remove = len(self.embedding_cache) - self.max_cache_size
                for key, _ in sorted_items[:items_to_remove]:
                    del self.embedding_cache[key]
            
            cache_data = {
                "embeddings": self.embedding_cache,
                "timestamp": datetime.now().isoformat(),
                "model_name": self.model_name
            }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save embedding cache: {e}")
    
    def _get_text_hash(self, text: str) -> str:
        """テキストのハッシュを生成"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def encode_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """テキストを埋め込みベクトルに変換"""
        
        if not text.strip():
            return np.zeros(self.sentence_model.get_sentence_embedding_dimension())
        
        # キャッシュチェック
        if use_cache:
            text_hash = self._get_text_hash(text)
            if text_hash in self.embedding_cache:
                cached_embedding = self.embedding_cache[text_hash]
                return np.array(cached_embedding['embedding'])
        
        try:
            # 埋め込み生成
            embedding = self.sentence_model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # キャッシュに保存
            if use_cache:
                text_hash = self._get_text_hash(text)
                self.embedding_cache[text_hash] = {
                    'embedding': embedding.tolist(),
                    'timestamp': datetime.now().timestamp(),
                    'text_length': len(text)
                }
                self._save_cache()
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Failed to encode text: {e}")
            return np.zeros(self.sentence_model.get_sentence_embedding_dimension())
    
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """複数テキストを一括で埋め込みベクトルに変換"""
        
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = []
            
            for text in batch_texts:
                embedding = self.encode_text(text)
                batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def calculate_similarity(self, 
                           embedding1: np.ndarray, 
                           embedding2: np.ndarray) -> float:
        """2つの埋め込みベクトルの類似度を計算"""
        
        try:
            # コサイン類似度計算
            similarity = cosine_similarity(
                embedding1.reshape(1, -1),
                embedding2.reshape(1, -1)
            )[0][0]
            
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate similarity: {e}")
            return 0.0
    
    def find_most_similar(self,
                         query_embedding: np.ndarray,
                         candidate_embeddings: List[np.ndarray],
                         top_k: int = 5) -> List[Tuple[int, float]]:
        """最も類似した埋め込みベクトルを検索"""
        
        try:
            similarities = []
            
            for i, candidate in enumerate(candidate_embeddings):
                similarity = self.calculate_similarity(query_embedding, candidate)
                similarities.append((i, similarity))
            
            # 類似度でソート
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            self.logger.error(f"Failed to find similar embeddings: {e}")
            return []
    
    def cluster_embeddings(self,
                          embeddings: List[np.ndarray],
                          n_clusters: int = 5) -> Dict[str, Any]:
        """埋め込みベクトルをクラスタリング"""
        
        try:
            if len(embeddings) < n_clusters:
                n_clusters = len(embeddings)
            
            # 埋め込みベクトルを配列に変換
            embedding_matrix = np.vstack(embeddings)
            
            # K-means クラスタリング
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embedding_matrix)
            
            # クラスター情報を構築
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(i)
            
            return {
                "cluster_labels": cluster_labels.tolist(),
                "cluster_centers": kmeans.cluster_centers_.tolist(),
                "clusters": clusters,
                "n_clusters": n_clusters
            }
            
        except Exception as e:
            self.logger.error(f"Failed to cluster embeddings: {e}")
            return {}
    
    def reduce_dimensionality(self,
                            embeddings: List[np.ndarray],
                            target_dim: int = 128) -> List[np.ndarray]:
        """埋め込みベクトルの次元削減"""
        
        try:
            from sklearn.decomposition import PCA
            
            if not embeddings:
                return []
            
            # 埋め込みベクトルを配列に変換
            embedding_matrix = np.vstack(embeddings)
            
            # 目標次元が元の次元より大きい場合はそのまま返す
            if target_dim >= embedding_matrix.shape[1]:
                return embeddings
            
            # PCA による次元削減
            pca = PCA(n_components=target_dim)
            reduced_embeddings = pca.fit_transform(embedding_matrix)
            
            return [emb for emb in reduced_embeddings]
            
        except Exception as e:
            self.logger.error(f"Failed to reduce dimensionality: {e}")
            return embeddings
    
    def get_embedding_dimension(self) -> int:
        """埋め込みベクトルの次元数を取得"""
        
        return self.sentence_model.get_sentence_embedding_dimension()
    
    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報を取得"""
        
        return {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dimension": self.get_embedding_dimension(),
            "cache_size": len(self.embedding_cache),
            "max_cache_size": self.max_cache_size
        }
    
    def clear_cache(self):
        """埋め込みキャッシュをクリア"""
        
        self.embedding_cache = {}
        self._save_cache()
        self.logger.info("Cleared embedding cache")
    
    def optimize_cache(self):
        """埋め込みキャッシュを最適化"""
        
        try:
            current_time = datetime.now().timestamp()
            cutoff_time = current_time - (30 * 24 * 60 * 60)  # 30日前
            
            # 古いエントリを削除
            keys_to_remove = []
            for key, value in self.embedding_cache.items():
                if value.get('timestamp', 0) < cutoff_time:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.embedding_cache[key]
            
            self._save_cache()
            self.logger.info(f"Optimized cache, removed {len(keys_to_remove)} old entries")
            
        except Exception as e:
            self.logger.error(f"Failed to optimize cache: {e}")
