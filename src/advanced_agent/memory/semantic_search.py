"""
ChromaDB + Sentence-Transformers Semantic Search System

ChromaDB と Sentence-Transformers を統合した意味的記憶検索システム
"""

import json
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import torch

from .persistent_memory import LangChainPersistentMemory
from .memory_models import MemoryItem, ConversationItem


class SentenceTransformersSearchEngine:
    """Sentence-Transformers による高度な意味検索エンジン"""
    
    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "auto"):
        
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Sentence-Transformers モデル
        self.sentence_model = SentenceTransformer(
            model_name,
            device=self.device
        )
        
        # 埋め込みキャッシュ
        self.embedding_cache = {}
        
        # 類似度閾値
        self.similarity_threshold = 0.7
        self.importance_threshold = 0.6
    
    def encode_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """テキストを埋め込みベクトルに変換"""
        
        if use_cache and text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # Sentence-Transformers による埋め込み生成
        embedding = self.sentence_model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        if use_cache:
            self.embedding_cache[text] = embedding
        
        return embedding
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """2つのテキスト間の類似度を計算"""
        
        embedding1 = self.encode_text(text1)
        embedding2 = self.encode_text(text2)
        
        # コサイン類似度計算
        similarity = cosine_similarity(
            embedding1.reshape(1, -1),
            embedding2.reshape(1, -1)
        )[0][0]
        
        return float(similarity)
    
    def calculate_importance_by_similarity(self, 
                                        text: str,
                                        reference_texts: List[str]) -> float:
        """参照テキストとの類似度に基づく重要度計算"""
        
        if not reference_texts:
            return 0.5  # デフォルト重要度
        
        text_embedding = self.encode_text(text)
        similarities = []
        
        for ref_text in reference_texts:
            ref_embedding = self.encode_text(ref_text)
            similarity = cosine_similarity(
                text_embedding.reshape(1, -1),
                ref_embedding.reshape(1, -1)
            )[0][0]
            similarities.append(similarity)
        
        # 最大類似度を重要度として使用
        max_similarity = max(similarities)
        
        # 類似度を重要度スコアに変換 (0.0-1.0)
        importance = min(1.0, max(0.0, max_similarity + 0.3))
        
        return importance
    
    def find_similar_memories(self,
                            query: str,
                            memory_texts: List[str],
                            memory_metadata: List[Dict[str, Any]],
                            max_results: int = 10) -> List[Dict[str, Any]]:
        """類似記憶の検索"""
        
        if not memory_texts:
            return []
        
        query_embedding = self.encode_text(query)
        similarities = []
        
        # 各記憶との類似度計算
        for i, memory_text in enumerate(memory_texts):
            memory_embedding = self.encode_text(memory_text)
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                memory_embedding.reshape(1, -1)
            )[0][0]
            
            similarities.append({
                "index": i,
                "similarity": float(similarity),
                "text": memory_text,
                "metadata": memory_metadata[i] if i < len(memory_metadata) else {}
            })
        
        # 類似度でソート
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        # 閾値以上の結果のみ返す
        filtered_results = [
            result for result in similarities 
            if result["similarity"] >= self.similarity_threshold
        ]
        
        return filtered_results[:max_results]
    
    def cluster_memories(self,
                        memory_texts: List[str],
                        n_clusters: int = 5) -> Dict[str, Any]:
        """記憶のクラスタリング"""
        
        if len(memory_texts) < n_clusters:
            n_clusters = max(1, len(memory_texts))
        
        # 埋め込み生成
        embeddings = np.array([
            self.encode_text(text) for text in memory_texts
        ])
        
        # K-means クラスタリング
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # クラスタ別に記憶を分類
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({
                "index": i,
                "text": memory_texts[i],
                "distance_to_center": float(
                    np.linalg.norm(embeddings[i] - kmeans.cluster_centers_[label])
                )
            })
        
        # 各クラスタの代表的な記憶を特定
        cluster_representatives = {}
        for label, memories in clusters.items():
            # 中心に最も近い記憶を代表とする
            representative = min(memories, key=lambda x: x["distance_to_center"])
            cluster_representatives[label] = representative
        
        return {
            "clusters": clusters,
            "representatives": cluster_representatives,
            "cluster_centers": kmeans.cluster_centers_.tolist()
        }


class ChromaDBSemanticMemory:
    """ChromaDB + Sentence-Transformers 統合記憶システム"""
    
    def __init__(self,
                 persistent_memory: LangChainPersistentMemory,
                 search_engine: Optional[SentenceTransformersSearchEngine] = None):
        
        self.persistent_memory = persistent_memory
        self.search_engine = search_engine or SentenceTransformersSearchEngine()
        
        # 経験学習パターン
        self.learned_patterns = {}
        self.pattern_usage_count = {}
        
        # 記憶統計
        self.search_stats = {
            "total_searches": 0,
            "successful_searches": 0,
            "pattern_matches": 0,
            "cluster_updates": 0
        }
    
    async def enhanced_similarity_search(self,
                                       query: str,
                                       session_id: Optional[str] = None,
                                       max_results: int = 10,
                                       use_patterns: bool = True) -> Dict[str, Any]:
        """拡張類似度検索"""
        
        self.search_stats["total_searches"] += 1
        
        # 1. 基本的な ChromaDB 検索
        basic_context = await self.persistent_memory.retrieve_relevant_context(
            query=query,
            session_id=session_id,
            max_results=max_results * 2  # より多くの候補を取得
        )
        
        similar_conversations = basic_context.get("similar_conversations", [])
        
        if not similar_conversations:
            return {
                "query": query,
                "enhanced_results": [],
                "pattern_matches": [],
                "search_stats": self.search_stats.copy()
            }
        
        # 2. Sentence-Transformers による再ランキング
        memory_texts = [conv["content"] for conv in similar_conversations]
        memory_metadata = [conv["metadata"] for conv in similar_conversations]
        
        enhanced_results = self.search_engine.find_similar_memories(
            query=query,
            memory_texts=memory_texts,
            memory_metadata=memory_metadata,
            max_results=max_results
        )
        
        # 3. 学習パターンとのマッチング
        pattern_matches = []
        if use_patterns:
            pattern_matches = await self._match_learned_patterns(query, enhanced_results)
        
        # 4. 重要度による再評価
        for result in enhanced_results:
            # 既存の重要度スコアと類似度を組み合わせ
            existing_importance = result["metadata"].get("importance_score", 0.5)
            similarity_score = result["similarity"]
            
            # 統合スコア計算
            combined_score = (existing_importance * 0.4) + (similarity_score * 0.6)
            result["combined_score"] = combined_score
        
        # 統合スコアで再ソート
        enhanced_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        if enhanced_results:
            self.search_stats["successful_searches"] += 1
        
        return {
            "query": query,
            "query_embedding_info": {
                "model": self.search_engine.sentence_model.get_sentence_embedding_dimension(),
                "similarity_threshold": self.search_engine.similarity_threshold
            },
            "enhanced_results": enhanced_results,
            "pattern_matches": pattern_matches,
            "basic_results_count": len(similar_conversations),
            "enhanced_results_count": len(enhanced_results),
            "search_stats": self.search_stats.copy()
        }
    
    async def _match_learned_patterns(self,
                                    query: str,
                                    search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """学習済みパターンとのマッチング"""
        
        pattern_matches = []
        
        for pattern_name, pattern_data in self.learned_patterns.items():
            pattern_query = pattern_data["query_pattern"]
            pattern_similarity = self.search_engine.calculate_similarity(query, pattern_query)
            
            if pattern_similarity >= 0.8:  # 高い類似度でパターンマッチ
                # パターンに関連する記憶を検索
                related_memories = []
                for result in search_results:
                    for related_text in pattern_data["related_memories"]:
                        memory_similarity = self.search_engine.calculate_similarity(
                            result["text"], related_text
                        )
                        if memory_similarity >= 0.7:
                            related_memories.append({
                                "memory": result,
                                "pattern_similarity": memory_similarity
                            })
                
                if related_memories:
                    pattern_matches.append({
                        "pattern_name": pattern_name,
                        "pattern_similarity": pattern_similarity,
                        "related_memories": related_memories,
                        "usage_count": self.pattern_usage_count.get(pattern_name, 0)
                    })
                    
                    # 使用回数更新
                    self.pattern_usage_count[pattern_name] = self.pattern_usage_count.get(pattern_name, 0) + 1
                    self.search_stats["pattern_matches"] += 1
        
        return pattern_matches
    
    async def learn_search_pattern(self,
                                 query: str,
                                 successful_results: List[Dict[str, Any]],
                                 pattern_name: Optional[str] = None) -> str:
        """検索パターンの学習"""
        
        if not successful_results:
            return ""
        
        # パターン名生成
        if pattern_name is None:
            pattern_name = f"pattern_{len(self.learned_patterns)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 成功した結果から関連記憶を抽出
        related_memories = [result["text"] for result in successful_results]
        
        # パターンデータ作成
        pattern_data = {
            "query_pattern": query,
            "related_memories": related_memories,
            "created_at": datetime.now().isoformat(),
            "success_count": len(successful_results),
            "average_similarity": np.mean([result["similarity"] for result in successful_results])
        }
        
        self.learned_patterns[pattern_name] = pattern_data
        self.pattern_usage_count[pattern_name] = 0
        
        return pattern_name
    
    async def cluster_session_memories(self,
                                     session_id: str,
                                     n_clusters: int = 5) -> Dict[str, Any]:
        """セッション記憶のクラスタリング"""
        
        # セッションの全記憶を取得
        context = await self.persistent_memory.retrieve_relevant_context(
            query="全ての記憶",
            session_id=session_id,
            max_results=100
        )
        
        conversations = context.get("similar_conversations", [])
        
        if len(conversations) < 2:
            return {
                "session_id": session_id,
                "clusters": {},
                "total_memories": len(conversations),
                "message": "クラスタリングには最低2つの記憶が必要です"
            }
        
        # テキスト抽出
        memory_texts = [conv["content"] for conv in conversations]
        
        # クラスタリング実行
        clustering_result = self.search_engine.cluster_memories(
            memory_texts=memory_texts,
            n_clusters=min(n_clusters, len(memory_texts))
        )
        
        # メタデータ付きクラスタ情報
        enhanced_clusters = {}
        for cluster_id, memories in clustering_result["clusters"].items():
            enhanced_memories = []
            for memory in memories:
                original_conv = conversations[memory["index"]]
                enhanced_memories.append({
                    **memory,
                    "metadata": original_conv.get("metadata", {}),
                    "timestamp": original_conv.get("metadata", {}).get("timestamp", "")
                })
            
            enhanced_clusters[cluster_id] = {
                "memories": enhanced_memories,
                "representative": clustering_result["representatives"][cluster_id],
                "size": len(enhanced_memories)
            }
        
        self.search_stats["cluster_updates"] += 1
        
        return {
            "session_id": session_id,
            "clusters": enhanced_clusters,
            "total_memories": len(conversations),
            "n_clusters": len(enhanced_clusters),
            "clustering_stats": {
                "model_dimension": self.search_engine.sentence_model.get_sentence_embedding_dimension(),
                "similarity_threshold": self.search_engine.similarity_threshold
            }
        }
    
    async def find_memory_evolution(self,
                                  topic: str,
                                  session_id: Optional[str] = None,
                                  time_window_days: int = 30) -> Dict[str, Any]:
        """記憶の進化・変化パターンの検出"""
        
        # 指定期間の記憶を取得
        cutoff_date = datetime.now() - timedelta(days=time_window_days)
        
        # トピック関連の記憶を検索
        search_result = await self.enhanced_similarity_search(
            query=topic,
            session_id=session_id,
            max_results=50
        )
        
        enhanced_results = search_result["enhanced_results"]
        
        # 時系列でソート
        timestamped_memories = []
        for result in enhanced_results:
            timestamp_str = result["metadata"].get("timestamp", "")
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    if timestamp >= cutoff_date:
                        timestamped_memories.append({
                            **result,
                            "parsed_timestamp": timestamp
                        })
                except ValueError:
                    continue
        
        timestamped_memories.sort(key=lambda x: x["parsed_timestamp"])
        
        # 進化パターン分析
        evolution_patterns = []
        if len(timestamped_memories) >= 2:
            for i in range(1, len(timestamped_memories)):
                prev_memory = timestamped_memories[i-1]
                curr_memory = timestamped_memories[i]
                
                # 内容の変化を分析
                content_similarity = self.search_engine.calculate_similarity(
                    prev_memory["text"],
                    curr_memory["text"]
                )
                
                time_diff = curr_memory["parsed_timestamp"] - prev_memory["parsed_timestamp"]
                
                evolution_patterns.append({
                    "from_memory": {
                        "text": prev_memory["text"][:100] + "...",
                        "timestamp": prev_memory["parsed_timestamp"].isoformat(),
                        "importance": prev_memory["metadata"].get("importance_score", 0.5)
                    },
                    "to_memory": {
                        "text": curr_memory["text"][:100] + "...",
                        "timestamp": curr_memory["parsed_timestamp"].isoformat(),
                        "importance": curr_memory["metadata"].get("importance_score", 0.5)
                    },
                    "content_similarity": content_similarity,
                    "time_difference_hours": time_diff.total_seconds() / 3600,
                    "evolution_type": self._classify_evolution_type(content_similarity)
                })
        
        return {
            "topic": topic,
            "time_window_days": time_window_days,
            "total_memories": len(timestamped_memories),
            "evolution_patterns": evolution_patterns,
            "summary": {
                "significant_changes": len([p for p in evolution_patterns if p["content_similarity"] < 0.5]),
                "gradual_changes": len([p for p in evolution_patterns if 0.5 <= p["content_similarity"] < 0.8]),
                "minor_updates": len([p for p in evolution_patterns if p["content_similarity"] >= 0.8])
            }
        }
    
    def _classify_evolution_type(self, similarity: float) -> str:
        """進化タイプの分類"""
        if similarity < 0.3:
            return "major_change"
        elif similarity < 0.6:
            return "significant_update"
        elif similarity < 0.8:
            return "gradual_evolution"
        else:
            return "minor_refinement"
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """検索統計情報"""
        
        success_rate = (
            self.search_stats["successful_searches"] / self.search_stats["total_searches"]
            if self.search_stats["total_searches"] > 0 else 0
        )
        
        return {
            "search_performance": {
                "total_searches": self.search_stats["total_searches"],
                "successful_searches": self.search_stats["successful_searches"],
                "success_rate": success_rate,
                "pattern_matches": self.search_stats["pattern_matches"],
                "cluster_updates": self.search_stats["cluster_updates"]
            },
            "learned_patterns": {
                "total_patterns": len(self.learned_patterns),
                "pattern_names": list(self.learned_patterns.keys()),
                "most_used_pattern": max(
                    self.pattern_usage_count.items(),
                    key=lambda x: x[1]
                )[0] if self.pattern_usage_count else None
            },
            "model_info": {
                "embedding_dimension": self.search_engine.sentence_model.get_sentence_embedding_dimension(),
                "similarity_threshold": self.search_engine.similarity_threshold,
                "importance_threshold": self.search_engine.importance_threshold,
                "cache_size": len(self.search_engine.embedding_cache)
            }
        }