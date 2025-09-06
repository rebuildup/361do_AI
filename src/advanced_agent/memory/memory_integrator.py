"""
Memory Integrator

短期・長期記憶の統合管理システム
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import deque

from .vector_store import ChromaVectorStore
from .embedding_manager import EmbeddingManager
from .importance_evaluator import ImportanceEvaluator
from .memory_cleaner import MemoryCleaner
from .memory_models import MemoryItem, ConversationItem


class MemoryIntegrator:
    """記憶統合管理クラス"""
    
    def __init__(self,
                 vector_store: Optional[ChromaVectorStore] = None,
                 embedding_manager: Optional[EmbeddingManager] = None,
                 importance_evaluator: Optional[ImportanceEvaluator] = None,
                 memory_cleaner: Optional[MemoryCleaner] = None,
                 short_term_capacity: int = 100,
                 consolidation_threshold: float = 0.7,
                 consolidation_interval_hours: int = 24):
        
        # コンポーネント初期化
        self.vector_store = vector_store or ChromaVectorStore()
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.importance_evaluator = importance_evaluator or ImportanceEvaluator()
        self.memory_cleaner = memory_cleaner or MemoryCleaner()
        
        # 短期記憶設定
        self.short_term_capacity = short_term_capacity
        self.short_term_memories = deque(maxlen=short_term_capacity)
        
        # 統合設定
        self.consolidation_threshold = consolidation_threshold
        self.consolidation_interval_hours = consolidation_interval_hours
        self.last_consolidation = datetime.now()
        
        # 記憶タイプ定義
        self.memory_types = {
            "conversation": {"priority": 0.8, "retention_days": 30},
            "fact": {"priority": 0.9, "retention_days": 90},
            "preference": {"priority": 0.95, "retention_days": 180},
            "skill": {"priority": 0.85, "retention_days": 120},
            "error": {"priority": 0.9, "retention_days": 60},
            "goal": {"priority": 0.95, "retention_days": 365},
            "temporary": {"priority": 0.3, "retention_days": 7}
        }
        
        self.logger = logging.getLogger(__name__)
    
    def add_memory(self,
                   content: str,
                   memory_type: str = "conversation",
                   session_id: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   immediate_consolidation: bool = False) -> str:
        """記憶を追加"""
        
        try:
            # メタデータ構築
            full_metadata = {
                "type": memory_type,
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "priority": self.memory_types.get(memory_type, {}).get("priority", 0.5),
                "retention_days": self.memory_types.get(memory_type, {}).get("retention_days", 30),
                **(metadata or {})
            }
            
            # 重要度評価
            temp_memory = {
                "content": content,
                "metadata": full_metadata
            }
            importance_score = self.importance_evaluator.calculate_importance_score(
                temp_memory, [temp_memory]
            )
            full_metadata["importance_score"] = importance_score
            
            # 短期記憶に追加
            memory_item = {
                "id": f"short_{datetime.now().timestamp()}_{hash(content) % 10000}",
                "content": content,
                "metadata": full_metadata,
                "added_at": datetime.now()
            }
            
            self.short_term_memories.append(memory_item)
            
            # 高重要度または即座統合の場合は長期記憶にも追加
            if importance_score >= self.consolidation_threshold or immediate_consolidation:
                long_term_id = self.vector_store.add_memory(
                    content=content,
                    memory_type=memory_type,
                    session_id=session_id,
                    importance_score=importance_score,
                    metadata=full_metadata
                )
                memory_item["long_term_id"] = long_term_id
            
            self.logger.info(f"Added memory: {memory_item['id']} (importance: {importance_score:.3f})")
            return memory_item["id"]
            
        except Exception as e:
            self.logger.error(f"Failed to add memory: {e}")
            raise
    
    def consolidate_memories(self) -> Dict[str, Any]:
        """短期記憶を長期記憶に統合"""
        
        try:
            consolidation_result = {
                "consolidated_count": 0,
                "skipped_count": 0,
                "error_count": 0,
                "consolidated_memories": [],
                "skipped_memories": [],
                "errors": []
            }
            
            # 統合対象の記憶を特定
            memories_to_consolidate = []
            memories_to_skip = []
            
            for memory in self.short_term_memories:
                importance_score = memory["metadata"].get("importance_score", 0)
                
                if importance_score >= self.consolidation_threshold:
                    memories_to_consolidate.append(memory)
                else:
                    memories_to_skip.append(memory)
            
            # 長期記憶に統合
            for memory in memories_to_consolidate:
                try:
                    long_term_id = self.vector_store.add_memory(
                        content=memory["content"],
                        memory_type=memory["metadata"].get("type", "conversation"),
                        session_id=memory["metadata"].get("session_id"),
                        importance_score=memory["metadata"].get("importance_score", 0),
                        metadata=memory["metadata"]
                    )
                    
                    memory["long_term_id"] = long_term_id
                    consolidation_result["consolidated_memories"].append(memory)
                    consolidation_result["consolidated_count"] += 1
                    
                except Exception as e:
                    consolidation_result["errors"].append({
                        "memory_id": memory["id"],
                        "error": str(e)
                    })
                    consolidation_result["error_count"] += 1
            
            # スキップされた記憶
            consolidation_result["skipped_memories"] = memories_to_skip
            consolidation_result["skipped_count"] = len(memories_to_skip)
            
            # 統合時刻を更新
            self.last_consolidation = datetime.now()
            
            self.logger.info(f"Memory consolidation completed: {consolidation_result['consolidated_count']} consolidated, {consolidation_result['skipped_count']} skipped")
            
            return consolidation_result
            
        except Exception as e:
            self.logger.error(f"Failed to consolidate memories: {e}")
            return {"error": str(e)}
    
    def auto_consolidate(self) -> Dict[str, Any]:
        """自動統合チェック"""
        
        try:
            current_time = datetime.now()
            time_since_last = current_time - self.last_consolidation
            
            # 統合間隔をチェック
            if time_since_last.total_seconds() < self.consolidation_interval_hours * 3600:
                return {
                    "message": "Consolidation not needed yet",
                    "time_since_last_hours": time_since_last.total_seconds() / 3600,
                    "consolidation_interval_hours": self.consolidation_interval_hours
                }
            
            # 統合実行
            return self.consolidate_memories()
            
        except Exception as e:
            self.logger.error(f"Failed to auto consolidate: {e}")
            return {"error": str(e)}
    
    def search_memories(self,
                       query: str,
                       search_scope: str = "all",
                       n_results: int = 10,
                       min_similarity: float = 0.0) -> List[Dict[str, Any]]:
        """記憶を検索"""
        
        try:
            results = []
            
            # 短期記憶を検索
            if search_scope in ["all", "short_term"]:
                short_term_results = self._search_short_term_memories(
                    query, n_results, min_similarity
                )
                results.extend(short_term_results)
            
            # 長期記憶を検索
            if search_scope in ["all", "long_term"]:
                long_term_results = self.vector_store.search_similar(
                    query=query,
                    n_results=n_results,
                    min_similarity=min_similarity
                )
                results.extend(long_term_results)
            
            # 類似度でソート
            results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
            
            # 結果数を制限
            return results[:n_results]
            
        except Exception as e:
            self.logger.error(f"Failed to search memories: {e}")
            return []
    
    def _search_short_term_memories(self,
                                   query: str,
                                   n_results: int,
                                   min_similarity: float) -> List[Dict[str, Any]]:
        """短期記憶を検索"""
        
        try:
            query_embedding = self.embedding_manager.encode_text(query)
            results = []
            
            for memory in self.short_term_memories:
                content = memory.get("content", "")
                content_embedding = self.embedding_manager.encode_text(content)
                
                similarity = self.embedding_manager.calculate_similarity(
                    query_embedding, content_embedding
                )
                
                if similarity >= min_similarity:
                    result = {
                        "content": content,
                        "metadata": memory.get("metadata", {}),
                        "similarity": similarity,
                        "id": memory.get("id"),
                        "source": "short_term"
                    }
                    results.append(result)
            
            # 類似度でソート
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            return results[:n_results]
            
        except Exception as e:
            self.logger.error(f"Failed to search short term memories: {e}")
            return []
    
    def get_memory_context(self,
                          query: str,
                          context_window: int = 5) -> Dict[str, Any]:
        """記憶コンテキストを構築"""
        
        try:
            # 関連記憶を検索
            relevant_memories = self.search_memories(
                query=query,
                n_results=context_window * 2,
                min_similarity=0.3
            )
            
            # コンテキストを構築
            context = {
                "query": query,
                "relevant_memories": relevant_memories[:context_window],
                "total_found": len(relevant_memories),
                "context_summary": self._generate_context_summary(relevant_memories),
                "memory_types": self._get_memory_type_distribution(relevant_memories),
                "temporal_distribution": self._get_temporal_distribution(relevant_memories)
            }
            
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to get memory context: {e}")
            return {"error": str(e)}
    
    def _generate_context_summary(self, memories: List[Dict[str, Any]]) -> str:
        """コンテキストサマリーを生成"""
        
        try:
            if not memories:
                return "No relevant memories found."
            
            # 高重要度の記憶を抽出
            high_importance = [
                mem for mem in memories
                if mem.get("metadata", {}).get("importance_score", 0) > 0.7
            ]
            
            if high_importance:
                return f"Found {len(high_importance)} highly important memories related to the query."
            else:
                return f"Found {len(memories)} relevant memories with moderate importance."
                
        except Exception as e:
            self.logger.error(f"Failed to generate context summary: {e}")
            return "Error generating context summary."
    
    def _get_memory_type_distribution(self, memories: List[Dict[str, Any]]) -> Dict[str, int]:
        """記憶タイプ分布を取得"""
        
        try:
            type_distribution = {}
            for memory in memories:
                memory_type = memory.get("metadata", {}).get("type", "unknown")
                type_distribution[memory_type] = type_distribution.get(memory_type, 0) + 1
            
            return type_distribution
            
        except Exception as e:
            self.logger.error(f"Failed to get memory type distribution: {e}")
            return {}
    
    def _get_temporal_distribution(self, memories: List[Dict[str, Any]]) -> Dict[str, int]:
        """時間分布を取得"""
        
        try:
            temporal_distribution = {
                "recent": 0,    # 1日以内
                "recent_week": 0,  # 1週間以内
                "recent_month": 0,  # 1ヶ月以内
                "older": 0      # 1ヶ月以上前
            }
            
            current_time = datetime.now()
            
            for memory in memories:
                timestamp_str = memory.get("metadata", {}).get("timestamp")
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        time_diff = current_time - timestamp
                        
                        if time_diff.days == 0:
                            temporal_distribution["recent"] += 1
                        elif time_diff.days <= 7:
                            temporal_distribution["recent_week"] += 1
                        elif time_diff.days <= 30:
                            temporal_distribution["recent_month"] += 1
                        else:
                            temporal_distribution["older"] += 1
                    except Exception:
                        temporal_distribution["older"] += 1
            
            return temporal_distribution
            
        except Exception as e:
            self.logger.error(f"Failed to get temporal distribution: {e}")
            return {}
    
    def cleanup_expired_memories(self) -> Dict[str, Any]:
        """期限切れ記憶をクリーンアップ"""
        
        try:
            cleanup_result = {
                "short_term_cleaned": 0,
                "long_term_cleaned": 0,
                "total_cleaned": 0
            }
            
            current_time = datetime.now()
            
            # 短期記憶のクリーンアップ
            memories_to_remove = []
            for memory in self.short_term_memories:
                retention_days = memory["metadata"].get("retention_days", 30)
                added_at = memory.get("added_at", current_time)
                
                if (current_time - added_at).days > retention_days:
                    memories_to_remove.append(memory)
            
            for memory in memories_to_remove:
                self.short_term_memories.remove(memory)
                cleanup_result["short_term_cleaned"] += 1
            
            # 長期記憶のクリーンアップ（ベクトルストアで実装）
            # ここでは統計のみ返す
            
            cleanup_result["total_cleaned"] = cleanup_result["short_term_cleaned"] + cleanup_result["long_term_cleaned"]
            
            self.logger.info(f"Memory cleanup completed: {cleanup_result['total_cleaned']} memories cleaned")
            
            return cleanup_result
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired memories: {e}")
            return {"error": str(e)}
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """記憶統計を取得"""
        
        try:
            # 短期記憶統計
            short_term_stats = {
                "count": len(self.short_term_memories),
                "capacity": self.short_term_capacity,
                "utilization": len(self.short_term_memories) / self.short_term_capacity
            }
            
            # 長期記憶統計
            long_term_stats = self.vector_store.get_collection_stats()
            
            # 統合統計
            consolidation_stats = {
                "last_consolidation": self.last_consolidation.isoformat(),
                "consolidation_threshold": self.consolidation_threshold,
                "consolidation_interval_hours": self.consolidation_interval_hours
            }
            
            # 記憶タイプ統計
            type_stats = {}
            for memory in self.short_term_memories:
                memory_type = memory["metadata"].get("type", "unknown")
                type_stats[memory_type] = type_stats.get(memory_type, 0) + 1
            
            return {
                "short_term": short_term_stats,
                "long_term": long_term_stats,
                "consolidation": consolidation_stats,
                "type_distribution": type_stats,
                "total_memories": short_term_stats["count"] + long_term_stats.get("total_memories", 0)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}
    
    def update_memory_type_config(self, memory_type: str, config: Dict[str, Any]):
        """記憶タイプ設定を更新"""
        
        try:
            if memory_type in self.memory_types:
                self.memory_types[memory_type].update(config)
            else:
                self.memory_types[memory_type] = config
            
            self.logger.info(f"Updated memory type config for {memory_type}: {config}")
            
        except Exception as e:
            self.logger.error(f"Failed to update memory type config: {e}")
    
    def get_integration_config(self) -> Dict[str, Any]:
        """統合設定を取得"""
        
        return {
            "short_term_capacity": self.short_term_capacity,
            "consolidation_threshold": self.consolidation_threshold,
            "consolidation_interval_hours": self.consolidation_interval_hours,
            "memory_types": self.memory_types,
            "last_consolidation": self.last_consolidation.isoformat()
        }
