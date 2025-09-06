"""
Context Builder

記憶からコンテキストを構築するシステム
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from .memory_integrator import MemoryIntegrator
from .embedding_manager import EmbeddingManager
from .memory_models import MemoryItem, ConversationItem


class ContextBuilder:
    """コンテキスト構築クラス"""
    
    def __init__(self,
                 memory_integrator: Optional[MemoryIntegrator] = None,
                 embedding_manager: Optional[EmbeddingManager] = None,
                 max_context_length: int = 4000,
                 context_window_size: int = 10):
        
        self.memory_integrator = memory_integrator or MemoryIntegrator()
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.max_context_length = max_context_length
        self.context_window_size = context_window_size
        
        # コンテキスト構築設定
        self.context_weights = {
            "relevance": 0.4,      # 関連性
            "recency": 0.2,        # 新しさ
            "importance": 0.2,     # 重要度
            "diversity": 0.1,      # 多様性
            "coherence": 0.1       # 一貫性
        }
        
        # コンテキストタイプ
        self.context_types = {
            "conversation": {"max_items": 20, "weight": 0.8},
            "factual": {"max_items": 15, "weight": 0.9},
            "preference": {"max_items": 10, "weight": 0.95},
            "procedural": {"max_items": 12, "weight": 0.85},
            "temporal": {"max_items": 8, "weight": 0.7}
        }
        
        self.logger = logging.getLogger(__name__)
    
    def build_context(self,
                     query: str,
                     session_id: Optional[str] = None,
                     context_type: str = "general",
                     include_metadata: bool = True) -> Dict[str, Any]:
        """コンテキストを構築"""
        
        try:
            # 関連記憶を検索
            relevant_memories = self.memory_integrator.search_memories(
                query=query,
                n_results=self.context_window_size * 2,
                min_similarity=0.3
            )
            
            # コンテキストタイプ別に記憶を分類
            classified_memories = self._classify_memories_by_type(relevant_memories)
            
            # コンテキストを構築
            context = {
                "query": query,
                "session_id": session_id,
                "context_type": context_type,
                "timestamp": datetime.now().isoformat(),
                "memories": self._select_context_memories(classified_memories, context_type),
                "summary": self._generate_context_summary(relevant_memories),
                "metadata": {
                    "total_found": len(relevant_memories),
                    "selected_count": 0,
                    "context_length": 0,
                    "memory_types": self._get_memory_type_distribution(relevant_memories),
                    "temporal_span": self._get_temporal_span(relevant_memories)
                }
            }
            
            # メタデータ計算
            context["metadata"]["selected_count"] = len(context["memories"])
            context["metadata"]["context_length"] = self._calculate_context_length(context["memories"])
            
            # メタデータを含めるかどうか
            if not include_metadata:
                context.pop("metadata", None)
            
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to build context: {e}")
            return {"error": str(e)}
    
    def _classify_memories_by_type(self, memories: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """記憶をタイプ別に分類"""
        
        try:
            classified = defaultdict(list)
            
            for memory in memories:
                memory_type = memory.get("metadata", {}).get("type", "conversation")
                
                # タイプ別分類
                if memory_type in ["conversation", "chat"]:
                    classified["conversation"].append(memory)
                elif memory_type in ["fact", "knowledge", "information"]:
                    classified["factual"].append(memory)
                elif memory_type in ["preference", "setting", "config"]:
                    classified["preference"].append(memory)
                elif memory_type in ["skill", "procedure", "method"]:
                    classified["procedural"].append(memory)
                elif memory_type in ["temporal", "event", "history"]:
                    classified["temporal"].append(memory)
                else:
                    classified["conversation"].append(memory)  # デフォルト
            
            return dict(classified)
            
        except Exception as e:
            self.logger.error(f"Failed to classify memories: {e}")
            return {}
    
    def _select_context_memories(self,
                                classified_memories: Dict[str, List[Dict[str, Any]]],
                                context_type: str) -> List[Dict[str, Any]]:
        """コンテキスト用の記憶を選択"""
        
        try:
            selected_memories = []
            
            # コンテキストタイプに応じて記憶を選択
            if context_type == "conversation":
                # 会話コンテキスト：会話記憶を優先
                selected_memories.extend(
                    self._select_top_memories(classified_memories.get("conversation", []), 15)
                )
                selected_memories.extend(
                    self._select_top_memories(classified_memories.get("preference", []), 5)
                )
            
            elif context_type == "factual":
                # 事実コンテキスト：事実記憶を優先
                selected_memories.extend(
                    self._select_top_memories(classified_memories.get("factual", []), 12)
                )
                selected_memories.extend(
                    self._select_top_memories(classified_memories.get("procedural", []), 8)
                )
            
            elif context_type == "preference":
                # 設定コンテキスト：設定記憶を優先
                selected_memories.extend(
                    self._select_top_memories(classified_memories.get("preference", []), 10)
                )
                selected_memories.extend(
                    self._select_top_memories(classified_memories.get("conversation", []), 5)
                )
            
            else:
                # 一般コンテキスト：バランス良く選択
                for memory_type, memories in classified_memories.items():
                    max_items = self.context_types.get(memory_type, {}).get("max_items", 5)
                    selected_memories.extend(
                        self._select_top_memories(memories, max_items)
                    )
            
            # 重複除去とスコア計算
            unique_memories = self._remove_duplicates(selected_memories)
            scored_memories = self._calculate_context_scores(unique_memories)
            
            # スコアでソート
            scored_memories.sort(key=lambda x: x.get("context_score", 0), reverse=True)
            
            # 最大長制限
            return self._limit_context_length(scored_memories)
            
        except Exception as e:
            self.logger.error(f"Failed to select context memories: {e}")
            return []
    
    def _select_top_memories(self, memories: List[Dict[str, Any]], max_count: int) -> List[Dict[str, Any]]:
        """上位記憶を選択"""
        
        try:
            if not memories:
                return []
            
            # 重要度と類似度でソート
            sorted_memories = sorted(
                memories,
                key=lambda x: (
                    x.get("metadata", {}).get("importance_score", 0) * 0.6 +
                    x.get("similarity", 0) * 0.4
                ),
                reverse=True
            )
            
            return sorted_memories[:max_count]
            
        except Exception as e:
            self.logger.error(f"Failed to select top memories: {e}")
            return memories[:max_count] if memories else []
    
    def _remove_duplicates(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """重複記憶を除去"""
        
        try:
            unique_memories = []
            seen_contents = set()
            
            for memory in memories:
                content = memory.get("content", "")
                content_hash = hash(content)
                
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    unique_memories.append(memory)
            
            return unique_memories
            
        except Exception as e:
            self.logger.error(f"Failed to remove duplicates: {e}")
            return memories
    
    def _calculate_context_scores(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """コンテキストスコアを計算"""
        
        try:
            scored_memories = []
            
            for memory in memories:
                # 各要素のスコア計算
                relevance_score = memory.get("similarity", 0)
                
                # 新しさスコア
                timestamp_str = memory.get("metadata", {}).get("timestamp")
                recency_score = self._calculate_recency_score(timestamp_str)
                
                # 重要度スコア
                importance_score = memory.get("metadata", {}).get("importance_score", 0)
                
                # 多様性スコア（他の記憶との違い）
                diversity_score = self._calculate_diversity_score(memory, memories)
                
                # 一貫性スコア（メタデータの完全性）
                coherence_score = self._calculate_coherence_score(memory)
                
                # 総合スコア
                context_score = (
                    relevance_score * self.context_weights["relevance"] +
                    recency_score * self.context_weights["recency"] +
                    importance_score * self.context_weights["importance"] +
                    diversity_score * self.context_weights["diversity"] +
                    coherence_score * self.context_weights["coherence"]
                )
                
                memory_copy = memory.copy()
                memory_copy["context_score"] = context_score
                memory_copy["score_breakdown"] = {
                    "relevance": relevance_score,
                    "recency": recency_score,
                    "importance": importance_score,
                    "diversity": diversity_score,
                    "coherence": coherence_score
                }
                
                scored_memories.append(memory_copy)
            
            return scored_memories
            
        except Exception as e:
            self.logger.error(f"Failed to calculate context scores: {e}")
            return memories
    
    def _calculate_recency_score(self, timestamp_str: Optional[str]) -> float:
        """新しさスコアを計算"""
        
        try:
            if not timestamp_str:
                return 0.0
            
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            current_time = datetime.now()
            time_diff = (current_time - timestamp).total_seconds() / (24 * 60 * 60)  # 日数
            
            # 指数減衰
            recency_score = max(0, 1 - (time_diff / 30))  # 30日で0になる
            return min(recency_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate recency score: {e}")
            return 0.0
    
    def _calculate_diversity_score(self,
                                  memory: Dict[str, Any],
                                  all_memories: List[Dict[str, Any]]) -> float:
        """多様性スコアを計算"""
        
        try:
            if len(all_memories) <= 1:
                return 1.0
            
            memory_type = memory.get("metadata", {}).get("type", "unknown")
            same_type_count = sum(
                1 for m in all_memories
                if m.get("metadata", {}).get("type", "unknown") == memory_type
            )
            
            # 同じタイプが少ないほど多様性が高い
            diversity_score = 1 - (same_type_count / len(all_memories))
            return max(diversity_score, 0.0)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate diversity score: {e}")
            return 0.0
    
    def _calculate_coherence_score(self, memory: Dict[str, Any]) -> float:
        """一貫性スコアを計算"""
        
        try:
            metadata = memory.get("metadata", {})
            coherence_score = 0.0
            
            # 必要なメタデータの存在チェック
            required_fields = ["timestamp", "type", "importance_score"]
            for field in required_fields:
                if field in metadata and metadata[field] is not None:
                    coherence_score += 0.3
            
            # 追加メタデータの存在
            optional_fields = ["session_id", "priority", "retention_days"]
            for field in optional_fields:
                if field in metadata and metadata[field] is not None:
                    coherence_score += 0.1
            
            return min(coherence_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate coherence score: {e}")
            return 0.0
    
    def _limit_context_length(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """コンテキスト長を制限"""
        
        try:
            limited_memories = []
            current_length = 0
            
            for memory in memories:
                content_length = len(memory.get("content", ""))
                
                if current_length + content_length <= self.max_context_length:
                    limited_memories.append(memory)
                    current_length += content_length
                else:
                    break
            
            return limited_memories
            
        except Exception as e:
            self.logger.error(f"Failed to limit context length: {e}")
            return memories
    
    def _generate_context_summary(self, memories: List[Dict[str, Any]]) -> str:
        """コンテキストサマリーを生成"""
        
        try:
            if not memories:
                return "No relevant memories found."
            
            # 高重要度の記憶数
            high_importance_count = sum(
                1 for memory in memories
                if memory.get("metadata", {}).get("importance_score", 0) > 0.7
            )
            
            # 記憶タイプ分布
            type_distribution = self._get_memory_type_distribution(memories)
            main_types = sorted(type_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
            
            summary_parts = [
                f"Found {len(memories)} relevant memories",
                f"({high_importance_count} highly important)"
            ]
            
            if main_types:
                type_names = [f"{count} {type_name}" for type_name, count in main_types]
                summary_parts.append(f"Types: {', '.join(type_names)}")
            
            return ". ".join(summary_parts) + "."
            
        except Exception as e:
            self.logger.error(f"Failed to generate context summary: {e}")
            return "Error generating context summary."
    
    def _get_memory_type_distribution(self, memories: List[Dict[str, Any]]) -> Dict[str, int]:
        """記憶タイプ分布を取得"""
        
        try:
            type_distribution = defaultdict(int)
            for memory in memories:
                memory_type = memory.get("metadata", {}).get("type", "unknown")
                type_distribution[memory_type] += 1
            
            return dict(type_distribution)
            
        except Exception as e:
            self.logger.error(f"Failed to get memory type distribution: {e}")
            return {}
    
    def _get_temporal_span(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """時間スパンを取得"""
        
        try:
            if not memories:
                return {"span_days": 0, "oldest": None, "newest": None}
            
            timestamps = []
            for memory in memories:
                timestamp_str = memory.get("metadata", {}).get("timestamp")
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        timestamps.append(timestamp)
                    except Exception:
                        continue
            
            if not timestamps:
                return {"span_days": 0, "oldest": None, "newest": None}
            
            oldest = min(timestamps)
            newest = max(timestamps)
            span_days = (newest - oldest).days
            
            return {
                "span_days": span_days,
                "oldest": oldest.isoformat(),
                "newest": newest.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get temporal span: {e}")
            return {"span_days": 0, "oldest": None, "newest": None}
    
    def _calculate_context_length(self, memories: List[Dict[str, Any]]) -> int:
        """コンテキスト長を計算"""
        
        try:
            total_length = 0
            for memory in memories:
                total_length += len(memory.get("content", ""))
            
            return total_length
            
        except Exception as e:
            self.logger.error(f"Failed to calculate context length: {e}")
            return 0
    
    def build_conversation_context(self,
                                  current_query: str,
                                  session_id: str,
                                  conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """会話コンテキストを構築"""
        
        try:
            # 会話履歴から関連記憶を検索
            context_memories = []
            
            for turn in conversation_history[-5:]:  # 直近5ターン
                user_input = turn.get("user_input", "")
                if user_input:
                    memories = self.memory_integrator.search_memories(
                        query=user_input,
                        n_results=3,
                        min_similarity=0.4
                    )
                    context_memories.extend(memories)
            
            # 現在のクエリからも検索
            current_memories = self.memory_integrator.search_memories(
                query=current_query,
                n_results=5,
                min_similarity=0.3
            )
            context_memories.extend(current_memories)
            
            # 重複除去
            unique_memories = self._remove_duplicates(context_memories)
            
            # 会話コンテキストを構築
            conversation_context = {
                "query": current_query,
                "session_id": session_id,
                "context_type": "conversation",
                "timestamp": datetime.now().isoformat(),
                "memories": unique_memories[:10],  # 最大10件
                "conversation_turns": len(conversation_history),
                "summary": f"Conversation context with {len(unique_memories)} relevant memories from {len(conversation_history)} turns"
            }
            
            return conversation_context
            
        except Exception as e:
            self.logger.error(f"Failed to build conversation context: {e}")
            return {"error": str(e)}
    
    def update_context_weights(self, new_weights: Dict[str, float]):
        """コンテキスト重みを更新"""
        
        try:
            # 重みの合計が1.0になるように正規化
            total_weight = sum(new_weights.values())
            if total_weight > 0:
                self.context_weights = {
                    key: value / total_weight
                    for key, value in new_weights.items()
                }
            
            self.logger.info(f"Updated context weights: {self.context_weights}")
            
        except Exception as e:
            self.logger.error(f"Failed to update context weights: {e}")
    
    def get_context_config(self) -> Dict[str, Any]:
        """コンテキスト設定を取得"""
        
        return {
            "max_context_length": self.max_context_length,
            "context_window_size": self.context_window_size,
            "context_weights": self.context_weights,
            "context_types": self.context_types
        }
