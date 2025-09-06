"""
Memory Importance Evaluator

記憶の重要度を評価し、自動整理を行うシステム
"""

import re
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .embedding_manager import EmbeddingManager
from .memory_models import MemoryItem, ConversationItem


class ImportanceEvaluator:
    """記憶重要度評価クラス"""
    
    def __init__(self,
                 embedding_manager: Optional[EmbeddingManager] = None,
                 importance_threshold: float = 0.6,
                 decay_factor: float = 0.1):
        
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.importance_threshold = importance_threshold
        self.decay_factor = decay_factor
        
        # 重要度計算の重み
        self.weights = {
            "frequency": 0.3,      # 頻度
            "recency": 0.2,        # 新しさ
            "uniqueness": 0.2,     # 独自性
            "context_richness": 0.15,  # 文脈の豊富さ
            "user_interaction": 0.15   # ユーザーインタラクション
        }
        
        # 重要キーワード（重み付き）
        self.important_keywords = {
            "error": 0.9,
            "bug": 0.9,
            "problem": 0.8,
            "solution": 0.8,
            "important": 0.7,
            "critical": 0.9,
            "urgent": 0.8,
            "remember": 0.7,
            "note": 0.6,
            "todo": 0.7,
            "task": 0.6,
            "goal": 0.7,
            "preference": 0.8,
            "setting": 0.7,
            "config": 0.7
        }
        
        self.logger = logging.getLogger(__name__)
    
    def calculate_frequency_score(self, 
                                memory_content: str,
                                all_memories: List[Dict[str, Any]]) -> float:
        """頻度スコアを計算"""
        
        try:
            # テキストを単語に分割
            words = re.findall(r'\b\w+\b', memory_content.lower())
            
            # 全記憶から単語頻度を計算
            all_words = []
            for memory in all_memories:
                all_words.extend(re.findall(r'\b\w+\b', memory.get('content', '').lower()))
            
            word_freq = Counter(all_words)
            total_words = len(all_words)
            
            if total_words == 0:
                return 0.0
            
            # 重要度スコア計算
            frequency_score = 0.0
            for word in words:
                if word in word_freq:
                    # 頻度が高いほど重要度が高い
                    freq_ratio = word_freq[word] / total_words
                    frequency_score += freq_ratio
            
            # 正規化
            return min(frequency_score / len(words), 1.0) if words else 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to calculate frequency score: {e}")
            return 0.0
    
    def calculate_recency_score(self, 
                              memory_timestamp: str,
                              current_time: Optional[datetime] = None) -> float:
        """新しさスコアを計算"""
        
        try:
            if current_time is None:
                current_time = datetime.now()
            
            # タイムスタンプを解析
            if isinstance(memory_timestamp, str):
                memory_time = datetime.fromisoformat(memory_timestamp.replace('Z', '+00:00'))
            else:
                memory_time = memory_timestamp
            
            # 時間差を計算（日数）
            time_diff = (current_time - memory_time).total_seconds() / (24 * 60 * 60)
            
            # 指数減衰関数で新しさスコアを計算
            recency_score = np.exp(-self.decay_factor * time_diff)
            
            return min(recency_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate recency score: {e}")
            return 0.0
    
    def calculate_uniqueness_score(self,
                                 memory_content: str,
                                 all_memories: List[Dict[str, Any]]) -> float:
        """独自性スコアを計算"""
        
        try:
            if not all_memories:
                return 1.0
            
            # 現在の記憶の埋め込み
            current_embedding = self.embedding_manager.encode_text(memory_content)
            
            # 他の記憶の埋め込み
            other_embeddings = []
            for memory in all_memories:
                if memory.get('content') != memory_content:
                    other_embedding = self.embedding_manager.encode_text(memory.get('content', ''))
                    other_embeddings.append(other_embedding)
            
            if not other_embeddings:
                return 1.0
            
            # 類似度計算
            similarities = []
            for other_embedding in other_embeddings:
                similarity = self.embedding_manager.calculate_similarity(
                    current_embedding, other_embedding
                )
                similarities.append(similarity)
            
            # 最大類似度が低いほど独自性が高い
            max_similarity = max(similarities) if similarities else 0.0
            uniqueness_score = 1.0 - max_similarity
            
            return max(uniqueness_score, 0.0)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate uniqueness score: {e}")
            return 0.0
    
    def calculate_context_richness_score(self, memory_content: str) -> float:
        """文脈の豊富さスコアを計算"""
        
        try:
            # テキスト長による基本スコア
            length_score = min(len(memory_content) / 500, 1.0)  # 500文字で最大
            
            # 重要キーワードの存在
            keyword_score = 0.0
            words = re.findall(r'\b\w+\b', memory_content.lower())
            
            for word in words:
                if word in self.important_keywords:
                    keyword_score += self.important_keywords[word]
            
            # 正規化
            keyword_score = min(keyword_score / len(words), 1.0) if words else 0.0
            
            # 文の複雑さ（句読点の数）
            punctuation_count = len(re.findall(r'[.!?]', memory_content))
            complexity_score = min(punctuation_count / 5, 1.0)  # 5文で最大
            
            # 数値やコードの存在
            has_numbers = bool(re.search(r'\d+', memory_content))
            has_code = bool(re.search(r'[{}();]', memory_content))
            technical_score = 0.3 if has_numbers else 0.0
            technical_score += 0.4 if has_code else 0.0
            
            # 総合スコア
            context_score = (
                length_score * 0.3 +
                keyword_score * 0.4 +
                complexity_score * 0.2 +
                technical_score * 0.1
            )
            
            return min(context_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate context richness score: {e}")
            return 0.0
    
    def calculate_user_interaction_score(self,
                                       memory_metadata: Dict[str, Any]) -> float:
        """ユーザーインタラクションスコアを計算"""
        
        try:
            interaction_score = 0.0
            
            # セッション回数
            session_count = memory_metadata.get('session_count', 0)
            interaction_score += min(session_count / 10, 1.0) * 0.3
            
            # 参照回数
            reference_count = memory_metadata.get('reference_count', 0)
            interaction_score += min(reference_count / 5, 1.0) * 0.3
            
            # ユーザー評価
            user_rating = memory_metadata.get('user_rating', 0)
            interaction_score += user_rating * 0.2
            
            # 修正回数（重要度の指標）
            modification_count = memory_metadata.get('modification_count', 0)
            interaction_score += min(modification_count / 3, 1.0) * 0.2
            
            return min(interaction_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate user interaction score: {e}")
            return 0.0
    
    def calculate_importance_score(self,
                                 memory: Dict[str, Any],
                                 all_memories: List[Dict[str, Any]]) -> float:
        """総合重要度スコアを計算"""
        
        try:
            content = memory.get('content', '')
            metadata = memory.get('metadata', {})
            timestamp = metadata.get('timestamp', datetime.now().isoformat())
            
            # 各スコアを計算
            frequency_score = self.calculate_frequency_score(content, all_memories)
            recency_score = self.calculate_recency_score(timestamp)
            uniqueness_score = self.calculate_uniqueness_score(content, all_memories)
            context_score = self.calculate_context_richness_score(content)
            interaction_score = self.calculate_user_interaction_score(metadata)
            
            # 重み付き総合スコア
            importance_score = (
                frequency_score * self.weights["frequency"] +
                recency_score * self.weights["recency"] +
                uniqueness_score * self.weights["uniqueness"] +
                context_score * self.weights["context_richness"] +
                interaction_score * self.weights["user_interaction"]
            )
            
            return min(importance_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate importance score: {e}")
            return 0.0
    
    def evaluate_memories(self,
                         memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """複数の記憶の重要度を評価"""
        
        try:
            evaluated_memories = []
            
            for memory in memories:
                importance_score = self.calculate_importance_score(memory, memories)
                
                # 重要度スコアをメタデータに追加
                memory_copy = memory.copy()
                if 'metadata' not in memory_copy:
                    memory_copy['metadata'] = {}
                
                memory_copy['metadata']['importance_score'] = importance_score
                memory_copy['metadata']['evaluation_timestamp'] = datetime.now().isoformat()
                
                evaluated_memories.append(memory_copy)
            
            # 重要度でソート
            evaluated_memories.sort(
                key=lambda x: x['metadata'].get('importance_score', 0),
                reverse=True
            )
            
            return evaluated_memories
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate memories: {e}")
            return memories
    
    def filter_important_memories(self,
                                memories: List[Dict[str, Any]],
                                threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """重要度が閾値以上の記憶をフィルタリング"""
        
        if threshold is None:
            threshold = self.importance_threshold
        
        try:
            # 重要度評価
            evaluated_memories = self.evaluate_memories(memories)
            
            # 閾値でフィルタリング
            important_memories = [
                memory for memory in evaluated_memories
                if memory['metadata'].get('importance_score', 0) >= threshold
            ]
            
            return important_memories
            
        except Exception as e:
            self.logger.error(f"Failed to filter important memories: {e}")
            return memories
    
    def get_memory_insights(self,
                          memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """記憶の洞察情報を生成"""
        
        try:
            if not memories:
                return {"error": "No memories provided"}
            
            # 重要度評価
            evaluated_memories = self.evaluate_memories(memories)
            
            # 統計情報
            importance_scores = [
                memory['metadata'].get('importance_score', 0)
                for memory in evaluated_memories
            ]
            
            insights = {
                "total_memories": len(memories),
                "average_importance": np.mean(importance_scores),
                "max_importance": np.max(importance_scores),
                "min_importance": np.min(importance_scores),
                "high_importance_count": len([
                    score for score in importance_scores
                    if score >= self.importance_threshold
                ]),
                "top_memories": evaluated_memories[:5],  # 上位5件
                "low_importance_memories": [
                    memory for memory in evaluated_memories
                    if memory['metadata'].get('importance_score', 0) < 0.3
                ]
            }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to generate memory insights: {e}")
            return {"error": str(e)}
    
    def update_importance_weights(self, new_weights: Dict[str, float]):
        """重要度計算の重みを更新"""
        
        try:
            # 重みの合計が1.0になるように正規化
            total_weight = sum(new_weights.values())
            if total_weight > 0:
                self.weights = {
                    key: value / total_weight
                    for key, value in new_weights.items()
                }
            
            self.logger.info(f"Updated importance weights: {self.weights}")
            
        except Exception as e:
            self.logger.error(f"Failed to update importance weights: {e}")
    
    def add_important_keyword(self, keyword: str, weight: float):
        """重要キーワードを追加"""
        
        self.important_keywords[keyword.lower()] = weight
        self.logger.info(f"Added important keyword: {keyword} (weight: {weight})")
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """評価設定を取得"""
        
        return {
            "importance_threshold": self.importance_threshold,
            "decay_factor": self.decay_factor,
            "weights": self.weights,
            "important_keywords": self.important_keywords
        }
