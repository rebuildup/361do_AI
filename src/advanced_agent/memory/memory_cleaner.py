"""
Memory Cleaner

記憶の自動整理とクリーンアップシステム
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from .importance_evaluator import ImportanceEvaluator
from .embedding_manager import EmbeddingManager
from .memory_models import MemoryItem


class MemoryCleaner:
    """記憶クリーンアップクラス"""
    
    def __init__(self,
                 importance_evaluator: Optional[ImportanceEvaluator] = None,
                 embedding_manager: Optional[EmbeddingManager] = None,
                 max_memories: int = 10000,
                 cleanup_threshold: float = 0.3):
        
        self.importance_evaluator = importance_evaluator or ImportanceEvaluator()
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.max_memories = max_memories
        self.cleanup_threshold = cleanup_threshold
        
        # クリーンアップ設定
        self.cleanup_rules = {
            "duplicate_threshold": 0.95,      # 重複判定閾値
            "age_threshold_days": 90,         # 古い記憶の削除閾値（日数）
            "low_importance_threshold": 0.2,  # 低重要度記憶の削除閾値
            "max_similar_memories": 5,        # 類似記憶の最大保持数
            "min_content_length": 10          # 最小コンテンツ長
        }
        
        self.logger = logging.getLogger(__name__)
    
    def identify_duplicates(self,
                          memories: List[Dict[str, Any]]) -> List[List[str]]:
        """重複記憶を特定"""
        
        try:
            duplicate_groups = []
            processed_ids = set()
            
            # Precompute embeddings for all memories
            embeddings = {}
            for memory in memories:
                content = memory.get('content', '')
                if len(content) >= self.cleanup_rules["min_content_length"]:
                    embeddings[memory.get('id')] = self.embedding_manager.encode_text(content)
            
            for i, memory1 in enumerate(memories):
                if memory1.get('id') in processed_ids:
                    continue
                
                memory1_id = memory1.get('id')
                if memory1_id not in embeddings:
                    continue
                
                embedding1 = embeddings[memory1_id]
                duplicate_group = [memory1_id]
                
                for _j, memory2 in enumerate(memories[i+1:], i+1):
                    if memory2.get('id') in processed_ids:
                        continue
                    
                    memory2_id = memory2.get('id')
                    if memory2_id not in embeddings:
                        continue
                    
                    embedding2 = embeddings[memory2_id]
                    similarity = self.embedding_manager.calculate_similarity(embedding1, embedding2)
                    
                    if similarity >= self.cleanup_rules["duplicate_threshold"]:
                        duplicate_group.append(memory2_id)
                        processed_ids.add(memory2_id)
                
                if len(duplicate_group) > 1:
                    duplicate_groups.append(duplicate_group)
                    for mem_id in duplicate_group:
                        processed_ids.add(mem_id)
            
            return duplicate_groups
            
        except Exception as e:
            self.logger.error(f"Failed to identify duplicates: {e}")
            return []
    
    def identify_old_memories(self,
                            memories: List[Dict[str, Any]]) -> List[str]:
        """古い記憶を特定"""
        
        try:
            old_memory_ids = []
            # UTC aware cutoff date
            from datetime import timezone
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.cleanup_rules["age_threshold_days"])
            
            for memory in memories:
                metadata = memory.get('metadata', {})
                timestamp_str = metadata.get('timestamp')
                
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        # Ensure timestamp is timezone aware
                        if timestamp.tzinfo is None:
                            timestamp = timestamp.replace(tzinfo=timezone.utc)
                        if timestamp < cutoff_date:
                            old_memory_ids.append(memory.get('id'))
                    except Exception:
                        # タイムスタンプ解析エラーの場合は削除しない（安全のため）
                        self.logger.warning(f"Failed to parse timestamp: {timestamp_str}")
                        continue
            
            return old_memory_ids
            
        except Exception as e:
            self.logger.error(f"Failed to identify old memories: {e}")
            return []
    
    def identify_low_importance_memories(self,
                                       memories: List[Dict[str, Any]]) -> List[str]:
        """低重要度記憶を特定"""
        
        try:
            # 重要度評価
            evaluated_memories = self.importance_evaluator.evaluate_memories(memories)
            
            low_importance_ids = []
            for memory in evaluated_memories:
                importance_score = memory['metadata'].get('importance_score', 0)
                if importance_score < self.cleanup_rules["low_importance_threshold"]:
                    low_importance_ids.append(memory.get('id'))
            
            return low_importance_ids
            
        except Exception as e:
            self.logger.error(f"Failed to identify low importance memories: {e}")
            return []
    
    def identify_similar_memories(self,
                                memories: List[Dict[str, Any]]) -> List[List[str]]:
        """類似記憶を特定"""
        
        try:
            similar_groups = []
            processed_ids = set()
            
            # Precompute embeddings for all memories
            embeddings = {}
            for memory in memories:
                content = memory.get('content', '')
                if len(content) >= self.cleanup_rules["min_content_length"]:
                    embeddings[memory.get('id')] = self.embedding_manager.encode_text(content)
            
            for i, memory1 in enumerate(memories):
                if memory1.get('id') in processed_ids:
                    continue
                
                memory1_id = memory1.get('id')
                if memory1_id not in embeddings:
                    continue
                
                embedding1 = embeddings[memory1_id]
                similar_group = [memory1_id]
                
                for _j, memory2 in enumerate(memories[i+1:], i+1):
                    if memory2.get('id') in processed_ids:
                        continue
                    
                    memory2_id = memory2.get('id')
                    if memory2_id not in embeddings:
                        continue
                    
                    embedding2 = embeddings[memory2_id]
                    similarity = self.embedding_manager.calculate_similarity(embedding1, embedding2)
                    
                    # 類似度が高いが重複ではない記憶
                    if (0.7 <= similarity < self.cleanup_rules["duplicate_threshold"]):
                        similar_group.append(memory2_id)
                        processed_ids.add(memory2_id)
                
                if len(similar_group) > self.cleanup_rules["max_similar_memories"]:
                    similar_groups.append(similar_group)
                    for mem_id in similar_group:
                        processed_ids.add(mem_id)
            
            return similar_groups
            
        except Exception as e:
            self.logger.error(f"Failed to identify similar memories: {e}")
            return []
    
    def select_memories_to_keep(self,
                               memory_group: List[Dict[str, Any]]) -> List[str]:
        """保持する記憶を選択"""
        
        try:
            if not memory_group:
                return []
            
            # 重要度でソート
            sorted_memories = sorted(
                memory_group,
                key=lambda x: x.get('metadata', {}).get('importance_score', 0),
                reverse=True
            )
            
            # 最大保持数を超える場合は上位を保持
            max_keep = self.cleanup_rules["max_similar_memories"]
            memories_to_keep = sorted_memories[:max_keep]
            
            return [memory.get('id') for memory in memories_to_keep]
            
        except Exception as e:
            self.logger.error(f"Failed to select memories to keep: {e}")
            return [memory_group[0].get('id')] if memory_group else []
    
    def generate_cleanup_plan(self,
                            memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """クリーンアップ計画を生成"""
        
        try:
            cleanup_plan = {
                "total_memories": len(memories),
                "duplicates": [],
                "old_memories": [],
                "low_importance": [],
                "similar_groups": [],
                "memories_to_delete": [],
                "memories_to_keep": [],
                "cleanup_summary": {}
            }
            
            # 重複記憶の特定
            duplicate_groups = self.identify_duplicates(memories)
            for group in duplicate_groups:
                # 各グループで最も重要度の高い記憶を保持
                group_memories = [m for m in memories if m.get('id') in group]
                keep_ids = self.select_memories_to_keep(group_memories)
                delete_ids = [mem_id for mem_id in group if mem_id not in keep_ids]
                
                cleanup_plan["duplicates"].append({
                    "group": group,
                    "keep": keep_ids,
                    "delete": delete_ids
                })
                cleanup_plan["memories_to_delete"].extend(delete_ids)
                cleanup_plan["memories_to_keep"].extend(keep_ids)
            
            # 古い記憶の特定
            old_memory_ids = self.identify_old_memories(memories)
            cleanup_plan["old_memories"] = old_memory_ids
            cleanup_plan["memories_to_delete"].extend(old_memory_ids)
            
            # 低重要度記憶の特定
            low_importance_ids = self.identify_low_importance_memories(memories)
            cleanup_plan["low_importance"] = low_importance_ids
            cleanup_plan["memories_to_delete"].extend(low_importance_ids)
            
            # 類似記憶の特定
            similar_groups = self.identify_similar_memories(memories)
            for group in similar_groups:
                group_memories = [m for m in memories if m.get('id') in group]
                keep_ids = self.select_memories_to_keep(group_memories)
                delete_ids = [mem_id for mem_id in group if mem_id not in keep_ids]
                
                cleanup_plan["similar_groups"].append({
                    "group": group,
                    "keep": keep_ids,
                    "delete": delete_ids
                })
                cleanup_plan["memories_to_delete"].extend(delete_ids)
                cleanup_plan["memories_to_keep"].extend(keep_ids)
            
            # 重複削除
            cleanup_plan["memories_to_delete"] = list(set(cleanup_plan["memories_to_delete"]))
            cleanup_plan["memories_to_keep"] = list(set(cleanup_plan["memories_to_keep"]))
            
            # サマリー
            cleanup_plan["cleanup_summary"] = {
                "duplicates_found": len(duplicate_groups),
                "old_memories_found": len(old_memory_ids),
                "low_importance_found": len(low_importance_ids),
                "similar_groups_found": len(similar_groups),
                "total_to_delete": len(cleanup_plan["memories_to_delete"]),
                "total_to_keep": len(cleanup_plan["memories_to_keep"]),
                "cleanup_ratio": len(cleanup_plan["memories_to_delete"]) / len(memories) if memories else 0
            }
            
            return cleanup_plan
            
        except Exception as e:
            self.logger.error(f"Failed to generate cleanup plan: {e}")
            return {"error": str(e)}
    
    def execute_cleanup(self,
                       memories: List[Dict[str, Any]],
                       cleanup_plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """クリーンアップを実行"""
        
        try:
            if cleanup_plan is None:
                cleanup_plan = self.generate_cleanup_plan(memories)
            
            if "error" in cleanup_plan:
                return cleanup_plan
            
            # 削除対象の記憶を除外
            memories_to_delete = set(cleanup_plan["memories_to_delete"])
            cleaned_memories = [
                memory for memory in memories
                if memory.get('id') not in memories_to_delete
            ]
            
            # クリーンアップ結果
            cleanup_result = {
                "original_count": len(memories),
                "cleaned_count": len(cleaned_memories),
                "deleted_count": len(memories_to_delete),
                "cleanup_plan": cleanup_plan,
                "cleaned_memories": cleaned_memories,
                "success": True
            }
            
            self.logger.info(f"Memory cleanup completed: {cleanup_result['deleted_count']} memories deleted")
            
            return cleanup_result
            
        except Exception as e:
            self.logger.error(f"Failed to execute cleanup: {e}")
            return {"error": str(e), "success": False}
    
    def auto_cleanup(self,
                    memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """自動クリーンアップを実行"""
        
        try:
            # メモリ数が上限を超えている場合のみクリーンアップ
            if len(memories) <= self.max_memories:
                return {
                    "message": "No cleanup needed",
                    "original_count": len(memories),
                    "cleaned_count": len(memories),
                    "deleted_count": 0,
                    "success": True
                }
            
            # クリーンアップ計画生成
            cleanup_plan = self.generate_cleanup_plan(memories)
            
            if "error" in cleanup_plan:
                return cleanup_plan
            
            # クリーンアップ実行
            cleanup_result = self.execute_cleanup(memories, cleanup_plan)
            
            return cleanup_result
            
        except Exception as e:
            self.logger.error(f"Failed to execute auto cleanup: {e}")
            return {"error": str(e), "success": False}
    
    def update_cleanup_rules(self, new_rules: Dict[str, Any]):
        """クリーンアップルールを更新"""
        
        try:
            self.cleanup_rules.update(new_rules)
            self.logger.info(f"Updated cleanup rules: {self.cleanup_rules}")
            
        except Exception as e:
            self.logger.error(f"Failed to update cleanup rules: {e}")
    
    def get_cleanup_stats(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """クリーンアップ統計を取得"""
        
        try:
            cleanup_plan = self.generate_cleanup_plan(memories)
            
            if "error" in cleanup_plan:
                return cleanup_plan
            
            stats = {
                "current_memory_count": len(memories),
                "max_memory_limit": self.max_memories,
                "cleanup_needed": len(memories) > self.max_memories,
                "cleanup_plan": cleanup_plan["cleanup_summary"],
                "cleanup_rules": self.cleanup_rules
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get cleanup stats: {e}")
            return {"error": str(e)}
