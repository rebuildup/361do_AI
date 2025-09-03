"""
LangChain Memory + HuggingFace Memory System

LangChain Memory と HuggingFace Transformers を統合した記憶システム
"""

import json
import torch
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline,
    Pipeline
)
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory
)
from langchain.schema import BaseMessage, HumanMessage, AIMessage

from .persistent_memory import LangChainPersistentMemory
from .memory_models import MemoryItem, ConversationItem


class HuggingFaceMemoryClassifier:
    """HuggingFace による記憶分類・重要度評価システム"""
    
    def __init__(self,
                 importance_model: str = "cardiffnlp/twitter-roberta-base-emotion",
                 classification_model: str = "microsoft/DialoGPT-medium",
                 device: str = "auto"):
        
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 重要度評価パイプライン
        self.importance_classifier = pipeline(
            "text-classification",
            model=importance_model,
            device=0 if self.device == "cuda" else -1,
            return_all_scores=True
        )
        
        # 記憶分類パイプライン
        self.memory_classifier = pipeline(
            "text-classification", 
            model="facebook/bart-large-mnli",
            device=0 if self.device == "cuda" else -1
        )
        
        # 記憶タイプの候補ラベル
        self.memory_type_labels = [
            "technical knowledge",
            "personal preference", 
            "factual information",
            "procedural knowledge",
            "conversational context",
            "error information",
            "learning pattern"
        ]
    
    async def evaluate_importance(self, text: str) -> float:
        """テキストの重要度を評価"""
        
        try:
            # 感情分析による重要度評価
            emotion_results = self.importance_classifier(text)
            
            # 感情の強度を重要度として使用
            max_score = max([result['score'] for result in emotion_results[0]])
            
            # 基本重要度計算
            base_importance = 0.3
            
            # 文字数による調整
            length_factor = min(len(text) / 1000, 0.3)
            
            # キーワードによる調整
            important_keywords = [
                "重要", "問題", "エラー", "学習", "改善", "最適化",
                "設定", "設計", "実装", "テスト", "デバッグ", "バグ",
                "性能", "メモリ", "GPU", "CPU", "モデル", "訓練"
            ]
            
            keyword_factor = sum(0.05 for keyword in important_keywords if keyword in text)
            
            # 最終重要度計算
            final_importance = base_importance + max_score * 0.4 + length_factor + keyword_factor
            
            return min(1.0, max(0.0, final_importance))
            
        except Exception as e:
            # エラー時はデフォルト重要度を返す
            return 0.5
    
    async def classify_memory_type(self, text: str) -> str:
        """記憶のタイプを分類"""
        
        try:
            # ゼロショット分類による記憶タイプ判定
            result = self.memory_classifier(text, self.memory_type_labels)
            
            # 最も確信度の高いラベルを返す
            return result['labels'][0]
            
        except Exception as e:
            # エラー時はデフォルトタイプを返す
            return "conversational context"
    
    async def extract_key_concepts(self, text: str) -> List[str]:
        """テキストから重要な概念を抽出"""
        
        # 簡単なキーワード抽出（実際の実装では NER や キーフレーズ抽出を使用）
        important_terms = []
        
        # 技術用語の検出
        tech_terms = [
            "Python", "JavaScript", "React", "Vue", "Angular", "Node.js",
            "Docker", "Kubernetes", "AWS", "Azure", "GCP",
            "機械学習", "深層学習", "AI", "LLM", "GPT", "BERT",
            "データベース", "SQL", "NoSQL", "MongoDB", "PostgreSQL",
            "API", "REST", "GraphQL", "HTTP", "HTTPS",
            "Git", "GitHub", "CI/CD", "DevOps"
        ]
        
        for term in tech_terms:
            if term.lower() in text.lower():
                important_terms.append(term)
        
        return important_terms[:10]  # 最大10個まで


class LangChainHuggingFaceMemory:
    """LangChain Memory + HuggingFace 統合記憶システム"""
    
    def __init__(self,
                 persistent_memory: LangChainPersistentMemory,
                 max_short_term_memories: int = 50,
                 max_long_term_memories: int = 1000,
                 importance_threshold: float = 0.7):
        
        self.persistent_memory = persistent_memory
        self.max_short_term_memories = max_short_term_memories
        self.max_long_term_memories = max_long_term_memories
        self.importance_threshold = importance_threshold
        
        # HuggingFace 記憶分類器
        self.hf_classifier = HuggingFaceMemoryClassifier()
        
        # LangChain Memory インスタンス
        self.short_term_memory = ConversationBufferWindowMemory(
            k=max_short_term_memories,
            return_messages=True,
            memory_key="short_term_history"
        )
        
        self.long_term_memory = ConversationSummaryMemory(
            llm=persistent_memory.summary_llm,
            return_messages=True,
            memory_key="long_term_summary"
        )
        
        # 記憶統計
        self.memory_stats = {
            "short_term_count": 0,
            "long_term_count": 0,
            "total_processed": 0,
            "promoted_to_long_term": 0
        }
    
    async def process_conversation(self,
                                 user_input: str,
                                 agent_response: str,
                                 session_id: str,
                                 metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """会話を処理して適切な記憶に保存"""
        
        conversation_text = f"User: {user_input}\nAgent: {agent_response}"
        
        # 1. HuggingFace による重要度評価
        importance_score = await self.hf_classifier.evaluate_importance(conversation_text)
        
        # 2. 記憶タイプ分類
        memory_type = await self.hf_classifier.classify_memory_type(conversation_text)
        
        # 3. キー概念抽出
        key_concepts = await self.hf_classifier.extract_key_concepts(conversation_text)
        
        # 4. 短期記憶に追加
        self.short_term_memory.save_context(
            {"input": user_input},
            {"output": agent_response}
        )
        
        # 5. 重要度に基づく長期記憶への昇格判定
        promoted_to_long_term = False
        if importance_score >= self.importance_threshold:
            self.long_term_memory.save_context(
                {"input": user_input},
                {"output": agent_response}
            )
            promoted_to_long_term = True
            self.memory_stats["promoted_to_long_term"] += 1
        
        # 6. 永続化記憶に保存
        enhanced_metadata = {
            **(metadata or {}),
            "importance_score": importance_score,
            "memory_type": memory_type,
            "key_concepts": json.dumps(key_concepts),  # JSON文字列に変換
            "promoted_to_long_term": promoted_to_long_term
        }
        
        conversation_id = await self.persistent_memory.store_conversation(
            user_input=user_input,
            agent_response=agent_response,
            metadata=enhanced_metadata
        )
        
        # 7. 統計更新
        self.memory_stats["total_processed"] += 1
        self.memory_stats["short_term_count"] = len(self.short_term_memory.chat_memory.messages)
        
        return {
            "conversation_id": conversation_id,
            "importance_score": importance_score,
            "memory_type": memory_type,
            "key_concepts": key_concepts,
            "promoted_to_long_term": promoted_to_long_term,
            "processing_stats": self.memory_stats.copy()
        }
    
    async def retrieve_contextual_memories(self,
                                         query: str,
                                         session_id: str,
                                         max_results: int = 10) -> Dict[str, Any]:
        """コンテキストに応じた記憶検索"""
        
        # 1. クエリの重要度・タイプ分析
        query_importance = await self.hf_classifier.evaluate_importance(query)
        query_type = await self.hf_classifier.classify_memory_type(query)
        query_concepts = await self.hf_classifier.extract_key_concepts(query)
        
        # 2. 永続化記憶から関連コンテキスト検索
        persistent_context = await self.persistent_memory.retrieve_relevant_context(
            query=query,
            session_id=session_id,
            max_results=max_results
        )
        
        # 3. 短期記憶から最新コンテキスト取得
        short_term_messages = self.short_term_memory.chat_memory.messages
        
        # 4. 長期記憶から要約コンテキスト取得
        long_term_summary = ""
        try:
            long_term_summary = self.long_term_memory.predict_new_summary(
                messages=short_term_messages,
                existing_summary=""
            )
        except Exception:
            long_term_summary = "長期記憶要約の生成に失敗しました"
        
        # 5. 記憶タイプ別フィルタリング
        filtered_memories = []
        for conv in persistent_context.get("similar_conversations", []):
            conv_metadata = conv.get("metadata", {})
            conv_type = conv_metadata.get("memory_type", "")
            
            # key_conceptsがJSON文字列の場合はパース
            conv_concepts_raw = conv_metadata.get("key_concepts", [])
            if isinstance(conv_concepts_raw, str):
                try:
                    conv_concepts = json.loads(conv_concepts_raw)
                except json.JSONDecodeError:
                    conv_concepts = []
            else:
                conv_concepts = conv_concepts_raw if isinstance(conv_concepts_raw, list) else []
            
            # 概念の重複度計算
            concept_overlap = len(set(query_concepts) & set(conv_concepts))
            
            filtered_memories.append({
                **conv,
                "type_match": query_type == conv_type,
                "concept_overlap": concept_overlap,
                "relevance_score": conv["score"] + (0.1 if query_type == conv_type else 0) + (concept_overlap * 0.05)
            })
        
        # 関連度でソート
        filtered_memories.sort(key=lambda x: x["relevance_score"])
        
        return {
            "query_analysis": {
                "importance": query_importance,
                "type": query_type,
                "concepts": query_concepts
            },
            "short_term_context": [
                {
                    "type": msg.type,
                    "content": msg.content
                }
                for msg in short_term_messages[-10:]  # 最新10件
            ],
            "long_term_summary": long_term_summary,
            "relevant_memories": filtered_memories[:max_results],
            "memory_statistics": self.get_memory_statistics()
        }
    
    async def optimize_memory_capacity(self) -> Dict[str, Any]:
        """記憶容量の最適化"""
        
        optimization_results = {
            "short_term_optimized": 0,
            "long_term_optimized": 0,
            "persistent_cleaned": 0
        }
        
        # 1. 短期記憶の最適化
        if len(self.short_term_memory.chat_memory.messages) > self.max_short_term_memories:
            # 古いメッセージを削除
            messages_to_remove = len(self.short_term_memory.chat_memory.messages) - self.max_short_term_memories
            self.short_term_memory.chat_memory.messages = self.short_term_memory.chat_memory.messages[messages_to_remove:]
            optimization_results["short_term_optimized"] = messages_to_remove
        
        # 2. 永続化記憶の自動整理
        cleaned_count = await self.persistent_memory.cleanup_old_memories(
            days_threshold=30,
            importance_threshold=0.3
        )
        optimization_results["persistent_cleaned"] = cleaned_count
        
        # 3. 統計更新
        self.memory_stats["short_term_count"] = len(self.short_term_memory.chat_memory.messages)
        
        return optimization_results
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """記憶システムの統計情報"""
        
        persistent_stats = self.persistent_memory.get_memory_statistics()
        
        return {
            "short_term_memory": {
                "current_messages": len(self.short_term_memory.chat_memory.messages),
                "max_capacity": self.max_short_term_memories,
                "usage_percent": (len(self.short_term_memory.chat_memory.messages) / self.max_short_term_memories) * 100
            },
            "long_term_memory": {
                "summary_available": bool(self.long_term_memory.buffer),
                "max_capacity": self.max_long_term_memories
            },
            "processing_stats": self.memory_stats,
            "persistent_memory": persistent_stats,
            "importance_threshold": self.importance_threshold
        }
    
    async def analyze_memory_patterns(self, session_id: str) -> Dict[str, Any]:
        """記憶パターンの分析"""
        
        # セッションの全会話を取得
        context = await self.persistent_memory.retrieve_relevant_context(
            query="全ての会話",
            session_id=session_id,
            max_results=100
        )
        
        conversations = context.get("similar_conversations", [])
        
        # パターン分析
        memory_types = {}
        concept_frequency = {}
        importance_distribution = []
        
        for conv in conversations:
            metadata = conv.get("metadata", {})
            
            # 記憶タイプ分布
            mem_type = metadata.get("memory_type", "unknown")
            memory_types[mem_type] = memory_types.get(mem_type, 0) + 1
            
            # 概念頻度
            concepts_raw = metadata.get("key_concepts", [])
            if isinstance(concepts_raw, str):
                try:
                    concepts = json.loads(concepts_raw)
                except json.JSONDecodeError:
                    concepts = []
            else:
                concepts = concepts_raw if isinstance(concepts_raw, list) else []
            
            for concept in concepts:
                concept_frequency[concept] = concept_frequency.get(concept, 0) + 1
            
            # 重要度分布
            importance = metadata.get("importance_score", 0.5)
            importance_distribution.append(importance)
        
        # 統計計算
        avg_importance = sum(importance_distribution) / len(importance_distribution) if importance_distribution else 0
        
        return {
            "total_conversations": len(conversations),
            "memory_type_distribution": memory_types,
            "top_concepts": sorted(concept_frequency.items(), key=lambda x: x[1], reverse=True)[:10],
            "importance_stats": {
                "average": avg_importance,
                "min": min(importance_distribution) if importance_distribution else 0,
                "max": max(importance_distribution) if importance_distribution else 0,
                "high_importance_count": sum(1 for i in importance_distribution if i >= self.importance_threshold)
            }
        }