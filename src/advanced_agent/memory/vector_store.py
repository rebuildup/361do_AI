"""
ChromaDB Vector Store Integration

ChromaDB を統合したベクトル記憶システム
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from .memory_models import MemoryItem, ConversationItem


class ChromaVectorStore:
    """ChromaDB ベクトルストア統合クラス"""
    
    def __init__(self,
                 persist_directory: str = "data/chroma_db",
                 collection_name: str = "agent_memory",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu"):
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # ディレクトリ作成
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # 埋め込みモデル初期化
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': device}
        )
        
        # ChromaDB クライアント初期化
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # コレクション取得または作成
        try:
            self.collection = self.client.get_collection(
                name=collection_name
            )
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Agent memory storage"}
            )
        
        # LangChain Chroma 統合
        self.langchain_chroma = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        
        self.logger = logging.getLogger(__name__)
    
    def add_memory(self, 
                   content: str,
                   memory_type: str = "conversation",
                   session_id: Optional[str] = None,
                   importance_score: float = 0.5,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """記憶をベクトルストアに追加"""
        
        # メタデータ構築
        doc_metadata = {
            "type": memory_type,
            "timestamp": datetime.now().isoformat(),
            "importance_score": importance_score,
            **(metadata or {})
        }
        
        if session_id:
            doc_metadata["session_id"] = session_id
        
        # ドキュメントID生成
        doc_id = f"{memory_type}_{datetime.now().timestamp()}_{hash(content) % 10000}"
        
        try:
            # ChromaDB に直接追加
            self.collection.add(
                documents=[content],
                metadatas=[doc_metadata],
                ids=[doc_id]
            )
            
            self.logger.info(f"Added memory to vector store: {doc_id}")
            return doc_id
            
        except Exception as e:
            self.logger.error(f"Failed to add memory: {e}")
            raise
    
    def search_similar(self,
                      query: str,
                      n_results: int = 5,
                      filter_metadata: Optional[Dict[str, Any]] = None,
                      min_similarity: float = 0.0) -> List[Dict[str, Any]]:
        """類似記憶を検索"""
        
        try:
            # ChromaDB で検索
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_metadata
            )
            
            # 結果を整形
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    result = {
                        "content": doc,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "distance": results['distances'][0][i] if results['distances'] else 0.0,
                        "similarity": 1 - results['distances'][0][i] if results['distances'] else 1.0,
                        "id": results['ids'][0][i] if results['ids'] else None
                    }
                    
                    # 類似度フィルタリング
                    if result["similarity"] >= min_similarity:
                        formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """IDで記憶を取得"""
        
        try:
            results = self.collection.get(ids=[memory_id])
            
            if results['documents'] and results['documents'][0]:
                return {
                    "content": results['documents'][0],
                    "metadata": results['metadatas'][0] if results['metadatas'] else {},
                    "id": memory_id
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get memory by ID: {e}")
            return None
    
    def update_memory(self,
                     memory_id: str,
                     content: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """記憶を更新"""
        
        try:
            # 既存の記憶を取得
            existing = self.get_memory_by_id(memory_id)
            if not existing:
                return False
            
            # 更新データを準備
            update_data = {}
            if content:
                update_data["documents"] = [content]
            if metadata:
                # 既存メタデータとマージ
                existing_metadata = existing.get("metadata", {})
                existing_metadata.update(metadata)
                update_data["metadatas"] = [existing_metadata]
            
            # 更新実行
            if update_data:
                self.collection.update(
                    ids=[memory_id],
                    **update_data
                )
            
            self.logger.info(f"Updated memory: {memory_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update memory: {e}")
            return False
    
    def delete_memory(self, memory_id: str) -> bool:
        """記憶を削除"""
        
        try:
            self.collection.delete(ids=[memory_id])
            self.logger.info(f"Deleted memory: {memory_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete memory: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """コレクション統計情報を取得"""
        
        try:
            count = self.collection.count()
            
            # メタデータ統計
            all_data = self.collection.get()
            metadata_stats = {}
            
            if all_data['metadatas']:
                for metadata in all_data['metadatas']:
                    for key, value in metadata.items():
                        if key not in metadata_stats:
                            metadata_stats[key] = {}
                        if value not in metadata_stats[key]:
                            metadata_stats[key][value] = 0
                        metadata_stats[key][value] += 1
            
            return {
                "total_memories": count,
                "collection_name": self.collection_name,
                "metadata_distribution": metadata_stats
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    def clear_collection(self) -> bool:
        """コレクションをクリア"""
        
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Agent memory storage"}
            )
            self.logger.info("Cleared collection")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear collection: {e}")
            return False
    
    def backup_collection(self, backup_path: str) -> bool:
        """コレクションをバックアップ"""
        
        try:
            # 全データを取得
            all_data = self.collection.get()
            
            # バックアップデータ構築
            backup_data = {
                "collection_name": self.collection_name,
                "timestamp": datetime.now().isoformat(),
                "documents": all_data.get('documents', []),
                "metadatas": all_data.get('metadatas', []),
                "ids": all_data.get('ids', [])
            }
            
            # ファイルに保存
            Path(backup_path).parent.mkdir(parents=True, exist_ok=True)
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Backed up collection to: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to backup collection: {e}")
            return False
    
    def restore_collection(self, backup_path: str) -> bool:
        """バックアップからコレクションを復元"""
        
        try:
            # バックアップファイル読み込み
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            # 現在のコレクションをクリア
            self.clear_collection()
            
            # データを復元
            if backup_data.get('documents'):
                self.collection.add(
                    documents=backup_data['documents'],
                    metadatas=backup_data.get('metadatas', []),
                    ids=backup_data.get('ids', [])
                )
            
            self.logger.info(f"Restored collection from: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore collection: {e}")
            return False
