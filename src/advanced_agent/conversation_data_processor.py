#!/usr/bin/env python3
"""
Conversation Data Processor
conversation.jsonの巨大な1行データを適切に処理するシステム
"""

import json
import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
from datetime import datetime
import sqlite3
import hashlib

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.advanced_agent.core.self_learning_agent import SelfLearningAgent
    from src.advanced_agent.core.logger import get_logger
except ImportError:
    # フォールバック用の簡単なロガー
    import logging
    def get_logger():
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


class ConversationDataProcessor:
    """会話データ処理システム"""
    
    def __init__(self, db_path: str = "data/conversation_processing.db"):
        self.logger = get_logger()
        self.db_path = db_path
        self._init_database()
        
        # データファイルパス
        self.chatgpt_file = "workspace/chat-gpt-data/conversations.json"
        self.claude_file = "workspace/claude-data/conversations.json"
        
        # 処理統計
        self.stats = {
            "chatgpt_processed": 0,
            "claude_processed": 0,
            "total_conversations": 0,
            "processing_errors": 0,
            "start_time": None,
            "end_time": None
        }
    
    def _init_database(self):
        """データベース初期化"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_conversations (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    conversation_id TEXT,
                    title TEXT,
                    content TEXT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    content_hash TEXT,
                    word_count INTEGER,
                    quality_score REAL DEFAULT 0.0
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processing_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    total_conversations INTEGER,
                    processed_conversations INTEGER,
                    errors INTEGER,
                    processing_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def _generate_conversation_id(self, source: str, original_id: str = None) -> str:
        """会話ID生成"""
        if original_id:
            return f"{source}_{original_id}"
        return f"{source}_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
    
    def _calculate_quality_score(self, conversation: Dict[str, Any]) -> float:
        """会話品質スコア計算"""
        score = 0.0
        
        # タイトルの存在
        if conversation.get("title"):
            score += 0.2
        
        # コンテンツの長さ
        content = conversation.get("content", "")
        if len(content) > 100:
            score += 0.3
        if len(content) > 500:
            score += 0.3
        if len(content) > 1000:
            score += 0.2
        
        return min(score, 1.0)
    
    def _extract_chatgpt_content(self, conversation: Dict[str, Any]) -> str:
        """ChatGPT会話からコンテンツを抽出"""
        content_parts = []
        
        # mappingフィールドから会話を抽出
        mapping = conversation.get("mapping", {})
        
        for node_id, node_data in mapping.items():
            if isinstance(node_data, dict):
                message = node_data.get("message")
                if message and isinstance(message, dict):
                    # メッセージの内容を取得
                    content = message.get("content", {})
                    if isinstance(content, dict):
                        parts = content.get("parts", [])
                        for part in parts:
                            if isinstance(part, str) and part.strip():
                                content_parts.append(part.strip())
                    elif isinstance(content, str) and content.strip():
                        content_parts.append(content.strip())
        
        return "\n".join(content_parts)
    
    def _process_chatgpt_conversation(self, conversation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """ChatGPT会話データ処理"""
        try:
            # 実際のコンテンツを抽出
            content = self._extract_chatgpt_content(conversation)
            
            # 空のコンテンツはスキップ
            if not content.strip():
                return None
            
            # 基本情報抽出
            processed = {
                "id": self._generate_conversation_id("chatgpt", conversation.get("id")),
                "source": "chatgpt",
                "conversation_id": conversation.get("id"),
                "title": conversation.get("title", "Untitled"),
                "content": content,
                "word_count": len(content.split()),
                "quality_score": self._calculate_quality_score({"title": conversation.get("title"), "content": content})
            }
            
            # コンテンツハッシュ生成
            content_hash = hashlib.md5(processed["content"].encode()).hexdigest()
            processed["content_hash"] = content_hash
            
            return processed
            
        except Exception as e:
            self.logger.error(f"ChatGPT会話処理エラー: {e}")
            self.stats["processing_errors"] += 1
            return None
    
    def _extract_claude_content(self, conversation: Dict[str, Any]) -> str:
        """Claude会話からコンテンツを抽出"""
        content_parts = []
        
        # chat_messagesフィールドから会話を抽出
        chat_messages = conversation.get("chat_messages", [])
        
        for message in chat_messages:
            if isinstance(message, dict):
                # contentフィールドをチェック（配列の場合）
                content = message.get("content", [])
                if isinstance(content, list):
                    # content配列内の各要素を処理
                    for content_item in content:
                        if isinstance(content_item, dict):
                            text = content_item.get("text", "")
                            if isinstance(text, str) and text.strip():
                                content_parts.append(text.strip())
                elif isinstance(content, str) and content.strip():
                    # contentが文字列の場合
                    content_parts.append(content.strip())
                
                # textフィールドも直接チェック
                text = message.get("text", "")
                if isinstance(text, str) and text.strip():
                    content_parts.append(text.strip())
        
        return "\n".join(content_parts)
    
    def _process_claude_conversation(self, conversation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Claude会話データ処理"""
        try:
            # 実際のコンテンツを抽出
            content = self._extract_claude_content(conversation)
            
            # 空のコンテンツはスキップ
            if not content.strip():
                return None
            
            # 基本情報抽出
            processed = {
                "id": self._generate_conversation_id("claude", conversation.get("uuid")),
                "source": "claude",
                "conversation_id": conversation.get("uuid"),
                "title": conversation.get("name", "Untitled"),
                "content": content,
                "word_count": len(content.split()),
                "quality_score": self._calculate_quality_score({"title": conversation.get("name"), "content": content})
            }
            
            # コンテンツハッシュ生成
            content_hash = hashlib.md5(processed["content"].encode()).hexdigest()
            processed["content_hash"] = content_hash
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Claude会話処理エラー: {e}")
            self.stats["processing_errors"] += 1
            return None
    
    def _load_json_streaming(self, file_path: str) -> Generator[Dict[str, Any], None, None]:
        """ストリーミング方式でJSONファイルを読み込み"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # ファイル全体を読み込み（メモリ効率を考慮）
                content = f.read()
                
                # JSONパース
                data = json.loads(content)
                
                # リストの場合
                if isinstance(data, list):
                    for item in data:
                        yield item
                # 単一オブジェクトの場合
                elif isinstance(data, dict):
                    yield data
                    
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析エラー {file_path}: {e}")
        except Exception as e:
            self.logger.error(f"ファイル読み込みエラー {file_path}: {e}")
    
    def _save_processed_conversation(self, conversation: Dict[str, Any]) -> bool:
        """処理済み会話をデータベースに保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO processed_conversations 
                    (id, source, conversation_id, title, content, content_hash, word_count, quality_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    conversation["id"],
                    conversation["source"],
                    conversation["conversation_id"],
                    conversation["title"],
                    conversation["content"],
                    conversation["content_hash"],
                    conversation["word_count"],
                    conversation["quality_score"]
                ))
                conn.commit()
            return True
        except Exception as e:
            self.logger.error(f"データベース保存エラー: {e}")
            return False
    
    def process_chatgpt_data(self) -> int:
        """ChatGPTデータ処理"""
        if not os.path.exists(self.chatgpt_file):
            self.logger.warning(f"ChatGPTファイルが見つかりません: {self.chatgpt_file}")
            return 0
        
        self.logger.info("ChatGPTデータ処理開始...")
        processed_count = 0
        
        try:
            for conversation in self._load_json_streaming(self.chatgpt_file):
                processed = self._process_chatgpt_conversation(conversation)
                if processed and self._save_processed_conversation(processed):
                    processed_count += 1
                    
                # 進捗表示
                if processed_count % 100 == 0:
                    self.logger.info(f"ChatGPT処理済み: {processed_count}件")
            
            self.stats["chatgpt_processed"] = processed_count
            self.logger.info(f"ChatGPTデータ処理完了: {processed_count}件")
            return processed_count
            
        except Exception as e:
            self.logger.error(f"ChatGPTデータ処理エラー: {e}")
            return 0
    
    def process_claude_data(self) -> int:
        """Claudeデータ処理"""
        if not os.path.exists(self.claude_file):
            self.logger.warning(f"Claudeファイルが見つかりません: {self.claude_file}")
            return 0
        
        self.logger.info("Claudeデータ処理開始...")
        processed_count = 0
        total_conversations = 0
        
        try:
            for conversation in self._load_json_streaming(self.claude_file):
                total_conversations += 1
                processed = self._process_claude_conversation(conversation)
                if processed and self._save_processed_conversation(processed):
                    processed_count += 1
                    
                # 進捗表示（100件ごと、または最後）
                if processed_count % 100 == 0 and processed_count > 0:
                    self.logger.info(f"Claude処理済み: {processed_count}件")
            
            # 空のデータの場合は1回だけログ出力
            if total_conversations == 0:
                self.logger.info("Claudeデータ: 0件")
            elif processed_count == 0:
                self.logger.info(f"Claudeデータ: {total_conversations}件（有効なコンテンツなし）")
            else:
                self.logger.info(f"Claudeデータ処理完了: {processed_count}件")
            
            self.stats["claude_processed"] = processed_count
            return processed_count
            
        except Exception as e:
            self.logger.error(f"Claudeデータ処理エラー: {e}")
            return 0
    
    def process_all_data(self) -> Dict[str, Any]:
        """全データ処理"""
        self.stats["start_time"] = datetime.now()
        
        self.logger.info("=" * 60)
        self.logger.info("会話データ処理開始")
        self.logger.info("=" * 60)
        
        # ChatGPTデータ処理
        chatgpt_count = self.process_chatgpt_data()
        
        # Claudeデータ処理
        claude_count = self.process_claude_data()
        
        # 統計更新
        self.stats["total_conversations"] = chatgpt_count + claude_count
        self.stats["end_time"] = datetime.now()
        
        # 処理時間計算
        if self.stats["start_time"] and self.stats["end_time"]:
            processing_time = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
            self.stats["processing_time"] = processing_time
        
        # 統計をデータベースに保存
        self._save_processing_stats()
        
        # 結果表示
        self.logger.info("=" * 60)
        self.logger.info("会話データ処理完了")
        self.logger.info("=" * 60)
        self.logger.info(f"ChatGPT処理済み: {chatgpt_count}件")
        self.logger.info(f"Claude処理済み: {claude_count}件")
        self.logger.info(f"総処理数: {self.stats['total_conversations']}件")
        self.logger.info(f"エラー数: {self.stats['processing_errors']}件")
        if self.stats.get("processing_time"):
            self.logger.info(f"処理時間: {self.stats['processing_time']:.2f}秒")
        
        return self.stats
    
    def _save_processing_stats(self):
        """処理統計をデータベースに保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # ChatGPT統計
                conn.execute("""
                    INSERT INTO processing_stats 
                    (source, total_conversations, processed_conversations, errors, processing_time)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    "chatgpt",
                    self.stats["chatgpt_processed"],
                    self.stats["chatgpt_processed"],
                    0,  # エラー数は別途管理
                    self.stats.get("processing_time", 0) / 2  # 半分の時間を割り当て
                ))
                
                # Claude統計
                conn.execute("""
                    INSERT INTO processing_stats 
                    (source, total_conversations, processed_conversations, errors, processing_time)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    "claude",
                    self.stats["claude_processed"],
                    self.stats["claude_processed"],
                    0,  # エラー数は別途管理
                    self.stats.get("processing_time", 0) / 2  # 半分の時間を割り当て
                ))
                
                conn.commit()
        except Exception as e:
            self.logger.error(f"統計保存エラー: {e}")
    
    def get_processed_conversations(self, source: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """処理済み会話を取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                if source:
                    cursor = conn.execute("""
                        SELECT * FROM processed_conversations 
                        WHERE source = ? 
                        ORDER BY processed_at DESC 
                        LIMIT ?
                    """, (source, limit))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM processed_conversations 
                        ORDER BY processed_at DESC 
                        LIMIT ?
                    """, (limit,))
                
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"会話取得エラー: {e}")
            return []
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """処理統計を取得"""
        return self.stats.copy()


async def main():
    """メイン関数"""
    processor = ConversationDataProcessor()
    
    # データ処理実行
    stats = processor.process_all_data()
    
    # 結果確認
    print("\n" + "=" * 60)
    print("処理結果確認")
    print("=" * 60)
    
    # サンプルデータ表示
    sample_conversations = processor.get_processed_conversations(limit=5)
    for i, conv in enumerate(sample_conversations, 1):
        print(f"\nサンプル {i}:")
        print(f"  ID: {conv['id']}")
        print(f"  ソース: {conv['source']}")
        print(f"  タイトル: {conv['title']}")
        print(f"  単語数: {conv['word_count']}")
        print(f"  品質スコア: {conv['quality_score']:.2f}")
        print(f"  コンテンツプレビュー: {conv['content'][:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
