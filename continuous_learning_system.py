#!/usr/bin/env python3
"""
Continuous Learning System
4時間継続学習システム

workspaceフォルダ内のconversation.jsonファイルを4時間継続してエージェントに学習させる
"""

import asyncio
import json
import logging
import time
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import sqlite3
import hashlib
import random

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.advanced_agent.core.self_learning_agent import SelfLearningAgent
from src.advanced_agent.config.settings import get_agent_config
from src.advanced_agent.core.logger import get_logger


class ConversationDataProcessor:
    """会話データ処理クラス"""
    
    def __init__(self):
        self.logger = get_logger()
        self.processed_count = 0
        self.total_conversations = 0
        
    def load_conversation_data(self, file_path: str) -> List[Dict[str, Any]]:
        """会話データを読み込み"""
        try:
            self.logger.info(f"Loading conversation data from: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # データ構造に応じて会話を抽出
            conversations = self._extract_conversations(data)
            self.total_conversations = len(conversations)
            
            self.logger.info(f"Loaded {self.total_conversations} conversations")
            return conversations
            
        except Exception as e:
            self.logger.error(f"Error loading conversation data: {e}")
            return []
    
    def _extract_conversations(self, data: Any) -> List[Dict[str, Any]]:
        """データから会話を抽出"""
        conversations = []
        
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # ChatGPT形式の会話データ（shared_conversations.json）
                    if 'conversation_id' in item:
                        conversations.append({
                            'id': item.get('id', ''),
                            'conversation_id': item.get('conversation_id', ''),
                            'title': item.get('title', ''),
                            'content': item.get('content', ''),
                            'source': 'chatgpt_shared'
                        })
                    # ChatGPT形式の会話データ（conversations.json）
                    elif 'id' in item and 'title' in item:
                        conversations.append({
                            'id': item.get('id', ''),
                            'conversation_id': item.get('id', ''),
                            'title': item.get('title', ''),
                            'content': item.get('content', ''),
                            'messages': item.get('messages', []),
                            'source': 'chatgpt'
                        })
                    # Claude形式の会話データ
                    elif 'id' in item and 'messages' in item:
                        conversations.append({
                            'id': item.get('id', ''),
                            'title': item.get('title', ''),
                            'messages': item.get('messages', []),
                            'source': 'claude'
                        })
        
        return conversations
    
    def process_conversation_batch(self, conversations: List[Dict[str, Any]], batch_size: int = 10) -> List[Dict[str, Any]]:
        """会話データをバッチ処理"""
        processed_batch = []
        
        for i in range(0, len(conversations), batch_size):
            batch = conversations[i:i + batch_size]
            processed_batch.append({
                'batch_id': f"batch_{i//batch_size}",
                'conversations': batch,
                'processed_at': datetime.now().isoformat()
            })
        
        return processed_batch


class ContinuousLearningSystem:
    """4時間継続学習システム"""
    
    def __init__(self, learning_duration_hours: int = 4):
        self.learning_duration_hours = learning_duration_hours
        self.logger = get_logger()
        self.agent: Optional[SelfLearningAgent] = None
        self.data_processor = ConversationDataProcessor()
        
        # 学習統計
        self.start_time = None
        self.end_time = None
        self.total_processed = 0
        self.learning_cycles = 0
        self.current_epoch = 0
        
        # 学習設定
        self.learning_config = {
            'batch_size': 5,
            'learning_interval': 30,  # 30秒間隔
            'max_conversations_per_cycle': 50,
            'learning_rate': 0.1,
            'memory_retention_days': 7
        }
        
        # データベース初期化
        self._init_learning_database()
    
    def _init_learning_database(self):
        """学習データベース初期化"""
        try:
            db_path = "data/continuous_learning.db"
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            with sqlite3.connect(db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS learning_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT UNIQUE,
                        start_time TEXT,
                        end_time TEXT,
                        duration_hours REAL,
                        total_conversations INTEGER,
                        total_cycles INTEGER,
                        status TEXT,
                        created_at TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS learning_progress (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        cycle_number INTEGER,
                        conversations_processed INTEGER,
                        learning_epoch INTEGER,
                        timestamp TEXT,
                        performance_metrics TEXT,
                        FOREIGN KEY (session_id) REFERENCES learning_sessions (session_id)
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_learning (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        conversation_id TEXT,
                        source TEXT,
                        content_hash TEXT,
                        learning_epoch INTEGER,
                        processed_at TEXT,
                        quality_score REAL,
                        FOREIGN KEY (session_id) REFERENCES learning_sessions (session_id)
                    )
                """)
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Database initialization error: {e}")
    
    async def initialize_agent(self) -> bool:
        """エージェント初期化"""
        try:
            self.logger.info("Initializing self-learning agent...")
            
            # エージェント作成
            self.agent = SelfLearningAgent(
                config_path="config/agent_config.yaml",
                db_path="data/self_learning_agent.db"
            )
            
            # セッション初期化
            session_id = await self.agent.initialize_session(
                session_id="continuous_learning_session",
                user_id="learning_system"
            )
            
            self.logger.info(f"Agent initialized successfully with session: {session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Agent initialization error: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    async def start_continuous_learning(self):
        """4時間継続学習開始"""
        try:
            self.logger.info(f"Starting {self.learning_duration_hours}-hour continuous learning session")
            
            # セッションID生成
            session_id = f"learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 学習データ読み込み
            chatgpt_data = self._load_chatgpt_data()
            claude_data = self._load_claude_data()
            
            all_conversations = chatgpt_data + claude_data
            self.logger.info(f"Total conversations to process: {len(all_conversations)}")
            
            # 学習セッション記録
            self._record_learning_session(session_id, len(all_conversations))
            
            # 学習開始時間
            self.start_time = datetime.now()
            self.end_time = self.start_time + timedelta(hours=self.learning_duration_hours)
            
            # 学習ループ
            await self._learning_loop(session_id, all_conversations)
            
            # 学習完了
            self._complete_learning_session(session_id)
            
            self.logger.info("Continuous learning completed successfully")
            
        except Exception as e:
            self.logger.error(f"Continuous learning error: {e}")
            raise
    
    def _load_chatgpt_data(self) -> List[Dict[str, Any]]:
        """ChatGPTデータ読み込み"""
        try:
            conversations = []
            
            # 新しいconversations.jsonファイル
            conversations_file = "workspace/chat-gpt-data/conversations.json"
            if os.path.exists(conversations_file):
                self.logger.info(f"Loading ChatGPT conversations from: {conversations_file}")
                conversations.extend(self.data_processor.load_conversation_data(conversations_file))
            
            # 既存のshared_conversations.jsonファイル
            shared_file = "workspace/chat-gpt-data/shared_conversations.json"
            if os.path.exists(shared_file):
                self.logger.info(f"Loading ChatGPT shared conversations from: {shared_file}")
                conversations.extend(self.data_processor.load_conversation_data(shared_file))
            
            if not conversations:
                self.logger.warning("No ChatGPT data files found")
            
            return conversations
            
        except Exception as e:
            self.logger.error(f"Error loading ChatGPT data: {e}")
            return []
    
    def _load_claude_data(self) -> List[Dict[str, Any]]:
        """Claudeデータ読み込み"""
        try:
            file_path = "workspace/claude-data/conversations.json"
            if os.path.exists(file_path):
                return self.data_processor.load_conversation_data(file_path)
            else:
                self.logger.warning(f"Claude data file not found: {file_path}")
                return []
        except Exception as e:
            self.logger.error(f"Error loading Claude data: {e}")
            return []
    
    def _record_learning_session(self, session_id: str, total_conversations: int):
        """学習セッション記録"""
        try:
            with sqlite3.connect("data/continuous_learning.db") as conn:
                conn.execute("""
                    INSERT INTO learning_sessions 
                    (session_id, start_time, duration_hours, total_conversations, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    datetime.now().isoformat(),
                    self.learning_duration_hours,
                    total_conversations,
                    'running',
                    datetime.now().isoformat()
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error recording learning session: {e}")
    
    async def _learning_loop(self, session_id: str, conversations: List[Dict[str, Any]]):
        """学習ループ"""
        cycle_number = 0
        
        while datetime.now() < self.end_time:
            try:
                cycle_number += 1
                self.learning_cycles = cycle_number
                
                self.logger.info(f"Starting learning cycle {cycle_number}")
                
                # 会話データをシャッフルして学習
                random.shuffle(conversations)
                
                # バッチ処理
                batch_size = self.learning_config['batch_size']
                max_conversations = self.learning_config['max_conversations_per_cycle']
                
                conversations_to_process = conversations[:max_conversations]
                
                for i in range(0, len(conversations_to_process), batch_size):
                    batch = conversations_to_process[i:i + batch_size]
                    await self._process_conversation_batch(session_id, batch, cycle_number)
                    
                    # 進捗更新
                    self.total_processed += len(batch)
                    
                    # 短い休憩
                    await asyncio.sleep(1)
                
                # 学習進捗記録
                self._record_learning_progress(session_id, cycle_number, len(conversations_to_process))
                
                # エージェントの学習エポック更新
                if self.agent and hasattr(self.agent, 'current_state'):
                    self.agent.current_state.learning_epoch += 1
                    self.current_epoch = self.agent.current_state.learning_epoch
                
                # 学習間隔待機
                await asyncio.sleep(self.learning_config['learning_interval'])
                
                # 残り時間チェック
                remaining_time = self.end_time - datetime.now()
                if remaining_time.total_seconds() <= 0:
                    break
                
                self.logger.info(f"Cycle {cycle_number} completed. Remaining time: {remaining_time}")
                
            except Exception as e:
                self.logger.error(f"Error in learning cycle {cycle_number}: {e}")
                continue
    
    async def _process_conversation_batch(self, session_id: str, batch: List[Dict[str, Any]], cycle_number: int):
        """会話バッチ処理"""
        try:
            for conversation in batch:
                # 会話データを学習データとして処理
                learning_data = self._convert_to_learning_data(conversation)
                
                if learning_data and self.agent:
                    # エージェントに学習データを送信
                    await self._send_learning_data_to_agent(learning_data)
                    
                    # 学習記録
                    self._record_conversation_learning(session_id, conversation, cycle_number)
                
        except Exception as e:
            self.logger.error(f"Error processing conversation batch: {e}")
    
    def _convert_to_learning_data(self, conversation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """会話データを学習データに変換"""
        try:
            source = conversation.get('source', '')
            
            if source in ['chatgpt', 'chatgpt_shared']:
                # ChatGPT形式のデータ処理
                content = conversation.get('content', '')
                messages = conversation.get('messages', [])
                
                # messagesがある場合は、それらからコンテンツを構築
                if messages and not content:
                    content = '\n'.join([msg.get('content', '') for msg in messages])
                
                if content:
                    return {
                        'type': 'conversation',
                        'content': content,
                        'metadata': {
                            'source': source,
                            'conversation_id': conversation.get('conversation_id', ''),
                            'title': conversation.get('title', '')
                        }
                    }
                    
            elif source == 'claude':
                # Claude形式のデータ処理
                messages = conversation.get('messages', [])
                if messages:
                    content = '\n'.join([msg.get('content', '') for msg in messages])
                    return {
                        'type': 'conversation',
                        'content': content,
                        'metadata': {
                            'source': 'claude',
                            'conversation_id': conversation.get('id', ''),
                            'title': conversation.get('title', '')
                        }
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error converting conversation to learning data: {e}")
            return None
    
    async def _send_learning_data_to_agent(self, learning_data: Dict[str, Any]):
        """エージェントに学習データを送信"""
        try:
            if not self.agent:
                return
            
            # 学習データをエージェントのメモリシステムに保存
            content = learning_data.get('content', '')
            metadata = learning_data.get('metadata', {})
            
            # コンテンツを分割してユーザー入力とレスポンスに分ける
            if len(content) > 1000:
                # 長いコンテンツの場合は分割
                user_input = content[:500]
                response = content[500:1000]
            else:
                # 短いコンテンツの場合は半分に分割
                mid_point = len(content) // 2
                user_input = content[:mid_point]
                response = content[mid_point:]
            
            # インタラクションID生成
            interaction_id = f"learning_{hashlib.md5(content.encode()).hexdigest()[:8]}"
            
            # エージェントのメモリシステムに直接保存
            if hasattr(self.agent, 'memory_system') and self.agent.memory_system:
                await self.agent.memory_system.store_conversation(
                    user_input=user_input,
                    agent_response=response,
                    metadata={
                        "interaction_id": interaction_id,
                        "learning_epoch": self.current_epoch,
                        "source": metadata.get('source', 'unknown'),
                        "conversation_id": metadata.get('conversation_id', ''),
                        "title": metadata.get('title', '')
                    }
                )
            
            # エージェント状態の更新
            if hasattr(self.agent, 'current_state') and self.agent.current_state:
                self.agent.current_state.total_interactions += 1
                self.agent.current_state.last_activity = datetime.now()
            
            self.logger.debug(f"Learning data processed: {interaction_id}")
            
        except Exception as e:
            self.logger.error(f"Error sending learning data to agent: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _record_conversation_learning(self, session_id: str, conversation: Dict[str, Any], cycle_number: int):
        """会話学習記録"""
        try:
            content = conversation.get('content', '') or str(conversation.get('messages', []))
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            with sqlite3.connect("data/continuous_learning.db") as conn:
                conn.execute("""
                    INSERT INTO conversation_learning 
                    (session_id, conversation_id, source, content_hash, learning_epoch, processed_at, quality_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    conversation.get('id', '') or conversation.get('conversation_id', ''),
                    conversation.get('source', ''),
                    content_hash,
                    self.current_epoch,
                    datetime.now().isoformat(),
                    0.5  # デフォルト品質スコア
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error recording conversation learning: {e}")
    
    def _record_learning_progress(self, session_id: str, cycle_number: int, conversations_processed: int):
        """学習進捗記録"""
        try:
            with sqlite3.connect("data/continuous_learning.db") as conn:
                conn.execute("""
                    INSERT INTO learning_progress 
                    (session_id, cycle_number, conversations_processed, learning_epoch, timestamp, performance_metrics)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    cycle_number,
                    conversations_processed,
                    self.current_epoch,
                    datetime.now().isoformat(),
                    json.dumps({
                        'total_processed': self.total_processed,
                        'learning_cycles': self.learning_cycles,
                        'remaining_time': str(self.end_time - datetime.now())
                    })
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error recording learning progress: {e}")
    
    def _complete_learning_session(self, session_id: str):
        """学習セッション完了"""
        try:
            with sqlite3.connect("data/continuous_learning.db") as conn:
                conn.execute("""
                    UPDATE learning_sessions 
                    SET end_time = ?, status = ?
                    WHERE session_id = ?
                """, (
                    datetime.now().isoformat(),
                    'completed',
                    session_id
                ))
                conn.commit()
                
            # 学習統計ログ
            duration = datetime.now() - self.start_time
            self.logger.info(f"Learning session completed:")
            self.logger.info(f"  Duration: {duration}")
            self.logger.info(f"  Total cycles: {self.learning_cycles}")
            self.logger.info(f"  Total processed: {self.total_processed}")
            self.logger.info(f"  Final epoch: {self.current_epoch}")
            
        except Exception as e:
            self.logger.error(f"Error completing learning session: {e}")
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """学習統計取得"""
        return {
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_processed': self.total_processed,
            'learning_cycles': self.learning_cycles,
            'current_epoch': self.current_epoch,
            'remaining_time': str(self.end_time - datetime.now()) if self.end_time else None
        }


async def main():
    """メイン関数"""
    try:
        # ログ設定
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/continuous_learning.log'),
                logging.StreamHandler()
            ]
        )
        
        # 継続学習システム作成
        learning_system = ContinuousLearningSystem(learning_duration_hours=4)
        
        # エージェント初期化
        if not await learning_system.initialize_agent():
            print("Failed to initialize agent")
            return
        
        # 4時間継続学習開始
        print("Starting 4-hour continuous learning session...")
        print("Press Ctrl+C to stop early")
        
        await learning_system.start_continuous_learning()
        
        # 学習完了
        stats = learning_system.get_learning_statistics()
        print("\nLearning completed!")
        print(f"Total processed: {stats['total_processed']}")
        print(f"Learning cycles: {stats['learning_cycles']}")
        print(f"Final epoch: {stats['current_epoch']}")
        
    except KeyboardInterrupt:
        print("\nLearning interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        logging.error(f"Main error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
