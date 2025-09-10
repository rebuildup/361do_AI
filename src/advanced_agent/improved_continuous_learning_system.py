#!/usr/bin/env python3
"""
Improved Continuous Learning System
改善された4時間継続学習システム
"""

import asyncio
import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import sqlite3
import random

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from conversation_data_processor import ConversationDataProcessor

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
    
    # フォールバック用の簡単なエージェント
    class SelfLearningAgent:
        def __init__(self, **kwargs):
            self.logger = get_logger()
        
        async def initialize_session(self, **kwargs):
            self.logger.info("フォールバックエージェント初期化")
            return True


class ImprovedContinuousLearningSystem:
    """改善された継続学習システム"""
    
    def __init__(self, learning_duration_hours: int = 4):
        self.logger = get_logger()
        self.learning_duration_hours = learning_duration_hours
        self.db_path = "data/improved_continuous_learning.db"
        
        # データプロセッサー（同じデータベースを使用）
        self.data_processor = ConversationDataProcessor(db_path="data/conversation_processing.db")
        
        # エージェント
        self.agent: Optional[SelfLearningAgent] = None
        
        # 学習設定
        self.learning_config = {
            "batch_size": 10,  # バッチサイズを増加
            "learning_interval": 20,  # 学習間隔を短縮
            "max_conversations_per_cycle": 100,  # サイクルあたり最大会話数を増加
            "learning_rate": 0.1,
            "memory_retention_days": 7,
            "quality_threshold": 0.1,  # 品質閾値を下げる（より多くのデータを使用）
            "max_content_length": 10000,  # 最大コンテンツ長を増加
            "min_content_length": 20   # 最小コンテンツ長を下げる
        }
        
        # 学習統計
        self.learning_stats = {
            "session_id": f"learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "start_time": None,
            "end_time": None,
            "total_processed": 0,
            "learning_cycles": 0,
            "current_epoch": 0,
            "total_conversations": 0,
            "quality_scores": [],
            "processing_errors": 0,
            "duplicate_errors": 0,  # 重複エラーを別途カウント
            "source_stats": {
                "chatgpt": {"processed": 0, "errors": 0},
                "claude": {"processed": 0, "errors": 0}
            }
        }
        
        # データベース初期化
        self._init_database()
    
    def _init_database(self):
        """データベース初期化"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    total_processed INTEGER,
                    learning_cycles INTEGER,
                    max_epoch INTEGER,
                    total_conversations INTEGER,
                    avg_quality_score REAL,
                    processing_errors INTEGER,
                    duplicate_errors INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'running'
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_cycles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    cycle_number INTEGER,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    conversations_processed INTEGER,
                    avg_quality_score REAL,
                    epoch INTEGER,
                    FOREIGN KEY (session_id) REFERENCES learning_sessions (session_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_conversations (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    cycle_number INTEGER,
                    source TEXT,
                    conversation_id TEXT,
                    title TEXT,
                    content TEXT,
                    quality_score REAL,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES learning_sessions (session_id)
                )
            """)
            
            conn.commit()
    
    async def initialize_agent(self) -> bool:
        """エージェント初期化"""
        try:
            self.logger.info("エージェント初期化中...")
            
            # エージェント作成
            self.agent = SelfLearningAgent()
            
            # エージェント初期化
            await self.agent.initialize_session()
            
            self.logger.info("✅ エージェント初期化完了")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ エージェント初期化失敗: {e}")
            return False
    
    def _load_processed_conversations(self, limit: int = None) -> List[Dict[str, Any]]:
        """処理済み会話データを読み込み"""
        try:
            # データベースファイルの存在確認
            db_path = self.data_processor.db_path
            if not os.path.exists(db_path):
                self.logger.error(f"データベースファイルが見つかりません: {db_path}")
                return []
            
            # データベースサイズ確認
            db_size = os.path.getsize(db_path)
            self.logger.info(f"データベースファイル: {db_path} (サイズ: {db_size:,} bytes)")
            
            # データプロセッサーから処理済み会話を取得
            conversations = self.data_processor.get_processed_conversations(limit=limit or 1000)
            self.logger.info(f"データベースから読み込み: {len(conversations)}件")
            
            # 品質フィルタリング
            filtered_conversations = []
            quality_filtered = 0
            length_filtered = 0
            
            for conv in conversations:
                # 品質チェック
                if conv.get("quality_score", 0) < self.learning_config["quality_threshold"]:
                    quality_filtered += 1
                    continue
                
                # コンテンツ長チェック
                content_length = len(conv.get("content", ""))
                if (content_length < self.learning_config["min_content_length"] or 
                    content_length > self.learning_config["max_content_length"]):
                    length_filtered += 1
                    continue
                
                filtered_conversations.append(conv)
            
            # シャッフル
            random.shuffle(filtered_conversations)
            
            self.logger.info(f"フィルタリング結果:")
            self.logger.info(f"  - 元データ: {len(conversations)}件")
            self.logger.info(f"  - 品質フィルタ: {quality_filtered}件除外")
            self.logger.info(f"  - 長さフィルタ: {length_filtered}件除外")
            self.logger.info(f"  - 最終結果: {len(filtered_conversations)}件")
            
            return filtered_conversations
            
        except Exception as e:
            self.logger.error(f"会話データ読み込みエラー: {e}")
            import traceback
            self.logger.error(f"詳細エラー: {traceback.format_exc()}")
            return []
    
    def _prepare_learning_batch(self, conversations: List[Dict[str, Any]], batch_size: int) -> List[Dict[str, Any]]:
        """学習バッチを準備"""
        batch = conversations[:batch_size]
        
        # バッチ内の会話を学習用にフォーマット
        formatted_batch = []
        for conv in batch:
            try:
                # 会話内容を学習用にフォーマット
                formatted_conv = {
                    "id": conv["id"],
                    "source": conv["source"],
                    "title": conv["title"],
                    "content": conv["content"],
                    "quality_score": conv["quality_score"],
                    "learning_prompt": self._create_learning_prompt(conv)
                }
                formatted_batch.append(formatted_conv)
                
            except Exception as e:
                self.logger.error(f"会話フォーマットエラー: {e}")
                self.learning_stats["processing_errors"] += 1
        
        return formatted_batch
    
    def _create_learning_prompt(self, conversation: Dict[str, Any]) -> str:
        """学習用プロンプトを作成"""
        content = conversation.get("content", "")
        title = conversation.get("title", "Untitled")
        source = conversation.get("source", "unknown")
        
        # 学習用プロンプトテンプレート
        prompt = f"""
学習データ: {source.upper()}会話
タイトル: {title}
内容: {content[:2000]}...

この会話から学習し、同様の品質とスタイルで応答できるよう改善してください。
"""
        return prompt.strip()
    
    async def _process_learning_batch(self, batch: List[Dict[str, Any]], cycle_number: int) -> Dict[str, Any]:
        """学習バッチを処理"""
        batch_stats = {
            "processed": 0,
            "errors": 0,
            "avg_quality": 0.0,
            "sources": {"chatgpt": 0, "claude": 0}
        }
        
        try:
            for conv in batch:
                try:
                    # エージェントに学習データを送信
                    if self.agent:
                        # 学習プロンプトをエージェントに送信
                        learning_prompt = conv["learning_prompt"]
                        
                        # エージェントの学習処理（実際の実装に応じて調整）
                        # ここでは模擬的な学習処理
                        await asyncio.sleep(0.1)  # 学習処理の模擬
                        
                        # 統計更新
                        batch_stats["processed"] += 1
                        batch_stats["sources"][conv["source"]] += 1
                        batch_stats["avg_quality"] += conv["quality_score"]
                        
                        # データベースに記録
                        self._save_processed_conversation(conv, cycle_number)
                        
                except Exception as e:
                    self.logger.error(f"会話学習エラー: {e}")
                    batch_stats["errors"] += 1
                    self.learning_stats["processing_errors"] += 1
            
            # 平均品質スコア計算
            if batch_stats["processed"] > 0:
                batch_stats["avg_quality"] /= batch_stats["processed"]
            
            return batch_stats
            
        except Exception as e:
            self.logger.error(f"バッチ処理エラー: {e}")
            return batch_stats
    
    def _save_processed_conversation(self, conversation: Dict[str, Any], cycle_number: int):
        """処理済み会話をデータベースに保存"""
        try:
            # 一意のIDを生成（会話ID + サイクル番号）
            unique_id = f"{conversation['id']}_{cycle_number}"
            
            with sqlite3.connect(self.db_path) as conn:
                # 既存のレコードをチェック
                cursor = conn.execute("SELECT id FROM processed_conversations WHERE id = ?", (unique_id,))
                if cursor.fetchone():
                    # 既に存在する場合はスキップ（重複エラーとしてカウント）
                    self.learning_stats["duplicate_errors"] += 1
                    return
                
                conn.execute("""
                    INSERT INTO processed_conversations 
                    (id, session_id, cycle_number, source, conversation_id, title, content, quality_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    unique_id,
                    self.learning_stats["session_id"],
                    cycle_number,
                    conversation["source"],
                    conversation.get("conversation_id", ""),
                    conversation["title"],
                    conversation["content"],
                    conversation["quality_score"]
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"会話保存エラー: {e}")
    
    def _save_cycle_stats(self, cycle_number: int, batch_stats: Dict[str, Any], start_time: datetime, end_time: datetime):
        """サイクル統計を保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO learning_cycles 
                    (session_id, cycle_number, start_time, end_time, conversations_processed, avg_quality_score, epoch)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    self.learning_stats["session_id"],
                    cycle_number,
                    start_time,
                    end_time,
                    batch_stats["processed"],
                    batch_stats["avg_quality"],
                    self.learning_stats["current_epoch"]
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"サイクル統計保存エラー: {e}")
    
    def _update_learning_stats(self, batch_stats: Dict[str, Any]):
        """学習統計を更新"""
        self.learning_stats["total_processed"] += batch_stats["processed"]
        self.learning_stats["learning_cycles"] += 1
        self.learning_stats["current_epoch"] += 1
        
        # ソース別統計更新
        for source, count in batch_stats["sources"].items():
            self.learning_stats["source_stats"][source]["processed"] += count
        
        # 品質スコア記録
        if batch_stats["avg_quality"] > 0:
            self.learning_stats["quality_scores"].append(batch_stats["avg_quality"])
    
    def _save_session_stats(self):
        """セッション統計を保存"""
        try:
            avg_quality = 0.0
            if self.learning_stats["quality_scores"]:
                avg_quality = sum(self.learning_stats["quality_scores"]) / len(self.learning_stats["quality_scores"])
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO learning_sessions 
                    (session_id, start_time, end_time, total_processed, learning_cycles, max_epoch, 
                     total_conversations, avg_quality_score, processing_errors, duplicate_errors, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    self.learning_stats["session_id"],
                    self.learning_stats["start_time"],
                    self.learning_stats["end_time"],
                    self.learning_stats["total_processed"],
                    self.learning_stats["learning_cycles"],
                    self.learning_stats["current_epoch"],
                    self.learning_stats["total_conversations"],
                    avg_quality,
                    self.learning_stats["processing_errors"],
                    self.learning_stats["duplicate_errors"],
                    "completed"
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"セッション統計保存エラー: {e}")
    
    async def start_continuous_learning(self):
        """継続学習開始"""
        self.learning_stats["start_time"] = datetime.now()
        end_time = self.learning_stats["start_time"] + timedelta(hours=self.learning_duration_hours)
        
        self.logger.info("=" * 60)
        self.logger.info("改善された4時間継続学習開始")
        self.logger.info("=" * 60)
        self.logger.info(f"学習時間: {self.learning_duration_hours}時間")
        self.logger.info(f"終了予定: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 60)
        
        # 処理済み会話データを読み込み
        conversations = self._load_processed_conversations()
        self.learning_stats["total_conversations"] = len(conversations)
        
        if not conversations:
            self.logger.error("❌ 学習データが見つかりません")
            return
        
        self.logger.info(f"📊 学習データ: {len(conversations)}件")
        
        cycle_number = 0
        
        try:
            while datetime.now() < end_time:
                cycle_start_time = datetime.now()
                cycle_number += 1
                
                # 学習バッチを準備
                batch = self._prepare_learning_batch(conversations, self.learning_config["batch_size"])
                
                if not batch:
                    self.logger.warning("学習バッチが空です")
                    break
                
                # バッチを処理
                batch_stats = await self._process_learning_batch(batch, cycle_number)
                cycle_end_time = datetime.now()
                
                # 統計更新
                self._update_learning_stats(batch_stats)
                self._save_cycle_stats(cycle_number, batch_stats, cycle_start_time, cycle_end_time)
                
                # 進捗表示
                elapsed_time = datetime.now() - self.learning_stats["start_time"]
                remaining_time = end_time - datetime.now()
                
                self.logger.info(f"サイクル {cycle_number}: 処理済み {batch_stats['processed']}件, "
                               f"品質スコア {batch_stats['avg_quality']:.3f}, "
                               f"経過時間 {elapsed_time}, 残り時間 {remaining_time}")
                
                # 学習間隔待機
                await asyncio.sleep(self.learning_config["learning_interval"])
                
                # 会話データをシャッフル（次のサイクル用）
                random.shuffle(conversations)
        
        except KeyboardInterrupt:
            self.logger.info("⚠️ 学習がユーザーによって中断されました")
        except Exception as e:
            self.logger.error(f"❌ 学習エラー: {e}")
        finally:
            # 学習完了処理
            self.learning_stats["end_time"] = datetime.now()
            self._save_session_stats()
            
            # 最終統計表示
            self._display_final_stats()
    
    def _display_final_stats(self):
        """最終統計を表示"""
        self.logger.info("=" * 60)
        self.logger.info("🎉 学習完了!")
        self.logger.info("=" * 60)
        self.logger.info(f"総処理数: {self.learning_stats['total_processed']}")
        self.logger.info(f"学習サイクル数: {self.learning_stats['learning_cycles']}")
        self.logger.info(f"最終エポック: {self.learning_stats['current_epoch']}")
        self.logger.info(f"開始時間: {self.learning_stats['start_time']}")
        self.logger.info(f"終了時間: {self.learning_stats['end_time']}")
        
        if self.learning_stats["quality_scores"]:
            avg_quality = sum(self.learning_stats["quality_scores"]) / len(self.learning_stats["quality_scores"])
            self.logger.info(f"平均品質スコア: {avg_quality:.3f}")
        
        self.logger.info(f"エラー数: {self.learning_stats['processing_errors']}")
        self.logger.info(f"重複エラー数: {self.learning_stats['duplicate_errors']}")
        
        # ソース別統計
        self.logger.info("ソース別統計:")
        for source, stats in self.learning_stats["source_stats"].items():
            self.logger.info(f"  {source}: {stats['processed']}件")
        
        self.logger.info("=" * 60)
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """学習統計を取得"""
        return self.learning_stats.copy()


async def main():
    """メイン関数"""
    print("=" * 60)
    print("361do_AI 改善された4時間継続学習システム")
    print("=" * 60)
    print()
    
    try:
        # 継続学習システム作成
        learning_system = ImprovedContinuousLearningSystem(learning_duration_hours=4)
        
        print("エージェント初期化中...")
        # エージェント初期化
        if not await learning_system.initialize_agent():
            print("❌ エージェント初期化に失敗しました")
            return
        
        print("✅ エージェント初期化完了")
        print()
        
        # データ処理確認
        print("データ処理確認中...")
        processed_conversations = learning_system._load_processed_conversations(limit=100)
        print(f"📊 処理済み会話数: {len(processed_conversations)}")
        
        if not processed_conversations:
            print("❌ 処理済み学習データが見つかりません")
            print("先に conversation_data_processor.py を実行してください")
            return
        
        # 学習設定表示
        print("学習設定:")
        print(f"   - 学習時間: 4時間")
        print(f"   - バッチサイズ: {learning_system.learning_config['batch_size']}")
        print(f"   - 学習間隔: {learning_system.learning_config['learning_interval']}秒")
        print(f"   - サイクルあたり最大会話数: {learning_system.learning_config['max_conversations_per_cycle']}")
        print(f"   - 品質閾値: {learning_system.learning_config['quality_threshold']}")
        print()
        
        # 確認
        response = input("改善された4時間継続学習を開始しますか？ (y/N): ")
        if response.lower() != 'y':
            print("学習をキャンセルしました")
            return
        
        print()
        print("🚀 改善された4時間継続学習開始...")
        print("   Ctrl+C で早期終了可能")
        print("=" * 60)
        
        # 4時間継続学習開始
        await learning_system.start_continuous_learning()
        
    except KeyboardInterrupt:
        print("\n⚠️  学習がユーザーによって中断されました")
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
