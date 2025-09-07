#!/usr/bin/env python3
"""
データベース初期化スクリプト
Database Initialization Script
"""

import sqlite3
import os
from pathlib import Path

def init_database():
    """データベースを初期化"""
    
    # データベースパス
    db_path = "data/self_learning_agent.db"
    
    # ディレクトリ作成
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # エージェント状態テーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_states (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    current_prompt_version TEXT,
                    learning_epoch INTEGER DEFAULT 0,
                    total_interactions INTEGER DEFAULT 0,
                    reward_score REAL DEFAULT 0.0,
                    evolution_generation INTEGER DEFAULT 0,
                    last_activity TEXT,
                    performance_metrics TEXT
                )
            """)
            
            # プロンプトテンプレートテーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prompt_templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    template TEXT NOT NULL,
                    description TEXT,
                    category TEXT,
                    version TEXT DEFAULT '1.0.0',
                    created_at TEXT,
                    updated_at TEXT,
                    usage_count INTEGER DEFAULT 0,
                    performance_score REAL DEFAULT 0.0
                )
            """)
            
            # チューニングデータテーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tuning_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    user_input TEXT,
                    agent_response TEXT,
                    reward_score REAL,
                    metadata TEXT,
                    created_at TEXT,
                    FOREIGN KEY (session_id) REFERENCES agent_states (session_id)
                )
            """)
            
            # 進化候補テーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evolution_candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    generation INTEGER,
                    prompt_template_id INTEGER,
                    fitness_score REAL,
                    mutation_history TEXT,
                    created_at TEXT,
                    FOREIGN KEY (prompt_template_id) REFERENCES prompt_templates (id)
                )
            """)
            
            # 報酬履歴テーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reward_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    reward_type TEXT,
                    reward_value REAL,
                    context TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (session_id) REFERENCES agent_states (session_id)
                )
            """)
            
            # 会話履歴テーブル（永続セッション用）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    user_input TEXT,
                    agent_response TEXT,
                    metadata TEXT,
                    created_at TEXT,
                    FOREIGN KEY (session_id) REFERENCES agent_states (session_id)
                )
            """)
            
            conn.commit()
            print("✅ データベース初期化完了")
            
    except Exception as e:
        print(f"❌ データベース初期化エラー: {e}")
        raise

if __name__ == "__main__":
    init_database()
