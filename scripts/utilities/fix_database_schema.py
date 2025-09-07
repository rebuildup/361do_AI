#!/usr/bin/env python3
"""
データベーススキーマ修正スクリプト
conversationsテーブルの不足カラムを追加
"""

import sqlite3
import os
from pathlib import Path

def fix_database_schema():
    """データベーススキーマを修正"""
    
    db_path = "data/self_learning_agent.db"
    print(f"データベースパス: {db_path}")
    
    # データベースファイルが存在しない場合は作成
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"データベースディレクトリを作成: {Path(db_path).parent}")
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # 現在のテーブル構造を確認
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            print(f"既存のテーブル: {[table[0] for table in tables]}")
            
            # conversationsテーブルの存在確認
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'")
            if cursor.fetchone():
                print("conversationsテーブルが存在します")
                
                # 現在のカラム構造を確認
                cursor.execute("PRAGMA table_info(conversations)")
                columns = cursor.fetchall()
                print("現在のカラム:")
                for col in columns:
                    print(f"  {col[1]} ({col[2]})")
                
                # 必要なカラムを追加
                required_columns = {
                    'user_input': 'TEXT',
                    'agent_response': 'TEXT',
                    'timestamp': 'TIMESTAMP',
                    'importance_score': 'REAL',
                    'extra_metadata': 'TEXT'
                }
                
                for col_name, col_type in required_columns.items():
                    # カラムが存在するかチェック
                    cursor.execute(f"PRAGMA table_info(conversations)")
                    existing_columns = [col[1] for col in cursor.fetchall()]
                    
                    if col_name not in existing_columns:
                        try:
                            cursor.execute(f"ALTER TABLE conversations ADD COLUMN {col_name} {col_type}")
                            print(f"カラム '{col_name}' を追加しました")
                        except sqlite3.Error as e:
                            print(f"カラム '{col_name}' の追加に失敗: {e}")
                    else:
                        print(f"カラム '{col_name}' は既に存在します")
            else:
                print("conversationsテーブルが存在しません。作成します。")
                
                # conversationsテーブルを作成
                cursor.execute("""
                    CREATE TABLE conversations (
                        id TEXT PRIMARY KEY,
                        session_id TEXT,
                        user_input TEXT,
                        agent_response TEXT,
                        timestamp TIMESTAMP,
                        importance_score REAL,
                        extra_metadata TEXT
                    )
                """)
                print("conversationsテーブルを作成しました")
            
            # 他の必要なテーブルも確認・作成
            required_tables = {
                'agent_states': """
                    CREATE TABLE IF NOT EXISTS agent_states (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT,
                        current_prompt_version TEXT,
                        learning_epoch INTEGER,
                        total_interactions INTEGER,
                        reward_score REAL,
                        evolution_generation INTEGER,
                        last_activity TIMESTAMP,
                        performance_metrics TEXT
                    )
                """,
                'prompt_templates': """
                    CREATE TABLE IF NOT EXISTS prompt_templates (
                        version TEXT PRIMARY KEY,
                        content TEXT,
                        metadata TEXT,
                        performance_score REAL,
                        usage_count INTEGER,
                        created_at TIMESTAMP,
                        last_modified TIMESTAMP
                    )
                """,
                'tuning_data': """
                    CREATE TABLE IF NOT EXISTS tuning_data (
                        id TEXT PRIMARY KEY,
                        content TEXT,
                        data_type TEXT,
                        quality_score REAL,
                        usage_count INTEGER,
                        created_at TIMESTAMP,
                        tags TEXT
                    )
                """,
                'evolution_candidates': """
                    CREATE TABLE IF NOT EXISTS evolution_candidates (
                        id TEXT PRIMARY KEY,
                        parent_ids TEXT,
                        prompt_template_version TEXT,
                        tuning_data_ids TEXT,
                        fitness_score REAL,
                        generation INTEGER,
                        created_at TIMESTAMP
                    )
                """,
                'reward_history': """
                    CREATE TABLE IF NOT EXISTS reward_history (
                        interaction_id TEXT PRIMARY KEY,
                        reward_type TEXT,
                        value REAL,
                        context TEXT,
                        timestamp TIMESTAMP
                    )
                """
            }
            
            for table_name, create_sql in required_tables.items():
                cursor.execute(create_sql)
                print(f"{table_name}テーブルを確認/作成しました")
            
            conn.commit()
            print("データベーススキーマの修正が完了しました")
            
    except Exception as e:
        print(f"データベース修正エラー: {e}")
        raise

if __name__ == "__main__":
    print("データベーススキーマ修正を開始...")
    fix_database_schema()
    print("修正完了！")
