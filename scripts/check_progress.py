#!/usr/bin/env python3
"""
Check Learning Progress
学習進捗確認スクリプト
"""

import sqlite3
import json
from datetime import datetime, timedelta

def check_progress():
    """学習進捗確認"""
    try:
        # データベース接続
        conn = sqlite3.connect('data/continuous_learning.db')
        
        # 現在のセッション取得
        cursor = conn.execute("SELECT * FROM learning_sessions WHERE status = 'running' ORDER BY created_at DESC LIMIT 1")
        session = cursor.fetchone()
        
        if session:
            print('=' * 60)
            print('📊 現在の学習セッション')
            print('=' * 60)
            print(f'セッションID: {session[1]}')
            print(f'開始時間: {session[2]}')
            print(f'学習時間: {session[3]}時間')
            print(f'ステータス: {session[6]}')
            print()
            
            # 進捗統計
            cursor = conn.execute('SELECT COUNT(*) FROM learning_progress WHERE session_id = ?', (session[1],))
            cycles = cursor.fetchone()[0]
            
            cursor = conn.execute('SELECT SUM(conversations_processed) FROM learning_progress WHERE session_id = ?', (session[1],))
            total_processed = cursor.fetchone()[0] or 0
            
            cursor = conn.execute('SELECT COUNT(*) FROM conversation_learning WHERE session_id = ?', (session[1],))
            conversations_learned = cursor.fetchone()[0]
            
            print('📈 進捗統計:')
            print(f'  完了サイクル数: {cycles}')
            print(f'  処理済み会話数: {total_processed}')
            print(f'  学習済み会話数: {conversations_learned}')
            print()
            
            # 最新の進捗
            cursor = conn.execute('SELECT * FROM learning_progress WHERE session_id = ? ORDER BY cycle_number DESC LIMIT 1', (session[1],))
            latest = cursor.fetchone()
            if latest:
                print('🔄 最新進捗:')
                print(f'  最新サイクル: {latest[2]}')
                print(f'  最新エポック: {latest[4]}')
                print(f'  最新タイムスタンプ: {latest[5]}')
                
                # パフォーマンスメトリクス
                try:
                    metrics = json.loads(latest[6])
                    print(f'  総処理数: {metrics.get("total_processed", 0)}')
                    print(f'  学習サイクル数: {metrics.get("learning_cycles", 0)}')
                    if 'remaining_time' in metrics:
                        print(f'  残り時間: {metrics["remaining_time"]}')
                except:
                    pass
            print()
            
            # ソース別統計
            cursor = conn.execute('SELECT source, COUNT(*) FROM conversation_learning WHERE session_id = ? GROUP BY source', (session[1],))
            source_stats = cursor.fetchall()
            if source_stats:
                print('📚 ソース別統計:')
                for source, count in source_stats:
                    print(f'  {source}: {count}会話')
                print()
            
            # 時間経過計算
            start_time = datetime.fromisoformat(session[2])
            current_time = datetime.now()
            elapsed = current_time - start_time
            duration_hours = session[3] or 4  # デフォルト4時間
            end_time = start_time + timedelta(hours=duration_hours)
            remaining = end_time - current_time
            
            print('⏰ 時間情報:')
            print(f'  経過時間: {elapsed}')
            print(f'  残り時間: {remaining}')
            print(f'  進捗率: {(elapsed.total_seconds() / (duration_hours * 3600)) * 100:.1f}%')
            
        else:
            print('❌ 実行中の学習セッションが見つかりません')
            
        conn.close()
        
    except Exception as e:
        print(f'❌ エラー: {e}')

if __name__ == "__main__":
    check_progress()
