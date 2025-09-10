#!/usr/bin/env python3
"""
Monitor Learning Progress
学習進捗監視スクリプト
"""

import sqlite3
import json
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any


class LearningProgressMonitor:
    """学習進捗監視クラス"""
    
    def __init__(self, db_path: str = "data/continuous_learning.db"):
        self.db_path = db_path
    
    def get_current_session(self) -> Dict[str, Any]:
        """現在の学習セッション取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM learning_sessions 
                    WHERE status = 'running' 
                    ORDER BY created_at DESC 
                    LIMIT 1
                """)
                
                row = cursor.fetchone()
                if row:
                    columns = [description[0] for description in cursor.description]
                    return dict(zip(columns, row))
                return {}
                
        except Exception as e:
            print(f"Error getting current session: {e}")
            return {}
    
    def get_learning_progress(self, session_id: str) -> List[Dict[str, Any]]:
        """学習進捗取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM learning_progress 
                    WHERE session_id = ? 
                    ORDER BY cycle_number DESC
                """, (session_id,))
                
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            print(f"Error getting learning progress: {e}")
            return []
    
    def get_conversation_stats(self, session_id: str) -> Dict[str, Any]:
        """会話統計取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 総会話数
                cursor = conn.execute("""
                    SELECT COUNT(*) as total_conversations 
                    FROM conversation_learning 
                    WHERE session_id = ?
                """, (session_id,))
                total_conversations = cursor.fetchone()[0]
                
                # ソース別統計
                cursor = conn.execute("""
                    SELECT source, COUNT(*) as count 
                    FROM conversation_learning 
                    WHERE session_id = ? 
                    GROUP BY source
                """, (session_id,))
                source_stats = dict(cursor.fetchall())
                
                # エポック別統計
                cursor = conn.execute("""
                    SELECT learning_epoch, COUNT(*) as count 
                    FROM conversation_learning 
                    WHERE session_id = ? 
                    GROUP BY learning_epoch 
                    ORDER BY learning_epoch
                """, (session_id,))
                epoch_stats = dict(cursor.fetchall())
                
                return {
                    'total_conversations': total_conversations,
                    'source_stats': source_stats,
                    'epoch_stats': epoch_stats
                }
                
        except Exception as e:
            print(f"Error getting conversation stats: {e}")
            return {}
    
    def display_progress(self):
        """進捗表示"""
        session = self.get_current_session()
        
        if not session:
            print("現在実行中の学習セッションが見つかりません")
            return
        
        session_id = session['session_id']
        start_time = datetime.fromisoformat(session['start_time'])
        duration_hours = session['duration_hours']
        end_time = start_time + timedelta(hours=duration_hours)
        current_time = datetime.now()
        
        # 進捗計算
        elapsed = current_time - start_time
        total_duration = timedelta(hours=duration_hours)
        progress_percent = min((elapsed / total_duration) * 100, 100)
        remaining = max(end_time - current_time, timedelta(0))
        
        print("=" * 80)
        print("📊 学習進捗監視")
        print("=" * 80)
        print(f"セッションID: {session_id}")
        print(f"開始時間: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"予定終了時間: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"現在時刻: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"経過時間: {elapsed}")
        print(f"残り時間: {remaining}")
        print(f"進捗: {progress_percent:.1f}%")
        print()
        
        # 学習サイクル進捗
        progress_data = self.get_learning_progress(session_id)
        if progress_data:
            latest = progress_data[0]
            print("📈 最新学習サイクル:")
            print(f"   サイクル番号: {latest['cycle_number']}")
            print(f"   処理会話数: {latest['conversations_processed']}")
            print(f"   学習エポック: {latest['learning_epoch']}")
            print(f"   タイムスタンプ: {latest['timestamp']}")
            
            # パフォーマンスメトリクス
            try:
                metrics = json.loads(latest['performance_metrics'])
                print(f"   総処理数: {metrics.get('total_processed', 0)}")
                print(f"   学習サイクル数: {metrics.get('learning_cycles', 0)}")
            except:
                pass
            print()
        
        # 会話統計
        stats = self.get_conversation_stats(session_id)
        if stats:
            print("📋 会話統計:")
            print(f"   総会話数: {stats['total_conversations']}")
            
            if stats['source_stats']:
                print("   ソース別:")
                for source, count in stats['source_stats'].items():
                    print(f"     - {source}: {count}")
            
            if stats['epoch_stats']:
                print("   エポック別:")
                for epoch, count in sorted(stats['epoch_stats'].items()):
                    print(f"     - エポック {epoch}: {count}")
            print()
        
        # 進捗バー
        bar_length = 50
        filled_length = int(bar_length * progress_percent / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"進捗バー: [{bar}] {progress_percent:.1f}%")
        print("=" * 80)


def main():
    """メイン関数"""
    monitor = LearningProgressMonitor()
    
    try:
        while True:
            # 画面クリア（Windows）
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # 進捗表示
            monitor.display_progress()
            
            # 5秒待機
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n監視を終了しました")


if __name__ == "__main__":
    main()
