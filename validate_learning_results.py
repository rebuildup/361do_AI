#!/usr/bin/env python3
"""
Validate Learning Results
学習結果検証スクリプト
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any


class LearningResultValidator:
    """学習結果検証クラス"""
    
    def __init__(self, db_path: str = "data/continuous_learning.db"):
        self.db_path = db_path
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """全学習セッション取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM learning_sessions 
                    ORDER BY created_at DESC
                """)
                
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            print(f"Error getting sessions: {e}")
            return []
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """セッション要約取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # セッション情報
                cursor = conn.execute("""
                    SELECT * FROM learning_sessions 
                    WHERE session_id = ?
                """, (session_id,))
                session = cursor.fetchone()
                
                if not session:
                    return {}
                
                columns = [description[0] for description in cursor.description]
                session_data = dict(zip(columns, session))
                
                # 進捗統計
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_cycles,
                        MAX(cycle_number) as max_cycle,
                        SUM(conversations_processed) as total_processed,
                        MAX(learning_epoch) as max_epoch
                    FROM learning_progress 
                    WHERE session_id = ?
                """, (session_id,))
                progress_stats = cursor.fetchone()
                
                # 会話統計
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_conversations,
                        COUNT(DISTINCT source) as unique_sources,
                        AVG(quality_score) as avg_quality_score
                    FROM conversation_learning 
                    WHERE session_id = ?
                """, (session_id,))
                conversation_stats = cursor.fetchone()
                
                # ソース別統計
                cursor = conn.execute("""
                    SELECT source, COUNT(*) as count 
                    FROM conversation_learning 
                    WHERE session_id = ? 
                    GROUP BY source
                """, (session_id,))
                source_stats = dict(cursor.fetchall())
                
                return {
                    'session': session_data,
                    'progress_stats': {
                        'total_cycles': progress_stats[0] or 0,
                        'max_cycle': progress_stats[1] or 0,
                        'total_processed': progress_stats[2] or 0,
                        'max_epoch': progress_stats[3] or 0
                    },
                    'conversation_stats': {
                        'total_conversations': conversation_stats[0] or 0,
                        'unique_sources': conversation_stats[1] or 0,
                        'avg_quality_score': conversation_stats[2] or 0
                    },
                    'source_stats': source_stats
                }
                
        except Exception as e:
            print(f"Error getting session summary: {e}")
            return {}
    
    def validate_learning_effectiveness(self, session_id: str) -> Dict[str, Any]:
        """学習効果検証"""
        try:
            summary = self.get_session_summary(session_id)
            if not summary:
                return {}
            
            session = summary['session']
            progress = summary['progress_stats']
            conversations = summary['conversation_stats']
            
            # 基本指標
            duration_hours = session.get('duration_hours', 0)
            total_conversations = conversations.get('total_conversations', 0)
            total_cycles = progress.get('total_cycles', 0)
            max_epoch = progress.get('max_epoch', 0)
            
            # 効果指標計算
            conversations_per_hour = total_conversations / duration_hours if duration_hours > 0 else 0
            cycles_per_hour = total_cycles / duration_hours if duration_hours > 0 else 0
            epochs_per_hour = max_epoch / duration_hours if duration_hours > 0 else 0
            
            # 品質評価
            avg_quality = conversations.get('avg_quality_score', 0)
            quality_rating = "高" if avg_quality >= 0.7 else "中" if avg_quality >= 0.4 else "低"
            
            # 効率評価
            efficiency_rating = "高" if conversations_per_hour >= 100 else "中" if conversations_per_hour >= 50 else "低"
            
            return {
                'basic_metrics': {
                    'duration_hours': duration_hours,
                    'total_conversations': total_conversations,
                    'total_cycles': total_cycles,
                    'max_epoch': max_epoch
                },
                'efficiency_metrics': {
                    'conversations_per_hour': conversations_per_hour,
                    'cycles_per_hour': cycles_per_hour,
                    'epochs_per_hour': epochs_per_hour
                },
                'quality_metrics': {
                    'avg_quality_score': avg_quality,
                    'quality_rating': quality_rating
                },
                'efficiency_rating': efficiency_rating,
                'source_diversity': len(summary.get('source_stats', {}))
            }
            
        except Exception as e:
            print(f"Error validating learning effectiveness: {e}")
            return {}
    
    def generate_report(self, session_id: str) -> str:
        """学習レポート生成"""
        try:
            summary = self.get_session_summary(session_id)
            effectiveness = self.validate_learning_effectiveness(session_id)
            
            if not summary or not effectiveness:
                return "レポート生成に失敗しました"
            
            session = summary['session']
            progress = summary['progress_stats']
            conversations = summary['conversation_stats']
            source_stats = summary['source_stats']
            
            report = []
            report.append("=" * 80)
            report.append("📊 学習結果検証レポート")
            report.append("=" * 80)
            report.append(f"セッションID: {session_id}")
            report.append(f"開始時間: {session.get('start_time', 'N/A')}")
            report.append(f"終了時間: {session.get('end_time', 'N/A')}")
            report.append(f"ステータス: {session.get('status', 'N/A')}")
            report.append("")
            
            # 基本統計
            report.append("📈 基本統計:")
            report.append(f"  学習時間: {session.get('duration_hours', 0):.1f}時間")
            report.append(f"  総会話数: {conversations.get('total_conversations', 0)}")
            report.append(f"  総サイクル数: {progress.get('total_cycles', 0)}")
            report.append(f"  最大エポック: {progress.get('max_epoch', 0)}")
            report.append("")
            
            # 効率指標
            report.append("⚡ 効率指標:")
            report.append(f"  時間あたり会話数: {effectiveness['efficiency_metrics']['conversations_per_hour']:.1f}")
            report.append(f"  時間あたりサイクル数: {effectiveness['efficiency_metrics']['cycles_per_hour']:.1f}")
            report.append(f"  時間あたりエポック数: {effectiveness['efficiency_metrics']['epochs_per_hour']:.1f}")
            report.append(f"  効率評価: {effectiveness['efficiency_rating']}")
            report.append("")
            
            # 品質指標
            report.append("🎯 品質指標:")
            report.append(f"  平均品質スコア: {effectiveness['quality_metrics']['avg_quality_score']:.3f}")
            report.append(f"  品質評価: {effectiveness['quality_metrics']['quality_rating']}")
            report.append("")
            
            # データソース
            report.append("📚 データソース:")
            for source, count in source_stats.items():
                report.append(f"  {source}: {count}会話")
            report.append(f"  ソース多様性: {effectiveness['source_diversity']}種類")
            report.append("")
            
            # 総合評価
            report.append("🏆 総合評価:")
            if effectiveness['efficiency_rating'] == "高" and effectiveness['quality_metrics']['quality_rating'] == "高":
                report.append("  ✅ 優秀: 効率と品質の両方が高い")
            elif effectiveness['efficiency_rating'] == "高" or effectiveness['quality_metrics']['quality_rating'] == "高":
                report.append("  ⚠️  良好: 効率または品質のいずれかが高い")
            else:
                report.append("  ❌ 要改善: 効率と品質の両方を向上させる必要がある")
            
            report.append("=" * 80)
            
            return "\n".join(report)
            
        except Exception as e:
            return f"レポート生成エラー: {e}"
    
    def save_report(self, session_id: str, output_file: str = None):
        """レポート保存"""
        try:
            report = self.generate_report(session_id)
            
            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"logs/learning_report_{session_id}_{timestamp}.txt"
            
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"レポートを保存しました: {output_file}")
            
        except Exception as e:
            print(f"レポート保存エラー: {e}")


def main():
    """メイン関数"""
    validator = LearningResultValidator()
    
    print("=" * 80)
    print("📊 学習結果検証システム")
    print("=" * 80)
    print()
    
    # 全セッション取得
    sessions = validator.get_all_sessions()
    
    if not sessions:
        print("学習セッションが見つかりません")
        return
    
    print("利用可能な学習セッション:")
    for i, session in enumerate(sessions):
        status_icon = "✅" if session['status'] == 'completed' else "🔄" if session['status'] == 'running' else "❌"
        print(f"  {i+1}. {status_icon} {session['session_id']} ({session['status']})")
    
    print()
    
    # セッション選択
    try:
        choice = int(input("検証するセッション番号を選択してください: ")) - 1
        if 0 <= choice < len(sessions):
            selected_session = sessions[choice]
            session_id = selected_session['session_id']
            
            print()
            print("検証中...")
            
            # レポート生成・表示
            report = validator.generate_report(session_id)
            print(report)
            
            # レポート保存
            save_choice = input("\nレポートをファイルに保存しますか？ (y/N): ")
            if save_choice.lower() == 'y':
                validator.save_report(session_id)
            
        else:
            print("無効な選択です")
            
    except ValueError:
        print("無効な入力です")
    except KeyboardInterrupt:
        print("\n検証を終了しました")


if __name__ == "__main__":
    main()
