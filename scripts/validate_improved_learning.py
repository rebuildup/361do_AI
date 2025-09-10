#!/usr/bin/env python3
"""
Validate Improved Learning Results
改善された学習結果検証スクリプト
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, Any, List
import statistics


class ImprovedLearningValidator:
    """改善された学習結果検証システム"""
    
    def __init__(self, db_path: str = "data/improved_continuous_learning.db"):
        self.db_path = db_path
        self.results = {}
    
    def validate_learning_results(self) -> Dict[str, Any]:
        """学習結果を検証"""
        print("=" * 60)
        print("📊 改善された学習結果検証")
        print("=" * 60)
        
        if not os.path.exists(self.db_path):
            print(f"❌ データベースファイルが見つかりません: {self.db_path}")
            return {}
        
        # セッション統計を取得
        session_stats = self._get_session_stats()
        if not session_stats:
            print("❌ 学習セッションが見つかりません")
            return {}
        
        # サイクル統計を取得
        cycle_stats = self._get_cycle_stats()
        
        # 会話統計を取得
        conversation_stats = self._get_conversation_stats()
        
        # 品質分析
        quality_analysis = self._analyze_quality()
        
        # 効率分析
        efficiency_analysis = self._analyze_efficiency(session_stats)
        
        # 結果統合
        self.results = {
            "session_stats": session_stats,
            "cycle_stats": cycle_stats,
            "conversation_stats": conversation_stats,
            "quality_analysis": quality_analysis,
            "efficiency_analysis": efficiency_analysis,
            "validation_timestamp": datetime.now().isoformat()
        }
        
        # 結果表示
        self._display_results()
        
        # レポート生成
        self._generate_report()
        
        return self.results
    
    def _get_session_stats(self) -> Dict[str, Any]:
        """セッション統計を取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM learning_sessions 
                    ORDER BY start_time DESC 
                    LIMIT 1
                """)
                row = cursor.fetchone()
                
                if row:
                    return dict(row)
                return {}
        except Exception as e:
            print(f"セッション統計取得エラー: {e}")
            return {}
    
    def _get_cycle_stats(self) -> List[Dict[str, Any]]:
        """サイクル統計を取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM learning_cycles 
                    ORDER BY cycle_number
                """)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"サイクル統計取得エラー: {e}")
            return []
    
    def _get_conversation_stats(self) -> Dict[str, Any]:
        """会話統計を取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 総会話数
                cursor = conn.execute("SELECT COUNT(*) as total FROM processed_conversations")
                total_conversations = cursor.fetchone()[0]
                
                # ソース別統計
                cursor = conn.execute("""
                    SELECT source, COUNT(*) as count, AVG(quality_score) as avg_quality
                    FROM processed_conversations 
                    GROUP BY source
                """)
                source_stats = {row[0]: {"count": row[1], "avg_quality": row[2]} for row in cursor.fetchall()}
                
                # 品質分布
                cursor = conn.execute("""
                    SELECT 
                        CASE 
                            WHEN quality_score >= 0.8 THEN 'high'
                            WHEN quality_score >= 0.5 THEN 'medium'
                            ELSE 'low'
                        END as quality_level,
                        COUNT(*) as count
                    FROM processed_conversations 
                    GROUP BY quality_level
                """)
                quality_distribution = {row[0]: row[1] for row in cursor.fetchall()}
                
                return {
                    "total_conversations": total_conversations,
                    "source_stats": source_stats,
                    "quality_distribution": quality_distribution
                }
        except Exception as e:
            print(f"会話統計取得エラー: {e}")
            return {}
    
    def _analyze_quality(self) -> Dict[str, Any]:
        """品質分析"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 品質スコア統計
                cursor = conn.execute("""
                    SELECT 
                        AVG(quality_score) as avg_quality,
                        MIN(quality_score) as min_quality,
                        MAX(quality_score) as max_quality,
                        COUNT(*) as total_count
                    FROM processed_conversations
                """)
                row = cursor.fetchone()
                
                if row and row[3] > 0:
                    return {
                        "average_quality": row[0],
                        "min_quality": row[1],
                        "max_quality": row[2],
                        "total_count": row[3],
                        "quality_trend": self._get_quality_trend()
                    }
                return {}
        except Exception as e:
            print(f"品質分析エラー: {e}")
            return {}
    
    def _get_quality_trend(self) -> List[float]:
        """品質トレンドを取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT AVG(quality_score) as avg_quality
                    FROM processed_conversations 
                    GROUP BY cycle_number 
                    ORDER BY cycle_number
                """)
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            print(f"品質トレンド取得エラー: {e}")
            return []
    
    def _analyze_efficiency(self, session_stats: Dict[str, Any]) -> Dict[str, Any]:
        """効率分析"""
        if not session_stats:
            return {}
        
        try:
            start_time = datetime.fromisoformat(session_stats["start_time"])
            end_time = datetime.fromisoformat(session_stats["end_time"])
            duration = (end_time - start_time).total_seconds() / 3600  # 時間
            
            total_processed = session_stats["total_processed"]
            learning_cycles = session_stats["learning_cycles"]
            
            return {
                "total_duration_hours": duration,
                "conversations_per_hour": total_processed / duration if duration > 0 else 0,
                "cycles_per_hour": learning_cycles / duration if duration > 0 else 0,
                "avg_conversations_per_cycle": total_processed / learning_cycles if learning_cycles > 0 else 0,
                "efficiency_rating": self._calculate_efficiency_rating(total_processed, duration)
            }
        except Exception as e:
            print(f"効率分析エラー: {e}")
            return {}
    
    def _calculate_efficiency_rating(self, total_processed: int, duration_hours: float) -> str:
        """効率評価を計算"""
        if duration_hours == 0:
            return "unknown"
        
        conversations_per_hour = total_processed / duration_hours
        
        if conversations_per_hour >= 50:
            return "excellent"
        elif conversations_per_hour >= 25:
            return "good"
        elif conversations_per_hour >= 10:
            return "average"
        else:
            return "poor"
    
    def _display_results(self):
        """結果を表示"""
        print("\n📈 基本統計:")
        session_stats = self.results.get("session_stats", {})
        if session_stats:
            print(f"  学習時間: {session_stats.get('total_processed', 0)}件処理")
            print(f"  総サイクル数: {session_stats.get('learning_cycles', 0)}")
            print(f"  最大エポック: {session_stats.get('max_epoch', 0)}")
            print(f"  平均品質スコア: {session_stats.get('avg_quality_score', 0):.3f}")
        
        print("\n⚡ 効率指標:")
        efficiency = self.results.get("efficiency_analysis", {})
        if efficiency:
            print(f"  時間あたり会話数: {efficiency.get('conversations_per_hour', 0):.1f}")
            print(f"  時間あたりサイクル数: {efficiency.get('cycles_per_hour', 0):.1f}")
            print(f"  効率評価: {efficiency.get('efficiency_rating', 'unknown')}")
        
        print("\n🎯 品質指標:")
        quality = self.results.get("quality_analysis", {})
        if quality:
            print(f"  平均品質スコア: {quality.get('average_quality', 0):.3f}")
            print(f"  品質範囲: {quality.get('min_quality', 0):.3f} - {quality.get('max_quality', 0):.3f}")
        
        print("\n📚 データソース:")
        conversation_stats = self.results.get("conversation_stats", {})
        if conversation_stats:
            source_stats = conversation_stats.get("source_stats", {})
            for source, stats in source_stats.items():
                print(f"  {source}: {stats['count']}件 (品質: {stats['avg_quality']:.3f})")
        
        print("\n🏆 総合評価:")
        self._display_overall_assessment()
    
    def _display_overall_assessment(self):
        """総合評価を表示"""
        efficiency = self.results.get("efficiency_analysis", {})
        quality = self.results.get("quality_analysis", {})
        
        efficiency_rating = efficiency.get("efficiency_rating", "unknown")
        avg_quality = quality.get("average_quality", 0)
        
        if efficiency_rating in ["excellent", "good"] and avg_quality >= 0.5:
            print("  ✅ 優秀: 効率と品質の両方が良好")
        elif efficiency_rating in ["excellent", "good"] or avg_quality >= 0.5:
            print("  ⚠️ 良好: 効率または品質のいずれかが良好")
        else:
            print("  ❌ 要改善: 効率と品質の両方を向上させる必要がある")
    
    def _generate_report(self):
        """レポートを生成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"logs/improved_learning_report_{timestamp}.txt"
        
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("📊 改善された学習結果検証レポート\n")
            f.write("=" * 80 + "\n")
            f.write(f"検証日時: {self.results.get('validation_timestamp', 'N/A')}\n\n")
            
            # 基本統計
            session_stats = self.results.get("session_stats", {})
            if session_stats:
                f.write("📈 基本統計:\n")
                f.write(f"  学習時間: {session_stats.get('total_processed', 0)}件処理\n")
                f.write(f"  総サイクル数: {session_stats.get('learning_cycles', 0)}\n")
                f.write(f"  最大エポック: {session_stats.get('max_epoch', 0)}\n")
                f.write(f"  平均品質スコア: {session_stats.get('avg_quality_score', 0):.3f}\n\n")
            
            # 効率指標
            efficiency = self.results.get("efficiency_analysis", {})
            if efficiency:
                f.write("⚡ 効率指標:\n")
                f.write(f"  時間あたり会話数: {efficiency.get('conversations_per_hour', 0):.1f}\n")
                f.write(f"  時間あたりサイクル数: {efficiency.get('cycles_per_hour', 0):.1f}\n")
                f.write(f"  効率評価: {efficiency.get('efficiency_rating', 'unknown')}\n\n")
            
            # 品質指標
            quality = self.results.get("quality_analysis", {})
            if quality:
                f.write("🎯 品質指標:\n")
                f.write(f"  平均品質スコア: {quality.get('average_quality', 0):.3f}\n")
                f.write(f"  品質範囲: {quality.get('min_quality', 0):.3f} - {quality.get('max_quality', 0):.3f}\n\n")
            
            # データソース
            conversation_stats = self.results.get("conversation_stats", {})
            if conversation_stats:
                f.write("📚 データソース:\n")
                source_stats = conversation_stats.get("source_stats", {})
                for source, stats in source_stats.items():
                    f.write(f"  {source}: {stats['count']}件 (品質: {stats['avg_quality']:.3f})\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
        
        print(f"\n📄 レポート生成完了: {report_file}")


def main():
    """メイン関数"""
    validator = ImprovedLearningValidator()
    results = validator.validate_learning_results()
    
    if results:
        print("\n✅ 学習結果検証完了")
    else:
        print("\n❌ 学習結果検証失敗")


if __name__ == "__main__":
    main()
