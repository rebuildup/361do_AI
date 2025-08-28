#!/usr/bin/env python3
"""
Learning Database Analysis
学習データベースの詳細分析
"""

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter

def analyze_learning_database():
    """学習データベースの詳細分析"""
    
    print("=" * 80)
    print("🗄️ 学習データベース詳細分析")
    print("=" * 80)
    
    db_path = "data/agent.db"
    if not Path(db_path).exists():
        print(f"❌ データベースファイルが見つかりません: {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 基本統計
        print("📊 基本統計:")
        
        # 学習データ総数
        cursor.execute("SELECT COUNT(*) FROM learning_data")
        total_learning_data = cursor.fetchone()[0]
        print(f"  学習データ総数: {total_learning_data}件")
        
        # 知識アイテム数
        cursor.execute("SELECT COUNT(*) FROM knowledge_items")
        total_knowledge = cursor.fetchone()[0]
        print(f"  知識アイテム数: {total_knowledge}件")
        
        # プロンプトテンプレート数
        cursor.execute("SELECT COUNT(*) FROM prompt_templates")
        total_prompts = cursor.fetchone()[0]
        print(f"  プロンプトテンプレート数: {total_prompts}件")
        
        # 学習データの品質分析
        print(f"\n📈 学習データ品質分析:")
        
        cursor.execute("""
            SELECT 
                AVG(quality_score) as avg_quality,
                MIN(quality_score) as min_quality,
                MAX(quality_score) as max_quality,
                COUNT(CASE WHEN quality_score >= 0.8 THEN 1 END) as high_quality,
                COUNT(CASE WHEN quality_score >= 0.6 THEN 1 END) as medium_quality,
                COUNT(CASE WHEN quality_score < 0.6 THEN 1 END) as low_quality
            FROM learning_data
        """)
        
        quality_stats = cursor.fetchone()
        if quality_stats[0]:  # avg_qualityがNoneでない場合
            avg_quality, min_quality, max_quality, high_quality, medium_quality, low_quality = quality_stats
            print(f"  平均品質スコア: {avg_quality:.3f}")
            print(f"  品質スコア範囲: {min_quality:.3f} - {max_quality:.3f}")
            print(f"  高品質データ (≥0.8): {high_quality}件 ({high_quality/total_learning_data*100:.1f}%)")
            print(f"  中品質データ (≥0.6): {medium_quality}件 ({medium_quality/total_learning_data*100:.1f}%)")
            print(f"  低品質データ (<0.6): {low_quality}件 ({low_quality/total_learning_data*100:.1f}%)")
        
        # カテゴリ別分析
        print(f"\n📂 カテゴリ別分析:")
        
        cursor.execute("""
            SELECT category, COUNT(*) as count, AVG(quality_score) as avg_quality
            FROM learning_data 
            GROUP BY category 
            ORDER BY count DESC
        """)
        
        categories = cursor.fetchall()
        for category, count, avg_quality in categories:
            quality_str = f"{avg_quality:.3f}" if avg_quality else "N/A"
            print(f"  {category}: {count}件 (平均品質: {quality_str})")
        
        # 時系列分析
        print(f"\n📅 時系列分析:")
        
        # 過去24時間のデータ
        cursor.execute("""
            SELECT COUNT(*) FROM learning_data 
            WHERE created_at > datetime('now', '-1 day')
        """)
        recent_24h = cursor.fetchone()[0]
        
        # 過去1週間のデータ
        cursor.execute("""
            SELECT COUNT(*) FROM learning_data 
            WHERE created_at > datetime('now', '-7 days')
        """)
        recent_week = cursor.fetchone()[0]
        
        # 8時間学習セッション中のデータ（推定）
        cursor.execute("""
            SELECT COUNT(*) FROM learning_data 
            WHERE created_at > datetime('now', '-10 hours')
        """)
        session_data = cursor.fetchone()[0]
        
        print(f"  過去24時間: {recent_24h}件")
        print(f"  過去1週間: {recent_week}件")
        print(f"  8時間セッション中（推定）: {session_data}件")
        
        # タグ分析
        print(f"\n🏷️ タグ分析:")
        
        cursor.execute("""
            SELECT tags, COUNT(*) as count
            FROM learning_data 
            WHERE tags IS NOT NULL AND tags != ''
            GROUP BY tags 
            ORDER BY count DESC 
            LIMIT 10
        """)
        
        tags = cursor.fetchall()
        if tags:
            for tag, count in tags:
                print(f"  {tag}: {count}件")
        else:
            print("  タグ付きデータなし")
        
        # 最新の学習データサンプル
        print(f"\n📝 最新学習データサンプル:")
        
        cursor.execute("""
            SELECT content, category, quality_score, created_at
            FROM learning_data 
            ORDER BY created_at DESC 
            LIMIT 5
        """)
        
        recent_samples = cursor.fetchall()
        for i, (content, category, quality, created_at) in enumerate(recent_samples, 1):
            content_preview = content[:100] + "..." if len(content) > 100 else content
            quality_str = f"{quality:.3f}" if quality else "N/A"
            print(f"  {i}. [{category}] 品質:{quality_str} - {content_preview}")
            print(f"     作成日時: {created_at}")
        
        # マルチエージェント学習の効果分析
        print(f"\n🤖 マルチエージェント学習効果:")
        
        # multi_agent_conversation カテゴリのデータ
        cursor.execute("""
            SELECT COUNT(*) FROM learning_data 
            WHERE category = 'multi_agent_conversation'
        """)
        multi_agent_data = cursor.fetchone()[0]
        
        if multi_agent_data > 0:
            print(f"  マルチエージェント会話データ: {multi_agent_data}件")
            
            cursor.execute("""
                SELECT AVG(quality_score) FROM learning_data 
                WHERE category = 'multi_agent_conversation'
            """)
            multi_agent_quality = cursor.fetchone()[0]
            
            if multi_agent_quality:
                print(f"  マルチエージェント会話データ平均品質: {multi_agent_quality:.3f}")
        else:
            print("  マルチエージェント会話データなし")
        
        # 学習データの成長率
        print(f"\n📊 学習データ成長分析:")
        
        if recent_24h > 0:
            daily_growth_rate = (recent_24h / total_learning_data) * 100
            print(f"  日次成長率: {daily_growth_rate:.1f}%")
            
            # 8時間セッションでの成長
            if session_data > 0:
                session_growth_rate = (session_data / total_learning_data) * 100
                print(f"  8時間セッション成長率: {session_growth_rate:.1f}%")
                print(f"  セッション中の時間あたり新規データ: {session_data/8:.1f}件/時間")
        
        # データベースサイズ情報
        print(f"\n💾 データベース情報:")
        
        db_size = Path(db_path).stat().st_size
        print(f"  データベースファイルサイズ: {db_size/1024/1024:.2f}MB")
        
        # テーブル別レコード数
        tables = ['learning_data', 'knowledge_items', 'prompt_templates', 'conversations']
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"  {table}テーブル: {count}件")
            except sqlite3.OperationalError:
                print(f"  {table}テーブル: 存在しません")
        
        conn.close()
        
        # 学習効果の評価
        print(f"\n🎯 学習効果評価:")
        
        if total_learning_data >= 300:
            print("  ✅ 豊富な学習データが蓄積されています")
        elif total_learning_data >= 100:
            print("  ✅ 十分な学習データが蓄積されています")
        else:
            print("  ⚠️ 学習データの蓄積が不十分です")
        
        if quality_stats[0] and quality_stats[0] >= 0.7:
            print("  ✅ 学習データの品質が高いです")
        elif quality_stats[0] and quality_stats[0] >= 0.5:
            print("  ✅ 学習データの品質は普通です")
        else:
            print("  ⚠️ 学習データの品質向上が必要です")
        
        if session_data >= 50:
            print("  ✅ 8時間セッションで大量の学習データが生成されました")
        elif session_data >= 20:
            print("  ✅ 8時間セッションで適度な学習データが生成されました")
        else:
            print("  ⚠️ 8時間セッションでの学習データ生成が少ないです")
        
    except Exception as e:
        print(f"❌ データベース分析エラー: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 80)

def main():
    """メイン関数"""
    analyze_learning_database()

if __name__ == "__main__":
    main()