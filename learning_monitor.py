#!/usr/bin/env python3
"""
Learning Monitor
学習状況リアルタイムモニタリングシステム
"""

import asyncio
import sys
import time
import os
from datetime import datetime, timedelta
from pathlib import Path
import json

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.core.config import Config
from agent.core.database import DatabaseManager


class LearningMonitor:
    """学習モニタリングシステム"""
    
    def __init__(self):
        self.config = None
        self.db_manager = None
        self.monitoring = False
        self.previous_stats = None
        
    async def initialize(self):
        """初期化"""
        try:
            self.config = Config()
            self.db_manager = DatabaseManager(self.config.database_url)
            await self.db_manager.initialize()
            return True
        except Exception as e:
            print(f"❌ 初期化エラー: {e}")
            return False
    
    async def shutdown(self):
        """終了処理"""
        if self.db_manager:
            await self.db_manager.close()
    
    async def get_current_stats(self):
        """現在の統計取得"""
        try:
            stats = await self.db_manager.get_learning_statistics()
            
            # 追加情報取得
            recent_data = await self.db_manager.get_learning_data(limit=5)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'total_learning_data': stats.get('total_learning_data', 0),
                'total_knowledge_items': stats.get('total_knowledge_items', 0),
                'average_quality_score': stats.get('average_quality_score', 0),
                'high_quality_count': stats.get('high_quality_count', 0),
                'recent_data_count': len(recent_data),
                'database_size': stats.get('database_size', 0)
            }
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def calculate_changes(self, current_stats, previous_stats):
        """変化量計算"""
        if not previous_stats or 'error' in previous_stats or 'error' in current_stats:
            return {}
        
        changes = {}
        for key in ['total_learning_data', 'total_knowledge_items', 'high_quality_count']:
            if key in current_stats and key in previous_stats:
                changes[f'{key}_change'] = current_stats[key] - previous_stats[key]
        
        # 品質スコアの変化
        if 'average_quality_score' in current_stats and 'average_quality_score' in previous_stats:
            changes['quality_change'] = current_stats['average_quality_score'] - previous_stats['average_quality_score']
        
        return changes
    
    def print_stats_display(self, stats, changes=None):
        """統計表示"""
        # 画面クリア（Windows/Linux対応）
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("🤖 自己学習システム リアルタイムモニター")
        print("=" * 60)
        print(f"📅 更新時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if 'error' in stats:
            print(f"❌ エラー: {stats['error']}")
            return
        
        print(f"\n📊 学習データ統計:")
        print(f"  📚 総学習データ数: {stats.get('total_learning_data', 0):,}件", end="")
        if changes and 'total_learning_data_change' in changes:
            change = changes['total_learning_data_change']
            if change > 0:
                print(f" (+{change})", end="")
            elif change < 0:
                print(f" ({change})", end="")
        print()
        
        print(f"  🧠 知識アイテム数: {stats.get('total_knowledge_items', 0):,}件", end="")
        if changes and 'total_knowledge_items_change' in changes:
            change = changes['total_knowledge_items_change']
            if change > 0:
                print(f" (+{change})", end="")
            elif change < 0:
                print(f" ({change})", end="")
        print()
        
        print(f"  ⭐ 高品質データ数: {stats.get('high_quality_count', 0):,}件", end="")
        if changes and 'high_quality_count_change' in changes:
            change = changes['high_quality_count_change']
            if change > 0:
                print(f" (+{change})", end="")
            elif change < 0:
                print(f" ({change})", end="")
        print()
        
        quality_score = stats.get('average_quality_score', 0)
        print(f"  📈 平均品質スコア: {quality_score:.4f}", end="")
        if changes and 'quality_change' in changes:
            change = changes['quality_change']
            if change > 0:
                print(f" (+{change:.4f})", end="")
            elif change < 0:
                print(f" ({change:.4f})", end="")
        print()
        
        # 品質レベル表示
        if quality_score >= 0.8:
            quality_level = "🟢 優秀"
        elif quality_score >= 0.6:
            quality_level = "🟡 良好"
        elif quality_score >= 0.4:
            quality_level = "🟠 普通"
        else:
            quality_level = "🔴 要改善"
        
        print(f"  📊 品質レベル: {quality_level}")
        
        # 品質比率
        total_data = stats.get('total_learning_data', 0)
        high_quality = stats.get('high_quality_count', 0)
        if total_data > 0:
            quality_ratio = (high_quality / total_data) * 100
            print(f"  📊 高品質比率: {quality_ratio:.1f}%")
        
        print(f"\n💾 データベース情報:")
        print(f"  📁 データベースサイズ: {stats.get('database_size', 0):,} bytes")
        print(f"  🕐 最新データ数: {stats.get('recent_data_count', 0)}件")
        
        print(f"\n🔄 監視状態:")
        print(f"  ✅ 監視中... (Ctrl+C で停止)")
        print(f"  🔄 次回更新まで: 10秒")
        
        print("=" * 60)
    
    async def run_monitor(self, update_interval=10):
        """モニタリング実行"""
        print("🔍 学習モニタリング開始...")
        self.monitoring = True
        
        try:
            while self.monitoring:
                # 現在の統計取得
                current_stats = await self.get_current_stats()
                
                # 変化量計算
                changes = None
                if self.previous_stats:
                    changes = self.calculate_changes(current_stats, self.previous_stats)
                
                # 表示
                self.print_stats_display(current_stats, changes)
                
                # 統計保存
                self.previous_stats = current_stats
                
                # 待機
                await asyncio.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\n👋 モニタリングを停止します...")
        except Exception as e:
            print(f"\n❌ モニタリングエラー: {e}")
        finally:
            self.monitoring = False


class LearningDashboard:
    """学習ダッシュボード（詳細表示）"""
    
    def __init__(self):
        self.config = None
        self.db_manager = None
    
    async def initialize(self):
        """初期化"""
        try:
            self.config = Config()
            self.db_manager = DatabaseManager(self.config.database_url)
            await self.db_manager.initialize()
            return True
        except Exception as e:
            print(f"❌ 初期化エラー: {e}")
            return False
    
    async def shutdown(self):
        """終了処理"""
        if self.db_manager:
            await self.db_manager.close()
    
    async def show_dashboard(self):
        """ダッシュボード表示"""
        try:
            # 基本統計
            stats = await self.db_manager.get_learning_statistics()
            
            # 最近の学習データ
            recent_data = await self.db_manager.get_learning_data(limit=10)
            
            # カテゴリ別統計（簡略化）
            categories = {}
            for item in recent_data:
                category = item.get('category', 'unknown')
                categories[category] = categories.get(category, 0) + 1
            
            print("🤖 自己学習システム ダッシュボード")
            print("=" * 80)
            print(f"📅 生成時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            print(f"\n📊 基本統計:")
            print(f"  📚 総学習データ数: {stats.get('total_learning_data', 0):,}件")
            print(f"  🧠 知識アイテム数: {stats.get('total_knowledge_items', 0):,}件")
            print(f"  ⭐ 高品質データ数: {stats.get('high_quality_count', 0):,}件")
            print(f"  📈 平均品質スコア: {stats.get('average_quality_score', 0):.4f}")
            
            print(f"\n📋 カテゴリ別データ数:")
            for category, count in sorted(categories.items()):
                print(f"  📁 {category}: {count}件")
            
            print(f"\n📝 最近の学習データ (最新{len(recent_data)}件):")
            for i, item in enumerate(recent_data, 1):
                content = item.get('content', '')[:60]
                category = item.get('category', 'unknown')
                quality = item.get('quality_score', 0)
                created = item.get('created_at', '')
                
                print(f"  {i:2d}. [{category}] {content}... (品質: {quality:.2f})")
                if created:
                    print(f"      作成: {created}")
            
            print("=" * 80)
            
        except Exception as e:
            print(f"❌ ダッシュボード表示エラー: {e}")


async def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="学習モニタリングシステム")
    parser.add_argument("--mode", choices=['monitor', 'dashboard'], default='monitor', 
                       help="実行モード")
    parser.add_argument("--interval", type=int, default=10, 
                       help="更新間隔（秒）")
    
    args = parser.parse_args()
    
    if args.mode == 'monitor':
        monitor = LearningMonitor()
        
        print("🔍 学習リアルタイムモニター")
        print("=" * 50)
        print("停止: Ctrl+C")
        print("=" * 50)
        
        try:
            if await monitor.initialize():
                await monitor.run_monitor(args.interval)
            else:
                print("❌ モニター初期化失敗")
        except KeyboardInterrupt:
            print("\n👋 モニタリング停止")
        finally:
            await monitor.shutdown()
    
    elif args.mode == 'dashboard':
        dashboard = LearningDashboard()
        
        try:
            if await dashboard.initialize():
                await dashboard.show_dashboard()
            else:
                print("❌ ダッシュボード初期化失敗")
        finally:
            await dashboard.shutdown()


if __name__ == "__main__":
    asyncio.run(main())