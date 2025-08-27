#!/usr/bin/env python3
"""
Continuous Learning System
継続的学習システム - シンプルな自走学習
"""

import asyncio
import sys
import time
import signal
import os
from datetime import datetime, timedelta
from pathlib import Path
import json

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.core.config import Config
from agent.core.database import DatabaseManager
from agent.core.agent_manager import AgentManager


class ContinuousLearning:
    """継続的学習システム"""
    
    def __init__(self):
        self.running = False
        self.cycle_count = 0
        self.start_time = None
        self.agent_manager = None
        self.db_manager = None
        self.config = None
        
        # 停止フラグ
        self.stop_requested = False
        
    def setup_signal_handler(self):
        """Ctrl+C での停止処理"""
        def signal_handler(signum, frame):
            print("\n🛑 停止シグナルを受信しました...")
            self.stop_requested = True
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def initialize(self):
        """システム初期化"""
        print("🚀 継続的学習システム初期化中...")
        
        try:
            self.config = Config()
            self.db_manager = DatabaseManager(self.config.database_url)
            await self.db_manager.initialize()
            
            self.agent_manager = AgentManager(self.config, self.db_manager)
            await self.agent_manager.initialize()
            
            print("✅ 初期化完了")
            return True
            
        except Exception as e:
            print(f"❌ 初期化エラー: {e}")
            return False
    
    async def shutdown(self):
        """システム終了"""
        print("🔄 システム終了中...")
        
        try:
            if self.agent_manager:
                await self.agent_manager.shutdown()
            if self.db_manager:
                await self.db_manager.close()
            print("✅ 終了完了")
        except Exception as e:
            print(f"⚠️ 終了エラー: {e}")
    
    async def learning_cycle(self):
        """学習サイクル実行"""
        cycle_start = time.time()
        print(f"\n🔄 学習サイクル {self.cycle_count + 1} 開始...")
        
        activities = []
        
        try:
            # 1. 学習データ統計確認
            print("  📊 学習データ統計確認中...")
            stats = await self.db_manager.get_learning_statistics()
            activities.append(f"学習データ: {stats.get('total_learning_data', 0)}件")
            
            # 2. 学習システム状態確認
            if hasattr(self.agent_manager, 'learning_tool') and self.agent_manager.learning_tool:
                print("  🧠 学習システム状態確認中...")
                status = await self.agent_manager.learning_tool.get_learning_status()
                activities.append(f"学習システム: {status.get('status', 'unknown')}")
                
                # 3. 新しい学習データ追加（テスト用）
                print("  ➕ 学習データ追加中...")
                test_content = f"継続学習サイクル{self.cycle_count + 1} - {datetime.now().isoformat()}"
                add_result = await self.agent_manager.learning_tool.add_custom_learning_data(
                    content=test_content,
                    category="continuous_learning",
                    tags=["auto_generated", f"cycle_{self.cycle_count + 1}"]
                )
                activities.append(f"データ追加: {add_result.get('status', 'failed')}")
                
                # 4. 学習サイクル手動実行
                print("  🔄 学習サイクル実行中...")
                cycle_result = await self.agent_manager.learning_tool.manually_trigger_learning_cycle()
                activities.append(f"学習サイクル: {cycle_result.get('status', 'failed')}")
            
            # 5. パフォーマンス測定
            print("  📈 パフォーマンス測定中...")
            end_stats = await self.db_manager.get_learning_statistics()
            quality_score = end_stats.get('average_quality_score', 0)
            activities.append(f"品質スコア: {quality_score:.3f}")
            
        except Exception as e:
            print(f"  ❌ サイクルエラー: {e}")
            activities.append(f"エラー: {str(e)}")
        
        cycle_duration = time.time() - cycle_start
        
        # 結果表示
        print(f"  ✅ サイクル完了 ({cycle_duration:.2f}秒)")
        for activity in activities:
            print(f"    - {activity}")
        
        return {
            'cycle': self.cycle_count + 1,
            'duration': cycle_duration,
            'activities': activities,
            'timestamp': datetime.now().isoformat()
        }
    
    async def run_continuous_learning(self, max_cycles=None, max_hours=None, cycle_interval=30):
        """継続的学習実行"""
        print("🎯 継続的学習開始...")
        print(f"設定: 最大サイクル={max_cycles}, 最大時間={max_hours}時間, 間隔={cycle_interval}秒")
        print("停止: Ctrl+C")
        print("=" * 60)
        
        self.running = True
        self.start_time = datetime.now()
        results = []
        
        try:
            while self.running and not self.stop_requested:
                # 停止条件チェック
                if max_cycles and self.cycle_count >= max_cycles:
                    print(f"🏁 最大サイクル数到達: {max_cycles}")
                    break
                
                if max_hours:
                    runtime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
                    if runtime_hours >= max_hours:
                        print(f"🏁 最大実行時間到達: {runtime_hours:.2f}時間")
                        break
                
                # 学習サイクル実行
                cycle_result = await self.learning_cycle()
                results.append(cycle_result)
                self.cycle_count += 1
                
                # 進捗表示
                if self.start_time:
                    total_runtime = (datetime.now() - self.start_time).total_seconds()
                    avg_cycle_time = total_runtime / self.cycle_count if self.cycle_count > 0 else 0
                    print(f"📊 進捗: {self.cycle_count}サイクル完了, 総実行時間: {total_runtime/60:.1f}分, 平均サイクル時間: {avg_cycle_time:.1f}秒")
                
                # 次のサイクルまで待機
                if not self.stop_requested:
                    print(f"⏳ {cycle_interval}秒待機中... (Ctrl+Cで停止)")
                    for i in range(cycle_interval):
                        if self.stop_requested:
                            break
                        await asyncio.sleep(1)
                        
        except Exception as e:
            print(f"❌ 継続学習エラー: {e}")
        finally:
            self.running = False
            
            # 結果保存
            if results:
                await self.save_results(results)
            
            # 最終レポート
            self.print_final_report(results)
    
    async def save_results(self, results):
        """結果保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"continuous_learning_results_{timestamp}.json"
        
        try:
            data = {
                'session_info': {
                    'start_time': self.start_time.isoformat() if self.start_time else None,
                    'end_time': datetime.now().isoformat(),
                    'total_cycles': len(results),
                    'total_runtime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
                },
                'results': results
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 結果保存: {filename}")
            
        except Exception as e:
            print(f"⚠️ 結果保存エラー: {e}")
    
    def print_final_report(self, results):
        """最終レポート"""
        if not results:
            print("📊 実行結果がありません")
            return
        
        total_runtime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        avg_cycle_time = sum(r['duration'] for r in results) / len(results)
        
        print(f"\n{'='*60}")
        print(f"🎉 継続的学習セッション完了")
        print(f"{'='*60}")
        print(f"📅 開始: {self.start_time.strftime('%Y-%m-%d %H:%M:%S') if self.start_time else 'Unknown'}")
        print(f"📅 終了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  総実行時間: {total_runtime/60:.1f}分")
        print(f"🔄 完了サイクル数: {len(results)}")
        print(f"⏱️  平均サイクル時間: {avg_cycle_time:.2f}秒")
        print(f"{'='*60}")


async def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="継続的学習システム")
    parser.add_argument("--max-cycles", type=int, help="最大サイクル数")
    parser.add_argument("--max-hours", type=float, help="最大実行時間（時間）")
    parser.add_argument("--interval", type=int, default=30, help="サイクル間隔（秒）")
    
    args = parser.parse_args()
    
    learning_system = ContinuousLearning()
    learning_system.setup_signal_handler()
    
    print("🤖 継続的学習システム")
    print("=" * 50)
    
    try:
        if await learning_system.initialize():
            await learning_system.run_continuous_learning(
                max_cycles=args.max_cycles,
                max_hours=args.max_hours,
                cycle_interval=args.interval
            )
        else:
            print("❌ システム初期化失敗")
            
    except KeyboardInterrupt:
        print("\n👋 ユーザーによって中断されました")
    except Exception as e:
        print(f"❌ 実行エラー: {e}")
    finally:
        await learning_system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())