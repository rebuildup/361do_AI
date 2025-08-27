#!/usr/bin/env python3
"""
Self-Learning Tests Runner
自己学習機能テストの実行管理スクリプト
"""

import asyncio
import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Tests should not contact a local ollama daemon; force skip mode
os.environ.setdefault('AGENT_SKIP_OLLAMA', '1')


class SelfLearningTestRunner:
    """自己学習テスト実行管理クラス"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.end_time = None
    
    async def run_quick_test(self):
        """クイックテスト実行"""
        print("🚀 クイックテスト実行中...")
        
        try:
            # quick_self_learning_test.pyを動的インポートして実行
            from quick_self_learning_test import quick_test
            await quick_test()
            return True
        except Exception as e:
            print(f"❌ クイックテストエラー: {e}")
            return False
    
    async def run_diagnostic_test(self):
        """診断テスト実行"""
        print("🔍 診断テスト実行中...")
        
        try:
            from self_learning_diagnostic_test import SelfLearningDiagnostic
            
            diagnostic = SelfLearningDiagnostic()
            
            if await diagnostic.initialize():
                results = await diagnostic.run_comprehensive_diagnostic()
                diagnostic.print_diagnostic_summary(results)
                
                # 結果保存
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"diagnostic_results_{timestamp}.json"
                diagnostic.save_diagnostic_results(results, filename)
                
                await diagnostic.shutdown()
                return results
            else:
                print("❌ 診断システム初期化失敗")
                return None
                
        except Exception as e:
            print(f"❌ 診断テストエラー: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def run_comprehensive_test(self):
        """包括テスト実行"""
        print("📋 包括テスト実行中...")
        
        try:
            from self_learning_test import SelfLearningTester
            
            tester = SelfLearningTester()
            
            if await tester.initialize():
                results = await tester.run_comprehensive_test()
                tester.print_comprehensive_summary(results)
                
                # 結果保存
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"comprehensive_results_{timestamp}.json"
                tester.save_test_results(results, filename)
                
                await tester.shutdown()
                return results
            else:
                print("❌ 包括テストシステム初期化失敗")
                return None
                
        except Exception as e:
            print(f"❌ 包括テストエラー: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def run_simple_test(self):
        """シンプルテスト実行"""
        print("🧪 シンプルテスト実行中...")
        
        try:
            from test_self_learning_simple import _test_learning_system_async
            await _test_learning_system_async()
            return True
        except Exception as e:
            print(f"❌ シンプルテストエラー: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def print_menu(self):
        """メニュー表示"""
        print("\n" + "="*60)
        print("🤖 自己学習機能テスト実行メニュー")
        print("="*60)
        print("1. クイック診断 (quick) - 基本的な問題を素早く特定")
        print("2. 詳細診断 (diagnostic) - 包括的な問題診断と推奨事項")
        print("3. 包括テスト (comprehensive) - 全機能の動作テスト")
        print("4. シンプルテスト (simple) - 基本的な学習システムテスト")
        print("5. 全テスト実行 (all) - 全てのテストを順次実行")
        print("6. 終了 (exit)")
        print("="*60)
    
    async def run_all_tests(self):
        """全テスト実行"""
        print("🎯 全テスト実行開始...")
        self.start_time = datetime.now()
        
        all_results = {}
        
        # 1. クイックテスト
        print("\n" + "="*40)
        print("1/4: クイック診断")
        print("="*40)
        quick_result = await self.run_quick_test()
        all_results['quick_test'] = {'success': quick_result}
        
        # 2. 診断テスト
        print("\n" + "="*40)
        print("2/4: 詳細診断")
        print("="*40)
        diagnostic_result = await self.run_diagnostic_test()
        all_results['diagnostic_test'] = diagnostic_result
        
        # 3. 包括テスト
        print("\n" + "="*40)
        print("3/4: 包括テスト")
        print("="*40)
        comprehensive_result = await self.run_comprehensive_test()
        all_results['comprehensive_test'] = comprehensive_result
        
        # 4. シンプルテスト
        print("\n" + "="*40)
        print("4/4: シンプルテスト")
        print("="*40)
        simple_result = await self.run_simple_test()
        all_results['simple_test'] = {'success': simple_result}
        
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        # 全体結果サマリー
        print("\n" + "="*60)
        print("🎉 全テスト実行完了")
        print("="*60)
        print(f"実行時間: {duration:.2f}秒")
        print(f"開始時刻: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"終了時刻: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 各テストの結果
        print("\n📊 テスト結果サマリー:")
        for test_name, result in all_results.items():
            if isinstance(result, dict) and 'success' in result:
                status = "✅ 成功" if result['success'] else "❌ 失敗"
                print(f"  {test_name}: {status}")
            elif result:
                print(f"  {test_name}: ✅ 完了")
            else:
                print(f"  {test_name}: ❌ 失敗")
        
        # 結果保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"all_tests_results_{timestamp}.json"
        
        final_results = {
            'execution_info': {
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat(),
                'duration_seconds': duration
            },
            'test_results': all_results
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, ensure_ascii=False, indent=2)
            print(f"\n💾 全テスト結果を保存: {filename}")
        except Exception as e:
            print(f"⚠️ 結果保存エラー: {e}")
        
        print("="*60)
        
        return all_results
    
    async def interactive_menu(self):
        """インタラクティブメニュー"""
        while True:
            self.print_menu()
            
            try:
                choice = input("\n選択してください (1-6): ").strip().lower()
                
                if choice in ['1', 'quick']:
                    await self.run_quick_test()
                elif choice in ['2', 'diagnostic']:
                    await self.run_diagnostic_test()
                elif choice in ['3', 'comprehensive']:
                    await self.run_comprehensive_test()
                elif choice in ['4', 'simple']:
                    await self.run_simple_test()
                elif choice in ['5', 'all']:
                    await self.run_all_tests()
                elif choice in ['6', 'exit', 'quit']:
                    print("👋 テストランナーを終了します")
                    break
                else:
                    print("❌ 無効な選択です。1-6の数字を入力してください。")
                
                input("\nEnterキーを押して続行...")
                
            except KeyboardInterrupt:
                print("\n👋 テストランナーを終了します")
                break
            except Exception as e:
                print(f"❌ エラー: {e}")
                input("\nEnterキーを押して続行...")


async def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="自己学習機能テスト実行管理")
    parser.add_argument("--test", choices=[
        'quick', 'diagnostic', 'comprehensive', 'simple', 'all', 'interactive'
    ], default='interactive', help="実行するテスト種類")
    
    args = parser.parse_args()
    
    runner = SelfLearningTestRunner()
    
    print("🤖 自己学習機能テスト実行管理システム")
    print("=" * 50)
    
    try:
        if args.test == 'quick':
            await runner.run_quick_test()
        elif args.test == 'diagnostic':
            await runner.run_diagnostic_test()
        elif args.test == 'comprehensive':
            await runner.run_comprehensive_test()
        elif args.test == 'simple':
            await runner.run_simple_test()
        elif args.test == 'all':
            await runner.run_all_tests()
        elif args.test == 'interactive':
            await runner.interactive_menu()
        
    except KeyboardInterrupt:
        print("\n👋 ユーザーによって中断されました")
    except Exception as e:
        print(f"❌ 実行エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())