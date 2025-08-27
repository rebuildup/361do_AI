#!/usr/bin/env python3
"""
Self-Learning Function Test
自己学習機能の詳細テストと診断
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.core.config import Config
from agent.core.database import DatabaseManager
from agent.core.ollama_client import OllamaClient
from agent.core.agent_manager import AgentManager


class SelfLearningTester:
    """自己学習機能テスター"""

    def __init__(self):
        self.config = None
        self.db_manager = None
        self.agent_manager = None
        self.test_results = []

    async def initialize(self):
        """システム初期化"""
        print("[SELF-LEARN] 自己学習機能テストシステムを初期化中...")
        
        try:
            self.config = Config()
            
            # データベース初期化
            self.db_manager = DatabaseManager(self.config.database_url)
            await self.db_manager.initialize()
            
            # エージェントマネージャー初期化
            self.agent_manager = AgentManager(self.config, self.db_manager)
            await self.agent_manager.initialize()
            
            print("[SELF-LEARN] 初期化完了")
            return True
            
        except Exception as e:
            print(f"[ERROR] 初期化エラー: {e}")
            return False

    async def shutdown(self):
        """システム終了処理"""
        print("[SELF-LEARN] システムを終了中...")
        
        if self.agent_manager:
            await self.agent_manager.shutdown()
        
        if self.db_manager:
            await self.db_manager.close()
        
        print("[SELF-LEARN] 終了完了")

    async def test_learning_data_access(self) -> Dict[str, Any]:
        """学習データアクセステスト"""
        print("\n[TEST] 学習データアクセステスト開始...")
        
        test_cases = [
            "学習データの統計を表示",
            "一番古い学習データを教えて",
            "学習システムの状態を確認",
            "学習データを一覧表示",
            "学習データの品質スコアを確認"
        ]
        
        results = []
        for i, question in enumerate(test_cases, 1):
            print(f"  テスト {i}/{len(test_cases)}: {question}")
            
            start_time = time.time()
            try:
                response = await self.agent_manager.process_message(question)
                execution_time = time.time() - start_time
                
                result = {
                    'test_case': question,
                    'response': response.get('response', ''),
                    'intent': response.get('intent', {}),
                    'tools_used': response.get('tools_used', []),
                    'execution_time': execution_time,
                    'success': True,
                    'error': None
                }
                
                print(f"    応答: {response.get('response', '')[:100]}...")
                print(f"    意図: {response.get('intent', {}).get('primary_intent', 'unknown')}")
                print(f"    ツール: {response.get('tools_used', [])}")
                
            except Exception as e:
                execution_time = time.time() - start_time
                result = {
                    'test_case': question,
                    'response': f"エラー: {str(e)}",
                    'intent': {},
                    'tools_used': [],
                    'execution_time': execution_time,
                    'success': False,
                    'error': str(e)
                }
                print(f"    エラー: {e}")
            
            results.append(result)
            await asyncio.sleep(1)
        
        return {
            'test_name': 'learning_data_access',
            'results': results,
            'summary': self._generate_test_summary(results)
        }

    async def test_self_edit_functions(self) -> Dict[str, Any]:
        """自己編集機能テスト"""
        print("\n[TEST] 自己編集機能テスト開始...")
        
        test_cases = [
            "read file src/data/prompts/system_prompt.txt",
            "write file test_output.txt\nテスト用のファイル内容です",
            "append file test_output.txt\n追記内容です",
            "update prompt test_prompt: これはテスト用のプロンプトです",
            "add learning data: テスト用の学習データです"
        ]
        
        results = []
        for i, question in enumerate(test_cases, 1):
            print(f"  テスト {i}/{len(test_cases)}: {question}")
            
            start_time = time.time()
            try:
                response = await self.agent_manager.process_message(question)
                execution_time = time.time() - start_time
                
                result = {
                    'test_case': question,
                    'response': response.get('response', ''),
                    'intent': response.get('intent', {}),
                    'tools_used': response.get('tools_used', []),
                    'execution_time': execution_time,
                    'success': True,
                    'error': None
                }
                
                print(f"    応答: {response.get('response', '')[:100]}...")
                print(f"    意図: {response.get('intent', {}).get('primary_intent', 'unknown')}")
                print(f"    ツール: {response.get('tools_used', [])}")
                
            except Exception as e:
                execution_time = time.time() - start_time
                result = {
                    'test_case': question,
                    'response': f"エラー: {str(e)}",
                    'intent': {},
                    'tools_used': [],
                    'execution_time': execution_time,
                    'success': False,
                    'error': str(e)
                }
                print(f"    エラー: {e}")
            
            results.append(result)
            await asyncio.sleep(1)
        
        return {
            'test_name': 'self_edit_functions',
            'results': results,
            'summary': self._generate_test_summary(results)
        }

    async def test_learning_system_integration(self) -> Dict[str, Any]:
        """学習システム統合テスト"""
        print("\n[TEST] 学習システム統合テスト開始...")
        
        # 学習ツールの直接テスト
        results = []
        
        if self.agent_manager.learning_tool:
            print("  学習ツールが利用可能です")
            
            # 学習データ取得テスト
            try:
                print("  学習データ取得テスト...")
                learning_data = await self.agent_manager.learning_tool.get_learning_data(limit=5)
                results.append({
                    'test': 'get_learning_data',
                    'success': learning_data.get('status') == 'success',
                    'data_count': len(learning_data.get('data', [])),
                    'response': learning_data
                })
                print(f"    結果: {len(learning_data.get('data', []))}件のデータを取得")
            except Exception as e:
                results.append({
                    'test': 'get_learning_data',
                    'success': False,
                    'error': str(e)
                })
                print(f"    エラー: {e}")
            
            # 学習状態取得テスト
            try:
                print("  学習状態取得テスト...")
                learning_status = await self.agent_manager.learning_tool.get_learning_status()
                results.append({
                    'test': 'get_learning_status',
                    'success': learning_status.get('status') == 'success',
                    'response': learning_status
                })
                print(f"    結果: {learning_status.get('status', 'unknown')}")
            except Exception as e:
                results.append({
                    'test': 'get_learning_status',
                    'success': False,
                    'error': str(e)
                })
                print(f"    エラー: {e}")
            
            # カスタム学習データ追加テスト
            try:
                print("  カスタム学習データ追加テスト...")
                add_result = await self.agent_manager.learning_tool.add_custom_learning_data(
                    content="テスト用学習データ - 自動テストで追加",
                    category="test_data",
                    tags=["automated_test", "self_learning_test"]
                )
                results.append({
                    'test': 'add_custom_learning_data',
                    'success': add_result.get('status') == 'success',
                    'response': add_result
                })
                print(f"    結果: {add_result.get('status', 'unknown')}")
            except Exception as e:
                results.append({
                    'test': 'add_custom_learning_data',
                    'success': False,
                    'error': str(e)
                })
                print(f"    エラー: {e}")
        
        else:
            print("  学習ツールが利用できません")
            results.append({
                'test': 'learning_tool_availability',
                'success': False,
                'error': 'Learning tool not available'
            })
        
        return {
            'test_name': 'learning_system_integration',
            'results': results,
            'summary': {
                'total_tests': len(results),
                'successful': sum(1 for r in results if r.get('success', False)),
                'failed': sum(1 for r in results if not r.get('success', False))
            }
        }

    async def test_database_operations(self) -> Dict[str, Any]:
        """データベース操作テスト"""
        print("\n[TEST] データベース操作テスト開始...")
        
        results = []
        
        try:
            # 学習統計取得テスト
            print("  学習統計取得テスト...")
            stats = await self.db_manager.get_learning_statistics()
            results.append({
                'test': 'get_learning_statistics',
                'success': True,
                'stats': stats
            })
            print(f"    結果: 学習データ {stats.get('total_learning_data', 0)}件")
            
        except Exception as e:
            results.append({
                'test': 'get_learning_statistics',
                'success': False,
                'error': str(e)
            })
            print(f"    エラー: {e}")
        
        try:
            # プロンプトテンプレート取得テスト
            print("  プロンプトテンプレート取得テスト...")
            system_prompt = await self.db_manager.get_prompt_template("system_prompt")
            results.append({
                'test': 'get_prompt_template',
                'success': system_prompt is not None,
                'has_system_prompt': system_prompt is not None
            })
            print(f"    結果: システムプロンプト {'存在' if system_prompt else '不存在'}")
            
        except Exception as e:
            results.append({
                'test': 'get_prompt_template',
                'success': False,
                'error': str(e)
            })
            print(f"    エラー: {e}")
        
        try:
            # アクティブ知識取得テスト
            print("  アクティブ知識取得テスト...")
            knowledge = await self.db_manager.get_active_knowledge()
            results.append({
                'test': 'get_active_knowledge',
                'success': True,
                'knowledge_count': len(knowledge)
            })
            print(f"    結果: アクティブ知識 {len(knowledge)}件")
            
        except Exception as e:
            results.append({
                'test': 'get_active_knowledge',
                'success': False,
                'error': str(e)
            })
            print(f"    エラー: {e}")
        
        return {
            'test_name': 'database_operations',
            'results': results,
            'summary': {
                'total_tests': len(results),
                'successful': sum(1 for r in results if r.get('success', False)),
                'failed': sum(1 for r in results if not r.get('success', False))
            }
        }

    async def test_intent_analysis(self) -> Dict[str, Any]:
        """意図分析テスト"""
        print("\n[TEST] 意図分析テスト開始...")
        
        test_cases = [
            {
                'input': '学習データの統計を表示',
                'expected_intent': 'learning_data_access'
            },
            {
                'input': '一番古い学習データを教えて',
                'expected_intent': 'learning_data_access'
            },
            {
                'input': 'read file test.txt',
                'expected_intent': 'file_operation'
            },
            {
                'input': 'write file output.txt\ncontent',
                'expected_intent': 'file_operation'
            },
            {
                'input': 'update prompt test: content',
                'expected_intent': 'file_operation'
            },
            {
                'input': 'add learning data: test content',
                'expected_intent': 'file_operation'
            }
        ]
        
        results = []
        for i, test_case in enumerate(test_cases, 1):
            print(f"  テスト {i}/{len(test_cases)}: {test_case['input']}")
            
            try:
                # 意図分析のみテスト（実際の処理は行わない）
                context = []
                intent = await self.agent_manager._analyze_intent(test_case['input'], context)
                
                detected_intent = intent.get('primary_intent', 'unknown')
                expected_intent = test_case['expected_intent']
                
                result = {
                    'input': test_case['input'],
                    'expected_intent': expected_intent,
                    'detected_intent': detected_intent,
                    'confidence': intent.get('confidence', 0),
                    'success': detected_intent == expected_intent,
                    'full_intent': intent
                }
                
                print(f"    期待: {expected_intent}, 検出: {detected_intent}, 一致: {result['success']}")
                
            except Exception as e:
                result = {
                    'input': test_case['input'],
                    'expected_intent': test_case['expected_intent'],
                    'detected_intent': 'error',
                    'success': False,
                    'error': str(e)
                }
                print(f"    エラー: {e}")
            
            results.append(result)
        
        return {
            'test_name': 'intent_analysis',
            'results': results,
            'summary': self._generate_test_summary(results)
        }

    def _generate_test_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """テスト結果サマリー生成"""
        if not results:
            return {}
        
        successful = sum(1 for r in results if r.get('success', False))
        failed = len(results) - successful
        
        return {
            'total_tests': len(results),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(results) if results else 0
        }

    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """包括的テスト実行"""
        print("[SELF-LEARN] 自己学習機能包括テスト開始...")
        
        test_start_time = datetime.now()
        all_results = {}
        
        # 各テストを実行
        test_functions = [
            self.test_intent_analysis,
            self.test_database_operations,
            self.test_learning_system_integration,
            self.test_learning_data_access,
            self.test_self_edit_functions
        ]
        
        for test_func in test_functions:
            try:
                result = await test_func()
                all_results[result['test_name']] = result
            except Exception as e:
                print(f"[ERROR] テスト {test_func.__name__} でエラー: {e}")
                all_results[test_func.__name__] = {
                    'test_name': test_func.__name__,
                    'error': str(e),
                    'success': False
                }
        
        test_end_time = datetime.now()
        test_duration = (test_end_time - test_start_time).total_seconds()
        
        # 総合サマリー生成
        comprehensive_summary = {
            'start_time': test_start_time.isoformat(),
            'end_time': test_end_time.isoformat(),
            'duration_seconds': test_duration,
            'tests_executed': len(all_results),
            'results': all_results
        }
        
        return comprehensive_summary

    def print_comprehensive_summary(self, results: Dict[str, Any]):
        """包括テスト結果サマリー表示"""
        print(f"\n{'='*80}")
        print(f"[SELF-LEARN] 自己学習機能テスト結果サマリー")
        print(f"{'='*80}")
        print(f"実行開始: {results['start_time']}")
        print(f"実行終了: {results['end_time']}")
        print(f"総実行時間: {results['duration_seconds']:.2f}秒")
        print(f"実行テスト数: {results['tests_executed']}")
        
        print(f"\n[RESULTS] テスト別結果:")
        for test_name, test_data in results['results'].items():
            if 'summary' in test_data:
                summary = test_data['summary']
                success_rate = summary.get('success_rate', 0)
                print(f"\n  📋 {test_name}:")
                print(f"     テスト数: {summary.get('total_tests', 0)}")
                print(f"     成功率: {success_rate:.1%}")
                print(f"     成功: {summary.get('successful', 0)}")
                print(f"     失敗: {summary.get('failed', 0)}")
                
                # 失敗したテストの詳細表示
                if 'results' in test_data and summary.get('failed', 0) > 0:
                    failed_tests = [r for r in test_data['results'] if not r.get('success', False)]
                    print(f"     失敗詳細:")
                    for failed in failed_tests[:3]:  # 最大3件まで表示
                        error = failed.get('error', 'Unknown error')
                        test_case = failed.get('test_case', failed.get('input', failed.get('test', 'Unknown')))
                        print(f"       - {test_case}: {error}")
            else:
                print(f"\n  📋 {test_name}: エラー - {test_data.get('error', 'Unknown error')}")
        
        print(f"{'='*80}")

    def save_test_results(self, results: Dict[str, Any], filename: str = None):
        """テスト結果保存"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"self_learning_test_results_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"[SELF-LEARN] テスト結果を保存: {filename}")
            return filename
            
        except Exception as e:
            print(f"[ERROR] テスト結果保存エラー: {e}")
            return None


async def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="自己学習機能テストシステム")
    parser.add_argument("--test", choices=[
        'intent', 'database', 'integration', 'data_access', 'self_edit', 'all'
    ], default='all', help="実行するテスト種類")
    parser.add_argument("--output", help="結果出力ファイル名")
    parser.add_argument("--no-save", action="store_true", help="結果を保存しない")
    
    args = parser.parse_args()
    
    tester = SelfLearningTester()
    
    try:
        if await tester.initialize():
            if args.test == 'all':
                results = await tester.run_comprehensive_test()
                tester.print_comprehensive_summary(results)
            else:
                # 個別テスト実行
                test_map = {
                    'intent': tester.test_intent_analysis,
                    'database': tester.test_database_operations,
                    'integration': tester.test_learning_system_integration,
                    'data_access': tester.test_learning_data_access,
                    'self_edit': tester.test_self_edit_functions
                }
                
                if args.test in test_map:
                    result = await test_map[args.test]()
                    print(f"\n[RESULT] {result['test_name']} 完了")
                    if 'summary' in result:
                        summary = result['summary']
                        print(f"成功率: {summary.get('success_rate', 0):.1%}")
                        print(f"成功: {summary.get('successful', 0)}/{summary.get('total_tests', 0)}")
            
            # 結果保存
            if not args.no_save and 'results' in locals():
                tester.save_test_results(results, args.output)
            
        else:
            print("[ERROR] システム初期化に失敗しました")
            
    except KeyboardInterrupt:
        print("\n[SELF-LEARN] ユーザーによって中断されました")
    except Exception as e:
        print(f"[ERROR] テスト実行エラー: {e}")
    finally:
        await tester.shutdown()


if __name__ == "__main__":
    asyncio.run(main())