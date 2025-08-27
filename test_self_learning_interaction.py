#!/usr/bin/env python3
"""
Self-Learning Interaction Test
実際のエージェントとの対話で自己学習機能をテスト
"""

import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.core.config import Config
from agent.core.database import DatabaseManager
from agent.core.agent_manager import AgentManager


async def test_self_learning_interaction():
    """自己学習機能の対話テスト"""
    print("🤖 自己学習機能対話テスト開始...")
    
    try:
        # システム初期化
        config = Config()
        db_manager = DatabaseManager(config.database_url)
        await db_manager.initialize()
        
        agent_manager = AgentManager(config, db_manager)
        await agent_manager.initialize()
        
        print("✅ システム初期化完了")
        
        # テストケース
        test_cases = [
            {
                'description': '学習データ統計の取得',
                'message': '学習データの統計を表示してください',
                'expected_tools': ['learning']
            },
            {
                'description': 'ファイル書き込みテスト',
                'message': 'write file src/data/prompts/interaction_test.txt\n対話テスト用ファイル\n作成日時: ' + datetime.now().isoformat(),
                'expected_tools': ['file']
            },
            {
                'description': 'ファイル読み取りテスト',
                'message': 'read file src/data/prompts/interaction_test.txt',
                'expected_tools': ['file']
            },
            {
                'description': 'プロンプト更新テスト',
                'message': 'update prompt interaction_test_prompt: これは対話テスト用のプロンプトです',
                'expected_tools': ['file']
            },
            {
                'description': '学習データ追加テスト',
                'message': 'add learning data: 対話テストで追加された学習データ - カテゴリ: interaction_test',
                'expected_tools': ['file']
            },
            {
                'description': '学習データ一覧取得',
                'message': '最近追加された学習データを5件表示してください',
                'expected_tools': ['learning']
            }
        ]
        
        print(f"\n📋 {len(test_cases)}個のテストケースを実行します...\n")
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"🧪 テスト {i}/{len(test_cases)}: {test_case['description']}")
            print(f"   メッセージ: {test_case['message']}")
            
            try:
                # エージェントにメッセージを送信
                response = await agent_manager.process_message(test_case['message'])
                
                # 結果の分析
                response_text = response.get('response', '')
                tools_used = response.get('tools_used', [])
                intent = response.get('intent', {})
                response_time = response.get('response_time', 0)
                
                # 期待されたツールが使用されたかチェック
                expected_tools = test_case.get('expected_tools', [])
                tools_match = any(tool in tools_used for tool in expected_tools) if expected_tools else True
                
                result = {
                    'test_case': test_case['description'],
                    'message': test_case['message'],
                    'response': response_text,
                    'tools_used': tools_used,
                    'expected_tools': expected_tools,
                    'tools_match': tools_match,
                    'intent': intent.get('primary_intent', 'unknown'),
                    'response_time': response_time,
                    'success': bool(response_text and (not expected_tools or tools_match))
                }
                
                results.append(result)
                
                # 結果表示
                print(f"   応答: {response_text[:100]}{'...' if len(response_text) > 100 else ''}")
                print(f"   使用ツール: {tools_used}")
                print(f"   期待ツール: {expected_tools}")
                print(f"   ツール一致: {'✅' if tools_match else '❌'}")
                print(f"   意図: {intent.get('primary_intent', 'unknown')}")
                print(f"   応答時間: {response_time:.2f}秒")
                print(f"   成功: {'✅' if result['success'] else '❌'}")
                
            except Exception as e:
                print(f"   ❌ エラー: {e}")
                results.append({
                    'test_case': test_case['description'],
                    'message': test_case['message'],
                    'error': str(e),
                    'success': False
                })
            
            print()  # 空行
            await asyncio.sleep(1)  # 少し待機
        
        # 結果サマリー
        print("="*60)
        print("📊 対話テスト結果サマリー")
        print("="*60)
        
        successful_tests = sum(1 for r in results if r.get('success', False))
        total_tests = len(results)
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        print(f"総テスト数: {total_tests}")
        print(f"成功: {successful_tests}")
        print(f"失敗: {total_tests - successful_tests}")
        print(f"成功率: {success_rate:.1%}")
        
        # 失敗したテストの詳細
        failed_tests = [r for r in results if not r.get('success', False)]
        if failed_tests:
            print(f"\n❌ 失敗したテスト:")
            for failed in failed_tests:
                print(f"  - {failed['test_case']}")
                if 'error' in failed:
                    print(f"    エラー: {failed['error']}")
                elif not failed.get('tools_match', True):
                    print(f"    期待ツール: {failed.get('expected_tools', [])}")
                    print(f"    使用ツール: {failed.get('tools_used', [])}")
        
        # 成功したテストの詳細
        successful_tests_list = [r for r in results if r.get('success', False)]
        if successful_tests_list:
            print(f"\n✅ 成功したテスト:")
            for success in successful_tests_list:
                print(f"  - {success['test_case']}")
                print(f"    使用ツール: {success.get('tools_used', [])}")
                print(f"    応答時間: {success.get('response_time', 0):.2f}秒")
        
        print("="*60)
        
        # クリーンアップ
        await agent_manager.shutdown()
        await db_manager.close()
        
        return results
        
    except Exception as e:
        print(f"❌ 対話テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return []


async def main():
    """メイン関数"""
    print("🤖 自己学習機能対話テストシステム")
    print("=" * 50)
    
    results = await test_self_learning_interaction()
    
    if results:
        print("\n🎉 対話テスト完了！")
        
        # 結果をファイルに保存
        import json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"interaction_test_results_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"💾 結果を保存: {filename}")
        except Exception as e:
            print(f"⚠️ 結果保存エラー: {e}")
    else:
        print("❌ 対話テストが完了しませんでした")


if __name__ == "__main__":
    asyncio.run(main())