#!/usr/bin/env python3
"""
Multi-Agent Learning System Test
マルチエージェント学習システムのテスト用スクリプト
"""

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from multi_agent_learning_system import MultiAgentLearningSystem


async def test_agent_initialization():
    """エージェント初期化テスト"""
    print("🧪 エージェント初期化テスト開始...")
    
    system = MultiAgentLearningSystem(time_limit_hours=0.01)  # 36秒制限
    
    try:
        success = await system.initialize_agents()
        
        if success:
            print("✅ エージェント初期化成功")
            
            # エージェント情報表示
            for agent_id, agent_data in system.agents.items():
                role = agent_data['role']
                print(f"  🤖 {agent_id}: {role['name']} - {role['focus']}")
            
            return True
        else:
            print("❌ エージェント初期化失敗")
            return False
            
    except Exception as e:
        print(f"❌ 初期化テストエラー: {e}")
        return False
    finally:
        await system.shutdown_agents()


async def test_single_conversation():
    """単一会話テスト"""
    print("\n🧪 単一会話テスト開始...")
    
    system = MultiAgentLearningSystem(time_limit_hours=0.01)
    
    try:
        if await system.initialize_agents():
            # テスト用トピック
            test_topic = "AIの学習効率について"
            
            # グループ会話実行（1ラウンドのみ）
            conversation_log = await system.conduct_group_conversation(test_topic, rounds=1)
            
            print(f"✅ 会話テスト成功: {len(conversation_log)}件の発言")
            
            # 発言内容表示
            for conv in conversation_log:
                if conv['success']:
                    print(f"  💬 {conv['agent_name']}: {conv['content'][:80]}...")
                else:
                    print(f"  ❌ {conv['agent_name']}: エラー")
            
            return True
        else:
            print("❌ 初期化失敗")
            return False
            
    except Exception as e:
        print(f"❌ 会話テストエラー: {e}")
        return False
    finally:
        await system.shutdown_agents()


async def test_cross_learning():
    """相互学習テスト"""
    print("\n🧪 相互学習テスト開始...")
    
    system = MultiAgentLearningSystem(time_limit_hours=0.01)
    
    try:
        if await system.initialize_agents():
            # まず会話を実行して学習データを生成
            test_topic = "効率的な問題解決手法"
            conversation_log = await system.conduct_group_conversation(test_topic, rounds=1)
            
            if conversation_log:
                # 相互学習実行
                learning_results = await system.cross_agent_learning()
                
                print(f"✅ 相互学習テスト成功: {len(learning_results)}エージェント")
                
                for result in learning_results:
                    status = result['learning_status']
                    data_count = result['data_added']
                    print(f"  🧠 {result['agent_id']}: {status} (データ追加: {data_count}件)")
                
                return True
            else:
                print("❌ 会話データなし")
                return False
        else:
            print("❌ 初期化失敗")
            return False
            
    except Exception as e:
        print(f"❌ 相互学習テストエラー: {e}")
        return False
    finally:
        await system.shutdown_agents()


async def test_time_limit():
    """時間制限テスト"""
    print("\n🧪 時間制限テスト開始...")
    
    # 30秒制限でテスト
    system = MultiAgentLearningSystem(time_limit_hours=30/3600)  # 30秒
    system.setup_signal_handlers()
    
    try:
        if await system.initialize_agents():
            start_time = time.time()
            
            # 短時間実行
            await system.run_multi_agent_learning()
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"✅ 時間制限テスト成功: {duration:.1f}秒実行")
            
            # 統計表示
            stats = system.learning_stats
            print(f"  📊 会話数: {stats['total_conversations']}")
            print(f"  🧠 学習サイクル: {stats['total_learning_cycles']}")
            print(f"  🤝 知識共有: {stats['knowledge_shared']}")
            
            return True
        else:
            print("❌ 初期化失敗")
            return False
            
    except Exception as e:
        print(f"❌ 時間制限テストエラー: {e}")
        return False
    finally:
        await system.shutdown_agents()


async def test_error_handling():
    """エラーハンドリングテスト"""
    print("\n🧪 エラーハンドリングテスト開始...")
    
    system = MultiAgentLearningSystem(time_limit_hours=0.01)
    
    try:
        # 初期化せずに会話を試行（エラーが発生するはず）
        conversation_log = await system.conduct_group_conversation("テストトピック", rounds=1)
        
        # エラーが適切に処理されているかチェック
        error_count = sum(1 for conv in conversation_log if not conv['success'])
        
        if error_count > 0:
            print(f"✅ エラーハンドリングテスト成功: {error_count}件のエラーを適切に処理")
            return True
        else:
            print("⚠️ エラーが発生しませんでした（予期しない結果）")
            return False
            
    except Exception as e:
        print(f"✅ エラーハンドリングテスト成功: 例外を適切にキャッチ ({e})")
        return True


async def run_all_tests():
    """全テスト実行"""
    print("🚀 マルチエージェント学習システム テストスイート")
    print("=" * 60)
    
    tests = [
        ("エージェント初期化", test_agent_initialization),
        ("単一会話", test_single_conversation),
        ("相互学習", test_cross_learning),
        ("時間制限", test_time_limit),
        ("エラーハンドリング", test_error_handling)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"テスト: {test_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            success = await test_func()
            duration = time.time() - start_time
            
            results.append({
                'name': test_name,
                'success': success,
                'duration': duration
            })
            
            status = "✅ 成功" if success else "❌ 失敗"
            print(f"\n{status} ({duration:.2f}秒)")
            
        except Exception as e:
            duration = time.time() - start_time
            results.append({
                'name': test_name,
                'success': False,
                'duration': duration,
                'error': str(e)
            })
            print(f"\n❌ テスト例外: {e} ({duration:.2f}秒)")
        
        # テスト間の間隔
        await asyncio.sleep(2)
    
    # 最終結果表示
    print(f"\n{'='*60}")
    print("🎯 テスト結果サマリー")
    print(f"{'='*60}")
    
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    total_time = sum(r['duration'] for r in results)
    
    print(f"成功: {successful}/{total}")
    print(f"総実行時間: {total_time:.2f}秒")
    
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"{status} {result['name']}: {result['duration']:.2f}秒")
        if 'error' in result:
            print(f"   エラー: {result['error']}")
    
    print(f"{'='*60}")
    
    if successful == total:
        print("🎉 全テスト成功！")
    else:
        print(f"⚠️ {total - successful}件のテストが失敗しました")


async def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="マルチエージェント学習システム テスト")
    parser.add_argument("--test", choices=[
        "init", "conversation", "learning", "time", "error", "all"
    ], default="all", help="実行するテスト")
    
    args = parser.parse_args()
    
    if args.test == "all":
        await run_all_tests()
    elif args.test == "init":
        await test_agent_initialization()
    elif args.test == "conversation":
        await test_single_conversation()
    elif args.test == "learning":
        await test_cross_learning()
    elif args.test == "time":
        await test_time_limit()
    elif args.test == "error":
        await test_error_handling()


if __name__ == "__main__":
    asyncio.run(main())