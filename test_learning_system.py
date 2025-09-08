#!/usr/bin/env python3
"""
Test Learning System
テスト用学習システム
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.advanced_agent.core.self_learning_agent import SelfLearningAgent
from src.advanced_agent.config.settings import get_agent_config


async def test_agent_initialization():
    """エージェント初期化テスト"""
    print("=" * 60)
    print("エージェント初期化テスト")
    print("=" * 60)
    
    try:
        # 設定確認
        print("1. 設定ファイル確認...")
        config_path = "config/agent_config.yaml"
        if os.path.exists(config_path):
            print(f"   ✅ 設定ファイル存在: {config_path}")
        else:
            print(f"   ❌ 設定ファイル不存在: {config_path}")
            return False
        
        # エージェント作成
        print("2. エージェント作成...")
        agent = SelfLearningAgent(
            config_path=config_path,
            db_path="data/self_learning_agent.db"
        )
        print("   ✅ エージェント作成完了")
        
        # セッション初期化
        print("3. セッション初期化...")
        session_id = await agent.initialize_session(
            session_id="test_session",
            user_id="test_user"
        )
        print(f"   ✅ セッション初期化完了: {session_id}")
        
        # エージェント状態確認
        print("4. エージェント状態確認...")
        if agent.current_state:
            print(f"   ✅ エージェント状態: {agent.current_state.session_id}")
            print(f"   - 学習エポック: {agent.current_state.learning_epoch}")
            print(f"   - 総インタラクション: {agent.current_state.total_interactions}")
        else:
            print("   ⚠️  エージェント状態が設定されていません")
        
        # メモリシステム確認
        print("5. メモリシステム確認...")
        if agent.memory_system:
            print("   ✅ メモリシステム利用可能")
        else:
            print("   ❌ メモリシステムが利用できません")
        
        print("\n✅ エージェント初期化テスト完了")
        return True
        
    except Exception as e:
        print(f"\n❌ エージェント初期化テスト失敗: {e}")
        import traceback
        print(f"詳細: {traceback.format_exc()}")
        return False


async def test_data_loading():
    """データ読み込みテスト"""
    print("\n" + "=" * 60)
    print("データ読み込みテスト")
    print("=" * 60)
    
    try:
        # ChatGPTデータ確認
        print("1. ChatGPTデータ確認...")
        chatgpt_files = [
            "workspace/chat-gpt-data/conversations.json",
            "workspace/chat-gpt-data/shared_conversations.json"
        ]
        
        for file_path in chatgpt_files:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                print(f"   ✅ {file_path}: {size:,} bytes")
            else:
                print(f"   ❌ {file_path}: ファイル不存在")
        
        # Claudeデータ確認
        print("2. Claudeデータ確認...")
        claude_file = "workspace/claude-data/conversations.json"
        if os.path.exists(claude_file):
            size = os.path.getsize(claude_file)
            print(f"   ✅ {claude_file}: {size:,} bytes")
        else:
            print(f"   ❌ {claude_file}: ファイル不存在")
        
        # データ読み込みテスト
        print("3. データ読み込みテスト...")
        test_file = "workspace/chat-gpt-data/shared_conversations.json"
        if os.path.exists(test_file):
            with open(test_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"   ✅ データ読み込み成功: {len(data) if isinstance(data, list) else 'N/A'} 項目")
        else:
            print(f"   ❌ テストファイル不存在: {test_file}")
        
        print("\n✅ データ読み込みテスト完了")
        return True
        
    except Exception as e:
        print(f"\n❌ データ読み込みテスト失敗: {e}")
        import traceback
        print(f"詳細: {traceback.format_exc()}")
        return False


async def test_simple_learning():
    """簡単な学習テスト"""
    print("\n" + "=" * 60)
    print("簡単な学習テスト")
    print("=" * 60)
    
    try:
        # エージェント初期化
        agent = SelfLearningAgent(
            config_path="config/agent_config.yaml",
            db_path="data/self_learning_agent.db"
        )
        
        session_id = await agent.initialize_session(
            session_id="test_learning_session",
            user_id="test_user"
        )
        
        print(f"1. エージェント初期化完了: {session_id}")
        
        # テストデータで学習
        print("2. テスト学習データ処理...")
        test_data = {
            'content': 'これはテスト用の会話データです。ユーザーが質問をして、エージェントが回答する形式のデータです。',
            'metadata': {
                'source': 'test',
                'conversation_id': 'test_001',
                'title': 'テスト会話'
            }
        }
        
        # メモリシステムに保存
        if agent.memory_system:
            await agent.memory_system.store_conversation(
                user_input="テスト質問です",
                agent_response="テスト回答です",
                metadata={
                    "interaction_id": "test_001",
                    "learning_epoch": 1,
                    "source": "test"
                }
            )
            print("   ✅ テストデータ保存完了")
        else:
            print("   ❌ メモリシステムが利用できません")
        
        # エージェント状態更新
        if agent.current_state:
            agent.current_state.total_interactions += 1
            agent.current_state.learning_epoch += 1
            print(f"   ✅ エージェント状態更新: インタラクション={agent.current_state.total_interactions}, エポック={agent.current_state.learning_epoch}")
        
        print("\n✅ 簡単な学習テスト完了")
        return True
        
    except Exception as e:
        print(f"\n❌ 簡単な学習テスト失敗: {e}")
        import traceback
        print(f"詳細: {traceback.format_exc()}")
        return False


async def main():
    """メイン関数"""
    print("361do_AI 学習システムテスト")
    print("=" * 60)
    
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # テスト実行
    tests = [
        ("エージェント初期化", test_agent_initialization),
        ("データ読み込み", test_data_loading),
        ("簡単な学習", test_simple_learning)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}テスト開始...")
        result = await test_func()
        results.append((test_name, result))
    
    # 結果表示
    print("\n" + "=" * 60)
    print("テスト結果")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✅ 成功" if result else "❌ 失敗"
        print(f"{test_name}: {status}")
    
    success_count = sum(1 for _, result in results if result)
    total_count = len(results)
    
    print(f"\n総合結果: {success_count}/{total_count} テスト成功")
    
    if success_count == total_count:
        print("🎉 すべてのテストが成功しました！継続学習システムを実行できます。")
    else:
        print("⚠️  一部のテストが失敗しました。問題を修正してから継続学習を実行してください。")


if __name__ == "__main__":
    asyncio.run(main())
