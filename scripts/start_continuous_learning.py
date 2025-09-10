#!/usr/bin/env python3
"""
Start Continuous Learning
4時間継続学習開始スクリプト
"""

import asyncio
import sys
import os
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from continuous_learning_system import ContinuousLearningSystem


async def main():
    """メイン関数"""
    print("=" * 60)
    print("361do_AI 4時間継続学習システム")
    print("=" * 60)
    print()
    
    try:
        # 継続学習システム作成
        learning_system = ContinuousLearningSystem(learning_duration_hours=4)
        
        print("エージェント初期化中...")
        # エージェント初期化
        if not await learning_system.initialize_agent():
            print("❌ エージェント初期化に失敗しました")
            return
        
        print("✅ エージェント初期化完了")
        print()
        
        # データファイル確認
        print("データファイル確認中...")
        chatgpt_conversations = learning_system._load_chatgpt_data()
        claude_conversations = learning_system._load_claude_data()
        
        total_conversations = len(chatgpt_conversations) + len(claude_conversations)
        print(f"📊 総会話数: {total_conversations}")
        print(f"   - ChatGPT: {len(chatgpt_conversations)}")
        print(f"   - Claude: {len(claude_conversations)}")
        print()
        
        if total_conversations == 0:
            print("❌ 学習データが見つかりません")
            return
        
        # 学習設定表示
        print("学習設定:")
        print(f"   - 学習時間: 4時間")
        print(f"   - バッチサイズ: {learning_system.learning_config['batch_size']}")
        print(f"   - 学習間隔: {learning_system.learning_config['learning_interval']}秒")
        print(f"   - サイクルあたり最大会話数: {learning_system.learning_config['max_conversations_per_cycle']}")
        print()
        
        # 確認
        response = input("4時間継続学習を開始しますか？ (y/N): ")
        if response.lower() != 'y':
            print("学習をキャンセルしました")
            return
        
        print()
        print("🚀 4時間継続学習開始...")
        print("   Ctrl+C で早期終了可能")
        print("=" * 60)
        
        # 4時間継続学習開始
        await learning_system.start_continuous_learning()
        
        # 学習完了
        stats = learning_system.get_learning_statistics()
        print()
        print("=" * 60)
        print("🎉 学習完了!")
        print("=" * 60)
        print(f"総処理数: {stats['total_processed']}")
        print(f"学習サイクル数: {stats['learning_cycles']}")
        print(f"最終エポック: {stats['current_epoch']}")
        print(f"開始時間: {stats['start_time']}")
        print(f"終了時間: {stats['end_time']}")
        print()
        
    except KeyboardInterrupt:
        print("\n⚠️  学習がユーザーによって中断されました")
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
