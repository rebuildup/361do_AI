#!/usr/bin/env python3
"""
Start Improved Learning
改善された学習システム開始スクリプト
"""

import asyncio
import sys
import os
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from conversation_data_processor import ConversationDataProcessor
from improved_continuous_learning_system import ImprovedContinuousLearningSystem


async def main():
    """メイン関数"""
    print("=" * 60)
    print("361do_AI 改善された学習システム")
    print("=" * 60)
    print()
    
    try:
        # ステップ1: データ処理
        print("ステップ1: 会話データ処理")
        print("-" * 40)
        
        processor = ConversationDataProcessor()
        
        # データ処理実行
        print("conversation.jsonファイルを処理中...")
        stats = processor.process_all_data()
        
        if stats["total_conversations"] == 0:
            print("❌ 処理された会話データがありません")
            return
        
        print(f"✅ データ処理完了: {stats['total_conversations']}件")
        print()
        
        # ステップ2: 学習システム初期化
        print("ステップ2: 学習システム初期化")
        print("-" * 40)
        
        learning_system = ImprovedContinuousLearningSystem(learning_duration_hours=4)
        
        print("エージェント初期化中...")
        if not await learning_system.initialize_agent():
            print("❌ エージェント初期化に失敗しました")
            return
        
        print("✅ エージェント初期化完了")
        print()
        
        # ステップ3: 学習データ確認
        print("ステップ3: 学習データ確認")
        print("-" * 40)
        
        processed_conversations = learning_system._load_processed_conversations(limit=100)
        print(f"📊 処理済み会話数: {len(processed_conversations)}")
        
        if processed_conversations:
            # サンプル表示
            sample = processed_conversations[0]
            print(f"サンプル会話:")
            print(f"  ID: {sample['id']}")
            print(f"  ソース: {sample['source']}")
            print(f"  タイトル: {sample['title']}")
            print(f"  品質スコア: {sample['quality_score']:.3f}")
            print(f"  コンテンツ長: {len(sample['content'])}文字")
            print()
        
        # ステップ4: 学習設定表示
        print("ステップ4: 学習設定")
        print("-" * 40)
        print(f"学習時間: 4時間")
        print(f"バッチサイズ: {learning_system.learning_config['batch_size']}")
        print(f"学習間隔: {learning_system.learning_config['learning_interval']}秒")
        print(f"品質閾値: {learning_system.learning_config['quality_threshold']}")
        print(f"最大コンテンツ長: {learning_system.learning_config['max_content_length']}")
        print(f"最小コンテンツ長: {learning_system.learning_config['min_content_length']}")
        print()
        
        # ステップ5: 学習開始確認
        print("ステップ5: 学習開始確認")
        print("-" * 40)
        
        if len(processed_conversations) < 10:
            print("⚠️ 学習データが少なすぎます（10件未満）")
            print("学習を続行しますか？")
            response = input("続行する場合は 'y' を入力: ")
            if response.lower() != 'y':
                print("学習をキャンセルしました")
                return
        
        response = input("改善された4時間継続学習を開始しますか？ (y/N): ")
        if response.lower() != 'y':
            print("学習をキャンセルしました")
            return
        
        print()
        print("🚀 改善された4時間継続学習開始...")
        print("   Ctrl+C で早期終了可能")
        print("=" * 60)
        
        # ステップ6: 学習実行
        await learning_system.start_continuous_learning()
        
        # ステップ7: 学習完了
        print()
        print("=" * 60)
        print("🎉 学習完了!")
        print("=" * 60)
        
        final_stats = learning_system.get_learning_statistics()
        print(f"総処理数: {final_stats['total_processed']}")
        print(f"学習サイクル数: {final_stats['learning_cycles']}")
        print(f"最終エポック: {final_stats['current_epoch']}")
        print(f"開始時間: {final_stats['start_time']}")
        print(f"終了時間: {final_stats['end_time']}")
        
        if final_stats["quality_scores"]:
            avg_quality = sum(final_stats["quality_scores"]) / len(final_stats["quality_scores"])
            print(f"平均品質スコア: {avg_quality:.3f}")
        
        print(f"エラー数: {final_stats['processing_errors']}")
        
        # ソース別統計
        print("ソース別統計:")
        for source, stats in final_stats["source_stats"].items():
            print(f"  {source}: {stats['processed']}件")
        
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⚠️  学習がユーザーによって中断されました")
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
