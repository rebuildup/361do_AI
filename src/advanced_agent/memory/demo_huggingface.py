"""
HuggingFace Memory System Demonstration

LangChain Memory + HuggingFace 統合記憶システムのデモンストレーション
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from .persistent_memory import LangChainPersistentMemory
from .huggingface_memory import LangChainHuggingFaceMemory, HuggingFaceMemoryClassifier


async def demo_huggingface_classifier():
    """HuggingFace 記憶分類器のデモ"""
    
    print("=== HuggingFace 記憶分類器 デモ ===\n")
    
    classifier = HuggingFaceMemoryClassifier(device="cpu")
    
    # テストテキスト
    test_texts = [
        "Pythonでの機械学習について教えてください。scikit-learnとTensorFlowの違いは何ですか？",
        "重要なエラーが発生しました！GPU メモリ不足でモデルの訓練が停止しています。",
        "こんにちは、今日はいい天気ですね。",
        "まず設定ファイルを開いて、パラメータを変更し、その後システムを再起動してください。",
        "データベースの正規化について説明します。第一正規形から第三正規形まで段階的に進めます。"
    ]
    
    print("1. 重要度評価とタイプ分類")
    for i, text in enumerate(test_texts, 1):
        print(f"\n   テキスト {i}: {text[:50]}...")
        
        # 重要度評価
        importance = await classifier.evaluate_importance(text)
        
        # タイプ分類
        memory_type = await classifier.classify_memory_type(text)
        
        # キー概念抽出
        concepts = await classifier.extract_key_concepts(text)
        
        print(f"     重要度: {importance:.3f}")
        print(f"     タイプ: {memory_type}")
        print(f"     概念: {concepts[:5]}")  # 最初の5個まで表示


async def demo_integrated_memory_system():
    """統合記憶システムのデモ"""
    
    print("\n=== LangChain + HuggingFace 統合記憶システム デモ ===\n")
    
    # システム初期化
    persistent_memory = LangChainPersistentMemory(
        db_path="data/demo_hf_memory.db",
        chroma_path="data/demo_hf_chroma"
    )
    
    hf_memory = LangChainHuggingFaceMemory(
        persistent_memory=persistent_memory,
        max_short_term_memories=5,
        importance_threshold=0.6
    )
    
    try:
        # セッション初期化
        session_id = await persistent_memory.initialize_session(user_id="demo_user")
        print(f"1. セッション初期化: {session_id[:20]}...\n")
        
        # 様々なタイプの会話を処理
        conversations = [
            ("Pythonでの機械学習について教えて", "Pythonでは scikit-learn, TensorFlow, PyTorch などのライブラリが人気です。"),
            ("重要なエラーが発生しました", "GPU メモリ不足エラーですね。バッチサイズを削減してみてください。"),
            ("今日の天気はどうですか？", "申し訳ありませんが、天気情報は提供できません。"),
            ("データベース設計のベストプラクティス", "正規化、インデックス設計、パフォーマンス最適化が重要です。"),
            ("Dockerの使い方を教えて", "Dockerfileを作成し、docker buildでイメージを構築します。"),
            ("簡単な挨拶", "こんにちは！何かお手伝いできることはありますか？")
        ]
        
        print("2. 会話処理と記憶分類")
        for i, (user_input, agent_response) in enumerate(conversations, 1):
            result = await hf_memory.process_conversation(
                user_input=user_input,
                agent_response=agent_response,
                session_id=session_id,
                metadata={"demo_step": i}
            )
            
            print(f"   会話 {i}: {user_input[:30]}...")
            print(f"     重要度: {result['importance_score']:.3f}")
            print(f"     タイプ: {result['memory_type']}")
            print(f"     概念: {result['key_concepts'][:3]}")
            print(f"     長期記憶昇格: {'Yes' if result['promoted_to_long_term'] else 'No'}")
        
        print()
        
        # 記憶統計表示
        print("3. 記憶システム統計")
        stats = hf_memory.get_memory_statistics()
        
        short_term = stats["short_term_memory"]
        print(f"   短期記憶: {short_term['current_messages']}/{short_term['max_capacity']} ({short_term['usage_percent']:.1f}%)")
        
        processing = stats["processing_stats"]
        print(f"   処理済み会話: {processing['total_processed']}")
        print(f"   長期記憶昇格: {processing['promoted_to_long_term']}")
        
        persistent = stats["persistent_memory"]
        print(f"   永続化会話: {persistent['structured_data']['total_conversations']}")
        
        print()
        
        # コンテキスト検索デモ
        print("4. コンテキスト記憶検索")
        queries = [
            "Python 機械学習",
            "エラー 問題",
            "データベース 設計"
        ]
        
        for query in queries:
            print(f"\n   クエリ: '{query}'")
            context = await hf_memory.retrieve_contextual_memories(
                query=query,
                session_id=session_id,
                max_results=3
            )
            
            query_analysis = context["query_analysis"]
            print(f"     クエリ重要度: {query_analysis['importance']:.3f}")
            print(f"     クエリタイプ: {query_analysis['type']}")
            print(f"     クエリ概念: {query_analysis['concepts'][:3]}")
            
            relevant_memories = context["relevant_memories"]
            print(f"     関連記憶数: {len(relevant_memories)}")
            
            for j, memory in enumerate(relevant_memories[:2], 1):
                print(f"       {j}. 関連度: {memory['relevance_score']:.3f}")
                print(f"          内容: {memory['content'][:60]}...")
        
        print()
        
        # 記憶パターン分析
        print("5. 記憶パターン分析")
        analysis = await hf_memory.analyze_memory_patterns(session_id)
        
        print(f"   総会話数: {analysis['total_conversations']}")
        print(f"   記憶タイプ分布:")
        for mem_type, count in analysis["memory_type_distribution"].items():
            print(f"     {mem_type}: {count}")
        
        print(f"   頻出概念:")
        for concept, freq in analysis["top_concepts"][:5]:
            print(f"     {concept}: {freq}")
        
        importance_stats = analysis["importance_stats"]
        print(f"   重要度統計:")
        print(f"     平均: {importance_stats['average']:.3f}")
        print(f"     最小: {importance_stats['min']:.3f}")
        print(f"     最大: {importance_stats['max']:.3f}")
        print(f"     高重要度: {importance_stats['high_importance_count']}")
        
        print()
        
        # 記憶最適化デモ
        print("6. 記憶最適化")
        optimization = await hf_memory.optimize_memory_capacity()
        
        print(f"   短期記憶最適化: {optimization['short_term_optimized']} 件削除")
        print(f"   永続化記憶整理: {optimization['persistent_cleaned']} 件削除")
        
        # 最適化後の統計
        final_stats = hf_memory.get_memory_statistics()
        final_short_term = final_stats["short_term_memory"]
        print(f"   最適化後短期記憶: {final_short_term['current_messages']}/{final_short_term['max_capacity']}")
        
    finally:
        persistent_memory.close()


async def demo_memory_comparison():
    """記憶システム比較デモ"""
    
    print("\n=== 記憶システム比較デモ ===\n")
    
    # 基本記憶システム
    basic_memory = LangChainPersistentMemory(
        db_path="data/demo_basic_memory.db",
        chroma_path="data/demo_basic_chroma"
    )
    
    # HuggingFace統合記憶システム
    persistent_memory = LangChainPersistentMemory(
        db_path="data/demo_enhanced_memory.db",
        chroma_path="data/demo_enhanced_chroma"
    )
    
    enhanced_memory = LangChainHuggingFaceMemory(
        persistent_memory=persistent_memory,
        importance_threshold=0.6
    )
    
    try:
        # 同じ会話を両システムで処理
        session_id1 = await basic_memory.initialize_session(user_id="basic_user")
        session_id2 = await persistent_memory.initialize_session(user_id="enhanced_user")
        
        test_conversation = (
            "重要なシステムエラーが発生しています。GPU メモリ不足でPythonの機械学習モデル訓練が失敗しました。",
            "GPU メモリ不足の解決策として、バッチサイズの削減、勾配蓄積、モデル量子化を試してください。"
        )
        
        user_input, agent_response = test_conversation
        
        # 基本システムで処理
        print("1. 基本記憶システム")
        basic_id = await basic_memory.store_conversation(
            user_input=user_input,
            agent_response=agent_response
        )
        basic_importance = await basic_memory._calculate_importance(f"User: {user_input}\nAgent: {agent_response}")
        print(f"   会話ID: {basic_id[:20]}...")
        print(f"   重要度: {basic_importance:.3f}")
        print(f"   分析機能: 基本的な重要度計算のみ")
        
        # 拡張システムで処理
        print("\n2. HuggingFace統合記憶システム")
        enhanced_result = await enhanced_memory.process_conversation(
            user_input=user_input,
            agent_response=agent_response,
            session_id=session_id2
        )
        
        print(f"   会話ID: {enhanced_result['conversation_id'][:20]}...")
        print(f"   重要度: {enhanced_result['importance_score']:.3f}")
        print(f"   記憶タイプ: {enhanced_result['memory_type']}")
        print(f"   キー概念: {enhanced_result['key_concepts']}")
        print(f"   長期記憶昇格: {'Yes' if enhanced_result['promoted_to_long_term'] else 'No'}")
        
        print("\n3. 比較結果")
        print(f"   重要度差: {enhanced_result['importance_score'] - basic_importance:.3f}")
        print(f"   拡張機能: タイプ分類、概念抽出、自動昇格")
        
    finally:
        basic_memory.close()
        persistent_memory.close()


async def main():
    """メインデモ実行"""
    
    # データディレクトリ作成
    Path("data").mkdir(exist_ok=True)
    
    try:
        await demo_huggingface_classifier()
        await demo_integrated_memory_system()
        await demo_memory_comparison()
        
        print("\n=== デモ完了 ===")
        print("LangChain Memory + HuggingFace 統合記憶システムが正常に動作することを確認しました。")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())