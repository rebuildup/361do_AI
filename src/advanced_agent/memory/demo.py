"""
Memory System Demonstration

記憶システムのデモンストレーション
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from .persistent_memory import LangChainPersistentMemory
from .conversation_manager import ConversationManager


async def demo_basic_memory_operations():
    """基本的な記憶操作のデモ"""
    
    print("=== LangChain + ChromaDB 永続的記憶システム デモ ===\n")
    
    # 記憶システム初期化
    memory_system = LangChainPersistentMemory(
        db_path="data/demo_memory.db",
        chroma_path="data/demo_chroma"
    )
    
    try:
        # 1. セッション初期化
        print("1. セッション初期化")
        session_id = await memory_system.initialize_session(user_id="demo_user")
        print(f"   セッションID: {session_id}\n")
        
        # 2. 会話保存
        print("2. 会話の保存")
        conversations = [
            ("Pythonでの機械学習について教えて", "Pythonでは scikit-learn, TensorFlow, PyTorch などのライブラリが人気です。"),
            ("ディープラーニングのモデル訓練のコツは？", "適切なデータ前処理、正則化、学習率調整が重要です。"),
            ("GPU メモリ不足の対処法は？", "バッチサイズ削減、勾配蓄積、モデル量子化などが効果的です。"),
            ("LangChain の使い方を知りたい", "LangChain は LLM アプリケーション開発のフレームワークです。"),
            ("ChromaDB とは何ですか？", "ChromaDB は埋め込みベクトル用のオープンソースデータベースです。")
        ]
        
        for i, (user_input, agent_response) in enumerate(conversations, 1):
            conversation_id = await memory_system.store_conversation(
                user_input=user_input,
                agent_response=agent_response,
                metadata={"demo_step": i}
            )
            print(f"   会話 {i}: {conversation_id[:20]}...")
        
        print()
        
        # 3. 関連コンテキスト検索
        print("3. 関連コンテキスト検索")
        queries = [
            "機械学習",
            "GPU メモリ",
            "データベース"
        ]
        
        for query in queries:
            print(f"\n   クエリ: '{query}'")
            context = await memory_system.retrieve_relevant_context(
                query=query,
                session_id=session_id,
                max_results=3
            )
            
            print(f"   関連会話数: {len(context['similar_conversations'])}")
            for j, conv in enumerate(context['similar_conversations'][:2], 1):
                print(f"     {j}. スコア: {conv['score']:.3f}")
                print(f"        内容: {conv['content'][:100]}...")
        
        print()
        
        # 4. 統計情報表示
        print("4. 記憶システム統計")
        stats = memory_system.get_memory_statistics()
        print(f"   ベクトルストア文書数: {stats['vector_store']['total_documents']}")
        print(f"   総会話数: {stats['structured_data']['total_conversations']}")
        print(f"   セッション数: {stats['structured_data']['unique_sessions']}")
        print(f"   現在のメモリバッファ: {stats['memory_buffer']['current_messages']} メッセージ")
        
        print()
        
        # 5. セッション要約
        print("5. セッション要約")
        context = await memory_system.retrieve_relevant_context(
            query="今日の会話の要約",
            session_id=session_id
        )
        print(f"   要約: {context['session_summary'][:200]}...")
        
    finally:
        memory_system.close()


async def demo_conversation_manager():
    """会話管理システムのデモ"""
    
    print("\n=== 会話管理システム デモ ===\n")
    
    # システム初期化
    memory_system = LangChainPersistentMemory(
        db_path="data/demo_memory.db",
        chroma_path="data/demo_chroma"
    )
    
    conversation_manager = ConversationManager(
        memory_system=memory_system,
        session_timeout_hours=24
    )
    
    try:
        # 1. 複数セッション管理
        print("1. 複数セッション管理")
        
        # セッション1: AI開発者
        session1 = await conversation_manager.start_conversation(user_id="ai_developer")
        await conversation_manager.continue_conversation(
            session_id=session1,
            user_input="PyTorch でのモデル最適化について",
            agent_response="PyTorch では torch.jit.script や量子化が効果的です。"
        )
        
        # セッション2: データサイエンティスト
        session2 = await conversation_manager.start_conversation(user_id="data_scientist")
        await conversation_manager.continue_conversation(
            session_id=session2,
            user_input="データ前処理のベストプラクティス",
            agent_response="欠損値処理、外れ値検出、特徴量スケーリングが重要です。"
        )
        
        print(f"   セッション1 (AI開発者): {session1[:20]}...")
        print(f"   セッション2 (データサイエンティスト): {session2[:20]}...")
        
        # 2. アクティブセッション一覧
        print("\n2. アクティブセッション一覧")
        active_sessions = conversation_manager.get_active_sessions()
        for session in active_sessions:
            print(f"   セッション: {session['session_id'][:20]}...")
            print(f"     ユーザー: {session['user_id']}")
            print(f"     会話数: {session['conversation_count']}")
        
        # 3. セッション別コンテキスト
        print("\n3. セッション別コンテキスト")
        for session_id in [session1, session2]:
            context = await conversation_manager.get_conversation_context(
                session_id=session_id,
                query="最新の話題"
            )
            session_info = context.get("session_info", {})
            print(f"   セッション {session_id[:20]}...")
            print(f"     ユーザー: {session_info.get('user_id', 'Unknown')}")
            print(f"     会話数: {session_info.get('conversation_count', 0)}")
        
        # 4. 統計情報
        print("\n4. 会話管理統計")
        stats = conversation_manager.get_session_statistics()
        print(f"   アクティブセッション数: {stats['active_sessions']}")
        print(f"   総会話数: {stats['total_conversations']}")
        
        # 5. セッション終了
        print("\n5. セッション終了")
        await conversation_manager.end_session(session1)
        await conversation_manager.end_session(session2)
        print("   全セッションを終了しました")
        
    finally:
        memory_system.close()


async def demo_memory_persistence():
    """記憶永続化のデモ"""
    
    print("\n=== 記憶永続化デモ ===\n")
    
    # 1回目: データ保存
    print("1. 初回実行 - データ保存")
    memory_system1 = LangChainPersistentMemory(
        db_path="data/persistence_test.db",
        chroma_path="data/persistence_chroma"
    )
    
    session_id = await memory_system1.initialize_session(user_id="persistence_test")
    await memory_system1.store_conversation(
        user_input="永続化テストです",
        agent_response="データが正常に保存されました。"
    )
    
    stats1 = memory_system1.get_memory_statistics()
    print(f"   保存された会話数: {stats1['structured_data']['total_conversations']}")
    memory_system1.close()
    
    # 2回目: データ復元
    print("\n2. 再実行 - データ復元")
    memory_system2 = LangChainPersistentMemory(
        db_path="data/persistence_test.db",
        chroma_path="data/persistence_chroma"
    )
    
    # 同じセッションIDで復元
    await memory_system2.initialize_session(session_id=session_id)
    
    # 過去の会話を検索
    context = await memory_system2.retrieve_relevant_context(
        query="永続化",
        session_id=session_id
    )
    
    stats2 = memory_system2.get_memory_statistics()
    print(f"   復元された会話数: {stats2['structured_data']['total_conversations']}")
    print(f"   関連会話検索結果: {len(context['similar_conversations'])} 件")
    
    if context['similar_conversations']:
        print(f"   検索された内容: {context['similar_conversations'][0]['content'][:100]}...")
    
    memory_system2.close()


async def main():
    """メインデモ実行"""
    
    # データディレクトリ作成
    Path("data").mkdir(exist_ok=True)
    
    try:
        await demo_basic_memory_operations()
        await demo_conversation_manager()
        await demo_memory_persistence()
        
        print("\n=== デモ完了 ===")
        print("記憶システムが正常に動作することを確認しました。")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())