"""
Session Management System Demonstration

SQLAlchemy + LangChain 統合セッション管理システムのデモンストレーション
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

from .session_manager import SQLAlchemySessionManager
from .persistent_memory import LangChainPersistentMemory


async def demo_user_and_session_management():
    """ユーザーとセッション管理のデモ"""
    
    print("=== SQLAlchemy + LangChain セッション管理システム デモ ===\n")
    
    # システム初期化
    session_manager = SQLAlchemySessionManager(
        db_path="data/demo_session_management.db"
    )
    
    try:
        # 1. ユーザー作成
        print("1. ユーザー管理")
        
        users = [
            ("alice_developer", "alice@example.com", {"theme": "dark", "language": "ja", "expertise": "Python"}),
            ("bob_researcher", "bob@example.com", {"theme": "light", "language": "en", "expertise": "ML"}),
            ("charlie_student", "charlie@example.com", {"theme": "auto", "language": "ja", "expertise": "beginner"})
        ]
        
        user_ids = {}
        for username, email, preferences in users:
            user_id = await session_manager.create_user(
                username=username,
                email=email,
                preferences=preferences
            )
            user_ids[username] = user_id
            print(f"   ユーザー作成: {username} ({user_id[:20]}...)")
        
        print()
        
        # 2. セッション作成
        print("2. セッション作成")
        
        sessions = {}
        
        # Alice のセッション
        alice_session1 = await session_manager.create_session(
            user_id=user_ids["alice_developer"],
            session_name="Python開発セッション",
            session_type="development",
            metadata={"project": "web_app", "priority": "high"}
        )
        sessions["alice_dev"] = alice_session1
        
        alice_session2 = await session_manager.create_session(
            user_id=user_ids["alice_developer"],
            session_name="コードレビューセッション",
            session_type="review"
        )
        sessions["alice_review"] = alice_session2
        
        # Bob のセッション
        bob_session = await session_manager.create_session(
            user_id=user_ids["bob_researcher"],
            session_name="機械学習研究",
            session_type="research",
            metadata={"domain": "NLP", "model": "transformer"}
        )
        sessions["bob_research"] = bob_session
        
        # Charlie のセッション
        charlie_session = await session_manager.create_session(
            user_id=user_ids["charlie_student"],
            session_name="学習セッション",
            session_type="learning"
        )
        sessions["charlie_learning"] = charlie_session
        
        for session_name, session_id in sessions.items():
            print(f"   セッション作成: {session_name} ({session_id[:20]}...)")
        
        print()
        
        # 3. 会話データの追加
        print("3. 会話データの追加")
        
        # Alice の開発セッション
        alice_conversations = [
            ("Pythonでのエラーハンドリングについて教えて", "try-except文を使用してエラーを適切に処理することが重要です。"),
            ("FastAPIでのバリデーション方法は？", "Pydanticモデルを使用してリクエストデータを検証できます。"),
            ("非同期処理の最適化方法", "asyncio.gather()やasyncio.create_task()を適切に使用しましょう。")
        ]
        
        for user_input, agent_response in alice_conversations:
            await session_manager.save_conversation(
                session_id=sessions["alice_dev"],
                user_input=user_input,
                agent_response=agent_response,
                response_time=1.2,
                importance_score=0.8
            )
        
        # Bob の研究セッション
        bob_conversations = [
            ("Transformerアーキテクチャの最新動向", "GPT-4、Claude、Geminiなどの大規模言語モデルが注目されています。"),
            ("注意機構の計算効率化", "Flash AttentionやLinear Attentionなどの手法があります。"),
            ("マルチモーダル学習の課題", "テキストと画像の効果的な統合が重要な研究テーマです。")
        ]
        
        for user_input, agent_response in bob_conversations:
            await session_manager.save_conversation(
                session_id=sessions["bob_research"],
                user_input=user_input,
                agent_response=agent_response,
                response_time=2.1,
                importance_score=0.9
            )
        
        # Charlie の学習セッション
        charlie_conversations = [
            ("プログラミングを始めたいです", "Pythonから始めることをお勧めします。文法がシンプルで学習しやすいです。"),
            ("変数とは何ですか？", "変数はデータを格納するための名前付きの箱のようなものです。"),
            ("関数の作り方を教えて", "def キーワードを使って関数を定義できます。")
        ]
        
        for user_input, agent_response in charlie_conversations:
            await session_manager.save_conversation(
                session_id=sessions["charlie_learning"],
                user_input=user_input,
                agent_response=agent_response,
                response_time=0.8,
                importance_score=0.6
            )
        
        print(f"   Alice開発セッション: {len(alice_conversations)} 会話")
        print(f"   Bob研究セッション: {len(bob_conversations)} 会話")
        print(f"   Charlie学習セッション: {len(charlie_conversations)} 会話")
        
        print()
        
        # 4. セッション状態の保存
        print("4. セッション状態管理")
        
        # Alice の開発セッション状態
        await session_manager.save_session_state(
            session_id=sessions["alice_dev"],
            state_type="context",
            state_data={
                "current_project": "web_application",
                "tech_stack": ["Python", "FastAPI", "PostgreSQL"],
                "current_issue": "error_handling",
                "progress": 0.7
            }
        )
        
        await session_manager.save_session_state(
            session_id=sessions["alice_dev"],
            state_type="preferences",
            state_data={
                "code_style": "PEP8",
                "testing_framework": "pytest",
                "documentation_level": "detailed"
            }
        )
        
        # Bob の研究セッション状態
        await session_manager.save_session_state(
            session_id=sessions["bob_research"],
            state_type="context",
            state_data={
                "research_topic": "efficient_transformers",
                "current_paper": "Flash Attention 2",
                "experiment_status": "running",
                "datasets": ["GLUE", "SuperGLUE"]
            }
        )
        
        print("   セッション状態を保存しました")
        
        print()
        
        # 5. セッションコンテキストの取得
        print("5. セッションコンテキスト取得")
        
        for session_name, session_id in sessions.items():
            print(f"\n   セッション: {session_name}")
            context = await session_manager.get_session_context(session_id)
            
            user_info = context["user_info"]
            session_info = context["session_info"]
            langchain_context = context["langchain_context"]
            
            print(f"     ユーザー: {user_info['username']}")
            print(f"     セッション名: {session_info['session_name']}")
            print(f"     セッションタイプ: {session_info['session_type']}")
            print(f"     会話数: {session_info['conversation_count']}")
            print(f"     LangChainメッセージ数: {langchain_context['message_count']}")
        
        print()
        
        # 6. ユーザー別セッション一覧
        print("6. ユーザー別セッション一覧")
        
        for username, user_id in user_ids.items():
            print(f"\n   ユーザー: {username}")
            user_sessions = session_manager.get_user_sessions(user_id)
            
            for session in user_sessions:
                print(f"     - {session['session_name']} ({session['session_type']})")
                print(f"       会話数: {session['conversation_count']}, アクティブ: {session['is_currently_active']}")
        
        print()
        
        # 7. システム統計
        print("7. システム統計")
        
        stats = session_manager.get_system_statistics()
        
        db_stats = stats["database_stats"]
        runtime_stats = stats["runtime_stats"]
        
        print(f"   データベース統計:")
        print(f"     総ユーザー数: {db_stats['total_users']}")
        print(f"     アクティブユーザー数: {db_stats['active_users']}")
        print(f"     総セッション数: {db_stats['total_sessions']}")
        print(f"     アクティブセッション数: {db_stats['active_sessions_db']}")
        print(f"     総会話数: {db_stats['total_conversations']}")
        print(f"     24時間以内の活動: {db_stats['recent_activity_24h']}")
        
        print(f"   ランタイム統計:")
        print(f"     メモリ内アクティブセッション: {stats['active_sessions_memory']}")
        print(f"     復元されたセッション: {runtime_stats['restored_sessions']}")
        print(f"     整合性修復回数: {runtime_stats['integrity_repairs']}")
        
        print()
        
        return sessions, user_ids
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return {}, {}


async def demo_session_restoration():
    """セッション復元のデモ"""
    
    print("=== セッション復元デモ ===\n")
    
    session_manager = SQLAlchemySessionManager(
        db_path="data/demo_session_management.db"
    )
    
    try:
        # 既存のセッションを取得（前のデモで作成されたもの）
        stats = session_manager.get_system_statistics()
        active_session_ids = stats["active_session_ids"]
        
        if not active_session_ids:
            print("   復元可能なセッションがありません。先にセッションを作成してください。")
            return
        
        # 最初のアクティブセッションを選択
        session_id = active_session_ids[0]
        print(f"1. セッション復元テスト: {session_id[:20]}...")
        
        # セッションをメモリから削除（復元テストのため）
        if session_id in session_manager.active_sessions:
            session_info = session_manager.active_sessions[session_id]
            print(f"   セッション '{session_info['session_name']}' をメモリから削除")
            del session_manager.active_sessions[session_id]
        
        # セッション復元
        print("   セッションを復元中...")
        restore_info = await session_manager.restore_session(session_id)
        
        print(f"   復元完了:")
        print(f"     セッションID: {restore_info['session_id'][:20]}...")
        print(f"     ユーザー: {restore_info['username']}")
        print(f"     セッション名: {restore_info['session_name']}")
        print(f"     復元された会話数: {restore_info['restored_conversations']}")
        print(f"     開始時間: {restore_info['start_time']}")
        
        # 復元されたコンテキストの確認
        print("\n2. 復元されたコンテキストの確認")
        context = await session_manager.get_session_context(session_id)
        
        langchain_context = context["langchain_context"]
        recent_conversations = context["recent_conversations"]
        
        print(f"   LangChainメッセージ数: {langchain_context['message_count']}")
        print(f"   最近の会話数: {len(recent_conversations)}")
        
        if recent_conversations:
            print(f"   最新の会話:")
            latest_conv = recent_conversations[0]
            print(f"     ユーザー: {latest_conv['user_input'][:50]}...")
            print(f"     エージェント: {latest_conv['agent_response'][:50]}...")
        
        # セッション状態の確認
        if "session_states" in restore_info and restore_info["session_states"]:
            print(f"\n3. 復元されたセッション状態")
            for state_type, state_data in restore_info["session_states"].items():
                print(f"   {state_type}: {json.dumps(state_data, ensure_ascii=False, indent=2)[:100]}...")
        
        print()
        
    except Exception as e:
        print(f"復元エラー: {e}")


async def demo_integrity_repair():
    """整合性修復のデモ"""
    
    print("=== セッション整合性修復デモ ===\n")
    
    session_manager = SQLAlchemySessionManager(
        db_path="data/demo_session_management.db"
    )
    
    try:
        # アクティブセッションを取得
        stats = session_manager.get_system_statistics()
        active_session_ids = stats["active_session_ids"]
        
        if not active_session_ids:
            print("   修復可能なセッションがありません。")
            return
        
        session_id = active_session_ids[0]
        print(f"1. 整合性チェック対象: {session_id[:20]}...")
        
        # 整合性修復実行
        repair_result = await session_manager.repair_session_integrity(session_id)
        
        print(f"   修復結果:")
        print(f"     修復成功: {repair_result['repair_success']}")
        print(f"     発見された問題数: {len(repair_result['issues_found'])}")
        print(f"     実行された修復数: {len(repair_result['repairs_performed'])}")
        
        if repair_result["issues_found"]:
            print(f"   発見された問題:")
            for issue in repair_result["issues_found"]:
                print(f"     - {issue}")
        
        if repair_result["repairs_performed"]:
            print(f"   実行された修復:")
            for repair in repair_result["repairs_performed"]:
                print(f"     - {repair}")
        
        if not repair_result["issues_found"]:
            print("   問題は見つかりませんでした。セッションは正常です。")
        
        print()
        
    except Exception as e:
        print(f"修復エラー: {e}")


async def demo_advanced_session_features():
    """高度なセッション機能のデモ"""
    
    print("=== 高度なセッション機能デモ ===\n")
    
    session_manager = SQLAlchemySessionManager(
        db_path="data/demo_session_management.db"
    )
    
    try:
        # 1. 長期セッションのシミュレーション
        print("1. 長期セッションシミュレーション")
        
        # 新しいユーザーとセッション作成
        user_id = await session_manager.create_user(
            username="long_session_user",
            email="long@example.com"
        )
        
        session_id = await session_manager.create_session(
            user_id=user_id,
            session_name="長期開発セッション",
            session_type="development"
        )
        
        # 複数の会話を時系列で追加
        conversations_timeline = [
            ("プロジェクト開始", "新しいWebアプリケーションの開発を始めます。"),
            ("要件定義", "ユーザー認証とデータ管理機能が必要です。"),
            ("技術選定", "Python + FastAPI + PostgreSQLを選択しました。"),
            ("開発開始", "基本的なAPIエンドポイントを実装中です。"),
            ("テスト実装", "単体テストとAPIテストを追加しています。"),
            ("デプロイ準備", "Docker化とCI/CDパイプラインを設定中です。"),
            ("本番リリース", "アプリケーションを本番環境にデプロイしました。")
        ]
        
        for i, (user_input, agent_response) in enumerate(conversations_timeline):
            await session_manager.save_conversation(
                session_id=session_id,
                user_input=user_input,
                agent_response=agent_response,
                response_time=1.0 + i * 0.1,
                importance_score=0.7 + i * 0.05,
                metadata={"phase": i + 1, "timeline": True}
            )
            
            # 各段階でセッション状態を更新
            await session_manager.save_session_state(
                session_id=session_id,
                state_type="project_progress",
                state_data={
                    "current_phase": user_input,
                    "completion_percentage": (i + 1) / len(conversations_timeline),
                    "next_steps": agent_response
                }
            )
        
        print(f"   {len(conversations_timeline)} 段階の開発プロセスを記録")
        
        # 2. セッション分析
        print("\n2. セッション分析")
        
        context = await session_manager.get_session_context(session_id)
        session_info = context["session_info"]
        
        print(f"   総会話数: {session_info['conversation_count']}")
        print(f"   総トークン数: {session_info['total_tokens']}")
        
        # 会話の重要度分析
        recent_conversations = context["recent_conversations"]
        if recent_conversations:
            avg_importance = sum(conv["importance_score"] for conv in recent_conversations) / len(recent_conversations)
            print(f"   平均重要度: {avg_importance:.3f}")
            
            high_importance_count = sum(1 for conv in recent_conversations if conv["importance_score"] > 0.8)
            print(f"   高重要度会話数: {high_importance_count}")
        
        # 3. セッション終了
        print("\n3. セッション終了処理")
        
        end_info = await session_manager.end_session(session_id)
        
        print(f"   セッション終了時間: {end_info['end_time']}")
        print(f"   総会話数: {end_info['total_conversations']}")
        print(f"   セッション継続時間: {end_info['duration_minutes']:.1f} 分")
        
        # 4. 最終統計
        print("\n4. 最終システム統計")
        
        final_stats = session_manager.get_system_statistics()
        db_stats = final_stats["database_stats"]
        
        print(f"   システム全体:")
        print(f"     総ユーザー数: {db_stats['total_users']}")
        print(f"     総セッション数: {db_stats['total_sessions']}")
        print(f"     総会話数: {db_stats['total_conversations']}")
        print(f"     現在アクティブなセッション: {final_stats['active_sessions_memory']}")
        
        print()
        
    except Exception as e:
        print(f"高度機能デモエラー: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """メインデモ実行"""
    
    # データディレクトリ作成
    Path("data").mkdir(exist_ok=True)
    
    try:
        # 基本的なセッション管理デモ
        sessions, user_ids = await demo_user_and_session_management()
        
        # セッション復元デモ
        await demo_session_restoration()
        
        # 整合性修復デモ
        await demo_integrity_repair()
        
        # 高度な機能デモ
        await demo_advanced_session_features()
        
        print("=== デモ完了 ===")
        print("SQLAlchemy + LangChain 統合セッション管理システムが正常に動作することを確認しました。")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())