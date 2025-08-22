#!/usr/bin/env python3
"""
Simple Self-Learning Test
自己学習システムの簡単なテスト
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.core.config import Config
from agent.core.database import DatabaseManager
from agent.core.ollama_client import OllamaClient
from agent.tools.learning_tool import LearningTool
from agent.core.agent_manager import AgentManager


async def _test_learning_system_async():
    """学習システムのテスト"""
    print("🧪 自己学習システムテスト開始")
    
    try:
        # 設定とデータベースの初期化
        config = Config()
        db_manager = DatabaseManager(config.database_url)
        await db_manager.initialize()
        
        # OLLAMAクライアント初期化
        ollama_client = OllamaClient(config.ollama_config)
        await ollama_client.initialize()
        
        # 学習ツール初期化
        learning_tool = LearningTool(
            db_manager=db_manager,
            config=config,
            ollama_client=ollama_client
        )
        
        # エージェントマネージャー初期化
        agent_manager = AgentManager(config, db_manager)
        await agent_manager.initialize()
        
        print("✅ 初期化完了")
        
        # テスト1: 学習データの追加
        print("\n📚 テスト1: 学習データの追加")
        test_data = [
            {
                "content": "Pythonは動的型付け言語で、読みやすく書きやすい言語です。",
                "category": "programming",
                "tags": ["python", "programming", "language"]
            },
            {
                "content": "常に丁寧で親切な口調で応答することが重要です。",
                "category": "conversation_rules",
                "tags": ["conversation", "politeness", "user_experience"]
            },
            {
                "content": "Webデザインではユーザビリティとアクセシビリティを重視すべきです。",
                "category": "web_design",
                "tags": ["web_design", "usability", "accessibility"]
            }
        ]
        
        for data in test_data:
            result = await learning_tool.add_custom_learning_data(
                content=data["content"],
                category=data["category"],
                tags=data["tags"]
            )
            print(f"  ✅ {data['category']}: {result.get('status', 'error')}")
        
        # テスト2: プロンプトテンプレートの追加
        print("\n📝 テスト2: プロンプトテンプレートの追加")
        test_prompts = [
            {
                "name": "greeting_prompt",
                "content": "こんにちは！何かお手伝いできることはありますか？",
                "description": "挨拶用のプロンプト"
            },
            {
                "name": "technical_help_prompt",
                "content": "技術的な質問ですね。詳しく説明させていただきます。",
                "description": "技術サポート用のプロンプト"
            }
        ]
        
        for prompt in test_prompts:
            result = await learning_tool.add_prompt_template(
                name=prompt["name"],
                content=prompt["content"],
                description=prompt["description"]
            )
            print(f"  ✅ {prompt['name']}: {result.get('status', 'error')}")
        
        # テスト3: 学習データの取得
        print("\n📖 テスト3: 学習データの取得")
        learning_data = await learning_tool.get_learning_data(limit=5)
        if learning_data.get('status') == 'success':
            data = learning_data.get('data', [])
            print(f"  📊 取得件数: {len(data)}件")
            for i, item in enumerate(data[:3], 1):
                print(f"    {i}. [{item.get('category', '')}] {item.get('content', '')[:50]}...")
        else:
            print(f"  ❌ 取得エラー: {learning_data.get('message', '')}")
        
        # テスト4: プロンプトテンプレートの取得
        print("\n📋 テスト4: プロンプトテンプレートの取得")
        prompt_templates = await learning_tool.get_prompt_templates()
        if prompt_templates.get('status') == 'success':
            templates = prompt_templates.get('data', [])
            print(f"  📊 テンプレート数: {len(templates)}件")
            for i, template in enumerate(templates[:3], 1):
                print(f"    {i}. {template.get('name', '')} - {template.get('description', '')[:30]}...")
        else:
            print(f"  ❌ 取得エラー: {prompt_templates.get('message', '')}")
        
        # テスト5: 学習システムの開始
        print("\n🚀 テスト5: 学習システムの開始")
        start_result = await learning_tool.start_learning_system()
        print(f"  📊 開始結果: {start_result.get('status', 'error')}")
        
        # テスト6: 手動学習サイクルの実行
        print("\n🔄 テスト6: 手動学習サイクルの実行")
        cycle_result = await learning_tool.manually_trigger_learning_cycle()
        print(f"  📊 サイクル実行結果: {cycle_result.get('status', 'error')}")
        
        # テスト7: エージェントとの会話テスト
        print("\n💬 テスト7: エージェントとの会話テスト")
        test_messages = [
            "こんにちは",
            "Pythonについて教えてください",
            "Webデザインのコツを教えてください"
        ]
        
        for message in test_messages:
            print(f"\n👤 ユーザー: {message}")
            response = await agent_manager.process_message(message)
            print(f"🤖 エージェント: {response.get('response', 'エラー')[:100]}...")
            print(f"   ⏱️ 応答時間: {response.get('response_time', 0):.2f}秒")
            print(f"   🎯 意図: {response.get('intent', {}).get('primary_intent', 'unknown')}")

            # Insert a self-edit quick test in the loop for the last message
            if message == test_messages[-1]:
                print("\n🔧 テスト: エージェントによる自己編集 (ファイル書き込み/読み取り)")
                # instruct the agent to write a file
                write_cmd = "write file src/data/prompts/test_agent_written.txt\nThis file was written by the agent for testing."
                write_resp = await agent_manager.process_message(write_cmd)
                print(f"  ✍️ 書き込みコマンド応答: {write_resp.get('response', '')}")

                # read back the file
                read_cmd = "read file src/data/prompts/test_agent_written.txt"
                read_resp = await agent_manager.process_message(read_cmd)
                print(f"  📖 読み取り結果先頭: {read_resp.get('response', '')[:80]}")

                # update a prompt template via self-edit
                print("\n🔧 テスト: プロンプト更新 (update prompt)")
                update_cmd = "update prompt greeting_prompt: こんにちは、私はエージェントによって更新されたプロンプトです。"
                update_resp = await agent_manager.process_message(update_cmd)
                print(f"  🔁 プロンプト更新応答: {update_resp.get('response', '')}")

                # add learning data via self-edit
                print("\n🔧 テスト: 学習データ追加 (add learning data)")
                add_learning_cmd = "add learning data: {\"content\": \"Agent added this learning item for test.\", \"category\": \"unit_test\", \"tags\": [\"agent\", \"test\"]}"
                add_learning_resp = await agent_manager.process_message(add_learning_cmd)
                print(f"  ➕ 学習データ追加応答: {add_learning_resp.get('response', '')}")
        
        # テスト8: 学習データのエクスポート
        print("\n📤 テスト8: 学習データのエクスポート")
        export_result = await learning_tool.export_learning_data("json")
        if export_result.get('status') == 'success':
            data = export_result.get('data', {})
            print(f"  📊 学習データ: {data.get('total_learning_items', 0)}件")
            print(f"  📊 知識アイテム: {data.get('total_knowledge_items', 0)}件")
        else:
            print(f"  ❌ エクスポートエラー: {export_result.get('message', '')}")
        
        # テスト9: プロンプトテンプレートのエクスポート
        print("\n📤 テスト9: プロンプトテンプレートのエクスポート")
        prompt_export_result = await learning_tool.export_prompt_templates()
        if prompt_export_result.get('status') == 'success':
            print(f"  📊 プロンプト数: {prompt_export_result.get('count', 0)}件")
        else:
            print(f"  ❌ エクスポートエラー: {prompt_export_result.get('message', '')}")
        
        # テスト10: パフォーマンスレポートの取得
        print("\n📊 テスト10: パフォーマンスレポートの取得")
        report_result = await learning_tool.get_performance_report(days=1)
        if report_result.get('status') == 'success':
            report = report_result.get('report', {})
            print(f"  📊 学習統計: {report.get('learning_stats', {})}")
            print(f"  📊 知識ベース統計: {report.get('knowledge_stats', {})}")
        else:
            print(f"  ❌ レポート取得エラー: {report_result.get('message', '')}")
        
        # テスト11: 学習システムの停止
        print("\n⏹️ テスト11: 学習システムの停止")
        stop_result = await learning_tool.stop_learning_system()
        print(f"  📊 停止結果: {stop_result.get('status', 'error')}")
        
        # テスト12: 統計情報の表示
        print("\n📈 テスト12: 統計情報の表示")
        stats = await db_manager.get_learning_statistics()
        print(f"  📊 学習データ総数: {stats.get('total_learning_data', 0)}件")
        print(f"  📊 知識アイテム総数: {stats.get('total_knowledge_items', 0)}件")
        print(f"  📊 平均品質スコア: {stats.get('average_quality_score', 0):.2f}")
        print(f"  📊 高品質データ数: {stats.get('high_quality_count', 0)}件")
        
        print("\n✅ 全てのテストが完了しました！")
        
        # クリーンアップ
        await agent_manager.shutdown()
        await ollama_client.close()
        await db_manager.close()
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()


async def _test_prompt_optimization_async():
    """プロンプト最適化のテスト"""
    print("\n🔧 プロンプト最適化テスト開始")
    
    try:
        # 設定とデータベースの初期化
        config = Config()
        db_manager = DatabaseManager(config.database_url)
        await db_manager.initialize()
        
        # OLLAMAクライアント初期化
        ollama_client = OllamaClient(config.ollama_config)
        await ollama_client.initialize()
        
        # 学習ツール初期化
        learning_tool = LearningTool(
            db_manager=db_manager,
            config=config,
            ollama_client=ollama_client
        )
        
        # テスト用プロンプトを追加
        test_prompt_name = "test_optimization_prompt"
        test_prompt_content = "ユーザーの質問に答えてください。"
        
        await learning_tool.add_prompt_template(
            name=test_prompt_name,
            content=test_prompt_content,
            description="最適化テスト用のプロンプト"
        )
        
        print(f"✅ テストプロンプト追加: {test_prompt_name}")
        
        # プロンプト最適化のテスト
        print(f"🔄 プロンプト最適化実行: {test_prompt_name}")
        optimization_result = await learning_tool.optimize_prompt_template(test_prompt_name)
        
        if optimization_result.get('status') == 'success':
            print(f"✅ 最適化成功")
            print(f"   📊 改善スコア: {optimization_result.get('improvement_score', 0):.2f}")
        else:
            print(f"❌ 最適化エラー: {optimization_result.get('message', '')}")
        
        # クリーンアップ
        await ollama_client.close()
        await db_manager.close()
        
    except Exception as e:
        print(f"❌ プロンプト最適化テストエラー: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """メイン関数"""
    print("🤖 自己学習型AIエージェント - テスト実行")
    print("=" * 50)
    
    # 基本学習システムテスト
    await _test_learning_system_async()

    # プロンプト最適化テスト
    await _test_prompt_optimization_async()
    
    print("\n🎉 全てのテストが完了しました！")


if __name__ == "__main__":
    asyncio.run(main())


# Pytest-friendly synchronous wrappers
def test_learning_system():
    """Synchronously run the async learning system test for pytest."""
    asyncio.run(_test_learning_system_async())


def test_prompt_optimization():
    """Synchronously run the async prompt optimization test for pytest."""
    asyncio.run(_test_prompt_optimization_async())
