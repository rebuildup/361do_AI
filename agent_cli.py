#!/usr/bin/env python3
"""
Agent CLI
自己学習システムを活用するCLIエージェント
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.core.config import Config
from agent.core.database import DatabaseManager
from agent.core.ollama_client import OllamaClient
from agent.tools.learning_tool import LearningTool
from agent.core.agent_manager import AgentManager


class AgentCLI:
    """CLIエージェントクラス"""
    
    def __init__(self):
        self.config = None
        self.db_manager = None
        self.ollama_client = None
        self.learning_tool = None
        self.agent_manager = None
        self.is_running = False
        
    async def initialize(self):
        """CLIエージェント初期化"""
        print("🤖 自己学習型AIエージェントCLIを初期化中...")
        
        try:
            # 設定とデータベースの初期化
            self.config = Config()
            self.db_manager = DatabaseManager(self.config.database_url)
            await self.db_manager.initialize()
            
            # OLLAMAクライアント初期化
            self.ollama_client = OllamaClient(self.config.ollama_config)
            await self.ollama_client.initialize()
            
            # 学習ツール初期化
            self.learning_tool = LearningTool(
                db_manager=self.db_manager,
                config=self.config,
                ollama_client=self.ollama_client
            )
            
            # エージェントマネージャー初期化
            self.agent_manager = AgentManager(self.config, self.db_manager)
            await self.agent_manager.initialize()
            
            print("✅ 初期化完了")
            return True
            
        except Exception as e:
            print(f"❌ 初期化エラー: {e}")
            return False
    
    async def shutdown(self):
        """CLIエージェント終了処理"""
        print("🔄 システムを終了中...")
        
        if self.learning_tool:
            await self.learning_tool.stop_learning_system()
        
        if self.agent_manager:
            await self.agent_manager.shutdown()
        
        if self.ollama_client:
            await self.ollama_client.close()
        
        if self.db_manager:
            await self.db_manager.close()
        
        print("✅ 終了完了")
    
    def show_help(self):
        """ヘルプ表示"""
        help_text = """
🤖 自己学習型AIエージェントCLI - ヘルプ

📝 基本コマンド:
  chat <メッセージ>     - エージェントとチャット
  help                  - このヘルプを表示
  quit                  - 終了

🧠 学習システムコマンド:
  learn start           - 学習システム開始
  learn stop            - 学習システム停止
  learn status          - 学習システム状態確認
  learn cycle           - 手動で学習サイクル実行
  
📚 学習データ管理:
  data add <カテゴリ> <内容>  - 学習データ追加
  data list [カテゴリ]        - 学習データ一覧表示
  data update <ID> <内容>     - 学習データ更新
  data delete <ID>            - 学習データ削除
  data stats                - 学習統計表示
  data export [json/csv]     - 学習データエクスポート
  data import <ファイル>      - 学習データインポート
  
🔧 プロンプト管理:
  prompt list              - プロンプト一覧表示
  prompt add <名前> <内容>   - カスタムプロンプト追加
  prompt update <名前> <内容> - プロンプト更新
  prompt delete <名前>      - プロンプト削除
  prompt optimize <名前>    - プロンプト最適化
  prompt export            - プロンプトエクスポート
  prompt import <ファイル>  - プロンプトインポート
  
💬 会話テスト:
  test conversation         - 会話テストモード
  test learning            - 学習機能テスト
  
📊 システム情報:
  status                   - システム状態確認
  stats                    - 統計情報表示
  report                   - 詳細レポート生成

例:
  chat こんにちは
  learn start
  data add conversation_rules "常に丁寧に応答する"
  prompt add greeting "こんにちは！何かお手伝いできることはありますか？"
  test conversation
        """
        print(help_text)
    
    async def handle_chat(self, message: str):
        """チャット処理"""
        try:
            print(f"👤 あなた: {message}")
            print("🤖 エージェント: 考え中...")
            
            response = await self.agent_manager.process_message(message)
            
            print(f"🤖 エージェント: {response['response']}")
            
            if response.get('intent'):
                intent = response['intent']
                print(f"📊 検出された意図: {intent.get('primary_intent', 'unknown')}")
            
            if response.get('tools_used'):
                print(f"🔧 使用ツール: {', '.join(response['tools_used'])}")
            
            print(f"⏱️  応答時間: {response.get('response_time', 0):.2f}秒")
            
        except Exception as e:
            print(f"❌ チャットエラー: {e}")
    
    async def handle_learning_command(self, subcommand: str):
        """学習システムコマンド処理"""
        try:
            if subcommand == "start":
                result = await self.learning_tool.start_learning_system()
                print(f"✅ 学習システム開始: {result.get('message', '')}")
                
            elif subcommand == "stop":
                result = await self.learning_tool.stop_learning_system()
                print(f"✅ 学習システム停止: {result.get('message', '')}")
                
            elif subcommand == "status":
                result = await self.learning_tool.get_learning_status()
                if result.get('status') == 'success':
                    status_data = result.get('data', {})
                    print("📊 学習システム状態:")
                    print(f"  実行中: {status_data.get('is_running', False)}")
                    print(f"  アクティブタスク: {len(status_data.get('active_tasks', []))}")
                else:
                    print(f"❌ 状態取得エラー: {result.get('message', '')}")
                    
            elif subcommand == "cycle":
                result = await self.learning_tool.manually_trigger_learning_cycle()
                print(f"✅ 学習サイクル実行: {result.get('message', '')}")
                
            else:
                print("❌ 不明な学習コマンド。'learn start/stop/status/cycle' を使用してください。")
                
        except Exception as e:
            print(f"❌ 学習コマンドエラー: {e}")
    
    async def handle_data_command(self, subcommand: str, *args):
        """学習データコマンド処理"""
        try:
            if subcommand == "add":
                if len(args) < 2:
                    print("❌ 使用方法: data add <カテゴリ> <内容>")
                    return
                
                category = args[0]
                content = " ".join(args[1:])
                
                result = await self.learning_tool.add_custom_learning_data(
                    content=content,
                    category=category,
                    tags=["cli_added"]
                )
                
                if result.get('status') == 'success':
                    print(f"✅ 学習データ追加完了: {result.get('data_id', '')}")
                else:
                    print(f"❌ 追加エラー: {result.get('message', '')}")
                    
            elif subcommand == "list":
                category = args[0] if args else None
                
                result = await self.learning_tool.get_learning_data(
                    category=category,
                    limit=10
                )
                
                if result.get('status') == 'success':
                    data = result.get('data', [])
                    print(f"📚 学習データ一覧 ({len(data)}件):")
                    for i, item in enumerate(data, 1):
                        print(f"  {i}. [{item.get('category', '')}] {item.get('content', '')[:50]}...")
                else:
                    print(f"❌ 取得エラー: {result.get('message', '')}")
                    
            elif subcommand == "update":
                if len(args) < 2:
                    print("❌ 使用方法: data update <ID> <内容>")
                    return
                
                data_id = args[0]
                content = " ".join(args[1:])
                
                result = await self.learning_tool.update_learning_data(
                    data_id=data_id,
                    content=content
                )
                
                if result.get('status') == 'success':
                    print(f"✅ 学習データ更新完了: {data_id}")
                else:
                    print(f"❌ 更新エラー: {result.get('message', '')}")
                    
            elif subcommand == "delete":
                if not args:
                    print("❌ 使用方法: data delete <ID>")
                    return
                
                data_id = args[0]
                result = await self.learning_tool.delete_learning_data(data_id)
                
                if result.get('status') == 'success':
                    print(f"✅ 学習データ削除完了: {data_id}")
                else:
                    print(f"❌ 削除エラー: {result.get('message', '')}")
                    
            elif subcommand == "stats":
                stats = await self.db_manager.get_learning_statistics()
                print("📊 学習統計:")
                print(f"  学習データ総数: {stats.get('total_learning_data', 0)}")
                print(f"  知識アイテム総数: {stats.get('total_knowledge_items', 0)}")
                print(f"  平均品質スコア: {stats.get('average_quality_score', 0):.2f}")
                print(f"  高品質データ数: {stats.get('high_quality_count', 0)}")
                
            elif subcommand == "export":
                format_type = args[0] if args else "json"
                result = await self.learning_tool.export_learning_data(format_type)
                
                if result.get('status') == 'success':
                    print(f"✅ 学習データエクスポート完了: {format_type}形式")
                    print(f"  学習データ: {result.get('data', {}).get('total_learning_items', 0)}件")
                    print(f"  知識アイテム: {result.get('data', {}).get('total_knowledge_items', 0)}件")
                else:
                    print(f"❌ エクスポートエラー: {result.get('message', '')}")
                    
            elif subcommand == "import":
                if not args:
                    print("❌ 使用方法: data import <ファイルパス>")
                    return
                
                file_path = args[0]
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        import_data = json.load(f)
                    
                    result = await self.learning_tool.import_learning_data(import_data)
                    
                    if result.get('status') == 'success':
                        print(f"✅ 学習データインポート完了: {result.get('imported_count', 0)}件")
                    else:
                        print(f"❌ インポートエラー: {result.get('message', '')}")
                except Exception as e:
                    print(f"❌ ファイル読み込みエラー: {e}")
                
            else:
                print("❌ 不明なデータコマンド。'data add/list/update/delete/stats/export/import' を使用してください。")
                
        except Exception as e:
            print(f"❌ データコマンドエラー: {e}")
    
    async def handle_prompt_command(self, subcommand: str, *args):
        """プロンプトコマンド処理"""
        try:
            if subcommand == "list":
                result = await self.learning_tool.get_prompt_templates()
                if result.get('status') == 'success':
                    prompts = result.get('data', [])
                    print(f"📝 プロンプト一覧 ({len(prompts)}件):")
                    for i, prompt in enumerate(prompts, 1):
                        print(f"  {i}. {prompt.get('name', '')} - {prompt.get('description', '')[:50]}...")
                else:
                    print(f"❌ 取得エラー: {result.get('message', '')}")
                    
            elif subcommand == "add":
                if len(args) < 2:
                    print("❌ 使用方法: prompt add <名前> <内容>")
                    return
                
                name = args[0]
                content = " ".join(args[1:])
                
                result = await self.learning_tool.add_prompt_template(
                    name=name,
                    content=content,
                    description=f"カスタムプロンプト: {name}"
                )
                
                if result.get('status') == 'success':
                    print(f"✅ プロンプト追加完了: {name}")
                else:
                    print(f"❌ 追加エラー: {result.get('message', '')}")
                    
            elif subcommand == "update":
                if len(args) < 2:
                    print("❌ 使用方法: prompt update <名前> <内容>")
                    return
                
                name = args[0]
                content = " ".join(args[1:])
                
                result = await self.learning_tool.update_prompt_template(
                    name=name,
                    content=content
                )
                
                if result.get('status') == 'success':
                    print(f"✅ プロンプト更新完了: {name}")
                else:
                    print(f"❌ 更新エラー: {result.get('message', '')}")
                    
            elif subcommand == "delete":
                if not args:
                    print("❌ 使用方法: prompt delete <名前>")
                    return
                
                name = args[0]
                result = await self.learning_tool.delete_prompt_template(name)
                
                if result.get('status') == 'success':
                    print(f"✅ プロンプト削除完了: {name}")
                else:
                    print(f"❌ 削除エラー: {result.get('message', '')}")
                    
            elif subcommand == "optimize":
                if not args:
                    print("❌ 使用方法: prompt optimize <名前>")
                    return
                
                name = args[0]
                result = await self.learning_tool.optimize_prompt_template(name)
                
                if result.get('status') == 'success':
                    print(f"✅ プロンプト最適化完了: {name}")
                    print(f"  改善スコア: {result.get('improvement_score', 0):.2f}")
                else:
                    print(f"❌ 最適化エラー: {result.get('message', '')}")
                    
            elif subcommand == "export":
                result = await self.learning_tool.export_prompt_templates()
                
                if result.get('status') == 'success':
                    print(f"✅ プロンプトエクスポート完了")
                    print(f"  プロンプト数: {result.get('count', 0)}件")
                else:
                    print(f"❌ エクスポートエラー: {result.get('message', '')}")
                    
            elif subcommand == "import":
                if not args:
                    print("❌ 使用方法: prompt import <ファイルパス>")
                    return
                
                file_path = args[0]
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        import_data = json.load(f)
                    
                    result = await self.learning_tool.import_prompt_templates(import_data)
                    
                    if result.get('status') == 'success':
                        print(f"✅ プロンプトインポート完了: {result.get('imported_count', 0)}件")
                    else:
                        print(f"❌ インポートエラー: {result.get('message', '')}")
                except Exception as e:
                    print(f"❌ ファイル読み込みエラー: {e}")
                
            else:
                print("❌ 不明なプロンプトコマンド。'prompt list/add/update/delete/optimize/export/import' を使用してください。")
                
        except Exception as e:
            print(f"❌ プロンプトコマンドエラー: {e}")
    
    async def handle_test_command(self, subcommand: str):
        """テストコマンド処理"""
        try:
            if subcommand == "conversation":
                print("💬 会話テストモード開始 (終了するには 'quit' と入力)")
                print("学習されたルールとプロンプトが適用されるかテストできます")
                
                while True:
                    try:
                        user_input = input("\n👤 あなた: ").strip()
                        if user_input.lower() in ['quit', 'exit', '終了']:
                            break
                        
                        await self.handle_chat(user_input)
                        
                    except KeyboardInterrupt:
                        break
                    except Exception as e:
                        print(f"❌ エラー: {e}")
                        
            elif subcommand == "learning":
                print("🧪 学習機能テスト開始...")
                
                # 学習システムを開始
                await self.learning_tool.start_learning_system()
                print("✅ 学習システムを開始しました")
                
                # テストデータを追加
                test_data = [
                    ("conversation_rules", "常に丁寧で親切に応答する"),
                    ("knowledge_base", "Pythonは動的型付け言語です"),
                    ("prompt_templates", "システムプロンプトの基本テンプレート")
                ]
                
                for category, content in test_data:
                    await self.learning_tool.add_custom_learning_data(
                        content=content,
                        category=category,
                        tags=["test_data"]
                    )
                    print(f"✅ テストデータ追加: {category}")
                
                # 学習サイクルを実行
                await self.learning_tool.manually_trigger_learning_cycle()
                print("✅ 学習サイクルを実行しました")
                
                # テスト会話
                test_messages = [
                    "こんにちは",
                    "自己紹介をしてください",
                    "Pythonについて教えてください"
                ]
                
                for message in test_messages:
                    print(f"\n--- テスト会話: {message} ---")
                    await self.handle_chat(message)
                    await asyncio.sleep(1)
                    
            else:
                print("❌ 不明なテストコマンド。'test conversation/learning' を使用してください。")
                
        except Exception as e:
            print(f"❌ テストコマンドエラー: {e}")
    
    async def handle_status_command(self):
        """システム状態確認"""
        try:
            print("📊 システム状態:")
            
            # OLLAMA状態
            try:
                ollama_status = await self.ollama_client.health_check()
                print(f"  🤖 OLLAMA: {'✅ 正常' if ollama_status.get('status') == 'ok' else '❌ 異常'}")
            except:
                print("  🤖 OLLAMA: ❌ 接続エラー")
            
            # データベース状態
            try:
                stats = await self.db_manager.get_learning_statistics()
                print(f"  💾 データベース: ✅ 正常 (学習データ: {stats.get('total_learning_data', 0)}件)")
            except:
                print("  💾 データベース: ❌ エラー")
            
            # 学習システム状態
            try:
                result = await self.learning_tool.get_learning_status()
                if result.get('status') == 'success':
                    status_data = result.get('data', {})
                    print(f"  🧠 学習システム: {'✅ 実行中' if status_data.get('is_running') else '⏸️ 停止中'}")
                else:
                    print("  🧠 学習システム: ❌ エラー")
            except:
                print("  🧠 学習システム: ❌ エラー")
                
        except Exception as e:
            print(f"❌ 状態確認エラー: {e}")
    
    async def handle_report_command(self):
        """詳細レポート生成"""
        try:
            print("📋 詳細レポート生成中...")
            
            result = await self.learning_tool.get_performance_report(days=7)
            
            if result.get('status') == 'success':
                report = result.get('report', {})
                
                print("\n📊 パフォーマンスレポート (過去7日間):")
                print(f"  学習データ統計: {report.get('learning_stats', {})}")
                print(f"  知識ベース統計: {report.get('knowledge_stats', {})}")
                print(f"  プロンプト最適化統計: {report.get('prompt_optimization_stats', {})}")
                
                print("\n📈 パフォーマンス指標:")
                metrics = report.get('performance_metrics', {})
                for key, value in metrics.items():
                    print(f"  {key}: {value}")
                    
            else:
                print(f"❌ レポート生成エラー: {result.get('message', '')}")
                
        except Exception as e:
            print(f"❌ レポート生成エラー: {e}")
    
    async def run(self):
        """CLIメインループ"""
        print("🤖 自己学習型AIエージェントCLI")
        print("入力してください (help でヘルプ表示):")
        
        self.is_running = True
        
        while self.is_running:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                parts = user_input.split()
                command = parts[0].lower()
                args = parts[1:] if len(parts) > 1 else []
                
                if command in ['quit', 'exit', '終了']:
                    self.is_running = False
                    break
                    
                elif command == 'help':
                    self.show_help()
                    
                elif command == 'chat':
                    if not args:
                        print("❌ 使用方法: chat <メッセージ>")
                        continue
                    message = " ".join(args)
                    await self.handle_chat(message)
                    
                elif command == 'learn':
                    if not args:
                        print("❌ 使用方法: learn <start/stop/status/cycle>")
                        continue
                    await self.handle_learning_command(args[0])
                    
                elif command == 'data':
                    if not args:
                        print("❌ 使用方法: data <add/list/update/delete/stats/export/import>")
                        continue
                    await self.handle_data_command(args[0], *args[1:])
                    
                elif command == 'prompt':
                    if not args:
                        print("❌ 使用方法: prompt <list/add/update/delete/optimize/export/import>")
                        continue
                    await self.handle_prompt_command(args[0], *args[1:])
                    
                elif command == 'test':
                    if not args:
                        print("❌ 使用方法: test <conversation/learning>")
                        continue
                    await self.handle_test_command(args[0])
                    
                elif command == 'status':
                    await self.handle_status_command()
                    
                elif command == 'stats':
                    await self.handle_data_command('stats')
                    
                elif command == 'report':
                    await self.handle_report_command()
                    
                else:
                    print(f"❌ 不明なコマンド: {command}")
                    print("'help' でヘルプを表示してください")
                    
            except KeyboardInterrupt:
                print("\n🔄 終了しますか？ (y/N): ", end="")
                try:
                    response = input().strip().lower()
                    if response in ['y', 'yes', 'はい']:
                        self.is_running = False
                except:
                    self.is_running = False
                    
            except Exception as e:
                print(f"❌ エラー: {e}")


async def main():
    """メイン関数"""
    cli = AgentCLI()
    
    try:
        if await cli.initialize():
            await cli.run()
        else:
            print("❌ 初期化に失敗しました")
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
    finally:
        await cli.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
