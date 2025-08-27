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
        # 設定とデータベースの初期化
        self.config = Config()
        
        # エージェントモードの表示
        print(f"[DEBUG] USE_CODEX_AGENT setting: {self.config.settings.use_codex_agent}")
        print(f"[DEBUG] ENABLE_WEB_SEARCH setting: {self.config.settings.enable_web_search}")
        print(f"[DEBUG] ENABLE_COMMAND_EXECUTION setting: {self.config.settings.enable_command_execution}")
        
        if self.config.is_codex_agent_enabled:
            print("[AI] Codex互換AIエージェントCLIを初期化中...")
        else:
            print("[AI] 自己学習型AIエージェントCLIを初期化中...")
        
        try:
            self.db_manager = DatabaseManager(self.config.database_url)
            await self.db_manager.initialize()
            
            # Codex互換エージェントを使用する場合は、シンプルな初期化
            if self.config.is_codex_agent_enabled:
                # エージェントマネージャーのみ初期化（内部でCodexエージェントが初期化される）
                self.agent_manager = AgentManager(self.config, self.db_manager)
                await self.agent_manager.initialize()
                print("[OK] Codex互換エージェント初期化完了")
                return True
            
            # 従来の複雑なシステム初期化
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
            
            print("[OK] 初期化完了")
            return True
            
        except Exception as e:
            print(f"[ERROR] 初期化エラー: {e}")
            return False
    
    async def shutdown(self):
        """CLIエージェント終了処理"""
        print("[INFO] システムを終了中...")
        
        if self.learning_tool:
            await self.learning_tool.stop_learning_system()
        
        if self.agent_manager:
            await self.agent_manager.shutdown()
        
        if self.ollama_client:
            await self.ollama_client.close()
        
        if self.db_manager:
            await self.db_manager.close()
        
        print("[OK] 終了完了")
    
    def show_help(self):
        """ヘルプ表示"""
        if self.config and self.config.is_codex_agent_enabled:
            help_text = """
[AI] Codex互換AIエージェントCLI - ヘルプ

[CMD] 基本コマンド:
  chat <メッセージ>     - エージェントとチャット
  help                  - このヘルプを表示
  quit                  - 終了
  status                - システム状態確認

[MODE] エージェントモード:
  現在: Codex互換モード (シンプル・安定)
  
  Codex互換モードでは以下の機能を提供:
  - 高速なコード補完
  - 安定したチャット機能
  - OLLAMA統合
  
  従来の学習システムを使用するには:
  環境変数 AGENT_USE_CODEX_AGENT=false を設定

例:
  chat こんにちは
  chat "Pythonでファイルを読み込む方法を教えて"
  status
        """
        else:
            help_text = """
[AI] 自己学習型AIエージェントCLI - ヘルプ

[CMD] 基本コマンド:
  chat <メッセージ>     - エージェントとチャット
  help                  - このヘルプを表示
  quit                  - 終了

[LEARN] 学習システムコマンド:
  learn start           - 学習システム開始
  learn stop            - 学習システム停止
  learn status          - 学習システム状態確認
  learn cycle           - 手動で学習サイクル実行
  
[DATA] 学習データ管理:
  data add <カテゴリ> <内容>  -学習データ追加
  data list [カテゴリ]        -学習データ一覧表示
  data update <ID> <内容>     -学習データ更新
  data delete <ID>            -学習データ削除
  data stats                -学習統計表示
  data export [json/csv]     -学習データエクスポート
  data import <ファイル>      -学習データインポート
  
[PROMPT] プロンプト管理:
  prompt list              - プロンプト一覧表示
  prompt add <名前> <内容>   - カスタムプロンプト追加
  prompt update <名前> <内容> - プロンプト更新
  prompt delete <名前>      - プロンプト削除
  prompt optimize <名前>    - プロンプト最適化
  prompt export            - プロンプトエクスポート
  prompt import <ファイル>  - プロンプトインポート
  
[TEST] 会話テスト:
  test conversation         - 会話テストモード
  test learning            - 学習機能テスト
  
[SYS] システム情報:
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
            print(f"[YOU] あなた: {message}")
            print("[AI] エージェント: 考え中...")
            
            response = await self.agent_manager.process_message(message)
            
            print(f"[AI] エージェント: {response['response']}")
            
            # Codex互換モードの場合
            if response.get('agent_type') == 'codex':
                if response.get('model_info'):
                    model_info = response['model_info']
                    print(f"[MODEL] {model_info.get('model', 'unknown')}")
                
                if response.get('usage'):
                    usage = response['usage']
                    if 'total_tokens' in usage:
                        print(f"[TOKENS] {usage['total_tokens']} tokens")
            
            # 従来モードの場合
            else:
                if response.get('intent'):
                    intent = response['intent']
                    print(f"[SYS] 検出された意図: {intent.get('primary_intent', 'unknown')}")
                
                if response.get('tools_used'):
                    print(f"[PROMPT] 使用ツール: {', '.join(response['tools_used'])}")
            
            print(f"[TIME] 応答時間: {response.get('response_time', 0):.2f}秒")
            
        except Exception as e:
            print(f"[ERROR] チャットエラー: {e}")
    
    async def handle_learning_command(self, subcommand: str):
        """学習システムコマンド処理"""
        # Codex互換モードでは学習コマンドは無効
        if self.config.is_codex_agent_enabled:
            print("[INFO] Codex互換モードでは学習システムコマンドは利用できません。")
            print("[INFO] 従来の学習システムを使用するには、環境変数 AGENT_USE_CODEX_AGENT=false を設定してください。")
            return
        
        try:
            if subcommand == "start":
                result = await self.learning_tool.start_learning_system()
                print(f"[OK] 学習システム開始: {result.get('message', '')}")
                
            elif subcommand == "stop":
                result = await self.learning_tool.stop_learning_system()
                print(f"[OK] 学習システム停止: {result.get('message', '')}")
                
            elif subcommand == "status":
                result = await self.learning_tool.get_learning_status()
                if result.get('status') == 'success':
                    status_data = result.get('data', {})
                    print("[SYS] 学習システム状態:")
                    print(f"  実行中: {status_data.get('is_running', False)}")
                    print(f"  アクティブタスク: {len(status_data.get('active_tasks', []))}")
                else:
                    print(f"[ERROR] 状態取得エラー: {result.get('message', '')}")
                    
            elif subcommand == "cycle":
                result = await self.learning_tool.manually_trigger_learning_cycle()
                print(f"[OK] 学習サイクル実行: {result.get('message', '')}")
                
            else:
                print("[ERROR] 不明な学習コマンド。'learn start/stop/status/cycle' を使用してください。")
                
        except Exception as e:
            print(f"[ERROR] 学習コマンドエラー: {e}")
    
    async def handle_data_command(self, subcommand: str, *args):
        """学習データコマンド処理"""
        try:
            if subcommand == "add":
                if len(args) < 2:
                    print("[ERROR] 使用方法: data add <カテゴリ> <内容>")
                    return
                
                category = args[0]
                content = " ".join(args[1:])
                
                result = await self.learning_tool.add_custom_learning_data(
                    content=content,
                    category=category,
                    tags=["cli_added"]
                )
                
                if result.get('status') == 'success':
                    print(f"[OK] 学習データ追加完了: {result.get('data_id', '')}")
                else:
                    print(f"[ERROR] 追加エラー: {result.get('message', '')}")
                    
            elif subcommand == "list":
                category = args[0] if args else None
                
                result = await self.learning_tool.get_learning_data(
                    category=category,
                    limit=10
                )
                
                if result.get('status') == 'success':
                    data = result.get('data', [])
                    print(f"[DATA] 学習データ一覧 ({len(data)}件):")
                    for i, item in enumerate(data, 1):
                        print(f"  {i}. [{item.get('category', '')}] {item.get('content', '')[:50]}...")
                else:
                    print(f"[ERROR] 取得エラー: {result.get('message', '')}")
                    
            elif subcommand == "update":
                if len(args) < 2:
                    print("[ERROR] 使用方法: data update <ID> <内容>")
                    return
                
                data_id = args[0]
                content = " ".join(args[1:])
                
                result = await self.learning_tool.update_learning_data(
                    data_id=data_id,
                    content=content
                )
                
                if result.get('status') == 'success':
                    print(f"[OK] 学習データ更新完了: {data_id}")
                else:
                    print(f"[ERROR] 更新エラー: {result.get('message', '')}")
                    
            elif subcommand == "delete":
                if not args:
                    print("[ERROR] 使用方法: data delete <ID>")
                    return
                
                data_id = args[0]
                result = await self.learning_tool.delete_learning_data(data_id)
                
                if result.get('status') == 'success':
                    print(f"[OK] 学習データ削除完了: {data_id}")
                else:
                    print(f"[ERROR] 削除エラー: {result.get('message', '')}")
                    
            elif subcommand == "stats":
                stats = await self.db_manager.get_learning_statistics()
                print("[SYS] 学習統計:")
                print(f"  学習データ総数: {stats.get('total_learning_data', 0)}")
                print(f"  知識アイテム総数: {stats.get('total_knowledge_items', 0)}")
                print(f"  平均品質スコア: {stats.get('average_quality_score', 0):.2f}")
                print(f"  高品質データ数: {stats.get('high_quality_count', 0)}")
                
            elif subcommand == "export":
                format_type = args[0] if args else "json"
                result = await self.learning_tool.export_learning_data(format_type)
                
                if result.get('status') == 'success':
                    print(f"[OK] 学習データエクスポート完了: {format_type}形式")
                    print(f"  学習データ: {result.get('data', {}).get('total_learning_items', 0)}件")
                    print(f"  知識アイテム: {result.get('data', {}).get('total_knowledge_items', 0)}件")
                else:
                    print(f"[ERROR] エクスポートエラー: {result.get('message', '')}")
                    
            elif subcommand == "import":
                if not args:
                    print("[ERROR] 使用方法: data import <ファイルパス>")
                    return
                
                file_path = args[0]
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        import_data = json.load(f)
                    
                    result = await self.learning_tool.import_learning_data(import_data)
                    
                    if result.get('status') == 'success':
                        print(f"[OK] 学習データインポート完了: {result.get('imported_count', 0)}件")
                    else:
                        print(f"[ERROR] インポートエラー: {result.get('message', '')}")
                except Exception as e:
                    print(f"[ERROR] ファイル読み込みエラー: {e}")
                
            else:
                print("[ERROR] 不明なデータコマンド。'data add/list/update/delete/stats/export/import' を使用してください。")
                
        except Exception as e:
            print(f"[ERROR] データコマンドエラー: {e}")
    
    async def handle_prompt_command(self, subcommand: str, *args):
        """プロンプトコマンド処理"""
        try:
            if subcommand == "list":
                result = await self.learning_tool.get_prompt_templates()
                if result.get('status') == 'success':
                    prompts = result.get('data', [])
                    print(f"[CMD] プロンプト一覧 ({len(prompts)}件):")
                    for i, prompt in enumerate(prompts, 1):
                        print(f"  {i}. {prompt.get('name', '')} - {prompt.get('description', '')[:50]}...")
                else:
                    print(f"[ERROR] 取得エラー: {result.get('message', '')}")
                    
            elif subcommand == "add":
                if len(args) < 2:
                    print("[ERROR] 使用方法: prompt add <名前> <内容>")
                    return
                
                name = args[0]
                content = " ".join(args[1:])
                
                result = await self.learning_tool.add_prompt_template(
                    name=name,
                    content=content,
                    description=f"カスタムプロンプト: {name}"
                )
                
                if result.get('status') == 'success':
                    print(f"[OK] プロンプト追加完了: {name}")
                else:
                    print(f"[ERROR] 追加エラー: {result.get('message', '')}")
                    
            elif subcommand == "update":
                if len(args) < 2:
                    print("[ERROR] 使用方法: prompt update <名前> <内容>")
                    return
                
                name = args[0]
                content = " ".join(args[1:])
                
                result = await self.learning_tool.update_prompt_template(
                    name=name,
                    content=content
                )
                
                if result.get('status') == 'success':
                    print(f"[OK] プロンプト更新完了: {name}")
                else:
                    print(f"[ERROR] 更新エラー: {result.get('message', '')}")
                    
            elif subcommand == "delete":
                if not args:
                    print("[ERROR] 使用方法: prompt delete <名前>")
                    return
                
                name = args[0]
                result = await self.learning_tool.delete_prompt_template(name)
                
                if result.get('status') == 'success':
                    print(f"[OK] プロンプト削除完了: {name}")
                else:
                    print(f"[ERROR] 削除エラー: {result.get('message', '')}")
                    
            elif subcommand == "optimize":
                if not args:
                    print("[ERROR] 使用方法: prompt optimize <名前>")
                    return
                
                name = args[0]
                result = await self.learning_tool.optimize_prompt_template(name)
                
                if result.get('status') == 'success':
                    print(f"[OK] プロンプト最適化完了: {name}")
                    print(f"  改善スコア: {result.get('improvement_score', 0):.2f}")
                else:
                    print(f"[ERROR] 最適化エラー: {result.get('message', '')}")
                    
            elif subcommand == "export":
                result = await self.learning_tool.export_prompt_templates()
                
                if result.get('status') == 'success':
                    print(f"[OK] プロンプトエクスポート完了")
                    print(f"  プロンプト数: {result.get('count', 0)}件")
                else:
                    print(f"[ERROR] エクスポートエラー: {result.get('message', '')}")
                    
            elif subcommand == "import":
                if not args:
                    print("[ERROR] 使用方法: prompt import <ファイルパス>")
                    return
                
                file_path = args[0]
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        import_data = json.load(f)
                    
                    result = await self.learning_tool.import_prompt_templates(import_data)
                    
                    if result.get('status') == 'success':
                        print(f"[OK] プロンプトインポート完了: {result.get('imported_count', 0)}件")
                    else:
                        print(f"[ERROR] インポートエラー: {result.get('message', '')}")
                except Exception as e:
                    print(f"[ERROR] ファイル読み込みエラー: {e}")
                
            else:
                print("[ERROR] 不明なプロンプトコマンド。'prompt list/add/update/delete/optimize/export/import' を使用してください。")
                
        except Exception as e:
            print(f"[ERROR] プロンプトコマンドエラー: {e}")
    
    async def handle_test_command(self, subcommand: str):
        """テストコマンド処理"""
        try:
            if subcommand == "conversation":
                print("[TEST] 会話テストモード開始 (終了するには 'quit' と入力)")
                print("学習されたルールとプロンプトが適用されるかテストできます")
                
                while True:
                    try:
                        user_input = input("\n[YOU] あなた: ").strip()
                        if user_input.lower() in ['quit', 'exit', '終了']:
                            break
                        
                        await self.handle_chat(user_input)
                        
                    except KeyboardInterrupt:
                        break
                    except Exception as e:
                        print(f"[ERROR] エラー: {e}")
                        
            elif subcommand == "learning":
                print("[TEST] 学習機能テスト開始...")
                
                # 学習システムを開始
                await self.learning_tool.start_learning_system()
                print("[OK] 学習システムを開始しました")
                
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
                    print(f"[OK] テストデータ追加: {category}")
                
                # 学習サイクルを実行
                await self.learning_tool.manually_trigger_learning_cycle()
                print("[OK] 学習サイクルを実行しました")
                
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
                print("[ERROR] 不明なテストコマンド。'test conversation/learning' を使用してください。")
                
        except Exception as e:
            print(f"[ERROR] テストコマンドエラー: {e}")
    
    async def handle_status_command(self):
        """システム状態確認"""
        try:
            print("[SYS] システム状態:")
            
            # Codex互換モードの場合
            if self.config.is_codex_agent_enabled:
                try:
                    health_status = await self.agent_manager.codex_agent.health_check()
                    print(f"  [AI] Codex Agent: {'[OK] 正常' if health_status.get('healthy') else '[ERROR] 異常'}")
                    print(f"  [MODEL] 現在のモデル: {health_status.get('current_model', 'unknown')}")
                    
                    # パフォーマンス情報表示
                    if 'performance' in health_status:
                        perf = health_status['performance']
                        print(f"  [PERF] 過去5分間の統計:")
                        print(f"    リクエスト数: {perf.get('total_requests', 0)}")
                        print(f"    平均応答時間: {perf.get('avg_response_time_ms', 0):.1f}ms")
                        print(f"    エラー率: {perf.get('error_rate', 0):.1f}%")
                        print(f"    アクティブリクエスト: {perf.get('active_requests', 0)}")
                        
                        if 'system_metrics' in perf:
                            sys_metrics = perf['system_metrics']
                            print(f"    CPU使用率: {sys_metrics.get('avg_cpu_percent', 0):.1f}%")
                            print(f"    メモリ使用率: {sys_metrics.get('avg_memory_percent', 0):.1f}%")
                    
                    # システム統計情報
                    system_stats = self.agent_manager.codex_agent.get_system_stats()
                    pool_stats = system_stats.get('request_pool_stats', {})
                    print(f"  [POOL] リクエストプール:")
                    print(f"    並行処理数: {pool_stats.get('active_tasks', 0)}/{pool_stats.get('max_concurrent', 0)}")
                    print(f"    成功率: {pool_stats.get('success_rate', 0):.1f}%")
                    
                except Exception as e:
                    print(f"  [AI] Codex Agent: [ERROR] {e}")
                
                return
            
            # 従来モードの場合
            # OLLAMA状態
            try:
                ollama_status = await self.ollama_client.health_check()
                print(f"  [AI] OLLAMA: {'[OK] 正常' if ollama_status.get('status') == 'ok' else '[ERROR] 異常'}")
            except:
                print("  [AI] OLLAMA: [ERROR] 接続エラー")
            
            # データベース状態
            try:
                stats = await self.db_manager.get_learning_statistics()
                print(f"  [DB] データベース: [OK] 正常 (学習データ: {stats.get('total_learning_data', 0)}件)")
            except:
                print("  [DB] データベース: [ERROR] エラー")
            
            # 学習システム状態
            try:
                result = await self.learning_tool.get_learning_status()
                if result.get('status') == 'success':
                    status_data = result.get('data', {})
                    print(f"  [LEARN] 学習システム: {'[OK] 実行中' if status_data.get('is_running') else '[PAUSED] 停止中'}")
                else:
                    print("  [LEARN] 学習システム: [ERROR] エラー")
            except:
                print("  [LEARN] 学習システム: [ERROR] エラー")
                
        except Exception as e:
            print(f"[ERROR] 状態確認エラー: {e}")
    
    async def handle_report_command(self):
        """詳細レポート生成"""
        try:
            print("[REPORT] 詳細レポート生成中...")
            
            result = await self.learning_tool.get_performance_report(days=7)
            
            if result.get('status') == 'success':
                report = result.get('report', {})
                
                print("\n[SYS] パフォーマンスレポート (過去7日間):")
                print(f"  学習データ統計: {report.get('learning_stats', {})}")
                print(f"  知識ベース統計: {report.get('knowledge_stats', {})}")
                print(f"  プロンプト最適化統計: {report.get('prompt_optimization_stats', {})}")
                
                print("\n[STATS] パフォーマンス指標:")
                metrics = report.get('performance_metrics', {})
                for key, value in metrics.items():
                    print(f"  {key}: {value}")
                    
            else:
                print(f"[ERROR] レポート生成エラー: {result.get('message', '')}")
                
        except Exception as e:
            print(f"[ERROR] レポート生成エラー: {e}")
    
    async def run(self):
        """CLIメインループ"""
        if self.config.is_codex_agent_enabled:
            print("[AI] Codex互換AIエージェントCLI")
            print("入力してください (help でヘルプ表示):")
        else:
            print("[AI] 自己学習型AIエージェントCLI")
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
                        print("[ERROR] 使用方法: chat <メッセージ>")
                        continue
                    message = " ".join(args)
                    await self.handle_chat(message)
                    
                elif command == 'status':
                    await self.handle_status_command()
                
                # Codex互換モードでは以下のコマンドは無効
                elif self.config.is_codex_agent_enabled and command in ['learn', 'data', 'prompt', 'test', 'stats', 'report']:
                    print(f"[INFO] コマンド '{command}' はCodex互換モードでは利用できません。")
                    print("[INFO] 利用可能なコマンド: chat, status, help, quit")
                    
                elif command == 'learn':
                    if not args:
                        print("[ERROR] 使用方法: learn <start/stop/status/cycle>")
                        continue
                    await self.handle_learning_command(args[0])
                    
                elif command == 'data':
                    if not args:
                        print("[ERROR] 使用方法: data <add/list/update/delete/stats/export/import>")
                        continue
                    await self.handle_data_command(args[0], *args[1:])
                    
                elif command == 'prompt':
                    if not args:
                        print("[ERROR] 使用方法: prompt <list/add/update/delete/optimize/export/import>")
                        continue
                    await self.handle_prompt_command(args[0], *args[1:])
                    
                elif command == 'test':
                    if not args:
                        print("[ERROR] 使用方法: test <conversation/learning>")
                        continue
                    await self.handle_test_command(args[0])
                    
                elif command == 'status':
                    await self.handle_status_command()
                    
                elif command == 'stats':
                    await self.handle_data_command('stats')
                    
                elif command == 'report':
                    await self.handle_report_command()
                    
                else:
                    print(f"[ERROR] 不明なコマンド: {command}")
                    print("'help' でヘルプを表示してください")
                    
            except KeyboardInterrupt:
                print("\n[INFO] 終了しますか？ (y/N): ", end="")
                try:
                    response = input().strip().lower()
                    if response in ['y', 'yes', 'はい']:
                        self.is_running = False
                except:
                    self.is_running = False
                    
            except Exception as e:
                print(f"[ERROR] エラー: {e}")


async def main():
    """メイン関数"""
    cli = AgentCLI()
    
    try:
        if await cli.initialize():
            await cli.run()
        else:
            print("[ERROR] 初期化に失敗しました")
    except Exception as e:
        print(f"[ERROR] 予期しないエラー: {e}")
    finally:
        await cli.shutdown()


if __name__ == "__main__":
    asyncio.run(main())