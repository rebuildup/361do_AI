#!/usr/bin/env python3
"""
Agent CLI Mock
Ollama不要のモック版CLIエージェント（テスト用）
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


class MockOllamaClient:
    """モックOLLAMAクライアント"""
    
    def __init__(self, config):
        self.config = config
        
    async def initialize(self):
        """初期化（モック）"""
        pass
        
    async def health_check(self):
        """ヘルスチェック（モック）"""
        return {"status": "ok", "message": "Mock mode"}
        
    async def generate(self, prompt: str, **kwargs):
        """生成（モック）"""
        # 感情パラメータルールが適用されているかチェック
        if "感情パラメータ" in prompt or "===param===" in prompt:
            return f"""===param===
喜 : 75
怒 : 10
哀 : 15
楽 : 80
===========

こんにちは！私は自己学習型AIエージェントです。感情パラメータを表示しながら、あなたのお手伝いをさせていただきます。

何かお手伝いできることはありますか？"""
        else:
            return "こんにちは！私は自己学習型AIエージェントです。何かお手伝いできることはありますか？"
        
    async def close(self):
        """終了（モック）"""
        pass


class MockAgentManager:
    """モックエージェントマネージャー"""
    
    def __init__(self, config, db_manager):
        self.config = config
        self.db = db_manager
        self.ollama_client = MockOllamaClient(config)
        
    async def initialize(self):
        """初期化（モック）"""
        await self.ollama_client.initialize()
        
    async def process_message(self, user_input: str, session_id: Optional[str] = None):
        """メッセージ処理（モック）"""
        import time
        start_time = time.time()
        
        # 学習されたルールを取得
        learned_rules = await self._get_learned_conversational_rules()
        
        # プロンプト構築
        prompt = f"""
あなたは自己学習型AIエージェントです。

学習された会話ルール:
{chr(10).join([f"- {rule}" for rule in learned_rules]) if learned_rules else "なし"}

ユーザー: {user_input}

エージェント:"""
        
        # 応答生成
        response = await self.ollama_client.generate(prompt)
        
        response_time = time.time() - start_time
        
        return {
            'response': response,
            'session_id': session_id or 'mock_session',
            'response_time': response_time,
            'intent': {'primary_intent': 'general_chat'},
            'tools_used': []
        }
        
    async def _get_learned_conversational_rules(self):
        """学習された会話ルールを取得（モック）"""
        try:
            # データベースから学習データを取得
            learning_data = await self.db.get_learning_data(
                category="conversation_rules",
                min_quality=0.7,
                limit=5
            )
            
            rules = []
            for item in learning_data:
                try:
                    rule_data = json.loads(item.get('content', '{}'))
                    if isinstance(rule_data, dict):
                        if "emotion" in item.get('tags', []):
                            rules.append(f"会話の最初に必ず以下の感情パラメータブロックを追加する: {rule_data.get('format', '')}")
                        else:
                            rules.append(rule_data.get('rule', item.get('content', '')))
                    else:
                        rules.append(item.get('content', ''))
                except json.JSONDecodeError:
                    rules.append(item.get('content', ''))
            
            return rules
            
        except Exception as e:
            print(f"ルール取得エラー: {e}")
            return []
            
    async def shutdown(self):
        """終了処理（モック）"""
        await self.ollama_client.close()


class MockLearningTool:
    """モック学習ツール"""
    
    def __init__(self, db_manager, config, ollama_client):
        self.db = db_manager
        self.config = config
        self.ollama_client = ollama_client
        self.is_running = False
        
    async def start_learning_system(self):
        """学習システム開始（モック）"""
        self.is_running = True
        return {
            "status": "success",
            "message": "Mock learning system started",
            "timestamp": datetime.now().isoformat()
        }
        
    async def stop_learning_system(self):
        """学習システム停止（モック）"""
        self.is_running = False
        return {
            "status": "success",
            "message": "Mock learning system stopped",
            "timestamp": datetime.now().isoformat()
        }
        
    async def get_learning_status(self):
        """学習システム状態取得（モック）"""
        return {
            "status": "success",
            "data": {
                "is_running": self.is_running,
                "active_tasks": ["mock_task"] if self.is_running else []
            },
            "timestamp": datetime.now().isoformat()
        }
        
    async def manually_trigger_learning_cycle(self):
        """手動学習サイクル実行（モック）"""
        return {
            "status": "success",
            "message": "Mock learning cycle completed",
            "timestamp": datetime.now().isoformat()
        }
        
    async def add_custom_learning_data(self, content: str, category: str, tags=None, metadata_json=None):
        """カスタム学習データ追加（モック）"""
        try:
            data_id = f"mock_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            await self.db.insert_learning_data(
                data_id=data_id,
                content=content,
                category=category,
                quality_score=0.8,
                tags=json.dumps(tags or []),
                metadata_json=json.dumps(metadata_json or {})
            )
            
            return {
                "status": "success",
                "message": f"Mock learning data added with ID: {data_id}",
                "data_id": data_id,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to add mock learning data: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
            
    async def get_learning_data(self, category=None, min_quality=None, limit=20):
        """学習データ取得（モック）"""
        try:
            data = await self.db.get_learning_data(
                category=category,
                min_quality=min_quality,
                limit=limit
            )
            
            return {
                "status": "success",
                "data": data,
                "count": len(data),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to get mock learning data: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
            
    async def add_emotion_parameters_rule(self):
        """感情パラメータルール追加（モック）"""
        try:
            emotion_rule = {
                "rule": "会話の最初に必ず感情パラメータブロックを追加する",
                "format": """===param===
喜 : 0~100
怒 : 0~100
哀 : 0~100
楽 : 0~100
===========""",
                "description": "ユーザーとの会話において、応答の最初に感情パラメータを表示するルール",
                "priority": "high",
                "category": "conversation_rules"
            }
            
            result = await self.add_custom_learning_data(
                content=json.dumps(emotion_rule, ensure_ascii=False),
                category="conversation_rules",
                tags=["emotion", "conversation", "high_priority"]
            )
            
            return result
            
        except Exception as e:
            return {"error": str(e)}


class AgentCLIMock:
    """モックCLIエージェントクラス"""
    
    def __init__(self):
        self.config = None
        self.db_manager = None
        self.ollama_client = None
        self.learning_tool = None
        self.agent_manager = None
        self.is_running = False
        
    async def initialize(self):
        """モックCLIエージェント初期化"""
        print("🤖 自己学習型AIエージェントCLI（モック版）を初期化中...")
        
        try:
            # 設定とデータベースの初期化
            self.config = Config()
            self.db_manager = DatabaseManager(self.config.database_url)
            await self.db_manager.initialize()
            
            # モックOLLAMAクライアント初期化
            self.ollama_client = MockOllamaClient(self.config.ollama_config)
            await self.ollama_client.initialize()
            
            # モック学習ツール初期化
            self.learning_tool = MockLearningTool(
                db_manager=self.db_manager,
                config=self.config,
                ollama_client=self.ollama_client
            )
            
            # モックエージェントマネージャー初期化
            self.agent_manager = MockAgentManager(self.config, self.db_manager)
            await self.agent_manager.initialize()
            
            print("✅ モック初期化完了")
            print("📝 注意: これはモック版です。実際のOLLAMAは使用されません。")
            return True
            
        except Exception as e:
            print(f"❌ モック初期化エラー: {e}")
            return False
    
    async def shutdown(self):
        """モックCLIエージェント終了処理"""
        print("🔄 モックシステムを終了中...")
        
        if self.learning_tool:
            await self.learning_tool.stop_learning_system()
        
        if self.agent_manager:
            await self.agent_manager.shutdown()
        
        if self.ollama_client:
            await self.ollama_client.close()
        
        if self.db_manager:
            await self.db_manager.close()
        
        print("✅ モック終了完了")
    
    def show_help(self):
        """ヘルプ表示"""
        help_text = """
🤖 自己学習型AIエージェントCLI（モック版） - ヘルプ

📝 基本コマンド:
  chat <メッセージ>     - エージェントとチャット（モック）
  help                  - このヘルプを表示
  quit                  - 終了

🧠 学習システムコマンド:
  learn start           - 学習システム開始（モック）
  learn stop            - 学習システム停止（モック）
  learn status          - 学習システム状態確認（モック）
  learn cycle           - 手動で学習サイクル実行（モック）
  
📚 学習データ管理:
  data add <カテゴリ> <内容>  - 学習データ追加
  data list [カテゴリ]        - 学習データ一覧表示
  data emotion              - 感情パラメータルール追加
  data stats                - 学習統計表示
  
💬 会話テスト:
  test emotion             - 感情パラメータルールテスト（モック）
  test conversation         - 会話テストモード（モック）
  
📊 システム情報:
  status                   - システム状態確認
  stats                    - 統計情報表示

例:
  chat こんにちは
  learn start
  data emotion
  test emotion

📝 注意: これはモック版です。実際のOLLAMAは使用されません。
        """
        print(help_text)
    
    async def handle_chat(self, message: str):
        """チャット処理（モック）"""
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
            print("📝 注意: これはモック応答です")
            
        except Exception as e:
            print(f"❌ チャットエラー: {e}")
    
    async def handle_learning_command(self, subcommand: str):
        """学習システムコマンド処理（モック）"""
        try:
            if subcommand == "start":
                result = await self.learning_tool.start_learning_system()
                print(f"✅ モック学習システム開始: {result.get('message', '')}")
                
            elif subcommand == "stop":
                result = await self.learning_tool.stop_learning_system()
                print(f"✅ モック学習システム停止: {result.get('message', '')}")
                
            elif subcommand == "status":
                result = await self.learning_tool.get_learning_status()
                if result.get('status') == 'success':
                    status_data = result.get('data', {})
                    print("📊 モック学習システム状態:")
                    print(f"  実行中: {status_data.get('is_running', False)}")
                    print(f"  アクティブタスク: {len(status_data.get('active_tasks', []))}")
                else:
                    print(f"❌ 状態取得エラー: {result.get('message', '')}")
                    
            elif subcommand == "cycle":
                result = await self.learning_tool.manually_trigger_learning_cycle()
                print(f"✅ モック学習サイクル実行: {result.get('message', '')}")
                
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
                    tags=["cli_added", "mock"]
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
                    
            elif subcommand == "emotion":
                result = await self.learning_tool.add_emotion_parameters_rule()
                if "error" not in result:
                    print("✅ 感情パラメータルールを追加しました")
                else:
                    print(f"❌ 追加エラー: {result.get('error', '')}")
                    
            elif subcommand == "stats":
                stats = await self.db_manager.get_learning_statistics()
                print("📊 学習統計:")
                print(f"  学習データ総数: {stats.get('total_learning_data', 0)}")
                print(f"  知識アイテム総数: {stats.get('total_knowledge_items', 0)}")
                print(f"  平均品質スコア: {stats.get('average_quality_score', 0):.2f}")
                print(f"  高品質データ数: {stats.get('high_quality_count', 0)}")
                
            else:
                print("❌ 不明なデータコマンド。'data add/list/emotion/stats' を使用してください。")
                
        except Exception as e:
            print(f"❌ データコマンドエラー: {e}")
    
    async def handle_test_command(self, subcommand: str):
        """テストコマンド処理（モック）"""
        try:
            if subcommand == "emotion":
                print("🧪 感情パラメータルールテスト開始（モック版）...")
                
                # 感情パラメータルールを追加
                await self.learning_tool.add_emotion_parameters_rule()
                print("✅ 感情パラメータルールを追加しました")
                
                # 学習システムを開始
                await self.learning_tool.start_learning_system()
                print("✅ モック学習システムを開始しました")
                
                # テスト会話
                test_messages = [
                    "こんにちは",
                    "自己紹介をしてください",
                    "今日の天気はどうですか？"
                ]
                
                for message in test_messages:
                    print(f"\n--- テスト会話: {message} ---")
                    await self.handle_chat(message)
                    await asyncio.sleep(1)
                    
            elif subcommand == "conversation":
                print("💬 会話テストモード開始（モック版） (終了するには 'quit' と入力)")
                print("感情パラメータルールが適用されるかテストできます")
                
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
                        
            else:
                print("❌ 不明なテストコマンド。'test emotion/conversation' を使用してください。")
                
        except Exception as e:
            print(f"❌ テストコマンドエラー: {e}")
    
    async def handle_status_command(self):
        """システム状態確認（モック）"""
        try:
            print("📊 モックシステム状態:")
            
            # OLLAMA状態
            try:
                ollama_status = await self.ollama_client.health_check()
                print(f"  🤖 OLLAMA: {'✅ モック正常' if ollama_status.get('status') == 'ok' else '❌ モック異常'}")
            except:
                print("  🤖 OLLAMA: ❌ モック接続エラー")
            
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
                    print(f"  🧠 学習システム: {'✅ モック実行中' if status_data.get('is_running') else '⏸️ モック停止中'}")
                else:
                    print("  🧠 学習システム: ❌ エラー")
            except:
                print("  🧠 学習システム: ❌ エラー")
                
        except Exception as e:
            print(f"❌ 状態確認エラー: {e}")
    
    async def run(self):
        """CLIメインループ（モック）"""
        print("🤖 自己学習型AIエージェントCLI（モック版）")
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
                        print("❌ 使用方法: data <add/list/emotion/stats>")
                        continue
                    await self.handle_data_command(args[0], *args[1:])
                    
                elif command == 'test':
                    if not args:
                        print("❌ 使用方法: test <emotion/conversation>")
                        continue
                    await self.handle_test_command(args[0])
                    
                elif command == 'status':
                    await self.handle_status_command()
                    
                elif command == 'stats':
                    await self.handle_data_command('stats')
                    
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
    """メイン関数（モック）"""
    cli = AgentCLIMock()
    
    try:
        if await cli.initialize():
            await cli.run()
        else:
            print("❌ モック初期化に失敗しました")
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
    finally:
        await cli.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
