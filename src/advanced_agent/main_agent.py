#!/usr/bin/env python3
"""
自己学習AIエージェント メインシステム
統合されたエージェントシステムのエントリーポイント
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.advanced_agent.core.self_learning_agent import SelfLearningAgent, create_self_learning_agent
from src.advanced_agent.config.settings import get_agent_config
from src.advanced_agent.core.environment import validate_environment_startup
from src.advanced_agent.core.logger import get_logger


class AgentMain:
    """エージェントメインクラス"""
    
    def __init__(self):
        self.logger = get_logger()
        self.agent: Optional[SelfLearningAgent] = None
        self.config = get_agent_config()
    
    async def initialize(self, config_path: Optional[str] = None) -> bool:
        """エージェント初期化"""
        try:
            self.logger.info("自己学習AIエージェント初期化開始")
            
            # 環境検証
            self.logger.info("環境検証実行中...")
            try:
                report = validate_environment_startup(self.config)
                if report.overall_status == "FAIL":
                    self.logger.error("環境検証失敗")
                    return False
                elif report.overall_status == "WARNING":
                    self.logger.warning("環境検証で警告が検出されましたが、続行します")
            except Exception as e:
                self.logger.warning(f"環境検証でエラーが発生しましたが、続行します: {e}")
            
            # エージェント作成
            self.agent = await create_self_learning_agent(config_path or "config/agent_config.yaml")
            
            # セッション初期化
            session_id = await self.agent.initialize_session()
            self.logger.info(f"セッション初期化完了: {session_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"エージェント初期化エラー: {e}")
            return False
    
    async def run_interactive_mode(self):
        """インタラクティブモード実行"""
        if not self.agent:
            self.logger.error("エージェントが初期化されていません")
            return
        
        print("\n" + "="*60)
        print("🤖 自己学習AIエージェント - インタラクティブモード")
        print("="*60)
        print("コマンド:")
        print("  /status  - エージェント状態表示")
        print("  /help    - ヘルプ表示")
        print("  /quit    - 終了")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n👤 あなた: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                    print("👋 さようなら！")
                    break
                
                if user_input.lower() == '/status':
                    await self._show_status()
                    continue
                
                if user_input.lower() == '/help':
                    self._show_help()
                    continue
                
                # エージェント処理
                print("🤖 エージェント思考中...")
                result = await self.agent.process_user_input(user_input)
                
                print(f"\n🤖 エージェント: {result['response']}")
                
                if 'agent_state' in result:
                    state = result['agent_state']
                    print(f"\n📊 状態: 学習エポック {state['learning_epoch']}, "
                          f"インタラクション {state['total_interactions']}, "
                          f"報酬スコア {state['reward_score']:.3f}")
                
            except KeyboardInterrupt:
                print("\n👋 さようなら！")
                break
            except Exception as e:
                self.logger.error(f"インタラクティブモードエラー: {e}")
                print(f"❌ エラーが発生しました: {e}")
    
    async def _show_status(self):
        """エージェント状態表示"""
        if not self.agent:
            print("❌ エージェントが初期化されていません")
            return
        
        status = await self.agent.get_agent_status()
        
        print("\n📊 エージェント状態:")
        print(f"  セッションID: {status['session_id']}")
        print(f"  学習エポック: {status['learning_epoch']}")
        print(f"  総インタラクション数: {status['total_interactions']}")
        print(f"  報酬スコア: {status['reward_score']:.3f}")
        print(f"  進化世代: {status['evolution_generation']}")
        print(f"  現在のプロンプトバージョン: {status['current_prompt_version']}")
        print(f"  プロンプトテンプレート数: {status['prompt_templates_count']}")
        print(f"  チューニングデータ数: {status['tuning_data_count']}")
        print(f"  進化候補数: {status['evolution_candidates_count']}")
        print(f"  最終活動: {status['last_activity']}")
    
    def _show_help(self):
        """ヘルプ表示"""
        print("\n📖 ヘルプ:")
        print("  このエージェントは以下の機能を持っています:")
        print("  • 永続的記憶: 過去の会話を記憶し、継続的な学習を行います")
        print("  • 自己改善: プロンプトとチューニングデータを動的に最適化します")
        print("  • 推論能力: Deepseekレベルの複雑な推論を実行します")
        print("  • ツール使用: ネット検索、コマンド実行、ファイル操作、MCP連携が可能です")
        print("  • 進化: SAKANA AIスタイルの交配進化により能力を向上させます")
        print("  • 報酬学習: ユーザーとの関わりを報酬として学習します")
        print("\n  コマンド:")
        print("    /status  - エージェント状態表示")
        print("    /help    - このヘルプ表示")
        print("    /quit    - 終了")
    
    async def run_single_query(self, query: str) -> Dict[str, Any]:
        """単一クエリ実行"""
        if not self.agent:
            raise RuntimeError("エージェントが初期化されていません")
        
        return await self.agent.process_user_input(query)
    
    async def cleanup(self):
        """クリーンアップ"""
        if self.agent:
            await self.agent.close()
            self.logger.info("エージェントクリーンアップ完了")


async def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="自己学習AIエージェント",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python -m src.advanced_agent.main_agent                    # インタラクティブモード
  python -m src.advanced_agent.main_agent --query "こんにちは"  # 単一クエリ
  python -m src.advanced_agent.main_agent --config custom.yaml # カスタム設定
        """
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="単一クエリを実行"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="設定ファイルのパス"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="詳細ログ出力"
    )
    
    args = parser.parse_args()
    
    # ログレベル設定
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # メインエージェント作成
    agent_main = AgentMain()
    
    try:
        # 初期化
        if not await agent_main.initialize(args.config):
            print("❌ エージェント初期化に失敗しました")
            sys.exit(1)
        
        if args.query:
            # 単一クエリ実行
            print(f"🤖 クエリ実行: {args.query}")
            result = await agent_main.run_single_query(args.query)
            print(f"🤖 回答: {result['response']}")
        else:
            # インタラクティブモード
            await agent_main.run_interactive_mode()
    
    except KeyboardInterrupt:
        print("\n👋 終了します...")
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        sys.exit(1)
    finally:
        await agent_main.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
