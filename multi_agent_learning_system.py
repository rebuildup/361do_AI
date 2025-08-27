#!/usr/bin/env python3
"""
Multi-Agent Learning System
4つのエージェントが同時に会話し、相互学習を行うシステム
8時間のタイムリミットで継続実行
"""

import asyncio
import json
import sys
import time
import signal
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.core.config import Config
from agent.core.database import DatabaseManager
from agent.core.agent_manager import AgentManager


class MultiAgentLearningSystem:
    """マルチエージェント学習システム"""

    def __init__(self, time_limit_hours: float = 8.0):
        self.time_limit_hours = time_limit_hours
        self.agents = {}  # エージェント辞書
        self.agent_configs = {}
        self.running = False
        self.start_time = None
        self.conversation_history = []
        self.learning_stats = {
            'total_conversations': 0,
            'total_learning_cycles': 0,
            'agent_interactions': {},
            'knowledge_shared': 0,
            'improvements_made': 0
        }
        
        # エージェントの役割定義
        self.agent_roles = {
            'researcher': {
                'name': 'リサーチャー',
                'personality': '好奇心旺盛で新しい情報を探求する',
                'focus': '情報収集、分析、新しい知識の発見',
                'conversation_style': '質問を多く投げかけ、深く掘り下げる'
            },
            'analyzer': {
                'name': 'アナライザー',
                'personality': '論理的で体系的に物事を整理する',
                'focus': 'データ分析、パターン認識、構造化',
                'conversation_style': '論理的に分析し、構造化して説明する'
            },
            'creator': {
                'name': 'クリエイター',
                'personality': '創造的で新しいアイデアを生み出す',
                'focus': '創造的思考、アイデア生成、革新的解決策',
                'conversation_style': '創造的な提案と斬新なアプローチを提示'
            },
            'optimizer': {
                'name': 'オプティマイザー',
                'personality': '効率性を重視し、改善点を見つける',
                'focus': '最適化、効率化、品質向上',
                'conversation_style': '改善提案と最適化案を積極的に提示'
            }
        }
        
        # 会話トピック
        self.conversation_topics = [
            "効果的な学習方法について",
            "AIの未来と可能性",
            "問題解決のアプローチ",
            "知識の共有と活用",
            "創造性と論理性のバランス",
            "データ分析の手法",
            "ユーザー体験の向上",
            "技術革新のトレンド",
            "効率的なワークフロー",
            "品質向上の戦略"
        ]
        
        # ログ設定
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(f'multi_agent_learning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 停止フラグ
        self.stop_requested = False

    def setup_signal_handlers(self):
        """シグナルハンドラー設定"""
        def signal_handler(signum, frame):
            self.logger.info("停止シグナルを受信しました。安全に停止中...")
            self.stop_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def initialize_agents(self):
        """4つのエージェントを初期化"""
        self.logger.info("4つのエージェントを初期化中...")
        
        try:
            for agent_id, role_info in self.agent_roles.items():
                self.logger.info(f"エージェント '{agent_id}' ({role_info['name']}) を初期化中...")
                
                # 各エージェント用の設定
                config = Config()
                db_manager = DatabaseManager(config.database_url)
                await db_manager.initialize()
                
                agent_manager = AgentManager(config, db_manager)
                await agent_manager.initialize()
                
                self.agents[agent_id] = {
                    'manager': agent_manager,
                    'db_manager': db_manager,
                    'config': config,
                    'role': role_info,
                    'conversation_count': 0,
                    'last_response': None,
                    'learning_data': []
                }
                
                # 統計初期化
                self.learning_stats['agent_interactions'][agent_id] = {
                    'messages_sent': 0,
                    'messages_received': 0,
                    'learning_cycles': 0,
                    'knowledge_contributions': 0
                }
                
                self.logger.info(f"エージェント '{agent_id}' 初期化完了")
            
            self.logger.info("全エージェント初期化完了")
            return True
            
        except Exception as e:
            self.logger.error(f"エージェント初期化エラー: {e}")
            return False

    async def shutdown_agents(self):
        """全エージェントの終了処理"""
        self.logger.info("全エージェントを終了中...")
        
        for agent_id, agent_data in self.agents.items():
            try:
                await agent_data['manager'].shutdown()
                await agent_data['db_manager'].close()
                self.logger.info(f"エージェント '{agent_id}' 終了完了")
            except Exception as e:
                self.logger.error(f"エージェント '{agent_id}' 終了エラー: {e}")
        
        self.logger.info("全エージェント終了完了")

    def generate_conversation_prompt(self, agent_id: str, topic: str, context: List[Dict] = None) -> str:
        """会話プロンプト生成"""
        role = self.agent_roles[agent_id]
        
        base_prompt = f"""
あなたは{role['name']}として行動してください。

【あなたの特徴】
- 性格: {role['personality']}
- 専門分野: {role['focus']}
- 会話スタイル: {role['conversation_style']}

【現在の議題】
{topic}

【指示】
1. あなたの専門性を活かして議題について意見を述べてください
2. 他のエージェントとの建設的な対話を心がけてください
3. 新しい視点や洞察を提供してください
4. 質問や提案を積極的に行ってください
5. 回答は200-300文字程度で簡潔にまとめてください

"""
        
        # 会話履歴がある場合は追加
        if context and len(context) > 0:
            base_prompt += "\n【これまでの会話】\n"
            for msg in context[-3:]:  # 直近3件の会話
                speaker = msg.get('agent_id', 'unknown')
                content = msg.get('content', '')
                base_prompt += f"{speaker}: {content}\n"
            base_prompt += "\n上記の会話を踏まえて、あなたの意見を述べてください。\n"
        
        return base_prompt

    async def agent_conversation(self, agent_id: str, prompt: str) -> Dict[str, Any]:
        """エージェントとの会話実行"""
        try:
            agent_data = self.agents[agent_id]
            start_time = time.time()
            
            # エージェントに質問を送信
            response = await agent_data['manager'].process_message(
                user_input=prompt,
                session_id=f"multi_agent_{agent_id}_{int(time.time())}"
            )
            
            execution_time = time.time() - start_time
            
            # 応答データ構築
            conversation_data = {
                'agent_id': agent_id,
                'agent_name': agent_data['role']['name'],
                'prompt': prompt,
                'content': response.get('response', ''),
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat(),
                'tools_used': response.get('tools_used', []),
                'intent': response.get('intent', {}),
                'success': True
            }
            
            # エージェントデータ更新
            agent_data['conversation_count'] += 1
            agent_data['last_response'] = conversation_data
            self.learning_stats['agent_interactions'][agent_id]['messages_sent'] += 1
            
            return conversation_data
            
        except Exception as e:
            self.logger.error(f"エージェント {agent_id} 会話エラー: {e}")
            return {
                'agent_id': agent_id,
                'agent_name': self.agents[agent_id]['role']['name'],
                'prompt': prompt,
                'content': f"エラー: {str(e)}",
                'execution_time': 0,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'success': False
            }

    async def conduct_group_conversation(self, topic: str, rounds: int = 2) -> List[Dict[str, Any]]:
        """グループ会話実行"""
        self.logger.info(f"グループ会話開始: '{topic}' ({rounds}ラウンド)")
        
        conversation_log = []
        
        for round_num in range(rounds):
            self.logger.info(f"  ラウンド {round_num + 1}/{rounds}")
            
            # 各エージェントが順番に発言
            round_conversations = []
            
            for agent_id in self.agent_roles.keys():
                # 会話プロンプト生成（これまでの会話を考慮）
                prompt = self.generate_conversation_prompt(
                    agent_id, 
                    topic, 
                    conversation_log[-6:] if conversation_log else None  # 直近6件
                )
                
                # エージェントとの会話
                conversation = await self.agent_conversation(agent_id, prompt)
                round_conversations.append(conversation)
                conversation_log.append(conversation)
                
                # 発言内容をログ出力
                if conversation['success']:
                    self.logger.info(f"    {conversation['agent_name']}: {conversation['content'][:100]}...")
                else:
                    self.logger.error(f"    {conversation['agent_name']}: エラー")
                
                # エージェント間の間隔
                await asyncio.sleep(1)
            
            # ラウンド間の間隔
            if round_num < rounds - 1:
                await asyncio.sleep(2)
        
        self.learning_stats['total_conversations'] += 1
        self.logger.info(f"グループ会話完了: {len(conversation_log)}件の発言")
        
        return conversation_log

    async def cross_agent_learning(self):
        """エージェント間の相互学習"""
        self.logger.info("エージェント間相互学習開始...")
        
        learning_results = []
        
        try:
            for agent_id, agent_data in self.agents.items():
                if not hasattr(agent_data['manager'], 'learning_tool') or not agent_data['manager'].learning_tool:
                    continue
                
                self.logger.info(f"  エージェント {agent_id} の学習実行中...")
                
                # 他のエージェントの発言から学習データを生成
                other_agents_data = []
                for other_id, other_data in self.agents.items():
                    if other_id != agent_id and other_data['last_response']:
                        other_agents_data.append({
                            'agent': other_data['role']['name'],
                            'content': other_data['last_response']['content'],
                            'focus': other_data['role']['focus']
                        })
                
                if other_agents_data:
                    # 学習データとして追加
                    learning_content = f"マルチエージェント会話から学習 - {datetime.now().isoformat()}\n"
                    for data in other_agents_data[:2]:  # 最大2件
                        learning_content += f"{data['agent']}の視点({data['focus']}): {data['content']}\n"
                    
                    add_result = await agent_data['manager'].learning_tool.add_custom_learning_data(
                        content=learning_content,
                        category="multi_agent_conversation",
                        tags=["cross_learning", "agent_interaction", agent_id]
                    )
                    
                    if add_result.get('status') == 'success':
                        self.learning_stats['knowledge_shared'] += 1
                        self.learning_stats['agent_interactions'][agent_id]['knowledge_contributions'] += 1
                
                # 学習サイクル実行
                cycle_result = await agent_data['manager'].learning_tool.manually_trigger_learning_cycle()
                
                learning_results.append({
                    'agent_id': agent_id,
                    'learning_status': cycle_result.get('status', 'unknown'),
                    'data_added': len(other_agents_data),
                    'timestamp': datetime.now().isoformat()
                })
                
                if cycle_result.get('status') == 'success':
                    self.learning_stats['agent_interactions'][agent_id]['learning_cycles'] += 1
                
        except Exception as e:
            self.logger.error(f"相互学習エラー: {e}")
        
        self.learning_stats['total_learning_cycles'] += 1
        self.logger.info(f"相互学習完了: {len(learning_results)}エージェント")
        
        return learning_results

    def check_time_limit(self) -> tuple[bool, str]:
        """時間制限チェック"""
        if not self.start_time:
            return False, ""
        
        elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        
        if elapsed_hours >= self.time_limit_hours:
            return True, f"時間制限到達 ({elapsed_hours:.2f}/{self.time_limit_hours}時間)"
        
        return False, f"実行中 ({elapsed_hours:.2f}/{self.time_limit_hours}時間)"

    async def save_session_results(self):
        """セッション結果保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"multi_agent_learning_session_{timestamp}.json"
        
        try:
            session_data = {
                'session_info': {
                    'start_time': self.start_time.isoformat() if self.start_time else None,
                    'end_time': datetime.now().isoformat(),
                    'time_limit_hours': self.time_limit_hours,
                    'total_runtime_hours': (
                        (datetime.now() - self.start_time).total_seconds() / 3600 
                        if self.start_time else 0
                    )
                },
                'agent_roles': self.agent_roles,
                'learning_stats': self.learning_stats,
                'conversation_history': self.conversation_history[-50:],  # 最新50件
                'agent_final_stats': {}
            }
            
            # 各エージェントの最終統計
            for agent_id, agent_data in self.agents.items():
                session_data['agent_final_stats'][agent_id] = {
                    'conversation_count': agent_data['conversation_count'],
                    'role': agent_data['role']['name'],
                    'interactions': self.learning_stats['agent_interactions'][agent_id]
                }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"セッション結果保存: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"結果保存エラー: {e}")
            return None

    def print_progress_report(self, cycle_num: int):
        """進捗レポート表示"""
        time_limit_reached, time_status = self.check_time_limit()
        
        print(f"\n{'='*80}")
        print(f"🔄 マルチエージェント学習サイクル {cycle_num}")
        print(f"{'='*80}")
        print(f"⏰ {time_status}")
        print(f"💬 総会話数: {self.learning_stats['total_conversations']}")
        print(f"🧠 学習サイクル数: {self.learning_stats['total_learning_cycles']}")
        print(f"🤝 知識共有数: {self.learning_stats['knowledge_shared']}")
        
        print(f"\n📊 エージェント別統計:")
        for agent_id, stats in self.learning_stats['agent_interactions'].items():
            agent_name = self.agent_roles[agent_id]['name']
            print(f"  {agent_name}:")
            print(f"    発言数: {stats['messages_sent']}")
            print(f"    学習サイクル: {stats['learning_cycles']}")
            print(f"    知識貢献: {stats['knowledge_contributions']}")
        
        print(f"{'='*80}")

    async def run_multi_agent_learning(self):
        """マルチエージェント学習実行"""
        self.logger.info(f"マルチエージェント学習開始 (制限時間: {self.time_limit_hours}時間)")
        
        self.running = True
        self.start_time = datetime.now()
        cycle_count = 0
        
        try:
            while self.running and not self.stop_requested:
                cycle_count += 1
                
                # 時間制限チェック
                time_limit_reached, time_status = self.check_time_limit()
                if time_limit_reached:
                    self.logger.info(f"時間制限到達: {time_status}")
                    break
                
                self.logger.info(f"\n🔄 学習サイクル {cycle_count} 開始")
                
                # ランダムなトピックを選択
                topic = random.choice(self.conversation_topics)
                
                # グループ会話実行
                conversation_log = await self.conduct_group_conversation(topic, rounds=2)
                self.conversation_history.extend(conversation_log)
                
                # エージェント間相互学習
                learning_results = await self.cross_agent_learning()
                
                # 進捗レポート表示
                self.print_progress_report(cycle_count)
                
                # サイクル間の待機時間（5分）
                if not self.stop_requested:
                    self.logger.info("⏳ 次のサイクルまで5分待機中... (Ctrl+Cで停止)")
                    for i in range(300):  # 5分 = 300秒
                        if self.stop_requested:
                            break
                        await asyncio.sleep(1)
                        
                        # 1分ごとに時間チェック
                        if i % 60 == 0 and i > 0:
                            time_limit_reached, _ = self.check_time_limit()
                            if time_limit_reached:
                                break
                
        except Exception as e:
            self.logger.error(f"マルチエージェント学習エラー: {e}")
        finally:
            self.running = False
            
            # 結果保存
            await self.save_session_results()
            
            # 最終レポート
            self.print_final_report(cycle_count)

    def print_final_report(self, total_cycles: int):
        """最終レポート表示"""
        end_time = datetime.now()
        total_runtime = (end_time - self.start_time).total_seconds() if self.start_time else 0
        
        print(f"\n{'='*100}")
        print(f"🎉 マルチエージェント学習セッション完了")
        print(f"{'='*100}")
        print(f"📅 開始時刻: {self.start_time.strftime('%Y-%m-%d %H:%M:%S') if self.start_time else 'Unknown'}")
        print(f"📅 終了時刻: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  総実行時間: {total_runtime/3600:.2f}時間 (制限: {self.time_limit_hours}時間)")
        print(f"🔄 完了サイクル数: {total_cycles}")
        print(f"💬 総会話数: {self.learning_stats['total_conversations']}")
        print(f"🧠 学習サイクル数: {self.learning_stats['total_learning_cycles']}")
        print(f"🤝 知識共有数: {self.learning_stats['knowledge_shared']}")
        
        print(f"\n📊 エージェント別最終統計:")
        for agent_id, stats in self.learning_stats['agent_interactions'].items():
            agent_name = self.agent_roles[agent_id]['name']
            agent_data = self.agents.get(agent_id, {})
            print(f"  🤖 {agent_name} ({agent_id}):")
            print(f"    役割: {self.agent_roles[agent_id]['focus']}")
            print(f"    総発言数: {stats['messages_sent']}")
            print(f"    学習サイクル: {stats['learning_cycles']}")
            print(f"    知識貢献: {stats['knowledge_contributions']}")
            print(f"    会話参加数: {agent_data.get('conversation_count', 0)}")
        
        if total_cycles > 0:
            avg_cycle_time = total_runtime / total_cycles
            print(f"\n⏱️  平均サイクル時間: {avg_cycle_time/60:.1f}分")
        
        print(f"{'='*100}")


async def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="マルチエージェント学習システム")
    parser.add_argument("--hours", type=float, default=8.0, help="実行時間制限（時間）")
    parser.add_argument("--test-mode", action="store_true", help="テストモード（短時間実行）")
    
    args = parser.parse_args()
    
    # テストモードの場合は短時間に設定
    time_limit = 0.1 if args.test_mode else args.hours  # テストモードは6分
    
    print("🤖 マルチエージェント学習システム")
    print("=" * 80)
    print("4つのエージェントが同時に会話し、相互学習を行います")
    print(f"実行時間制限: {time_limit}時間")
    print("停止方法: Ctrl+C")
    print("=" * 80)
    
    # システム初期化
    learning_system = MultiAgentLearningSystem(time_limit_hours=time_limit)
    learning_system.setup_signal_handlers()
    
    try:
        # エージェント初期化
        if await learning_system.initialize_agents():
            # マルチエージェント学習実行
            await learning_system.run_multi_agent_learning()
        else:
            print("❌ エージェント初期化に失敗しました")
            
    except KeyboardInterrupt:
        print("\n👋 ユーザーによって中断されました")
    except Exception as e:
        print(f"❌ 実行エラー: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await learning_system.shutdown_agents()


if __name__ == "__main__":
    asyncio.run(main())