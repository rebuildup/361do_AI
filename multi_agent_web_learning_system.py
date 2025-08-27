#!/usr/bin/env python3
"""
Multi-Agent Web Learning System with Playwright MCP
4つのエージェントがPlaywright MCPを活用してWebブラウザを操作しながら学習するシステム
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

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.core.config import Config
from agent.core.database import DatabaseManager
from agent.core.agent_manager import AgentManager


class MultiAgentWebLearningSystem:
    """Playwright MCP対応マルチエージェント学習システム"""

    def __init__(self, time_limit_hours: float = 8.0):
        self.time_limit_hours = time_limit_hours
        self.agents = {}
        self.running = False
        self.start_time = None
        self.conversation_history = []
        self.web_research_history = []
        self.learning_stats = {
            'total_conversations': 0,
            'total_web_researches': 0,
            'total_learning_cycles': 0,
            'agent_interactions': {},
            'knowledge_shared': 0,
            'web_pages_visited': 0,
            'improvements_made': 0
        }
        
        # エージェントの役割定義（Web活動を含む）
        self.agent_roles = {
            'web_researcher': {
                'name': 'Webリサーチャー',
                'personality': 'インターネットから最新情報を積極的に収集する',
                'focus': 'Web検索、情報収集、トレンド分析',
                'conversation_style': '最新の情報を引用して議論を深める',
                'web_tasks': ['search', 'browse', 'analyze_content']
            },
            'data_analyzer': {
                'name': 'データアナライザー',
                'personality': 'Webデータを構造化して分析する',
                'focus': 'データ抽出、統計分析、パターン認識',
                'conversation_style': 'データに基づいた論理的な分析を提示',
                'web_tasks': ['extract_data', 'analyze_structure', 'compare_sources']
            },
            'content_creator': {
                'name': 'コンテンツクリエイター',
                'personality': 'Web上の情報から創造的なアイデアを生成',
                'focus': '創造的思考、コンテンツ生成、アイデア統合',
                'conversation_style': 'Web情報を基に創造的な提案を行う',
                'web_tasks': ['content_analysis', 'inspiration_gathering', 'trend_monitoring']
            },
            'quality_optimizer': {
                'name': '品質オプティマイザー',
                'personality': 'Web情報の品質を評価し最適化する',
                'focus': '情報品質評価、信頼性チェック、最適化提案',
                'conversation_style': '情報の信頼性と品質向上を重視',
                'web_tasks': ['quality_check', 'source_verification', 'optimization_analysis']
            }
        }  
      
        # Web研究用のトピックとURL
        self.web_research_topics = {
            'ai_trends': {
                'topic': 'AI技術の最新トレンド',
                'search_terms': ['AI trends 2025', 'artificial intelligence news', 'machine learning updates'],
                'target_sites': ['https://www.google.com', 'https://news.ycombinator.com']
            },
            'programming_best_practices': {
                'topic': 'プログラミングのベストプラクティス',
                'search_terms': ['programming best practices', 'clean code principles', 'software development'],
                'target_sites': ['https://stackoverflow.com', 'https://github.com']
            },
            'web_development': {
                'topic': 'Web開発の最新動向',
                'search_terms': ['web development trends', 'frontend frameworks', 'backend technologies'],
                'target_sites': ['https://developer.mozilla.org', 'https://css-tricks.com']
            },
            'data_science': {
                'topic': 'データサイエンスの手法',
                'search_terms': ['data science methods', 'machine learning algorithms', 'data analysis'],
                'target_sites': ['https://kaggle.com', 'https://towardsdatascience.com']
            }
        }
        
        # ログ設定
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(f'multi_agent_web_learning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.stop_requested = False

    def setup_signal_handlers(self):
        """シグナルハンドラー設定"""
        def signal_handler(signum, frame):
            self.logger.info("停止シグナルを受信しました。安全に停止中...")
            self.stop_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def initialize_agents(self):
        """4つのエージェントを初期化（Playwright MCP対応）"""
        self.logger.info("Playwright MCP対応エージェントを初期化中...")
        
        try:
            for agent_id, role_info in self.agent_roles.items():
                self.logger.info(f"エージェント '{agent_id}' ({role_info['name']}) を初期化中...")
                
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
                    'web_research_count': 0,
                    'last_response': None,
                    'last_web_research': None,
                    'learning_data': []
                }
                
                # 統計初期化
                self.learning_stats['agent_interactions'][agent_id] = {
                    'messages_sent': 0,
                    'web_researches': 0,
                    'learning_cycles': 0,
                    'knowledge_contributions': 0,
                    'pages_visited': 0
                }
                
                self.logger.info(f"エージェント '{agent_id}' 初期化完了")
            
            # Playwright MCP接続テスト
            await self.test_playwright_mcp()
            
            self.logger.info("全エージェント初期化完了")
            return True
            
        except Exception as e:
            self.logger.error(f"エージェント初期化エラー: {e}")
            return False

    async def test_playwright_mcp(self):
        """Playwright MCP接続テスト"""
        self.logger.info("Playwright MCP接続テスト中...")
        
        try:
            # 最初のエージェントでテスト
            first_agent = list(self.agents.values())[0]
            
            # ブラウザスナップショット取得テスト
            test_prompt = """
Playwright MCPを使用してブラウザの動作確認を行ってください。
以下の手順を実行してください：
1. ブラウザのスナップショットを取得
2. 動作確認の結果を報告

簡潔に結果を報告してください。
"""
            
            response = await first_agent['manager'].process_message(
                user_input=test_prompt,
                session_id=f"playwright_test_{int(time.time())}"
            )
            
            if 'playwright' in str(response).lower() or 'browser' in str(response).lower():
                self.logger.info("✅ Playwright MCP接続確認")
            else:
                self.logger.warning("⚠️ Playwright MCP応答が不明確")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Playwright MCPテストエラー: {e}")

    async def shutdown_agents(self):
        """全エージェントの終了処理"""
        self.logger.info("全エージェントを終了中...")
        
        # ブラウザを閉じる
        try:
            await self.close_all_browsers()
        except Exception as e:
            self.logger.warning(f"ブラウザ終了エラー: {e}")
        
        for agent_id, agent_data in self.agents.items():
            try:
                await agent_data['manager'].shutdown()
                await agent_data['db_manager'].close()
                self.logger.info(f"エージェント '{agent_id}' 終了完了")
            except Exception as e:
                self.logger.error(f"エージェント '{agent_id}' 終了エラー: {e}")
        
        self.logger.info("全エージェント終了完了")

    async def close_all_browsers(self):
        """全ブラウザを閉じる"""
        self.logger.info("全ブラウザを閉じています...")
        
        for agent_id, agent_data in self.agents.items():
            try:
                close_prompt = "Playwright MCPを使用してブラウザを閉じてください。"
                await agent_data['manager'].process_message(
                    user_input=close_prompt,
                    session_id=f"browser_close_{agent_id}_{int(time.time())}"
                )
            except Exception as e:
                self.logger.warning(f"エージェント {agent_id} のブラウザ終了エラー: {e}")

    async def conduct_web_research(self, agent_id: str, research_topic: Dict[str, Any]) -> Dict[str, Any]:
        """Web研究実行"""
        self.logger.info(f"エージェント {agent_id} がWeb研究開始: {research_topic['topic']}")
        
        agent_data = self.agents[agent_id]
        role = agent_data['role']
        
        # エージェントの役割に応じたWeb研究プロンプト生成
        web_research_prompt = f"""
あなたは{role['name']}として、以下のトピックについてWeb研究を行ってください。

【研究トピック】
{research_topic['topic']}

【あなたの専門性】
- 専門分野: {role['focus']}
- 担当タスク: {', '.join(role['web_tasks'])}

【指示】
1. Playwright MCPを使用してブラウザを開いてください
2. 以下の検索キーワードで情報を検索してください: {', '.join(research_topic['search_terms'][:2])}
3. 関連する情報を収集し、あなたの専門性を活かして分析してください
4. 重要な発見や洞察を200-300文字でまとめてください
5. 情報源のURLも含めて報告してください

効率的に情報収集を行い、質の高い洞察を提供してください。
"""
        
        try:
            start_time = time.time()
            
            response = await agent_data['manager'].process_message(
                user_input=web_research_prompt,
                session_id=f"web_research_{agent_id}_{int(time.time())}"
            )
            
            execution_time = time.time() - start_time
            
            research_result = {
                'agent_id': agent_id,
                'agent_name': role['name'],
                'topic': research_topic['topic'],
                'research_content': response.get('response', ''),
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat(),
                'tools_used': response.get('tools_used', []),
                'success': True
            }
            
            # 統計更新
            agent_data['web_research_count'] += 1
            agent_data['last_web_research'] = research_result
            self.learning_stats['agent_interactions'][agent_id]['web_researches'] += 1
            self.learning_stats['web_pages_visited'] += 1
            
            self.logger.info(f"Web研究完了: {agent_id} ({execution_time:.2f}秒)")
            
            return research_result
            
        except Exception as e:
            self.logger.error(f"Web研究エラー {agent_id}: {e}")
            return {
                'agent_id': agent_id,
                'agent_name': role['name'],
                'topic': research_topic['topic'],
                'research_content': f"エラー: {str(e)}",
                'execution_time': 0,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'success': False
            }

    async def generate_web_informed_conversation_prompt(self, agent_id: str, topic: str, 
                                                       web_research: Dict[str, Any] = None,
                                                       context: List[Dict] = None) -> str:
        """Web研究結果を含む会話プロンプト生成"""
        role = self.agent_roles[agent_id]
        
        base_prompt = f"""
あなたは{role['name']}として行動してください。

【あなたの特徴】
- 性格: {role['personality']}
- 専門分野: {role['focus']}
- 会話スタイル: {role['conversation_style']}

【現在の議題】
{topic}

"""
        
        # Web研究結果がある場合は追加
        if web_research and web_research.get('success'):
            base_prompt += f"""
【あなたが収集したWeb情報】
{web_research['research_content']}

上記のWeb研究結果を活用して議論に参加してください。
"""
        
        # 会話履歴がある場合は追加
        if context and len(context) > 0:
            base_prompt += "\n【これまでの会話】\n"
            for msg in context[-3:]:
                speaker = msg.get('agent_id', 'unknown')
                content = msg.get('content', '')
                base_prompt += f"{speaker}: {content}\n"
        
        base_prompt += """
【指示】
1. あなたの専門性とWeb研究結果を活かして議題について意見を述べてください
2. 他のエージェントとの建設的な対話を心がけてください
3. 新しい視点や洞察を提供してください
4. 必要に応じて追加のWeb検索を提案してください
5. 回答は200-300文字程度で簡潔にまとめてください

Web情報を基にした具体的で価値のある発言をしてください。
"""
        
        return base_prompt    async
 def agent_conversation_with_web(self, agent_id: str, prompt: str) -> Dict[str, Any]:
        """Web情報を活用したエージェント会話"""
        try:
            agent_data = self.agents[agent_id]
            start_time = time.time()
            
            response = await agent_data['manager'].process_message(
                user_input=prompt,
                session_id=f"web_conversation_{agent_id}_{int(time.time())}"
            )
            
            execution_time = time.time() - start_time
            
            conversation_data = {
                'agent_id': agent_id,
                'agent_name': agent_data['role']['name'],
                'prompt': prompt,
                'content': response.get('response', ''),
                'executio