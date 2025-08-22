"""
Agent Manager
エージェントの中核機能を管理するクラス
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp
from loguru import logger
import os

from agent.core.config import Config
from agent.core.database import DatabaseManager
from agent.core.ollama_client import OllamaClient
from agent.tools.file_tool import FileTool
from agent.tools.learning_tool import LearningTool
from agent.web_design.design_generator import WebDesignGenerator


class AgentManager:
    """エージェント管理クラス"""
    
    def __init__(self, config: Config, db_manager: DatabaseManager):
        self.config = config
        self.db = db_manager
        self.ollama_client = None
        self.tools = {}
        self.web_design_generator = None
        self.learning_tool = None
        self.active_sessions = {}
        
    async def initialize(self):
        """エージェントマネージャー初期化"""
        logger.info("Initializing Agent Manager...")
        
        # OLLAMAクライアント初期化
        self.ollama_client = OllamaClient(self.config.ollama_config)
        await self.ollama_client.initialize()
        
        # ツール初期化
        await self._initialize_tools()
        
        # 学習ツール初期化
        if self.config.is_learning_enabled:
            self.learning_tool = LearningTool(
                db_manager=self.db,
                config=self.config,
                ollama_client=self.ollama_client,
                agent_manager=self
            )
        
        # Webデザイン生成器初期化
        if self.config.is_web_design_enabled:
            self.web_design_generator = WebDesignGenerator(
                llm_client=self.ollama_client,
                db_manager=self.db,
                config=self.config
            )
            await self.web_design_generator.initialize()
        
        logger.info("Agent Manager initialized successfully")
    
    async def _initialize_tools(self):
        """ツール初期化"""
        # 検索ツール
        if self.config.settings.enable_web_search:
            try:
                from agent.tools.search_tool import SearchTool
                self.tools['search'] = SearchTool()
                await self.tools['search'].initialize()
            except Exception as e:
                logger.warning(f"SearchTool could not be initialized (optional): {e}")
        
        # ファイルツール
        # プロジェクトのルートディレクトリを取得
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        self.tools['file'] = FileTool(project_root=project_root)
        await self.tools['file'].initialize()
        
        logger.info(f"Initialized {len(self.tools)} tools")
    
    async def shutdown(self):
        """エージェントマネージャー終了処理"""
        logger.info("Shutting down Agent Manager...")
        
        if self.ollama_client:
            await self.ollama_client.close()
        
        for tool in self.tools.values():
            if hasattr(tool, 'close'):
                await tool.close()
        
        # 学習ツール終了処理
        if self.learning_tool:
            await self.learning_tool.stop_learning_system()
        
        logger.info("Agent Manager shutdown complete")
    
    async def process_message(
        self,
        user_input: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """メッセージ処理"""
        start_time = time.time()
        
        # セッション管理
        if not session_id:
            session_id = str(uuid.uuid4())
        
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                'created_at': datetime.now(),
                'message_count': 0,
                'context': []
            }
        
        session = self.active_sessions[session_id]
        session['message_count'] += 1
        
        try:
            # 意図分析
            intent = await self._analyze_intent(user_input, session['context'])
            
            # 適切なハンドラーを選択
            response = await self._route_to_handler(user_input, intent, session)
            
            # 応答時間計算
            response_time = time.time() - start_time
            
            # 会話記録
            conversation_id = await self.db.insert_conversation(
                session_id=session_id,
                user_input=user_input,
                agent_response=response['content'],
                response_time=response_time,
                context_data={
                    'intent': intent,
                    'tools_used': response.get('tools_used', []),
                    'message_count': session['message_count']
                }
            )
            
            # セッション更新
            session['context'].append({
                'user': user_input,
                'agent': response['content'],
                'intent': intent,
                'timestamp': datetime.now().isoformat()
            })
            
            # コンテキスト制限（最新10件まで）
            if len(session['context']) > 10:
                session['context'] = session['context'][-10:]
            
            return {
                'response': response['content'],
                'session_id': session_id,
                'conversation_id': conversation_id,
                'response_time': response_time,
                'intent': intent,
                'tools_used': response.get('tools_used', [])
            }
            
        except Exception as e:
            logger.error(f"Message processing failed: {e}")
            
            error_response = "申し訳ありませんが、処理中にエラーが発生しました。もう一度お試しください。"
            
            # エラーも記録
            await self.db.insert_conversation(
                session_id=session_id,
                user_input=user_input,
                agent_response=error_response,
                response_time=time.time() - start_time,
                context_data={'error': str(e)}
            )
            
            return {
                'response': error_response,
                'session_id': session_id,
                'error': str(e)
            }
    
    async def _analyze_intent(
        self,
        user_input: str,
        context: List[Dict]
    ) -> Dict[str, Any]:
        """ユーザー意図分析"""
        
        # コンテキスト情報を構築
        context_str = ""
        if context:
            recent_context = context[-3:]  # 最新3件のみ
            context_str = "\n".join([
                f"ユーザー: {item['user']}\nエージェント: {item['agent']}"
                for item in recent_context
            ])
        
        intent_prompt = f"""
        以下のユーザー入力の意図を分析してください。
        
        現在の会話コンテキスト:
        {context_str}
        
        ユーザー入力: {user_input}
        
        以下の観点から分析し、JSON形式で回答してください:
        1. primary_intent: 主要な意図 (general_chat, web_search, web_design, file_operation, self_edit, technical_help)
        2. confidence: 確信度 (0.0-1.0)
        3. entities: 抽出されたエンティティ
        4. requires_tools: 必要なツール (search, file, web_design, self_edit)
        5. complexity: 複雑度 (simple, medium, complex)
        
        回答例:
        {{
            "primary_intent": "web_design",
            "confidence": 0.9,
            "entities": ["ランディングページ", "レスポンシブ"],
            "requires_tools": ["web_design"],
            "complexity": "medium"
        }}
        """
        
        try:
            response = await self.ollama_client.generate(
                prompt=intent_prompt,
                max_tokens=300,
                temperature=0.1
            )
            
            # JSONパース試行
            import json
            intent = json.loads(response)
            
            return intent
            
        except Exception as e:
            logger.error(f"Intent analysis failed: {e}")
            
            # フォールバック: シンプルな意図分析
            return await self._simple_intent_analysis(user_input)
    
    async def _simple_intent_analysis(self, user_input: str) -> Dict[str, Any]:
        """シンプルな意図分析（フォールバック）"""
        user_lower = user_input.lower()
        
        # Webデザイン関連キーワード
        web_design_keywords = [
            'ウェブサイト', 'webサイト', 'ホームページ', 'ランディング',
            'デザイン', 'html', 'css', 'レスポンシブ'
        ]
        
        # 検索関連キーワード
        search_keywords = [
            '検索', '調べて', '探して', '情報', '最新'
        ]
        
        if any(keyword in user_lower for keyword in web_design_keywords):
            return {
                "primary_intent": "web_design",
                "confidence": 0.8,
                "entities": [],
                "requires_tools": ["web_design"],
                "complexity": "medium"
            }
        elif any(keyword in user_lower for keyword in search_keywords):
            return {
                "primary_intent": "web_search",
                "confidence": 0.7,
                "entities": [],
                "requires_tools": ["search"],
                "complexity": "simple"
            }
        else:
            return {
                "primary_intent": "general_chat",
                "confidence": 0.6,
                "entities": [],
                "requires_tools": [],
                "complexity": "simple"
            }
    
    async def _route_to_handler(
        self,
        user_input: str,
        intent: Dict[str, Any],
        session: Dict
    ) -> Dict[str, Any]:
        """適切なハンドラーにルーティング"""
        
        primary_intent = intent.get('primary_intent', 'general_chat')
        tools_used = []
        
        if primary_intent == 'web_design' and self.web_design_generator:
            response = await self._handle_web_design(user_input, session)
            tools_used.append('web_design')
            
        elif primary_intent == 'self_edit' and 'file' in self.tools:
            response = await self._handle_self_edit(user_input, session)
            tools_used.append('file')

        elif primary_intent == 'web_search' and 'search' in self.tools:
            response = await self._handle_web_search(user_input, session)
            tools_used.append('search')
            
        else:
            response = await self._handle_general_chat(user_input, session, intent)
        
        return {
            'content': response,
            'tools_used': tools_used
        }
    
    async def _handle_web_design(
        self,
        user_input: str,
        session: Dict
    ) -> str:
        """Webデザイン処理"""
        try:
            design_result = await self.web_design_generator.generate_design(
                requirements=user_input,
                session_context=session.get('context', [])
            )
            
            return design_result.get('summary', 'Webデザインを生成しました。')
            
        except Exception as e:
            logger.error(f"Web design generation failed: {e}")
            return "Webデザインの生成中にエラーが発生しました。要件を再度お聞かせください。"
    
    async def _handle_web_search(
        self,
        user_input: str,
        session: Dict
    ) -> str:
        """Web検索処理"""
        try:
            search_tool = self.tools['search']
            search_results = await search_tool.search(user_input)
            
            # 検索結果を要約
            summary_prompt = f"""
            以下の検索結果を基に、ユーザーの質問に答えてください。
            
            ユーザーの質問: {user_input}
            
            検索結果:
            {search_results}
            
            検索結果を参考に、正確で有用な回答を提供してください。
            情報源も含めて回答してください。
            """
            
            response = await self.ollama_client.generate(
                prompt=summary_prompt,
                max_tokens=800,
                temperature=0.3
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return "検索中にエラーが発生しました。別の検索語句で試してみてください。"

    async def _handle_self_edit(
        self,
        user_input: str,
        session: Dict
    ) -> str:
        """自己修正 / ファイル・プロンプト・学習データの簡易操作ハンドラ

        サポートするコマンド（簡易）:
        - read file <path>
        - write file <path>\n<content>
        - append file <path>\n<content>
        - update prompt <name>: <content>
        - add learning data: <content>  (カテゴリは任意に指定可能)
        """
        try:
            file_tool = self.tools.get('file')
            lt = self.learning_tool

            text = user_input.strip()
            lower = text.lower()

            # read file
            if lower.startswith('read file'):
                parts = text.split(None, 2)
                if len(parts) < 3:
                    return "読み取り先のファイルパスが指定されていません。例: read file src/path/file.txt"
                path = parts[2].strip()
                if not file_tool:
                    return "ファイルツールが利用できません。"
                result = await file_tool.read_file(path)
                if result.get('error'):
                    return f"ファイル読み取りエラー: {result['error']}"
                return result.get('content', '')

            # write file (expects newline-separated: write file <path>\n<content>)
            if lower.startswith('write file'):
                # split into header and body
                header, _, body = text.partition('\n')
                parts = header.split(None, 2)
                if len(parts) < 3:
                    return "書き込み先のファイルパスが指定されていません。例: write file src/path/file.txt\nコンテンツ"
                path = parts[2].strip()
                if not file_tool:
                    return "ファイルツールが利用できません。"
                write_res = await file_tool.write_file(path, body)
                if write_res.get('error'):
                    return f"ファイル書き込みエラー: {write_res['error']}"
                return write_res.get('message', 'ファイルを書き込みました。')

            # append file
            if lower.startswith('append file'):
                header, _, body = text.partition('\n')
                parts = header.split(None, 2)
                if len(parts) < 3:
                    return "追記先のファイルパスが指定されていません。例: append file src/path/file.txt\n追記内容"
                path = parts[2].strip()
                if not file_tool:
                    return "ファイルツールが利用できません。"
                # read existing
                existing = await file_tool.read_file(path)
                if existing.get('error'):
                    # if file doesn't exist, create it
                    new_content = body
                else:
                    new_content = existing.get('content', '') + '\n' + body
                write_res = await file_tool.write_file(path, new_content)
                if write_res.get('error'):
                    return f"ファイル追記エラー: {write_res['error']}"
                return write_res.get('message', 'ファイルに追記しました。')

            # update prompt: update prompt <name>: <content>
            if lower.startswith('update prompt') or lower.startswith('set prompt'):
                # accept formats: update prompt name: content
                m = text.split(':', 1)
                if len(m) < 2:
                    return "プロンプト更新のフォーマットが不正です。例: update prompt greeting_prompt: 新しい内容"
                left = m[0]
                content = m[1].strip()
                parts = left.split(None, 2)
                if len(parts) < 3:
                    return "プロンプト名が指定されていません。例: update prompt greeting_prompt: 新しい内容"
                name = parts[2].strip()
                if not lt:
                    return "学習ツールが利用できません。"
                res = await lt.update_prompt_template(name=name, content=content)
                return res.get('message', str(res))

            # add learning data: either JSON or plain text after the colon
            if lower.startswith('add learning data') or lower.startswith('add learning'):
                # allow: add learning data: {json} or add learning data: content text
                _, _, rest = text.partition(':')
                payload = rest.strip()
                if not payload:
                    return "追加する学習データの内容が指定されていません。例: add learning data: 学習内容"
                # try to parse JSON
                try:
                    data_obj = json.loads(payload)
                    content = data_obj.get('content') or data_obj.get('text') or json.dumps(data_obj, ensure_ascii=False)
                    category = data_obj.get('category', 'custom')
                    tags = data_obj.get('tags', [])
                except Exception:
                    content = payload
                    category = 'custom'
                    tags = []

                if not lt:
                    return "学習ツールが利用できません。"
                add_res = await lt.add_custom_learning_data(content=content, category=category, tags=tags)
                return add_res.get('message', str(add_res))

            return "サポートされていない自己編集コマンドです。"

        except Exception as e:
            logger.error(f"Self-edit handler failed: {e}")
            return f"自己編集処理でエラーが発生しました: {e}"
    
    async def _handle_general_chat(
        self,
        user_input: str,
        session: Dict,
        intent: Dict
    ) -> str:
        """一般的な会話処理"""
        
        # 学習システムから適用すべきルールとプロンプトを取得
        learned_rules = await self._get_learned_conversational_rules()
        optimized_prompt = await self._get_optimized_system_prompt()
        
        # システムプロンプト取得（最適化されたものを優先）
        system_prompt_template = await self.db.get_prompt_template("system_prompt")
        base_system_prompt = system_prompt_template['template_content'] if system_prompt_template else ""
        
        # 最適化されたプロンプトがあれば使用
        if optimized_prompt:
            system_prompt = optimized_prompt
        else:
            system_prompt = base_system_prompt
        
        # 学習された会話ルールをシステムプロンプトに追加
        if learned_rules:
            rules_text = "\n".join([f"- {rule}" for rule in learned_rules])
            system_prompt += f"\n\n適用すべき会話ルール:\n{rules_text}"
        
        # コンテキスト構築
        context = session.get('context', [])
        context_str = ""
        if context:
            recent_context = context[-5:]  # 最新5件
            context_str = "\n".join([
                f"ユーザー: {item['user']}\nエージェント: {item['agent']}"
                for item in recent_context
            ])
        
        # 知識ベースから関連知識を取得
        relevant_knowledge = await self._get_relevant_knowledge(user_input)
        knowledge_str = ""
        if relevant_knowledge:
            knowledge_str = "\n".join([
                f"- {item['content']}"
                for item in relevant_knowledge[:3]  # 上位3件
            ])
        
        # プロンプト構築
        full_prompt = f"""
        {system_prompt}
        
        関連する知識:
        {knowledge_str}
        
        会話履歴:
        {context_str}
        
        ユーザー: {user_input}
        
        エージェント:"""
        
        try:
            response = await self.ollama_client.generate(
                prompt=full_prompt,
                max_tokens=1000,
                temperature=0.7
            )
            
            # 学習システムに会話データを記録
            await self._record_conversation_for_learning(user_input, response, session)
            
            return response
            
        except Exception as e:
            logger.error(f"General chat processing failed: {e}")
            return "申し訳ありませんが、応答の生成中にエラーが発生しました。"
    
    async def _get_learned_conversational_rules(self) -> List[str]:
        """学習された会話ルールを取得"""
        try:
            if not self.learning_tool:
                return []
            
            # 学習データから会話ルールを取得
            learning_data = await self.learning_tool.get_learning_data(category="conversation_rules")
            
            rules = []
            for item in learning_data.get('data', []):
                if item.get('quality_score', 0) > 0.7:  # 高品質なルールのみ
                    try:
                        # JSONとしてパースを試行
                        rule_data = json.loads(item['content'])
                        if isinstance(rule_data, dict):
                            rules.append(rule_data.get('rule', item['content']))
                        else:
                            rules.append(item['content'])
                    except json.JSONDecodeError:
                        # JSONでない場合はそのまま使用
                        rules.append(item['content'])
            
            return rules[:5]  # 上位5件まで
            
        except Exception as e:
            logger.error(f"Failed to get learned rules: {e}")
            return []
    
    async def _get_optimized_system_prompt(self) -> Optional[str]:
        """最適化されたシステムプロンプトを取得"""
        try:
            if not self.learning_tool:
                return None
            
            # プロンプト最適化履歴から最新の最適化されたプロンプトを取得
            # ここでは簡易的に学習データから取得
            learning_data = await self.learning_tool.get_learning_data(category="prompt_optimization")
            
            for item in learning_data.get('data', []):
                if item.get('quality_score', 0) > 0.8:  # 高品質なプロンプトのみ
                    return item['content']
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get optimized prompt: {e}")
            return None
    
    async def _record_conversation_for_learning(
        self,
        user_input: str,
        agent_response: str,
        session: Dict
    ):
        """学習のための会話データを記録"""
        try:
            if not self.learning_tool:
                return
            
            # 会話データを学習システムに送信
            conversation_data = {
                'user_input': user_input,
                'agent_response': agent_response,
                'session_context': session.get('context', []),
                'timestamp': datetime.now().isoformat()
            }
            
            # 学習データとして追加
            await self.learning_tool.add_custom_learning_data(
                content=json.dumps(conversation_data, ensure_ascii=False),
                category="conversation_history",
                quality_score=0.8,
                tags=["conversation", "learning"]
            )
            
        except Exception as e:
            logger.error(f"Failed to record conversation for learning: {e}")
    
    async def _get_relevant_knowledge(self, user_input: str) -> List[Dict]:
        """関連する知識をデータベースから取得"""
        try:
            # 簡単なキーワードマッチング（将来的にはベクトル検索に改善）
            knowledge_items = await self.db.get_active_knowledge()
            
            relevant = []
            user_lower = user_input.lower()
            
            for item in knowledge_items:
                content_lower = item['content'].lower()
                # シンプルなキーワードマッチング
                if any(word in content_lower for word in user_lower.split() if len(word) > 2):
                    relevant.append(item)
            
            # 信頼度順でソート
            relevant.sort(key=lambda x: x['confidence_score'], reverse=True)
            
            return relevant
            
        except Exception as e:
            logger.error(f"Knowledge retrieval failed: {e}")
            return []
    
    async def generate_web_design(self, requirements: str) -> Dict[str, Any]:
        """Webデザイン生成（外部API用）"""
        if not self.web_design_generator:
            raise ValueError("Web design functionality is not enabled")
        
        return await self.web_design_generator.generate_design(requirements)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """システムステータス取得"""
        try:
            # OLLAMA接続確認
            ollama_status = await self.ollama_client.health_check()
            
            # データベース統計取得
            db_stats = await self.db.get_performance_metrics()
            
            # アクティブセッション数
            active_sessions = len(self.active_sessions)
            
            # ツールステータス
            tools_status = {}
            for name, tool in self.tools.items():
                if hasattr(tool, 'get_status'):
                    tools_status[name] = await tool.get_status()
                else:
                    tools_status[name] = "active"
            
            return {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'ollama_status': ollama_status,
                'database_stats': db_stats,
                'active_sessions': active_sessions,
                'tools_status': tools_status,
                'web_design_enabled': self.config.is_web_design_enabled,
                'learning_enabled': self.config.is_learning_enabled
            }
            
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
