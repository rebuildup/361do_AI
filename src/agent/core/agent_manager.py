"""
Agent Manager
エージェントの中核機能を管理するクラス
"""

import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from loguru import logger
import os

from agent.core.config import Config
from agent.core.database import DatabaseManager
from agent.core.ollama_client import OllamaClient
from agent.tools.file_tool import FileTool
from agent.tools.learning_tool import LearningTool
from agent.tools.command_tool import CommandTool
from agent.tools.tool_manager import ToolManager
from agent.tools.help_system import HelpSystem

# Codex互換エージェントのインポート
try:
    from codex_agent import CodexConfig, CodexAgentInterface
    CODEX_AGENT_AVAILABLE = True
except ImportError:
    CODEX_AGENT_AVAILABLE = False


class AgentManager:
    """エージェント管理クラス"""

    def __init__(self, config: Config, db_manager: DatabaseManager):
        self.config: Config = config
        self.db: DatabaseManager = db_manager
        self.ollama_client: Optional[OllamaClient] = None
        self.tools: Dict[str, Any] = {}
        self.tool_manager: Optional[ToolManager] = None
        self.help_system: Optional[HelpSystem] = None
        self.learning_tool: Optional[LearningTool] = None
        self.codex_agent: Optional[CodexAgentInterface] = None
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

    async def initialize(self):
        """エージェントマネージャー初期化"""
        logger.info("Initializing Agent Manager...")

        # Codex互換エージェントを使用する場合
        if self.config.is_codex_agent_enabled:
            logger.info("Using Codex-compatible agent mode")
            await self._initialize_codex_agent()
            return

        # 従来の複雑なエージェントシステム
        logger.info("Using traditional learning agent mode")
        
        # OLLAMAクライアント初期化
        self.ollama_client = OllamaClient(self.config.ollama_config)
        await self.ollama_client.initialize()

        # ツール初期化
        await self._initialize_tools()
        
        # 統合ツールマネージャー初期化
        self.tool_manager = ToolManager(self.config, self.db, self.ollama_client)
        await self.tool_manager.initialize()
        
        # ヘルプシステム初期化
        self.help_system = HelpSystem(self.tool_manager)

        # 学習ツール初期化
        if self.config.is_learning_enabled:
            self.learning_tool = LearningTool(
                db_manager=self.db,
                config=self.config,
                ollama_client=self.ollama_client,
                agent_manager=self
            )
            try:
                # 自動で学習システムを開始する
                await self.learning_tool.start_learning_system()
                logger.info("LearningTool started successfully")
            except Exception as e:
                logger.error(f"Failed to start LearningTool: {e}")

        # システムプロンプトの初期化
        await self._initialize_system_prompt()

        logger.info("Agent Manager initialized successfully")

    async def _initialize_codex_agent(self):
        """Codex互換エージェントの初期化"""
        if not CODEX_AGENT_AVAILABLE:
            raise RuntimeError("Codex agent not available. Please install codex_agent module.")
        
        try:
            # Codex設定を作成
            codex_config = CodexConfig(
                model=self.config.settings.ollama_model,
                ollama_base_url=self.config.settings.ollama_base_url,
                cwd=self.config.paths.base_dir
            )
            
            # Codexエージェントを初期化
            self.codex_agent = CodexAgentInterface(codex_config)
            await self.codex_agent.initialize()
            
            logger.info("Codex-compatible agent initialized successfully")
        
        except Exception as e:
            logger.error(f"Failed to initialize Codex agent: {e}")
            raise

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
        # restrict file operations to safe data directories from config.paths
        allowed_dirs = [
            self.config.paths.prompts_dir,
            self.config.paths.learning_data_dir,
        ]
        self.tools['file'] = FileTool(
            project_root=project_root,
            allowed_dirs=allowed_dirs,
            auto_apply=getattr(self.config.settings, 'auto_apply_self_edits', False),
            proposals_dir=self.config.paths.proposals_dir
        )
        await self.tools['file'].initialize()

        # コマンドツール（セキュリティ制限付き）
        if getattr(self.config.settings, 'enable_command_execution', False):
            try:
                self.tools['command'] = CommandTool()
                await self.tools['command'].initialize()
                logger.info("CommandTool initialized with security restrictions")
            except Exception as e:
                logger.warning(f"CommandTool could not be initialized: {e}")

        logger.info(f"Initialized {len(self.tools)} tools")

    async def shutdown(self):
        """エージェントマネージャー終了処理"""
        logger.info("Shutting down Agent Manager...")

        # Codexエージェント終了処理
        if self.codex_agent:
            await self.codex_agent.shutdown()

        if self.ollama_client:
            await self.ollama_client.close()

        for tool in self.tools.values():
            if hasattr(tool, 'close'):
                await tool.close()

        # 学習ツール終了処理
        if self.learning_tool:
            await self.learning_tool.stop_learning_system()

        logger.info("Agent Manager shutdown complete")

    async def apply_proposal(self, proposal_filename: str) -> Dict[str, Any]:
        """
        Apply a saved proposal by filename (located in config.paths.proposals_dir).
        Returns the result of the forced write.
        """
        try:
            import json
            proposal_path = os.path.join(self.config.paths.proposals_dir, proposal_filename)
            if not os.path.exists(proposal_path):
                return {"status": "error", "message": f"Proposal not found: {proposal_filename}"}

            with open(proposal_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)

            target = meta.get('target')
            content = meta.get('content')

            file_tool = self.tools.get('file')
            if not file_tool:
                return {"status": "error", "message": "File tool not available"}

            # force apply by writing directly to target path
            result = await file_tool.write_file(target, content)

            # If applied (status success), move proposal to applied subdir
            if result.get('status') == 'success':
                applied_dir = os.path.join(self.config.paths.proposals_dir, 'applied')
                os.makedirs(applied_dir, exist_ok=True)
                os.replace(proposal_path, os.path.join(applied_dir, proposal_filename))

            return result
        except Exception as e:
            logger.error(f"Failed to apply proposal {proposal_filename}: {e}")
            return {"status": "error", "message": str(e)}

    async def process_message(
        self,
        user_input: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """メッセージ処理"""
        start_time = time.time()

        # Codex互換エージェントを使用する場合
        if self.config.is_codex_agent_enabled and self.codex_agent:
            return await self._process_message_codex(user_input, session_id, start_time)

        # 従来の複雑なエージェントシステム
        return await self._process_message_traditional(user_input, session_id, start_time)

    async def _process_message_codex(
        self,
        user_input: str,
        session_id: Optional[str],
        start_time: float
    ) -> Dict[str, Any]:
        """Codex互換エージェントでのメッセージ処理"""
        try:
            # チャット形式でメッセージを処理
            messages = [{"role": "user", "content": user_input}]
            
            response = await self.codex_agent.chat(
                messages=messages,
                session_id=session_id
            )
            
            # 応答時間計算
            response_time = time.time() - start_time
            
            # レスポンス形式を統一
            if "choices" in response and response["choices"]:
                agent_response = response["choices"][0].get("text", "").strip()
            else:
                agent_response = "申し訳ありませんが、応答を生成できませんでした。"
            
            return {
                'response': agent_response,
                'session_id': response.get('session_id', session_id),
                'response_time': response_time,
                'agent_type': 'codex',
                'model_info': response.get('model_info', {}),
                'usage': response.get('usage', {})
            }
        
        except Exception as e:
            logger.error(f"Codex agent message processing failed: {e}")
            return {
                'response': f"エラーが発生しました: {str(e)}",
                'session_id': session_id,
                'response_time': time.time() - start_time,
                'error': str(e),
                'agent_type': 'codex'
            }

    async def _process_message_traditional(
        self,
        user_input: str,
        session_id: Optional[str],
        start_time: float
    ) -> Dict[str, Any]:
        """従来のエージェントシステムでのメッセージ処理"""
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
        1. primary_intent: 主要な意図 (general_chat, web_search, web_design, file_operation, self_edit, command_execution, technical_help)
        2. confidence: 確信度 (0.0-1.0)
        3. entities: 抽出されたエンティティ
        4. requires_tools: 必要なツール (search, file, web_design, self_edit, command)
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
            if not self.ollama_client:
                raise RuntimeError("LLM client is not initialized")
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
        """シンプルな意図分析（フォールバック）- 動的ツール提案システム"""
        user_lower = user_input.lower().strip()

        # 統合ツールマネージャーを使用した動的意図分析
        if self.tool_manager:
            tool_suggestions = self.tool_manager.get_tool_suggestions(user_input)
            
            if tool_suggestions:
                best_suggestion = tool_suggestions[0]
                tool_name = best_suggestion['tool']
                
                # ツール名を意図にマッピング
                intent_mapping = {
                    'search': 'web_search',
                    'command': 'command_execution', 
                    'file': 'file_operation',
                    'learning': 'learning_data_access'
                }
                
                primary_intent = intent_mapping.get(tool_name, 'tool_execution')
                
                return {
                    "primary_intent": primary_intent,
                    "confidence": best_suggestion['confidence'],
                    "entities": [],
                    "requires_tools": [tool_name],
                    "complexity": "simple",
                    "tool_suggestions": tool_suggestions,
                    "recommended_tool": best_suggestion
                }

        # フォールバック: 従来のキーワードベース分析
        # Self-edit / file commands
        if user_lower.startswith('read file') or user_lower.startswith('write file') or user_lower.startswith('append file'):
            return {
                "primary_intent": "file_operation",
                "confidence": 0.95,
                "entities": [],
                "requires_tools": ["file"],
                "complexity": "simple"
            }

        # Command execution heuristics
        command_keywords = [
            '実行', 'コマンド', 'run', 'execute', 'cmd', 'systeminfo', 'dir', 'ls', 'pwd',
            'tasklist', 'whoami', 'hostname', 'ipconfig', 'netstat', 'ping',
            'システム情報', 'プロセス', 'タスク', 'ネットワーク', 'ディレクトリ',
            '使用できますか', '実行できますか', '動作しますか', 'コマンドを', 'を実行'
        ]
        if any(keyword in user_lower for keyword in command_keywords):
            return {
                "primary_intent": "command_execution",
                "confidence": 0.8,
                "entities": [],
                "requires_tools": ["command"],
                "complexity": "simple"
            }

        # Web search heuristics
        search_keywords = [
            '検索', '調べて', '探して', '情報', '最新', 'search', 'find', 'look up',
            '結果が発表', '優勝者', 'について情報', 'ニュース', '発表された'
        ]
        if any(keyword in user_lower for keyword in search_keywords):
            return {
                "primary_intent": "web_search",
                "confidence": 0.7,
                "entities": [],
                "requires_tools": ["search"],
                "complexity": "simple"
            }

        # Content analysis
        analysis_keywords = [
            'まとめて', '説明して', '内容を', '結果を', 'について説明',
            'わかりやすく', '整理して', '分析して', '先程の', '先ほどの'
        ]
        if any(keyword in user_lower for keyword in analysis_keywords):
            if not any(cmd in user_lower for cmd in ['実行', 'run', 'execute', 'コマンド']):
                return {
                    "primary_intent": "content_analysis",
                    "confidence": 0.9,
                    "entities": [],
                    "requires_tools": [],
                    "complexity": "simple"
                }

        # Learning data access
        learning_keywords = [
            '学習データ', '学習機能', '一番古い', '最古の', 'データの中',
            '存在する学習', '学習した内容', '記憶している'
        ]
        if any(keyword in user_lower for keyword in learning_keywords):
            return {
                "primary_intent": "learning_data_access",
                "confidence": 0.8,
                "entities": [],
                "requires_tools": ["learning"],
                "complexity": "simple"
            }

        # Help system
        help_keywords = [
            'help', 'ヘルプ', '使い方', '機能一覧', '何ができる', 'できること',
            '使用方法', '操作方法', 'コマンド一覧', 'ツール一覧'
        ]
        if any(keyword in user_lower for keyword in help_keywords):
            return {
                "primary_intent": "help_request",
                "confidence": 0.9,
                "entities": [],
                "requires_tools": ["help"],
                "complexity": "simple"
            }

        # Default - 動的ツール実行を試行
        return {
            "primary_intent": "dynamic_tool_execution",
            "confidence": 0.5,
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
        """適切なハンドラーにルーティング - 動的ツール実行対応"""

        primary_intent = intent.get('primary_intent', 'general_chat')
        tools_used = []

        # 動的ツール実行を最優先で試行
        if primary_intent == 'dynamic_tool_execution' or intent.get('tool_suggestions'):
            if self.tool_manager:
                try:
                    result = await self.tool_manager.auto_execute_best_tool(user_input, session)
                    if result['success']:
                        response = await self._format_tool_result(result, user_input)
                        tools_used.append(result.get('type', 'unknown'))
                        return {
                            'content': response,
                            'tools_used': tools_used,
                            'tool_result': result
                        }
                    else:
                        # 動的実行が失敗した場合は従来のハンドラーにフォールバック
                        logger.warning(f"Dynamic tool execution failed: {result.get('error')}")
                except Exception as e:
                    logger.error(f"Dynamic tool execution error: {e}")

        # 従来のハンドラー
        if primary_intent == 'file_operation' and 'file' in self.tools:
            response = await self._handle_self_edit(user_input, session)
            tools_used.append('file')

        elif primary_intent == 'web_search' and 'search' in self.tools:
            response = await self._handle_web_search(user_input, session)
            tools_used.append('search')

        elif primary_intent == 'command_execution' and 'command' in self.tools:
            response = await self._handle_command_execution(user_input, session)
            tools_used.append('command')

        elif primary_intent == 'content_analysis':
            response = await self._handle_content_analysis(user_input, session)

        elif primary_intent == 'learning_data_access':
            response = await self._handle_learning_data_access(user_input, session)

        elif primary_intent == 'help_request':
            response = await self._handle_help_request(user_input, session)

        else:
            response = await self._handle_general_chat(user_input, session, intent)

        return {
            'content': response,
            'tools_used': tools_used
        }

    async def _format_tool_result(self, result: Dict[str, Any], user_input: str) -> str:
        """ツール実行結果を自然な形式でフォーマット"""
        try:
            result_type = result.get('type', 'unknown')
            
            if result_type == 'search':
                return await self._format_search_result(result, user_input)
            elif result_type == 'command':
                return await self._format_command_result(result, user_input)
            elif result_type == 'file_read':
                return await self._format_file_read_result(result, user_input)
            elif result_type == 'file_write':
                return await self._format_file_write_result(result, user_input)
            elif result_type == 'learning_data':
                return await self._format_learning_result(result, user_input)
            else:
                return f"ツール実行が完了しました。結果: {result}"

        except Exception as e:
            logger.error(f"Tool result formatting failed: {e}")
            return f"ツールの実行は成功しましたが、結果の表示でエラーが発生しました: {e}"

    async def _format_search_result(self, result: Dict[str, Any], user_input: str) -> str:
        """検索結果のフォーマット"""
        query = result.get('query', user_input)
        results = result.get('results', [])
        total = result.get('total_results', 0)
        
        if not results:
            return f"「{query}」の検索を実行しましたが、関連する結果が見つかりませんでした。"
        
        formatted = f"「{query}」の検索結果 ({total}件):\n\n"
        
        for i, item in enumerate(results[:5], 1):
            title = item.get('title', '無題')
            body = item.get('body', '')[:200]
            url = item.get('href', '')
            
            formatted += f"{i}. **{title}**\n"
            if body:
                formatted += f"   {body}...\n"
            if url:
                formatted += f"   URL: {url}\n"
            formatted += "\n"
        
        return formatted

    async def _format_command_result(self, result: Dict[str, Any], user_input: str) -> str:
        """コマンド実行結果のフォーマット"""
        command = result.get('command', '')
        output = result.get('output', '')
        error = result.get('error', '')
        return_code = result.get('return_code', 0)
        
        formatted = f"コマンド '{command}' の実行結果:\n\n"
        
        if return_code == 0:
            formatted += "```\n" + output + "\n```"
            if error:
                formatted += f"\n\n警告: {error}"
        else:
            formatted += f"エラー (終了コード: {return_code}):\n{error}"
        
        return formatted

    async def _format_file_read_result(self, result: Dict[str, Any], user_input: str) -> str:
        """ファイル読み取り結果のフォーマット"""
        path = result.get('path', '')
        content = result.get('content', '')
        
        if len(content) > 1000:
            content = content[:1000] + "\n\n... (内容が長いため省略されました)"
        
        return f"ファイル '{path}' の内容:\n\n```\n{content}\n```"

    async def _format_file_write_result(self, result: Dict[str, Any], user_input: str) -> str:
        """ファイル書き込み結果のフォーマット"""
        path = result.get('path', '')
        message = result.get('message', '')
        
        return f"ファイル '{path}' への書き込みが完了しました。\n{message}"

    async def _format_learning_result(self, result: Dict[str, Any], user_input: str) -> str:
        """学習データ結果のフォーマット"""
        if 'oldest_data' in result:
            data = result['oldest_data']
            return f"最古の学習データ:\n\n" \
                   f"作成日時: {data.get('created_at', '不明')}\n" \
                   f"カテゴリ: {data.get('category', '不明')}\n" \
                   f"内容: {data.get('content', '')[:200]}...\n" \
                   f"品質スコア: {data.get('quality_score', 0)}"
        elif 'data' in result:
            data_list = result['data']
            formatted = f"学習データ一覧 ({len(data_list)}件):\n\n"
            for i, item in enumerate(data_list[:5], 1):
                formatted += f"{i}. [{item.get('category', '')}] {item.get('content', '')[:50]}...\n"
            return formatted
        else:
            return "学習データの取得が完了しました。"

    async def _handle_web_search(
        self,
        user_input: str,
        session: Dict
    ) -> str:
        """Web検索処理"""
        try:
            search_tool = self.tools['search']
            
            # 検索クエリを最適化
            optimized_query = await self._optimize_search_query(user_input)
            logger.info(f"Optimized search query: {optimized_query}")
            
            search_results = await search_tool.search(optimized_query)

            # 検索結果を要約
            summary_prompt = f"""
            以下の検索結果を基に、ユーザーの質問に答えてください。

            ユーザーの質問: {user_input}

            検索結果:
            {search_results}

            検索結果を参考に、正確で有用な回答を提供してください。
            情報源も含めて回答してください。
            """

            if not self.ollama_client:
                raise RuntimeError("LLM client is not initialized")
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

    async def _handle_command_execution(
        self,
        user_input: str,
        session: Dict
    ) -> str:
        """コマンド実行処理"""
        try:
            command_tool = self.tools.get('command')
            if not command_tool:
                return "コマンドツールが利用できません。"

            # ユーザー入力からコマンドを抽出
            user_lower = user_input.lower().strip()
            
            # 直接的なコマンド指定の場合
            if user_lower.startswith('run ') or user_lower.startswith('execute '):
                command = user_input.split(None, 1)[1] if len(user_input.split(None, 1)) > 1 else ""
            elif user_lower.startswith('コマンド実行'):
                command = user_input.replace('コマンド実行', '').strip()
            else:
                # より詳細なコマンド推測
                if 'システム情報' in user_input or 'system info' in user_lower or 'systeminfo' in user_lower:
                    command = 'systeminfo'
                elif 'ディレクトリ' in user_input or 'dir' in user_lower or 'ls' in user_lower:
                    command = 'dir'
                elif 'ユーザー' in user_input or 'whoami' in user_lower:
                    command = 'whoami'
                elif 'ホスト名' in user_input or 'hostname' in user_lower:
                    command = 'hostname'
                elif 'tasklist' in user_lower or 'プロセス' in user_input or 'タスク' in user_input:
                    command = 'tasklist'
                elif 'ipconfig' in user_lower or 'ネットワーク' in user_input:
                    command = 'ipconfig'
                elif 'netstat' in user_lower:
                    command = 'netstat'
                elif 'ping' in user_lower:
                    # pingの場合は対象が必要なので、デフォルトでlocalhostをping
                    command = 'ping localhost -n 4'
                elif '使用できますか' in user_input or '実行できますか' in user_input or '動作しますか' in user_input:
                    # コマンドの可用性を確認する質問の場合
                    # 質問に含まれるコマンド名を抽出
                    for cmd in ['tasklist', 'systeminfo', 'dir', 'whoami', 'hostname', 'ipconfig', 'netstat', 'ping']:
                        if cmd in user_lower:
                            return f"はい、'{cmd}'コマンドは使用可能です。実行しますか？\n\n利用可能なコマンド一覧:\n" + \
                                   "\n".join([f"- {c}" for c in command_tool.get_allowed_commands()[:10]])
                    return "利用可能なコマンド一覧:\n" + "\n".join([f"- {c}" for c in command_tool.get_allowed_commands()[:10]])
                else:
                    return "実行するコマンドが明確ではありません。\n\n例:\n- 'run systeminfo'\n- 'システム情報を表示'\n- 'tasklistコマンドを実行'\n\n利用可能なコマンド: " + ", ".join(command_tool.get_allowed_commands()[:8])

            if not command:
                return "実行するコマンドが指定されていません。"

            # コマンド実行
            result = await command_tool.execute_command(command)
            
            if result['success']:
                output = result['stdout']
                if result['stderr']:
                    output += f"\n警告: {result['stderr']}"
                return f"コマンド '{command}' の実行結果:\n```\n{output}\n```"
            else:
                error_msg = result.get('error', result.get('stderr', '不明なエラー'))
                return f"コマンド '{command}' の実行に失敗しました: {error_msg}"

        except Exception as e:
            logger.error(f"Command execution handler failed: {e}")
            return f"コマンド実行処理でエラーが発生しました: {e}"

    async def _handle_content_analysis(
        self,
        user_input: str,
        session: Dict
    ) -> str:
        """コンテンツ分析・要約処理"""
        try:
            # 最近の会話履歴から関連するコマンド実行結果を探す
            context = session.get('context', [])
            recent_command_output = None
            
            # 最新5件の会話から最後のコマンド実行結果を探す
            for item in reversed(context[-5:]):
                if 'コマンド' in item.get('agent', '') and '実行結果:' in item.get('agent', ''):
                    recent_command_output = item['agent']
                    break
            
            if recent_command_output:
                # コマンド実行結果がある場合、それを分析
                analysis_prompt = f"""
以下のコマンド実行結果を分析し、わかりやすくまとめてください。

ユーザーの要求: {user_input}

コマンド実行結果:
{recent_command_output}

以下の観点で分析してください：
1. 実行されたコマンドの種類
2. 主要な結果の要約
3. 注目すべきポイント
4. システムの状態や特徴

技術的な詳細は適度に簡略化し、一般的に理解しやすい形で説明してください。
"""
            else:
                # コマンド実行結果がない場合、一般的な分析
                analysis_prompt = f"""
ユーザーの要求: {user_input}

会話履歴:
{chr(10).join([f"ユーザー: {item['user'][:100]}..." if len(item['user']) > 100 else f"ユーザー: {item['user']}" for item in context[-3:]])}

上記の内容について、ユーザーが求めている分析や説明を提供してください。
"""

            if not self.ollama_client:
                return "申し訳ありませんが、分析機能が利用できません。"

            response = await self.ollama_client.generate(
                prompt=analysis_prompt,
                max_tokens=800,
                temperature=0.3
            )

            return response

        except Exception as e:
            logger.error(f"Content analysis handler failed: {e}")
            return f"コンテンツ分析処理でエラーが発生しました: {e}"

    async def _handle_learning_data_access(
        self,
        user_input: str,
        session: Dict
    ) -> str:
        """学習データアクセス処理"""
        try:
            if not self.learning_tool:
                return "学習システムが利用できません。"

            user_lower = user_input.lower()
            
            if '一番古い' in user_lower or '最古の' in user_lower:
                # 最古の学習データを取得
                result = await self.learning_tool.get_learning_data(limit=1000)
                if result.get('status') == 'success':
                    data_list = result.get('data', [])
                    if data_list:
                        # 作成日時でソート（最古のものを取得）
                        oldest_data = min(data_list, key=lambda x: x.get('created_at', ''))
                        return f"最古の学習データ:\n\n" \
                               f"作成日時: {oldest_data.get('created_at', '不明')}\n" \
                               f"カテゴリ: {oldest_data.get('category', '不明')}\n" \
                               f"内容: {oldest_data.get('content', '')[:200]}...\n" \
                               f"品質スコア: {oldest_data.get('quality_score', 0)}"
                    else:
                        return "学習データが見つかりませんでした。"
                else:
                    return f"学習データの取得に失敗しました: {result.get('message', '')}"
            
            elif '学習データ' in user_lower:
                # 学習データの統計情報を取得
                stats = await self.db.get_learning_statistics()
                return f"学習データの統計情報:\n\n" \
                       f"総学習データ数: {stats.get('total_learning_data', 0)}件\n" \
                       f"知識アイテム数: {stats.get('total_knowledge_items', 0)}件\n" \
                       f"平均品質スコア: {stats.get('average_quality_score', 0):.2f}\n" \
                       f"高品質データ数: {stats.get('high_quality_count', 0)}件"
            
            else:
                return "学習機能に関する具体的な質問をお聞かせください。例：\n" \
                       "- 一番古い学習データについて\n" \
                       "- 学習データの統計情報\n" \
                       "- 学習システムの状態"

        except Exception as e:
            logger.error(f"Learning data access handler failed: {e}")
            return f"学習データアクセス処理でエラーが発生しました: {e}"

    async def _handle_help_request(
        self,
        user_input: str,
        session: Dict
    ) -> str:
        """ヘルプ要求処理"""
        try:
            if not self.help_system:
                return "ヘルプシステムが利用できません。"

            user_lower = user_input.lower()
            
            if 'tools' in user_lower or 'ツール' in user_lower:
                return self.help_system.get_tool_help()
            elif 'examples' in user_lower or '例' in user_lower or '使用例' in user_lower:
                return self.help_system.get_examples()
            elif 'commands' in user_lower or 'コマンド' in user_lower:
                if self.tool_manager:
                    tools = self.tool_manager.get_available_tools()
                    command_help = "📋 利用可能なコマンド:\n\n"
                    if 'command' in tools and tools['command']['available']:
                        command_tool = self.tool_manager.tools.get('command')
                        if command_tool:
                            allowed_commands = command_tool.get_allowed_commands()
                            command_help += "**システムコマンド:**\n"
                            command_help += ", ".join(allowed_commands[:10])
                            if len(allowed_commands) > 10:
                                command_help += f" (他{len(allowed_commands)-10}個)"
                    return command_help
                else:
                    return "コマンド情報が取得できません。"
            else:
                return self.help_system.get_comprehensive_help()

        except Exception as e:
            logger.error(f"Help request handler failed: {e}")
            return f"ヘルプ処理でエラーが発生しました: {e}"

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

        # ツール使用の指示を追加
        available_tools = []
        if 'search' in self.tools:
            available_tools.append("Web検索")
        if 'command' in self.tools:
            available_tools.append("システムコマンド実行")
        if 'file' in self.tools:
            available_tools.append("ファイル操作")

        if available_tools:
            tools_instruction = f"""

あなたは以下のツールを使用できます: {', '.join(available_tools)}

ユーザーが以下のような要求をした場合は、適切なツールを使用してください：
- システム情報やコマンド実行の要求 → コマンド実行ツールを使用
- 最新情報の検索や調査の要求 → Web検索ツールを使用
- ファイルの読み書きの要求 → ファイル操作ツールを使用

ツールを使用する場合は、その旨をユーザーに伝えてから実行してください。"""
            system_prompt += tools_instruction

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

        # ユーザー入力を再分析してツール使用の必要性を判断
        should_use_tools = await self._should_use_tools_for_general_chat(user_input)
        
        if should_use_tools:
            tool_suggestion = should_use_tools
            system_prompt += f"\n\n重要: ユーザーの要求に対して{tool_suggestion}の使用を検討してください。"

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
            if not self.ollama_client:
                raise RuntimeError("LLM client is not initialized")
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

    async def _should_use_tools_for_general_chat(self, user_input: str) -> Optional[str]:
        """一般的な会話でツール使用が必要かどうかを判断"""
        user_lower = user_input.lower()
        
        # コマンド実行が必要そうな場合
        command_hints = [
            'システム', 'プロセス', 'タスク', 'ネットワーク', 'ディレクトリ',
            'コンピュータ', 'PC', 'マシン', '情報を取得', '確認して',
            'tasklist', 'systeminfo', 'dir', 'whoami', 'hostname', 'ipconfig'
        ]
        if any(hint in user_lower for hint in command_hints):
            return "システムコマンド実行ツール"
        
        # Web検索が必要そうな場合
        search_hints = [
            '最新', '今日', '昨日', '最近', '調べて', '検索', '情報',
            'ニュース', '発表', '優勝', '結果', 'について教えて'
        ]
        if any(hint in user_lower for hint in search_hints):
            return "Web検索ツール"
        
        # ファイル操作が必要そうな場合
        file_hints = [
            'ファイル', '読み', '書き', '保存', 'テキスト', 'データ'
        ]
        if any(hint in user_lower for hint in file_hints):
            return "ファイル操作ツール"
        
        return None

    async def _initialize_system_prompt(self):
        """システムプロンプトの初期化"""
        try:
            # 既存のシステムプロンプトを確認
            existing_prompt = await self.db.get_prompt_template("system_prompt")
            
            if not existing_prompt:
                # システムプロンプトファイルから読み込み
                system_prompt_file = os.path.join(self.config.paths.prompts_dir, "system_prompt.txt")
                if os.path.exists(system_prompt_file):
                    with open(system_prompt_file, 'r', encoding='utf-8') as f:
                        system_prompt_content = f.read()
                    
                    # データベースに保存
                    await self.db.insert_prompt_template(
                        name="system_prompt",
                        template_content=system_prompt_content,
                        description="デフォルトシステムプロンプト",
                        category="system"
                    )
                    logger.info("System prompt initialized from file")
                else:
                    # デフォルトのシステムプロンプトを作成
                    default_prompt = """あなたは高度なAIエージェントです。ユーザーの要求に応じて、様々なツールを使用して情報を取得し、タスクを実行することができます。

基本的な行動指針:
1. ユーザーの要求を正確に理解し、適切なツールを選択して使用する
2. ツールを使用する前に、何を行うかをユーザーに説明する
3. 結果を分かりやすく整理して提示する
4. 常に丁寧で親切な対応を心がける

あなたは自律的にツールを使用して、ユーザーの要求に最適な回答を提供してください。"""
                    
                    await self.db.insert_prompt_template(
                        name="system_prompt",
                        template_content=default_prompt,
                        description="デフォルトシステムプロンプト",
                        category="system"
                    )
                    logger.info("Default system prompt created")
            else:
                logger.info("System prompt already exists")
                
        except Exception as e:
            logger.error(f"Failed to initialize system prompt: {e}")

    async def _optimize_search_query(self, user_input: str) -> str:
        """検索クエリを最適化"""
        try:
            # 重要なキーワードを抽出するための簡易ロジック
            user_lower = user_input.lower()
            
            # 検索に不要な語句を除去
            noise_words = [
                'ありがとうございます', '次は', '検索機能をテストします',
                'について情報を調べ', '教えて下さい', '教えてください',
                'について', 'を調べて', 'について調べて', 'の情報',
                'をテストします', 'をテスト', 'します'
            ]
            
            optimized = user_input
            for noise in noise_words:
                optimized = optimized.replace(noise, '')
            
            # 重要なキーワードを抽出
            important_keywords = []
            
            # 固有名詞や重要な語句を検出
            if 'ボカコレ' in optimized or 'ボーカロイド' in optimized:
                important_keywords.extend(['ボカコレ', 'ボーカロイド', '楽曲投稿祭'])
            
            if 'ルーキー部門' in optimized:
                important_keywords.append('ルーキー部門')
                
            if '優勝者' in optimized or '優勝' in optimized:
                important_keywords.append('優勝者')
                
            if '昨日' in optimized or '結果が発表' in optimized:
                important_keywords.extend(['2025年', '最新', '結果'])
            
            # キーワードがある場合はそれを使用、なければ元の文章をクリーンアップ
            if important_keywords:
                return ' '.join(important_keywords)
            else:
                # 不要な語句を除去した結果を返す
                return optimized.strip()
                
        except Exception as e:
            logger.error(f"Search query optimization failed: {e}")
            return user_input

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

    async def get_system_status(self) -> Dict[str, Any]:
        """システムステータス取得"""
        try:
            # OLLAMA接続確認
            if not self.ollama_client:
                ollama_status = {'status': 'uninitialized'}
            else:
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
                'learning_enabled': self.config.is_learning_enabled
            }

        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
