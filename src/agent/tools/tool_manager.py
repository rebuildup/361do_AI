"""
Tool Manager
全ツールの統合管理と動的実行を提供するクラス
"""

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional, Union
from loguru import logger

from agent.tools.command_tool import CommandTool
from agent.tools.search_tool import SearchTool
from agent.tools.file_tool import FileTool
from agent.tools.learning_tool import LearningTool


class ToolManager:
    """統合ツール管理クラス"""

    def __init__(self, config, db_manager, ollama_client=None):
        self.config = config
        self.db_manager = db_manager
        self.ollama_client = ollama_client
        self.tools: Dict[str, Any] = {}
        self.tool_capabilities: Dict[str, Dict] = {}

    async def initialize(self):
        """全ツールの初期化"""
        logger.info("Initializing Tool Manager...")

        # 検索ツール
        if self.config.settings.enable_web_search:
            try:
                self.tools['search'] = SearchTool()
                await self.tools['search'].initialize()
                self.tool_capabilities['search'] = {
                    'name': 'Web検索',
                    'description': 'インターネット検索、最新情報の取得',
                    'methods': ['search', 'search_news', 'search_images'],
                    'keywords': ['検索', '調べて', '探して', '情報', '最新', 'search', 'find']
                }
                logger.info("SearchTool initialized successfully")
            except Exception as e:
                logger.warning(f"SearchTool initialization failed: {e}")

        # ファイルツール
        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
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
            self.tool_capabilities['file'] = {
                'name': 'ファイル操作',
                'description': 'ファイルの読み書き、データ管理',
                'methods': ['read_file', 'write_file', 'list_files', 'delete_file'],
                'keywords': ['ファイル', '読み', '書き', '保存', 'file', 'read', 'write']
            }
            logger.info("FileTool initialized successfully")
        except Exception as e:
            logger.error(f"FileTool initialization failed: {e}")

        # コマンドツール
        if getattr(self.config.settings, 'enable_command_execution', False):
            try:
                self.tools['command'] = CommandTool()
                await self.tools['command'].initialize()
                self.tool_capabilities['command'] = {
                    'name': 'システムコマンド実行',
                    'description': 'システム情報取得、コマンド実行',
                    'methods': ['execute_command', 'get_system_info'],
                    'keywords': ['実行', 'コマンド', 'システム', 'run', 'execute', 'systeminfo', 'tasklist']
                }
                logger.info("CommandTool initialized successfully")
            except Exception as e:
                logger.warning(f"CommandTool initialization failed: {e}")

        # 学習ツール
        if self.config.is_learning_enabled and self.ollama_client:
            try:
                self.tools['learning'] = LearningTool(
                    db_manager=self.db_manager,
                    config=self.config,
                    ollama_client=self.ollama_client
                )
                self.tool_capabilities['learning'] = {
                    'name': '学習システム',
                    'description': '学習データ管理、プロンプト最適化',
                    'methods': ['get_learning_data', 'add_custom_learning_data', 'get_learning_status'],
                    'keywords': ['学習', 'データ', 'プロンプト', '最適化', 'learning', 'data']
                }
                logger.info("LearningTool initialized successfully")
            except Exception as e:
                logger.warning(f"LearningTool initialization failed: {e}")

        logger.info(f"Tool Manager initialized with {len(self.tools)} tools")

    async def execute_tool_function(
        self,
        tool_name: str,
        method_name: str,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """ツール機能の動的実行"""
        try:
            if tool_name not in self.tools:
                return {
                    'success': False,
                    'error': f'Tool "{tool_name}" not available',
                    'available_tools': list(self.tools.keys())
                }

            tool = self.tools[tool_name]
            if not hasattr(tool, method_name):
                return {
                    'success': False,
                    'error': f'Method "{method_name}" not found in tool "{tool_name}"',
                    'available_methods': [m for m in dir(tool) if not m.startswith('_')]
                }

            method = getattr(tool, method_name)
            if asyncio.iscoroutinefunction(method):
                result = await method(*args, **kwargs)
            else:
                result = method(*args, **kwargs)

            return {
                'success': True,
                'result': result,
                'tool': tool_name,
                'method': method_name
            }

        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name}.{method_name} - {e}")
            return {
                'success': False,
                'error': str(e),
                'tool': tool_name,
                'method': method_name
            }

    def get_tool_suggestions(self, user_input: str) -> List[Dict[str, Any]]:
        """ユーザー入力に基づいてツール使用を提案"""
        suggestions = []
        user_lower = user_input.lower()

        for tool_name, capability in self.tool_capabilities.items():
            if tool_name not in self.tools:
                continue

            # キーワードマッチング
            keyword_matches = sum(1 for keyword in capability['keywords'] if keyword in user_lower)
            if keyword_matches > 0:
                suggestions.append({
                    'tool': tool_name,
                    'name': capability['name'],
                    'description': capability['description'],
                    'confidence': min(keyword_matches * 0.3, 1.0),
                    'methods': capability['methods']
                })

        # 信頼度順でソート
        suggestions.sort(key=lambda x: x['confidence'], reverse=True)
        return suggestions

    async def auto_execute_best_tool(self, user_input: str, context: Dict = None) -> Dict[str, Any]:
        """最適なツールを自動選択・実行"""
        suggestions = self.get_tool_suggestions(user_input)
        
        if not suggestions:
            return {
                'success': False,
                'error': 'No suitable tool found',
                'user_input': user_input
            }

        best_suggestion = suggestions[0]
        tool_name = best_suggestion['tool']
        
        # ツール固有の実行ロジック
        try:
            if tool_name == 'search':
                return await self._auto_execute_search(user_input)
            elif tool_name == 'command':
                return await self._auto_execute_command(user_input)
            elif tool_name == 'file':
                return await self._auto_execute_file(user_input, context)
            elif tool_name == 'learning':
                return await self._auto_execute_learning(user_input)
            else:
                return {
                    'success': False,
                    'error': f'Auto-execution not implemented for tool: {tool_name}'
                }

        except Exception as e:
            logger.error(f"Auto-execution failed for {tool_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'tool': tool_name
            }

    async def _auto_execute_search(self, user_input: str) -> Dict[str, Any]:
        """検索ツールの自動実行"""
        # 検索クエリを最適化
        optimized_query = self._optimize_search_query(user_input)
        
        result = await self.execute_tool_function('search', 'search', optimized_query)
        if result['success']:
            search_data = result['result']
            return {
                'success': True,
                'type': 'search',
                'query': optimized_query,
                'results': search_data.get('results', []),
                'total_results': search_data.get('total_results', 0)
            }
        return result

    async def _auto_execute_command(self, user_input: str) -> Dict[str, Any]:
        """コマンドツールの自動実行"""
        command = self._extract_command_from_input(user_input)
        
        if not command:
            return {
                'success': False,
                'error': 'Could not determine command to execute',
                'suggestion': 'Try: "run systeminfo" or "execute tasklist"'
            }

        result = await self.execute_tool_function('command', 'execute_command', command)
        if result['success']:
            cmd_result = result['result']
            return {
                'success': True,
                'type': 'command',
                'command': command,
                'output': cmd_result.get('stdout', ''),
                'error': cmd_result.get('stderr', ''),
                'return_code': cmd_result.get('return_code', 0)
            }
        return result

    async def _auto_execute_file(self, user_input: str, context: Dict = None) -> Dict[str, Any]:
        """ファイルツールの自動実行"""
        user_lower = user_input.lower()
        
        if 'read' in user_lower or '読み' in user_lower:
            # ファイル読み取り
            file_path = self._extract_file_path_from_input(user_input)
            if file_path:
                result = await self.execute_tool_function('file', 'read_file', file_path)
                if result['success']:
                    return {
                        'success': True,
                        'type': 'file_read',
                        'path': file_path,
                        'content': result['result'].get('content', '')
                    }
                return result
        
        elif 'write' in user_lower or '書き' in user_lower:
            # ファイル書き込み
            file_path, content = self._extract_file_write_params(user_input)
            if file_path and content:
                result = await self.execute_tool_function('file', 'write_file', file_path, content)
                return {
                    'success': result['success'],
                    'type': 'file_write',
                    'path': file_path,
                    'message': result.get('result', {}).get('message', '')
                }

        return {
            'success': False,
            'error': 'Could not determine file operation',
            'suggestion': 'Try: "read file path/to/file.txt" or "write file path/to/file.txt\\ncontent"'
        }

    async def _auto_execute_learning(self, user_input: str) -> Dict[str, Any]:
        """学習ツールの自動実行"""
        user_lower = user_input.lower()
        
        if '学習データ' in user_lower or 'learning data' in user_lower:
            if '一番古い' in user_lower or '最古' in user_lower:
                result = await self.execute_tool_function('learning', 'get_learning_data', limit=1000)
                if result['success']:
                    data_list = result['result'].get('data', [])
                    if data_list:
                        oldest = min(data_list, key=lambda x: x.get('created_at', ''))
                        return {
                            'success': True,
                            'type': 'learning_data',
                            'oldest_data': oldest
                        }
            else:
                result = await self.execute_tool_function('learning', 'get_learning_data', limit=10)
                if result['success']:
                    return {
                        'success': True,
                        'type': 'learning_data',
                        'data': result['result'].get('data', [])
                    }

        elif '学習状態' in user_lower or 'learning status' in user_lower:
            result = await self.execute_tool_function('learning', 'get_learning_status')
            return {
                'success': result['success'],
                'type': 'learning_status',
                'status': result.get('result', {})
            }

        return {
            'success': False,
            'error': 'Could not determine learning operation'
        }

    def _optimize_search_query(self, user_input: str) -> str:
        """検索クエリの最適化"""
        # 不要な語句を除去
        noise_words = [
            'について', '調べて', '検索して', '教えて', 'について教えて',
            'の情報', 'を調べて', 'について調べて', 'を検索'
        ]
        
        optimized = user_input
        for noise in noise_words:
            optimized = optimized.replace(noise, '')
        
        return optimized.strip()

    def _extract_command_from_input(self, user_input: str) -> Optional[str]:
        """ユーザー入力からコマンドを抽出"""
        user_lower = user_input.lower()
        
        # 直接的なコマンド指定
        if user_lower.startswith('run ') or user_lower.startswith('execute '):
            return user_input.split(None, 1)[1] if len(user_input.split(None, 1)) > 1 else None
        
        # コマンド推測
        command_mappings = {
            'システム情報': 'systeminfo',
            'system info': 'systeminfo',
            'systeminfo': 'systeminfo',
            'タスク': 'tasklist',
            'プロセス': 'tasklist',
            'tasklist': 'tasklist',
            'ディレクトリ': 'dir',
            'dir': 'dir',
            'ユーザー': 'whoami',
            'whoami': 'whoami',
            'ホスト名': 'hostname',
            'hostname': 'hostname',
            'ネットワーク': 'ipconfig',
            'ipconfig': 'ipconfig'
        }
        
        for keyword, command in command_mappings.items():
            if keyword in user_lower:
                return command
        
        return None

    def _extract_file_path_from_input(self, user_input: str) -> Optional[str]:
        """ユーザー入力からファイルパスを抽出"""
        # "read file path/to/file.txt" のようなパターンを検出
        patterns = [
            r'read\s+file\s+([^\s]+)',
            r'ファイル\s*([^\s]+)\s*を読み',
            r'([^\s]+\.txt|[^\s]+\.json|[^\s]+\.py|[^\s]+\.md)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None

    def _extract_file_write_params(self, user_input: str) -> tuple[Optional[str], Optional[str]]:
        """ユーザー入力からファイル書き込みパラメータを抽出"""
        # "write file path/to/file.txt\ncontent" のようなパターンを検出
        if '\n' in user_input:
            header, content = user_input.split('\n', 1)
            path_match = re.search(r'write\s+file\s+([^\s]+)', header, re.IGNORECASE)
            if path_match:
                return path_match.group(1), content
        
        return None, None

    def get_available_tools(self) -> Dict[str, Dict]:
        """利用可能なツール一覧を取得"""
        return {
            tool_name: {
                'available': tool_name in self.tools,
                'capability': self.tool_capabilities.get(tool_name, {}),
                'status': 'active' if tool_name in self.tools else 'unavailable'
            }
            for tool_name in self.tool_capabilities.keys()
        }

    async def close(self):
        """全ツールの終了処理"""
        for tool in self.tools.values():
            if hasattr(tool, 'close'):
                await tool.close()
        logger.info("Tool Manager closed")