"""
Tool Registry

ツールの登録と管理を行うシステム
"""

import logging
from typing import Dict, Any, List, Optional, Type, Union
from datetime import datetime
import asyncio
from abc import ABC, abstractmethod

from langchain.tools import BaseTool

from .web_search import WebSearchTool, WebScrapingTool, NewsSearchTool
from .command_executor import CommandExecutorTool, PythonExecutorTool, SystemInfoTool
from .file_manager import FileManagerTool, TextProcessorTool
from .mcp_client import MCPManager, MCPTool, MCPResource


class ToolRegistry:
    """ツール登録システム"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.tool_metadata: Dict[str, Dict[str, Any]] = {}
        self.mcp_manager = MCPManager()
        self.logger = logging.getLogger(__name__)
        
        # デフォルトツールを登録
        self._register_default_tools()
    
    def _register_default_tools(self):
        """デフォルトツールを登録"""
        
        try:
            # Web検索ツール
            self.register_tool(WebSearchTool(search_engine="duckduckgo"), {
                "category": "web",
                "description": "インターネット検索",
                "version": "1.0.0",
                "author": "advanced-agent"
            })
            
            # Webスクレイピングツール
            self.register_tool(WebScrapingTool(), {
                "category": "web",
                "description": "Webページスクレイピング",
                "version": "1.0.0",
                "author": "advanced-agent"
            })
            
            # ニュース検索ツール
            self.register_tool(NewsSearchTool(), {
                "category": "web",
                "description": "ニュース検索",
                "version": "1.0.0",
                "author": "advanced-agent"
            })
            
            # コマンド実行ツール
            self.register_tool(CommandExecutorTool(), {
                "category": "system",
                "description": "システムコマンド実行",
                "version": "1.0.0",
                "author": "advanced-agent"
            })
            
            # Python実行ツール
            self.register_tool(PythonExecutorTool(), {
                "category": "system",
                "description": "Pythonコード実行",
                "version": "1.0.0",
                "author": "advanced-agent"
            })
            
            # システム情報ツール
            self.register_tool(SystemInfoTool(), {
                "category": "system",
                "description": "システム情報取得",
                "version": "1.0.0",
                "author": "advanced-agent"
            })
            
            # ファイル管理ツール
            self.register_tool(FileManagerTool(), {
                "category": "file",
                "description": "ファイル・ディレクトリ操作",
                "version": "1.0.0",
                "author": "advanced-agent"
            })
            
            # テキスト処理ツール
            self.register_tool(TextProcessorTool(), {
                "category": "file",
                "description": "テキスト処理",
                "version": "1.0.0",
                "author": "advanced-agent"
            })
            
            self.logger.info("Default tools registered successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to register default tools: {e}")
    
    def register_tool(self, tool: BaseTool, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """ツールを登録"""
        
        try:
            if not isinstance(tool, BaseTool):
                self.logger.error("Tool must be an instance of BaseTool")
                return False
            
            tool_name = tool.name
            if tool_name in self.tools:
                self.logger.warning(f"Tool already registered: {tool_name}")
                return False
            
            self.tools[tool_name] = tool
            self.tool_metadata[tool_name] = metadata or {}
            self.tool_metadata[tool_name].update({
                "registered_at": datetime.now().isoformat(),
                "type": type(tool).__name__
            })
            
            self.logger.info(f"Tool registered: {tool_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register tool: {e}")
            return False
    
    def unregister_tool(self, tool_name: str) -> bool:
        """ツールを登録解除"""
        
        try:
            if tool_name not in self.tools:
                self.logger.warning(f"Tool not found: {tool_name}")
                return False
            
            del self.tools[tool_name]
            del self.tool_metadata[tool_name]
            
            self.logger.info(f"Tool unregistered: {tool_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister tool: {e}")
            return False
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """ツールを取得"""
        
        return self.tools.get(tool_name)
    
    def get_tool_metadata(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """ツールメタデータを取得"""
        
        return self.tool_metadata.get(tool_name)
    
    def list_tools(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """ツール一覧を取得"""
        
        tools = []
        for tool_name, tool in self.tools.items():
            metadata = self.tool_metadata.get(tool_name, {})
            
            if category and metadata.get("category") != category:
                continue
            
            tools.append({
                "name": tool_name,
                "description": tool.description,
                "category": metadata.get("category", "unknown"),
                "version": metadata.get("version", "1.0.0"),
                "author": metadata.get("author", "unknown"),
                "registered_at": metadata.get("registered_at"),
                "type": metadata.get("type", "unknown")
            })
        
        return tools
    
    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """カテゴリ別にツールを取得"""
        
        tools = []
        for tool_name, tool in self.tools.items():
            metadata = self.tool_metadata.get(tool_name, {})
            if metadata.get("category") == category:
                tools.append(tool)
        
        return tools
    
    def search_tools(self, query: str) -> List[Dict[str, Any]]:
        """ツールを検索"""
        
        query_lower = query.lower()
        results = []
        
        for tool_name, tool in self.tools.items():
            metadata = self.tool_metadata.get(tool_name, {})
            
            # 名前、説明、カテゴリで検索
            if (query_lower in tool_name.lower() or
                query_lower in tool.description.lower() or
                query_lower in metadata.get("category", "").lower()):
                
                results.append({
                    "name": tool_name,
                    "description": tool.description,
                    "category": metadata.get("category", "unknown"),
                    "version": metadata.get("version", "1.0.0"),
                    "author": metadata.get("author", "unknown"),
                    "registered_at": metadata.get("registered_at"),
                    "type": metadata.get("type", "unknown")
                })
        
        return results
    
    async def add_mcp_server(self, 
                           server_name: str,
                           server_command: List[str],
                           server_args: Optional[List[str]] = None,
                           timeout: int = 30,
                           working_directory: Optional[str] = None) -> bool:
        """MCPサーバーを追加"""
        
        try:
            success = await self.mcp_manager.add_server(
                server_name=server_name,
                server_command=server_command,
                server_args=server_args,
                timeout=timeout,
                working_directory=working_directory
            )
            
            if success:
                # MCPツールとリソースを登録
                await self._register_mcp_tools(server_name)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to add MCP server: {e}")
            return False
    
    async def remove_mcp_server(self, server_name: str) -> bool:
        """MCPサーバーを削除"""
        
        try:
            success = await self.mcp_manager.remove_server(server_name)
            
            if success:
                # MCPツールとリソースを登録解除
                await self._unregister_mcp_tools(server_name)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to remove MCP server: {e}")
            return False
    
    async def _register_mcp_tools(self, server_name: str):
        """MCPツールとリソースを登録"""
        
        try:
            # MCPツールを登録
            for tool_info in self.mcp_manager.get_available_tools():
                tool_name = tool_info["name"]
                mcp_tool = self.mcp_manager.get_tool(tool_name)
                
                if mcp_tool:
                    self.register_tool(mcp_tool, {
                        "category": "mcp",
                        "description": f"MCP tool from {server_name}",
                        "version": "1.0.0",
                        "author": "mcp-server",
                        "server": server_name
                    })
            
            # MCPリソースを登録
            for resource_info in self.mcp_manager.get_available_resources():
                resource_name = resource_info["name"]
                mcp_resource = self.mcp_manager.get_resource(resource_name)
                
                if mcp_resource:
                    self.register_tool(mcp_resource, {
                        "category": "mcp",
                        "description": f"MCP resource from {server_name}",
                        "version": "1.0.0",
                        "author": "mcp-server",
                        "server": server_name
                    })
            
            self.logger.info(f"MCP tools and resources registered for {server_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to register MCP tools: {e}")
    
    async def _unregister_mcp_tools(self, server_name: str):
        """MCPツールとリソースを登録解除"""
        
        try:
            # サーバーに関連するツールを削除
            tools_to_remove = []
            for tool_name, metadata in self.tool_metadata.items():
                if metadata.get("server") == server_name:
                    tools_to_remove.append(tool_name)
            
            for tool_name in tools_to_remove:
                self.unregister_tool(tool_name)
            
            self.logger.info(f"MCP tools and resources unregistered for {server_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to unregister MCP tools: {e}")
    
    def get_mcp_status(self) -> Dict[str, Any]:
        """MCP状態を取得"""
        
        return self.mcp_manager.get_stats()
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """登録システムの統計を取得"""
        
        categories = {}
        for metadata in self.tool_metadata.values():
            category = metadata.get("category", "unknown")
            categories[category] = categories.get(category, 0) + 1
        
        return {
            "total_tools": len(self.tools),
            "categories": categories,
            "mcp_servers": len(self.mcp_manager.clients),
            "mcp_tools": len([t for t in self.tools.values() if isinstance(t, (MCPTool, MCPResource))])
        }
    
    def validate_tool(self, tool: BaseTool) -> Dict[str, Any]:
        """ツールを検証"""
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # 基本チェック
            if not hasattr(tool, 'name') or not tool.name:
                validation_result["errors"].append("Tool name is required")
                validation_result["valid"] = False
            
            if not hasattr(tool, 'description') or not tool.description:
                validation_result["warnings"].append("Tool description is recommended")
            
            if not hasattr(tool, '_run') and not hasattr(tool, '_arun'):
                validation_result["errors"].append("Tool must implement _run or _arun method")
                validation_result["valid"] = False
            
            # 名前の重複チェック
            if tool.name in self.tools:
                validation_result["warnings"].append(f"Tool name '{tool.name}' already exists")
            
            # 実行テスト（オプション）
            try:
                if hasattr(tool, '_run'):
                    # 簡単なテスト実行
                    test_result = tool._run("test")
                    if not isinstance(test_result, str):
                        validation_result["warnings"].append("Tool should return string result")
            except Exception as e:
                validation_result["warnings"].append(f"Tool execution test failed: {str(e)}")
            
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {str(e)}")
            validation_result["valid"] = False
        
        return validation_result
    
    def export_tools_config(self) -> Dict[str, Any]:
        """ツール設定をエクスポート"""
        
        config = {
            "tools": {},
            "mcp_servers": {},
            "exported_at": datetime.now().isoformat()
        }
        
        # ツール設定
        for tool_name, metadata in self.tool_metadata.items():
            config["tools"][tool_name] = {
                "name": tool_name,
                "description": self.tools[tool_name].description,
                "metadata": metadata
            }
        
        # MCPサーバー設定
        for server_name, client in self.mcp_manager.clients.items():
            config["mcp_servers"][server_name] = {
                "server_command": client.server_command,
                "server_args": client.server_args,
                "timeout": client.timeout,
                "working_directory": client.working_directory
            }
        
        return config
    
    def import_tools_config(self, config: Dict[str, Any]) -> bool:
        """ツール設定をインポート"""
        
        try:
            # MCPサーバーを復元
            if "mcp_servers" in config:
                for server_name, server_config in config["mcp_servers"].items():
                    asyncio.create_task(self.add_mcp_server(
                        server_name=server_name,
                        server_command=server_config["server_command"],
                        server_args=server_config.get("server_args"),
                        timeout=server_config.get("timeout", 30),
                        working_directory=server_config.get("working_directory")
                    ))
            
            self.logger.info("Tools configuration imported successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import tools configuration: {e}")
            return False


class ToolExecutor:
    """ツール実行システム"""
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.logger = logging.getLogger(__name__)
        self.execution_history: List[Dict[str, Any]] = []
    
    async def execute_tool(self, 
                          tool_name: str, 
                          arguments: Dict[str, Any],
                          timeout: int = 30) -> Dict[str, Any]:
        """ツールを実行"""
        
        try:
            tool = self.registry.get_tool(tool_name)
            if not tool:
                return {
                    "success": False,
                    "error": f"Tool not found: {tool_name}",
                    "result": None
                }
            
            self.logger.info(f"Executing tool: {tool_name} with arguments: {arguments}")
            
            # 実行履歴に記録
            execution_record = {
                "tool_name": tool_name,
                "arguments": arguments,
                "started_at": datetime.now().isoformat(),
                "timeout": timeout
            }
            
            # ツール実行
            try:
                if hasattr(tool, '_arun'):
                    # 非同期実行
                    result = await asyncio.wait_for(
                        tool._arun(**arguments),
                        timeout=timeout
                    )
                else:
                    # 同期実行
                    result = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, tool._run, **arguments
                        ),
                        timeout=timeout
                    )
                
                execution_record.update({
                    "success": True,
                    "result": result,
                    "completed_at": datetime.now().isoformat()
                })
                
                self.execution_history.append(execution_record)
                
                return {
                    "success": True,
                    "error": None,
                    "result": result
                }
                
            except asyncio.TimeoutError:
                execution_record.update({
                    "success": False,
                    "error": f"Tool execution timeout ({timeout}s)",
                    "completed_at": datetime.now().isoformat()
                })
                
                self.execution_history.append(execution_record)
                
                return {
                    "success": False,
                    "error": f"Tool execution timeout ({timeout}s)",
                    "result": None
                }
            
        except Exception as e:
            execution_record.update({
                "success": False,
                "error": str(e),
                "completed_at": datetime.now().isoformat()
            })
            
            self.execution_history.append(execution_record)
            
            self.logger.error(f"Tool execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "result": None
            }
    
    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """実行履歴を取得"""
        
        return self.execution_history[-limit:]
    
    def clear_execution_history(self):
        """実行履歴をクリア"""
        
        self.execution_history.clear()
        self.logger.info("Execution history cleared")