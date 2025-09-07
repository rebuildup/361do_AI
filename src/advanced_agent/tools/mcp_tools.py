"""
MCP Tools

Model Context Protocol ツールの実装
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json

from langchain.tools import BaseTool

from .mcp_client import MCPClient, MCPManager, MCPTool, MCPResource


class MCPToolWrapper(BaseTool):
    """MCPツールラッパー"""
    
    name: str = "mcp_tool_wrapper"
    description: str = "MCPツールのラッパー"
    
    def __init__(self, mcp_client: MCPClient, tool_name: str):
        super().__init__()
        self.mcp_client = mcp_client
        self.tool_name = tool_name
        self.name = f"mcp_{tool_name}"
        self.logger = logging.getLogger(__name__)
        
        # ツール情報を取得
        tool_info = mcp_client.get_tool_info(tool_name)
        if tool_info:
            self.description = tool_info.get("description", f"MCP tool: {tool_name}")
        else:
            self.description = f"MCP tool: {tool_name}"
    
    def _run(self, **kwargs) -> str:
        """同期実行"""
        return asyncio.run(self._arun(**kwargs))
    
    async def _arun(self, **kwargs) -> str:
        """非同期実行"""
        
        try:
            result = await self.mcp_client.call_tool(self.tool_name, kwargs)
            
            if "error" in result:
                return f"MCP tool error: {result['error']}"
            
            # 結果を整形
            if "content" in result:
                content = result["content"]
                if isinstance(content, list):
                    return "\n".join(str(item) for item in content)
                else:
                    return str(content)
            else:
                return str(result)
                
        except Exception as e:
            self.logger.error(f"MCP tool execution failed: {e}")
            return f"MCP tool execution error: {str(e)}"


class MCPResourceWrapper(BaseTool):
    """MCPリソースラッパー"""
    
    name: str = "mcp_resource_wrapper"
    description: str = "MCPリソースのラッパー"
    
    def __init__(self, mcp_client: MCPClient, resource_uri: str):
        super().__init__()
        self.mcp_client = mcp_client
        self.resource_uri = resource_uri
        self.name = f"mcp_resource_{resource_uri.replace('/', '_').replace(':', '_')}"
        self.logger = logging.getLogger(__name__)
        
        # リソース情報を取得
        resource_info = mcp_client.get_resource_info(resource_uri)
        if resource_info:
            self.description = resource_info.get("description", f"MCP resource: {resource_uri}")
        else:
            self.description = f"MCP resource: {resource_uri}"
    
    def _run(self, **kwargs) -> str:
        """同期実行"""
        return asyncio.run(self._arun(**kwargs))
    
    async def _arun(self, **kwargs) -> str:
        """非同期実行"""
        
        try:
            result = await self.mcp_client.read_resource(self.resource_uri)
            
            if "error" in result:
                return f"MCP resource error: {result['error']}"
            
            # 結果を整形
            if "contents" in result:
                contents = result["contents"]
                if isinstance(contents, list):
                    return "\n".join(str(content) for content in contents)
                else:
                    return str(contents)
            else:
                return str(result)
                
        except Exception as e:
            self.logger.error(f"MCP resource access failed: {e}")
            return f"MCP resource access error: {str(e)}"


class MCPToolChain(BaseTool):
    """MCPツールチェーン"""
    
    name: str = "mcp_tool_chain"
    description: str = "複数のMCPツールを連鎖実行"
    
    def __init__(self, mcp_manager: MCPManager):
        super().__init__()
        self.mcp_manager = mcp_manager
        self.logger = logging.getLogger(__name__)
    
    def _run(self, tool_chain: str, **kwargs) -> str:
        """同期実行"""
        return asyncio.run(self._arun(tool_chain, **kwargs))
    
    async def _arun(self, tool_chain: str, **kwargs) -> str:
        """非同期実行"""
        
        try:
            if not tool_chain:
                return "ツールチェーンが指定されていません。"
            
            # ツールチェーンを解析
            chain_steps = self._parse_tool_chain(tool_chain)
            if not chain_steps:
                return "無効なツールチェーンです。"
            
            results = []
            current_input = kwargs.get("input", "")
            
            for step in chain_steps:
                tool_name = step["tool"]
                arguments = step.get("arguments", {})
                
                # 前のステップの結果を入力として使用
                if current_input:
                    arguments["input"] = current_input
                
                # ツールを実行
                tool = self.mcp_manager.get_tool(tool_name)
                if not tool:
                    return f"ツールが見つかりません: {tool_name}"
                
                result = await tool._arun(**arguments)
                results.append({
                    "tool": tool_name,
                    "result": result,
                    "arguments": arguments
                })
                
                # 結果を次のステップの入力として使用
                current_input = result
            
            # 結果を整形
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append(
                    f"ステップ {i}: {result['tool']}\n"
                    f"引数: {result['arguments']}\n"
                    f"結果: {result['result']}\n"
                )
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            self.logger.error(f"MCP tool chain execution failed: {e}")
            return f"MCP tool chain execution error: {str(e)}"
    
    def _parse_tool_chain(self, tool_chain: str) -> List[Dict[str, Any]]:
        """ツールチェーンを解析"""
        
        try:
            # JSON形式のツールチェーン
            if tool_chain.startswith('[') or tool_chain.startswith('{'):
                return json.loads(tool_chain)
            
            # シンプルな形式: "tool1:arg1=value1,arg2=value2;tool2:arg3=value3"
            steps = []
            for step_str in tool_chain.split(';'):
                if ':' in step_str:
                    tool_name, args_str = step_str.split(':', 1)
                    arguments = {}
                    
                    if args_str:
                        for arg_pair in args_str.split(','):
                            if '=' in arg_pair:
                                key, value = arg_pair.split('=', 1)
                                arguments[key.strip()] = value.strip()
                    
                    steps.append({
                        "tool": tool_name.strip(),
                        "arguments": arguments
                    })
                else:
                    steps.append({
                        "tool": step_str.strip(),
                        "arguments": {}
                    })
            
            return steps
            
        except Exception as e:
            self.logger.error(f"Tool chain parsing error: {e}")
            return []


class MCPToolSelector(BaseTool):
    """MCPツール選択器"""
    
    name: str = "mcp_tool_selector"
    description: str = "適切なMCPツールを選択"
    
    def __init__(self, mcp_manager: MCPManager):
        super().__init__()
        self.mcp_manager = mcp_manager
        self.logger = logging.getLogger(__name__)
    
    def _run(self, task_description: str, **kwargs) -> str:
        """同期実行"""
        return asyncio.run(self._arun(task_description, **kwargs))
    
    async def _arun(self, task_description: str, **kwargs) -> str:
        """非同期実行"""
        
        try:
            if not task_description:
                return "タスクの説明を指定してください。"
            
            # 利用可能なツールを取得
            available_tools = self.mcp_manager.get_available_tools()
            if not available_tools:
                return "利用可能なMCPツールがありません。"
            
            # タスクに適したツールを選択
            selected_tools = self._select_tools_for_task(task_description, available_tools)
            
            if not selected_tools:
                return "タスクに適したツールが見つかりませんでした。"
            
            # 選択されたツールの情報を返す
            tool_info = []
            for tool in selected_tools:
                tool_info.append(
                    f"ツール: {tool['name']}\n"
                    f"説明: {tool['description']}\n"
                    f"タイプ: {tool['type']}\n"
                )
            
            return f"タスク '{task_description}' に適したツール:\n\n" + "\n".join(tool_info)
            
        except Exception as e:
            self.logger.error(f"MCP tool selection failed: {e}")
            return f"MCP tool selection error: {str(e)}"
    
    def _select_tools_for_task(self, task_description: str, available_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """タスクに適したツールを選択（ワード検出なし）"""
        
        selected_tools = []
        
        # ワード検出を使わず、文脈と構造から判断
        # タスクの複雑さに基づいてツールを選択
        task_length = len(task_description)
        
        # 複雑なタスクの場合は、より多くのツールを提案
        if task_length > 100:
            # 複雑なタスクには複数のツールを提案
            selected_tools = available_tools[:3]  # 最初の3つのツール
        elif task_length > 50:
            # 中程度のタスクには2つのツールを提案
            selected_tools = available_tools[:2]
        else:
            # 簡単なタスクには1つのツールを提案
            selected_tools = available_tools[:1]
        
        return selected_tools


class MCPToolExecutor(BaseTool):
    """MCPツール実行器"""
    
    name: str = "mcp_tool_executor"
    description: str = "MCPツールを実行"
    
    def __init__(self, mcp_manager: MCPManager):
        super().__init__()
        self.mcp_manager = mcp_manager
        self.logger = logging.getLogger(__name__)
    
    def _run(self, tool_name: str, arguments: str = "{}", **kwargs) -> str:
        """同期実行"""
        return asyncio.run(self._arun(tool_name, arguments, **kwargs))
    
    async def _arun(self, tool_name: str, arguments: str = "{}", **kwargs) -> str:
        """非同期実行"""
        
        try:
            if not tool_name:
                return "ツール名を指定してください。"
            
            # 引数を解析
            try:
                if isinstance(arguments, str):
                    parsed_arguments = json.loads(arguments)
                else:
                    parsed_arguments = arguments
            except json.JSONDecodeError:
                parsed_arguments = {}
            
            # 追加の引数をマージ
            parsed_arguments.update(kwargs)
            
            # ツールを取得
            tool = self.mcp_manager.get_tool(tool_name)
            if not tool:
                return f"ツールが見つかりません: {tool_name}"
            
            # ツールを実行
            result = await tool._arun(**parsed_arguments)
            
            return f"ツール '{tool_name}' の実行結果:\n\n{result}"
            
        except Exception as e:
            self.logger.error(f"MCP tool execution failed: {e}")
            return f"MCP tool execution error: {str(e)}"


class MCPToolMonitor(BaseTool):
    """MCPツール監視器"""
    
    name: str = "mcp_tool_monitor"
    description: str = "MCPツールの状態を監視"
    
    def __init__(self, mcp_manager: MCPManager):
        super().__init__()
        self.mcp_manager = mcp_manager
        self.logger = logging.getLogger(__name__)
    
    def _run(self, monitor_type: str = "all", **kwargs) -> str:
        """同期実行"""
        return asyncio.run(self._arun(monitor_type, **kwargs))
    
    async def _arun(self, monitor_type: str = "all", **kwargs) -> str:
        """非同期実行"""
        
        try:
            if monitor_type == "all":
                return await self._monitor_all()
            elif monitor_type == "servers":
                return await self._monitor_servers()
            elif monitor_type == "tools":
                return await self._monitor_tools()
            elif monitor_type == "resources":
                return await self._monitor_resources()
            elif monitor_type == "performance":
                return await self._monitor_performance()
            else:
                return f"サポートされていない監視タイプ: {monitor_type}"
            
        except Exception as e:
            self.logger.error(f"MCP tool monitoring failed: {e}")
            return f"MCP tool monitoring error: {str(e)}"
    
    async def _monitor_all(self) -> str:
        """全監視情報を取得"""
        
        info_parts = []
        
        # サーバー監視
        server_info = await self._monitor_servers()
        info_parts.append(f"=== サーバー状態 ===\n{server_info}")
        
        # ツール監視
        tool_info = await self._monitor_tools()
        info_parts.append(f"\n=== ツール状態 ===\n{tool_info}")
        
        # リソース監視
        resource_info = await self._monitor_resources()
        info_parts.append(f"\n=== リソース状態 ===\n{resource_info}")
        
        return "\n".join(info_parts)
    
    async def _monitor_servers(self) -> str:
        """サーバー状態を監視"""
        
        try:
            stats = self.mcp_manager.get_stats()
            server_status = stats.get("server_status", {})
            
            if not server_status:
                return "接続中のMCPサーバーがありません。"
            
            server_info = []
            for server_name, status in server_status.items():
                server_info.append(
                    f"サーバー: {server_name}\n"
                    f"  接続状態: {'接続中' if status['connected'] else '切断'}\n"
                    f"  コマンド: {' '.join(status['server_command'])}\n"
                    f"  利用可能ツール数: {status['available_tools_count']}\n"
                    f"  利用可能リソース数: {status['available_resources_count']}\n"
                )
            
            return "\n".join(server_info)
            
        except Exception as e:
            return f"サーバー監視エラー: {str(e)}"
    
    async def _monitor_tools(self) -> str:
        """ツール状態を監視"""
        
        try:
            tools = self.mcp_manager.get_available_tools()
            
            if not tools:
                return "利用可能なMCPツールがありません。"
            
            tool_info = []
            for tool in tools:
                tool_info.append(
                    f"ツール: {tool['name']}\n"
                    f"  説明: {tool['description']}\n"
                    f"  タイプ: {tool['type']}\n"
                )
            
            return f"利用可能なツール ({len(tools)}個):\n\n" + "\n".join(tool_info)
            
        except Exception as e:
            return f"ツール監視エラー: {str(e)}"
    
    async def _monitor_resources(self) -> str:
        """リソース状態を監視"""
        
        try:
            resources = self.mcp_manager.get_available_resources()
            
            if not resources:
                return "利用可能なMCPリソースがありません。"
            
            resource_info = []
            for resource in resources:
                resource_info.append(
                    f"リソース: {resource['name']}\n"
                    f"  説明: {resource['description']}\n"
                    f"  タイプ: {resource['type']}\n"
                )
            
            return f"利用可能なリソース ({len(resources)}個):\n\n" + "\n".join(resource_info)
            
        except Exception as e:
            return f"リソース監視エラー: {str(e)}"
    
    async def _monitor_performance(self) -> str:
        """パフォーマンスを監視"""
        
        try:
            stats = self.mcp_manager.get_stats()
            
            performance_info = [
                f"サーバー数: {stats['servers_count']}",
                f"ツール数: {stats['tools_count']}",
                f"リソース数: {stats['resources_count']}",
                f"サーバー一覧: {', '.join(stats['servers'])}"
            ]
            
            return "パフォーマンス情報:\n" + "\n".join(performance_info)
            
        except Exception as e:
            return f"パフォーマンス監視エラー: {str(e)}"