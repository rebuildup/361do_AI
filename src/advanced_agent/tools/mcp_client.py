"""
MCP Client

Model Context Protocol クライアント実装
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime
import subprocess
import os
from pathlib import Path

from langchain.tools import BaseTool


class MCPClient:
    """MCP クライアント"""
    
    def __init__(self,
                 server_command: List[str],
                 server_args: Optional[List[str]] = None,
                 timeout: int = 30,
                 working_directory: Optional[str] = None):
        
        self.server_command = server_command
        self.server_args = server_args or []
        self.timeout = timeout
        self.working_directory = working_directory or os.getcwd()
        self.logger = logging.getLogger(__name__)
        
        # サーバープロセス
        self.server_process: Optional[subprocess.Popen] = None
        self.is_connected = False
        
        # ツールとリソース
        self.available_tools: Dict[str, Dict[str, Any]] = {}
        self.available_resources: Dict[str, Dict[str, Any]] = {}
        
        # リクエストID管理
        self.request_id = 0
    
    async def connect(self) -> bool:
        """MCPサーバーに接続"""
        
        try:
            if self.is_connected:
                return True
            
            self.logger.info(f"Connecting to MCP server: {' '.join(self.server_command)}")
            
            # サーバープロセス起動
            full_command = self.server_command + self.server_args
            self.server_process = subprocess.Popen(
                full_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.working_directory
            )
            
            # 初期化メッセージ送信
            init_message = {
                "jsonrpc": "2.0",
                "id": self._get_next_request_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                        "resources": {}
                    },
                    "clientInfo": {
                        "name": "advanced-agent",
                        "version": "1.0.0"
                    }
                }
            }
            
            response = await self._send_message(init_message)
            
            if response and "result" in response:
                # 初期化完了メッセージ送信
                initialized_message = {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized"
                }
                await self._send_message(initialized_message)
                
                # ツールとリソースを取得
                await self._load_tools_and_resources()
                
                self.is_connected = True
                self.logger.info("MCP server connected successfully")
                return True
            else:
                self.logger.error(f"Failed to initialize MCP server: {response}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to connect to MCP server: {e}")
            return False
    
    async def disconnect(self):
        """MCPサーバーから切断"""
        
        try:
            if self.server_process:
                self.server_process.terminate()
                await asyncio.sleep(1)
                
                if self.server_process.poll() is None:
                    self.server_process.kill()
                
                self.server_process = None
            
            self.is_connected = False
            self.available_tools.clear()
            self.available_resources.clear()
            
            self.logger.info("MCP server disconnected")
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from MCP server: {e}")
    
    async def _send_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """メッセージを送信"""
        
        try:
            if not self.server_process:
                return None
            
            message_str = json.dumps(message) + "\n"
            self.server_process.stdin.write(message_str)
            self.server_process.stdin.flush()
            
            # レスポンスを読み取り
            response_line = self.server_process.stdout.readline()
            if response_line:
                return json.loads(response_line.strip())
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
            return None
    
    def _get_next_request_id(self) -> int:
        """次のリクエストIDを取得"""
        
        self.request_id += 1
        return self.request_id
    
    async def _load_tools_and_resources(self):
        """ツールとリソースを読み込み"""
        
        try:
            # ツール一覧取得
            tools_message = {
                "jsonrpc": "2.0",
                "id": self._get_next_request_id(),
                "method": "tools/list"
            }
            
            tools_response = await self._send_message(tools_message)
            if tools_response and "result" in tools_response:
                self.available_tools = {
                    tool["name"]: tool
                    for tool in tools_response["result"].get("tools", [])
                }
            
            # リソース一覧取得
            resources_message = {
                "jsonrpc": "2.0",
                "id": self._get_next_request_id(),
                "method": "resources/list"
            }
            
            resources_response = await self._send_message(resources_message)
            if resources_response and "result" in resources_response:
                self.available_resources = {
                    resource["uri"]: resource
                    for resource in resources_response["result"].get("resources", [])
                }
            
            self.logger.info(f"Loaded {len(self.available_tools)} tools and {len(self.available_resources)} resources")
            
        except Exception as e:
            self.logger.error(f"Error loading tools and resources: {e}")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """ツールを呼び出し"""
        
        try:
            if not self.is_connected:
                return {"error": "Not connected to MCP server"}
            
            if tool_name not in self.available_tools:
                return {"error": f"Tool not found: {tool_name}"}
            
            call_message = {
                "jsonrpc": "2.0",
                "id": self._get_next_request_id(),
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            response = await self._send_message(call_message)
            
            if response and "result" in response:
                return response["result"]
            elif response and "error" in response:
                return {"error": response["error"]}
            else:
                return {"error": "Invalid response from MCP server"}
                
        except Exception as e:
            self.logger.error(f"Error calling tool {tool_name}: {e}")
            return {"error": str(e)}
    
    async def read_resource(self, resource_uri: str) -> Dict[str, Any]:
        """リソースを読み取り"""
        
        try:
            if not self.is_connected:
                return {"error": "Not connected to MCP server"}
            
            if resource_uri not in self.available_resources:
                return {"error": f"Resource not found: {resource_uri}"}
            
            read_message = {
                "jsonrpc": "2.0",
                "id": self._get_next_request_id(),
                "method": "resources/read",
                "params": {
                    "uri": resource_uri
                }
            }
            
            response = await self._send_message(read_message)
            
            if response and "result" in response:
                return response["result"]
            elif response and "error" in response:
                return {"error": response["error"]}
            else:
                return {"error": "Invalid response from MCP server"}
                
        except Exception as e:
            self.logger.error(f"Error reading resource {resource_uri}: {e}")
            return {"error": str(e)}
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """利用可能なツール一覧を取得"""
        
        return list(self.available_tools.values())
    
    def get_available_resources(self) -> List[Dict[str, Any]]:
        """利用可能なリソース一覧を取得"""
        
        return list(self.available_resources.values())
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """ツール情報を取得"""
        
        return self.available_tools.get(tool_name)
    
    def get_resource_info(self, resource_uri: str) -> Optional[Dict[str, Any]]:
        """リソース情報を取得"""
        
        return self.available_resources.get(resource_uri)
    
    def is_tool_available(self, tool_name: str) -> bool:
        """ツールが利用可能かチェック"""
        
        return tool_name in self.available_tools
    
    def is_resource_available(self, resource_uri: str) -> bool:
        """リソースが利用可能かチェック"""
        
        return resource_uri in self.available_resources
    
    def get_connection_status(self) -> Dict[str, Any]:
        """接続状態を取得"""
        
        return {
            "connected": self.is_connected,
            "server_command": self.server_command,
            "server_args": self.server_args,
            "available_tools_count": len(self.available_tools),
            "available_resources_count": len(self.available_resources),
            "working_directory": self.working_directory
        }


class MCPTool(BaseTool):
    """MCP ツールラッパー"""
    
    name: str = "mcp_tool"
    description: str = "MCP ツール"
    
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


class MCPResource(BaseTool):
    """MCP リソースラッパー"""
    
    name: str = "mcp_resource"
    description: str = "MCP リソース"
    
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


class MCPManager:
    """MCP 管理クラス"""
    
    def __init__(self):
        self.clients: Dict[str, MCPClient] = {}
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.logger = logging.getLogger(__name__)
    
    async def add_server(self,
                        server_name: str,
                        server_command: List[str],
                        server_args: Optional[List[str]] = None,
                        timeout: int = 30,
                        working_directory: Optional[str] = None) -> bool:
        """MCPサーバーを追加"""
        
        try:
            if server_name in self.clients:
                self.logger.warning(f"MCP server already exists: {server_name}")
                return False
            
            client = MCPClient(
                server_command=server_command,
                server_args=server_args,
                timeout=timeout,
                working_directory=working_directory
            )
            
            # 接続
            if await client.connect():
                self.clients[server_name] = client
                
                # ツールとリソースを登録
                await self._register_tools_and_resources(server_name, client)
                
                self.logger.info(f"MCP server added: {server_name}")
                return True
            else:
                self.logger.error(f"Failed to connect to MCP server: {server_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error adding MCP server: {e}")
            return False
    
    async def remove_server(self, server_name: str) -> bool:
        """MCPサーバーを削除"""
        
        try:
            if server_name not in self.clients:
                return False
            
            client = self.clients[server_name]
            await client.disconnect()
            
            # 関連するツールとリソースを削除
            tools_to_remove = [name for name in self.tools.keys() if name.startswith(f"mcp_{server_name}_")]
            for tool_name in tools_to_remove:
                del self.tools[tool_name]
            
            resources_to_remove = [name for name in self.resources.keys() if name.startswith(f"mcp_{server_name}_")]
            for resource_name in resources_to_remove:
                del self.resources[resource_name]
            
            del self.clients[server_name]
            
            self.logger.info(f"MCP server removed: {server_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing MCP server: {e}")
            return False
    
    async def _register_tools_and_resources(self, server_name: str, client: MCPClient):
        """ツールとリソースを登録"""
        
        try:
            # ツールを登録
            for tool_info in client.get_available_tools():
                tool_name = tool_info["name"]
                full_tool_name = f"mcp_{server_name}_{tool_name}"
                
                mcp_tool = MCPTool(client, tool_name)
                self.tools[full_tool_name] = mcp_tool
            
            # リソースを登録
            for resource_info in client.get_available_resources():
                resource_uri = resource_info["uri"]
                full_resource_name = f"mcp_{server_name}_{resource_uri.replace('/', '_').replace(':', '_')}"
                
                mcp_resource = MCPResource(client, resource_uri)
                self.resources[full_resource_name] = mcp_resource
            
            self.logger.info(f"Registered {len(client.get_available_tools())} tools and {len(client.get_available_resources())} resources from {server_name}")
            
        except Exception as e:
            self.logger.error(f"Error registering tools and resources: {e}")
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """利用可能なツール一覧を取得"""
        
        tools = []
        for tool_name, tool in self.tools.items():
            tools.append({
                "name": tool_name,
                "description": tool.description,
                "type": "mcp_tool"
            })
        
        return tools
    
    def get_available_resources(self) -> List[Dict[str, Any]]:
        """利用可能なリソース一覧を取得"""
        
        resources = []
        for resource_name, resource in self.resources.items():
            resources.append({
                "name": resource_name,
                "description": resource.description,
                "type": "mcp_resource"
            })
        
        return resources
    
    def get_tool(self, tool_name: str) -> Optional[MCPTool]:
        """ツールを取得"""
        
        return self.tools.get(tool_name)
    
    def get_resource(self, resource_name: str) -> Optional[MCPResource]:
        """リソースを取得"""
        
        return self.resources.get(resource_name)
    
    def get_server_status(self) -> Dict[str, Any]:
        """サーバー状態を取得"""
        
        status = {}
        for server_name, client in self.clients.items():
            status[server_name] = client.get_connection_status()
        
        return status
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
        
        return {
            "servers_count": len(self.clients),
            "tools_count": len(self.tools),
            "resources_count": len(self.resources),
            "servers": list(self.clients.keys()),
            "server_status": self.get_server_status()
        }
