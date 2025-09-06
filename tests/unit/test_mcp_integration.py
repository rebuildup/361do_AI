"""
Tests for MCP Integration

MCP統合システムのテスト
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import json

from src.advanced_agent.tools.mcp_client import MCPClient, MCPManager, MCPTool, MCPResource
from src.advanced_agent.tools.mcp_tools import (
    MCPToolWrapper, MCPResourceWrapper, MCPToolChain, 
    MCPToolSelector, MCPToolExecutor, MCPToolMonitor
)


class TestMCPToolWrapper:
    """MCPツールラッパーのテスト"""
    
    def test_mcp_tool_wrapper_initialization(self):
        """MCPツールラッパーの初期化テスト"""
        mock_client = Mock()
        mock_client.get_tool_info.return_value = {
            "description": "Test MCP tool"
        }
        
        wrapper = MCPToolWrapper(mock_client, "test_tool")
        assert wrapper.name == "mcp_test_tool"
        assert wrapper.description == "Test MCP tool"
        assert wrapper.tool_name == "test_tool"
    
    def test_mcp_tool_wrapper_no_tool_info(self):
        """ツール情報なしの初期化テスト"""
        mock_client = Mock()
        mock_client.get_tool_info.return_value = None
        
        wrapper = MCPToolWrapper(mock_client, "test_tool")
        assert wrapper.name == "mcp_test_tool"
        assert wrapper.description == "MCP tool: test_tool"
    
    @pytest.mark.asyncio
    async def test_mcp_tool_wrapper_execution_success(self):
        """MCPツールラッパー実行成功テスト"""
        mock_client = Mock()
        mock_client.get_tool_info.return_value = {"description": "Test tool"}
        mock_client.call_tool = AsyncMock(return_value={
            "content": "Test result"
        })
        
        wrapper = MCPToolWrapper(mock_client, "test_tool")
        result = await wrapper._arun(arg1="value1")
        
        assert result == "Test result"
        mock_client.call_tool.assert_called_once_with("test_tool", {"arg1": "value1"})
    
    @pytest.mark.asyncio
    async def test_mcp_tool_wrapper_execution_error(self):
        """MCPツールラッパー実行エラーテスト"""
        mock_client = Mock()
        mock_client.get_tool_info.return_value = {"description": "Test tool"}
        mock_client.call_tool = AsyncMock(return_value={
            "error": "Test error"
        })
        
        wrapper = MCPToolWrapper(mock_client, "test_tool")
        result = await wrapper._arun(arg1="value1")
        
        assert "MCP tool error: Test error" in result
    
    @pytest.mark.asyncio
    async def test_mcp_tool_wrapper_execution_exception(self):
        """MCPツールラッパー実行例外テスト"""
        mock_client = Mock()
        mock_client.get_tool_info.return_value = {"description": "Test tool"}
        mock_client.call_tool = AsyncMock(side_effect=Exception("Test exception"))
        
        wrapper = MCPToolWrapper(mock_client, "test_tool")
        result = await wrapper._arun(arg1="value1")
        
        assert "MCP tool execution error: Test exception" in result


class TestMCPResourceWrapper:
    """MCPリソースラッパーのテスト"""
    
    def test_mcp_resource_wrapper_initialization(self):
        """MCPリソースラッパーの初期化テスト"""
        mock_client = Mock()
        mock_client.get_resource_info.return_value = {
            "description": "Test MCP resource"
        }
        
        wrapper = MCPResourceWrapper(mock_client, "test://resource")
        assert wrapper.name == "mcp_resource_test___resource"
        assert wrapper.description == "Test MCP resource"
        assert wrapper.resource_uri == "test://resource"
    
    @pytest.mark.asyncio
    async def test_mcp_resource_wrapper_access_success(self):
        """MCPリソースラッパーアクセス成功テスト"""
        mock_client = Mock()
        mock_client.get_resource_info.return_value = {"description": "Test resource"}
        mock_client.read_resource = AsyncMock(return_value={
            "contents": "Test content"
        })
        
        wrapper = MCPResourceWrapper(mock_client, "test://resource")
        result = await wrapper._arun()
        
        assert result == "Test content"
        mock_client.read_resource.assert_called_once_with("test://resource")
    
    @pytest.mark.asyncio
    async def test_mcp_resource_wrapper_access_error(self):
        """MCPリソースラッパーアクセスエラーテスト"""
        mock_client = Mock()
        mock_client.get_resource_info.return_value = {"description": "Test resource"}
        mock_client.read_resource = AsyncMock(return_value={
            "error": "Test error"
        })
        
        wrapper = MCPResourceWrapper(mock_client, "test://resource")
        result = await wrapper._arun()
        
        assert "MCP resource error: Test error" in result


class TestMCPToolChain:
    """MCPツールチェーンのテスト"""
    
    def test_mcp_tool_chain_initialization(self):
        """MCPツールチェーンの初期化テスト"""
        mock_manager = Mock()
        chain = MCPToolChain(mock_manager)
        assert chain.name == "mcp_tool_chain"
        assert "複数のMCPツールを連鎖実行" in chain.description
    
    def test_parse_tool_chain_json(self):
        """JSON形式のツールチェーン解析テスト"""
        mock_manager = Mock()
        chain = MCPToolChain(mock_manager)
        
        tool_chain = '[{"tool": "tool1", "arguments": {"arg1": "value1"}}, {"tool": "tool2"}]'
        steps = chain._parse_tool_chain(tool_chain)
        
        assert len(steps) == 2
        assert steps[0]["tool"] == "tool1"
        assert steps[0]["arguments"]["arg1"] == "value1"
        assert steps[1]["tool"] == "tool2"
    
    def test_parse_tool_chain_simple(self):
        """シンプル形式のツールチェーン解析テスト"""
        mock_manager = Mock()
        chain = MCPToolChain(mock_manager)
        
        tool_chain = "tool1:arg1=value1,arg2=value2;tool2:arg3=value3"
        steps = chain._parse_tool_chain(tool_chain)
        
        assert len(steps) == 2
        assert steps[0]["tool"] == "tool1"
        assert steps[0]["arguments"]["arg1"] == "value1"
        assert steps[0]["arguments"]["arg2"] == "value2"
        assert steps[1]["tool"] == "tool2"
        assert steps[1]["arguments"]["arg3"] == "value3"
    
    def test_parse_tool_chain_invalid(self):
        """無効なツールチェーン解析テスト"""
        mock_manager = Mock()
        chain = MCPToolChain(mock_manager)
        
        steps = chain._parse_tool_chain("invalid json")
        assert len(steps) == 0
    
    @pytest.mark.asyncio
    async def test_mcp_tool_chain_execution_success(self):
        """MCPツールチェーン実行成功テスト"""
        mock_manager = Mock()
        
        # モックツール
        mock_tool = Mock()
        mock_tool._arun = AsyncMock(return_value="Tool result")
        mock_manager.get_tool.return_value = mock_tool
        
        chain = MCPToolChain(mock_manager)
        result = await chain._arun("tool1:arg1=value1;tool2", input="initial input")
        
        assert "ステップ 1: tool1" in result
        assert "ステップ 2: tool2" in result
        assert "Tool result" in result
    
    @pytest.mark.asyncio
    async def test_mcp_tool_chain_execution_tool_not_found(self):
        """MCPツールチェーン実行 - ツール未発見テスト"""
        mock_manager = Mock()
        mock_manager.get_tool.return_value = None
        
        chain = MCPToolChain(mock_manager)
        result = await chain._arun("nonexistent_tool")
        
        assert "ツールが見つかりません: nonexistent_tool" in result
    
    @pytest.mark.asyncio
    async def test_mcp_tool_chain_execution_empty_chain(self):
        """MCPツールチェーン実行 - 空チェーンテスト"""
        mock_manager = Mock()
        chain = MCPToolChain(mock_manager)
        
        result = await chain._arun("")
        assert "ツールチェーンが指定されていません" in result


class TestMCPToolSelector:
    """MCPツール選択器のテスト"""
    
    def test_mcp_tool_selector_initialization(self):
        """MCPツール選択器の初期化テスト"""
        mock_manager = Mock()
        selector = MCPToolSelector(mock_manager)
        assert selector.name == "mcp_tool_selector"
        assert "適切なMCPツールを選択" in selector.description
    
    def test_select_tools_for_task_keyword_match(self):
        """キーワードマッチングによるツール選択テスト"""
        mock_manager = Mock()
        selector = MCPToolSelector(mock_manager)
        
        available_tools = [
            {"name": "web_search", "description": "Search the web", "type": "tool"},
            {"name": "file_manager", "description": "Manage files", "type": "tool"},
            {"name": "database_query", "description": "Query database", "type": "tool"}
        ]
        
        # Web検索タスク
        selected = selector._select_tools_for_task("search for information on the web", available_tools)
        assert len(selected) > 0
        assert any(tool["name"] == "web_search" for tool in selected)
        
        # ファイル管理タスク
        selected = selector._select_tools_for_task("manage files in directory", available_tools)
        assert len(selected) > 0
        assert any(tool["name"] == "file_manager" for tool in selected)
    
    def test_select_tools_for_task_no_match(self):
        """マッチしないタスクのテスト"""
        mock_manager = Mock()
        selector = MCPToolSelector(mock_manager)
        
        available_tools = [
            {"name": "web_search", "description": "Search the web", "type": "tool"}
        ]
        
        selected = selector._select_tools_for_task("completely unrelated task", available_tools)
        assert len(selected) == 0
    
    @pytest.mark.asyncio
    async def test_mcp_tool_selector_execution_success(self):
        """MCPツール選択器実行成功テスト"""
        mock_manager = Mock()
        mock_manager.get_available_tools.return_value = [
            {"name": "web_search", "description": "Search the web", "type": "tool"}
        ]
        
        selector = MCPToolSelector(mock_manager)
        result = await selector._arun("search for information")
        
        assert "タスク 'search for information' に適したツール:" in result
        assert "web_search" in result
    
    @pytest.mark.asyncio
    async def test_mcp_tool_selector_execution_no_tools(self):
        """MCPツール選択器実行 - ツールなしテスト"""
        mock_manager = Mock()
        mock_manager.get_available_tools.return_value = []
        
        selector = MCPToolSelector(mock_manager)
        result = await selector._arun("search for information")
        
        assert "利用可能なMCPツールがありません" in result
    
    @pytest.mark.asyncio
    async def test_mcp_tool_selector_execution_empty_task(self):
        """MCPツール選択器実行 - 空タスクテスト"""
        mock_manager = Mock()
        selector = MCPToolSelector(mock_manager)
        
        result = await selector._arun("")
        assert "タスクの説明を指定してください" in result


class TestMCPToolExecutor:
    """MCPツール実行器のテスト"""
    
    def test_mcp_tool_executor_initialization(self):
        """MCPツール実行器の初期化テスト"""
        mock_manager = Mock()
        executor = MCPToolExecutor(mock_manager)
        assert executor.name == "mcp_tool_executor"
        assert "MCPツールを実行" in executor.description
    
    @pytest.mark.asyncio
    async def test_mcp_tool_executor_execution_success(self):
        """MCPツール実行器実行成功テスト"""
        mock_manager = Mock()
        
        # モックツール
        mock_tool = Mock()
        mock_tool._arun = AsyncMock(return_value="Execution result")
        mock_manager.get_tool.return_value = mock_tool
        
        executor = MCPToolExecutor(mock_manager)
        result = await executor._arun("test_tool", '{"arg1": "value1"}')
        
        assert "ツール 'test_tool' の実行結果:" in result
        assert "Execution result" in result
    
    @pytest.mark.asyncio
    async def test_mcp_tool_executor_execution_tool_not_found(self):
        """MCPツール実行器実行 - ツール未発見テスト"""
        mock_manager = Mock()
        mock_manager.get_tool.return_value = None
        
        executor = MCPToolExecutor(mock_manager)
        result = await executor._arun("nonexistent_tool")
        
        assert "ツールが見つかりません: nonexistent_tool" in result
    
    @pytest.mark.asyncio
    async def test_mcp_tool_executor_execution_invalid_json(self):
        """MCPツール実行器実行 - 無効JSONテスト"""
        mock_manager = Mock()
        
        # モックツール
        mock_tool = Mock()
        mock_tool._arun = AsyncMock(return_value="Execution result")
        mock_manager.get_tool.return_value = mock_tool
        
        executor = MCPToolExecutor(mock_manager)
        result = await executor._arun("test_tool", "invalid json")
        
        # 無効なJSONの場合は空の辞書として扱われる
        assert "ツール 'test_tool' の実行結果:" in result
    
    @pytest.mark.asyncio
    async def test_mcp_tool_executor_execution_empty_tool_name(self):
        """MCPツール実行器実行 - 空ツール名テスト"""
        mock_manager = Mock()
        executor = MCPToolExecutor(mock_manager)
        
        result = await executor._arun("")
        assert "ツール名を指定してください" in result


class TestMCPToolMonitor:
    """MCPツール監視器のテスト"""
    
    def test_mcp_tool_monitor_initialization(self):
        """MCPツール監視器の初期化テスト"""
        mock_manager = Mock()
        monitor = MCPToolMonitor(mock_manager)
        assert monitor.name == "mcp_tool_monitor"
        assert "MCPツールの状態を監視" in monitor.description
    
    @pytest.mark.asyncio
    async def test_mcp_tool_monitor_all(self):
        """MCPツール監視器 - 全監視テスト"""
        mock_manager = Mock()
        mock_manager.get_stats.return_value = {
            "server_status": {
                "server1": {
                    "connected": True,
                    "server_command": ["python", "server.py"],
                    "available_tools_count": 5,
                    "available_resources_count": 3
                }
            }
        }
        mock_manager.get_available_tools.return_value = [
            {"name": "tool1", "description": "Tool 1", "type": "tool"}
        ]
        mock_manager.get_available_resources.return_value = [
            {"name": "resource1", "description": "Resource 1", "type": "resource"}
        ]
        
        monitor = MCPToolMonitor(mock_manager)
        result = await monitor._arun("all")
        
        assert "=== サーバー状態 ===" in result
        assert "=== ツール状態 ===" in result
        assert "=== リソース状態 ===" in result
    
    @pytest.mark.asyncio
    async def test_mcp_tool_monitor_servers(self):
        """MCPツール監視器 - サーバー監視テスト"""
        mock_manager = Mock()
        mock_manager.get_stats.return_value = {
            "server_status": {
                "server1": {
                    "connected": True,
                    "server_command": ["python", "server.py"],
                    "available_tools_count": 5,
                    "available_resources_count": 3
                }
            }
        }
        
        monitor = MCPToolMonitor(mock_manager)
        result = await monitor._arun("servers")
        
        assert "サーバー: server1" in result
        assert "接続状態: 接続中" in result
        assert "利用可能ツール数: 5" in result
    
    @pytest.mark.asyncio
    async def test_mcp_tool_monitor_servers_no_servers(self):
        """MCPツール監視器 - サーバーなしテスト"""
        mock_manager = Mock()
        mock_manager.get_stats.return_value = {"server_status": {}}
        
        monitor = MCPToolMonitor(mock_manager)
        result = await monitor._arun("servers")
        
        assert "接続中のMCPサーバーがありません" in result
    
    @pytest.mark.asyncio
    async def test_mcp_tool_monitor_tools(self):
        """MCPツール監視器 - ツール監視テスト"""
        mock_manager = Mock()
        mock_manager.get_available_tools.return_value = [
            {"name": "tool1", "description": "Tool 1", "type": "tool"},
            {"name": "tool2", "description": "Tool 2", "type": "tool"}
        ]
        
        monitor = MCPToolMonitor(mock_manager)
        result = await monitor._arun("tools")
        
        assert "利用可能なツール (2個):" in result
        assert "ツール: tool1" in result
        assert "ツール: tool2" in result
    
    @pytest.mark.asyncio
    async def test_mcp_tool_monitor_tools_no_tools(self):
        """MCPツール監視器 - ツールなしテスト"""
        mock_manager = Mock()
        mock_manager.get_available_tools.return_value = []
        
        monitor = MCPToolMonitor(mock_manager)
        result = await monitor._arun("tools")
        
        assert "利用可能なMCPツールがありません" in result
    
    @pytest.mark.asyncio
    async def test_mcp_tool_monitor_resources(self):
        """MCPツール監視器 - リソース監視テスト"""
        mock_manager = Mock()
        mock_manager.get_available_resources.return_value = [
            {"name": "resource1", "description": "Resource 1", "type": "resource"}
        ]
        
        monitor = MCPToolMonitor(mock_manager)
        result = await monitor._arun("resources")
        
        assert "利用可能なリソース (1個):" in result
        assert "リソース: resource1" in result
    
    @pytest.mark.asyncio
    async def test_mcp_tool_monitor_performance(self):
        """MCPツール監視器 - パフォーマンス監視テスト"""
        mock_manager = Mock()
        mock_manager.get_stats.return_value = {
            "servers_count": 2,
            "tools_count": 10,
            "resources_count": 5,
            "servers": ["server1", "server2"]
        }
        
        monitor = MCPToolMonitor(mock_manager)
        result = await monitor._arun("performance")
        
        assert "サーバー数: 2" in result
        assert "ツール数: 10" in result
        assert "リソース数: 5" in result
        assert "server1" in result
        assert "server2" in result
    
    @pytest.mark.asyncio
    async def test_mcp_tool_monitor_invalid_type(self):
        """MCPツール監視器 - 無効タイプテスト"""
        mock_manager = Mock()
        monitor = MCPToolMonitor(mock_manager)
        
        result = await monitor._arun("invalid_type")
        assert "サポートされていない監視タイプ: invalid_type" in result


if __name__ == "__main__":
    pytest.main([__file__])
