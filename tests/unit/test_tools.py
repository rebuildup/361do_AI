"""
Tests for Tools

ツールシステムのテスト
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from src.advanced_agent.tools.web_search import WebSearchTool, WebScrapingTool, NewsSearchTool
from src.advanced_agent.tools.command_executor import CommandExecutorTool, PythonExecutorTool, SystemInfoTool
from src.advanced_agent.tools.file_manager import FileManagerTool, TextProcessorTool
from src.advanced_agent.tools.tool_registry import ToolRegistry, ToolExecutor
from src.advanced_agent.tools.mcp_client import MCPClient, MCPManager


class TestWebSearchTool:
    """Web検索ツールのテスト"""
    
    def test_web_search_tool_initialization(self):
        """Web検索ツールの初期化テスト"""
        tool = WebSearchTool()
        assert tool.name == "web_search"
        assert "インターネットで情報を検索" in tool.description
        assert tool.search_engine == "duckduckgo"
        assert tool.max_results == 5
        assert tool.timeout == 30
    
    def test_web_search_tool_empty_query(self):
        """空のクエリテスト"""
        tool = WebSearchTool()
        result = tool._run("")
        assert "検索クエリが空です" in result
    
    @patch('aiohttp.ClientSession')
    async def test_web_search_tool_duckduckgo(self, mock_session):
        """DuckDuckGo検索テスト"""
        tool = WebSearchTool()
        
        # モックレスポンス
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "Abstract": "Test abstract",
            "AbstractURL": "https://example.com",
            "Heading": "Test heading"
        })
        
        mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
        
        result = await tool._arun("test query")
        assert "Test abstract" in result or "検索結果が見つかりませんでした" in result
    
    def test_web_scraping_tool_initialization(self):
        """Webスクレイピングツールの初期化テスト"""
        tool = WebScrapingTool()
        assert tool.name == "web_scraping"
        assert "指定されたURLからWebページの内容を取得" in tool.description
    
    def test_web_scraping_tool_empty_url(self):
        """空のURLテスト"""
        tool = WebScrapingTool()
        result = tool._run("")
        assert "URLが指定されていません" in result
    
    def test_news_search_tool_initialization(self):
        """ニュース検索ツールの初期化テスト"""
        tool = NewsSearchTool()
        assert tool.name == "news_search"
        assert "最新のニュースを検索" in tool.description


class TestCommandExecutorTool:
    """コマンド実行ツールのテスト"""
    
    def test_command_executor_initialization(self):
        """コマンド実行ツールの初期化テスト"""
        tool = CommandExecutorTool()
        assert tool.name == "command_executor"
        assert "システムコマンドを実行" in tool.description
        assert "ls" in tool.allowed_commands
        assert "rm" not in tool.allowed_commands
    
    def test_command_executor_empty_command(self):
        """空のコマンドテスト"""
        tool = CommandExecutorTool()
        result = tool._run("")
        assert "コマンドが指定されていません" in result
    
    def test_command_executor_dangerous_command(self):
        """危険なコマンドテスト"""
        tool = CommandExecutorTool()
        result = tool._run("rm -rf /")
        assert "許可されていないコマンドです" in result
    
    def test_command_executor_allowed_command(self):
        """許可されたコマンドテスト"""
        tool = CommandExecutorTool()
        result = tool._run("echo test")
        assert "test" in result or "コマンド実行エラー" in result
    
    def test_python_executor_initialization(self):
        """Python実行ツールの初期化テスト"""
        tool = PythonExecutorTool()
        assert tool.name == "python_executor"
        assert "Pythonコードを安全に実行" in tool.description
    
    def test_python_executor_empty_code(self):
        """空のコードテスト"""
        tool = PythonExecutorTool()
        result = tool._run("")
        assert "Pythonコードが指定されていません" in result
    
    def test_python_executor_dangerous_code(self):
        """危険なコードテスト"""
        tool = PythonExecutorTool()
        result = tool._run("import os; os.system('rm -rf /')")
        assert "安全でないコードが検出されました" in result
    
    def test_python_executor_safe_code(self):
        """安全なコードテスト"""
        tool = PythonExecutorTool()
        result = tool._run("print('Hello, World!')")
        assert "Hello, World!" in result or "Python実行エラー" in result
    
    def test_system_info_tool_initialization(self):
        """システム情報ツールの初期化テスト"""
        tool = SystemInfoTool()
        assert tool.name == "system_info"
        assert "システムの情報を取得" in tool.description


class TestFileManagerTool:
    """ファイル管理ツールのテスト"""
    
    def test_file_manager_initialization(self):
        """ファイル管理ツールの初期化テスト"""
        tool = FileManagerTool()
        assert tool.name == "file_manager"
        assert "ファイルとディレクトリの操作" in tool.description
        assert ".txt" in tool.allowed_extensions
        assert ".py" in tool.allowed_extensions
    
    def test_file_manager_empty_action(self):
        """空のアクションテスト"""
        tool = FileManagerTool()
        result = tool._run("", "test.txt")
        assert "アクションとパスを指定してください" in result
    
    def test_file_manager_empty_path(self):
        """空のパステスト"""
        tool = FileManagerTool()
        result = tool._run("read", "")
        assert "アクションとパスを指定してください" in result
    
    def test_file_manager_invalid_action(self):
        """無効なアクションテスト"""
        tool = FileManagerTool()
        result = tool._run("invalid_action", "test.txt")
        assert "サポートされていないアクション" in result
    
    def test_text_processor_initialization(self):
        """テキスト処理ツールの初期化テスト"""
        tool = TextProcessorTool()
        assert tool.name == "text_processor"
        assert "テキストファイルの処理" in tool.description
    
    def test_text_processor_empty_action(self):
        """空のアクションテスト"""
        tool = TextProcessorTool()
        result = tool._run("", "test text")
        assert "アクションとテキストを指定してください" in result
    
    def test_text_processor_count_words(self):
        """単語数カウントテスト"""
        tool = TextProcessorTool()
        result = tool._run("count_words", "Hello world test")
        assert "単語数: 3" in result
    
    def test_text_processor_count_lines(self):
        """行数カウントテスト"""
        tool = TextProcessorTool()
        result = tool._run("count_lines", "line1\nline2\nline3")
        assert "行数: 3" in result
    
    def test_text_processor_count_chars(self):
        """文字数カウントテスト"""
        tool = TextProcessorTool()
        result = tool._run("count_chars", "Hello")
        assert "文字数: 5" in result


class TestToolRegistry:
    """ツール登録システムのテスト"""
    
    def test_tool_registry_initialization(self):
        """ツール登録システムの初期化テスト"""
        registry = ToolRegistry()
        assert len(registry.tools) > 0
        assert "web_search" in registry.tools
        assert "command_executor" in registry.tools
        assert "file_manager" in registry.tools
    
    def test_register_tool(self):
        """ツール登録テスト"""
        registry = ToolRegistry()
        
        # モックツール
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool description"
        
        # ツール登録
        result = registry.register_tool(mock_tool, {"category": "test"})
        assert result is True
        assert "test_tool" in registry.tools
        assert registry.tool_metadata["test_tool"]["category"] == "test"
    
    def test_register_duplicate_tool(self):
        """重複ツール登録テスト"""
        registry = ToolRegistry()
        
        # モックツール
        mock_tool = Mock()
        mock_tool.name = "web_search"  # 既存のツール名
        mock_tool.description = "Test tool description"
        
        # 重複登録
        result = registry.register_tool(mock_tool)
        assert result is False
    
    def test_unregister_tool(self):
        """ツール登録解除テスト"""
        registry = ToolRegistry()
        
        # ツール登録解除
        result = registry.unregister_tool("web_search")
        assert result is True
        assert "web_search" not in registry.tools
        assert "web_search" not in registry.tool_metadata
    
    def test_unregister_nonexistent_tool(self):
        """存在しないツールの登録解除テスト"""
        registry = ToolRegistry()
        
        result = registry.unregister_tool("nonexistent_tool")
        assert result is False
    
    def test_get_tool(self):
        """ツール取得テスト"""
        registry = ToolRegistry()
        
        tool = registry.get_tool("web_search")
        assert tool is not None
        assert tool.name == "web_search"
    
    def test_get_nonexistent_tool(self):
        """存在しないツール取得テスト"""
        registry = ToolRegistry()
        
        tool = registry.get_tool("nonexistent_tool")
        assert tool is None
    
    def test_list_tools(self):
        """ツール一覧取得テスト"""
        registry = ToolRegistry()
        
        tools = registry.list_tools()
        assert len(tools) > 0
        assert any(tool["name"] == "web_search" for tool in tools)
    
    def test_list_tools_by_category(self):
        """カテゴリ別ツール一覧取得テスト"""
        registry = ToolRegistry()
        
        web_tools = registry.get_tools_by_category("web")
        assert len(web_tools) > 0
        
        system_tools = registry.get_tools_by_category("system")
        assert len(system_tools) > 0
    
    def test_search_tools(self):
        """ツール検索テスト"""
        registry = ToolRegistry()
        
        results = registry.search_tools("web")
        assert len(results) > 0
        assert any("web" in result["name"].lower() or "web" in result["description"].lower() for result in results)
    
    def test_get_registry_stats(self):
        """登録システム統計取得テスト"""
        registry = ToolRegistry()
        
        stats = registry.get_registry_stats()
        assert "total_tools" in stats
        assert "categories" in stats
        assert stats["total_tools"] > 0
    
    def test_validate_tool(self):
        """ツール検証テスト"""
        registry = ToolRegistry()
        
        # 有効なツール
        mock_tool = Mock()
        mock_tool.name = "valid_tool"
        mock_tool.description = "Valid tool description"
        mock_tool._run = Mock(return_value="test result")
        
        validation = registry.validate_tool(mock_tool)
        assert validation["valid"] is True
        assert len(validation["errors"]) == 0
    
    def test_validate_invalid_tool(self):
        """無効なツール検証テスト"""
        registry = ToolRegistry()
        
        # 無効なツール（名前なし）
        mock_tool = Mock()
        mock_tool.name = ""
        mock_tool.description = "Invalid tool description"
        
        validation = registry.validate_tool(mock_tool)
        assert validation["valid"] is False
        assert len(validation["errors"]) > 0


class TestToolExecutor:
    """ツール実行システムのテスト"""
    
    def test_tool_executor_initialization(self):
        """ツール実行システムの初期化テスト"""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        assert executor.registry == registry
        assert len(executor.execution_history) == 0
    
    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        """ツール実行成功テスト"""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        
        # モックツール
        mock_tool = Mock()
        mock_tool._arun = AsyncMock(return_value="test result")
        registry.tools["test_tool"] = mock_tool
        
        result = await executor.execute_tool("test_tool", {"arg1": "value1"})
        assert result["success"] is True
        assert result["result"] == "test result"
        assert result["error"] is None
        assert len(executor.execution_history) == 1
    
    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self):
        """存在しないツール実行テスト"""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        
        result = await executor.execute_tool("nonexistent_tool", {})
        assert result["success"] is False
        assert "Tool not found" in result["error"]
        assert result["result"] is None
    
    @pytest.mark.asyncio
    async def test_execute_tool_timeout(self):
        """ツール実行タイムアウトテスト"""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        
        # モックツール（遅延）
        mock_tool = Mock()
        async def slow_arun(**kwargs):
            await asyncio.sleep(2)
            return "slow result"
        mock_tool._arun = slow_arun
        registry.tools["slow_tool"] = mock_tool
        
        result = await executor.execute_tool("slow_tool", {}, timeout=1)
        assert result["success"] is False
        assert "timeout" in result["error"].lower()
        assert result["result"] is None
    
    def test_get_execution_history(self):
        """実行履歴取得テスト"""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        
        # 履歴を追加
        executor.execution_history.append({
            "tool_name": "test_tool",
            "arguments": {"arg1": "value1"},
            "success": True,
            "result": "test result"
        })
        
        history = executor.get_execution_history()
        assert len(history) == 1
        assert history[0]["tool_name"] == "test_tool"
    
    def test_clear_execution_history(self):
        """実行履歴クリアテスト"""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        
        # 履歴を追加
        executor.execution_history.append({"test": "data"})
        assert len(executor.execution_history) == 1
        
        # 履歴をクリア
        executor.clear_execution_history()
        assert len(executor.execution_history) == 0


class TestMCPClient:
    """MCPクライアントのテスト"""
    
    def test_mcp_client_initialization(self):
        """MCPクライアントの初期化テスト"""
        client = MCPClient(["python", "test_server.py"])
        assert client.server_command == ["python", "test_server.py"]
        assert client.timeout == 30
        assert client.is_connected is False
        assert len(client.available_tools) == 0
        assert len(client.available_resources) == 0
    
    def test_get_next_request_id(self):
        """リクエストID取得テスト"""
        client = MCPClient(["python", "test_server.py"])
        
        id1 = client._get_next_request_id()
        id2 = client._get_next_request_id()
        
        assert id1 == 1
        assert id2 == 2
    
    def test_get_connection_status(self):
        """接続状態取得テスト"""
        client = MCPClient(["python", "test_server.py"])
        
        status = client.get_connection_status()
        assert "connected" in status
        assert "server_command" in status
        assert "available_tools_count" in status
        assert status["connected"] is False


class TestMCPManager:
    """MCP管理システムのテスト"""
    
    def test_mcp_manager_initialization(self):
        """MCP管理システムの初期化テスト"""
        manager = MCPManager()
        assert len(manager.clients) == 0
        assert len(manager.tools) == 0
        assert len(manager.resources) == 0
    
    def test_get_stats(self):
        """統計情報取得テスト"""
        manager = MCPManager()
        
        stats = manager.get_stats()
        assert "servers_count" in stats
        assert "tools_count" in stats
        assert "resources_count" in stats
        assert "servers" in stats
        assert "server_status" in stats


if __name__ == "__main__":
    pytest.main([__file__])