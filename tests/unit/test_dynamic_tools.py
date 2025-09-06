"""
Tests for Dynamic Tools

動的ツール生成システムのテスト
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import os

from src.advanced_agent.tools.dynamic_tool_generator import (
    DynamicTool, DynamicToolGenerator, ToolValidator
)
from src.advanced_agent.reasoning.ollama_client import OllamaClient


class TestDynamicTool:
    """動的ツールのテスト"""
    
    def test_dynamic_tool_initialization(self):
        """動的ツールの初期化テスト"""
        def test_function(**kwargs):
            return "test result"
        
        tool = DynamicTool(
            name="test_tool",
            description="Test dynamic tool",
            function_code="def test_tool(**kwargs): return 'test result'",
            function_obj=test_function
        )
        
        assert tool.name == "test_tool"
        assert tool.description == "Test dynamic tool"
        assert tool.function_code == "def test_tool(**kwargs): return 'test result'"
        assert tool.function_obj == test_function
    
    def test_dynamic_tool_sync_execution(self):
        """動的ツール同期実行テスト"""
        def test_function(**kwargs):
            return f"Hello {kwargs.get('name', 'World')}"
        
        tool = DynamicTool(
            name="test_tool",
            description="Test dynamic tool",
            function_code="def test_tool(**kwargs): return f'Hello {kwargs.get(\"name\", \"World\")}'",
            function_obj=test_function
        )
        
        result = tool._run(name="Alice")
        assert result == "Hello Alice"
    
    @pytest.mark.asyncio
    async def test_dynamic_tool_async_execution(self):
        """動的ツール非同期実行テスト"""
        async def test_function(**kwargs):
            return f"Hello {kwargs.get('name', 'World')}"
        
        tool = DynamicTool(
            name="test_tool",
            description="Test dynamic tool",
            function_code="async def test_tool(**kwargs): return f'Hello {kwargs.get(\"name\", \"World\")}'",
            function_obj=test_function
        )
        
        result = await tool._arun(name="Bob")
        assert result == "Hello Bob"
    
    def test_dynamic_tool_execution_with_dict_result(self):
        """動的ツール辞書結果テスト"""
        def test_function(**kwargs):
            return {"message": "Hello", "name": kwargs.get("name", "World")}
        
        tool = DynamicTool(
            name="test_tool",
            description="Test dynamic tool",
            function_code="def test_tool(**kwargs): return {'message': 'Hello', 'name': kwargs.get('name', 'World')}",
            function_obj=test_function
        )
        
        result = tool._run(name="Charlie")
        assert '"message": "Hello"' in result
        assert '"name": "Charlie"' in result
    
    def test_dynamic_tool_execution_error(self):
        """動的ツール実行エラーテスト"""
        def test_function(**kwargs):
            raise ValueError("Test error")
        
        tool = DynamicTool(
            name="test_tool",
            description="Test dynamic tool",
            function_code="def test_tool(**kwargs): raise ValueError('Test error')",
            function_obj=test_function
        )
        
        result = tool._run()
        assert "ツール実行エラー: Test error" in result


class TestDynamicToolGenerator:
    """動的ツール生成器のテスト"""
    
    def test_dynamic_tool_generator_initialization(self):
        """動的ツール生成器の初期化テスト"""
        mock_ollama = Mock(spec=OllamaClient)
        generator = DynamicToolGenerator(mock_ollama)
        
        assert generator.ollama_client == mock_ollama
        assert len(generator.generated_tools) == 0
        assert len(generator.generation_history) == 0
    
    def test_dynamic_tool_generator_default_initialization(self):
        """動的ツール生成器デフォルト初期化テスト"""
        generator = DynamicToolGenerator()
        
        assert generator.ollama_client is not None
        assert len(generator.generated_tools) == 0
        assert len(generator.generation_history) == 0
    
    @pytest.mark.asyncio
    async def test_generate_tool_name(self):
        """ツール名生成テスト"""
        mock_ollama = Mock(spec=OllamaClient)
        mock_ollama.generate_response = AsyncMock(return_value="test_tool_name")
        
        generator = DynamicToolGenerator(mock_ollama)
        tool_name = await generator._generate_tool_name("Test tool description")
        
        assert tool_name == "test_tool_name"
        mock_ollama.generate_response.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_tool_name_with_duplicate(self):
        """重複ツール名生成テスト"""
        mock_ollama = Mock(spec=OllamaClient)
        mock_ollama.generate_response = AsyncMock(return_value="test_tool")
        
        generator = DynamicToolGenerator(mock_ollama)
        
        # 最初のツール名を生成
        tool_name1 = await generator._generate_tool_name("Test tool 1")
        assert tool_name1 == "test_tool"
        
        # ダミーのツールを追加
        generator.generated_tools["test_tool"] = Mock()
        
        # 重複するツール名を生成
        tool_name2 = await generator._generate_tool_name("Test tool 2")
        assert tool_name2 == "test_tool_1"
    
    @pytest.mark.asyncio
    async def test_generate_tool_code(self):
        """ツールコード生成テスト"""
        mock_ollama = Mock(spec=OllamaClient)
        mock_ollama.generate_response = AsyncMock(return_value="""
```python
def test_tool(**kwargs):
    try:
        result = "Test result"
        return result
    except Exception as e:
        return f"Error: {str(e)}"
```
""")
        
        generator = DynamicToolGenerator(mock_ollama)
        code = await generator._generate_tool_code("Test tool description", "test_tool")
        
        assert "def test_tool(**kwargs):" in code
        assert "return result" in code
        mock_ollama.generate_response.assert_called_once()
    
    def test_extract_code_from_response_with_code_block(self):
        """コードブロック付きレスポンスからコード抽出テスト"""
        mock_ollama = Mock(spec=OllamaClient)
        generator = DynamicToolGenerator(mock_ollama)
        
        response = """
Here's the code:

```python
def test_tool(**kwargs):
    return "test"
```
"""
        
        code = generator._extract_code_from_response(response)
        assert "def test_tool(**kwargs):" in code
        assert "return \"test\"" in code
    
    def test_extract_code_from_response_without_code_block(self):
        """コードブロックなしレスポンスからコード抽出テスト"""
        mock_ollama = Mock(spec=OllamaClient)
        generator = DynamicToolGenerator(mock_ollama)
        
        response = """
def test_tool(**kwargs):
    return "test"
"""
        
        code = generator._extract_code_from_response(response)
        assert "def test_tool(**kwargs):" in code
        assert "return \"test\"" in code
    
    def test_validate_tool_code_valid(self):
        """有効なツールコード検証テスト"""
        mock_ollama = Mock(spec=OllamaClient)
        generator = DynamicToolGenerator(mock_ollama)
        
        code = """
def test_tool(**kwargs):
    try:
        result = "test result"
        return result
    except Exception as e:
        return f"Error: {str(e)}"
"""
        
        validation = generator._validate_tool_code(code)
        assert validation["valid"] is True
        assert len(validation["errors"]) == 0
    
    def test_validate_tool_code_invalid_syntax(self):
        """無効な構文のツールコード検証テスト"""
        mock_ollama = Mock(spec=OllamaClient)
        generator = DynamicToolGenerator(mock_ollama)
        
        code = """
def test_tool(**kwargs):
    result = "test result"
    return result  # Missing closing parenthesis
"""
        
        validation = generator._validate_tool_code(code)
        assert validation["valid"] is False
        assert len(validation["errors"]) > 0
    
    def test_validate_tool_code_dangerous_operation(self):
        """危険な操作を含むツールコード検証テスト"""
        mock_ollama = Mock(spec=OllamaClient)
        generator = DynamicToolGenerator(mock_ollama)
        
        code = """
def test_tool(**kwargs):
    import os
    return os.getcwd()
"""
        
        validation = generator._validate_tool_code(code)
        assert validation["valid"] is False
        assert len(validation["errors"]) > 0
        assert any("Dangerous operation" in error for error in validation["errors"])
    
    def test_validate_tool_code_no_function(self):
        """関数定義なしのツールコード検証テスト"""
        mock_ollama = Mock(spec=OllamaClient)
        generator = DynamicToolGenerator(mock_ollama)
        
        code = """
result = "test result"
print(result)
"""
        
        validation = generator._validate_tool_code(code)
        assert validation["valid"] is False
        assert any("No function definition" in error for error in validation["errors"])
    
    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_compile_tool_code_success(self, mock_module_from_spec, mock_spec_from_file):
        """ツールコードコンパイル成功テスト"""
        mock_ollama = Mock(spec=OllamaClient)
        generator = DynamicToolGenerator(mock_ollama)
        
        # モックの設定
        mock_spec = Mock()
        mock_spec_from_file.return_value = mock_spec
        
        mock_module = Mock()
        mock_module_from_spec.return_value = mock_module
        
        mock_loader = Mock()
        mock_spec.loader = mock_loader
        
        def test_function(**kwargs):
            return "test result"
        
        mock_module.test_tool = test_function
        
        code = """
def test_tool(**kwargs):
    return "test result"
"""
        
        function_obj = generator._compile_tool_code(code, "test_tool")
        
        assert function_obj == test_function
        mock_spec_from_file.assert_called_once()
        mock_loader.exec_module.assert_called_once_with(mock_module)
    
    @patch('importlib.util.spec_from_file_location')
    def test_compile_tool_code_function_not_found(self, mock_spec_from_file):
        """ツールコードコンパイル - 関数未発見テスト"""
        mock_ollama = Mock(spec=OllamaClient)
        generator = DynamicToolGenerator(mock_ollama)
        
        # モックの設定
        mock_spec = Mock()
        mock_spec_from_file.return_value = mock_spec
        
        mock_module = Mock()
        mock_spec.loader = Mock()
        
        # 関数が存在しないモジュール
        mock_module.test_tool = None
        
        code = """
def other_function(**kwargs):
    return "test result"
"""
        
        function_obj = generator._compile_tool_code(code, "test_tool")
        
        assert function_obj is None
    
    def test_get_generated_tool(self):
        """生成されたツール取得テスト"""
        mock_ollama = Mock(spec=OllamaClient)
        generator = DynamicToolGenerator(mock_ollama)
        
        # ダミーのツールを追加
        mock_tool = Mock()
        generator.generated_tools["test_tool"] = mock_tool
        
        retrieved_tool = generator.get_generated_tool("test_tool")
        assert retrieved_tool == mock_tool
        
        # 存在しないツール
        retrieved_tool = generator.get_generated_tool("nonexistent_tool")
        assert retrieved_tool is None
    
    def test_list_generated_tools(self):
        """生成されたツール一覧取得テスト"""
        mock_ollama = Mock(spec=OllamaClient)
        generator = DynamicToolGenerator(mock_ollama)
        
        # ダミーのツールと履歴を追加
        mock_tool = Mock()
        mock_tool.description = "Test tool description"
        generator.generated_tools["test_tool"] = mock_tool
        
        generator.generation_history.append({
            "tool_name": "test_tool",
            "generated_at": "2024-01-01T00:00:00"
        })
        
        tools = generator.list_generated_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "test_tool"
        assert tools[0]["description"] == "Test tool description"
        assert tools[0]["generated_at"] == "2024-01-01T00:00:00"
    
    def test_delete_generated_tool(self):
        """生成されたツール削除テスト"""
        mock_ollama = Mock(spec=OllamaClient)
        generator = DynamicToolGenerator(mock_ollama)
        
        # ダミーのツールを追加
        mock_tool = Mock()
        generator.generated_tools["test_tool"] = mock_tool
        
        # ツールを削除
        result = generator.delete_generated_tool("test_tool")
        assert result is True
        assert "test_tool" not in generator.generated_tools
        
        # 存在しないツールを削除
        result = generator.delete_generated_tool("nonexistent_tool")
        assert result is False
    
    def test_get_generation_history(self):
        """生成履歴取得テスト"""
        mock_ollama = Mock(spec=OllamaClient)
        generator = DynamicToolGenerator(mock_ollama)
        
        # 履歴を追加
        generator.generation_history.extend([
            {"tool_name": "tool1", "generated_at": "2024-01-01T00:00:00"},
            {"tool_name": "tool2", "generated_at": "2024-01-02T00:00:00"},
            {"tool_name": "tool3", "generated_at": "2024-01-03T00:00:00"}
        ])
        
        # 制限なしで取得
        history = generator.get_generation_history()
        assert len(history) == 3
        
        # 制限ありで取得
        history = generator.get_generation_history(limit=2)
        assert len(history) == 2
        assert history[0]["tool_name"] == "tool2"
        assert history[1]["tool_name"] == "tool3"
    
    def test_clear_generation_history(self):
        """生成履歴クリアテスト"""
        mock_ollama = Mock(spec=OllamaClient)
        generator = DynamicToolGenerator(mock_ollama)
        
        # 履歴を追加
        generator.generation_history.append({"tool_name": "test_tool"})
        assert len(generator.generation_history) == 1
        
        # 履歴をクリア
        generator.clear_generation_history()
        assert len(generator.generation_history) == 0


class TestToolValidator:
    """ツール検証器のテスト"""
    
    def test_tool_validator_initialization(self):
        """ツール検証器の初期化テスト"""
        validator = ToolValidator()
        assert validator is not None
    
    def test_validate_tool_valid(self):
        """有効なツール検証テスト"""
        validator = ToolValidator()
        
        # モックツール
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool description"
        mock_tool._run = Mock(return_value="test result")
        
        validation = validator.validate_tool(mock_tool)
        
        assert validation["valid"] is True
        assert len(validation["errors"]) == 0
        assert validation["score"] > 0
    
    def test_validate_tool_no_name(self):
        """名前なしツール検証テスト"""
        validator = ToolValidator()
        
        # モックツール（名前なし）
        mock_tool = Mock()
        mock_tool.name = ""
        mock_tool.description = "Test tool description"
        mock_tool._run = Mock(return_value="test result")
        
        validation = validator.validate_tool(mock_tool)
        
        assert validation["valid"] is False
        assert any("Tool name is required" in error for error in validation["errors"])
    
    def test_validate_tool_no_description(self):
        """説明なしツール検証テスト"""
        validator = ToolValidator()
        
        # モックツール（説明なし）
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = ""
        mock_tool._run = Mock(return_value="test result")
        
        validation = validator.validate_tool(mock_tool)
        
        assert validation["valid"] is True  # 説明は警告のみ
        assert any("Tool description is recommended" in warning for warning in validation["warnings"])
    
    def test_validate_tool_no_run_method(self):
        """実行メソッドなしツール検証テスト"""
        validator = ToolValidator()
        
        # モックツール（実行メソッドなし）
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool description"
        # _run と _arun の両方を削除
        del mock_tool._run
        del mock_tool._arun
        
        validation = validator.validate_tool(mock_tool)
        
        assert validation["valid"] is False
        assert any("Tool must implement _run or _arun method" in error for error in validation["errors"])
    
    def test_validate_dynamic_tool(self):
        """動的ツール検証テスト"""
        validator = ToolValidator()
        
        # 動的ツール
        def test_function(**kwargs):
            return "test result"
        
        dynamic_tool = DynamicTool(
            name="test_tool",
            description="Test dynamic tool",
            function_code="def test_tool(**kwargs): return 'test result'",
            function_obj=test_function
        )
        
        validation = validator.validate_tool(dynamic_tool)
        
        assert validation["valid"] is True
        assert validation["score"] > 0  # 動的ツールのボーナススコア
    
    def test_validate_code_safety_safe(self):
        """安全なコード検証テスト"""
        validator = ToolValidator()
        
        code = """
def test_tool(**kwargs):
    try:
        result = "test result"
        return result
    except Exception as e:
        return f"Error: {str(e)}"
"""
        
        safety = validator._validate_code_safety(code)
        
        assert safety["safe"] is True
        assert len(safety["issues"]) == 0
    
    def test_validate_code_safety_dangerous(self):
        """危険なコード検証テスト"""
        validator = ToolValidator()
        
        code = """
def test_tool(**kwargs):
    import os
    return os.getcwd()
"""
        
        safety = validator._validate_code_safety(code)
        
        assert safety["safe"] is False
        assert len(safety["issues"]) > 0
        assert any("OS module import" in issue for issue in safety["issues"])
    
    def test_validate_code_safety_multiple_dangerous(self):
        """複数の危険な操作を含むコード検証テスト"""
        validator = ToolValidator()
        
        code = """
def test_tool(**kwargs):
    import os
    import subprocess
    exec("print('hello')")
    return "test"
"""
        
        safety = validator._validate_code_safety(code)
        
        assert safety["safe"] is False
        assert len(safety["issues"]) >= 3  # 複数の危険な操作


if __name__ == "__main__":
    pytest.main([__file__])
