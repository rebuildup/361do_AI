"""
Dynamic Tool Generator

LLMによる動的ツール生成システム
"""

import asyncio
import logging
import ast
import inspect
import types
from typing import Dict, Any, List, Optional, Union, Callable, Type
from datetime import datetime
import json
import re
import tempfile
import os

from langchain.tools import BaseTool

from ..reasoning.ollama_client import OllamaClient


class DynamicTool(BaseTool):
    """動的生成されたツール"""
    
    name: str = "dynamic_tool"
    description: str = "動的生成されたツール"
    
    def __init__(self, name: str, description: str, function_code: str, function_obj: Callable):
        super().__init__()
        self.name = name
        self.description = description
        self.function_code = function_code
        self.function_obj = function_obj
        self.logger = logging.getLogger(__name__)
    
    def _run(self, **kwargs) -> str:
        """同期実行"""
        return asyncio.run(self._arun(**kwargs))
    
    async def _arun(self, **kwargs) -> str:
        """非同期実行"""
        
        try:
            # 関数を実行
            if asyncio.iscoroutinefunction(self.function_obj):
                result = await self.function_obj(**kwargs)
            else:
                result = self.function_obj(**kwargs)
            
            # 結果を文字列に変換
            if isinstance(result, str):
                return result
            elif isinstance(result, (dict, list)):
                return json.dumps(result, ensure_ascii=False, indent=2)
            else:
                return str(result)
                
        except Exception as e:
            self.logger.error(f"Dynamic tool execution failed: {e}")
            return f"ツール実行エラー: {str(e)}"


class DynamicToolGenerator:
    """動的ツール生成器"""
    
    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        self.ollama_client = ollama_client or OllamaClient()
        self.logger = logging.getLogger(__name__)
        self.generated_tools: Dict[str, DynamicTool] = {}
        self.generation_history: List[Dict[str, Any]] = []
    
    async def generate_tool(self, 
                          tool_description: str,
                          tool_name: Optional[str] = None,
                          parameters: Optional[Dict[str, Any]] = None) -> Optional[DynamicTool]:
        """ツールを動的生成"""
        
        try:
            if not tool_description:
                self.logger.error("Tool description is required")
                return None
            
            self.logger.info(f"Generating tool: {tool_description}")
            
            # ツール名を生成
            if not tool_name:
                tool_name = await self._generate_tool_name(tool_description)
            
            # ツールコードを生成
            tool_code = await self._generate_tool_code(tool_description, tool_name, parameters)
            if not tool_code:
                return None
            
            # ツールを検証
            validation_result = self._validate_tool_code(tool_code)
            if not validation_result["valid"]:
                self.logger.error(f"Tool validation failed: {validation_result['errors']}")
                return None
            
            # ツールを実行可能な形に変換
            function_obj = self._compile_tool_code(tool_code, tool_name)
            if not function_obj:
                return None
            
            # 動的ツールを作成
            dynamic_tool = DynamicTool(
                name=tool_name,
                description=tool_description,
                function_code=tool_code,
                function_obj=function_obj
            )
            
            # 生成されたツールを記録
            self.generated_tools[tool_name] = dynamic_tool
            self.generation_history.append({
                "tool_name": tool_name,
                "description": tool_description,
                "generated_at": datetime.now().isoformat(),
                "parameters": parameters,
                "code": tool_code
            })
            
            self.logger.info(f"Tool generated successfully: {tool_name}")
            return dynamic_tool
            
        except Exception as e:
            self.logger.error(f"Tool generation failed: {e}")
            return None
    
    async def _generate_tool_name(self, description: str) -> str:
        """ツール名を生成"""
        
        try:
            prompt = f"""
以下のツールの説明に基づいて、適切なツール名を生成してください。

ツールの説明: {description}

要件:
- 英数字とアンダースコアのみ使用
- 小文字で開始
- 意味が明確で簡潔
- 既存のツール名と重複しない

ツール名のみを返してください（説明やその他のテキストは不要）:
"""
            
            response = await self.ollama_client.generate_response(prompt)
            tool_name = response.strip().lower()
            
            # ツール名を正規化
            tool_name = re.sub(r'[^a-z0-9_]', '_', tool_name)
            tool_name = re.sub(r'_+', '_', tool_name)
            tool_name = tool_name.strip('_')
            
            # 既存のツール名と重複しないように調整
            original_name = tool_name
            counter = 1
            while tool_name in self.generated_tools:
                tool_name = f"{original_name}_{counter}"
                counter += 1
            
            return tool_name
            
        except Exception as e:
            self.logger.error(f"Tool name generation failed: {e}")
            return f"generated_tool_{len(self.generated_tools) + 1}"
    
    async def _generate_tool_code(self, 
                                description: str, 
                                tool_name: str,
                                parameters: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """ツールコードを生成"""
        
        try:
            # パラメータ情報を準備
            params_info = ""
            if parameters:
                params_info = f"\nパラメータ情報:\n{json.dumps(parameters, ensure_ascii=False, indent=2)}"
            
            prompt = f"""
以下の要件に基づいて、Python関数を生成してください。

ツール名: {tool_name}
ツールの説明: {description}{params_info}

要件:
1. 関数名は '{tool_name}' であること
2. 引数は **kwargs で受け取ること
3. 戻り値は文字列または辞書/リストであること
4. エラーハンドリングを含むこと
5. 安全で実行可能なコードであること
6. 外部ライブラリは標準ライブラリのみ使用すること
7. ファイル操作やシステムコマンドは実行しないこと

生成する関数の例:
```python
def {tool_name}(**kwargs):
    try:
        # ツールの実装
        result = "処理結果"
        return result
    except Exception as e:
        return f"エラー: {{str(e)}}"
```

関数のコードのみを返してください（説明やその他のテキストは不要）:
"""
            
            response = await self.ollama_client.generate_response(prompt)
            
            # コードブロックを抽出
            code = self._extract_code_from_response(response)
            if not code:
                return None
            
            return code
            
        except Exception as e:
            self.logger.error(f"Tool code generation failed: {e}")
            return None
    
    def _extract_code_from_response(self, response: str) -> Optional[str]:
        """レスポンスからコードを抽出"""
        
        try:
            # コードブロックを検索
            code_pattern = r'```(?:python)?\s*(.*?)\s*```'
            matches = re.findall(code_pattern, response, re.DOTALL)
            
            if matches:
                return matches[0].strip()
            
            # コードブロックがない場合は、関数定義を検索
            function_pattern = r'def\s+\w+\([^)]*\):.*?(?=\n\ndef|\n\nclass|\Z)'
            match = re.search(function_pattern, response, re.DOTALL)
            
            if match:
                return match.group(0).strip()
            
            # それでも見つからない場合は、レスポンス全体を返す
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Code extraction failed: {e}")
            return None
    
    def _validate_tool_code(self, code: str) -> Dict[str, Any]:
        """ツールコードを検証"""
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # 構文チェック
            try:
                ast.parse(code)
            except SyntaxError as e:
                validation_result["errors"].append(f"Syntax error: {str(e)}")
                validation_result["valid"] = False
                return validation_result
            
            # 危険な操作をチェック
            dangerous_patterns = [
                r'import\s+os',
                r'import\s+sys',
                r'import\s+subprocess',
                r'import\s+shutil',
                r'import\s+socket',
                r'import\s+urllib',
                r'import\s+http',
                r'__import__',
                r'exec\s*\(',
                r'eval\s*\(',
                r'compile\s*\(',
                r'open\s*\(',
                r'file\s*\(',
                r'input\s*\(',
                r'raw_input\s*\(',
                r'exit\s*\(',
                r'quit\s*\('
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    validation_result["errors"].append(f"Dangerous operation detected: {pattern}")
                    validation_result["valid"] = False
            
            # 関数定義のチェック
            if 'def ' not in code:
                validation_result["errors"].append("No function definition found")
                validation_result["valid"] = False
            
            # 戻り値のチェック
            if 'return' not in code:
                validation_result["warnings"].append("No return statement found")
            
            # エラーハンドリングのチェック
            if 'try:' not in code and 'except' not in code:
                validation_result["warnings"].append("No error handling found")
            
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {str(e)}")
            validation_result["valid"] = False
        
        return validation_result
    
    def _compile_tool_code(self, code: str, tool_name: str) -> Optional[Callable]:
        """ツールコードをコンパイル"""
        
        try:
            # 一時ファイルにコードを書き込み
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # モジュールとして読み込み
                spec = importlib.util.spec_from_file_location(tool_name, temp_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # 関数を取得
                if hasattr(module, tool_name):
                    return getattr(module, tool_name)
                else:
                    self.logger.error(f"Function '{tool_name}' not found in generated code")
                    return None
                    
            finally:
                # 一時ファイルを削除
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass
                    
        except Exception as e:
            self.logger.error(f"Tool compilation failed: {e}")
            return None
    
    def get_generated_tool(self, tool_name: str) -> Optional[DynamicTool]:
        """生成されたツールを取得"""
        
        return self.generated_tools.get(tool_name)
    
    def list_generated_tools(self) -> List[Dict[str, Any]]:
        """生成されたツール一覧を取得"""
        
        tools = []
        for tool_name, tool in self.generated_tools.items():
            tools.append({
                "name": tool_name,
                "description": tool.description,
                "generated_at": next(
                    (h["generated_at"] for h in self.generation_history if h["tool_name"] == tool_name),
                    "Unknown"
                )
            })
        
        return tools
    
    def delete_generated_tool(self, tool_name: str) -> bool:
        """生成されたツールを削除"""
        
        try:
            if tool_name in self.generated_tools:
                del self.generated_tools[tool_name]
                self.logger.info(f"Generated tool deleted: {tool_name}")
                return True
            else:
                self.logger.warning(f"Generated tool not found: {tool_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to delete generated tool: {e}")
            return False
    
    def get_generation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """生成履歴を取得"""
        
        return self.generation_history[-limit:]
    
    def clear_generation_history(self):
        """生成履歴をクリア"""
        
        self.generation_history.clear()
        self.logger.info("Generation history cleared")


class ToolValidator:
    """ツール検証器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_tool(self, tool: BaseTool) -> Dict[str, Any]:
        """ツールを検証"""
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "score": 0
        }
        
        try:
            # 基本チェック
            if not hasattr(tool, 'name') or not tool.name:
                validation_result["errors"].append("Tool name is required")
                validation_result["valid"] = False
            else:
                validation_result["score"] += 10
            
            if not hasattr(tool, 'description') or not tool.description:
                validation_result["warnings"].append("Tool description is recommended")
            else:
                validation_result["score"] += 10
            
            # 実行メソッドのチェック
            if not hasattr(tool, '_run') and not hasattr(tool, '_arun'):
                validation_result["errors"].append("Tool must implement _run or _arun method")
                validation_result["valid"] = False
            else:
                validation_result["score"] += 20
            
            # 動的ツールの場合は追加チェック
            if isinstance(tool, DynamicTool):
                validation_result["score"] += 10
                
                # コードの安全性チェック
                if hasattr(tool, 'function_code'):
                    code_validation = self._validate_code_safety(tool.function_code)
                    if not code_validation["safe"]:
                        validation_result["errors"].extend(code_validation["issues"])
                        validation_result["valid"] = False
                    else:
                        validation_result["score"] += 20
            
            # 実行テスト
            try:
                if hasattr(tool, '_run'):
                    test_result = tool._run("test")
                    if isinstance(test_result, str):
                        validation_result["score"] += 20
                    else:
                        validation_result["warnings"].append("Tool should return string result")
            except Exception as e:
                validation_result["warnings"].append(f"Tool execution test failed: {str(e)}")
            
            # スコアの正規化
            validation_result["score"] = min(validation_result["score"], 100)
            
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {str(e)}")
            validation_result["valid"] = False
        
        return validation_result
    
    def _validate_code_safety(self, code: str) -> Dict[str, Any]:
        """コードの安全性を検証"""
        
        safety_result = {
            "safe": True,
            "issues": []
        }
        
        try:
            # 危険なパターンをチェック
            dangerous_patterns = [
                (r'import\s+os', "OS module import"),
                (r'import\s+sys', "System module import"),
                (r'import\s+subprocess', "Subprocess module import"),
                (r'import\s+shutil', "Shutil module import"),
                (r'import\s+socket', "Socket module import"),
                (r'import\s+urllib', "URLLib module import"),
                (r'import\s+http', "HTTP module import"),
                (r'__import__', "Dynamic import"),
                (r'exec\s*\(', "Exec function call"),
                (r'eval\s*\(', "Eval function call"),
                (r'compile\s*\(', "Compile function call"),
                (r'open\s*\(', "File open operation"),
                (r'file\s*\(', "File function call"),
                (r'input\s*\(', "Input function call"),
                (r'raw_input\s*\(', "Raw input function call"),
                (r'exit\s*\(', "Exit function call"),
                (r'quit\s*\(', "Quit function call")
            ]
            
            for pattern, description in dangerous_patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    safety_result["issues"].append(f"Potentially dangerous: {description}")
                    safety_result["safe"] = False
            
        except Exception as e:
            safety_result["issues"].append(f"Safety validation error: {str(e)}")
            safety_result["safe"] = False
        
        return safety_result


# 必要なインポートを追加
import importlib.util
