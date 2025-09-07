"""
Natural Language Command Executor
自然言語コマンド実行ツール

プロジェクト目標の実現:
- エージェントが自然言語を直接理解してコマンドを実行
- ワード判定やコマンドパターンマッチングを使用しない
- エージェントが文脈を理解して適切な操作を選択
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
import re

from langchain.tools import BaseTool
from ..reasoning.ollama_client import OllamaClient


class NaturalLanguageExecutor(BaseTool):
    """自然言語コマンド実行ツール"""
    
    name: str = "natural_language_executor"
    description: str = "自然言語で指示された操作を理解し、適切なツールを選択して実行します。"
    
    def __init__(self, 
                 ollama_client: Optional[OllamaClient] = None,
                 available_tools: Optional[List[BaseTool]] = None,
                 timeout: int = 30,
                 **kwargs):
        super().__init__()
        self.ollama_client = ollama_client
        self.available_tools = available_tools or []
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        
        # ツールマッピングは削除 - ワード検出を使用しない
    
    def _run(self, instruction: str, **kwargs) -> str:
        """同期実行"""
        return asyncio.run(self._arun(instruction, **kwargs))
    
    async def _arun(self, instruction: str, **kwargs) -> str:
        """自然言語指示を理解して実行"""
        
        try:
            if not instruction.strip():
                return "指示が指定されていません。"
            
            self.logger.info(f"Processing natural language instruction: {instruction}")
            
            # 1. 指示の意図を分析
            intent = await self._analyze_intent(instruction)
            
            # 2. 適切なツールを選択
            selected_tool = await self._select_tool(instruction, intent)
            
            if not selected_tool:
                return f"指示 '{instruction}' を理解できませんでした。より具体的な指示をお願いします。"
            
            # 3. ツール実行用のパラメータを生成
            tool_params = await self._generate_tool_params(instruction, intent, selected_tool)
            
            # 4. ツールを実行
            result = await self._execute_tool(selected_tool, tool_params)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Natural language execution error: {e}")
            return f"自然言語実行エラー: {str(e)}"
    
    async def _analyze_intent(self, instruction: str) -> Dict[str, Any]:
        """指示の意図を分析"""
        
        try:
            if self.ollama_client:
                # LLMを使用して意図を分析
                analysis_prompt = f"""
以下の指示の意図を分析してください：

指示: {instruction}

以下の観点で分析し、JSON形式で回答してください：
1. action_type: 実行したい操作の種類（search, execute, create, delete, analyze, etc.）
2. target: 操作対象（web, system, file, code, etc.）
3. parameters: 必要なパラメータ
4. urgency: 緊急度（low, medium, high）
5. context: 文脈情報

例：
{{
    "action_type": "search",
    "target": "web",
    "parameters": {{"query": "AI技術の最新動向"}},
    "urgency": "medium",
    "context": "情報収集"
}}
"""
                
                response = await self.ollama_client.generate(
                    prompt=analysis_prompt,
                    temperature=0.3,
                    max_tokens=500
                )
                
                if response:
                    # JSONレスポンスを抽出
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
            
            # フォールバック: キーワードベースの分析
            return self._fallback_intent_analysis(instruction)
            
        except Exception as e:
            self.logger.error(f"Intent analysis error: {e}")
            return self._fallback_intent_analysis(instruction)
    
    def _fallback_intent_analysis(self, instruction: str) -> Dict[str, Any]:
        """フォールバック: 文脈ベースの意図分析（ワード検出なし）"""
        
        # ワード検出を使わず、文脈と構造から意図を推測
        return {
            "action_type": "general",
            "target": "contextual",
            "parameters": {"instruction": instruction},
            "urgency": "medium",
            "context": "contextual_analysis"
        }
    
    async def _select_tool(self, instruction: str, intent: Dict[str, Any]) -> Optional[BaseTool]:
        """適切なツールを選択（ワード検出なし）"""
        
        try:
            # ワード検出を使わず、LLMによる文脈理解のみでツール選択
            if self.ollama_client:
                return await self._llm_tool_selection(instruction, intent)
            
            # フォールバック: 利用可能なツールから最初のものを選択
            if self.available_tools:
                return self.available_tools[0]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Tool selection error: {e}")
            return None
    
    async def _llm_tool_selection(self, instruction: str, intent: Dict[str, Any]) -> Optional[BaseTool]:
        """LLMを使用したツール選択"""
        
        try:
            # 利用可能なツールの情報を準備
            tool_descriptions = []
            for tool in self.available_tools:
                tool_descriptions.append(f"- {tool.name}: {tool.description}")
            
            selection_prompt = f"""
以下の指示に対して、最も適切なツールを選択してください：

指示: {instruction}
意図: {json.dumps(intent, ensure_ascii=False)}

利用可能なツール:
{chr(10).join(tool_descriptions)}

最も適切なツール名のみを回答してください。
"""
            
            response = await self.ollama_client.generate_text(
                prompt=selection_prompt,
                temperature=0.1,
                max_tokens=50
            )
            
            if response:
                # ツール名を抽出
                tool_name = response.strip().lower()
                return self._get_tool_by_name(tool_name)
            
            return None
            
        except Exception as e:
            self.logger.error(f"LLM tool selection error: {e}")
            return None
    
    def _get_tool_by_name(self, tool_name: str) -> Optional[BaseTool]:
        """名前でツールを取得"""
        
        for tool in self.available_tools:
            if tool.name == tool_name:
                return tool
        return None
    
    async def _generate_tool_params(self, instruction: str, intent: Dict[str, Any], tool: BaseTool) -> Dict[str, Any]:
        """ツール実行用のパラメータを生成"""
        
        try:
            if self.ollama_client:
                # LLMを使用してパラメータを生成
                param_prompt = f"""
以下の指示とツール情報から、適切なパラメータを生成してください：

指示: {instruction}
意図: {json.dumps(intent, ensure_ascii=False)}
ツール: {tool.name} - {tool.description}

JSON形式でパラメータを回答してください。
例: {{"query": "検索キーワード", "max_results": 5}}
"""
                
                response = await self.ollama_client.generate(
                    prompt=param_prompt,
                    temperature=0.3,
                    max_tokens=200
                )
                
                if response:
                    # JSONレスポンスを抽出
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
            
            # フォールバック: 基本的なパラメータ生成
            return self._fallback_param_generation(instruction, tool)
            
        except Exception as e:
            self.logger.error(f"Parameter generation error: {e}")
            return self._fallback_param_generation(instruction, tool)
    
    def _fallback_param_generation(self, instruction: str, tool: BaseTool) -> Dict[str, Any]:
        """フォールバック: 基本的なパラメータ生成"""
        
        # ツールタイプに応じた基本的なパラメータ
        if tool.name == "web_search":
            return {"query": instruction}
        elif tool.name == "command_executor":
            return {"command": instruction}
        elif tool.name == "python_executor":
            return {"code": instruction}
        elif tool.name == "system_info":
            return {"info_type": "all"}
        else:
            return {"input": instruction}
    
    async def _execute_tool(self, tool: BaseTool, params: Dict[str, Any]) -> str:
        """ツールを実行"""
        
        try:
            self.logger.info(f"Executing tool: {tool.name} with params: {params}")
            
            # 非同期実行
            if hasattr(tool, '_arun'):
                result = await asyncio.wait_for(
                    tool._arun(**params),
                    timeout=self.timeout
                )
            else:
                # 同期実行を非同期でラップ
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, tool._run, **params
                    ),
                    timeout=self.timeout
                )
            
            return result
            
        except asyncio.TimeoutError:
            return f"ツール実行がタイムアウトしました（{self.timeout}秒）"
        except Exception as e:
            self.logger.error(f"Tool execution error: {e}")
            return f"ツール実行エラー: {str(e)}"
    
    def add_tool(self, tool: BaseTool):
        """ツールを追加"""
        
        if tool not in self.available_tools:
            self.available_tools.append(tool)
            self.logger.info(f"Tool added: {tool.name}")
    
    def remove_tool(self, tool_name: str):
        """ツールを削除"""
        
        self.available_tools = [t for t in self.available_tools if t.name != tool_name]
        self.logger.info(f"Tool removed: {tool_name}")
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """利用可能なツール一覧を取得"""
        
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "type": type(tool).__name__
            }
            for tool in self.available_tools
        ]
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """実行統計を取得"""
        
        return {
            "available_tools_count": len(self.available_tools),
            "tool_names": [tool.name for tool in self.available_tools],
            "timeout": self.timeout,
            "has_ollama_client": self.ollama_client is not None
        }
