"""
LangChain Tools による推論結果構造化
カスタムツールとチェーン実装
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Type, Union
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod

from langchain_core.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser, PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains import LLMChain
from langchain.schema import BaseMessage

from .ollama_client import OllamaClient, InferenceRequest, InferenceResponse
from ..core.logger import get_logger


# 構造化出力用のPydanticモデル
class ReasoningStep(BaseModel):
    """推論ステップ"""
    step_number: int = Field(description="ステップ番号")
    thought: str = Field(description="思考内容")
    action: Optional[str] = Field(description="実行するアクション", default=None)
    observation: Optional[str] = Field(description="観察結果", default=None)
    confidence: float = Field(description="信頼度 (0-1)", ge=0, le=1)


class StructuredResponse(BaseModel):
    """構造化レスポンス"""
    summary: str = Field(description="回答の要約")
    reasoning_steps: List[ReasoningStep] = Field(description="推論ステップ")
    final_answer: str = Field(description="最終回答")
    confidence_score: float = Field(description="全体の信頼度", ge=0, le=1)
    sources: List[str] = Field(description="参考情報源", default_factory=list)
    tags: List[str] = Field(description="タグ", default_factory=list)


class CodeAnalysis(BaseModel):
    """コード解析結果"""
    language: str = Field(description="プログラミング言語")
    complexity: str = Field(description="複雑度 (low/medium/high)")
    issues: List[str] = Field(description="発見された問題", default_factory=list)
    suggestions: List[str] = Field(description="改善提案", default_factory=list)
    quality_score: float = Field(description="品質スコア (0-10)", ge=0, le=10)


class TaskBreakdown(BaseModel):
    """タスク分解結果"""
    main_task: str = Field(description="メインタスク")
    subtasks: List[str] = Field(description="サブタスク")
    dependencies: Dict[str, List[str]] = Field(description="依存関係", default_factory=dict)
    estimated_time: str = Field(description="推定時間")
    difficulty: str = Field(description="難易度 (easy/medium/hard)")


# カスタムツール実装
class StructuredReasoningTool(BaseTool):
    """構造化推論ツール"""
    
    name: str = "structured_reasoning"
    description: str = "複雑な問題を段階的に推論し、構造化された回答を生成します"
    
    def __init__(self, ollama_client: OllamaClient):
        super().__init__()
        self.ollama_client = ollama_client
        self.logger = get_logger()
        
        # 構造化推論用プロンプトテンプレート
        self.prompt_template = PromptTemplate(
            input_variables=["question", "context"],
            template="""
あなたは論理的思考に優れたAIアシスタントです。以下の質問に対して、段階的に推論を行い、構造化された回答を提供してください。

質問: {question}

コンテキスト: {context}

以下の形式で回答してください：

## 推論プロセス
1. **ステップ1**: [思考内容] (信頼度: X.X)
2. **ステップ2**: [思考内容] (信頼度: X.X)
...

## 最終回答
[明確で簡潔な回答]

## 信頼度
全体の信頼度: X.X

## 参考情報
- [情報源1]
- [情報源2]

回答は日本語で行ってください。
"""
        )
    
    def _run(self, question: str, context: str = "") -> str:
        """同期実行（非推奨）"""
        return asyncio.run(self._arun(question, context))
    
    async def _arun(self, question: str, context: str = "") -> str:
        """非同期実行"""
        try:
            # プロンプト生成
            prompt = self.prompt_template.format(question=question, context=context)
            
            # 推論実行
            request = InferenceRequest(
                prompt=prompt,
                temperature=0.1,
                system_message="あなたは論理的で構造化された思考を行うAIアシスタントです。"
            )
            
            response = await self.ollama_client.generate(request)
            
            # ログ記録
            self.logger.log_performance_metric(
                metric_name="structured_reasoning_time",
                value=response.processing_time,
                unit="seconds",
                component="structured_reasoning_tool"
            )
            
            return response.content
            
        except Exception as e:
            self.logger.log_alert(
                alert_type="tool_execution_error",
                severity="ERROR",
                message=f"Structured reasoning tool failed: {e}"
            )
            return f"推論中にエラーが発生しました: {e}"


class CodeAnalysisTool(BaseTool):
    """コード解析ツール"""
    
    name: str = "code_analysis"
    description: str = "プログラムコードを解析し、品質評価と改善提案を行います"
    
    def __init__(self, ollama_client: OllamaClient):
        super().__init__()
        self.ollama_client = ollama_client
        self.logger = get_logger()
        
        self.prompt_template = PromptTemplate(
            input_variables=["code", "language"],
            template="""
以下のコードを解析し、品質評価と改善提案を行ってください。

プログラミング言語: {language}
コード:
```{language}
{code}
```

以下の観点で分析してください：
1. コードの複雑度 (low/medium/high)
2. 潜在的な問題やバグ
3. パフォーマンスの改善点
4. 可読性の改善提案
5. セキュリティ上の懸念
6. 品質スコア (0-10点)

分析結果を構造化して回答してください。
"""
        )
    
    def _run(self, code: str, language: str = "python") -> str:
        """同期実行"""
        return asyncio.run(self._arun(code, language))
    
    async def _arun(self, code: str, language: str = "python") -> str:
        """非同期実行"""
        try:
            prompt = self.prompt_template.format(code=code, language=language)
            
            request = InferenceRequest(
                prompt=prompt,
                temperature=0.1,
                system_message="あなたはコード品質の専門家です。詳細で建設的な分析を行ってください。"
            )
            
            response = await self.ollama_client.generate(request)
            
            self.logger.log_performance_metric(
                metric_name="code_analysis_time",
                value=response.processing_time,
                unit="seconds",
                component="code_analysis_tool"
            )
            
            return response.content
            
        except Exception as e:
            self.logger.log_alert(
                alert_type="tool_execution_error",
                severity="ERROR",
                message=f"Code analysis tool failed: {e}"
            )
            return f"コード解析中にエラーが発生しました: {e}"


class TaskBreakdownTool(BaseTool):
    """タスク分解ツール"""
    
    name: str = "task_breakdown"
    description: str = "複雑なタスクを実行可能なサブタスクに分解します"
    
    def __init__(self, ollama_client: OllamaClient):
        super().__init__()
        self.ollama_client = ollama_client
        self.logger = get_logger()
        
        self.prompt_template = PromptTemplate(
            input_variables=["task", "constraints"],
            template="""
以下のタスクを実行可能なサブタスクに分解してください。

メインタスク: {task}
制約条件: {constraints}

以下の形式で分解してください：

## メインタスク
{task}

## サブタスク
1. [サブタスク1]
2. [サブタスク2]
...

## 依存関係
- サブタスク1 → サブタスク2
- サブタスク2 → サブタスク3

## 推定時間
各サブタスクの推定時間と全体の推定時間

## 難易度評価
easy/medium/hard で評価

## 注意点
実行時の注意点や考慮事項

実用的で実行可能な分解を行ってください。
"""
        )
    
    def _run(self, task: str, constraints: str = "") -> str:
        """同期実行"""
        return asyncio.run(self._arun(task, constraints))
    
    async def _arun(self, task: str, constraints: str = "") -> str:
        """非同期実行"""
        try:
            prompt = self.prompt_template.format(task=task, constraints=constraints)
            
            request = InferenceRequest(
                prompt=prompt,
                temperature=0.2,
                system_message="あなたはプロジェクト管理の専門家です。実用的なタスク分解を行ってください。"
            )
            
            response = await self.ollama_client.generate(request)
            
            self.logger.log_performance_metric(
                metric_name="task_breakdown_time",
                value=response.processing_time,
                unit="seconds",
                component="task_breakdown_tool"
            )
            
            return response.content
            
        except Exception as e:
            self.logger.log_alert(
                alert_type="tool_execution_error",
                severity="ERROR",
                message=f"Task breakdown tool failed: {e}"
            )
            return f"タスク分解中にエラーが発生しました: {e}"


class MemorySearchTool(BaseTool):
    """記憶検索ツール（プレースホルダー）"""
    
    name: str = "memory_search"
    description: str = "過去の会話や学習内容から関連情報を検索します"
    
    def __init__(self, ollama_client: OllamaClient):
        super().__init__()
        self.ollama_client = ollama_client
        self.logger = get_logger()
    
    def _run(self, query: str, limit: int = 5) -> str:
        """同期実行"""
        return asyncio.run(self._arun(query, limit))
    
    async def _arun(self, query: str, limit: int = 5) -> str:
        """非同期実行"""
        # TODO: 実際の記憶システムと統合
        self.logger.log_performance_metric(
            metric_name="memory_search_time",
            value=0.1,
            unit="seconds",
            component="memory_search_tool"
        )
        
        return f"記憶検索結果（クエリ: {query}）:\n- 関連する記憶が見つかりませんでした。"


class SystemMonitorTool(BaseTool):
    """システム監視ツール"""
    
    name: str = "system_monitor"
    description: str = "システムの状態（CPU、メモリ、GPU使用率）を確認します"
    
    def __init__(self, ollama_client: OllamaClient):
        super().__init__()
        self.ollama_client = ollama_client
        self.logger = get_logger()
    
    def _run(self) -> str:
        """同期実行"""
        return asyncio.run(self._arun())
    
    async def _arun(self) -> str:
        """非同期実行"""
        try:
            # システム監視モジュールから情報取得（簡略化）
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            result = f"""システム状態:
- CPU使用率: {cpu_percent:.1f}%
- メモリ使用率: {memory.percent:.1f}%
- 利用可能メモリ: {memory.available / (1024**3):.1f}GB
"""
            
            self.logger.log_system_stats({
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "available_memory_gb": memory.available / (1024**3)
            })
            
            return result
            
        except Exception as e:
            return f"システム監視エラー: {e}"


class ToolManager:
    """ツール管理クラス"""
    
    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client
        self.logger = get_logger()
        
        # 利用可能ツール
        self.tools = {
            "structured_reasoning": StructuredReasoningTool(ollama_client),
            "code_analysis": CodeAnalysisTool(ollama_client),
            "task_breakdown": TaskBreakdownTool(ollama_client),
            "memory_search": MemorySearchTool(ollama_client),
            "system_monitor": SystemMonitorTool(ollama_client)
        }
        
        self.logger.log_startup(
            component="tool_manager",
            version="1.0.0",
            config_summary={
                "available_tools": len(self.tools),
                "tool_names": list(self.tools.keys())
            }
        )
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """ツール取得"""
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[Dict[str, str]]:
        """利用可能ツール一覧"""
        return [
            {
                "name": tool.name,
                "description": tool.description
            }
            for tool in self.tools.values()
        ]
    
    async def execute_tool(self, tool_name: str, **kwargs) -> str:
        """ツール実行"""
        tool = self.get_tool(tool_name)
        if not tool:
            return f"ツール '{tool_name}' が見つかりません。"
        
        try:
            start_time = datetime.now()
            
            # ツール実行
            if hasattr(tool, '_arun'):
                result = await tool._arun(**kwargs)
            else:
                result = tool._run(**kwargs)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.log_performance_metric(
                metric_name=f"{tool_name}_execution_time",
                value=execution_time,
                unit="seconds",
                component="tool_manager"
            )
            
            return result
            
        except Exception as e:
            self.logger.log_alert(
                alert_type="tool_execution_error",
                severity="ERROR",
                message=f"Tool {tool_name} execution failed: {e}"
            )
            return f"ツール実行エラー: {e}"
    
    async def auto_select_tool(self, user_input: str) -> Optional[str]:
        """ユーザー入力に基づく自動ツール選択"""
        # 簡単なキーワードベース選択
        input_lower = user_input.lower()
        
        if any(keyword in input_lower for keyword in ["分析", "解析", "コード", "プログラム"]):
            return "code_analysis"
        elif any(keyword in input_lower for keyword in ["タスク", "分解", "計画", "ステップ"]):
            return "task_breakdown"
        elif any(keyword in input_lower for keyword in ["推論", "考え", "理由", "なぜ"]):
            return "structured_reasoning"
        elif any(keyword in input_lower for keyword in ["記憶", "過去", "履歴", "検索"]):
            return "memory_search"
        elif any(keyword in input_lower for keyword in ["システム", "状態", "監視", "cpu", "メモリ"]):
            return "system_monitor"
        
        return None
    
    async def process_with_tools(self, user_input: str, context: str = "") -> Dict[str, Any]:
        """ツールを使用した処理"""
        result = {
            "user_input": user_input,
            "selected_tool": None,
            "tool_result": None,
            "direct_response": None,
            "processing_time": 0
        }
        
        start_time = datetime.now()
        
        try:
            # 自動ツール選択
            selected_tool = await self.auto_select_tool(user_input)
            result["selected_tool"] = selected_tool
            
            if selected_tool:
                # ツール実行
                if selected_tool == "structured_reasoning":
                    tool_result = await self.execute_tool(selected_tool, question=user_input, context=context)
                elif selected_tool == "code_analysis":
                    # コードが含まれているかチェック
                    if "```" in user_input or "def " in user_input or "class " in user_input:
                        tool_result = await self.execute_tool(selected_tool, code=user_input)
                    else:
                        tool_result = "コードが検出されませんでした。コード解析を行うには、コードブロックを含めてください。"
                elif selected_tool == "task_breakdown":
                    tool_result = await self.execute_tool(selected_tool, task=user_input, constraints=context)
                elif selected_tool == "memory_search":
                    tool_result = await self.execute_tool(selected_tool, query=user_input)
                elif selected_tool == "system_monitor":
                    tool_result = await self.execute_tool(selected_tool)
                else:
                    tool_result = await self.execute_tool(selected_tool)
                
                result["tool_result"] = tool_result
            else:
                # 直接推論
                request = InferenceRequest(
                    prompt=user_input,
                    system_message="あなたは親切で知識豊富なAIアシスタントです。",
                    context=[context] if context else None
                )
                
                response = await self.ollama_client.generate(request)
                result["direct_response"] = response.content
            
        except Exception as e:
            result["error"] = str(e)
            self.logger.log_alert(
                alert_type="tool_processing_error",
                severity="ERROR",
                message=f"Tool processing failed: {e}"
            )
        
        result["processing_time"] = (datetime.now() - start_time).total_seconds()
        return result


# 便利関数
async def create_tool_manager(ollama_client: OllamaClient) -> ToolManager:
    """ツールマネージャー作成"""
    return ToolManager(ollama_client)


# 使用例
async def main():
    """テスト用メイン関数"""
    from .ollama_client import create_ollama_client
    
    try:
        # クライアントとツールマネージャー作成
        ollama_client = await create_ollama_client()
        tool_manager = ToolManager(ollama_client)
        
        # 利用可能ツール表示
        tools = tool_manager.list_tools()
        print("Available Tools:")
        for tool in tools:
            print(f"  - {tool['name']}: {tool['description']}")
        
        # ツール実行テスト
        test_cases = [
            "Pythonでフィボナッチ数列を計算する関数を分析してください",
            "Webアプリケーションを作成するタスクを分解してください",
            "なぜ機械学習が重要なのか推論してください",
            "現在のシステム状態を確認してください"
        ]
        
        for test_input in test_cases:
            print(f"\n--- Test: {test_input} ---")
            result = await tool_manager.process_with_tools(test_input)
            
            if result["selected_tool"]:
                print(f"Selected Tool: {result['selected_tool']}")
                print(f"Tool Result: {result['tool_result'][:200]}...")
            else:
                print(f"Direct Response: {result['direct_response'][:200]}...")
            
            print(f"Processing Time: {result['processing_time']:.2f}s")
        
        await ollama_client.shutdown()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())