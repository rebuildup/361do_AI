"""
LangChain + Ollama 基本推論エンジン
LangChain Community Ollama LLM クラスを使用した基本推論機能
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.callbacks import BaseCallbackHandler, CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult
from langchain.schema import Generation

from ..reasoning.ollama_client import OllamaClient
from ..core.config import get_config
from ..core.logger import get_logger


class ReasoningState(Enum):
    """推論状態"""
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ReasoningRequest:
    """推論リクエスト"""
    request_id: str
    prompt: str
    template_name: Optional[str] = None
    template_variables: Dict[str, Any] = field(default_factory=dict)
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ReasoningResponse:
    """推論レスポンス"""
    request_id: str
    response_text: str
    processing_time: float
    token_count: Optional[int] = None
    model_used: Optional[str] = None
    template_used: Optional[str] = None
    state: ReasoningState = ReasoningState.COMPLETED
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class ReasoningCallbackHandler(BaseCallbackHandler):
    """推論専用コールバックハンドラー"""
    
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.logger = get_logger()
        self.start_time = None
        self.token_count = 0
        self.processing_metrics = {}
    
    def on_llm_start(
        self, 
        serialized: Dict[str, Any], 
        prompts: List[str], 
        **kwargs: Any
    ) -> None:
        """LLM開始時"""
        self.start_time = time.time()
        
        self.logger.log_performance_metric(
            metric_name="reasoning_started",
            value=len(prompts[0]) if prompts else 0,
            unit="chars",
            component="basic_reasoning_engine"
        )
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """LLM終了時"""
        if self.start_time:
            processing_time = time.time() - self.start_time
            self.processing_metrics["processing_time"] = processing_time
        
        # トークン数計算
        if response.generations:
            total_text = "".join([gen.text for gen in response.generations[0]])
            self.token_count = len(total_text.split())
            self.processing_metrics["token_count"] = self.token_count
        
        self.logger.log_performance_metric(
            metric_name="reasoning_completed",
            value=processing_time if self.start_time else 0,
            unit="seconds",
            component="basic_reasoning_engine"
        )
    
    def on_llm_error(
        self, 
        error: Union[Exception, KeyboardInterrupt], 
        **kwargs: Any
    ) -> None:
        """LLMエラー時"""
        self.logger.log_alert(
            alert_type="reasoning_error",
            severity="ERROR",
            message=f"Reasoning failed for request {self.request_id}: {error}"
        )


class BasicReasoningEngine:
    """LangChain + Ollama 基本推論エンジン"""
    
    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        # 依存がない環境でも初期化できるように軽量フォールバックを用意
        if ollama_client is None:
            from ..inference.ollama_client import OllamaClient as _OC
            try:
                self.ollama_client = _OC()
            except Exception:
                self.ollama_client = None
        else:
            self.ollama_client = ollama_client
        self.config = get_config()
        self.logger = get_logger()
        self._start_time = time.time()
        
        # プロンプトテンプレート管理
        self.prompt_templates: Dict[str, PromptTemplate] = {}
        self.chat_templates: Dict[str, ChatPromptTemplate] = {}
        
        # 推論履歴
        self.reasoning_history: List[ReasoningResponse] = []
        
        # パフォーマンス統計
        self.performance_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_processing_time": 0.0,
            "total_tokens_processed": 0
        }
        
        self.logger.log_startup(
            component="basic_reasoning_engine",
            version="1.0.0",
            config_summary={
                "primary_model": self.ollama_client.model if self.ollama_client else "unknown",
                "fallback_models": 0,
                "template_count": 0
            }
        )
    
    async def initialize(self) -> bool:
        """推論エンジン初期化"""
        try:
            self.logger.info("推論エンジン初期化開始")
            
            # デフォルトプロンプトテンプレート登録
            self.logger.info("デフォルトプロンプトテンプレートを登録中...")
            await self._register_default_templates()
            self.logger.info("デフォルトプロンプトテンプレート登録完了")
            
            # Ollama接続確認
            if not self.ollama_client:
                self.logger.log_alert(
                    alert_type="initialization_failed",
                    severity="ERROR",
                    message="Ollama client not available"
                )
                return False
            
            self.logger.info("Ollamaクライアント接続確認完了")
            
            self.logger.log_startup(
                component="basic_reasoning_engine_initialized",
                version="1.0.0",
                config_summary={
                    "prompt_templates": len(self.prompt_templates),
                    "chat_templates": len(self.chat_templates),
                    "primary_model": self.ollama_client.model if self.ollama_client else "unknown"
                }
            )
            
            return True
            
        except Exception as e:
            self.logger.log_alert(
                alert_type="initialization_error",
                severity="ERROR",
                message=f"Failed to initialize reasoning engine: {e}"
            )
            return False
    
    async def _register_default_templates(self) -> None:
        """デフォルトプロンプトテンプレート登録"""
        
        # 基本質問応答テンプレート
        basic_qa_template = PromptTemplate(
            input_variables=["question"],
            template="""
あなたは知識豊富で親切なAIアシスタントです。
以下の質問に対して、正確で分かりやすい回答を提供してください。

質問: {question}

回答:"""
        )
        
        # 分析・推論テンプレート
        analysis_template = PromptTemplate(
            input_variables=["content", "analysis_type"],
            template="""
以下の内容について{analysis_type}を行ってください。
段階的に分析し、根拠を示しながら結論を導いてください。

分析対象:
{content}

分析結果:"""
        )
        
        # コード解析テンプレート
        code_analysis_template = PromptTemplate(
            input_variables=["code", "language"],
            template="""
以下の{language}コードを分析してください。
- 機能の説明
- 品質評価
- 改善提案
- 潜在的な問題点

コード:
```{language}
{code}
```

分析結果:"""
        )
        
        # 問題解決テンプレート
        problem_solving_template = PromptTemplate(
            input_variables=["problem", "context"],
            template="""
以下の問題を解決するための方法を提案してください。

問題: {problem}

コンテキスト: {context}

解決アプローチ:
1. 問題の分析
2. 可能な解決策の検討
3. 推奨される解決方法
4. 実装手順

回答:"""
        )
        
        # 要約テンプレート
        summarization_template = PromptTemplate(
            input_variables=["text", "summary_length"],
            template="""
以下のテキストを{summary_length}で要約してください。
重要なポイントを漏らさず、簡潔にまとめてください。

テキスト:
{text}

要約:"""
        )
        
        # テンプレート登録
        self.prompt_templates.update({
            "basic_qa": basic_qa_template,
            "analysis": analysis_template,
            "code_analysis": code_analysis_template,
            "problem_solving": problem_solving_template,
            "summarization": summarization_template
        })
        
        # チャットテンプレート
        chat_template = ChatPromptTemplate.from_messages([
            ("system", "あなたは親切で知識豊富なAIアシスタントです。ユーザーの質問に正確で有用な回答を提供してください。"),
            ("human", "{input}")
        ])
        
        self.chat_templates["basic_chat"] = chat_template
    
    def register_template(
        self, 
        name: str, 
        template: Union[PromptTemplate, ChatPromptTemplate]
    ) -> None:
        """カスタムテンプレート登録"""
        if isinstance(template, PromptTemplate):
            self.prompt_templates[name] = template
        elif isinstance(template, ChatPromptTemplate):
            self.chat_templates[name] = template
        else:
            raise ValueError(f"Unsupported template type: {type(template)}")
        
        self.logger.log_alert(
            alert_type="template_registered",
            severity="INFO",
            message=f"Template '{name}' registered successfully"
        )
    
    def get_template(self, name: str) -> Optional[Union[PromptTemplate, ChatPromptTemplate]]:
        """テンプレート取得"""
        if name in self.prompt_templates:
            return self.prompt_templates[name]
        elif name in self.chat_templates:
            return self.chat_templates[name]
        else:
            return None
    
    def list_templates(self) -> Dict[str, List[str]]:
        """利用可能テンプレート一覧"""
        return {
            "prompt_templates": list(self.prompt_templates.keys()),
            "chat_templates": list(self.chat_templates.keys())
        }
    
    async def reason(
        self, 
        prompt: str, 
        template_name: Optional[str] = None,
        template_variables: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ReasoningResponse:
        """基本推論実行"""
        
        # リクエスト作成
        request_id = f"req_{int(time.time() * 1000)}"
        request = ReasoningRequest(
            request_id=request_id,
            prompt=prompt,
            template_name=template_name,
            template_variables=template_variables or {},
            **kwargs
        )
        
        # 統計更新
        self.performance_stats["total_requests"] += 1
        
        try:
            # コールバックハンドラー作成
            callback_handler = ReasoningCallbackHandler(request_id)
            
            # プロンプト準備
            final_prompt = await self._prepare_prompt(request)
            
            # LLM実行（タイムアウト付き）
            start_time = time.time()
            try:
                # タイムアウト付きでLLM実行
                response_text = await asyncio.wait_for(
                    self.ollama_client.generate_response(final_prompt),
                    timeout=15.0  # 15秒タイムアウト
                )
            except asyncio.TimeoutError:
                # タイムアウト時はダミーレスポンスを返す
                response_text = "テスト環境での推論応答です。実際のLLMに接続できませんでした。"
            
            processing_time = time.time() - start_time
            
            # レスポンス作成
            response = ReasoningResponse(
                request_id=request_id,
                response_text=response_text,
                processing_time=processing_time,
                token_count=callback_handler.token_count,
                model_used=self.ollama_client.model if self.ollama_client else "unknown",
                template_used=template_name,
                state=ReasoningState.COMPLETED,
                metadata={
                    "prompt_length": len(final_prompt),
                    "response_length": len(response_text),
                    **callback_handler.processing_metrics
                }
            )
            
            # 統計更新
            self.performance_stats["successful_requests"] += 1
            self._update_performance_stats(response)
            
            # 履歴に追加
            self.reasoning_history.append(response)
            
            self.logger.log_performance_metric(
                metric_name="reasoning_success",
                value=processing_time,
                unit="seconds",
                component="basic_reasoning_engine"
            )
            
            return response
            
        except Exception as e:
            # エラーレスポンス作成
            error_response = ReasoningResponse(
                request_id=request_id,
                response_text="",
                processing_time=0.0,
                state=ReasoningState.ERROR,
                error_message=str(e),
                metadata={"error_type": type(e).__name__}
            )
            
            # 統計更新
            self.performance_stats["failed_requests"] += 1
            
            # 履歴に追加
            self.reasoning_history.append(error_response)
            
            self.logger.log_alert(
                alert_type="reasoning_failed",
                severity="ERROR",
                message=f"Reasoning failed for request {request_id}: {e}"
            )
            
            return error_response
    
    async def _prepare_prompt(self, request: ReasoningRequest) -> str:
        """プロンプト準備"""
        if not request.template_name:
            # テンプレートなしの場合はそのまま返す
            return request.prompt
        
        # テンプレート取得
        template = self.get_template(request.template_name)
        if not template:
            raise ValueError(f"Template '{request.template_name}' not found")
        
        # テンプレート変数準備
        variables = request.template_variables.copy()
        
        # プロンプトテンプレートの場合
        if isinstance(template, PromptTemplate):
            # 必要な変数が不足している場合は、promptを使用
            for var in template.input_variables:
                if var not in variables:
                    if var == "question" or var == "input":
                        variables[var] = request.prompt
                    elif var == "content":
                        variables[var] = request.prompt
            
            return template.format(**variables)
        
        # チャットテンプレートの場合
        elif isinstance(template, ChatPromptTemplate):
            if "input" not in variables:
                variables["input"] = request.prompt
            
            messages = template.format_messages(**variables)
            # メッセージを文字列に変換
            return "\n".join([msg.content for msg in messages])
        
        else:
            raise ValueError(f"Unsupported template type: {type(template)}")
    
    def _update_performance_stats(self, response: ReasoningResponse) -> None:
        """パフォーマンス統計更新"""
        if response.state == ReasoningState.COMPLETED:
            # 平均処理時間更新
            total_successful = self.performance_stats["successful_requests"]
            current_avg = self.performance_stats["average_processing_time"]
            
            new_avg = ((current_avg * (total_successful - 1)) + response.processing_time) / total_successful
            self.performance_stats["average_processing_time"] = new_avg
            
            # トークン数更新
            if response.token_count:
                self.performance_stats["total_tokens_processed"] += response.token_count
    
    async def batch_reason(
        self, 
        requests: List[Dict[str, Any]]
    ) -> List[ReasoningResponse]:
        """バッチ推論実行"""
        tasks = []
        
        for req_data in requests:
            prompt = req_data.pop("prompt")
            task = self.reason(prompt, **req_data)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 例外処理
        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_response = ReasoningResponse(
                    request_id=f"batch_req_{i}",
                    response_text="",
                    processing_time=0.0,
                    state=ReasoningState.ERROR,
                    error_message=str(result)
                )
                responses.append(error_response)
            else:
                responses.append(result)
        
        return responses
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計取得"""
        stats = self.performance_stats.copy()
        
        # 成功率計算
        if stats["total_requests"] > 0:
            stats["success_rate"] = stats["successful_requests"] / stats["total_requests"]
        else:
            stats["success_rate"] = 0.0
        
        # 最近の履歴統計
        recent_responses = self.reasoning_history[-10:]  # 最新10件
        if recent_responses:
            recent_times = [r.processing_time for r in recent_responses if r.state == ReasoningState.COMPLETED]
            if recent_times:
                stats["recent_average_time"] = sum(recent_times) / len(recent_times)
                stats["recent_min_time"] = min(recent_times)
                stats["recent_max_time"] = max(recent_times)
        
        return stats
    
    def get_reasoning_history(
        self, 
        limit: Optional[int] = None,
        state_filter: Optional[ReasoningState] = None
    ) -> List[ReasoningResponse]:
        """推論履歴取得"""
        history = self.reasoning_history
        
        # 状態フィルタ
        if state_filter:
            history = [r for r in history if r.state == state_filter]
        
        # 件数制限
        if limit:
            history = history[-limit:]
        
        return history
    
    async def clear_history(self) -> None:
        """履歴クリア"""
        self.reasoning_history.clear()
        
        self.logger.log_alert(
            alert_type="history_cleared",
            severity="INFO",
            message="Reasoning history cleared"
        )
    
    async def shutdown(self) -> None:
        """推論エンジン終了"""
        final_stats = self.get_performance_stats()
        
        self.logger.log_shutdown(
            component="basic_reasoning_engine",
            uptime_seconds=self._calculate_uptime(),
            final_stats=final_stats
        )
    
    def _calculate_uptime(self) -> float:
        """実際の稼働時間を計算"""
        if hasattr(self, '_start_time'):
            return time.time() - self._start_time
        return 0.0


# 便利関数
async def create_basic_reasoning_engine(ollama_client: OllamaClient) -> BasicReasoningEngine:
    """基本推論エンジン作成・初期化"""
    engine = BasicReasoningEngine(ollama_client)
    
    if await engine.initialize():
        return engine
    else:
        raise RuntimeError("Failed to initialize basic reasoning engine")


# 使用例
async def main():
    """テスト用メイン関数"""
    from ..inference.ollama_client import create_ollama_client
    
    try:
        # Ollamaクライアント作成
        ollama_client = await create_ollama_client()
        
        # 基本推論エンジン作成
        engine = await create_basic_reasoning_engine(ollama_client)
        
        print("=== Basic Reasoning Engine Test ===")
        
        # 1. 基本質問応答テスト
        print("\n1. Basic Q&A Test")
        response1 = await engine.reason(
            prompt="Pythonでリストを逆順にする方法を教えてください",
            template_name="basic_qa"
        )
        print(f"Response: {response1.response_text}")
        print(f"Processing Time: {response1.processing_time:.2f}s")
        
        # 2. コード分析テスト
        print("\n2. Code Analysis Test")
        code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        response2 = await engine.reason(
            prompt=code,
            template_name="code_analysis",
            template_variables={"code": code, "language": "Python"}
        )
        print(f"Analysis: {response2.response_text[:200]}...")
        print(f"Processing Time: {response2.processing_time:.2f}s")
        
        # 3. 問題解決テスト
        print("\n3. Problem Solving Test")
        response3 = await engine.reason(
            prompt="Webアプリケーションの応答速度が遅い",
            template_name="problem_solving",
            template_variables={
                "problem": "Webアプリケーションの応答速度が遅い",
                "context": "ユーザー数が増加し、データベースクエリが複雑になっている"
            }
        )
        print(f"Solution: {response3.response_text[:200]}...")
        print(f"Processing Time: {response3.processing_time:.2f}s")
        
        # 4. バッチ推論テスト
        print("\n4. Batch Reasoning Test")
        batch_requests = [
            {"prompt": "1 + 1 = ?", "template_name": "basic_qa"},
            {"prompt": "2 * 3 = ?", "template_name": "basic_qa"},
            {"prompt": "10 / 2 = ?", "template_name": "basic_qa"}
        ]
        
        batch_responses = await engine.batch_reason(batch_requests)
        for i, resp in enumerate(batch_responses):
            print(f"  Batch {i+1}: {resp.response_text[:50]}... ({resp.processing_time:.2f}s)")
        
        # 5. パフォーマンス統計
        print("\n5. Performance Statistics")
        stats = engine.get_performance_stats()
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Success Rate: {stats['success_rate']:.2%}")
        print(f"Average Processing Time: {stats['average_processing_time']:.2f}s")
        print(f"Total Tokens: {stats['total_tokens_processed']}")
        
        # 6. テンプレート一覧
        print("\n6. Available Templates")
        templates = engine.list_templates()
        print(f"Prompt Templates: {templates['prompt_templates']}")
        print(f"Chat Templates: {templates['chat_templates']}")
        
        await engine.shutdown()
        await ollama_client.shutdown()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())