"""
LangChain + Ollama 基本推論エンジン
PromptTemplate とコールバック統合
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler, CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser

from ..inference.ollama_client import OllamaClient, InferenceRequest, InferenceResponse
from ..core.config import get_config
from ..core.logger import get_logger


@dataclass
class ReasoningContext:
    """推論コンテキスト"""
    session_id: str
    user_id: Optional[str] = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    system_context: Optional[str] = None
    domain_context: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningRequest:
    """推論リクエスト"""
    prompt: str
    context: ReasoningContext
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    use_memory: bool = True
    reasoning_type: str = "general"  # general, analytical, creative, factual
    output_format: str = "text"  # text, structured, json
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningStep:
    """推論ステップ"""
    step_number: int
    thought: str
    action: Optional[str] = None
    observation: Optional[str] = None
    confidence: float = 1.0
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningResult:
    """推論結果"""
    request_id: str
    final_answer: str
    reasoning_steps: List[ReasoningStep] = field(default_factory=list)
    confidence_score: float = 1.0
    processing_time: float = 0.0
    model_used: str = ""
    tokens_used: Optional[int] = None
    memory_usage_mb: float = 0.0
    context_used: Optional[ReasoningContext] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReasoningCallbackHandler(BaseCallbackHandler):
    """推論専用コールバックハンドラー"""
    
    def __init__(self, request_id: str):
        super().__init__()
        self.request_id = request_id
        self.start_time = None
        self.step_count = 0
        self.logger = get_logger()
        self.reasoning_steps: List[ReasoningStep] = []
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """推論開始時"""
        self.start_time = time.time()
        self.step_count = 0
        
        model_name = serialized.get("name", "unknown")
        prompt_length = len(prompts[0]) if prompts else 0
        
        self.logger.log_inference_start(
            model_name=model_name,
            prompt_length=prompt_length,
            context_length=prompt_length
        )
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """新しいトークン生成時"""
        # ストリーミング時のトークン処理
        pass
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """推論終了時"""
        if self.start_time:
            processing_time = time.time() - self.start_time
            
            # レスポンス長計算
            response_length = 0
            if response.generations:
                for generation in response.generations:
                    for gen in generation:
                        response_length += len(gen.text)
            
            self.logger.log_inference_complete(
                model_name="reasoning_engine",
                response_length=response_length,
                processing_time=processing_time,
                memory_used_mb=0  # TODO: 実際のメモリ使用量
            )
    
    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        """推論エラー時"""
        self.logger.log_inference_error(
            model_name="reasoning_engine",
            error=error,
            fallback_used=False
        )
    
    def add_reasoning_step(self, step: ReasoningStep) -> None:
        """推論ステップ追加"""
        self.reasoning_steps.append(step)
        self.step_count += 1


class BaseReasoningEngine(ABC):
    """基本推論エンジン抽象クラス"""
    
    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client
        self.config = get_config()
        self.logger = get_logger()
    
    @abstractmethod
    async def reason(self, request: ReasoningRequest) -> ReasoningResult:
        """推論実行"""
        pass
    
    @abstractmethod
    def get_prompt_template(self, reasoning_type: str) -> PromptTemplate:
        """プロンプトテンプレート取得"""
        pass


class BasicReasoningEngine(BaseReasoningEngine):
    """基本推論エンジン実装"""
    
    def __init__(self, ollama_client: OllamaClient):
        super().__init__(ollama_client)
        
        # プロンプトテンプレート定義
        self.prompt_templates = {
            "general": PromptTemplate(
                input_variables=["context", "question", "constraints"],
                template="""あなたは知識豊富で論理的思考に優れたAIアシスタントです。

コンテキスト: {context}

制約条件:
{constraints}

質問: {question}

上記の質問に対して、論理的で正確な回答を提供してください。必要に応じて段階的に考えを整理し、根拠を示してください。

回答:"""
            ),
            
            "analytical": PromptTemplate(
                input_variables=["context", "question", "constraints"],
                template="""あなたは分析的思考に特化したAIアシスタントです。

コンテキスト: {context}

制約条件:
{constraints}

分析対象: {question}

以下の手順で分析を行ってください：
1. 問題の構造化と要素分解
2. 各要素の詳細分析
3. 要素間の関係性分析
4. 結論と推奨事項

分析結果:"""
            ),
            
            "creative": PromptTemplate(
                input_variables=["context", "question", "constraints"],
                template="""あなたは創造的思考に優れたAIアシスタントです。

コンテキスト: {context}

制約条件:
{constraints}

創造的課題: {question}

創造的で革新的なアイデアを生成してください。既存の枠にとらわれず、多角的な視点から新しい解決策や提案を考えてください。

創造的回答:"""
            ),
            
            "factual": PromptTemplate(
                input_variables=["context", "question", "constraints"],
                template="""あなたは事実に基づく正確な情報提供に特化したAIアシスタントです。

コンテキスト: {context}

制約条件:
{constraints}

質問: {question}

事実に基づいて正確で客観的な回答を提供してください。不確実な情報については明確に示し、可能な限り信頼できる根拠を示してください。

事実に基づく回答:"""
            )
        }
        
        self.logger.log_startup(
            component="basic_reasoning_engine",
            version="1.0.0",
            config_summary={
                "available_templates": len(self.prompt_templates),
                "template_types": list(self.prompt_templates.keys())
            }
        )
    
    def get_prompt_template(self, reasoning_type: str) -> PromptTemplate:
        """プロンプトテンプレート取得"""
        return self.prompt_templates.get(reasoning_type, self.prompt_templates["general"])
    
    async def reason(self, request: ReasoningRequest) -> ReasoningResult:
        """推論実行"""
        start_time = time.time()
        request_id = f"reasoning_{int(time.time() * 1000)}"
        
        # コールバックハンドラー作成
        callback_handler = ReasoningCallbackHandler(request_id)
        
        try:
            # プロンプトテンプレート取得
            template = self.get_prompt_template(request.reasoning_type)
            
            # コンテキスト構築
            context_str = self._build_context_string(request.context)
            constraints_str = self._build_constraints_string(request.context.constraints)
            
            # プロンプト生成
            prompt = template.format(
                context=context_str,
                question=request.prompt,
                constraints=constraints_str
            )
            
            # LangChain LLM チェーン作成
            llm = await self._get_llm_with_callbacks([callback_handler])
            chain = LLMChain(llm=llm, prompt=template)
            
            # 推論実行
            self.logger.log_performance_metric(
                metric_name="reasoning_request_start",
                value=time.time(),
                unit="timestamp",
                component="basic_reasoning_engine"
            )
            
            response = await chain.arun(
                context=context_str,
                question=request.prompt,
                constraints=constraints_str
            )
            
            processing_time = time.time() - start_time
            
            # 結果構築
            result = ReasoningResult(
                request_id=request_id,
                final_answer=response,
                reasoning_steps=callback_handler.reasoning_steps,
                confidence_score=self._calculate_confidence(response),
                processing_time=processing_time,
                model_used=self.config.models.primary,
                tokens_used=len(response.split()),  # 簡易トークン数
                context_used=request.context,
                metadata={
                    "reasoning_type": request.reasoning_type,
                    "output_format": request.output_format,
                    "template_used": request.reasoning_type
                }
            )
            
            # パフォーマンスログ
            self.logger.log_performance_metric(
                metric_name="reasoning_processing_time",
                value=processing_time,
                unit="seconds",
                component="basic_reasoning_engine"
            )
            
            return result
            
        except Exception as e:
            # エラー処理
            self.logger.log_inference_error(
                model_name="basic_reasoning_engine",
                error=e,
                fallback_used=False
            )
            
            # エラー結果返却
            return ReasoningResult(
                request_id=request_id,
                final_answer=f"推論中にエラーが発生しました: {e}",
                processing_time=time.time() - start_time,
                model_used="error",
                confidence_score=0.0,
                context_used=request.context,
                metadata={"error": str(e)}
            )
    
    async def _get_llm_with_callbacks(self, callbacks: List[BaseCallbackHandler]) -> Ollama:
        """コールバック付きLLM取得"""
        # Ollama LLM インスタンス作成
        llm = Ollama(
            model=self.config.models.primary,
            base_url=self.config.models.ollama_base_url,
            temperature=0.1,
            callbacks=callbacks
        )
        return llm
    
    def _build_context_string(self, context: ReasoningContext) -> str:
        """コンテキスト文字列構築"""
        context_parts = []
        
        # システムコンテキスト
        if context.system_context:
            context_parts.append(f"システム情報: {context.system_context}")
        
        # ドメインコンテキスト
        if context.domain_context:
            context_parts.append(f"ドメイン情報: {context.domain_context}")
        
        # 会話履歴（最新5件）
        if context.conversation_history:
            context_parts.append("最近の会話:")
            for i, conv in enumerate(context.conversation_history[-5:], 1):
                role = conv.get("role", "user")
                content = conv.get("content", "")
                context_parts.append(f"  {i}. {role}: {content}")
        
        # メタデータ
        if context.metadata:
            relevant_metadata = {k: v for k, v in context.metadata.items() 
                               if k in ["topic", "priority", "urgency"]}
            if relevant_metadata:
                context_parts.append(f"追加情報: {relevant_metadata}")
        
        return "\n".join(context_parts) if context_parts else "特別なコンテキストはありません。"
    
    def _build_constraints_string(self, constraints: List[str]) -> str:
        """制約条件文字列構築"""
        if not constraints:
            return "特別な制約はありません。"
        
        constraint_parts = []
        for i, constraint in enumerate(constraints, 1):
            constraint_parts.append(f"{i}. {constraint}")
        
        return "\n".join(constraint_parts)
    
    def _calculate_confidence(self, response: str) -> float:
        """信頼度計算（簡易版）"""
        # 簡易的な信頼度計算
        # 実際の実装では、より高度な手法を使用
        
        confidence = 0.8  # ベース信頼度
        
        # レスポンス長による調整
        if len(response) < 50:
            confidence -= 0.2
        elif len(response) > 500:
            confidence += 0.1
        
        # 不確実性を示すキーワードの検出
        uncertainty_keywords = ["わからない", "不明", "確実ではない", "おそらく", "かもしれない"]
        for keyword in uncertainty_keywords:
            if keyword in response:
                confidence -= 0.1
        
        # 確実性を示すキーワードの検出
        certainty_keywords = ["確実に", "明確に", "間違いなく", "事実として"]
        for keyword in certainty_keywords:
            if keyword in response:
                confidence += 0.1
        
        return max(0.0, min(1.0, confidence))


class MemoryAwareReasoningEngine(BasicReasoningEngine):
    """記憶対応推論エンジン"""
    
    def __init__(self, ollama_client: OllamaClient):
        super().__init__(ollama_client)
        # TODO: 記憶システムとの統合
        self.memory_enabled = False
    
    async def reason(self, request: ReasoningRequest) -> ReasoningResult:
        """記憶を考慮した推論実行"""
        if request.use_memory and self.memory_enabled:
            # TODO: 記憶システムから関連情報を取得
            # relevant_memories = await self.memory_system.search(request.prompt)
            # request.context に記憶情報を追加
            pass
        
        # 基本推論実行
        result = await super().reason(request)
        
        if request.use_memory and self.memory_enabled:
            # TODO: 推論結果を記憶システムに保存
            # await self.memory_system.store(request, result)
            pass
        
        return result


# 便利関数
async def create_reasoning_engine(ollama_client: OllamaClient, 
                                engine_type: str = "basic") -> BaseReasoningEngine:
    """推論エンジン作成"""
    if engine_type == "basic":
        return BasicReasoningEngine(ollama_client)
    elif engine_type == "memory_aware":
        return MemoryAwareReasoningEngine(ollama_client)
    else:
        raise ValueError(f"Unknown engine type: {engine_type}")


async def quick_reasoning(prompt: str, 
                        reasoning_type: str = "general",
                        ollama_client: Optional[OllamaClient] = None) -> str:
    """簡易推論実行"""
    if ollama_client is None:
        from ..inference.ollama_client import create_ollama_client
        ollama_client = await create_ollama_client()
    
    engine = await create_reasoning_engine(ollama_client)
    
    context = ReasoningContext(session_id="quick_session")
    request = ReasoningRequest(
        prompt=prompt,
        context=context,
        reasoning_type=reasoning_type
    )
    
    result = await engine.reason(request)
    return result.final_answer


# 使用例
async def main():
    """テスト用メイン関数"""
    from ..inference.ollama_client import create_ollama_client
    
    try:
        # Ollama クライアント作成
        ollama_client = await create_ollama_client()
        
        # 推論エンジン作成
        engine = await create_reasoning_engine(ollama_client, "basic")
        
        # テストケース
        test_cases = [
            {
                "prompt": "人工知能の未来について教えてください",
                "reasoning_type": "general",
                "context": ReasoningContext(
                    session_id="test_session_1",
                    domain_context="AI技術の発展について議論中"
                )
            },
            {
                "prompt": "Pythonでソートアルゴリズムを実装する最適な方法を分析してください",
                "reasoning_type": "analytical",
                "context": ReasoningContext(
                    session_id="test_session_2",
                    domain_context="プログラミング学習",
                    constraints=["実行効率を重視", "可読性も考慮"]
                )
            },
            {
                "prompt": "環境問題を解決する革新的なアイデアを提案してください",
                "reasoning_type": "creative",
                "context": ReasoningContext(
                    session_id="test_session_3",
                    domain_context="持続可能な社会の実現"
                )
            }
        ]
        
        print("Basic Reasoning Engine Test")
        print("=" * 50)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest Case {i}: {test_case['reasoning_type'].upper()}")
            print(f"Prompt: {test_case['prompt']}")
            
            request = ReasoningRequest(
                prompt=test_case["prompt"],
                context=test_case["context"],
                reasoning_type=test_case["reasoning_type"]
            )
            
            result = await engine.reason(request)
            
            print(f"Processing Time: {result.processing_time:.2f}s")
            print(f"Confidence: {result.confidence_score:.2f}")
            print(f"Model Used: {result.model_used}")
            print(f"Answer: {result.final_answer[:200]}...")
            print("-" * 50)
        
        # 簡易推論テスト
        print("\nQuick Reasoning Test")
        print("=" * 30)
        
        quick_answer = await quick_reasoning(
            "量子コンピューターの基本原理を簡潔に説明してください",
            "factual"
        )
        print(f"Quick Answer: {quick_answer[:200]}...")
        
        await ollama_client.shutdown()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())