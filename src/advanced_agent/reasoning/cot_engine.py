"""
Chain-of-Thought 推論エンジン

LangChain ReAct Agent による段階的思考プロセスを統合し、
推論ステップの可視化と構造化出力を実装
"""

import asyncio
import time
import json
import re
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.react.base import DocstoreExplorer
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool, BaseTool
from langchain_core.callbacks import BaseCallbackHandler
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import AgentAction, AgentFinish

from .basic_engine import BasicReasoningEngine, ReasoningResponse, ReasoningState
from ..inference.ollama_client import OllamaClient
from ..core.config import get_config
from ..core.logger import get_logger


class CoTStep(Enum):
    """Chain-of-Thought ステップタイプ"""
    OBSERVATION = "observation"
    THOUGHT = "thought"
    ACTION = "action"
    REFLECTION = "reflection"
    CONCLUSION = "conclusion"


@dataclass
class ReasoningStep:
    """推論ステップ"""
    step_number: int
    step_type: CoTStep
    content: str
    confidence: float = 0.0
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CoTResponse(ReasoningResponse):
    """Chain-of-Thought 推論レスポンス"""
    reasoning_steps: List[ReasoningStep] = field(default_factory=list)
    final_confidence: float = 0.0
    step_count: int = 0
    total_thinking_time: float = 0.0
    quality_score: float = 0.0


class CoTCallbackHandler(BaseCallbackHandler):
    """Chain-of-Thought 専用コールバックハンドラー"""
    
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.logger = get_logger()
        self.reasoning_steps: List[ReasoningStep] = []
        self.current_step = 0
        self.step_start_time = None
        self.total_start_time = time.time()
    
    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        """エージェントアクション時"""
        self.current_step += 1
        
        if self.step_start_time:
            step_time = time.time() - self.step_start_time
        else:
            step_time = 0.0
        
        # Thought ステップを記録
        thought_step = ReasoningStep(
            step_number=self.current_step,
            step_type=CoTStep.THOUGHT,
            content=action.log,
            processing_time=step_time,
            metadata={
                "tool": action.tool,
                "tool_input": str(action.tool_input)
            }
        )
        self.reasoning_steps.append(thought_step)
        
        self.step_start_time = time.time()
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """エージェント終了時"""
        if self.step_start_time:
            step_time = time.time() - self.step_start_time
        else:
            step_time = 0.0
        
        # 最終結論ステップを記録
        conclusion_step = ReasoningStep(
            step_number=self.current_step + 1,
            step_type=CoTStep.CONCLUSION,
            content=finish.return_values.get("output", ""),
            processing_time=step_time,
            metadata=finish.return_values
        )
        self.reasoning_steps.append(conclusion_step)
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        """ツール開始時"""
        action_step = ReasoningStep(
            step_number=self.current_step,
            step_type=CoTStep.ACTION,
            content=f"実行中: {serialized.get('name', 'unknown')}({input_str})",
            metadata={"tool_name": serialized.get('name'), "input": input_str}
        )
        self.reasoning_steps.append(action_step)
    
    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """ツール終了時"""
        if self.reasoning_steps and self.reasoning_steps[-1].step_type == CoTStep.ACTION:
            # 観察ステップを追加
            observation_step = ReasoningStep(
                step_number=self.current_step,
                step_type=CoTStep.OBSERVATION,
                content=f"結果: {output}",
                metadata={"tool_output": output}
            )
            self.reasoning_steps.append(observation_step)


class ChainOfThoughtEngine:
    """Chain-of-Thought 推論エンジン"""
    
    def __init__(self, basic_engine: BasicReasoningEngine):
        self.basic_engine = basic_engine
        self.ollama_client = basic_engine.ollama_client
        self.config = get_config()
        self.logger = get_logger()
        
        # ReAct Agent 設定
        self.agent_executor: Optional[AgentExecutor] = None
        self.tools: List[BaseTool] = []
        self.memory = ConversationBufferWindowMemory(
            k=10,  # 直近10回の会話を保持
            return_messages=True,
            memory_key="chat_history"
        )
        
        # CoT プロンプトテンプレート
        self.cot_prompt_template = None
        
        # 品質評価設定
        self.quality_thresholds = {
            "min_steps": 3,
            "min_confidence": 0.6,
            "max_processing_time": 30.0
        }
        
        self.logger.log_startup(
            component="cot_engine",
            version="1.0.0",
            config_summary={
                "quality_thresholds": self.quality_thresholds
            }
        )
    
    async def initialize(self) -> bool:
        """Chain-of-Thought エンジン初期化"""
        try:
            # ツール初期化
            await self._initialize_tools()
            
            # ReAct プロンプト作成
            self._create_react_prompt()
            
            # エージェント作成
            await self._create_agent()
            
            self.logger.log_startup(
                component="cot_engine_initialized",
                version="1.0.0",
                config_summary={
                    "tools_count": len(self.tools),
                    "memory_enabled": True,
                    "agent_ready": self.agent_executor is not None
                }
            )
            
            return True
            
        except Exception as e:
            self.logger.log_alert(
                alert_type="cot_initialization_error",
                severity="ERROR",
                message=f"Failed to initialize CoT engine: {e}"
            )
            return False
    
    async def _initialize_tools(self) -> None:
        """推論ツール初期化"""
        
        # 1. 計算ツール
        def calculate(expression: str) -> str:
            """数式計算ツール"""
            try:
                # 安全な計算のため、evalの代わりに制限された計算を実行
                import ast
                import operator
                
                # サポートする演算子
                ops = {
                    ast.Add: operator.add,
                    ast.Sub: operator.sub,
                    ast.Mult: operator.mul,
                    ast.Div: operator.truediv,
                    ast.Pow: operator.pow,
                    ast.USub: operator.neg,
                }
                
                def eval_expr(node):
                    if isinstance(node, ast.Num):
                        return node.n
                    elif isinstance(node, ast.BinOp):
                        return ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
                    elif isinstance(node, ast.UnaryOp):
                        return ops[type(node.op)](eval_expr(node.operand))
                    else:
                        raise TypeError(node)
                
                result = eval_expr(ast.parse(expression, mode='eval').body)
                return f"計算結果: {result}"
                
            except Exception as e:
                return f"計算エラー: {str(e)}"
        
        calc_tool = Tool(
            name="calculator",
            description="数式を計算します。例: 2+3*4, 10/2, 2**3",
            func=calculate
        )
        
        # 2. 分析ツール
        def analyze_text(text: str) -> str:
            """テキスト分析ツール"""
            try:
                word_count = len(text.split())
                char_count = len(text)
                sentence_count = len([s for s in text.split('.') if s.strip()])
                
                # 簡単な感情分析
                positive_words = ['良い', '素晴らしい', '優秀', '成功', '効果的', 'good', 'great', 'excellent']
                negative_words = ['悪い', '問題', '失敗', '困難', '危険', 'bad', 'problem', 'fail']
                
                positive_count = sum(1 for word in positive_words if word in text.lower())
                negative_count = sum(1 for word in negative_words if word in text.lower())
                
                sentiment = "中性"
                if positive_count > negative_count:
                    sentiment = "ポジティブ"
                elif negative_count > positive_count:
                    sentiment = "ネガティブ"
                
                return f"""テキスト分析結果:
- 文字数: {char_count}
- 単語数: {word_count}
- 文数: {sentence_count}
- 感情: {sentiment}
- ポジティブ語: {positive_count}
- ネガティブ語: {negative_count}"""
                
            except Exception as e:
                return f"分析エラー: {str(e)}"
        
        analysis_tool = Tool(
            name="text_analyzer",
            description="テキストを分析して統計情報と感情を返します",
            func=analyze_text
        )
        
        # 3. 知識検索ツール（簡易版）
        def search_knowledge(query: str) -> str:
            """知識検索ツール"""
            # 簡易的な知識ベース
            knowledge_base = {
                "python": "Pythonは高水準プログラミング言語で、読みやすく書きやすい構文が特徴です。",
                "ai": "人工知能（AI）は、人間の知能を模倣するコンピューターシステムです。",
                "machine learning": "機械学習は、データからパターンを学習してタスクを実行するAIの手法です。",
                "deep learning": "深層学習は、多層ニューラルネットワークを使用する機械学習の手法です。",
                "langchain": "LangChainは、大規模言語モデルを使用したアプリケーション開発のためのフレームワークです。"
            }
            
            query_lower = query.lower()
            for key, value in knowledge_base.items():
                if key in query_lower:
                    return f"知識: {value}"
            
            return f"'{query}'に関する情報が見つかりませんでした。"
        
        knowledge_tool = Tool(
            name="knowledge_search",
            description="知識ベースから情報を検索します",
            func=search_knowledge
        )
        
        # 4. 推論検証ツール
        def verify_reasoning(reasoning: str) -> str:
            """推論検証ツール"""
            try:
                # 推論の品質をチェック
                issues = []
                
                if len(reasoning) < 50:
                    issues.append("推論が短すぎます")
                
                if not any(word in reasoning for word in ['なぜなら', 'したがって', 'そのため', 'because', 'therefore']):
                    issues.append("論理的接続詞が不足しています")
                
                if reasoning.count('。') < 2:
                    issues.append("推論ステップが不足している可能性があります")
                
                if issues:
                    return f"推論の改善点: {', '.join(issues)}"
                else:
                    return "推論は適切に構成されています"
                    
            except Exception as e:
                return f"検証エラー: {str(e)}"
        
        verify_tool = Tool(
            name="reasoning_verifier",
            description="推論の品質と論理性を検証します",
            func=verify_reasoning
        )
        
        # ツールリストに追加
        self.tools = [calc_tool, analysis_tool, knowledge_tool, verify_tool]
    
    def _create_react_prompt(self) -> None:
        """ReAct プロンプトテンプレート作成"""
        
        self.cot_prompt_template = PromptTemplate(
            input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
            template="""あなたは段階的に考える優秀なAIアシスタントです。
問題を解決するために、以下のツールを使用して段階的に推論してください。

利用可能なツール:
{tools}

ツール名: {tool_names}

推論形式:
Question: 解決すべき問題
Thought: 何を考え、どのような行動を取るべきか
Action: 実行するアクション
Action Input: アクションへの入力
Observation: アクションの結果
... (必要に応じてThought/Action/Action Input/Observationを繰り返し)
Thought: 最終的な答えがわかりました
Final Answer: 最終回答

重要な指示:
1. 各ステップで明確に思考過程を示してください
2. 複雑な問題は小さな部分に分解してください
3. 各ステップの結果を次のステップに活用してください
4. 最終回答では、推論過程を要約してください

Question: {input}
{agent_scratchpad}"""
        )
    
    async def _create_agent(self) -> None:
        """ReAct エージェント作成"""
        try:
            # LLM取得
            llm = self.ollama_client.primary_llm
            if not llm:
                raise RuntimeError("Primary LLM not available")
            
            # ReAct エージェント作成
            agent = create_react_agent(
                llm=llm,
                tools=self.tools,
                prompt=self.cot_prompt_template
            )
            
            # エージェント実行器作成
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                memory=self.memory,
                verbose=True,
                max_iterations=10,
                max_execution_time=30,
                handle_parsing_errors=True,
                return_intermediate_steps=True
            )
            
        except Exception as e:
            self.logger.log_alert(
                alert_type="agent_creation_error",
                severity="ERROR",
                message=f"Failed to create ReAct agent: {e}"
            )
            raise
    
    async def reason_with_cot(
        self,
        prompt: str,
        max_steps: int = 10,
        confidence_threshold: float = 0.7,
        **kwargs
    ) -> CoTResponse:
        """Chain-of-Thought 推論実行"""
        
        request_id = f"cot_req_{int(time.time() * 1000)}"
        start_time = time.time()
        
        try:
            if not self.agent_executor:
                raise RuntimeError("Agent executor not initialized")
            
            # コールバックハンドラー作成
            callback_handler = CoTCallbackHandler(request_id)
            
            # エージェント実行
            result = await asyncio.to_thread(
                self.agent_executor.invoke,
                {"input": prompt},
                config={"callbacks": [callback_handler]}
            )
            
            processing_time = time.time() - start_time
            
            # 推論ステップ処理
            reasoning_steps = callback_handler.reasoning_steps
            
            # 信頼度計算
            final_confidence = self._calculate_confidence(reasoning_steps, result)
            
            # 品質スコア計算
            quality_score = self._calculate_quality_score(reasoning_steps, processing_time)
            
            # CoT レスポンス作成
            cot_response = CoTResponse(
                request_id=request_id,
                response_text=result.get("output", ""),
                processing_time=processing_time,
                reasoning_steps=reasoning_steps,
                final_confidence=final_confidence,
                step_count=len(reasoning_steps),
                total_thinking_time=sum(step.processing_time for step in reasoning_steps),
                quality_score=quality_score,
                model_used=self.ollama_client.primary_model,
                state=ReasoningState.COMPLETED,
                metadata={
                    "intermediate_steps": result.get("intermediate_steps", []),
                    "max_steps_reached": len(reasoning_steps) >= max_steps,
                    "confidence_threshold_met": final_confidence >= confidence_threshold
                }
            )
            
            self.logger.log_performance_metric(
                metric_name="cot_reasoning_success",
                value=processing_time,
                unit="seconds",
                component="cot_engine"
            )
            
            return cot_response
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            error_response = CoTResponse(
                request_id=request_id,
                response_text="",
                processing_time=processing_time,
                state=ReasoningState.ERROR,
                error_message=str(e),
                metadata={"error_type": type(e).__name__}
            )
            
            self.logger.log_alert(
                alert_type="cot_reasoning_failed",
                severity="ERROR",
                message=f"CoT reasoning failed for request {request_id}: {e}"
            )
            
            return error_response
    
    def _calculate_confidence(self, steps: List[ReasoningStep], result: Dict[str, Any]) -> float:
        """信頼度計算"""
        try:
            confidence_factors = []
            
            # ステップ数による信頼度
            step_confidence = min(len(steps) / 5.0, 1.0)  # 5ステップで最大
            confidence_factors.append(step_confidence)
            
            # 各ステップの処理時間による信頼度（適度な時間をかけているか）
            if steps:
                avg_step_time = sum(step.processing_time for step in steps) / len(steps)
                time_confidence = min(avg_step_time / 2.0, 1.0)  # 2秒で最大
                confidence_factors.append(time_confidence)
            
            # 結論ステップの存在
            has_conclusion = any(step.step_type == CoTStep.CONCLUSION for step in steps)
            conclusion_confidence = 1.0 if has_conclusion else 0.5
            confidence_factors.append(conclusion_confidence)
            
            # 平均信頼度
            return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.0
            
        except Exception:
            return 0.5  # デフォルト値
    
    def _calculate_quality_score(self, steps: List[ReasoningStep], processing_time: float) -> float:
        """品質スコア計算"""
        try:
            quality_factors = []
            
            # ステップ多様性
            step_types = set(step.step_type for step in steps)
            diversity_score = len(step_types) / len(CoTStep)
            quality_factors.append(diversity_score)
            
            # 処理時間効率
            time_efficiency = 1.0 - min(processing_time / self.quality_thresholds["max_processing_time"], 1.0)
            quality_factors.append(time_efficiency)
            
            # ステップ数適正性
            step_count = len(steps)
            if step_count >= self.quality_thresholds["min_steps"]:
                step_score = min(step_count / 8.0, 1.0)  # 8ステップで最大
            else:
                step_score = step_count / self.quality_thresholds["min_steps"]
            quality_factors.append(step_score)
            
            return sum(quality_factors) / len(quality_factors) if quality_factors else 0.0
            
        except Exception:
            return 0.5  # デフォルト値
    
    def extract_reasoning_steps_from_text(self, text: str) -> List[ReasoningStep]:
        """テキストから推論ステップを抽出"""
        steps = []
        step_number = 1
        
        # パターンマッチングで推論ステップを抽出
        patterns = {
            CoTStep.THOUGHT: r"Thought:\s*(.+?)(?=Action:|Observation:|Final Answer:|$)",
            CoTStep.ACTION: r"Action:\s*(.+?)(?=Action Input:|Thought:|Observation:|$)",
            CoTStep.OBSERVATION: r"Observation:\s*(.+?)(?=Thought:|Action:|Final Answer:|$)",
            CoTStep.CONCLUSION: r"Final Answer:\s*(.+?)(?=$)"
        }
        
        for step_type, pattern in patterns.items():
            matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                content = match.group(1).strip()
                if content:
                    step = ReasoningStep(
                        step_number=step_number,
                        step_type=step_type,
                        content=content,
                        confidence=0.8  # デフォルト信頼度
                    )
                    steps.append(step)
                    step_number += 1
        
        return steps
    
    async def analyze_reasoning_quality(self, response: CoTResponse) -> Dict[str, Any]:
        """推論品質分析"""
        analysis = {
            "overall_quality": response.quality_score,
            "confidence_level": response.final_confidence,
            "step_analysis": {},
            "recommendations": []
        }
        
        # ステップ分析
        step_types_count = {}
        for step in response.reasoning_steps:
            step_type = step.step_type.value
            step_types_count[step_type] = step_types_count.get(step_type, 0) + 1
        
        analysis["step_analysis"] = {
            "total_steps": response.step_count,
            "step_types": step_types_count,
            "average_step_time": response.total_thinking_time / response.step_count if response.step_count > 0 else 0,
            "processing_efficiency": response.processing_time / response.step_count if response.step_count > 0 else 0
        }
        
        # 推奨事項
        if response.step_count < self.quality_thresholds["min_steps"]:
            analysis["recommendations"].append("より詳細な推論ステップが必要です")
        
        if response.final_confidence < self.quality_thresholds["min_confidence"]:
            analysis["recommendations"].append("推論の信頼度を向上させる必要があります")
        
        if response.processing_time > self.quality_thresholds["max_processing_time"]:
            analysis["recommendations"].append("処理時間の最適化が必要です")
        
        if CoTStep.CONCLUSION.value not in step_types_count:
            analysis["recommendations"].append("明確な結論ステップが不足しています")
        
        return analysis
    
    async def shutdown(self) -> None:
        """CoT エンジン終了"""
        self.logger.log_shutdown(
            component="cot_engine",
            uptime_seconds=0,
            final_stats={}
        )


# 便利関数
async def create_cot_engine(basic_engine: BasicReasoningEngine) -> ChainOfThoughtEngine:
    """Chain-of-Thought エンジン作成・初期化"""
    engine = ChainOfThoughtEngine(basic_engine)
    
    if await engine.initialize():
        return engine
    else:
        raise RuntimeError("Failed to initialize Chain-of-Thought engine")


# 使用例
async def main():
    """テスト用メイン関数"""
    from .basic_engine import create_basic_reasoning_engine
    from ..inference.ollama_client import create_ollama_client
    
    try:
        # 基本エンジン作成
        ollama_client = await create_ollama_client()
        basic_engine = await create_basic_reasoning_engine(ollama_client)
        
        # CoT エンジン作成
        cot_engine = await create_cot_engine(basic_engine)
        
        print("=== Chain-of-Thought Reasoning Test ===")
        
        # 1. 数学問題
        print("\n1. Math Problem Test")
        math_response = await cot_engine.reason_with_cot(
            "田中さんは本を12冊持っています。そのうち3分の1を友達に貸しました。その後、新しく5冊買いました。田中さんは今何冊の本を持っていますか？"
        )
        
        print(f"Final Answer: {math_response.response_text}")
        print(f"Steps: {math_response.step_count}")
        print(f"Confidence: {math_response.final_confidence:.2f}")
        print(f"Quality Score: {math_response.quality_score:.2f}")
        
        print("\nReasoning Steps:")
        for step in math_response.reasoning_steps:
            print(f"  {step.step_number}. [{step.step_type.value}] {step.content[:100]}...")
        
        # 2. 論理問題
        print("\n2. Logic Problem Test")
        logic_response = await cot_engine.reason_with_cot(
            "AさんはBさんより背が高く、BさんはCさんより背が高いです。CさんはDさんより背が高いです。この4人の中で最も背が高いのは誰ですか？理由も説明してください。"
        )
        
        print(f"Final Answer: {logic_response.response_text}")
        print(f"Steps: {logic_response.step_count}")
        print(f"Confidence: {logic_response.final_confidence:.2f}")
        
        # 3. 品質分析
        print("\n3. Quality Analysis")
        quality_analysis = await cot_engine.analyze_reasoning_quality(math_response)
        print(f"Overall Quality: {quality_analysis['overall_quality']:.2f}")
        print(f"Step Analysis: {quality_analysis['step_analysis']}")
        print(f"Recommendations: {quality_analysis['recommendations']}")
        
        await cot_engine.shutdown()
        await basic_engine.shutdown()
        await ollama_client.shutdown()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())