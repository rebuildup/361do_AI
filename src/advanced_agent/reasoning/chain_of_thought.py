"""
LangChain ReAct Agent Chain-of-Thought 統合
段階的思考プロセスと中間ステップ構造化
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import re

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish
from langchain.memory import ConversationBufferWindowMemory

from ..inference.ollama_client import OllamaClient
from ..inference.tools import ToolManager
from ..core.config import get_config
from ..core.logger import get_logger


class ThoughtState(Enum):
    """思考状態"""
    INITIAL = "initial"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    ACTING = "acting"
    OBSERVING = "observing"
    REFLECTING = "reflecting"
    CONCLUDING = "concluding"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ThoughtStep:
    """思考ステップ"""
    step_number: int
    state: ThoughtState
    thought_content: str
    reasoning: Optional[str] = None
    action_plan: Optional[str] = None
    action_taken: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    reflection: Optional[str] = None
    confidence_score: float = 0.0
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChainOfThoughtSession:
    """Chain-of-Thought セッション"""
    session_id: str
    initial_query: str
    thought_steps: List[ThoughtStep] = field(default_factory=list)
    final_conclusion: Optional[str] = None
    reasoning_chain: List[str] = field(default_factory=list)
    total_processing_time: float = 0.0
    confidence_score: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChainOfThoughtCallbackHandler(BaseCallbackHandler):
    """Chain-of-Thought 専用コールバックハンドラー"""
    
    def __init__(self, session: ChainOfThoughtSession):
        self.session = session
        self.logger = get_logger()
        self.current_step = 0
        self.step_start_time = None
        self.thought_pattern = re.compile(r"Thought:\s*(.*?)(?=\n(?:Action|Final Answer))", re.DOTALL)
        self.action_pattern = re.compile(r"Action:\s*(.*?)(?=\n)", re.DOTALL)
        self.observation_pattern = re.compile(r"Observation:\s*(.*?)(?=\n(?:Thought|Final Answer))", re.DOTALL)
    
    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """エージェントアクション開始時"""
        self.step_start_time = time.time()
        self.current_step += 1
        
        # 思考内容を解析
        thought_content = self._extract_thought(action.log)
        reasoning = self._extract_reasoning(action.log)
        
        step = ThoughtStep(
            step_number=self.current_step,
            state=ThoughtState.ACTING,
            thought_content=thought_content,
            reasoning=reasoning,
            action_taken=action.tool,
            action_input=action.tool_input if isinstance(action.tool_input, dict) else {"input": action.tool_input}
        )
        
        self.session.thought_steps.append(step)
        self.session.reasoning_chain.append(f"Step {self.current_step}: {thought_content}")
        
        self.logger.log_performance_metric(
            metric_name="cot_step_started",
            value=self.current_step,
            unit="step",
            component="chain_of_thought"
        )
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """エージェント完了時"""
        if self.step_start_time:
            processing_time = time.time() - self.step_start_time
            
            if self.session.thought_steps:
                self.session.thought_steps[-1].processing_time = processing_time
                self.session.thought_steps[-1].state = ThoughtState.COMPLETED
        
        # 最終結論を設定
        self.session.final_conclusion = finish.return_values.get("output", "")
        self.session.end_time = datetime.now()
        self.session.total_processing_time = (self.session.end_time - self.session.start_time).total_seconds()
        
        # 全体の信頼度スコア計算
        self.session.confidence_score = self._calculate_overall_confidence()
        
        # 最終的な推論チェーンに結論を追加
        self.session.reasoning_chain.append(f"Conclusion: {self.session.final_conclusion}")
        
        self.logger.log_performance_metric(
            metric_name="cot_session_completed",
            value=self.session.total_processing_time,
            unit="seconds",
            component="chain_of_thought"
        )
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """ツール終了時"""
        if self.session.thought_steps:
            current_step = self.session.thought_steps[-1]
            current_step.observation = output
            current_step.state = ThoughtState.OBSERVING
            
            # 観察に基づく反省を生成
            current_step.reflection = self._generate_reflection(current_step)
            
            # 信頼度スコア計算
            current_step.confidence_score = self._calculate_step_confidence(current_step)
            
            if self.step_start_time:
                current_step.processing_time = time.time() - self.step_start_time
        
        self.logger.log_performance_metric(
            metric_name="cot_observation_recorded",
            value=len(output),
            unit="chars",
            component="chain_of_thought"
        )
    
    def _extract_thought(self, log_text: str) -> str:
        """思考内容抽出"""
        match = self.thought_pattern.search(log_text)
        if match:
            return match.group(1).strip()
        return log_text.split('\n')[0] if log_text else ""
    
    def _extract_reasoning(self, log_text: str) -> str:
        """推論内容抽出"""
        lines = log_text.split('\n')
        reasoning_lines = []
        
        for line in lines:
            if line.strip() and not line.startswith(('Action:', 'Action Input:', 'Observation:')):
                if not line.startswith('Thought:'):
                    reasoning_lines.append(line.strip())
        
        return ' '.join(reasoning_lines) if reasoning_lines else ""
    
    def _generate_reflection(self, step: ThoughtStep) -> str:
        """反省内容生成"""
        if not step.observation:
            return ""
        
        # 簡単な反省ロジック
        if "error" in step.observation.lower() or "failed" in step.observation.lower():
            return "このアクションは期待した結果を得られませんでした。別のアプローチを検討する必要があります。"
        elif len(step.observation) > 100:
            return "詳細な情報を取得できました。この結果を基に次のステップを計画します。"
        else:
            return "アクションは成功しました。得られた情報を活用して進めます。"
    
    def _calculate_step_confidence(self, step: ThoughtStep) -> float:
        """ステップ信頼度計算"""
        confidence = 0.5  # ベース信頼度
        
        # 思考内容の質
        if step.thought_content and len(step.thought_content) > 20:
            confidence += 0.1
        
        # 推論の存在
        if step.reasoning:
            confidence += 0.1
        
        # 観察結果の質
        if step.observation:
            if "error" not in step.observation.lower():
                confidence += 0.2
            if len(step.observation) > 50:
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_overall_confidence(self) -> float:
        """全体信頼度計算"""
        if not self.session.thought_steps:
            return 0.0
        
        step_confidences = [step.confidence_score for step in self.session.thought_steps if step.confidence_score > 0]
        
        if not step_confidences:
            return 0.5
        
        # 平均信頼度に調整を加える
        avg_confidence = sum(step_confidences) / len(step_confidences)
        
        # ステップ数による調整（多すぎると信頼度下がる）
        step_count_factor = max(0.8, 1.0 - (len(self.session.thought_steps) - 3) * 0.05)
        
        return min(avg_confidence * step_count_factor, 1.0)


class ChainOfThoughtEngine:
    """Chain-of-Thought 推論エンジン"""
    
    def __init__(self, ollama_client: OllamaClient, tool_manager: ToolManager):
        self.ollama_client = ollama_client
        self.tool_manager = tool_manager
        self.config = get_config()
        self.logger = get_logger()
        
        # エージェント設定
        self.max_iterations = 8
        self.max_execution_time = 180  # 3分
        
        # LangChain エージェント
        self.agent = None
        self.agent_executor = None
        
        # メモリ
        self.memory = ConversationBufferWindowMemory(
            k=3,  # 最新3回の会話を保持
            memory_key="chat_history",
            return_messages=True
        )
        
        # セッション管理
        self.active_sessions: Dict[str, ChainOfThoughtSession] = {}
        self.session_history: List[ChainOfThoughtSession] = []
        
        self.logger.log_startup(
            component="chain_of_thought_engine",
            version="1.0.0",
            config_summary={
                "max_iterations": self.max_iterations,
                "max_execution_time": self.max_execution_time,
                "memory_window": 3
            }
        )
    
    async def initialize(self) -> bool:
        """エンジン初期化"""
        try:
            # ツール準備
            tools = await self._prepare_tools()
            
            # Chain-of-Thought プロンプトテンプレート
            cot_prompt = self._create_cot_prompt()
            
            # LangChain LLM
            llm = self.ollama_client.primary_llm
            if not llm:
                raise ValueError("Primary LLM not available")
            
            # ReAct エージェント作成（Chain-of-Thought用に調整）
            self.agent = create_react_agent(
                llm=llm,
                tools=tools,
                prompt=cot_prompt
            )
            
            # エージェントエグゼキューター作成
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=tools,
                memory=self.memory,
                max_iterations=self.max_iterations,
                max_execution_time=self.max_execution_time,
                verbose=True,
                handle_parsing_errors=True,
                return_intermediate_steps=True
            )
            
            self.logger.log_startup(
                component="cot_engine_initialized",
                version="1.0.0",
                config_summary={
                    "tools_count": len(tools),
                    "llm_model": self.ollama_client.primary_model
                }
            )
            
            return True
            
        except Exception as e:
            self.logger.log_alert(
                alert_type="cot_initialization_failed",
                severity="ERROR",
                message=f"Chain-of-Thought engine initialization failed: {e}"
            )
            return False
    
    async def _prepare_tools(self) -> List[BaseTool]:
        """ツール準備"""
        tools = []
        
        # ツールマネージャーからツール取得
        for tool_name in ["structured_reasoning", "code_analysis", "task_breakdown"]:
            tool = self.tool_manager.get_tool(tool_name)
            if tool:
                tools.append(tool)
        
        # Chain-of-Thought専用ツール
        tools.extend(await self._create_cot_tools())
        
        return tools
    
    async def _create_cot_tools(self) -> List[BaseTool]:
        """Chain-of-Thought専用ツール作成"""
        cot_tools = []
        
        # 思考整理ツール
        class ThoughtOrganizerTool(BaseTool):
            name: str = "thought_organizer"
            description: str = "複雑な思考を整理し、論理的な構造に変換します"
            
            def _run(self, thoughts: str) -> str:
                try:
                    # 思考を段階的に整理
                    lines = thoughts.split('\n')
                    organized_thoughts = []
                    
                    current_section = ""
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # セクション判定
                        if any(keyword in line.lower() for keyword in ['まず', 'first', '最初に']):
                            current_section = "1. 初期分析"
                        elif any(keyword in line.lower() for keyword in ['次に', 'then', 'next']):
                            current_section = "2. 詳細検討"
                        elif any(keyword in line.lower() for keyword in ['最後に', 'finally', '結論']):
                            current_section = "3. 結論"
                        
                        if current_section and current_section not in organized_thoughts:
                            organized_thoughts.append(current_section)
                        
                        organized_thoughts.append(f"   - {line}")
                    
                    return "整理された思考:\n" + "\n".join(organized_thoughts)
                    
                except Exception as e:
                    return f"思考整理エラー: {e}"
            
            async def _arun(self, thoughts: str) -> str:
                return self._run(thoughts)
        
        # 論理検証ツール
        class LogicValidatorTool(BaseTool):
            name: str = "logic_validator"
            description: str = "推論の論理的妥当性を検証します"
            
            def _run(self, reasoning: str) -> str:
                try:
                    # 簡単な論理検証
                    issues = []
                    
                    # 矛盾チェック
                    if "しかし" in reasoning and "ただし" in reasoning:
                        issues.append("複数の反論が含まれており、論理が複雑になっています")
                    
                    # 根拠チェック
                    if not any(word in reasoning for word in ["なぜなら", "理由は", "根拠は", "because"]):
                        issues.append("根拠が明示されていません")
                    
                    # 結論チェック
                    if not any(word in reasoning for word in ["したがって", "よって", "結論として", "therefore"]):
                        issues.append("明確な結論が示されていません")
                    
                    if not issues:
                        return "論理的妥当性: ✅ 問題なし\n推論は論理的に一貫しています。"
                    else:
                        return f"論理的妥当性: ⚠️ 改善点あり\n" + "\n".join(f"- {issue}" for issue in issues)
                    
                except Exception as e:
                    return f"論理検証エラー: {e}"
            
            async def _arun(self, reasoning: str) -> str:
                return self._run(reasoning)
        
        # 信頼度評価ツール
        class ConfidenceEvaluatorTool(BaseTool):
            name: str = "confidence_evaluator"
            description: str = "推論結果の信頼度を評価します"
            
            def _run(self, conclusion: str) -> str:
                try:
                    confidence_factors = []
                    confidence_score = 0.5
                    
                    # 具体性チェック
                    if len(conclusion) > 100:
                        confidence_factors.append("詳細な結論 (+0.2)")
                        confidence_score += 0.2
                    
                    # 数値・データの存在
                    if any(char.isdigit() for char in conclusion):
                        confidence_factors.append("数値的根拠 (+0.1)")
                        confidence_score += 0.1
                    
                    # 不確実性の表現
                    uncertainty_words = ["おそらく", "可能性", "推測", "probably", "might"]
                    if any(word in conclusion for word in uncertainty_words):
                        confidence_factors.append("不確実性の認識 (+0.1)")
                        confidence_score += 0.1
                    else:
                        confidence_factors.append("断定的表現 (-0.1)")
                        confidence_score -= 0.1
                    
                    confidence_score = max(0.0, min(1.0, confidence_score))
                    
                    return f"信頼度評価: {confidence_score:.2f}\n" + "\n".join(confidence_factors)
                    
                except Exception as e:
                    return f"信頼度評価エラー: {e}"
            
            async def _arun(self, conclusion: str) -> str:
                return self._run(conclusion)
        
        cot_tools.extend([
            ThoughtOrganizerTool(),
            LogicValidatorTool(),
            ConfidenceEvaluatorTool()
        ])
        
        return cot_tools
    
    def _create_cot_prompt(self) -> PromptTemplate:
        """Chain-of-Thought プロンプトテンプレート作成"""
        template = """
あなたは段階的思考に優れたAI推論エンジンです。複雑な問題を論理的に分析し、明確な推論プロセスを示してください。

利用可能なツール:
{tools}

以下の形式で段階的に思考してください:

Thought: 現在の状況を分析し、何を考えるべきかを明確にします
Action: 必要に応じて適切なツールを使用します
Action Input: ツールへの具体的な入力
Observation: ツールの実行結果を観察します
... (必要に応じて思考プロセスを繰り返し)
Thought: 全ての情報を統合し、最終的な結論を導きます
Final Answer: 論理的で実用的な最終回答

重要な指針:
- 各思考ステップで「なぜそう考えるのか」を明確にしてください
- 仮定や推測は明示してください
- 複数の観点から問題を検討してください
- 結論に至る論理的な道筋を示してください
- 不確実な部分は正直に認めてください

過去の会話履歴:
{chat_history}

質問: {input}

{agent_scratchpad}
"""
        
        return PromptTemplate(
            input_variables=["tools", "tool_names", "input", "agent_scratchpad", "chat_history"],
            template=template
        )
    
    async def think_through(
        self, 
        query: str, 
        session_id: Optional[str] = None
    ) -> ChainOfThoughtSession:
        """Chain-of-Thought 推論実行"""
        
        # セッション作成
        if not session_id:
            session_id = f"cot_{int(time.time() * 1000)}"
        
        session = ChainOfThoughtSession(
            session_id=session_id,
            initial_query=query
        )
        
        self.active_sessions[session_id] = session
        
        # コールバックハンドラー作成
        callback_handler = ChainOfThoughtCallbackHandler(session)
        
        try:
            self.logger.log_performance_metric(
                metric_name="cot_session_started",
                value=len(query),
                unit="chars",
                component="chain_of_thought"
            )
            
            # エージェント実行
            result = await asyncio.to_thread(
                self.agent_executor.invoke,
                {
                    "input": query,
                    "chat_history": self.memory.chat_memory.messages
                },
                config={"callbacks": [callback_handler]}
            )
            
            # 結果処理
            session.final_conclusion = result.get("output", "")
            session.end_time = datetime.now()
            session.total_processing_time = (session.end_time - session.start_time).total_seconds()
            
            # 中間ステップ処理
            intermediate_steps = result.get("intermediate_steps", [])
            self._process_intermediate_steps(session, intermediate_steps)
            
            # メモリ更新
            self.memory.chat_memory.add_user_message(query)
            self.memory.chat_memory.add_ai_message(session.final_conclusion)
            
            # セッション履歴に追加
            self.session_history.append(session)
            
            # アクティブセッションから削除
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            self.logger.log_performance_metric(
                metric_name="cot_session_success",
                value=session.total_processing_time,
                unit="seconds",
                component="chain_of_thought"
            )
            
        except Exception as e:
            session.final_conclusion = f"推論中にエラーが発生しました: {e}"
            session.end_time = datetime.now()
            
            # エラーステップ追加
            error_step = ThoughtStep(
                step_number=len(session.thought_steps) + 1,
                state=ThoughtState.ERROR,
                thought_content=f"エラーが発生: {e}",
                observation=str(e)
            )
            session.thought_steps.append(error_step)
            
            self.logger.log_alert(
                alert_type="cot_processing_error",
                severity="ERROR",
                message=f"Chain-of-Thought processing failed: {e}"
            )
        
        return session
    
    def _process_intermediate_steps(
        self, 
        session: ChainOfThoughtSession, 
        intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> None:
        """中間ステップ処理"""
        for i, (action, observation) in enumerate(intermediate_steps):
            if i < len(session.thought_steps):
                # 既存のステップを更新
                step = session.thought_steps[i]
                if not step.observation:
                    step.observation = observation
                    step.state = ThoughtState.REFLECTING
            else:
                # 新しいステップを追加
                step = ThoughtStep(
                    step_number=i + 1,
                    state=ThoughtState.REFLECTING,
                    thought_content=getattr(action, 'log', ''),
                    action_taken=getattr(action, 'tool', ''),
                    action_input=getattr(action, 'tool_input', {}),
                    observation=observation
                )
                session.thought_steps.append(step)
    
    def get_reasoning_chain(self, session: ChainOfThoughtSession) -> List[str]:
        """推論チェーン取得"""
        return session.reasoning_chain.copy()
    
    def get_thought_analysis(self, session: ChainOfThoughtSession) -> Dict[str, Any]:
        """思考分析取得"""
        if not session.thought_steps:
            return {"error": "No thought steps found"}
        
        analysis = {
            "total_steps": len(session.thought_steps),
            "processing_time": session.total_processing_time,
            "confidence_score": session.confidence_score,
            "thought_states": {},
            "step_details": []
        }
        
        # 状態別集計
        for step in session.thought_steps:
            state = step.state.value
            if state not in analysis["thought_states"]:
                analysis["thought_states"][state] = 0
            analysis["thought_states"][state] += 1
        
        # ステップ詳細
        for step in session.thought_steps:
            step_detail = {
                "step": step.step_number,
                "state": step.state.value,
                "thought_length": len(step.thought_content),
                "has_reasoning": bool(step.reasoning),
                "has_action": bool(step.action_taken),
                "has_observation": bool(step.observation),
                "has_reflection": bool(step.reflection),
                "confidence": step.confidence_score,
                "processing_time": step.processing_time
            }
            analysis["step_details"].append(step_detail)
        
        return analysis
    
    def compare_sessions(
        self, 
        session1: ChainOfThoughtSession, 
        session2: ChainOfThoughtSession
    ) -> Dict[str, Any]:
        """セッション比較"""
        comparison = {
            "session1_id": session1.session_id,
            "session2_id": session2.session_id,
            "step_count_diff": len(session1.thought_steps) - len(session2.thought_steps),
            "processing_time_diff": session1.total_processing_time - session2.total_processing_time,
            "confidence_diff": session1.confidence_score - session2.confidence_score,
            "reasoning_complexity": {
                "session1": len(session1.reasoning_chain),
                "session2": len(session2.reasoning_chain)
            }
        }
        
        return comparison
    
    async def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """セッションサマリー取得"""
        # アクティブセッションから検索
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
        else:
            # 履歴から検索
            session = next((s for s in self.session_history if s.session_id == session_id), None)
        
        if not session:
            return None
        
        return {
            "session_id": session.session_id,
            "initial_query": session.initial_query,
            "final_conclusion": session.final_conclusion,
            "total_steps": len(session.thought_steps),
            "processing_time": session.total_processing_time,
            "confidence_score": session.confidence_score,
            "start_time": session.start_time.isoformat(),
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "reasoning_chain_length": len(session.reasoning_chain)
        }
    
    async def shutdown(self) -> None:
        """エンジン終了"""
        final_stats = {
            "total_sessions": len(self.session_history),
            "active_sessions": len(self.active_sessions),
            "memory_messages": len(self.memory.chat_memory.messages)
        }
        
        self.logger.log_shutdown(
            component="chain_of_thought_engine",
            uptime_seconds=0,  # TODO: 実際の稼働時間計算
            final_stats=final_stats
        )


# 便利関数
async def create_chain_of_thought_engine(
    ollama_client: OllamaClient, 
    tool_manager: ToolManager
) -> ChainOfThoughtEngine:
    """Chain-of-Thought エンジン作成・初期化"""
    engine = ChainOfThoughtEngine(ollama_client, tool_manager)
    
    if await engine.initialize():
        return engine
    else:
        raise RuntimeError("Failed to initialize Chain-of-Thought engine")


# 使用例
async def main():
    """テスト用メイン関数"""
    from ..inference.ollama_client import create_ollama_client
    from ..inference.tools import create_tool_manager
    
    try:
        # 依存関係作成
        ollama_client = await create_ollama_client()
        tool_manager = await create_tool_manager(ollama_client)
        
        # Chain-of-Thought エンジン作成
        cot_engine = await create_chain_of_thought_engine(ollama_client, tool_manager)
        
        # テスト質問
        complex_questions = [
            "Webアプリケーションのパフォーマンスが低下している問題を段階的に分析し、解決策を提案してください",
            "機械学習プロジェクトで精度が向上しない場合の原因を論理的に特定し、改善方法を考えてください",
            "新しいプログラミング言語を学習する最適な戦略を、学習者のレベル別に段階的に設計してください"
        ]
        
        for i, question in enumerate(complex_questions, 1):
            print(f"\n=== Chain-of-Thought Test {i} ===")
            print(f"Question: {question}")
            
            session = await cot_engine.think_through(question)
            
            print(f"\nFinal Conclusion: {session.final_conclusion}")
            print(f"Processing Time: {session.total_processing_time:.2f}s")
            print(f"Confidence Score: {session.confidence_score:.2f}")
            print(f"Thought Steps: {len(session.thought_steps)}")
            
            # 推論チェーン表示
            reasoning_chain = cot_engine.get_reasoning_chain(session)
            print(f"\nReasoning Chain:")
            for j, reasoning in enumerate(reasoning_chain, 1):
                print(f"  {j}. {reasoning}")
            
            # 思考分析
            analysis = cot_engine.get_thought_analysis(session)
            print(f"\nThought Analysis:")
            print(f"  States: {analysis['thought_states']}")
            print(f"  Average Confidence: {sum(s['confidence'] for s in analysis['step_details']) / len(analysis['step_details']):.2f}")
        
        await cot_engine.shutdown()
        await ollama_client.shutdown()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())