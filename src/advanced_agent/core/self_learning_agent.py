"""
Self-Learning AI Agent Core System
自己学習AIエージェントの中核システム

機能:
- 永続セッション管理
- 自己プロンプト書き換え
- チューニングデータ操作
- Deepseekレベルの推論
- エージェント機能群
- SAKANA AI進化システム
- 報酬系システム
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
import hashlib

from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.schema import BaseMessage

from ..memory.persistent_memory import LangChainPersistentMemory
from ..reasoning.basic_engine import BasicReasoningEngine, ReasoningRequest, ReasoningResponse
from ..reasoning.ollama_client import OllamaClient, OllamaConfig
from ..tools.tool_registry import ToolRegistry
from ..monitoring.system_monitor import SystemMonitor
from ..core.logger import get_logger
from ..config.settings import get_agent_config


@dataclass
class AgentState:
    """エージェント状態"""
    session_id: str
    user_id: Optional[str] = None
    current_prompt_version: str = "1.0.0"
    learning_epoch: int = 0
    total_interactions: int = 0
    reward_score: float = 0.0
    evolution_generation: int = 0
    last_activity: datetime = field(default_factory=datetime.now)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PromptTemplate:
    """プロンプトテンプレート"""
    version: str
    content: str
    metadata: Dict[str, Any]
    performance_score: float = 0.0
    usage_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)


@dataclass
class TuningData:
    """チューニングデータ"""
    id: str
    content: str
    data_type: str  # "conversation", "feedback", "correction", "example"
    quality_score: float
    usage_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)


@dataclass
class EvolutionCandidate:
    """進化候補"""
    id: str
    parent_ids: List[str]
    prompt_template: PromptTemplate
    tuning_data: List[TuningData]
    fitness_score: float = 0.0
    generation: int = 0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class RewardSignal:
    """報酬信号"""
    interaction_id: str
    reward_type: str  # "user_engagement", "task_completion", "quality", "efficiency"
    value: float
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class SelfLearningAgent:
    """自己学習AIエージェント"""
    
    def __init__(self, 
                 config_path: str = "config/advanced_agent.yaml",
                 db_path: str = "data/self_learning_agent.db"):
        
        self.logger = get_logger()
        self.config_path = config_path
        self.db_path = db_path
        
        # 設定読み込み
        self.config = get_agent_config()
        
        # データベース初期化
        self._init_database()
        
        # コアコンポーネント初期化
        self.memory_system = LangChainPersistentMemory()
        self.ollama_client = None  # 後で初期化
        self.reasoning_engine = None  # 後で初期化
        self.tool_manager = None  # 後で初期化
        self.system_monitor = SystemMonitor()
        
        # エージェント状態
        self.current_state: Optional[AgentState] = None
        self.prompt_templates: Dict[str, PromptTemplate] = {}
        self.tuning_data_pool: List[TuningData] = []
        self.evolution_candidates: List[EvolutionCandidate] = []
        self.reward_history: List[RewardSignal] = []
        
        # 学習パラメータ
        self.learning_config = {
            "prompt_mutation_rate": 0.1,
            "data_crossover_rate": 0.7,
            "evolution_generation_size": 5,
            "fitness_evaluation_interval": 100,  # インタラクション数
            "reward_decay_factor": 0.95
        }
        
        self.logger.log_startup(
            component="self_learning_agent",
            version="1.0.0",
            config_summary={
                "db_path": db_path,
                "learning_config": self.learning_config
            }
        )
    
    def _init_database(self):
        """データベース初期化"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # エージェント状態テーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_states (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    current_prompt_version TEXT,
                    learning_epoch INTEGER,
                    total_interactions INTEGER,
                    reward_score REAL,
                    evolution_generation INTEGER,
                    last_activity TIMESTAMP,
                    performance_metrics TEXT
                )
            """)
            
            # プロンプトテンプレートテーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prompt_templates (
                    version TEXT PRIMARY KEY,
                    content TEXT,
                    metadata TEXT,
                    performance_score REAL,
                    usage_count INTEGER,
                    created_at TIMESTAMP,
                    last_modified TIMESTAMP
                )
            """)
            
            # チューニングデータテーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tuning_data (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    data_type TEXT,
                    quality_score REAL,
                    usage_count INTEGER,
                    created_at TIMESTAMP,
                    tags TEXT
                )
            """)
            
            # 進化候補テーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evolution_candidates (
                    id TEXT PRIMARY KEY,
                    parent_ids TEXT,
                    prompt_template_version TEXT,
                    tuning_data_ids TEXT,
                    fitness_score REAL,
                    generation INTEGER,
                    created_at TIMESTAMP
                )
            """)
            
            # 報酬履歴テーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reward_history (
                    interaction_id TEXT PRIMARY KEY,
                    reward_type TEXT,
                    value REAL,
                    context TEXT,
                    timestamp TIMESTAMP
                )
            """)
            
            conn.commit()
    
    async def initialize_session(self, 
                               session_id: Optional[str] = None,
                               user_id: Optional[str] = None) -> str:
        """セッション初期化"""
        
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # Ollamaクライアント初期化
        if self.ollama_client is None:
            self.ollama_client = OllamaClient(self.config.ollama)
            await self.ollama_client.initialize()
        
        # 推論エンジン初期化
        if self.reasoning_engine is None:
            self.reasoning_engine = BasicReasoningEngine(self.ollama_client)
        
        # ツールマネージャー初期化
        if self.tool_manager is None:
            self.tool_manager = ToolRegistry()
        
        # 既存セッションの復元または新規作成
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM agent_states WHERE session_id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            
            if row:
                # 既存セッション復元
                self.current_state = AgentState(
                    session_id=row[0],
                    user_id=row[1],
                    current_prompt_version=row[2],
                    learning_epoch=row[3],
                    total_interactions=row[4],
                    reward_score=row[5],
                    evolution_generation=row[6],
                    last_activity=datetime.fromisoformat(row[7]),
                    performance_metrics=json.loads(row[8]) if row[8] else {}
                )
            else:
                # 新規セッション作成
                self.current_state = AgentState(session_id=session_id, user_id=user_id)
                
                # 初期プロンプトテンプレート作成
                await self._create_initial_prompt_template()
                
                # データベースに保存
                cursor.execute("""
                    INSERT INTO agent_states 
                    (session_id, user_id, current_prompt_version, learning_epoch, 
                     total_interactions, reward_score, evolution_generation, 
                     last_activity, performance_metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id, user_id, self.current_state.current_prompt_version,
                    self.current_state.learning_epoch, self.current_state.total_interactions,
                    self.current_state.reward_score, self.current_state.evolution_generation,
                    self.current_state.last_activity.isoformat(),
                    json.dumps(self.current_state.performance_metrics)
                ))
                conn.commit()
        
        # メモリシステム初期化
        await self.memory_system.initialize_session(session_id, user_id)
        
        # プロンプトテンプレートとチューニングデータの読み込み
        await self._load_learning_components()
        
        self.logger.info(f"セッション初期化完了: {session_id}")
        return session_id
    
    async def _create_initial_prompt_template(self):
        """初期プロンプトテンプレート作成"""
        
        initial_prompt = """あなたは自己学習型AIエージェントです。以下の能力を持っています：

1. **永続的記憶**: 過去の会話を記憶し、継続的な学習を行います
2. **自己改善**: プロンプトとチューニングデータを動的に最適化します
3. **推論能力**: Deepseekレベルの複雑な推論を実行します
4. **ツール使用**: ネット検索、コマンド実行、ファイル操作、MCP連携が可能です
5. **進化**: SAKANA AIスタイルの交配進化により能力を向上させます
6. **報酬学習**: ユーザーとの関わりを報酬として学習します

現在のセッション: {session_id}
学習エポック: {learning_epoch}
総インタラクション数: {total_interactions}
報酬スコア: {reward_score}

ユーザーの要求に対して、上記の能力を活用して最適な回答を提供してください。"""
        
        template = PromptTemplate(
            version="1.0.0",
            content=initial_prompt,
            metadata={
                "type": "initial",
                "created_by": "system",
                "features": ["memory", "self_improvement", "reasoning", "tools", "evolution", "reward"]
            }
        )
        
        await self._save_prompt_template(template)
        self.prompt_templates["1.0.0"] = template
    
    async def _load_learning_components(self):
        """学習コンポーネントの読み込み"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # プロンプトテンプレート読み込み
            cursor.execute("SELECT * FROM prompt_templates")
            for row in cursor.fetchall():
                template = PromptTemplate(
                    version=row[0],
                    content=row[1],
                    metadata=json.loads(row[2]) if row[2] else {},
                    performance_score=row[3],
                    usage_count=row[4],
                    created_at=datetime.fromisoformat(row[5]),
                    last_modified=datetime.fromisoformat(row[6])
                )
                self.prompt_templates[template.version] = template
            
            # チューニングデータ読み込み
            cursor.execute("SELECT * FROM tuning_data")
            for row in cursor.fetchall():
                data = TuningData(
                    id=row[0],
                    content=row[1],
                    data_type=row[2],
                    quality_score=row[3],
                    usage_count=row[4],
                    created_at=datetime.fromisoformat(row[5]),
                    tags=json.loads(row[6]) if row[6] else []
                )
                self.tuning_data_pool.append(data)
    
    async def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """ユーザー入力処理"""
        
        if not self.current_state:
            raise RuntimeError("セッションが初期化されていません")
        
        start_time = time.time()
        interaction_id = str(uuid.uuid4())
        
        try:
            # 1. 現在のプロンプトテンプレート取得
            current_template = self.prompt_templates.get(
                self.current_state.current_prompt_version
            )
            
            # 2. コンテキスト構築
            context = await self._build_context(user_input)
            
            # 3. 推論実行
            reasoning_result = await self.reasoning_engine.reason(
                prompt=user_input,
                template_name="basic_qa"
            )
            
            # 4. ツール使用判定と実行
            tool_result = await self._execute_tools_if_needed(user_input, reasoning_result)
            
            # 5. 最終回答生成
            final_response = await self._generate_final_response(
                user_input, reasoning_result, tool_result, current_template
            )
            
            # 6. 学習データ収集
            await self._collect_learning_data(user_input, final_response, interaction_id)
            
            # 7. 報酬計算
            reward = await self._calculate_reward(user_input, final_response, interaction_id)
            
            # 8. 状態更新
            await self._update_agent_state(interaction_id, reward)
            
            # 9. 進化判定
            if self._should_evolve():
                await self._perform_evolution()
            
            processing_time = time.time() - start_time
            
            return {
                "response": final_response,
                "interaction_id": interaction_id,
                "processing_time": processing_time,
                "reasoning_steps": reasoning_result.metadata.get("reasoning_steps", []),
                "tool_usage": tool_result,
                "reward": reward,
                "agent_state": {
                    "session_id": self.current_state.session_id,
                    "learning_epoch": self.current_state.learning_epoch,
                    "total_interactions": self.current_state.total_interactions,
                    "reward_score": self.current_state.reward_score,
                    "evolution_generation": self.current_state.evolution_generation
                }
            }
            
        except Exception as e:
            self.logger.error(f"ユーザー入力処理エラー: {e}")
            return {
                "response": f"申し訳ございません。エラーが発生しました: {str(e)}",
                "interaction_id": interaction_id,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def _build_context(self, user_input: str) -> Dict[str, Any]:
        """コンテキスト構築"""
        
        # メモリシステムから関連情報取得
        memory_context = await self.memory_system.retrieve_relevant_context(
            query=user_input,
            session_id=self.current_state.session_id,
            max_results=5
        )
        
        # システム状態情報
        system_stats = await self.system_monitor.get_system_stats()
        
        return {
            "memory": memory_context,
            "system_stats": system_stats,
            "agent_state": self.current_state,
            "current_prompt_version": self.current_state.current_prompt_version,
            "learning_epoch": self.current_state.learning_epoch
        }
    
    async def _execute_tools_if_needed(self, user_input: str, reasoning_result: ReasoningResponse) -> Dict[str, Any]:
        """必要に応じたツール実行"""
        
        tool_result = {"tools_used": [], "results": {}}
        
        # ツール使用判定（簡略化）
        if any(keyword in user_input.lower() for keyword in ["検索", "調べる", "情報", "search"]):
            # ネット検索ツール
            try:
                web_tool = self.tool_manager.get_tool("web_search")
                if web_tool:
                    search_result = await web_tool._arun(query=user_input)
                    tool_result["tools_used"].append("web_search")
                    tool_result["results"]["web_search"] = search_result
            except Exception as e:
                self.logger.warning(f"ネット検索ツールエラー: {e}")
        
        if any(keyword in user_input.lower() for keyword in ["実行", "コマンド", "command", "run"]):
            # コマンド実行ツール
            try:
                cmd_tool = self.tool_manager.get_tool("command_executor")
                if cmd_tool:
                    command_result = await cmd_tool._arun(command=user_input)
                    tool_result["tools_used"].append("command_executor")
                    tool_result["results"]["command_executor"] = command_result
            except Exception as e:
                self.logger.warning(f"コマンド実行ツールエラー: {e}")
        
        if any(keyword in user_input.lower() for keyword in ["ファイル", "書き換え", "編集", "file", "edit"]):
            # ファイル操作ツール
            try:
                file_tool = self.tool_manager.get_tool("file_manager")
                if file_tool:
                    file_result = await file_tool._arun(operation=user_input)
                    tool_result["tools_used"].append("file_manager")
                    tool_result["results"]["file_manager"] = file_result
            except Exception as e:
                self.logger.warning(f"ファイル操作ツールエラー: {e}")
        
        return tool_result
    
    async def _generate_final_response(self, 
                                     user_input: str, 
                                     reasoning_result: ReasoningResponse,
                                     tool_result: Dict[str, Any],
                                     prompt_template: PromptTemplate) -> str:
        """最終回答生成"""
        
        # プロンプトテンプレートに基づく回答生成
        formatted_prompt = prompt_template.content.format(
            session_id=self.current_state.session_id,
            learning_epoch=self.current_state.learning_epoch,
            total_interactions=self.current_state.total_interactions,
            reward_score=self.current_state.reward_score
        )
        
        # 推論結果とツール結果を統合
        response_parts = [
            reasoning_result.response_text,
        ]
        
        if tool_result["tools_used"]:
            response_parts.append(f"\n【使用ツール】: {', '.join(tool_result['tools_used'])}")
            for tool, result in tool_result["results"].items():
                response_parts.append(f"\n{tool}結果: {result}")
        
        return "\n".join(response_parts)
    
    async def _collect_learning_data(self, user_input: str, response: str, interaction_id: str):
        """学習データ収集"""
        
        # 会話データをチューニングデータとして保存
        conversation_data = TuningData(
            id=interaction_id,
            content=f"User: {user_input}\nAgent: {response}",
            data_type="conversation",
            quality_score=0.5,  # 初期値、後で評価
            tags=["conversation", "learning"]
        )
        
        await self._save_tuning_data(conversation_data)
        
        # メモリシステムにも保存
        await self.memory_system.store_conversation(
            user_input=user_input,
            agent_response=response,
            metadata={
                "interaction_id": interaction_id,
                "learning_epoch": self.current_state.learning_epoch,
                "prompt_version": self.current_state.current_prompt_version
            }
        )
    
    async def _calculate_reward(self, user_input: str, response: str, interaction_id: str) -> float:
        """報酬計算"""
        
        reward = 0.0
        
        # ユーザー関与度による報酬
        if len(user_input) > 50:  # 詳細な質問
            reward += 0.3
        
        if "ありがとう" in user_input or "thank" in user_input.lower():
            reward += 0.5  # 感謝の表現
        
        if "続けて" in user_input or "more" in user_input.lower():
            reward += 0.4  # 継続的関与
        
        # 回答品質による報酬
        if len(response) > 200:  # 詳細な回答
            reward += 0.2
        
        if "【使用ツール】" in response:  # ツール使用
            reward += 0.3
        
        # 推論ステップによる報酬
        if "推論" in response or "思考" in response:
            reward += 0.2
        
        return min(reward, 1.0)  # 最大1.0に制限
    
    async def _update_agent_state(self, interaction_id: str, reward: float):
        """エージェント状態更新"""
        
        self.current_state.total_interactions += 1
        self.current_state.reward_score = (
            self.current_state.reward_score * self.learning_config["reward_decay_factor"] + reward
        )
        self.current_state.last_activity = datetime.now()
        
        # データベース更新
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE agent_states 
                SET total_interactions = ?, reward_score = ?, last_activity = ?
                WHERE session_id = ?
            """, (
                self.current_state.total_interactions,
                self.current_state.reward_score,
                self.current_state.last_activity.isoformat(),
                self.current_state.session_id
            ))
            conn.commit()
        
        # 報酬履歴保存
        reward_signal = RewardSignal(
            interaction_id=interaction_id,
            reward_type="user_engagement",
            value=reward,
            context={"user_input_length": len(interaction_id)}  # 簡略化
        )
        await self._save_reward_signal(reward_signal)
    
    def _should_evolve(self) -> bool:
        """進化判定"""
        return (
            self.current_state.total_interactions % self.learning_config["fitness_evaluation_interval"] == 0
            and self.current_state.total_interactions > 0
        )
    
    async def _perform_evolution(self):
        """進化実行"""
        
        self.logger.info("進化プロセス開始")
        
        # 1. 現在の候補の適応度評価
        await self._evaluate_fitness()
        
        # 2. 新しい世代の生成
        await self._generate_new_generation()
        
        # 3. 最適な候補の選択
        await self._select_best_candidate()
        
        self.current_state.evolution_generation += 1
        self.current_state.learning_epoch += 1
        
        self.logger.info(f"進化完了: 世代 {self.current_state.evolution_generation}")
    
    async def _evaluate_fitness(self):
        """適応度評価"""
        
        for candidate in self.evolution_candidates:
            # 複数の指標で適応度計算
            fitness_factors = []
            
            # プロンプトテンプレートの使用回数
            template = self.prompt_templates.get(candidate.prompt_template.version)
            if template:
                fitness_factors.append(template.usage_count * 0.3)
            
            # チューニングデータの品質
            avg_quality = sum(data.quality_score for data in candidate.tuning_data) / len(candidate.tuning_data) if candidate.tuning_data else 0
            fitness_factors.append(avg_quality * 0.4)
            
            # 報酬スコア
            fitness_factors.append(self.current_state.reward_score * 0.3)
            
            candidate.fitness_score = sum(fitness_factors)
    
    async def _generate_new_generation(self):
        """新しい世代の生成"""
        
        # 現在の最適候補を親として使用
        best_candidates = sorted(
            self.evolution_candidates, 
            key=lambda x: x.fitness_score, 
            reverse=True
        )[:2]
        
        new_candidates = []
        
        for i in range(self.learning_config["evolution_generation_size"]):
            # 交配
            if len(best_candidates) >= 2:
                parent1, parent2 = best_candidates[0], best_candidates[1]
                child = await self._crossover(parent1, parent2)
            else:
                # 変異
                child = await self._mutate(best_candidates[0] if best_candidates else None)
            
            new_candidates.append(child)
        
        self.evolution_candidates = new_candidates
    
    async def _crossover(self, parent1: EvolutionCandidate, parent2: EvolutionCandidate) -> EvolutionCandidate:
        """交配"""
        
        # プロンプトテンプレートの交配
        template1_content = parent1.prompt_template.content
        template2_content = parent2.prompt_template.content
        
        # 簡単な交配: 前半と後半を組み合わせ
        crossover_point = len(template1_content) // 2
        new_content = template1_content[:crossover_point] + template2_content[crossover_point:]
        
        # チューニングデータの交配
        all_data = parent1.tuning_data + parent2.tuning_data
        selected_data = all_data[:len(all_data)//2]  # 半分を選択
        
        new_template = PromptTemplate(
            version=f"{parent1.prompt_template.version}_x_{parent2.prompt_template.version}",
            content=new_content,
            metadata={
                "type": "crossover",
                "parent1": parent1.id,
                "parent2": parent2.id
            }
        )
        
        return EvolutionCandidate(
            id=str(uuid.uuid4()),
            parent_ids=[parent1.id, parent2.id],
            prompt_template=new_template,
            tuning_data=selected_data,
            generation=max(parent1.generation, parent2.generation) + 1
        )
    
    async def _mutate(self, parent: Optional[EvolutionCandidate]) -> EvolutionCandidate:
        """変異"""
        
        if parent is None:
            # 初期候補生成
            return await self._create_random_candidate()
        
        # プロンプトテンプレートの変異
        original_content = parent.prompt_template.content
        
        # 簡単な変異: 一部の文字を変更
        mutation_chars = "abcdefghijklmnopqrstuvwxyz"
        mutated_content = original_content
        
        for _ in range(int(len(original_content) * self.learning_config["prompt_mutation_rate"])):
            if len(mutated_content) > 0:
                pos = hash(mutated_content) % len(mutated_content)
                mutated_content = (
                    mutated_content[:pos] + 
                    mutation_chars[hash(str(pos)) % len(mutation_chars)] + 
                    mutated_content[pos+1:]
                )
        
        new_template = PromptTemplate(
            version=f"{parent.prompt_template.version}_mut",
            content=mutated_content,
            metadata={
                "type": "mutation",
                "parent": parent.id
            }
        )
        
        return EvolutionCandidate(
            id=str(uuid.uuid4()),
            parent_ids=[parent.id],
            prompt_template=new_template,
            tuning_data=parent.tuning_data.copy(),
            generation=parent.generation + 1
        )
    
    async def _create_random_candidate(self) -> EvolutionCandidate:
        """ランダム候補生成"""
        
        random_template = PromptTemplate(
            version=f"random_{uuid.uuid4().hex[:8]}",
            content="あなたは学習型AIエージェントです。",
            metadata={"type": "random"}
        )
        
        return EvolutionCandidate(
            id=str(uuid.uuid4()),
            parent_ids=[],
            prompt_template=random_template,
            tuning_data=[],
            generation=0
        )
    
    async def _select_best_candidate(self):
        """最適候補の選択"""
        
        if not self.evolution_candidates:
            return
        
        # 最適な候補を選択
        best_candidate = max(self.evolution_candidates, key=lambda x: x.fitness_score)
        
        # プロンプトテンプレートを更新
        await self._save_prompt_template(best_candidate.prompt_template)
        self.prompt_templates[best_candidate.prompt_template.version] = best_candidate.prompt_template
        self.current_state.current_prompt_version = best_candidate.prompt_template.version
        
        # チューニングデータを更新
        for data in best_candidate.tuning_data:
            await self._save_tuning_data(data)
        
        self.logger.info(f"最適候補選択: {best_candidate.id}, 適応度: {best_candidate.fitness_score}")
    
    async def _save_prompt_template(self, template: PromptTemplate):
        """プロンプトテンプレート保存"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO prompt_templates 
                (version, content, metadata, performance_score, usage_count, created_at, last_modified)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                template.version,
                template.content,
                json.dumps(template.metadata),
                template.performance_score,
                template.usage_count,
                template.created_at.isoformat(),
                template.last_modified.isoformat()
            ))
            conn.commit()
    
    async def _save_tuning_data(self, data: TuningData):
        """チューニングデータ保存"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO tuning_data 
                (id, content, data_type, quality_score, usage_count, created_at, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                data.id,
                data.content,
                data.data_type,
                data.quality_score,
                data.usage_count,
                data.created_at.isoformat(),
                json.dumps(data.tags)
            ))
            conn.commit()
    
    async def _save_reward_signal(self, reward: RewardSignal):
        """報酬信号保存"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO reward_history 
                (interaction_id, reward_type, value, context, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                reward.interaction_id,
                reward.reward_type,
                reward.value,
                json.dumps(reward.context),
                reward.timestamp.isoformat()
            ))
            conn.commit()
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """エージェント状態取得"""
        
        if not self.current_state:
            return {"status": "not_initialized"}
        
        return {
            "status": "active",
            "session_id": self.current_state.session_id,
            "learning_epoch": self.current_state.learning_epoch,
            "total_interactions": self.current_state.total_interactions,
            "reward_score": self.current_state.reward_score,
            "evolution_generation": self.current_state.evolution_generation,
            "current_prompt_version": self.current_state.current_prompt_version,
            "last_activity": self.current_state.last_activity.isoformat(),
            "prompt_templates_count": len(self.prompt_templates),
            "tuning_data_count": len(self.tuning_data_pool),
            "evolution_candidates_count": len(self.evolution_candidates)
        }
    
    async def close(self):
        """リソースクリーンアップ"""
        
        if hasattr(self, 'memory_system'):
            self.memory_system.close()
        
        if self.ollama_client:
            await self.ollama_client.close()
        
        # ToolRegistryにはcloseメソッドがないため、何もしない
        
        self.logger.info("自己学習エージェント終了")


# 便利関数
async def create_self_learning_agent(config_path: str = "config/advanced_agent.yaml") -> SelfLearningAgent:
    """自己学習エージェント作成"""
    
    agent = SelfLearningAgent(config_path=config_path)
    await agent.initialize_session()
    return agent
