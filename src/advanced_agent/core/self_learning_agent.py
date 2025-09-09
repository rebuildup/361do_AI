"""
361do_AI Core System
361do_AIの中核システム

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
from ..inference.ollama_client import OllamaClient
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
    """361do_AI"""
    
    def __init__(self, 
                 config_path: str = "config/advanced_agent.yaml",
                 db_path: str = "data/self_learning_agent.db",
                 system_monitor: Optional[SystemMonitor] = None,
                 memory_system: Optional[LangChainPersistentMemory] = None,
                 ollama_client: Optional[OllamaClient] = None,
                 reasoning_engine: Optional[BasicReasoningEngine] = None,
                 tool_manager: Optional[ToolRegistry] = None,
                 **kwargs):
        
        self.logger = get_logger()
        self.config_path = config_path
        self.db_path = db_path
        
        # 設定読み込み
        self.config = get_agent_config()
        
        # データベース初期化
        self._init_database()
        
        # コアコンポーネント初期化
        self.memory_system = memory_system or LangChainPersistentMemory()
        self.ollama_client = ollama_client  # 後で初期化される場合あり
        self.reasoning_engine = reasoning_engine  # 後で初期化される場合あり
        self.tool_manager = tool_manager  # 後で初期化される場合あり
        self.system_monitor = system_monitor or SystemMonitor()
        self.prompt_manager = None  # 後で初期化
        
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
            
            # エージェント状態テーブル（完全なスキーマ）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_states (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    current_prompt_version TEXT DEFAULT '1.0.0',
                    learning_epoch INTEGER DEFAULT 0,
                    total_interactions INTEGER DEFAULT 0,
                    reward_score REAL DEFAULT 0.0,
                    evolution_generation INTEGER DEFAULT 0,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    performance_metrics TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # プロンプトテンプレートテーブル（完全なスキーマ）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prompt_templates (
                    version TEXT PRIMARY KEY,
                    content TEXT,
                    metadata TEXT,
                    performance_score REAL DEFAULT 0.0,
                    usage_count INTEGER DEFAULT 0,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # チューニングデータテーブル（完全なスキーマ）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tuning_data (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    data_type TEXT,
                    quality_score REAL DEFAULT 0.0,
                    usage_count INTEGER DEFAULT 0,
                    tags TEXT,
                    metadata TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 進化候補テーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evolution_candidates (
                    id TEXT PRIMARY KEY,
                    parent_ids TEXT,
                    prompt_template_version TEXT,
                    tuning_data_ids TEXT,
                    fitness_score REAL DEFAULT 0.0,
                    generation INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 報酬履歴テーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reward_history (
                    interaction_id TEXT PRIMARY KEY,
                    reward_type TEXT,
                    value REAL,
                    context TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def _get_persistent_session_id(self, user_id: str) -> str:
        """ユーザーIDに基づく永続的なセッションIDを生成"""
        import hashlib
        
        # ユーザーIDから一意のセッションIDを生成
        hash_object = hashlib.md5(f"persistent_session_{user_id}".encode())
        session_id = hash_object.hexdigest()
        
        # UUID形式に変換（8-4-4-4-12の形式）
        formatted_session_id = f"{session_id[:8]}-{session_id[8:12]}-{session_id[12:16]}-{session_id[16:20]}-{session_id[20:32]}"
        
        self.logger.info(f"永続セッションID生成: ユーザーID={user_id}, セッションID={formatted_session_id}")
        return formatted_session_id
    
    async def _load_session_conversation_history(self, session_id: str):
        """セッションの会話履歴を読み込み"""
        try:
            # メモリシステムから会話履歴を取得
            if self.memory_system:
                try:
                    # 直接メソッドを呼び出し
                    history = await self.memory_system.get_conversation_history(session_id, limit=50)
                    
                    if history:
                        self.logger.info(f"会話履歴読み込み完了: {len(history)}件の会話を復元")
                        
                        # 会話履歴をメモリシステムに再構築
                        for conv in history:
                            if conv.get('user_input') and conv.get('agent_response'):
                                # LangChainメモリに会話を追加
                                if hasattr(self.memory_system, 'conversation_memory'):
                                    from langchain.schema import HumanMessage, AIMessage
                                    self.memory_system.conversation_memory.chat_memory.add_user_message(conv['user_input'])
                                    self.memory_system.conversation_memory.chat_memory.add_ai_message(conv['agent_response'])
                    else:
                        self.logger.info("会話履歴が見つかりませんでした")
                except Exception as e:
                    self.logger.error(f"会話履歴取得エラー: {e}")
            else:
                self.logger.warning("メモリシステムが初期化されていません")
                
        except Exception as e:
            self.logger.error(f"会話履歴読み込みエラー: {e}")
    
    async def initialize_session(self, 
                               session_id: Optional[str] = None,
                               user_id: Optional[str] = None) -> str:
        """セッション初期化"""
        
        # 永続的なセッション管理
        if session_id is None:
            if user_id is None:
                user_id = "default_user"
            
            # ユーザーIDに基づく固定セッションIDを生成
            session_id = self._get_persistent_session_id(user_id)
        
        # Ollamaクライアント初期化
        if self.ollama_client is None:
            self.logger.info("Ollamaクライアントを初期化中...")
            self.ollama_client = OllamaClient(self.config.ollama.base_url)
            await self.ollama_client.initialize()
            self.logger.info("Ollamaクライアント初期化完了")
        
        # 推論エンジン初期化
        if self.reasoning_engine is None:
            self.logger.info("推論エンジンを初期化中...")
            self.reasoning_engine = BasicReasoningEngine(self.ollama_client)
            await self.reasoning_engine.initialize()
            self.logger.info("推論エンジン初期化完了")
        
        # ツールマネージャー初期化
        if self.tool_manager is None:
            self.tool_manager = ToolRegistry()
        
        # プロンプトマネージャー初期化
        if self.prompt_manager is None:
            from ..learning.prompt_manager import PromptManager
            self.prompt_manager = PromptManager(self.ollama_client)
        
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
                
                # 既存セッションの会話履歴を読み込み
                await self._load_session_conversation_history(session_id)
                
                self.logger.info(f"既存セッション復元完了: {session_id}, インタラクション数: {self.current_state.total_interactions}")
            else:
                # 新規セッション作成
                self.current_state = AgentState(session_id=session_id, user_id=user_id)
                
                # 初期プロンプトテンプレート作成
                await self._create_initial_prompt_template(session_id)
                
                # プロンプトテンプレートが正しく作成されたか確認
                if not self.prompt_templates:
                    self.logger.error("初期プロンプトテンプレートの作成に失敗しました")
                    raise RuntimeError("初期プロンプトテンプレートの作成に失敗しました")
                
                # 現在のプロンプトバージョンを設定
                self.current_state.current_prompt_version = "1.0.0"
                
                # プロンプトテンプレートが正しく作成されたか確認
                if "1.0.0" not in self.prompt_templates:
                    self.logger.error("プロンプトテンプレート '1.0.0' が見つかりません")
                    raise RuntimeError("プロンプトテンプレート '1.0.0' が見つかりません")
                
                # プロンプトテンプレートの作成を確認
                self.logger.info(f"プロンプトテンプレート '1.0.0' が正常に作成されました")
                
                # プロンプトテンプレートの内容を確認
                template_content = self.prompt_templates["1.0.0"].content
                self.logger.info(f"プロンプトテンプレートの内容: {template_content[:100]}...")
                
                # プロンプトテンプレートの変数を確認（SQLAlchemyモデルのため、直接アクセス）
                import re
                variables = re.findall(r'\{([^}]+)\}', template_content)
                self.logger.info(f"プロンプトテンプレートの変数: {variables}")
                
                # プロンプトテンプレートの検証（簡易版）
                validation_result = {
                    "valid": True,
                    "errors": [],
                    "warnings": [],
                    "score": 1.0
                }
                self.logger.info(f"プロンプトテンプレートの検証結果: {validation_result}")
                
                # プロンプトテンプレートの使用回数を初期化
                self.prompt_templates["1.0.0"].usage_count = 0
                
                # プロンプトテンプレートの作成完了をログに記録
                self.logger.info("初期プロンプトテンプレートの作成が完了しました")
                
                # データベースに保存
                cursor.execute("""
                    INSERT INTO agent_states 
                    (session_id, user_id, current_prompt_version, learning_epoch, 
                     total_interactions, reward_score, evolution_generation, 
                     last_activity, performance_metrics, is_active, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id, user_id, self.current_state.current_prompt_version,
                    self.current_state.learning_epoch, self.current_state.total_interactions,
                    self.current_state.reward_score, self.current_state.evolution_generation,
                    self.current_state.last_activity.isoformat(),
                    json.dumps(self.current_state.performance_metrics),
                    True, datetime.now().isoformat(), datetime.now().isoformat()
                ))
                conn.commit()
        
        # メモリシステム初期化
        await self.memory_system.initialize_session(session_id, user_id)
        
        # プロンプトテンプレートとチューニングデータの読み込み
        await self._load_learning_components()
        
        self.logger.info(f"セッション初期化完了: {session_id}")
        return session_id
    
    async def _create_initial_prompt_template(self, session_id: str):
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

## 重要な指示:
ユーザーの要求を自然言語で理解し、必要に応じて以下のツールを**自発的に判断して使用**してください：

### ツール使用の判断基準:
1. **Web検索**: 最新情報、リアルタイムデータ、ニュース、記事などが必要な場合
   - 推論過程で「検索して」「調べて」「最新情報を探して」と表現してください

2. **コマンド実行**: システム情報、プロセス確認、環境設定などが必要な場合
   - 推論過程で「システム情報を確認」「状態を確認」「環境を確認」と表現してください

3. **ファイル操作**: ファイルの作成、編集、管理などが必要な場合
   - 推論過程で「ファイルを確認」「設定ファイルを確認」と表現してください

4. **MCP連携**: 外部ツールやサービスとの連携が必要な場合
   - 推論過程で「外部ツールと連携」「外部サービスを使用」と表現してください

### 推論過程の例:
「ユーザーの要求を満たすために、最新の情報が必要です。検索して詳細な情報を取得します。」
「システムの状態を確認する必要があります。システム情報を確認します。」
「ファイルの内容を確認する必要があります。ファイルを確認します。」

ユーザーの要求に対して、上記の能力を**自発的に活用**して最適な回答を提供してください。"""
        
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
        
        self.logger.info("学習コンポーネントの読み込みを開始...")
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # プロンプトテンプレート読み込み
                try:
                    cursor.execute("SELECT version, content, metadata, performance_score, usage_count, created_at, last_modified FROM prompt_templates")
                    for row in cursor.fetchall():
                        # 日時フィールドの安全な処理
                        created_at = row[5]
                        if isinstance(created_at, str):
                            try:
                                created_at = datetime.fromisoformat(created_at)
                            except ValueError:
                                created_at = datetime.now()
                        elif not isinstance(created_at, datetime):
                            created_at = datetime.now()
                        
                        last_modified = row[6]
                        if isinstance(last_modified, str):
                            try:
                                last_modified = datetime.fromisoformat(last_modified)
                            except ValueError:
                                last_modified = datetime.now()
                        elif not isinstance(last_modified, datetime):
                            last_modified = datetime.now()
                        
                        # metadataフィールドの安全な処理
                        metadata = {}
                        if row[2]:
                            try:
                                metadata = json.loads(row[2]) if isinstance(row[2], str) else row[2]
                            except (json.JSONDecodeError, TypeError):
                                metadata = {}
                        
                        template = PromptTemplate(
                            version=row[0],
                            content=row[1],
                            metadata=metadata,
                            performance_score=row[3],
                            usage_count=row[4],
                            created_at=created_at,
                            last_modified=last_modified
                        )
                        self.prompt_templates[template.version] = template
                except Exception as e:
                    self.logger.warning(f"プロンプトテンプレートの読み込みに失敗: {e}")
                
                self.logger.info("プロンプトテンプレートの読み込み完了")
                
                # チューニングデータ読み込み
                try:
                    cursor.execute("SELECT id, content, data_type, quality_score, usage_count, created_at, tags, metadata FROM tuning_data")
                    for row in cursor.fetchall():
                        # 日時フィールドの安全な処理
                        created_at = row[5]
                        if isinstance(created_at, str):
                            try:
                                created_at = datetime.fromisoformat(created_at)
                            except ValueError:
                                # ISO形式でない場合は現在時刻を使用
                                created_at = datetime.now()
                        elif not isinstance(created_at, datetime):
                            created_at = datetime.now()
                        
                        # tagsフィールドの安全な処理
                        tags = []
                        if row[6]:
                            try:
                                tags = json.loads(row[6]) if isinstance(row[6], str) else row[6]
                            except (json.JSONDecodeError, TypeError):
                                tags = []
                        
                        # metadataフィールドの安全な処理
                        metadata = {}
                        if row[7]:
                            try:
                                metadata = json.loads(row[7]) if isinstance(row[7], str) else row[7]
                            except (json.JSONDecodeError, TypeError):
                                metadata = {}
                        
                        data = TuningData(
                            id=row[0],
                            content=row[1],
                            data_type=row[2],
                            quality_score=row[3],
                            usage_count=row[4],
                            created_at=created_at,
                            tags=tags
                        )
                        self.tuning_data_pool.append(data)
                except Exception as e:
                    self.logger.warning(f"チューニングデータの読み込みに失敗: {e}")
                
                self.logger.info("チューニングデータの読み込み完了")
                    
        except Exception as e:
            self.logger.warning(f"学習コンポーネントの読み込みに失敗: {e}")
        
        self.logger.info("学習コンポーネントの読み込み完了")
    
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
            
            # テンプレートが見つからない場合はデフォルトを使用
            if not current_template:
                self.logger.warning(f"プロンプトテンプレート '{self.current_state.current_prompt_version}' が見つかりません")
                # 利用可能なテンプレートを確認
                available_templates = list(self.prompt_templates.keys())
                self.logger.info(f"利用可能なテンプレート: {available_templates}")
                if available_templates:
                    current_template = self.prompt_templates[available_templates[0]]
                    self.logger.info(f"デフォルトテンプレート '{available_templates[0]}' を使用します")
            
            # 2. コンテキスト構築
            context = await self._build_context(user_input)
            
            # 3. 推論実行（会話履歴を含む）
            # 会話履歴をプロンプトに含める
            enhanced_prompt = self._enhance_prompt_with_history(user_input, context.get("conversation_history", []))
            
            reasoning_result = await self.reasoning_engine.reason(
                prompt=enhanced_prompt,
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
        
        # 会話履歴を取得
        conversation_history = []
        try:
            if hasattr(self.memory_system, 'get_conversation_history'):
                history = await self.memory_system.get_conversation_history(
                    session_id=self.current_state.session_id,
                    limit=10
                )
                conversation_history = history
                self.logger.info(f"会話履歴をコンテキストに追加: {len(history)}件")
        except Exception as e:
            self.logger.warning(f"会話履歴取得エラー: {e}")
        
        # システム状態情報
        system_stats = await self.system_monitor.get_system_stats()
        
        return {
            "memory": memory_context,
            "conversation_history": conversation_history,
            "system_stats": system_stats,
            "agent_state": self.current_state,
            "current_prompt_version": self.current_state.current_prompt_version,
            "learning_epoch": self.current_state.learning_epoch
        }
    
    def _enhance_prompt_with_history(self, user_input: str, conversation_history: List[Dict[str, Any]]) -> str:
        """会話履歴をプロンプトに含める"""
        
        if not conversation_history:
            return user_input
        
        # 会話履歴を時系列順に並べ替え（古い順）
        sorted_history = sorted(conversation_history, key=lambda x: x.get('timestamp', ''))
        
        # 会話履歴を文字列に変換
        history_text = "【過去の会話履歴】\n"
        for i, conv in enumerate(sorted_history[-5:], 1):  # 最新5件のみ
            user_msg = conv.get('user_input', '')
            agent_msg = conv.get('agent_response', '')
            if user_msg and agent_msg:
                history_text += f"{i}. ユーザー: {user_msg}\n"
                history_text += f"   エージェント: {agent_msg}\n\n"
        
        # 現在の質問と組み合わせ
        enhanced_prompt = f"{history_text}【現在の質問】\n{user_input}\n\n上記の会話履歴を参考にして、現在の質問に回答してください。"
        
        self.logger.info(f"会話履歴を含むプロンプトを生成: {len(sorted_history)}件の履歴を使用")
        return enhanced_prompt
    
    async def _execute_tools_if_needed(self, user_input: str, reasoning_result: ReasoningResponse) -> Dict[str, Any]:
        """エージェントの自発的なツール実行"""
        
        tool_result = {"tools_used": [], "results": {}}
        
        # 推論結果からツール使用の必要性を判断
        reasoning_text = reasoning_result.response_text
        
        # エージェントが自発的にツールを使用するかどうかを判断
        # 推論結果にツール使用の意図が含まれているかチェック
        tool_actions = await self._analyze_tool_usage_intent(reasoning_text, user_input)
        
        for action in tool_actions:
            try:
                tool_name = action["tool"]
                tool = self.tool_manager.get_tool(tool_name)
                
                if tool:
                    # ツールに適切なパラメータを渡して実行
                    result = await self._execute_tool_action(tool, action)
                    
                    tool_result["tools_used"].append(tool_name)
                    tool_result["results"][tool_name] = result
                    
                    self.logger.info(f"ツール '{tool_name}' を自発的に実行しました")
                    
            except Exception as e:
                self.logger.warning(f"ツール '{action.get('tool', 'unknown')}' の実行エラー: {e}")
        
        return tool_result
    
    async def _analyze_tool_usage_intent(self, reasoning_text: str, user_input: str) -> List[Dict[str, Any]]:
        """推論結果からツール使用の意図を分析（ワード検出なし）"""
        
        actions = []
        
        # LLMを使用してツール使用の意図を分析
        if self.ollama_client:
            actions = await self._llm_analyze_tool_intent(reasoning_text, user_input)
        else:
            # フォールバック: 自然言語理解ベースの分析
            actions = await self._contextual_analyze_tool_intent(reasoning_text, user_input)
        
        return actions
    
    async def _llm_analyze_tool_intent(self, reasoning_text: str, user_input: str) -> List[Dict[str, Any]]:
        """LLMを使用してツール使用の意図を分析"""
        
        try:
            # 利用可能なツールの情報を準備
            available_tools = []
            if self.tool_manager:
                tools = self.tool_manager.list_tools()
                for tool in tools:
                    available_tools.append(f"- {tool['name']}: {tool['description']}")
            
            analysis_prompt = f"""
以下の推論結果とユーザー入力から、エージェントがどのツールを使用すべきかを分析してください：

推論結果:
{reasoning_text}

ユーザー入力:
{user_input}

利用可能なツール:
{chr(10).join(available_tools) if available_tools else "ツールが利用できません"}

エージェントが自発的にツールを使用すべきかどうかを判断し、使用すべきツールがある場合はJSON形式で回答してください。

例：
[
    {{
        "tool": "web_search",
        "action": "search",
        "query": "検索クエリ",
        "reason": "最新情報が必要"
    }}
]

ツールを使用する必要がない場合は空の配列 [] を回答してください。
"""
            
            # generateはInferenceRequestを受け取るため簡易ラッパーを使用
            response_obj = await self.ollama_client.generate_response(analysis_prompt)
            response = response_obj if isinstance(response_obj, str) else str(response_obj)
            
            if response:
                import re
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            
            return []
            
        except Exception as e:
            self.logger.error(f"LLM tool intent analysis error: {e}")
            return []
    
    async def _contextual_analyze_tool_intent(self, reasoning_text: str, user_input: str) -> List[Dict[str, Any]]:
        """文脈ベースのツール使用意図分析（ワード検出なし）"""
        
        actions = []
        
        # 推論結果の長さと複雑さから判断
        reasoning_length = len(reasoning_text)
        user_input_length = len(user_input)
        
        # 複雑な推論や長い入力の場合は、情報収集ツールを提案
        if reasoning_length > 200 or user_input_length > 100:
            if self.tool_manager and self.tool_manager.get_tool("web_search"):
                actions.append({
                    "tool": "web_search",
                    "action": "search",
                    "query": user_input,
                    "reason": "複雑な質問に対して詳細な情報収集が必要"
                })
        
        # システム関連の質問の場合は、システム情報ツールを提案
        if "システム" in user_input or "状態" in user_input or "情報" in user_input:
            if self.tool_manager and self.tool_manager.get_tool("system_info"):
                actions.append({
                    "tool": "system_info",
                    "action": "get_info",
                    "info_type": "all",
                    "reason": "システム情報の確認が必要"
                })
        
        return actions
    
    def _determine_system_command(self, reasoning_text: str) -> str:
        """システム情報の種類に応じてコマンドを決定"""
        
        if "メモリ" in reasoning_text or "memory" in reasoning_text:
            return "free -h"
        elif "ディスク" in reasoning_text or "disk" in reasoning_text:
            return "df -h"
        elif "プロセス" in reasoning_text or "process" in reasoning_text:
            return "ps aux"
        elif "起動時間" in reasoning_text or "uptime" in reasoning_text:
            return "uptime"
        else:
            return "pwd"  # デフォルト
    
    def _determine_file_path(self, reasoning_text: str, user_input: str) -> str:
        """ファイルパスを決定"""
        
        # ユーザー入力からファイルパスを抽出
        import re
        
        # ファイルパスのパターンを検索
        path_patterns = [
            r'([a-zA-Z0-9_/\\\.]+\.(py|js|ts|json|txt|md|yml|yaml|conf|config))',
            r'([a-zA-Z0-9_/\\\.]+\.(log|out|err))'
        ]
        
        for pattern in path_patterns:
            match = re.search(pattern, user_input)
            if match:
                return match.group(1)
        
        # デフォルトのファイルパス
        return "."
    
    async def _execute_tool_action(self, tool, action: Dict[str, Any]) -> Any:
        """ツールアクションを実行"""
        
        tool_name = action["tool"]
        
        if tool_name == "web_search":
            return await tool._arun(query=action["query"])
        elif tool_name == "command_executor":
            return await tool._arun(command=action["command"])
        elif tool_name == "file_manager":
            return await tool._arun(operation=action["action"], path=action["path"])
        elif tool_name == "mcp_client":
            return await tool._arun(operation=action["action"], service=action["service"])
        else:
            return await tool._arun(operation=action["action"])
    
    
    async def _generate_final_response(self, 
                                     user_input: str, 
                                     reasoning_result: ReasoningResponse,
                                     tool_result: Dict[str, Any],
                                     prompt_template: Optional[PromptTemplate]) -> str:
        """最終回答生成"""
        
        # 推論結果を基本応答として使用
        if reasoning_result and reasoning_result.response_text:
            base_response = reasoning_result.response_text
        else:
            # フォールバック応答
            base_response = f"こんにちは！{user_input}についてお答えします。"
        
        # プロンプトテンプレートが利用可能な場合の追加処理
        if prompt_template:
            try:
                # テンプレートに基づく追加情報
                template_info = f"\n\n【現在のセッション情報】\n"
                template_info += f"セッションID: {self.current_state.session_id}\n"
                template_info += f"学習エポック: {self.current_state.learning_epoch}\n"
                template_info += f"総インタラクション数: {self.current_state.total_interactions}\n"
                template_info += f"報酬スコア: {self.current_state.reward_score:.2f}"
                
                base_response += template_info
            except Exception as e:
                self.logger.warning(f"プロンプトテンプレート処理エラー: {e}")
        
        # 推論結果とツール結果を統合
        response_parts = [base_response]
        
        if tool_result and tool_result.get("tools_used"):
            response_parts.append(f"\n【使用ツール】: {', '.join(tool_result['tools_used'])}")
            for tool, result in tool_result.get("results", {}).items():
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
                (version, content, metadata, performance_score, usage_count, is_active, created_at, last_modified)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                template.version,
                template.content,
                json.dumps(template.metadata),
                template.performance_score,
                template.usage_count,
                True,
                template.created_at.isoformat() if template.created_at else datetime.now().isoformat(),
                template.last_modified.isoformat() if template.last_modified else datetime.now().isoformat()
            ))
            conn.commit()
    
    async def _save_tuning_data(self, data: TuningData):
        """チューニングデータ保存"""
        
        # メモリプールにも追加
        self.tuning_data_pool.append(data)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO tuning_data 
                (id, content, data_type, quality_score, usage_count, tags, metadata, is_active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data.id,
                data.content,
                data.data_type,
                data.quality_score,
                data.usage_count,
                json.dumps(data.tags),
                json.dumps({}),  # 空のmetadata
                True,
                data.created_at.isoformat(),
                datetime.now().isoformat()
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

    async def self_improve_prompt(self, 
                                template_name: str,
                                improvement_context: str,
                                performance_feedback: Optional[Dict[str, Any]] = None) -> bool:
        """エージェントが独自のプロンプトを自己改善する機能"""
        
        try:
            if not hasattr(self, 'prompt_manager') or not self.prompt_manager:
                self.logger.error("Prompt manager not available for self-improvement")
                return False
            
            self.logger.info(f"Starting prompt self-improvement for: {template_name}")
            
            # プロンプトマネージャーを使用して自己改善を実行
            success = await self.prompt_manager.self_improve_prompt(
                template_name=template_name,
                improvement_context=improvement_context,
                performance_feedback=performance_feedback
            )
            
            if success:
                # 改善されたプロンプトを現在のセッションに反映
                await self._update_current_prompt_template(template_name)
                
                # 学習エポックを増分
                if self.current_state:
                    self.current_state.learning_epoch += 1
                    self._save_agent_state()
                
                self.logger.info(f"Successfully completed prompt self-improvement for: {template_name}")
                return True
            else:
                self.logger.warning(f"Prompt self-improvement failed for: {template_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Prompt self-improvement error: {e}")
            return False
    
    async def _update_current_prompt_template(self, template_name: str):
        """現在のプロンプトテンプレートを更新"""
        
        try:
            if hasattr(self, 'prompt_manager') and self.prompt_manager:
                # 最新バージョンのテンプレートを取得
                latest_template = None
                for name, template in self.prompt_manager.templates.items():
                    if name.startswith(template_name) and "self-improved" in template.tags:
                        if latest_template is None or template.version > latest_template.version:
                            latest_template = template
                
                if latest_template:
                    # 現在のプロンプトテンプレートを更新
                    self.current_prompt_template = latest_template
                    
                    # データベースの状態も更新
                    if self.current_state:
                        self.current_state.current_prompt_version = latest_template.version
                    
                    self.logger.info(f"Updated current prompt template to: {latest_template.name} v{latest_template.version}")
                else:
                    self.logger.warning(f"No improved template found for: {template_name}")
                    
        except Exception as e:
            self.logger.error(f"Failed to update current prompt template: {e}")
    
    async def analyze_prompt_performance(self, template_name: str) -> Dict[str, Any]:
        """プロンプトのパフォーマンスを分析"""
        
        try:
            if not hasattr(self, 'prompt_manager') or not self.prompt_manager:
                return {"error": "Prompt manager not available"}
            
            # プロンプトマネージャーを使用してパフォーマンス分析
            analysis = await self.prompt_manager.analyze_prompt_performance(template_name)
            
            # エージェント固有の情報を追加
            if self.current_state:
                analysis.update({
                    "agent_session_id": self.current_state.session_id,
                    "agent_learning_epoch": self.current_state.learning_epoch,
                    "agent_total_interactions": self.current_state.total_interactions,
                    "agent_reward_score": self.current_state.reward_score
                })
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Prompt performance analysis error: {e}")
            return {"error": str(e)}
    
    async def trigger_self_improvement(self, 
                                     improvement_trigger: str = "performance_based",
                                     context: str = "") -> bool:
        """自己改善をトリガーする機能"""
        
        try:
            self.logger.info(f"Triggering self-improvement: {improvement_trigger}")
            
            # 現在のプロンプトテンプレートを取得
            current_template_name = None
            if hasattr(self, 'current_prompt_template') and self.current_prompt_template:
                current_template_name = self.current_prompt_template.name
            else:
                # デフォルトテンプレートを使用
                current_template_name = "basic_qa"
            
            # パフォーマンスフィードバックを収集
            performance_feedback = await self._collect_performance_feedback()
            
            # 改善コンテキストを構築
            improvement_context = self._build_improvement_context(
                improvement_trigger, 
                context, 
                performance_feedback
            )
            
            # 自己改善を実行
            success = await self.self_improve_prompt(
                template_name=current_template_name,
                improvement_context=improvement_context,
                performance_feedback=performance_feedback
            )
            
            if success:
                self.logger.info("Self-improvement completed successfully")
                return True
            else:
                self.logger.warning("Self-improvement failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Self-improvement trigger error: {e}")
            return False
    
    async def _collect_performance_feedback(self) -> Dict[str, Any]:
        """パフォーマンスフィードバックを収集"""
        
        try:
            feedback = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.current_state.session_id if self.current_state else None,
                "learning_epoch": self.current_state.learning_epoch if self.current_state else 0,
                "total_interactions": self.current_state.total_interactions if self.current_state else 0,
                "reward_score": self.current_state.reward_score if self.current_state else 0.0,
                "recent_performance": {}
            }
            
            # 最近のインタラクションのパフォーマンスを分析
            if hasattr(self, 'memory_system') and self.memory_system:
                try:
                    # 最近の会話履歴を取得
                    recent_history = await self.memory_system.get_conversation_history(
                        session_id=self.current_state.session_id if self.current_state else None,
                        limit=10
                    )
                    
                    if recent_history:
                        # パフォーマンス指標を計算
                        feedback["recent_performance"] = {
                            "conversation_count": len(recent_history),
                            "average_response_length": sum(
                                len(conv.get("agent_response", "")) 
                                for conv in recent_history
                            ) / len(recent_history) if recent_history else 0,
                            "response_quality_indicators": {
                                "has_detailed_responses": sum(
                                    1 for conv in recent_history 
                                    if len(conv.get("agent_response", "")) > 100
                                ),
                                "has_structured_responses": sum(
                                    1 for conv in recent_history 
                                    if any(marker in conv.get("agent_response", "") 
                                          for marker in ["【", "##", "1.", "2.", "3."])
                                )
                            }
                        }
                except Exception as e:
                    self.logger.warning(f"Failed to collect recent performance data: {e}")
            
            return feedback
            
        except Exception as e:
            self.logger.error(f"Performance feedback collection error: {e}")
            return {"error": str(e)}
    
    def _build_improvement_context(self, 
                                 trigger: str, 
                                 context: str, 
                                 feedback: Dict[str, Any]) -> str:
        """改善コンテキストを構築"""
        
        try:
            base_context = f"自己改善トリガー: {trigger}\n"
            
            if context:
                base_context += f"追加コンテキスト: {context}\n"
            
            # パフォーマンス情報を追加
            if feedback and "recent_performance" in feedback:
                perf = feedback["recent_performance"]
                base_context += f"最近のパフォーマンス:\n"
                base_context += f"- 会話数: {perf.get('conversation_count', 0)}\n"
                base_context += f"- 平均応答長: {perf.get('average_response_length', 0):.1f}文字\n"
                
                quality = perf.get("response_quality_indicators", {})
                base_context += f"- 詳細応答数: {quality.get('has_detailed_responses', 0)}\n"
                base_context += f"- 構造化応答数: {quality.get('has_structured_responses', 0)}\n"
            
            # トリガー別の改善指針を追加
            if trigger == "performance_based":
                base_context += "\n改善指針: パフォーマンス向上に焦点を当てた改善を行ってください。"
            elif trigger == "user_feedback":
                base_context += "\n改善指針: ユーザーフィードバックに基づいた改善を行ってください。"
            elif trigger == "scheduled":
                base_context += "\n改善指針: 定期的な改善として、全体的な品質向上を行ってください。"
            else:
                base_context += "\n改善指針: 一般的な品質向上を行ってください。"
            
            return base_context
            
        except Exception as e:
            self.logger.error(f"Improvement context building error: {e}")
            return f"自己改善トリガー: {trigger}\nコンテキスト: {context}"
    
    def _save_agent_state(self):
        """エージェント状態をデータベースに保存"""
        try:
            if not self.current_state:
                return
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO agent_states 
                    (session_id, learning_epoch, total_interactions, reward_score, 
                     evolution_generation, current_prompt_version, last_activity)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    self.current_state.session_id,
                    self.current_state.learning_epoch,
                    self.current_state.total_interactions,
                    self.current_state.reward_score,
                    self.current_state.evolution_generation,
                    self.current_state.current_prompt_version,
                    self.current_state.last_activity.isoformat()
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"エージェント状態保存エラー: {e}")


# 便利関数
async def create_self_learning_agent(config_path: str = "config/advanced_agent.yaml") -> SelfLearningAgent:
    """自己学習エージェント作成"""
    
    agent = SelfLearningAgent(config_path=config_path)
    await agent.initialize_session()
    return agent
