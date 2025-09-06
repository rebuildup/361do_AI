"""
Pydantic settings for self-learning AI agent
自己学習AIエージェント用Pydantic設定
"""

import os
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from pydantic import Field, field_validator, model_validator, ConfigDict
from pydantic_settings import BaseSettings
from enum import Enum


class LogLevel(str, Enum):
    """ログレベル"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseType(str, Enum):
    """データベースタイプ"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"


class MemoryBackend(str, Enum):
    """メモリバックエンド"""
    CHROMA = "chroma"
    HUGGINGFACE = "huggingface"
    SQLITE = "sqlite"


class EvolutionStrategy(str, Enum):
    """進化戦略"""
    GENETIC = "genetic"
    REINFORCEMENT = "reinforcement"
    HYBRID = "hybrid"


class DatabaseConfig(BaseSettings):
    """データベース設定"""
    
    type: DatabaseType = Field(default=DatabaseType.SQLITE, description="データベースタイプ")
    path: str = Field(default="data/self_learning_agent.db", description="データベースパス")
    host: Optional[str] = Field(default=None, description="データベースホスト")
    port: Optional[int] = Field(default=None, description="データベースポート")
    username: Optional[str] = Field(default=None, description="データベースユーザー名")
    password: Optional[str] = Field(default=None, description="データベースパスワード")
    database: Optional[str] = Field(default=None, description="データベース名")
    echo: bool = Field(default=False, description="SQLクエリログ出力")
    pool_size: int = Field(default=5, description="コネクションプールサイズ")
    max_overflow: int = Field(default=10, description="最大オーバーフロー接続数")
    
    model_config = ConfigDict(env_prefix="DB_")


class OllamaConfig(BaseSettings):
    """Ollama設定"""
    
    base_url: str = Field(default="http://localhost:11434", description="OllamaベースURL")
    model: str = Field(default="qwen2:7b-instruct", description="使用モデル")
    temperature: float = Field(default=0.7, description="温度パラメータ")
    top_p: float = Field(default=0.9, description="Top-pパラメータ")
    top_k: int = Field(default=40, description="Top-kパラメータ")
    max_tokens: int = Field(default=2048, description="最大トークン数")
    timeout: int = Field(default=30, description="タイムアウト（秒）")
    retry_attempts: int = Field(default=3, description="リトライ回数")
    retry_delay: float = Field(default=1.0, description="リトライ間隔（秒）")
    
    model_config = ConfigDict(env_prefix="OLLAMA_")


class MemoryConfig(BaseSettings):
    """メモリ設定"""
    
    backend: MemoryBackend = Field(default=MemoryBackend.CHROMA, description="メモリバックエンド")
    persist_directory: str = Field(default="data/memory", description="永続化ディレクトリ")
    collection_name: str = Field(default="agent_memory", description="コレクション名")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="埋め込みモデル")
    chunk_size: int = Field(default=1000, description="チャンクサイズ")
    chunk_overlap: int = Field(default=200, description="チャンクオーバーラップ")
    max_memory_size: int = Field(default=10000, description="最大メモリサイズ")
    similarity_threshold: float = Field(default=0.7, description="類似度閾値")
    cleanup_interval: int = Field(default=3600, description="クリーンアップ間隔（秒）")
    
    model_config = ConfigDict(env_prefix="MEMORY_")


class LearningConfig(BaseSettings):
    """学習設定"""
    
    prompt_mutation_rate: float = Field(default=0.1, description="プロンプト変異率")
    data_crossover_rate: float = Field(default=0.7, description="データ交配率")
    evolution_generation_size: int = Field(default=5, description="進化世代サイズ")
    fitness_evaluation_interval: int = Field(default=100, description="適応度評価間隔")
    reward_decay_factor: float = Field(default=0.95, description="報酬減衰係数")
    learning_rate: float = Field(default=0.001, description="学習率")
    batch_size: int = Field(default=32, description="バッチサイズ")
    max_epochs: int = Field(default=10, description="最大エポック数")
    early_stopping_patience: int = Field(default=3, description="早期停止パティエンス")
    validation_split: float = Field(default=0.2, description="検証分割率")
    
    model_config = ConfigDict(env_prefix="LEARNING_")


class EvolutionConfig(BaseSettings):
    """進化設定"""
    
    strategy: EvolutionStrategy = Field(default=EvolutionStrategy.GENETIC, description="進化戦略")
    population_size: int = Field(default=20, description="個体群サイズ")
    elite_size: int = Field(default=5, description="エリートサイズ")
    mutation_rate: float = Field(default=0.1, description="変異率")
    crossover_rate: float = Field(default=0.8, description="交配率")
    selection_pressure: float = Field(default=2.0, description="選択圧")
    diversity_threshold: float = Field(default=0.1, description="多様性閾値")
    convergence_threshold: float = Field(default=0.01, description="収束閾値")
    max_generations: int = Field(default=100, description="最大世代数")
    fitness_weights: Dict[str, float] = Field(
        default={
            "accuracy": 0.4,
            "efficiency": 0.3,
            "user_satisfaction": 0.3
        },
        description="適応度重み"
    )
    
    model_config = ConfigDict(env_prefix="EVOLUTION_")


class MonitoringConfig(BaseSettings):
    """監視設定"""
    
    enabled: bool = Field(default=True, description="監視有効化")
    prometheus_port: int = Field(default=9090, description="Prometheusポート")
    grafana_port: int = Field(default=3000, description="Grafanaポート")
    metrics_interval: int = Field(default=60, description="メトリクス収集間隔（秒）")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="ログレベル")
    log_file: str = Field(default="logs/agent.log", description="ログファイル")
    max_log_size: int = Field(default=100 * 1024 * 1024, description="最大ログサイズ（バイト）")
    backup_count: int = Field(default=5, description="ログバックアップ数")
    gpu_monitoring: bool = Field(default=True, description="GPU監視")
    memory_monitoring: bool = Field(default=True, description="メモリ監視")
    performance_monitoring: bool = Field(default=True, description="パフォーマンス監視")
    alert_thresholds: Dict[str, float] = Field(
        default={
            "gpu_usage": 0.9,
            "memory_usage": 0.8,
            "response_time": 10.0,
            "error_rate": 0.1
        },
        description="アラート閾値"
    )
    
    model_config = ConfigDict(env_prefix="MONITORING_")


class UIConfig(BaseSettings):
    """UI設定"""
    
    streamlit_port: int = Field(default=8501, description="Streamlitポート")
    fastapi_port: int = Field(default=8000, description="FastAPIポート")
    websocket_port: int = Field(default=8001, description="WebSocketポート")
    host: str = Field(default="localhost", description="ホスト")
    debug: bool = Field(default=False, description="デバッグモード")
    auto_reload: bool = Field(default=False, description="自動リロード")
    cors_origins: List[str] = Field(default=["*"], description="CORSオリジン")
    max_request_size: int = Field(default=10 * 1024 * 1024, description="最大リクエストサイズ（バイト）")
    request_timeout: int = Field(default=30, description="リクエストタイムアウト（秒）")
    
    model_config = ConfigDict(env_prefix="UI_")


class AgentConfig(BaseSettings):
    """エージェント設定"""
    
    # 基本設定
    name: str = Field(default="SelfLearningAgent", description="エージェント名")
    version: str = Field(default="1.0.0", description="バージョン")
    description: str = Field(default="自己学習AIエージェント", description="説明")
    author: str = Field(default="AI Agent", description="作者")
    
    # サブ設定
    database: DatabaseConfig = Field(default_factory=DatabaseConfig, description="データベース設定")
    ollama: OllamaConfig = Field(default_factory=OllamaConfig, description="Ollama設定")
    memory: MemoryConfig = Field(default_factory=MemoryConfig, description="メモリ設定")
    learning: LearningConfig = Field(default_factory=LearningConfig, description="学習設定")
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig, description="進化設定")
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig, description="監視設定")
    ui: UIConfig = Field(default_factory=UIConfig, description="UI設定")
    
    # システム設定
    data_dir: str = Field(default="data", description="データディレクトリ")
    logs_dir: str = Field(default="logs", description="ログディレクトリ")
    models_dir: str = Field(default="models", description="モデルディレクトリ")
    cache_dir: str = Field(default="cache", description="キャッシュディレクトリ")
    temp_dir: str = Field(default="temp", description="一時ディレクトリ")
    
    # パフォーマンス設定
    max_concurrent_requests: int = Field(default=10, description="最大同時リクエスト数")
    request_queue_size: int = Field(default=100, description="リクエストキューサイズ")
    worker_threads: int = Field(default=4, description="ワーカースレッド数")
    enable_caching: bool = Field(default=True, description="キャッシュ有効化")
    cache_ttl: int = Field(default=3600, description="キャッシュTTL（秒）")
    
    # セキュリティ設定
    enable_auth: bool = Field(default=False, description="認証有効化")
    secret_key: Optional[str] = Field(default=None, description="秘密鍵")
    token_expiry: int = Field(default=3600, description="トークン有効期限（秒）")
    rate_limit: int = Field(default=100, description="レート制限（リクエスト/分）")
    
    model_config = ConfigDict(env_prefix="AGENT_", case_sensitive=False)
    
    @field_validator('data_dir', 'logs_dir', 'models_dir', 'cache_dir', 'temp_dir')
    @classmethod
    def create_directories(cls, v):
        """ディレクトリ作成"""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    @field_validator('secret_key')
    @classmethod
    def validate_secret_key(cls, v, info):
        """秘密鍵検証"""
        if info.data.get('enable_auth', False) and not v:
            raise ValueError("認証が有効な場合、秘密鍵が必要です")
        return v
    
    @model_validator(mode='after')
    def validate_config(self):
        """設定全体検証"""
        # データベース設定検証
        if self.database.type != 'sqlite' and not all([
            self.database.host,
            self.database.port,
            self.database.username,
            self.database.password,
            self.database.database
        ]):
            raise ValueError("SQLite以外のデータベースの場合、接続情報が必要です")
        
        # ポート競合チェック
        ports = [
            self.monitoring.prometheus_port,
            self.monitoring.grafana_port,
            self.ui.streamlit_port,
            self.ui.fastapi_port,
            self.ui.websocket_port
        ]
        
        if len(ports) != len(set(ports)):
            raise ValueError("ポート番号が重複しています")
        
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return self.model_dump()
    
    def save_to_file(self, file_path: str):
        """ファイルに保存"""
        import yaml
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'AgentConfig':
        """ファイルから読み込み"""
        import yaml
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def get_data_dir(self) -> Path:
        """データディレクトリ取得"""
        return Path(self.data_dir)
    
    def get_logs_dir(self) -> Path:
        """ログディレクトリ取得"""
        return Path(self.logs_dir)
    
    @property
    def models(self) -> List[str]:
        """利用可能なモデル一覧"""
        return [self.ollama.model] if self.ollama.model else []


# グローバル設定インスタンス
_agent_config: Optional[AgentConfig] = None


def get_agent_config() -> AgentConfig:
    """エージェント設定取得（シングルトン）"""
    global _agent_config
    
    if _agent_config is None:
        _agent_config = AgentConfig()
    
    return _agent_config


def set_agent_config(config: AgentConfig):
    """エージェント設定設定"""
    global _agent_config
    _agent_config = config


def reset_agent_config():
    """エージェント設定リセット"""
    global _agent_config
    _agent_config = None
