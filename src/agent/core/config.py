"""
Configuration Management
アプリケーション設定管理
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """環境変数ベースの設定"""
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AGENT_", extra="ignore")

    # OLLAMA settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2:7b-instruct"

    # Database
    database_url: str = "sqlite:///data/agent.db"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Learning settings
    learning_enabled: bool = True
    learning_interval_minutes: int = 30
    quality_threshold: float = 0.8

    # Security
    secret_key: str = "your-secret-key-change-this"

    # Logging
    log_level: str = "INFO"

    # Whether self-edits are auto-applied or saved as proposals
    auto_apply_self_edits: bool = False

    # Web search flags
    enable_web_search: bool = False
    duckduckgo_enabled: bool = True
    
    # Command execution (security restricted)
    enable_command_execution: bool = False
    
    # Agent mode selection
    use_codex_agent: bool = False  # Use Codex-compatible agent instead of complex learning system

@dataclass
class PathConfig:
    """Paths configuration"""
    base_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

    prompts_dir: str = field(init=False)
    learning_data_dir: str = field(init=False)
    logs_dir: str = field(init=False)
    knowledge_base_dir: str = field(init=False)
    proposals_dir: str = field(init=False)

    custom_prompt_file: str = field(init=False)
    learning_data_file: str = field(init=False)
    log_file: str = field(init=False)

    def __post_init__(self):
        self.prompts_dir = os.path.join(self.base_dir, "prompts")
        self.learning_data_dir = os.path.join(self.base_dir, "learning_data")
        self.logs_dir = os.path.join(self.base_dir, "logs")
        self.knowledge_base_dir = os.path.join(self.base_dir, "knowledge_base")
        self.proposals_dir = os.path.join(self.base_dir, "proposals")

        self.custom_prompt_file = os.path.join(self.prompts_dir, "custom_prompt.txt")
        self.learning_data_file = os.path.join(self.learning_data_dir, "learning_data.jsonl")
        self.log_file = os.path.join(self.logs_dir, "agent.log")

        # Ensure directories exist
        os.makedirs(self.prompts_dir, exist_ok=True)
        os.makedirs(self.learning_data_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.knowledge_base_dir, exist_ok=True)
        os.makedirs(self.proposals_dir, exist_ok=True)

class LearningConfig(BaseSettings):
    """Learning subsystem configuration"""
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AGENT_LEARNING_", extra="ignore")

    auto_evaluation_enabled: bool = True
    min_quality_score_for_learning: float = 0.7
    max_conversations_per_learning_cycle: int = 100

    prompt_optimization_enabled: bool = True
    ab_test_duration_hours: int = 24
    min_samples_for_statistical_significance: int = 30

    knowledge_extraction_enabled: bool = True
    knowledge_confidence_threshold: float = 0.8
    max_knowledge_items_per_category: int = 1000

    safety_check_enabled: bool = True
    safety_threshold: float = 0.8
    harmful_content_detection: bool = True

class Config:
    """メイン設定クラス"""

    def __init__(self):
        self.settings = Settings()
        self.paths = PathConfig()
        self.learning = LearningConfig()

        # Docker環境での調整
        if os.getenv("DOCKER_CONTAINER"):
            self.settings.database_url = "sqlite:////app/data/agent.db"
            self.paths.base_dir = "/app/data"
            self.paths.__post_init__() # パスを再構築

    @property
    def database_url(self) -> str:
        """データベースURL取得"""
        return self.settings.database_url

    @property
    def ollama_config(self) -> dict:
        """OLLAMA設定取得"""
        return {
            "base_url": self.settings.ollama_base_url,
            "model": self.settings.ollama_model
        }

    @property
    def is_learning_enabled(self) -> bool:
        """学習機能が有効かどうか"""
        return self.settings.learning_enabled and self.learning.auto_evaluation_enabled and not self.settings.use_codex_agent
    
    @property
    def is_codex_agent_enabled(self) -> bool:
        """Codex互換エージェントが有効かどうか"""
        return self.settings.use_codex_agent

    def get_log_config(self) -> dict:
        """ログ設定取得"""
        return {
            "level": self.settings.log_level,
            "file": self.paths.log_file
        }

    def validate_config(self) -> list:
        """設定の妥当性チェック"""
        errors = []

        if not self.settings.ollama_base_url:
            errors.append("OLLAMA base URL is required")

        if not self.settings.database_url:
            errors.append("Database URL is required")

        if not (0 <= self.learning.min_quality_score_for_learning <= 1):
            errors.append("Quality score threshold must be between 0 and 1")

        if self.settings.secret_key == "your-secret-key-change-this":
            errors.append("Secret key must be changed from default value")

        return errors

    def to_dict(self) -> dict:
        """設定を辞書形式で取得（デバッグ用）"""
        return {
            "settings": self.settings.model_dump(),
            "paths": self.paths.__dict__,
            "learning": self.learning.model_dump()
        }