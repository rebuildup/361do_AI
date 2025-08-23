"""
Configuration Management
アプリケーション設定管理
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Type, Any

Settings: Type[Any]  # set below depending on availability of pydantic_settings

try:
    from pydantic_settings import BaseSettings

    class PydanticSettings(BaseSettings):
        """環境変数ベースの設定 (pydantic_settings ベース)"""
        # OLLAMA設定
        ollama_base_url: str = "http://localhost:11434"
        ollama_model: str = "qwen2:7b-instruct"

        # データベース設定
        database_url: str = "sqlite:///data/agent.db"

        # API設定
        api_host: str = "0.0.0.0"
        api_port: int = 8000

        # 学習設定
        learning_enabled: bool = True
        learning_interval_minutes: int = 30
        quality_threshold: float = 0.8

        # セキュリティ設定
        secret_key: str = "your-secret-key-change-this"

        # ログ設定
        log_level: str = "INFO"
        # 検索API設定 (テスト実行の簡易化のためデフォルトは無効)
        enable_web_search: bool = False
        duckduckgo_enabled: bool = True

    # Webデザイン機能は削除

    class PydanticConfig:
        env_file = ".env"
        env_prefix = "AGENT_"

    Settings = PydanticSettings

except Exception:
    # フォールバック: pydantic_settings が無い環境向けの軽量実装
    from dataclasses import dataclass, asdict

    @dataclass
    class DataclassSettings:
        """環境変数ベースの設定 (フォールバック dataclass 実装)"""
        # OLLAMA設定
        ollama_base_url: str = "http://localhost:11434"
        ollama_model: str = "qwen2:7b-instruct"

        # データベース設定
        database_url: str = "sqlite:///data/agent.db"

        # API設定
        api_host: str = "0.0.0.0"
        api_port: int = 8000

        # 学習設定
        learning_enabled: bool = True
        learning_interval_minutes: int = 30
        quality_threshold: float = 0.8

        # セキュリティ設定
        secret_key: str = "your-secret-key-change-this"

        # ログ設定
        log_level: str = "INFO"

        # 検索API設定
        enable_web_search: bool = True
        duckduckgo_enabled: bool = True

    # Webデザイン機能は削除

        class Config:
            env_file = ".env"
            env_prefix = "AGENT_"

        def dict(self):
            return asdict(self)

    Settings = DataclassSettings


@dataclass
class PathConfig:
    """パス設定"""
    base_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))

    prompts_dir: str = field(init=False)
    learning_data_dir: str = field(init=False)
    logs_dir: str = field(init=False)
    knowledge_base_dir: str = field(init=False)

    custom_prompt_file: str = field(init=False)
    learning_data_file: str = field(init=False)
    log_file: str = field(init=False)

    def __post_init__(self):
        self.prompts_dir = os.path.join(self.base_dir, "prompts")
        self.learning_data_dir = os.path.join(self.base_dir, "learning_data")
        self.logs_dir = os.path.join(self.base_dir, "logs")
        self.knowledge_base_dir = os.path.join(self.base_dir, "knowledge_base")

        self.custom_prompt_file = os.path.join(self.prompts_dir, "custom_prompt.txt")
        self.learning_data_file = os.path.join(self.learning_data_dir, "learning_data.jsonl")
        self.log_file = os.path.join(self.logs_dir, "agent.log")

        # ディレクトリが存在しない場合は作成
        os.makedirs(self.prompts_dir, exist_ok=True)
        os.makedirs(self.learning_data_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.knowledge_base_dir, exist_ok=True)


@dataclass
class LearningConfig:
    """学習機能の詳細設定"""

    # 品質評価設定
    auto_evaluation_enabled: bool = True
    min_quality_score_for_learning: float = 0.7
    max_conversations_per_learning_cycle: int = 100

    # プロンプト最適化設定
    prompt_optimization_enabled: bool = True
    ab_test_duration_hours: int = 24
    min_samples_for_statistical_significance: int = 30

    # 知識抽出設定
    knowledge_extraction_enabled: bool = True
    knowledge_confidence_threshold: float = 0.8
    max_knowledge_items_per_category: int = 1000

    # セーフガード設定
    safety_check_enabled: bool = True
    safety_threshold: float = 0.8
    harmful_content_detection: bool = True


@dataclass
class WebDesignConfig:
    """Webデザイン機能の設定"""

    # デザイン生成設定
    max_design_options: int = 3
    default_responsive_breakpoints: Optional[dict] = None

    # コード生成設定
    generate_html: bool = True
    generate_css: bool = True
    generate_javascript: bool = True

    # 最適化設定
    optimize_performance: bool = True
    minify_code: bool = True
    accessibility_check: bool = True

    # プレビュー設定
    preview_enabled: bool = True
    screenshot_devices: Optional[list] = None

    def __post_init__(self):
        if self.default_responsive_breakpoints is None:
            self.default_responsive_breakpoints = {
                "mobile": "768px",
                "tablet": "1024px",
                "desktop": "1440px"
            }

        if self.screenshot_devices is None:
            self.screenshot_devices = [
                {"name": "desktop", "width": 1920, "height": 1080},
                {"name": "tablet", "width": 768, "height": 1024},
                {"name": "mobile", "width": 375, "height": 667}
            ]


class Config:
    """メイン設定クラス"""

    def __init__(self):
        self.settings = Settings()
        self.paths = PathConfig()
        self.learning = LearningConfig()

        # 環境変数から追加設定を読み込み
        self._load_environment_config()

    def _load_environment_config(self):
        """環境変数から追加設定を読み込み"""

        # Docker環境での調整
        if os.getenv("DOCKER_CONTAINER"):
            self.settings.database_url = "sqlite:////app/data/agent.db"
            self.paths.base_dir = "/app/data"
            self.paths.__post_init__() # パスを再構築

        # 開発環境での調整
        if os.getenv("DEVELOPMENT"):
            self.settings.log_level = "DEBUG"
            self.learning.ab_test_duration_hours = 1  # テスト用に短縮

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
        return self.settings.learning_enabled and self.learning.auto_evaluation_enabled

    @property
    def is_web_design_enabled(self) -> bool:
        """Webデザイン機能は削除されたため常に False を返す"""
        return False

    def get_log_config(self) -> dict:
        """ログ設定取得"""
        return {
            "level": self.settings.log_level,
            "file": self.paths.log_file
        }

    def validate_config(self) -> list:
        """設定の妥当性チェック"""
        errors = []

        # 必須設定のチェック
        if not self.settings.ollama_base_url:
            errors.append("OLLAMA base URL is required")

        if not self.settings.database_url:
            errors.append("Database URL is required")

        # 学習設定のチェック
        if self.learning.min_quality_score_for_learning < 0 or self.learning.min_quality_score_for_learning > 1:
            errors.append("Quality score threshold must be between 0 and 1")

        # セキュリティチェック
        if self.settings.secret_key == "your-secret-key-change-this":
            errors.append("Secret key must be changed from default value")

        return errors

    def to_dict(self) -> dict:
        """設定を辞書形式で取得（デバッグ用）"""
        return {
            "settings": self.settings.dict(),
            "paths": self.paths.__dict__,
            "learning": self.learning.__dict__
        }
