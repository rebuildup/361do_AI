"""
統合設定管理システム
Unified Configuration Management System

複数の設定ファイルを統合し、一元管理するシステム
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class UnifiedConfig(BaseSettings):
    """統合設定クラス"""
    
    # 基本設定
    name: str = Field(default="361do_AI", description="エージェント名")
    version: str = Field(default="1.0.0", description="バージョン")
    description: str = Field(default="361do_AI - 自己学習AIエージェント", description="説明")
    author: str = Field(default="361do_AI", description="作者")
    environment: str = Field(default="development", description="環境")
    
    # データベース設定
    database: Dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "sqlite",
            "path": "data/self_learning_agent.db",
            "echo": False,
            "pool_size": 5,
            "max_overflow": 10
        }
    )
    
    # Ollama設定
    ollama: Dict[str, Any] = Field(
        default_factory=lambda: {
            "base_url": "http://localhost:11434",
            "model": "qwen2:1.5b-instruct-q4_k_m",
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_tokens": 2048,
            "timeout": 30,
            "retry_attempts": 3,
            "retry_delay": 1.0
        }
    )
    
    # メモリ設定
    memory: Dict[str, Any] = Field(
        default_factory=lambda: {
            "backend": "chroma",
            "persist_directory": "data/memory",
            "collection_name": "agent_memory",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "max_memory_size": 10000,
            "similarity_threshold": 0.7,
            "cleanup_interval": 3600
        }
    )
    
    # 学習設定
    learning: Dict[str, Any] = Field(
        default_factory=lambda: {
            "prompt_mutation_rate": 0.1,
            "data_crossover_rate": 0.7,
            "evolution_generation_size": 5,
            "fitness_evaluation_interval": 100,
            "reward_decay_factor": 0.95,
            "learning_rate": 0.001,
            "batch_size": 32,
            "max_epochs": 10,
            "early_stopping_patience": 3,
            "validation_split": 0.2
        }
    )
    
    # 進化設定
    evolution: Dict[str, Any] = Field(
        default_factory=lambda: {
            "strategy": "genetic",
            "population_size": 20,
            "elite_size": 5,
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
            "selection_pressure": 2.0,
            "diversity_threshold": 0.1,
            "convergence_threshold": 0.01,
            "max_generations": 100,
            "fitness_weights": {
                "accuracy": 0.4,
                "efficiency": 0.3,
                "user_satisfaction": 0.3
            }
        }
    )
    
    # 監視設定
    monitoring: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "prometheus_port": 9090,
            "grafana_port": 3000,
            "metrics_interval": 60,
            "log_level": "INFO",
            "log_file": "logs/agent.log",
            "max_log_size": 104857600,  # 100MB
            "backup_count": 5,
            "gpu_monitoring": True,
            "memory_monitoring": True,
            "performance_monitoring": True,
            "alert_thresholds": {
                "gpu_usage": 0.9,
                "memory_usage": 0.8,
                "response_time": 10.0,
                "error_rate": 0.1
            }
        }
    )
    
    # UI設定
    ui: Dict[str, Any] = Field(
        default_factory=lambda: {
            "streamlit_port": 8501,
            "fastapi_port": 8000,
            "websocket_port": 8001,
            "host": "0.0.0.0",
            "debug": False,
            "auto_reload": False,
            "cors_origins": ["http://localhost", "http://localhost:3000"],
            "max_request_size": 10485760,  # 10MB
            "request_timeout": 30
        }
    )
    
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
    
    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8"
    )
    
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
        # ポート競合チェック
        ports = [
            self.monitoring.get("prometheus_port", 9090),
            self.monitoring.get("grafana_port", 3000),
            self.ui.get("streamlit_port", 8501),
            self.ui.get("fastapi_port", 8000),
            self.ui.get("websocket_port", 8001)
        ]
        
        if len(ports) != len(set(ports)):
            raise ValueError("ポート番号が重複しています")
        
        return self
    
    def get(self, key: str, default: Any = None) -> Any:
        """設定値取得（ドット記法対応）"""
        keys = key.split('.')
        value = self.model_dump()
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """設定値設定（ドット記法対応）"""
        keys = key.split('.')
        config_dict = self.model_dump()
        
        # ネストした辞書の更新
        current = config_dict
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
        
        # 新しい設定でオブジェクトを更新
        self.__dict__.update(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return self.model_dump()
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """ファイルに保存"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"設定を保存しました: {file_path}")
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'UnifiedConfig':
        """ファイルから読み込み"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"設定ファイルが見つかりません: {file_path}")
            return cls()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)


class UnifiedConfigManager:
    """統合設定管理システム"""
    
    def __init__(self, config_file: Union[str, Path] = "config/unified_config.yaml"):
        self.config_file = Path(config_file)
        self._config: Optional[UnifiedConfig] = None
        self._backup_dir = Path("config/backups")
        self._backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"統合設定管理システム初期化: {self.config_file}")
    
    def load_config(self) -> UnifiedConfig:
        """設定を読み込み"""
        if self._config is None:
            self._config = UnifiedConfig.load_from_file(self.config_file)
            logger.info("設定を読み込みました")
        
        return self._config
    
    def save_config(self, config: Optional[UnifiedConfig] = None) -> bool:
        """設定を保存"""
        try:
            if config:
                self._config = config
            
            if not self._config:
                logger.error("保存する設定がありません")
                return False
            
            # バックアップ作成
            self._create_backup()
            
            # 設定保存
            self._config.save_to_file(self.config_file)
            return True
            
        except Exception as e:
            logger.error(f"設定保存エラー: {e}")
            return False
    
    def get_config(self) -> UnifiedConfig:
        """現在の設定を取得"""
        if not self._config:
            return self.load_config()
        return self._config
    
    def update_config(self, **kwargs) -> bool:
        """設定を更新"""
        try:
            config = self.get_config()
            
            for key, value in kwargs.items():
                config.set(key, value)
            
            return self.save_config(config)
            
        except Exception as e:
            logger.error(f"設定更新エラー: {e}")
            return False
    
    def _create_backup(self) -> None:
        """設定のバックアップを作成"""
        if self.config_file.exists():
            from datetime import datetime
            backup_file = self._backup_dir / f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
            
            import shutil
            shutil.copy2(self.config_file, backup_file)
            logger.info(f"設定バックアップを作成: {backup_file}")
    
    def reset_to_defaults(self) -> bool:
        """設定をデフォルトにリセット"""
        try:
            # 現在の設定をバックアップ
            self._create_backup()
            
            # デフォルト設定を作成
            self._config = UnifiedConfig()
            return self.save_config()
            
        except Exception as e:
            logger.error(f"設定リセットエラー: {e}")
            return False
    
    def validate_config(self) -> bool:
        """設定の妥当性を検証"""
        try:
            config = self.get_config()
            config.validate_config()
            return True
        except Exception as e:
            logger.error(f"設定検証エラー: {e}")
            return False


# グローバル設定管理インスタンス
_config_manager: Optional[UnifiedConfigManager] = None


def get_config_manager() -> UnifiedConfigManager:
    """設定管理システムのシングルトンインスタンスを取得"""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = UnifiedConfigManager()
    
    return _config_manager


def get_config() -> UnifiedConfig:
    """現在の設定を取得"""
    return get_config_manager().get_config()


def update_config(**kwargs) -> bool:
    """設定を更新"""
    return get_config_manager().update_config(**kwargs)
