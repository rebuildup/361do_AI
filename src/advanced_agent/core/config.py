"""
設定管理モジュール
Pydantic Settings による統合設定管理
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from pydantic import Field, validator
from pydantic_settings import BaseSettings
import yaml


class GPUConfig(BaseSettings):
    """GPU設定"""
    max_vram_gb: float = Field(default=5.0, description="最大VRAM使用量(GB)")
    quantization_levels: List[int] = Field(default=[8, 4, 3], description="量子化レベル")
    temperature_threshold: int = Field(default=80, description="温度閾値(℃)")
    memory_threshold_percent: float = Field(default=90.0, description="メモリ使用率閾値(%)")
    offload_threshold_percent: float = Field(default=95.0, description="CPUオフロード閾値(%)")


class CPUConfig(BaseSettings):
    """CPU設定"""
    max_threads: int = Field(default=16, description="最大スレッド数")
    offload_threshold: float = Field(default=0.8, description="オフロード閾値")
    memory_limit_gb: float = Field(default=24.0, description="メモリ制限(GB)")


class MemoryConfig(BaseSettings):
    """メモリ設定"""
    system_ram_gb: float = Field(default=32.0, description="システムRAM(GB)")
    swap_limit_gb: float = Field(default=8.0, description="スワップ制限(GB)")
    cache_size_mb: int = Field(default=1024, description="キャッシュサイズ(MB)")


class ModelConfig(BaseSettings):
    """モデル設定"""
    primary: str = Field(default="deepseek-r1:7b", description="プライマリモデル")
    fallback: str = Field(default="qwen2.5:7b-instruct-q4_k_m", description="フォールバックモデル")
    emergency: str = Field(default="qwen2:1.5b-instruct-q4_k_m", description="緊急時モデル")
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama ベースURL")
    context_length: int = Field(default=4096, description="コンテキスト長")


class LearningConfig(BaseSettings):
    """学習設定"""
    adapter_pool_size: int = Field(default=10, description="アダプタプールサイズ")
    generation_size: int = Field(default=5, description="世代サイズ")
    mutation_rate: float = Field(default=0.1, description="変異率")
    crossover_rate: float = Field(default=0.7, description="交配率")
    learning_rate: float = Field(default=1e-4, description="学習率")
    batch_size: int = Field(default=4, description="バッチサイズ")
    max_epochs: int = Field(default=3, description="最大エポック数")


class MonitoringConfig(BaseSettings):
    """監視設定"""
    interval_seconds: float = Field(default=1.0, description="監視間隔(秒)")
    prometheus_port: int = Field(default=8000, description="Prometheusポート")
    enable_prometheus: bool = Field(default=True, description="Prometheus有効化")
    log_level: str = Field(default="INFO", description="ログレベル")
    metrics_retention_days: int = Field(default=7, description="メトリクス保持日数")


class DatabaseConfig(BaseSettings):
    """データベース設定"""
    sqlite_path: str = Field(default="data/agent_memory.db", description="SQLiteパス")
    chroma_path: str = Field(default="data/chroma_db", description="ChromaDBパス")
    backup_interval_hours: int = Field(default=24, description="バックアップ間隔(時間)")
    max_memory_items: int = Field(default=10000, description="最大記憶項目数")


class APIConfig(BaseSettings):
    """API設定"""
    host: str = Field(default="0.0.0.0", description="APIホスト")
    port: int = Field(default=8080, description="APIポート")
    workers: int = Field(default=1, description="ワーカー数")
    timeout_seconds: int = Field(default=300, description="タイムアウト(秒)")
    max_request_size_mb: int = Field(default=100, description="最大リクエストサイズ(MB)")


class UIConfig(BaseSettings):
    """UI設定"""
    streamlit_port: int = Field(default=8501, description="Streamlitポート")
    theme: str = Field(default="dark", description="テーマ")
    auto_refresh_seconds: int = Field(default=5, description="自動更新間隔(秒)")


class SecurityConfig(BaseSettings):
    """セキュリティ設定"""
    enable_auth: bool = Field(default=False, description="認証有効化")
    api_key: Optional[str] = Field(default=None, description="APIキー")
    rate_limit_per_minute: int = Field(default=60, description="レート制限(分)")
    encrypt_memory: bool = Field(default=False, description="記憶暗号化")


class AdvancedAgentConfig(BaseSettings):
    """統合設定クラス"""
    
    # 基本情報
    project_name: str = Field(default="Advanced Self-Learning Agent", description="プロジェクト名")
    version: str = Field(default="1.0.0", description="バージョン")
    environment: str = Field(default="development", description="環境")
    
    # 各種設定
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    cpu: CPUConfig = Field(default_factory=CPUConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        case_sensitive = False
    
    @validator("environment")
    def validate_environment(cls, v):
        allowed = ["development", "testing", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v
    
    @classmethod
    def load_from_yaml(cls, yaml_path: str) -> "AdvancedAgentConfig":
        """YAMLファイルから設定読み込み"""
        yaml_file = Path(yaml_path)
        if not yaml_file.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_file, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        
        return cls(**yaml_data)
    
    def save_to_yaml(self, yaml_path: str) -> None:
        """YAML ファイルに設定保存"""
        yaml_file = Path(yaml_path)
        yaml_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Pydantic モデルを辞書に変換
        config_dict = self.dict()
        
        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    def get_data_dir(self) -> Path:
        """データディレクトリパス取得"""
        return Path("data")
    
    def get_logs_dir(self) -> Path:
        """ログディレクトリパス取得"""
        return Path("logs")
    
    def get_config_dir(self) -> Path:
        """設定ディレクトリパス取得"""
        return Path("config")
    
    def ensure_directories(self) -> None:
        """必要なディレクトリ作成"""
        directories = [
            self.get_data_dir(),
            self.get_logs_dir(),
            self.get_config_dir(),
            Path(self.database.chroma_path).parent,
            Path(self.database.sqlite_path).parent
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def is_gpu_available(self) -> bool:
        """GPU利用可能性チェック"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def get_optimal_batch_size(self) -> int:
        """最適バッチサイズ計算"""
        if self.is_gpu_available():
            # GPU VRAM に基づく動的調整
            if self.gpu.max_vram_gb <= 4:
                return 1
            elif self.gpu.max_vram_gb <= 6:
                return 2
            elif self.gpu.max_vram_gb <= 8:
                return 4
            else:
                return 8
        else:
            return 1
    
    def get_memory_optimization_config(self) -> Dict[str, Any]:
        """メモリ最適化設定取得"""
        return {
            "gradient_checkpointing": True,
            "fp16": True,
            "dataloader_pin_memory": False,
            "dataloader_num_workers": min(4, self.cpu.max_threads // 4),
            "max_memory_mb": int(self.gpu.max_vram_gb * 1024 * 0.9)  # 90% 使用
        }


# グローバル設定インスタンス
_config: Optional[AdvancedAgentConfig] = None


def get_config() -> AdvancedAgentConfig:
    """グローバル設定取得"""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def load_config(config_path: Optional[str] = None) -> AdvancedAgentConfig:
    """設定読み込み"""
    global _config
    
    if config_path:
        _config = AdvancedAgentConfig.load_from_yaml(config_path)
    else:
        # デフォルト設定またはYAMLファイルから読み込み
        default_config_path = Path("config/advanced_agent.yaml")
        if default_config_path.exists():
            _config = AdvancedAgentConfig.load_from_yaml(str(default_config_path))
        else:
            _config = AdvancedAgentConfig()
    
    # 必要なディレクトリ作成
    _config.ensure_directories()
    
    return _config


def reload_config(config_path: Optional[str] = None) -> AdvancedAgentConfig:
    """設定再読み込み"""
    global _config
    _config = None
    return load_config(config_path)


# 使用例
if __name__ == "__main__":
    # デフォルト設定作成・保存
    config = AdvancedAgentConfig()
    config.save_to_yaml("config/advanced_agent.yaml")
    
    print("Configuration saved to config/advanced_agent.yaml")
    print(f"GPU available: {config.is_gpu_available()}")
    print(f"Optimal batch size: {config.get_optimal_batch_size()}")
    print(f"Memory optimization: {config.get_memory_optimization_config()}")