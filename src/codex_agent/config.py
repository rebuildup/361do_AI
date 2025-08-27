"""
Codex Compatible Configuration Management
Codexの設定システムをベースにしたOLLAMA対応設定管理
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class ModelProviderInfo:
    """モデルプロバイダー情報 (Codex ModelProviderInfo相当)"""
    base_url: Optional[str] = None
    wire_api: str = "chat"  # "chat" or "completions"
    api_key: Optional[str] = None
    timeout: int = 30


@dataclass
class CodexConfig:
    """
    Codex互換設定管理クラス
    Rust版のConfigを参考にしたOLLAMA対応設定
    """
    
    # モデル設定 (Codex Config相当)
    model: str = "qwen2:7b-instruct"
    model_provider_id: str = "ollama"
    model_context_window: Optional[int] = None
    model_max_output_tokens: Optional[int] = 1000
    
    # OLLAMA設定
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: int = 30
    
    # GPU設定
    gpu_enabled: bool = True
    gpu_memory_fraction: float = 0.8  # GPUメモリの使用割合
    gpu_layers: Optional[int] = None  # GPU層数（自動設定の場合はNone）
    parallel_requests: int = 4  # 並列リクエスト数
    
    # 作業ディレクトリ
    cwd: Path = field(default_factory=Path.cwd)
    
    # エージェント設定
    hide_agent_reasoning: bool = False
    disable_response_storage: bool = False
    
    # プロバイダー情報
    model_providers: Dict[str, ModelProviderInfo] = field(default_factory=dict)
    
    def __post_init__(self):
        """初期化後の設定処理"""
        # 環境変数からの設定読み込み
        self._load_from_env()
        
        # デフォルトプロバイダーの設定
        if not self.model_providers:
            self._setup_default_providers()
    
    def _load_from_env(self):
        """環境変数から設定を読み込み (Codex load_config相当)"""
        # OLLAMA設定
        if ollama_url := os.getenv("OLLAMA_BASE_URL"):
            self.ollama_base_url = ollama_url
        
        if ollama_model := os.getenv("OLLAMA_MODEL"):
            self.model = ollama_model
        
        if timeout := os.getenv("OLLAMA_TIMEOUT"):
            try:
                self.ollama_timeout = int(timeout)
            except ValueError:
                pass
        
        # GPU設定
        if gpu_enabled := os.getenv("OLLAMA_GPU_ENABLED"):
            self.gpu_enabled = gpu_enabled.lower() in ("true", "1", "yes")
        
        if gpu_memory := os.getenv("OLLAMA_GPU_MEMORY_FRACTION"):
            try:
                self.gpu_memory_fraction = float(gpu_memory)
            except ValueError:
                pass
        
        if gpu_layers := os.getenv("OLLAMA_GPU_LAYERS"):
            try:
                self.gpu_layers = int(gpu_layers)
            except ValueError:
                pass
        
        if parallel := os.getenv("OLLAMA_PARALLEL_REQUESTS"):
            try:
                self.parallel_requests = int(parallel)
            except ValueError:
                pass
        
        # 作業ディレクトリ
        if cwd := os.getenv("CODEX_CWD"):
            self.cwd = Path(cwd)
    
    def _setup_default_providers(self):
        """デフォルトプロバイダーの設定 (Codex built_in_model_providers相当)"""
        self.model_providers = {
            "ollama": ModelProviderInfo(
                base_url=self.ollama_base_url,
                wire_api="chat",
                timeout=self.ollama_timeout
            )
        }
    
    @property
    def model_provider(self) -> ModelProviderInfo:
        """現在のモデルプロバイダー情報を取得"""
        return self.model_providers.get(
            self.model_provider_id,
            self.model_providers["ollama"]
        )
    
    def get_ollama_config(self) -> Dict[str, Any]:
        """OLLAMA設定を辞書形式で取得"""
        config = {
            "base_url": self.ollama_base_url,
            "model": self.model,
            "timeout": self.ollama_timeout,
            "max_tokens": self.model_max_output_tokens
        }
        
        # GPU設定を追加
        if self.gpu_enabled:
            config["gpu_enabled"] = True
            config["gpu_memory_fraction"] = self.gpu_memory_fraction
            if self.gpu_layers is not None:
                config["gpu_layers"] = self.gpu_layers
            config["parallel_requests"] = self.parallel_requests
        
        return config
    
    def validate(self) -> bool:
        """設定の妥当性チェック (Codex Config::validate相当)"""
        errors = []
        
        # 必須設定のチェック
        if not self.model:
            errors.append("Model name is required")
        
        if not self.ollama_base_url:
            errors.append("OLLAMA base URL is required")
        
        if self.model_provider_id not in self.model_providers:
            errors.append(f"Model provider '{self.model_provider_id}' not found")
        
        # 作業ディレクトリの存在チェック
        if not self.cwd.exists():
            errors.append(f"Working directory does not exist: {self.cwd}")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        return True
    
    @classmethod
    def load_from_file(cls, config_path: Optional[Path] = None) -> "CodexConfig":
        """
        設定ファイルから設定を読み込み
        将来的にTOMLファイル対応を追加予定
        """
        # 現在は環境変数ベースのみ実装
        config = cls()
        config.validate()
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書形式で出力 (デバッグ用)"""
        return {
            "model": self.model,
            "model_provider_id": self.model_provider_id,
            "ollama_base_url": self.ollama_base_url,
            "ollama_timeout": self.ollama_timeout,
            "cwd": str(self.cwd),
            "model_providers": {
                k: {
                    "base_url": v.base_url,
                    "wire_api": v.wire_api,
                    "timeout": v.timeout
                }
                for k, v in self.model_providers.items()
            }
        }