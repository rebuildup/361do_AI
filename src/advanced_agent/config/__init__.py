"""
Configuration module for self-learning AI agent
自己学習AIエージェント用設定モジュール
"""

from .settings import (
    AgentConfig,
    DatabaseConfig,
    OllamaConfig,
    MemoryConfig,
    LearningConfig,
    EvolutionConfig,
    MonitoringConfig,
    UIConfig,
    get_agent_config
)

from .loader import ConfigLoader, load_config_from_file

__all__ = [
    "AgentConfig",
    "DatabaseConfig", 
    "OllamaConfig",
    "MemoryConfig",
    "LearningConfig",
    "EvolutionConfig",
    "MonitoringConfig",
    "UIConfig",
    "get_agent_config",
    "ConfigLoader",
    "load_config_from_file"
]
