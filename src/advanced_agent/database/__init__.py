"""
Database module for self-learning AI agent
自己学習AIエージェント用データベースモジュール
"""

from .connection import DatabaseManager, get_database_manager, initialize_database, close_database
from .migrations import MigrationManager, initialize_database_with_migrations
from .models import (
    Base,
    AgentState,
    PromptTemplate,
    TuningData,
    EvolutionCandidate,
    EvolutionTuningData,
    RewardSignal,
    Interaction,
    LearningSession,
    SystemMetrics,
    Configuration
)

__all__ = [
    # Connection management
    'DatabaseManager',
    'get_database_manager',
    'initialize_database',
    'close_database',
    
    # Migration management
    'MigrationManager',
    'initialize_database_with_migrations',
    
    # Models
    'Base',
    'AgentState',
    'PromptTemplate',
    'TuningData',
    'EvolutionCandidate',
    'EvolutionTuningData',
    'RewardSignal',
    'Interaction',
    'LearningSession',
    'SystemMetrics',
    'Configuration'
]