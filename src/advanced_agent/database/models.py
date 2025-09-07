"""
SQLAlchemy models for self-learning AI agent
自己学習AIエージェント用SQLAlchemyモデル
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Text, 
    Boolean, ForeignKey, JSON, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON
import uuid

Base = declarative_base()


class AgentState(Base):
    """エージェント状態テーブル"""
    __tablename__ = 'agent_states'
    
    session_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(255), nullable=True, index=True)
    current_prompt_version = Column(String(50), nullable=False, default="1.0.0")
    learning_epoch = Column(Integer, nullable=False, default=0)
    total_interactions = Column(Integer, nullable=False, default=0)
    reward_score = Column(Float, nullable=False, default=0.0)
    evolution_generation = Column(Integer, nullable=False, default=0)
    last_activity = Column(DateTime, nullable=False, default=datetime.utcnow)
    performance_metrics = Column(SQLiteJSON, nullable=True, default=dict)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # リレーションシップ
    interactions = relationship("Interaction", back_populates="agent_state", cascade="all, delete-orphan")
    learning_sessions = relationship("LearningSession", back_populates="agent_state", cascade="all, delete-orphan")
    
    # インデックス
    __table_args__ = (
        Index('idx_agent_states_user_id', 'user_id'),
        Index('idx_agent_states_last_activity', 'last_activity'),
        Index('idx_agent_states_reward_score', 'reward_score'),
    )


class PromptTemplate(Base):
    """プロンプトテンプレートテーブル"""
    __tablename__ = 'prompt_templates'
    
    version = Column(String(50), primary_key=True)
    content = Column(Text, nullable=False)
    prompt_metadata = Column(SQLiteJSON, nullable=True, default=dict)
    performance_score = Column(Float, nullable=False, default=0.0)
    usage_count = Column(Integer, nullable=False, default=0)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_modified = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # リレーションシップ
    evolution_candidates = relationship("EvolutionCandidate", back_populates="prompt_template")
    
    # インデックス
    __table_args__ = (
        Index('idx_prompt_templates_performance', 'performance_score'),
        Index('idx_prompt_templates_usage', 'usage_count'),
        Index('idx_prompt_templates_active', 'is_active'),
    )


class TuningData(Base):
    """チューニングデータテーブル"""
    __tablename__ = 'tuning_data'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    content = Column(Text, nullable=False)
    data_type = Column(String(50), nullable=False)  # "conversation", "feedback", "correction", "example"
    quality_score = Column(Float, nullable=False, default=0.0)
    usage_count = Column(Integer, nullable=False, default=0)
    tags = Column(SQLiteJSON, nullable=True, default=list)
    data_metadata = Column(SQLiteJSON, nullable=True, default=dict)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # リレーションシップ
    evolution_candidates = relationship("EvolutionCandidate", secondary="evolution_tuning_data", back_populates="tuning_data")
    
    # インデックス
    __table_args__ = (
        Index('idx_tuning_data_type', 'data_type'),
        Index('idx_tuning_data_quality', 'quality_score'),
        Index('idx_tuning_data_active', 'is_active'),
    )


class EvolutionCandidate(Base):
    """進化候補テーブル"""
    __tablename__ = 'evolution_candidates'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    parent_ids = Column(SQLiteJSON, nullable=True, default=list)
    prompt_template_version = Column(String(50), ForeignKey('prompt_templates.version'), nullable=False)
    fitness_score = Column(Float, nullable=False, default=0.0)
    generation = Column(Integer, nullable=False, default=0)
    evaluation_metrics = Column(SQLiteJSON, nullable=True, default=dict)
    is_selected = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    evaluated_at = Column(DateTime, nullable=True)
    
    # リレーションシップ
    prompt_template = relationship("PromptTemplate", back_populates="evolution_candidates")
    tuning_data = relationship("TuningData", secondary="evolution_tuning_data", back_populates="evolution_candidates")
    
    # インデックス
    __table_args__ = (
        Index('idx_evolution_candidates_fitness', 'fitness_score'),
        Index('idx_evolution_candidates_generation', 'generation'),
        Index('idx_evolution_candidates_selected', 'is_selected'),
    )


# 進化候補とチューニングデータの多対多関係テーブル
class EvolutionTuningData(Base):
    """進化候補とチューニングデータの関連テーブル"""
    __tablename__ = 'evolution_tuning_data'
    
    evolution_candidate_id = Column(String(36), ForeignKey('evolution_candidates.id'), primary_key=True)
    tuning_data_id = Column(String(36), ForeignKey('tuning_data.id'), primary_key=True)
    weight = Column(Float, nullable=False, default=1.0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class RewardSignal(Base):
    """報酬信号テーブル"""
    __tablename__ = 'reward_signals'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    interaction_id = Column(String(36), ForeignKey('interactions.id'), nullable=False)
    reward_type = Column(String(50), nullable=False)  # "user_engagement", "task_completion", "quality", "efficiency"
    value = Column(Float, nullable=False)
    context = Column(SQLiteJSON, nullable=True, default=dict)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # リレーションシップ
    interaction = relationship("Interaction", back_populates="reward_signals")
    
    # インデックス
    __table_args__ = (
        Index('idx_reward_signals_type', 'reward_type'),
        Index('idx_reward_signals_value', 'value'),
        Index('idx_reward_signals_timestamp', 'timestamp'),
    )


class Interaction(Base):
    """インタラクションテーブル"""
    __tablename__ = 'interactions'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey('agent_states.session_id'), nullable=False)
    user_input = Column(Text, nullable=False)
    agent_response = Column(Text, nullable=False)
    reasoning_steps = Column(SQLiteJSON, nullable=True, default=list)
    tool_usage = Column(SQLiteJSON, nullable=True, default=dict)
    processing_time = Column(Float, nullable=True)
    quality_score = Column(Float, nullable=True)
    interaction_metadata = Column(SQLiteJSON, nullable=True, default=dict)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # リレーションシップ
    agent_state = relationship("AgentState", back_populates="interactions")
    reward_signals = relationship("RewardSignal", back_populates="interaction", cascade="all, delete-orphan")
    
    # インデックス
    __table_args__ = (
        Index('idx_interactions_session', 'session_id'),
        Index('idx_interactions_created', 'created_at'),
        Index('idx_interactions_quality', 'quality_score'),
    )


class LearningSession(Base):
    """学習セッションテーブル"""
    __tablename__ = 'learning_sessions'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey('agent_states.session_id'), nullable=False)
    session_type = Column(String(50), nullable=False)  # "evolution", "fine_tuning", "evaluation"
    status = Column(String(20), nullable=False, default="pending")  # "pending", "running", "completed", "failed"
    parameters = Column(SQLiteJSON, nullable=True, default=dict)
    results = Column(SQLiteJSON, nullable=True, default=dict)
    metrics = Column(SQLiteJSON, nullable=True, default=dict)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # リレーションシップ
    agent_state = relationship("AgentState", back_populates="learning_sessions")
    
    # インデックス
    __table_args__ = (
        Index('idx_learning_sessions_type', 'session_type'),
        Index('idx_learning_sessions_status', 'status'),
        Index('idx_learning_sessions_created', 'created_at'),
    )


class SystemMetrics(Base):
    """システムメトリクステーブル"""
    __tablename__ = 'system_metrics'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_type = Column(String(50), nullable=False)  # "gpu", "memory", "cpu", "performance"
    tags = Column(SQLiteJSON, nullable=True, default=dict)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # インデックス
    __table_args__ = (
        Index('idx_system_metrics_name', 'metric_name'),
        Index('idx_system_metrics_type', 'metric_type'),
        Index('idx_system_metrics_timestamp', 'timestamp'),
        UniqueConstraint('metric_name', 'timestamp', name='uq_metric_name_timestamp'),
    )


class Configuration(Base):
    """設定テーブル"""
    __tablename__ = 'configurations'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    config_key = Column(String(100), nullable=False, unique=True)
    config_value = Column(Text, nullable=False)
    config_type = Column(String(50), nullable=False)  # "string", "integer", "float", "boolean", "json"
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # インデックス
    __table_args__ = (
        Index('idx_configurations_key', 'config_key'),
        Index('idx_configurations_active', 'is_active'),
    )


class Conversation(Base):
    """会話記録テーブル"""
    __tablename__ = 'conversations'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey('agent_states.session_id'), nullable=False)
    user_input = Column(Text, nullable=False)
    agent_response = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    importance_score = Column(Float, nullable=False, default=0.5)
    response_time = Column(Float, nullable=True)  # 応答時間（秒）
    token_count = Column(Integer, nullable=False, default=0)
    conversation_metadata = Column(SQLiteJSON, nullable=True, default=dict)
    is_deleted = Column(Boolean, nullable=False, default=False)
    
    # リレーションシップ
    agent_state = relationship("AgentState", backref="conversations")
    
    # インデックス
    __table_args__ = (
        Index('idx_conversations_session', 'session_id'),
        Index('idx_conversations_timestamp', 'timestamp'),
        Index('idx_conversations_importance', 'importance_score'),
        Index('idx_conversations_deleted', 'is_deleted'),
    )