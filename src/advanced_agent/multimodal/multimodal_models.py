"""
Multimodal Processing Data Models

マルチモーダル処理のデータモデル定義
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from enum import Enum


class TaskType(Enum):
    """処理タスクタイプ"""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    IMAGE_PROCESSING = "image_processing"
    DOCUMENT_ANALYSIS = "document_analysis"
    MULTIMODAL_FUSION = "multimodal_fusion"
    BATCH_PROCESSING = "batch_processing"


class DeviceType(Enum):
    """デバイスタイプ"""
    GPU = "gpu"
    CPU = "cpu"
    AUTO = "auto"


class ProcessingStatus(Enum):
    """処理ステータス"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ResourceAllocation:
    """リソース配分情報"""
    device_type: DeviceType
    device_id: Optional[int] = None
    memory_allocated_mb: float = 0.0
    cpu_cores: int = 1
    gpu_memory_mb: float = 0.0
    estimated_processing_time: float = 0.0
    priority: int = 1  # 1=低, 5=高


@dataclass
class ProcessingTask:
    """処理タスク"""
    task_id: str
    task_type: TaskType
    input_data: Dict[str, Any]
    parameters: Dict[str, Any] = field(default_factory=dict)
    resource_requirements: Optional[ResourceAllocation] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingResult:
    """処理結果"""
    task_id: str
    status: ProcessingStatus
    output_data: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    resource_usage: Optional[ResourceAllocation] = None
    error_message: Optional[str] = None
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class BatchProcessingJob:
    """バッチ処理ジョブ"""
    job_id: str
    tasks: List[ProcessingTask]
    batch_size: int = 1
    parallel_workers: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    results: List[ProcessingResult] = field(default_factory=list)


@dataclass
class SystemResources:
    """システムリソース情報"""
    total_gpu_memory_mb: float
    available_gpu_memory_mb: float
    gpu_utilization_percent: float
    total_cpu_cores: int
    available_cpu_cores: int
    cpu_utilization_percent: float
    total_system_memory_mb: float
    available_system_memory_mb: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ModelInfo:
    """モデル情報"""
    model_name: str
    model_type: str
    device: str
    memory_usage_mb: float
    parameters_count: int
    quantization: Optional[str] = None
    loaded_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)


@dataclass
class PipelineConfig:
    """パイプライン設定"""
    pipeline_type: str
    model_name: str
    device: str = "auto"
    batch_size: int = 1
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    custom_parameters: Dict[str, Any] = field(default_factory=dict)