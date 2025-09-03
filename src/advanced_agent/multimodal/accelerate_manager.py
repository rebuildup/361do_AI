"""
HuggingFace Accelerate Hybrid Processing Manager

HuggingFace Accelerate による GPU/CPU ハイブリッド処理管理システム
"""

import uuid
import torch
import psutil
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path

from accelerate import Accelerator, PartialState
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM,
    pipeline, Pipeline
)
import numpy as np

from .multimodal_models import (
    ProcessingTask, ProcessingResult, BatchProcessingJob,
    TaskType, DeviceType, ProcessingStatus, ResourceAllocation,
    SystemResources, ModelInfo, PipelineConfig
)


class AccelerateHybridProcessor:
    """HuggingFace Accelerate による GPU/CPU ハイブリッド処理システム"""
    
    def __init__(self,
                 max_gpu_memory_gb: float = 5.0,
                 max_cpu_workers: int = 4,
                 auto_device_map: bool = True):
        
        self.max_gpu_memory_gb = max_gpu_memory_gb
        self.max_cpu_workers = max_cpu_workers
        self.auto_device_map = auto_device_map
        
        # Accelerate 初期化
        self.accelerator = Accelerator()
        self.partial_state = PartialState()
        
        # リソース管理
        self.loaded_models: Dict[str, ModelInfo] = {}
        self.active_pipelines: Dict[str, Pipeline] = {}
        self.task_queue: List[ProcessingTask] = []
        self.processing_tasks: Dict[str, ProcessingTask] = {}
        
        # 実行プール
        self.thread_executor = ThreadPoolExecutor(max_workers=max_cpu_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=max_cpu_workers)
        
        # 統計情報
        self.processing_stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "gpu_tasks": 0,
            "cpu_tasks": 0,
            "hybrid_tasks": 0,
            "total_processing_time": 0.0
        }
    
    def get_system_resources(self) -> SystemResources:
        """システムリソース情報取得"""
        
        # GPU情報
        gpu_memory_total = 0.0
        gpu_memory_available = 0.0
        gpu_utilization = 0.0
        
        if torch.cuda.is_available():
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                gpu_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_total = gpu_info.total / (1024**3)  # GB
                gpu_memory_available = gpu_info.free / (1024**3)
                
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization = gpu_util.gpu
                
            except ImportError:
                # pynvml が利用できない場合は torch で代替
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_memory_available = (torch.cuda.get_device_properties(0).total_memory - 
                                      torch.cuda.memory_allocated(0)) / (1024**3)
        
        # CPU情報
        cpu_info = psutil.cpu_percent(interval=None, percpu=False)
        cpu_cores = psutil.cpu_count()
        
        # メモリ情報
        memory_info = psutil.virtual_memory()
        
        return SystemResources(
            total_gpu_memory_mb=gpu_memory_total * 1024,
            available_gpu_memory_mb=gpu_memory_available * 1024,
            gpu_utilization_percent=gpu_utilization,
            total_cpu_cores=cpu_cores,
            available_cpu_cores=max(1, cpu_cores - 1),  # 1コアは予約
            cpu_utilization_percent=cpu_info,
            total_system_memory_mb=memory_info.total / (1024**2),
            available_system_memory_mb=memory_info.available / (1024**2)
        )
    
    def determine_optimal_device(self, 
                               task: ProcessingTask,
                               system_resources: Optional[SystemResources] = None) -> ResourceAllocation:
        """最適なデバイス配分の決定"""
        
        if system_resources is None:
            system_resources = self.get_system_resources()
        
        # タスクタイプ別の推奨リソース
        task_requirements = {
            TaskType.TEXT_GENERATION: {"gpu_memory": 2.0, "cpu_intensive": False},
            TaskType.CODE_GENERATION: {"gpu_memory": 3.0, "cpu_intensive": False},
            TaskType.IMAGE_PROCESSING: {"gpu_memory": 4.0, "cpu_intensive": True},
            TaskType.DOCUMENT_ANALYSIS: {"gpu_memory": 1.5, "cpu_intensive": True},
            TaskType.MULTIMODAL_FUSION: {"gpu_memory": 5.0, "cpu_intensive": False},
            TaskType.BATCH_PROCESSING: {"gpu_memory": 2.0, "cpu_intensive": True}
        }
        
        requirements = task_requirements.get(task.task_type, {"gpu_memory": 2.0, "cpu_intensive": False})
        required_gpu_memory = requirements["gpu_memory"] * 1024  # MB
        
        # デバイス選択ロジック
        if (system_resources.available_gpu_memory_mb >= required_gpu_memory and 
            system_resources.gpu_utilization_percent < 80):
            # GPU使用
            device_type = DeviceType.GPU
            device_id = 0
            memory_allocated = required_gpu_memory
            cpu_cores = 1
        elif (system_resources.available_cpu_cores >= 2 and 
              system_resources.cpu_utilization_percent < 70):
            # CPU使用
            device_type = DeviceType.CPU
            device_id = None
            memory_allocated = min(2048, system_resources.available_system_memory_mb * 0.3)
            cpu_cores = min(4, system_resources.available_cpu_cores)
        else:
            # ハイブリッド使用（CPU主体）
            device_type = DeviceType.CPU
            device_id = None
            memory_allocated = min(1024, system_resources.available_system_memory_mb * 0.2)
            cpu_cores = 2
        
        # 処理時間推定
        base_time = 1.0
        if device_type == DeviceType.GPU:
            estimated_time = base_time * 0.3  # GPU は高速
        else:
            estimated_time = base_time * (2.0 if requirements["cpu_intensive"] else 1.5)
        
        return ResourceAllocation(
            device_type=device_type,
            device_id=device_id,
            memory_allocated_mb=memory_allocated,
            cpu_cores=cpu_cores,
            gpu_memory_mb=required_gpu_memory if device_type == DeviceType.GPU else 0.0,
            estimated_processing_time=estimated_time,
            priority=task.metadata.get("priority", 1)
        )
    
    async def load_model_with_accelerate(self,
                                       model_name: str,
                                       model_type: str = "auto",
                                       device_allocation: Optional[ResourceAllocation] = None) -> str:
        """Accelerate を使用したモデル読み込み"""
        
        model_key = f"{model_name}_{model_type}"
        
        if model_key in self.loaded_models:
            # 既に読み込み済み
            self.loaded_models[model_key].last_used = datetime.now()
            return model_key
        
        try:
            # デバイス設定
            if device_allocation and device_allocation.device_type == DeviceType.GPU:
                device_map = "auto" if self.auto_device_map else {"": 0}
            else:
                device_map = {"": "cpu"}
            
            # モデル読み込み
            if model_type == "causal_lm":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map=device_map,
                    torch_dtype=torch.float16 if device_allocation and device_allocation.device_type == DeviceType.GPU else torch.float32,
                    low_cpu_mem_usage=True
                )
            else:
                model = AutoModel.from_pretrained(
                    model_name,
                    device_map=device_map,
                    torch_dtype=torch.float16 if device_allocation and device_allocation.device_type == DeviceType.GPU else torch.float32
                )
            
            # Accelerate で準備
            model = self.accelerator.prepare(model)
            
            # トークナイザー読み込み
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # モデル情報記録
            model_info = ModelInfo(
                model_name=model_name,
                model_type=model_type,
                device=str(model.device) if hasattr(model, 'device') else "unknown",
                memory_usage_mb=self._estimate_model_memory(model),
                parameters_count=sum(p.numel() for p in model.parameters()),
                quantization=None,
                loaded_at=datetime.now(),
                last_used=datetime.now()
            )
            
            self.loaded_models[model_key] = model_info
            
            return model_key
            
        except Exception as e:
            raise RuntimeError(f"モデル読み込みに失敗しました: {e}")
    
    def _estimate_model_memory(self, model) -> float:
        """モデルのメモリ使用量推定"""
        try:
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            return (param_size + buffer_size) / (1024**2)  # MB
        except Exception:
            return 0.0
    
    async def create_pipeline_with_accelerate(self,
                                            config: PipelineConfig,
                                            resource_allocation: Optional[ResourceAllocation] = None) -> str:
        """Accelerate を使用したパイプライン作成"""
        
        pipeline_key = f"{config.pipeline_type}_{config.model_name}_{config.device}"
        
        if pipeline_key in self.active_pipelines:
            return pipeline_key
        
        try:
            # デバイス設定
            if resource_allocation and resource_allocation.device_type == DeviceType.GPU:
                device = 0 if torch.cuda.is_available() else -1
            else:
                device = -1  # CPU
            
            # パイプライン作成
            pipe = pipeline(
                config.pipeline_type,
                model=config.model_name,
                device=device,
                torch_dtype=torch.float16 if device >= 0 else torch.float32,
                model_kwargs={
                    "low_cpu_mem_usage": True,
                    "device_map": "auto" if self.auto_device_map and device >= 0 else None
                }
            )
            
            self.active_pipelines[pipeline_key] = pipe
            
            return pipeline_key
            
        except Exception as e:
            raise RuntimeError(f"パイプライン作成に失敗しました: {e}")
    
    async def process_task(self, task: ProcessingTask) -> ProcessingResult:
        """単一タスクの処理"""
        
        task.started_at = datetime.now()
        task.status = ProcessingStatus.RUNNING
        self.processing_tasks[task.task_id] = task
        
        try:
            # リソース配分決定
            resource_allocation = self.determine_optimal_device(task)
            task.resource_requirements = resource_allocation
            
            # タスクタイプ別処理
            if task.task_type == TaskType.TEXT_GENERATION:
                result = await self._process_text_generation(task, resource_allocation)
            elif task.task_type == TaskType.CODE_GENERATION:
                result = await self._process_code_generation(task, resource_allocation)
            elif task.task_type == TaskType.BATCH_PROCESSING:
                result = await self._process_batch_task(task, resource_allocation)
            else:
                raise ValueError(f"未対応のタスクタイプ: {task.task_type}")
            
            # 処理完了
            task.completed_at = datetime.now()
            task.status = ProcessingStatus.COMPLETED
            
            result.processing_time = (task.completed_at - task.started_at).total_seconds()
            result.resource_usage = resource_allocation
            
            # 統計更新
            self.processing_stats["completed_tasks"] += 1
            self.processing_stats["total_processing_time"] += result.processing_time
            
            if resource_allocation.device_type == DeviceType.GPU:
                self.processing_stats["gpu_tasks"] += 1
            else:
                self.processing_stats["cpu_tasks"] += 1
            
            return result
            
        except Exception as e:
            # エラー処理
            task.status = ProcessingStatus.FAILED
            task.completed_at = datetime.now()
            
            self.processing_stats["failed_tasks"] += 1
            
            return ProcessingResult(
                task_id=task.task_id,
                status=ProcessingStatus.FAILED,
                error_message=str(e),
                processing_time=(task.completed_at - task.started_at).total_seconds() if task.started_at else 0.0
            )
        
        finally:
            # クリーンアップ
            if task.task_id in self.processing_tasks:
                del self.processing_tasks[task.task_id]
    
    async def _process_text_generation(self, 
                                     task: ProcessingTask,
                                     resource_allocation: ResourceAllocation) -> ProcessingResult:
        """テキスト生成処理"""
        
        input_text = task.input_data.get("text", "")
        max_length = task.parameters.get("max_length", 100)
        temperature = task.parameters.get("temperature", 0.7)
        
        # パイプライン設定
        config = PipelineConfig(
            pipeline_type="text-generation",
            model_name=task.parameters.get("model_name", "gpt2"),
            device="gpu" if resource_allocation.device_type == DeviceType.GPU else "cpu",
            max_length=max_length,
            temperature=temperature
        )
        
        # パイプライン作成・実行
        pipeline_key = await self.create_pipeline_with_accelerate(config, resource_allocation)
        pipe = self.active_pipelines[pipeline_key]
        
        # 非同期実行
        loop = asyncio.get_event_loop()
        
        def generate_text():
            return pipe(
                input_text,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=pipe.tokenizer.eos_token_id
            )
        
        if resource_allocation.device_type == DeviceType.CPU:
            # CPU処理は別スレッドで実行
            result = await loop.run_in_executor(self.thread_executor, generate_text)
        else:
            # GPU処理は直接実行
            result = generate_text()
        
        return ProcessingResult(
            task_id=task.task_id,
            status=ProcessingStatus.COMPLETED,
            output_data={
                "generated_text": result[0]["generated_text"] if result else "",
                "input_text": input_text
            },
            confidence_score=0.8,
            metadata={
                "model_name": config.model_name,
                "device_used": config.device,
                "parameters": task.parameters
            }
        )
    
    async def _process_code_generation(self,
                                     task: ProcessingTask,
                                     resource_allocation: ResourceAllocation) -> ProcessingResult:
        """コード生成処理"""
        
        prompt = task.input_data.get("prompt", "")
        language = task.parameters.get("language", "python")
        
        # コード生成用のプロンプト調整
        code_prompt = f"# {language} code\n{prompt}\n"
        
        # パイプライン設定
        config = PipelineConfig(
            pipeline_type="text-generation",
            model_name=task.parameters.get("model_name", "microsoft/CodeGPT-small-py"),
            device="gpu" if resource_allocation.device_type == DeviceType.GPU else "cpu",
            max_length=task.parameters.get("max_length", 200)
        )
        
        try:
            # パイプライン作成・実行
            pipeline_key = await self.create_pipeline_with_accelerate(config, resource_allocation)
            pipe = self.active_pipelines[pipeline_key]
            
            # 非同期実行
            loop = asyncio.get_event_loop()
            
            def generate_code():
                return pipe(
                    code_prompt,
                    max_length=config.max_length,
                    temperature=0.2,  # コード生成は低温度
                    do_sample=True,
                    pad_token_id=pipe.tokenizer.eos_token_id
                )
            
            if resource_allocation.device_type == DeviceType.CPU:
                result = await loop.run_in_executor(self.thread_executor, generate_code)
            else:
                result = generate_code()
            
            generated_code = result[0]["generated_text"] if result else ""
            
            # 簡単な構文チェック
            syntax_valid = self._validate_code_syntax(generated_code, language)
            
            return ProcessingResult(
                task_id=task.task_id,
                status=ProcessingStatus.COMPLETED,
                output_data={
                    "generated_code": generated_code,
                    "language": language,
                    "syntax_valid": syntax_valid,
                    "prompt": prompt
                },
                confidence_score=0.7 if syntax_valid else 0.4,
                metadata={
                    "model_name": config.model_name,
                    "device_used": config.device,
                    "language": language
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                task_id=task.task_id,
                status=ProcessingStatus.FAILED,
                error_message=f"コード生成エラー: {e}"
            )
    
    def _validate_code_syntax(self, code: str, language: str) -> bool:
        """簡単な構文チェック"""
        try:
            if language.lower() == "python":
                compile(code, "<string>", "exec")
                return True
            else:
                # 他の言語は基本的なチェックのみ
                return len(code.strip()) > 0 and not code.strip().startswith("Error")
        except SyntaxError:
            return False
        except Exception:
            return False
    
    async def _process_batch_task(self,
                                task: ProcessingTask,
                                resource_allocation: ResourceAllocation) -> ProcessingResult:
        """バッチ処理"""
        
        batch_items = task.input_data.get("batch_items", [])
        batch_size = task.parameters.get("batch_size", 4)
        
        results = []
        
        # バッチ処理実行
        for i in range(0, len(batch_items), batch_size):
            batch = batch_items[i:i + batch_size]
            
            # 並列処理
            batch_tasks = []
            for item in batch:
                sub_task = ProcessingTask(
                    task_id=f"{task.task_id}_batch_{i}_{len(batch_tasks)}",
                    task_type=TaskType.TEXT_GENERATION,
                    input_data=item,
                    parameters=task.parameters
                )
                batch_tasks.append(self.process_task(sub_task))
            
            # バッチ結果待機
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)
        
        # 成功・失敗カウント
        successful_results = [r for r in results if isinstance(r, ProcessingResult) and r.status == ProcessingStatus.COMPLETED]
        failed_results = [r for r in results if not isinstance(r, ProcessingResult) or r.status == ProcessingStatus.FAILED]
        
        return ProcessingResult(
            task_id=task.task_id,
            status=ProcessingStatus.COMPLETED,
            output_data={
                "batch_results": [r.output_data if isinstance(r, ProcessingResult) else {"error": str(r)} for r in results],
                "successful_count": len(successful_results),
                "failed_count": len(failed_results),
                "total_items": len(batch_items)
            },
            confidence_score=len(successful_results) / len(batch_items) if batch_items else 0.0,
            metadata={
                "batch_size": batch_size,
                "device_used": resource_allocation.device_type.value
            }
        )
    
    async def process_batch_job(self, job: BatchProcessingJob) -> BatchProcessingJob:
        """バッチジョブ処理"""
        
        job.started_at = datetime.now()
        job.status = ProcessingStatus.RUNNING
        
        try:
            # 並列処理数制限
            semaphore = asyncio.Semaphore(job.parallel_workers)
            
            async def process_with_semaphore(task):
                async with semaphore:
                    return await self.process_task(task)
            
            # 全タスクを並列実行
            tasks = [process_with_semaphore(task) for task in job.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 結果をジョブに格納
            job.results = [r if isinstance(r, ProcessingResult) else 
                          ProcessingResult(task_id="unknown", status=ProcessingStatus.FAILED, error_message=str(r))
                          for r in results]
            
            job.completed_at = datetime.now()
            job.status = ProcessingStatus.COMPLETED
            
        except Exception as e:
            job.status = ProcessingStatus.FAILED
            job.completed_at = datetime.now()
        
        return job
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """処理統計情報取得"""
        
        system_resources = self.get_system_resources()
        
        return {
            "processing_stats": self.processing_stats,
            "system_resources": {
                "gpu_memory_usage_percent": (
                    (system_resources.total_gpu_memory_mb - system_resources.available_gpu_memory_mb) /
                    system_resources.total_gpu_memory_mb * 100
                ) if system_resources.total_gpu_memory_mb > 0 else 0,
                "cpu_utilization_percent": system_resources.cpu_utilization_percent,
                "memory_usage_percent": (
                    (system_resources.total_system_memory_mb - system_resources.available_system_memory_mb) /
                    system_resources.total_system_memory_mb * 100
                )
            },
            "loaded_models": len(self.loaded_models),
            "active_pipelines": len(self.active_pipelines),
            "processing_tasks": len(self.processing_tasks),
            "accelerator_info": {
                "device": str(self.accelerator.device),
                "num_processes": self.accelerator.num_processes,
                "process_index": self.accelerator.process_index,
                "is_main_process": self.accelerator.is_main_process
            }
        }
    
    def cleanup_resources(self):
        """リソースクリーンアップ"""
        
        # パイプラインクリーンアップ
        for pipeline_key in list(self.active_pipelines.keys()):
            del self.active_pipelines[pipeline_key]
        
        # モデルクリーンアップ
        self.loaded_models.clear()
        
        # GPU メモリクリーンアップ
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 実行プール終了
        self.thread_executor.shutdown(wait=False)
        self.process_executor.shutdown(wait=False)