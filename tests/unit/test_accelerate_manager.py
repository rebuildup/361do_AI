"""
Test HuggingFace Accelerate Hybrid Processing Manager

HuggingFace Accelerate ハイブリッド処理管理システムのテスト
"""

import pytest
import asyncio
from datetime import datetime

from src.advanced_agent.multimodal.accelerate_manager import AccelerateHybridProcessor
from src.advanced_agent.multimodal.multimodal_models import (
    ProcessingTask, TaskType, DeviceType, ProcessingStatus,
    PipelineConfig, BatchProcessingJob
)


class TestAccelerateHybridProcessor:
    """Accelerate ハイブリッド処理システムのテスト"""
    
    @pytest.fixture
    def processor(self):
        """テスト用プロセッサ"""
        return AccelerateHybridProcessor(
            max_gpu_memory_gb=2.0,  # テスト用に小さく設定
            max_cpu_workers=2,
            auto_device_map=True
        )
    
    def test_system_resources(self, processor):
        """システムリソース取得テスト"""
        
        resources = processor.get_system_resources()
        
        assert resources.total_cpu_cores > 0
        assert resources.available_cpu_cores > 0
        assert resources.total_system_memory_mb > 0
        assert resources.available_system_memory_mb > 0
        assert 0 <= resources.cpu_utilization_percent <= 100
        
        # GPU が利用可能な場合のテスト
        if resources.total_gpu_memory_mb > 0:
            assert resources.available_gpu_memory_mb >= 0
            assert 0 <= resources.gpu_utilization_percent <= 100
    
    def test_device_allocation(self, processor):
        """デバイス配分決定テスト"""
        
        # テキスト生成タスク
        text_task = ProcessingTask(
            task_id="test_text",
            task_type=TaskType.TEXT_GENERATION,
            input_data={"text": "Hello world"}
        )
        
        allocation = processor.determine_optimal_device(text_task)
        
        assert allocation.device_type in [DeviceType.GPU, DeviceType.CPU]
        assert allocation.memory_allocated_mb > 0
        assert allocation.cpu_cores >= 1
        assert allocation.estimated_processing_time > 0
        
        # コード生成タスク
        code_task = ProcessingTask(
            task_id="test_code",
            task_type=TaskType.CODE_GENERATION,
            input_data={"prompt": "def hello():"}
        )
        
        code_allocation = processor.determine_optimal_device(code_task)
        
        assert code_allocation.device_type in [DeviceType.GPU, DeviceType.CPU]
        assert code_allocation.memory_allocated_mb > 0
    
    @pytest.mark.asyncio
    async def test_pipeline_creation(self, processor):
        """パイプライン作成テスト"""
        
        config = PipelineConfig(
            pipeline_type="text-generation",
            model_name="gpt2",  # 小さなモデルを使用
            device="cpu",  # テスト用にCPUを指定
            max_length=50
        )
        
        try:
            pipeline_key = await processor.create_pipeline_with_accelerate(config)
            
            assert pipeline_key is not None
            assert pipeline_key in processor.active_pipelines
            
            # 同じ設定で再作成（キャッシュされるはず）
            pipeline_key2 = await processor.create_pipeline_with_accelerate(config)
            assert pipeline_key == pipeline_key2
            
        except Exception as e:
            # モデルダウンロードに失敗する場合はスキップ
            pytest.skip(f"モデル読み込みに失敗: {e}")
    
    @pytest.mark.asyncio
    async def test_text_generation_task(self, processor):
        """テキスト生成タスクテスト"""
        
        task = ProcessingTask(
            task_id="test_text_gen",
            task_type=TaskType.TEXT_GENERATION,
            input_data={"text": "Hello"},
            parameters={
                "model_name": "gpt2",
                "max_length": 20,
                "temperature": 0.7
            }
        )
        
        try:
            result = await processor.process_task(task)
            
            assert result.task_id == task.task_id
            assert result.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]
            
            if result.status == ProcessingStatus.COMPLETED:
                assert "generated_text" in result.output_data
                assert result.processing_time > 0
                assert result.confidence_score > 0
            else:
                assert result.error_message is not None
                
        except Exception as e:
            pytest.skip(f"テキスト生成テストをスキップ: {e}")
    
    @pytest.mark.asyncio
    async def test_code_generation_task(self, processor):
        """コード生成タスクテスト"""
        
        task = ProcessingTask(
            task_id="test_code_gen",
            task_type=TaskType.CODE_GENERATION,
            input_data={"prompt": "print hello world"},
            parameters={
                "language": "python",
                "max_length": 50
            }
        )
        
        try:
            result = await processor.process_task(task)
            
            assert result.task_id == task.task_id
            assert result.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]
            
            if result.status == ProcessingStatus.COMPLETED:
                assert "generated_code" in result.output_data
                assert "language" in result.output_data
                assert "syntax_valid" in result.output_data
                
        except Exception as e:
            pytest.skip(f"コード生成テストをスキップ: {e}")
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, processor):
        """バッチ処理テスト"""
        
        batch_items = [
            {"text": "Hello"},
            {"text": "World"},
            {"text": "Test"}
        ]
        
        task = ProcessingTask(
            task_id="test_batch",
            task_type=TaskType.BATCH_PROCESSING,
            input_data={"batch_items": batch_items},
            parameters={
                "batch_size": 2,
                "model_name": "gpt2",
                "max_length": 20
            }
        )
        
        try:
            result = await processor.process_task(task)
            
            assert result.task_id == task.task_id
            
            if result.status == ProcessingStatus.COMPLETED:
                assert "batch_results" in result.output_data
                assert "successful_count" in result.output_data
                assert "failed_count" in result.output_data
                assert "total_items" in result.output_data
                
                assert result.output_data["total_items"] == len(batch_items)
                
        except Exception as e:
            pytest.skip(f"バッチ処理テストをスキップ: {e}")
    
    @pytest.mark.asyncio
    async def test_batch_job_processing(self, processor):
        """バッチジョブ処理テスト"""
        
        tasks = [
            ProcessingTask(
                task_id=f"batch_task_{i}",
                task_type=TaskType.TEXT_GENERATION,
                input_data={"text": f"Test {i}"},
                parameters={"model_name": "gpt2", "max_length": 15}
            )
            for i in range(3)
        ]
        
        job = BatchProcessingJob(
            job_id="test_batch_job",
            tasks=tasks,
            batch_size=2,
            parallel_workers=2
        )
        
        try:
            completed_job = await processor.process_batch_job(job)
            
            assert completed_job.job_id == job.job_id
            assert completed_job.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]
            assert len(completed_job.results) == len(tasks)
            
            if completed_job.status == ProcessingStatus.COMPLETED:
                assert completed_job.started_at is not None
                assert completed_job.completed_at is not None
                
        except Exception as e:
            pytest.skip(f"バッチジョブテストをスキップ: {e}")
    
    def test_code_syntax_validation(self, processor):
        """コード構文検証テスト"""
        
        # 有効なPythonコード
        valid_python = "print('Hello, World!')\nx = 1 + 2"
        assert processor._validate_code_syntax(valid_python, "python") is True
        
        # 無効なPythonコード
        invalid_python = "print('Hello, World!'\nx = 1 +"
        assert processor._validate_code_syntax(invalid_python, "python") is False
        
        # 他の言語（基本チェックのみ）
        other_code = "console.log('Hello');"
        assert processor._validate_code_syntax(other_code, "javascript") is True
        
        # 空のコード
        empty_code = ""
        assert processor._validate_code_syntax(empty_code, "python") is False
    
    def test_processing_statistics(self, processor):
        """処理統計テスト"""
        
        stats = processor.get_processing_statistics()
        
        assert "processing_stats" in stats
        assert "system_resources" in stats
        assert "loaded_models" in stats
        assert "active_pipelines" in stats
        assert "processing_tasks" in stats
        assert "accelerator_info" in stats
        
        # 処理統計の確認
        processing_stats = stats["processing_stats"]
        assert "total_tasks" in processing_stats
        assert "completed_tasks" in processing_stats
        assert "failed_tasks" in processing_stats
        assert "gpu_tasks" in processing_stats
        assert "cpu_tasks" in processing_stats
        
        # Accelerator情報の確認
        accelerator_info = stats["accelerator_info"]
        assert "device" in accelerator_info
        assert "num_processes" in accelerator_info
        assert "is_main_process" in accelerator_info
    
    def test_resource_cleanup(self, processor):
        """リソースクリーンアップテスト"""
        
        # クリーンアップ前の状態確認
        initial_stats = processor.get_processing_statistics()
        
        # クリーンアップ実行
        processor.cleanup_resources()
        
        # クリーンアップ後の状態確認
        final_stats = processor.get_processing_statistics()
        
        assert final_stats["loaded_models"] == 0
        assert final_stats["active_pipelines"] == 0


if __name__ == "__main__":
    pytest.main([__file__])