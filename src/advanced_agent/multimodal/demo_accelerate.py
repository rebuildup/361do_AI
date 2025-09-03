"""
Accelerate Hybrid Processing Demonstration

HuggingFace Accelerate ハイブリッド処理システムのデモンストレーション
"""

import asyncio
import time
from datetime import datetime
from pathlib import Path

from .accelerate_manager import AccelerateHybridProcessor
from .multimodal_models import (
    ProcessingTask, TaskType, PipelineConfig, BatchProcessingJob
)


async def demo_system_resources():
    """システムリソース監視のデモ"""
    
    print("=== HuggingFace Accelerate ハイブリッド処理システム デモ ===\n")
    
    processor = AccelerateHybridProcessor(
        max_gpu_memory_gb=4.0,
        max_cpu_workers=4,
        auto_device_map=True
    )
    
    try:
        # 1. システムリソース確認
        print("1. システムリソース情報")
        
        resources = processor.get_system_resources()
        
        print(f"   CPU情報:")
        print(f"     総コア数: {resources.total_cpu_cores}")
        print(f"     利用可能コア数: {resources.available_cpu_cores}")
        print(f"     CPU使用率: {resources.cpu_utilization_percent:.1f}%")
        
        print(f"   メモリ情報:")
        print(f"     総メモリ: {resources.total_system_memory_mb/1024:.1f} GB")
        print(f"     利用可能メモリ: {resources.available_system_memory_mb/1024:.1f} GB")
        
        print(f"   GPU情報:")
        if resources.total_gpu_memory_mb > 0:
            print(f"     総GPU メモリ: {resources.total_gpu_memory_mb/1024:.1f} GB")
            print(f"     利用可能GPU メモリ: {resources.available_gpu_memory_mb/1024:.1f} GB")
            print(f"     GPU使用率: {resources.gpu_utilization_percent:.1f}%")
        else:
            print(f"     GPU: 利用不可")
        
        print()
        
        # 2. デバイス配分テスト
        print("2. デバイス配分決定")
        
        test_tasks = [
            ("テキスト生成", TaskType.TEXT_GENERATION, {"text": "Hello world"}),
            ("コード生成", TaskType.CODE_GENERATION, {"prompt": "def hello():"}),
            ("画像処理", TaskType.IMAGE_PROCESSING, {"image_path": "test.jpg"}),
            ("バッチ処理", TaskType.BATCH_PROCESSING, {"batch_items": [1, 2, 3]})
        ]
        
        for task_name, task_type, input_data in test_tasks:
            task = ProcessingTask(
                task_id=f"test_{task_type.value}",
                task_type=task_type,
                input_data=input_data
            )
            
            allocation = processor.determine_optimal_device(task)
            
            print(f"   {task_name}:")
            print(f"     推奨デバイス: {allocation.device_type.value}")
            print(f"     メモリ配分: {allocation.memory_allocated_mb:.0f} MB")
            print(f"     CPU コア: {allocation.cpu_cores}")
            print(f"     推定処理時間: {allocation.estimated_processing_time:.2f} 秒")
        
        print()
        
        return processor
        
    except Exception as e:
        print(f"システムリソースデモエラー: {e}")
        return processor


async def demo_text_generation(processor):
    """テキスト生成のデモ"""
    
    print("3. テキスト生成デモ")
    
    try:
        # 軽量モデルを使用
        tasks = [
            {
                "name": "短文生成",
                "input": "The future of AI is",
                "params": {"max_length": 30, "temperature": 0.7}
            },
            {
                "name": "質問応答",
                "input": "What is machine learning?",
                "params": {"max_length": 50, "temperature": 0.5}
            },
            {
                "name": "創作文",
                "input": "Once upon a time",
                "params": {"max_length": 40, "temperature": 0.9}
            }
        ]
        
        for i, task_config in enumerate(tasks, 1):
            print(f"\n   テスト {i}: {task_config['name']}")
            print(f"     入力: {task_config['input']}")
            
            task = ProcessingTask(
                task_id=f"text_gen_{i}",
                task_type=TaskType.TEXT_GENERATION,
                input_data={"text": task_config["input"]},
                parameters={
                    "model_name": "gpt2",  # 軽量モデル
                    **task_config["params"]
                }
            )
            
            start_time = time.time()
            result = await processor.process_task(task)
            end_time = time.time()
            
            print(f"     処理時間: {end_time - start_time:.2f} 秒")
            print(f"     ステータス: {result.status.value}")
            
            if result.status.name == "COMPLETED":
                generated_text = result.output_data.get("generated_text", "")
                print(f"     生成結果: {generated_text[:100]}...")
                print(f"     信頼度: {result.confidence_score:.2f}")
                print(f"     使用デバイス: {result.metadata.get('device_used', 'unknown')}")
            else:
                print(f"     エラー: {result.error_message}")
        
        print()
        
    except Exception as e:
        print(f"テキスト生成デモエラー: {e}")


async def demo_code_generation(processor):
    """コード生成のデモ"""
    
    print("4. コード生成デモ")
    
    try:
        code_tasks = [
            {
                "name": "Hello World関数",
                "prompt": "def hello_world():",
                "language": "python"
            },
            {
                "name": "リスト処理",
                "prompt": "# Create a function to filter even numbers from a list\ndef filter_even(numbers):",
                "language": "python"
            },
            {
                "name": "クラス定義",
                "prompt": "class Calculator:",
                "language": "python"
            }
        ]
        
        for i, task_config in enumerate(code_tasks, 1):
            print(f"\n   コード生成 {i}: {task_config['name']}")
            print(f"     プロンプト: {task_config['prompt']}")
            
            task = ProcessingTask(
                task_id=f"code_gen_{i}",
                task_type=TaskType.CODE_GENERATION,
                input_data={"prompt": task_config["prompt"]},
                parameters={
                    "language": task_config["language"],
                    "max_length": 100
                }
            )
            
            start_time = time.time()
            result = await processor.process_task(task)
            end_time = time.time()
            
            print(f"     処理時間: {end_time - start_time:.2f} 秒")
            print(f"     ステータス: {result.status.value}")
            
            if result.status.name == "COMPLETED":
                generated_code = result.output_data.get("generated_code", "")
                syntax_valid = result.output_data.get("syntax_valid", False)
                
                print(f"     生成コード:")
                for line in generated_code.split('\n')[:5]:  # 最初の5行のみ表示
                    print(f"       {line}")
                if len(generated_code.split('\n')) > 5:
                    print(f"       ... (省略)")
                
                print(f"     構文チェック: {'✓' if syntax_valid else '✗'}")
                print(f"     信頼度: {result.confidence_score:.2f}")
            else:
                print(f"     エラー: {result.error_message}")
        
        print()
        
    except Exception as e:
        print(f"コード生成デモエラー: {e}")


async def demo_batch_processing(processor):
    """バッチ処理のデモ"""
    
    print("5. バッチ処理デモ")
    
    try:
        # バッチアイテム作成
        batch_items = [
            {"text": f"Generate a creative story about {topic}"}
            for topic in ["robots", "space", "ocean", "forest", "city"]
        ]
        
        print(f"   バッチサイズ: {len(batch_items)} アイテム")
        
        # バッチタスク作成
        batch_task = ProcessingTask(
            task_id="batch_demo",
            task_type=TaskType.BATCH_PROCESSING,
            input_data={"batch_items": batch_items},
            parameters={
                "batch_size": 2,
                "model_name": "gpt2",
                "max_length": 50,
                "temperature": 0.8
            }
        )
        
        start_time = time.time()
        result = await processor.process_task(batch_task)
        end_time = time.time()
        
        print(f"   総処理時間: {end_time - start_time:.2f} 秒")
        print(f"   ステータス: {result.status.value}")
        
        if result.status.name == "COMPLETED":
            output_data = result.output_data
            print(f"   成功: {output_data['successful_count']} / {output_data['total_items']}")
            print(f"   失敗: {output_data['failed_count']} / {output_data['total_items']}")
            print(f"   成功率: {result.confidence_score:.1%}")
            
            # いくつかの結果を表示
            batch_results = output_data.get("batch_results", [])
            for i, batch_result in enumerate(batch_results[:3], 1):
                if "generated_text" in batch_result:
                    text = batch_result["generated_text"]
                    print(f"   結果 {i}: {text[:60]}...")
                else:
                    print(f"   結果 {i}: エラー - {batch_result.get('error', 'Unknown')}")
        else:
            print(f"   エラー: {result.error_message}")
        
        print()
        
    except Exception as e:
        print(f"バッチ処理デモエラー: {e}")


async def demo_batch_job_processing(processor):
    """バッチジョブ処理のデモ"""
    
    print("6. バッチジョブ処理デモ")
    
    try:
        # 複数のタスクを作成
        tasks = []
        
        # テキスト生成タスク
        for i in range(3):
            task = ProcessingTask(
                task_id=f"job_text_{i}",
                task_type=TaskType.TEXT_GENERATION,
                input_data={"text": f"Story {i+1}: In a distant galaxy"},
                parameters={"model_name": "gpt2", "max_length": 40}
            )
            tasks.append(task)
        
        # コード生成タスク
        for i in range(2):
            task = ProcessingTask(
                task_id=f"job_code_{i}",
                task_type=TaskType.CODE_GENERATION,
                input_data={"prompt": f"def function_{i+1}():"},
                parameters={"language": "python", "max_length": 60}
            )
            tasks.append(task)
        
        # バッチジョブ作成
        job = BatchProcessingJob(
            job_id="demo_batch_job",
            tasks=tasks,
            batch_size=3,
            parallel_workers=2
        )
        
        print(f"   ジョブID: {job.job_id}")
        print(f"   総タスク数: {len(job.tasks)}")
        print(f"   並列ワーカー数: {job.parallel_workers}")
        
        start_time = time.time()
        completed_job = await processor.process_batch_job(job)
        end_time = time.time()
        
        print(f"   総処理時間: {end_time - start_time:.2f} 秒")
        print(f"   ジョブステータス: {completed_job.status.value}")
        
        if completed_job.status.name == "COMPLETED":
            successful_results = [r for r in completed_job.results if r.status.name == "COMPLETED"]
            failed_results = [r for r in completed_job.results if r.status.name == "FAILED"]
            
            print(f"   成功タスク: {len(successful_results)} / {len(completed_job.tasks)}")
            print(f"   失敗タスク: {len(failed_results)} / {len(completed_job.tasks)}")
            
            # 成功したタスクの詳細
            for i, result in enumerate(successful_results[:3], 1):
                task_type = result.task_id.split('_')[1]
                print(f"   成功 {i} ({task_type}): 処理時間 {result.processing_time:.2f}秒")
        
        print()
        
    except Exception as e:
        print(f"バッチジョブデモエラー: {e}")


async def demo_performance_monitoring(processor):
    """パフォーマンス監視のデモ"""
    
    print("7. パフォーマンス監視")
    
    try:
        # 処理統計取得
        stats = processor.get_processing_statistics()
        
        print(f"   処理統計:")
        processing_stats = stats["processing_stats"]
        print(f"     総タスク数: {processing_stats['total_tasks']}")
        print(f"     完了タスク数: {processing_stats['completed_tasks']}")
        print(f"     失敗タスク数: {processing_stats['failed_tasks']}")
        print(f"     GPU タスク数: {processing_stats['gpu_tasks']}")
        print(f"     CPU タスク数: {processing_stats['cpu_tasks']}")
        print(f"     総処理時間: {processing_stats['total_processing_time']:.2f} 秒")
        
        print(f"   システムリソース:")
        system_resources = stats["system_resources"]
        print(f"     GPU メモリ使用率: {system_resources['gpu_memory_usage_percent']:.1f}%")
        print(f"     CPU 使用率: {system_resources['cpu_utilization_percent']:.1f}%")
        print(f"     システムメモリ使用率: {system_resources['memory_usage_percent']:.1f}%")
        
        print(f"   リソース状況:")
        print(f"     読み込み済みモデル数: {stats['loaded_models']}")
        print(f"     アクティブパイプライン数: {stats['active_pipelines']}")
        print(f"     処理中タスク数: {stats['processing_tasks']}")
        
        print(f"   Accelerator情報:")
        accelerator_info = stats["accelerator_info"]
        print(f"     デバイス: {accelerator_info['device']}")
        print(f"     プロセス数: {accelerator_info['num_processes']}")
        print(f"     メインプロセス: {accelerator_info['is_main_process']}")
        
        print()
        
    except Exception as e:
        print(f"パフォーマンス監視デモエラー: {e}")


async def demo_resource_optimization():
    """リソース最適化のデモ"""
    
    print("8. リソース最適化デモ")
    
    try:
        # 異なる設定でプロセッサを作成
        configs = [
            {"name": "GPU優先", "max_gpu_memory_gb": 6.0, "max_cpu_workers": 2},
            {"name": "CPU優先", "max_gpu_memory_gb": 1.0, "max_cpu_workers": 8},
            {"name": "バランス", "max_gpu_memory_gb": 3.0, "max_cpu_workers": 4}
        ]
        
        for config in configs:
            print(f"\n   設定: {config['name']}")
            
            processor = AccelerateHybridProcessor(
                max_gpu_memory_gb=config["max_gpu_memory_gb"],
                max_cpu_workers=config["max_cpu_workers"],
                auto_device_map=True
            )
            
            # テストタスク
            test_task = ProcessingTask(
                task_id="optimization_test",
                task_type=TaskType.TEXT_GENERATION,
                input_data={"text": "Optimization test"}
            )
            
            allocation = processor.determine_optimal_device(test_task)
            
            print(f"     推奨デバイス: {allocation.device_type.value}")
            print(f"     メモリ配分: {allocation.memory_allocated_mb:.0f} MB")
            print(f"     CPU コア: {allocation.cpu_cores}")
            print(f"     推定処理時間: {allocation.estimated_processing_time:.2f} 秒")
            
            # クリーンアップ
            processor.cleanup_resources()
        
        print()
        
    except Exception as e:
        print(f"リソース最適化デモエラー: {e}")


async def main():
    """メインデモ実行"""
    
    try:
        # システムリソース確認
        processor = await demo_system_resources()
        
        # テキスト生成デモ
        await demo_text_generation(processor)
        
        # コード生成デモ
        await demo_code_generation(processor)
        
        # バッチ処理デモ
        await demo_batch_processing(processor)
        
        # バッチジョブ処理デモ
        await demo_batch_job_processing(processor)
        
        # パフォーマンス監視デモ
        await demo_performance_monitoring(processor)
        
        # リソース最適化デモ
        await demo_resource_optimization()
        
        print("=== デモ完了 ===")
        print("HuggingFace Accelerate ハイブリッド処理システムが正常に動作することを確認しました。")
        
        # 最終クリーンアップ
        processor.cleanup_resources()
        
    except Exception as e:
        print(f"\nデモ実行エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())