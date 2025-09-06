"""
Performance Integration Tests
パフォーマンス統合テスト
"""

import pytest
import asyncio
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging

from src.advanced_agent.core.self_learning_agent import SelfLearningAgent
from src.advanced_agent.monitoring.system_monitor import SystemMonitor
from src.advanced_agent.memory.persistent_memory import PersistentMemoryManager
from src.advanced_agent.reasoning.basic_engine import BasicReasoningEngine

logger = logging.getLogger(__name__)


class PerformanceIntegrationTests:
    """パフォーマンス統合テストクラス"""
    
    def __init__(self):
        self.agent: Optional[SelfLearningAgent] = None
        self.system_monitor: Optional[SystemMonitor] = None
        self.memory_manager: Optional[PersistentMemoryManager] = None
        self.reasoning_engine: Optional[BasicReasoningEngine] = None
        self.performance_metrics: Dict[str, Any] = {}
    
    async def setup_performance_test(self):
        """パフォーマンステストのセットアップ"""
        logger.info("⚡ パフォーマンステストセットアップ開始")
        
        try:
            # システム監視器の初期化
            self.system_monitor = SystemMonitor(
                collection_interval=0.1,
                enable_gpu_monitoring=False
            )
            
            # メモリ管理の初期化
            self.memory_manager = PersistentMemoryManager()
            
            # 推論エンジンの初期化
            self.reasoning_engine = BasicReasoningEngine()
            
            # 自己学習エージェントの初期化
            self.agent = SelfLearningAgent(
                system_monitor=self.system_monitor,
                memory_manager=self.memory_manager,
                reasoning_engine=self.reasoning_engine
            )
            
            logger.info("✅ パフォーマンステストセットアップ完了")
            return True
            
        except Exception as e:
            logger.error(f"❌ パフォーマンステストセットアップ失敗: {e}")
            return False
    
    async def test_memory_performance(self) -> Dict[str, Any]:
        """メモリパフォーマンステスト"""
        logger.info("🧠 メモリパフォーマンステスト開始")
        
        try:
            # 初期メモリ使用量
            initial_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
            
            # 大量のメモリ操作を実行
            start_time = time.time()
            
            # 100個のメモリを保存
            for i in range(100):
                await self.memory_manager.save_memory(
                    content=f"パフォーマンステスト用メモリ {i}",
                    metadata={"test": True, "index": i},
                    importance=0.5 + (i % 50) / 100
                )
            
            save_time = time.time() - start_time
            
            # メモリ検索のパフォーマンステスト
            start_time = time.time()
            
            for i in range(10):
                await self.memory_manager.search_memories(
                    query=f"テスト {i}",
                    limit=20
                )
            
            search_time = time.time() - start_time
            
            # 最終メモリ使用量
            final_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            return {
                "save_operations": 100,
                "search_operations": 10,
                "save_time_seconds": save_time,
                "search_time_seconds": search_time,
                "avg_save_time_ms": (save_time / 100) * 1000,
                "avg_search_time_ms": (search_time / 10) * 1000,
                "memory_increase_mb": memory_increase,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"メモリパフォーマンステスト失敗: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def test_reasoning_performance(self) -> Dict[str, Any]:
        """推論パフォーマンステスト"""
        logger.info("🧮 推論パフォーマンステスト開始")
        
        try:
            # 推論テスト用のプロンプト
            test_prompts = [
                "簡単な計算をしてください: 2 + 2 = ?",
                "短い文章を生成してください。",
                "現在の時刻を教えてください。",
                "1から10までの数字を列挙してください。",
                "色の名前を5つ教えてください。"
            ]
            
            reasoning_times = []
            reasoning_results = []
            
            for prompt in test_prompts:
                start_time = time.time()
                
                result = await self.reasoning_engine.reasoning_inference(
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=50
                )
                
                end_time = time.time()
                reasoning_time = end_time - start_time
                
                reasoning_times.append(reasoning_time)
                reasoning_results.append({
                    "prompt": prompt,
                    "response_length": len(str(result)) if result else 0,
                    "time_seconds": reasoning_time
                })
            
            avg_reasoning_time = sum(reasoning_times) / len(reasoning_times)
            min_reasoning_time = min(reasoning_times)
            max_reasoning_time = max(reasoning_times)
            
            return {
                "total_prompts": len(test_prompts),
                "avg_reasoning_time_seconds": avg_reasoning_time,
                "min_reasoning_time_seconds": min_reasoning_time,
                "max_reasoning_time_seconds": max_reasoning_time,
                "reasoning_results": reasoning_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"推論パフォーマンステスト失敗: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def test_monitoring_performance(self) -> Dict[str, Any]:
        """監視パフォーマンステスト"""
        logger.info("📊 監視パフォーマンステスト開始")
        
        try:
            # 監視開始
            await self.system_monitor.start_monitoring()
            
            # 監視データ収集のパフォーマンステスト
            collection_times = []
            metrics_counts = []
            
            for i in range(20):
                start_time = time.time()
                
                # メトリクス取得
                metrics = self.system_monitor.get_latest_metrics()
                
                end_time = time.time()
                collection_time = end_time - start_time
                
                collection_times.append(collection_time)
                metrics_counts.append(1 if metrics else 0)
                
                # 少し待機
                await asyncio.sleep(0.1)
            
            # 監視停止
            await self.system_monitor.cleanup()
            
            # 履歴取得のパフォーマンステスト
            start_time = time.time()
            history = self.system_monitor.get_metrics_history()
            history_time = time.time() - start_time
            
            avg_collection_time = sum(collection_times) / len(collection_times)
            successful_collections = sum(metrics_counts)
            
            return {
                "collection_operations": 20,
                "successful_collections": successful_collections,
                "avg_collection_time_seconds": avg_collection_time,
                "history_retrieval_time_seconds": history_time,
                "history_size": len(history),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"監視パフォーマンステスト失敗: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def test_concurrent_operations(self) -> Dict[str, Any]:
        """並行操作パフォーマンステスト"""
        logger.info("🔄 並行操作パフォーマンステスト開始")
        
        try:
            # 並行操作のテスト
            start_time = time.time()
            
            # 複数のタスクを並行実行
            tasks = [
                self._concurrent_memory_operation(i) for i in range(10)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # 成功したタスク数
            successful_tasks = sum(1 for result in results if not isinstance(result, Exception))
            failed_tasks = len(results) - successful_tasks
            
            return {
                "concurrent_tasks": 10,
                "successful_tasks": successful_tasks,
                "failed_tasks": failed_tasks,
                "total_time_seconds": total_time,
                "avg_time_per_task_seconds": total_time / 10,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"並行操作パフォーマンステスト失敗: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def test_system_resource_usage(self) -> Dict[str, Any]:
        """システムリソース使用量テスト"""
        logger.info("💻 システムリソース使用量テスト開始")
        
        try:
            # 初期リソース使用量
            initial_cpu = psutil.cpu_percent()
            initial_memory = psutil.virtual_memory().percent
            initial_disk = psutil.disk_usage('/').percent
            
            # 重い操作を実行
            await self._heavy_operations()
            
            # 最終リソース使用量
            final_cpu = psutil.cpu_percent()
            final_memory = psutil.virtual_memory().percent
            final_disk = psutil.disk_usage('/').percent
            
            return {
                "initial_cpu_percent": initial_cpu,
                "final_cpu_percent": final_cpu,
                "cpu_increase": final_cpu - initial_cpu,
                "initial_memory_percent": initial_memory,
                "final_memory_percent": final_memory,
                "memory_increase": final_memory - initial_memory,
                "initial_disk_percent": initial_disk,
                "final_disk_percent": final_disk,
                "disk_increase": final_disk - initial_disk,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"システムリソース使用量テスト失敗: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _concurrent_memory_operation(self, task_id: int) -> Dict[str, Any]:
        """並行メモリ操作"""
        try:
            # メモリ保存
            await self.memory_manager.save_memory(
                content=f"並行テスト用メモリ {task_id}",
                metadata={"task_id": task_id, "concurrent": True},
                importance=0.7
            )
            
            # メモリ検索
            search_result = await self.memory_manager.search_memories(
                query=f"並行テスト {task_id}",
                limit=5
            )
            
            return {
                "task_id": task_id,
                "success": True,
                "search_results": len(search_result.get("results", []))
            }
            
        except Exception as e:
            return {
                "task_id": task_id,
                "success": False,
                "error": str(e)
            }
    
    async def _heavy_operations(self):
        """重い操作の実行"""
        # 大量のメモリ操作
        for i in range(50):
            await self.memory_manager.save_memory(
                content=f"重い操作テスト用メモリ {i}",
                metadata={"heavy_operation": True, "index": i},
                importance=0.6
            )
        
        # 大量の推論操作
        for i in range(10):
            await self.reasoning_engine.reasoning_inference(
                prompt=f"重い操作テスト用推論 {i}",
                temperature=0.7,
                max_tokens=30
            )
    
    async def cleanup_performance_test(self):
        """パフォーマンステストのクリーンアップ"""
        logger.info("🧹 パフォーマンステストクリーンアップ開始")
        
        try:
            if self.system_monitor:
                await self.system_monitor.cleanup()
            
            if self.memory_manager:
                await self.memory_manager.cleanup()
            
            logger.info("✅ パフォーマンステストクリーンアップ完了")
            
        except Exception as e:
            logger.error(f"❌ パフォーマンステストクリーンアップ失敗: {e}")


@pytest.mark.asyncio
async def test_performance_integration_suite():
    """パフォーマンス統合テストスイート"""
    performance_tests = PerformanceIntegrationTests()
    
    try:
        # パフォーマンステストセットアップ
        setup_success = await performance_tests.setup_performance_test()
        assert setup_success, "パフォーマンステストのセットアップに失敗しました"
        
        # 各テストを実行
        memory_results = await performance_tests.test_memory_performance()
        reasoning_results = await performance_tests.test_reasoning_performance()
        monitoring_results = await performance_tests.test_monitoring_performance()
        concurrent_results = await performance_tests.test_concurrent_operations()
        resource_results = await performance_tests.test_system_resource_usage()
        
        # 結果をまとめる
        all_results = {
            "memory_performance": memory_results,
            "reasoning_performance": reasoning_results,
            "monitoring_performance": monitoring_results,
            "concurrent_operations": concurrent_results,
            "system_resource_usage": resource_results
        }
        
        # 基本的な検証
        assert "memory_performance" in all_results, "メモリパフォーマンステストが実行されていません"
        assert "reasoning_performance" in all_results, "推論パフォーマンステストが実行されていません"
        assert "monitoring_performance" in all_results, "監視パフォーマンステストが実行されていません"
        
        return all_results
        
    finally:
        # パフォーマンステストクリーンアップ
        await performance_tests.cleanup_performance_test()


if __name__ == "__main__":
    # 直接実行時のテスト
    async def main():
        performance_tests = PerformanceIntegrationTests()
        
        print("⚡ パフォーマンステストセットアップ中...")
        setup_success = await performance_tests.setup_performance_test()
        print(f"セットアップ結果: {setup_success}")
        
        if setup_success:
            print("🧠 メモリパフォーマンステスト実行中...")
            memory_results = await performance_tests.test_memory_performance()
            print(f"メモリ結果: {memory_results.get('save_operations', 0)}個の操作")
            
            print("🧮 推論パフォーマンステスト実行中...")
            reasoning_results = await performance_tests.test_reasoning_performance()
            print(f"推論結果: {reasoning_results.get('total_prompts', 0)}個のプロンプト")
            
            print("📊 監視パフォーマンステスト実行中...")
            monitoring_results = await performance_tests.test_monitoring_performance()
            print(f"監視結果: {monitoring_results.get('collection_operations', 0)}個の操作")
            
            print("🔄 並行操作パフォーマンステスト実行中...")
            concurrent_results = await performance_tests.test_concurrent_operations()
            print(f"並行結果: {concurrent_results.get('concurrent_tasks', 0)}個のタスク")
            
            print("💻 システムリソース使用量テスト実行中...")
            resource_results = await performance_tests.test_system_resource_usage()
            print(f"リソース結果: CPU {resource_results.get('final_cpu_percent', 0):.1f}%")
            
            print("✅ 全パフォーマンス統合テスト完了")
        
        # クリーンアップ
        await performance_tests.cleanup_performance_test()
    
    asyncio.run(main())
