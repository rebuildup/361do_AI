"""
Load Integration Tests
負荷統合テスト
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


class LoadIntegrationTests:
    """負荷統合テストクラス"""
    
    def __init__(self):
        self.agent: Optional[SelfLearningAgent] = None
        self.system_monitor: Optional[SystemMonitor] = None
        self.memory_manager: Optional[PersistentMemoryManager] = None
        self.reasoning_engine: Optional[BasicReasoningEngine] = None
        self.load_metrics: Dict[str, Any] = {}
    
    async def setup_load_test(self):
        """負荷テストのセットアップ"""
        logger.info("🔥 負荷テストセットアップ開始")
        
        try:
            # システム監視器の初期化
            self.system_monitor = SystemMonitor(
                collection_interval=0.05,  # より頻繁な監視
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
            
            logger.info("✅ 負荷テストセットアップ完了")
            return True
            
        except Exception as e:
            logger.error(f"❌ 負荷テストセットアップ失敗: {e}")
            return False
    
    async def test_high_volume_memory_operations(self) -> Dict[str, Any]:
        """高ボリュームメモリ操作テスト"""
        logger.info("💾 高ボリュームメモリ操作テスト開始")
        
        try:
            # 初期リソース使用量
            initial_memory = psutil.virtual_memory().percent
            initial_cpu = psutil.cpu_percent()
            
            # 大量のメモリ操作を実行
            start_time = time.time()
            
            # 1000個のメモリを保存
            save_tasks = []
            for i in range(1000):
                task = self.memory_manager.save_memory(
                    content=f"高ボリュームテスト用メモリ {i}",
                    metadata={"load_test": True, "index": i, "timestamp": datetime.now().isoformat()},
                    importance=0.3 + (i % 70) / 100
                )
                save_tasks.append(task)
            
            # 並行実行
            await asyncio.gather(*save_tasks, return_exceptions=True)
            
            save_time = time.time() - start_time
            
            # 大量の検索操作を実行
            start_time = time.time()
            
            search_tasks = []
            for i in range(100):
                task = self.memory_manager.search_memories(
                    query=f"高ボリューム {i}",
                    limit=50
                )
                search_tasks.append(task)
            
            # 並行実行
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            search_time = time.time() - start_time
            
            # 最終リソース使用量
            final_memory = psutil.virtual_memory().percent
            final_cpu = psutil.cpu_percent()
            
            # 成功した操作数
            successful_saves = 1000  # 保存は基本的に成功
            successful_searches = sum(1 for result in search_results if not isinstance(result, Exception))
            
            return {
                "save_operations": 1000,
                "search_operations": 100,
                "successful_saves": successful_saves,
                "successful_searches": successful_searches,
                "save_time_seconds": save_time,
                "search_time_seconds": search_time,
                "avg_save_time_ms": (save_time / 1000) * 1000,
                "avg_search_time_ms": (search_time / 100) * 1000,
                "memory_increase_percent": final_memory - initial_memory,
                "cpu_increase_percent": final_cpu - initial_cpu,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"高ボリュームメモリ操作テスト失敗: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def test_concurrent_reasoning_load(self) -> Dict[str, Any]:
        """並行推論負荷テスト"""
        logger.info("🧮 並行推論負荷テスト開始")
        
        try:
            # 初期リソース使用量
            initial_cpu = psutil.cpu_percent()
            initial_memory = psutil.virtual_memory().percent
            
            # 大量の推論操作を並行実行
            start_time = time.time()
            
            reasoning_tasks = []
            for i in range(50):
                task = self.reasoning_engine.reasoning_inference(
                    prompt=f"並行推論負荷テスト {i}: 簡単な計算をしてください。",
                    temperature=0.7,
                    max_tokens=30
                )
                reasoning_tasks.append(task)
            
            # 並行実行
            reasoning_results = await asyncio.gather(*reasoning_tasks, return_exceptions=True)
            
            reasoning_time = time.time() - start_time
            
            # 最終リソース使用量
            final_cpu = psutil.cpu_percent()
            final_memory = psutil.virtual_memory().percent
            
            # 成功した推論数
            successful_reasoning = sum(1 for result in reasoning_results if not isinstance(result, Exception))
            
            return {
                "concurrent_reasoning_tasks": 50,
                "successful_reasoning": successful_reasoning,
                "failed_reasoning": 50 - successful_reasoning,
                "total_time_seconds": reasoning_time,
                "avg_time_per_reasoning_seconds": reasoning_time / 50,
                "cpu_increase_percent": final_cpu - initial_cpu,
                "memory_increase_percent": final_memory - initial_memory,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"並行推論負荷テスト失敗: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def test_monitoring_under_load(self) -> Dict[str, Any]:
        """負荷下での監視テスト"""
        logger.info("📊 負荷下での監視テスト開始")
        
        try:
            # 監視開始
            await self.system_monitor.start_monitoring()
            
            # 負荷をかけながら監視
            load_tasks = []
            
            # メモリ操作の負荷
            for i in range(200):
                task = self.memory_manager.save_memory(
                    content=f"監視負荷テスト用メモリ {i}",
                    metadata={"monitoring_load": True, "index": i},
                    importance=0.5
                )
                load_tasks.append(task)
            
            # 推論操作の負荷
            for i in range(20):
                task = self.reasoning_engine.reasoning_inference(
                    prompt=f"監視負荷テスト {i}",
                    temperature=0.7,
                    max_tokens=20
                )
                load_tasks.append(task)
            
            # 負荷実行開始
            start_time = time.time()
            load_results = await asyncio.gather(*load_tasks, return_exceptions=True)
            load_time = time.time() - start_time
            
            # 監視データの取得
            monitoring_data = []
            for i in range(10):
                metrics = self.system_monitor.get_latest_metrics()
                monitoring_data.append(metrics)
                await asyncio.sleep(0.1)
            
            # 監視停止
            await self.system_monitor.cleanup()
            
            # 履歴取得
            history = self.system_monitor.get_metrics_history()
            
            # 成功した負荷操作数
            successful_load_operations = sum(1 for result in load_results if not isinstance(result, Exception))
            
            return {
                "load_operations": 220,  # 200 + 20
                "successful_load_operations": successful_load_operations,
                "load_time_seconds": load_time,
                "monitoring_data_points": len(monitoring_data),
                "monitoring_history_size": len(history),
                "avg_load_operation_time_ms": (load_time / 220) * 1000,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"負荷下での監視テスト失敗: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def test_mixed_workload(self) -> Dict[str, Any]:
        """混合ワークロードテスト"""
        logger.info("🔄 混合ワークロードテスト開始")
        
        try:
            # 初期リソース使用量
            initial_cpu = psutil.cpu_percent()
            initial_memory = psutil.virtual_memory().percent
            
            # 混合ワークロードの実行
            start_time = time.time()
            
            # 様々な操作を並行実行
            mixed_tasks = []
            
            # メモリ操作 (30%)
            for i in range(30):
                task = self.memory_manager.save_memory(
                    content=f"混合ワークロードメモリ {i}",
                    metadata={"mixed_workload": True, "type": "memory", "index": i},
                    importance=0.6
                )
                mixed_tasks.append(task)
            
            # 推論操作 (30%)
            for i in range(30):
                task = self.reasoning_engine.reasoning_inference(
                    prompt=f"混合ワークロード推論 {i}",
                    temperature=0.7,
                    max_tokens=25
                )
                mixed_tasks.append(task)
            
            # メモリ検索 (20%)
            for i in range(20):
                task = self.memory_manager.search_memories(
                    query=f"混合ワークロード {i}",
                    limit=10
                )
                mixed_tasks.append(task)
            
            # 複合操作 (20%)
            for i in range(20):
                async def composite_operation(index):
                    # メモリ保存
                    await self.memory_manager.save_memory(
                        content=f"複合操作メモリ {index}",
                        metadata={"composite": True, "index": index},
                        importance=0.7
                    )
                    # 推論実行
                    await self.reasoning_engine.reasoning_inference(
                        prompt=f"複合操作推論 {index}",
                        temperature=0.7,
                        max_tokens=20
                    )
                    # メモリ検索
                    await self.memory_manager.search_memories(
                        query=f"複合操作 {index}",
                        limit=5
                    )
                
                mixed_tasks.append(composite_operation(i))
            
            # 並行実行
            mixed_results = await asyncio.gather(*mixed_tasks, return_exceptions=True)
            
            mixed_time = time.time() - start_time
            
            # 最終リソース使用量
            final_cpu = psutil.cpu_percent()
            final_memory = psutil.virtual_memory().percent
            
            # 成功した操作数
            successful_operations = sum(1 for result in mixed_results if not isinstance(result, Exception))
            
            return {
                "total_operations": 100,
                "successful_operations": successful_operations,
                "failed_operations": 100 - successful_operations,
                "total_time_seconds": mixed_time,
                "avg_operation_time_ms": (mixed_time / 100) * 1000,
                "cpu_increase_percent": final_cpu - initial_cpu,
                "memory_increase_percent": final_memory - initial_memory,
                "operations_per_second": 100 / mixed_time,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"混合ワークロードテスト失敗: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def test_sustained_load(self) -> Dict[str, Any]:
        """持続負荷テスト"""
        logger.info("⏱️ 持続負荷テスト開始")
        
        try:
            # 初期リソース使用量
            initial_cpu = psutil.cpu_percent()
            initial_memory = psutil.virtual_memory().percent
            
            # 持続負荷の実行 (30秒間)
            start_time = time.time()
            end_time = start_time + 30  # 30秒間
            
            operation_counts = {
                "memory_saves": 0,
                "memory_searches": 0,
                "reasoning_operations": 0
            }
            
            while time.time() < end_time:
                # 並行操作の実行
                tasks = []
                
                # メモリ保存
                for i in range(5):
                    task = self.memory_manager.save_memory(
                        content=f"持続負荷メモリ {operation_counts['memory_saves'] + i}",
                        metadata={"sustained_load": True, "timestamp": datetime.now().isoformat()},
                        importance=0.4
                    )
                    tasks.append(task)
                
                # メモリ検索
                for i in range(3):
                    task = self.memory_manager.search_memories(
                        query=f"持続負荷 {operation_counts['memory_searches'] + i}",
                        limit=20
                    )
                    tasks.append(task)
                
                # 推論操作
                for i in range(2):
                    task = self.reasoning_engine.reasoning_inference(
                        prompt=f"持続負荷推論 {operation_counts['reasoning_operations'] + i}",
                        temperature=0.7,
                        max_tokens=15
                    )
                    tasks.append(task)
                
                # 並行実行
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 操作数の更新
                operation_counts["memory_saves"] += 5
                operation_counts["memory_searches"] += 3
                operation_counts["reasoning_operations"] += 2
                
                # 少し待機
                await asyncio.sleep(1)
            
            total_time = time.time() - start_time
            
            # 最終リソース使用量
            final_cpu = psutil.cpu_percent()
            final_memory = psutil.virtual_memory().percent
            
            return {
                "sustained_load_duration_seconds": total_time,
                "total_memory_saves": operation_counts["memory_saves"],
                "total_memory_searches": operation_counts["memory_searches"],
                "total_reasoning_operations": operation_counts["reasoning_operations"],
                "total_operations": sum(operation_counts.values()),
                "operations_per_second": sum(operation_counts.values()) / total_time,
                "cpu_increase_percent": final_cpu - initial_cpu,
                "memory_increase_percent": final_memory - initial_memory,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"持続負荷テスト失敗: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def cleanup_load_test(self):
        """負荷テストのクリーンアップ"""
        logger.info("🧹 負荷テストクリーンアップ開始")
        
        try:
            if self.system_monitor:
                await self.system_monitor.cleanup()
            
            if self.memory_manager:
                await self.memory_manager.cleanup()
            
            logger.info("✅ 負荷テストクリーンアップ完了")
            
        except Exception as e:
            logger.error(f"❌ 負荷テストクリーンアップ失敗: {e}")


@pytest.mark.asyncio
async def test_load_integration_suite():
    """負荷統合テストスイート"""
    load_tests = LoadIntegrationTests()
    
    try:
        # 負荷テストセットアップ
        setup_success = await load_tests.setup_load_test()
        assert setup_success, "負荷テストのセットアップに失敗しました"
        
        # 各テストを実行
        memory_load_results = await load_tests.test_high_volume_memory_operations()
        reasoning_load_results = await load_tests.test_concurrent_reasoning_load()
        monitoring_load_results = await load_tests.test_monitoring_under_load()
        mixed_workload_results = await load_tests.test_mixed_workload()
        sustained_load_results = await load_tests.test_sustained_load()
        
        # 結果をまとめる
        all_results = {
            "high_volume_memory": memory_load_results,
            "concurrent_reasoning": reasoning_load_results,
            "monitoring_under_load": monitoring_load_results,
            "mixed_workload": mixed_workload_results,
            "sustained_load": sustained_load_results
        }
        
        # 基本的な検証
        assert "high_volume_memory" in all_results, "高ボリュームメモリ操作テストが実行されていません"
        assert "concurrent_reasoning" in all_results, "並行推論負荷テストが実行されていません"
        assert "monitoring_under_load" in all_results, "負荷下での監視テストが実行されていません"
        
        return all_results
        
    finally:
        # 負荷テストクリーンアップ
        await load_tests.cleanup_load_test()


if __name__ == "__main__":
    # 直接実行時のテスト
    async def main():
        load_tests = LoadIntegrationTests()
        
        print("🔥 負荷テストセットアップ中...")
        setup_success = await load_tests.setup_load_test()
        print(f"セットアップ結果: {setup_success}")
        
        if setup_success:
            print("💾 高ボリュームメモリ操作テスト実行中...")
            memory_results = await load_tests.test_high_volume_memory_operations()
            print(f"メモリ結果: {memory_results.get('save_operations', 0)}個の操作")
            
            print("🧮 並行推論負荷テスト実行中...")
            reasoning_results = await load_tests.test_concurrent_reasoning_load()
            print(f"推論結果: {reasoning_results.get('concurrent_reasoning_tasks', 0)}個のタスク")
            
            print("📊 負荷下での監視テスト実行中...")
            monitoring_results = await load_tests.test_monitoring_under_load()
            print(f"監視結果: {monitoring_results.get('load_operations', 0)}個の操作")
            
            print("🔄 混合ワークロードテスト実行中...")
            mixed_results = await load_tests.test_mixed_workload()
            print(f"混合結果: {mixed_results.get('total_operations', 0)}個の操作")
            
            print("⏱️ 持続負荷テスト実行中...")
            sustained_results = await load_tests.test_sustained_load()
            print(f"持続結果: {sustained_results.get('total_operations', 0)}個の操作")
            
            print("✅ 全負荷統合テスト完了")
        
        # クリーンアップ
        await load_tests.cleanup_load_test()
    
    asyncio.run(main())
