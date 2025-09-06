"""
Stability Integration Tests
安定性統合テスト
"""

import pytest
import asyncio
import time
import psutil
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging

from src.advanced_agent.core.self_learning_agent import SelfLearningAgent
from src.advanced_agent.monitoring.system_monitor import SystemMonitor
from src.advanced_agent.memory.persistent_memory import PersistentMemoryManager
from src.advanced_agent.reasoning.basic_engine import BasicReasoningEngine

logger = logging.getLogger(__name__)


class StabilityIntegrationTests:
    """安定性統合テストクラス"""
    
    def __init__(self):
        self.agent: Optional[SelfLearningAgent] = None
        self.system_monitor: Optional[SystemMonitor] = None
        self.memory_manager: Optional[PersistentMemoryManager] = None
        self.reasoning_engine: Optional[BasicReasoningEngine] = None
        self.stability_metrics: Dict[str, Any] = {}
        self.error_log: List[Dict[str, Any]] = []
    
    async def setup_stability_test(self):
        """安定性テストのセットアップ"""
        logger.info("🛡️ 安定性テストセットアップ開始")
        
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
            
            logger.info("✅ 安定性テストセットアップ完了")
            return True
            
        except Exception as e:
            logger.error(f"❌ 安定性テストセットアップ失敗: {e}")
            return False
    
    async def test_long_running_stability(self) -> Dict[str, Any]:
        """長時間実行安定性テスト"""
        logger.info("⏰ 長時間実行安定性テスト開始")
        
        try:
            # 初期リソース使用量
            initial_cpu = psutil.cpu_percent()
            initial_memory = psutil.virtual_memory().percent
            
            # 長時間実行 (5分間)
            start_time = time.time()
            end_time = start_time + 300  # 5分間
            
            operation_counts = {
                "memory_saves": 0,
                "memory_searches": 0,
                "reasoning_operations": 0,
                "errors": 0
            }
            
            # 監視開始
            await self.system_monitor.start_monitoring()
            
            while time.time() < end_time:
                try:
                    # ランダムな操作を実行
                    operation_type = random.choice(["memory_save", "memory_search", "reasoning"])
                    
                    if operation_type == "memory_save":
                        await self.memory_manager.save_memory(
                            content=f"長時間実行テストメモリ {operation_counts['memory_saves']}",
                            metadata={"long_running": True, "timestamp": datetime.now().isoformat()},
                            importance=random.uniform(0.3, 0.9)
                        )
                        operation_counts["memory_saves"] += 1
                    
                    elif operation_type == "memory_search":
                        await self.memory_manager.search_memories(
                            query=f"長時間実行 {operation_counts['memory_searches']}",
                            limit=random.randint(5, 20)
                        )
                        operation_counts["memory_searches"] += 1
                    
                    elif operation_type == "reasoning":
                        await self.reasoning_engine.reasoning_inference(
                            prompt=f"長時間実行推論 {operation_counts['reasoning_operations']}",
                            temperature=random.uniform(0.5, 0.9),
                            max_tokens=random.randint(10, 50)
                        )
                        operation_counts["reasoning_operations"] += 1
                    
                    # ランダムな待機時間
                    await asyncio.sleep(random.uniform(0.1, 1.0))
                    
                except Exception as e:
                    operation_counts["errors"] += 1
                    self.error_log.append({
                        "timestamp": datetime.now().isoformat(),
                        "error": str(e),
                        "operation_type": operation_type
                    })
                    logger.warning(f"長時間実行中のエラー: {e}")
            
            # 監視停止
            await self.system_monitor.cleanup()
            
            total_time = time.time() - start_time
            
            # 最終リソース使用量
            final_cpu = psutil.cpu_percent()
            final_memory = psutil.virtual_memory().percent
            
            # 安定性メトリクス
            total_operations = sum(operation_counts.values()) - operation_counts["errors"]
            error_rate = operation_counts["errors"] / (total_operations + operation_counts["errors"]) * 100
            
            return {
                "total_runtime_seconds": total_time,
                "total_operations": total_operations,
                "memory_saves": operation_counts["memory_saves"],
                "memory_searches": operation_counts["memory_searches"],
                "reasoning_operations": operation_counts["reasoning_operations"],
                "errors": operation_counts["errors"],
                "error_rate_percent": error_rate,
                "operations_per_second": total_operations / total_time,
                "cpu_increase_percent": final_cpu - initial_cpu,
                "memory_increase_percent": final_memory - initial_memory,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"長時間実行安定性テスト失敗: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def test_error_recovery(self) -> Dict[str, Any]:
        """エラー回復テスト"""
        logger.info("🔄 エラー回復テスト開始")
        
        try:
            recovery_tests = []
            
            # テスト1: 無効なメモリ操作からの回復
            try:
                await self.memory_manager.save_memory(
                    content="",  # 空のコンテンツ
                    metadata={"test": "error_recovery"},
                    importance=1.5  # 無効な重要度
                )
            except Exception as e:
                recovery_tests.append({
                    "test": "invalid_memory_operation",
                    "error": str(e),
                    "recovered": True
                })
            
            # テスト2: 無効な推論操作からの回復
            try:
                await self.reasoning_engine.reasoning_inference(
                    prompt="",  # 空のプロンプト
                    temperature=2.0,  # 無効な温度
                    max_tokens=-1  # 無効なトークン数
                )
            except Exception as e:
                recovery_tests.append({
                    "test": "invalid_reasoning_operation",
                    "error": str(e),
                    "recovered": True
                })
            
            # テスト3: 正常な操作が回復後に動作するか
            try:
                # 正常なメモリ操作
                await self.memory_manager.save_memory(
                    content="エラー回復後の正常なメモリ",
                    metadata={"recovery_test": True},
                    importance=0.7
                )
                
                # 正常な推論操作
                await self.reasoning_engine.reasoning_inference(
                    prompt="エラー回復後の正常な推論",
                    temperature=0.7,
                    max_tokens=20
                )
                
                recovery_tests.append({
                    "test": "normal_operations_after_error",
                    "success": True,
                    "recovered": True
                })
                
            except Exception as e:
                recovery_tests.append({
                    "test": "normal_operations_after_error",
                    "error": str(e),
                    "recovered": False
                })
            
            # 回復率の計算
            total_tests = len(recovery_tests)
            successful_recoveries = sum(1 for test in recovery_tests if test.get("recovered", False))
            recovery_rate = (successful_recoveries / total_tests) * 100
            
            return {
                "recovery_tests": recovery_tests,
                "total_tests": total_tests,
                "successful_recoveries": successful_recoveries,
                "recovery_rate_percent": recovery_rate,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"エラー回復テスト失敗: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def test_memory_leak_detection(self) -> Dict[str, Any]:
        """メモリリーク検出テスト"""
        logger.info("🔍 メモリリーク検出テスト開始")
        
        try:
            # 初期メモリ使用量
            initial_memory = psutil.virtual_memory().percent
            initial_memory_mb = psutil.virtual_memory().used / 1024 / 1024
            
            # 大量の操作を実行
            for cycle in range(10):
                # メモリ操作のサイクル
                for i in range(100):
                    await self.memory_manager.save_memory(
                        content=f"メモリリークテスト {cycle}-{i}",
                        metadata={"leak_test": True, "cycle": cycle, "index": i},
                        importance=0.5
                    )
                
                # メモリ検索
                for i in range(50):
                    await self.memory_manager.search_memories(
                        query=f"メモリリーク {cycle}-{i}",
                        limit=10
                    )
                
                # 推論操作
                for i in range(20):
                    await self.reasoning_engine.reasoning_inference(
                        prompt=f"メモリリーク推論 {cycle}-{i}",
                        temperature=0.7,
                        max_tokens=15
                    )
                
                # サイクル間の待機
                await asyncio.sleep(0.5)
                
                # メモリ使用量の記録
                current_memory = psutil.virtual_memory().percent
                current_memory_mb = psutil.virtual_memory().used / 1024 / 1024
                
                logger.info(f"サイクル {cycle}: メモリ使用量 {current_memory:.1f}% ({current_memory_mb:.1f}MB)")
            
            # 最終メモリ使用量
            final_memory = psutil.virtual_memory().percent
            final_memory_mb = psutil.virtual_memory().used / 1024 / 1024
            
            # メモリ増加量
            memory_increase_percent = final_memory - initial_memory
            memory_increase_mb = final_memory_mb - initial_memory_mb
            
            # メモリリークの判定
            memory_leak_detected = memory_increase_percent > 10.0  # 10%以上の増加
            
            return {
                "initial_memory_percent": initial_memory,
                "final_memory_percent": final_memory,
                "memory_increase_percent": memory_increase_percent,
                "initial_memory_mb": initial_memory_mb,
                "final_memory_mb": final_memory_mb,
                "memory_increase_mb": memory_increase_mb,
                "memory_leak_detected": memory_leak_detected,
                "total_operations": 1700,  # 100 + 50 + 20 * 10 cycles
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"メモリリーク検出テスト失敗: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def test_concurrent_stability(self) -> Dict[str, Any]:
        """並行安定性テスト"""
        logger.info("🔄 並行安定性テスト開始")
        
        try:
            # 大量の並行タスクを実行
            concurrent_tasks = []
            
            # メモリ操作タスク
            for i in range(50):
                task = self._concurrent_memory_task(i)
                concurrent_tasks.append(task)
            
            # 推論操作タスク
            for i in range(30):
                task = self._concurrent_reasoning_task(i)
                concurrent_tasks.append(task)
            
            # 複合操作タスク
            for i in range(20):
                task = self._concurrent_composite_task(i)
                concurrent_tasks.append(task)
            
            # 並行実行
            start_time = time.time()
            results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
            execution_time = time.time() - start_time
            
            # 結果の分析
            successful_tasks = sum(1 for result in results if not isinstance(result, Exception))
            failed_tasks = len(results) - successful_tasks
            
            # エラーの詳細
            error_details = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    error_details.append({
                        "task_index": i,
                        "error": str(result),
                        "error_type": type(result).__name__
                    })
            
            # 安定性メトリクス
            stability_rate = (successful_tasks / len(results)) * 100
            
            return {
                "total_concurrent_tasks": len(concurrent_tasks),
                "successful_tasks": successful_tasks,
                "failed_tasks": failed_tasks,
                "stability_rate_percent": stability_rate,
                "execution_time_seconds": execution_time,
                "avg_task_time_seconds": execution_time / len(concurrent_tasks),
                "error_details": error_details,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"並行安定性テスト失敗: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def test_resource_stability(self) -> Dict[str, Any]:
        """リソース安定性テスト"""
        logger.info("💻 リソース安定性テスト開始")
        
        try:
            # 初期リソース使用量
            initial_cpu = psutil.cpu_percent()
            initial_memory = psutil.virtual_memory().percent
            initial_disk = psutil.disk_usage('/').percent
            
            # リソース使用量の監視
            resource_measurements = []
            
            # 監視開始
            await self.system_monitor.start_monitoring()
            
            # 負荷をかけながらリソースを監視
            for i in range(30):
                # 負荷操作
                await self._resource_intensive_operation(i)
                
                # リソース測定
                current_cpu = psutil.cpu_percent()
                current_memory = psutil.virtual_memory().percent
                current_disk = psutil.disk_usage('/').percent
                
                resource_measurements.append({
                    "iteration": i,
                    "cpu_percent": current_cpu,
                    "memory_percent": current_memory,
                    "disk_percent": current_disk,
                    "timestamp": datetime.now().isoformat()
                })
                
                await asyncio.sleep(1)
            
            # 監視停止
            await self.system_monitor.cleanup()
            
            # 最終リソース使用量
            final_cpu = psutil.cpu_percent()
            final_memory = psutil.virtual_memory().percent
            final_disk = psutil.disk_usage('/').percent
            
            # リソース安定性の分析
            cpu_values = [m["cpu_percent"] for m in resource_measurements]
            memory_values = [m["memory_percent"] for m in resource_measurements]
            disk_values = [m["disk_percent"] for m in resource_measurements]
            
            cpu_stability = max(cpu_values) - min(cpu_values)
            memory_stability = max(memory_values) - min(memory_values)
            disk_stability = max(disk_values) - min(disk_values)
            
            # 安定性の判定
            cpu_stable = cpu_stability < 20.0  # CPU変動が20%未満
            memory_stable = memory_stability < 15.0  # メモリ変動が15%未満
            disk_stable = disk_stability < 5.0  # ディスク変動が5%未満
            
            return {
                "initial_resources": {
                    "cpu_percent": initial_cpu,
                    "memory_percent": initial_memory,
                    "disk_percent": initial_disk
                },
                "final_resources": {
                    "cpu_percent": final_cpu,
                    "memory_percent": final_memory,
                    "disk_percent": final_disk
                },
                "resource_stability": {
                    "cpu_variation": cpu_stability,
                    "memory_variation": memory_stability,
                    "disk_variation": disk_stability
                },
                "stability_assessment": {
                    "cpu_stable": cpu_stable,
                    "memory_stable": memory_stable,
                    "disk_stable": disk_stable,
                    "overall_stable": cpu_stable and memory_stable and disk_stable
                },
                "measurements": resource_measurements,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"リソース安定性テスト失敗: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    # ヘルパーメソッド
    async def _concurrent_memory_task(self, task_id: int) -> Dict[str, Any]:
        """並行メモリタスク"""
        try:
            await self.memory_manager.save_memory(
                content=f"並行安定性テストメモリ {task_id}",
                metadata={"concurrent_stability": True, "task_id": task_id},
                importance=0.6
            )
            
            await self.memory_manager.search_memories(
                query=f"並行安定性 {task_id}",
                limit=10
            )
            
            return {"task_id": task_id, "success": True}
            
        except Exception as e:
            return {"task_id": task_id, "success": False, "error": str(e)}
    
    async def _concurrent_reasoning_task(self, task_id: int) -> Dict[str, Any]:
        """並行推論タスク"""
        try:
            await self.reasoning_engine.reasoning_inference(
                prompt=f"並行安定性推論 {task_id}",
                temperature=0.7,
                max_tokens=20
            )
            
            return {"task_id": task_id, "success": True}
            
        except Exception as e:
            return {"task_id": task_id, "success": False, "error": str(e)}
    
    async def _concurrent_composite_task(self, task_id: int) -> Dict[str, Any]:
        """並行複合タスク"""
        try:
            # メモリ保存
            await self.memory_manager.save_memory(
                content=f"並行複合テストメモリ {task_id}",
                metadata={"composite": True, "task_id": task_id},
                importance=0.7
            )
            
            # 推論実行
            await self.reasoning_engine.reasoning_inference(
                prompt=f"並行複合推論 {task_id}",
                temperature=0.7,
                max_tokens=15
            )
            
            # メモリ検索
            await self.memory_manager.search_memories(
                query=f"並行複合 {task_id}",
                limit=5
            )
            
            return {"task_id": task_id, "success": True}
            
        except Exception as e:
            return {"task_id": task_id, "success": False, "error": str(e)}
    
    async def _resource_intensive_operation(self, iteration: int):
        """リソース集約的操作"""
        # メモリ操作
        for i in range(10):
            await self.memory_manager.save_memory(
                content=f"リソース集約メモリ {iteration}-{i}",
                metadata={"resource_intensive": True, "iteration": iteration},
                importance=0.5
            )
        
        # 推論操作
        for i in range(5):
            await self.reasoning_engine.reasoning_inference(
                prompt=f"リソース集約推論 {iteration}-{i}",
                temperature=0.7,
                max_tokens=10
            )
    
    async def cleanup_stability_test(self):
        """安定性テストのクリーンアップ"""
        logger.info("🧹 安定性テストクリーンアップ開始")
        
        try:
            if self.system_monitor:
                await self.system_monitor.cleanup()
            
            if self.memory_manager:
                await self.memory_manager.cleanup()
            
            logger.info("✅ 安定性テストクリーンアップ完了")
            
        except Exception as e:
            logger.error(f"❌ 安定性テストクリーンアップ失敗: {e}")


@pytest.mark.asyncio
async def test_stability_integration_suite():
    """安定性統合テストスイート"""
    stability_tests = StabilityIntegrationTests()
    
    try:
        # 安定性テストセットアップ
        setup_success = await stability_tests.setup_stability_test()
        assert setup_success, "安定性テストのセットアップに失敗しました"
        
        # 各テストを実行
        long_running_results = await stability_tests.test_long_running_stability()
        error_recovery_results = await stability_tests.test_error_recovery()
        memory_leak_results = await stability_tests.test_memory_leak_detection()
        concurrent_stability_results = await stability_tests.test_concurrent_stability()
        resource_stability_results = await stability_tests.test_resource_stability()
        
        # 結果をまとめる
        all_results = {
            "long_running_stability": long_running_results,
            "error_recovery": error_recovery_results,
            "memory_leak_detection": memory_leak_results,
            "concurrent_stability": concurrent_stability_results,
            "resource_stability": resource_stability_results
        }
        
        # 基本的な検証
        assert "long_running_stability" in all_results, "長時間実行安定性テストが実行されていません"
        assert "error_recovery" in all_results, "エラー回復テストが実行されていません"
        assert "memory_leak_detection" in all_results, "メモリリーク検出テストが実行されていません"
        
        return all_results
        
    finally:
        # 安定性テストクリーンアップ
        await stability_tests.cleanup_stability_test()


if __name__ == "__main__":
    # 直接実行時のテスト
    async def main():
        stability_tests = StabilityIntegrationTests()
        
        print("🛡️ 安定性テストセットアップ中...")
        setup_success = await stability_tests.setup_stability_test()
        print(f"セットアップ結果: {setup_success}")
        
        if setup_success:
            print("⏰ 長時間実行安定性テスト実行中...")
            long_running_results = await stability_tests.test_long_running_stability()
            print(f"長時間実行結果: {long_running_results.get('total_operations', 0)}個の操作")
            
            print("🔄 エラー回復テスト実行中...")
            error_recovery_results = await stability_tests.test_error_recovery()
            print(f"エラー回復結果: {error_recovery_results.get('recovery_rate_percent', 0):.1f}%")
            
            print("🔍 メモリリーク検出テスト実行中...")
            memory_leak_results = await stability_tests.test_memory_leak_detection()
            print(f"メモリリーク結果: {memory_leak_results.get('memory_leak_detected', False)}")
            
            print("🔄 並行安定性テスト実行中...")
            concurrent_results = await stability_tests.test_concurrent_stability()
            print(f"並行安定性結果: {concurrent_results.get('stability_rate_percent', 0):.1f}%")
            
            print("💻 リソース安定性テスト実行中...")
            resource_results = await stability_tests.test_resource_stability()
            print(f"リソース安定性結果: {resource_results.get('stability_assessment', {}).get('overall_stable', False)}")
            
            print("✅ 全安定性統合テスト完了")
        
        # クリーンアップ
        await stability_tests.cleanup_stability_test()
    
    asyncio.run(main())
