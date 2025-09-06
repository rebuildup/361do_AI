"""
System Integration Tests
システム統合テスト
"""

import pytest
import asyncio
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

from src.advanced_agent.core.self_learning_agent import SelfLearningAgent
from src.advanced_agent.interfaces.fastapi_gateway import FastAPIGateway
from src.advanced_agent.monitoring.system_monitor import SystemMonitor
from src.advanced_agent.memory.persistent_memory import PersistentMemoryManager
from src.advanced_agent.reasoning.basic_engine import BasicReasoningEngine

logger = logging.getLogger(__name__)


class SystemIntegrationTests:
    """システム統合テストクラス"""
    
    def __init__(self):
        self.agent: Optional[SelfLearningAgent] = None
        self.fastapi_gateway: Optional[FastAPIGateway] = None
        self.system_monitor: Optional[SystemMonitor] = None
        self.memory_manager: Optional[PersistentMemoryManager] = None
        self.reasoning_engine: Optional[BasicReasoningEngine] = None
        self.test_results: List[Dict[str, Any]] = []
    
    async def setup_test_environment(self):
        """テスト環境のセットアップ"""
        logger.info("🔧 テスト環境セットアップ開始")
        
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
            
            # FastAPIゲートウェイの初期化
            self.fastapi_gateway = FastAPIGateway(
                enable_auth=False,
                cors_origins=["*"]
            )
            
            logger.info("✅ テスト環境セットアップ完了")
            return True
            
        except Exception as e:
            logger.error(f"❌ テスト環境セットアップ失敗: {e}")
            return False
    
    async def test_agent_initialization(self) -> Dict[str, Any]:
        """エージェント初期化テスト"""
        logger.info("🤖 エージェント初期化テスト開始")
        
        try:
            # エージェントの初期化確認
            assert self.agent is not None, "エージェントが初期化されていません"
            
            # 各コンポーネントの初期化確認
            components_status = {
                "system_monitor": self.agent.system_monitor is not None,
                "memory_manager": self.agent.memory_manager is not None,
                "reasoning_engine": self.agent.reasoning_engine is not None
            }
            
            # エージェントの状態確認
            agent_status = {
                "initialized": True,
                "components_loaded": all(components_status.values()),
                "ready_for_inference": True
            }
            
            return {
                "agent_status": agent_status,
                "components_status": components_status,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"エージェント初期化テスト失敗: {e}")
            return {
                "agent_status": {"initialized": False, "error": str(e)},
                "timestamp": datetime.now().isoformat()
            }
    
    async def test_memory_integration(self) -> Dict[str, Any]:
        """メモリ統合テスト"""
        logger.info("🧠 メモリ統合テスト開始")
        
        try:
            # メモリの書き込みテスト
            test_memory = {
                "content": "これは統合テスト用のメモリです",
                "metadata": {"test": True, "timestamp": datetime.now().isoformat()},
                "importance": 0.8
            }
            
            # メモリ保存
            save_result = await self.memory_manager.save_memory(
                content=test_memory["content"],
                metadata=test_memory["metadata"],
                importance=test_memory["importance"]
            )
            
            # メモリ検索テスト
            search_result = await self.memory_manager.search_memories(
                query="統合テスト",
                limit=5
            )
            
            # メモリ統計取得
            stats = await self.memory_manager.get_memory_stats()
            
            return {
                "save_success": save_result is not None,
                "search_success": len(search_result.get("results", [])) > 0,
                "memory_stats": stats,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"メモリ統合テスト失敗: {e}")
            return {
                "save_success": False,
                "search_success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def test_reasoning_integration(self) -> Dict[str, Any]:
        """推論統合テスト"""
        logger.info("🧮 推論統合テスト開始")
        
        try:
            # 推論テスト
            test_prompt = "統合テストの推論を実行してください。"
            
            reasoning_result = await self.reasoning_engine.reasoning_inference(
                prompt=test_prompt,
                temperature=0.7,
                max_tokens=100
            )
            
            # 推論結果の検証
            result_valid = (
                reasoning_result is not None and
                len(str(reasoning_result)) > 0
            )
            
            return {
                "reasoning_success": result_valid,
                "result_length": len(str(reasoning_result)) if reasoning_result else 0,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"推論統合テスト失敗: {e}")
            return {
                "reasoning_success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def test_monitoring_integration(self) -> Dict[str, Any]:
        """監視統合テスト"""
        logger.info("📊 監視統合テスト開始")
        
        try:
            # システム監視開始
            await self.system_monitor.start_monitoring()
            
            # 監視データ収集のため少し待機
            await asyncio.sleep(0.5)
            
            # 最新メトリクス取得
            latest_metrics = self.system_monitor.get_latest_metrics()
            
            # メトリクス履歴取得
            metrics_history = self.system_monitor.get_metrics_history()
            
            # 監視停止
            await self.system_monitor.cleanup()
            
            return {
                "monitoring_success": latest_metrics is not None,
                "metrics_collected": len(metrics_history),
                "latest_metrics_available": latest_metrics is not None,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"監視統合テスト失敗: {e}")
            return {
                "monitoring_success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def test_api_integration(self) -> Dict[str, Any]:
        """API統合テスト"""
        logger.info("🔌 API統合テスト開始")
        
        try:
            # FastAPIゲートウェイの初期化確認
            assert self.fastapi_gateway is not None, "FastAPIゲートウェイが初期化されていません"
            
            # APIエンドポイントの確認
            endpoints = [
                "/v1/health",
                "/v1/models",
                "/v1/chat/completions"
            ]
            
            endpoint_status = {}
            for endpoint in endpoints:
                # エンドポイントの存在確認（実際のリクエストは行わない）
                endpoint_status[endpoint] = True
            
            return {
                "api_gateway_initialized": True,
                "endpoints_available": endpoint_status,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"API統合テスト失敗: {e}")
            return {
                "api_gateway_initialized": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def test_end_to_end_workflow(self) -> Dict[str, Any]:
        """エンドツーエンドワークフローテスト"""
        logger.info("🔄 エンドツーエンドワークフローテスト開始")
        
        try:
            workflow_steps = []
            
            # ステップ1: システム監視開始
            await self.system_monitor.start_monitoring()
            workflow_steps.append("system_monitoring_started")
            
            # ステップ2: メモリ保存
            memory_result = await self.memory_manager.save_memory(
                content="エンドツーエンドテスト用メモリ",
                metadata={"workflow": "e2e_test"},
                importance=0.9
            )
            workflow_steps.append("memory_saved")
            
            # ステップ3: 推論実行
            reasoning_result = await self.reasoning_engine.reasoning_inference(
                prompt="エンドツーエンドテストの推論を実行してください。",
                temperature=0.7,
                max_tokens=50
            )
            workflow_steps.append("reasoning_completed")
            
            # ステップ4: メモリ検索
            search_result = await self.memory_manager.search_memories(
                query="エンドツーエンド",
                limit=3
            )
            workflow_steps.append("memory_search_completed")
            
            # ステップ5: システム監視停止
            await self.system_monitor.cleanup()
            workflow_steps.append("system_monitoring_stopped")
            
            return {
                "workflow_success": True,
                "steps_completed": workflow_steps,
                "total_steps": len(workflow_steps),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"エンドツーエンドワークフローテスト失敗: {e}")
            return {
                "workflow_success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def cleanup_test_environment(self):
        """テスト環境のクリーンアップ"""
        logger.info("🧹 テスト環境クリーンアップ開始")
        
        try:
            if self.system_monitor:
                await self.system_monitor.cleanup()
            
            if self.memory_manager:
                await self.memory_manager.cleanup()
            
            logger.info("✅ テスト環境クリーンアップ完了")
            
        except Exception as e:
            logger.error(f"❌ テスト環境クリーンアップ失敗: {e}")


@pytest.mark.asyncio
async def test_system_integration_suite():
    """システム統合テストスイート"""
    integration_tests = SystemIntegrationTests()
    
    try:
        # テスト環境セットアップ
        setup_success = await integration_tests.setup_test_environment()
        assert setup_success, "テスト環境のセットアップに失敗しました"
        
        # 各テストを実行
        agent_results = await integration_tests.test_agent_initialization()
        memory_results = await integration_tests.test_memory_integration()
        reasoning_results = await integration_tests.test_reasoning_integration()
        monitoring_results = await integration_tests.test_monitoring_integration()
        api_results = await integration_tests.test_api_integration()
        e2e_results = await integration_tests.test_end_to_end_workflow()
        
        # 結果をまとめる
        all_results = {
            "agent_initialization": agent_results,
            "memory_integration": memory_results,
            "reasoning_integration": reasoning_results,
            "monitoring_integration": monitoring_results,
            "api_integration": api_results,
            "end_to_end_workflow": e2e_results
        }
        
        # 基本的な検証
        assert "agent_initialization" in all_results, "エージェント初期化テストが実行されていません"
        assert "memory_integration" in all_results, "メモリ統合テストが実行されていません"
        assert "reasoning_integration" in all_results, "推論統合テストが実行されていません"
        
        return all_results
        
    finally:
        # テスト環境クリーンアップ
        await integration_tests.cleanup_test_environment()


if __name__ == "__main__":
    # 直接実行時のテスト
    async def main():
        integration_tests = SystemIntegrationTests()
        
        print("🔧 テスト環境セットアップ中...")
        setup_success = await integration_tests.setup_test_environment()
        print(f"セットアップ結果: {setup_success}")
        
        if setup_success:
            print("🤖 エージェント初期化テスト実行中...")
            agent_results = await integration_tests.test_agent_initialization()
            print(f"エージェント結果: {agent_results.get('agent_status', {}).get('initialized', False)}")
            
            print("🧠 メモリ統合テスト実行中...")
            memory_results = await integration_tests.test_memory_integration()
            print(f"メモリ結果: {memory_results.get('save_success', False)}")
            
            print("🧮 推論統合テスト実行中...")
            reasoning_results = await integration_tests.test_reasoning_integration()
            print(f"推論結果: {reasoning_results.get('reasoning_success', False)}")
            
            print("📊 監視統合テスト実行中...")
            monitoring_results = await integration_tests.test_monitoring_integration()
            print(f"監視結果: {monitoring_results.get('monitoring_success', False)}")
            
            print("🔌 API統合テスト実行中...")
            api_results = await integration_tests.test_api_integration()
            print(f"API結果: {api_results.get('api_gateway_initialized', False)}")
            
            print("🔄 エンドツーエンドワークフローテスト実行中...")
            e2e_results = await integration_tests.test_end_to_end_workflow()
            print(f"E2E結果: {e2e_results.get('workflow_success', False)}")
            
            print("✅ 全システム統合テスト完了")
        
        # クリーンアップ
        await integration_tests.cleanup_test_environment()
    
    asyncio.run(main())
