"""
復旧システムのテスト
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from src.advanced_agent.monitoring.recovery_system import (
    SystemRecoveryManager,
    RecoveryStatus,
    RecoveryStrategy,
    RecoveryPlan,
    RecoveryExecution
)
from src.advanced_agent.monitoring.anomaly_detector import (
    DetectedAnomaly,
    AnomalyPattern,
    AnomalyType,
    SeverityLevel
)


class TestSystemRecoveryManager:
    """システム復旧管理のテスト"""
    
    @pytest.fixture
    def recovery_manager(self):
        """テスト用復旧管理"""
        return SystemRecoveryManager(
            grafana_url="http://localhost:3000",
            max_concurrent_recoveries=2
        )
    
    @pytest.fixture
    def test_anomaly(self):
        """テスト用異常"""
        pattern = AnomalyPattern(
            pattern_id="gpu_memory_high",
            name="GPU Memory High Usage",
            description="GPU メモリ使用率が異常に高い",
            anomaly_type=AnomalyType.GPU_MEMORY_HIGH,
            severity=SeverityLevel.HIGH,
            detection_query='gpu_memory_usage_percent > 90',
            threshold=90.0,
            recovery_actions=["increase_quantization", "clear_cache"]
        )
        
        return DetectedAnomaly(
            anomaly_id="test_anomaly_001",
            pattern=pattern,
            detected_at=datetime.now(),
            current_value=95.5,
            threshold=90.0,
            severity=SeverityLevel.HIGH
        )
    
    def test_initialize_action_handlers(self, recovery_manager):
        """アクションハンドラー初期化テスト"""
        handlers = recovery_manager.action_handlers
        
        assert "increase_quantization" in handlers
        assert "clear_cache" in handlers
        assert "offload_to_cpu" in handlers
        assert "reduce_gpu_load" in handlers
        
        # ハンドラーが呼び出し可能であることを確認
        assert callable(handlers["increase_quantization"])
    
    @pytest.mark.asyncio
    async def test_create_recovery_plan(self, recovery_manager, test_anomaly):
        """復旧計画作成テスト"""
        plan = await recovery_manager.create_recovery_plan(test_anomaly)
        
        assert plan.anomaly == test_anomaly
        assert plan.strategy == RecoveryStrategy.GRADUAL  # HIGH severity
        assert len(plan.actions) > 0
        assert plan.estimated_duration > 0
        assert plan.rollback_plan is not None
    
    @pytest.mark.asyncio
    async def test_select_recovery_actions(self, recovery_manager, test_anomaly):
        """復旧アクション選択テスト"""
        actions = await recovery_manager._select_recovery_actions(test_anomaly)
        
        assert len(actions) == 2  # increase_quantization, clear_cache
        
        action_ids = [action.action_id for action in actions]
        assert "increase_quantization" in action_ids
        assert "clear_cache" in action_ids
    
    @pytest.mark.asyncio
    async def test_create_rollback_plan(self, recovery_manager):
        """ロールバック計画作成テスト"""
        from src.advanced_agent.monitoring.recovery_system import RecoveryAction
        
        actions = [
            RecoveryAction(
                action_id="increase_quantization",
                name="Increase Quantization",
                description="Test action",
                action_type="model",
                priority=1,
                estimated_duration=30,
                rollback_possible=True
            ),
            RecoveryAction(
                action_id="clear_cache",
                name="Clear Cache",
                description="Test action",
                action_type="system",
                priority=2,
                estimated_duration=10,
                rollback_possible=False
            )
        ]
        
        rollback_plan = await recovery_manager._create_rollback_plan(actions)
        
        # ロールバック可能なアクションのみが含まれる（逆順）
        assert len(rollback_plan) == 1
        assert rollback_plan[0] == "rollback_increase_quantization"
    
    @pytest.mark.asyncio
    async def test_execute_recovery_plan_success(self, recovery_manager, test_anomaly):
        """復旧計画実行成功テスト"""
        plan = await recovery_manager.create_recovery_plan(test_anomaly)
        
        # アクション実行をモック
        with patch.object(recovery_manager, '_execute_gradual') as mock_execute:
            mock_execute.return_value = None
            
            with patch.object(recovery_manager, '_capture_metrics') as mock_metrics:
                mock_metrics.return_value = {"test": "metrics"}
                
                execution = await recovery_manager.execute_recovery_plan(plan)
                
                assert execution.status == RecoveryStatus.SUCCESS
                assert execution.completed_at is not None
                assert execution.metrics_before == {"test": "metrics"}
                assert execution.metrics_after == {"test": "metrics"}
    
    @pytest.mark.asyncio
    async def test_execute_recovery_plan_failure(self, recovery_manager, test_anomaly):
        """復旧計画実行失敗テスト"""
        plan = await recovery_manager.create_recovery_plan(test_anomaly)
        
        # アクション実行で例外発生をシミュレート
        with patch.object(recovery_manager, '_execute_gradual') as mock_execute:
            mock_execute.side_effect = Exception("Test failure")
            
            with patch.object(recovery_manager, '_capture_metrics', return_value={}):
                with patch.object(recovery_manager, '_execute_rollback') as mock_rollback:
                    execution = await recovery_manager.execute_recovery_plan(plan)
                    
                    assert execution.status == RecoveryStatus.FAILED
                    assert len(execution.logs) > 0
                    mock_rollback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_immediate_strategy(self, recovery_manager, test_anomaly):
        """即座実行戦略テスト"""
        plan = await recovery_manager.create_recovery_plan(test_anomaly)
        plan.strategy = RecoveryStrategy.IMMEDIATE
        
        execution = RecoveryExecution(
            execution_id="test_exec",
            plan=plan,
            started_at=datetime.now()
        )
        
        with patch.object(recovery_manager, '_execute_action') as mock_action:
            mock_action.return_value = None
            
            await recovery_manager._execute_immediate(execution)
            
            # 全アクションが実行される
            assert mock_action.call_count == len(plan.actions)
            assert len(execution.executed_actions) == len(plan.actions)
    
    @pytest.mark.asyncio
    async def test_execute_gradual_strategy(self, recovery_manager, test_anomaly):
        """段階的実行戦略テスト"""
        plan = await recovery_manager.create_recovery_plan(test_anomaly)
        
        execution = RecoveryExecution(
            execution_id="test_exec",
            plan=plan,
            started_at=datetime.now()
        )
        
        with patch.object(recovery_manager, '_execute_action') as mock_action:
            mock_action.return_value = None
            
            with patch.object(recovery_manager, '_check_improvement', return_value=False):
                with patch('asyncio.sleep'):  # sleep をスキップ
                    await recovery_manager._execute_gradual(execution)
                    
                    assert mock_action.call_count == len(plan.actions)
    
    @pytest.mark.asyncio
    async def test_execute_conservative_strategy(self, recovery_manager, test_anomaly):
        """保守的実行戦略テスト"""
        plan = await recovery_manager.create_recovery_plan(test_anomaly)
        
        execution = RecoveryExecution(
            execution_id="test_exec",
            plan=plan,
            started_at=datetime.now()
        )
        
        with patch.object(recovery_manager, '_execute_action') as mock_action:
            mock_action.return_value = None
            
            with patch.object(recovery_manager, '_pre_action_check', return_value=True):
                with patch('asyncio.sleep'):  # sleep をスキップ
                    await recovery_manager._execute_conservative(execution)
                    
                    assert mock_action.call_count == len(plan.actions)
    
    @pytest.mark.asyncio
    async def test_increase_quantization_action(self, recovery_manager, test_anomaly):
        """量子化レベル上昇アクションテスト"""
        mock_gpu_stats = {"memory_percent": 95.0}
        
        with patch.object(recovery_manager.system_monitor, 'get_gpu_stats', return_value=mock_gpu_stats):
            with patch('torch.cuda.is_available', return_value=True):
                with patch('torch.cuda.empty_cache') as mock_cache:
                    result = await recovery_manager._increase_quantization(test_anomaly)
                    
                    assert result["action"] == "quantization_increased"
                    assert result["level"] == "4bit"
                    mock_cache.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_clear_gpu_cache_action(self, recovery_manager, test_anomaly):
        """GPU キャッシュクリアアクションテスト"""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.empty_cache') as mock_cache:
                with patch('torch.cuda.synchronize') as mock_sync:
                    with patch('gc.collect', return_value=10) as mock_gc:
                        result = await recovery_manager._clear_gpu_cache(test_anomaly)
                        
                        assert result["action"] == "cache_cleared"
                        mock_cache.assert_called_once()
                        mock_sync.assert_called_once()
                        mock_gc.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_offload_to_cpu_action(self, recovery_manager, test_anomaly):
        """CPU オフロードアクションテスト"""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.empty_cache') as mock_cache:
                with patch('torch.cuda.synchronize') as mock_sync:
                    result = await recovery_manager._offload_to_cpu(test_anomaly)
                    
                    assert result["action"] == "cpu_offload_enabled"
                    mock_cache.assert_called_once()
                    mock_sync.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_garbage_collection_action(self, recovery_manager, test_anomaly):
        """ガベージコレクションアクションテスト"""
        with patch('gc.collect', return_value=15) as mock_gc:
            with patch('torch.cuda.is_available', return_value=True):
                with patch('torch.cuda.empty_cache') as mock_cache:
                    result = await recovery_manager._garbage_collection(test_anomaly)
                    
                    assert result["action"] == "garbage_collected"
                    assert result["objects_collected"] == 15
                    mock_gc.assert_called_once()
                    mock_cache.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_capture_metrics(self, recovery_manager):
        """メトリクス取得テスト"""
        mock_system_stats = {"cpu_percent": 75.0}
        mock_gpu_stats = {"memory_percent": 85.0}
        
        with patch.object(recovery_manager.system_monitor, 'get_system_stats', return_value=mock_system_stats):
            with patch.object(recovery_manager.system_monitor, 'get_gpu_stats', return_value=mock_gpu_stats):
                metrics = await recovery_manager._capture_metrics()
                
                assert "timestamp" in metrics
                assert metrics["system"] == mock_system_stats
                assert metrics["gpu"] == mock_gpu_stats
    
    @pytest.mark.asyncio
    async def test_check_improvement(self, recovery_manager, test_anomaly):
        """改善確認テスト"""
        plan = await recovery_manager.create_recovery_plan(test_anomaly)
        execution = RecoveryExecution(
            execution_id="test_exec",
            plan=plan,
            started_at=datetime.now()
        )
        
        # 改善された場合
        mock_metrics = {"gpu": {"memory_percent": 85.0}}  # 閾値以下
        with patch.object(recovery_manager, '_capture_metrics', return_value=mock_metrics):
            improvement = await recovery_manager._check_improvement(execution)
            assert improvement is True
        
        # 改善されていない場合
        mock_metrics = {"gpu": {"memory_percent": 95.0}}  # 閾値以上
        with patch.object(recovery_manager, '_capture_metrics', return_value=mock_metrics):
            improvement = await recovery_manager._check_improvement(execution)
            assert improvement is False
    
    @pytest.mark.asyncio
    async def test_pre_action_check(self, recovery_manager):
        """事前アクションチェックテスト"""
        from src.advanced_agent.monitoring.recovery_system import RecoveryAction
        
        action = RecoveryAction(
            action_id="test_action",
            name="Test Action",
            description="Test",
            action_type="system",
            priority=1,
            estimated_duration=30
        )
        
        execution = Mock()
        
        # CPU 使用率が低い場合（実行可能）
        mock_metrics = {"system": {"cpu_percent": 80.0}}
        with patch.object(recovery_manager, '_capture_metrics', return_value=mock_metrics):
            result = await recovery_manager._pre_action_check(action, execution)
            assert result is True
        
        # CPU 使用率が高い場合（実行不可）
        mock_metrics = {"system": {"cpu_percent": 98.0}}
        with patch.object(recovery_manager, '_capture_metrics', return_value=mock_metrics):
            result = await recovery_manager._pre_action_check(action, execution)
            assert result is False
    
    @pytest.mark.asyncio
    async def test_execute_rollback(self, recovery_manager, test_anomaly):
        """ロールバック実行テスト"""
        plan = await recovery_manager.create_recovery_plan(test_anomaly)
        plan.rollback_plan = ["rollback_action_1", "rollback_action_2"]
        
        execution = RecoveryExecution(
            execution_id="test_exec",
            plan=plan,
            started_at=datetime.now()
        )
        
        with patch('asyncio.sleep'):  # sleep をスキップ
            await recovery_manager._execute_rollback(execution)
            
            assert execution.status == RecoveryStatus.ROLLED_BACK
            assert len(execution.rollback_actions) == 2
            assert "rollback_action_1" in execution.rollback_actions
            assert "rollback_action_2" in execution.rollback_actions
    
    def test_get_recovery_status(self, recovery_manager):
        """復旧ステータス取得テスト"""
        # テスト用実行履歴を追加
        test_execution = Mock()
        test_execution.execution_id = "test_exec"
        test_execution.status = RecoveryStatus.SUCCESS
        test_execution.started_at = datetime.now()
        test_execution.plan.anomaly.pattern.anomaly_type = AnomalyType.GPU_MEMORY_HIGH
        
        recovery_manager.recovery_history = [test_execution]
        
        status = recovery_manager.get_recovery_status()
        
        assert status["active_recoveries"] == 0
        assert status["total_recoveries"] == 1
        assert status["success_rate"] == 100.0
        assert len(status["recent_recoveries"]) == 1
    
    def test_calculate_success_rate(self, recovery_manager):
        """成功率計算テスト"""
        # 空の履歴
        assert recovery_manager._calculate_success_rate() == 0.0
        
        # 成功と失敗の混合
        success_exec = Mock()
        success_exec.status = RecoveryStatus.SUCCESS
        
        failed_exec = Mock()
        failed_exec.status = RecoveryStatus.FAILED
        
        recovery_manager.recovery_history = [success_exec, failed_exec, success_exec]
        
        success_rate = recovery_manager._calculate_success_rate()
        assert success_rate == 66.66666666666666  # 2/3 * 100
    
    @pytest.mark.asyncio
    async def test_concurrent_recovery_limit(self, recovery_manager, test_anomaly):
        """同時復旧制限テスト"""
        recovery_manager.max_concurrent_recoveries = 1
        
        plan1 = await recovery_manager.create_recovery_plan(test_anomaly)
        plan2 = await recovery_manager.create_recovery_plan(test_anomaly)
        
        # 最初の復旧を開始（完了させない）
        recovery_manager.active_recoveries["test1"] = Mock()
        
        # 2番目の復旧は制限により失敗するはず
        with pytest.raises(RuntimeError, match="最大同時復旧数に達しています"):
            await recovery_manager.execute_recovery_plan(plan2)


if __name__ == "__main__":
    pytest.main([__file__])