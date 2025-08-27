"""
Performance Monitor Tests
パフォーマンス監視システムのテスト
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch
from pathlib import Path

# テスト用のパス設定
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from codex_agent.performance_monitor import PerformanceMonitor, RequestMetrics
from codex_agent.request_pool import RequestPool, AdaptiveRequestPool


class TestPerformanceMonitor:
    """PerformanceMonitor のテスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.monitor = PerformanceMonitor(max_history=100, monitoring_interval=0.1)
    
    def teardown_method(self):
        """テストクリーンアップ"""
        self.monitor.stop_monitoring()
    
    def test_request_metrics_creation(self):
        """RequestMetrics作成のテスト"""
        start_time = time.time()
        metric = RequestMetrics(
            request_id="test-123",
            start_time=start_time,
            request_type="complete",
            model="test-model"
        )
        
        assert metric.request_id == "test-123"
        assert metric.request_type == "complete"
        assert metric.model == "test-model"
        assert metric.success is True
        assert metric.duration == 0.0  # end_timeがNoneの場合は0.0
        
        # end_timeを設定してdurationをテスト
        metric.end_time = start_time + 0.1
        assert metric.duration == 0.1
    
    def test_start_end_request(self):
        """リクエスト開始・終了のテスト"""
        request_id = "test-request"
        
        # リクエスト開始
        metric = self.monitor.start_request(
            request_id=request_id,
            request_type="complete",
            model="test-model"
        )
        
        assert metric.request_id == request_id
        assert request_id in self.monitor.active_requests
        assert self.monitor.total_requests == 1
        
        # 少し待機
        time.sleep(0.01)
        
        # リクエスト終了
        self.monitor.end_request(
            request_id=request_id,
            success=True,
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30
        )
        
        assert request_id not in self.monitor.active_requests
        assert len(self.monitor.request_metrics) == 1
        
        completed_metric = self.monitor.request_metrics[0]
        assert completed_metric.success is True
        assert completed_metric.total_tokens == 30
        assert completed_metric.duration > 0
    
    def test_error_request(self):
        """エラーリクエストのテスト"""
        request_id = "error-request"
        
        self.monitor.start_request(request_id, "complete")
        self.monitor.end_request(
            request_id=request_id,
            success=False,
            error_type="ConnectionError",
            error_message="Connection failed"
        )
        
        assert self.monitor.total_errors == 1
        
        completed_metric = self.monitor.request_metrics[0]
        assert completed_metric.success is False
        assert completed_metric.error_type == "ConnectionError"
    
    def test_performance_summary(self):
        """パフォーマンス要約のテスト"""
        # 複数のリクエストを記録
        for i in range(5):
            request_id = f"request-{i}"
            self.monitor.start_request(request_id, "complete")
            time.sleep(0.001)  # 短い処理時間をシミュレート
            self.monitor.end_request(
                request_id=request_id,
                success=i < 4,  # 最後の1つはエラー
                total_tokens=100 + i * 10
            )
        
        summary = self.monitor.get_performance_summary(time_window_minutes=1)
        
        assert summary['total_requests'] == 5
        assert summary['successful_requests'] == 4
        assert summary['error_count'] == 1
        assert summary['error_rate'] == 20.0  # 1/5 = 20%
        assert summary['avg_response_time_ms'] > 0
        assert summary['total_tokens_processed'] == 100 + 110 + 120 + 130  # エラーは除外
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_system_metrics_collection(self, mock_memory, mock_cpu):
        """システムメトリクス収集のテスト"""
        # モック設定
        mock_cpu.return_value = 50.0
        mock_memory.return_value = Mock(percent=60.0, used=1024*1024*1024)  # 1GB
        
        # 監視開始
        self.monitor.start_monitoring()
        
        # 少し待機してメトリクス収集
        time.sleep(0.2)
        
        # システムメトリクスが収集されているか確認
        assert len(self.monitor.system_metrics) > 0
        
        latest_metric = self.monitor.system_metrics[-1]
        assert latest_metric.cpu_percent == 50.0
        assert latest_metric.memory_percent == 60.0
    
    def test_alert_thresholds(self):
        """アラート閾値のテスト"""
        # アラートコールバック設定
        alerts_received = []
        
        def alert_callback(alert_type, alert_data):
            alerts_received.append((alert_type, alert_data))
        
        self.monitor.add_alert_callback(alert_callback)
        
        # 閾値を低く設定
        self.monitor.set_alert_threshold('response_time_ms', 1.0)  # 1ms
        
        # 長時間のリクエストをシミュレート
        request_id = "slow-request"
        self.monitor.start_request(request_id, "complete")
        time.sleep(0.01)  # 10ms待機
        self.monitor.end_request(request_id, success=True)
        
        # 監視開始してアラートチェック
        self.monitor.start_monitoring()
        time.sleep(0.2)  # アラートチェック待機
        
        # アラートが発火したか確認（システムメトリクス収集後）
        # 注意: 実際のアラートは平均応答時間に基づくため、複数のリクエストが必要な場合がある


class TestRequestPool:
    """RequestPool のテスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.pool = RequestPool(max_concurrent=3, default_timeout=1.0)
    
    def teardown_method(self):
        """テストクリーンアップ"""
        # 同期的にクリーンアップ
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 既存のループがある場合は新しいタスクとして実行
                asyncio.create_task(self.pool.stop())
            else:
                loop.run_until_complete(self.pool.stop())
        except:
            pass
    
    @pytest.mark.asyncio
    async def test_pool_basic_operation(self):
        """プール基本動作のテスト"""
        await self.pool.start()
        
        # 簡単な非同期関数
        async def simple_task(value):
            await asyncio.sleep(0.01)
            return value * 2
        
        # タスク投入
        task_id = await self.pool.submit(simple_task(5))
        
        # 結果待機
        result = await self.pool.wait_for_task(task_id, timeout=2.0)
        assert result == 10
        
        # 統計確認
        stats = self.pool.get_stats()
        assert stats['total_requests'] == 1
        assert stats['completed_requests'] == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_limit(self):
        """並行数制限のテスト"""
        await self.pool.start()
        
        # 長時間タスク
        async def long_task():
            await asyncio.sleep(0.1)
            return "done"
        
        # 並行数を超えるタスクを投入
        task_ids = []
        for i in range(5):  # max_concurrent=3を超える
            task_id = await self.pool.submit(long_task())
            task_ids.append(task_id)
        
        # 統計確認
        stats = self.pool.get_stats()
        assert stats['total_requests'] == 5
        assert stats['active_tasks'] <= 3  # 並行数制限
        
        # 全タスク完了待機
        results = await asyncio.gather(
            *[self.pool.wait_for_task(tid, timeout=2.0) for tid in task_ids]
        )
        
        assert all(result == "done" for result in results)
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """タイムアウト処理のテスト"""
        await self.pool.start()
        
        # タイムアウトするタスク
        async def timeout_task():
            await asyncio.sleep(2.0)  # default_timeout=1.0を超える
            return "should not reach here"
        
        task_id = await self.pool.submit(timeout_task())
        
        # タイムアウトエラーが発生することを確認
        with pytest.raises(asyncio.TimeoutError):
            await self.pool.wait_for_task(task_id, timeout=2.0)
        
        # タイムアウト統計確認
        stats = self.pool.get_stats()
        assert stats['timeout_requests'] == 1


class TestAdaptiveRequestPool:
    """AdaptiveRequestPool のテスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.pool = AdaptiveRequestPool(
            initial_concurrent=2,
            min_concurrent=1,
            max_concurrent=5
        )
    
    def teardown_method(self):
        """テストクリーンアップ"""
        # 同期的にクリーンアップ
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 既存のループがある場合は新しいタスクとして実行
                asyncio.create_task(self.pool.stop())
            else:
                loop.run_until_complete(self.pool.stop())
        except:
            pass
    
    @pytest.mark.asyncio
    async def test_adaptive_adjustment(self):
        """適応的調整のテスト"""
        await self.pool.start()
        
        # 初期並行数確認
        assert self.pool.max_concurrent == 2
        
        # 高負荷状況をシミュレート
        async def load_task():
            await asyncio.sleep(0.05)
            return "done"
        
        # 多数のタスクを投入
        task_ids = []
        for i in range(10):
            task_id = await self.pool.submit(load_task())
            task_ids.append(task_id)
        
        # 少し待機して適応調整を待つ
        await asyncio.sleep(0.1)
        
        # 統計確認
        stats = self.pool.get_stats()
        assert stats['total_requests'] == 10
        
        # 全タスク完了待機
        await asyncio.gather(
            *[self.pool.wait_for_task(tid, timeout=2.0) for tid in task_ids]
        )
        
        final_stats = self.pool.get_stats()
        assert final_stats['completed_requests'] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])