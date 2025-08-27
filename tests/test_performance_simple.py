"""
Simple Performance Monitor Tests
パフォーマンス監視システムの簡単なテスト
"""

import pytest
import time
from pathlib import Path

# テスト用のパス設定
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from codex_agent.performance_monitor import PerformanceMonitor, RequestMetrics


class TestPerformanceMonitorSimple:
    """PerformanceMonitor の簡単なテスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.monitor = PerformanceMonitor(max_history=100, monitoring_interval=0.1)
    
    def teardown_method(self):
        """テストクリーンアップ"""
        self.monitor.stop_monitoring()
    
    def test_request_lifecycle(self):
        """リクエストライフサイクルのテスト"""
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
    
    def test_performance_summary_basic(self):
        """基本的なパフォーマンス要約のテスト"""
        # 成功リクエスト
        self.monitor.start_request("req1", "complete")
        time.sleep(0.001)
        self.monitor.end_request("req1", success=True, total_tokens=100)
        
        # エラーリクエスト
        self.monitor.start_request("req2", "complete")
        time.sleep(0.001)
        self.monitor.end_request("req2", success=False, error_type="TestError")
        
        summary = self.monitor.get_performance_summary(time_window_minutes=1)
        
        assert summary['total_requests'] == 2
        assert summary['successful_requests'] == 1
        assert summary['error_count'] == 1
        assert summary['error_rate'] == 50.0
        assert summary['avg_response_time_ms'] > 0
        assert summary['total_tokens_processed'] == 100
    
    def test_alert_threshold_setting(self):
        """アラート閾値設定のテスト"""
        # 初期閾値確認
        assert 'response_time_ms' in self.monitor.alert_thresholds
        
        # 閾値変更
        self.monitor.set_alert_threshold('response_time_ms', 5000.0)
        assert self.monitor.alert_thresholds['response_time_ms'] == 5000.0
        
        # アラートコールバック追加
        alerts = []
        def test_callback(alert_type, alert_data):
            alerts.append((alert_type, alert_data))
        
        self.monitor.add_alert_callback(test_callback)
        assert len(self.monitor.alert_callbacks) == 1
    
    def test_request_history(self):
        """リクエスト履歴のテスト"""
        # 複数のリクエストを実行
        for i in range(3):
            request_id = f"req-{i}"
            self.monitor.start_request(request_id, "complete", model=f"model-{i}")
            time.sleep(0.001)
            self.monitor.end_request(request_id, success=True, total_tokens=100 + i)
        
        # 履歴取得
        history = self.monitor.get_request_history(limit=5)
        
        assert len(history) == 3
        assert all('request_id' in item for item in history)
        assert all('duration_ms' in item for item in history)
        assert all(item['success'] for item in history)
        
        # 最新のリクエストが最後に来ることを確認
        assert history[-1]['request_id'] == "req-2"
        assert history[-1]['total_tokens'] == 102


if __name__ == "__main__":
    pytest.main([__file__, "-v"])