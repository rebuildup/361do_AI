"""
異常検出システムのテスト
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.advanced_agent.monitoring.anomaly_detector import (
    PrometheusAnomalyDetector,
    AnomalyType,
    SeverityLevel,
    AnomalyPattern,
    DetectedAnomaly
)


class TestPrometheusAnomalyDetector:
    """Prometheus 異常検出システムのテスト"""
    
    @pytest.fixture
    def detector(self):
        """テスト用検出器"""
        return PrometheusAnomalyDetector(
            prometheus_url="http://localhost:9090",
            grafana_url="http://localhost:3000"
        )
    
    @pytest.fixture
    def mock_prometheus_response(self):
        """モック Prometheus レスポンス"""
        return {
            "status": "success",
            "data": {
                "result": [
                    {
                        "value": [1640995200, "95.5"]  # 95.5% GPU memory usage
                    }
                ]
            }
        }
    
    def test_initialize_patterns(self, detector):
        """異常パターン初期化テスト"""
        patterns = detector.anomaly_patterns
        
        assert "gpu_memory_high" in patterns
        assert "gpu_temperature_high" in patterns
        assert "inference_slow" in patterns
        assert "memory_leak" in patterns
        
        gpu_pattern = patterns["gpu_memory_high"]
        assert gpu_pattern.anomaly_type == AnomalyType.GPU_MEMORY_HIGH
        assert gpu_pattern.severity == SeverityLevel.HIGH
        assert gpu_pattern.threshold == 90.0
    
    @pytest.mark.asyncio
    async def test_execute_prometheus_query_success(self, detector, mock_prometheus_response):
        """Prometheus クエリ成功テスト"""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_prometheus_response)
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await detector._execute_prometheus_query("gpu_memory_usage_percent")
            
            assert result == 95.5
            mock_get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_prometheus_query_failure(self, detector):
        """Prometheus クエリ失敗テスト"""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await detector._execute_prometheus_query("invalid_query")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_check_pattern_anomaly_detected(self, detector):
        """異常検出テスト"""
        pattern = detector.anomaly_patterns["gpu_memory_high"]
        
        with patch.object(detector, '_execute_prometheus_query', return_value=95.5):
            with patch.object(detector, '_gather_context', return_value={}):
                anomaly = await detector._check_pattern(pattern)
                
                assert anomaly is not None
                assert anomaly.pattern == pattern
                assert anomaly.current_value == 95.5
                assert anomaly.threshold == 90.0
                assert not anomaly.resolved
    
    @pytest.mark.asyncio
    async def test_check_pattern_no_anomaly(self, detector):
        """正常状態テスト"""
        pattern = detector.anomaly_patterns["gpu_memory_high"]
        
        with patch.object(detector, '_execute_prometheus_query', return_value=85.0):
            anomaly = await detector._check_pattern(pattern)
            
            assert anomaly is None
    
    @pytest.mark.asyncio
    async def test_detect_anomalies(self, detector):
        """異常検出実行テスト"""
        # GPU メモリ高使用率をシミュレート
        query_results = {
            'gpu_memory_usage_percent > 90': 95.5,
            'gpu_temperature_celsius > 85': 82.0,  # 正常
            'inference_duration_seconds > 10': 15.0,  # 異常
            'increase(system_memory_used_bytes[30m]) > 1073741824': 500000000  # 正常
        }
        
        async def mock_query(query):
            return query_results.get(query, 0.0)
        
        with patch.object(detector, '_execute_prometheus_query', side_effect=mock_query):
            with patch.object(detector, '_gather_context', return_value={}):
                anomalies = await detector.detect_anomalies()
                
                # GPU メモリと推論速度の異常が検出されるはず
                assert len(anomalies) == 2
                
                anomaly_types = [a.pattern.anomaly_type for a in anomalies]
                assert AnomalyType.GPU_MEMORY_HIGH in anomaly_types
                assert AnomalyType.INFERENCE_SLOW in anomaly_types
    
    @pytest.mark.asyncio
    async def test_gather_context(self, detector):
        """コンテキスト収集テスト"""
        pattern = detector.anomaly_patterns["gpu_memory_high"]
        
        mock_system_stats = {"cpu_percent": 75.0, "memory_percent": 60.0}
        mock_gpu_stats = {"memory_percent": 95.0, "temperature": 78.0}
        mock_related_metrics = {"gpu_utilization": 90.0}
        
        with patch.object(detector.system_monitor, 'get_system_stats', return_value=mock_system_stats):
            with patch.object(detector.system_monitor, 'get_gpu_stats', return_value=mock_gpu_stats):
                with patch.object(detector, '_get_related_metrics', return_value=mock_related_metrics):
                    context = await detector._gather_context(pattern)
                    
                    assert "system_stats" in context
                    assert "gpu_stats" in context
                    assert "related_metrics" in context
                    assert context["system_stats"] == mock_system_stats
                    assert context["gpu_stats"] == mock_gpu_stats
    
    @pytest.mark.asyncio
    async def test_get_related_metrics_gpu_memory(self, detector):
        """GPU メモリ関連メトリクス取得テスト"""
        query_results = {
            "gpu_utilization_percent": 90.0,
            "model_memory_usage_bytes": 4000000000,
            "gpu_cache_memory_bytes": 500000000
        }
        
        async def mock_query(query):
            return query_results.get(query, 0.0)
        
        with patch.object(detector, '_execute_prometheus_query', side_effect=mock_query):
            metrics = await detector._get_related_metrics(AnomalyType.GPU_MEMORY_HIGH)
            
            assert "gpu_utilization" in metrics
            assert "model_memory" in metrics
            assert "cache_memory" in metrics
            assert metrics["gpu_utilization"] == 90.0
    
    @pytest.mark.asyncio
    async def test_send_alert(self, detector):
        """アラート送信テスト"""
        detector.grafana_api_key = "test_api_key"
        
        anomaly = DetectedAnomaly(
            anomaly_id="test_anomaly",
            pattern=detector.anomaly_patterns["gpu_memory_high"],
            detected_at=datetime.now(),
            current_value=95.5,
            threshold=90.0,
            severity=SeverityLevel.HIGH
        )
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_post.return_value.__aenter__.return_value = mock_response
            
            await detector._send_alert(anomaly)
            
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "api/alerts" in call_args[1]["url"]
            assert "Bearer test_api_key" in call_args[1]["headers"]["Authorization"]
    
    def test_get_anomaly_summary(self, detector):
        """異常サマリー取得テスト"""
        # テスト用異常を追加
        test_anomaly = DetectedAnomaly(
            anomaly_id="test_anomaly",
            pattern=detector.anomaly_patterns["gpu_memory_high"],
            detected_at=datetime.now(),
            current_value=95.5,
            threshold=90.0,
            severity=SeverityLevel.HIGH
        )
        
        detector.detected_anomalies["test_anomaly"] = test_anomaly
        detector.active_anomalies["gpu_memory_high"] = test_anomaly
        
        summary = detector.get_anomaly_summary()
        
        assert summary["total_patterns"] == len(detector.anomaly_patterns)
        assert summary["active_anomalies"] == 1
        assert summary["total_detected"] == 1
        assert summary["anomaly_types"][AnomalyType.GPU_MEMORY_HIGH.value] == 1
        assert summary["severity_distribution"][SeverityLevel.HIGH.value] == 1
    
    @pytest.mark.asyncio
    async def test_anomaly_resolution(self, detector):
        """異常解決テスト"""
        pattern = detector.anomaly_patterns["gpu_memory_high"]
        
        # 最初に異常を検出
        with patch.object(detector, '_execute_prometheus_query', return_value=95.5):
            with patch.object(detector, '_gather_context', return_value={}):
                anomaly = await detector._check_pattern(pattern)
                assert anomaly is not None
                assert pattern.pattern_id in detector.active_anomalies
        
        # 次に正常値を返して解決をテスト
        with patch.object(detector, '_execute_prometheus_query', return_value=85.0):
            result = await detector._check_pattern(pattern)
            assert result is None
            assert pattern.pattern_id not in detector.active_anomalies
            
            # 解決された異常は履歴に残る
            resolved_anomaly = detector.detected_anomalies[anomaly.anomaly_id]
            assert resolved_anomaly.resolved
            assert resolved_anomaly.resolved_at is not None


@pytest.mark.asyncio
async def test_continuous_monitoring():
    """継続的監視テスト"""
    detector = PrometheusAnomalyDetector()
    
    # 短時間の監視をテスト
    monitoring_task = asyncio.create_task(detector.start_monitoring(interval_seconds=1))
    
    # 2秒後に停止
    await asyncio.sleep(2)
    monitoring_task.cancel()
    
    try:
        await monitoring_task
    except asyncio.CancelledError:
        pass  # 期待される例外


if __name__ == "__main__":
    pytest.main([__file__])