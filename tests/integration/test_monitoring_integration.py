"""
監視システム統合テスト

Prometheus + Grafana 監視・最適化システムの統合テスト
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from unittest.mock import Mock, patch

# テスト対象をインポート
from src.advanced_agent.monitoring.grafana_dashboard import (
    GrafanaDashboardManager,
    AnomalyDetector,
    AutoRecoverySystem,
    create_monitoring_system
)


@pytest.fixture
def temp_dir():
    """テスト用一時ディレクトリ"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_grafana_response():
    """モック Grafana レスポンス"""
    return {
        "status_code": 200,
        "json": {
            "id": 1,
            "slug": "advanced-agent-dashboard",
            "status": "success",
            "uid": "advanced-agent-uid",
            "url": "/d/advanced-agent-uid/advanced-agent-dashboard",
            "version": 1
        }
    }


@pytest.fixture
def mock_prometheus_response():
    """モック Prometheus レスポンス"""
    return {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": [
                {
                    "metric": {"__name__": "cpu_usage_percent", "instance": "localhost:8001"},
                    "value": [1640995200, "75.5"]
                }
            ]
        }
    }


class TestGrafanaDashboardManager:
    """Grafana ダッシュボード管理テスト"""
    
    def test_dashboard_manager_initialization(self):
        """ダッシュボード管理システム初期化テスト"""
        
        manager = GrafanaDashboardManager(
            grafana_url="http://localhost:3000",
            api_key="test-api-key"
        )
        
        assert manager.grafana_url == "http://localhost:3000"
        assert manager.api_key == "test-api-key"
        assert "advanced_agent_dashboard" in manager.dashboard_config
    
    def test_dashboard_config_loading(self):
        """ダッシュボード設定読み込みテスト"""
        
        manager = GrafanaDashboardManager()
        config = manager.dashboard_config["advanced_agent_dashboard"]
        
        assert config["title"] == "Advanced AI Agent - System Monitoring"
        assert "ai-agent" in config["tags"]
        assert config["refresh"] == "5s"
        assert len(config["panels"]) > 0
        
        # パネル設定の確認
        panel = config["panels"][0]
        assert "title" in panel
        assert "type" in panel
        assert "targets" in panel
    
    def test_dashboard_json_building(self):
        """ダッシュボード JSON 構築テスト"""
        
        manager = GrafanaDashboardManager()
        config = manager.dashboard_config["advanced_agent_dashboard"]
        
        dashboard_json = manager._build_dashboard_json(config)
        
        assert dashboard_json["title"] == config["title"]
        assert dashboard_json["tags"] == config["tags"]
        assert dashboard_json["refresh"] == config["refresh"]
        assert len(dashboard_json["panels"]) == len(config["panels"])
        
        # パネル構造の確認
        panel = dashboard_json["panels"][0]
        assert "id" in panel
        assert "title" in panel
        assert "type" in panel
        assert "gridPos" in panel
        assert "targets" in panel
    
    @patch('requests.post')
    def test_dashboard_creation_with_api(self, mock_post, mock_grafana_response):
        """API を使用したダッシュボード作成テスト"""
        
        # モックレスポンス設定
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_grafana_response["json"]
        mock_post.return_value = mock_response
        
        manager = GrafanaDashboardManager(api_key="test-key")
        result = manager.create_dashboard("advanced_agent_dashboard")
        
        assert result is True
        mock_post.assert_called_once()
        
        # API 呼び出しの確認
        call_args = mock_post.call_args
        assert "dashboard" in call_args[1]["json"]
        assert call_args[1]["json"]["overwrite"] is True
    
    def test_dashboard_creation_without_api(self, temp_dir):
        """API なしでのダッシュボード作成テスト"""
        
        # 一時ディレクトリに移動
        import os
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            manager = GrafanaDashboardManager(api_key=None)
            result = manager.create_dashboard("advanced_agent_dashboard")
            
            assert result is True
            
            # ファイルが作成されていることを確認
            dashboard_file = temp_dir / "grafana_dashboard_advanced_agent_dashboard.json"
            assert dashboard_file.exists()
            
            # ファイル内容の確認
            with open(dashboard_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert "dashboard" in data
            assert data["overwrite"] is True
            
        finally:
            os.chdir(original_cwd)
    
    def test_alert_rules_creation(self, temp_dir):
        """アラートルール作成テスト"""
        
        # 一時ディレクトリに移動
        import os
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            manager = GrafanaDashboardManager()
            result = manager.create_alert_rules()
            
            assert result is True
            
            # アラートファイルが作成されていることを確認
            alert_file = temp_dir / "prometheus_alerts.yml"
            assert alert_file.exists()
            
            # ファイル内容の確認
            import yaml
            with open(alert_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            assert "groups" in data
            assert len(data["groups"]) > 0
            assert "rules" in data["groups"][0]
            
            # アラートルールの確認
            rules = data["groups"][0]["rules"]
            alert_names = [rule["alert"] for rule in rules]
            
            expected_alerts = ["HighCPUUsage", "HighMemoryUsage", "GPUMemoryExhaustion"]
            for expected in expected_alerts:
                assert expected in alert_names
            
        finally:
            os.chdir(original_cwd)
    
    def test_dashboard_url_generation(self):
        """ダッシュボード URL 生成テスト"""
        
        manager = GrafanaDashboardManager(grafana_url="http://localhost:3000")
        
        url = manager.get_dashboard_url("advanced_agent_dashboard")
        expected_url = "http://localhost:3000/d/advanced-agent-dashboard/advanced-agent-dashboard"
        
        assert url == expected_url
        
        # スペースを含む名前のテスト
        url = manager.get_dashboard_url("My Dashboard Name")
        expected_url = "http://localhost:3000/d/my-dashboard-name/my-dashboard-name"
        
        assert url == expected_url


class TestAnomalyDetector:
    """異常検出システムテスト"""
    
    def test_anomaly_detector_initialization(self):
        """異常検出システム初期化テスト"""
        
        detector = AnomalyDetector(prometheus_url="http://localhost:9090")
        
        assert detector.prometheus_url == "http://localhost:9090"
        assert "cpu_usage_percent" in detector.thresholds
        assert "memory_usage_percent" in detector.thresholds
        assert "gpu_memory_percent" in detector.thresholds
    
    def test_metric_status_evaluation(self):
        """メトリクスステータス評価テスト"""
        
        detector = AnomalyDetector()
        thresholds = {"warning": 80, "critical": 95}
        
        # 正常値
        status = detector._evaluate_metric_status(50, thresholds)
        assert status == "healthy"
        
        # 警告値
        status = detector._evaluate_metric_status(85, thresholds)
        assert status == "warning"
        
        # 危険値
        status = detector._evaluate_metric_status(98, thresholds)
        assert status == "critical"
    
    @patch('requests.get')
    def test_metric_value_retrieval(self, mock_get, mock_prometheus_response):
        """メトリクス値取得テスト"""
        
        # モックレスポンス設定
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_prometheus_response
        mock_get.return_value = mock_response
        
        detector = AnomalyDetector()
        value = detector._get_current_metric_value("cpu_usage_percent")
        
        assert value == 75.5
        mock_get.assert_called_once()
    
    @patch('requests.get')
    def test_metric_value_retrieval_failure(self, mock_get):
        """メトリクス値取得失敗テスト"""
        
        # エラーレスポンス設定
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        detector = AnomalyDetector()
        value = detector._get_current_metric_value("cpu_usage_percent")
        
        assert value is None
    
    def test_recommendations_generation(self):
        """推奨事項生成テスト"""
        
        detector = AnomalyDetector()
        
        alerts = [
            {"metric": "cpu_usage_percent", "severity": "critical", "value": 98},
            {"metric": "memory_usage_percent", "severity": "warning", "value": 87},
            {"metric": "gpu_memory_percent", "severity": "critical", "value": 99}
        ]
        
        recommendations = detector._generate_recommendations(alerts)
        
        assert len(recommendations) == 3
        assert any("CPU使用率が危険レベル" in rec for rec in recommendations)
        assert any("メモリ使用量が増加" in rec for rec in recommendations)
        assert any("GPU メモリが不足" in rec for rec in recommendations)
    
    @patch.object(AnomalyDetector, '_get_current_metric_value')
    def test_system_health_check(self, mock_get_metric):
        """システムヘルスチェックテスト"""
        
        # メトリクス値をモック
        metric_values = {
            "cpu_usage_percent": 85,
            "memory_usage_percent": 70,
            "gpu_memory_percent": 95,
            "inference_time_seconds": 3,
            "error_rate": 0.02
        }
        
        def mock_metric_side_effect(metric_name):
            return metric_values.get(metric_name)
        
        mock_get_metric.side_effect = mock_metric_side_effect
        
        detector = AnomalyDetector()
        health_status = detector.check_system_health()
        
        assert health_status["overall_status"] in ["healthy", "warning", "critical"]
        assert "metrics" in health_status
        assert "alerts" in health_status
        assert "recommendations" in health_status
        
        # 警告アラートの確認
        alerts = health_status["alerts"]
        alert_metrics = [alert["metric"] for alert in alerts]
        
        assert "cpu_usage_percent" in alert_metrics  # 85 > 80 (warning)
        assert "gpu_memory_percent" in alert_metrics  # 95 > 90 (warning)


class TestAutoRecoverySystem:
    """自動復旧システムテスト"""
    
    def test_recovery_system_initialization(self):
        """自動復旧システム初期化テスト"""
        
        recovery_system = AutoRecoverySystem()
        
        assert len(recovery_system.recovery_strategies) > 0
        assert "high_cpu" in recovery_system.recovery_strategies
        assert "high_memory" in recovery_system.recovery_strategies
        assert "gpu_memory_exhaustion" in recovery_system.recovery_strategies
        assert len(recovery_system.recovery_history) == 0
    
    def test_recovery_strategy_determination(self):
        """復旧戦略決定テスト"""
        
        recovery_system = AutoRecoverySystem()
        
        # CPU 使用率高
        strategy = recovery_system._determine_recovery_strategy("cpu_usage_percent", "critical")
        assert strategy == "high_cpu"
        
        # メモリ使用率高
        strategy = recovery_system._determine_recovery_strategy("memory_usage_percent", "warning")
        assert strategy == "high_memory"
        
        # GPU メモリ不足
        strategy = recovery_system._determine_recovery_strategy("gpu_memory_percent", "critical")
        assert strategy == "gpu_memory_exhaustion"
        
        # 推論タイムアウト
        strategy = recovery_system._determine_recovery_strategy("inference_time_seconds", "warning")
        assert strategy == "inference_timeout"
        
        # 該当なし
        strategy = recovery_system._determine_recovery_strategy("unknown_metric", "warning")
        assert strategy is None
    
    def test_high_cpu_recovery(self):
        """高CPU使用率復旧テスト"""
        
        recovery_system = AutoRecoverySystem()
        alert = {"metric": "cpu_usage_percent", "value": 95, "severity": "critical"}
        
        result = recovery_system._recover_high_cpu(alert)
        
        assert result["success"] is True
        assert "actions" in result
        assert len(result["actions"]) > 0
        assert "message" in result
    
    def test_high_memory_recovery(self):
        """高メモリ使用率復旧テスト"""
        
        recovery_system = AutoRecoverySystem()
        alert = {"metric": "memory_usage_percent", "value": 97, "severity": "critical"}
        
        result = recovery_system._recover_high_memory(alert)
        
        assert result["success"] is True
        assert "actions" in result
        assert len(result["actions"]) > 0
    
    def test_gpu_memory_recovery(self):
        """GPU メモリ不足復旧テスト"""
        
        recovery_system = AutoRecoverySystem()
        alert = {"metric": "gpu_memory_percent", "value": 99, "severity": "critical"}
        
        result = recovery_system._recover_gpu_memory(alert)
        
        assert result["success"] is True
        assert "actions" in result
        assert len(result["actions"]) > 0
    
    def test_recovery_execution(self):
        """復旧実行テスト"""
        
        recovery_system = AutoRecoverySystem()
        
        alerts = [
            {"metric": "cpu_usage_percent", "value": 95, "severity": "critical"},
            {"metric": "memory_usage_percent", "value": 88, "severity": "warning"}
        ]
        
        result = recovery_system.execute_recovery(alerts)
        
        assert "timestamp" in result
        assert "actions_taken" in result
        assert "success" in result
        assert len(result["actions_taken"]) == 2
        
        # 復旧履歴の確認
        assert len(recovery_system.recovery_history) == 1
        
        # 履歴取得テスト
        history = recovery_system.get_recovery_history(limit=5)
        assert len(history) == 1
        assert history[0] == result


class TestMonitoringSystemIntegration:
    """監視システム統合テスト"""
    
    def test_monitoring_system_creation(self):
        """監視システム作成テスト"""
        
        dashboard_manager, anomaly_detector, recovery_system = create_monitoring_system()
        
        assert isinstance(dashboard_manager, GrafanaDashboardManager)
        assert isinstance(anomaly_detector, AnomalyDetector)
        assert isinstance(recovery_system, AutoRecoverySystem)
    
    @patch.object(AnomalyDetector, 'check_system_health')
    @patch.object(AutoRecoverySystem, 'execute_recovery')
    def test_full_monitoring_workflow(self, mock_recovery, mock_health_check):
        """完全監視ワークフローテスト"""
        
        # ヘルスチェック結果をモック
        mock_health_result = {
            "overall_status": "warning",
            "alerts": [
                {"metric": "cpu_usage_percent", "value": 85, "severity": "warning"}
            ],
            "recommendations": ["CPU使用率を削減してください"]
        }
        mock_health_check.return_value = mock_health_result
        
        # 復旧結果をモック
        mock_recovery_result = {
            "success": True,
            "actions_taken": [{"strategy": "high_cpu", "result": {"success": True}}]
        }
        mock_recovery.return_value = mock_recovery_result
        
        # 監視システム作成
        dashboard_manager, anomaly_detector, recovery_system = create_monitoring_system()
        
        # ワークフロー実行
        health_status = anomaly_detector.check_system_health()
        
        if health_status["alerts"]:
            recovery_result = recovery_system.execute_recovery(health_status["alerts"])
            
            assert recovery_result["success"] is True
            assert len(recovery_result["actions_taken"]) > 0
        
        # モック呼び出しの確認
        mock_health_check.assert_called_once()
        mock_recovery.assert_called_once()
    
    def test_dashboard_and_alerts_integration(self, temp_dir):
        """ダッシュボードとアラート統合テスト"""
        
        # 一時ディレクトリに移動
        import os
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            dashboard_manager = GrafanaDashboardManager(api_key=None)
            
            # ダッシュボード作成
            dashboard_result = dashboard_manager.create_dashboard("advanced_agent_dashboard")
            assert dashboard_result is True
            
            # アラートルール作成
            alert_result = dashboard_manager.create_alert_rules()
            assert alert_result is True
            
            # ファイルが作成されていることを確認
            dashboard_file = temp_dir / "grafana_dashboard_advanced_agent_dashboard.json"
            alert_file = temp_dir / "prometheus_alerts.yml"
            
            assert dashboard_file.exists()
            assert alert_file.exists()
            
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])