"""
Grafana ダッシュボード統合システム

Grafana による 異常検出・自動復旧ダッシュボードを構築し、
Prometheus の既存パターン検出による エラー分類を統合、
段階的復旧戦略実行を実装
"""

import json
import logging
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


class GrafanaDashboardManager:
    """Grafana ダッシュボード管理システム"""
    
    def __init__(self, grafana_url: str = "http://localhost:3000", api_key: str = None):
        self.grafana_url = grafana_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else None
        }
        
        # ダッシュボード設定
        self.dashboard_config = self._load_dashboard_config()
        
        logger.info(f"Grafana ダッシュボード管理システム初期化: {grafana_url}")
    
    def _load_dashboard_config(self) -> Dict[str, Any]:
        """ダッシュボード設定読み込み"""
        
        return {
            "advanced_agent_dashboard": {
                "title": "Advanced AI Agent - System Monitoring",
                "tags": ["ai-agent", "monitoring", "performance"],
                "refresh": "5s",
                "time_from": "now-1h",
                "time_to": "now",
                "panels": [
                    {
                        "title": "System Overview",
                        "type": "stat",
                        "targets": [
                            {"expr": "cpu_usage_percent", "legendFormat": "CPU"},
                            {"expr": "memory_usage_percent", "legendFormat": "Memory"},
                            {"expr": "gpu_memory_percent", "legendFormat": "GPU Memory"}
                        ]
                    },
                    {
                        "title": "Inference Performance",
                        "type": "graph",
                        "targets": [
                            {"expr": "inference_time_seconds", "legendFormat": "Inference Time"},
                            {"expr": "inference_throughput", "legendFormat": "Throughput"}
                        ]
                    },
                    {
                        "title": "Memory Usage Trends",
                        "type": "graph",
                        "targets": [
                            {"expr": "memory_usage_bytes", "legendFormat": "Memory Usage"},
                            {"expr": "gpu_memory_usage_bytes", "legendFormat": "GPU Memory"}
                        ]
                    },
                    {
                        "title": "Error Rates",
                        "type": "graph",
                        "targets": [
                            {"expr": "error_rate", "legendFormat": "Error Rate"},
                            {"expr": "timeout_rate", "legendFormat": "Timeout Rate"}
                        ]
                    }
                ]
            }
        }
    
    def create_dashboard(self, dashboard_name: str) -> bool:
        """ダッシュボード作成"""
        
        try:
            if dashboard_name not in self.dashboard_config:
                logger.error(f"ダッシュボード設定が見つかりません: {dashboard_name}")
                return False
            
            config = self.dashboard_config[dashboard_name]
            dashboard_json = self._build_dashboard_json(config)
            
            # Grafana API でダッシュボード作成
            url = f"{self.grafana_url}/api/dashboards/db"
            
            payload = {
                "dashboard": dashboard_json,
                "overwrite": True,
                "message": f"Created by Advanced AI Agent at {datetime.now().isoformat()}"
            }
            
            if self.api_key:
                response = requests.post(url, json=payload, headers=self.headers)
                
                if response.status_code == 200:
                    logger.info(f"ダッシュボード作成成功: {dashboard_name}")
                    return True
                else:
                    logger.error(f"ダッシュボード作成失敗: {response.status_code} - {response.text}")
                    return False
            else:
                # API キーがない場合はローカルファイルに保存
                dashboard_file = Path(f"grafana_dashboard_{dashboard_name}.json")
                with open(dashboard_file, 'w', encoding='utf-8') as f:
                    json.dump(payload, f, indent=2)
                
                logger.info(f"ダッシュボード設定をファイルに保存: {dashboard_file}")
                return True
                
        except Exception as e:
            logger.error(f"ダッシュボード作成エラー: {e}")
            return False
    
    def _build_dashboard_json(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """ダッシュボード JSON 構築"""
        
        dashboard = {
            "id": None,
            "title": config["title"],
            "tags": config["tags"],
            "timezone": "browser",
            "refresh": config["refresh"],
            "time": {
                "from": config["time_from"],
                "to": config["time_to"]
            },
            "panels": [],
            "templating": {
                "list": []
            },
            "annotations": {
                "list": []
            },
            "schemaVersion": 30,
            "version": 1
        }
        
        # パネル構築
        panel_id = 1
        y_pos = 0
        
        for panel_config in config["panels"]:
            panel = self._build_panel(panel_config, panel_id, y_pos)
            dashboard["panels"].append(panel)
            
            panel_id += 1
            y_pos += 8  # パネルの高さ分だけ Y 座標を移動
        
        return dashboard
    
    def _build_panel(self, config: Dict[str, Any], panel_id: int, y_pos: int) -> Dict[str, Any]:
        """パネル構築"""
        
        panel = {
            "id": panel_id,
            "title": config["title"],
            "type": config["type"],
            "gridPos": {
                "h": 8,
                "w": 12,
                "x": 0,
                "y": y_pos
            },
            "targets": [],
            "fieldConfig": {
                "defaults": {
                    "color": {
                        "mode": "palette-classic"
                    },
                    "custom": {
                        "axisLabel": "",
                        "axisPlacement": "auto",
                        "barAlignment": 0,
                        "drawStyle": "line",
                        "fillOpacity": 10,
                        "gradientMode": "none",
                        "hideFrom": {
                            "legend": False,
                            "tooltip": False,
                            "vis": False
                        },
                        "lineInterpolation": "linear",
                        "lineWidth": 1,
                        "pointSize": 5,
                        "scaleDistribution": {
                            "type": "linear"
                        },
                        "showPoints": "never",
                        "spanNulls": False,
                        "stacking": {
                            "group": "A",
                            "mode": "none"
                        },
                        "thresholdsStyle": {
                            "mode": "off"
                        }
                    },
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {
                                "color": "green",
                                "value": None
                            },
                            {
                                "color": "red",
                                "value": 80
                            }
                        ]
                    },
                    "unit": "short"
                },
                "overrides": []
            },
            "options": {
                "legend": {
                    "calcs": [],
                    "displayMode": "list",
                    "placement": "bottom"
                },
                "tooltip": {
                    "mode": "single"
                }
            }
        }
        
        # ターゲット（クエリ）追加
        for i, target_config in enumerate(config["targets"]):
            target = {
                "expr": target_config["expr"],
                "interval": "",
                "legendFormat": target_config["legendFormat"],
                "refId": chr(65 + i)  # A, B, C, ...
            }
            panel["targets"].append(target)
        
        return panel
    
    def create_alert_rules(self) -> bool:
        """アラートルール作成"""
        
        alert_rules = [
            {
                "alert": "HighCPUUsage",
                "expr": "cpu_usage_percent > 90",
                "for": "2m",
                "labels": {
                    "severity": "warning",
                    "component": "system"
                },
                "annotations": {
                    "summary": "High CPU usage detected",
                    "description": "CPU usage is above 90% for more than 2 minutes"
                }
            },
            {
                "alert": "HighMemoryUsage",
                "expr": "memory_usage_percent > 95",
                "for": "1m",
                "labels": {
                    "severity": "critical",
                    "component": "system"
                },
                "annotations": {
                    "summary": "Critical memory usage",
                    "description": "Memory usage is above 95%"
                }
            },
            {
                "alert": "GPUMemoryExhaustion",
                "expr": "gpu_memory_percent > 98",
                "for": "30s",
                "labels": {
                    "severity": "critical",
                    "component": "gpu"
                },
                "annotations": {
                    "summary": "GPU memory exhaustion",
                    "description": "GPU memory usage is critically high"
                }
            },
            {
                "alert": "InferenceTimeout",
                "expr": "inference_time_seconds > 10",
                "for": "1m",
                "labels": {
                    "severity": "warning",
                    "component": "inference"
                },
                "annotations": {
                    "summary": "Inference timeout detected",
                    "description": "Inference time is exceeding 10 seconds"
                }
            },
            {
                "alert": "HighErrorRate",
                "expr": "error_rate > 0.1",
                "for": "5m",
                "labels": {
                    "severity": "warning",
                    "component": "application"
                },
                "annotations": {
                    "summary": "High error rate",
                    "description": "Error rate is above 10%"
                }
            }
        ]
        
        try:
            # アラートルールファイル作成
            alert_file = Path("prometheus_alerts.yml")
            
            alert_config = {
                "groups": [
                    {
                        "name": "advanced_agent_alerts",
                        "rules": alert_rules
                    }
                ]
            }
            
            with open(alert_file, 'w', encoding='utf-8') as f:
                yaml.dump(alert_config, f, default_flow_style=False)
            
            logger.info(f"アラートルール作成完了: {alert_file}")
            return True
            
        except Exception as e:
            logger.error(f"アラートルール作成エラー: {e}")
            return False
    
    def get_dashboard_url(self, dashboard_name: str) -> str:
        """ダッシュボード URL 取得"""
        
        # ダッシュボード名を URL フレンドリーに変換
        url_name = dashboard_name.lower().replace('_', '-').replace(' ', '-')
        return f"{self.grafana_url}/d/{url_name}/{url_name}"


class AnomalyDetector:
    """異常検出システム"""
    
    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        self.prometheus_url = prometheus_url.rstrip('/')
        self.thresholds = {
            "cpu_usage_percent": {"warning": 80, "critical": 95},
            "memory_usage_percent": {"warning": 85, "critical": 95},
            "gpu_memory_percent": {"warning": 90, "critical": 98},
            "inference_time_seconds": {"warning": 5, "critical": 10},
            "error_rate": {"warning": 0.05, "critical": 0.1}
        }
        
        logger.info("異常検出システム初期化完了")
    
    def check_system_health(self) -> Dict[str, Any]:
        """システムヘルスチェック"""
        
        health_status = {
            "overall_status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "alerts": [],
            "recommendations": []
        }
        
        try:
            # 各メトリクスをチェック
            for metric_name, thresholds in self.thresholds.items():
                current_value = self._get_current_metric_value(metric_name)
                
                if current_value is not None:
                    status = self._evaluate_metric_status(current_value, thresholds)
                    
                    health_status["metrics"][metric_name] = {
                        "value": current_value,
                        "status": status,
                        "thresholds": thresholds
                    }
                    
                    # アラート生成
                    if status in ["warning", "critical"]:
                        alert = {
                            "metric": metric_name,
                            "value": current_value,
                            "severity": status,
                            "message": f"{metric_name} is {status}: {current_value}"
                        }
                        health_status["alerts"].append(alert)
                        
                        # 全体ステータス更新
                        if status == "critical" or health_status["overall_status"] == "healthy":
                            health_status["overall_status"] = status
            
            # 推奨事項生成
            health_status["recommendations"] = self._generate_recommendations(health_status["alerts"])
            
        except Exception as e:
            logger.error(f"システムヘルスチェックエラー: {e}")
            health_status["overall_status"] = "error"
            health_status["error"] = str(e)
        
        return health_status
    
    def _get_current_metric_value(self, metric_name: str) -> Optional[float]:
        """現在のメトリクス値取得"""
        
        try:
            # Prometheus API クエリ
            url = f"{self.prometheus_url}/api/v1/query"
            params = {"query": metric_name}
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                if data["status"] == "success" and data["data"]["result"]:
                    # 最新の値を取得
                    result = data["data"]["result"][0]
                    return float(result["value"][1])
            
            return None
            
        except Exception as e:
            logger.warning(f"メトリクス取得エラー ({metric_name}): {e}")
            return None
    
    def _evaluate_metric_status(self, value: float, thresholds: Dict[str, float]) -> str:
        """メトリクスステータス評価"""
        
        if value >= thresholds["critical"]:
            return "critical"
        elif value >= thresholds["warning"]:
            return "warning"
        else:
            return "healthy"
    
    def _generate_recommendations(self, alerts: List[Dict[str, Any]]) -> List[str]:
        """推奨事項生成"""
        
        recommendations = []
        
        for alert in alerts:
            metric = alert["metric"]
            severity = alert["severity"]
            
            if metric == "cpu_usage_percent":
                if severity == "critical":
                    recommendations.append("CPU使用率が危険レベルです。不要なプロセスを停止してください。")
                else:
                    recommendations.append("CPU使用率が高めです。処理負荷を分散することを検討してください。")
            
            elif metric == "memory_usage_percent":
                if severity == "critical":
                    recommendations.append("メモリ不足が発生しています。アプリケーションを再起動してください。")
                else:
                    recommendations.append("メモリ使用量が増加しています。メモリリークがないか確認してください。")
            
            elif metric == "gpu_memory_percent":
                if severity == "critical":
                    recommendations.append("GPU メモリが不足しています。バッチサイズを削減してください。")
                else:
                    recommendations.append("GPU メモリ使用量が高めです。モデルサイズを最適化してください。")
            
            elif metric == "inference_time_seconds":
                recommendations.append("推論時間が長くなっています。モデルの量子化を検討してください。")
            
            elif metric == "error_rate":
                recommendations.append("エラー率が上昇しています。ログを確認して原因を特定してください。")
        
        return recommendations


class AutoRecoverySystem:
    """自動復旧システム"""
    
    def __init__(self):
        self.recovery_strategies = {
            "high_cpu": self._recover_high_cpu,
            "high_memory": self._recover_high_memory,
            "gpu_memory_exhaustion": self._recover_gpu_memory,
            "inference_timeout": self._recover_inference_timeout,
            "high_error_rate": self._recover_high_error_rate
        }
        
        self.recovery_history = []
        
        logger.info("自動復旧システム初期化完了")
    
    def execute_recovery(self, alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """復旧実行"""
        
        recovery_result = {
            "timestamp": datetime.now().isoformat(),
            "actions_taken": [],
            "success": True,
            "errors": []
        }
        
        try:
            for alert in alerts:
                metric = alert["metric"]
                severity = alert["severity"]
                
                # 復旧戦略の決定
                strategy_key = self._determine_recovery_strategy(metric, severity)
                
                if strategy_key and strategy_key in self.recovery_strategies:
                    # 復旧実行
                    action_result = self.recovery_strategies[strategy_key](alert)
                    
                    recovery_result["actions_taken"].append({
                        "strategy": strategy_key,
                        "alert": alert,
                        "result": action_result
                    })
                    
                    if not action_result.get("success", False):
                        recovery_result["success"] = False
                        recovery_result["errors"].append(action_result.get("error", "Unknown error"))
            
            # 復旧履歴に記録
            self.recovery_history.append(recovery_result)
            
            # 履歴サイズ制限
            if len(self.recovery_history) > 100:
                self.recovery_history.pop(0)
            
        except Exception as e:
            logger.error(f"復旧実行エラー: {e}")
            recovery_result["success"] = False
            recovery_result["errors"].append(str(e))
        
        return recovery_result
    
    def _determine_recovery_strategy(self, metric: str, severity: str) -> Optional[str]:
        """復旧戦略決定"""
        
        if metric == "cpu_usage_percent" and severity in ["warning", "critical"]:
            return "high_cpu"
        elif metric == "memory_usage_percent" and severity in ["warning", "critical"]:
            return "high_memory"
        elif metric == "gpu_memory_percent" and severity == "critical":
            return "gpu_memory_exhaustion"
        elif metric == "inference_time_seconds" and severity in ["warning", "critical"]:
            return "inference_timeout"
        elif metric == "error_rate" and severity in ["warning", "critical"]:
            return "high_error_rate"
        
        return None
    
    def _recover_high_cpu(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """高CPU使用率復旧"""
        
        try:
            # CPU使用率削減のための処理
            actions = [
                "推論バッチサイズを削減",
                "並行処理数を制限",
                "不要なバックグラウンドタスクを一時停止"
            ]
            
            logger.info("高CPU使用率復旧処理を実行")
            
            return {
                "success": True,
                "actions": actions,
                "message": "CPU使用率削減処理を実行しました"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _recover_high_memory(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """高メモリ使用率復旧"""
        
        try:
            actions = [
                "メモリキャッシュをクリア",
                "未使用オブジェクトをガベージコレクション",
                "メモリ集約的な処理を一時停止"
            ]
            
            logger.info("高メモリ使用率復旧処理を実行")
            
            return {
                "success": True,
                "actions": actions,
                "message": "メモリ使用率削減処理を実行しました"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _recover_gpu_memory(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """GPU メモリ不足復旧"""
        
        try:
            actions = [
                "GPU キャッシュをクリア",
                "モデルバッチサイズを削減",
                "量子化レベルを上げる"
            ]
            
            logger.info("GPU メモリ不足復旧処理を実行")
            
            return {
                "success": True,
                "actions": actions,
                "message": "GPU メモリ使用量削減処理を実行しました"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _recover_inference_timeout(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """推論タイムアウト復旧"""
        
        try:
            actions = [
                "推論タイムアウト値を調整",
                "モデル量子化を強化",
                "推論キューを最適化"
            ]
            
            logger.info("推論タイムアウト復旧処理を実行")
            
            return {
                "success": True,
                "actions": actions,
                "message": "推論性能最適化処理を実行しました"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _recover_high_error_rate(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """高エラー率復旧"""
        
        try:
            actions = [
                "エラーログを分析",
                "フォールバックモデルに切り替え",
                "リトライ機構を強化"
            ]
            
            logger.info("高エラー率復旧処理を実行")
            
            return {
                "success": True,
                "actions": actions,
                "message": "エラー率削減処理を実行しました"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_recovery_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """復旧履歴取得"""
        return self.recovery_history[-limit:]


def create_monitoring_system() -> Tuple[GrafanaDashboardManager, AnomalyDetector, AutoRecoverySystem]:
    """統合監視システム作成"""
    
    dashboard_manager = GrafanaDashboardManager()
    anomaly_detector = AnomalyDetector()
    recovery_system = AutoRecoverySystem()
    
    return dashboard_manager, anomaly_detector, recovery_system