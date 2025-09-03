#!/usr/bin/env python3
"""
統合監視システムデモ

Prometheus + Grafana 監視・最適化システムの統合デモ
"""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
import sys

# パス設定
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.advanced_agent.monitoring.grafana_dashboard import (
    GrafanaDashboardManager,
    AnomalyDetector,
    AutoRecoverySystem,
    create_monitoring_system
)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_monitoring_system():
    """統合監視システムデモ"""
    
    print("=" * 60)
    print("Advanced AI Agent - 統合監視システムデモ")
    print("=" * 60)
    
    # 監視システム作成
    print("\n1. 監視システム初期化...")
    dashboard_manager, anomaly_detector, recovery_system = create_monitoring_system()
    
    print("✅ Grafana ダッシュボード管理システム初期化完了")
    print("✅ 異常検出システム初期化完了")
    print("✅ 自動復旧システム初期化完了")
    
    # ダッシュボード作成デモ
    print("\n2. Grafana ダッシュボード作成...")
    
    dashboard_created = dashboard_manager.create_dashboard("advanced_agent_dashboard")
    if dashboard_created:
        print("✅ ダッシュボード作成成功")
        dashboard_url = dashboard_manager.get_dashboard_url("advanced_agent_dashboard")
        print(f"📊 ダッシュボード URL: {dashboard_url}")
    else:
        print("❌ ダッシュボード作成失敗")
    
    # アラートルール作成デモ
    print("\n3. アラートルール作成...")
    
    alerts_created = dashboard_manager.create_alert_rules()
    if alerts_created:
        print("✅ アラートルール作成成功")
        print("📋 prometheus_alerts.yml ファイルを確認してください")
    else:
        print("❌ アラートルール作成失敗")
    
    # システムヘルスチェックデモ
    print("\n4. システムヘルスチェック...")
    
    health_status = anomaly_detector.check_system_health()
    
    print(f"🏥 全体ステータス: {health_status['overall_status']}")
    print(f"📊 監視メトリクス数: {len(health_status['metrics'])}")
    print(f"🚨 アクティブアラート数: {len(health_status['alerts'])}")
    
    # メトリクス詳細表示
    if health_status['metrics']:
        print("\n📈 メトリクス詳細:")
        for metric_name, metric_data in health_status['metrics'].items():
            status_emoji = {
                "healthy": "🟢",
                "warning": "🟡", 
                "critical": "🔴"
            }.get(metric_data['status'], "⚪")
            
            print(f"  {status_emoji} {metric_name}: {metric_data['value']:.2f} ({metric_data['status']})")
    
    # アラート詳細表示
    if health_status['alerts']:
        print("\n🚨 アクティブアラート:")
        for alert in health_status['alerts']:
            severity_emoji = {
                "warning": "⚠️",
                "critical": "🚨"
            }.get(alert['severity'], "ℹ️")
            
            print(f"  {severity_emoji} {alert['metric']}: {alert['value']:.2f} ({alert['severity']})")
    
    # 推奨事項表示
    if health_status['recommendations']:
        print("\n💡 推奨事項:")
        for i, recommendation in enumerate(health_status['recommendations'], 1):
            print(f"  {i}. {recommendation}")
    
    # 自動復旧デモ
    if health_status['alerts']:
        print("\n5. 自動復旧システム実行...")
        
        recovery_result = recovery_system.execute_recovery(health_status['alerts'])
        
        if recovery_result['success']:
            print("✅ 自動復旧処理成功")
            print(f"🔧 実行されたアクション数: {len(recovery_result['actions_taken'])}")
            
            for action in recovery_result['actions_taken']:
                strategy = action['strategy']
                result = action['result']
                
                if result.get('success'):
                    print(f"  ✅ {strategy}: {result.get('message', '処理完了')}")
                    
                    if 'actions' in result:
                        for sub_action in result['actions']:
                            print(f"    - {sub_action}")
                else:
                    print(f"  ❌ {strategy}: {result.get('error', '処理失敗')}")
        else:
            print("❌ 自動復旧処理失敗")
            for error in recovery_result.get('errors', []):
                print(f"  ❌ エラー: {error}")
    else:
        print("\n5. 自動復旧システム...")
        print("ℹ️ アラートがないため、復旧処理はスキップされました")
    
    # 復旧履歴表示
    print("\n6. 復旧履歴...")
    
    recovery_history = recovery_system.get_recovery_history(limit=3)
    
    if recovery_history:
        print(f"📚 復旧履歴 (最新 {len(recovery_history)} 件):")
        
        for i, record in enumerate(recovery_history, 1):
            timestamp = record['timestamp']
            success = record['success']
            actions_count = len(record['actions_taken'])
            
            status_emoji = "✅" if success else "❌"
            print(f"  {status_emoji} {i}. {timestamp} - {actions_count} アクション実行")
    else:
        print("📚 復旧履歴: なし")
    
    print("\n7. 監視システム統合確認...")
    
    # 設定ファイル確認
    config_files = [
        "config/prometheus.yml",
        "config/alertmanager.yml", 
        "docker-compose.monitoring.yml",
        "prometheus_alerts.yml"
    ]
    
    print("📁 設定ファイル確認:")
    for config_file in config_files:
        file_path = Path(config_file)
        if file_path.exists():
            print(f"  ✅ {config_file}")
        else:
            print(f"  ❌ {config_file} (見つかりません)")
    
    print("\n8. 監視システム起動手順...")
    
    print("🚀 Docker Compose で監視システムを起動:")
    print("  docker-compose -f docker-compose.monitoring.yml up -d")
    print()
    print("🌐 アクセス URL:")
    print("  - Grafana ダッシュボード: http://localhost:3000 (admin/admin123)")
    print("  - Prometheus: http://localhost:9090")
    print("  - AlertManager: http://localhost:9093")
    print("  - Advanced AI Agent API: http://localhost:8000")
    print("  - Streamlit UI: http://localhost:8501")
    
    print("\n" + "=" * 60)
    print("統合監視システムデモ完了")
    print("=" * 60)


async def demo_continuous_monitoring():
    """継続監視デモ"""
    
    print("\n🔄 継続監視デモ開始 (30秒間)")
    
    dashboard_manager, anomaly_detector, recovery_system = create_monitoring_system()
    
    start_time = time.time()
    check_count = 0
    
    while time.time() - start_time < 30:  # 30秒間実行
        check_count += 1
        
        print(f"\n--- 監視チェック #{check_count} ---")
        
        # ヘルスチェック実行
        health_status = anomaly_detector.check_system_health()
        
        print(f"ステータス: {health_status['overall_status']}")
        print(f"アラート数: {len(health_status['alerts'])}")
        
        # アラートがある場合は復旧実行
        if health_status['alerts']:
            print("🚨 アラート検出 - 自動復旧実行中...")
            
            recovery_result = recovery_system.execute_recovery(health_status['alerts'])
            
            if recovery_result['success']:
                print("✅ 復旧処理完了")
            else:
                print("❌ 復旧処理失敗")
        
        # 5秒待機
        await asyncio.sleep(5)
    
    print("\n🔄 継続監視デモ終了")


def demo_configuration_files():
    """設定ファイルデモ"""
    
    print("\n📋 監視システム設定ファイル概要:")
    
    configs = {
        "config/prometheus.yml": "Prometheus メイン設定",
        "config/alertmanager.yml": "AlertManager アラート設定", 
        "docker-compose.monitoring.yml": "Docker Compose 監視スタック",
        "prometheus_alerts.yml": "Prometheus アラートルール"
    }
    
    for config_file, description in configs.items():
        file_path = Path(config_file)
        
        print(f"\n📄 {config_file}")
        print(f"   説明: {description}")
        
        if file_path.exists():
            print(f"   ステータス: ✅ 存在")
            print(f"   サイズ: {file_path.stat().st_size} bytes")
            print(f"   更新日時: {datetime.fromtimestamp(file_path.stat().st_mtime)}")
        else:
            print(f"   ステータス: ❌ 見つかりません")


async def main():
    """メイン関数"""
    
    try:
        # 基本デモ実行
        await demo_monitoring_system()
        
        # 設定ファイル確認
        demo_configuration_files()
        
        # ユーザー選択
        print("\n" + "=" * 60)
        print("追加デモオプション:")
        print("1. 継続監視デモ (30秒間)")
        print("2. 終了")
        
        choice = input("\n選択してください (1-2): ").strip()
        
        if choice == "1":
            await demo_continuous_monitoring()
        
        print("\nデモを終了します。")
        
    except KeyboardInterrupt:
        print("\n\nデモが中断されました。")
    except Exception as e:
        logger.error(f"デモ実行エラー: {e}")
        print(f"\n❌ エラーが発生しました: {e}")


if __name__ == "__main__":
    asyncio.run(main())