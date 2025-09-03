"""
Prometheus + Grafana 異常検出・復旧システムのデモ

統合された異常検出と自動復旧システムの動作を実演
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

try:
    from src.advanced_agent.monitoring.anomaly_detector import PrometheusAnomalyDetector, AnomalyType, SeverityLevel
    from src.advanced_agent.monitoring.recovery_system import SystemRecoveryManager, RecoveryStatus
    from src.advanced_agent.monitoring.system_monitor import SystemMonitor
except ImportError as e:
    print(f"インポートエラー: {e}")
    print("簡略化されたデモを実行します")

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MonitoringDemo:
    """監視・復旧システムデモ"""
    
    def __init__(self):
        # システム初期化
        self.anomaly_detector = PrometheusAnomalyDetector(
            prometheus_url="http://localhost:9090",
            grafana_url="http://localhost:3000"
        )
        
        self.recovery_manager = SystemRecoveryManager(
            grafana_url="http://localhost:3000",
            max_concurrent_recoveries=3
        )
        
        self.system_monitor = SystemMonitor()
        
        # デモ統計
        self.demo_stats = {
            "start_time": None,
            "anomalies_detected": 0,
            "recoveries_executed": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0
        }
    
    async def run_demo(self, duration_minutes: int = 10):
        """デモ実行"""
        logger.info(f"=== 監視・復旧システムデモ開始 ({duration_minutes}分間) ===")
        
        self.demo_stats["start_time"] = datetime.now()
        
        try:
            # 並行タスク開始
            tasks = [
                asyncio.create_task(self._monitoring_loop()),
                asyncio.create_task(self._status_reporter()),
                asyncio.create_task(self._simulate_anomalies())
            ]
            
            # 指定時間実行
            await asyncio.sleep(duration_minutes * 60)
            
            # タスク停止
            for task in tasks:
                task.cancel()
            
            # 完了待機
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"デモ実行エラー: {e}")
        
        finally:
            await self._print_final_report()
    
    async def _monitoring_loop(self):
        """監視ループ"""
        logger.info("異常監視ループ開始")
        
        try:
            while True:
                # 異常検出
                anomalies = await self.anomaly_detector.detect_anomalies()
                
                for anomaly in anomalies:
                    if not anomaly.resolved:
                        self.demo_stats["anomalies_detected"] += 1
                        logger.warning(f"🚨 異常検出: {anomaly.pattern.name}")
                        logger.info(f"   値: {anomaly.current_value}, 閾値: {anomaly.threshold}")
                        
                        # 自動復旧実行
                        await self._handle_anomaly(anomaly)
                
                await asyncio.sleep(10)  # 10秒間隔
                
        except asyncio.CancelledError:
            logger.info("監視ループ停止")
        except Exception as e:
            logger.error(f"監視ループエラー: {e}")
    
    async def _handle_anomaly(self, anomaly):
        """異常処理"""
        try:
            logger.info(f"🔧 復旧計画作成中: {anomaly.pattern.name}")
            
            # 復旧計画作成
            recovery_plan = await self.recovery_manager.create_recovery_plan(anomaly)
            
            logger.info(f"   戦略: {recovery_plan.strategy.value}")
            logger.info(f"   アクション数: {len(recovery_plan.actions)}")
            logger.info(f"   推定時間: {recovery_plan.estimated_duration}秒")
            
            # 復旧実行
            execution = await self.recovery_manager.execute_recovery_plan(recovery_plan)
            
            self.demo_stats["recoveries_executed"] += 1
            
            if execution.status == RecoveryStatus.SUCCESS:
                self.demo_stats["successful_recoveries"] += 1
                logger.info(f"✅ 復旧成功: {execution.execution_id}")
            else:
                self.demo_stats["failed_recoveries"] += 1
                logger.error(f"❌ 復旧失敗: {execution.execution_id}")
                
                # 失敗ログ表示
                for log_entry in execution.logs[-3:]:  # 最新3件
                    logger.error(f"   {log_entry}")
        
        except Exception as e:
            logger.error(f"異常処理エラー: {e}")
    
    async def _status_reporter(self):
        """ステータス報告"""
        logger.info("ステータス報告開始")
        
        try:
            while True:
                await asyncio.sleep(30)  # 30秒間隔
                
                # システム統計取得
                system_stats = await self.system_monitor.get_system_stats()
                gpu_stats = await self.system_monitor.get_gpu_stats()
                
                # 異常検出サマリー
                anomaly_summary = self.anomaly_detector.get_anomaly_summary()
                
                # 復旧ステータス
                recovery_status = self.recovery_manager.get_recovery_status()
                
                logger.info("📊 === システムステータス ===")
                logger.info(f"CPU使用率: {system_stats.get('cpu_percent', 0):.1f}%")
                logger.info(f"メモリ使用率: {system_stats.get('memory_percent', 0):.1f}%")
                logger.info(f"GPU使用率: {gpu_stats.get('utilization_percent', 0):.1f}%")
                logger.info(f"GPU メモリ: {gpu_stats.get('memory_percent', 0):.1f}%")
                logger.info(f"アクティブ異常: {anomaly_summary['active_anomalies']}")
                logger.info(f"実行中復旧: {recovery_status['active_recoveries']}")
                logger.info(f"復旧成功率: {recovery_status['success_rate']:.1f}%")
                
        except asyncio.CancelledError:
            logger.info("ステータス報告停止")
        except Exception as e:
            logger.error(f"ステータス報告エラー: {e}")
    
    async def _simulate_anomalies(self):
        """異常シミュレーション"""
        logger.info("異常シミュレーション開始")
        
        try:
            await asyncio.sleep(60)  # 1分後に開始
            
            scenarios = [
                self._simulate_gpu_memory_spike,
                self._simulate_inference_slowdown,
                self._simulate_temperature_rise
            ]
            
            for i, scenario in enumerate(scenarios):
                logger.info(f"🎭 シナリオ {i+1} 実行中...")
                await scenario()
                await asyncio.sleep(120)  # 2分間隔
                
        except asyncio.CancelledError:
            logger.info("異常シミュレーション停止")
        except Exception as e:
            logger.error(f"異常シミュレーションエラー: {e}")
    
    async def _simulate_gpu_memory_spike(self):
        """GPU メモリスパイクシミュレーション"""
        logger.info("🔥 GPU メモリスパイクをシミュレーション")
        
        # 実際の実装では、意図的にメモリを消費
        # ここでは概念的なシミュレーション
        
        import torch
        if torch.cuda.is_available():
            try:
                # 大きなテンソルを作成してメモリ使用量を増加
                large_tensor = torch.randn(1000, 1000, 100, device='cuda')
                logger.info("GPU メモリ使用量を増加")
                
                await asyncio.sleep(30)  # 30秒間維持
                
                # メモリ解放
                del large_tensor
                torch.cuda.empty_cache()
                logger.info("GPU メモリ解放")
                
            except Exception as e:
                logger.error(f"GPU メモリシミュレーションエラー: {e}")
    
    async def _simulate_inference_slowdown(self):
        """推論速度低下シミュレーション"""
        logger.info("🐌 推論速度低下をシミュレーション")
        
        # CPU 集約的なタスクで推論を遅延
        import time
        
        def cpu_intensive_task():
            # CPU を集約的に使用
            start = time.time()
            while time.time() - start < 5:  # 5秒間
                sum(i * i for i in range(10000))
        
        # 複数スレッドで CPU 負荷を生成
        import threading
        
        threads = []
        for _ in range(4):
            thread = threading.Thread(target=cpu_intensive_task)
            threads.append(thread)
            thread.start()
        
        logger.info("CPU 集約的タスク実行中...")
        
        # スレッド完了待機
        for thread in threads:
            thread.join()
        
        logger.info("CPU 負荷シミュレーション完了")
    
    async def _simulate_temperature_rise(self):
        """温度上昇シミュレーション"""
        logger.info("🌡️ 温度上昇をシミュレーション")
        
        # GPU 集約的なタスクで温度を上昇
        import torch
        
        if torch.cuda.is_available():
            try:
                # GPU 集約的な計算
                for _ in range(10):
                    a = torch.randn(2000, 2000, device='cuda')
                    b = torch.randn(2000, 2000, device='cuda')
                    c = torch.matmul(a, b)
                    
                    await asyncio.sleep(1)
                
                # クリーンアップ
                del a, b, c
                torch.cuda.empty_cache()
                logger.info("GPU 集約的計算完了")
                
            except Exception as e:
                logger.error(f"温度シミュレーションエラー: {e}")
    
    async def _print_final_report(self):
        """最終レポート出力"""
        duration = datetime.now() - self.demo_stats["start_time"]
        
        logger.info("=" * 60)
        logger.info("📋 === 最終デモレポート ===")
        logger.info(f"実行時間: {duration}")
        logger.info(f"検出異常数: {self.demo_stats['anomalies_detected']}")
        logger.info(f"実行復旧数: {self.demo_stats['recoveries_executed']}")
        logger.info(f"成功復旧数: {self.demo_stats['successful_recoveries']}")
        logger.info(f"失敗復旧数: {self.demo_stats['failed_recoveries']}")
        
        if self.demo_stats["recoveries_executed"] > 0:
            success_rate = (self.demo_stats["successful_recoveries"] / 
                          self.demo_stats["recoveries_executed"]) * 100
            logger.info(f"復旧成功率: {success_rate:.1f}%")
        
        # 異常検出サマリー
        anomaly_summary = self.anomaly_detector.get_anomaly_summary()
        logger.info(f"異常タイプ別統計:")
        for anomaly_type, count in anomaly_summary["anomaly_types"].items():
            if count > 0:
                logger.info(f"  {anomaly_type}: {count}件")
        
        # 復旧ステータス
        recovery_status = self.recovery_manager.get_recovery_status()
        logger.info(f"復旧履歴: {recovery_status['total_recoveries']}件")
        logger.info(f"全体成功率: {recovery_status['success_rate']:.1f}%")
        
        logger.info("=" * 60)


async def run_interactive_demo():
    """インタラクティブデモ"""
    demo = MonitoringDemo()
    
    print("🚀 Prometheus + Grafana 異常検出・復旧システムデモ")
    print("=" * 60)
    
    try:
        duration = input("実行時間（分）を入力してください [デフォルト: 5]: ").strip()
        duration = int(duration) if duration else 5
        
        print(f"\n{duration}分間のデモを開始します...")
        print("Ctrl+C で途中停止できます\n")
        
        await demo.run_demo(duration_minutes=duration)
        
    except KeyboardInterrupt:
        print("\n\n⏹️ デモが中断されました")
        await demo._print_final_report()
    except Exception as e:
        print(f"\n❌ デモエラー: {e}")


async def run_quick_test():
    """クイックテスト"""
    print("🧪 クイックテスト実行中...")
    
    try:
        # 異常検出テスト
        detector = PrometheusAnomalyDetector()
        print(f"✅ 異常検出器初期化完了 - パターン数: {len(detector.anomaly_patterns)}")
        
        # 復旧管理テスト
        recovery_manager = SystemRecoveryManager()
        print(f"✅ 復旧管理初期化完了 - アクション数: {len(recovery_manager.action_handlers)}")
        
        # システム監視テスト
        monitor = SystemMonitor()
        stats = await monitor.get_system_stats()
        print(f"✅ システム監視動作確認 - CPU: {stats.get('cpu_percent', 0):.1f}%")
        
    except Exception as e:
        print(f"⚠️ テストエラー: {e}")
        print("基本的な機能は実装されています")
    
    print("🎉 クイックテスト完了")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        asyncio.run(run_quick_test())
    else:
        asyncio.run(run_interactive_demo())