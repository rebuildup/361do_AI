"""
システム監視デモンストレーション
PSUtil + NVIDIA-ML + Prometheus 統合テスト
"""

import asyncio
import time
from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.advanced_agent.core.config import get_config, load_config
from src.advanced_agent.core.logging import setup_logging, get_logger
from src.advanced_agent.monitoring.system_monitor import SystemMonitor


async def demo_system_monitoring():
    """システム監視デモ"""
    print("=== Advanced Agent System Monitoring Demo ===\n")
    
    # 設定とログ初期化
    config = load_config("config/advanced_agent.yaml")
    logger = setup_logging(config.monitoring.log_level)
    
    print(f"Project: {config.project_name} v{config.version}")
    print(f"Environment: {config.environment}")
    print(f"GPU VRAM Limit: {config.gpu.max_vram_gb}GB")
    print(f"Monitoring Interval: {config.monitoring.interval_seconds}s")
    print(f"Prometheus Port: {config.monitoring.prometheus_port}")
    print()
    
    # システム監視初期化
    monitor = SystemMonitor(
        monitoring_interval=config.monitoring.interval_seconds,
        prometheus_port=config.monitoring.prometheus_port,
        enable_prometheus=config.monitoring.enable_prometheus
    )
    
    # 起動ログ
    logger.log_startup(
        component="system_monitor_demo",
        version=config.version,
        config_summary={
            "gpu_limit_gb": config.gpu.max_vram_gb,
            "monitoring_interval": config.monitoring.interval_seconds,
            "prometheus_enabled": config.monitoring.enable_prometheus
        }
    )
    
    try:
        print("Starting system monitoring...")
        await monitor.start_monitoring()
        
        # 初期統計表示
        print("\n=== Initial System Statistics ===")
        stats = monitor.get_combined_stats()
        
        print(f"Timestamp: {stats.timestamp}")
        print(f"CPU Usage: {stats.system.cpu_percent:.1f}% ({stats.system.cpu_count} cores)")
        print(f"Memory: {stats.system.memory_used_gb:.1f}GB / {stats.system.memory_total_gb:.1f}GB ({stats.system.memory_percent:.1f}%)")
        print(f"Disk Usage: {stats.system.disk_usage_percent:.1f}%")
        print(f"Network: ↑{stats.system.network_sent_mb:.2f}MB ↓{stats.system.network_recv_mb:.2f}MB")
        
        if stats.gpus:
            print("\n=== GPU Statistics ===")
            for gpu in stats.gpus:
                print(f"GPU{gpu.gpu_index}: {gpu.name}")
                print(f"  Memory: {gpu.used_memory_gb:.1f}GB / {gpu.total_memory_gb:.1f}GB ({gpu.memory_percent:.1f}%)")
                print(f"  Utilization: {gpu.utilization_percent}%")
                print(f"  Temperature: {gpu.temperature_celsius}°C")
                print(f"  Power: {gpu.power_draw_watts:.1f}W / {gpu.power_limit_watts:.1f}W")
        else:
            print("\n=== GPU Statistics ===")
            print("No GPU detected or NVIDIA-ML not available")
        
        if stats.alerts:
            print(f"\n=== Alerts ===")
            for alert in stats.alerts:
                print(f"⚠️  {alert}")
        else:
            print(f"\n=== Alerts ===")
            print("✅ No alerts")
        
        # メモリ圧迫レベル表示
        memory_level = monitor.get_memory_pressure_level()
        gpu_memory_level = monitor.get_gpu_memory_pressure_level()
        
        print(f"\n=== Memory Pressure Levels ===")
        print(f"System Memory: {memory_level}")
        print(f"GPU Memory: {gpu_memory_level}")
        
        optimization_needed = monitor.should_trigger_memory_optimization()
        print(f"Memory Optimization Needed: {'Yes' if optimization_needed else 'No'}")
        
        if config.monitoring.enable_prometheus:
            print(f"\n=== Prometheus Metrics ===")
            print(f"Metrics available at: http://localhost:{config.monitoring.prometheus_port}/metrics")
        
        # 継続監視デモ
        print(f"\n=== Continuous Monitoring (10 seconds) ===")
        print("Monitoring system metrics... (Press Ctrl+C to stop)")
        
        start_time = time.time()
        cycle_count = 0
        
        while time.time() - start_time < 10:
            await asyncio.sleep(2)
            cycle_count += 1
            
            # 現在の統計取得
            current_stats = monitor.get_combined_stats()
            
            print(f"\nCycle {cycle_count}:")
            print(f"  CPU: {current_stats.system.cpu_percent:.1f}%")
            print(f"  Memory: {current_stats.system.memory_percent:.1f}%")
            
            if current_stats.gpus:
                gpu = current_stats.gpus[0]
                print(f"  GPU Memory: {gpu.memory_percent:.1f}%")
                print(f"  GPU Util: {gpu.utilization_percent}%")
                print(f"  GPU Temp: {gpu.temperature_celsius}°C")
            
            if current_stats.alerts:
                print(f"  Alerts: {len(current_stats.alerts)}")
                for alert in current_stats.alerts:
                    print(f"    ⚠️  {alert}")
            
            # パフォーマンスメトリクスログ
            logger.log_system_stats({
                "cpu_percent": current_stats.system.cpu_percent,
                "memory_percent": current_stats.system.memory_percent,
                "disk_percent": current_stats.system.disk_usage_percent
            })
            
            if current_stats.gpus:
                logger.log_gpu_stats({
                    "gpu_memory_percent": current_stats.gpus[0].memory_percent,
                    "gpu_utilization": current_stats.gpus[0].utilization_percent,
                    "gpu_temperature": current_stats.gpus[0].temperature_celsius
                })
        
        print(f"\n=== Final Statistics ===")
        final_stats = monitor.get_combined_stats()
        
        print(f"Total Monitoring Cycles: {cycle_count}")
        print(f"Final CPU Usage: {final_stats.system.cpu_percent:.1f}%")
        print(f"Final Memory Usage: {final_stats.system.memory_percent:.1f}%")
        
        if final_stats.gpus:
            gpu = final_stats.gpus[0]
            print(f"Final GPU Memory: {gpu.memory_percent:.1f}%")
            print(f"Final GPU Temperature: {gpu.temperature_celsius}°C")
        
        # アラート統計
        total_alerts = sum(len(stats.alerts) for stats in [stats, final_stats])
        print(f"Total Alerts Generated: {total_alerts}")
        
    except KeyboardInterrupt:
        print("\n\nMonitoring interrupted by user")
    except Exception as e:
        print(f"\nError during monitoring: {e}")
        logger.log_alert("monitoring_error", "ERROR", str(e))
    finally:
        print("\nStopping system monitoring...")
        await monitor.stop_monitoring()
        
        # 終了ログ
        uptime = time.time() - start_time
        logger.log_shutdown(
            component="system_monitor_demo",
            uptime_seconds=uptime,
            final_stats={
                "cycles_completed": cycle_count,
                "final_cpu_percent": final_stats.system.cpu_percent if 'final_stats' in locals() else 0,
                "final_memory_percent": final_stats.system.memory_percent if 'final_stats' in locals() else 0
            }
        )
        
        print("Demo completed!")
        print(f"\nCheck the following for detailed logs:")
        print(f"  - logs/agent_*.log (general logs)")
        print(f"  - logs/agent_structured_*.json (structured logs)")
        print(f"  - logs/agent_performance_*.log (performance logs)")
        
        if config.monitoring.enable_prometheus:
            print(f"  - http://localhost:{config.monitoring.prometheus_port}/metrics (Prometheus metrics)")


async def demo_memory_pressure_simulation():
    """メモリ圧迫シミュレーションデモ"""
    print("\n=== Memory Pressure Simulation Demo ===")
    
    config = get_config()
    logger = get_logger()
    
    monitor = SystemMonitor(
        monitoring_interval=0.5,
        prometheus_port=config.monitoring.prometheus_port + 1,
        enable_prometheus=False
    )
    
    try:
        await monitor.start_monitoring()
        
        print("Simulating different memory pressure scenarios...")
        
        # 各圧迫レベルをシミュレーション
        scenarios = [
            ("Normal Usage", 50, 40),
            ("Medium Pressure", 75, 70),
            ("High Pressure", 90, 85),
            ("Critical Pressure", 97, 96)
        ]
        
        for scenario_name, sys_mem, gpu_mem in scenarios:
            print(f"\n--- {scenario_name} ---")
            
            # メモリ使用率をシミュレート（実際のメモリ使用量は変更しない）
            with monitor:
                # 現在の統計を取得し、値を上書き
                stats = monitor.get_combined_stats()
                stats.system.memory_percent = sys_mem
                
                if stats.gpus:
                    stats.gpus[0].memory_percent = gpu_mem
                
                # 圧迫レベル判定
                memory_level = "LOW"
                if sys_mem > 95:
                    memory_level = "CRITICAL"
                elif sys_mem > 85:
                    memory_level = "HIGH"
                elif sys_mem > 70:
                    memory_level = "MEDIUM"
                
                gpu_memory_level = "LOW"
                if gpu_mem > 95:
                    gpu_memory_level = "CRITICAL"
                elif gpu_mem > 85:
                    gpu_memory_level = "HIGH"
                elif gpu_mem > 70:
                    gpu_memory_level = "MEDIUM"
                
                print(f"System Memory: {sys_mem}% ({memory_level})")
                print(f"GPU Memory: {gpu_mem}% ({gpu_memory_level})")
                
                optimization_needed = memory_level in ["HIGH", "CRITICAL"] or gpu_memory_level in ["HIGH", "CRITICAL"]
                print(f"Optimization Trigger: {'YES' if optimization_needed else 'NO'}")
                
                # アラート生成
                alerts = []
                if sys_mem > 90:
                    alerts.append(f"HIGH_MEMORY_USAGE: {sys_mem}%")
                if gpu_mem > 95:
                    alerts.append(f"HIGH_GPU_MEMORY_USAGE: {gpu_mem}%")
                
                if alerts:
                    print("Alerts:")
                    for alert in alerts:
                        print(f"  ⚠️  {alert}")
                        logger.log_alert("memory_pressure", "WARNING", alert)
                else:
                    print("Alerts: None")
                
                # メモリ圧迫ログ
                logger.log_memory_pressure(
                    pressure_level=max(memory_level, gpu_memory_level, key=lambda x: ["LOW", "MEDIUM", "HIGH", "CRITICAL"].index(x)),
                    system_memory_percent=sys_mem,
                    gpu_memory_percent=gpu_mem,
                    action_taken="simulation" if optimization_needed else None
                )
            
            await asyncio.sleep(1)
        
    finally:
        await monitor.stop_monitoring()
        print("\nMemory pressure simulation completed!")


if __name__ == "__main__":
    print("Advanced Agent System Monitoring Demo")
    print("=====================================")
    
    try:
        # メインデモ実行
        asyncio.run(demo_system_monitoring())
        
        # メモリ圧迫シミュレーション
        asyncio.run(demo_memory_pressure_simulation())
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback
        traceback.print_exc()