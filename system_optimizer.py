#!/usr/bin/env python3
"""
System Optimizer for Multi-Agent Learning
マルチエージェント学習システム用最適化ツール
"""

import asyncio
import sys
import os
import gc
import psutil
import time
from datetime import datetime
from pathlib import Path
import json

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


class SystemOptimizer:
    """システム最適化クラス"""
    
    def __init__(self):
        self.optimization_log = []
        self.initial_stats = {}
        self.optimized_stats = {}
    
    def log_optimization(self, action: str, result: str, improvement: str = None):
        """最適化ログを記録"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'result': result,
            'improvement': improvement
        }
        self.optimization_log.append(entry)
        print(f"  ✅ {action}: {result}")
        if improvement:
            print(f"    💡 {improvement}")
    
    def collect_system_stats(self) -> dict:
        """システム統計収集"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'disk_usage_percent': psutil.disk_usage('.').percent,
            'process_count': len(psutil.pids()),
            'timestamp': datetime.now().isoformat()
        }
    
    def optimize_memory(self):
        """メモリ最適化"""
        print("🧠 メモリ最適化中...")
        
        # 初期メモリ状態
        initial_memory = psutil.virtual_memory()
        
        # Python ガベージコレクション強制実行
        collected = gc.collect()
        self.log_optimization(
            "ガベージコレクション",
            f"{collected}オブジェクト回収",
            "メモリ断片化を解消"
        )
        
        # メモリ統計更新
        optimized_memory = psutil.virtual_memory()
        memory_freed = (initial_memory.used - optimized_memory.used) / (1024**2)
        
        if memory_freed > 0:
            self.log_optimization(
                "メモリ解放",
                f"{memory_freed:.1f}MB解放",
                f"利用可能メモリ: {optimized_memory.available / (1024**3):.1f}GB"
            )
    
    def optimize_processes(self):
        """プロセス最適化"""
        print("⚙️ プロセス最適化中...")
        
        current_process = psutil.Process()
        
        # プロセス優先度を高に設定（Windows）
        try:
            if sys.platform == "win32":
                current_process.nice(psutil.HIGH_PRIORITY_CLASS)
                self.log_optimization(
                    "プロセス優先度",
                    "高優先度に設定",
                    "CPU時間の優先割り当て"
                )
            else:
                current_process.nice(-5)  # Unix系
                self.log_optimization(
                    "プロセス優先度",
                    "高優先度に設定 (nice -5)",
                    "CPU時間の優先割り当て"
                )
        except (psutil.AccessDenied, OSError) as e:
            self.log_optimization(
                "プロセス優先度",
                f"設定失敗: {e}",
                "管理者権限が必要な場合があります"
            )
        
        # CPU親和性設定（利用可能な全CPUコアを使用）
        try:
            cpu_count = psutil.cpu_count()
            current_process.cpu_affinity(list(range(cpu_count)))
            self.log_optimization(
                "CPU親和性",
                f"全{cpu_count}コアを使用",
                "並列処理性能向上"
            )
        except (psutil.AccessDenied, OSError, AttributeError) as e:
            self.log_optimization(
                "CPU親和性",
                f"設定失敗: {e}",
                "プラットフォーム制限の可能性"
            )
    
    def optimize_environment_variables(self):
        """環境変数最適化"""
        print("🌍 環境変数最適化中...")
        
        optimizations = [
            # Python最適化
            ("PYTHONUNBUFFERED", "1", "Python出力バッファリング無効化"),
            ("PYTHONDONTWRITEBYTECODE", "1", "pycファイル生成無効化"),
            
            # Ollama最適化
            ("OLLAMA_NUM_PARALLEL", "4", "並列リクエスト数増加"),
            ("OLLAMA_MAX_LOADED_MODELS", "2", "同時ロードモデル数制限"),
            ("OLLAMA_FLASH_ATTENTION", "1", "Flash Attention有効化"),
            
            # システム最適化
            ("OMP_NUM_THREADS", str(psutil.cpu_count()), "OpenMP並列スレッド数最適化"),
        ]
        
        for var_name, var_value, description in optimizations:
            old_value = os.environ.get(var_name)
            os.environ[var_name] = var_value
            
            if old_value != var_value:
                self.log_optimization(
                    f"環境変数 {var_name}",
                    f"{old_value or 'None'} → {var_value}",
                    description
                )
    
    def optimize_file_system(self):
        """ファイルシステム最適化"""
        print("📁 ファイルシステム最適化中...")
        
        # 一時ファイル削除
        temp_dirs = [
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache"
        ]
        
        total_freed = 0
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    import shutil
                    dir_size = sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, dirnames, filenames in os.walk(temp_dir)
                        for filename in filenames
                    )
                    shutil.rmtree(temp_dir)
                    total_freed += dir_size
                    self.log_optimization(
                        f"一時ディレクトリ削除",
                        f"{temp_dir} ({dir_size / (1024**2):.1f}MB)",
                        "ディスク容量解放"
                    )
                except Exception as e:
                    self.log_optimization(
                        f"一時ディレクトリ削除",
                        f"{temp_dir} 削除失敗: {e}",
                        None
                    )
        
        if total_freed > 0:
            self.log_optimization(
                "総ディスク解放",
                f"{total_freed / (1024**2):.1f}MB",
                "ストレージ最適化完了"
            )
    
    def optimize_database_settings(self):
        """データベース最適化設定"""
        print("🗄️ データベース最適化中...")
        
        # SQLite最適化設定
        sqlite_optimizations = {
            "PRAGMA journal_mode": "WAL",
            "PRAGMA synchronous": "NORMAL", 
            "PRAGMA cache_size": "10000",
            "PRAGMA temp_store": "MEMORY",
            "PRAGMA mmap_size": "268435456"  # 256MB
        }
        
        optimization_file = "data/sqlite_optimizations.sql"
        os.makedirs("data", exist_ok=True)
        
        try:
            with open(optimization_file, 'w') as f:
                for pragma, value in sqlite_optimizations.items():
                    f.write(f"{pragma} = {value};\n")
            
            self.log_optimization(
                "データベース設定",
                f"最適化設定を {optimization_file} に保存",
                "SQLite性能向上設定"
            )
        except Exception as e:
            self.log_optimization(
                "データベース設定",
                f"設定保存失敗: {e}",
                None
            )
    
    def create_performance_config(self):
        """パフォーマンス設定ファイル作成"""
        print("⚡ パフォーマンス設定作成中...")
        
        # システム情報に基づく最適化設定
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        
        # メモリ使用量の調整
        if memory_gb >= 16:
            max_agents = 4
            agent_memory_limit = "2GB"
            batch_size = 10
        elif memory_gb >= 8:
            max_agents = 4
            agent_memory_limit = "1GB"
            batch_size = 5
        else:
            max_agents = 2
            agent_memory_limit = "512MB"
            batch_size = 3
        
        performance_config = {
            "system_info": {
                "memory_gb": round(memory_gb, 2),
                "cpu_count": cpu_count,
                "optimization_timestamp": datetime.now().isoformat()
            },
            "multi_agent_settings": {
                "max_concurrent_agents": max_agents,
                "agent_memory_limit": agent_memory_limit,
                "conversation_batch_size": batch_size,
                "cycle_interval_seconds": 300,  # 5分
                "max_conversation_history": 100
            },
            "ollama_settings": {
                "num_parallel": min(4, cpu_count),
                "max_loaded_models": 2,
                "context_length": 4096,
                "temperature": 0.7,
                "timeout_seconds": 30
            },
            "database_settings": {
                "connection_pool_size": 5,
                "max_overflow": 10,
                "pool_timeout": 30,
                "pool_recycle": 3600
            },
            "logging_settings": {
                "level": "INFO",
                "max_file_size_mb": 100,
                "backup_count": 5,
                "console_output": True
            }
        }
        
        config_file = "config/performance_config.json"
        os.makedirs("config", exist_ok=True)
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(performance_config, f, ensure_ascii=False, indent=2)
            
            self.log_optimization(
                "パフォーマンス設定",
                f"設定を {config_file} に保存",
                f"最大エージェント数: {max_agents}, メモリ制限: {agent_memory_limit}"
            )
        except Exception as e:
            self.log_optimization(
                "パフォーマンス設定",
                f"設定保存失敗: {e}",
                None
            )
    
    def setup_monitoring(self):
        """監視設定"""
        print("📊 監視設定中...")
        
        # 監視スクリプト作成
        monitoring_script = """#!/usr/bin/env python3
import psutil
import time
import json
from datetime import datetime

def monitor_system():
    while True:
        stats = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('.').percent,
            'process_count': len(psutil.pids())
        }
        
        with open('system_monitor.log', 'a') as f:
            f.write(json.dumps(stats) + '\\n')
        
        time.sleep(60)  # 1分間隔

if __name__ == "__main__":
    monitor_system()
"""
        
        try:
            with open("system_monitor.py", 'w') as f:
                f.write(monitoring_script)
            
            self.log_optimization(
                "システム監視",
                "監視スクリプト作成完了",
                "system_monitor.py で実行可能"
            )
        except Exception as e:
            self.log_optimization(
                "システム監視",
                f"スクリプト作成失敗: {e}",
                None
            )
    
    def save_optimization_report(self):
        """最適化レポート保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimization_report_{timestamp}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'initial_stats': self.initial_stats,
            'optimized_stats': self.optimized_stats,
            'optimization_log': self.optimization_log,
            'summary': {
                'total_optimizations': len(self.optimization_log),
                'successful_optimizations': sum(1 for log in self.optimization_log if '失敗' not in log['result']),
                'memory_improvement': None,
                'cpu_improvement': None
            }
        }
        
        # 改善度計算
        if self.initial_stats and self.optimized_stats:
            memory_improvement = self.initial_stats.get('memory_percent', 0) - self.optimized_stats.get('memory_percent', 0)
            cpu_improvement = self.initial_stats.get('cpu_percent', 0) - self.optimized_stats.get('cpu_percent', 0)
            
            report['summary']['memory_improvement'] = f"{memory_improvement:.1f}%"
            report['summary']['cpu_improvement'] = f"{cpu_improvement:.1f}%"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"\n📄 最適化レポートを保存: {filename}")
        except Exception as e:
            print(f"\n⚠️ レポート保存エラー: {e}")
    
    def print_summary(self):
        """最適化サマリー表示"""
        successful = sum(1 for log in self.optimization_log if '失敗' not in log['result'])
        total = len(self.optimization_log)
        
        print(f"\n{'='*60}")
        print(f"⚡ システム最適化完了")
        print(f"{'='*60}")
        print(f"実行した最適化: {total}")
        print(f"成功: {successful}")
        print(f"失敗: {total - successful}")
        
        if self.initial_stats and self.optimized_stats:
            memory_improvement = self.initial_stats.get('memory_percent', 0) - self.optimized_stats.get('memory_percent', 0)
            print(f"メモリ使用量改善: {memory_improvement:.1f}%")
        
        print(f"\n💡 次のステップ:")
        print(f"1. python pre_launch_checklist.py でシステム状態を確認")
        print(f"2. python multi_agent_learning_system.py で8時間学習を開始")
        print(f"3. python system_monitor.py でリアルタイム監視（別ターミナル）")
        print(f"{'='*60}")


async def main():
    """メイン関数"""
    print("⚡ マルチエージェント学習システム 最適化ツール")
    print("=" * 60)
    print("8時間実行に向けてシステムを最適化します")
    print("=" * 60)
    
    optimizer = SystemOptimizer()
    
    try:
        # 初期統計収集
        print("📊 初期システム状態を記録中...")
        optimizer.initial_stats = optimizer.collect_system_stats()
        print(f"  初期メモリ使用量: {optimizer.initial_stats['memory_percent']:.1f}%")
        print(f"  初期CPU使用量: {optimizer.initial_stats['cpu_percent']:.1f}%")
        print()
        
        # 各種最適化実行
        optimizer.optimize_memory()
        print()
        
        optimizer.optimize_processes()
        print()
        
        optimizer.optimize_environment_variables()
        print()
        
        optimizer.optimize_file_system()
        print()
        
        optimizer.optimize_database_settings()
        print()
        
        optimizer.create_performance_config()
        print()
        
        optimizer.setup_monitoring()
        print()
        
        # 最適化後統計収集
        print("📊 最適化後システム状態を記録中...")
        optimizer.optimized_stats = optimizer.collect_system_stats()
        print(f"  最適化後メモリ使用量: {optimizer.optimized_stats['memory_percent']:.1f}%")
        print(f"  最適化後CPU使用量: {optimizer.optimized_stats['cpu_percent']:.1f}%")
        
        # サマリー表示
        optimizer.print_summary()
        
        # レポート保存
        optimizer.save_optimization_report()
        
    except Exception as e:
        print(f"\n❌ 最適化エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())