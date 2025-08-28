#!/usr/bin/env python3
"""
Pre-Launch Checklist for Multi-Agent Learning System
マルチエージェント学習システム実行前チェックリスト
"""

import asyncio
import sys
import os
import time
import json
import psutil
from datetime import datetime
from pathlib import Path
import subprocess
import requests

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from agent.core.config import Config
    from agent.core.database import DatabaseManager
    from agent.core.ollama_client import OllamaClient
except ImportError as e:
    print(f"⚠️ インポートエラー: {e}")
    print("src/ディレクトリが正しく設定されているか確認してください")


class PreLaunchChecker:
    """実行前チェッククラス"""
    
    def __init__(self):
        self.checks = []
        self.warnings = []
        self.errors = []
        self.system_info = {}
        
    def add_check(self, name: str, status: bool, message: str, critical: bool = False):
        """チェック結果を追加"""
        self.checks.append({
            'name': name,
            'status': status,
            'message': message,
            'critical': critical,
            'timestamp': datetime.now().isoformat()
        })
        
        if not status:
            if critical:
                self.errors.append(f"{name}: {message}")
            else:
                self.warnings.append(f"{name}: {message}")
    
    def check_system_resources(self):
        """システムリソースチェック"""
        print("🖥️  システムリソースをチェック中...")
        
        # CPU情報
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        self.system_info['cpu'] = {'count': cpu_count, 'usage': cpu_percent}
        
        # メモリ情報
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        self.system_info['memory'] = {
            'total_gb': round(memory_gb, 2),
            'available_gb': round(memory_available_gb, 2),
            'usage_percent': memory.percent
        }
        
        # ディスク情報
        disk = psutil.disk_usage('.')
        disk_free_gb = disk.free / (1024**3)
        self.system_info['disk'] = {
            'free_gb': round(disk_free_gb, 2),
            'usage_percent': round((disk.used / disk.total) * 100, 2)
        }
        
        # リソースチェック
        self.add_check(
            "CPU数", 
            cpu_count >= 4, 
            f"CPU数: {cpu_count} (推奨: 4コア以上)",
            critical=False
        )
        
        self.add_check(
            "メモリ容量", 
            memory_gb >= 8, 
            f"メモリ: {memory_gb:.1f}GB (推奨: 8GB以上)",
            critical=True
        )
        
        self.add_check(
            "利用可能メモリ", 
            memory_available_gb >= 4, 
            f"利用可能メモリ: {memory_available_gb:.1f}GB (推奨: 4GB以上)",
            critical=True
        )
        
        self.add_check(
            "ディスク容量", 
            disk_free_gb >= 2, 
            f"空きディスク: {disk_free_gb:.1f}GB (推奨: 2GB以上)",
            critical=True
        )
        
        print(f"  ✅ CPU: {cpu_count}コア ({cpu_percent:.1f}%使用中)")
        print(f"  ✅ メモリ: {memory_gb:.1f}GB ({memory_available_gb:.1f}GB利用可能)")
        print(f"  ✅ ディスク: {disk_free_gb:.1f}GB空き")
    
    def check_python_environment(self):
        """Python環境チェック"""
        print("🐍 Python環境をチェック中...")
        
        # Pythonバージョン
        python_version = sys.version_info
        version_str = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
        
        self.add_check(
            "Pythonバージョン",
            python_version >= (3, 8),
            f"Python {version_str} (推奨: 3.8以上)",
            critical=True
        )
        
        # 仮想環境チェック
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        
        self.add_check(
            "仮想環境",
            in_venv,
            "仮想環境が有効" if in_venv else "仮想環境が無効（推奨: 仮想環境の使用）",
            critical=False
        )
        
        # 必要なパッケージチェック
        required_packages = [
            'asyncio', 'aiohttp', 'sqlalchemy', 'loguru', 'psutil'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        self.add_check(
            "必要パッケージ",
            len(missing_packages) == 0,
            f"不足パッケージ: {missing_packages}" if missing_packages else "全パッケージ利用可能",
            critical=True
        )
        
        print(f"  ✅ Python {version_str}")
        print(f"  ✅ 仮想環境: {'有効' if in_venv else '無効'}")
        if missing_packages:
            print(f"  ❌ 不足パッケージ: {missing_packages}")
    
    def check_ollama_connection(self):
        """Ollama接続チェック"""
        print("🤖 Ollama接続をチェック中...")
        
        try:
            # Ollama API接続テスト
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                self.add_check(
                    "Ollama接続",
                    True,
                    f"接続成功 ({len(models)}モデル利用可能)",
                    critical=True
                )
                
                # 推奨モデルチェック
                recommended_models = ['qwen2:7b-instruct', 'llama3:8b', 'codellama:7b']
                available_recommended = [m for m in recommended_models if any(m in name for name in model_names)]
                
                self.add_check(
                    "推奨モデル",
                    len(available_recommended) > 0,
                    f"利用可能な推奨モデル: {available_recommended}" if available_recommended else "推奨モデルなし",
                    critical=False
                )
                
                print(f"  ✅ Ollama接続成功")
                print(f"  ✅ 利用可能モデル: {len(models)}個")
                if available_recommended:
                    print(f"  ✅ 推奨モデル: {available_recommended}")
                
            else:
                self.add_check(
                    "Ollama接続",
                    False,
                    f"HTTP {response.status_code}",
                    critical=True
                )
                print(f"  ❌ Ollama接続失敗: HTTP {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            self.add_check(
                "Ollama接続",
                False,
                "接続拒否 - Ollamaが起動していない可能性",
                critical=True
            )
            print("  ❌ Ollama接続失敗: サーバーが起動していません")
            
        except Exception as e:
            self.add_check(
                "Ollama接続",
                False,
                f"接続エラー: {str(e)}",
                critical=True
            )
            print(f"  ❌ Ollama接続エラー: {e}")
    
    async def check_database_connection(self):
        """データベース接続チェック"""
        print("🗄️  データベース接続をチェック中...")
        
        try:
            config = Config()
            db_manager = DatabaseManager(config.database_url)
            
            await db_manager.initialize()
            
            # 基本的なクエリテスト
            stats = await db_manager.get_learning_statistics()
            
            await db_manager.close()
            
            self.add_check(
                "データベース接続",
                True,
                f"接続成功 (学習データ: {stats.get('total_learning_data', 0)}件)",
                critical=True
            )
            
            print(f"  ✅ データベース接続成功")
            print(f"  ✅ 学習データ: {stats.get('total_learning_data', 0)}件")
            
        except Exception as e:
            self.add_check(
                "データベース接続",
                False,
                f"接続エラー: {str(e)}",
                critical=True
            )
            print(f"  ❌ データベース接続エラー: {e}")
    
    def check_file_permissions(self):
        """ファイル権限チェック"""
        print("📁 ファイル権限をチェック中...")
        
        # 重要なディレクトリの書き込み権限チェック
        important_dirs = ['data', 'src', '.']
        
        for dir_path in important_dirs:
            if os.path.exists(dir_path):
                writable = os.access(dir_path, os.W_OK)
                self.add_check(
                    f"{dir_path}ディレクトリ書き込み権限",
                    writable,
                    "書き込み可能" if writable else "書き込み不可",
                    critical=dir_path in ['data', '.']
                )
                
                if writable:
                    print(f"  ✅ {dir_path}: 書き込み可能")
                else:
                    print(f"  ❌ {dir_path}: 書き込み不可")
            else:
                print(f"  ⚠️ {dir_path}: ディレクトリが存在しません")
    
    def check_network_connectivity(self):
        """ネットワーク接続チェック"""
        print("🌐 ネットワーク接続をチェック中...")
        
        # インターネット接続テスト
        try:
            response = requests.get("https://www.google.com", timeout=5)
            internet_ok = response.status_code == 200
        except:
            internet_ok = False
        
        self.add_check(
            "インターネット接続",
            internet_ok,
            "接続可能" if internet_ok else "接続不可（Web検索機能が制限される可能性）",
            critical=False
        )
        
        print(f"  {'✅' if internet_ok else '⚠️'} インターネット接続: {'可能' if internet_ok else '不可'}")
    
    def estimate_performance(self):
        """パフォーマンス予測"""
        print("⚡ パフォーマンス予測中...")
        
        cpu_count = self.system_info.get('cpu', {}).get('count', 1)
        memory_gb = self.system_info.get('memory', {}).get('total_gb', 0)
        
        # パフォーマンス予測
        if cpu_count >= 8 and memory_gb >= 16:
            performance_level = "高性能"
            estimated_response_time = "2-5秒"
        elif cpu_count >= 4 and memory_gb >= 8:
            performance_level = "標準"
            estimated_response_time = "5-10秒"
        else:
            performance_level = "低性能"
            estimated_response_time = "10-20秒"
        
        print(f"  📊 予測パフォーマンス: {performance_level}")
        print(f"  ⏱️ 予測応答時間: {estimated_response_time}")
        
        # 8時間実行の推定リソース使用量
        estimated_conversations = 96  # 5分間隔で8時間
        estimated_learning_cycles = 96
        estimated_disk_usage = estimated_conversations * 0.1  # MB
        
        print(f"  📈 8時間実行予測:")
        print(f"    - 会話数: 約{estimated_conversations}回")
        print(f"    - 学習サイクル: 約{estimated_learning_cycles}回")
        print(f"    - ディスク使用量: 約{estimated_disk_usage:.1f}MB")
    
    def generate_recommendations(self):
        """推奨事項生成"""
        print("\n💡 推奨事項:")
        
        recommendations = []
        
        # メモリ不足の場合
        memory_gb = self.system_info.get('memory', {}).get('total_gb', 0)
        if memory_gb < 8:
            recommendations.append("メモリ不足: 他のアプリケーションを終了してください")
        
        # CPU不足の場合
        cpu_count = self.system_info.get('cpu', {}).get('count', 1)
        if cpu_count < 4:
            recommendations.append("CPU性能不足: 実行時間が長くなる可能性があります")
        
        # Ollama接続問題
        if any("Ollama接続" in check['name'] and not check['status'] for check in self.checks):
            recommendations.append("Ollamaを起動してください: ollama serve")
        
        # ディスク容量不足
        disk_free_gb = self.system_info.get('disk', {}).get('free_gb', 0)
        if disk_free_gb < 2:
            recommendations.append("ディスク容量不足: 不要なファイルを削除してください")
        
        # 一般的な推奨事項
        recommendations.extend([
            "実行前にシステムを再起動することを推奨",
            "8時間の実行中は他の重いタスクを避ける",
            "定期的にログファイルを確認",
            "異常を検知したらCtrl+Cで安全に停止"
        ])
        
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    def save_report(self):
        """チェック結果をファイルに保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pre_launch_check_{timestamp}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self.system_info,
            'checks': self.checks,
            'warnings': self.warnings,
            'errors': self.errors,
            'summary': {
                'total_checks': len(self.checks),
                'passed_checks': sum(1 for c in self.checks if c['status']),
                'failed_checks': sum(1 for c in self.checks if not c['status']),
                'critical_errors': len(self.errors),
                'warnings': len(self.warnings)
            }
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"\n📄 チェック結果を保存: {filename}")
        except Exception as e:
            print(f"\n⚠️ レポート保存エラー: {e}")
    
    def print_summary(self):
        """サマリー表示"""
        total_checks = len(self.checks)
        passed_checks = sum(1 for c in self.checks if c['status'])
        failed_checks = total_checks - passed_checks
        critical_errors = len(self.errors)
        
        print(f"\n{'='*60}")
        print(f"📋 事前チェック結果サマリー")
        print(f"{'='*60}")
        print(f"総チェック数: {total_checks}")
        print(f"成功: {passed_checks}")
        print(f"失敗: {failed_checks}")
        print(f"重要エラー: {critical_errors}")
        print(f"警告: {len(self.warnings)}")
        
        if critical_errors == 0:
            print(f"\n✅ システム準備完了！マルチエージェント学習を開始できます。")
        else:
            print(f"\n❌ 重要エラーがあります。以下を修正してから実行してください:")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print(f"\n⚠️ 警告:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        print(f"{'='*60}")


async def main():
    """メイン関数"""
    print("🚀 マルチエージェント学習システム 事前チェック")
    print("=" * 60)
    print("8時間の自己学習実行前にシステム状態を確認します")
    print("=" * 60)
    
    checker = PreLaunchChecker()
    
    try:
        # 各種チェック実行
        checker.check_system_resources()
        print()
        
        checker.check_python_environment()
        print()
        
        checker.check_ollama_connection()
        print()
        
        await checker.check_database_connection()
        print()
        
        checker.check_file_permissions()
        print()
        
        checker.check_network_connectivity()
        print()
        
        checker.estimate_performance()
        
        # 推奨事項表示
        checker.generate_recommendations()
        
        # サマリー表示
        checker.print_summary()
        
        # レポート保存
        checker.save_report()
        
    except Exception as e:
        print(f"\n❌ チェック実行エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())