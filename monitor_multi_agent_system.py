#!/usr/bin/env python3
"""
Real-time Monitor for Multi-Agent Learning System
マルチエージェント学習システムのリアルタイム監視
"""

import asyncio
import sys
import os
import time
import json
import psutil
from datetime import datetime
from pathlib import Path

def monitor_system_resources():
    """システムリソース監視"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('.')
    
    return {
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'memory_available_gb': memory.available / (1024**3),
        'disk_free_gb': disk.free / (1024**3),
        'process_count': len(psutil.pids())
    }

def monitor_log_files():
    """ログファイル監視"""
    log_files = []
    for file in os.listdir('.'):
        if file.startswith('multi_agent_learning_') and file.endswith('.log'):
            try:
                stat = os.stat(file)
                log_files.append({
                    'filename': file,
                    'size_mb': stat.st_size / (1024**2),
                    'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%H:%M:%S')
                })
            except:
                pass
    
    return log_files

def get_latest_log_lines(filename, lines=5):
    """最新のログ行を取得"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            return all_lines[-lines:] if len(all_lines) >= lines else all_lines
    except:
        return []

def print_status_dashboard():
    """ステータスダッシュボード表示"""
    # 画面クリア
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("=" * 80)
    print("マルチエージェント学習システム リアルタイム監視")
    print("=" * 80)
    print(f"監視時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # システムリソース
    resources = monitor_system_resources()
    print("システムリソース:")
    print(f"  CPU使用率: {resources['cpu_percent']:.1f}%")
    print(f"  メモリ使用率: {resources['memory_percent']:.1f}%")
    print(f"  利用可能メモリ: {resources['memory_available_gb']:.1f}GB")
    print(f"  空きディスク: {resources['disk_free_gb']:.1f}GB")
    print(f"  プロセス数: {resources['process_count']}")
    print()
    
    # ログファイル状況
    log_files = monitor_log_files()
    if log_files:
        print("ログファイル:")
        for log_file in log_files:
            print(f"  {log_file['filename']}: {log_file['size_mb']:.1f}MB (更新: {log_file['modified']})")
        print()
        
        # 最新ログ内容
        latest_log = max(log_files, key=lambda x: x['modified'])
        print(f"最新ログ ({latest_log['filename']}):")
        latest_lines = get_latest_log_lines(latest_log['filename'], 10)
        for line in latest_lines:
            print(f"  {line.strip()}")
    else:
        print("ログファイルが見つかりません")
    
    print()
    print("=" * 80)
    print("監視を停止するには Ctrl+C を押してください")
    print("=" * 80)

async def continuous_monitoring():
    """継続監視"""
    try:
        while True:
            print_status_dashboard()
            await asyncio.sleep(10)  # 10秒間隔で更新
    except KeyboardInterrupt:
        print("\n監視を停止しました")

def check_multi_agent_process():
    """マルチエージェントプロセスの確認"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['cmdline'] and any('multi_agent_learning_system.py' in arg for arg in proc.info['cmdline']):
                return {
                    'pid': proc.info['pid'],
                    'status': 'running',
                    'cpu_percent': proc.cpu_percent(),
                    'memory_mb': proc.memory_info().rss / (1024**2)
                }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None

def main():
    """メイン関数"""
    print("マルチエージェント学習システム 監視ツール")
    print("=" * 50)
    
    # プロセス確認
    process_info = check_multi_agent_process()
    if process_info:
        print(f"マルチエージェントシステムが実行中です (PID: {process_info['pid']})")
        print(f"CPU使用率: {process_info['cpu_percent']:.1f}%")
        print(f"メモリ使用量: {process_info['memory_mb']:.1f}MB")
        print()
        print("リアルタイム監視を開始します...")
        print()
        
        # 継続監視開始
        asyncio.run(continuous_monitoring())
    else:
        print("マルチエージェントシステムが実行されていません")
        print()
        print("現在のシステム状況:")
        print_status_dashboard()

if __name__ == "__main__":
    main()