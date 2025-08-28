#!/usr/bin/env python3
"""
Restart Multi-Agent Learning System
マルチエージェント学習システム再起動スクリプト
"""

import asyncio
import sys
import os
import time
import subprocess
from datetime import datetime
from pathlib import Path

def setup_windows_encoding():
    """Windows環境での文字エンコーディング設定"""
    if sys.platform == "win32":
        # 環境変数設定
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        os.environ['PYTHONUTF8'] = '1'
        
        # コンソールのコードページをUTF-8に設定
        try:
            subprocess.run(['chcp', '65001'], shell=True, capture_output=True)
            print("Windows コンソールをUTF-8に設定しました")
        except Exception as e:
            print(f"コンソール設定エラー: {e}")

def check_system_status():
    """システム状態確認"""
    print("システム状態を確認中...")
    
    # Ollamaプロセス確認
    try:
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq ollama.exe'], 
                              capture_output=True, text=True)
        if 'ollama.exe' in result.stdout:
            print("✅ Ollama プロセスが実行中")
        else:
            print("⚠️ Ollama プロセスが見つかりません")
            print("   ollama serve コマンドでOllamaを起動してください")
    except Exception as e:
        print(f"Ollama確認エラー: {e}")
    
    # ログファイル確認
    log_files = [f for f in os.listdir('.') if f.startswith('multi_agent_learning_') and f.endswith('.log')]
    if log_files:
        latest_log = max(log_files, key=lambda x: os.path.getmtime(x))
        print(f"✅ 最新ログファイル: {latest_log}")
        
        # ログファイルサイズ確認
        size_mb = os.path.getsize(latest_log) / (1024**2)
        print(f"   ファイルサイズ: {size_mb:.1f}MB")
    else:
        print("⚠️ ログファイルが見つかりません")

def clean_old_logs():
    """古いログファイルの整理"""
    print("古いログファイルを整理中...")
    
    log_files = [f for f in os.listdir('.') if f.startswith('multi_agent_learning_') and f.endswith('.log')]
    
    if len(log_files) > 3:  # 3個以上ある場合は古いものを削除
        log_files.sort(key=lambda x: os.path.getmtime(x))
        for old_log in log_files[:-2]:  # 最新2個を残す
            try:
                os.remove(old_log)
                print(f"   削除: {old_log}")
            except Exception as e:
                print(f"   削除失敗: {old_log} - {e}")

def restart_system(hours=8.0, test_mode=False):
    """システム再起動"""
    print(f"\nマルチエージェント学習システムを再起動します")
    print(f"実行時間: {'テストモード (6分)' if test_mode else f'{hours}時間'}")
    print("=" * 60)
    
    # コマンド構築
    cmd = [sys.executable, "multi_agent_learning_system.py"]
    if test_mode:
        cmd.append("--test-mode")
    else:
        cmd.extend(["--hours", str(hours)])
    
    try:
        # システム実行
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 text=True, encoding='utf-8', bufsize=1, universal_newlines=True)
        
        print("システムが起動しました。出力を監視中...")
        print("停止するには Ctrl+C を押してください")
        print("=" * 60)
        
        # リアルタイム出力表示
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                # 文字化け対策：エラーを無視して表示
                try:
                    print(output.strip())
                except UnicodeEncodeError:
                    print(output.encode('ascii', 'ignore').decode('ascii').strip())
        
        # 終了コード確認
        return_code = process.poll()
        
        if return_code == 0:
            print("\n✅ システムが正常に終了しました")
        else:
            print(f"\n⚠️ システムが終了コード {return_code} で終了しました")
        
        return return_code == 0
        
    except KeyboardInterrupt:
        print("\nユーザーによって中断されました")
        if 'process' in locals():
            process.terminate()
        return True
    except Exception as e:
        print(f"\nシステム起動エラー: {e}")
        return False

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="マルチエージェント学習システム再起動")
    parser.add_argument("--hours", type=float, default=8.0, help="実行時間（時間）")
    parser.add_argument("--test-mode", action="store_true", help="テストモード（6分間）")
    parser.add_argument("--no-cleanup", action="store_true", help="ログファイル整理をスキップ")
    
    args = parser.parse_args()
    
    print("マルチエージェント学習システム 再起動ツール")
    print("=" * 60)
    print(f"起動時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Windows環境設定
    setup_windows_encoding()
    print()
    
    # システム状態確認
    check_system_status()
    print()
    
    # ログファイル整理
    if not args.no_cleanup:
        clean_old_logs()
        print()
    
    # システム再起動
    success = restart_system(args.hours, args.test_mode)
    
    if success:
        print("\n🎉 実行完了")
    else:
        print("\n❌ 実行中にエラーが発生しました")

if __name__ == "__main__":
    main()