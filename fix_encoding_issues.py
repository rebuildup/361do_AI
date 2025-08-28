#!/usr/bin/env python3
"""
Fix Encoding Issues for Multi-Agent Learning System
マルチエージェント学習システムの文字エンコーディング問題修正
"""

import os
import sys
import locale

def fix_windows_encoding():
    """Windows環境での文字エンコーディング問題を修正"""
    
    print("🔧 Windows文字エンコーディング修正中...")
    
    # 環境変数設定
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
    
    # コンソールのコードページをUTF-8に設定（Windows）
    if sys.platform == "win32":
        try:
            import subprocess
            subprocess.run(['chcp', '65001'], shell=True, capture_output=True)
            print("✅ Windows コンソールをUTF-8に設定")
        except Exception as e:
            print(f"⚠️ コンソール設定エラー: {e}")
    
    # ロケール設定
    try:
        if sys.platform == "win32":
            locale.setlocale(locale.LC_ALL, 'Japanese_Japan.UTF-8')
        else:
            locale.setlocale(locale.LC_ALL, 'ja_JP.UTF-8')
        print("✅ ロケールをUTF-8に設定")
    except Exception as e:
        print(f"⚠️ ロケール設定エラー: {e}")
    
    print("✅ エンコーディング修正完了")

if __name__ == "__main__":
    fix_windows_encoding()