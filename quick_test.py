#!/usr/bin/env python3
"""
Quick Test Script
簡単なテスト実行用スクリプト
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from automated_chat import AutomatedChatSystem


async def quick_test():
    """クイックテスト実行"""
    
    # テスト質問
    test_questions = [
        "こんにちは",
        "systeminfoコマンドを実行してください",
        "help tools",
        "最新のAI技術について調べて",
        "ありがとうございました"
    ]
    
    print("[QUICK] クイックテスト開始...")
    
    chat_system = AutomatedChatSystem()
    
    try:
        if await chat_system.initialize():
            results = await chat_system.execute_chat_sequence(test_questions)
            chat_system.print_summary(results)
            
            # 簡易結果保存
            chat_system.save_results(results, "quick_test_results.json")
            
        else:
            print("[ERROR] システム初期化に失敗しました")
            
    except Exception as e:
        print(f"[ERROR] テスト実行エラー: {e}")
    finally:
        await chat_system.shutdown()


if __name__ == "__main__":
    asyncio.run(quick_test())