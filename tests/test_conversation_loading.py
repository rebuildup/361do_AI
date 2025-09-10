#!/usr/bin/env python3
"""
Test Conversation Loading
会話データ読み込みテストスクリプト
"""

import sys
import os
sys.path.insert(0, '.')

from continuous_learning_system import ContinuousLearningSystem
import asyncio

async def test_conversation_loading():
    """会話データ読み込みテスト"""
    try:
        system = ContinuousLearningSystem()
        
        # ChatGPTデータの読み込みテスト
        print('Testing ChatGPT data loading...')
        chatgpt_data = system._load_chatgpt_data()
        print(f'Loaded {len(chatgpt_data)} ChatGPT conversations')
        
        # Claudeデータの読み込みテスト
        print('Testing Claude data loading...')
        claude_data = system._load_claude_data()
        print(f'Loaded {len(claude_data)} Claude conversations')
        
        # 合計
        total = len(chatgpt_data) + len(claude_data)
        print(f'Total conversations loaded: {total}')
        
        # サンプル表示
        if chatgpt_data:
            print('\nSample ChatGPT conversation:')
            sample = chatgpt_data[0]
            print(f'ID: {sample.get("id", "N/A")}')
            print(f'Title: {sample.get("title", "N/A")}')
            print(f'Content length: {len(sample.get("content", ""))}')
            print(f'Source: {sample.get("source", "N/A")}')
            
            # コンテンツの一部を表示
            content = sample.get("content", "")
            if content:
                preview = content[:200] + "..." if len(content) > 200 else content
                print(f'Content preview: {preview}')
        
        if claude_data:
            print('\nSample Claude conversation:')
            sample = claude_data[0]
            print(f'ID: {sample.get("id", "N/A")}')
            print(f'Title: {sample.get("title", "N/A")}')
            print(f'Content length: {len(sample.get("content", ""))}')
            print(f'Source: {sample.get("source", "N/A")}')
        
        return total > 0
        
    except Exception as e:
        print(f'Error during testing: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_conversation_loading())
    if success:
        print('\n✅ Conversation loading test completed successfully!')
    else:
        print('\n❌ Conversation loading test failed!')
