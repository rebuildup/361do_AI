#!/usr/bin/env python3
"""
GPU Performance Test Script
GPUを使用したエージェントのパフォーマンステスト
"""

import asyncio
import time
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from codex_agent.config import CodexConfig
from codex_agent.ollama_client import CodexOllamaClient


async def test_gpu_performance():
    """GPU パフォーマンステスト"""
    print("🚀 GPU パフォーマンステスト開始")
    print("=" * 50)
    
    # GPU設定を有効化
    os.environ["OLLAMA_GPU_ENABLED"] = "true"
    os.environ["OLLAMA_GPU_MEMORY_FRACTION"] = "0.8"
    os.environ["OLLAMA_GPU_LAYERS"] = "32"
    os.environ["OLLAMA_PARALLEL_REQUESTS"] = "4"
    
    # 設定を読み込み
    config = CodexConfig()
    print(f"📊 設定情報:")
    print(f"  モデル: {config.model}")
    print(f"  GPU有効: {config.gpu_enabled}")
    print(f"  GPUメモリ使用率: {config.gpu_memory_fraction}")
    print(f"  GPU層数: {config.gpu_layers}")
    print(f"  並列リクエスト数: {config.parallel_requests}")
    print()
    
    # OLLAMAクライアント初期化
    async with CodexOllamaClient(config) as client:
        print("✅ OLLAMAクライアント初期化完了")
        
        # モデル情報を表示
        model_info = client.get_model_info()
        print(f"📋 モデル情報:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        print()
        
        # テストプロンプト
        test_prompts = [
            "Pythonでファイルを読み込む簡単なコードを書いてください。",
            "機械学習の基本概念を3つ説明してください。",
            "Webアプリケーションの基本的なアーキテクチャについて説明してください。",
            "データベース設計の基本原則を教えてください。"
        ]
        
        print("🧪 パフォーマンステスト実行中...")
        
        # 単一リクエストテスト
        print("\n📈 単一リクエストテスト:")
        for i, prompt in enumerate(test_prompts[:2], 1):
            print(f"  テスト {i}: {prompt[:30]}...")
            
            start_time = time.time()
            response = await client.generate(prompt, max_tokens=200)
            end_time = time.time()
            
            response_time = end_time - start_time
            tokens_per_second = len(response.response.split()) / response_time if response_time > 0 else 0
            
            print(f"    応答時間: {response_time:.2f}秒")
            print(f"    トークン/秒: {tokens_per_second:.1f}")
            print(f"    応答長: {len(response.response)}文字")
            print()
        
        # 並列リクエストテスト
        print("🔄 並列リクエストテスト:")
        start_time = time.time()
        
        tasks = []
        for i, prompt in enumerate(test_prompts, 1):
            task = client.generate(f"簡潔に答えてください: {prompt}", max_tokens=100)
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_request = total_time / len(tasks)
        
        print(f"  並列リクエスト数: {len(tasks)}")
        print(f"  総実行時間: {total_time:.2f}秒")
        print(f"  平均応答時間: {avg_time_per_request:.2f}秒")
        print(f"  スループット: {len(tasks)/total_time:.2f} req/sec")
        print()
        
        # ストリーミングテスト
        print("📡 ストリーミングテスト:")
        stream_prompt = "Pythonの特徴について詳しく説明してください。"
        
        start_time = time.time()
        first_token_time = None
        token_count = 0
        
        print(f"  プロンプト: {stream_prompt}")
        print("  応答: ", end="", flush=True)
        
        async for chunk in client.generate_stream(stream_prompt, max_tokens=300):
            if first_token_time is None:
                first_token_time = time.time()
            
            if chunk.response:
                print(chunk.response, end="", flush=True)
                token_count += len(chunk.response.split())
            
            if chunk.done:
                break
        
        end_time = time.time()
        
        if first_token_time:
            time_to_first_token = first_token_time - start_time
            total_stream_time = end_time - start_time
            tokens_per_second = token_count / total_stream_time if total_stream_time > 0 else 0
            
            print(f"\n  初回トークン時間: {time_to_first_token:.2f}秒")
            print(f"  総ストリーミング時間: {total_stream_time:.2f}秒")
            print(f"  ストリーミング速度: {tokens_per_second:.1f} tokens/sec")
        
        print("\n" + "=" * 50)
        print("✅ GPU パフォーマンステスト完了")


async def compare_cpu_gpu_performance():
    """CPU vs GPU パフォーマンス比較"""
    print("⚖️  CPU vs GPU パフォーマンス比較")
    print("=" * 50)
    
    test_prompt = "Pythonでファイルを読み込むコードを書いてください。"
    
    # CPU設定でテスト
    print("🖥️  CPU モードテスト:")
    os.environ["OLLAMA_GPU_ENABLED"] = "false"
    config_cpu = CodexConfig()
    
    async with CodexOllamaClient(config_cpu) as client_cpu:
        start_time = time.time()
        response_cpu = await client_cpu.generate(test_prompt, max_tokens=200)
        cpu_time = time.time() - start_time
        
        print(f"  応答時間: {cpu_time:.2f}秒")
        print(f"  応答長: {len(response_cpu.response)}文字")
    
    # GPU設定でテスト
    print("\n🚀 GPU モードテスト:")
    os.environ["OLLAMA_GPU_ENABLED"] = "true"
    config_gpu = CodexConfig()
    
    async with CodexOllamaClient(config_gpu) as client_gpu:
        start_time = time.time()
        response_gpu = await client_gpu.generate(test_prompt, max_tokens=200)
        gpu_time = time.time() - start_time
        
        print(f"  応答時間: {gpu_time:.2f}秒")
        print(f"  応答長: {len(response_gpu.response)}文字")
    
    # 比較結果
    if cpu_time > 0 and gpu_time > 0:
        speedup = cpu_time / gpu_time
        print(f"\n📊 比較結果:")
        print(f"  GPU高速化倍率: {speedup:.2f}x")
        print(f"  時間短縮: {((cpu_time - gpu_time) / cpu_time * 100):.1f}%")


async def main():
    """メイン関数"""
    try:
        await test_gpu_performance()
        print("\n" + "=" * 50)
        await compare_cpu_gpu_performance()
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())