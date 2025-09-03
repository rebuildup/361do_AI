"""
HuggingFace Code Generator デモスクリプト

RTX 4050 6GB VRAM環境でのコード生成機能をデモンストレーションします。

使用方法:
    python -m src.advanced_agent.multimodal.demo_code_generator

要件: 3.2, 3.5
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import logging

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.advanced_agent.multimodal.code_generator import (
    HuggingFaceCodeGenerator,
    CodeGenerationResult
)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CodeGeneratorDemo:
    """コード生成デモクラス"""
    
    def __init__(self):
        self.generator = HuggingFaceCodeGenerator(
            model_name="microsoft/CodeGPT-small-py",  # 軽量モデル使用
            max_vram_gb=4.0  # RTX 4050用設定
        )
        
        # デモ用プロンプト
        self.demo_prompts = {
            "basic": [
                "Create a function to calculate the factorial of a number",
                "Write a function to check if a number is prime",
                "Implement a simple calculator class"
            ],
            "algorithms": [
                "Implement bubble sort algorithm",
                "Create a binary search function",
                "Write a function to find the longest common subsequence"
            ],
            "data_structures": [
                "Implement a stack class with push, pop, and peek methods",
                "Create a simple linked list class",
                "Write a binary tree class with insert and search methods"
            ],
            "web": [
                "Create a simple Flask web application",
                "Write a function to make HTTP requests",
                "Implement a basic REST API endpoint"
            ]
        }
    
    async def run_demo(self):
        """デモ実行"""
        print("=" * 60)
        print("HuggingFace Code Generator Demo")
        print("RTX 4050 6GB VRAM Optimized")
        print("=" * 60)
        
        try:
            # 初期化
            print("\n🔧 Initializing code generator...")
            start_time = time.time()
            
            if not await self.generator.initialize():
                print("❌ Failed to initialize code generator")
                return
            
            init_time = time.time() - start_time
            print(f"✅ Initialization completed in {init_time:.2f} seconds")
            
            # メモリ使用量表示
            memory_usage = self.generator.get_memory_usage()
            if memory_usage:
                print(f"📊 GPU Memory: {memory_usage.get('gpu_allocated_mb', 0):.1f} MB allocated")
            
            # インタラクティブデモ
            await self._interactive_demo()
            
        except KeyboardInterrupt:
            print("\n\n⏹️  Demo interrupted by user")
        except Exception as e:
            print(f"\n❌ Demo error: {e}")
            logger.error(f"Demo error: {e}", exc_info=True)
        finally:
            print("\n🧹 Cleaning up...")
            await self.generator.cleanup()
            print("✅ Cleanup completed")
    
    async def _interactive_demo(self):
        """インタラクティブデモ"""
        while True:
            print("\n" + "=" * 50)
            print("Choose demo mode:")
            print("1. Basic Functions")
            print("2. Algorithms")
            print("3. Data Structures")
            print("4. Web Development")
            print("5. Custom Prompt")
            print("6. Batch Generation")
            print("7. Performance Test")
            print("0. Exit")
            print("=" * 50)
            
            try:
                choice = input("\nEnter your choice (0-7): ").strip()
                
                if choice == "0":
                    break
                elif choice == "1":
                    await self._demo_category("basic")
                elif choice == "2":
                    await self._demo_category("algorithms")
                elif choice == "3":
                    await self._demo_category("data_structures")
                elif choice == "4":
                    await self._demo_category("web")
                elif choice == "5":
                    await self._custom_prompt_demo()
                elif choice == "6":
                    await self._batch_demo()
                elif choice == "7":
                    await self._performance_demo()
                else:
                    print("❌ Invalid choice. Please try again.")
                    
            except (EOFError, KeyboardInterrupt):
                break
    
    async def _demo_category(self, category: str):
        """カテゴリ別デモ"""
        prompts = self.demo_prompts[category]
        
        print(f"\n📝 {category.title()} Code Generation Demo")
        print("-" * 40)
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n{i}. {prompt}")
            
            if input("Generate code? (y/n): ").lower() != 'y':
                continue
            
            await self._generate_and_display(prompt)
    
    async def _custom_prompt_demo(self):
        """カスタムプロンプトデモ"""
        print("\n✏️  Custom Prompt Demo")
        print("-" * 30)
        
        prompt = input("Enter your code generation prompt: ").strip()
        if not prompt:
            print("❌ Empty prompt")
            return
        
        # 言語選択
        print("\nSelect language:")
        print("1. Python (default)")
        print("2. JavaScript")
        print("3. Java")
        print("4. C++")
        
        lang_choice = input("Enter choice (1-4): ").strip()
        language_map = {"1": "python", "2": "javascript", "3": "java", "4": "cpp"}
        language = language_map.get(lang_choice, "python")
        
        await self._generate_and_display(prompt, language)
    
    async def _batch_demo(self):
        """バッチ生成デモ"""
        print("\n🔄 Batch Generation Demo")
        print("-" * 30)
        
        # 複数プロンプト入力
        prompts = []
        print("Enter prompts (empty line to finish):")
        
        while True:
            prompt = input(f"Prompt {len(prompts) + 1}: ").strip()
            if not prompt:
                break
            prompts.append(prompt)
        
        if not prompts:
            print("❌ No prompts entered")
            return
        
        print(f"\n🚀 Generating code for {len(prompts)} prompts...")
        start_time = time.time()
        
        results = await self.generator.batch_generate(prompts, "python", max_concurrent=2)
        
        batch_time = time.time() - start_time
        print(f"⏱️  Batch generation completed in {batch_time:.2f} seconds")
        
        # 結果表示
        for i, (prompt, result) in enumerate(zip(prompts, results), 1):
            print(f"\n--- Result {i}: {prompt} ---")
            self._display_result(result)
    
    async def _performance_demo(self):
        """パフォーマンステスト"""
        print("\n⚡ Performance Test Demo")
        print("-" * 30)
        
        test_prompts = [
            "Create a simple function",
            "Write a basic class",
            "Implement a sorting algorithm"
        ]
        
        print(f"Testing with {len(test_prompts)} prompts...")
        
        # 個別生成時間測定
        individual_times = []
        for prompt in test_prompts:
            start_time = time.time()
            result = await self.generator.generate_code(prompt, "python")
            end_time = time.time()
            
            individual_times.append(end_time - start_time)
            print(f"✅ '{prompt}': {end_time - start_time:.2f}s (confidence: {result.confidence_score:.2f})")
        
        # バッチ生成時間測定
        start_time = time.time()
        batch_results = await self.generator.batch_generate(test_prompts, "python")
        batch_time = time.time() - start_time
        
        # 統計表示
        print(f"\n📊 Performance Statistics:")
        print(f"Individual total time: {sum(individual_times):.2f}s")
        print(f"Batch total time: {batch_time:.2f}s")
        print(f"Average individual time: {sum(individual_times) / len(individual_times):.2f}s")
        print(f"Average batch time per prompt: {batch_time / len(test_prompts):.2f}s")
        print(f"Speedup: {sum(individual_times) / batch_time:.2f}x")
        
        # メモリ使用量
        memory_usage = self.generator.get_memory_usage()
        if memory_usage:
            print(f"GPU Memory: {memory_usage.get('gpu_allocated_mb', 0):.1f} MB")
    
    async def _generate_and_display(self, prompt: str, language: str = "python"):
        """コード生成と結果表示"""
        print(f"\n🤖 Generating {language} code...")
        print(f"Prompt: {prompt}")
        
        start_time = time.time()
        result = await self.generator.generate_code(prompt, language)
        generation_time = time.time() - start_time
        
        print(f"⏱️  Generation time: {generation_time:.2f} seconds")
        
        self._display_result(result)
    
    def _display_result(self, result: CodeGenerationResult):
        """結果表示"""
        print("\n" + "=" * 50)
        print("📄 Generated Code:")
        print("-" * 20)
        print(result.generated_code)
        
        print(f"\n📊 Analysis:")
        print(f"  Language: {result.language}")
        print(f"  Confidence: {result.confidence_score:.2f}")
        print(f"  Syntax Valid: {'✅' if result.syntax_valid else '❌'}")
        
        if result.quality_metrics:
            print(f"  Quality Metrics:")
            for metric, score in result.quality_metrics.items():
                print(f"    {metric}: {score:.2f}")
        
        if result.execution_result:
            print(f"\n🏃 Execution Result:")
            print(result.execution_result)
        
        if result.error_message:
            print(f"\n❌ Error: {result.error_message}")
        
        if result.suggestions:
            print(f"\n💡 Suggestions:")
            for suggestion in result.suggestions:
                print(f"  • {suggestion}")
        
        print("=" * 50)


async def main():
    """メイン関数"""
    demo = CodeGeneratorDemo()
    await demo.run_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Main error: {e}", exc_info=True)