"""
基本推論エンジンデモンストレーション
LangChain + Ollama 基本推論機能のテスト
"""

import asyncio
import sys
from pathlib import Path
import time
import json

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.advanced_agent.reasoning.basic_engine import (
    BasicReasoningEngine, create_basic_reasoning_engine, ReasoningState
)
from src.advanced_agent.inference.ollama_client import create_ollama_client
from src.advanced_agent.core.config import get_config
from src.advanced_agent.core.logger import setup_logging, get_logger


async def demo_basic_reasoning():
    """基本推論機能デモ"""
    print("\n" + "="*70)
    print("BASIC REASONING ENGINE DEMO")
    print("="*70)
    
    logger = get_logger()
    
    try:
        # 依存関係作成
        print("\n1. Initializing Dependencies")
        print("-" * 35)
        
        print("   Creating Ollama client...")
        ollama_client = await create_ollama_client()
        
        print("   Creating basic reasoning engine...")
        engine = await create_basic_reasoning_engine(ollama_client)
        
        print("   ✅ All dependencies initialized")
        
        print("\n2. Template Management Test")
        print("-" * 35)
        
        # 利用可能テンプレート表示
        templates = engine.list_templates()
        print(f"   Available Prompt Templates: {len(templates['prompt_templates'])}")
        for template_name in templates['prompt_templates']:
            print(f"      - {template_name}")
        
        print(f"   Available Chat Templates: {len(templates['chat_templates'])}")
        for template_name in templates['chat_templates']:
            print(f"      - {template_name}")
        
        print("\n3. Basic Q&A Test")
        print("-" * 25)
        
        question = "Pythonでリストを逆順にする方法を3つ教えてください"
        print(f"Question: {question}")
        
        response = await engine.reason(
            prompt=question,
            template_name="basic_qa"
        )
        
        print(f"\n📊 Results:")
        print(f"   Response: {response.response_text[:200]}...")
        print(f"   Processing Time: {response.processing_time:.2f}s")
        print(f"   Token Count: {response.token_count}")
        print(f"   Model Used: {response.model_used}")
        print(f"   Template Used: {response.template_used}")
        
        print("\n4. Code Analysis Test")
        print("-" * 30)
        
        code_sample = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
"""
        
        print("Analyzing bubble sort implementation...")
        
        code_response = await engine.reason(
            prompt=code_sample,
            template_name="code_analysis",
            template_variables={
                "code": code_sample,
                "language": "Python"
            }
        )
        
        print(f"\n📊 Code Analysis Results:")
        print(f"   Analysis: {code_response.response_text[:300]}...")
        print(f"   Processing Time: {code_response.processing_time:.2f}s")
        
        print("\n5. Problem Solving Test")
        print("-" * 30)
        
        problem = "データベースクエリが遅い"
        context = "ユーザー数が1万人を超え、複雑なJOINクエリが多用されている"
        
        print(f"Problem: {problem}")
        print(f"Context: {context}")
        
        solution_response = await engine.reason(
            prompt=problem,
            template_name="problem_solving",
            template_variables={
                "problem": problem,
                "context": context
            }
        )
        
        print(f"\n📊 Problem Solving Results:")
        print(f"   Solution: {solution_response.response_text[:300]}...")
        print(f"   Processing Time: {solution_response.processing_time:.2f}s")
        
        print("\n6. Summarization Test")
        print("-" * 25)
        
        long_text = """
機械学習は人工知能の一分野であり、コンピュータがデータから学習し、
明示的にプログラムされることなく予測や決定を行う能力を指します。
機械学習には教師あり学習、教師なし学習、強化学習の3つの主要なタイプがあります。
教師あり学習では、入力と正解のペアからなる訓練データを使用してモデルを学習させます。
教師なし学習では、正解のないデータからパターンや構造を発見します。
強化学習では、エージェントが環境との相互作用を通じて最適な行動を学習します。
近年、深層学習の発展により、画像認識、自然言語処理、音声認識などの分野で
大きな進歩が見られています。
"""
        
        print("Summarizing machine learning text...")
        
        summary_response = await engine.reason(
            prompt=long_text,
            template_name="summarization",
            template_variables={
                "text": long_text,
                "summary_length": "3行程度"
            }
        )
        
        print(f"\n📊 Summarization Results:")
        print(f"   Summary: {summary_response.response_text}")
        print(f"   Processing Time: {summary_response.processing_time:.2f}s")
        
        print("\n7. Batch Processing Test")
        print("-" * 30)
        
        batch_requests = [
            {
                "prompt": "Pythonの基本的なデータ型を教えてください",
                "template_name": "basic_qa"
            },
            {
                "prompt": "JavaScriptとPythonの違いは何ですか？",
                "template_name": "basic_qa"
            },
            {
                "prompt": "機械学習の基本概念",
                "template_name": "analysis",
                "template_variables": {
                    "content": "機械学習の基本概念",
                    "analysis_type": "概要説明"
                }
            }
        ]
        
        print(f"Processing {len(batch_requests)} requests in batch...")
        
        batch_start_time = time.time()
        batch_responses = await engine.batch_reason(batch_requests)
        batch_total_time = time.time() - batch_start_time
        
        print(f"\n📊 Batch Processing Results:")
        print(f"   Total Batch Time: {batch_total_time:.2f}s")
        print(f"   Average Time per Request: {batch_total_time / len(batch_requests):.2f}s")
        
        for i, resp in enumerate(batch_responses, 1):
            print(f"   Request {i}:")
            print(f"      State: {resp.state.value}")
            print(f"      Response: {resp.response_text[:100]}...")
            print(f"      Processing Time: {resp.processing_time:.2f}s")
        
        print("\n8. Performance Statistics")
        print("-" * 30)
        
        stats = engine.get_performance_stats()
        print(f"   Total Requests: {stats['total_requests']}")
        print(f"   Successful Requests: {stats['successful_requests']}")
        print(f"   Failed Requests: {stats['failed_requests']}")
        print(f"   Success Rate: {stats['success_rate']:.2%}")
        print(f"   Average Processing Time: {stats['average_processing_time']:.2f}s")
        print(f"   Total Tokens Processed: {stats['total_tokens_processed']}")
        
        if 'recent_average_time' in stats:
            print(f"   Recent Average Time: {stats['recent_average_time']:.2f}s")
            print(f"   Recent Min Time: {stats['recent_min_time']:.2f}s")
            print(f"   Recent Max Time: {stats['recent_max_time']:.2f}s")
        
        print("\n9. Reasoning History Analysis")
        print("-" * 35)
        
        # 全履歴
        all_history = engine.get_reasoning_history()
        print(f"   Total History Entries: {len(all_history)}")
        
        # 成功した推論のみ
        successful_history = engine.get_reasoning_history(
            state_filter=ReasoningState.COMPLETED
        )
        print(f"   Successful Reasoning: {len(successful_history)}")
        
        # 最新5件
        recent_history = engine.get_reasoning_history(limit=5)
        print(f"   Recent 5 Entries:")
        for i, entry in enumerate(recent_history, 1):
            print(f"      {i}. {entry.request_id}: {entry.state.value} ({entry.processing_time:.2f}s)")
        
        await engine.shutdown()
        await ollama_client.shutdown()
        
        return True
        
    except Exception as e:
        print(f"❌ Basic reasoning demo failed: {e}")
        return False


async def demo_custom_templates():
    """カスタムテンプレートデモ"""
    print("\n" + "="*70)
    print("CUSTOM TEMPLATES DEMO")
    print("="*70)
    
    try:
        # 依存関係作成
        ollama_client = await create_ollama_client()
        engine = await create_basic_reasoning_engine(ollama_client)
        
        print("\n1. Creating Custom Templates")
        print("-" * 35)
        
        # カスタムプロンプトテンプレート
        from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
        
        # 技術記事作成テンプレート
        tech_article_template = PromptTemplate(
            input_variables=["topic", "target_audience", "length"],
            template="""
技術記事を作成してください。

トピック: {topic}
対象読者: {target_audience}
記事の長さ: {length}

以下の構成で記事を作成してください：
1. 導入
2. 基本概念の説明
3. 実践的な例
4. まとめ

記事:"""
        )
        
        # レビューテンプレート
        review_template = PromptTemplate(
            input_variables=["content", "review_type"],
            template="""
以下の内容について{review_type}レビューを行ってください。

レビュー対象:
{content}

レビュー観点:
- 正確性
- 完全性
- 明確性
- 改善提案

レビュー結果:"""
        )
        
        # テンプレート登録
        engine.register_template("tech_article", tech_article_template)
        engine.register_template("review", review_template)
        
        print("   ✅ Custom templates registered")
        
        print("\n2. Using Tech Article Template")
        print("-" * 35)
        
        article_response = await engine.reason(
            prompt="Python async/await",
            template_name="tech_article",
            template_variables={
                "topic": "Python async/await",
                "target_audience": "中級プログラマー",
                "length": "1000文字程度"
            }
        )
        
        print(f"   Article: {article_response.response_text[:300]}...")
        print(f"   Processing Time: {article_response.processing_time:.2f}s")
        
        print("\n3. Using Review Template")
        print("-" * 30)
        
        code_to_review = """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)
"""
        
        review_response = await engine.reason(
            prompt=code_to_review,
            template_name="review",
            template_variables={
                "content": code_to_review,
                "review_type": "コード品質"
            }
        )
        
        print(f"   Review: {review_response.response_text[:300]}...")
        print(f"   Processing Time: {review_response.processing_time:.2f}s")
        
        print("\n4. Template List After Custom Addition")
        print("-" * 45)
        
        templates = engine.list_templates()
        print(f"   Total Prompt Templates: {len(templates['prompt_templates'])}")
        for template_name in templates['prompt_templates']:
            print(f"      - {template_name}\")")
        
        await engine.shutdown()
        await ollama_client.shutdown()
        
        return True
        
    except Exception as e:
        print(f"❌ Custom templates demo failed: {e}")
        return False


async def demo_error_handling():
    """エラーハンドリングデモ"""
    print("\n" + "="*70)
    print("ERROR HANDLING DEMO")
    print("="*70)
    
    try:
        # 依存関係作成
        ollama_client = await create_ollama_client()
        engine = await create_basic_reasoning_engine(ollama_client)
        
        print("\n1. Invalid Template Test")
        print("-" * 30)
        
        # 存在しないテンプレート使用
        invalid_response = await engine.reason(
            prompt="Test prompt",
            template_name="nonexistent_template"
        )
        
        print(f"   State: {invalid_response.state.value}")
        print(f"   Error: {invalid_response.error_message}")
        
        print("\n2. Template Variable Missing Test")
        print("-" * 40)
        
        # 必要な変数が不足
        missing_var_response = await engine.reason(
            prompt="Test analysis",
            template_name="analysis",
            template_variables={"content": "Test content"}  # analysis_type が不足
        )
        
        print(f"   State: {missing_var_response.state.value}")
        if missing_var_response.state == ReasoningState.ERROR:
            print(f"   Error: {missing_var_response.error_message}")
        else:
            print(f"   Response: {missing_var_response.response_text[:100]}...")
        
        print("\n3. Batch Error Handling Test")
        print("-" * 35)
        
        # 一部エラーを含むバッチ
        mixed_batch = [
            {"prompt": "Normal question", "template_name": "basic_qa"},
            {"prompt": "Error question", "template_name": "nonexistent"},
            {"prompt": "Another normal question", "template_name": "basic_qa"}
        ]
        
        batch_responses = await engine.batch_reason(mixed_batch)
        
        print(f"   Batch Results:")
        for i, resp in enumerate(batch_responses, 1):
            print(f"      Request {i}: {resp.state.value}")
            if resp.state == ReasoningState.ERROR:
                print(f"         Error: {resp.error_message}")
        
        print("\n4. Performance Stats with Errors")
        print("-" * 40)
        
        stats = engine.get_performance_stats()
        print(f"   Total Requests: {stats['total_requests']}")
        print(f"   Successful: {stats['successful_requests']}")
        print(f"   Failed: {stats['failed_requests']}")
        print(f"   Success Rate: {stats['success_rate']:.2%}")
        
        await engine.shutdown()
        await ollama_client.shutdown()
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling demo failed: {e}")
        return False


async def main():
    """メインデモ実行"""
    print("🚀 Starting Basic Reasoning Engine Demonstrations")
    
    # ログ設定
    setup_logging()
    
    demos = [
        ("Basic Reasoning", demo_basic_reasoning),
        ("Custom Templates", demo_custom_templates),
        ("Error Handling", demo_error_handling)
    ]
    
    results = {}
    
    for demo_name, demo_func in demos:
        print(f"\n🎯 Running {demo_name} Demo...")
        try:
            result = await demo_func()
            results[demo_name] = "✅ Success" if result else "❌ Failed"
        except Exception as e:
            results[demo_name] = f"❌ Error: {e}"
    
    print("\n" + "="*70)
    print("DEMO RESULTS SUMMARY")
    print("="*70)
    
    for demo_name, result in results.items():
        print(f"   {demo_name}: {result}")
    
    success_count = sum(1 for result in results.values() if "✅" in result)
    total_count = len(results)
    
    print(f"\n📊 Overall Success Rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    if success_count == total_count:
        print("🎉 All demos completed successfully!")
    else:
        print("⚠️  Some demos failed. Check the logs for details.")


if __name__ == "__main__":
    asyncio.run(main())