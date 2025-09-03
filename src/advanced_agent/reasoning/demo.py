"""
LangChain + Ollama 基本推論エンジン統合デモ
PromptTemplate とコールバック統合テスト
"""

import asyncio
import sys
from pathlib import Path
import time

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.advanced_agent.reasoning.base_engine import (
    BasicReasoningEngine, MemoryAwareReasoningEngine, ReasoningRequest, 
    ReasoningContext, create_reasoning_engine, quick_reasoning
)
from src.advanced_agent.reasoning.prompt_manager import (
    PromptManager, PromptConfig, PromptType, PromptExample, get_prompt_manager
)
from src.advanced_agent.reasoning.callbacks import (
    PerformanceCallbackHandler, create_performance_callback, get_performance_statistics
)
from src.advanced_agent.inference.ollama_client import create_ollama_client
from src.advanced_agent.core.config import get_config
from src.advanced_agent.core.logger import setup_logging, get_logger


async def demo_basic_reasoning_engine():
    """基本推論エンジンデモ"""
    print("\n" + "="*70)
    print("BASIC REASONING ENGINE DEMO")
    print("="*70)
    
    logger = get_logger()
    
    try:
        # Ollama クライアント作成
        ollama_client = await create_ollama_client()
        
        # 推論エンジン作成
        engine = await create_reasoning_engine(ollama_client, "basic")
        
        print("\n1. General Reasoning Test")
        print("-" * 35)
        
        # 一般的な推論テスト
        context = ReasoningContext(
            session_id="demo_session_1",
            system_context="AI技術の発展について議論中",
            domain_context="人工知能と機械学習",
            constraints=["客観的な視点を保つ", "具体例を含める"]
        )
        
        request = ReasoningRequest(
            prompt="人工知能が社会に与える影響について分析してください",
            context=context,
            reasoning_type="general",
            temperature=0.1
        )
        
        print(f"Question: {request.prompt}")
        print(f"Reasoning Type: {request.reasoning_type}")
        print("Processing...")
        
        result = await engine.reason(request)
        
        print(f"\n📊 Results:")
        print(f"   Processing Time: {result.processing_time:.2f}s")
        print(f"   Confidence Score: {result.confidence_score:.2f}")
        print(f"   Model Used: {result.model_used}")
        print(f"   Tokens Used: {result.tokens_used}")
        print(f"   Memory Usage: {result.memory_usage_mb:.2f}MB")
        print(f"\n📝 Answer:")
        print(f"   {result.final_answer[:300]}...")
        
        print("\n2. Analytical Reasoning Test")
        print("-" * 35)
        
        # 分析的推論テスト
        analytical_context = ReasoningContext(
            session_id="demo_session_2",
            domain_context="プログラミング教育",
            constraints=["初心者向けに説明", "実用的な観点を重視"]
        )
        
        analytical_request = ReasoningRequest(
            prompt="Pythonプログラミング言語の学習方法を体系的に分析してください",
            context=analytical_context,
            reasoning_type="analytical"
        )
        
        print(f"Question: {analytical_request.prompt}")
        print("Processing...")
        
        analytical_result = await engine.reason(analytical_request)
        
        print(f"\n📊 Results:")
        print(f"   Processing Time: {analytical_result.processing_time:.2f}s")
        print(f"   Confidence Score: {analytical_result.confidence_score:.2f}")
        print(f"\n📝 Analysis:")
        print(f"   {analytical_result.final_answer[:300]}...")
        
        print("\n3. Creative Reasoning Test")
        print("-" * 30)
        
        # 創造的推論テスト
        creative_context = ReasoningContext(
            session_id="demo_session_3",
            domain_context="環境問題解決",
            constraints=["実現可能性を考慮", "革新的なアプローチ"]
        )
        
        creative_request = ReasoningRequest(
            prompt="気候変動問題を解決する革新的なテクノロジーソリューションを提案してください",
            context=creative_context,
            reasoning_type="creative"
        )
        
        print(f"Question: {creative_request.prompt}")
        print("Processing...")
        
        creative_result = await engine.reason(creative_request)
        
        print(f"\n📊 Results:")
        print(f"   Processing Time: {creative_result.processing_time:.2f}s")
        print(f"   Confidence Score: {creative_result.confidence_score:.2f}")
        print(f"\n💡 Creative Ideas:")
        print(f"   {creative_result.final_answer[:300]}...")
        
        await ollama_client.shutdown()
        return True
        
    except Exception as e:
        print(f"❌ Basic reasoning engine demo failed: {e}")
        return False


async def demo_prompt_management():
    """プロンプト管理デモ"""
    print("\n" + "="*70)
    print("PROMPT MANAGEMENT DEMO")
    print("="*70)
    
    logger = get_logger()
    
    try:
        # プロンプトマネージャー取得
        manager = get_prompt_manager()
        
        print("\n1. Available Prompts")
        print("-" * 25)
        
        prompts = manager.list_prompts()
        print(f"Total prompts: {len(prompts)}")
        
        for prompt_type in PromptType:
            type_prompts = manager.list_prompts(prompt_type)
            if type_prompts:
                print(f"   {prompt_type.value}: {len(type_prompts)} prompts")
                for prompt_name in type_prompts[:2]:  # 最初の2つだけ表示
                    print(f"     - {prompt_name}")
        
        print("\n2. Prompt Template Usage")
        print("-" * 30)
        
        # プロンプトテンプレート使用例
        reasoning_template = manager.get_prompt_template("general_reasoning")
        if reasoning_template:
            formatted_prompt = reasoning_template.format(
                context="機械学習の基礎について学習中",
                question="ニューラルネットワークの仕組みを説明してください"
            )
            
            print("Template: general_reasoning")
            print("Formatted prompt:")
            print(f"   {formatted_prompt[:200]}...")
        
        print("\n3. Chat Prompt Creation")
        print("-" * 30)
        
        # チャットプロンプト作成
        chat_prompt = manager.create_chat_prompt(
            "analytical_thinking",
            subject="データサイエンス",
            context="ビジネス分析での活用",
            aspects="効率性、正確性、実用性"
        )
        
        print("Chat prompt created successfully")
        print(f"Messages count: {len(chat_prompt.messages)}")
        
        print("\n4. Prompt Optimization")
        print("-" * 25)
        
        # プロンプト最適化テスト
        test_prompts = [
            ("何かいい方法を教えてください", "specificity"),
            ("あいまいな説明をしてください", "clarity"),
            ("いくつかの例を示してください。詳細に説明してください。", "structure")
        ]
        
        for original_prompt, optimization_type in test_prompts:
            result = manager.optimize_prompt(original_prompt, optimization_type)
            
            print(f"\nOptimization: {optimization_type}")
            print(f"   Original: {result.original_prompt}")
            print(f"   Optimized: {result.optimized_prompt}")
            print(f"   Score: {result.improvement_score:.2f}")
            print(f"   Reasoning: {result.reasoning}")
        
        print("\n5. Few-Shot Prompt Creation")
        print("-" * 35)
        
        # Few-shot プロンプト作成
        examples = [
            PromptExample(
                input="日本の首都は？",
                output="日本の首都は東京です。"
            ),
            PromptExample(
                input="フランスの首都は？",
                output="フランスの首都はパリです。"
            )
        ]
        
        few_shot_config = manager.create_few_shot_prompt(
            name="geography_qa",
            base_template="地理に関する質問に答えてください。\n\n質問: {question}\n\n回答:",
            examples=examples,
            input_variables=["question"]
        )
        
        print(f"Few-shot prompt created: {few_shot_config.name}")
        print(f"Examples count: {len(few_shot_config.examples)}")
        
        # 統計情報
        stats = manager.get_prompt_statistics()
        print(f"\n📊 Prompt Statistics:")
        print(f"   Total prompts: {stats['total_prompts']}")
        print(f"   Total examples: {stats['total_examples']}")
        print(f"   Average examples per prompt: {stats['average_examples_per_prompt']:.1f}")
        print(f"   Prompts by type: {stats['prompts_by_type']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prompt management demo failed: {e}")
        return False


async def demo_performance_callbacks():
    """パフォーマンスコールバックデモ"""
    print("\n" + "="*70)
    print("PERFORMANCE CALLBACKS DEMO")
    print("="*70)
    
    logger = get_logger()
    
    try:
        print("\n1. Performance Callback Creation")
        print("-" * 40)
        
        # パフォーマンスコールバック作成
        callback = create_performance_callback("demo_request_001", enable_gpu=True)
        
        print(f"Callback created for request: {callback.request_id}")
        print(f"GPU monitoring enabled: {callback.enable_gpu_monitoring}")
        
        print("\n2. Simulated LLM Processing")
        print("-" * 35)
        
        # 模擬的なLLM処理
        print("Starting LLM simulation...")
        
        # LLM開始
        callback.on_llm_start(
            {"name": "deepseek-r1:7b"},
            ["これは推論エンジンのパフォーマンステストです。複雑な質問に対して詳細な回答を生成してください。"]
        )
        
        # 処理時間シミュレート
        print("Processing tokens...")
        for i in range(100):
            callback.on_llm_new_token(f"token_{i}")
            await asyncio.sleep(0.005)  # 5ms間隔
        
        # エージェントアクション シミュレート
        from langchain_core.agents import AgentAction
        action = AgentAction(
            tool="reasoning_tool",
            tool_input="analyze the problem",
            log="Thinking about the problem..."
        )
        callback.on_agent_action(action)
        
        # LLM終了
        from langchain_core.outputs import LLMResult, Generation
        result = LLMResult(generations=[[Generation(
            text="これは詳細な回答です。推論プロセスを経て生成された結果となります。"
        )]])
        callback.on_llm_end(result)
        
        print("LLM processing completed")
        
        print("\n3. Performance Metrics")
        print("-" * 25)
        
        # メトリクス取得
        metrics = callback.get_metrics()
        
        print(f"📊 Performance Metrics:")
        print(f"   Request ID: {metrics.request_id}")
        print(f"   Model: {metrics.model_name}")
        print(f"   Duration: {metrics.duration:.3f}s")
        print(f"   Input Tokens: {metrics.input_tokens}")
        print(f"   Output Tokens: {metrics.output_tokens}")
        print(f"   Total Tokens: {metrics.total_tokens}")
        print(f"   Memory Delta: {metrics.memory_delta_mb:.2f}MB")
        print(f"   Memory Peak: {metrics.memory_peak_mb:.2f}MB")
        print(f"   Reasoning Steps: {metrics.reasoning_steps}")
        print(f"   Tool Calls: {metrics.tool_calls}")
        print(f"   Errors: {metrics.errors}")
        
        if metrics.gpu_utilization_avg > 0:
            print(f"   GPU Utilization: {metrics.gpu_utilization_avg:.1f}%")
            print(f"   GPU Memory Peak: {metrics.gpu_memory_peak_mb:.2f}MB")
        
        print("\n4. Event Timeline")
        print("-" * 20)
        
        events = callback.get_events()
        print(f"Total events: {len(events)}")
        
        for event in events:
            elapsed = event.timestamp - callback.metrics.start_time
            print(f"   {elapsed:.3f}s: {event.event_type}")
        
        print("\n5. Performance Summary")
        print("-" * 25)
        
        summary = callback.get_summary()
        print(f"Summary: {summary}")
        
        print("\n6. Aggregated Statistics")
        print("-" * 30)
        
        # 集約統計に追加
        from src.advanced_agent.reasoning.callbacks import get_aggregated_handler
        aggregated = get_aggregated_handler()
        aggregated.add_performance_record(metrics)
        
        stats = get_performance_statistics()
        print(f"📈 Aggregated Statistics:")
        print(f"   Total Requests: {stats['total_requests']}")
        print(f"   Average Duration: {stats['average_duration']:.3f}s")
        print(f"   Average Tokens: {stats['average_tokens']:.0f}")
        print(f"   Error Rate: {stats['error_rate']:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance callbacks demo failed: {e}")
        return False


async def demo_integrated_reasoning():
    """統合推論デモ"""
    print("\n" + "="*70)
    print("INTEGRATED REASONING DEMO")
    print("="*70)
    
    logger = get_logger()
    
    try:
        print("\n1. Quick Reasoning Test")
        print("-" * 30)
        
        # 簡易推論テスト
        quick_questions = [
            ("量子コンピューターの基本原理を説明してください", "factual"),
            ("新しいプログラミング言語を学ぶ効率的な方法を分析してください", "analytical"),
            ("持続可能な都市設計のアイデアを提案してください", "creative")
        ]
        
        for question, reasoning_type in quick_questions:
            print(f"\nQuestion ({reasoning_type}): {question}")
            print("Processing...")
            
            start_time = time.time()
            answer = await quick_reasoning(question, reasoning_type)
            processing_time = time.time() - start_time
            
            print(f"Time: {processing_time:.2f}s")
            print(f"Answer: {answer[:150]}...")
        
        print("\n2. Complex Reasoning with Context")
        print("-" * 40)
        
        # 複雑な推論テスト
        ollama_client = await create_ollama_client()
        engine = await create_reasoning_engine(ollama_client, "basic")
        
        # 会話履歴を含むコンテキスト
        complex_context = ReasoningContext(
            session_id="complex_demo",
            system_context="技術コンサルティングセッション",
            domain_context="企業のデジタル変革",
            conversation_history=[
                {"role": "user", "content": "我が社はAI導入を検討しています"},
                {"role": "assistant", "content": "AI導入には段階的なアプローチが重要です"},
                {"role": "user", "content": "具体的にはどのような手順で進めるべきでしょうか"}
            ],
            constraints=[
                "実用的で実現可能な提案",
                "コスト効率を考慮",
                "段階的な実装アプローチ"
            ],
            metadata={
                "industry": "manufacturing",
                "company_size": "medium",
                "urgency": "high"
            }
        )
        
        complex_request = ReasoningRequest(
            prompt="製造業の中規模企業におけるAI導入戦略を詳細に分析し、具体的な実装計画を提案してください",
            context=complex_context,
            reasoning_type="analytical",
            use_memory=True
        )
        
        print("Processing complex reasoning request...")
        print(f"Context includes: {len(complex_context.conversation_history)} conversation turns")
        print(f"Constraints: {len(complex_context.constraints)} items")
        
        # パフォーマンスコールバック付きで実行
        callback = create_performance_callback("complex_demo_001")
        
        complex_result = await engine.reason(complex_request)
        
        print(f"\n📊 Complex Reasoning Results:")
        print(f"   Processing Time: {complex_result.processing_time:.2f}s")
        print(f"   Confidence Score: {complex_result.confidence_score:.2f}")
        print(f"   Reasoning Steps: {len(complex_result.reasoning_steps)}")
        print(f"   Context Used: {complex_result.context_used.session_id}")
        
        print(f"\n📝 Strategic Analysis:")
        print(f"   {complex_result.final_answer[:400]}...")
        
        await ollama_client.shutdown()
        return True
        
    except Exception as e:
        print(f"❌ Integrated reasoning demo failed: {e}")
        return False


async def main():
    """メインデモ実行"""
    print("LANGCHAIN + OLLAMA BASIC REASONING ENGINE DEMO")
    print("=" * 70)
    print("RTX 4050 6GB VRAM 最適化推論システム")
    print("PromptTemplate + Callbacks + Performance Monitoring")
    
    # ログシステム初期化
    logger = setup_logging("INFO")
    
    results = []
    
    try:
        # 1. 基本推論エンジンデモ
        reasoning_result = await demo_basic_reasoning_engine()
        results.append(("Basic Reasoning Engine", reasoning_result))
        
        # 2. プロンプト管理デモ
        prompt_result = await demo_prompt_management()
        results.append(("Prompt Management", prompt_result))
        
        # 3. パフォーマンスコールバックデモ
        callback_result = await demo_performance_callbacks()
        results.append(("Performance Callbacks", callback_result))
        
        # 4. 統合推論デモ
        integrated_result = await demo_integrated_reasoning()
        results.append(("Integrated Reasoning", integrated_result))
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        return
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 結果サマリー
    print("\n" + "="*70)
    print("DEMO RESULTS SUMMARY")
    print("="*70)
    
    success_count = sum(1 for _, result in results if result)
    total_count = len(results)
    
    print(f"\n📊 Overall Results: {success_count}/{total_count} demos successful")
    
    for demo_name, result in results:
        status = "✅ SUCCESS" if result else "❌ FAILED"
        print(f"   {status} {demo_name}")
    
    if success_count == total_count:
        print(f"\n🎉 All demos completed successfully!")
        print(f"   • Basic reasoning engine working")
        print(f"   • Prompt management system working")
        print(f"   • Performance callbacks working")
        print(f"   • Integrated reasoning working")
    else:
        print(f"\n⚠️  {total_count - success_count} demo(s) failed")
        print(f"   Please check the error messages above")
    
    print(f"\n💡 Next Steps:")
    print(f"   • Ensure Ollama server is running for full functionality")
    print(f"   • Test with different reasoning types and contexts")
    print(f"   • Monitor performance metrics for optimization")
    print(f"   • Integrate with memory system for enhanced reasoning")
    
    # 最終ログ
    logger.log_shutdown(
        component="basic_reasoning_demo",
        uptime_seconds=0,  # デモなので0
        final_stats={
            "demos_run": total_count,
            "demos_successful": success_count,
            "success_rate": success_count / total_count if total_count > 0 else 0
        }
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback
        traceback.print_exc()