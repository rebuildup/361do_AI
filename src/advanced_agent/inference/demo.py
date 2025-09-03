"""
LangChain + Ollama 統合デモンストレーション
推論エンジンとツールシステムのテスト
"""

import asyncio
import sys
from pathlib import Path
import time

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.advanced_agent.inference.ollama_client import (
    OllamaClient, InferenceRequest, create_ollama_client
)
from src.advanced_agent.inference.tools import ToolManager, create_tool_manager
from src.advanced_agent.core.config import get_config
from src.advanced_agent.core.logger import setup_logging, get_logger


async def demo_ollama_connection():
    """Ollama 接続デモ"""
    print("\n" + "="*70)
    print("OLLAMA CONNECTION DEMO")
    print("="*70)
    
    logger = get_logger()
    
    print("\n1. Ollama Server Connection Test")
    print("-" * 40)
    
    try:
        # クライアント作成（初期化なし）
        client = OllamaClient()
        
        # サーバー接続テスト
        connected = await client._check_server_connection()
        print(f"Server Connection: {'✅ SUCCESS' if connected else '❌ FAILED'}")
        
        if not connected:
            print("⚠️  Ollama server is not running or not accessible")
            print("   Please start Ollama server with: ollama serve")
            return False
        
        # 完全初期化
        initialized = await client.initialize()
        print(f"Client Initialization: {'✅ SUCCESS' if initialized else '❌ FAILED'}")
        
        if not initialized:
            return False
        
        # 利用可能モデル表示
        models = await client.list_models()
        print(f"\n📋 Available Models ({len(models)}):")
        
        if models:
            for model in models:
                print(f"   • {model.name} ({model.size_gb:.1f}GB) - {model.status.value}")
        else:
            print("   No models found. Please download models with: ollama pull <model_name>")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False


async def demo_basic_inference():
    """基本推論デモ"""
    print("\n" + "="*70)
    print("BASIC INFERENCE DEMO")
    print("="*70)
    
    logger = get_logger()
    
    try:
        # クライアント作成
        client = await create_ollama_client()
        
        print("\n1. Simple Text Generation")
        print("-" * 35)
        
        # 簡単な質問
        request = InferenceRequest(
            prompt="What is the capital of Japan?",
            system_message="You are a helpful and knowledgeable assistant."
        )
        
        print(f"Question: {request.prompt}")
        print("Generating response...")
        
        start_time = time.time()
        response = await client.generate(request)
        
        print(f"\n📝 Response:")
        print(f"   Model: {response.model_used}")
        print(f"   Time: {response.processing_time:.2f}s")
        print(f"   Content: {response.content}")
        
        print("\n2. Chat Format Conversation")
        print("-" * 35)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that provides concise answers."},
            {"role": "user", "content": "Hello! Can you help me understand what AI is?"},
            {"role": "assistant", "content": "Hello! I'd be happy to help. AI (Artificial Intelligence) refers to computer systems that can perform tasks typically requiring human intelligence."},
            {"role": "user", "content": "What are some practical applications of AI?"}
        ]
        
        print("Chat Messages:")
        for msg in messages[-2:]:  # 最後の2メッセージを表示
            role_icon = "🤖" if msg["role"] == "assistant" else "👤"
            print(f"   {role_icon} {msg['role']}: {msg['content']}")
        
        chat_response = await client.chat(messages)
        
        print(f"\n🤖 AI Response:")
        print(f"   Model: {chat_response.model_used}")
        print(f"   Time: {chat_response.processing_time:.2f}s")
        print(f"   Content: {chat_response.content}")
        
        print("\n3. Context-Aware Generation")
        print("-" * 35)
        
        context_request = InferenceRequest(
            prompt="How can this technology be used in healthcare?",
            system_message="You are an expert in AI applications.",
            context=[
                "We are discussing artificial intelligence and machine learning",
                "Previous topics covered: AI definition, practical applications",
                "Focus on healthcare and medical applications"
            ]
        )
        
        print(f"Question: {context_request.prompt}")
        print("Context provided: AI discussion, healthcare focus")
        
        context_response = await client.generate(context_request)
        
        print(f"\n🏥 Healthcare AI Response:")
        print(f"   Model: {context_response.model_used}")
        print(f"   Time: {context_response.processing_time:.2f}s")
        print(f"   Content: {context_response.content[:300]}...")
        
        await client.shutdown()
        return True
        
    except Exception as e:
        print(f"❌ Basic inference demo failed: {e}")
        return False


async def demo_fallback_system():
    """フォールバックシステムデモ"""
    print("\n" + "="*70)
    print("FALLBACK SYSTEM DEMO")
    print("="*70)
    
    logger = get_logger()
    
    try:
        client = await create_ollama_client()
        
        print("\n1. Normal Operation (Primary Model)")
        print("-" * 45)
        
        # 通常の推論
        normal_request = InferenceRequest(
            prompt="Explain quantum computing in simple terms.",
            model_name=client.primary_model
        )
        
        print(f"Using primary model: {client.primary_model}")
        
        try:
            normal_response = await client.generate(normal_request)
            print(f"✅ Primary model response received")
            print(f"   Time: {normal_response.processing_time:.2f}s")
            print(f"   Fallback used: {normal_response.metadata.get('fallback_used', False)}")
            print(f"   Content preview: {normal_response.content[:150]}...")
        except Exception as e:
            print(f"❌ Primary model failed: {e}")
        
        print("\n2. Fallback Model Test")
        print("-" * 30)
        
        # フォールバックモデルを直接指定
        fallback_request = InferenceRequest(
            prompt="What is machine learning?",
            model_name=client.fallback_model
        )
        
        print(f"Using fallback model: {client.fallback_model}")
        
        try:
            fallback_response = await client.generate(fallback_request)
            print(f"✅ Fallback model response received")
            print(f"   Time: {fallback_response.processing_time:.2f}s")
            print(f"   Content preview: {fallback_response.content[:150]}...")
        except Exception as e:
            print(f"❌ Fallback model failed: {e}")
        
        print("\n3. Model Availability Check")
        print("-" * 35)
        
        config = get_config()
        models_to_check = [
            config.models.primary,
            config.models.fallback,
            config.models.emergency
        ]
        
        for model_name in models_to_check:
            model_info = await client.get_model_info(model_name)
            if model_info:
                print(f"   ✅ {model_name} - Available ({model_info.size_gb:.1f}GB)")
            else:
                print(f"   ❌ {model_name} - Not available")
        
        await client.shutdown()
        return True
        
    except Exception as e:
        print(f"❌ Fallback system demo failed: {e}")
        return False


async def demo_tools_integration():
    """ツール統合デモ"""
    print("\n" + "="*70)
    print("TOOLS INTEGRATION DEMO")
    print("="*70)
    
    logger = get_logger()
    
    try:
        # クライアントとツールマネージャー作成
        client = await create_ollama_client()
        tool_manager = await create_tool_manager(client)
        
        print("\n1. Available Tools")
        print("-" * 25)
        
        tools = tool_manager.list_tools()
        for i, tool in enumerate(tools, 1):
            print(f"   {i}. {tool['name']}: {tool['description']}")
        
        print("\n2. Structured Reasoning Tool")
        print("-" * 35)
        
        reasoning_question = "Why is renewable energy important for the future?"
        print(f"Question: {reasoning_question}")
        
        reasoning_result = await tool_manager.execute_tool(
            "structured_reasoning",
            question=reasoning_question,
            context="Environmental and economic considerations"
        )
        
        print(f"📊 Structured Reasoning Result:")
        print(f"   {reasoning_result[:300]}...")
        
        print("\n3. Code Analysis Tool")
        print("-" * 25)
        
        sample_code = '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Calculate fibonacci numbers
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")
'''
        
        print("Analyzing Python code...")
        code_analysis = await tool_manager.execute_tool(
            "code_analysis",
            code=sample_code,
            language="python"
        )
        
        print(f"🔍 Code Analysis Result:")
        print(f"   {code_analysis[:300]}...")
        
        print("\n4. Task Breakdown Tool")
        print("-" * 30)
        
        complex_task = "Create a web application for task management"
        print(f"Task: {complex_task}")
        
        task_breakdown = await tool_manager.execute_tool(
            "task_breakdown",
            task=complex_task,
            constraints="Use Python Flask, SQLite database, responsive design"
        )
        
        print(f"📋 Task Breakdown Result:")
        print(f"   {task_breakdown[:300]}...")
        
        print("\n5. System Monitor Tool")
        print("-" * 30)
        
        system_status = await tool_manager.execute_tool("system_monitor")
        
        print(f"💻 System Status:")
        print(f"   {system_status}")
        
        print("\n6. Auto Tool Selection")
        print("-" * 30)
        
        test_inputs = [
            "分析してください: def hello(): print('world')",
            "タスクを分解してください: AIチャットボットを作成する",
            "なぜPythonが人気なのか推論してください",
            "システムの状態を確認してください"
        ]
        
        for test_input in test_inputs:
            print(f"\nInput: {test_input}")
            
            result = await tool_manager.process_with_tools(test_input)
            
            if result["selected_tool"]:
                print(f"   🔧 Selected Tool: {result['selected_tool']}")
                print(f"   ⏱️  Processing Time: {result['processing_time']:.2f}s")
                
                if result["tool_result"]:
                    preview = result["tool_result"][:100].replace('\n', ' ')
                    print(f"   📄 Result Preview: {preview}...")
            else:
                print(f"   💬 Direct Response: {result['direct_response'][:100]}...")
        
        await client.shutdown()
        return True
        
    except Exception as e:
        print(f"❌ Tools integration demo failed: {e}")
        return False


async def demo_health_monitoring():
    """ヘルス監視デモ"""
    print("\n" + "="*70)
    print("HEALTH MONITORING DEMO")
    print("="*70)
    
    logger = get_logger()
    
    try:
        client = await create_ollama_client()
        
        print("\n1. Comprehensive Health Check")
        print("-" * 40)
        
        health_status = await client.health_check()
        
        print(f"🏥 Health Check Results:")
        print(f"   Server Connected: {'✅' if health_status['server_connected'] else '❌'}")
        print(f"   Models Available: {health_status['models_available']}")
        print(f"   Primary Model: {'✅' if health_status['primary_model_available'] else '❌'}")
        print(f"   Fallback Model: {'✅' if health_status['fallback_model_available'] else '❌'}")
        
        if "inference_test" in health_status:
            test_status = health_status["inference_test"]
            print(f"   Inference Test: {'✅' if test_status == 'passed' else '❌'} ({test_status})")
            
            if "test_response_time" in health_status:
                print(f"   Response Time: {health_status['test_response_time']:.2f}s")
        
        if "error" in health_status:
            print(f"   ❌ Error: {health_status['error']}")
        
        print(f"   Last Check: {health_status['last_check']}")
        
        print("\n2. Performance Metrics")
        print("-" * 30)
        
        # 複数回の推論でパフォーマンス測定
        test_prompts = [
            "Hello",
            "What is 2+2?",
            "Explain AI briefly",
            "Name three colors"
        ]
        
        response_times = []
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"   Test {i}/4: {prompt}")
            
            request = InferenceRequest(prompt=prompt)
            response = await client.generate(request)
            
            response_times.append(response.processing_time)
            print(f"      Time: {response.processing_time:.2f}s")
        
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        
        print(f"\n📊 Performance Summary:")
        print(f"   Average Response Time: {avg_time:.2f}s")
        print(f"   Fastest Response: {min_time:.2f}s")
        print(f"   Slowest Response: {max_time:.2f}s")
        print(f"   Total Tests: {len(test_prompts)}")
        
        await client.shutdown()
        return True
        
    except Exception as e:
        print(f"❌ Health monitoring demo failed: {e}")
        return False


async def main():
    """メインデモ実行"""
    print("LANGCHAIN + OLLAMA INTEGRATION DEMO")
    print("=" * 70)
    print("RTX 4050 6GB VRAM 最適化推論システム")
    print("DeepSeek-R1 + フォールバック機能 + LangChain Tools")
    
    # ログシステム初期化
    logger = setup_logging("INFO")
    
    results = []
    
    try:
        # 1. Ollama 接続デモ
        connection_result = await demo_ollama_connection()
        results.append(("Ollama Connection", connection_result))
        
        if not connection_result:
            print("\n⚠️  Ollama server connection failed. Skipping other demos.")
            print("   Please ensure Ollama is running: ollama serve")
            print("   And download required models: ollama pull deepseek-r1:7b")
            return
        
        # 2. 基本推論デモ
        inference_result = await demo_basic_inference()
        results.append(("Basic Inference", inference_result))
        
        # 3. フォールバックシステムデモ
        fallback_result = await demo_fallback_system()
        results.append(("Fallback System", fallback_result))
        
        # 4. ツール統合デモ
        tools_result = await demo_tools_integration()
        results.append(("Tools Integration", tools_result))
        
        # 5. ヘルス監視デモ
        health_result = await demo_health_monitoring()
        results.append(("Health Monitoring", health_result))
        
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
        print(f"   • Ollama server connection working")
        print(f"   • Basic inference functionality working")
        print(f"   • Fallback system working")
        print(f"   • LangChain tools integration working")
        print(f"   • Health monitoring working")
    else:
        print(f"\n⚠️  {total_count - success_count} demo(s) failed")
        print(f"   Please check the error messages above")
    
    print(f"\n💡 Next Steps:")
    print(f"   • Ensure Ollama server is running for full functionality")
    print(f"   • Download required models: deepseek-r1:7b, qwen2.5:7b-instruct-q4_k_m")
    print(f"   • Check system resources (6GB VRAM recommended)")
    print(f"   • Review logs for detailed performance metrics")
    
    # 最終ログ
    logger.log_shutdown(
        component="langchain_ollama_demo",
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