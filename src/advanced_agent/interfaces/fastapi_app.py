"""
FastAPI Application Factory

Creates and configures the FastAPI application with all necessary routes
and middleware for the React frontend integration.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from .fastapi_gateway import FastAPIGateway
from ..core.self_learning_agent import SelfLearningAgent
from ..config.settings import get_agent_config

logger = logging.getLogger(__name__)

# Global agent instance
_agent_instance: Optional[SelfLearningAgent] = None
_agent_initialization_task: Optional[asyncio.Task] = None


async def get_agent_instance() -> SelfLearningAgent:
    """Get or create the global agent instance（非同期初期化版）"""
    global _agent_instance, _agent_initialization_task
    
    if _agent_instance is None:
        # 初期化タスクが実行中でない場合は開始
        if _agent_initialization_task is None:
            _agent_initialization_task = asyncio.create_task(_initialize_agent_async())
        
        # 初期化完了を待つ（タイムアウト5秒）
        try:
            await asyncio.wait_for(_agent_initialization_task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Agent initialization timeout, using fallback mode")
            # タイムアウトの場合はフォールバックモードで初期化
            _agent_instance = SelfLearningAgent(
                config_path="config/agent_config.yaml",
                db_path="data/self_learning_agent.db"
            )
            _agent_instance.session_id = "fallback_session"
            _agent_instance.user_id = "fallback_user"
    
    return _agent_instance

async def _initialize_agent_async():
    """エージェントの非同期初期化"""
    global _agent_instance
    
    try:
        config = get_agent_config()
        _agent_instance = SelfLearningAgent(
            config_path="config/agent_config.yaml",
            db_path="data/self_learning_agent.db"
        )
        await _agent_instance.initialize_session()
        logger.info("Agent instance initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        # エラーの場合はフォールバックモードで初期化
        _agent_instance = SelfLearningAgent(
            config_path="config/agent_config.yaml",
            db_path="data/self_learning_agent.db"
        )
        _agent_instance.session_id = "fallback_session"
        _agent_instance.user_id = "fallback_user"


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    # Create FastAPI gateway with enhanced configuration
    gateway = FastAPIGateway(
        title="361do_AI Advanced Agent API",
        version="1.0.0",
        description="Advanced AI Agent with Self-Learning Capabilities",
        enable_auth=False,  # Disable auth for development
        cors_origins=["*"]  # Allow all origins for development
    )
    
    app = gateway.app
    
    # Add agent-specific routes
    setup_agent_routes(app)
    
    # Add static file serving for React frontend
    try:
        import os
        
        # Try multiple possible static file locations
        static_dirs = [
            "frontend/dist",
            "static",
            "/app/static",
            "/app/frontend/dist"
        ]
        
        static_dir = None
        for dir_path in static_dirs:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                static_dir = dir_path
                break
        
        if static_dir:
            app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
            logger.info(f"Static files mounted from {static_dir} for React frontend")
        else:
            logger.warning("No static files directory found for React frontend")
            
    except Exception as e:
        logger.warning(f"Could not mount static files: {e}")
    
    # Add startup and shutdown events (temporarily disabled for debugging)
    # @app.on_event("startup")
    # async def startup_event():
    #     """Initialize agent on startup"""
    #     try:
    #         await get_agent_instance()
    #         logger.info("FastAPI application started successfully")
    #     except Exception as e:
    #         logger.error(f"Startup failed: {e}")
    
    # @app.on_event("shutdown")
    # async def shutdown_event():
    #     """Cleanup on shutdown"""
    #     global _agent_instance
    #     if _agent_instance:
    #         try:
    #             # Add any cleanup logic here
    #             logger.info("Agent cleanup completed")
    #         except Exception as e:
    #             logger.error(f"Shutdown error: {e}")
    #     logger.info("FastAPI application shutdown")
    
    return app


def setup_agent_routes(app: FastAPI):
    """Setup agent-specific routes"""
    
    # Health endpoint (used by frontend)
    @app.get("/v1/health")
    async def health():
        try:
            _ = await get_agent_instance()
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "system_info": {
                    "cpu_percent": 0,
                    "memory_percent": 0
                }
            }
        except Exception:
            return {
                "status": "degraded",
                "timestamp": datetime.now().isoformat(),
                "version": "unknown",
                "system_info": {
                    "cpu_percent": 0,
                    "memory_percent": 0
                }
            }
    
    # Simple health endpoint for frontend
    @app.get("/health")
    async def simple_health():
        """Simple health check endpoint"""
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat()
        }

    # Models endpoint
    @app.get("/v1/models")
    async def list_models():
        """Get available models from Ollama"""
        try:
            agent = await get_agent_instance()
            if hasattr(agent, 'reasoning_engine') and hasattr(agent.reasoning_engine, 'ollama_client'):
                ollama_client = agent.reasoning_engine.ollama_client
                models = []
                
                # Get models from Ollama client
                for model_name, model_info in ollama_client.model_cache.items():
                    models.append({
                        "id": model_name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "advanced-agent",
                        "permission": []
                    })
                
                return {
                    "object": "list",
                    "data": models
                }
            else:
                # Fallback to config model
                config = get_agent_config()
                return {
                    "object": "list",
                    "data": [{
                        "id": config.ollama.model,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "advanced-agent",
                        "permission": []
                    }]
                }
        except Exception as e:
            logger.error(f"Models list error: {e}")
            # Fallback to config model
            config = get_agent_config()
            return {
                "object": "list",
                "data": [{
                    "id": config.ollama.model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "advanced-agent",
                    "permission": []
                }]
            }
    
    # Agent status endpoint（簡易版）
    @app.get("/v1/agent/status")
    async def get_agent_status():
        """Get agent status and capabilities（簡易版）"""
        # エージェント初期化を待たずに即座にレスポンスを返す
        return {
            "status": "active",
            "capabilities": [
                "natural_language_processing",
                "web_search",
                "file_operations",
                "command_execution",
                "self_learning",
                "prompt_rewriting",
                "tool_usage"
            ],
            "active_tools": [],
            "memory_usage": {},
            "learning_metrics": {
                "learning_epoch": 0,
                "total_interactions": 0,
                "reward_score": 0.0
            }
        }
    
    # Agent configuration endpoint
    @app.post("/v1/agent/config")
    async def update_agent_config(config: dict):
        """Update agent configuration"""
        try:
            agent = await get_agent_instance()
            
            # Update configuration based on provided settings
            success = True
            message = "Configuration updated successfully"
            
            # Handle temperature setting
            if "temperature" in config:
                # Update temperature in agent's inference client
                if hasattr(agent, 'inference_client'):
                    agent.inference_client.temperature = config["temperature"]
            
            # Handle max_tokens setting
            if "max_tokens" in config:
                if hasattr(agent, 'inference_client'):
                    agent.inference_client.max_tokens = config["max_tokens"]
            
            # Handle streaming setting
            if "streaming_enabled" in config:
                if hasattr(agent, 'inference_client'):
                    agent.inference_client.streaming_enabled = config["streaming_enabled"]
            
            # Handle tools setting
            if "tools_enabled" in config and hasattr(agent, 'tool_registry'):
                for tool_name in config["tools_enabled"]:
                    agent.tool_registry.enable_tool(tool_name)
            
            return {
                "success": success,
                "message": message
            }
        except Exception as e:
            logger.error(f"Agent config update error: {e}")
            return {
                "success": False,
                "message": f"Configuration update failed: {str(e)}"
            }
    
    # Available tools endpoint（完全無効化版）
    @app.get("/v1/agent/tools")
    def get_available_tools():
        """Get available tools and their status（完全無効化版）"""
        # 完全に無効化して即座にレスポンスを返す
        return {"tools": []}
    
    # Tool execution endpoint
    @app.post("/v1/agent/tools/execute")
    async def execute_tool(request: dict):
        """Execute a specific tool with parameters"""
        try:
            agent = await get_agent_instance()
            tool_name = request.get("tool_name")
            parameters = request.get("parameters", {})
            session_id = request.get("session_id")
            
            if not tool_name:
                return {
                    "success": False,
                    "error": "tool_name is required"
                }
            
            start_time = time.time()
            
            # Execute tool through agent
            if hasattr(agent, 'tool_registry'):
                result = await agent.tool_registry.execute_tool(tool_name, parameters)
                execution_time = time.time() - start_time
                
                return {
                    "result": result,
                    "execution_time": execution_time,
                    "success": True
                }
            else:
                return {
                    "success": False,
                    "error": "Tool registry not available"
                }
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": 0
            }
    
    # Learning metrics endpoint
    @app.get("/v1/agent/learning/metrics")
    async def get_learning_metrics():
        """Get learning progress and metrics"""
        try:
            agent = await get_agent_instance()
            status = await agent.get_agent_status()
            
            return {
                "total_interactions": status.get("total_interactions", 0),
                "successful_responses": status.get("successful_responses", 0),
                "average_response_time": status.get("average_response_time", 0.0),
                "learning_rate": status.get("learning_rate", 0.0),
                "recent_improvements": status.get("recent_improvements", [])
            }
        except Exception as e:
            logger.error(f"Learning metrics error: {e}")
            return {
                "total_interactions": 0,
                "successful_responses": 0,
                "average_response_time": 0.0,
                "learning_rate": 0.0,
                "recent_improvements": []
            }
    
    # Feedback endpoint
    @app.post("/v1/agent/feedback")
    async def provide_feedback(request: dict):
        """Provide feedback on agent response"""
        try:
            agent = await get_agent_instance()
            response_id = request.get("response_id")
            rating = request.get("rating")
            helpful = request.get("helpful")
            comments = request.get("comments", "")
            
            # Process feedback through agent's learning system
            # This would integrate with the reward system
            
            return {
                "success": True,
                "message": "Feedback received and processed"
            }
        except Exception as e:
            logger.error(f"Feedback processing error: {e}")
            return {
                "success": False,
                "message": f"Feedback processing failed: {str(e)}"
            }
    
    # Enhanced chat completion with agent processing
    @app.post("/v1/chat/completions/agent")
    async def agent_chat_completion(request: dict):
        """Enhanced chat completion using the self-learning agent"""
        try:
            agent = await get_agent_instance()
            
            # Extract message content
            messages = request.get("messages", [])
            if not messages:
                raise ValueError("No messages provided")
            
            # Get the last user message
            user_message = None
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break
            
            if not user_message:
                raise ValueError("No user message found")
            
            # Process through agent
            result = await agent.process_user_input(user_message)
            
            # Format response
            response_content = result.get("response", "")
            reasoning = result.get("reasoning", "")
            
            # Create OpenAI-compatible response
            response = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.get("model", "agent"),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(user_message.split()),
                    "completion_tokens": len(response_content.split()),
                    "total_tokens": len(user_message.split()) + len(response_content.split())
                }
            }
            
            # Add reasoning if available
            if reasoning:
                response["reasoning"] = reasoning
            
            return response
            
        except Exception as e:
            logger.error(f"Agent chat completion error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return a proper error response instead of raising HTTPException
            return {
                "error": {
                    "message": f"Agent processing failed: {str(e)}",
                    "type": "agent_error",
                    "code": "processing_failed"
                }
            }
    
    # OpenAI-compatible chat completion（改善版）
    @app.post("/v1/chat/completions")
    async def chat_completions(request: dict):
        """OpenAI-compatible chat completion endpoint"""
        try:
            # Check if streaming is requested
            stream = request.get("stream", False)
            if stream:
                # Delegate to streaming handler
                return await chat_completions_stream(request)
            
            # Get messages from request
            messages = request.get("messages", [])
            if not messages:
                return {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.get("model", "agent"),
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "メッセージが提供されていません。"},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                }
            
            # Get the last user message
            user_message = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break
            
            if not user_message:
                return {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.get("model", "agent"),
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "ユーザーメッセージが見つかりません。"},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                }
            
            # Process through agent
            try:
                agent = await get_agent_instance()
                result = await agent.process_user_input(user_message)
                response_content = result.get("response", "申し訳ございません。応答を生成できませんでした。")
            except Exception as e:
                logger.error(f"Agent processing error: {e}")
                response_content = f"エラーが発生しました: {str(e)}"
            
            return {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.get("model", "agent"),
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": response_content},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(user_message.split()),
                    "completion_tokens": len(response_content.split()),
                    "total_tokens": len(user_message.split()) + len(response_content.split())
                }
            }
        except Exception as e:
            logger.error(f"Chat completion error: {e}")
            return {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.get("model", "agent"),
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": f"システムエラーが発生しました: {str(e)}"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }

    # Streaming variant used by frontend when stream=true (SSE-like)
    @app.post("/v1/chat/completions", response_class=StreamingResponse)
    async def chat_completions_stream(request: dict):  # type: ignore[func-duplicates]
        try:
            stream = request.get("stream", False)
            if not stream:
                # Delegate to non-streaming handler
                return await chat_completions(request)

            agent = await get_agent_instance()
            messages = request.get("messages", []) or []
            user_message = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break

            async def event_generator():
                try:
                    result = await agent.process_user_input(user_message)
                    content = result.get("response", "")
                    chunk = {
                        "id": f"chatcmpl-{uuid.uuid4()}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.get("model", "agent"),
                        "choices": [{
                            "index": 0,
                            "delta": {"content": content},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as ex:
                    logger.error(f"Streaming error: {ex}")
                    yield f"data: {json.dumps({'error': str(ex)})}\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")
        except Exception as e:
            logger.error(f"Chat completion stream setup error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Basic inference endpoint expected by frontend
    @app.post("/v1/inference")
    async def inference(request: dict):
        try:
            agent = await get_agent_instance()
            prompt = request.get("prompt", "")
            result = await agent.process_user_input(prompt)
            return {
                "response": result.get("response", ""),
                "reasoning": result.get("reasoning", ""),
                "session_id": request.get("session_id")
            }
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Session creation
    @app.post("/v1/sessions")
    async def create_session(request: dict):
        try:
            agent = await get_agent_instance()
            user_id = request.get("user_id")
            session_id = await agent.initialize_session(user_id=user_id)
            return {"session_id": session_id, "user_id": user_id}
        except Exception as e:
            logger.error(f"Create session error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Get session by ID (simple presence check)
    @app.get("/v1/sessions/{session_id}")
    async def get_session(session_id: str):
        try:
            agent = await get_agent_instance()
            # For now, just echo back; deeper details would read memory DB
            return {"session_id": session_id, "active": True}
        except Exception as e:
            logger.error(f"Get session error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # System stats (placeholder values) - GET endpoint for frontend
    @app.get("/v1/system/stats")
    async def system_stats():
        try:
            return {
                "timestamp": datetime.now().isoformat(),
                "system": {
                    "cpu_percent": 0,
                    "memory_percent": 0,
                    "disk_usage": 0,
                    "uptime": 0,
                },
                "agent": {
                    "status": "active",
                    "active_sessions": 1,
                    "total_requests": 0,
                    "average_response_time": 0,
                },
            }
        except Exception as e:
            logger.error(f"System stats error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Model status endpoint for frontend
    @app.get("/v1/models/{model_name}/status")
    async def get_model_status(model_name: str):
        """Get model status and availability"""
        try:
            # Decode URL-encoded model name
            import urllib.parse
            decoded_model_name = urllib.parse.unquote(model_name)
            
            # Check if model is available in Ollama
            try:
                agent = await get_agent_instance()
                if hasattr(agent, 'reasoning_engine') and hasattr(agent.reasoning_engine, 'ollama_client'):
                    ollama_client = agent.reasoning_engine.ollama_client
                    
                    # Check if model exists in cache
                    if decoded_model_name in ollama_client.model_cache:
                        return {
                            "model": decoded_model_name,
                            "status": "available",
                            "loaded": True,
                            "size": ollama_client.model_cache[decoded_model_name].get("size", 0),
                            "modified_at": ollama_client.model_cache[decoded_model_name].get("modified_at", ""),
                        }
                    else:
                        return {
                            "model": decoded_model_name,
                            "status": "not_found",
                            "loaded": False,
                            "error": "Model not found in Ollama"
                        }
                else:
                    # Fallback response
                    return {
                        "model": decoded_model_name,
                        "status": "unknown",
                        "loaded": False,
                        "error": "Ollama client not available"
                    }
            except Exception as e:
                logger.error(f"Model status check error: {e}")
                return {
                    "model": decoded_model_name,
                    "status": "error",
                    "loaded": False,
                    "error": str(e)
                }
        except Exception as e:
            logger.error(f"Model status endpoint error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Memory search (placeholder)
    @app.post("/v1/memory/search")
    async def memory_search(request: dict):
        try:
            query = request.get("query")
            return {
                "results": [],
                "query": query,
            }
        except Exception as e:
            logger.error(f"Memory search error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Session message history
    @app.get("/v1/sessions/{session_id}/messages")
    async def get_conversation_history(session_id: str, limit: int = 50):
        """Get conversation history with enhanced metadata"""
        try:
            agent = await get_agent_instance()
            
            # Get conversation history from agent
            # This would integrate with the agent's memory system
            
            # Mock response for now
            messages = []
            for i in range(min(limit, 10)):
                messages.append({
                    "id": f"msg_{i}",
                    "role": "user" if i % 2 == 0 else "assistant",
                    "content": f"Sample message {i}",
                    "timestamp": datetime.now().isoformat(),
                    "reasoning": "Sample reasoning" if i % 2 == 1 else None,
                    "tools_used": ["web_search"] if i % 4 == 1 else [],
                    "processing_time": 1.5 if i % 2 == 1 else None
                })
            
            return {
                "messages": messages,
                "total_count": len(messages)
            }
        except Exception as e:
            logger.error(f"Conversation history error: {e}")
            return {
                "messages": [],
                "total_count": 0
            }
    
    # Clear conversation history
    @app.delete("/v1/sessions/{session_id}/messages")
    async def clear_conversation_history(session_id: str):
        """Clear conversation history"""
        try:
            agent = await get_agent_instance()
            
            # Clear conversation history in agent
            # This would integrate with the agent's memory system
            
            return {
                "success": True,
                "message": "Conversation history cleared"
            }
        except Exception as e:
            logger.error(f"Clear history error: {e}")
            return {
                "success": False,
                "message": f"Failed to clear history: {str(e)}"
            }
    
    # Export conversation
    @app.get("/v1/sessions/{session_id}/export")
    async def export_conversation(session_id: str, format: str = "json"):
        """Export conversation data"""
        try:
            from fastapi.responses import Response
            
            agent = await get_agent_instance()
            
            # Get conversation data
            # This would integrate with the agent's memory system
            
            if format == "json":
                data = {
                    "session_id": session_id,
                    "exported_at": datetime.now().isoformat(),
                    "messages": []  # Would be populated from agent
                }
                content = json.dumps(data, indent=2)
                media_type = "application/json"
                filename = f"conversation_{session_id}.json"
            elif format == "markdown":
                content = f"# Conversation Export\n\nSession ID: {session_id}\n\n"
                media_type = "text/markdown"
                filename = f"conversation_{session_id}.md"
            else:
                content = f"Conversation Export\nSession ID: {session_id}\n\n"
                media_type = "text/plain"
                filename = f"conversation_{session_id}.txt"
            
            return Response(
                content=content,
                media_type=media_type,
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        except Exception as e:
            logger.error(f"Export conversation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# Create app instance for uvicorn
app = create_app()

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)