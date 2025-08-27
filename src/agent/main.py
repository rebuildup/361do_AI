#!/usr/bin/env python3
"""
AI Agent Studio - Main Application
自己学習型AIエージェントのメインアプリケーション
"""

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

# プロジェクトルートをPATHに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.core.agent_manager import AgentManager
from agent.core.config import Config
from agent.core.database import DatabaseManager
from agent.self_tuning.continuous_learning import ContinuousLearningEngine


class AgentApplication:
    """メインアプリケーションクラス"""

    def __init__(self):
        self.config = Config()
        self.db_manager = None
        self.agent_manager = None
        self.learning_engine = None

    async def initialize(self):
        """アプリケーションの初期化"""
        logger.info("Initializing AI Agent Studio...")

        # データベース初期化
        self.db_manager = DatabaseManager(self.config.database_url)
        await self.db_manager.initialize()
        logger.info("Database initialized")

        # エージェントマネージャー初期化
        self.agent_manager = AgentManager(
            config=self.config,
            db_manager=self.db_manager
        )
        await self.agent_manager.initialize()
        logger.info("Agent Manager initialized")

        # 継続学習エンジン初期化
        self.learning_engine = ContinuousLearningEngine(
            agent_manager=self.agent_manager,
            db_manager=self.db_manager,
            config=self.config
        )
        await self.learning_engine.start_learning_cycle()
        logger.info("Continuous Learning Engine started")

        logger.info("AI Agent Studio initialized successfully!")

    async def shutdown(self):
        """アプリケーションのシャットダウン"""
        logger.info("Shutting down AI Agent Studio...")

        if self.learning_engine:
            await self.learning_engine.stop()

        if self.agent_manager:
            await self.agent_manager.shutdown()

        if self.db_manager:
            await self.db_manager.close()

        logger.info("AI Agent Studio shutdown complete")


# グローバルアプリケーションインスタンス
app_instance = AgentApplication()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションライフサイクル管理"""
    # 起動時
    await app_instance.initialize()
    yield
    # シャットダウン時
    await app_instance.shutdown()


# FastAPIアプリケーション作成
app = FastAPI(
    title="AI Agent Studio",
    description="自己学習型AIエージェントシステム",
    version="1.0.0",
    lifespan=lifespan
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント"""
    return {"status": "healthy", "service": "AI Agent Studio"}


@app.get("/api/status")
async def get_status():
    """システムステータス取得"""
    try:
        if app_instance.agent_manager is None:
            raise HTTPException(status_code=503, detail="Agent manager is not initialized")

        status = await app_instance.agent_manager.get_system_status()
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(request: dict):
    """チャットエンドポイント"""
    try:
        user_input = request.get("message", "")
        session_id = request.get("session_id")

        if not user_input:
            raise HTTPException(status_code=400, detail="Message is required")

        if app_instance.agent_manager is None:
            raise HTTPException(status_code=503, detail="Agent manager is not initialized")

        response = await app_instance.agent_manager.process_message(
            user_input=user_input,
            session_id=session_id
        )

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Chat processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback")
async def submit_feedback(request: dict):
    """フィードバック送信エンドポイント"""
    try:
        conversation_id = request.get("conversation_id")
        feedback_score = request.get("feedback_score")  # -1, 0, 1
        feedback_comment = request.get("feedback_comment", "")

        if conversation_id is None or feedback_score is None:
            raise HTTPException(
                status_code=400,
                detail="conversation_id and feedback_score are required"
            )

        if app_instance.learning_engine is None:
            raise HTTPException(status_code=503, detail="Learning engine is not initialized")

        await app_instance.learning_engine.process_user_feedback(
            conversation_id=conversation_id,
            feedback_score=feedback_score,
            feedback_comment=feedback_comment
        )

        return JSONResponse(content={"status": "feedback_received"})

    except Exception as e:
        logger.error(f"Feedback processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/learning/report")
async def get_learning_report():
    """学習レポート取得"""
    try:
        if app_instance.learning_engine is None:
            raise HTTPException(status_code=503, detail="Learning engine is not initialized")

        report = await app_instance.learning_engine.generate_learning_report()
        return JSONResponse(content=report)

    except Exception as e:
        logger.error(f"Learning report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/learning/run_once')
async def run_learning_once():
    """学習エンジンを一回だけ実行する（同期）。テストや管理用。"""
    try:
        if app_instance.learning_engine is None:
            raise HTTPException(status_code=503, detail="Learning engine is not initialized")

        result = await app_instance.learning_engine.run_once()
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Run-once learning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/learning/status')
async def learning_status():
    """学習エンジンの状態取得"""
    try:
        if app_instance.learning_engine is None:
            raise HTTPException(status_code=503, detail="Learning engine is not initialized")

        status = await app_instance.learning_engine.get_status()
        return JSONResponse(content=status)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get learning status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/web-design/generate")
async def generate_web_design(request: dict):
    """Webデザイン生成エンドポイント（削除）"""
    raise HTTPException(status_code=404, detail="web_design feature has been removed")


def setup_logging():
    """ログ設定"""
    # Loguruの設定
    logger.remove()  # デフォルトハンドラーを削除

    # コンソール出力
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        level="INFO"
    )

    # ファイル出力
    # パスはConfigから取得するように変更する可能性
    log_file_path = os.path.join(os.path.dirname(__file__), "..", "data", "logs", "agent.log")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logger.add(
        log_file_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days"
    )


async def run_manual_learning():
    """手動で自己学習サイクルを実行する"""
    logger.info("手動学習サイクルを開始します...")

    config = Config()
    db_manager = None
    agent_manager_instance = None

    try:
        db_manager = DatabaseManager(config.database_url)
        await db_manager.initialize()

        agent_manager_instance = AgentManager(config=config, db_manager=db_manager)
        await agent_manager_instance.initialize()

        # self_tuningモジュールはここでインポート
        from agent.self_tuning.learning_loop import LearningLoop

        if agent_manager_instance.ollama_client is None:
            logger.warning("OLLAMA client not initialized; skipping learning loop")
        else:
            learning_loop = LearningLoop(config, agent_manager_instance.ollama_client)

            logger.info("手動学習のための初期化が完了しました。")
            await learning_loop.run_learning_cycle()

    except Exception as e:
        logger.error(f"手動学習サイクル中にエラーが発生しました: {e}", exc_info=True)
    finally:
        logger.info("シャットダウン処理を開始します。")
        if agent_manager_instance:
            await agent_manager_instance.shutdown()
        if db_manager:
            await db_manager.close()

    logger.info("手動学習サイクルが完了しました。")


if __name__ == "__main__":
    setup_logging()

    if len(sys.argv) > 1 and sys.argv[1] == 'learn':
        # `python -m agent.main learn` のように実行された場合
        asyncio.run(run_manual_learning())
    else:
        # 通常のサーバー起動
        logger.info("Starting AI Agent Studio...")
        uvicorn.run(
                app,
                host="0.0.0.0",
                port=8000,
                reload=False,
                log_level="info"
        )
