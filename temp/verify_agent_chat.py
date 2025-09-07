import asyncio
import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.getcwd())

async def main():
    from src.advanced_agent.core.self_learning_agent import SelfLearningAgent
    agent = SelfLearningAgent(config_path="config/agent_config.yaml", db_path="data/test_agent.db")
    # Initialize persistent session
    await agent.initialize_session(session_id=None, user_id="verification_user")
    result = await agent.process_user_input("テストです。短く1文で応答してください。")
    resp = result.get("response", "")
    # Print a safe ASCII-only prefix to avoid Windows cp932 emoji issues
    out = resp.replace("\n", " ")
    print((out[:500]).encode("cp932", errors="ignore").decode("cp932", errors="ignore"))
    # Avoid calling agent.close() due to potential missing close on underlying clients

if __name__ == "__main__":
    asyncio.run(main())
