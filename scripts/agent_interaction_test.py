#!/usr/bin/env python3
"""
Agent interaction test script
"""
import asyncio
import os
from pathlib import Path

# Ensure package import path (add the repository's `src` directory)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.core.config import Config
from agent.core.database import DatabaseManager
from agent.core.agent_manager import AgentManager


async def main():
    # Skip real Ollama for tests
    os.environ.setdefault('AGENT_SKIP_OLLAMA', '1')

    print("Starting agent interaction test...")

    config = Config()
    db = DatabaseManager(config.database_url)
    await db.initialize()

    manager = AgentManager(config, db)
    await manager.initialize()

    try:
        messages = [
            "こんにちは",
            "Pythonについて教えてください",
            "write file src/data/prompts/test_agent_written.txt\nThis file was written by the agent for testing.",
            "read file src/data/prompts/test_agent_written.txt",
            "update prompt greeting_prompt: こんにちは、私はエージェントによって更新されたプロンプトです。",
            "add learning data: {\"content\": \"Agent added this learning item for test.\", \"category\": \"unit_test\", \"tags\": [\"agent\", \"test\"]}"
        ]

        session_id = None
        for msg in messages:
            print('\n---')
            print(f'USER: {msg}')
            resp = await manager.process_message(msg, session_id=session_id)
            session_id = resp.get('session_id', session_id)
            print(f"AGENT (response): {resp.get('response')[:400]}")
            print(f"AGENT (tools_used): {resp.get('tools_used')}")

        # Verify file was written
        file_path = Path('src/data/prompts/test_agent_written.txt')
        if file_path.exists():
            print('\nFile exists; reading back contents:')
            print(file_path.read_text(encoding='utf-8')[:400])
        else:
            print('\nFile not created by agent')

        # Check learning data was added (via DB)
        # Use db.get_learning_data
        learning_items = await db.get_learning_data(limit=10)
        print(f'\nLearning items found: {len(learning_items)}')
        for item in learning_items[:5]:
            print(f" - {item.get('id')} | {item.get('category')} | {str(item.get('content'))[:60]}")

        # Try a disallowed write (should be denied by FileTool whitelist)
        print('\n🔒 テスト: 禁止されているファイルへの書き込みを試みます（拒否されるべき）')
        bad_path_cmd = "write file src/agent/core/database.py\n# test malicious change"
        bad_resp = await manager.process_message(bad_path_cmd)
        print(f"DISALLOWED WRITE RESPONSE: {bad_resp.get('response', '')[:400]}")

        # Create a proposal by writing to prompts (auto_apply is False by default in config)
        print('\n📝 テスト: 提案作成（プロンプトを書いて提案を作る）')
        prop_cmd = "write file src/data/prompts/proposed_prompt.txt\nThis is a proposed prompt change."
        prop_resp = await manager.process_message(prop_cmd)
        print(f"PROPOSAL RESPONSE: {prop_resp.get('response', '')}")

        # List proposals and apply the newest one
        proposals_dir = Path(config.paths.proposals_dir)
        proposals = sorted([p.name for p in proposals_dir.glob('proposal_*.json')])
        if proposals:
            latest = proposals[-1]
            print(f"Applying proposal: {latest}")
            apply_res = await manager.apply_proposal(latest)
            print(f"APPLY RESULT: {apply_res}")
            # Verify applied file exists
            applied_file = Path('src/data/prompts/proposed_prompt.txt')
            if applied_file.exists():
                print('Applied file content:')
                print(applied_file.read_text(encoding='utf-8')[:400])
        else:
            print('No proposals found')

    finally:
        print('\nShutting down...')
        await manager.shutdown()
        await db.close()


if __name__ == '__main__':
    asyncio.run(main())
