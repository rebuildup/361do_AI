import os
import sys
import asyncio
from datetime import datetime

import pytest

# Ensure 'src' is on sys.path so package imports like `agent.*` resolve when running tests from repo root
ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from agent.core.config import Config
from agent.core.database import DatabaseManager
from agent.core.agent_manager import AgentManager
from agent.tools.file_tool import FileTool
from agent.tools.learning_tool import LearningTool

# Avoid contacting local ollama daemon during tests
os.environ.setdefault('AGENT_SKIP_OLLAMA', '1')

@pytest.mark.asyncio
async def test_agent_self_edit_write_and_read():
    # リポジトリルートをプロジェクトルートとして使用
    project_root = os.getcwd()
    test_rel_path = "src/data/prompts/self_edit_test_agent.txt"
    test_full_path = os.path.join(project_root, test_rel_path)
    content = f"agent-wrote: {datetime.utcnow().isoformat()}"

    config = Config()
    db = DatabaseManager(config.database_url)

    # 初期化と後始末は確実に行う
    try:
        await db.initialize()

        agent = AgentManager(config, db)

        # ファイルツールを手動で差し込む（AgentManager.initialize を呼ばず、外部依存を避ける）
        file_tool = FileTool(project_root=project_root)
        agent.tools['file'] = file_tool
        await file_tool.initialize()

        # write via self-edit handler
        write_cmd = f"write file {test_rel_path}\n{content}"
        write_res = await agent._handle_self_edit(write_cmd, {})
        assert isinstance(write_res, str)
        assert "書き込" in write_res or "正常" in write_res

        # read via self-edit handler
        read_cmd = f"read file {test_rel_path}"
        read_res = await agent._handle_self_edit(read_cmd, {})
        assert isinstance(read_res, str)
        assert read_res.strip() == content

    finally:
        # cleanup
        try:
            if os.path.exists(test_full_path):
                os.remove(test_full_path)
        except Exception:
            pass
        try:
            await db.close()
        except Exception:
            pass
        try:
            if 'file_tool' in locals():
                await file_tool.close()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_agent_self_edit_update_prompt_and_add_learning():
    project_root = os.getcwd()
    config = Config()
    db = DatabaseManager(config.database_url)

    try:
        await db.initialize()

        agent = AgentManager(config, db)

        # Prepare LearningTool and attach to agent
        from agent.core.ollama_client import OllamaClient
        # Dummy Ollama client to avoid network calls in tests
        class DummyOllama(OllamaClient):
            def __init__(self, config):
                self.config = config
                self.base_url = config.get('base_url', '')
                self.model = config.get('model', '')
                self.session = None

            async def initialize(self):
                return

            async def close(self):
                return

            async def generate(self, *args, **kwargs):
                return "0.0"

            async def generate_response(self, *args, **kwargs):
                return "0.0"

        dummy_ollama = DummyOllama(config.ollama_config)
        learning_tool = LearningTool(db_manager=db, config=config, ollama_client=dummy_ollama, agent_manager=agent)
        agent.learning_tool = learning_tool

        # 1) Add initial prompt via the LearningTool directly
        name = "test_greeting_prompt"
        initial = "Hello, I am the agent."
        await learning_tool.add_prompt_template(name=name, content=initial, description="test init")

        # 2) Update prompt via agent self-edit handler
        new_content = "Hello, I am the updated agent."
        update_cmd = f"update prompt {name}: {new_content}"
        update_res = await agent._handle_self_edit(update_cmd, {})
        assert isinstance(update_res, str)

        # Verify DB reflects updated content
        tmpl = await db.get_prompt_template_by_name(name)
        assert tmpl is not None
        assert new_content in tmpl.get('template_content', '')

        # 3) Add learning data via agent self-edit
        learning_text = "This is a test learning item added by the agent."
        add_cmd = f"add learning data: {learning_text}"
        add_res = await agent._handle_self_edit(add_cmd, {})
        assert isinstance(add_res, str)

        # Verify learning data exists in DB under category 'custom'
        items = await db.get_learning_data(category='custom', min_quality=None, limit=50)
        # items is a list of dicts
        found = any(learning_text in it.get('content', '') for it in items)
        assert found, "Added learning data not found in DB"

    finally:
        try:
            await db.close()
        except Exception:
            pass
