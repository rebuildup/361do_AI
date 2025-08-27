
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call, mock_open
from datetime import datetime, timedelta

from agent.core.config import Config, PathConfig
from agent.self_tuning.advanced_learning import AdvancedLearningSystem, LearningData

@pytest.fixture
def mock_config():
    config = MagicMock(spec=Config)
    config.paths = MagicMock(spec=PathConfig)
    config.settings = MagicMock()
    config.paths.learning_data_dir = "/fake/learning_data"
    config.paths.prompts_dir = "/fake/prompts"
    config.settings.learning_interval_minutes = 30
    return config

@pytest.fixture
def mock_db_manager():
    return AsyncMock()

@pytest.fixture
def mock_ollama_client():
    client = AsyncMock()
    client.generate_response.return_value = '{"key": "value"}'
    return client

@pytest.fixture
def mock_agent_manager():
    return AsyncMock()

@pytest.fixture
def learning_system(mock_config, mock_db_manager, mock_ollama_client, mock_agent_manager):
    with patch('pathlib.Path.mkdir'):
        system = AdvancedLearningSystem(
            config=mock_config,
            db_manager=mock_db_manager,
            ollama_client=mock_ollama_client,
            agent_manager=mock_agent_manager,
        )
    return system

@pytest.mark.asyncio
async def test_analyze_and_improve_learning_data(learning_system, mock_db_manager, mock_ollama_client):
    # Arrange
    mock_db_manager.get_conversations_by_quality.return_value = [
        {"id": 1, "user_input": "test_user", "agent_response": "test_agent", "quality_score": 0.9}
    ]
    mock_db_manager.get_learning_data_by_quality.return_value = [
        {"id": "low_quality_1", "content": "low quality content", "quality_score": 0.5, "category": "general"}
    ]
    mock_db_manager.get_unused_learning_data.return_value = []
    mock_ollama_client.generate_response.side_effect = [
        '{"category": "extracted", "content": "extracted_content", "tags": []}', # For extraction
        'improved_content' # For improvement
    ]

    # Act
    with patch("builtins.open", mock_open()) as mock_file:
        await learning_system._analyze_and_improve_learning_data()

    # Assert
    mock_db_manager.get_conversations_by_quality.assert_called_once_with(min_score=0.7, max_score=1.0, limit=50)
    mock_ollama_client.generate_response.assert_any_call(A_MATCHING_PROMPT_FOR_EXTRACTION)
    mock_db_manager.insert_learning_data.assert_called_once()
    mock_db_manager.get_learning_data_by_quality.assert_called_once_with(max_score=0.6, limit=20)
    mock_ollama_client.generate_response.assert_any_call(A_MATCHING_PROMPT_FOR_IMPROVEMENT)
    mock_db_manager.update_learning_data.assert_called_once_with(
        data_id="low_quality_1",
        content="improved_content",
        quality_score=pytest.approx(0.7)
    )
    mock_db_manager.get_unused_learning_data.assert_called_once()

@pytest.mark.asyncio
async def test_optimize_prompts(learning_system, mock_db_manager, mock_ollama_client):
    # Arrange
    mock_db_manager.get_all_prompt_templates.return_value = [
        {"name": "test_prompt", "template_content": "original", "description": "desc"}
    ]
    mock_ollama_client.generate_response.side_effect = [
        "optimized_prompt", # For optimization
        "0.5" # For evaluation
    ]

    # Act
    await learning_system._optimize_prompts()

    # Assert
    mock_db_manager.get_all_prompt_templates.assert_called_once()
    mock_ollama_client.generate_response.assert_any_call(A_MATCHING_PROMPT_FOR_OPTIMIZATION)
    mock_ollama_client.generate_response.assert_any_call(A_MATCHING_PROMPT_FOR_EVALUATION)
    mock_db_manager.update_prompt_template.assert_called_once_with(
        name="test_prompt",
        template_content="optimized_prompt"
    )
    mock_db_manager.insert_prompt_optimization_history.assert_called_once()

@pytest.mark.asyncio
async def test_cleanup_unused_learning_data(learning_system, mock_db_manager):
    # Arrange
    cutoff_date = datetime.now() - timedelta(days=30)
    mock_db_manager.get_unused_learning_data.return_value = [
        {"id": "unused_1", "category": "general"}
    ]

    # Act
    with patch('pathlib.Path.exists', return_value=True), patch('pathlib.Path.unlink') as mock_unlink:
        await learning_system._cleanup_unused_learning_data()

    # Assert
    mock_db_manager.get_unused_learning_data.assert_called_with(
        ANY,
        limit=50
    )
    mock_db_manager.delete_learning_data.assert_called_once_with("unused_1")
    mock_unlink.assert_called_once()

from unittest.mock import ANY

# Helper constants for matching prompts
A_MATCHING_PROMPT_FOR_EXTRACTION = ANY
A_MATCHING_PROMPT_FOR_IMPROVEMENT = ANY
A_MATCHING_PROMPT_FOR_OPTIMIZATION = ANY
A_MATCHING_PROMPT_FOR_EVALUATION = ANY
