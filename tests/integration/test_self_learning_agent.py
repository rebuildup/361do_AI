"""
自己学習AIエージェント統合テスト
Integration tests for Self-Learning AI Agent
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

from src.advanced_agent.core.self_learning_agent import SelfLearningAgent, create_self_learning_agent
from src.advanced_agent.config.settings import AgentConfig


class TestSelfLearningAgent:
    """自己学習AIエージェントテスト"""
    
    @pytest.fixture
    async def temp_agent(self):
        """一時エージェント作成"""
        # 一時ディレクトリ作成
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test_agent.db"
        
        try:
            # エージェント作成
            agent = SelfLearningAgent(db_path=str(db_path))
            await agent.initialize_session()
            
            yield agent
            
        finally:
            # クリーンアップ
            await agent.close()
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, temp_agent):
        """エージェント初期化テスト"""
        assert temp_agent.current_state is not None
        assert temp_agent.current_state.session_id is not None
        assert temp_agent.current_state.learning_epoch == 0
        assert temp_agent.current_state.total_interactions == 0
        assert temp_agent.current_state.reward_score == 0.0
    
    @pytest.mark.asyncio
    async def test_initial_prompt_template_creation(self, temp_agent):
        """初期プロンプトテンプレート作成テスト"""
        assert len(temp_agent.prompt_templates) > 0
        assert "1.0.0" in temp_agent.prompt_templates
        
        template = temp_agent.prompt_templates["1.0.0"]
        assert "自己学習型AIエージェント" in template.content
        assert "永続的記憶" in template.content
        assert "自己改善" in template.content
    
    @pytest.mark.asyncio
    async def test_user_input_processing(self, temp_agent):
        """ユーザー入力処理テスト"""
        user_input = "こんにちは、自己学習AIエージェントです。"
        
        result = await temp_agent.process_user_input(user_input)
        
        assert "response" in result
        assert "interaction_id" in result
        assert "processing_time" in result
        assert "agent_state" in result
        
        # エージェント状態の更新確認
        assert temp_agent.current_state.total_interactions == 1
        assert temp_agent.current_state.reward_score > 0
    
    @pytest.mark.asyncio
    async def test_memory_storage(self, temp_agent):
        """メモリ保存テスト"""
        user_input = "私の名前は田中太郎です。"
        
        result = await temp_agent.process_user_input(user_input)
        
        # チューニングデータの保存確認
        assert len(temp_agent.tuning_data_pool) > 0
        
        # 最新のデータを確認
        latest_data = temp_agent.tuning_data_pool[-1]
        assert latest_data.data_type == "conversation"
        assert "田中太郎" in latest_data.content
    
    @pytest.mark.asyncio
    async def test_reward_calculation(self, temp_agent):
        """報酬計算テスト"""
        # 感謝の表現を含む入力
        user_input = "ありがとうございます。とても役に立ちました。"
        
        result = await temp_agent.process_user_input(user_input)
        
        # 報酬が計算されていることを確認
        assert "reward" in result
        assert result["reward"] > 0
        
        # エージェント状態の報酬スコアが更新されていることを確認
        assert temp_agent.current_state.reward_score > 0
    
    @pytest.mark.asyncio
    async def test_evolution_trigger(self, temp_agent):
        """進化トリガーテスト"""
        # 進化間隔を短く設定
        temp_agent.learning_config["fitness_evaluation_interval"] = 2
        
        # 複数回のインタラクション
        for i in range(3):
            await temp_agent.process_user_input(f"テストメッセージ {i+1}")
        
        # 進化が実行されていることを確認
        assert temp_agent.current_state.evolution_generation > 0
        assert temp_agent.current_state.learning_epoch > 0
    
    @pytest.mark.asyncio
    async def test_agent_status(self, temp_agent):
        """エージェント状態取得テスト"""
        # いくつかのインタラクションを実行
        await temp_agent.process_user_input("テスト1")
        await temp_agent.process_user_input("テスト2")
        
        status = await temp_agent.get_agent_status()
        
        assert status["status"] == "active"
        assert status["session_id"] == temp_agent.current_state.session_id
        assert status["total_interactions"] == 2
        assert status["learning_epoch"] == 0
        assert status["reward_score"] > 0
        assert "prompt_templates_count" in status
        assert "tuning_data_count" in status
    
    @pytest.mark.asyncio
    async def test_session_persistence(self):
        """セッション永続化テスト"""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "persistence_test.db"
        
        try:
            # 最初のエージェント
            agent1 = SelfLearningAgent(db_path=str(db_path))
            session_id = await agent1.initialize_session()
            
            # インタラクション実行
            await agent1.process_user_input("永続化テストメッセージ")
            original_interactions = agent1.current_state.total_interactions
            
            await agent1.close()
            
            # 新しいエージェントで同じセッションを復元
            agent2 = SelfLearningAgent(db_path=str(db_path))
            await agent2.initialize_session(session_id=session_id)
            
            # 状態が復元されていることを確認
            assert agent2.current_state.session_id == session_id
            assert agent2.current_state.total_interactions == original_interactions
            
            await agent2.close()
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, temp_agent):
        """エラーハンドリングテスト"""
        # 空の入力
        result = await temp_agent.process_user_input("")
        assert "response" in result
        
        # 非常に長い入力
        long_input = "テスト" * 1000
        result = await temp_agent.process_user_input(long_input)
        assert "response" in result
        
        # 特殊文字を含む入力
        special_input = "特殊文字テスト: !@#$%^&*()_+{}|:<>?[]\\;'\",./"
        result = await temp_agent.process_user_input(special_input)
        assert "response" in result


class TestAgentIntegration:
    """エージェント統合テスト"""
    
    @pytest.mark.asyncio
    async def test_create_self_learning_agent(self):
        """自己学習エージェント作成テスト"""
        temp_dir = tempfile.mkdtemp()
        config_path = Path(temp_dir) / "test_config.yaml"
        
        try:
            # テスト用設定ファイル作成
            config_content = """
name: "TestAgent"
version: "1.0.0"
database:
  type: "sqlite"
  path: "data/test_agent.db"
ollama:
  base_url: "http://localhost:11434"
  model: "qwen2:7b-instruct"
"""
            config_path.write_text(config_content)
            
            # エージェント作成
            agent = await create_self_learning_agent(str(config_path))
            
            assert agent is not None
            assert agent.current_state is not None
            
            await agent.close()
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_agent_workflow(self):
        """エージェントワークフローテスト"""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "workflow_test.db"
        
        try:
            agent = SelfLearningAgent(db_path=str(db_path))
            await agent.initialize_session()
            
            # 複数のインタラクションを実行
            test_inputs = [
                "こんにちは",
                "私の趣味は読書です",
                "プログラミングについて教えてください",
                "ありがとうございました"
            ]
            
            for i, user_input in enumerate(test_inputs):
                result = await agent.process_user_input(user_input)
                
                assert "response" in result
                assert result["agent_state"]["total_interactions"] == i + 1
            
            # 最終状態確認
            final_status = await agent.get_agent_status()
            assert final_status["total_interactions"] == len(test_inputs)
            assert final_status["reward_score"] > 0
            
            await agent.close()
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# パフォーマンステスト
class TestAgentPerformance:
    """エージェントパフォーマンステスト"""
    
    @pytest.mark.asyncio
    async def test_response_time(self):
        """レスポンス時間テスト"""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "performance_test.db"
        
        try:
            agent = SelfLearningAgent(db_path=str(db_path))
            await agent.initialize_session()
            
            user_input = "パフォーマンステスト用のメッセージです。"
            
            import time
            start_time = time.time()
            result = await agent.process_user_input(user_input)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # レスポンス時間が妥当な範囲内であることを確認
            assert processing_time < 30.0  # 30秒以内
            assert result["processing_time"] < 30.0
            
            await agent.close()
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """メモリ使用量テスト"""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "memory_test.db"
        
        try:
            agent = SelfLearningAgent(db_path=str(db_path))
            await agent.initialize_session()
            
            # 複数のインタラクションを実行してメモリ使用量を確認
            for i in range(10):
                await agent.process_user_input(f"メモリテストメッセージ {i+1}")
            
            # メモリ使用量が妥当な範囲内であることを確認
            assert len(agent.tuning_data_pool) <= 10
            assert len(agent.prompt_templates) >= 1
            
            await agent.close()
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
