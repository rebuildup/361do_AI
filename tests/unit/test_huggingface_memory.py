"""
Test LangChain Memory + HuggingFace Memory System

LangChain Memory + HuggingFace 統合記憶システムのテスト
"""

import pytest
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

from src.advanced_agent.memory.persistent_memory import LangChainPersistentMemory
from src.advanced_agent.memory.huggingface_memory import (
    HuggingFaceMemoryClassifier,
    LangChainHuggingFaceMemory
)


class TestHuggingFaceMemoryClassifier:
    """HuggingFace 記憶分類器のテスト"""
    
    @pytest.fixture
    def classifier(self):
        """テスト用分類器"""
        return HuggingFaceMemoryClassifier(device="cpu")
    
    @pytest.mark.asyncio
    async def test_importance_evaluation(self, classifier):
        """重要度評価テスト"""
        
        # 高重要度テキスト
        high_importance_text = "重要なエラーが発生しました。システムの性能に大きな問題があります。"
        high_score = await classifier.evaluate_importance(high_importance_text)
        
        # 低重要度テキスト
        low_importance_text = "こんにちは"
        low_score = await classifier.evaluate_importance(low_importance_text)
        
        # 技術的テキスト
        tech_text = "Pythonでの機械学習モデルの訓練において、GPU メモリ不足が発生しています。"
        tech_score = await classifier.evaluate_importance(tech_text)
        
        assert 0.0 <= high_score <= 1.0
        assert 0.0 <= low_score <= 1.0
        assert 0.0 <= tech_score <= 1.0
        assert tech_score > low_score  # 技術的内容は重要度が高い
    
    @pytest.mark.asyncio
    async def test_memory_type_classification(self, classifier):
        """記憶タイプ分類テスト"""
        
        # 技術的知識
        tech_text = "Pythonでリストを作成するには[]を使用します"
        tech_type = await classifier.classify_memory_type(tech_text)
        
        # 手続き的知識
        procedure_text = "まず設定ファイルを開いて、次にパラメータを変更します"
        procedure_type = await classifier.classify_memory_type(procedure_text)
        
        # エラー情報
        error_text = "ImportError: No module named 'numpy'"
        error_type = await classifier.classify_memory_type(error_text)
        
        assert isinstance(tech_type, str)
        assert isinstance(procedure_type, str)
        assert isinstance(error_type, str)
    
    @pytest.mark.asyncio
    async def test_key_concept_extraction(self, classifier):
        """キー概念抽出テスト"""
        
        text = "PythonとJavaScriptを使ってReactアプリケーションを開発し、AWSにデプロイしました。"
        concepts = await classifier.extract_key_concepts(text)
        
        assert isinstance(concepts, list)
        assert len(concepts) <= 10
        
        # 期待される概念が含まれているかチェック
        expected_concepts = ["Python", "JavaScript", "React", "AWS"]
        found_concepts = [c for c in expected_concepts if c in concepts]
        assert len(found_concepts) > 0


class TestLangChainHuggingFaceMemory:
    """LangChain + HuggingFace 統合記憶システムのテスト"""
    
    @pytest.fixture
    def temp_dir(self):
        """テスト用一時ディレクトリ"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def memory_system(self, temp_dir):
        """テスト用記憶システム"""
        db_path = Path(temp_dir) / "test_memory.db"
        chroma_path = Path(temp_dir) / "test_chroma"
        
        # 永続化記憶システム
        persistent_memory = LangChainPersistentMemory(
            db_path=str(db_path),
            chroma_path=str(chroma_path)
        )
        
        # HuggingFace統合記憶システム
        hf_memory = LangChainHuggingFaceMemory(
            persistent_memory=persistent_memory,
            max_short_term_memories=10,
            importance_threshold=0.6
        )
        
        yield hf_memory
        persistent_memory.close()
    
    @pytest.mark.asyncio
    async def test_conversation_processing(self, memory_system):
        """会話処理テスト"""
        
        # セッション初期化
        session_id = await memory_system.persistent_memory.initialize_session(user_id="test_user")
        
        # 会話処理
        result = await memory_system.process_conversation(
            user_input="Pythonでの機械学習について教えて",
            agent_response="Pythonでは scikit-learn, TensorFlow, PyTorch などが人気です。",
            session_id=session_id,
            metadata={"test": True}
        )
        
        assert "conversation_id" in result
        assert "importance_score" in result
        assert "memory_type" in result
        assert "key_concepts" in result
        assert "promoted_to_long_term" in result
        assert "processing_stats" in result
        
        assert 0.0 <= result["importance_score"] <= 1.0
        assert isinstance(result["memory_type"], str)
        assert isinstance(result["key_concepts"], list)
        assert isinstance(result["promoted_to_long_term"], bool)
    
    @pytest.mark.asyncio
    async def test_contextual_memory_retrieval(self, memory_system):
        """コンテキスト記憶検索テスト"""
        
        # セッション初期化
        session_id = await memory_system.persistent_memory.initialize_session()
        
        # 複数の会話を処理
        conversations = [
            ("Pythonについて教えて", "Pythonは汎用プログラミング言語です。"),
            ("機械学習のライブラリは？", "scikit-learn, TensorFlow, PyTorchが人気です。"),
            ("データベースの設計について", "正規化とインデックスが重要です。")
        ]
        
        for user_input, agent_response in conversations:
            await memory_system.process_conversation(
                user_input=user_input,
                agent_response=agent_response,
                session_id=session_id
            )
        
        # コンテキスト検索
        context = await memory_system.retrieve_contextual_memories(
            query="Python 機械学習",
            session_id=session_id,
            max_results=5
        )
        
        assert "query_analysis" in context
        assert "short_term_context" in context
        assert "long_term_summary" in context
        assert "relevant_memories" in context
        assert "memory_statistics" in context
        
        # クエリ分析結果の確認
        query_analysis = context["query_analysis"]
        assert "importance" in query_analysis
        assert "type" in query_analysis
        assert "concepts" in query_analysis
        
        # 関連記憶の確認
        relevant_memories = context["relevant_memories"]
        assert isinstance(relevant_memories, list)
        
        # Python関連の記憶が上位に来ることを確認
        if relevant_memories:
            top_memory = relevant_memories[0]
            assert "relevance_score" in top_memory
            assert "type_match" in top_memory
            assert "concept_overlap" in top_memory
    
    @pytest.mark.asyncio
    async def test_memory_optimization(self, memory_system):
        """記憶最適化テスト"""
        
        # セッション初期化
        session_id = await memory_system.persistent_memory.initialize_session()
        
        # 短期記憶の容量を超える会話を追加
        for i in range(15):  # max_short_term_memories=10を超える
            await memory_system.process_conversation(
                user_input=f"質問 {i}",
                agent_response=f"回答 {i}",
                session_id=session_id
            )
        
        # 最適化実行
        optimization_result = await memory_system.optimize_memory_capacity()
        
        assert "short_term_optimized" in optimization_result
        assert "long_term_optimized" in optimization_result
        assert "persistent_cleaned" in optimization_result
        
        # 短期記憶が制限内に収まっていることを確認
        stats = memory_system.get_memory_statistics()
        assert stats["short_term_memory"]["current_messages"] <= memory_system.max_short_term_memories
    
    @pytest.mark.asyncio
    async def test_memory_pattern_analysis(self, memory_system):
        """記憶パターン分析テスト"""
        
        # セッション初期化
        session_id = await memory_system.persistent_memory.initialize_session()
        
        # 様々なタイプの会話を追加
        conversations = [
            ("Pythonのエラーが出ました", "ImportErrorの解決方法を説明します。"),
            ("機械学習の設定方法は？", "まずライブラリをインストールしてください。"),
            ("データベースの最適化", "インデックスを適切に設定することが重要です。"),
            ("Pythonでのテスト方法", "pytestを使用することを推奨します。")
        ]
        
        for user_input, agent_response in conversations:
            await memory_system.process_conversation(
                user_input=user_input,
                agent_response=agent_response,
                session_id=session_id
            )
        
        # パターン分析実行
        analysis = await memory_system.analyze_memory_patterns(session_id)
        
        assert "total_conversations" in analysis
        assert "memory_type_distribution" in analysis
        assert "top_concepts" in analysis
        assert "importance_stats" in analysis
        
        # 統計の妥当性確認
        assert analysis["total_conversations"] >= len(conversations)
        assert isinstance(analysis["memory_type_distribution"], dict)
        assert isinstance(analysis["top_concepts"], list)
        
        importance_stats = analysis["importance_stats"]
        assert "average" in importance_stats
        assert "min" in importance_stats
        assert "max" in importance_stats
        assert "high_importance_count" in importance_stats
    
    def test_memory_statistics(self, memory_system):
        """記憶統計テスト"""
        
        stats = memory_system.get_memory_statistics()
        
        assert "short_term_memory" in stats
        assert "long_term_memory" in stats
        assert "processing_stats" in stats
        assert "persistent_memory" in stats
        assert "importance_threshold" in stats
        
        # 短期記憶統計
        short_term = stats["short_term_memory"]
        assert "current_messages" in short_term
        assert "max_capacity" in short_term
        assert "usage_percent" in short_term
        
        # 処理統計
        processing = stats["processing_stats"]
        assert "total_processed" in processing
        assert "promoted_to_long_term" in processing


if __name__ == "__main__":
    pytest.main([__file__])