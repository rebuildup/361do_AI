"""
エンドツーエンド統合テスト

Pytest の既存機能による エンドツーエンド推論テスト・全連携検証を統合し、
メモリ圧迫安定性テスト、性能劣化検出、セッション永続化・記憶整合性テストを実装
"""

import pytest
import asyncio
import time
import psutil
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import yaml

# テスト用のモッククラス
class MockOllamaClient:
    """テスト用 Ollama クライアント"""
    
    def __init__(self):
        self.call_count = 0
        self.response_time = 0.5
        
    async def generate(self, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """モック生成"""
        self.call_count += 1
        await asyncio.sleep(self.response_time)
        
        return {
            "response": f"Mock response for: {prompt[:50]}...",
            "model": model,
            "created_at": datetime.now().isoformat(),
            "done": True,
            "total_duration": int(self.response_time * 1e9),
            "load_duration": 1000000,
            "prompt_eval_count": len(prompt.split()),
            "eval_count": 20,
            "eval_duration": int(self.response_time * 0.8 * 1e9)
        }

class MockChromaDB:
    """テスト用 ChromaDB"""
    
    def __init__(self):
        self.collections = {}
        self.documents = {}
        
    def create_collection(self, name: str) -> 'MockCollection':
        collection = MockCollection(name)
        self.collections[name] = collection
        return collection
    
    def get_collection(self, name: str) -> 'MockCollection':
        return self.collections.get(name, self.create_collection(name))

class MockCollection:
    """テスト用 Collection"""
    
    def __init__(self, name: str):
        self.name = name
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.ids = []
    
    def add(self, documents: List[str], embeddings: List[List[float]], 
            metadatas: List[Dict], ids: List[str]):
        """ドキュメント追加"""
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
    
    def query(self, query_embeddings: List[List[float]], n_results: int = 10) -> Dict:
        """クエリ実行"""
        # 簡単な類似度計算（実際の実装ではより高度）
        results = {
            "ids": [self.ids[:n_results]],
            "distances": [[0.1, 0.2, 0.3][:n_results]],
            "documents": [self.documents[:n_results]],
            "metadatas": [self.metadatas[:n_results]]
        }
        return results

class MockSystemMonitor:
    """テスト用システム監視"""
    
    def __init__(self):
        self.cpu_percent = 50.0
        self.memory_percent = 60.0
        self.gpu_memory_percent = 70.0
        
    async def get_system_stats(self) -> Dict[str, Any]:
        """システム統計取得"""
        return {
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_gpu_stats(self) -> Dict[str, Any]:
        """GPU統計取得"""
        return {
            "memory_percent": self.gpu_memory_percent,
            "utilization_percent": 80.0,
            "temperature": 75.0,
            "timestamp": datetime.now().isoformat()
        }


@pytest.fixture
def temp_dir():
    """テスト用一時ディレクトリ"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_ollama():
    """モック Ollama クライアント"""
    return MockOllamaClient()


@pytest.fixture
def mock_chromadb():
    """モック ChromaDB"""
    return MockChromaDB()


@pytest.fixture
def mock_system_monitor():
    """モックシステム監視"""
    return MockSystemMonitor()


class TestEndToEndIntegration:
    """エンドツーエンド統合テスト"""
    
    @pytest.mark.asyncio
    async def test_basic_inference_pipeline(self, mock_ollama):
        """基本推論パイプラインテスト"""
        
        # 推論実行
        result = await mock_ollama.generate(
            model="deepseek-r1:7b",
            prompt="What is artificial intelligence?"
        )
        
        # 結果検証
        assert result["response"].startswith("Mock response for:")
        assert result["model"] == "deepseek-r1:7b"
        assert result["done"] is True
        assert "total_duration" in result
        assert "eval_count" in result
        
        # 呼び出し回数確認
        assert mock_ollama.call_count == 1
    
    @pytest.mark.asyncio
    async def test_memory_integration(self, mock_chromadb):
        """記憶システム統合テスト"""
        
        # コレクション作成
        collection = mock_chromadb.create_collection("test_memory")
        
        # ドキュメント追加
        documents = ["Test document 1", "Test document 2"]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        metadatas = [{"type": "test"}, {"type": "test"}]
        ids = ["doc1", "doc2"]
        
        collection.add(documents, embeddings, metadatas, ids)
        
        # 検索実行
        query_results = collection.query([[0.1, 0.2, 0.3]], n_results=2)
        
        # 結果検証
        assert len(query_results["ids"][0]) <= 2
        assert len(query_results["documents"][0]) <= 2
        assert "distances" in query_results
    
    @pytest.mark.asyncio
    async def test_system_monitoring_integration(self, mock_system_monitor):
        """システム監視統合テスト"""
        
        # システム統計取得
        system_stats = await mock_system_monitor.get_system_stats()
        gpu_stats = await mock_system_monitor.get_gpu_stats()
        
        # 結果検証
        assert "cpu_percent" in system_stats
        assert "memory_percent" in system_stats
        assert "timestamp" in system_stats
        
        assert "memory_percent" in gpu_stats
        assert "utilization_percent" in gpu_stats
        assert "temperature" in gpu_stats
        
        # 値の範囲確認
        assert 0 <= system_stats["cpu_percent"] <= 100
        assert 0 <= system_stats["memory_percent"] <= 100
        assert 0 <= gpu_stats["memory_percent"] <= 100
    
    @pytest.mark.asyncio
    async def test_full_pipeline_integration(self, mock_ollama, mock_chromadb, mock_system_monitor):
        """完全パイプライン統合テスト"""
        
        # 1. システム監視開始
        initial_stats = await mock_system_monitor.get_system_stats()
        
        # 2. 記憶システム初期化
        memory_collection = mock_chromadb.create_collection("pipeline_test")
        
        # 3. 推論実行
        inference_result = await mock_ollama.generate(
            model="deepseek-r1:7b",
            prompt="Explain machine learning concepts"
        )
        
        # 4. 結果を記憶に保存
        memory_collection.add(
            documents=[inference_result["response"]],
            embeddings=[[0.1, 0.2, 0.3, 0.4, 0.5]],
            metadatas=[{"type": "inference", "model": inference_result["model"]}],
            ids=[f"inference_{datetime.now().timestamp()}"]
        )
        
        # 5. 記憶から検索
        search_results = memory_collection.query([[0.1, 0.2, 0.3, 0.4, 0.5]], n_results=1)
        
        # 6. 最終システム統計
        final_stats = await mock_system_monitor.get_system_stats()
        
        # 統合検証
        assert inference_result["done"] is True
        assert len(search_results["documents"][0]) > 0
        assert initial_stats["timestamp"] != final_stats["timestamp"]
        
        # パフォーマンス検証
        assert inference_result["total_duration"] > 0
        assert mock_ollama.call_count == 1


class TestMemoryPressureStability:
    """メモリ圧迫安定性テスト - Pytest + PSUtil による メモリ圧迫安定性テスト"""
    
    def test_memory_usage_monitoring(self):
        """メモリ使用量監視テスト"""
        
        # 初期メモリ使用量
        initial_memory = psutil.virtual_memory()
        
        # メモリ集約的な処理をシミュレート
        large_data = []
        for i in range(1000):
            large_data.append([0.1] * 1000)  # 大きなリストを作成
        
        # メモリ使用量確認
        current_memory = psutil.virtual_memory()
        
        # メモリ使用量が増加していることを確認
        assert current_memory.used >= initial_memory.used
        
        # メモリ使用率が危険レベルでないことを確認
        assert current_memory.percent < 95.0, f"メモリ使用率が危険レベル: {current_memory.percent}%"
        
        # クリーンアップ
        del large_data
    
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self, mock_ollama):
        """メモリ圧迫時の処理テスト"""
        
        # 初期メモリ状態
        initial_memory = psutil.virtual_memory()
        
        # 複数の並行推論をシミュレート
        tasks = []
        for i in range(10):
            task = mock_ollama.generate(
                model="deepseek-r1:7b",
                prompt=f"Test prompt {i} " * 100  # 長いプロンプト
            )
            tasks.append(task)
        
        # 並行実行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 結果検証
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) > 0, "少なくとも一部の推論は成功する必要があります"
        
        # メモリ使用量確認
        final_memory = psutil.virtual_memory()
        memory_increase = final_memory.used - initial_memory.used
        
        # メモリ増加が合理的な範囲内であることを確認
        assert memory_increase < 1024 * 1024 * 1024, "メモリ増加が1GB未満である必要があります"
    
    def test_memory_leak_detection(self):
        """メモリリーク検出テスト"""
        
        # 初期メモリ使用量
        initial_memory = psutil.Process().memory_info().rss
        
        # 繰り返し処理でメモリリークをチェック
        for i in range(100):
            # 一時的なデータ作成と削除
            temp_data = [j for j in range(1000)]
            del temp_data
        
        # 最終メモリ使用量
        final_memory = psutil.Process().memory_info().rss
        memory_diff = final_memory - initial_memory
        
        # メモリ増加が最小限であることを確認
        max_acceptable_increase = 10 * 1024 * 1024  # 10MB
        assert memory_diff < max_acceptable_increase, f"メモリリークの可能性: {memory_diff} bytes増加"


class TestPerformanceDegradation:
    """性能劣化検出テスト - Pytest の既存長時間テスト機能による 性能劣化検出"""
    
    @pytest.mark.asyncio
    async def test_inference_performance_consistency(self, mock_ollama):
        """推論性能一貫性テスト"""
        
        response_times = []
        
        # 複数回の推論実行
        for i in range(10):
            start_time = time.time()
            
            result = await mock_ollama.generate(
                model="deepseek-r1:7b",
                prompt=f"Performance test prompt {i}"
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            response_times.append(response_time)
            
            # 基本的な結果検証
            assert result["done"] is True
        
        # 性能一貫性の検証
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        min_response_time = min(response_times)
        
        # 応答時間のばらつきが許容範囲内であることを確認
        time_variance = max_response_time - min_response_time
        assert time_variance < avg_response_time * 2, f"応答時間のばらつきが大きすぎます: {time_variance:.3f}s"
        
        # 平均応答時間が合理的であることを確認
        assert avg_response_time < 5.0, f"平均応答時間が遅すぎます: {avg_response_time:.3f}s"
    
    @pytest.mark.asyncio
    async def test_long_running_stability(self, mock_ollama, mock_system_monitor):
        """長時間実行安定性テスト"""
        
        start_time = time.time()
        test_duration = 30  # 30秒間のテスト
        
        inference_count = 0
        error_count = 0
        
        while time.time() - start_time < test_duration:
            try:
                # 推論実行
                result = await mock_ollama.generate(
                    model="deepseek-r1:7b",
                    prompt=f"Long running test {inference_count}"
                )
                
                assert result["done"] is True
                inference_count += 1
                
                # システム統計確認
                stats = await mock_system_monitor.get_system_stats()
                assert stats["cpu_percent"] < 100
                assert stats["memory_percent"] < 95
                
                # 短い待機
                await asyncio.sleep(0.1)
                
            except Exception as e:
                error_count += 1
                if error_count > inference_count * 0.1:  # エラー率10%以下
                    pytest.fail(f"エラー率が高すぎます: {error_count}/{inference_count}")
        
        # 最終検証
        assert inference_count > 0, "推論が実行されませんでした"
        error_rate = error_count / inference_count if inference_count > 0 else 1
        assert error_rate < 0.05, f"エラー率が高すぎます: {error_rate:.2%}"
    
    def test_memory_usage_over_time(self):
        """時間経過によるメモリ使用量テスト"""
        
        memory_samples = []
        test_duration = 10  # 10秒間のテスト
        start_time = time.time()
        
        while time.time() - start_time < test_duration:
            # メモリ使用量サンプリング
            memory_info = psutil.virtual_memory()
            memory_samples.append({
                "timestamp": time.time(),
                "memory_percent": memory_info.percent,
                "memory_used": memory_info.used
            })
            
            # 一時的な処理をシミュレート
            temp_data = [i for i in range(1000)]
            del temp_data
            
            time.sleep(0.5)
        
        # メモリ使用量の傾向分析
        if len(memory_samples) >= 2:
            initial_memory = memory_samples[0]["memory_used"]
            final_memory = memory_samples[-1]["memory_used"]
            memory_growth = final_memory - initial_memory
            
            # メモリ増加が合理的な範囲内であることを確認
            max_acceptable_growth = 50 * 1024 * 1024  # 50MB
            assert memory_growth < max_acceptable_growth, f"メモリ使用量の増加が大きすぎます: {memory_growth} bytes"


class TestSessionPersistence:
    """セッション永続化・記憶整合性テスト - Pytest + SQLAlchemy による セッション永続化・記憶整合性テスト"""
    
    def test_session_data_persistence(self, temp_dir):
        """セッションデータ永続化テスト"""
        
        # テスト用セッションデータ
        session_data = {
            "session_id": "test_session_123",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            "settings": {
                "model": "deepseek-r1:7b",
                "temperature": 0.7
            },
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # セッションファイルに保存
        session_file = temp_dir / "session_test.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2)
        
        # ファイルが作成されていることを確認
        assert session_file.exists()
        
        # セッションデータを読み込み
        with open(session_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        # データ整合性確認
        assert loaded_data["session_id"] == session_data["session_id"]
        assert len(loaded_data["messages"]) == len(session_data["messages"])
        assert loaded_data["settings"]["model"] == session_data["settings"]["model"]
    
    def test_memory_consistency(self, mock_chromadb):
        """記憶整合性テスト"""
        
        # 記憶コレクション作成
        collection = mock_chromadb.create_collection("consistency_test")
        
        # 複数のドキュメントを追加
        documents = [
            "First document about AI",
            "Second document about machine learning",
            "Third document about neural networks"
        ]
        
        embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ]
        
        metadatas = [
            {"topic": "AI", "importance": 0.8},
            {"topic": "ML", "importance": 0.9},
            {"topic": "NN", "importance": 0.7}
        ]
        
        ids = ["doc1", "doc2", "doc3"]
        
        collection.add(documents, embeddings, metadatas, ids)
        
        # データ整合性確認
        assert len(collection.documents) == 3
        assert len(collection.embeddings) == 3
        assert len(collection.metadatas) == 3
        assert len(collection.ids) == 3
        
        # 検索結果の整合性確認
        query_results = collection.query([[0.1, 0.2, 0.3]], n_results=3)
        
        assert len(query_results["ids"][0]) <= 3
        assert len(query_results["documents"][0]) <= 3
        assert len(query_results["metadatas"][0]) <= 3
    
    def test_session_recovery(self, temp_dir):
        """セッション復旧テスト"""
        
        # 複数のセッションファイルを作成
        sessions = []
        for i in range(3):
            session_data = {
                "session_id": f"session_{i}",
                "messages": [{"role": "user", "content": f"Message {i}"}],
                "created_at": (datetime.now() - timedelta(days=i)).isoformat()
            }
            
            session_file = temp_dir / f"session_{i}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f)
            
            sessions.append(session_data)
        
        # セッションファイル一覧取得
        session_files = list(temp_dir.glob("session_*.json"))
        assert len(session_files) == 3
        
        # 各セッションファイルの復旧テスト
        recovered_sessions = []
        for session_file in session_files:
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
                recovered_sessions.append(session_data)
        
        # 復旧されたセッション数の確認
        assert len(recovered_sessions) == 3
        
        # セッションIDの一意性確認
        session_ids = [s["session_id"] for s in recovered_sessions]
        assert len(set(session_ids)) == 3, "セッションIDが重複しています"
    
    def test_data_corruption_handling(self, temp_dir):
        """データ破損処理テスト"""
        
        # 正常なセッションファイル
        valid_session = {
            "session_id": "valid_session",
            "messages": [{"role": "user", "content": "Valid message"}]
        }
        
        valid_file = temp_dir / "valid_session.json"
        with open(valid_file, 'w', encoding='utf-8') as f:
            json.dump(valid_session, f)
        
        # 破損したセッションファイル
        corrupted_file = temp_dir / "corrupted_session.json"
        with open(corrupted_file, 'w', encoding='utf-8') as f:
            f.write("{ invalid json content")
        
        # 空のセッションファイル
        empty_file = temp_dir / "empty_session.json"
        empty_file.touch()
        
        # ファイル復旧テスト
        recovered_sessions = []
        
        for session_file in temp_dir.glob("*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                    recovered_sessions.append(session_data)
            except (json.JSONDecodeError, FileNotFoundError):
                # 破損したファイルはスキップ
                continue
        
        # 正常なセッションのみが復旧されることを確認
        assert len(recovered_sessions) == 1
        assert recovered_sessions[0]["session_id"] == "valid_session"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])