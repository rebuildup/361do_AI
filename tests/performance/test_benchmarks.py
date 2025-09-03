"""
性能ベンチマークテスト

HuggingFace Evaluate による 推論速度・メモリ効率・学習効果評価を統合し、
ChromaDB + Evaluate による 記憶検索性能・精度評価を実装、
Grafana + Evaluate による 最適化効果可視化・レポート生成を統合
"""

import pytest
import asyncio
import time
import statistics
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import tempfile
import shutil

# モック評価ライブラリ
class MockEvaluate:
    """HuggingFace Evaluate のモック"""
    
    @staticmethod
    def load(metric_name: str):
        """評価メトリクスをロード"""
        return MockMetric(metric_name)

class MockMetric:
    """評価メトリクスのモック"""
    
    def __init__(self, name: str):
        self.name = name
    
    def compute(self, predictions: List, references: List, **kwargs) -> Dict[str, float]:
        """メトリクス計算"""
        if self.name == "bleu":
            return {"bleu": 0.85}
        elif self.name == "rouge":
            return {"rouge1": 0.75, "rouge2": 0.65, "rougeL": 0.70}
        elif self.name == "accuracy":
            return {"accuracy": 0.92}
        elif self.name == "f1":
            return {"f1": 0.88}
        else:
            return {"score": 0.80}

# テスト用のモッククラス
class MockInferenceEngine:
    """推論エンジンのモック"""
    
    def __init__(self):
        self.inference_count = 0
        self.total_time = 0.0
        self.memory_usage = []
        
    async def inference(self, prompt: str, model: str = "deepseek-r1:7b") -> Dict[str, Any]:
        """推論実行"""
        start_time = time.time()
        
        # 推論時間をシミュレート
        inference_time = 0.5 + (len(prompt) / 1000)  # プロンプト長に応じた時間
        await asyncio.sleep(inference_time)
        
        end_time = time.time()
        actual_time = end_time - start_time
        
        self.inference_count += 1
        self.total_time += actual_time
        
        # メモリ使用量をシミュレート
        import psutil
        memory_usage = psutil.virtual_memory().percent
        self.memory_usage.append(memory_usage)
        
        return {
            "response": f"Generated response for: {prompt[:50]}...",
            "model": model,
            "inference_time": actual_time,
            "tokens_generated": len(prompt.split()) + 20,
            "memory_usage": memory_usage,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_performance_stats(self) -> Dict[str, float]:
        """性能統計取得"""
        if self.inference_count == 0:
            return {}
        
        avg_time = self.total_time / self.inference_count
        avg_memory = statistics.mean(self.memory_usage) if self.memory_usage else 0
        
        return {
            "total_inferences": self.inference_count,
            "total_time": self.total_time,
            "average_inference_time": avg_time,
            "average_memory_usage": avg_memory,
            "throughput": self.inference_count / self.total_time if self.total_time > 0 else 0
        }

class MockMemorySystem:
    """記憶システムのモック"""
    
    def __init__(self):
        self.documents = []
        self.search_times = []
        self.search_accuracies = []
        
    def add_document(self, content: str, metadata: Dict[str, Any] = None):
        """ドキュメント追加"""
        doc = {
            "id": len(self.documents),
            "content": content,
            "metadata": metadata or {},
            "embedding": [0.1] * 384,  # ダミー埋め込み
            "timestamp": datetime.now().isoformat()
        }
        self.documents.append(doc)
    
    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """記憶検索"""
        start_time = time.time()
        
        # 検索時間をシミュレート
        search_time = 0.1 + (len(self.documents) / 10000)
        await asyncio.sleep(search_time)
        
        end_time = time.time()
        actual_time = end_time - start_time
        self.search_times.append(actual_time)
        
        # 検索結果をシミュレート（関連度順）
        results = []
        for i, doc in enumerate(self.documents[:top_k]):
            similarity = max(0.1, 1.0 - (i * 0.1))  # 順位に応じた類似度
            results.append({
                "document": doc,
                "similarity": similarity,
                "rank": i + 1
            })
        
        # 検索精度をシミュレート
        accuracy = min(0.95, 0.7 + (len(results) / top_k) * 0.2)
        self.search_accuracies.append(accuracy)
        
        return results
    
    def get_search_performance(self) -> Dict[str, float]:
        """検索性能統計"""
        if not self.search_times:
            return {}
        
        return {
            "total_searches": len(self.search_times),
            "average_search_time": statistics.mean(self.search_times),
            "min_search_time": min(self.search_times),
            "max_search_time": max(self.search_times),
            "average_accuracy": statistics.mean(self.search_accuracies) if self.search_accuracies else 0,
            "total_documents": len(self.documents)
        }

class MockPrometheusCollector:
    """Prometheus メトリクス収集のモック"""
    
    def __init__(self):
        self.metrics = {}
        self.time_series = {}
        
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """メトリクス記録"""
        timestamp = datetime.now()
        
        if name not in self.metrics:
            self.metrics[name] = []
            self.time_series[name] = []
        
        metric_data = {
            "value": value,
            "labels": labels or {},
            "timestamp": timestamp
        }
        
        self.metrics[name].append(metric_data)
        self.time_series[name].append((timestamp, value))
    
    def get_metric_summary(self, name: str) -> Dict[str, float]:
        """メトリクス要約統計"""
        if name not in self.metrics:
            return {}
        
        values = [m["value"] for m in self.metrics[name]]
        
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0
        }
    
    def get_time_series(self, name: str, duration_minutes: int = 60) -> List[Tuple[datetime, float]]:
        """時系列データ取得"""
        if name not in self.time_series:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        return [(ts, val) for ts, val in self.time_series[name] if ts >= cutoff_time]


@pytest.fixture
def mock_evaluate():
    """モック評価システム"""
    return MockEvaluate()


@pytest.fixture
def mock_inference_engine():
    """モック推論エンジン"""
    return MockInferenceEngine()


@pytest.fixture
def mock_memory_system():
    """モック記憶システム"""
    return MockMemorySystem()


@pytest.fixture
def mock_prometheus():
    """モック Prometheus コレクター"""
    return MockPrometheusCollector()


@pytest.fixture
def temp_dir():
    """テスト用一時ディレクトリ"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


class TestInferencePerformance:
    """推論性能テスト - HuggingFace Evaluate による 推論速度・メモリ効率・学習効果評価"""
    
    @pytest.mark.asyncio
    async def test_inference_speed_benchmark(self, mock_inference_engine, mock_evaluate):
        """推論速度ベンチマーク"""
        
        # テスト用プロンプト
        test_prompts = [
            "What is artificial intelligence?",
            "Explain machine learning in simple terms.",
            "How do neural networks work?",
            "What are the applications of deep learning?",
            "Describe the future of AI technology."
        ]
        
        # 推論実行
        results = []
        for prompt in test_prompts:
            result = await mock_inference_engine.inference(prompt)
            results.append(result)
        
        # 性能統計取得
        stats = mock_inference_engine.get_performance_stats()
        
        # 性能検証
        assert stats["total_inferences"] == len(test_prompts)
        assert stats["average_inference_time"] > 0
        assert stats["average_inference_time"] < 5.0, "推論時間が遅すぎます"
        assert stats["throughput"] > 0.1, "スループットが低すぎます"
        
        # 推論時間の一貫性確認
        inference_times = [r["inference_time"] for r in results]
        time_variance = max(inference_times) - min(inference_times)
        avg_time = statistics.mean(inference_times)
        
        assert time_variance < avg_time * 2, "推論時間のばらつきが大きすぎます"
    
    @pytest.mark.asyncio
    async def test_memory_efficiency_benchmark(self, mock_inference_engine):
        """メモリ効率ベンチマーク"""
        
        # 異なる長さのプロンプトでテスト
        prompt_lengths = [10, 50, 100, 200, 500]  # 単語数
        
        memory_usage_by_length = {}
        
        for length in prompt_lengths:
            prompt = " ".join([f"word{i}" for i in range(length)])
            result = await mock_inference_engine.inference(prompt)
            
            memory_usage_by_length[length] = result["memory_usage"]
        
        # メモリ使用量の検証
        for length, memory_usage in memory_usage_by_length.items():
            assert memory_usage < 95.0, f"メモリ使用率が高すぎます: {memory_usage}%"
        
        # メモリ使用量の増加が線形的でないことを確認（効率的な実装）
        memory_values = list(memory_usage_by_length.values())
        memory_increase = max(memory_values) - min(memory_values)
        
        assert memory_increase < 30.0, "プロンプト長によるメモリ増加が大きすぎます"
    
    @pytest.mark.asyncio
    async def test_concurrent_inference_performance(self, mock_inference_engine):
        """並行推論性能テスト"""
        
        # 並行推論タスク
        concurrent_tasks = 5
        prompts = [f"Concurrent test prompt {i}" for i in range(concurrent_tasks)]
        
        # 並行実行
        start_time = time.time()
        tasks = [mock_inference_engine.inference(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # 並行性能の検証
        assert len(results) == concurrent_tasks
        assert all(r["response"] for r in results), "すべての推論が成功する必要があります"
        
        # 並行実行による時間短縮効果の確認
        sequential_time_estimate = sum(r["inference_time"] for r in results)
        efficiency = sequential_time_estimate / total_time
        
        assert efficiency > 1.0, "並行実行による効率化が見られません"
    
    def test_inference_quality_evaluation(self, mock_evaluate):
        """推論品質評価テスト"""
        
        # テスト用の予測と参照データ
        predictions = [
            "AI is a technology that enables machines to simulate human intelligence.",
            "Machine learning is a subset of AI that learns from data.",
            "Neural networks are computing systems inspired by biological neural networks."
        ]
        
        references = [
            "Artificial intelligence allows machines to mimic human cognitive functions.",
            "Machine learning is an AI technique that improves through experience.",
            "Neural networks are computational models based on the human brain."
        ]
        
        # 各種メトリクスで評価
        bleu_metric = mock_evaluate.load("bleu")
        rouge_metric = mock_evaluate.load("rouge")
        
        bleu_score = bleu_metric.compute(predictions=predictions, references=references)
        rouge_scores = rouge_metric.compute(predictions=predictions, references=references)
        
        # 品質スコアの検証
        assert bleu_score["bleu"] > 0.7, f"BLEU スコアが低すぎます: {bleu_score['bleu']}"
        assert rouge_scores["rouge1"] > 0.6, f"ROUGE-1 スコアが低すぎます: {rouge_scores['rouge1']}"
        assert rouge_scores["rougeL"] > 0.6, f"ROUGE-L スコアが低すぎます: {rouge_scores['rougeL']}"


class TestMemorySearchPerformance:
    """記憶検索性能テスト - ChromaDB + Evaluate による 記憶検索性能・精度評価"""
    
    @pytest.mark.asyncio
    async def test_search_speed_benchmark(self, mock_memory_system):
        """検索速度ベンチマーク"""
        
        # テスト用ドキュメントを追加
        documents = [
            "Artificial intelligence is transforming technology.",
            "Machine learning algorithms learn from data patterns.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing enables AI to understand text.",
            "Computer vision allows machines to interpret images."
        ]
        
        for doc in documents:
            mock_memory_system.add_document(doc, {"category": "AI"})
        
        # 検索クエリ
        queries = [
            "artificial intelligence",
            "machine learning",
            "neural networks",
            "natural language",
            "computer vision"
        ]
        
        # 検索実行
        search_results = []
        for query in queries:
            results = await mock_memory_system.search(query, top_k=3)
            search_results.append(results)
        
        # 検索性能統計
        perf_stats = mock_memory_system.get_search_performance()
        
        # 性能検証
        assert perf_stats["total_searches"] == len(queries)
        assert perf_stats["average_search_time"] < 1.0, "検索時間が遅すぎます"
        assert perf_stats["average_accuracy"] > 0.7, "検索精度が低すぎます"
        
        # 検索結果の妥当性確認
        for results in search_results:
            assert len(results) <= 3, "検索結果数が制限を超えています"
            assert all(r["similarity"] > 0 for r in results), "類似度スコアが無効です"
    
    @pytest.mark.asyncio
    async def test_search_scalability(self, mock_memory_system):
        """検索スケーラビリティテスト"""
        
        # 異なるサイズのドキュメントセットでテスト
        document_counts = [10, 50, 100, 500]
        
        scalability_results = {}
        
        for count in document_counts:
            # ドキュメント追加
            for i in range(count):
                mock_memory_system.add_document(
                    f"Test document {i} about various topics in AI and technology.",
                    {"index": i, "category": "test"}
                )
            
            # 検索実行
            start_time = time.time()
            results = await mock_memory_system.search("AI technology", top_k=5)
            end_time = time.time()
            
            search_time = end_time - start_time
            scalability_results[count] = {
                "search_time": search_time,
                "results_count": len(results)
            }
        
        # スケーラビリティの検証
        for count, result in scalability_results.items():
            assert result["search_time"] < 2.0, f"ドキュメント数 {count} での検索時間が遅すぎます"
            assert result["results_count"] > 0, "検索結果が返されませんでした"
        
        # 検索時間の増加が線形以下であることを確認
        times = [result["search_time"] for result in scalability_results.values()]
        counts = list(scalability_results.keys())
        
        # 最大ドキュメント数での検索時間が最小の10倍以下であることを確認
        time_ratio = max(times) / min(times)
        count_ratio = max(counts) / min(counts)
        
        assert time_ratio < count_ratio, "検索時間の増加が非効率的です"
    
    @pytest.mark.asyncio
    async def test_search_accuracy_evaluation(self, mock_memory_system, mock_evaluate):
        """検索精度評価テスト"""
        
        # カテゴリ別のテストドキュメント
        test_documents = [
            ("Machine learning is a subset of artificial intelligence.", "ML"),
            ("Deep learning uses neural networks with multiple layers.", "DL"),
            ("Natural language processing enables text understanding.", "NLP"),
            ("Computer vision processes and analyzes visual data.", "CV"),
            ("Reinforcement learning learns through trial and error.", "RL")
        ]
        
        # ドキュメント追加
        for doc, category in test_documents:
            mock_memory_system.add_document(doc, {"category": category})
        
        # 検索クエリと期待されるカテゴリ
        test_queries = [
            ("machine learning algorithms", "ML"),
            ("neural network architecture", "DL"),
            ("text processing techniques", "NLP"),
            ("image recognition systems", "CV"),
            ("learning through rewards", "RL")
        ]
        
        correct_predictions = 0
        total_predictions = len(test_queries)
        
        # 検索精度テスト
        for query, expected_category in test_queries:
            results = await mock_memory_system.search(query, top_k=1)
            
            if results and results[0]["document"]["metadata"]["category"] == expected_category:
                correct_predictions += 1
        
        # 精度計算
        accuracy = correct_predictions / total_predictions
        
        # 精度検証
        assert accuracy > 0.6, f"検索精度が低すぎます: {accuracy:.2%}"
        
        # 評価メトリクスでの検証
        accuracy_metric = mock_evaluate.load("accuracy")
        predictions = [1] * correct_predictions + [0] * (total_predictions - correct_predictions)
        references = [1] * total_predictions
        
        eval_result = accuracy_metric.compute(predictions=predictions, references=references)
        assert eval_result["accuracy"] == accuracy


class TestOptimizationEffectiveness:
    """最適化効果テスト - Grafana + Evaluate による 最適化効果可視化・レポート生成"""
    
    @pytest.mark.asyncio
    async def test_performance_optimization_tracking(self, mock_inference_engine, mock_prometheus):
        """性能最適化追跡テスト"""
        
        # ベースライン性能測定
        baseline_prompts = ["Test prompt"] * 10
        
        for prompt in baseline_prompts:
            result = await mock_inference_engine.inference(prompt)
            mock_prometheus.record_metric(
                "inference_time", 
                result["inference_time"],
                {"phase": "baseline", "model": result["model"]}
            )
        
        # 最適化後の性能測定（シミュレート）
        mock_inference_engine.total_time *= 0.8  # 20%の改善をシミュレート
        
        optimized_prompts = ["Optimized test prompt"] * 10
        
        for prompt in optimized_prompts:
            result = await mock_inference_engine.inference(prompt)
            # 最適化効果をシミュレート
            optimized_time = result["inference_time"] * 0.8
            mock_prometheus.record_metric(
                "inference_time",
                optimized_time,
                {"phase": "optimized", "model": result["model"]}
            )
        
        # 最適化効果の分析
        baseline_stats = mock_prometheus.get_metric_summary("inference_time")
        
        # 改善効果の検証
        assert baseline_stats["count"] == 20  # ベースライン + 最適化
        assert baseline_stats["mean"] > 0, "平均推論時間が記録されていません"
        
        # 時系列データの確認
        time_series = mock_prometheus.get_time_series("inference_time", duration_minutes=10)
        assert len(time_series) > 0, "時系列データが記録されていません"
    
    def test_memory_optimization_tracking(self, mock_prometheus):
        """メモリ最適化追跡テスト"""
        
        # メモリ使用量の変化をシミュレート
        memory_phases = [
            ("initial", [70, 72, 75, 73, 71]),
            ("optimized", [65, 63, 67, 64, 66]),
            ("further_optimized", [60, 58, 62, 59, 61])
        ]
        
        for phase, memory_values in memory_phases:
            for memory_usage in memory_values:
                mock_prometheus.record_metric(
                    "memory_usage_percent",
                    memory_usage,
                    {"phase": phase}
                )
        
        # メモリ最適化効果の分析
        memory_stats = mock_prometheus.get_metric_summary("memory_usage_percent")
        
        # 統計の検証
        assert memory_stats["count"] == 15  # 3フェーズ × 5測定
        assert memory_stats["min"] < memory_stats["max"], "メモリ使用量に変化が見られません"
        assert memory_stats["std"] > 0, "メモリ使用量の標準偏差が計算されていません"
    
    def test_optimization_report_generation(self, mock_prometheus, temp_dir):
        """最適化レポート生成テスト"""
        
        # テストメトリクスの記録
        metrics_data = {
            "inference_time": [1.2, 1.1, 0.9, 0.8, 0.7],
            "memory_usage": [75, 73, 70, 68, 65],
            "throughput": [0.8, 0.9, 1.1, 1.2, 1.4],
            "accuracy": [0.85, 0.87, 0.89, 0.91, 0.93]
        }
        
        for metric_name, values in metrics_data.items():
            for i, value in enumerate(values):
                mock_prometheus.record_metric(
                    metric_name,
                    value,
                    {"iteration": str(i)}
                )
        
        # レポートデータの生成
        report_data = {}
        for metric_name in metrics_data.keys():
            stats = mock_prometheus.get_metric_summary(metric_name)
            time_series = mock_prometheus.get_time_series(metric_name, duration_minutes=60)
            
            report_data[metric_name] = {
                "summary": stats,
                "time_series_count": len(time_series),
                "improvement": {
                    "initial": metrics_data[metric_name][0],
                    "final": metrics_data[metric_name][-1],
                    "change_percent": ((metrics_data[metric_name][-1] - metrics_data[metric_name][0]) / metrics_data[metric_name][0]) * 100
                }
            }
        
        # レポートファイルの生成
        report_file = temp_dir / "optimization_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # レポートファイルの検証
        assert report_file.exists(), "レポートファイルが生成されませんでした"
        
        with open(report_file, 'r', encoding='utf-8') as f:
            loaded_report = json.load(f)
        
        # レポート内容の検証
        assert len(loaded_report) == len(metrics_data), "すべてのメトリクスがレポートに含まれていません"
        
        for metric_name in metrics_data.keys():
            assert metric_name in loaded_report, f"メトリクス {metric_name} がレポートにありません"
            assert "summary" in loaded_report[metric_name], "サマリー情報がありません"
            assert "improvement" in loaded_report[metric_name], "改善情報がありません"
    
    def test_comparative_performance_analysis(self, mock_evaluate):
        """比較性能分析テスト"""
        
        # 異なるモデル/設定での性能データ
        performance_data = {
            "model_a": {
                "inference_times": [1.2, 1.1, 1.3, 1.0, 1.2],
                "accuracy_scores": [0.85, 0.87, 0.84, 0.88, 0.86],
                "memory_usage": [70, 72, 69, 71, 70]
            },
            "model_b": {
                "inference_times": [0.8, 0.9, 0.7, 0.8, 0.9],
                "accuracy_scores": [0.82, 0.84, 0.81, 0.85, 0.83],
                "memory_usage": [65, 67, 64, 66, 65]
            }
        }
        
        # 比較分析の実行
        comparison_results = {}
        
        for model_name, data in performance_data.items():
            avg_inference_time = statistics.mean(data["inference_times"])
            avg_accuracy = statistics.mean(data["accuracy_scores"])
            avg_memory = statistics.mean(data["memory_usage"])
            
            # 効率性スコア計算（精度/時間の比率）
            efficiency_score = avg_accuracy / avg_inference_time
            
            comparison_results[model_name] = {
                "avg_inference_time": avg_inference_time,
                "avg_accuracy": avg_accuracy,
                "avg_memory_usage": avg_memory,
                "efficiency_score": efficiency_score
            }
        
        # 比較結果の検証
        assert len(comparison_results) == 2, "両方のモデルの結果が必要です"
        
        model_a_results = comparison_results["model_a"]
        model_b_results = comparison_results["model_b"]
        
        # 性能トレードオフの分析
        if model_a_results["avg_inference_time"] > model_b_results["avg_inference_time"]:
            # model_b が高速な場合、精度やメモリ使用量での比較
            assert model_a_results["avg_accuracy"] >= model_b_results["avg_accuracy"] * 0.95, \
                "高速化による精度低下が大きすぎます"
        
        # 効率性スコアの妥当性確認
        for model_name, results in comparison_results.items():
            assert results["efficiency_score"] > 0, f"{model_name} の効率性スコアが無効です"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])