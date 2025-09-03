"""
Adapter Evaluator のテスト
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
import numpy as np

from src.advanced_agent.adaptation.adapter_evaluator import (
    AdapterEvaluator, EvaluationConfig, EvaluationResult, ComparisonResult,
    EvaluationTask, EvaluationMetric, create_adapter_evaluator
)
from src.advanced_agent.adaptation.peft_manager import PEFTAdapterPool, AdapterInfo, AdapterConfig, AdapterType, AdapterStatus
from src.advanced_agent.monitoring.system_monitor import SystemMonitor


class TestAdapterEvaluator:
    """AdapterEvaluator クラスのテスト"""
    
    @pytest.fixture
    def mock_peft_pool(self):
        """モックPEFTプール"""
        pool = Mock(spec=PEFTAdapterPool)
        pool.adapters = {}
        pool.active_adapters = {}
        pool.tokenizer = Mock()
        pool.tokenizer.eos_token_id = 50256
        pool.tokenizer.decode = Mock(return_value="Generated text")
        
        # モックアダプタ情報
        adapter_info = AdapterInfo(
            config=AdapterConfig(name="test_adapter", adapter_type=AdapterType.LORA),
            status=AdapterStatus.INACTIVE
        )
        pool.adapters["test_adapter"] = adapter_info
        pool.get_adapter_info = Mock(return_value=adapter_info)
        pool.load_adapter = AsyncMock()
        
        return pool
    
    @pytest.fixture
    def mock_system_monitor(self):
        """モックシステムモニター"""
        return Mock(spec=SystemMonitor)
    
    @pytest.fixture
    def adapter_evaluator(self, mock_peft_pool, mock_system_monitor):
        """AdapterEvaluator インスタンス"""
        with patch('src.advanced_agent.adaptation.adapter_evaluator.get_config') as mock_get_config, \
             patch('src.advanced_agent.adaptation.adapter_evaluator.EVALUATE_AVAILABLE', True):
            from src.advanced_agent.core.config import AdvancedAgentConfig
            mock_get_config.return_value = AdvancedAgentConfig()
            
            return AdapterEvaluator(mock_peft_pool, mock_system_monitor)
    
    def test_init(self, adapter_evaluator, mock_peft_pool, mock_system_monitor):
        """初期化テスト"""
        assert adapter_evaluator.peft_pool == mock_peft_pool
        assert adapter_evaluator.system_monitor == mock_system_monitor
        assert len(adapter_evaluator.evaluators) == 0
        assert len(adapter_evaluator.evaluation_history) == 0
        assert len(adapter_evaluator.comparison_history) == 0
        assert adapter_evaluator.evaluation_stats["total_evaluations"] == 0
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, adapter_evaluator):
        """初期化成功テスト"""
        with patch.object(adapter_evaluator, '_preload_evaluators') as mock_preload:
            mock_preload.return_value = None
            
            result = await adapter_evaluator.initialize()
            
            assert result is True
            mock_preload.assert_called_once()
    
    @patch('src.advanced_agent.adaptation.adapter_evaluator.EVALUATE_AVAILABLE', False)
    @pytest.mark.asyncio
    async def test_initialize_unavailable(self, adapter_evaluator):
        """Evaluate利用不可時の初期化テスト"""
        result = await adapter_evaluator.initialize()
        
        assert result is False
    
    @patch('src.advanced_agent.adaptation.adapter_evaluator.evaluate')
    @pytest.mark.asyncio
    async def test_preload_evaluators(self, mock_evaluate, adapter_evaluator):
        """評価器事前ロードテスト"""
        # モック評価器
        mock_evaluator = Mock()
        mock_evaluate.load = AsyncMock(return_value=mock_evaluator)
        
        await adapter_evaluator._preload_evaluators()
        
        # 基本メトリクスがロードされることを確認
        expected_metrics = ["accuracy", "f1", "bleu", "rouge"]
        assert len(adapter_evaluator.evaluators) <= len(expected_metrics)
    
    @patch('src.advanced_agent.adaptation.adapter_evaluator.evaluate')
    def test_get_evaluator_cached(self, mock_evaluate, adapter_evaluator):
        """キャッシュされた評価器取得テスト"""
        mock_evaluator = Mock()
        adapter_evaluator.evaluators["accuracy"] = mock_evaluator
        
        result = adapter_evaluator.get_evaluator("accuracy")
        
        assert result == mock_evaluator
        mock_evaluate.load.assert_not_called()
    
    @patch('src.advanced_agent.adaptation.adapter_evaluator.evaluate')
    def test_get_evaluator_new(self, mock_evaluate, adapter_evaluator):
        """新しい評価器取得テスト"""
        mock_evaluator = Mock()
        mock_evaluate.load.return_value = mock_evaluator
        
        result = adapter_evaluator.get_evaluator("new_metric")
        
        assert result == mock_evaluator
        assert adapter_evaluator.evaluators["new_metric"] == mock_evaluator
        mock_evaluate.load.assert_called_once_with("new_metric")
    
    @patch('src.advanced_agent.adaptation.adapter_evaluator.HFDataset')
    def test_create_default_dataset_perplexity(self, mock_dataset, adapter_evaluator):
        """パープレキシティ用デフォルトデータセット作成テスト"""
        mock_dataset_instance = Mock()
        mock_dataset.from_dict.return_value = mock_dataset_instance
        
        result = adapter_evaluator._create_default_dataset(EvaluationTask.PERPLEXITY)
        
        assert result == mock_dataset_instance
        mock_dataset.from_dict.assert_called_once()
        
        # 呼び出し引数確認
        call_args = mock_dataset.from_dict.call_args[0][0]
        assert "text" in call_args
        assert len(call_args["text"]) > 0
    
    @patch('src.advanced_agent.adaptation.adapter_evaluator.HFDataset')
    def test_create_default_dataset_bleu(self, mock_dataset, adapter_evaluator):
        """BLEU用デフォルトデータセット作成テスト"""
        mock_dataset_instance = Mock()
        mock_dataset.from_dict.return_value = mock_dataset_instance
        
        result = adapter_evaluator._create_default_dataset(EvaluationTask.BLEU)
        
        assert result == mock_dataset_instance
        
        # 呼び出し引数確認
        call_args = mock_dataset.from_dict.call_args[0][0]
        assert "predictions" in call_args
        assert "references" in call_args
    
    def test_is_float(self, adapter_evaluator):
        """浮動小数点数判定テスト"""
        assert adapter_evaluator._is_float("3.14") is True
        assert adapter_evaluator._is_float("42") is True
        assert adapter_evaluator._is_float("-1.5") is True
        assert adapter_evaluator._is_float("not_a_number") is False
        assert adapter_evaluator._is_float("") is False
    
    @patch('src.advanced_agent.adaptation.adapter_evaluator.torch')
    def test_get_peak_memory_cuda(self, mock_torch, adapter_evaluator):
        """CUDA環境でのピークメモリ取得テスト"""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.max_memory_allocated.return_value = 2048 * 1024 * 1024  # 2GB
        
        memory_mb = adapter_evaluator._get_peak_memory()
        
        assert memory_mb == 2048.0
    
    @patch('src.advanced_agent.adaptation.adapter_evaluator.torch')
    def test_get_peak_memory_cpu(self, mock_torch, adapter_evaluator):
        """CPU環境でのピークメモリ取得テスト"""
        mock_torch.cuda.is_available.return_value = False
        
        memory_mb = adapter_evaluator._get_peak_memory()
        
        assert memory_mb == 0.0
    
    @pytest.mark.asyncio
    async def test_calculate_metrics_accuracy(self, adapter_evaluator):
        """精度メトリクス計算テスト"""
        # モック評価器設定
        mock_evaluator = Mock()
        mock_evaluator.compute.return_value = {"accuracy": 0.85}
        adapter_evaluator.evaluators["accuracy"] = mock_evaluator
        
        predictions = ["1", "0", "1", "1"]
        references = ["1", "0", "0", "1"]
        metrics = [EvaluationMetric.ACCURACY]
        
        result = await adapter_evaluator._calculate_metrics(predictions, references, metrics)
        
        assert "accuracy" in result
        assert result["accuracy"] == 0.85
        mock_evaluator.compute.assert_called_once_with(
            predictions=[1, 0, 1, 1],
            references=[1, 0, 0, 1]
        )
    
    @pytest.mark.asyncio
    async def test_calculate_metrics_f1(self, adapter_evaluator):
        """F1スコアメトリクス計算テスト"""
        mock_evaluator = Mock()
        mock_evaluator.compute.return_value = {"f1": 0.75}
        adapter_evaluator.evaluators["f1"] = mock_evaluator
        
        predictions = ["1", "0", "1"]
        references = ["1", "1", "1"]
        metrics = [EvaluationMetric.F1_SCORE]
        
        result = await adapter_evaluator._calculate_metrics(predictions, references, metrics)
        
        assert "f1" in result
        assert result["f1"] == 0.75
    
    @pytest.mark.asyncio
    async def test_calculate_metrics_bleu(self, adapter_evaluator):
        """BLEUスコアメトリクス計算テスト"""
        mock_evaluator = Mock()
        mock_evaluator.compute.return_value = {"bleu": 0.65}
        adapter_evaluator.evaluators["bleu"] = mock_evaluator
        
        predictions = ["The cat is on the mat", "Hello world"]
        references = ["The cat sits on the mat", "Hello world"]
        metrics = [EvaluationMetric.BLEU_SCORE]
        
        result = await adapter_evaluator._calculate_metrics(predictions, references, metrics)
        
        assert "bleu" in result
        assert result["bleu"] == 0.65
    
    @pytest.mark.asyncio
    async def test_calculate_metrics_score(self, adapter_evaluator):
        """スコアメトリクス計算テスト"""
        predictions = ["2.5", "3.0", "2.8"]
        references = ["text1", "text2", "text3"]
        metrics = [EvaluationMetric.SCORE]
        
        result = await adapter_evaluator._calculate_metrics(predictions, references, metrics)
        
        assert "score" in result
        assert "perplexity" in result
        expected_score = np.mean([2.5, 3.0, 2.8])
        assert abs(result["score"] - expected_score) < 0.001
    
    def test_analyze_comparison(self, adapter_evaluator):
        """比較分析テスト"""
        # ベースライン結果
        baseline_result = EvaluationResult(
            adapter_name="baseline",
            task=EvaluationTask.ACCURACY,
            config=EvaluationConfig(task=EvaluationTask.ACCURACY, metrics=[]),
            metrics={"accuracy": 0.8, "f1": 0.75}
        )
        
        # 比較結果
        comparison_result = EvaluationResult(
            adapter_name="improved",
            task=EvaluationTask.ACCURACY,
            config=EvaluationConfig(task=EvaluationTask.ACCURACY, metrics=[]),
            metrics={"accuracy": 0.85, "f1": 0.80}
        )
        
        result = adapter_evaluator._analyze_comparison(
            "baseline", baseline_result,
            "improved", comparison_result,
            EvaluationTask.ACCURACY
        )
        
        assert result.baseline_adapter == "baseline"
        assert result.comparison_adapter == "improved"
        assert result.baseline_metrics == {"accuracy": 0.8, "f1": 0.75}
        assert result.comparison_metrics == {"accuracy": 0.85, "f1": 0.80}
        
        # 改善度確認
        assert "accuracy" in result.improvement
        assert "f1" in result.improvement
        assert result.improvement["accuracy"] == 0.0625  # (0.85 - 0.8) / 0.8
        assert abs(result.improvement["f1"] - 0.0667) < 0.001  # (0.80 - 0.75) / 0.75
        
        # 総合改善度確認
        assert result.overall_improvement > 0
        assert "improvement" in result.recommendation.lower()
    
    def test_get_evaluation_stats_empty(self, adapter_evaluator):
        """空の評価統計取得テスト"""
        stats = adapter_evaluator.get_evaluation_stats()
        
        assert stats["total_evaluations"] == 0
        assert stats["successful_evaluations"] == 0
        assert stats["failed_evaluations"] == 0
        assert stats["success_rate"] == 0.0
    
    def test_get_evaluation_stats_with_history(self, adapter_evaluator):
        """履歴ありの評価統計取得テスト"""
        # テスト履歴追加
        results = [
            EvaluationResult(
                adapter_name="test1",
                task=EvaluationTask.ACCURACY,
                config=EvaluationConfig(task=EvaluationTask.ACCURACY, metrics=[]),
                evaluation_time=10.0,
                total_samples=100,
                success=True
            ),
            EvaluationResult(
                adapter_name="test2",
                task=EvaluationTask.ACCURACY,
                config=EvaluationConfig(task=EvaluationTask.ACCURACY, metrics=[]),
                evaluation_time=15.0,
                total_samples=150,
                success=True
            ),
            EvaluationResult(
                adapter_name="test3",
                task=EvaluationTask.ACCURACY,
                config=EvaluationConfig(task=EvaluationTask.ACCURACY, metrics=[]),
                success=False
            )
        ]
        
        adapter_evaluator.evaluation_history.extend(results)
        adapter_evaluator.evaluation_stats.update({
            "total_evaluations": 3,
            "successful_evaluations": 2,
            "failed_evaluations": 1
        })
        
        stats = adapter_evaluator.get_evaluation_stats()
        
        assert stats["total_evaluations"] == 3
        assert stats["successful_evaluations"] == 2
        assert stats["failed_evaluations"] == 1
        assert stats["success_rate"] == 2/3
        assert "recent_average_time" in stats
        assert "recent_average_samples" in stats
    
    def test_get_evaluation_history_no_filter(self, adapter_evaluator):
        """フィルタなし評価履歴取得テスト"""
        # テスト履歴追加
        results = [
            EvaluationResult(
                adapter_name="adapter1",
                task=EvaluationTask.ACCURACY,
                config=EvaluationConfig(task=EvaluationTask.ACCURACY, metrics=[])
            ),
            EvaluationResult(
                adapter_name="adapter2",
                task=EvaluationTask.BLEU,
                config=EvaluationConfig(task=EvaluationTask.BLEU, metrics=[])
            )
        ]
        adapter_evaluator.evaluation_history.extend(results)
        
        history = adapter_evaluator.get_evaluation_history()
        
        assert len(history) == 2
    
    def test_get_evaluation_history_adapter_filter(self, adapter_evaluator):
        """アダプタフィルタ評価履歴取得テスト"""
        # テスト履歴追加
        results = [
            EvaluationResult(
                adapter_name="adapter1",
                task=EvaluationTask.ACCURACY,
                config=EvaluationConfig(task=EvaluationTask.ACCURACY, metrics=[])
            ),
            EvaluationResult(
                adapter_name="adapter2",
                task=EvaluationTask.ACCURACY,
                config=EvaluationConfig(task=EvaluationTask.ACCURACY, metrics=[])
            ),
            EvaluationResult(
                adapter_name="adapter1",
                task=EvaluationTask.BLEU,
                config=EvaluationConfig(task=EvaluationTask.BLEU, metrics=[])
            )
        ]
        adapter_evaluator.evaluation_history.extend(results)
        
        history = adapter_evaluator.get_evaluation_history(adapter_name="adapter1")
        
        assert len(history) == 2
        assert all(e.adapter_name == "adapter1" for e in history)
    
    def test_get_evaluation_history_task_filter(self, adapter_evaluator):
        """タスクフィルタ評価履歴取得テスト"""
        # テスト履歴追加
        results = [
            EvaluationResult(
                adapter_name="adapter1",
                task=EvaluationTask.ACCURACY,
                config=EvaluationConfig(task=EvaluationTask.ACCURACY, metrics=[])
            ),
            EvaluationResult(
                adapter_name="adapter2",
                task=EvaluationTask.BLEU,
                config=EvaluationConfig(task=EvaluationTask.BLEU, metrics=[])
            )
        ]
        adapter_evaluator.evaluation_history.extend(results)
        
        history = adapter_evaluator.get_evaluation_history(task=EvaluationTask.ACCURACY)
        
        assert len(history) == 1
        assert history[0].task == EvaluationTask.ACCURACY
    
    def test_get_evaluation_history_limit(self, adapter_evaluator):
        """件数制限評価履歴取得テスト"""
        # テスト履歴追加
        results = [
            EvaluationResult(
                adapter_name=f"adapter{i}",
                task=EvaluationTask.ACCURACY,
                config=EvaluationConfig(task=EvaluationTask.ACCURACY, metrics=[])
            )
            for i in range(5)
        ]
        adapter_evaluator.evaluation_history.extend(results)
        
        history = adapter_evaluator.get_evaluation_history(limit=3)
        
        assert len(history) == 3
    
    def test_get_best_adapter(self, adapter_evaluator):
        """最高性能アダプタ取得テスト"""
        # テスト履歴追加
        results = [
            EvaluationResult(
                adapter_name="adapter1",
                task=EvaluationTask.ACCURACY,
                config=EvaluationConfig(task=EvaluationTask.ACCURACY, metrics=[]),
                metrics={"accuracy": 0.8},
                success=True
            ),
            EvaluationResult(
                adapter_name="adapter2",
                task=EvaluationTask.ACCURACY,
                config=EvaluationConfig(task=EvaluationTask.ACCURACY, metrics=[]),
                metrics={"accuracy": 0.9},
                success=True
            ),
            EvaluationResult(
                adapter_name="adapter3",
                task=EvaluationTask.BLEU,
                config=EvaluationConfig(task=EvaluationTask.BLEU, metrics=[]),
                metrics={"accuracy": 0.7},
                success=True
            )
        ]
        adapter_evaluator.evaluation_history.extend(results)
        
        best_adapter = adapter_evaluator.get_best_adapter(EvaluationTask.ACCURACY, "accuracy")
        
        assert best_adapter == "adapter2"
    
    def test_get_best_adapter_no_results(self, adapter_evaluator):
        """結果なし最高性能アダプタ取得テスト"""
        best_adapter = adapter_evaluator.get_best_adapter(EvaluationTask.ACCURACY, "accuracy")
        
        assert best_adapter is None


class TestEvaluationDataClasses:
    """評価データクラスのテスト"""
    
    def test_evaluation_config(self):
        """EvaluationConfig テスト"""
        config = EvaluationConfig(
            task=EvaluationTask.ACCURACY,
            metrics=[EvaluationMetric.ACCURACY, EvaluationMetric.F1_SCORE],
            dataset_name="test_dataset",
            batch_size=16,
            max_length=256
        )
        
        assert config.task == EvaluationTask.ACCURACY
        assert len(config.metrics) == 2
        assert EvaluationMetric.ACCURACY in config.metrics
        assert EvaluationMetric.F1_SCORE in config.metrics
        assert config.dataset_name == "test_dataset"
        assert config.batch_size == 16
        assert config.max_length == 256
    
    def test_evaluation_result(self):
        """EvaluationResult テスト"""
        config = EvaluationConfig(
            task=EvaluationTask.ACCURACY,
            metrics=[EvaluationMetric.ACCURACY]
        )
        
        result = EvaluationResult(
            adapter_name="test_adapter",
            task=EvaluationTask.ACCURACY,
            config=config,
            metrics={"accuracy": 0.85},
            total_samples=100,
            evaluation_time=30.0,
            success=True
        )
        
        assert result.adapter_name == "test_adapter"
        assert result.task == EvaluationTask.ACCURACY
        assert result.config == config
        assert result.metrics["accuracy"] == 0.85
        assert result.total_samples == 100
        assert result.evaluation_time == 30.0
        assert result.success is True
        assert isinstance(result.timestamp, datetime)
    
    def test_comparison_result(self):
        """ComparisonResult テスト"""
        result = ComparisonResult(
            baseline_adapter="baseline",
            comparison_adapter="improved",
            task=EvaluationTask.ACCURACY,
            baseline_metrics={"accuracy": 0.8},
            comparison_metrics={"accuracy": 0.85},
            improvement={"accuracy": 0.0625},
            overall_improvement=0.0625,
            recommendation="improved shows improvement"
        )
        
        assert result.baseline_adapter == "baseline"
        assert result.comparison_adapter == "improved"
        assert result.task == EvaluationTask.ACCURACY
        assert result.baseline_metrics["accuracy"] == 0.8
        assert result.comparison_metrics["accuracy"] == 0.85
        assert result.improvement["accuracy"] == 0.0625
        assert result.overall_improvement == 0.0625
        assert "improvement" in result.recommendation
        assert isinstance(result.timestamp, datetime)


class TestEvaluationEnums:
    """評価列挙型のテスト"""
    
    def test_evaluation_task_enum(self):
        """EvaluationTask 列挙型テスト"""
        assert EvaluationTask.PERPLEXITY.value == "perplexity"
        assert EvaluationTask.BLEU.value == "bleu"
        assert EvaluationTask.ROUGE.value == "rouge"
        assert EvaluationTask.BERTSCORE.value == "bertscore"
        assert EvaluationTask.ACCURACY.value == "accuracy"
        assert EvaluationTask.F1.value == "f1"
        assert EvaluationTask.EXACT_MATCH.value == "exact_match"
        assert EvaluationTask.CUSTOM.value == "custom"
    
    def test_evaluation_metric_enum(self):
        """EvaluationMetric 列挙型テスト"""
        assert EvaluationMetric.SCORE.value == "score"
        assert EvaluationMetric.LOSS.value == "loss"
        assert EvaluationMetric.ACCURACY.value == "accuracy"
        assert EvaluationMetric.PRECISION.value == "precision"
        assert EvaluationMetric.RECALL.value == "recall"
        assert EvaluationMetric.F1_SCORE.value == "f1"
        assert EvaluationMetric.BLEU_SCORE.value == "bleu"
        assert EvaluationMetric.ROUGE_L.value == "rouge_l"
        assert EvaluationMetric.BERTSCORE_F1.value == "bertscore_f1"


class TestCreateAdapterEvaluator:
    """create_adapter_evaluator 関数のテスト"""
    
    @pytest.mark.asyncio
    async def test_create_success(self):
        """作成成功テスト"""
        mock_pool = Mock(spec=PEFTAdapterPool)
        
        with patch('src.advanced_agent.adaptation.adapter_evaluator.AdapterEvaluator') as MockEvaluator:
            mock_evaluator = Mock()
            mock_evaluator.initialize = AsyncMock(return_value=True)
            MockEvaluator.return_value = mock_evaluator
            
            result = await create_adapter_evaluator(mock_pool)
            
            assert result == mock_evaluator
            MockEvaluator.assert_called_once_with(mock_pool, None)
            mock_evaluator.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_failure(self):
        """作成失敗テスト"""
        mock_pool = Mock(spec=PEFTAdapterPool)
        
        with patch('src.advanced_agent.adaptation.adapter_evaluator.AdapterEvaluator') as MockEvaluator:
            mock_evaluator = Mock()
            mock_evaluator.initialize = AsyncMock(return_value=False)
            MockEvaluator.return_value = mock_evaluator
            
            with pytest.raises(RuntimeError, match="Failed to initialize adapter evaluator"):
                await create_adapter_evaluator(mock_pool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])