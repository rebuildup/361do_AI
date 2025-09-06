"""
Unit tests for quality evaluation module
品質評価モジュールの単体テスト
"""

import pytest
import time
from datetime import datetime
from unittest.mock import Mock, patch

from src.advanced_agent.reasoning.quality_evaluator import (
    QualityEvaluator, QualityEvaluation, QualityScore, QualityDimension,
    evaluate_response_quality, get_quality_summary
)
from src.advanced_agent.reasoning.metrics import (
    MetricsCollector, MetricPoint, MetricSummary, MetricType, MetricCategory,
    get_metrics_collector, record_metric, record_quality_evaluation
)
from src.advanced_agent.reasoning.cot_engine import (
    CoTResponse, ReasoningStep, CoTStep, ReasoningState
)


class TestQualityEvaluator:
    """品質評価器テスト"""
    
    @pytest.fixture
    def quality_evaluator(self):
        """品質評価器"""
        return QualityEvaluator()
    
    @pytest.fixture
    def sample_cot_response(self):
        """サンプルCoTレスポンス"""
        steps = [
            ReasoningStep(1, CoTStep.THOUGHT, "問題を理解します"),
            ReasoningStep(2, CoTStep.ACTION, "計算を実行します"),
            ReasoningStep(3, CoTStep.OBSERVATION, "結果を確認します"),
            ReasoningStep(4, CoTStep.CONCLUSION, "最終回答を導きます")
        ]
        
        return CoTResponse(
            request_id="test_123",
            response_text="この問題を段階的に解決します。まず、与えられた条件を整理し、次に適切な計算方法を選択します。計算結果を検証し、最終的な答えを導きます。答えは42です。",
            processing_time=5.0,
            reasoning_steps=steps,
            final_confidence=0.8,
            step_count=4,
            total_thinking_time=4.5,
            quality_score=0.7,
            model_used="qwen2:7b-instruct",
            state=ReasoningState.COMPLETED
        )
    
    def test_quality_evaluator_initialization(self, quality_evaluator):
        """品質評価器初期化テスト"""
        assert quality_evaluator is not None
        assert len(quality_evaluator.evaluation_rules) > 0
        assert len(quality_evaluator.quality_thresholds) > 0
        assert quality_evaluator.evaluation_metrics["total_evaluations"] == 0
    
    def test_evaluate_response(self, quality_evaluator, sample_cot_response):
        """レスポンス評価テスト"""
        evaluation = quality_evaluator.evaluate_response(
            sample_cot_response, 
            "2+2は何ですか？"
        )
        
        assert isinstance(evaluation, QualityEvaluation)
        assert 0.0 <= evaluation.overall_score <= 1.0
        assert len(evaluation.dimension_scores) > 0
        assert isinstance(evaluation.strengths, list)
        assert isinstance(evaluation.weaknesses, list)
        assert isinstance(evaluation.recommendations, list)
        assert evaluation.evaluation_time > 0.0
    
    def test_evaluate_accuracy(self, quality_evaluator, sample_cot_response):
        """正確性評価テスト"""
        score = quality_evaluator._evaluate_accuracy(
            sample_cot_response.response_text.lower(),
            sample_cot_response,
            quality_evaluator.evaluation_rules[QualityDimension.ACCURACY]
        )
        
        assert isinstance(score, QualityScore)
        assert score.dimension == QualityDimension.ACCURACY
        assert 0.0 <= score.score <= 1.0
        assert 0.0 <= score.confidence <= 1.0
        assert isinstance(score.explanation, str)
        assert isinstance(score.evidence, list)
        assert isinstance(score.suggestions, list)
    
    def test_evaluate_completeness(self, quality_evaluator, sample_cot_response):
        """完全性評価テスト"""
        score = quality_evaluator._evaluate_completeness(
            sample_cot_response.response_text.lower(),
            sample_cot_response,
            quality_evaluator.evaluation_rules[QualityDimension.COMPLETENESS],
            "2+2は何ですか？"
        )
        
        assert isinstance(score, QualityScore)
        assert score.dimension == QualityDimension.COMPLETENESS
        assert 0.0 <= score.score <= 1.0
    
    def test_evaluate_clarity(self, quality_evaluator, sample_cot_response):
        """明確性評価テスト"""
        score = quality_evaluator._evaluate_clarity(
            sample_cot_response.response_text.lower(),
            sample_cot_response,
            quality_evaluator.evaluation_rules[QualityDimension.CLARITY]
        )
        
        assert isinstance(score, QualityScore)
        assert score.dimension == QualityDimension.CLARITY
        assert 0.0 <= score.score <= 1.0
    
    def test_evaluate_logical_consistency(self, quality_evaluator, sample_cot_response):
        """論理的一貫性評価テスト"""
        score = quality_evaluator._evaluate_logical_consistency(
            sample_cot_response.response_text.lower(),
            sample_cot_response,
            quality_evaluator.evaluation_rules[QualityDimension.LOGICAL_CONSISTENCY]
        )
        
        assert isinstance(score, QualityScore)
        assert score.dimension == QualityDimension.LOGICAL_CONSISTENCY
        assert 0.0 <= score.score <= 1.0
    
    def test_evaluate_usefulness(self, quality_evaluator, sample_cot_response):
        """有用性評価テスト"""
        score = quality_evaluator._evaluate_usefulness(
            sample_cot_response.response_text.lower(),
            sample_cot_response,
            quality_evaluator.evaluation_rules[QualityDimension.USEFULNESS],
            "2+2は何ですか？"
        )
        
        assert isinstance(score, QualityScore)
        assert score.dimension == QualityDimension.USEFULNESS
        assert 0.0 <= score.score <= 1.0
    
    def test_evaluate_efficiency(self, quality_evaluator, sample_cot_response):
        """効率性評価テスト"""
        score = quality_evaluator._evaluate_efficiency(
            sample_cot_response.response_text.lower(),
            sample_cot_response,
            quality_evaluator.evaluation_rules[QualityDimension.EFFICIENCY]
        )
        
        assert isinstance(score, QualityScore)
        assert score.dimension == QualityDimension.EFFICIENCY
        assert 0.0 <= score.score <= 1.0
    
    def test_evaluate_creativity(self, quality_evaluator, sample_cot_response):
        """創造性評価テスト"""
        score = quality_evaluator._evaluate_creativity(
            sample_cot_response.response_text.lower(),
            sample_cot_response,
            quality_evaluator.evaluation_rules[QualityDimension.CREATIVITY]
        )
        
        assert isinstance(score, QualityScore)
        assert score.dimension == QualityDimension.CREATIVITY
        assert 0.0 <= score.score <= 1.0
    
    def test_evaluate_safety(self, quality_evaluator, sample_cot_response):
        """安全性評価テスト"""
        score = quality_evaluator._evaluate_safety(
            sample_cot_response.response_text.lower(),
            sample_cot_response,
            quality_evaluator.evaluation_rules[QualityDimension.SAFETY]
        )
        
        assert isinstance(score, QualityScore)
        assert score.dimension == QualityDimension.SAFETY
        assert 0.0 <= score.score <= 1.0
    
    def test_calculate_overall_score(self, quality_evaluator):
        """総合スコア計算テスト"""
        dimension_scores = {
            QualityDimension.ACCURACY: QualityScore(QualityDimension.ACCURACY, 0.8, 0.7, "Good accuracy"),
            QualityDimension.COMPLETENESS: QualityScore(QualityDimension.COMPLETENESS, 0.7, 0.6, "Good completeness"),
            QualityDimension.CLARITY: QualityScore(QualityDimension.CLARITY, 0.6, 0.5, "Moderate clarity")
        }
        
        overall_score = quality_evaluator._calculate_overall_score(dimension_scores)
        
        assert 0.0 <= overall_score <= 1.0
        assert overall_score > 0.0
    
    def test_check_step_consistency(self, quality_evaluator):
        """ステップ一貫性チェックテスト"""
        steps = [
            ReasoningStep(1, CoTStep.THOUGHT, "問題を理解します"),
            ReasoningStep(2, CoTStep.ACTION, "計算を実行します"),
            ReasoningStep(3, CoTStep.OBSERVATION, "結果を確認します")
        ]
        
        consistency = quality_evaluator._check_step_consistency(steps)
        
        assert 0.0 <= consistency <= 1.0
    
    def test_is_question_answered(self, quality_evaluator):
        """質問回答チェックテスト"""
        text = "2+2の答えは4です。"
        question = "2+2は何ですか？"
        
        is_answered = quality_evaluator._is_question_answered(text, question)
        
        assert isinstance(is_answered, bool)
    
    def test_has_redundancy(self, quality_evaluator):
        """冗長性チェックテスト"""
        # 冗長なテキスト
        redundant_text = "これはテストです。これはテストです。これはテストです。"
        non_redundant_text = "これはテストです。別の文です。最後の文です。"
        
        assert quality_evaluator._has_redundancy(redundant_text) is True
        assert quality_evaluator._has_redundancy(non_redundant_text) is False
    
    def test_are_sentences_similar(self, quality_evaluator):
        """文の類似性チェックテスト"""
        sentence1 = "これはテストの文です"
        sentence2 = "これはテストの文章です"
        sentence3 = "全く違う内容の文です"
        
        assert quality_evaluator._are_sentences_similar(sentence1, sentence2) is True
        assert quality_evaluator._are_sentences_similar(sentence1, sentence3) is False
    
    def test_has_original_insights(self, quality_evaluator):
        """独創的洞察チェックテスト"""
        creative_text = "これは新しい方法で問題を解決します"
        normal_text = "これは普通の説明です"
        
        assert quality_evaluator._has_original_insights(creative_text) is True
        assert quality_evaluator._has_original_insights(normal_text) is False
    
    def test_get_evaluation_statistics(self, quality_evaluator, sample_cot_response):
        """評価統計取得テスト"""
        # 評価実行
        quality_evaluator.evaluate_response(sample_cot_response, "テスト質問")
        
        stats = quality_evaluator.get_evaluation_statistics()
        
        assert "total_evaluations" in stats
        assert "average_evaluation_time" in stats
        assert "dimension_averages" in stats
        assert "dimension_performance" in stats
        assert stats["total_evaluations"] > 0
    
    def test_reset_metrics(self, quality_evaluator, sample_cot_response):
        """メトリクスリセットテスト"""
        # 評価実行
        quality_evaluator.evaluate_response(sample_cot_response, "テスト質問")
        
        # リセット前の確認
        assert quality_evaluator.evaluation_metrics["total_evaluations"] > 0
        
        # リセット
        quality_evaluator.reset_metrics()
        
        # リセット後の確認
        assert quality_evaluator.evaluation_metrics["total_evaluations"] == 0


class TestQualityEvaluationConvenienceFunctions:
    """品質評価便利関数テスト"""
    
    @pytest.fixture
    def sample_cot_response(self):
        """サンプルCoTレスポンス"""
        steps = [
            ReasoningStep(1, CoTStep.THOUGHT, "問題を理解します"),
            ReasoningStep(2, CoTStep.CONCLUSION, "答えは4です")
        ]
        
        return CoTResponse(
            request_id="test_123",
            response_text="2+2の答えは4です。",
            processing_time=2.0,
            reasoning_steps=steps,
            final_confidence=0.9,
            step_count=2,
            total_thinking_time=1.5,
            quality_score=0.8,
            model_used="qwen2:7b-instruct",
            state=ReasoningState.COMPLETED
        )
    
    def test_evaluate_response_quality_function(self, sample_cot_response):
        """evaluate_response_quality関数テスト"""
        evaluation = evaluate_response_quality(sample_cot_response, "2+2は何ですか？")
        
        assert isinstance(evaluation, QualityEvaluation)
        assert 0.0 <= evaluation.overall_score <= 1.0
    
    def test_get_quality_summary_function(self, sample_cot_response):
        """get_quality_summary関数テスト"""
        evaluation = evaluate_response_quality(sample_cot_response, "2+2は何ですか？")
        summary = get_quality_summary(evaluation)
        
        assert "overall_score" in summary
        assert "grade" in summary
        assert "top_strengths" in summary
        assert "main_weaknesses" in summary
        assert "key_recommendations" in summary
        assert "evaluation_time" in summary
        
        assert 0.0 <= summary["overall_score"] <= 1.0
        assert summary["grade"] in ["A+", "A", "B+", "B", "C+", "C", "D"]


class TestMetricsCollector:
    """メトリクスコレクターテスト"""
    
    @pytest.fixture
    def metrics_collector(self):
        """メトリクスコレクター"""
        return MetricsCollector()
    
    def test_metrics_collector_initialization(self, metrics_collector):
        """メトリクスコレクター初期化テスト"""
        assert metrics_collector is not None
        assert metrics_collector.retention_hours == 24
        assert len(metrics_collector.metric_configs) > 0
        assert len(metrics_collector.metrics) == 0
    
    def test_record_metric(self, metrics_collector):
        """メトリクス記録テスト"""
        success = metrics_collector.record_metric("test_metric", 1.0, {"label": "value"})
        
        assert success is True
        assert len(metrics_collector.metrics) == 1
        
        metric = metrics_collector.metrics[0]
        assert metric.name == "test_metric"
        assert metric.value == 1.0
        assert metric.labels == {"label": "value"}
        assert isinstance(metric.timestamp, datetime)
    
    def test_record_metric_validation(self, metrics_collector):
        """メトリクス記録検証テスト"""
        # 無効な値（範囲外）
        success = metrics_collector.record_metric("overall_quality", 2.0)  # 範囲外
        assert success is False
        
        # 有効な値
        success = metrics_collector.record_metric("overall_quality", 0.8)
        assert success is True
    
    def test_record_quality_evaluation(self, metrics_collector):
        """品質評価メトリクス記録テスト"""
        from src.advanced_agent.reasoning.quality_evaluator import QualityEvaluation, QualityScore, QualityDimension
        
        evaluation = QualityEvaluation(
            overall_score=0.85,
            dimension_scores={
                QualityDimension.ACCURACY: QualityScore(QualityDimension.ACCURACY, 0.9, 0.8, "High accuracy"),
                QualityDimension.COMPLETENESS: QualityScore(QualityDimension.COMPLETENESS, 0.8, 0.7, "Good completeness")
            },
            strengths=["High accuracy"],
            weaknesses=[],
            recommendations=[],
            evaluation_time=0.5,
            metadata={"response_length": 100, "step_count": 3, "processing_time": 2.0, "confidence": 0.8}
        )
        
        metrics_collector.record_quality_evaluation(evaluation, {"model": "test"})
        
        # 記録されたメトリクスを確認
        assert len(metrics_collector.metrics) > 0
        
        # 総合品質スコアが記録されているか確認
        quality_metrics = [m for m in metrics_collector.metrics if m.name == "overall_quality"]
        assert len(quality_metrics) == 1
        assert quality_metrics[0].value == 0.85
    
    def test_record_request_metrics(self, metrics_collector):
        """リクエストメトリクス記録テスト"""
        metrics_collector.record_request_metrics(True, 2.5, {"model": "test"})
        
        # リクエストメトリクスが記録されているか確認
        total_requests = [m for m in metrics_collector.metrics if m.name == "total_requests"]
        successful_requests = [m for m in metrics_collector.metrics if m.name == "successful_requests"]
        reasoning_time = [m for m in metrics_collector.metrics if m.name == "reasoning_time"]
        
        assert len(total_requests) == 1
        assert len(successful_requests) == 1
        assert len(reasoning_time) == 1
        assert reasoning_time[0].value == 2.5
    
    def test_get_metric_summary(self, metrics_collector):
        """メトリクスサマリー取得テスト"""
        # 複数のメトリクスを記録
        for i in range(5):
            metrics_collector.record_metric("test_metric", float(i), {"label": "value"})
        
        summary = metrics_collector.get_metric_summary("test_metric")
        
        assert summary is not None
        assert summary.name == "test_metric"
        assert summary.count == 5
        assert summary.sum == 10.0  # 0+1+2+3+4
        assert summary.min == 0.0
        assert summary.max == 4.0
        assert summary.mean == 2.0
        assert summary.median == 2.0
    
    def test_get_metric_summary_not_found(self, metrics_collector):
        """存在しないメトリクスのサマリー取得テスト"""
        summary = metrics_collector.get_metric_summary("nonexistent_metric")
        assert summary is None
    
    def test_filter_metrics(self, metrics_collector):
        """メトリクスフィルタリングテスト"""
        # 異なるラベルでメトリクスを記録
        metrics_collector.record_metric("test_metric", 1.0, {"label1": "value1"})
        metrics_collector.record_metric("test_metric", 2.0, {"label1": "value2"})
        metrics_collector.record_metric("other_metric", 3.0, {"label1": "value1"})
        
        # 名前でフィルタ
        filtered = metrics_collector._filter_metrics("test_metric")
        assert len(filtered) == 2
        
        # ラベルでフィルタ
        filtered = metrics_collector._filter_metrics("test_metric", labels_filter={"label1": "value1"})
        assert len(filtered) == 1
        assert filtered[0].value == 1.0
    
    def test_get_metrics_by_category(self, metrics_collector):
        """カテゴリ別メトリクス取得テスト"""
        metrics_collector.record_metric("overall_quality", 0.8)  # QUALITY
        metrics_collector.record_metric("reasoning_time", 2.0)   # PERFORMANCE
        
        quality_metrics = metrics_collector.get_metrics_by_category(MetricCategory.QUALITY)
        performance_metrics = metrics_collector.get_metrics_by_category(MetricCategory.PERFORMANCE)
        
        assert len(quality_metrics) == 1
        assert len(performance_metrics) == 1
        assert quality_metrics[0].name == "overall_quality"
        assert performance_metrics[0].name == "reasoning_time"
    
    def test_get_metrics_by_type(self, metrics_collector):
        """タイプ別メトリクス取得テスト"""
        metrics_collector.record_metric("overall_quality", 0.8)  # GAUGE
        metrics_collector.record_metric("total_requests", 1)     # COUNTER
        
        gauge_metrics = metrics_collector.get_metrics_by_type(MetricType.GAUGE)
        counter_metrics = metrics_collector.get_metrics_by_type(MetricType.COUNTER)
        
        assert len(gauge_metrics) == 1
        assert len(counter_metrics) == 1
        assert gauge_metrics[0].name == "overall_quality"
        assert counter_metrics[0].name == "total_requests"
    
    def test_get_latest_metric(self, metrics_collector):
        """最新メトリクス取得テスト"""
        metrics_collector.record_metric("test_metric", 1.0)
        time.sleep(0.01)  # 時間差を作る
        metrics_collector.record_metric("test_metric", 2.0)
        
        latest = metrics_collector.get_latest_metric("test_metric")
        
        assert latest is not None
        assert latest.value == 2.0
    
    def test_get_metric_trend(self, metrics_collector):
        """メトリクストレンド取得テスト"""
        # 複数のメトリクスを記録
        for i in range(10):
            metrics_collector.record_metric("test_metric", float(i))
            time.sleep(0.001)  # 時間差を作る
        
        trend = metrics_collector.get_metric_trend("test_metric")
        
        assert len(trend) > 0
        assert all(isinstance(point[0], datetime) for point in trend)
        assert all(isinstance(point[1], float) for point in trend)
    
    def test_export_metrics(self, metrics_collector):
        """メトリクスエクスポートテスト"""
        metrics_collector.record_metric("test_metric", 1.0, {"label": "value"})
        
        # JSON形式でエクスポート
        exported = metrics_collector.export_metrics("json")
        
        assert isinstance(exported, str)
        assert "test_metric" in exported
        assert "1.0" in exported
        
        # Prometheus形式でエクスポート
        exported_prom = metrics_collector.export_metrics("prometheus")
        
        assert isinstance(exported_prom, str)
        assert "test_metric" in exported_prom
    
    def test_export_metrics_unsupported_format(self, metrics_collector):
        """サポートされていない形式でのエクスポートテスト"""
        with pytest.raises(ValueError):
            metrics_collector.export_metrics("unsupported")
    
    def test_get_collector_stats(self, metrics_collector):
        """コレクター統計取得テスト"""
        metrics_collector.record_metric("test_metric", 1.0)
        
        stats = metrics_collector.get_collector_stats()
        
        assert "total_metrics" in stats
        assert "retention_hours" in stats
        assert "stats" in stats
        assert "metric_configs_count" in stats
        assert stats["total_metrics"] == 1
    
    def test_reset_metrics(self, metrics_collector):
        """メトリクスリセットテスト"""
        metrics_collector.record_metric("test_metric", 1.0)
        
        assert len(metrics_collector.metrics) == 1
        
        metrics_collector.reset_metrics()
        
        assert len(metrics_collector.metrics) == 0
        assert metrics_collector.stats["total_metrics"] == 0


class TestMetricsCollectorConvenienceFunctions:
    """メトリクスコレクター便利関数テスト"""
    
    def test_get_metrics_collector_singleton(self):
        """メトリクスコレクターシングルトンテスト"""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        
        assert collector1 is collector2
    
    def test_record_metric_function(self):
        """record_metric関数テスト"""
        success = record_metric("test_metric", 1.0, {"label": "value"})
        
        assert success is True
        
        collector = get_metrics_collector()
        assert len(collector.metrics) > 0
    
    def test_record_quality_evaluation_function(self):
        """record_quality_evaluation関数テスト"""
        from src.advanced_agent.reasoning.quality_evaluator import QualityEvaluation, QualityScore, QualityDimension
        
        evaluation = QualityEvaluation(
            overall_score=0.8,
            dimension_scores={
                QualityDimension.ACCURACY: QualityScore(QualityDimension.ACCURACY, 0.9, 0.8, "High accuracy")
            },
            strengths=["High accuracy"],
            weaknesses=[],
            recommendations=[],
            evaluation_time=0.5
        )
        
        record_quality_evaluation(evaluation, {"model": "test"})
        
        collector = get_metrics_collector()
        quality_metrics = [m for m in collector.metrics if m.name == "overall_quality"]
        assert len(quality_metrics) > 0
    
    def test_record_request_metrics_function(self):
        """record_request_metrics関数テスト"""
        record_request_metrics(True, 2.0, {"model": "test"})
        
        collector = get_metrics_collector()
        request_metrics = [m for m in collector.metrics if m.name == "total_requests"]
        assert len(request_metrics) > 0


class TestQualityEvaluationIntegration:
    """品質評価統合テスト"""
    
    def test_quality_evaluation_with_metrics_integration(self):
        """品質評価とメトリクスの統合テスト"""
        # サンプルCoTレスポンス作成
        steps = [
            ReasoningStep(1, CoTStep.THOUGHT, "問題を理解します"),
            ReasoningStep(2, CoTStep.CONCLUSION, "答えは4です")
        ]
        
        cot_response = CoTResponse(
            request_id="test_123",
            response_text="2+2の答えは4です。段階的に計算しました。",
            processing_time=2.0,
            reasoning_steps=steps,
            final_confidence=0.9,
            step_count=2,
            total_thinking_time=1.5,
            quality_score=0.8,
            model_used="qwen2:7b-instruct",
            state=ReasoningState.COMPLETED
        )
        
        # 品質評価実行
        evaluator = QualityEvaluator()
        evaluation = evaluator.evaluate_response(cot_response, "2+2は何ですか？")
        
        # メトリクス記録
        record_quality_evaluation(evaluation, {"model": "qwen2:7b-instruct", "session": "test"})
        
        # メトリクス確認
        collector = get_metrics_collector()
        quality_metrics = [m for m in collector.metrics if m.name == "overall_quality"]
        
        assert len(quality_metrics) > 0
        assert quality_metrics[0].value == evaluation.overall_score
        assert quality_metrics[0].labels["model"] == "qwen2:7b-instruct"
    
    def test_quality_evaluation_performance(self):
        """品質評価パフォーマンステスト"""
        # 大量の評価を実行してパフォーマンスをテスト
        evaluator = QualityEvaluator()
        
        steps = [ReasoningStep(1, CoTStep.CONCLUSION, "テスト回答")]
        cot_response = CoTResponse(
            request_id="test",
            response_text="これはテスト回答です。",
            processing_time=1.0,
            reasoning_steps=steps,
            final_confidence=0.8,
            step_count=1,
            total_thinking_time=0.5,
            quality_score=0.7,
            model_used="test",
            state=ReasoningState.COMPLETED
        )
        
        start_time = time.time()
        
        # 複数回評価実行
        for i in range(10):
            evaluation = evaluator.evaluate_response(cot_response, f"テスト質問{i}")
            assert evaluation.overall_score >= 0.0
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # パフォーマンス確認（10回の評価が5秒以内に完了することを期待）
        assert total_time < 5.0
        
        # 統計確認
        stats = evaluator.get_evaluation_statistics()
        assert stats["total_evaluations"] == 10


if __name__ == "__main__":
    pytest.main([__file__])
