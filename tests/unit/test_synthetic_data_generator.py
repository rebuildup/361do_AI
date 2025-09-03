"""
Synthetic Data Generator のテスト
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime

from src.advanced_agent.evolution.synthetic_data_generator import (
    SyntheticDataGenerator, GenerationConfig, GeneratedSample, GenerationResult,
    QualityScore, QualityFilter, DataType, GenerationStrategy, QualityMetric,
    create_synthetic_data_generator
)
from src.advanced_agent.adaptation.peft_manager import PEFTAdapterPool
from src.advanced_agent.monitoring.system_monitor import SystemMonitor


class TestSyntheticDataGenerator:
    """SyntheticDataGenerator クラスのテスト"""
    
    @pytest.fixture
    def mock_peft_pool(self):
        """モックPEFTプール"""
        pool = Mock(spec=PEFTAdapterPool)
        pool.active_adapters = {}
        pool.tokenizer = Mock()
        pool.tokenizer.eos_token_id = 50256
        return pool
    
    @pytest.fixture
    def mock_system_monitor(self):
        """モックシステムモニター"""
        return Mock(spec=SystemMonitor)
    
    @pytest.fixture
    def data_generator(self, mock_peft_pool, mock_system_monitor):
        """SyntheticDataGenerator インスタンス"""
        with patch('src.advanced_agent.evolution.synthetic_data_generator.get_config') as mock_get_config, \
             patch('src.advanced_agent.evolution.synthetic_data_generator.DATASETS_AVAILABLE', True):
            from src.advanced_agent.core.config import AdvancedAgentConfig
            mock_get_config.return_value = AdvancedAgentConfig()
            
            return SyntheticDataGenerator(mock_peft_pool, mock_system_monitor)
    
    def test_init(self, data_generator, mock_peft_pool, mock_system_monitor):
        """初期化テスト"""
        assert data_generator.peft_pool == mock_peft_pool
        assert data_generator.system_monitor == mock_system_monitor
        assert data_generator.generation_pipeline is None
        assert data_generator.tokenizer is None
        assert isinstance(data_generator.quality_filter, QualityFilter)
        assert len(data_generator.templates) > 0
        assert len(data_generator.generation_history) == 0
        assert data_generator.generation_stats["total_generations"] == 0
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, data_generator):
        """初期化成功テスト"""
        with patch.object(data_generator.quality_filter, 'initialize', return_value=True):
            result = await data_generator.initialize()
            
            assert result is True
    
    @patch('src.advanced_agent.evolution.synthetic_data_generator.DATASETS_AVAILABLE', False)
    @pytest.mark.asyncio
    async def test_initialize_unavailable(self, data_generator):
        """Datasets利用不可時の初期化テスト"""
        result = await data_generator.initialize()
        
        assert result is False
    
    def test_load_default_templates(self, data_generator):
        """デフォルトテンプレート読み込みテスト"""
        templates = data_generator._load_default_templates()
        
        # 各データタイプにテンプレートが存在することを確認
        assert DataType.TEXT_GENERATION in templates
        assert DataType.TEXT_CLASSIFICATION in templates
        assert DataType.QUESTION_ANSWERING in templates
        assert DataType.SUMMARIZATION in templates
        assert DataType.CONVERSATION in templates
        
        # テンプレートが空でないことを確認
        for data_type, template_list in templates.items():
            assert len(template_list) > 0
            assert all(isinstance(template, str) for template in template_list)
    
    def test_fill_template(self, data_generator):
        """テンプレート変数埋め込みテスト"""
        template = "{topic}について説明してください。"
        config = GenerationConfig(
            data_type=DataType.TEXT_GENERATION,
            strategy=GenerationStrategy.TEMPLATE_BASED,
            domain="technology"
        )
        
        filled = data_generator._fill_template(template, config)
        
        assert "{topic}" not in filled
        assert "について説明してください。" in filled
        assert len(filled) > len("について説明してください。")
    
    def test_generate_prompts(self, data_generator):
        """プロンプト生成テスト"""
        config = GenerationConfig(
            data_type=DataType.TEXT_GENERATION,
            strategy=GenerationStrategy.MODEL_BASED,
            domain="technology",
            num_samples=5
        )
        
        prompts = data_generator._generate_prompts(config)
        
        assert len(prompts) == 5
        assert all(isinstance(prompt, str) for prompt in prompts)
        assert all("technology" in prompt for prompt in prompts)
    
    def test_synonym_replacement(self, data_generator):
        """同義語置換テスト"""
        text = "これは重要な問題です。"
        
        result = data_generator._synonym_replacement(text)
        
        # 同義語が置換されているか確認
        assert result != text or "重要" not in text  # 置換されるか、元々対象語がない
        if "重要" in text:
            assert "大切" in result or "課題" in result
    
    def test_sentence_reorder(self, data_generator):
        """文の順序変更テスト"""
        text = "最初の文です。2番目の文です。3番目の文です。"
        
        result = data_generator._sentence_reorder(text)
        
        # 文の数は同じ
        original_sentences = [s.strip() for s in text.split('。') if s.strip()]
        result_sentences = [s.strip() for s in result.split('。') if s.strip()]
        
        assert len(original_sentences) == len(result_sentences)
        
        # 全ての文が含まれている
        for sentence in original_sentences:
            assert any(sentence in result_sentence for result_sentence in result_sentences)
    
    def test_paraphrase(self, data_generator):
        """パラフレーズテスト"""
        text1 = "これは重要です。"
        text2 = "これは重要である。"
        
        result1 = data_generator._paraphrase(text1)
        result2 = data_generator._paraphrase(text2)
        
        # 文体が変更されることを確認
        if text1.endswith('です。'):
            assert result1.endswith('である。')
        
        if text2.endswith('である。'):
            assert result2.endswith('です。')
    
    def test_augment_text(self, data_generator):
        """テキスト拡張テスト"""
        text = "これは重要な問題です。解決が必要です。"
        config = GenerationConfig(
            data_type=DataType.TEXT_GENERATION,
            strategy=GenerationStrategy.AUGMENTATION
        )
        
        augmented = data_generator._augment_text(text, config)
        
        assert isinstance(augmented, list)
        assert len(augmented) > 0
        assert all(isinstance(aug_text, str) for aug_text in augmented)
        assert all(aug_text != text for aug_text in augmented if aug_text)
    
    @pytest.mark.asyncio
    async def test_generate_template_based(self, data_generator):
        """テンプレートベース生成テスト"""
        config = GenerationConfig(
            data_type=DataType.TEXT_GENERATION,
            strategy=GenerationStrategy.TEMPLATE_BASED,
            num_samples=3,
            domain="technology"
        )
        
        samples = await data_generator._generate_template_based(config)
        
        assert len(samples) == 3
        assert all(isinstance(sample, GeneratedSample) for sample in samples)
        assert all(sample.generation_method == "template_based" for sample in samples)
        assert all(sample.template_used is not None for sample in samples)
        assert all("technology" in sample.metadata.get("domain", "") for sample in samples)
    
    @patch('src.advanced_agent.evolution.synthetic_data_generator.HFDataset')
    def test_create_dataset(self, mock_dataset, data_generator):
        """データセット作成テスト"""
        mock_dataset_instance = Mock()
        mock_dataset.from_dict.return_value = mock_dataset_instance
        
        samples = [
            GeneratedSample(
                text="Sample 1",
                generation_method="template_based",
                quality_score=QualityScore(overall=0.8)
            ),
            GeneratedSample(
                text="Sample 2",
                label="positive",
                generation_method="model_based",
                quality_score=QualityScore(overall=0.7)
            )
        ]
        
        config = GenerationConfig(
            data_type=DataType.TEXT_CLASSIFICATION,
            strategy=GenerationStrategy.TEMPLATE_BASED
        )
        
        result = data_generator._create_dataset(samples, config)
        
        assert result == mock_dataset_instance
        mock_dataset.from_dict.assert_called_once()
        
        # 呼び出し引数確認
        call_args = mock_dataset.from_dict.call_args[0][0]
        assert "text" in call_args
        assert "generation_method" in call_args
        assert "timestamp" in call_args
        assert "label" in call_args  # ラベルがあるサンプルが含まれているため
        assert "quality_score" in call_args  # 品質スコアがあるサンプルが含まれているため
        
        assert len(call_args["text"]) == 2
        assert call_args["text"] == ["Sample 1", "Sample 2"]
    
    def test_calculate_quality_stats(self, data_generator):
        """品質統計計算テスト"""
        samples = [
            GeneratedSample(
                text="High quality",
                quality_score=QualityScore(overall=0.9)
            ),
            GeneratedSample(
                text="Medium quality",
                quality_score=QualityScore(overall=0.6)
            ),
            GeneratedSample(
                text="Low quality",
                quality_score=QualityScore(overall=0.3)
            )
        ]
        
        stats = data_generator._calculate_quality_stats(samples)
        
        assert "average" in stats
        assert "distribution" in stats
        
        # 平均品質確認
        expected_average = (0.9 + 0.6 + 0.3) / 3
        assert abs(stats["average"] - expected_average) < 0.01
        
        # 分布確認
        distribution = stats["distribution"]
        assert distribution["high"] == 1  # 0.9
        assert distribution["medium"] == 1  # 0.6
        assert distribution["low"] == 1  # 0.3
    
    def test_update_generation_stats(self, data_generator):
        """生成統計更新テスト"""
        # 初期状態
        assert data_generator.generation_stats["successful_generations"] == 0
        assert data_generator.generation_stats["average_quality"] == 0.0
        
        # 最初の結果
        result1 = GenerationResult(
            config=GenerationConfig(data_type=DataType.TEXT_GENERATION, strategy=GenerationStrategy.TEMPLATE_BASED),
            average_quality=0.8,
            success=True
        )
        
        data_generator.generation_stats["successful_generations"] = 1
        data_generator._update_generation_stats(result1)
        
        assert data_generator.generation_stats["average_quality"] == 0.8
        
        # 2番目の結果
        result2 = GenerationResult(
            config=GenerationConfig(data_type=DataType.TEXT_GENERATION, strategy=GenerationStrategy.TEMPLATE_BASED),
            average_quality=0.6,
            success=True
        )
        
        data_generator.generation_stats["successful_generations"] = 2
        data_generator._update_generation_stats(result2)
        
        # 平均品質確認
        expected_avg = (0.8 + 0.6) / 2
        assert abs(data_generator.generation_stats["average_quality"] - expected_avg) < 0.01
    
    def test_get_generation_stats_empty(self, data_generator):
        """空の生成統計取得テスト"""
        stats = data_generator.get_generation_stats()
        
        assert stats["total_generations"] == 0
        assert stats["successful_generations"] == 0
        assert stats["failed_generations"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["filter_rate"] == 0.0
    
    def test_get_generation_stats_with_data(self, data_generator):
        """データありの生成統計取得テスト"""
        # 統計設定
        data_generator.generation_stats.update({
            "total_generations": 5,
            "successful_generations": 4,
            "failed_generations": 1,
            "total_samples_generated": 100,
            "total_samples_filtered": 80,
            "average_quality": 0.75
        })
        
        stats = data_generator.get_generation_stats()
        
        assert stats["total_generations"] == 5
        assert stats["successful_generations"] == 4
        assert stats["failed_generations"] == 1
        assert stats["success_rate"] == 0.8  # 4/5
        assert stats["filter_rate"] == 0.8  # 80/100
        assert stats["average_quality"] == 0.75
    
    def test_get_generation_history_no_filter(self, data_generator):
        """フィルタなし生成履歴取得テスト"""
        # テスト履歴追加
        results = [
            GenerationResult(
                config=GenerationConfig(data_type=DataType.TEXT_GENERATION, strategy=GenerationStrategy.TEMPLATE_BASED),
                success=True
            ),
            GenerationResult(
                config=GenerationConfig(data_type=DataType.TEXT_GENERATION, strategy=GenerationStrategy.TEMPLATE_BASED),
                success=False
            )
        ]
        data_generator.generation_history.extend(results)
        
        history = data_generator.get_generation_history()
        
        assert len(history) == 2
    
    def test_get_generation_history_success_filter(self, data_generator):
        """成功フィルタ生成履歴取得テスト"""
        # テスト履歴追加
        results = [
            GenerationResult(
                config=GenerationConfig(data_type=DataType.TEXT_GENERATION, strategy=GenerationStrategy.TEMPLATE_BASED),
                success=True
            ),
            GenerationResult(
                config=GenerationConfig(data_type=DataType.TEXT_GENERATION, strategy=GenerationStrategy.TEMPLATE_BASED),
                success=False
            )
        ]
        data_generator.generation_history.extend(results)
        
        history = data_generator.get_generation_history(success_only=True)
        
        assert len(history) == 1
        assert history[0].success is True
    
    def test_get_generation_history_limit(self, data_generator):
        """件数制限生成履歴取得テスト"""
        # テスト履歴追加
        results = [
            GenerationResult(
                config=GenerationConfig(data_type=DataType.TEXT_GENERATION, strategy=GenerationStrategy.TEMPLATE_BASED),
                success=True
            )
            for _ in range(5)
        ]
        data_generator.generation_history.extend(results)
        
        history = data_generator.get_generation_history(limit=3)
        
        assert len(history) == 3


class TestQualityFilter:
    """QualityFilter クラスのテスト"""
    
    @pytest.fixture
    def quality_filter(self):
        """QualityFilter インスタンス"""
        return QualityFilter()
    
    @pytest.mark.asyncio
    async def test_initialize(self, quality_filter):
        """初期化テスト"""
        with patch('src.advanced_agent.evolution.synthetic_data_generator.DATASETS_AVAILABLE', True):
            result = await quality_filter.initialize()
            
            assert result is True
    
    def test_evaluate_fluency(self, quality_filter):
        """流暢性評価テスト"""
        # 良い例
        good_text = "これは適切な長さの文章です。読みやすく、理解しやすい内容になっています。"
        fluency_good = quality_filter._evaluate_fluency(good_text)
        
        # 悪い例（短すぎる）
        bad_text = "短い。"
        fluency_bad = quality_filter._evaluate_fluency(bad_text)
        
        assert 0.0 <= fluency_good <= 1.0
        assert 0.0 <= fluency_bad <= 1.0
        assert fluency_good > fluency_bad
    
    def test_evaluate_coherence(self, quality_filter):
        """一貫性評価テスト"""
        # 接続詞を含む文章
        coherent_text = "まず問題を分析します。しかし、解決策は複雑です。したがって、段階的なアプローチが必要です。"
        coherence_good = quality_filter._evaluate_coherence(coherent_text)
        
        # 接続詞のない文章
        incoherent_text = "問題があります。解決策があります。"
        coherence_bad = quality_filter._evaluate_coherence(incoherent_text)
        
        assert 0.0 <= coherence_good <= 1.0
        assert 0.0 <= coherence_bad <= 1.0
        assert coherence_good >= coherence_bad
    
    def test_evaluate_diversity(self, quality_filter):
        """多様性評価テスト"""
        # 多様な語彙
        diverse_text = "革新的な技術が社会に与える影響について考察します。"
        diversity_good = quality_filter._evaluate_diversity(diverse_text)
        
        # 重複する語彙
        repetitive_text = "技術技術技術について技術技術を説明します。"
        diversity_bad = quality_filter._evaluate_diversity(repetitive_text)
        
        assert 0.0 <= diversity_good <= 1.0
        assert 0.0 <= diversity_bad <= 1.0
        assert diversity_good > diversity_bad
    
    def test_evaluate_relevance(self, quality_filter):
        """関連性評価テスト"""
        # 技術ドメインに関連
        tech_text = "AIとデータサイエンスの技術革新について"
        tech_metadata = {"domain": "technology"}
        relevance_good = quality_filter._evaluate_relevance(tech_text, tech_metadata)
        
        # 技術ドメインに無関係
        irrelevant_text = "料理のレシピについて"
        relevance_bad = quality_filter._evaluate_relevance(irrelevant_text, tech_metadata)
        
        assert 0.0 <= relevance_good <= 1.0
        assert 0.0 <= relevance_bad <= 1.0
        assert relevance_good > relevance_bad
    
    def test_evaluate_toxicity(self, quality_filter):
        """毒性評価テスト"""
        # 適切な文章
        clean_text = "これは建設的で有用な情報です。"
        toxicity_good = quality_filter._evaluate_toxicity(clean_text)
        
        # 有害語句を含む文章
        toxic_text = "バカな考えだ。"
        toxicity_bad = quality_filter._evaluate_toxicity(toxic_text)
        
        assert 0.0 <= toxicity_good <= 1.0
        assert 0.0 <= toxicity_bad <= 1.0
        assert toxicity_good < toxicity_bad  # 毒性は低い方が良い
    
    def test_evaluate_bias(self, quality_filter):
        """バイアス評価テスト"""
        # 中立的な文章
        neutral_text = "多様な観点から検討することが重要です。"
        bias_good = quality_filter._evaluate_bias(neutral_text)
        
        # バイアスを含む文章
        biased_text = "男性はすべき、女性は当然こうに違いない。"
        bias_bad = quality_filter._evaluate_bias(biased_text)
        
        assert 0.0 <= bias_good <= 1.0
        assert 0.0 <= bias_bad <= 1.0
        assert bias_good < bias_bad  # バイアスは低い方が良い
    
    def test_calculate_overall_score(self, quality_filter):
        """総合スコア計算テスト"""
        score = QualityScore(
            fluency=0.8,
            coherence=0.7,
            diversity=0.6,
            relevance=0.9,
            toxicity=0.1,  # 低い方が良い
            bias=0.2       # 低い方が良い
        )
        
        overall = quality_filter._calculate_overall_score(score)
        
        assert 0.0 <= overall <= 1.0
        
        # 手動計算で確認
        expected = (
            0.25 * 0.8 +  # fluency
            0.20 * 0.7 +  # coherence
            0.20 * 0.6 +  # diversity
            0.15 * 0.9 +  # relevance
            0.10 * (1.0 - 0.1) +  # toxicity (reversed)
            0.10 * (1.0 - 0.2)    # bias (reversed)
        )
        
        assert abs(overall - expected) < 0.01
    
    def test_evaluate_quality_complete(self, quality_filter):
        """完全な品質評価テスト"""
        sample = GeneratedSample(
            text="これは技術革新について説明する適切な文章です。AIとデータサイエンスの発展により、社会に大きな変化をもたらしています。",
            metadata={"domain": "technology"}
        )
        
        quality_score = quality_filter.evaluate_quality(sample)
        
        assert isinstance(quality_score, QualityScore)
        assert 0.0 <= quality_score.fluency <= 1.0
        assert 0.0 <= quality_score.coherence <= 1.0
        assert 0.0 <= quality_score.diversity <= 1.0
        assert 0.0 <= quality_score.relevance <= 1.0
        assert 0.0 <= quality_score.toxicity <= 1.0
        assert 0.0 <= quality_score.bias <= 1.0
        assert 0.0 <= quality_score.overall <= 1.0


class TestDataClasses:
    """データクラスのテスト"""
    
    def test_generation_config(self):
        """GenerationConfig テスト"""
        config = GenerationConfig(
            data_type=DataType.TEXT_GENERATION,
            strategy=GenerationStrategy.TEMPLATE_BASED,
            num_samples=100,
            max_length=256,
            temperature=0.8,
            quality_threshold=0.7,
            domain="technology"
        )
        
        assert config.data_type == DataType.TEXT_GENERATION
        assert config.strategy == GenerationStrategy.TEMPLATE_BASED
        assert config.num_samples == 100
        assert config.max_length == 256
        assert config.temperature == 0.8
        assert config.quality_threshold == 0.7
        assert config.domain == "technology"
    
    def test_quality_score(self):
        """QualityScore テスト"""
        score = QualityScore(
            fluency=0.8,
            coherence=0.7,
            diversity=0.9,
            relevance=0.6,
            toxicity=0.1,
            bias=0.2,
            overall=0.75
        )
        
        assert score.fluency == 0.8
        assert score.coherence == 0.7
        assert score.diversity == 0.9
        assert score.relevance == 0.6
        assert score.toxicity == 0.1
        assert score.bias == 0.2
        assert score.overall == 0.75
        
        # to_dict テスト
        score_dict = score.to_dict()
        assert isinstance(score_dict, dict)
        assert len(score_dict) == 7
        assert score_dict["fluency"] == 0.8
        assert score_dict["overall"] == 0.75
    
    def test_generated_sample(self):
        """GeneratedSample テスト"""
        quality_score = QualityScore(overall=0.8)
        
        sample = GeneratedSample(
            text="Generated text",
            label="positive",
            metadata={"domain": "technology"},
            quality_score=quality_score,
            generation_method="template_based",
            template_used="Template {topic}"
        )
        
        assert sample.text == "Generated text"
        assert sample.label == "positive"
        assert sample.metadata == {"domain": "technology"}
        assert sample.quality_score == quality_score
        assert sample.generation_method == "template_based"
        assert sample.template_used == "Template {topic}"
        assert isinstance(sample.timestamp, datetime)
    
    def test_generation_result(self):
        """GenerationResult テスト"""
        config = GenerationConfig(
            data_type=DataType.TEXT_GENERATION,
            strategy=GenerationStrategy.TEMPLATE_BASED
        )
        
        samples = [
            GeneratedSample(text="Sample 1"),
            GeneratedSample(text="Sample 2")
        ]
        
        result = GenerationResult(
            config=config,
            samples=samples,
            total_generated=10,
            filtered_count=2,
            final_count=8,
            average_quality=0.75,
            generation_time=5.0,
            filtering_time=1.0,
            total_time=6.0,
            success=True
        )
        
        assert result.config == config
        assert result.samples == samples
        assert result.total_generated == 10
        assert result.filtered_count == 2
        assert result.final_count == 8
        assert result.average_quality == 0.75
        assert result.generation_time == 5.0
        assert result.filtering_time == 1.0
        assert result.total_time == 6.0
        assert result.success is True
        assert isinstance(result.timestamp, datetime)


class TestEnums:
    """列挙型のテスト"""
    
    def test_data_type_enum(self):
        """DataType 列挙型テスト"""
        assert DataType.TEXT_GENERATION.value == "text_generation"
        assert DataType.TEXT_CLASSIFICATION.value == "text_classification"
        assert DataType.QUESTION_ANSWERING.value == "question_answering"
        assert DataType.SUMMARIZATION.value == "summarization"
        assert DataType.TRANSLATION.value == "translation"
        assert DataType.CONVERSATION.value == "conversation"
    
    def test_generation_strategy_enum(self):
        """GenerationStrategy 列挙型テスト"""
        assert GenerationStrategy.TEMPLATE_BASED.value == "template_based"
        assert GenerationStrategy.MODEL_BASED.value == "model_based"
        assert GenerationStrategy.HYBRID.value == "hybrid"
        assert GenerationStrategy.AUGMENTATION.value == "augmentation"
    
    def test_quality_metric_enum(self):
        """QualityMetric 列挙型テスト"""
        assert QualityMetric.FLUENCY.value == "fluency"
        assert QualityMetric.COHERENCE.value == "coherence"
        assert QualityMetric.DIVERSITY.value == "diversity"
        assert QualityMetric.RELEVANCE.value == "relevance"
        assert QualityMetric.TOXICITY.value == "toxicity"
        assert QualityMetric.BIAS.value == "bias"


class TestCreateSyntheticDataGenerator:
    """create_synthetic_data_generator 関数のテスト"""
    
    @pytest.mark.asyncio
    async def test_create_success(self):
        """作成成功テスト"""
        with patch('src.advanced_agent.evolution.synthetic_data_generator.SyntheticDataGenerator') as MockGenerator:
            mock_generator = Mock()
            mock_generator.initialize = AsyncMock(return_value=True)
            MockGenerator.return_value = mock_generator
            
            result = await create_synthetic_data_generator()
            
            assert result == mock_generator
            MockGenerator.assert_called_once_with(None, None)
            mock_generator.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_failure(self):
        """作成失敗テスト"""
        with patch('src.advanced_agent.evolution.synthetic_data_generator.SyntheticDataGenerator') as MockGenerator:
            mock_generator = Mock()
            mock_generator.initialize = AsyncMock(return_value=False)
            MockGenerator.return_value = mock_generator
            
            with pytest.raises(RuntimeError, match="Failed to initialize synthetic data generator"):
                await create_synthetic_data_generator()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])