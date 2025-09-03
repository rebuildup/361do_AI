"""
HuggingFace Datasets 合成データ生成統合システム
HuggingFace Datasets による 自動データ生成パイプラインとTransformers Pipeline による 生成データ品質フィルタリング
"""

import asyncio
import time
import json
import random
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import numpy as np

try:
    from datasets import Dataset as HFDataset, DatasetDict, load_dataset
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, pipeline,
        TextGenerationPipeline, TextClassificationPipeline
    )
    import torch
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    HFDataset = None

from ..core.config import get_config
from ..core.logger import get_logger
from ..monitoring.system_monitor import SystemMonitor
from ..adaptation.peft_manager import PEFTAdapterPool


class DataType(Enum):
    """データタイプ"""
    TEXT_GENERATION = "text_generation"
    TEXT_CLASSIFICATION = "text_classification"
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CONVERSATION = "conversation"


class GenerationStrategy(Enum):
    """生成戦略"""
    TEMPLATE_BASED = "template_based"
    MODEL_BASED = "model_based"
    HYBRID = "hybrid"
    AUGMENTATION = "augmentation"


class QualityMetric(Enum):
    """品質指標"""
    FLUENCY = "fluency"
    COHERENCE = "coherence"
    DIVERSITY = "diversity"
    RELEVANCE = "relevance"
    TOXICITY = "toxicity"
    BIAS = "bias"


@dataclass
class GenerationConfig:
    """生成設定"""
    data_type: DataType
    strategy: GenerationStrategy
    
    # 生成パラメータ
    num_samples: int = 1000
    max_length: int = 512
    min_length: int = 10
    
    # モデル生成設定
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    
    # 品質フィルタ設定
    quality_threshold: float = 0.7
    diversity_threshold: float = 0.5
    enable_quality_filter: bool = True
    
    # テンプレート設定
    templates: List[str] = field(default_factory=list)
    seed_data: Optional[HFDataset] = None
    
    # ドメイン設定
    domain: str = "general"
    language: str = "ja"
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityScore:
    """品質スコア"""
    fluency: float = 0.0
    coherence: float = 0.0
    diversity: float = 0.0
    relevance: float = 0.0
    toxicity: float = 0.0
    bias: float = 0.0
    overall: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "fluency": self.fluency,
            "coherence": self.coherence,
            "diversity": self.diversity,
            "relevance": self.relevance,
            "toxicity": self.toxicity,
            "bias": self.bias,
            "overall": self.overall
        }


@dataclass
class GeneratedSample:
    """生成サンプル"""
    text: str
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: Optional[QualityScore] = None
    generation_method: str = ""
    template_used: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class GenerationResult:
    """生成結果"""
    config: GenerationConfig
    
    # 生成データ
    samples: List[GeneratedSample] = field(default_factory=list)
    dataset: Optional[HFDataset] = None
    
    # 統計
    total_generated: int = 0
    filtered_count: int = 0
    final_count: int = 0
    
    # 品質統計
    average_quality: float = 0.0
    quality_distribution: Dict[str, int] = field(default_factory=dict)
    
    # 時間統計
    generation_time: float = 0.0
    filtering_time: float = 0.0
    total_time: float = 0.0
    
    # 結果
    success: bool = True
    error_message: Optional[str] = None
    
    timestamp: datetime = field(default_factory=datetime.now)


class QualityFilter:
    """品質フィルタ"""
    
    def __init__(self):
        self.fluency_pipeline = None
        self.toxicity_pipeline = None
        self.diversity_cache = set()
        
    async def initialize(self) -> bool:
        """フィルタ初期化"""
        try:
            if not DATASETS_AVAILABLE:
                return False
            
            # 流暢性評価パイプライン（言語モデルベース）
            # 実際の実装では適切なモデルを使用
            # self.fluency_pipeline = pipeline("text-classification", model="fluency-model")
            
            # 毒性検出パイプライン
            # self.toxicity_pipeline = pipeline("text-classification", model="toxicity-model")
            
            return True
            
        except Exception:
            return False
    
    def evaluate_quality(self, sample: GeneratedSample) -> QualityScore:
        """品質評価"""
        
        score = QualityScore()
        
        # 流暢性評価（簡易実装）
        score.fluency = self._evaluate_fluency(sample.text)
        
        # 一貫性評価
        score.coherence = self._evaluate_coherence(sample.text)
        
        # 多様性評価
        score.diversity = self._evaluate_diversity(sample.text)
        
        # 関連性評価
        score.relevance = self._evaluate_relevance(sample.text, sample.metadata)
        
        # 毒性評価
        score.toxicity = self._evaluate_toxicity(sample.text)
        
        # バイアス評価
        score.bias = self._evaluate_bias(sample.text)
        
        # 総合スコア計算
        score.overall = self._calculate_overall_score(score)
        
        return score
    
    def _evaluate_fluency(self, text: str) -> float:
        """流暢性評価"""
        # 簡易実装：文の長さと構造をチェック
        sentences = text.split('。')
        if not sentences:
            return 0.0
        
        # 平均文長
        avg_length = sum(len(s.strip()) for s in sentences if s.strip()) / len([s for s in sentences if s.strip()])
        
        # 適切な文長範囲（10-100文字）
        if 10 <= avg_length <= 100:
            fluency = 0.8
        elif avg_length < 10:
            fluency = 0.3
        else:
            fluency = 0.6
        
        # 句読点の使用
        if '、' in text or '。' in text:
            fluency += 0.1
        
        return min(1.0, fluency)
    
    def _evaluate_coherence(self, text: str) -> float:
        """一貫性評価"""
        # 簡易実装：文の接続性をチェック
        sentences = [s.strip() for s in text.split('。') if s.strip()]
        
        if len(sentences) < 2:
            return 0.5
        
        # 接続詞の使用
        connectors = ['しかし', 'そして', 'また', 'さらに', 'つまり', 'したがって']
        connector_count = sum(1 for conn in connectors if conn in text)
        
        coherence = 0.5 + min(0.4, connector_count * 0.1)
        
        return coherence
    
    def _evaluate_diversity(self, text: str) -> float:
        """多様性評価"""
        # 既存テキストとの類似度チェック
        text_hash = hash(text.lower().replace(' ', ''))
        
        if text_hash in self.diversity_cache:
            return 0.0  # 重複
        
        self.diversity_cache.add(text_hash)
        
        # 語彙の多様性
        words = text.split()
        unique_words = set(words)
        
        if len(words) == 0:
            return 0.0
        
        diversity = len(unique_words) / len(words)
        return min(1.0, diversity * 1.5)  # 調整係数
    
    def _evaluate_relevance(self, text: str, metadata: Dict[str, Any]) -> float:
        """関連性評価"""
        # メタデータに基づく関連性チェック
        domain = metadata.get('domain', 'general')
        
        # ドメイン固有キーワード
        domain_keywords = {
            'technology': ['技術', 'システム', 'データ', 'AI', 'コンピュータ'],
            'business': ['ビジネス', '企業', '売上', '利益', '戦略'],
            'science': ['研究', '実験', '理論', '分析', '発見'],
            'general': ['問題', '解決', '方法', '結果', '効果']
        }
        
        keywords = domain_keywords.get(domain, domain_keywords['general'])
        keyword_count = sum(1 for keyword in keywords if keyword in text)
        
        relevance = min(1.0, keyword_count / len(keywords) * 2)
        return relevance
    
    def _evaluate_toxicity(self, text: str) -> float:
        """毒性評価（低いほど良い）"""
        # 簡易実装：有害語句チェック
        toxic_words = ['バカ', 'アホ', '死ね', 'クソ']  # 実際はより包括的なリスト
        
        toxic_count = sum(1 for word in toxic_words if word in text)
        
        # 毒性スコア（0が最良、1が最悪）
        toxicity = min(1.0, toxic_count * 0.5)
        
        return toxicity
    
    def _evaluate_bias(self, text: str) -> float:
        """バイアス評価（低いほど良い）"""
        # 簡易実装：性別・年齢・職業バイアスチェック
        bias_indicators = [
            '男性は', '女性は', '高齢者は', '若者は',
            '〜すべき', '〜に違いない', '当然〜'
        ]
        
        bias_count = sum(1 for indicator in bias_indicators if indicator in text)
        
        # バイアススコア（0が最良、1が最悪）
        bias = min(1.0, bias_count * 0.3)
        
        return bias
    
    def _calculate_overall_score(self, score: QualityScore) -> float:
        """総合スコア計算"""
        # 重み付き平均（毒性とバイアスは逆転）
        weights = {
            'fluency': 0.25,
            'coherence': 0.20,
            'diversity': 0.20,
            'relevance': 0.15,
            'toxicity': 0.10,  # 低い方が良い
            'bias': 0.10       # 低い方が良い
        }
        
        overall = (
            weights['fluency'] * score.fluency +
            weights['coherence'] * score.coherence +
            weights['diversity'] * score.diversity +
            weights['relevance'] * score.relevance +
            weights['toxicity'] * (1.0 - score.toxicity) +  # 逆転
            weights['bias'] * (1.0 - score.bias)            # 逆転
        )
        
        return overall


class SyntheticDataGenerator:
    """HuggingFace Datasets 合成データ生成システム"""
    
    def __init__(
        self,
        peft_pool: Optional[PEFTAdapterPool] = None,
        system_monitor: Optional[SystemMonitor] = None
    ):
        self.peft_pool = peft_pool
        self.system_monitor = system_monitor
        
        self.config = get_config()
        self.logger = get_logger()
        
        # 生成パイプライン
        self.generation_pipeline = None
        self.tokenizer = None
        
        # 品質フィルタ
        self.quality_filter = QualityFilter()
        
        # テンプレート
        self.templates = self._load_default_templates()
        
        # 生成履歴
        self.generation_history: List[GenerationResult] = []
        
        # 統計
        self.generation_stats = {
            "total_generations": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "total_samples_generated": 0,
            "total_samples_filtered": 0,
            "average_quality": 0.0
        }
        
        self.logger.log_startup(
            component="synthetic_data_generator",
            version="1.0.0",
            config_summary={
                "datasets_available": DATASETS_AVAILABLE,
                "templates_loaded": len(self.templates)
            }
        )
    
    async def initialize(self) -> bool:
        """データ生成システム初期化"""
        try:
            if not DATASETS_AVAILABLE:
                self.logger.log_alert(
                    alert_type="datasets_unavailable",
                    severity="WARNING",
                    message="HuggingFace Datasets not available"
                )
                return False
            
            # 品質フィルタ初期化
            await self.quality_filter.initialize()
            
            self.logger.log_startup(
                component="synthetic_data_generator_initialized",
                version="1.0.0",
                config_summary={
                    "initialization_complete": True
                }
            )
            
            return True
            
        except Exception as e:
            self.logger.log_alert(
                alert_type="data_generator_initialization_failed",
                severity="ERROR",
                message=f"Synthetic data generator initialization failed: {e}"
            )
            return False   
 
    def _load_default_templates(self) -> Dict[DataType, List[str]]:
        """デフォルトテンプレート読み込み"""
        
        templates = {
            DataType.TEXT_GENERATION: [
                "{topic}について説明してください。",
                "{topic}の特徴を教えてください。",
                "{topic}はなぜ重要なのですか？",
                "{topic}の利点と欠点を比較してください。",
                "{topic}の将来性について考察してください。"
            ],
            
            DataType.TEXT_CLASSIFICATION: [
                "この文章は{category}に分類されます：{text}",
                "以下のテキストは{category}カテゴリです：{text}",
                "{category}の例：{text}",
                "カテゴリ{category}：{text}"
            ],
            
            DataType.QUESTION_ANSWERING: [
                "質問：{question}\n回答：{answer}",
                "Q: {question}\nA: {answer}",
                "{question}について教えてください。\n{answer}",
                "{question}の答えは{answer}です。"
            ],
            
            DataType.SUMMARIZATION: [
                "以下の文章を要約してください：\n{text}\n要約：{summary}",
                "要約：{summary}\n元の文章：{text}",
                "{text}\n上記の要約：{summary}"
            ],
            
            DataType.CONVERSATION: [
                "ユーザー：{user_message}\nアシスタント：{assistant_message}",
                "人間：{user_message}\nAI：{assistant_message}",
                "質問者：{user_message}\n回答者：{assistant_message}"
            ]
        }
        
        return templates
    
    async def generate_synthetic_data(
        self,
        generation_config: GenerationConfig
    ) -> GenerationResult:
        """合成データ生成実行"""
        
        start_time = time.time()
        
        try:
            # 生成パイプライン準備
            await self._prepare_generation_pipeline(generation_config)
            
            # データ生成
            generation_start = time.time()
            samples = await self._generate_samples(generation_config)
            generation_time = time.time() - generation_start
            
            # 品質フィルタリング
            filtering_start = time.time()
            filtered_samples = await self._filter_samples(samples, generation_config)
            filtering_time = time.time() - filtering_start
            
            # データセット作成
            dataset = self._create_dataset(filtered_samples, generation_config)
            
            # 品質統計計算
            quality_stats = self._calculate_quality_stats(filtered_samples)
            
            # 結果作成
            result = GenerationResult(
                config=generation_config,
                samples=filtered_samples,
                dataset=dataset,
                total_generated=len(samples),
                filtered_count=len(samples) - len(filtered_samples),
                final_count=len(filtered_samples),
                average_quality=quality_stats["average"],
                quality_distribution=quality_stats["distribution"],
                generation_time=generation_time,
                filtering_time=filtering_time,
                total_time=time.time() - start_time,
                success=True
            )
            
            # 統計更新
            self.generation_stats["total_generations"] += 1
            self.generation_stats["successful_generations"] += 1
            self.generation_stats["total_samples_generated"] += len(samples)
            self.generation_stats["total_samples_filtered"] += len(filtered_samples)
            self._update_generation_stats(result)
            
            # 履歴に追加
            self.generation_history.append(result)
            
            self.logger.log_performance_metric(
                metric_name="synthetic_data_generation_completed",
                value=result.total_time,
                unit="seconds",
                component="synthetic_data_generator"
            )
            
            return result
            
        except Exception as e:
            # エラー結果作成
            error_result = GenerationResult(
                config=generation_config,
                total_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
            
            # 統計更新
            self.generation_stats["total_generations"] += 1
            self.generation_stats["failed_generations"] += 1
            
            # 履歴に追加
            self.generation_history.append(error_result)
            
            self.logger.log_alert(
                alert_type="synthetic_data_generation_failed",
                severity="ERROR",
                message=f"Synthetic data generation failed: {e}"
            )
            
            raise
    
    async def _prepare_generation_pipeline(self, config: GenerationConfig) -> None:
        """生成パイプライン準備"""
        
        if config.strategy == GenerationStrategy.MODEL_BASED:
            # モデルベース生成の場合
            if self.peft_pool and self.peft_pool.active_adapters:
                # PEFTアダプタを使用
                adapter_name = list(self.peft_pool.active_adapters.keys())[0]
                model = self.peft_pool.active_adapters[adapter_name]
                tokenizer = self.peft_pool.tokenizer
            else:
                # デフォルトモデル使用
                model_name = "microsoft/DialoGPT-medium"  # 軽量モデル
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)
            
            self.generation_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            self.tokenizer = tokenizer
    
    async def _generate_samples(self, config: GenerationConfig) -> List[GeneratedSample]:
        """サンプル生成"""
        
        samples = []
        
        if config.strategy == GenerationStrategy.TEMPLATE_BASED:
            samples = await self._generate_template_based(config)
        elif config.strategy == GenerationStrategy.MODEL_BASED:
            samples = await self._generate_model_based(config)
        elif config.strategy == GenerationStrategy.HYBRID:
            template_samples = await self._generate_template_based(config)
            model_samples = await self._generate_model_based(config)
            samples = template_samples + model_samples
        elif config.strategy == GenerationStrategy.AUGMENTATION:
            samples = await self._generate_augmentation_based(config)
        
        return samples
    
    async def _generate_template_based(self, config: GenerationConfig) -> List[GeneratedSample]:
        """テンプレートベース生成"""
        
        samples = []
        templates = config.templates or self.templates.get(config.data_type, [])
        
        if not templates:
            return samples
        
        # サンプル生成
        for i in range(config.num_samples):
            template = random.choice(templates)
            
            # テンプレート変数を置換
            filled_template = self._fill_template(template, config)
            
            sample = GeneratedSample(
                text=filled_template,
                metadata={
                    "domain": config.domain,
                    "language": config.language,
                    "generation_method": "template"
                },
                generation_method="template_based",
                template_used=template
            )
            
            samples.append(sample)
        
        return samples
    
    async def _generate_model_based(self, config: GenerationConfig) -> List[GeneratedSample]:
        """モデルベース生成"""
        
        if not self.generation_pipeline:
            return []
        
        samples = []
        
        # プロンプト生成
        prompts = self._generate_prompts(config)
        
        for prompt in prompts[:config.num_samples]:
            try:
                # テキスト生成
                generated = self.generation_pipeline(
                    prompt,
                    max_length=config.max_length,
                    min_length=config.min_length,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    do_sample=config.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
                
                generated_text = generated[0]["generated_text"]
                
                # プロンプト部分を除去
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                
                sample = GeneratedSample(
                    text=generated_text,
                    metadata={
                        "domain": config.domain,
                        "language": config.language,
                        "prompt": prompt,
                        "generation_method": "model"
                    },
                    generation_method="model_based"
                )
                
                samples.append(sample)
                
            except Exception as e:
                self.logger.log_alert(
                    alert_type="model_generation_failed",
                    severity="WARNING",
                    message=f"Model generation failed for prompt '{prompt}': {e}"
                )
        
        return samples
    
    async def _generate_augmentation_based(self, config: GenerationConfig) -> List[GeneratedSample]:
        """拡張ベース生成"""
        
        if not config.seed_data:
            return []
        
        samples = []
        
        # シードデータから拡張
        for i, seed_sample in enumerate(config.seed_data):
            if i >= config.num_samples:
                break
            
            # データ拡張技術適用
            augmented_texts = self._augment_text(
                seed_sample.get("text", ""),
                config
            )
            
            for aug_text in augmented_texts:
                sample = GeneratedSample(
                    text=aug_text,
                    label=seed_sample.get("label"),
                    metadata={
                        "domain": config.domain,
                        "language": config.language,
                        "original_text": seed_sample.get("text", ""),
                        "generation_method": "augmentation"
                    },
                    generation_method="augmentation_based"
                )
                
                samples.append(sample)
        
        return samples
    
    def _fill_template(self, template: str, config: GenerationConfig) -> str:
        """テンプレート変数埋め込み"""
        
        # ドメイン固有の変数辞書
        variables = {
            "technology": {
                "topic": random.choice(["AI", "機械学習", "データサイエンス", "クラウド", "IoT"]),
                "category": random.choice(["技術", "IT", "革新"]),
                "question": random.choice(["AIとは何ですか？", "機械学習の仕組みは？", "データの重要性は？"]),
                "answer": random.choice(["人工知能技術です", "データから学習する技術です", "意思決定に重要です"])
            },
            "business": {
                "topic": random.choice(["マーケティング", "経営戦略", "財務", "人事", "営業"]),
                "category": random.choice(["ビジネス", "経営", "戦略"]),
                "question": random.choice(["売上向上の方法は？", "効率的な経営とは？", "チームワークの重要性は？"]),
                "answer": random.choice(["顧客満足度の向上です", "リソースの最適化です", "協力が成功の鍵です"])
            },
            "general": {
                "topic": random.choice(["教育", "健康", "環境", "社会", "文化"]),
                "category": random.choice(["一般", "社会", "生活"]),
                "question": random.choice(["健康的な生活とは？", "環境保護の方法は？", "教育の意義は？"]),
                "answer": random.choice(["バランスの取れた生活です", "持続可能な行動です", "知識と成長です"])
            }
        }
        
        domain_vars = variables.get(config.domain, variables["general"])
        
        # テンプレート変数を置換
        filled = template
        for var, value in domain_vars.items():
            filled = filled.replace(f"{{{var}}}", value)
        
        return filled
    
    def _generate_prompts(self, config: GenerationConfig) -> List[str]:
        """プロンプト生成"""
        
        prompts = []
        
        # データタイプ別プロンプト
        if config.data_type == DataType.TEXT_GENERATION:
            base_prompts = [
                f"{config.domain}について詳しく説明してください。",
                f"{config.domain}の重要なポイントを教えてください。",
                f"{config.domain}に関する興味深い事実を紹介してください。"
            ]
        elif config.data_type == DataType.CONVERSATION:
            base_prompts = [
                f"{config.domain}について質問があります。",
                f"{config.domain}で困っています。助けてください。",
                f"{config.domain}の専門家として回答してください。"
            ]
        else:
            base_prompts = [
                f"{config.domain}に関する内容を生成してください。",
                f"{config.domain}について書いてください。"
            ]
        
        # プロンプトを複製して数を増やす
        while len(prompts) < config.num_samples:
            prompts.extend(base_prompts)
        
        return prompts[:config.num_samples]
    
    def _augment_text(self, text: str, config: GenerationConfig) -> List[str]:
        """テキスト拡張"""
        
        augmented = []
        
        # 同義語置換
        augmented.append(self._synonym_replacement(text))
        
        # 文の順序変更
        augmented.append(self._sentence_reorder(text))
        
        # パラフレーズ
        augmented.append(self._paraphrase(text))
        
        return [aug for aug in augmented if aug and aug != text]
    
    def _synonym_replacement(self, text: str) -> str:
        """同義語置換"""
        # 簡易実装：基本的な同義語辞書
        synonyms = {
            "重要": "大切",
            "問題": "課題",
            "解決": "解消",
            "方法": "手段",
            "効果": "成果",
            "技術": "テクノロジー",
            "システム": "仕組み"
        }
        
        result = text
        for original, synonym in synonyms.items():
            if original in result:
                result = result.replace(original, synonym)
                break  # 1つだけ置換
        
        return result
    
    def _sentence_reorder(self, text: str) -> str:
        """文の順序変更"""
        sentences = [s.strip() for s in text.split('。') if s.strip()]
        
        if len(sentences) > 1:
            random.shuffle(sentences)
            return '。'.join(sentences) + '。'
        
        return text
    
    def _paraphrase(self, text: str) -> str:
        """パラフレーズ"""
        # 簡易実装：文体変更
        if text.endswith('です。'):
            return text.replace('です。', 'である。')
        elif text.endswith('である。'):
            return text.replace('である。', 'です。')
        
        return text
    
    async def _filter_samples(
        self,
        samples: List[GeneratedSample],
        config: GenerationConfig
    ) -> List[GeneratedSample]:
        """サンプルフィルタリング"""
        
        if not config.enable_quality_filter:
            return samples
        
        filtered_samples = []
        
        for sample in samples:
            # 品質評価
            quality_score = self.quality_filter.evaluate_quality(sample)
            sample.quality_score = quality_score
            
            # 閾値チェック
            if quality_score.overall >= config.quality_threshold:
                filtered_samples.append(sample)
        
        return filtered_samples
    
    def _create_dataset(
        self,
        samples: List[GeneratedSample],
        config: GenerationConfig
    ) -> HFDataset:
        """データセット作成"""
        
        if not samples:
            return HFDataset.from_dict({"text": [], "label": []})
        
        # データ構造準備
        data_dict = {
            "text": [sample.text for sample in samples],
            "generation_method": [sample.generation_method for sample in samples],
            "timestamp": [sample.timestamp.isoformat() for sample in samples]
        }
        
        # ラベルがある場合
        if any(sample.label for sample in samples):
            data_dict["label"] = [sample.label or "" for sample in samples]
        
        # 品質スコアがある場合
        if any(sample.quality_score for sample in samples):
            data_dict["quality_score"] = [
                sample.quality_score.overall if sample.quality_score else 0.0
                for sample in samples
            ]
        
        return HFDataset.from_dict(data_dict)
    
    def _calculate_quality_stats(self, samples: List[GeneratedSample]) -> Dict[str, Any]:
        """品質統計計算"""
        
        if not samples or not any(sample.quality_score for sample in samples):
            return {"average": 0.0, "distribution": {}}
        
        quality_scores = [
            sample.quality_score.overall
            for sample in samples
            if sample.quality_score
        ]
        
        if not quality_scores:
            return {"average": 0.0, "distribution": {}}
        
        # 平均品質
        average_quality = sum(quality_scores) / len(quality_scores)
        
        # 品質分布
        distribution = {
            "high": sum(1 for score in quality_scores if score >= 0.8),
            "medium": sum(1 for score in quality_scores if 0.5 <= score < 0.8),
            "low": sum(1 for score in quality_scores if score < 0.5)
        }
        
        return {
            "average": average_quality,
            "distribution": distribution
        }
    
    def _update_generation_stats(self, result: GenerationResult) -> None:
        """生成統計更新"""
        
        if result.success:
            # 平均品質更新
            successful_count = self.generation_stats["successful_generations"]
            current_avg = self.generation_stats["average_quality"]
            
            new_avg = ((current_avg * (successful_count - 1)) + result.average_quality) / successful_count
            self.generation_stats["average_quality"] = new_avg
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """生成統計取得"""
        stats = self.generation_stats.copy()
        
        # 成功率計算
        if stats["total_generations"] > 0:
            stats["success_rate"] = stats["successful_generations"] / stats["total_generations"]
        else:
            stats["success_rate"] = 0.0
        
        # フィルタ率計算
        if stats["total_samples_generated"] > 0:
            stats["filter_rate"] = stats["total_samples_filtered"] / stats["total_samples_generated"]
        else:
            stats["filter_rate"] = 0.0
        
        return stats
    
    def get_generation_history(
        self,
        limit: Optional[int] = None,
        success_only: bool = False
    ) -> List[GenerationResult]:
        """生成履歴取得"""
        
        history = self.generation_history
        
        if success_only:
            history = [r for r in history if r.success]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    async def shutdown(self) -> None:
        """データ生成システム終了"""
        
        final_stats = self.get_generation_stats()
        
        self.logger.log_shutdown(
            component="synthetic_data_generator",
            uptime_seconds=0,  # TODO: 実際の稼働時間計算
            final_stats=final_stats
        )


# 便利関数
async def create_synthetic_data_generator(
    peft_pool: Optional[PEFTAdapterPool] = None,
    system_monitor: Optional[SystemMonitor] = None
) -> SyntheticDataGenerator:
    """合成データ生成システム作成・初期化"""
    
    generator = SyntheticDataGenerator(peft_pool, system_monitor)
    
    if await generator.initialize():
        return generator
    else:
        raise RuntimeError("Failed to initialize synthetic data generator")


# 使用例
async def main():
    """テスト用メイン関数"""
    try:
        # データ生成システム作成
        generator = await create_synthetic_data_generator()
        
        print("=== Synthetic Data Generator Test ===")
        
        # 生成設定作成
        generation_config = GenerationConfig(
            data_type=DataType.TEXT_GENERATION,
            strategy=GenerationStrategy.TEMPLATE_BASED,
            num_samples=10,
            domain="technology",
            quality_threshold=0.5
        )
        
        print(f"Generation Config: {generation_config}")
        
        # データ生成実行
        result = await generator.generate_synthetic_data(generation_config)
        
        print(f"Generation Result:")
        print(f"  Total Generated: {result.total_generated}")
        print(f"  Final Count: {result.final_count}")
        print(f"  Average Quality: {result.average_quality:.2f}")
        print(f"  Generation Time: {result.generation_time:.2f}s")
        
        # 統計取得
        stats = generator.get_generation_stats()
        print(f"Generation Stats: {stats}")
        
        await generator.shutdown()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())