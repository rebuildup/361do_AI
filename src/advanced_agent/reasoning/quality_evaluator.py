"""
Quality evaluation system for reasoning responses
推論レスポンス品質評価システム
"""

import re
import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import statistics

from .cot_engine import CoTResponse, ReasoningStep, CoTStep

logger = logging.getLogger(__name__)


class QualityDimension(Enum):
    """品質評価次元"""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    LOGICAL_CONSISTENCY = "logical_consistency"
    USEFULNESS = "usefulness"
    EFFICIENCY = "efficiency"
    CREATIVITY = "creativity"
    SAFETY = "safety"


@dataclass
class QualityScore:
    """品質スコア"""
    dimension: QualityDimension
    score: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0
    explanation: str
    evidence: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class QualityEvaluation:
    """品質評価結果"""
    overall_score: float
    dimension_scores: Dict[QualityDimension, QualityScore]
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    evaluation_time: float
    evaluator_version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)


class QualityEvaluator:
    """推論品質評価器"""
    
    def __init__(self):
        self.evaluation_rules = self._initialize_evaluation_rules()
        self.quality_thresholds = {
            QualityDimension.ACCURACY: 0.7,
            QualityDimension.COMPLETENESS: 0.6,
            QualityDimension.CLARITY: 0.6,
            QualityDimension.LOGICAL_CONSISTENCY: 0.7,
            QualityDimension.USEFULNESS: 0.6,
            QualityDimension.EFFICIENCY: 0.5,
            QualityDimension.CREATIVITY: 0.4,
            QualityDimension.SAFETY: 0.8
        }
        
        # 評価メトリクス
        self.evaluation_metrics = {
            "total_evaluations": 0,
            "average_scores": {},
            "dimension_performance": {},
            "evaluation_times": []
        }
        
        logger.info("Quality evaluator initialized")
    
    def _initialize_evaluation_rules(self) -> Dict[QualityDimension, Dict[str, Any]]:
        """評価ルール初期化"""
        return {
            QualityDimension.ACCURACY: {
                "keywords": ["正確", "事実", "データ", "根拠", "証拠"],
                "negative_keywords": ["間違い", "誤り", "不正確", "推測", "憶測"],
                "patterns": [
                    r"\d+",  # 数値の存在
                    r"なぜなら|したがって|そのため|because|therefore",  # 論理的接続詞
                ]
            },
            QualityDimension.COMPLETENESS: {
                "required_elements": ["問題の理解", "解決手順", "結論"],
                "length_thresholds": {"min": 100, "optimal": 500},
                "coverage_indicators": ["すべて", "全体", "包括的", "詳細"]
            },
            QualityDimension.CLARITY: {
                "sentence_length_threshold": 50,
                "complexity_indicators": ["つまり", "要するに", "簡単に言うと"],
                "structure_indicators": ["1.", "2.", "3.", "•", "-", "・"]
            },
            QualityDimension.LOGICAL_CONSISTENCY: {
                "logical_connectors": ["なぜなら", "したがって", "そのため", "一方で", "しかし"],
                "contradiction_indicators": ["しかし", "一方で", "逆に", "反対に"],
                "reasoning_patterns": ["if-then", "cause-effect", "premise-conclusion"]
            },
            QualityDimension.USEFULNESS: {
                "actionable_indicators": ["方法", "手順", "ステップ", "コツ", "ポイント"],
                "practical_keywords": ["実用的", "役立つ", "参考", "活用"],
                "user_benefit_indicators": ["理解", "解決", "改善", "向上"]
            },
            QualityDimension.EFFICIENCY: {
                "conciseness_indicators": ["簡潔", "要点", "まとめ"],
                "redundancy_penalty": 0.1,
                "optimal_length_range": (200, 800)
            },
            QualityDimension.CREATIVITY: {
                "creative_indicators": ["新しい", "独創的", "革新的", "創造的"],
                "alternative_indicators": ["別の方法", "代替案", "異なる視点"],
                "innovation_keywords": ["アイデア", "発想", "工夫", "改善"]
            },
            QualityDimension.SAFETY: {
                "safety_keywords": ["安全", "注意", "危険", "リスク"],
                "harmful_indicators": ["危険", "有害", "違法", "不適切"],
                "ethical_indicators": ["倫理", "道徳", "責任", "配慮"]
            }
        }
    
    def evaluate_response(self, 
                         response: CoTResponse,
                         question: str,
                         context: Optional[Dict[str, Any]] = None) -> QualityEvaluation:
        """推論レスポンスの品質評価"""
        
        start_time = time.time()
        
        try:
            # 各次元の評価
            dimension_scores = {}
            for dimension in QualityDimension:
                score = self._evaluate_dimension(dimension, response, question, context)
                dimension_scores[dimension] = score
            
            # 総合スコア計算
            overall_score = self._calculate_overall_score(dimension_scores)
            
            # 強み・弱み・推奨事項の分析
            strengths, weaknesses, recommendations = self._analyze_quality_aspects(
                dimension_scores, response, question
            )
            
            evaluation_time = time.time() - start_time
            
            # 評価メトリクス更新
            self._update_evaluation_metrics(dimension_scores, evaluation_time)
            
            evaluation = QualityEvaluation(
                overall_score=overall_score,
                dimension_scores=dimension_scores,
                strengths=strengths,
                weaknesses=weaknesses,
                recommendations=recommendations,
                evaluation_time=evaluation_time,
                metadata={
                    "question": question,
                    "response_length": len(response.response_text),
                    "step_count": response.step_count,
                    "processing_time": response.processing_time,
                    "confidence": response.final_confidence
                }
            )
            
            logger.info(f"Quality evaluation completed: {overall_score:.3f} in {evaluation_time:.3f}s")
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Quality evaluation failed: {e}")
            return self._create_error_evaluation(str(e), time.time() - start_time)
    
    def _evaluate_dimension(self, 
                           dimension: QualityDimension,
                           response: CoTResponse,
                           question: str,
                           context: Optional[Dict[str, Any]]) -> QualityScore:
        """特定次元の品質評価"""
        
        rules = self.evaluation_rules[dimension]
        text = response.response_text.lower()
        
        if dimension == QualityDimension.ACCURACY:
            return self._evaluate_accuracy(text, response, rules)
        elif dimension == QualityDimension.COMPLETENESS:
            return self._evaluate_completeness(text, response, rules, question)
        elif dimension == QualityDimension.CLARITY:
            return self._evaluate_clarity(text, response, rules)
        elif dimension == QualityDimension.LOGICAL_CONSISTENCY:
            return self._evaluate_logical_consistency(text, response, rules)
        elif dimension == QualityDimension.USEFULNESS:
            return self._evaluate_usefulness(text, response, rules, question)
        elif dimension == QualityDimension.EFFICIENCY:
            return self._evaluate_efficiency(text, response, rules)
        elif dimension == QualityDimension.CREATIVITY:
            return self._evaluate_creativity(text, response, rules)
        elif dimension == QualityDimension.SAFETY:
            return self._evaluate_safety(text, response, rules)
        else:
            return QualityScore(
                dimension=dimension,
                score=0.5,
                confidence=0.0,
                explanation="Unknown dimension"
            )
    
    def _evaluate_accuracy(self, text: str, response: CoTResponse, rules: Dict[str, Any]) -> QualityScore:
        """正確性評価"""
        score = 0.5
        evidence = []
        suggestions = []
        
        # 数値の存在チェック
        if re.search(r'\d+', text):
            score += 0.1
            evidence.append("数値データが含まれています")
        
        # 論理的接続詞の存在
        logical_connectors = re.findall(r'なぜなら|したがって|そのため|because|therefore', text)
        if logical_connectors:
            score += 0.2
            evidence.append(f"論理的接続詞が{len(logical_connectors)}個使用されています")
        
        # 推論ステップの存在
        if response.step_count >= 3:
            score += 0.2
            evidence.append(f"詳細な推論ステップ({response.step_count}ステップ)が含まれています")
        
        # 信頼度の考慮
        if response.final_confidence > 0.7:
            score += 0.1
            evidence.append("高い信頼度を示しています")
        
        # 改善提案
        if score < 0.7:
            suggestions.append("より具体的な数値やデータを含めてください")
            suggestions.append("論理的根拠を明確に示してください")
        
        return QualityScore(
            dimension=QualityDimension.ACCURACY,
            score=min(score, 1.0),
            confidence=0.8,
            explanation=f"正確性スコア: {score:.2f}",
            evidence=evidence,
            suggestions=suggestions
        )
    
    def _evaluate_completeness(self, text: str, response: CoTResponse, rules: Dict[str, Any], question: str) -> QualityScore:
        """完全性評価"""
        score = 0.5
        evidence = []
        suggestions = []
        
        # 必須要素のチェック
        required_elements = rules["required_elements"]
        found_elements = 0
        
        for element in required_elements:
            if element in text:
                found_elements += 1
                evidence.append(f"'{element}'が含まれています")
        
        score += (found_elements / len(required_elements)) * 0.3
        
        # 長さの評価
        text_length = len(text)
        length_thresholds = rules["length_thresholds"]
        
        if text_length >= length_thresholds["min"]:
            score += 0.1
            evidence.append(f"適切な長さ({text_length}文字)です")
        
        if text_length >= length_thresholds["optimal"]:
            score += 0.1
            evidence.append("詳細な説明が含まれています")
        
        # 推論ステップの完全性
        if response.step_count >= 5:
            score += 0.1
            evidence.append("十分な推論ステップが含まれています")
        
        # 改善提案
        if score < 0.6:
            suggestions.append("問題の理解、解決手順、結論を明確に示してください")
            suggestions.append("より詳細な説明を追加してください")
        
        return QualityScore(
            dimension=QualityDimension.COMPLETENESS,
            score=min(score, 1.0),
            confidence=0.7,
            explanation=f"完全性スコア: {score:.2f}",
            evidence=evidence,
            suggestions=suggestions
        )
    
    def _evaluate_clarity(self, text: str, response: CoTResponse, rules: Dict[str, Any]) -> QualityScore:
        """明確性評価"""
        score = 0.5
        evidence = []
        suggestions = []
        
        # 文の長さチェック
        sentences = text.split('。')
        long_sentences = [s for s in sentences if len(s) > rules["sentence_length_threshold"]]
        
        if len(long_sentences) / len(sentences) < 0.3:
            score += 0.2
            evidence.append("適切な文の長さです")
        else:
            suggestions.append("長すぎる文を短く分割してください")
        
        # 構造化の評価
        structure_indicators = rules["structure_indicators"]
        structure_count = sum(text.count(indicator) for indicator in structure_indicators)
        
        if structure_count > 0:
            score += 0.2
            evidence.append("構造化された説明です")
        
        # 簡潔性の評価
        complexity_indicators = rules["complexity_indicators"]
        if any(indicator in text for indicator in complexity_indicators):
            score += 0.1
            evidence.append("複雑な概念を簡潔に説明しています")
        
        # 改善提案
        if score < 0.6:
            suggestions.append("箇条書きや番号付きリストを使用してください")
            suggestions.append("一文を短くして読みやすくしてください")
        
        return QualityScore(
            dimension=QualityDimension.CLARITY,
            score=min(score, 1.0),
            confidence=0.7,
            explanation=f"明確性スコア: {score:.2f}",
            evidence=evidence,
            suggestions=suggestions
        )
    
    def _evaluate_logical_consistency(self, text: str, response: CoTResponse, rules: Dict[str, Any]) -> QualityScore:
        """論理的一貫性評価"""
        score = 0.5
        evidence = []
        suggestions = []
        
        # 論理的接続詞の使用
        logical_connectors = rules["logical_connectors"]
        connector_count = sum(text.count(connector) for connector in logical_connectors)
        
        if connector_count > 0:
            score += 0.2
            evidence.append(f"論理的接続詞が{connector_count}個使用されています")
        
        # 推論ステップの一貫性
        if response.step_count >= 3:
            # 各ステップの論理的つながりをチェック
            step_consistency = self._check_step_consistency(response.reasoning_steps)
            score += step_consistency * 0.2
            evidence.append("推論ステップが論理的に一貫しています")
        
        # 矛盾の検出
        contradiction_indicators = rules["contradiction_indicators"]
        contradiction_count = sum(text.count(indicator) for indicator in contradiction_indicators)
        
        if contradiction_count == 0:
            score += 0.1
            evidence.append("明らかな矛盾は見つかりませんでした")
        else:
            score -= 0.1
            suggestions.append("矛盾する内容がないか確認してください")
        
        # 改善提案
        if score < 0.7:
            suggestions.append("論理的接続詞を適切に使用してください")
            suggestions.append("推論ステップのつながりを明確にしてください")
        
        return QualityScore(
            dimension=QualityDimension.LOGICAL_CONSISTENCY,
            score=max(0.0, min(score, 1.0)),
            confidence=0.8,
            explanation=f"論理的一貫性スコア: {score:.2f}",
            evidence=evidence,
            suggestions=suggestions
        )
    
    def _evaluate_usefulness(self, text: str, response: CoTResponse, rules: Dict[str, Any], question: str) -> QualityScore:
        """有用性評価"""
        score = 0.5
        evidence = []
        suggestions = []
        
        # 実用的な要素の存在
        actionable_indicators = rules["actionable_indicators"]
        actionable_count = sum(text.count(indicator) for indicator in actionable_indicators)
        
        if actionable_count > 0:
            score += 0.2
            evidence.append("実用的な情報が含まれています")
        
        # ユーザー利益の考慮
        user_benefit_indicators = rules["user_benefit_indicators"]
        benefit_count = sum(text.count(indicator) for indicator in user_benefit_indicators)
        
        if benefit_count > 0:
            score += 0.2
            evidence.append("ユーザーの利益を考慮した内容です")
        
        # 質問への適切な回答
        if self._is_question_answered(text, question):
            score += 0.1
            evidence.append("質問に適切に答えています")
        
        # 改善提案
        if score < 0.6:
            suggestions.append("より実用的なアドバイスを含めてください")
            suggestions.append("ユーザーが活用できる具体的な方法を示してください")
        
        return QualityScore(
            dimension=QualityDimension.USEFULNESS,
            score=min(score, 1.0),
            confidence=0.6,
            explanation=f"有用性スコア: {score:.2f}",
            evidence=evidence,
            suggestions=suggestions
        )
    
    def _evaluate_efficiency(self, text: str, response: CoTResponse, rules: Dict[str, Any]) -> QualityScore:
        """効率性評価"""
        score = 0.5
        evidence = []
        suggestions = []
        
        # 長さの効率性
        text_length = len(text)
        optimal_range = rules["optimal_length_range"]
        
        if optimal_range[0] <= text_length <= optimal_range[1]:
            score += 0.3
            evidence.append("適切な長さで効率的です")
        elif text_length < optimal_range[0]:
            score += 0.1
            evidence.append("簡潔で効率的です")
        else:
            suggestions.append("より簡潔にまとめてください")
        
        # 冗長性のチェック
        redundancy_penalty = rules["redundancy_penalty"]
        if self._has_redundancy(text):
            score -= redundancy_penalty
            suggestions.append("重複する内容を削除してください")
        else:
            evidence.append("冗長性が少なく効率的です")
        
        # 処理時間の効率性
        if response.processing_time < 10.0:
            score += 0.1
            evidence.append("迅速な処理です")
        
        # 改善提案
        if score < 0.5:
            suggestions.append("要点を絞って簡潔に説明してください")
            suggestions.append("不要な繰り返しを避けてください")
        
        return QualityScore(
            dimension=QualityDimension.EFFICIENCY,
            score=max(0.0, min(score, 1.0)),
            confidence=0.7,
            explanation=f"効率性スコア: {score:.2f}",
            evidence=evidence,
            suggestions=suggestions
        )
    
    def _evaluate_creativity(self, text: str, response: CoTResponse, rules: Dict[str, Any]) -> QualityScore:
        """創造性評価"""
        score = 0.3  # 創造性は基本的に低めに設定
        evidence = []
        suggestions = []
        
        # 創造的要素の存在
        creative_indicators = rules["creative_indicators"]
        creative_count = sum(text.count(indicator) for indicator in creative_indicators)
        
        if creative_count > 0:
            score += 0.2
            evidence.append("創造的な要素が含まれています")
        
        # 代替案の提示
        alternative_indicators = rules["alternative_indicators"]
        alternative_count = sum(text.count(indicator) for indicator in alternative_indicators)
        
        if alternative_count > 0:
            score += 0.2
            evidence.append("複数の視点や代替案が提示されています")
        
        # 独創性の評価
        if self._has_original_insights(text):
            score += 0.3
            evidence.append("独創的な洞察が含まれています")
        
        # 改善提案
        if score < 0.4:
            suggestions.append("より創造的なアプローチを検討してください")
            suggestions.append("複数の視点から問題を捉えてください")
        
        return QualityScore(
            dimension=QualityDimension.CREATIVITY,
            score=min(score, 1.0),
            confidence=0.5,
            explanation=f"創造性スコア: {score:.2f}",
            evidence=evidence,
            suggestions=suggestions
        )
    
    def _evaluate_safety(self, text: str, response: CoTResponse, rules: Dict[str, Any]) -> QualityScore:
        """安全性評価"""
        score = 0.8  # 安全性は基本的に高めに設定
        evidence = []
        suggestions = []
        
        # 有害な内容の検出
        harmful_indicators = rules["harmful_indicators"]
        harmful_count = sum(text.count(indicator) for indicator in harmful_indicators)
        
        if harmful_count > 0:
            score -= 0.3
            suggestions.append("有害な内容が含まれている可能性があります")
        else:
            evidence.append("有害な内容は検出されませんでした")
        
        # 安全配慮の存在
        safety_keywords = rules["safety_keywords"]
        safety_count = sum(text.count(keyword) for keyword in safety_keywords)
        
        if safety_count > 0:
            score += 0.1
            evidence.append("安全配慮が含まれています")
        
        # 倫理的配慮
        ethical_indicators = rules["ethical_indicators"]
        ethical_count = sum(text.count(indicator) for indicator in ethical_indicators)
        
        if ethical_count > 0:
            score += 0.1
            evidence.append("倫理的配慮が含まれています")
        
        # 改善提案
        if score < 0.8:
            suggestions.append("安全性と倫理性をより重視してください")
            suggestions.append("潜在的なリスクについて言及してください")
        
        return QualityScore(
            dimension=QualityDimension.SAFETY,
            score=max(0.0, min(score, 1.0)),
            confidence=0.9,
            explanation=f"安全性スコア: {score:.2f}",
            evidence=evidence,
            suggestions=suggestions
        )
    
    def _check_step_consistency(self, steps: List[ReasoningStep]) -> float:
        """推論ステップの一貫性チェック"""
        if len(steps) < 2:
            return 0.5
        
        consistency_score = 0.0
        total_pairs = 0
        
        for i in range(len(steps) - 1):
            current_step = steps[i]
            next_step = steps[i + 1]
            
            # ステップ間の論理的つながりをチェック
            if self._are_steps_logically_connected(current_step, next_step):
                consistency_score += 1.0
            
            total_pairs += 1
        
        return consistency_score / total_pairs if total_pairs > 0 else 0.5
    
    def _are_steps_logically_connected(self, step1: ReasoningStep, step2: ReasoningStep) -> bool:
        """2つのステップが論理的に接続されているかチェック"""
        # 簡単な接続性チェック
        content1 = step1.content.lower()
        content2 = step2.content.lower()
        
        # 共通のキーワードや概念の存在
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        common_words = words1.intersection(words2)
        return len(common_words) > 2  # 3つ以上の共通単語
    
    def _is_question_answered(self, text: str, question: str) -> bool:
        """質問が適切に答えられているかチェック"""
        # 簡単なキーワードマッチング
        question_words = set(question.lower().split())
        answer_words = set(text.lower().split())
        
        # 質問の主要な単語が回答に含まれているか
        important_words = [word for word in question_words if len(word) > 3]
        matched_words = [word for word in important_words if word in answer_words]
        
        return len(matched_words) / len(important_words) > 0.5 if important_words else True
    
    def _has_redundancy(self, text: str) -> bool:
        """冗長性のチェック"""
        sentences = text.split('。')
        if len(sentences) < 3:
            return False
        
        # 類似した文の検出
        for i in range(len(sentences) - 1):
            for j in range(i + 1, len(sentences)):
                if self._are_sentences_similar(sentences[i], sentences[j]):
                    return True
        
        return False
    
    def _are_sentences_similar(self, sentence1: str, sentence2: str) -> bool:
        """2つの文が類似しているかチェック"""
        words1 = set(sentence1.lower().split())
        words2 = set(sentence2.lower().split())
        
        if len(words1) < 3 or len(words2) < 3:
            return False
        
        common_words = words1.intersection(words2)
        similarity = len(common_words) / min(len(words1), len(words2))
        
        return similarity > 0.7
    
    def _has_original_insights(self, text: str) -> bool:
        """独創的な洞察の存在チェック"""
        # 創造的な表現や新しい視点の検出
        creative_patterns = [
            r"新しい.*方法",
            r"別の.*視点",
            r"興味深い.*発見",
            r"意外な.*結果",
            r"革新的.*アプローチ"
        ]
        
        for pattern in creative_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _calculate_overall_score(self, dimension_scores: Dict[QualityDimension, QualityScore]) -> float:
        """総合スコア計算"""
        # 重み付き平均
        weights = {
            QualityDimension.ACCURACY: 0.25,
            QualityDimension.COMPLETENESS: 0.20,
            QualityDimension.CLARITY: 0.15,
            QualityDimension.LOGICAL_CONSISTENCY: 0.15,
            QualityDimension.USEFULNESS: 0.15,
            QualityDimension.EFFICIENCY: 0.05,
            QualityDimension.CREATIVITY: 0.03,
            QualityDimension.SAFETY: 0.02
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for dimension, score in dimension_scores.items():
            weight = weights.get(dimension, 0.0)
            weighted_sum += score.score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _analyze_quality_aspects(self, 
                                dimension_scores: Dict[QualityDimension, QualityScore],
                                response: CoTResponse,
                                question: str) -> Tuple[List[str], List[str], List[str]]:
        """品質面の分析"""
        strengths = []
        weaknesses = []
        recommendations = []
        
        # 各次元の分析
        for dimension, score in dimension_scores.items():
            if score.score >= self.quality_thresholds[dimension]:
                strengths.append(f"{dimension.value}: {score.explanation}")
            else:
                weaknesses.append(f"{dimension.value}: {score.explanation}")
                recommendations.extend(score.suggestions)
        
        # 全体的な分析
        if response.final_confidence > 0.8:
            strengths.append("高い信頼度を示しています")
        
        if response.step_count >= 5:
            strengths.append("詳細な推論プロセスが含まれています")
        
        if response.processing_time < 5.0:
            strengths.append("迅速な処理です")
        
        # 重複する推奨事項を削除
        recommendations = list(set(recommendations))
        
        return strengths, weaknesses, recommendations
    
    def _update_evaluation_metrics(self, dimension_scores: Dict[QualityDimension, QualityScore], evaluation_time: float):
        """評価メトリクス更新"""
        self.evaluation_metrics["total_evaluations"] += 1
        self.evaluation_metrics["evaluation_times"].append(evaluation_time)
        
        # 平均スコア更新
        for dimension, score in dimension_scores.items():
            if dimension not in self.evaluation_metrics["average_scores"]:
                self.evaluation_metrics["average_scores"][dimension] = []
            self.evaluation_metrics["average_scores"][dimension].append(score.score)
        
        # パフォーマンス更新
        for dimension in dimension_scores:
            if dimension not in self.evaluation_metrics["dimension_performance"]:
                self.evaluation_metrics["dimension_performance"][dimension] = {"high": 0, "medium": 0, "low": 0}
            
            score = dimension_scores[dimension].score
            if score >= 0.8:
                self.evaluation_metrics["dimension_performance"][dimension]["high"] += 1
            elif score >= 0.6:
                self.evaluation_metrics["dimension_performance"][dimension]["medium"] += 1
            else:
                self.evaluation_metrics["dimension_performance"][dimension]["low"] += 1
    
    def _create_error_evaluation(self, error_message: str, evaluation_time: float) -> QualityEvaluation:
        """エラー評価の作成"""
        return QualityEvaluation(
            overall_score=0.0,
            dimension_scores={},
            strengths=[],
            weaknesses=[f"評価エラー: {error_message}"],
            recommendations=["評価システムの確認が必要です"],
            evaluation_time=evaluation_time,
            metadata={"error": error_message}
        )
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """評価統計取得"""
        stats = {
            "total_evaluations": self.evaluation_metrics["total_evaluations"],
            "average_evaluation_time": statistics.mean(self.evaluation_metrics["evaluation_times"]) if self.evaluation_metrics["evaluation_times"] else 0.0,
            "dimension_averages": {},
            "dimension_performance": self.evaluation_metrics["dimension_performance"]
        }
        
        # 次元別平均スコア
        for dimension, scores in self.evaluation_metrics["average_scores"].items():
            stats["dimension_averages"][dimension.value] = statistics.mean(scores)
        
        return stats
    
    def reset_metrics(self):
        """メトリクスリセット"""
        self.evaluation_metrics = {
            "total_evaluations": 0,
            "average_scores": {},
            "dimension_performance": {},
            "evaluation_times": []
        }


# 便利関数
def evaluate_response_quality(response: CoTResponse, question: str, context: Optional[Dict[str, Any]] = None) -> QualityEvaluation:
    """レスポンス品質評価（便利関数）"""
    evaluator = QualityEvaluator()
    return evaluator.evaluate_response(response, question, context)


def get_quality_summary(evaluation: QualityEvaluation) -> Dict[str, Any]:
    """品質評価サマリー取得"""
    return {
        "overall_score": evaluation.overall_score,
        "grade": _score_to_grade(evaluation.overall_score),
        "top_strengths": evaluation.strengths[:3],
        "main_weaknesses": evaluation.weaknesses[:3],
        "key_recommendations": evaluation.recommendations[:3],
        "evaluation_time": evaluation.evaluation_time
    }


def _score_to_grade(score: float) -> str:
    """スコアをグレードに変換"""
    if score >= 0.9:
        return "A+"
    elif score >= 0.8:
        return "A"
    elif score >= 0.7:
        return "B+"
    elif score >= 0.6:
        return "B"
    elif score >= 0.5:
        return "C+"
    elif score >= 0.4:
        return "C"
    else:
        return "D"


# 使用例
if __name__ == "__main__":
    # テスト用のCoTResponse作成
    from .cot_engine import CoTResponse, ReasoningStep, CoTStep, ReasoningState
    
    test_steps = [
        ReasoningStep(1, CoTStep.THOUGHT, "問題を理解します"),
        ReasoningStep(2, CoTStep.ACTION, "計算を実行します"),
        ReasoningStep(3, CoTStep.OBSERVATION, "結果を確認します"),
        ReasoningStep(4, CoTStep.CONCLUSION, "最終回答を導きます")
    ]
    
    test_response = CoTResponse(
        request_id="test_123",
        response_text="この問題を段階的に解決します。まず、与えられた条件を整理し、次に適切な計算方法を選択します。計算結果を検証し、最終的な答えを導きます。答えは42です。",
        processing_time=5.0,
        reasoning_steps=test_steps,
        final_confidence=0.8,
        step_count=4,
        total_thinking_time=4.5,
        quality_score=0.7,
        model_used="qwen2:7b-instruct",
        state=ReasoningState.COMPLETED
    )
    
    # 品質評価実行
    evaluator = QualityEvaluator()
    evaluation = evaluator.evaluate_response(test_response, "2+2は何ですか？")
    
    print("=== Quality Evaluation Results ===")
    print(f"Overall Score: {evaluation.overall_score:.3f}")
    print(f"Grade: {_score_to_grade(evaluation.overall_score)}")
    print(f"Evaluation Time: {evaluation.evaluation_time:.3f}s")
    
    print("\n=== Dimension Scores ===")
    for dimension, score in evaluation.dimension_scores.items():
        print(f"{dimension.value}: {score.score:.3f} - {score.explanation}")
    
    print("\n=== Strengths ===")
    for strength in evaluation.strengths:
        print(f"- {strength}")
    
    print("\n=== Weaknesses ===")
    for weakness in evaluation.weaknesses:
        print(f"- {weakness}")
    
    print("\n=== Recommendations ===")
    for recommendation in evaluation.recommendations:
        print(f"- {recommendation}")
    
    # 統計表示
    stats = evaluator.get_evaluation_statistics()
    print(f"\n=== Statistics ===")
    print(f"Total Evaluations: {stats['total_evaluations']}")
    print(f"Average Evaluation Time: {stats['average_evaluation_time']:.3f}s")
