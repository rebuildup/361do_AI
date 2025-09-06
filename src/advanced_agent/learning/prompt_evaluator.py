"""
Prompt Evaluator

プロンプトの効果を評価するシステム
"""

import asyncio
import logging
import json
import statistics
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import uuid

from ..reasoning.ollama_client import OllamaClient
from ..reasoning.quality_evaluator import QualityEvaluator
from .prompt_manager import PromptTemplate


class PromptEvaluation:
    """プロンプト評価結果"""
    
    def __init__(self, 
                 template_name: str,
                 evaluation_id: str = None):
        self.evaluation_id = evaluation_id or str(uuid.uuid4())
        self.template_name = template_name
        self.evaluated_at = datetime.now()
        self.metrics: Dict[str, float] = {}
        self.scores: Dict[str, float] = {}
        self.feedback: List[str] = []
        self.recommendations: List[str] = []
        self.test_cases: List[Dict[str, Any]] = []
        self.overall_score: float = 0.0
        self.logger = logging.getLogger(__name__)
    
    def add_metric(self, name: str, value: float):
        """メトリクスを追加"""
        self.metrics[name] = value
    
    def add_score(self, category: str, score: float):
        """スコアを追加"""
        self.scores[category] = score
    
    def add_feedback(self, feedback: str):
        """フィードバックを追加"""
        self.feedback.append(feedback)
    
    def add_recommendation(self, recommendation: str):
        """推奨事項を追加"""
        self.recommendations.append(recommendation)
    
    def add_test_case(self, test_case: Dict[str, Any]):
        """テストケースを追加"""
        self.test_cases.append(test_case)
    
    def calculate_overall_score(self):
        """総合スコアを計算"""
        if self.scores:
            self.overall_score = statistics.mean(self.scores.values())
        else:
            self.overall_score = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "evaluation_id": self.evaluation_id,
            "template_name": self.template_name,
            "evaluated_at": self.evaluated_at.isoformat(),
            "metrics": self.metrics,
            "scores": self.scores,
            "feedback": self.feedback,
            "recommendations": self.recommendations,
            "test_cases": self.test_cases,
            "overall_score": self.overall_score
        }


class PromptEvaluator:
    """プロンプト評価システム"""
    
    def __init__(self, 
                 ollama_client: Optional[OllamaClient] = None,
                 quality_evaluator: Optional[QualityEvaluator] = None):
        self.ollama_client = ollama_client or OllamaClient()
        self.quality_evaluator = quality_evaluator or QualityEvaluator()
        self.logger = logging.getLogger(__name__)
        self.evaluation_history: List[PromptEvaluation] = []
    
    async def evaluate_template(self, 
                              template: PromptTemplate,
                              test_cases: Optional[List[Dict[str, Any]]] = None,
                              evaluation_criteria: Optional[Dict[str, Any]] = None) -> PromptEvaluation:
        """プロンプトテンプレートを評価"""
        
        try:
            self.logger.info(f"Evaluating template: {template.name}")
            
            evaluation = PromptEvaluation(template.name)
            
            # テストケースを準備
            if not test_cases:
                test_cases = await self._generate_test_cases(template)
            
            evaluation.test_cases = test_cases
            
            # 各評価項目を実行
            await self._evaluate_clarity(template, evaluation)
            await self._evaluate_completeness(template, evaluation)
            await self._evaluate_consistency(template, evaluation, test_cases)
            await self._evaluate_effectiveness(template, evaluation, test_cases)
            await self._evaluate_safety(template, evaluation)
            await self._evaluate_efficiency(template, evaluation)
            
            # 総合スコアを計算
            evaluation.calculate_overall_score()
            
            # 推奨事項を生成
            await self._generate_recommendations(template, evaluation)
            
            # 評価履歴に追加
            self.evaluation_history.append(evaluation)
            
            self.logger.info(f"Template evaluation completed: {template.name} (Score: {evaluation.overall_score:.2f})")
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Template evaluation failed: {e}")
            raise
    
    async def _generate_test_cases(self, template: PromptTemplate) -> List[Dict[str, Any]]:
        """テストケースを生成"""
        
        try:
            # 変数に基づいてテストケースを生成
            test_cases = []
            
            if not template.variables:
                # 変数がない場合は基本的なテストケース
                test_cases.append({
                    "input": {},
                    "expected_output_type": "string",
                    "description": "Basic test case"
                })
            else:
                # 各変数に対してテストケースを生成
                for i, variable in enumerate(template.variables):
                    test_case = {
                        "input": {var: f"test_{var}_{i}" for var in template.variables},
                        "expected_output_type": "string",
                        "description": f"Test case for {variable}"
                    }
                    test_cases.append(test_case)
            
            # LLMを使用してより詳細なテストケースを生成
            if len(template.variables) > 0:
                additional_cases = await self._generate_llm_test_cases(template)
                test_cases.extend(additional_cases)
            
            return test_cases[:10]  # 最大10ケース
            
        except Exception as e:
            self.logger.error(f"Test case generation failed: {e}")
            return []
    
    async def _generate_llm_test_cases(self, template: PromptTemplate) -> List[Dict[str, Any]]:
        """LLMを使用してテストケースを生成"""
        
        try:
            prompt = f"""
以下のプロンプトテンプレートのテストケースを生成してください。

テンプレート名: {template.name}
テンプレート内容: {template.template}
変数: {template.variables}
カテゴリ: {template.category}

以下のJSON形式で3つのテストケースを生成してください:
[
    {{
        "input": {{"variable1": "value1", "variable2": "value2"}},
        "expected_output_type": "string",
        "description": "テストケースの説明"
    }}
]

各テストケースは異なるシナリオを表し、テンプレートの効果を適切に評価できるようにしてください。
"""
            
            response = await self.ollama_client.generate_response(prompt)
            
            try:
                test_cases = json.loads(response)
                if isinstance(test_cases, list):
                    return test_cases
            except json.JSONDecodeError:
                pass
            
            return []
            
        except Exception as e:
            self.logger.error(f"LLM test case generation failed: {e}")
            return []
    
    async def _evaluate_clarity(self, template: PromptTemplate, evaluation: PromptEvaluation):
        """明確性を評価"""
        
        try:
            clarity_score = 0.0
            feedback = []
            
            # 指示の明確性
            if "please" in template.template.lower() or "you should" in template.template.lower():
                clarity_score += 20
            else:
                feedback.append("明確な指示語（please, you should等）の使用を推奨")
            
            # 構造の明確性
            if "\n" in template.template and len(template.template.split("\n")) > 2:
                clarity_score += 20
            else:
                feedback.append("構造化されたプロンプト（段落分け）の使用を推奨")
            
            # 具体性
            if any(word in template.template.lower() for word in ["specific", "detailed", "example", "format"]):
                clarity_score += 20
            else:
                feedback.append("具体的な指示や例の提供を推奨")
            
            # 変数の明確性
            if template.variables:
                clarity_score += 20
                for var in template.variables:
                    if var in template.template:
                        clarity_score += 10 / len(template.variables)
            
            # 長さの適切性
            if 50 <= len(template.template) <= 1000:
                clarity_score += 20
            elif len(template.template) < 50:
                feedback.append("プロンプトが短すぎる可能性があります")
            else:
                feedback.append("プロンプトが長すぎる可能性があります")
            
            evaluation.add_score("clarity", min(clarity_score, 100))
            evaluation.add_metric("clarity_score", clarity_score)
            
            for fb in feedback:
                evaluation.add_feedback(fb)
            
        except Exception as e:
            self.logger.error(f"Clarity evaluation failed: {e}")
    
    async def _evaluate_completeness(self, template: PromptTemplate, evaluation: PromptEvaluation):
        """完全性を評価"""
        
        try:
            completeness_score = 0.0
            feedback = []
            
            # 必要な要素の存在チェック
            required_elements = {
                "context": ["context", "background", "situation"],
                "task": ["task", "goal", "objective", "purpose"],
                "output_format": ["format", "output", "result", "response"],
                "constraints": ["constraint", "limit", "requirement", "rule"]
            }
            
            template_lower = template.template.lower()
            
            for element, keywords in required_elements.items():
                if any(keyword in template_lower for keyword in keywords):
                    completeness_score += 25
                else:
                    feedback.append(f"{element}の明示的な指定を推奨")
            
            evaluation.add_score("completeness", min(completeness_score, 100))
            evaluation.add_metric("completeness_score", completeness_score)
            
            for fb in feedback:
                evaluation.add_feedback(fb)
            
        except Exception as e:
            self.logger.error(f"Completeness evaluation failed: {e}")
    
    async def _evaluate_consistency(self, 
                                  template: PromptTemplate, 
                                  evaluation: PromptEvaluation,
                                  test_cases: List[Dict[str, Any]]):
        """一貫性を評価"""
        
        try:
            consistency_score = 0.0
            feedback = []
            
            # テンプレートの一貫性チェック
            if template.template.count("{") == template.template.count("}"):
                consistency_score += 30
            else:
                feedback.append("変数の括弧の対応が正しくありません")
            
            # 変数の一貫性
            if template.variables:
                for var in template.variables:
                    if f"{{{var}}}" in template.template:
                        consistency_score += 70 / len(template.variables)
                    else:
                        feedback.append(f"変数 '{var}' がテンプレート内で使用されていません")
            
            # 実際の出力の一貫性（テストケース実行）
            if test_cases:
                consistency_scores = []
                for test_case in test_cases[:3]:  # 最初の3ケースでテスト
                    try:
                        rendered = template.render(**test_case["input"])
                        if rendered and len(rendered) > 0:
                            consistency_scores.append(100)
                        else:
                            consistency_scores.append(0)
                    except Exception:
                        consistency_scores.append(0)
                
                if consistency_scores:
                    avg_consistency = statistics.mean(consistency_scores)
                    consistency_score = (consistency_score + avg_consistency) / 2
            
            evaluation.add_score("consistency", min(consistency_score, 100))
            evaluation.add_metric("consistency_score", consistency_score)
            
            for fb in feedback:
                evaluation.add_feedback(fb)
            
        except Exception as e:
            self.logger.error(f"Consistency evaluation failed: {e}")
    
    async def _evaluate_effectiveness(self, 
                                    template: PromptTemplate, 
                                    evaluation: PromptEvaluation,
                                    test_cases: List[Dict[str, Any]]):
        """効果性を評価"""
        
        try:
            effectiveness_score = 0.0
            feedback = []
            
            # プロンプトの効果性をLLMで評価
            evaluation_prompt = f"""
以下のプロンプトテンプレートの効果性を評価してください。

テンプレート名: {template.name}
テンプレート内容: {template.template}
カテゴリ: {template.category}

評価基準:
1. 目的の達成可能性 (0-25点)
2. 指示の明確さ (0-25点)
3. 期待される出力の品質 (0-25点)
4. 実用性 (0-25点)

各項目の点数と総合評価（0-100点）を以下のJSON形式で返してください:
{{
    "purpose_achievement": 点数,
    "instruction_clarity": 点数,
    "output_quality": 点数,
    "practicality": 点数,
    "overall_effectiveness": 総合点数,
    "feedback": "評価コメント"
}}
"""
            
            try:
                response = await self.ollama_client.generate_response(evaluation_prompt)
                eval_result = json.loads(response)
                
                effectiveness_score = eval_result.get("overall_effectiveness", 0)
                evaluation.add_metric("purpose_achievement", eval_result.get("purpose_achievement", 0))
                evaluation.add_metric("instruction_clarity", eval_result.get("instruction_clarity", 0))
                evaluation.add_metric("output_quality", eval_result.get("output_quality", 0))
                evaluation.add_metric("practicality", eval_result.get("practicality", 0))
                
                if "feedback" in eval_result:
                    evaluation.add_feedback(eval_result["feedback"])
                
            except Exception as e:
                self.logger.warning(f"LLM effectiveness evaluation failed: {e}")
                # フォールバック評価
                effectiveness_score = 50.0
            
            evaluation.add_score("effectiveness", effectiveness_score)
            evaluation.add_metric("effectiveness_score", effectiveness_score)
            
        except Exception as e:
            self.logger.error(f"Effectiveness evaluation failed: {e}")
    
    async def _evaluate_safety(self, template: PromptTemplate, evaluation: PromptEvaluation):
        """安全性を評価"""
        
        try:
            safety_score = 100.0
            feedback = []
            
            # 危険なパターンをチェック
            dangerous_patterns = [
                (r"ignore\s+safety", "安全性を無視する指示"),
                (r"bypass\s+security", "セキュリティを回避する指示"),
                (r"illegal\s+activity", "違法行為の指示"),
                (r"harmful\s+content", "有害コンテンツの生成指示"),
                (r"personal\s+information", "個人情報の要求"),
                (r"password|secret|key", "機密情報の要求")
            ]
            
            template_lower = template.template.lower()
            
            for pattern, description in dangerous_patterns:
                if re.search(pattern, template_lower):
                    safety_score -= 20
                    feedback.append(f"潜在的な安全性問題: {description}")
            
            # 倫理的配慮のチェック
            ethical_indicators = [
                "ethical", "responsible", "appropriate", "respectful",
                "consider", "think about", "be mindful"
            ]
            
            if any(indicator in template_lower for indicator in ethical_indicators):
                safety_score += 10
            
            evaluation.add_score("safety", max(safety_score, 0))
            evaluation.add_metric("safety_score", safety_score)
            
            for fb in feedback:
                evaluation.add_feedback(fb)
            
        except Exception as e:
            self.logger.error(f"Safety evaluation failed: {e}")
    
    async def _evaluate_efficiency(self, template: PromptTemplate, evaluation: PromptEvaluation):
        """効率性を評価"""
        
        try:
            efficiency_score = 0.0
            feedback = []
            
            # トークン数の効率性
            token_count = len(template.template.split())
            
            if token_count <= 100:
                efficiency_score += 40
            elif token_count <= 200:
                efficiency_score += 30
            elif token_count <= 500:
                efficiency_score += 20
            else:
                feedback.append("プロンプトが長すぎて効率が低下する可能性があります")
            
            # 変数の効率性
            if template.variables:
                variable_efficiency = len(template.variables) / max(token_count, 1) * 100
                if variable_efficiency > 0.1:  # 10%以上が変数
                    efficiency_score += 30
                else:
                    feedback.append("変数の使用率が低い可能性があります")
            
            # 重複のチェック
            words = template.template.lower().split()
            unique_words = set(words)
            if len(unique_words) / len(words) > 0.7:  # 70%以上がユニーク
                efficiency_score += 30
            else:
                feedback.append("重複する単語が多く、効率が低下する可能性があります")
            
            evaluation.add_score("efficiency", min(efficiency_score, 100))
            evaluation.add_metric("efficiency_score", efficiency_score)
            evaluation.add_metric("token_count", token_count)
            
            for fb in feedback:
                evaluation.add_feedback(fb)
            
        except Exception as e:
            self.logger.error(f"Efficiency evaluation failed: {e}")
    
    async def _generate_recommendations(self, template: PromptTemplate, evaluation: PromptEvaluation):
        """推奨事項を生成"""
        
        try:
            recommendations = []
            
            # スコアに基づく推奨事項
            for category, score in evaluation.scores.items():
                if score < 60:
                    if category == "clarity":
                        recommendations.append("プロンプトの指示をより明確にしてください")
                    elif category == "completeness":
                        recommendations.append("必要な要素（コンテキスト、タスク、出力形式等）を追加してください")
                    elif category == "consistency":
                        recommendations.append("変数の使用とテンプレート構造を一貫させてください")
                    elif category == "effectiveness":
                        recommendations.append("プロンプトの目的達成能力を向上させてください")
                    elif category == "safety":
                        recommendations.append("安全性と倫理性を考慮したプロンプトに修正してください")
                    elif category == "efficiency":
                        recommendations.append("プロンプトの長さと構造を最適化してください")
            
            # フィードバックに基づく推奨事項
            for feedback in evaluation.feedback:
                if "推奨" in feedback:
                    recommendations.append(feedback)
            
            # LLMによる推奨事項生成
            if evaluation.overall_score < 70:
                recommendation_prompt = f"""
以下のプロンプトテンプレートの改善点を提案してください。

テンプレート名: {template.name}
テンプレート内容: {template.template}
評価スコア: {evaluation.overall_score:.1f}/100

評価結果:
- 明確性: {evaluation.scores.get('clarity', 0):.1f}
- 完全性: {evaluation.scores.get('completeness', 0):.1f}
- 一貫性: {evaluation.scores.get('consistency', 0):.1f}
- 効果性: {evaluation.scores.get('effectiveness', 0):.1f}
- 安全性: {evaluation.scores.get('safety', 0):.1f}
- 効率性: {evaluation.scores.get('efficiency', 0):.1f}

具体的な改善提案を3つ以上提供してください。
"""
                
                try:
                    response = await self.ollama_client.generate_response(recommendation_prompt)
                    recommendations.append(f"LLM推奨: {response}")
                except Exception as e:
                    self.logger.warning(f"LLM recommendation generation failed: {e}")
            
            for rec in recommendations:
                evaluation.add_recommendation(rec)
            
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {e}")
    
    def get_evaluation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """評価履歴を取得"""
        
        return [eval.to_dict() for eval in self.evaluation_history[-limit:]]
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """評価統計を取得"""
        
        if not self.evaluation_history:
            return {"total_evaluations": 0}
        
        scores = [eval.overall_score for eval in self.evaluation_history]
        
        return {
            "total_evaluations": len(self.evaluation_history),
            "average_score": statistics.mean(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "score_distribution": {
                "excellent": len([s for s in scores if s >= 90]),
                "good": len([s for s in scores if 70 <= s < 90]),
                "fair": len([s for s in scores if 50 <= s < 70]),
                "poor": len([s for s in scores if s < 50])
            }
        }
