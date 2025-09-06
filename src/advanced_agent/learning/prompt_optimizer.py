"""
Prompt Optimizer

プロンプトの最適化システム
"""

import asyncio
import logging
import json
import re
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import uuid
import statistics

from ..reasoning.ollama_client import OllamaClient
from .prompt_manager import PromptTemplate
from .prompt_evaluator import PromptEvaluator, PromptEvaluation


class OptimizationResult:
    """最適化結果"""
    
    def __init__(self, 
                 original_template: PromptTemplate,
                 optimized_template: PromptTemplate,
                 optimization_id: str = None):
        self.optimization_id = optimization_id or str(uuid.uuid4())
        self.original_template = original_template
        self.optimized_template = optimized_template
        self.optimized_at = datetime.now()
        self.improvements: List[str] = []
        self.score_improvement: float = 0.0
        self.optimization_steps: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
    
    def add_improvement(self, improvement: str):
        """改善点を追加"""
        self.improvements.append(improvement)
    
    def add_optimization_step(self, step: Dict[str, Any]):
        """最適化ステップを追加"""
        self.optimization_steps.append(step)
    
    def set_score_improvement(self, original_score: float, optimized_score: float):
        """スコア改善を設定"""
        self.score_improvement = optimized_score - original_score
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "optimization_id": self.optimization_id,
            "original_template": self.original_template.to_dict(),
            "optimized_template": self.optimized_template.to_dict(),
            "optimized_at": self.optimized_at.isoformat(),
            "improvements": self.improvements,
            "score_improvement": self.score_improvement,
            "optimization_steps": self.optimization_steps
        }


class PromptOptimizer:
    """プロンプト最適化システム"""
    
    def __init__(self, 
                 ollama_client: Optional[OllamaClient] = None,
                 evaluator: Optional[PromptEvaluator] = None):
        self.ollama_client = ollama_client or OllamaClient()
        self.evaluator = evaluator or PromptEvaluator(ollama_client)
        self.logger = logging.getLogger(__name__)
        self.optimization_history: List[OptimizationResult] = []
    
    async def optimize_template(self, 
                              template: PromptTemplate,
                              optimization_goals: Optional[List[str]] = None,
                              max_iterations: int = 5) -> OptimizationResult:
        """プロンプトテンプレートを最適化"""
        
        try:
            self.logger.info(f"Optimizing template: {template.name}")
            
            # 最適化目標を設定
            if not optimization_goals:
                optimization_goals = ["clarity", "effectiveness", "efficiency"]
            
            # 元のテンプレートを評価
            original_evaluation = await self.evaluator.evaluate_template(template)
            original_score = original_evaluation.overall_score
            
            # 最適化結果オブジェクトを作成
            result = OptimizationResult(template, template)
            
            # 反復的最適化
            current_template = template
            current_score = original_score
            
            for iteration in range(max_iterations):
                self.logger.info(f"Optimization iteration {iteration + 1}/{max_iterations}")
                
                # 最適化ステップを実行
                optimized_template = await self._optimize_iteration(
                    current_template, 
                    optimization_goals, 
                    original_evaluation
                )
                
                if not optimized_template:
                    break
                
                # 最適化されたテンプレートを評価
                optimized_evaluation = await self.evaluator.evaluate_template(optimized_template)
                optimized_score = optimized_evaluation.overall_score
                
                # 改善があったかチェック
                if optimized_score > current_score:
                    current_template = optimized_template
                    current_score = optimized_score
                    
                    # 最適化ステップを記録
                    step = {
                        "iteration": iteration + 1,
                        "improvement": optimized_score - current_score,
                        "optimizations": self._get_optimization_summary(original_evaluation, optimized_evaluation)
                    }
                    result.add_optimization_step(step)
                    
                    self.logger.info(f"Iteration {iteration + 1}: Score improved from {current_score:.2f} to {optimized_score:.2f}")
                else:
                    self.logger.info(f"Iteration {iteration + 1}: No improvement, stopping optimization")
                    break
            
            # 最終結果を設定
            result.optimized_template = current_template
            result.set_score_improvement(original_score, current_score)
            
            # 改善点を生成
            await self._generate_improvement_summary(result, original_evaluation)
            
            # 最適化履歴に追加
            self.optimization_history.append(result)
            
            self.logger.info(f"Template optimization completed: {template.name} (Improvement: {result.score_improvement:.2f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Template optimization failed: {e}")
            raise
    
    async def _optimize_iteration(self, 
                                template: PromptTemplate,
                                goals: List[str],
                                evaluation: PromptEvaluation) -> Optional[PromptTemplate]:
        """最適化の1回の反復を実行"""
        
        try:
            # 最も改善が必要な項目を特定
            target_category = self._identify_weakest_category(evaluation.scores)
            
            # カテゴリ別の最適化を実行
            if target_category == "clarity":
                return await self._optimize_clarity(template)
            elif target_category == "completeness":
                return await self._optimize_completeness(template)
            elif target_category == "consistency":
                return await self._optimize_consistency(template)
            elif target_category == "effectiveness":
                return await self._optimize_effectiveness(template)
            elif target_category == "safety":
                return await self._optimize_safety(template)
            elif target_category == "efficiency":
                return await self._optimize_efficiency(template)
            else:
                return await self._optimize_general(template)
            
        except Exception as e:
            self.logger.error(f"Optimization iteration failed: {e}")
            return None
    
    def _identify_weakest_category(self, scores: Dict[str, float]) -> str:
        """最も改善が必要なカテゴリを特定"""
        
        if not scores:
            return "effectiveness"
        
        return min(scores.items(), key=lambda x: x[1])[0]
    
    async def _optimize_clarity(self, template: PromptTemplate) -> Optional[PromptTemplate]:
        """明確性を最適化"""
        
        try:
            optimization_prompt = f"""
以下のプロンプトテンプレートの明確性を向上させてください。

現在のテンプレート:
{template.template}

改善点:
1. 指示をより明確にする
2. 構造を整理する
3. 具体例を追加する
4. 曖昧な表現を避ける

最適化されたテンプレートを返してください。変数は {{variable_name}} 形式を維持してください。
"""
            
            response = await self.ollama_client.generate_response(optimization_prompt)
            
            # 新しいテンプレートを作成
            optimized_template = PromptTemplate(
                name=f"{template.name}_optimized",
                template=response.strip(),
                description=f"Optimized version of {template.description}",
                category=template.category,
                tags=template.tags + ["optimized", "clarity"],
                metadata={**template.metadata, "optimization_type": "clarity"}
            )
            
            # 変数を抽出
            optimized_template.extract_variables()
            
            return optimized_template
            
        except Exception as e:
            self.logger.error(f"Clarity optimization failed: {e}")
            return None
    
    async def _optimize_completeness(self, template: PromptTemplate) -> Optional[PromptTemplate]:
        """完全性を最適化"""
        
        try:
            optimization_prompt = f"""
以下のプロンプトテンプレートの完全性を向上させてください。

現在のテンプレート:
{template.template}
カテゴリ: {template.category}

以下の要素を含むように改善してください:
1. 明確なコンテキスト設定
2. 具体的なタスク定義
3. 期待される出力形式
4. 制約条件や注意事項

最適化されたテンプレートを返してください。変数は {{variable_name}} 形式を維持してください。
"""
            
            response = await self.ollama_client.generate_response(optimization_prompt)
            
            optimized_template = PromptTemplate(
                name=f"{template.name}_optimized",
                template=response.strip(),
                description=f"Optimized version of {template.description}",
                category=template.category,
                tags=template.tags + ["optimized", "completeness"],
                metadata={**template.metadata, "optimization_type": "completeness"}
            )
            
            optimized_template.extract_variables()
            
            return optimized_template
            
        except Exception as e:
            self.logger.error(f"Completeness optimization failed: {e}")
            return None
    
    async def _optimize_consistency(self, template: PromptTemplate) -> Optional[PromptTemplate]:
        """一貫性を最適化"""
        
        try:
            # 変数の一貫性をチェック
            template_text = template.template
            variables = template.variables
            
            # 変数の使用を統一
            for var in variables:
                # 変数の表記を統一
                template_text = re.sub(rf'\{{{var}\}}', f'{{{var}}}', template_text)
            
            # 構造の一貫性を改善
            optimization_prompt = f"""
以下のプロンプトテンプレートの一貫性を向上させてください。

現在のテンプレート:
{template_text}

改善点:
1. 変数の使用を統一する
2. 構造を一貫させる
3. 文体を統一する
4. フォーマットを統一する

最適化されたテンプレートを返してください。変数は {{variable_name}} 形式を維持してください。
"""
            
            response = await self.ollama_client.generate_response(optimization_prompt)
            
            optimized_template = PromptTemplate(
                name=f"{template.name}_optimized",
                template=response.strip(),
                description=f"Optimized version of {template.description}",
                category=template.category,
                tags=template.tags + ["optimized", "consistency"],
                metadata={**template.metadata, "optimization_type": "consistency"}
            )
            
            optimized_template.extract_variables()
            
            return optimized_template
            
        except Exception as e:
            self.logger.error(f"Consistency optimization failed: {e}")
            return None
    
    async def _optimize_effectiveness(self, template: PromptTemplate) -> Optional[PromptTemplate]:
        """効果性を最適化"""
        
        try:
            optimization_prompt = f"""
以下のプロンプトテンプレートの効果性を向上させてください。

現在のテンプレート:
{template.template}
カテゴリ: {template.category}

改善点:
1. 目的達成のための明確な指示
2. より効果的な例や説明
3. 期待される結果の明確化
4. 実用性の向上

最適化されたテンプレートを返してください。変数は {{variable_name}} 形式を維持してください。
"""
            
            response = await self.ollama_client.generate_response(optimization_prompt)
            
            optimized_template = PromptTemplate(
                name=f"{template.name}_optimized",
                template=response.strip(),
                description=f"Optimized version of {template.description}",
                category=template.category,
                tags=template.tags + ["optimized", "effectiveness"],
                metadata={**template.metadata, "optimization_type": "effectiveness"}
            )
            
            optimized_template.extract_variables()
            
            return optimized_template
            
        except Exception as e:
            self.logger.error(f"Effectiveness optimization failed: {e}")
            return None
    
    async def _optimize_safety(self, template: PromptTemplate) -> Optional[PromptTemplate]:
        """安全性を最適化"""
        
        try:
            optimization_prompt = f"""
以下のプロンプトテンプレートの安全性と倫理性を向上させてください。

現在のテンプレート:
{template.template}

改善点:
1. 倫理的配慮の追加
2. 安全性の確保
3. 責任ある使用の促進
4. 有害コンテンツの防止

最適化されたテンプレートを返してください。変数は {{variable_name}} 形式を維持してください。
"""
            
            response = await self.ollama_client.generate_response(optimization_prompt)
            
            optimized_template = PromptTemplate(
                name=f"{template.name}_optimized",
                template=response.strip(),
                description=f"Optimized version of {template.description}",
                category=template.category,
                tags=template.tags + ["optimized", "safety"],
                metadata={**template.metadata, "optimization_type": "safety"}
            )
            
            optimized_template.extract_variables()
            
            return optimized_template
            
        except Exception as e:
            self.logger.error(f"Safety optimization failed: {e}")
            return None
    
    async def _optimize_efficiency(self, template: PromptTemplate) -> Optional[PromptTemplate]:
        """効率性を最適化"""
        
        try:
            optimization_prompt = f"""
以下のプロンプトテンプレートの効率性を向上させてください。

現在のテンプレート:
{template.template}

改善点:
1. 冗長な表現の削除
2. 簡潔で明確な指示
3. 不要な情報の除去
4. トークン数の最適化

最適化されたテンプレートを返してください。変数は {{variable_name}} 形式を維持してください。
"""
            
            response = await self.ollama_client.generate_response(optimization_prompt)
            
            optimized_template = PromptTemplate(
                name=f"{template.name}_optimized",
                template=response.strip(),
                description=f"Optimized version of {template.description}",
                category=template.category,
                tags=template.tags + ["optimized", "efficiency"],
                metadata={**template.metadata, "optimization_type": "efficiency"}
            )
            
            optimized_template.extract_variables()
            
            return optimized_template
            
        except Exception as e:
            self.logger.error(f"Efficiency optimization failed: {e}")
            return None
    
    async def _optimize_general(self, template: PromptTemplate) -> Optional[PromptTemplate]:
        """一般的な最適化"""
        
        try:
            optimization_prompt = f"""
以下のプロンプトテンプレートを全体的に改善してください。

現在のテンプレート:
{template.template}
カテゴリ: {template.category}

改善点:
1. 全体的な品質向上
2. ユーザビリティの改善
3. 効果性の向上
4. 明確性の向上

最適化されたテンプレートを返してください。変数は {{variable_name}} 形式を維持してください。
"""
            
            response = await self.ollama_client.generate_response(optimization_prompt)
            
            optimized_template = PromptTemplate(
                name=f"{template.name}_optimized",
                template=response.strip(),
                description=f"Optimized version of {template.description}",
                category=template.category,
                tags=template.tags + ["optimized", "general"],
                metadata={**template.metadata, "optimization_type": "general"}
            )
            
            optimized_template.extract_variables()
            
            return optimized_template
            
        except Exception as e:
            self.logger.error(f"General optimization failed: {e}")
            return None
    
    def _get_optimization_summary(self, 
                                original_eval: PromptEvaluation, 
                                optimized_eval: PromptEvaluation) -> List[str]:
        """最適化の要約を取得"""
        
        improvements = []
        
        for category in original_eval.scores:
            original_score = original_eval.scores[category]
            optimized_score = optimized_eval.scores.get(category, 0)
            
            if optimized_score > original_score:
                improvement = optimized_score - original_score
                improvements.append(f"{category}: +{improvement:.1f}")
        
        return improvements
    
    async def _generate_improvement_summary(self, 
                                          result: OptimizationResult,
                                          original_evaluation: PromptEvaluation):
        """改善要約を生成"""
        
        try:
            # スコア改善の要約
            if result.score_improvement > 0:
                result.add_improvement(f"総合スコアが {result.score_improvement:.2f} ポイント向上")
            
            # カテゴリ別の改善
            for step in result.optimization_steps:
                if "optimizations" in step:
                    for opt in step["optimizations"]:
                        result.add_improvement(opt)
            
            # LLMによる改善要約
            if result.score_improvement > 5:  # 大幅な改善があった場合
                summary_prompt = f"""
以下のプロンプト最適化の改善点を要約してください。

元のテンプレート:
{result.original_template.template}

最適化されたテンプレート:
{result.optimized_template.template}

改善スコア: {result.score_improvement:.2f} ポイント

主な改善点を3つ挙げてください。
"""
                
                try:
                    response = await self.ollama_client.generate_response(summary_prompt)
                    result.add_improvement(f"LLM分析: {response}")
                except Exception as e:
                    self.logger.warning(f"LLM improvement summary failed: {e}")
            
        except Exception as e:
            self.logger.error(f"Improvement summary generation failed: {e}")
    
    async def batch_optimize(self, 
                           templates: List[PromptTemplate],
                           optimization_goals: Optional[List[str]] = None) -> List[OptimizationResult]:
        """複数のテンプレートを一括最適化"""
        
        try:
            self.logger.info(f"Batch optimizing {len(templates)} templates")
            
            results = []
            
            for template in templates:
                try:
                    result = await self.optimize_template(template, optimization_goals)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to optimize template {template.name}: {e}")
                    continue
            
            self.logger.info(f"Batch optimization completed: {len(results)}/{len(templates)} successful")
            return results
            
        except Exception as e:
            self.logger.error(f"Batch optimization failed: {e}")
            return []
    
    def get_optimization_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """最適化履歴を取得"""
        
        return [result.to_dict() for result in self.optimization_history[-limit:]]
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """最適化統計を取得"""
        
        if not self.optimization_history:
            return {"total_optimizations": 0}
        
        improvements = [result.score_improvement for result in self.optimization_history]
        
        return {
            "total_optimizations": len(self.optimization_history),
            "average_improvement": statistics.mean(improvements),
            "max_improvement": max(improvements),
            "min_improvement": min(improvements),
            "successful_optimizations": len([i for i in improvements if i > 0]),
            "improvement_distribution": {
                "significant": len([i for i in improvements if i >= 10]),
                "moderate": len([i for i in improvements if 5 <= i < 10]),
                "minor": len([i for i in improvements if 0 < i < 5]),
                "no_improvement": len([i for i in improvements if i <= 0])
            }
        }
