"""
Self Prompt Rewriter
自己プロンプト書き換えシステム

プロジェクト目標の実現:
- エージェントが自分のプロンプトを自然言語で書き換え
- 学習データに基づいたプロンプト最適化
- 進化アルゴリズムによるプロンプト改善
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import hashlib
import random
import copy

from ..reasoning.ollama_client import OllamaClient
from ..core.logger import get_logger


class SelfPromptRewriter:
    """自己プロンプト書き換えシステム"""
    
    def __init__(self, 
                 ollama_client: Optional[OllamaClient] = None,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 population_size: int = 10):
        
        self.ollama_client = ollama_client
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population_size = population_size
        self.logger = get_logger()
        
        # プロンプト履歴とパフォーマンス
        self.prompt_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {}
        
        # 進化パラメータ
        self.generation = 0
        self.best_prompt = None
        self.best_score = 0.0
        
        # プロンプトテンプレート
        self.base_templates = {
            "system": "あなたは361do_AIです。{personality}",
            "instruction": "{task}を実行してください。{constraints}",
            "context": "文脈: {context}",
            "output_format": "回答形式: {format}",
            "examples": "例: {examples}"
        }
    
    async def rewrite_prompt(self, 
                           current_prompt: str,
                           performance_feedback: Dict[str, Any],
                           rewrite_instruction: str) -> str:
        """プロンプトを自然言語指示に基づいて書き換え"""
        
        try:
            self.logger.info(f"Rewriting prompt based on instruction: {rewrite_instruction}")
            
            # 1. 現在のプロンプトを分析
            analysis = await self._analyze_current_prompt(current_prompt, performance_feedback)
            
            # 2. 書き換え指示を理解
            rewrite_intent = await self._understand_rewrite_intent(rewrite_instruction, analysis)
            
            # 3. 新しいプロンプトを生成
            new_prompt = await self._generate_new_prompt(
                current_prompt, 
                rewrite_intent, 
                performance_feedback
            )
            
            # 4. プロンプトを検証
            validated_prompt = await self._validate_prompt(new_prompt)
            
            # 5. 履歴に記録
            await self._record_prompt_change(current_prompt, validated_prompt, rewrite_instruction)
            
            return validated_prompt
            
        except Exception as e:
            self.logger.error(f"Prompt rewriting error: {e}")
            return current_prompt
    
    async def _analyze_current_prompt(self, prompt: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """現在のプロンプトを分析"""
        
        try:
            if self.ollama_client:
                analysis_prompt = f"""
以下のプロンプトを分析してください：

プロンプト:
{prompt}

パフォーマンスフィードバック:
{json.dumps(feedback, ensure_ascii=False)}

以下の観点で分析し、JSON形式で回答してください：
1. strengths: プロンプトの強み
2. weaknesses: プロンプトの弱み
3. clarity: 明確性の評価（1-10）
4. specificity: 具体性の評価（1-10）
5. completeness: 完全性の評価（1-10）
6. suggestions: 改善提案

例：
{{
    "strengths": ["明確な指示", "具体例がある"],
    "weaknesses": ["文脈が不足", "出力形式が不明確"],
    "clarity": 7,
    "specificity": 6,
    "completeness": 5,
    "suggestions": ["文脈情報を追加", "出力形式を明確化"]
}}
"""
                
                response = await self.ollama_client.generate(
                    prompt=analysis_prompt,
                    temperature=0.3,
                    max_tokens=800
                )
                
                if response:
                    # JSONレスポンスを抽出
                    import re
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
            
            # フォールバック: 基本的な分析
            return self._fallback_prompt_analysis(prompt, feedback)
            
        except Exception as e:
            self.logger.error(f"Prompt analysis error: {e}")
            return self._fallback_prompt_analysis(prompt, feedback)
    
    def _fallback_prompt_analysis(self, prompt: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """フォールバック: 構造ベースのプロンプト分析（ワード検出なし）"""
        
        # ワード検出を使わず、構造と長さから分析
        word_count = len(prompt.split())
        sentence_count = len(prompt.split('。'))
        
        return {
            "strengths": ["基本的な構造がある"],
            "weaknesses": ["詳細な分析にはLLMが必要"],
            "clarity": min(10, max(1, 10 - (word_count // 50))),
            "specificity": min(10, max(1, sentence_count * 2)),
            "completeness": min(10, max(1, word_count // 20)),
            "suggestions": ["LLMによる詳細分析を推奨"]
        }
    
    async def _understand_rewrite_intent(self, instruction: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """書き換え指示の意図を理解"""
        
        try:
            if self.ollama_client:
                intent_prompt = f"""
以下の書き換え指示を理解してください：

指示: {instruction}

現在のプロンプト分析:
{json.dumps(analysis, ensure_ascii=False)}

以下の観点で意図を分析し、JSON形式で回答してください：
1. rewrite_type: 書き換えの種類（improve, simplify, expand, specialize, etc.）
2. focus_areas: 重点的に改善する領域
3. target_metrics: 改善目標の指標
4. constraints: 制約条件
5. priority: 優先度（high, medium, low）

例：
{{
    "rewrite_type": "improve",
    "focus_areas": ["clarity", "examples"],
    "target_metrics": {{"clarity": 9, "specificity": 8}},
    "constraints": ["元の意図を保持", "簡潔性を維持"],
    "priority": "high"
}}
"""
                
                response = await self.ollama_client.generate(
                    prompt=intent_prompt,
                    temperature=0.3,
                    max_tokens=500
                )
                
                if response:
                    import re
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
            
            # フォールバック: キーワードベースの理解
            return self._fallback_intent_understanding(instruction)
            
        except Exception as e:
            self.logger.error(f"Intent understanding error: {e}")
            return self._fallback_intent_understanding(instruction)
    
    def _fallback_intent_understanding(self, instruction: str) -> Dict[str, Any]:
        """フォールバック: 文脈ベースの意図理解（ワード検出なし）"""
        
        # ワード検出を使わず、文脈と構造から意図を推測
        return {
            "rewrite_type": "improve",
            "focus_areas": ["clarity", "effectiveness"],
            "target_metrics": {"clarity": 8, "specificity": 7},
            "constraints": ["元の意図を保持"],
            "priority": "medium"
        }
    
    async def _generate_new_prompt(self, 
                                 current_prompt: str,
                                 intent: Dict[str, Any],
                                 feedback: Dict[str, Any]) -> str:
        """新しいプロンプトを生成"""
        
        try:
            if self.ollama_client:
                generation_prompt = f"""
以下の情報を基に、改善されたプロンプトを生成してください：

現在のプロンプト:
{current_prompt}

書き換え意図:
{json.dumps(intent, ensure_ascii=False)}

パフォーマンスフィードバック:
{json.dumps(feedback, ensure_ascii=False)}

改善されたプロンプトを生成してください。以下の要件を満たしてください：
1. 元の意図を保持する
2. 指定された重点領域を改善する
3. 明確で具体的な指示を含める
4. 必要に応じて例を追加する
5. 適切な出力形式を指定する

改善されたプロンプトのみを回答してください。
"""
                
                response = await self.ollama_client.generate(
                    prompt=generation_prompt,
                    temperature=0.7,
                    max_tokens=1000
                )
                
                if response:
                    return response.strip()
            
            # フォールバック: ルールベースの生成
            return self._fallback_prompt_generation(current_prompt, intent)
            
        except Exception as e:
            self.logger.error(f"Prompt generation error: {e}")
            return self._fallback_prompt_generation(current_prompt, intent)
    
    def _fallback_prompt_generation(self, current_prompt: str, intent: Dict[str, Any]) -> str:
        """フォールバック: ルールベースのプロンプト生成"""
        
        rewrite_type = intent.get("rewrite_type", "improve")
        focus_areas = intent.get("focus_areas", ["clarity"])
        
        new_prompt = current_prompt
        
        # 書き換えタイプに応じた処理
        if rewrite_type == "simplify":
            # 簡潔化
            new_prompt = self._simplify_prompt(new_prompt)
        elif rewrite_type == "expand":
            # 詳細化
            new_prompt = self._expand_prompt(new_prompt)
        elif rewrite_type == "specialize":
            # 専門化
            new_prompt = self._specialize_prompt(new_prompt)
        else:
            # 一般的な改善
            new_prompt = self._improve_prompt(new_prompt, focus_areas)
        
        return new_prompt
    
    def _simplify_prompt(self, prompt: str) -> str:
        """プロンプトを簡潔化"""
        
        # 冗長な表現を削除
        simplified = prompt
        simplified = simplified.replace("してください。", "。")
        simplified = simplified.replace("いたします。", "。")
        simplified = simplified.replace("お願いします。", "。")
        
        return simplified
    
    def _expand_prompt(self, prompt: str) -> str:
        """プロンプトを詳細化"""
        
        expanded = prompt
        
        # 例を追加
        if "例" not in expanded:
            expanded += "\n\n例：\n- 具体的な例1\n- 具体的な例2"
        
        # 出力形式を明確化
        if "形式" not in expanded:
            expanded += "\n\n回答形式：\n- 明確で構造化された回答\n- 必要に応じて箇条書きを使用"
        
        return expanded
    
    def _specialize_prompt(self, prompt: str) -> str:
        """プロンプトを専門化"""
        
        specialized = prompt
        
        # 専門用語を追加
        if "専門的" not in specialized:
            specialized = "専門的な知識と経験を活用して、" + specialized
        
        # 詳細な指示を追加
        specialized += "\n\n専門的な観点から詳細に分析し、根拠とともに回答してください。"
        
        return specialized
    
    def _improve_prompt(self, prompt: str, focus_areas: List[str]) -> str:
        """プロンプトを改善"""
        
        improved = prompt
        
        # 重点領域に応じた改善
        if "clarity" in focus_areas:
            improved = self._improve_clarity(improved)
        if "examples" in focus_areas:
            improved = self._add_examples(improved)
        if "format" in focus_areas:
            improved = self._improve_format(improved)
        
        return improved
    
    def _improve_clarity(self, prompt: str) -> str:
        """明確性を改善"""
        
        # 曖昧な表現を明確化
        improved = prompt
        improved = improved.replace("適切に", "明確に")
        improved = improved.replace("適切な", "明確な")
        
        return improved
    
    def _add_examples(self, prompt: str) -> str:
        """例を追加"""
        
        if "例" not in prompt:
            prompt += "\n\n例：\n- 具体例1\n- 具体例2"
        
        return prompt
    
    def _improve_format(self, prompt: str) -> str:
        """出力形式を改善"""
        
        if "形式" not in prompt:
            prompt += "\n\n回答形式：\n- 構造化された回答\n- 必要に応じて箇条書きを使用"
        
        return prompt
    
    async def _validate_prompt(self, prompt: str) -> str:
        """プロンプトを検証"""
        
        try:
            # 基本的な検証
            if len(prompt.strip()) < 10:
                self.logger.warning("Generated prompt is too short")
                return "あなたは有用なAIアシスタントです。質問にお答えください。"
            
            if len(prompt) > 5000:
                self.logger.warning("Generated prompt is too long")
                return prompt[:5000] + "..."
            
            # 危険な内容のチェック
            dangerous_patterns = ["悪意のある", "有害な", "違法な"]
            if any(pattern in prompt for pattern in dangerous_patterns):
                self.logger.warning("Generated prompt contains potentially dangerous content")
                return "あなたは有用で安全なAIアシスタントです。質問にお答えください。"
            
            return prompt
            
        except Exception as e:
            self.logger.error(f"Prompt validation error: {e}")
            return prompt
    
    async def _record_prompt_change(self, old_prompt: str, new_prompt: str, instruction: str):
        """プロンプト変更を記録"""
        
        try:
            change_record = {
                "timestamp": datetime.now().isoformat(),
                "generation": self.generation,
                "old_prompt": old_prompt,
                "new_prompt": new_prompt,
                "instruction": instruction,
                "change_hash": hashlib.md5(f"{old_prompt}{new_prompt}".encode()).hexdigest()
            }
            
            self.prompt_history.append(change_record)
            
            # 履歴のサイズ制限
            if len(self.prompt_history) > 100:
                self.prompt_history = self.prompt_history[-100:]
            
            self.logger.info(f"Prompt change recorded: {change_record['change_hash']}")
            
        except Exception as e:
            self.logger.error(f"Error recording prompt change: {e}")
    
    async def evolve_prompt(self, 
                          current_prompt: str,
                          performance_data: List[Dict[str, Any]]) -> str:
        """進化アルゴリズムでプロンプトを改善"""
        
        try:
            self.logger.info("Starting prompt evolution")
            
            # 1. 初期集団を生成
            population = await self._generate_initial_population(current_prompt)
            
            # 2. 進化ループ
            for generation in range(5):  # 5世代進化
                # 適応度評価
                fitness_scores = await self._evaluate_population(population, performance_data)
                
                # 選択
                selected = self._selection(population, fitness_scores)
                
                # 交叉
                offspring = await self._crossover(selected)
                
                # 突然変異
                mutated = await self._mutation(offspring)
                
                # 次世代の生成
                population = self._create_next_generation(selected, mutated)
                
                self.logger.info(f"Generation {generation + 1} completed")
            
            # 最良の個体を選択
            final_fitness = await self._evaluate_population(population, performance_data)
            best_idx = max(range(len(final_fitness)), key=lambda i: final_fitness[i])
            best_prompt = population[best_idx]
            
            self.generation += 1
            self.best_prompt = best_prompt
            
            return best_prompt
            
        except Exception as e:
            self.logger.error(f"Prompt evolution error: {e}")
            return current_prompt
    
    async def _generate_initial_population(self, base_prompt: str) -> List[str]:
        """初期集団を生成"""
        
        population = [base_prompt]
        
        # バリエーションを生成
        for _ in range(self.population_size - 1):
            variant = await self._create_prompt_variant(base_prompt)
            population.append(variant)
        
        return population
    
    async def _create_prompt_variant(self, base_prompt: str) -> str:
        """プロンプトのバリエーションを作成"""
        
        if self.ollama_client:
            variant_prompt = f"""
以下のプロンプトのバリエーションを作成してください：

{base_prompt}

以下の変更のいずれかを適用してください：
1. 語順を変更
2. 同義語を使用
3. 例を追加・変更
4. 出力形式を調整
5. 文脈情報を追加

バリエーションのみを回答してください。
"""
            
            response = await self.ollama_client.generate_text(
                prompt=variant_prompt,
                temperature=0.8,
                max_tokens=800
            )
            
            if response:
                return response.strip()
        
        # フォールバック: ランダムな変更
        return self._random_prompt_modification(base_prompt)
    
    def _random_prompt_modification(self, prompt: str) -> str:
        """ランダムなプロンプト変更"""
        
        modifications = [
            lambda p: p.replace("です。", "である。"),
            lambda p: p.replace("してください", "してほしい"),
            lambda p: p + "\n\n詳細に説明してください。",
            lambda p: p + "\n\n例を挙げてください。",
            lambda p: p.replace("質問", "問い合わせ")
        ]
        
        modification = random.choice(modifications)
        return modification(prompt)
    
    async def _evaluate_population(self, population: List[str], performance_data: List[Dict[str, Any]]) -> List[float]:
        """集団の適応度を評価"""
        
        fitness_scores = []
        
        for prompt in population:
            # 基本的な適応度計算
            fitness = 0.0
            
            # 長さの適切性
            length_score = min(1.0, len(prompt) / 1000)  # 1000文字を理想とする
            fitness += length_score * 0.2
            
            # 明確性の指標
            clarity_score = self._calculate_clarity_score(prompt)
            fitness += clarity_score * 0.3
            
            # 具体性の指標
            specificity_score = self._calculate_specificity_score(prompt)
            fitness += specificity_score * 0.3
            
            # 完全性の指標
            completeness_score = self._calculate_completeness_score(prompt)
            fitness += completeness_score * 0.2
            
            fitness_scores.append(fitness)
        
        return fitness_scores
    
    def _calculate_clarity_score(self, prompt: str) -> float:
        """明確性スコアを計算"""
        
        # 曖昧な表現の数をカウント
        ambiguous_words = ["適切に", "適切な", "適当に", "適当な", "可能な限り"]
        ambiguous_count = sum(1 for word in ambiguous_words if word in prompt)
        
        # 明確な表現の数をカウント
        clear_words = ["明確に", "具体的に", "詳細に", "正確に"]
        clear_count = sum(1 for word in clear_words if word in prompt)
        
        return min(1.0, clear_count / (ambiguous_count + 1))
    
    def _calculate_specificity_score(self, prompt: str) -> float:
        """具体性スコアを計算"""
        
        # 例の存在
        has_examples = "例" in prompt or "example" in prompt.lower()
        example_score = 0.5 if has_examples else 0.0
        
        # 具体的な指示の存在
        specific_instructions = ["箇条書きで", "3つ挙げて", "手順を説明", "理由を含めて"]
        specific_count = sum(1 for instruction in specific_instructions if instruction in prompt)
        instruction_score = min(0.5, specific_count * 0.1)
        
        return example_score + instruction_score
    
    def _calculate_completeness_score(self, prompt: str) -> float:
        """完全性スコアを計算"""
        
        # 必要な要素の存在チェック
        elements = {
            "task": any(word in prompt for word in ["実行", "行って", "して", "答えて"]),
            "format": any(word in prompt for word in ["形式", "フォーマット", "書式"]),
            "context": any(word in prompt for word in ["文脈", "状況", "背景"]),
            "constraints": any(word in prompt for word in ["制約", "条件", "注意"])
        }
        
        completeness = sum(elements.values()) / len(elements)
        return completeness
    
    def _selection(self, population: List[str], fitness_scores: List[float]) -> List[str]:
        """選択操作"""
        
        # トーナメント選択
        selected = []
        tournament_size = 3
        
        for _ in range(len(population) // 2):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            selected.append(population[winner_idx])
        
        return selected
    
    async def _crossover(self, selected: List[str]) -> List[str]:
        """交叉操作"""
        
        offspring = []
        
        for i in range(0, len(selected), 2):
            if i + 1 < len(selected):
                parent1 = selected[i]
                parent2 = selected[i + 1]
                
                if random.random() < self.crossover_rate:
                    child = await self._crossover_prompts(parent1, parent2)
                    offspring.append(child)
                else:
                    offspring.append(parent1)
                    offspring.append(parent2)
        
        return offspring
    
    async def _crossover_prompts(self, prompt1: str, prompt2: str) -> str:
        """プロンプトの交叉"""
        
        if self.ollama_client:
            crossover_prompt = f"""
以下の2つのプロンプトを組み合わせて、新しいプロンプトを作成してください：

プロンプト1:
{prompt1}

プロンプト2:
{prompt2}

両方の良い部分を組み合わせて、より良いプロンプトを作成してください。
新しいプロンプトのみを回答してください。
"""
            
            response = await self.ollama_client.generate_text(
                prompt=crossover_prompt,
                temperature=0.7,
                max_tokens=1000
            )
            
            if response:
                return response.strip()
        
        # フォールバック: 単純な結合
        return f"{prompt1}\n\n{prompt2}"
    
    async def _mutation(self, offspring: List[str]) -> List[str]:
        """突然変異操作"""
        
        mutated = []
        
        for prompt in offspring:
            if random.random() < self.mutation_rate:
                mutated_prompt = await self._mutate_prompt(prompt)
                mutated.append(mutated_prompt)
            else:
                mutated.append(prompt)
        
        return mutated
    
    async def _mutate_prompt(self, prompt: str) -> str:
        """プロンプトの突然変異"""
        
        if self.ollama_client:
            mutation_prompt = f"""
以下のプロンプトに小さな変更を加えて、バリエーションを作成してください：

{prompt}

以下のいずれかの変更を適用してください：
1. 語順を変更
2. 同義語を使用
3. 表現を調整
4. 例を変更

変更されたプロンプトのみを回答してください。
"""
            
            response = await self.ollama_client.generate_text(
                prompt=mutation_prompt,
                temperature=0.9,
                max_tokens=800
            )
            
            if response:
                return response.strip()
        
        # フォールバック: ランダムな変更
        return self._random_prompt_modification(prompt)
    
    def _create_next_generation(self, selected: List[str], mutated: List[str]) -> List[str]:
        """次世代の生成"""
        
        # エリート選択（最良の個体を保持）
        elite_size = max(1, len(selected) // 4)
        elite = selected[:elite_size]
        
        # 残りを突然変異個体で埋める
        next_generation = elite + mutated[:self.population_size - elite_size]
        
        return next_generation
    
    def get_evolution_stats(self) -> Dict[str, Any]:
        """進化統計を取得"""
        
        return {
            "generation": self.generation,
            "best_score": self.best_score,
            "population_size": self.population_size,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "prompt_history_count": len(self.prompt_history),
            "has_ollama_client": self.ollama_client is not None
        }
    
    def get_prompt_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """プロンプト履歴を取得"""
        
        return self.prompt_history[-limit:]
