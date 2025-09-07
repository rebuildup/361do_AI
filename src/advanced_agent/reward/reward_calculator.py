"""
Reward Calculator
報酬計算システム
"""

import asyncio
import logging
import json
import statistics
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import uuid

from ..reasoning.ollama_client import OllamaClient
from ..memory.importance_evaluator import ImportanceEvaluator
from ..config import get_agent_config


@dataclass
class RewardMetrics:
    """報酬メトリクス"""
    
    # 基本メトリクス
    user_engagement: float = 0.0  # ユーザー関与度 (0-1)
    response_quality: float = 0.0  # 回答品質 (0-1)
    task_completion: float = 0.0  # タスク完了度 (0-1)
    creativity_score: float = 0.0  # 創造性スコア (0-1)
    helpfulness_score: float = 0.0  # 有用性スコア (0-1)
    
    # 時間関連メトリクス
    response_time: float = 0.0  # 応答時間（秒）
    interaction_duration: float = 0.0  # インタラクション時間（秒）
    
    # 学習関連メトリクス
    learning_progress: float = 0.0  # 学習進捗 (0-1)
    adaptation_score: float = 0.0  # 適応スコア (0-1)
    
    # システム関連メトリクス
    resource_efficiency: float = 0.0  # リソース効率 (0-1)
    error_rate: float = 0.0  # エラー率 (0-1)
    
    # 総合スコア
    total_reward: float = 0.0  # 総合報酬スコア (0-1)
    
    # メタデータ
    timestamp: datetime = None
    session_id: str = ""
    interaction_id: str = ""
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if not self.interaction_id:
            self.interaction_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "user_engagement": self.user_engagement,
            "response_quality": self.response_quality,
            "task_completion": self.task_completion,
            "creativity_score": self.creativity_score,
            "helpfulness_score": self.helpfulness_score,
            "response_time": self.response_time,
            "interaction_duration": self.interaction_duration,
            "learning_progress": self.learning_progress,
            "adaptation_score": self.adaptation_score,
            "resource_efficiency": self.resource_efficiency,
            "error_rate": self.error_rate,
            "total_reward": self.total_reward,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "interaction_id": self.interaction_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RewardMetrics':
        """辞書から作成"""
        metrics = cls()
        for key, value in data.items():
            if key == "timestamp" and isinstance(value, str):
                setattr(metrics, key, datetime.fromisoformat(value))
            elif hasattr(metrics, key):
                setattr(metrics, key, value)
        return metrics


class RewardCalculator:
    """報酬計算システム"""
    
    def __init__(self, 
                 ollama_client: Optional[OllamaClient] = None,
                 importance_evaluator: Optional[ImportanceEvaluator] = None):
        config = get_agent_config()
        self.ollama_client = ollama_client or OllamaClient(config.ollama)
        self.importance_evaluator = importance_evaluator or ImportanceEvaluator()
        self.logger = logging.getLogger(__name__)
        
        # 報酬履歴
        self.reward_history: List[RewardMetrics] = []
        
        # 重み設定
        self.weights = {
            "user_engagement": 0.25,
            "response_quality": 0.20,
            "task_completion": 0.15,
            "creativity_score": 0.10,
            "helpfulness_score": 0.10,
            "learning_progress": 0.10,
            "adaptation_score": 0.05,
            "resource_efficiency": 0.03,
            "error_rate": -0.02  # エラー率は負の重み
        }
    
    async def calculate_reward(self, 
                             user_input: str,
                             agent_response: str,
                             interaction_context: Dict[str, Any],
                             session_id: str = "") -> RewardMetrics:
        """報酬を計算"""
        
        try:
            self.logger.info("Calculating reward metrics")
            
            metrics = RewardMetrics(session_id=session_id)
            
            # 各メトリクスを計算
            metrics.user_engagement = await self._calculate_user_engagement(
                user_input, agent_response, interaction_context
            )
            
            metrics.response_quality = await self._calculate_response_quality(
                user_input, agent_response, interaction_context
            )
            
            metrics.task_completion = await self._calculate_task_completion(
                user_input, agent_response, interaction_context
            )
            
            metrics.creativity_score = await self._calculate_creativity_score(
                agent_response, interaction_context
            )
            
            metrics.helpfulness_score = await self._calculate_helpfulness_score(
                user_input, agent_response, interaction_context
            )
            
            metrics.learning_progress = await self._calculate_learning_progress(
                interaction_context
            )
            
            metrics.adaptation_score = await self._calculate_adaptation_score(
                interaction_context
            )
            
            metrics.resource_efficiency = await self._calculate_resource_efficiency(
                interaction_context
            )
            
            metrics.error_rate = await self._calculate_error_rate(
                interaction_context
            )
            
            # 時間メトリクス
            metrics.response_time = interaction_context.get("response_time", 0.0)
            metrics.interaction_duration = interaction_context.get("interaction_duration", 0.0)
            
            # 総合報酬を計算
            metrics.total_reward = self._calculate_total_reward(metrics)
            
            # 履歴に追加
            self.reward_history.append(metrics)
            
            self.logger.info(f"Reward calculated: {metrics.total_reward:.3f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Reward calculation failed: {e}")
            # エラー時はデフォルト値を返す
            return RewardMetrics(session_id=session_id)
    
    async def _calculate_user_engagement(self, 
                                       user_input: str, 
                                       agent_response: str,
                                       context: Dict[str, Any]) -> float:
        """ユーザー関与度を計算"""
        
        try:
            # 入力の長さと複雑さ
            input_length = len(user_input)
            input_complexity = len(user_input.split())
            
            # 質問の種類を分析
            question_indicators = ["?", "？", "どう", "なぜ", "何", "どの", "いつ", "どこ", "誰"]
            has_question = any(indicator in user_input for indicator in question_indicators)
            
            # 感情的な表現
            emotional_indicators = ["!", "！", "すごい", "ありがとう", "助かった", "困った", "嬉しい", "悲しい"]
            has_emotion = any(indicator in user_input for indicator in emotional_indicators)
            
            # 関与度スコアを計算
            engagement_score = 0.0
            
            # 基本スコア（入力の長さと複雑さ）
            if input_length > 100:
                engagement_score += 0.3
            elif input_length > 50:
                engagement_score += 0.2
            else:
                engagement_score += 0.1
            
            # 質問の有無
            if has_question:
                engagement_score += 0.2
            
            # 感情的な表現
            if has_emotion:
                engagement_score += 0.2
            
            # 複雑さ
            if input_complexity > 20:
                engagement_score += 0.2
            elif input_complexity > 10:
                engagement_score += 0.1
            
            # セッションの継続性
            session_length = context.get("session_length", 0)
            if session_length > 10:
                engagement_score += 0.1
            
            return min(engagement_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"User engagement calculation failed: {e}")
            return 0.5
    
    async def _calculate_response_quality(self, 
                                        user_input: str, 
                                        agent_response: str,
                                        context: Dict[str, Any]) -> float:
        """回答品質を計算"""
        
        try:
            # LLMを使用して品質を評価
            quality_prompt = f"""
以下のユーザーの質問とAIの回答を評価してください。

ユーザーの質問:
{user_input}

AIの回答:
{agent_response}

以下の観点で0-1のスコアを付けてください:
1. 正確性: 回答が正確で事実に基づいているか
2. 関連性: 質問に関連した回答になっているか
3. 完全性: 質問に対して十分な情報を提供しているか
4. 明確性: 回答が明確で理解しやすいか
5. 有用性: ユーザーにとって有用な情報か

JSON形式で回答してください:
{{
    "accuracy": 0.0-1.0,
    "relevance": 0.0-1.0,
    "completeness": 0.0-1.0,
    "clarity": 0.0-1.0,
    "usefulness": 0.0-1.0,
    "overall_quality": 0.0-1.0
}}
"""
            
            response = await self.ollama_client.generate_response(quality_prompt)
            
            try:
                quality_data = json.loads(response)
                return quality_data.get("overall_quality", 0.5)
            except json.JSONDecodeError:
                # JSON解析に失敗した場合は、回答の長さと構造で評価
                return self._fallback_quality_assessment(agent_response)
            
        except Exception as e:
            self.logger.error(f"Response quality calculation failed: {e}")
            return self._fallback_quality_assessment(agent_response)
    
    def _fallback_quality_assessment(self, response: str) -> float:
        """フォールバック品質評価"""
        
        try:
            score = 0.0
            
            # 回答の長さ
            if len(response) > 200:
                score += 0.3
            elif len(response) > 100:
                score += 0.2
            else:
                score += 0.1
            
            # 構造的な要素
            if "\n" in response:  # 改行がある
                score += 0.2
            
            if any(marker in response for marker in ["1.", "2.", "3.", "•", "-"]):  # リスト形式
                score += 0.2
            
            if "。" in response:  # 日本語の句読点
                score += 0.1
            
            # 専門用語や詳細な説明
            if len(response.split()) > 20:
                score += 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Fallback quality assessment failed: {e}")
            return 0.5
    
    async def _calculate_task_completion(self, 
                                       user_input: str, 
                                       agent_response: str,
                                       context: Dict[str, Any]) -> float:
        """タスク完了度を計算"""
        
        try:
            # タスクの種類を判定
            task_type = self._identify_task_type(user_input)
            
            # タスクタイプ別の完了度評価
            if task_type == "question":
                return self._evaluate_question_completion(user_input, agent_response)
            elif task_type == "request":
                return self._evaluate_request_completion(user_input, agent_response)
            elif task_type == "conversation":
                return self._evaluate_conversation_completion(user_input, agent_response)
            else:
                return 0.5  # デフォルト値
            
        except Exception as e:
            self.logger.error(f"Task completion calculation failed: {e}")
            return 0.5
    
    def _identify_task_type(self, user_input: str) -> str:
        """自然言語理解に基づくタスクの種類を判定"""
        
        # より自然な表現パターンを使用
        question_patterns = [
            "?", "？", "どう", "なぜ", "何", "どの", "いつ", "どこ", "誰",
            "教えて", "説明して", "どうやって", "なぜ", "どのように",
            "について", "とは", "ですか", "でしょうか"
        ]
        
        request_patterns = [
            "お願い", "してください", "して", "作って", "作成して", "生成して",
            "実行して", "起動して", "設定して", "変更して", "更新して",
            "検索して", "調べて", "探して", "確認して", "チェックして"
        ]
        
        conversation_patterns = [
            "ありがとう", "どうも", "こんにちは", "はじめまして", "よろしく",
            "続けて", "さらに", "それから", "また", "追加で"
        ]
        
        user_input_lower = user_input.lower()
        
        # リクエストの判定を優先
        if any(pattern in user_input_lower for pattern in request_patterns):
            return "request"
        elif any(pattern in user_input_lower for pattern in question_patterns):
            return "question"
        elif any(pattern in user_input_lower for pattern in conversation_patterns):
            return "conversation"
        else:
            # デフォルトは会話として扱う
            return "conversation"
    
    def _evaluate_question_completion(self, user_input: str, agent_response: str) -> float:
        """質問の完了度を評価"""
        
        score = 0.0
        
        # 回答の長さ
        if len(agent_response) > len(user_input) * 2:
            score += 0.3
        
        # 質問のキーワードが回答に含まれているか
        input_words = set(user_input.lower().split())
        response_words = set(agent_response.lower().split())
        common_words = input_words.intersection(response_words)
        
        if len(common_words) > 0:
            score += 0.3
        
        # 回答の構造
        if len(agent_response.split()) > 10:
            score += 0.2
        
        # 具体的な情報の提供
        if any(char.isdigit() for char in agent_response):  # 数字が含まれている
            score += 0.2
        
        return min(score, 1.0)
    
    def _evaluate_request_completion(self, user_input: str, agent_response: str) -> float:
        """リクエストの完了度を評価"""
        
        score = 0.0
        
        # リクエストの内容が実行されているか
        if "コード" in user_input and ("```" in agent_response or "def " in agent_response):
            score += 0.4
        elif "説明" in user_input and len(agent_response) > 100:
            score += 0.4
        elif "作成" in user_input and len(agent_response) > 50:
            score += 0.4
        
        # 回答の詳細度
        if len(agent_response) > 200:
            score += 0.3
        elif len(agent_response) > 100:
            score += 0.2
        
        # 構造化された回答
        if "\n" in agent_response:
            score += 0.3
        
        return min(score, 1.0)
    
    def _evaluate_conversation_completion(self, user_input: str, agent_response: str) -> float:
        """会話の完了度を評価"""
        
        score = 0.0
        
        # 会話の継続性
        if len(agent_response) > 50:
            score += 0.4
        
        # 感情的な応答
        if any(word in agent_response for word in ["ありがとう", "どういたしまして", "お役に立てて", "嬉しい"]):
            score += 0.3
        
        # 質問の返し
        if "?" in agent_response or "？" in agent_response:
            score += 0.3
        
        return min(score, 1.0)
    
    async def _calculate_creativity_score(self, 
                                        agent_response: str,
                                        context: Dict[str, Any]) -> float:
        """創造性スコアを計算"""
        
        try:
            score = 0.0
            
            # 独創的な表現
            creative_indicators = ["例えば", "別の方法として", "アイデア", "提案", "工夫", "新しい"]
            if any(indicator in agent_response for indicator in creative_indicators):
                score += 0.3
            
            # 多様な視点
            if len(agent_response.split("\n")) > 3:  # 複数の段落
                score += 0.2
            
            # 比喩や例え
            if any(word in agent_response for word in ["のような", "みたいな", "例えると"]):
                score += 0.2
            
            # 詳細な説明
            if len(agent_response) > 300:
                score += 0.2
            
            # 構造化された創造性
            if any(marker in agent_response for marker in ["1.", "2.", "3.", "•", "-"]):
                score += 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Creativity score calculation failed: {e}")
            return 0.3
    
    async def _calculate_helpfulness_score(self, 
                                         user_input: str, 
                                         agent_response: str,
                                         context: Dict[str, Any]) -> float:
        """有用性スコアを計算"""
        
        try:
            score = 0.0
            
            # 実用的な情報の提供
            if any(word in agent_response for word in ["方法", "手順", "ステップ", "コツ", "ポイント"]):
                score += 0.3
            
            # 具体的な例
            if "例" in agent_response or "例えば" in agent_response:
                score += 0.2
            
            # 追加の情報
            if len(agent_response) > len(user_input) * 3:
                score += 0.2
            
            # 構造化された情報
            if any(marker in agent_response for marker in ["1.", "2.", "3.", "•", "-"]):
                score += 0.2
            
            # 関連する情報の提供
            if len(agent_response.split()) > 50:
                score += 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Helpfulness score calculation failed: {e}")
            return 0.4
    
    async def _calculate_learning_progress(self, context: Dict[str, Any]) -> float:
        """学習進捗を計算"""
        
        try:
            # セッション内での学習進捗
            session_progress = context.get("session_learning_progress", 0.0)
            
            # 過去のインタラクションからの学習
            historical_learning = context.get("historical_learning_progress", 0.0)
            
            # 新しい知識の獲得
            new_knowledge = context.get("new_knowledge_acquired", 0.0)
            
            # 総合学習進捗
            total_progress = (session_progress * 0.4 + 
                            historical_learning * 0.4 + 
                            new_knowledge * 0.2)
            
            return min(total_progress, 1.0)
            
        except Exception as e:
            self.logger.error(f"Learning progress calculation failed: {e}")
            return 0.2
    
    async def _calculate_adaptation_score(self, context: Dict[str, Any]) -> float:
        """適応スコアを計算"""
        
        try:
            # ユーザーの好みへの適応
            preference_adaptation = context.get("preference_adaptation", 0.0)
            
            # コンテキストへの適応
            context_adaptation = context.get("context_adaptation", 0.0)
            
            # パフォーマンスの改善
            performance_improvement = context.get("performance_improvement", 0.0)
            
            # 総合適応スコア
            total_adaptation = (preference_adaptation * 0.4 + 
                              context_adaptation * 0.3 + 
                              performance_improvement * 0.3)
            
            return min(total_adaptation, 1.0)
            
        except Exception as e:
            self.logger.error(f"Adaptation score calculation failed: {e}")
            return 0.1
    
    async def _calculate_resource_efficiency(self, context: Dict[str, Any]) -> float:
        """リソース効率を計算"""
        
        try:
            # 応答時間の効率性
            response_time = context.get("response_time", 0.0)
            if response_time < 2.0:
                time_efficiency = 1.0
            elif response_time < 5.0:
                time_efficiency = 0.8
            elif response_time < 10.0:
                time_efficiency = 0.6
            else:
                time_efficiency = 0.4
            
            # メモリ使用効率
            memory_efficiency = context.get("memory_efficiency", 0.8)
            
            # CPU使用効率
            cpu_efficiency = context.get("cpu_efficiency", 0.8)
            
            # 総合効率
            total_efficiency = (time_efficiency * 0.5 + 
                              memory_efficiency * 0.3 + 
                              cpu_efficiency * 0.2)
            
            return min(total_efficiency, 1.0)
            
        except Exception as e:
            self.logger.error(f"Resource efficiency calculation failed: {e}")
            return 0.7
    
    async def _calculate_error_rate(self, context: Dict[str, Any]) -> float:
        """エラー率を計算"""
        
        try:
            # エラーの発生回数
            error_count = context.get("error_count", 0)
            total_operations = context.get("total_operations", 1)
            
            error_rate = error_count / total_operations
            
            # エラー率が低いほど良い（0に近いほど良い）
            return min(error_rate, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error rate calculation failed: {e}")
            return 0.1
    
    def _calculate_total_reward(self, metrics: RewardMetrics) -> float:
        """総合報酬を計算"""
        
        try:
            total_reward = 0.0
            
            # 重み付きスコアの計算
            for metric_name, weight in self.weights.items():
                if hasattr(metrics, metric_name):
                    value = getattr(metrics, metric_name)
                    total_reward += value * weight
            
            # 正規化（0-1の範囲に収める）
            return max(0.0, min(total_reward, 1.0))
            
        except Exception as e:
            self.logger.error(f"Total reward calculation failed: {e}")
            return 0.5
    
    def get_reward_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """報酬履歴を取得"""
        
        return [metrics.to_dict() for metrics in self.reward_history[-limit:]]
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """報酬統計を取得"""
        
        if not self.reward_history:
            return {"total_interactions": 0}
        
        total_rewards = [metrics.total_reward for metrics in self.reward_history]
        
        return {
            "total_interactions": len(self.reward_history),
            "average_reward": statistics.mean(total_rewards),
            "max_reward": max(total_rewards),
            "min_reward": min(total_rewards),
            "reward_std": statistics.stdev(total_rewards) if len(total_rewards) > 1 else 0.0,
            "recent_trend": self._calculate_recent_trend()
        }
    
    def _calculate_recent_trend(self) -> str:
        """最近の傾向を計算"""
        
        if len(self.reward_history) < 10:
            return "insufficient_data"
        
        recent_rewards = [metrics.total_reward for metrics in self.reward_history[-10:]]
        older_rewards = [metrics.total_reward for metrics in self.reward_history[-20:-10]]
        
        if not older_rewards:
            return "insufficient_data"
        
        recent_avg = statistics.mean(recent_rewards)
        older_avg = statistics.mean(older_rewards)
        
        if recent_avg > older_avg * 1.05:
            return "improving"
        elif recent_avg < older_avg * 0.95:
            return "declining"
        else:
            return "stable"
    
    def update_weights(self, new_weights: Dict[str, float]):
        """重みを更新"""
        
        try:
            # 重みの正規化
            total_weight = sum(new_weights.values())
            if total_weight > 0:
                normalized_weights = {k: v / total_weight for k, v in new_weights.items()}
                self.weights.update(normalized_weights)
                self.logger.info(f"Reward weights updated: {self.weights}")
            else:
                self.logger.error("Invalid weights: sum is zero")
                
        except Exception as e:
            self.logger.error(f"Weight update failed: {e}")
    
    def export_reward_data(self, file_path: str) -> bool:
        """報酬データをエクスポート"""
        
        try:
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "weights": self.weights,
                "statistics": self.get_reward_statistics(),
                "history": self.get_reward_history()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Reward data exported to: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Reward data export failed: {e}")
            return False

