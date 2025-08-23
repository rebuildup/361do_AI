"""
Continuous Learning Engine
継続的学習エンジン（基本実装）
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from agent.core.config import Config
from agent.core.database import DatabaseManager


class ContinuousLearningEngine:
    """継続的学習エンジン"""

    def __init__(
        self,
        agent_manager,
        db_manager: DatabaseManager,
        config: Config
    ):
        self.agent_manager = agent_manager
        self.db = db_manager
        self.config = config
        self.learning_tasks = {}
        self.is_running = False

    async def start_learning_cycle(self):
        """学習サイクル開始"""
        if not self.config.is_learning_enabled:
            logger.info("Learning is disabled, skipping learning cycle")
            return

        logger.info("Starting continuous learning cycle...")
        self.is_running = True

        # 基本的な学習タスクをスケジュール
        self.learning_tasks['quality_evaluation'] = asyncio.create_task(
            self._quality_evaluation_loop()
        )

        logger.info("Continuous learning cycle started")

    async def stop(self):
        """学習エンジン停止"""
        logger.info("Stopping continuous learning engine...")
        self.is_running = False

        # 実行中のタスクをキャンセル
        for task_name, task in self.learning_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.info(f"Learning task {task_name} cancelled")

        logger.info("Continuous learning engine stopped")

    async def _quality_evaluation_loop(self):
        """品質評価ループ"""
        while self.is_running:
            try:
                await self._evaluate_recent_conversations()
                await asyncio.sleep(self.config.settings.learning_interval_minutes * 60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Quality evaluation loop error: {e}")
                await asyncio.sleep(300)  # 5分待機してリトライ

    async def _evaluate_recent_conversations(self):
        """最近の会話を評価"""
        try:
            # 未評価の会話を取得（簡単な実装）
            conversations = await self.db.get_conversations_by_quality(
                min_score=None, max_score=None, limit=10
            )

            unevaluated = [
                conv for conv in conversations
                if conv.get('quality_score') is None
            ]

            if not unevaluated:
                logger.debug("No unevaluated conversations found")
                return

            logger.info(f"Evaluating {len(unevaluated)} conversations")

            for conv in unevaluated:
                try:
                    # 簡単な品質評価（実際の実装では LLM を使用）
                    quality_score = await self._simple_quality_evaluation(conv)

                    # 品質指標をデータベースに保存
                    await self.db.insert_quality_metrics(
                        conversation_id=conv['id'],
                        relevance_score=quality_score,
                        accuracy_score=quality_score,
                        helpfulness_score=quality_score,
                        clarity_score=quality_score,
                        overall_score=quality_score,
                        evaluation_method="auto_simple"
                    )

                    logger.debug(f"Evaluated conversation {conv['id']}: {quality_score}")

                except Exception as e:
                    logger.error(f"Failed to evaluate conversation {conv['id']}: {e}")

        except Exception as e:
            logger.error(f"Recent conversations evaluation failed: {e}")

    async def _simple_quality_evaluation(self, conversation: Dict) -> float:
        """簡単な品質評価"""
        try:
            user_input = conversation.get('user_input', '')
            agent_response = conversation.get('agent_response', '')
            response_time = conversation.get('response_time', 0)
            user_feedback = conversation.get('user_feedback')

            score = 0.5  # ベーススコア

            # ユーザーフィードバックがある場合
            if user_feedback is not None:
                if user_feedback == 1:  # 良い
                    score = 0.8
                elif user_feedback == -1:  # 悪い
                    score = 0.2
                else:  # 普通
                    score = 0.5
            else:
                # フィードバックがない場合の推定
                # 応答時間チェック
                if response_time > 10:  # 10秒以上は遅い
                    score -= 0.1

                # 応答長チェック
                if len(agent_response) < 10:  # 短すぎる応答
                    score -= 0.2
                elif len(agent_response) > 2000:  # 長すぎる応答
                    score -= 0.1

                # エラーメッセージチェック
                error_keywords = ['エラー', 'error', '申し訳', '分かりません']
                if any(keyword in agent_response.lower() for keyword in error_keywords):
                    score -= 0.2

                # 質問への応答性チェック（簡単な実装）
                if '?' in user_input or '？' in user_input:
                    if len(agent_response) > 50:  # 質問に対して十分な長さの回答
                        score += 0.1

            # スコアを 0-1 の範囲に制限
            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.error(f"Quality evaluation failed: {e}")
            return 0.5  # デフォルトスコア

    async def process_user_feedback(
        self,
        conversation_id: int,
        feedback_score: int,
        feedback_comment: str = ""
    ):
        """ユーザーフィードバック処理"""
        try:
            logger.info(f"Processing user feedback for conversation {conversation_id}")

            # フィードバックをデータベースに保存
            await self.db.update_conversation_feedback(
                conversation_id=conversation_id,
                feedback_score=feedback_score,
                feedback_comment=feedback_comment
            )

            # フィードバックに基づく学習（簡単な実装）
            if feedback_score == 1:  # 良いフィードバック
                await self._learn_from_positive_feedback(conversation_id)
            elif feedback_score == -1:  # 悪いフィードバック
                await self._learn_from_negative_feedback(conversation_id, feedback_comment)

            logger.info(f"User feedback processed for conversation {conversation_id}")

        except Exception as e:
            logger.error(f"Failed to process user feedback: {e}")

    async def _learn_from_positive_feedback(self, conversation_id: int):
        """ポジティブフィードバックからの学習"""
        try:
            # 成功パターンを知識ベースに追加（簡単な実装）
            logger.debug(f"Learning from positive feedback: {conversation_id}")

            # 学習履歴に記録
            await self.db.insert_learning_history(
                learning_type="positive_feedback",
                description=f"Learned from positive feedback for conversation {conversation_id}",
                performance_impact=0.1
            )

        except Exception as e:
            logger.error(f"Failed to learn from positive feedback: {e}")

    async def _learn_from_negative_feedback(
        self,
        conversation_id: int,
        feedback_comment: str
    ):
        """ネガティブフィードバックからの学習"""
        try:
            # 改善点を特定（簡単な実装）
            logger.debug(f"Learning from negative feedback: {conversation_id}")

            # 学習履歴に記録
            await self.db.insert_learning_history(
                learning_type="negative_feedback",
                description=f"Identified improvement area from conversation {conversation_id}: {feedback_comment}",
                performance_impact=-0.1
            )

        except Exception as e:
            logger.error(f"Failed to learn from negative feedback: {e}")

    async def generate_learning_report(self) -> Dict[str, Any]:
        """学習レポート生成"""
        try:
            # パフォーマンス指標取得
            performance_metrics = await self.db.get_performance_metrics(days=30)

            # 学習履歴取得（簡単な実装）
            # 実際の実装では、より詳細な分析を行う

            report = {
                'timestamp': datetime.now().isoformat(),
                'learning_enabled': self.config.is_learning_enabled,
                'performance_metrics': performance_metrics,
                'learning_summary': {
                    'total_conversations': performance_metrics.get('total_conversations', 0),
                    'average_quality': performance_metrics.get('avg_quality', 0),
                    'positive_feedback_rate': self._calculate_positive_feedback_rate(performance_metrics),
                    'improvement_trend': 'stable'  # 簡単な実装
                },
                'recommendations': [
                    "継続的な学習が実行されています",
                    "ユーザーフィードバックの収集を継続してください"
                ]
            }

            return report

        except Exception as e:
            logger.error(f"Failed to generate learning report: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'learning_enabled': self.config.is_learning_enabled
            }

    def _calculate_positive_feedback_rate(self, metrics: Dict) -> float:
        """ポジティブフィードバック率計算"""
        try:
            positive = metrics.get('positive_feedback', 0)
            negative = metrics.get('negative_feedback', 0)
            total = positive + negative

            if total == 0:
                return 0.0

            return positive / total

        except Exception:
            return 0.0
