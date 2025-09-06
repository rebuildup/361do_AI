"""
Engagement Analyzer
関与度分析システム
"""

import asyncio
import logging
import json
import statistics
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import uuid
import re

from ..reasoning.ollama_client import OllamaClient
from ..config import get_agent_config


@dataclass
class EngagementMetrics:
    """関与度メトリクス"""
    
    # 基本関与度
    interaction_frequency: float = 0.0  # インタラクション頻度 (0-1)
    session_duration: float = 0.0  # セッション継続時間 (0-1)
    message_length: float = 0.0  # メッセージ長 (0-1)
    response_time: float = 0.0  # 応答時間 (0-1)
    
    # 感情的関与度
    emotional_intensity: float = 0.0  # 感情的強度 (0-1)
    sentiment_score: float = 0.0  # 感情スコア (-1 to 1)
    enthusiasm_level: float = 0.0  # 熱意レベル (0-1)
    
    # 認知的関与度
    question_complexity: float = 0.0  # 質問の複雑さ (0-1)
    topic_depth: float = 0.0  # トピックの深さ (0-1)
    learning_indicators: float = 0.0  # 学習指標 (0-1)
    
    # 行動的関与度
    follow_up_questions: float = 0.0  # フォローアップ質問 (0-1)
    topic_switches: float = 0.0  # トピック切り替え (0-1)
    clarification_requests: float = 0.0  # 明確化要求 (0-1)
    
    # 総合関与度
    overall_engagement: float = 0.0  # 総合関与度 (0-1)
    
    # メタデータ
    timestamp: datetime = None
    session_id: str = ""
    user_id: str = ""
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "interaction_frequency": self.interaction_frequency,
            "session_duration": self.session_duration,
            "message_length": self.message_length,
            "response_time": self.response_time,
            "emotional_intensity": self.emotional_intensity,
            "sentiment_score": self.sentiment_score,
            "enthusiasm_level": self.enthusiasm_level,
            "question_complexity": self.question_complexity,
            "topic_depth": self.topic_depth,
            "learning_indicators": self.learning_indicators,
            "follow_up_questions": self.follow_up_questions,
            "topic_switches": self.topic_switches,
            "clarification_requests": self.clarification_requests,
            "overall_engagement": self.overall_engagement,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "user_id": self.user_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EngagementMetrics':
        """辞書から作成"""
        metrics = cls()
        for key, value in data.items():
            if key == "timestamp" and isinstance(value, str):
                setattr(metrics, key, datetime.fromisoformat(value))
            elif hasattr(metrics, key):
                setattr(metrics, key, value)
        return metrics


class EngagementAnalyzer:
    """関与度分析システム"""
    
    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        config = get_agent_config()
        self.ollama_client = ollama_client or OllamaClient(config.ollama)
        self.logger = logging.getLogger(__name__)
        
        # 関与度履歴
        self.engagement_history: List[EngagementMetrics] = []
        
        # 重み設定
        self.weights = {
            "interaction_frequency": 0.15,
            "session_duration": 0.10,
            "message_length": 0.10,
            "response_time": 0.05,
            "emotional_intensity": 0.15,
            "sentiment_score": 0.10,
            "enthusiasm_level": 0.10,
            "question_complexity": 0.10,
            "topic_depth": 0.10,
            "learning_indicators": 0.05
        }
        
        # 感情分析用の辞書
        self.emotional_indicators = {
            "positive": ["嬉しい", "楽しい", "面白い", "すごい", "素晴らしい", "ありがとう", "助かった"],
            "negative": ["悲しい", "困った", "難しい", "分からない", "疲れた", "嫌い", "つらい"],
            "excitement": ["!", "！", "すごい", "驚いた", "びっくり", "興奮", "ワクワク"],
            "curiosity": ["?", "？", "なぜ", "どうして", "どう", "何", "どの", "いつ", "どこ", "誰"]
        }
    
    async def analyze_engagement(self, 
                               user_input: str,
                               session_context: Dict[str, Any],
                               session_id: str = "",
                               user_id: str = "") -> EngagementMetrics:
        """関与度を分析"""
        
        try:
            self.logger.info("Analyzing user engagement")
            
            metrics = EngagementMetrics(session_id=session_id, user_id=user_id)
            
            # 各メトリクスを計算
            metrics.interaction_frequency = await self._calculate_interaction_frequency(session_context)
            metrics.session_duration = await self._calculate_session_duration(session_context)
            metrics.message_length = await self._calculate_message_length(user_input)
            metrics.response_time = await self._calculate_response_time(session_context)
            
            metrics.emotional_intensity = await self._calculate_emotional_intensity(user_input)
            metrics.sentiment_score = await self._calculate_sentiment_score(user_input)
            metrics.enthusiasm_level = await self._calculate_enthusiasm_level(user_input)
            
            metrics.question_complexity = await self._calculate_question_complexity(user_input)
            metrics.topic_depth = await self._calculate_topic_depth(user_input, session_context)
            metrics.learning_indicators = await self._calculate_learning_indicators(user_input, session_context)
            
            metrics.follow_up_questions = await self._calculate_follow_up_questions(session_context)
            metrics.topic_switches = await self._calculate_topic_switches(session_context)
            metrics.clarification_requests = await self._calculate_clarification_requests(user_input)
            
            # 総合関与度を計算
            metrics.overall_engagement = self._calculate_overall_engagement(metrics)
            
            # 履歴に追加
            self.engagement_history.append(metrics)
            
            self.logger.info(f"Engagement analyzed: {metrics.overall_engagement:.3f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Engagement analysis failed: {e}")
            # エラー時はデフォルト値を返す
            return EngagementMetrics(session_id=session_id, user_id=user_id)
    
    async def _calculate_interaction_frequency(self, context: Dict[str, Any]) -> float:
        """インタラクション頻度を計算"""
        
        try:
            # セッション内のインタラクション数
            interaction_count = context.get("interaction_count", 1)
            session_duration_minutes = context.get("session_duration_minutes", 1)
            
            # 1分あたりのインタラクション数
            frequency = interaction_count / session_duration_minutes
            
            # 正規化（0-1の範囲に収める）
            if frequency >= 2.0:  # 1分に2回以上
                return 1.0
            elif frequency >= 1.0:  # 1分に1回
                return 0.8
            elif frequency >= 0.5:  # 2分に1回
                return 0.6
            elif frequency >= 0.25:  # 4分に1回
                return 0.4
            else:
                return 0.2
            
        except Exception as e:
            self.logger.error(f"Interaction frequency calculation failed: {e}")
            return 0.5
    
    async def _calculate_session_duration(self, context: Dict[str, Any]) -> float:
        """セッション継続時間を計算"""
        
        try:
            duration_minutes = context.get("session_duration_minutes", 0)
            
            # セッション時間による関与度
            if duration_minutes >= 60:  # 1時間以上
                return 1.0
            elif duration_minutes >= 30:  # 30分以上
                return 0.8
            elif duration_minutes >= 15:  # 15分以上
                return 0.6
            elif duration_minutes >= 5:  # 5分以上
                return 0.4
            else:
                return 0.2
            
        except Exception as e:
            self.logger.error(f"Session duration calculation failed: {e}")
            return 0.3
    
    async def _calculate_message_length(self, user_input: str) -> float:
        """メッセージ長を計算"""
        
        try:
            length = len(user_input)
            word_count = len(user_input.split())
            
            # 文字数による関与度
            if length >= 500:  # 500文字以上
                char_score = 1.0
            elif length >= 200:  # 200文字以上
                char_score = 0.8
            elif length >= 100:  # 100文字以上
                char_score = 0.6
            elif length >= 50:  # 50文字以上
                char_score = 0.4
            else:
                char_score = 0.2
            
            # 単語数による関与度
            if word_count >= 100:  # 100単語以上
                word_score = 1.0
            elif word_count >= 50:  # 50単語以上
                word_score = 0.8
            elif word_count >= 20:  # 20単語以上
                word_score = 0.6
            elif word_count >= 10:  # 10単語以上
                word_score = 0.4
            else:
                word_score = 0.2
            
            # 文字数と単語数の平均
            return (char_score + word_score) / 2
            
        except Exception as e:
            self.logger.error(f"Message length calculation failed: {e}")
            return 0.4
    
    async def _calculate_response_time(self, context: Dict[str, Any]) -> float:
        """応答時間を計算"""
        
        try:
            response_time_seconds = context.get("response_time_seconds", 0)
            
            # 応答時間が短いほど関与度が高い（ユーザーが待っている）
            if response_time_seconds <= 1.0:  # 1秒以内
                return 1.0
            elif response_time_seconds <= 3.0:  # 3秒以内
                return 0.8
            elif response_time_seconds <= 5.0:  # 5秒以内
                return 0.6
            elif response_time_seconds <= 10.0:  # 10秒以内
                return 0.4
            else:
                return 0.2
            
        except Exception as e:
            self.logger.error(f"Response time calculation failed: {e}")
            return 0.5
    
    async def _calculate_emotional_intensity(self, user_input: str) -> float:
        """感情的強度を計算"""
        
        try:
            intensity = 0.0
            
            # 感情的な表現の検出
            for emotion_type, indicators in self.emotional_indicators.items():
                for indicator in indicators:
                    if indicator in user_input:
                        if emotion_type == "excitement":
                            intensity += 0.3
                        elif emotion_type in ["positive", "negative"]:
                            intensity += 0.2
                        elif emotion_type == "curiosity":
                            intensity += 0.1
            
            # 感嘆符の使用
            exclamation_count = user_input.count("!") + user_input.count("！")
            intensity += min(exclamation_count * 0.1, 0.3)
            
            # 大文字の使用（英語の場合）
            uppercase_ratio = sum(1 for c in user_input if c.isupper()) / len(user_input) if user_input else 0
            intensity += min(uppercase_ratio * 2, 0.2)
            
            return min(intensity, 1.0)
            
        except Exception as e:
            self.logger.error(f"Emotional intensity calculation failed: {e}")
            return 0.3
    
    async def _calculate_sentiment_score(self, user_input: str) -> float:
        """感情スコアを計算"""
        
        try:
            # 簡易的な感情分析
            positive_words = self.emotional_indicators["positive"]
            negative_words = self.emotional_indicators["negative"]
            
            positive_count = sum(1 for word in positive_words if word in user_input)
            negative_count = sum(1 for word in negative_words if word in user_input)
            
            # 感情スコアの計算（-1 to 1）
            if positive_count > negative_count:
                sentiment = min(positive_count * 0.2, 1.0)
            elif negative_count > positive_count:
                sentiment = max(-negative_count * 0.2, -1.0)
            else:
                sentiment = 0.0
            
            return sentiment
            
        except Exception as e:
            self.logger.error(f"Sentiment score calculation failed: {e}")
            return 0.0
    
    async def _calculate_enthusiasm_level(self, user_input: str) -> float:
        """熱意レベルを計算"""
        
        try:
            enthusiasm = 0.0
            
            # 熱意を示す表現
            enthusiasm_indicators = [
                "すごい", "素晴らしい", "驚いた", "びっくり", "興奮", "ワクワク",
                "楽しい", "面白い", "興味深い", "知りたい", "教えて", "詳しく"
            ]
            
            for indicator in enthusiasm_indicators:
                if indicator in user_input:
                    enthusiasm += 0.2
            
            # 感嘆符の使用
            exclamation_count = user_input.count("!") + user_input.count("！")
            enthusiasm += min(exclamation_count * 0.15, 0.3)
            
            # 繰り返し表現
            if re.search(r'(.)\1{2,}', user_input):  # 同じ文字が3回以上連続
                enthusiasm += 0.2
            
            return min(enthusiasm, 1.0)
            
        except Exception as e:
            self.logger.error(f"Enthusiasm level calculation failed: {e}")
            return 0.2
    
    async def _calculate_question_complexity(self, user_input: str) -> float:
        """質問の複雑さを計算"""
        
        try:
            complexity = 0.0
            
            # 質問の種類
            question_words = ["なぜ", "どうして", "どのように", "何故", "なぜなら"]
            for word in question_words:
                if word in user_input:
                    complexity += 0.3
            
            # 複数の質問
            question_marks = user_input.count("?") + user_input.count("？")
            if question_marks > 1:
                complexity += 0.2
            
            # 条件文
            if any(word in user_input for word in ["もし", "仮に", "場合", "とき"]):
                complexity += 0.2
            
            # 比較表現
            if any(word in user_input for word in ["より", "比較", "違い", "対比"]):
                complexity += 0.2
            
            # 長い文章
            if len(user_input.split()) > 30:
                complexity += 0.1
            
            return min(complexity, 1.0)
            
        except Exception as e:
            self.logger.error(f"Question complexity calculation failed: {e}")
            return 0.3
    
    async def _calculate_topic_depth(self, user_input: str, context: Dict[str, Any]) -> float:
        """トピックの深さを計算"""
        
        try:
            depth = 0.0
            
            # 専門用語の使用
            technical_terms = ["アルゴリズム", "データベース", "API", "フレームワーク", "アーキテクチャ"]
            for term in technical_terms:
                if term in user_input:
                    depth += 0.2
            
            # 詳細な説明要求
            detail_indicators = ["詳しく", "具体的に", "詳細", "例を", "手順", "方法"]
            for indicator in detail_indicators:
                if indicator in user_input:
                    depth += 0.15
            
            # セッション内でのトピックの継続
            topic_continuity = context.get("topic_continuity", 0.0)
            depth += topic_continuity * 0.3
            
            return min(depth, 1.0)
            
        except Exception as e:
            self.logger.error(f"Topic depth calculation failed: {e}")
            return 0.2
    
    async def _calculate_learning_indicators(self, user_input: str, context: Dict[str, Any]) -> float:
        """学習指標を計算"""
        
        try:
            learning_score = 0.0
            
            # 学習を示す表現
            learning_indicators = [
                "理解", "分かった", "覚えた", "学んだ", "習得", "身につけた",
                "復習", "確認", "整理", "まとめ", "整理", "要約"
            ]
            
            for indicator in learning_indicators:
                if indicator in user_input:
                    learning_score += 0.2
            
            # 質問の進化
            question_evolution = context.get("question_evolution", 0.0)
            learning_score += question_evolution * 0.3
            
            # 知識の応用
            if any(word in user_input for word in ["応用", "活用", "実践", "使って", "利用"]):
                learning_score += 0.3
            
            return min(learning_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Learning indicators calculation failed: {e}")
            return 0.1
    
    async def _calculate_follow_up_questions(self, context: Dict[str, Any]) -> float:
        """フォローアップ質問を計算"""
        
        try:
            follow_up_count = context.get("follow_up_questions", 0)
            total_questions = context.get("total_questions", 1)
            
            follow_up_ratio = follow_up_count / total_questions
            
            # フォローアップ質問が多いほど関与度が高い
            if follow_up_ratio >= 0.5:
                return 1.0
            elif follow_up_ratio >= 0.3:
                return 0.8
            elif follow_up_ratio >= 0.2:
                return 0.6
            elif follow_up_ratio >= 0.1:
                return 0.4
            else:
                return 0.2
            
        except Exception as e:
            self.logger.error(f"Follow-up questions calculation failed: {e}")
            return 0.3
    
    async def _calculate_topic_switches(self, context: Dict[str, Any]) -> float:
        """トピック切り替えを計算"""
        
        try:
            topic_switches = context.get("topic_switches", 0)
            session_duration_minutes = context.get("session_duration_minutes", 1)
            
            # 1分あたりのトピック切り替え数
            switch_rate = topic_switches / session_duration_minutes
            
            # 適度なトピック切り替えが関与度を示す
            if 0.1 <= switch_rate <= 0.5:  # 適度な切り替え
                return 0.8
            elif switch_rate > 0.5:  # 頻繁すぎる切り替え
                return 0.4
            else:  # 切り替えが少ない
                return 0.6
            
        except Exception as e:
            self.logger.error(f"Topic switches calculation failed: {e}")
            return 0.5
    
    async def _calculate_clarification_requests(self, user_input: str) -> float:
        """明確化要求を計算"""
        
        try:
            clarification_score = 0.0
            
            # 明確化を求める表現
            clarification_indicators = [
                "もう一度", "詳しく", "具体的に", "例を", "説明して", "教えて",
                "分からない", "理解できない", "意味が", "どういうこと"
            ]
            
            for indicator in clarification_indicators:
                if indicator in user_input:
                    clarification_score += 0.2
            
            # 質問の繰り返し
            if "?" in user_input and user_input.count("?") > 1:
                clarification_score += 0.2
            
            return min(clarification_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Clarification requests calculation failed: {e}")
            return 0.2
    
    def _calculate_overall_engagement(self, metrics: EngagementMetrics) -> float:
        """総合関与度を計算"""
        
        try:
            total_engagement = 0.0
            
            # 重み付きスコアの計算
            for metric_name, weight in self.weights.items():
                if hasattr(metrics, metric_name):
                    value = getattr(metrics, metric_name)
                    # 感情スコアは絶対値を使用
                    if metric_name == "sentiment_score":
                        value = abs(value)
                    total_engagement += value * weight
            
            # 正規化（0-1の範囲に収める）
            return max(0.0, min(total_engagement, 1.0))
            
        except Exception as e:
            self.logger.error(f"Overall engagement calculation failed: {e}")
            return 0.5
    
    def get_engagement_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """関与度履歴を取得"""
        
        return [metrics.to_dict() for metrics in self.engagement_history[-limit:]]
    
    def get_engagement_statistics(self) -> Dict[str, Any]:
        """関与度統計を取得"""
        
        if not self.engagement_history:
            return {"total_sessions": 0}
        
        overall_engagements = [metrics.overall_engagement for metrics in self.engagement_history]
        
        return {
            "total_sessions": len(self.engagement_history),
            "average_engagement": statistics.mean(overall_engagements),
            "max_engagement": max(overall_engagements),
            "min_engagement": min(overall_engagements),
            "engagement_std": statistics.stdev(overall_engagements) if len(overall_engagements) > 1 else 0.0,
            "engagement_trend": self._calculate_engagement_trend(),
            "user_segments": self._analyze_user_segments()
        }
    
    def _calculate_engagement_trend(self) -> str:
        """関与度の傾向を計算"""
        
        if len(self.engagement_history) < 10:
            return "insufficient_data"
        
        recent_engagements = [metrics.overall_engagement for metrics in self.engagement_history[-10:]]
        older_engagements = [metrics.overall_engagement for metrics in self.engagement_history[-20:-10]]
        
        if not older_engagements:
            return "insufficient_data"
        
        recent_avg = statistics.mean(recent_engagements)
        older_avg = statistics.mean(older_engagements)
        
        if recent_avg > older_avg * 1.05:
            return "increasing"
        elif recent_avg < older_avg * 0.95:
            return "decreasing"
        else:
            return "stable"
    
    def _analyze_user_segments(self) -> Dict[str, int]:
        """ユーザーセグメントを分析"""
        
        if not self.engagement_history:
            return {}
        
        segments = {
            "high_engagement": 0,  # 0.8以上
            "medium_engagement": 0,  # 0.5-0.8
            "low_engagement": 0  # 0.5未満
        }
        
        for metrics in self.engagement_history:
            if metrics.overall_engagement >= 0.8:
                segments["high_engagement"] += 1
            elif metrics.overall_engagement >= 0.5:
                segments["medium_engagement"] += 1
            else:
                segments["low_engagement"] += 1
        
        return segments
    
    def update_weights(self, new_weights: Dict[str, float]):
        """重みを更新"""
        
        try:
            # 重みの正規化
            total_weight = sum(new_weights.values())
            if total_weight > 0:
                normalized_weights = {k: v / total_weight for k, v in new_weights.items()}
                self.weights.update(normalized_weights)
                self.logger.info(f"Engagement weights updated: {self.weights}")
            else:
                self.logger.error("Invalid weights: sum is zero")
                
        except Exception as e:
            self.logger.error(f"Weight update failed: {e}")
    
    def export_engagement_data(self, file_path: str) -> bool:
        """関与度データをエクスポート"""
        
        try:
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "weights": self.weights,
                "statistics": self.get_engagement_statistics(),
                "history": self.get_engagement_history()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Engagement data exported to: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Engagement data export failed: {e}")
            return False

