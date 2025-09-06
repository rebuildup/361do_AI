"""
Reinforcement Learning Agent
強化学習エージェント
"""

import asyncio
import logging
import json
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass
import uuid

from .reward_calculator import RewardCalculator, RewardMetrics
from .engagement_analyzer import EngagementAnalyzer, EngagementMetrics


@dataclass
class RLState:
    """強化学習の状態"""
    
    # 環境状態
    user_engagement: float = 0.0
    session_context: Dict[str, Any] = None
    conversation_history: List[str] = None
    
    # エージェント状態
    current_prompt: str = ""
    response_quality: float = 0.0
    learning_progress: float = 0.0
    
    # メタデータ
    timestamp: datetime = None
    state_id: str = ""
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if not self.state_id:
            self.state_id = str(uuid.uuid4())
        if self.session_context is None:
            self.session_context = {}
        if self.conversation_history is None:
            self.conversation_history = []
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "user_engagement": self.user_engagement,
            "session_context": self.session_context,
            "conversation_history": self.conversation_history,
            "current_prompt": self.current_prompt,
            "response_quality": self.response_quality,
            "learning_progress": self.learning_progress,
            "timestamp": self.timestamp.isoformat(),
            "state_id": self.state_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RLState':
        """辞書から作成"""
        state = cls()
        for key, value in data.items():
            if key == "timestamp" and isinstance(value, str):
                setattr(state, key, datetime.fromisoformat(value))
            elif hasattr(state, key):
                setattr(state, key, value)
        return state


@dataclass
class RLAction:
    """強化学習の行動"""
    
    # 行動の種類
    action_type: str = ""  # "prompt_selection", "response_generation", "learning_update"
    
    # 行動のパラメータ
    parameters: Dict[str, Any] = None
    
    # メタデータ
    timestamp: datetime = None
    action_id: str = ""
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if not self.action_id:
            self.action_id = str(uuid.uuid4())
        if self.parameters is None:
            self.parameters = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "action_type": self.action_type,
            "parameters": self.parameters,
            "timestamp": self.timestamp.isoformat(),
            "action_id": self.action_id
        }


class RLAgent:
    """強化学習エージェント"""
    
    def __init__(self, 
                 reward_calculator: Optional[RewardCalculator] = None,
                 engagement_analyzer: Optional[EngagementAnalyzer] = None):
        self.reward_calculator = reward_calculator or RewardCalculator()
        self.engagement_analyzer = engagement_analyzer or EngagementAnalyzer()
        self.logger = logging.getLogger(__name__)
        
        # 強化学習パラメータ
        self.learning_rate = 0.01
        self.discount_factor = 0.95
        self.epsilon = 0.1  # ε-greedy探索
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Q値テーブル（状態-行動価値）
        self.q_table: Dict[str, Dict[str, float]] = {}
        
        # 経験バッファ
        self.experience_buffer: List[Tuple[RLState, RLAction, float, RLState]] = []
        self.buffer_size = 10000
        
        # 学習履歴
        self.learning_history: List[Dict[str, Any]] = []
        
        # 行動空間
        self.action_space = [
            "use_simple_prompt",
            "use_detailed_prompt", 
            "use_creative_prompt",
            "use_analytical_prompt",
            "ask_clarification",
            "provide_example",
            "suggest_alternatives",
            "continue_conversation"
        ]
    
    async def select_action(self, state: RLState) -> RLAction:
        """行動を選択"""
        
        try:
            state_key = self._get_state_key(state)
            
            # Q値テーブルを初期化
            if state_key not in self.q_table:
                self.q_table[state_key] = {action: 0.0 for action in self.action_space}
            
            # ε-greedy探索
            if np.random.random() < self.epsilon:
                # ランダム行動
                action_type = np.random.choice(self.action_space)
            else:
                # 最適行動
                q_values = self.q_table[state_key]
                action_type = max(q_values, key=q_values.get)
            
            # 行動パラメータを設定
            parameters = self._get_action_parameters(action_type, state)
            
            action = RLAction(
                action_type=action_type,
                parameters=parameters
            )
            
            self.logger.info(f"Action selected: {action_type}")
            return action
            
        except Exception as e:
            self.logger.error(f"Action selection failed: {e}")
            # デフォルト行動を返す
            return RLAction(action_type="continue_conversation")
    
    def _get_state_key(self, state: RLState) -> str:
        """状態のキーを生成"""
        
        try:
            # 状態を離散化
            engagement_level = "high" if state.user_engagement > 0.7 else "medium" if state.user_engagement > 0.4 else "low"
            quality_level = "high" if state.response_quality > 0.7 else "medium" if state.response_quality > 0.4 else "low"
            progress_level = "high" if state.learning_progress > 0.7 else "medium" if state.learning_progress > 0.4 else "low"
            
            return f"{engagement_level}_{quality_level}_{progress_level}"
            
        except Exception as e:
            self.logger.error(f"State key generation failed: {e}")
            return "default_state"
    
    def _get_action_parameters(self, action_type: str, state: RLState) -> Dict[str, Any]:
        """行動パラメータを取得"""
        
        parameters = {}
        
        if action_type == "use_simple_prompt":
            parameters = {
                "prompt_style": "simple",
                "max_length": 100,
                "tone": "friendly"
            }
        elif action_type == "use_detailed_prompt":
            parameters = {
                "prompt_style": "detailed",
                "max_length": 500,
                "tone": "professional"
            }
        elif action_type == "use_creative_prompt":
            parameters = {
                "prompt_style": "creative",
                "max_length": 300,
                "tone": "engaging"
            }
        elif action_type == "use_analytical_prompt":
            parameters = {
                "prompt_style": "analytical",
                "max_length": 400,
                "tone": "logical"
            }
        elif action_type == "ask_clarification":
            parameters = {
                "clarification_type": "general",
                "max_questions": 2
            }
        elif action_type == "provide_example":
            parameters = {
                "example_type": "practical",
                "max_examples": 2
            }
        elif action_type == "suggest_alternatives":
            parameters = {
                "alternative_count": 3,
                "include_pros_cons": True
            }
        elif action_type == "continue_conversation":
            parameters = {
                "continuation_style": "natural",
                "max_length": 200
            }
        
        return parameters
    
    async def update_q_value(self, 
                           state: RLState, 
                           action: RLAction, 
                           reward: float, 
                           next_state: RLState):
        """Q値を更新"""
        
        try:
            state_key = self._get_state_key(state)
            action_type = action.action_type
            
            # 現在のQ値
            current_q = self.q_table.get(state_key, {}).get(action_type, 0.0)
            
            # 次の状態の最大Q値
            next_state_key = self._get_state_key(next_state)
            next_q_values = self.q_table.get(next_state_key, {})
            max_next_q = max(next_q_values.values()) if next_q_values else 0.0
            
            # Q学習の更新式
            new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
            
            # Q値テーブルを更新
            if state_key not in self.q_table:
                self.q_table[state_key] = {}
            self.q_table[state_key][action_type] = new_q
            
            # 経験をバッファに追加
            self.experience_buffer.append((state, action, reward, next_state))
            if len(self.experience_buffer) > self.buffer_size:
                self.experience_buffer.pop(0)
            
            # εの減衰
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            self.logger.info(f"Q-value updated: {state_key} -> {action_type} = {new_q:.4f}")
            
        except Exception as e:
            self.logger.error(f"Q-value update failed: {e}")
    
    async def learn_from_experience(self, batch_size: int = 32):
        """経験から学習"""
        
        try:
            if len(self.experience_buffer) < batch_size:
                return
            
            # ランダムにバッチを選択
            batch = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
            
            for idx in batch:
                state, action, reward, next_state = self.experience_buffer[idx]
                await self.update_q_value(state, action, reward, next_state)
            
            # 学習履歴に記録
            learning_record = {
                "timestamp": datetime.now().isoformat(),
                "batch_size": batch_size,
                "epsilon": self.epsilon,
                "q_table_size": len(self.q_table),
                "buffer_size": len(self.experience_buffer)
            }
            self.learning_history.append(learning_record)
            
            self.logger.info(f"Learned from {batch_size} experiences")
            
        except Exception as e:
            self.logger.error(f"Experience learning failed: {e}")
    
    async def get_policy(self, state: RLState) -> Dict[str, float]:
        """現在のポリシーを取得"""
        
        try:
            state_key = self._get_state_key(state)
            q_values = self.q_table.get(state_key, {})
            
            if not q_values:
                # デフォルトポリシー
                return {action: 1.0 / len(self.action_space) for action in self.action_space}
            
            # ソフトマックス関数でポリシーを計算
            temperature = 1.0
            exp_values = np.exp(np.array(list(q_values.values())) / temperature)
            probabilities = exp_values / np.sum(exp_values)
            
            policy = dict(zip(q_values.keys(), probabilities))
            return policy
            
        except Exception as e:
            self.logger.error(f"Policy calculation failed: {e}")
            return {action: 1.0 / len(self.action_space) for action in self.action_space}
    
    async def evaluate_policy(self, test_states: List[RLState]) -> Dict[str, float]:
        """ポリシーを評価"""
        
        try:
            total_reward = 0.0
            total_actions = 0
            
            for state in test_states:
                action = await self.select_action(state)
                # 実際の報酬は環境から取得する必要がある
                # ここでは簡易的な評価を行う
                reward = self._estimate_reward(state, action)
                total_reward += reward
                total_actions += 1
            
            if total_actions == 0:
                return {"average_reward": 0.0, "total_episodes": 0}
            
            return {
                "average_reward": total_reward / total_actions,
                "total_episodes": total_actions,
                "epsilon": self.epsilon,
                "q_table_size": len(self.q_table)
            }
            
        except Exception as e:
            self.logger.error(f"Policy evaluation failed: {e}")
            return {"average_reward": 0.0, "total_episodes": 0}
    
    def _estimate_reward(self, state: RLState, action: RLAction) -> float:
        """報酬を推定"""
        
        try:
            # 簡易的な報酬推定
            base_reward = 0.5
            
            # 状態に基づく報酬調整
            if state.user_engagement > 0.7:
                base_reward += 0.2
            elif state.user_engagement < 0.3:
                base_reward -= 0.2
            
            # 行動に基づく報酬調整
            if action.action_type in ["use_detailed_prompt", "provide_example"]:
                base_reward += 0.1
            elif action.action_type == "ask_clarification":
                base_reward += 0.05
            
            return max(0.0, min(base_reward, 1.0))
            
        except Exception as e:
            self.logger.error(f"Reward estimation failed: {e}")
            return 0.5
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """学習統計を取得"""
        
        try:
            recent_episodes = self.learning_history[-10:] if len(self.learning_history) >= 10 else self.learning_history
            
            return {
                "total_learning_episodes": len(self.learning_history),
                "current_epsilon": self.epsilon,
                "q_table_size": len(self.q_table),
                "buffer_size": len(self.experience_buffer),
                "recent_learning_rate": len(recent_episodes),
                "learning_trend": self._calculate_learning_trend()
            }
            
        except Exception as e:
            self.logger.error(f"Learning statistics calculation failed: {e}")
            return {
                "total_learning_episodes": 0,
                "current_epsilon": self.epsilon,
                "q_table_size": 0,
                "buffer_size": 0,
                "recent_learning_rate": 0,
                "learning_trend": "unknown"
            }
    
    def _calculate_learning_trend(self) -> str:
        """学習の傾向を計算"""
        
        try:
            if len(self.learning_history) < 5:
                return "insufficient_data"
            
            recent_episodes = self.learning_history[-5:]
            older_episodes = self.learning_history[-10:-5] if len(self.learning_history) >= 10 else []
            
            if not older_episodes:
                return "insufficient_data"
            
            recent_q_size = np.mean([ep["q_table_size"] for ep in recent_episodes])
            older_q_size = np.mean([ep["q_table_size"] for ep in older_episodes])
            
            if recent_q_size > older_q_size * 1.1:
                return "improving"
            elif recent_q_size < older_q_size * 0.9:
                return "declining"
            else:
                return "stable"
                
        except Exception as e:
            self.logger.error(f"Learning trend calculation failed: {e}")
            return "unknown"
    
    def export_model(self, file_path: str) -> bool:
        """モデルをエクスポート"""
        
        try:
            model_data = {
                "exported_at": datetime.now().isoformat(),
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "epsilon": self.epsilon,
                "epsilon_decay": self.epsilon_decay,
                "epsilon_min": self.epsilon_min,
                "q_table": self.q_table,
                "action_space": self.action_space,
                "learning_history": self.learning_history[-100:]  # 最近の100件のみ
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(model_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Model exported to: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model export failed: {e}")
            return False
    
    def load_model(self, file_path: str) -> bool:
        """モデルを読み込み"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
            
            self.learning_rate = model_data.get("learning_rate", self.learning_rate)
            self.discount_factor = model_data.get("discount_factor", self.discount_factor)
            self.epsilon = model_data.get("epsilon", self.epsilon)
            self.epsilon_decay = model_data.get("epsilon_decay", self.epsilon_decay)
            self.epsilon_min = model_data.get("epsilon_min", self.epsilon_min)
            self.q_table = model_data.get("q_table", {})
            self.action_space = model_data.get("action_space", self.action_space)
            self.learning_history = model_data.get("learning_history", [])
            
            self.logger.info(f"Model loaded from: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            return False
    
    def reset_learning(self):
        """学習をリセット"""
        
        try:
            self.q_table.clear()
            self.experience_buffer.clear()
            self.learning_history.clear()
            self.epsilon = 0.1
            
            self.logger.info("Learning reset completed")
            
        except Exception as e:
            self.logger.error(f"Learning reset failed: {e}")

