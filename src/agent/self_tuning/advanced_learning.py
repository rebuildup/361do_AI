"""
Advanced Self-Learning System
高度な自己学習システム
エージェント自身が学習データやプロンプトを書き換える機能
"""

import asyncio
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from loguru import logger

from agent.core.config import Config
from agent.core.database import DatabaseManager
from agent.core.ollama_client import OllamaClient


@dataclass
class LearningData:
    """学習データの構造"""
    id: str
    content: str
    category: str
    quality_score: float
    usage_count: int
    created_at: datetime
    last_used: datetime
    tags: List[str]
    metadata: Dict[str, Any]


@dataclass
class PromptOptimization:
    """プロンプト最適化の結果"""
    original_prompt: str
    optimized_prompt: str
    improvement_score: float
    changes_made: List[str]
    test_results: Dict[str, float]
    applied_at: datetime


class AdvancedLearningSystem:
    """高度な自己学習システム"""
    
    def __init__(
        self,
        agent_manager,
        db_manager: DatabaseManager,
        config: Config,
        ollama_client: OllamaClient
    ):
        self.agent_manager = agent_manager
        self.db = db_manager
        self.config = config
        self.ollama_client = ollama_client
        self.learning_data_path = Path("src/data/learning_data")
        self.prompts_path = Path("src/data/prompts")
        self.is_running = False
        self.learning_tasks = {}
        
        # 学習データディレクトリの作成
        self.learning_data_path.mkdir(parents=True, exist_ok=True)
        self.prompts_path.mkdir(parents=True, exist_ok=True)
        
    async def start_advanced_learning(self):
        """高度な学習システム開始"""
        if not self.config.is_learning_enabled:
            logger.info("Advanced learning is disabled")
            return
        
        logger.info("Starting advanced self-learning system...")
        self.is_running = True
        
        # 学習タスクを開始
        self.learning_tasks['data_analysis'] = asyncio.create_task(
            self._learning_data_analysis_loop()
        )
        self.learning_tasks['prompt_optimization'] = asyncio.create_task(
            self._prompt_optimization_loop()
        )
        self.learning_tasks['knowledge_extraction'] = asyncio.create_task(
            self._knowledge_extraction_loop()
        )
        self.learning_tasks['performance_analysis'] = asyncio.create_task(
            self._performance_analysis_loop()
        )
        
        logger.info("Advanced self-learning system started")
    
    async def stop(self):
        """学習システム停止"""
        logger.info("Stopping advanced learning system...")
        self.is_running = False
        
        for task_name, task in self.learning_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.info(f"Learning task {task_name} cancelled")
        
        logger.info("Advanced learning system stopped")
    
    async def _learning_data_analysis_loop(self):
        """学習データ分析ループ"""
        while self.is_running:
            try:
                await self._analyze_and_improve_learning_data()
                await asyncio.sleep(self.config.settings.learning_interval_minutes * 60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Learning data analysis error: {e}")
                await asyncio.sleep(300)
    
    async def _prompt_optimization_loop(self):
        """プロンプト最適化ループ"""
        while self.is_running:
            try:
                await self._optimize_prompts()
                await asyncio.sleep(self.config.settings.learning_interval_minutes * 60 * 2)  # 2倍の間隔
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Prompt optimization error: {e}")
                await asyncio.sleep(600)
    
    async def _knowledge_extraction_loop(self):
        """知識抽出ループ"""
        while self.is_running:
            try:
                await self._extract_knowledge_from_conversations()
                await asyncio.sleep(self.config.settings.learning_interval_minutes * 60 * 3)  # 3倍の間隔
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Knowledge extraction error: {e}")
                await asyncio.sleep(900)
    
    async def _performance_analysis_loop(self):
        """パフォーマンス分析ループ"""
        while self.is_running:
            try:
                await self._analyze_performance_and_adapt()
                await asyncio.sleep(self.config.settings.learning_interval_minutes * 60 * 4)  # 4倍の間隔
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance analysis error: {e}")
                await asyncio.sleep(1200)
    
    async def _analyze_and_improve_learning_data(self):
        """学習データの分析と改善"""
        try:
            # 最近の会話から学習データを抽出
            conversations = await self.db.get_conversations_by_quality(
                min_score=0.7, max_score=1.0, limit=50
            )
            
            for conv in conversations:
                await self._extract_learning_data_from_conversation(conv)
            
            # 既存の学習データを分析・改善
            await self._improve_existing_learning_data()
            
            logger.info("Learning data analysis completed")
            
        except Exception as e:
            logger.error(f"Learning data analysis failed: {e}")
    
    async def _extract_learning_data_from_conversation(self, conversation: Dict):
        """会話から学習データを抽出"""
        try:
            user_input = conversation.get('user_input', '')
            agent_response = conversation.get('agent_response', '')
            quality_score = conversation.get('quality_score', 0.5)
            
            if quality_score < 0.7:
                return  # 低品質な会話はスキップ
            
            # LLMを使用して学習データを抽出
            extraction_prompt = f"""
以下の会話から学習に有用な情報を抽出してください：

ユーザー入力: {user_input}
エージェント応答: {agent_response}
品質スコア: {quality_score}

以下の形式でJSONを返してください：
{{
    "category": "カテゴリ名",
    "content": "学習内容",
    "tags": ["タグ1", "タグ2"],
    "metadata": {{
        "context": "文脈情報",
        "difficulty": "難易度",
        "usefulness": "有用性"
    }}
}}
"""
            
            response = await self.ollama_client.generate_response(extraction_prompt)
            
            try:
                learning_data = json.loads(response)
                await self._save_learning_data(learning_data, conversation['id'])
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse learning data from conversation {conversation['id']}")
            
        except Exception as e:
            logger.error(f"Failed to extract learning data: {e}")
    
    async def _save_learning_data(self, learning_data: Dict, conversation_id: int):
        """学習データを保存"""
        try:
            data = LearningData(
                id=f"conv_{conversation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                content=learning_data.get('content', ''),
                category=learning_data.get('category', 'general'),
                quality_score=0.8,  # 高品質な会話から抽出されたため
                usage_count=0,
                created_at=datetime.now(),
                last_used=datetime.now(),
                tags=learning_data.get('tags', []),
                metadata=learning_data.get('metadata', {})
            )
            
            # ファイルに保存
            file_path = self.learning_data_path / f"{data.category}_{data.id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(data), f, ensure_ascii=False, indent=2, default=str)
            
            # データベースにも保存
            await self.db.insert_learning_data(
                data_id=data.id,
                content=data.content,
                category=data.category,
                quality_score=data.quality_score,
                tags=json.dumps(data.tags),
                metadata=json.dumps(data.metadata)
            )
            
            logger.debug(f"Learning data saved: {data.id}")
            
        except Exception as e:
            logger.error(f"Failed to save learning data: {e}")
    
    async def _improve_existing_learning_data(self):
        """既存の学習データを改善"""
        try:
            # 低品質な学習データを特定
            low_quality_data = await self.db.get_learning_data_by_quality(
                max_score=0.6, limit=20
            )
            
            for data in low_quality_data:
                await self._improve_learning_data_item(data)
            
            # 使用頻度の低いデータを削除
            await self._cleanup_unused_learning_data()
            
        except Exception as e:
            logger.error(f"Failed to improve learning data: {e}")
    
    async def _improve_learning_data_item(self, data: Dict):
        """個別の学習データを改善"""
        try:
            improvement_prompt = f"""
以下の学習データを改善してください：

カテゴリ: {data.get('category')}
内容: {data.get('content')}
現在の品質スコア: {data.get('quality_score')}

より有用で正確な学習データに改善してください。
改善された内容を返してください。
"""
            
            improved_content = await self.ollama_client.generate_response(improvement_prompt)
            
            # 改善されたデータで更新
            await self.db.update_learning_data(
                data_id=data['id'],
                content=improved_content,
                quality_score=min(1.0, data.get('quality_score', 0.5) + 0.2)
            )
            
            logger.debug(f"Improved learning data: {data['id']}")
            
        except Exception as e:
            logger.error(f"Failed to improve learning data item: {e}")
    
    async def _cleanup_unused_learning_data(self):
        """使用されていない学習データを削除"""
        try:
            # 30日以上使用されていないデータを削除
            cutoff_date = datetime.now() - timedelta(days=30)
            
            unused_data = await self.db.get_unused_learning_data(cutoff_date, limit=50)
            
            for data in unused_data:
                await self.db.delete_learning_data(data['id'])
                
                # ファイルも削除
                file_path = self.learning_data_path / f"{data['category']}_{data['id']}.json"
                if file_path.exists():
                    file_path.unlink()
            
            logger.info(f"Cleaned up {len(unused_data)} unused learning data items")
            
        except Exception as e:
            logger.error(f"Failed to cleanup unused learning data: {e}")
    
    async def _optimize_prompts(self):
        """プロンプトの最適化"""
        try:
            # データベースからプロンプトテンプレートを取得
            prompt_templates = await self.db.get_all_prompt_templates()
            
            # 各プロンプトを最適化
            optimized_prompts = []
            
            for template in prompt_templates:
                optimized_prompt = await self._optimize_single_prompt(
                    template['name'], template
                )
                if optimized_prompt:
                    optimized_prompts.append(optimized_prompt)
            
            # 最適化されたプロンプトを保存
            if optimized_prompts:
                await self._save_optimized_prompts(optimized_prompts)
            
            logger.info("Prompt optimization completed")
            
        except Exception as e:
            logger.error(f"Prompt optimization failed: {e}")
    
    async def _optimize_single_prompt(self, prompt_key: str, prompt_info: Dict) -> Optional[Dict]:
        """単一のプロンプトを最適化"""
        try:
            original_prompt = prompt_info.get('template_content', '')
            
            optimization_prompt = f"""
以下のプロンプトを最適化してください：

プロンプト名: {prompt_info.get('name', '')}
説明: {prompt_info.get('description', '')}
現在のプロンプト: {original_prompt}

以下の点を考慮して最適化してください：
1. 明確性と簡潔性
2. 効果的な指示
3. 期待される出力の明確化
4. エラーハンドリングの改善

最適化されたプロンプトを返してください。
"""
            
            optimized_prompt_text = await self.ollama_client.generate_response(optimization_prompt)
            
            # 最適化の効果を評価
            improvement_score = await self._evaluate_prompt_improvement(
                original_prompt, optimized_prompt_text
            )
            
            if improvement_score > 0.1:  # 10%以上の改善がある場合のみ適用
                return {
                    **prompt_info,
                    'template_content': optimized_prompt_text,
                    'optimization_score': improvement_score,
                    'optimized_at': datetime.now().isoformat()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to optimize prompt {prompt_key}: {e}")
            return None
    
    async def _evaluate_prompt_improvement(self, original: str, optimized: str) -> float:
        """プロンプト改善の効果を評価"""
        try:
            evaluation_prompt = f"""
以下の2つのプロンプトを比較し、改善度を0.0から1.0のスコアで評価してください：

元のプロンプト:
{original}

最適化されたプロンプト:
{optimized}

評価基準：
- 明確性の向上
- 指示の効果性
- 出力の予測可能性
- エラーの減少

改善度スコア（0.0-1.0）のみを返してください。
"""
            
            response = await self.ollama_client.generate_response(evaluation_prompt)
            
            try:
                score = float(response.strip())
                return max(0.0, min(1.0, score))
            except ValueError:
                return 0.0
            
        except Exception as e:
            logger.error(f"Failed to evaluate prompt improvement: {e}")
            return 0.0
    
    async def _save_optimized_prompts(self, optimized_prompts: List[Dict]):
        """最適化されたプロンプトを保存"""
        try:
            for prompt in optimized_prompts:
                # データベースを更新
                await self.db.update_prompt_template(
                    name=prompt['name'],
                    template_content=prompt['template_content']
                )
                
                # 最適化履歴を記録
                await self.db.insert_prompt_optimization_history(
                    template_id=prompt.get('template_id', ''),
                    original_content=prompt.get('original_content', ''),
                    optimized_content=prompt['template_content'],
                    improvement_score=prompt.get('optimization_score', 0),
                    optimized_at=datetime.now()
                )
            
            logger.info(f"Optimized {len(optimized_prompts)} prompts")
            
        except Exception as e:
            logger.error(f"Failed to save optimized prompts: {e}")
    
    async def _extract_knowledge_from_conversations(self):
        """会話から知識を抽出"""
        try:
            # 高品質な会話から知識を抽出
            conversations = await self.db.get_conversations_by_quality(
                min_score=0.8, max_score=1.0, limit=30
            )
            
            for conv in conversations:
                await self._extract_knowledge_from_conversation(conv)
            
            # 知識ベースを整理・統合
            await self._consolidate_knowledge_base()
            
        except Exception as e:
            logger.error(f"Knowledge extraction failed: {e}")
    
    async def _extract_knowledge_from_conversation(self, conversation: Dict):
        """個別の会話から知識を抽出"""
        try:
            user_input = conversation.get('user_input', '')
            agent_response = conversation.get('agent_response', '')
            
            knowledge_prompt = f"""
以下の会話から有用な知識を抽出してください：

ユーザー: {user_input}
エージェント: {agent_response}

以下の形式で知識を抽出してください：
{{
    "fact": "事実や情報",
    "category": "カテゴリ",
    "confidence": 0.0-1.0,
    "source_context": "文脈",
    "applicability": "適用範囲"
}}

有用な知識がない場合は空のJSONを返してください。
"""
            
            response = await self.ollama_client.generate_response(knowledge_prompt)
            
            try:
                knowledge = json.loads(response)
                if knowledge and knowledge.get('fact'):
                    await self._save_knowledge_item(knowledge, conversation['id'])
            except json.JSONDecodeError:
                pass
            
        except Exception as e:
            logger.error(f"Failed to extract knowledge: {e}")
    
    async def _save_knowledge_item(self, knowledge: Dict, conversation_id: int):
        """知識アイテムを保存"""
        try:
            await self.db.insert_knowledge_item(
                fact=knowledge.get('fact', ''),
                category=knowledge.get('category', 'general'),
                confidence=knowledge.get('confidence', 0.5),
                source_context=knowledge.get('source_context', ''),
                applicability=knowledge.get('applicability', ''),
                source_conversation_id=conversation_id
            )
            
        except Exception as e:
            logger.error(f"Failed to save knowledge item: {e}")
    
    async def _consolidate_knowledge_base(self):
        """知識ベースの整理・統合"""
        try:
            # 重複する知識を統合
            duplicate_knowledge = await self.db.get_duplicate_knowledge_items()
            
            for group in duplicate_knowledge:
                await self._merge_knowledge_items(group)
            
            # 低信頼度の知識を削除
            await self.db.delete_low_confidence_knowledge(threshold=0.3)
            
            logger.info("Knowledge base consolidation completed")
            
        except Exception as e:
            logger.error(f"Knowledge consolidation failed: {e}")
    
    async def _merge_knowledge_items(self, knowledge_group: List[Dict]):
        """知識アイテムを統合"""
        try:
            if len(knowledge_group) < 2:
                return
            
            # 最も信頼度の高いアイテムをベースにする
            base_item = max(knowledge_group, key=lambda x: x.get('confidence', 0))
            
            # 他のアイテムの情報を統合
            merged_fact = base_item.get('fact', '')
            merged_confidence = base_item.get('confidence', 0)
            
            for item in knowledge_group:
                if item['id'] != base_item['id']:
                    # 信頼度を重み付き平均で更新
                    merged_confidence = (merged_confidence + item.get('confidence', 0)) / 2
                    
                    # データベースから削除
                    await self.db.delete_knowledge_item(item['id'])
            
            # ベースアイテムを更新
            await self.db.update_knowledge_item(
                knowledge_id=base_item['id'],
                fact=merged_fact,
                confidence=merged_confidence
            )
            
        except Exception as e:
            logger.error(f"Failed to merge knowledge items: {e}")
    
    async def _analyze_performance_and_adapt(self):
        """パフォーマンス分析と適応"""
        try:
            # パフォーマンス指標を取得
            performance_metrics = await self.db.get_performance_metrics(days=7)
            
            # パフォーマンスの傾向を分析
            trend_analysis = await self._analyze_performance_trends(performance_metrics)
            
            # 必要に応じて学習パラメータを調整
            await self._adapt_learning_parameters(trend_analysis)
            
            # 学習レポートを生成
            await self._generate_learning_report(performance_metrics, trend_analysis)
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
    
    async def _analyze_performance_trends(self, metrics: Dict) -> Dict:
        """パフォーマンス傾向を分析"""
        try:
            analysis_prompt = f"""
以下のパフォーマンス指標を分析し、傾向と改善提案を提供してください：

{json.dumps(metrics, indent=2, default=str)}

以下の形式で分析結果を返してください：
{{
    "trend": "改善/悪化/安定",
    "key_issues": ["問題1", "問題2"],
    "improvements": ["改善案1", "改善案2"],
    "recommendations": ["推奨事項1", "推奨事項2"]
}}
"""
            
            response = await self.ollama_client.generate_response(analysis_prompt)
            
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return {
                    "trend": "stable",
                    "key_issues": [],
                    "improvements": [],
                    "recommendations": []
                }
            
        except Exception as e:
            logger.error(f"Failed to analyze performance trends: {e}")
            return {"trend": "unknown", "key_issues": [], "improvements": [], "recommendations": []}
    
    async def _adapt_learning_parameters(self, trend_analysis: Dict):
        """学習パラメータを適応的に調整"""
        try:
            trend = trend_analysis.get('trend', 'stable')
            
            if trend == 'improving':
                # 改善傾向の場合は学習間隔を短縮
                current_interval = self.config.settings.learning_interval_minutes
                new_interval = max(5, current_interval - 5)
                logger.info(f"Reducing learning interval to {new_interval} minutes")
                
            elif trend == 'declining':
                # 悪化傾向の場合は学習間隔を延長
                current_interval = self.config.settings.learning_interval_minutes
                new_interval = min(120, current_interval + 10)
                logger.info(f"Increasing learning interval to {new_interval} minutes")
            
            # 設定を更新（実際の実装では設定ファイルを更新）
            
        except Exception as e:
            logger.error(f"Failed to adapt learning parameters: {e}")
    
    async def _generate_learning_report(self, metrics: Dict, analysis: Dict):
        """学習レポートを生成"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'performance_metrics': metrics,
                'trend_analysis': analysis,
                'learning_data_stats': await self._get_learning_data_stats(),
                'knowledge_base_stats': await self._get_knowledge_base_stats(),
                'prompt_optimization_stats': await self._get_prompt_optimization_stats()
            }
            
            # レポートをファイルに保存
            report_path = Path("src/data/logs/learning_report.json")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info("Learning report generated")
            
        except Exception as e:
            logger.error(f"Failed to generate learning report: {e}")
    
    async def _get_learning_data_stats(self) -> Dict:
        """学習データの統計を取得"""
        try:
            return await self.db.get_learning_data_stats()
        except Exception:
            return {}
    
    async def _get_knowledge_base_stats(self) -> Dict:
        """知識ベースの統計を取得"""
        try:
            return await self.db.get_knowledge_base_stats()
        except Exception:
            return {}
    
    async def _get_prompt_optimization_stats(self) -> Dict:
        """プロンプト最適化の統計を取得"""
        try:
            return await self.db.get_prompt_optimization_stats()
        except Exception:
            return {}
    
    async def get_learning_status(self) -> Dict[str, Any]:
        """学習システムの状態を取得"""
        return {
            'is_running': self.is_running,
            'active_tasks': list(self.learning_tasks.keys()),
            'learning_data_count': await self._get_learning_data_stats(),
            'knowledge_base_count': await self._get_knowledge_base_stats(),
            'last_optimization': await self._get_last_optimization_time()
        }
    
    async def _get_last_optimization_time(self) -> Optional[str]:
        """最後の最適化時刻を取得"""
        try:
            # 実際の実装ではデータベースから取得
            return datetime.now().isoformat()
        except Exception:
            return None
