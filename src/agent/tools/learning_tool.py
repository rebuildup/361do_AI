"""
Learning Tool
自己学習システムを操作するためのツール
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from agent.core.config import Config
from agent.core.database import DatabaseManager
from agent.core.ollama_client import OllamaClient
from agent.self_tuning.advanced_learning import AdvancedLearningSystem


class LearningTool:
    """自己学習ツール"""

    def __init__(
        self,
        db_manager: DatabaseManager,
        config: Config,
        ollama_client: OllamaClient,
        agent_manager=None
    ):
        self.db = db_manager
        self.config = config
        self.ollama_client = ollama_client
        self.agent_manager = agent_manager
        self.learning_system = AdvancedLearningSystem(
            agent_manager=agent_manager,
            db_manager=db_manager,
            config=config,
            ollama_client=ollama_client
        )

    async def start_learning_system(self) -> Dict[str, Any]:
        """学習システムを開始"""
        try:
            await self.learning_system.start_advanced_learning()
            return {
                "status": "success",
                "message": "Advanced learning system started successfully",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to start learning system: {e}")
            return {
                "status": "error",
                "message": f"Failed to start learning system: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    async def stop_learning_system(self) -> Dict[str, Any]:
        """学習システムを停止"""
        try:
            await self.learning_system.stop()
            return {
                "status": "success",
                "message": "Learning system stopped successfully",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to stop learning system: {e}")
            return {
                "status": "error",
                "message": f"Failed to stop learning system: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    async def get_learning_status(self) -> Dict[str, Any]:
        """学習システムの状態を取得"""
        try:
            status = await self.learning_system.get_learning_status()
            return {
                "status": "success",
                "data": status,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get learning status: {e}")
            return {
                "status": "error",
                "message": f"Failed to get learning status: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    async def manually_trigger_learning_cycle(self) -> Dict[str, Any]:
        """手動で学習サイクルを実行"""
        try:
            # 学習データ分析
            await self.learning_system._analyze_and_improve_learning_data()

            # プロンプト最適化
            await self.learning_system._optimize_prompts()

            # 知識抽出
            await self.learning_system._extract_knowledge_from_conversations()

            # パフォーマンス分析
            await self.learning_system._analyze_performance_and_adapt()

            return {
                "status": "success",
                "message": "Manual learning cycle completed successfully",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to trigger manual learning cycle: {e}")
            return {
                "status": "error",
                "message": f"Failed to trigger manual learning cycle: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    async def add_custom_learning_data(
        self,
        content: str,
        category: str,
        tags: Optional[List[str]] = None,
        metadata_json: Optional[Dict[str, Any]] = None,
        quality_score: float = 0.8
    ) -> Dict[str, Any]:
        """カスタム学習データを追加"""
        try:
            data_id = f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

            await self.db.insert_learning_data(
                data_id=data_id,
                content=content,
                category=category,
                quality_score=quality_score,
                tags=json.dumps(tags or []),
                metadata_json=json.dumps(metadata_json or {})
            )

            return {
                "status": "success",
                "message": f"Custom learning data added with ID: {data_id}",
                "data_id": data_id,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to add custom learning data: {e}")
            return {
                "status": "error",
                "message": f"Failed to add custom learning data: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    async def get_learning_data(
        self,
        category: Optional[str] = None,
        min_quality: Optional[float] = None,
        limit: int = 20
    ) -> Dict[str, Any]:
        """学習データを取得"""
        try:
            data = await self.db.get_learning_data_by_quality(
                min_score=min_quality,
                limit=limit
            )

            # カテゴリでフィルタリング
            if category:
                data = [item for item in data if item.get('category') == category]

            return {
                "status": "success",
                "data": data,
                "count": len(data),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get learning data: {e}")
            return {
                "status": "error",
                "message": f"Failed to get learning data: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    async def update_learning_data(
        self,
        data_id: str,
        content: Optional[str] = None,
        quality_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """学習データを更新"""
        try:
            await self.db.update_learning_data(
                data_id=data_id,
                content=content,
                quality_score=quality_score
            )

            return {
                "status": "success",
                "message": f"Learning data {data_id} updated successfully",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to update learning data: {e}")
            return {
                "status": "error",
                "message": f"Failed to update learning data: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    async def delete_learning_data(self, data_id: str) -> Dict[str, Any]:
        """学習データを削除"""
        try:
            await self.db.delete_learning_data(data_id)

            return {
                "status": "success",
                "message": f"Learning data {data_id} deleted successfully",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to delete learning data: {e}")
            return {
                "status": "error",
                "message": f"Failed to delete learning data: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    async def get_knowledge_base(self, category: Optional[str] = None) -> Dict[str, Any]:
        """知識ベースを取得"""
        try:
            # 知識ベースの統計を取得
            stats = await self.db.get_knowledge_base_stats()

            # 知識アイテムを取得（簡易実装）
            query = "SELECT * FROM knowledge_items"
            params = []

            if category:
                query += " WHERE category = ?"
                params.append(category)

            query += " ORDER BY confidence DESC LIMIT 50"

            rows = await self.db.execute_query(query, tuple(params))
            knowledge_items = [dict(row) for row in rows] if rows else []

            return {
                "status": "success",
                "stats": stats,
                "knowledge_items": knowledge_items,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get knowledge base: {e}")
            return {
                "status": "error",
                "message": f"Failed to get knowledge base: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    async def add_knowledge_item(
        self,
        fact: str,
        category: str,
        confidence: float,
        source_context: Optional[str] = None,
        applicability: Optional[str] = None
    ) -> Dict[str, Any]:
        """知識アイテムを追加"""
        try:
            await self.db.insert_knowledge_item(
                fact=fact,
                category=category,
                confidence=confidence,
                source_context=source_context,
                applicability=applicability
            )

            return {
                "status": "success",
                "message": "Knowledge item added successfully",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to add knowledge item: {e}")
            return {
                "status": "error",
                "message": f"Failed to add knowledge item: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    async def optimize_specific_prompt(
        self,
        prompt_key: str,
        prompt_content: str
    ) -> Dict[str, Any]:
        """特定のプロンプトを最適化"""
        try:
            prompt_info = {
                "name": prompt_key,
                "description": f"Optimized version of {prompt_key}",
                "prompt": prompt_content
            }

            optimized_prompt = await self.learning_system._optimize_single_prompt(
                prompt_key, prompt_info
            )

            if optimized_prompt:
                return {
                    "status": "success",
                    "message": "Prompt optimized successfully",
                    "original_prompt": prompt_content,
                    "optimized_prompt": optimized_prompt['prompt'],
                    "improvement_score": optimized_prompt.get('optimization_score', 0),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "info",
                    "message": "No significant improvement found for this prompt",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Failed to optimize prompt: {e}")
            return {
                "status": "error",
                "message": f"Failed to optimize prompt: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    async def get_performance_report(self, days: int = 7) -> Dict[str, Any]:
        """パフォーマンスレポートを取得"""
        try:
            # パフォーマンス指標を取得
            metrics = await self.db.get_performance_metrics(days=days)

            # 学習データ統計
            learning_stats = await self.db.get_learning_data_stats()

            # 知識ベース統計
            knowledge_stats = await self.db.get_knowledge_base_stats()

            # プロンプト最適化統計
            prompt_stats = await self.db.get_prompt_optimization_stats()

            report = {
                "period_days": days,
                "performance_metrics": metrics,
                "learning_stats": learning_stats,
                "knowledge_stats": knowledge_stats,
                "prompt_optimization_stats": prompt_stats,
                "timestamp": datetime.now().isoformat()
            }

            return {
                "status": "success",
                "report": report
            }
        except Exception as e:
            logger.error(f"Failed to get performance report: {e}")
            return {
                "status": "error",
                "message": f"Failed to get performance report: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    async def export_learning_data(self, format: str = "json") -> Dict[str, Any]:
        """学習データをエクスポート"""
        try:
            if format.lower() == "json":
                # 学習データを取得
                learning_data = await self.db.get_learning_data_by_quality(limit=1000)

                # 知識ベースを取得
                knowledge_query = "SELECT * FROM knowledge_items ORDER BY confidence DESC LIMIT 1000"
                knowledge_rows = await self.db.execute_query(knowledge_query)
                knowledge_items = [dict(row) for row in knowledge_rows] if knowledge_rows else []

                export_data = {
                    "learning_data": learning_data,
                    "knowledge_items": knowledge_items,
                    "export_timestamp": datetime.now().isoformat(),
                    "total_learning_items": len(learning_data),
                    "total_knowledge_items": len(knowledge_items)
                }

                return {
                    "status": "success",
                    "format": "json",
                    "data": export_data,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "error",
                    "message": f"Unsupported export format: {format}",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Failed to export learning data: {e}")
            return {
                "status": "error",
                "message": f"Failed to export learning data: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    async def import_learning_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """学習データをインポート"""
        try:
            imported_count = 0

            # 学習データをインポート
            if "learning_data" in data:
                for item in data["learning_data"]:
                    try:
                        await self.db.insert_learning_data(
                            data_id=item.get("id", f"imported_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                            content=item.get("content", ""),
                            category=item.get("category", "imported"),
                            quality_score=item.get("quality_score", 0.5),
                            tags=item.get("tags"),
                            metadata_json=item.get("metadata_json")
                        )
                        imported_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to import learning data item: {e}")

            # 知識アイテムをインポート
            if "knowledge_items" in data:
                for item in data["knowledge_items"]:
                    try:
                        await self.db.insert_knowledge_item(
                            fact=item.get("fact", ""),
                            category=item.get("category", "imported"),
                            confidence=item.get("confidence", 0.5),
                            source_context=item.get("source_context"),
                            applicability=item.get("applicability")
                        )
                        imported_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to import knowledge item: {e}")

            return {
                "status": "success",
                "message": f"Successfully imported {imported_count} items",
                "imported_count": imported_count,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to import learning data: {e}")
            return {
                "status": "error",
                "message": f"Failed to import learning data: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    # プロンプト管理機能
    async def get_prompt_templates(self) -> Dict[str, Any]:
        """プロンプトテンプレート一覧を取得"""
        try:
            # データベースからプロンプトテンプレートを取得
            query = "SELECT * FROM prompt_templates ORDER BY created_at DESC"
            rows = await self.db.execute_query(query)
            templates = [dict(row) for row in rows] if rows else []

            return {
                "status": "success",
                "data": templates,
                "count": len(templates),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get prompt templates: {e}")
            return {
                "status": "error",
                "message": f"Failed to get prompt templates: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    async def add_prompt_template(
        self,
        name: str,
        content: str,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """プロンプトテンプレートを追加"""
        try:
            # If a template with the same name exists, treat this as an update (create new version)
            existing = await self.db.get_prompt_template_by_name(name)
            if existing:
                await self.db.update_prompt_template(name=name, template_content=content, description=description)
                return {
                    "status": "success",
                    "message": f"Prompt template '{name}' updated (new version) successfully",
                    "template_id": existing.get('template_id'),
                    "timestamp": datetime.now().isoformat()
                }

            # Use UUID to avoid collisions when creating a brand-new template
            import uuid
            template_id = f"prompt_{uuid.uuid4().hex}"

            try:
                await self.db.insert_prompt_template(
                    template_id=template_id,
                    name=name,
                    template_content=content,
                    description=description or f"Custom prompt: {name}",
                    category="custom"
                )
            except Exception as e:
                logger.warning(f"Insert prompt template failed, attempting update: {e}")
                # Fallback: try to update existing template (create new version)
                try:
                    await self.db.update_prompt_template(name=name, template_content=content, description=description)
                    return {
                        "status": "success",
                        "message": f"Prompt template '{name}' updated (fallback) successfully",
                        "template_id": template_id,
                        "timestamp": datetime.now().isoformat()
                    }
                except Exception as e2:
                    logger.error(f"Failed to update prompt template after insert failure: {e2}")
                    raise

            return {
                "status": "success",
                "message": f"Prompt template '{name}' added successfully",
                "template_id": template_id,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to add prompt template: {e}")
            return {
                "status": "error",
                "message": f"Failed to add prompt template: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    async def update_prompt_template(
        self,
        name: str,
        content: str
    ) -> Dict[str, Any]:
        """プロンプトテンプレートを更新"""
        try:
            await self.db.update_prompt_template(
                name=name,
                template_content=content
            )

            return {
                "status": "success",
                "message": f"Prompt template '{name}' updated successfully",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to update prompt template: {e}")
            return {
                "status": "error",
                "message": f"Failed to update prompt template: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    async def delete_prompt_template(self, name: str) -> Dict[str, Any]:
        """プロンプトテンプレートを削除"""
        try:
            await self.db.delete_prompt_template(name)

            return {
                "status": "success",
                "message": f"Prompt template '{name}' deleted successfully",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to delete prompt template: {e}")
            return {
                "status": "error",
                "message": f"Failed to delete prompt template: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    async def optimize_prompt_template(self, name: str) -> Dict[str, Any]:
        """プロンプトテンプレートを最適化"""
        try:
            # 既存のプロンプトを取得
            template = await self.db.get_prompt_template_by_name(name)
            if not template:
                # テンプレートが存在しない場合は自動で追加して再取得（テストの繰り返し実行に備える）
                await self.add_prompt_template(name=name, content="ユーザーの質問に答えてください。", description="Auto-created for optimization")
                template = await self.db.get_prompt_template_by_name(name)
                if not template:
                    return {
                        "status": "error",
                        "message": f"Prompt template '{name}' not found after auto-create",
                        "timestamp": datetime.now().isoformat()
                    }

            # 最適化実行
            result = await self.optimize_specific_prompt(
                prompt_key=name,
                prompt_content=template['template_content']
            )

            if result.get('status') == 'success':
                # 最適化されたプロンプトで更新
                await self.db.update_prompt_template(
                    name=name,
                    template_content=result['optimized_prompt']
                )

                return {
                    "status": "success",
                    "message": f"Prompt template '{name}' optimized successfully",
                    "improvement_score": result.get('improvement_score', 0),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return result

        except Exception as e:
            logger.error(f"Failed to optimize prompt template: {e}")
            return {
                "status": "error",
                "message": f"Failed to optimize prompt template: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    async def export_prompt_templates(self) -> Dict[str, Any]:
        """プロンプトテンプレートをエクスポート"""
        try:
            templates = await self.get_prompt_templates()

            if templates.get('status') == 'success':
                export_data = {
                    "prompt_templates": templates.get('data', []),
                    "export_timestamp": datetime.now().isoformat(),
                    "total_templates": templates.get('count', 0)
                }

                return {
                    "status": "success",
                    "data": export_data,
                    "count": templates.get('count', 0),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return templates

        except Exception as e:
            logger.error(f"Failed to export prompt templates: {e}")
            return {
                "status": "error",
                "message": f"Failed to export prompt templates: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    async def import_prompt_templates(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """プロンプトテンプレートをインポート"""
        try:
            imported_count = 0

            if "prompt_templates" in data:
                for template in data["prompt_templates"]:
                    try:
                        await self.db.insert_prompt_template(
                            template_id=template.get("template_id", f"imported_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                            name=template.get("name", ""),
                            template_content=template.get("template_content", ""),
                            description=template.get("description", ""),
                            category=template.get("category", "imported")
                        )
                        imported_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to import prompt template: {e}")

            return {
                "status": "success",
                "message": f"Successfully imported {imported_count} prompt templates",
                "imported_count": imported_count,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to import prompt templates: {e}")
            return {
                "status": "error",
                "message": f"Failed to import prompt templates: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
