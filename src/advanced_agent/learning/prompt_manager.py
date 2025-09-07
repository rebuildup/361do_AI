"""
Prompt Manager

プロンプトテンプレートの動的管理システム
"""

import asyncio
import logging
import json
import re
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path
import hashlib
import uuid

from ..reasoning.ollama_client import OllamaClient
from ..database.models import PromptTemplate
from ..database.connection import get_database_manager


class PromptTemplate:
    """プロンプトテンプレートクラス"""
    
    def __init__(self, 
                 name: str,
                 template: str,
                 description: str = "",
                 category: str = "general",
                 version: str = "1.0.0",
                 tags: Optional[List[str]] = None,
                 variables: Optional[List[str]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.id = str(uuid.uuid4())
        self.name = name
        self.template = template
        self.description = description
        self.category = category
        self.version = version
        self.tags = tags or []
        self.variables = variables or []
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.usage_count = 0
        self.success_rate = 0.0
        self.logger = logging.getLogger(__name__)
    
    def render(self, **kwargs) -> str:
        """プロンプトテンプレートをレンダリング"""
        
        try:
            # 変数の検証
            missing_vars = set(self.variables) - set(kwargs.keys())
            if missing_vars:
                self.logger.warning(f"Missing variables: {missing_vars}")
            
            # テンプレートをレンダリング
            rendered = self.template.format(**kwargs)
            
            # 使用回数を増加
            self.usage_count += 1
            
            return rendered
            
        except KeyError as e:
            self.logger.error(f"Missing required variable: {e}")
            raise ValueError(f"Missing required variable: {e}")
        except Exception as e:
            self.logger.error(f"Template rendering error: {e}")
            raise
    
    def extract_variables(self) -> List[str]:
        """テンプレートから変数を抽出"""
        
        try:
            # {variable} 形式の変数を抽出
            pattern = r'\{([^}]+)\}'
            variables = re.findall(pattern, self.template)
            
            # 重複を除去
            unique_variables = list(set(variables))
            
            # 変数リストを更新
            self.variables = unique_variables
            
            return unique_variables
            
        except Exception as e:
            self.logger.error(f"Variable extraction error: {e}")
            return []
    
    def validate(self) -> Dict[str, Any]:
        """プロンプトテンプレートを検証"""
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "score": 0
        }
        
        try:
            # 基本チェック
            if not self.name:
                validation_result["errors"].append("Template name is required")
                validation_result["valid"] = False
            else:
                validation_result["score"] += 10
            
            if not self.template:
                validation_result["errors"].append("Template content is required")
                validation_result["valid"] = False
            else:
                validation_result["score"] += 20
            
            if not self.description:
                validation_result["warnings"].append("Template description is recommended")
            else:
                validation_result["score"] += 10
            
            # 変数の整合性チェック
            extracted_vars = self.extract_variables()
            if self.variables and set(self.variables) != set(extracted_vars):
                validation_result["warnings"].append("Variable list doesn't match template variables")
            
            # テンプレートの構文チェック
            try:
                # サンプルデータでレンダリングテスト
                sample_data = {var: f"sample_{var}" for var in extracted_vars}
                self.render(**sample_data)
                validation_result["score"] += 20
            except Exception as e:
                validation_result["errors"].append(f"Template rendering test failed: {str(e)}")
                validation_result["valid"] = False
            
            # 長さチェック
            if len(self.template) > 10000:
                validation_result["warnings"].append("Template is very long (>10KB)")
            elif len(self.template) < 10:
                validation_result["warnings"].append("Template is very short (<10 chars)")
            
            # スコアの正規化
            validation_result["score"] = min(validation_result["score"], 100)
            
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {str(e)}")
            validation_result["valid"] = False
        
        return validation_result
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        
        return {
            "id": self.id,
            "name": self.name,
            "template": self.template,
            "description": self.description,
            "category": self.category,
            "version": self.version,
            "tags": self.tags,
            "variables": self.variables,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "usage_count": self.usage_count,
            "success_rate": self.success_rate
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptTemplate':
        """辞書から作成"""
        
        template = cls(
            name=data["name"],
            template=data["template"],
            description=data.get("description", ""),
            category=data.get("category", "general"),
            version=data.get("version", "1.0.0"),
            tags=data.get("tags", []),
            variables=data.get("variables", []),
            metadata=data.get("metadata", {})
        )
        
        template.id = data.get("id", str(uuid.uuid4()))
        template.created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        template.updated_at = datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat()))
        template.usage_count = data.get("usage_count", 0)
        template.success_rate = data.get("success_rate", 0.0)
        
        return template


class PromptManager:
    """プロンプト管理システム"""
    
    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        self.ollama_client = ollama_client
        self.templates: Dict[str, PromptTemplate] = {}
        self.logger = logging.getLogger(__name__)
        
        # デフォルトテンプレートを読み込み
        self._load_default_templates()
    
    def _load_default_templates(self):
        """デフォルトテンプレートを読み込み"""
        
        try:
            default_templates = [
                {
                    "name": "general_chat",
                    "template": "You are a helpful AI assistant. Please respond to the following question: {question}",
                    "description": "一般的なチャット用プロンプト",
                    "category": "chat",
                    "variables": ["question"]
                },
                {
                    "name": "code_generation",
                    "template": "You are an expert programmer. Generate code for the following task: {task}\n\nRequirements: {requirements}\n\nLanguage: {language}",
                    "description": "コード生成用プロンプト",
                    "category": "coding",
                    "variables": ["task", "requirements", "language"]
                },
                {
                    "name": "text_analysis",
                    "template": "Analyze the following text and provide insights:\n\nText: {text}\n\nAnalysis type: {analysis_type}\n\nPlease provide: {output_format}",
                    "description": "テキスト分析用プロンプト",
                    "category": "analysis",
                    "variables": ["text", "analysis_type", "output_format"]
                },
                {
                    "name": "problem_solving",
                    "template": "Solve the following problem step by step:\n\nProblem: {problem}\n\nContext: {context}\n\nApproach: {approach}",
                    "description": "問題解決用プロンプト",
                    "category": "reasoning",
                    "variables": ["problem", "context", "approach"]
                }
            ]
            
            for template_data in default_templates:
                template = PromptTemplate.from_dict(template_data)
                self.templates[template.name] = template
            
            self.logger.info(f"Loaded {len(default_templates)} default templates")
            
        except Exception as e:
            self.logger.error(f"Failed to load default templates: {e}")
    
    def create_template(self, 
                       name: str,
                       template: str,
                       description: str = "",
                       category: str = "general",
                       tags: Optional[List[str]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> Optional[PromptTemplate]:
        """プロンプトテンプレートを作成"""
        
        try:
            if name in self.templates:
                self.logger.warning(f"Template already exists: {name}")
                return None
            
            prompt_template = PromptTemplate(
                name=name,
                template=template,
                description=description,
                category=category,
                tags=tags or [],
                metadata=metadata or {}
            )
            
            # 変数を自動抽出
            prompt_template.extract_variables()
            
            # 検証
            validation = prompt_template.validate()
            if not validation["valid"]:
                self.logger.error(f"Template validation failed: {validation['errors']}")
                return None
            
            self.templates[name] = prompt_template
            self.logger.info(f"Template created: {name}")
            
            return prompt_template
            
        except Exception as e:
            self.logger.error(f"Failed to create template: {e}")
            return None
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """プロンプトテンプレートを取得"""
        
        return self.templates.get(name)
    
    def update_template(self, 
                       name: str,
                       template: Optional[str] = None,
                       description: Optional[str] = None,
                       category: Optional[str] = None,
                       tags: Optional[List[str]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """プロンプトテンプレートを更新"""
        
        try:
            if name not in self.templates:
                self.logger.error(f"Template not found: {name}")
                return False
            
            prompt_template = self.templates[name]
            
            # 更新
            if template is not None:
                prompt_template.template = template
                prompt_template.extract_variables()  # 変数を再抽出
            
            if description is not None:
                prompt_template.description = description
            
            if category is not None:
                prompt_template.category = category
            
            if tags is not None:
                prompt_template.tags = tags
            
            if metadata is not None:
                prompt_template.metadata.update(metadata)
            
            prompt_template.updated_at = datetime.now()
            
            # バージョンを更新
            version_parts = prompt_template.version.split('.')
            version_parts[-1] = str(int(version_parts[-1]) + 1)
            prompt_template.version = '.'.join(version_parts)
            
            # 検証
            validation = prompt_template.validate()
            if not validation["valid"]:
                self.logger.error(f"Template validation failed: {validation['errors']}")
                return False
            
            self.logger.info(f"Template updated: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update template: {e}")
            return False
    
    def delete_template(self, name: str) -> bool:
        """プロンプトテンプレートを削除"""
        
        try:
            if name not in self.templates:
                self.logger.warning(f"Template not found: {name}")
                return False
            
            del self.templates[name]
            self.logger.info(f"Template deleted: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete template: {e}")
            return False
    
    def list_templates(self, 
                      category: Optional[str] = None,
                      tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """プロンプトテンプレート一覧を取得"""
        
        templates = []
        
        for template in self.templates.values():
            # カテゴリフィルタ
            if category and template.category != category:
                continue
            
            # タグフィルタ
            if tags and not any(tag in template.tags for tag in tags):
                continue
            
            templates.append(template.to_dict())
        
        # 使用回数でソート
        templates.sort(key=lambda x: x["usage_count"], reverse=True)
        
        return templates
    
    def search_templates(self, query: str) -> List[Dict[str, Any]]:
        """プロンプトテンプレートを検索"""
        
        query_lower = query.lower()
        results = []
        
        for template in self.templates.values():
            # 名前、説明、カテゴリ、タグで検索
            if (query_lower in template.name.lower() or
                query_lower in template.description.lower() or
                query_lower in template.category.lower() or
                any(query_lower in tag.lower() for tag in template.tags)):
                
                results.append(template.to_dict())
        
        return results
    
    async def generate_template(self, 
                              description: str,
                              category: str = "general",
                              example_inputs: Optional[Dict[str, Any]] = None) -> Optional[PromptTemplate]:
        """LLMを使用してプロンプトテンプレートを生成"""
        
        try:
            if not description:
                self.logger.error("Description is required for template generation")
                return None
            
            self.logger.info(f"Generating template: {description}")
            
            # プロンプト生成用のプロンプト
            generation_prompt = f"""
以下の要件に基づいて、効果的なプロンプトテンプレートを生成してください。

要件:
- カテゴリ: {category}
- 説明: {description}
- 例の入力: {example_inputs or "なし"}

生成するプロンプトテンプレートの要件:
1. 明確で具体的な指示を含む
2. 変数は {{variable_name}} 形式で定義
3. 適切なコンテキストと制約を提供
4. 期待される出力形式を指定
5. 長すぎず短すぎない適切な長さ

以下のJSON形式で返してください:
{{
    "name": "template_name",
    "template": "プロンプトテンプレートの内容",
    "description": "テンプレートの説明",
    "category": "{category}",
    "tags": ["tag1", "tag2"],
    "variables": ["variable1", "variable2"]
}}
"""
            
            response = await self.ollama_client.generate_response(generation_prompt)
            
            # JSONを解析
            try:
                template_data = json.loads(response)
            except json.JSONDecodeError:
                # JSON解析に失敗した場合は、レスポンスから情報を抽出
                template_data = self._extract_template_from_response(response, description, category)
            
            if not template_data:
                return None
            
            # テンプレートを作成
            template = PromptTemplate(
                name=template_data["name"],
                template=template_data["template"],
                description=template_data.get("description", description),
                category=template_data.get("category", category),
                tags=template_data.get("tags", []),
                metadata={"generated": True, "generation_prompt": generation_prompt}
            )
            
            # 変数を自動抽出
            template.extract_variables()
            
            # 検証
            validation = template.validate()
            if not validation["valid"]:
                self.logger.error(f"Generated template validation failed: {validation['errors']}")
                return None
            
            # テンプレートを保存
            self.templates[template.name] = template
            
            self.logger.info(f"Template generated successfully: {template.name}")
            return template
            
        except Exception as e:
            self.logger.error(f"Template generation failed: {e}")
            return None
    
    def _extract_template_from_response(self, 
                                      response: str, 
                                      description: str, 
                                      category: str) -> Optional[Dict[str, Any]]:
        """レスポンスからテンプレート情報を抽出"""
        
        try:
            # テンプレート名を生成
            name = f"generated_{category}_{len(self.templates) + 1}"
            name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
            
            # テンプレート内容を抽出
            template_content = response.strip()
            
            # 変数を抽出
            variables = re.findall(r'\{([^}]+)\}', template_content)
            unique_variables = list(set(variables))
            
            return {
                "name": name,
                "template": template_content,
                "description": description,
                "category": category,
                "tags": ["generated"],
                "variables": unique_variables
            }
            
        except Exception as e:
            self.logger.error(f"Template extraction failed: {e}")
            return None
    
    def render_template(self, name: str, **kwargs) -> Optional[str]:
        """プロンプトテンプレートをレンダリング"""
        
        try:
            template = self.get_template(name)
            if not template:
                self.logger.error(f"Template not found: {name}")
                return None
            
            return template.render(**kwargs)
            
        except Exception as e:
            self.logger.error(f"Template rendering failed: {e}")
            return None
    
    def get_template_stats(self) -> Dict[str, Any]:
        """テンプレート統計を取得"""
        
        stats = {
            "total_templates": len(self.templates),
            "categories": {},
            "total_usage": 0,
            "average_success_rate": 0.0
        }
        
        total_success_rate = 0.0
        
        for template in self.templates.values():
            # カテゴリ統計
            category = template.category
            if category not in stats["categories"]:
                stats["categories"][category] = 0
            stats["categories"][category] += 1
            
            # 使用統計
            stats["total_usage"] += template.usage_count
            total_success_rate += template.success_rate
        
        # 平均成功率
        if self.templates:
            stats["average_success_rate"] = total_success_rate / len(self.templates)
        
        return stats
    
    def export_templates(self, file_path: str) -> bool:
        """テンプレートをエクスポート"""
        
        try:
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "templates": [template.to_dict() for template in self.templates.values()]
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Templates exported to: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Template export failed: {e}")
            return False
    
    def import_templates(self, file_path: str) -> bool:
        """テンプレートをインポート"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            imported_count = 0
            
            for template_data in import_data.get("templates", []):
                template = PromptTemplate.from_dict(template_data)
                
                # 既存のテンプレートと重複しないように名前を調整
                original_name = template.name
                counter = 1
                while template.name in self.templates:
                    template.name = f"{original_name}_imported_{counter}"
                    counter += 1
                
                self.templates[template.name] = template
                imported_count += 1
            
            self.logger.info(f"Imported {imported_count} templates from: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Template import failed: {e}")
            return False
    
    async def self_improve_prompt(self, 
                                template_name: str,
                                improvement_context: str,
                                performance_feedback: Optional[Dict[str, Any]] = None) -> bool:
        """エージェントが独自のプロンプトを自己改善する機能"""
        
        try:
            if template_name not in self.templates:
                self.logger.error(f"Template '{template_name}' not found for self-improvement")
                return False
            
            original_template = self.templates[template_name]
            self.logger.info(f"Starting self-improvement for template: {template_name}")
            
            # 改善プロンプトを構築
            improvement_prompt = self._build_improvement_prompt(
                original_template, 
                improvement_context, 
                performance_feedback
            )
            
            # LLMを使用してプロンプトを改善
            if self.ollama_client:
                improved_content = await self._generate_improved_prompt(improvement_prompt)
                
                if improved_content:
                    # 新しいバージョンのテンプレートを作成
                    new_version = self._increment_version(original_template.version)
                    improved_template = PromptTemplate(
                        name=f"{template_name}_v{new_version}",
                        template=improved_content,
                        description=f"Self-improved version of {template_name}",
                        category=original_template.category,
                        version=new_version,
                        tags=original_template.tags + ["self-improved"],
                        variables=original_template.variables,
                        metadata={
                            **original_template.metadata,
                            "improvement_context": improvement_context,
                            "performance_feedback": performance_feedback,
                            "improvement_timestamp": datetime.now().isoformat(),
                            "original_template_id": original_template.id
                        }
                    )
                    
                    # 改善されたテンプレートを保存
                    self.templates[improved_template.name] = improved_template
                    
                    # データベースに保存
                    await self._save_improved_template(improved_template)
                    
                    self.logger.info(f"Successfully improved template: {template_name} -> {improved_template.name}")
                    return True
                else:
                    self.logger.warning(f"Failed to generate improved content for: {template_name}")
                    return False
            else:
                self.logger.error("Ollama client not available for self-improvement")
                return False
                
        except Exception as e:
            self.logger.error(f"Self-improvement failed for template '{template_name}': {e}")
            return False
    
    def _build_improvement_prompt(self, 
                                template: PromptTemplate,
                                context: str,
                                feedback: Optional[Dict[str, Any]]) -> str:
        """改善プロンプトを構築"""
        
        prompt = f"""あなたはAIエージェントのプロンプト改善専門家です。以下のプロンプトテンプレートを改善してください。

【現在のプロンプトテンプレート】
名前: {template.name}
説明: {template.description}
カテゴリ: {template.category}
バージョン: {template.version}

【現在のテンプレート内容】
{template.template}

【改善のコンテキスト】
{context}

【パフォーマンスフィードバック】
{json.dumps(feedback, ensure_ascii=False, indent=2) if feedback else "フィードバックなし"}

【改善の指針】
1. 現在のテンプレートの良い点を維持する
2. 明確性と効果性を向上させる
3. より具体的で実行可能な指示を含める
4. コンテキストに基づいて最適化する
5. 日本語で自然で理解しやすい表現を使用する

【改善されたプロンプトテンプレート】
改善されたプロンプトテンプレートのみを出力してください。説明やコメントは不要です。"""
        
        return prompt
    
    async def _generate_improved_prompt(self, improvement_prompt: str) -> Optional[str]:
        """LLMを使用して改善されたプロンプトを生成"""
        
        try:
            # Ollamaクライアントを直接使用
            if self.ollama_client and hasattr(self.ollama_client, 'generate'):
                # 直接テキスト生成
                response = await self.ollama_client.generate(
                    prompt=improvement_prompt,
                    temperature=0.7,
                    max_tokens=2000,
                    system_message="あなたはプロンプトエンジニアリングの専門家です。"
                )
                
                if response:
                    # レスポンスからプロンプト部分のみを抽出
                    improved_content = response.strip()
                    
                    # 基本的な検証
                    if len(improved_content) > 50:
                        return improved_content
                    else:
                        self.logger.warning("Generated content seems too short")
                        return None
                else:
                    self.logger.warning("No response from LLM for prompt improvement")
                    return None
            else:
                # フォールバック: シンプルな改善
                self.logger.warning("Ollama client not available, using fallback improvement")
                return self._fallback_improve_prompt(improvement_prompt)
                
        except Exception as e:
            self.logger.error(f"Failed to generate improved prompt: {e}")
            return None
    
    def _fallback_improve_prompt(self, improvement_prompt: str) -> str:
        """フォールバック改善（Ollamaクライアントが利用できない場合）"""
        
        # シンプルな改善ロジック
        improved_prompt = improvement_prompt
        
        # 基本的な改善を適用
        improvements = [
            ("より具体的で実行可能な指示を含める", "具体的で実行可能な指示を含めてください。"),
            ("日本語で自然で理解しやすい表現を使用する", "日本語で自然で理解しやすい表現を使用してください。"),
            ("ユーザーの理解を促進する", "ユーザーの理解を促進するように説明してください。"),
            ("エラーハンドリングを強化する", "エラーが発生した場合の適切な処理を含めてください。")
        ]
        
        for old_text, new_text in improvements:
            if old_text in improved_prompt:
                improved_prompt = improved_prompt.replace(old_text, new_text)
        
        # 改善されたプロンプトに追加の指示を追加
        improved_prompt += "\n\n【改善されたプロンプトテンプレート】\n"
        improved_prompt += "あなたは自己学習型AIエージェントです。ユーザーの要求を理解し、適切な応答を生成してください。\n"
        improved_prompt += "応答は明確で構造化され、ユーザーにとって有用な情報を含めてください。\n"
        improved_prompt += "必要に応じて、具体的な例や詳細な説明を提供してください。"
        
        return improved_prompt
    
    def _increment_version(self, current_version: str) -> str:
        """バージョン番号を増分"""
        
        try:
            # シンプルなバージョン管理: v1.0.0 -> v1.0.1
            if current_version.startswith('v'):
                version_parts = current_version[1:].split('.')
                if len(version_parts) >= 3:
                    patch = int(version_parts[2]) + 1
                    return f"v{version_parts[0]}.{version_parts[1]}.{patch}"
            
            # デフォルト: v1.0.1
            return "v1.0.1"
            
        except Exception:
            return "v1.0.1"
    
    async def _save_improved_template(self, template: PromptTemplate):
        """改善されたテンプレートをデータベースに保存"""
        
        try:
            db_manager = get_database_manager()
            
            # SQLAlchemyモデルに変換
            from ..database.models import PromptTemplate as DBPromptTemplate
            db_template = DBPromptTemplate(
                name=template.name,
                content=template.template,
                description=template.description,
                category=template.category,
                version=template.version,
                variables=json.dumps(template.variables),
                metadata=json.dumps(template.metadata),
                created_at=template.created_at,
                updated_at=template.updated_at
            )
            
            with db_manager.SessionLocal() as db:
                db.add(db_template)
                db.commit()
                
            self.logger.info(f"Saved improved template to database: {template.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to save improved template: {e}")
    
    async def analyze_prompt_performance(self, template_name: str) -> Dict[str, Any]:
        """プロンプトのパフォーマンスを分析"""
        
        try:
            if template_name not in self.templates:
                return {"error": f"Template '{template_name}' not found"}
            
            template = self.templates[template_name]
            
            # パフォーマンス分析
            analysis = {
                "template_name": template_name,
                "usage_count": template.usage_count,
                "success_rate": template.success_rate,
                "version": template.version,
                "last_updated": template.updated_at.isoformat(),
                "variables_count": len(template.variables),
                "template_length": len(template.template),
                "complexity_score": self._calculate_complexity_score(template),
                "improvement_suggestions": self._generate_improvement_suggestions(template)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Performance analysis failed: {e}")
            return {"error": str(e)}
    
    def _calculate_complexity_score(self, template: PromptTemplate) -> float:
        """テンプレートの複雑度スコアを計算"""
        
        try:
            score = 0.0
            
            # 長さによる複雑度
            length = len(template.template)
            if length > 1000:
                score += 0.3
            elif length > 500:
                score += 0.2
            elif length > 200:
                score += 0.1
            
            # 変数の数による複雑度
            var_count = len(template.variables)
            if var_count > 5:
                score += 0.3
            elif var_count > 3:
                score += 0.2
            elif var_count > 1:
                score += 0.1
            
            # 条件分岐の数による複雑度
            if_count = template.template.count('if')
            if if_count > 3:
                score += 0.2
            elif if_count > 1:
                score += 0.1
            
            return min(score, 1.0)
            
        except Exception:
            return 0.0
    
    def _generate_improvement_suggestions(self, template: PromptTemplate) -> List[str]:
        """改善提案を生成"""
        
        suggestions = []
        
        try:
            # 長さに関する提案
            if len(template.template) > 1000:
                suggestions.append("テンプレートが長すぎる可能性があります。簡潔にまとめることを検討してください。")
            elif len(template.template) < 50:
                suggestions.append("テンプレートが短すぎる可能性があります。より詳細な指示を追加することを検討してください。")
            
            # 変数に関する提案
            if len(template.variables) > 5:
                suggestions.append("変数が多すぎる可能性があります。テンプレートを分割することを検討してください。")
            elif len(template.variables) == 0:
                suggestions.append("変数が定義されていません。動的な内容を含めることを検討してください。")
            
            # 説明に関する提案
            if not template.description:
                suggestions.append("テンプレートの説明を追加することを推奨します。")
            
            # カテゴリに関する提案
            if template.category == "general":
                suggestions.append("より具体的なカテゴリを設定することを検討してください。")
            
            return suggestions
            
        except Exception:
            return ["分析中にエラーが発生しました。"]
