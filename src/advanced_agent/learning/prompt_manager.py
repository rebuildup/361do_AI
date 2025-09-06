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
from ..database.models import PromptTemplate, PromptVersion, PromptUsage
from ..database.connection import get_db_session


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
        self.ollama_client = ollama_client or OllamaClient()
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
