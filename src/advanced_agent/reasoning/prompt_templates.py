"""
Prompt templates for self-learning AI agent reasoning
自己学習AIエージェント推論用プロンプトテンプレート
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class PromptType(Enum):
    """プロンプトタイプ"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    REASONING = "reasoning"
    EVALUATION = "evaluation"
    LEARNING = "learning"


@dataclass
class PromptTemplate:
    """プロンプトテンプレート"""
    name: str
    template: str
    prompt_type: PromptType
    variables: List[str]
    description: str
    version: str = "1.0.0"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PromptTemplateManager:
    """プロンプトテンプレート管理クラス"""
    
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """デフォルトテンプレート初期化"""
        
        # 1. システムプロンプト
        system_prompt = PromptTemplate(
            name="system_base",
            template="""あなたは361do_AIです。以下の特徴を持っています：

1. **段階的思考**: 複雑な問題を小さな部分に分解して解決します
2. **論理的推論**: 明確な根拠に基づいて結論を導きます
3. **継続的学習**: 各インタラクションから学習し、改善します
4. **透明性**: 思考プロセスを明確に示します

あなたの目標は、ユーザーの質問に対して正確で有用な回答を提供し、同時に自身の能力を向上させることです。

現在のセッション情報:
- セッションID: {session_id}
- 学習エポック: {learning_epoch}
- 総インタラクション数: {total_interactions}
- 現在の報酬スコア: {reward_score}

推論スタイル: Chain-of-Thought（段階的思考）を使用し、各ステップで明確に思考過程を示してください。""",
            prompt_type=PromptType.SYSTEM,
            variables=["session_id", "learning_epoch", "total_interactions", "reward_score"],
            description="基本システムプロンプト",
            metadata={
                "category": "system",
                "priority": "high",
                "usage_count": 0
            }
        )
        
        # 2. 推論プロンプト
        reasoning_prompt = PromptTemplate(
            name="reasoning_cot",
            template="""以下の問題を段階的に解決してください。

問題: {question}

推論形式:
1. **理解**: 問題を理解し、何が求められているかを明確にします
2. **分析**: 問題を分析し、必要な情報や手順を特定します
3. **計画**: 解決手順を計画します
4. **実行**: 計画に従って段階的に解決します
5. **検証**: 答えが正しいか検証します
6. **結論**: 最終的な答えと根拠を提示します

各ステップで明確に思考過程を示し、必要に応じて計算や論理的な推論を含めてください。

回答形式:
```
理解: [問題の理解]
分析: [問題の分析]
計画: [解決計画]
実行: [段階的解決]
検証: [答えの検証]
結論: [最終回答]
```""",
            prompt_type=PromptType.REASONING,
            variables=["question"],
            description="Chain-of-Thought推論プロンプト",
            metadata={
                "category": "reasoning",
                "priority": "high",
                "usage_count": 0
            }
        )
        
        # 3. 数学問題用プロンプト
        math_prompt = PromptTemplate(
            name="math_reasoning",
            template="""以下の数学問題を段階的に解決してください。

問題: {math_problem}

解決手順:
1. **問題の理解**: 何を求められているかを明確にします
2. **既知情報の整理**: 与えられた情報を整理します
3. **解法の選択**: 適切な解法を選択します
4. **計算の実行**: 段階的に計算を実行します
5. **答えの検証**: 答えが妥当かチェックします

計算過程は必ず示し、各ステップで何をしているかを説明してください。

回答形式:
```
理解: [問題の理解]
既知情報: [与えられた情報]
解法: [選択した解法]
計算: [段階的計算]
検証: [答えの検証]
答え: [最終的な答え]
```""",
            prompt_type=PromptType.REASONING,
            variables=["math_problem"],
            description="数学問題推論プロンプト",
            metadata={
                "category": "math",
                "priority": "medium",
                "usage_count": 0
            }
        )
        
        # 4. 論理問題用プロンプト
        logic_prompt = PromptTemplate(
            name="logic_reasoning",
            template="""以下の論理問題を段階的に解決してください。

問題: {logic_problem}

解決手順:
1. **前提の整理**: 与えられた前提条件を整理します
2. **論理関係の分析**: 前提間の論理関係を分析します
3. **推論の実行**: 論理的な推論を段階的に実行します
4. **結論の導出**: 論理的に正しい結論を導きます
5. **検証**: 結論が前提と矛盾しないか検証します

各推論ステップで根拠を明確に示してください。

回答形式:
```
前提: [与えられた前提]
分析: [論理関係の分析]
推論: [段階的推論]
結論: [導出された結論]
検証: [結論の検証]
```""",
            prompt_type=PromptType.REASONING,
            variables=["logic_problem"],
            description="論理問題推論プロンプト",
            metadata={
                "category": "logic",
                "priority": "medium",
                "usage_count": 0
            }
        )
        
        # 5. 評価プロンプト
        evaluation_prompt = PromptTemplate(
            name="response_evaluation",
            template="""以下の回答を評価してください。

質問: {question}
回答: {response}

評価基準:
1. **正確性** (0-10): 事実的に正確か
2. **完全性** (0-10): 質問に完全に答えているか
3. **明確性** (0-10): 分かりやすく説明されているか
4. **論理性** (0-10): 論理的に一貫しているか
5. **有用性** (0-10): ユーザーにとって有用か

各基準について0-10で評価し、総合スコアを計算してください。

評価形式:
```
正確性: [スコア]/10 - [理由]
完全性: [スコア]/10 - [理由]
明確性: [スコア]/10 - [理由]
論理性: [スコア]/10 - [理由]
有用性: [スコア]/10 - [理由]

総合スコア: [総合スコア]/50
改善提案: [改善点があれば]
```""",
            prompt_type=PromptType.EVALUATION,
            variables=["question", "response"],
            description="回答評価プロンプト",
            metadata={
                "category": "evaluation",
                "priority": "high",
                "usage_count": 0
            }
        )
        
        # 6. 学習プロンプト
        learning_prompt = PromptTemplate(
            name="learning_reflection",
            template="""以下のインタラクションから学習してください。

質問: {question}
回答: {response}
評価: {evaluation}

学習観点:
1. **成功要因**: 何が良かったか
2. **改善点**: 何を改善すべきか
3. **知識の更新**: 新しい知識や洞察はあるか
4. **推論の改善**: 推論プロセスをどう改善できるか
5. **今後の方針**: 今後どのように改善するか

学習結果を構造化して出力してください。

学習形式:
```
成功要因:
- [成功要因1]
- [成功要因2]

改善点:
- [改善点1]
- [改善点2]

知識の更新:
- [新しい知識1]
- [新しい知識2]

推論の改善:
- [推論改善点1]
- [推論改善点2]

今後の方針:
- [方針1]
- [方針2]
```""",
            prompt_type=PromptType.LEARNING,
            variables=["question", "response", "evaluation"],
            description="学習振り返りプロンプト",
            metadata={
                "category": "learning",
                "priority": "medium",
                "usage_count": 0
            }
        )
        
        # 7. プロンプト最適化用
        optimization_prompt = PromptTemplate(
            name="prompt_optimization",
            template="""以下のプロンプトを最適化してください。

現在のプロンプト: {current_prompt}
使用統計: {usage_stats}
評価結果: {evaluation_results}

最適化方針:
1. **明確性の向上**: より明確で理解しやすい表現に
2. **効率性の向上**: より効率的な推論を促す
3. **品質の向上**: より高品質な回答を生成
4. **一貫性の向上**: より一貫した回答を生成

最適化されたプロンプトを提案してください。

最適化形式:
```
現在の問題点:
- [問題点1]
- [問題点2]

最適化案:
[最適化されたプロンプト]

改善点:
- [改善点1]
- [改善点2]

期待される効果:
- [効果1]
- [効果2]
```""",
            prompt_type=PromptType.LEARNING,
            variables=["current_prompt", "usage_stats", "evaluation_results"],
            description="プロンプト最適化プロンプト",
            metadata={
                "category": "optimization",
                "priority": "low",
                "usage_count": 0
            }
        )
        
        # テンプレート登録
        templates = [
            system_prompt,
            reasoning_prompt,
            math_prompt,
            logic_prompt,
            evaluation_prompt,
            learning_prompt,
            optimization_prompt
        ]
        
        for template in templates:
            self.templates[template.name] = template
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """テンプレート取得"""
        return self.templates.get(name)
    
    def get_templates_by_type(self, prompt_type: PromptType) -> List[PromptTemplate]:
        """タイプ別テンプレート取得"""
        return [template for template in self.templates.values() 
                if template.prompt_type == prompt_type]
    
    def get_templates_by_category(self, category: str) -> List[PromptTemplate]:
        """カテゴリ別テンプレート取得"""
        return [template for template in self.templates.values() 
                if template.metadata.get("category") == category]
    
    def format_template(self, name: str, **kwargs) -> str:
        """テンプレートフォーマット"""
        template = self.get_template(name)
        if not template:
            raise ValueError(f"Template not found: {name}")
        
        try:
            # 使用回数更新
            template.metadata["usage_count"] = template.metadata.get("usage_count", 0) + 1
            
            return template.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing variable {e} for template {name}")
    
    def add_template(self, template: PromptTemplate):
        """テンプレート追加"""
        self.templates[template.name] = template
    
    def remove_template(self, name: str) -> bool:
        """テンプレート削除"""
        if name in self.templates:
            del self.templates[name]
            return True
        return False
    
    def update_template(self, name: str, **updates):
        """テンプレート更新"""
        template = self.get_template(name)
        if not template:
            raise ValueError(f"Template not found: {name}")
        
        for key, value in updates.items():
            if hasattr(template, key):
                setattr(template, key, value)
            else:
                raise ValueError(f"Invalid template attribute: {key}")
    
    def get_template_stats(self) -> Dict[str, Any]:
        """テンプレート統計取得"""
        stats = {
            "total_templates": len(self.templates),
            "by_type": {},
            "by_category": {},
            "most_used": [],
            "least_used": []
        }
        
        # タイプ別統計
        for prompt_type in PromptType:
            count = len(self.get_templates_by_type(prompt_type))
            stats["by_type"][prompt_type.value] = count
        
        # カテゴリ別統計
        categories = set()
        for template in self.templates.values():
            category = template.metadata.get("category", "unknown")
            categories.add(category)
        
        for category in categories:
            count = len(self.get_templates_by_category(category))
            stats["by_category"][category] = count
        
        # 使用頻度統計
        usage_counts = [(name, template.metadata.get("usage_count", 0)) 
                       for name, template in self.templates.items()]
        usage_counts.sort(key=lambda x: x[1], reverse=True)
        
        stats["most_used"] = usage_counts[:5]
        stats["least_used"] = usage_counts[-5:]
        
        return stats
    
    def export_templates(self) -> Dict[str, Any]:
        """テンプレートエクスポート"""
        return {
            name: {
                "name": template.name,
                "template": template.template,
                "prompt_type": template.prompt_type.value,
                "variables": template.variables,
                "description": template.description,
                "version": template.version,
                "metadata": template.metadata
            }
            for name, template in self.templates.items()
        }
    
    def import_templates(self, templates_data: Dict[str, Any]):
        """テンプレートインポート"""
        for name, data in templates_data.items():
            template = PromptTemplate(
                name=data["name"],
                template=data["template"],
                prompt_type=PromptType(data["prompt_type"]),
                variables=data["variables"],
                description=data["description"],
                version=data.get("version", "1.0.0"),
                metadata=data.get("metadata", {})
            )
            self.templates[name] = template


# グローバルテンプレートマネージャー
_template_manager: Optional[PromptTemplateManager] = None


def get_template_manager() -> PromptTemplateManager:
    """テンプレートマネージャー取得（シングルトン）"""
    global _template_manager
    
    if _template_manager is None:
        _template_manager = PromptTemplateManager()
    
    return _template_manager


def format_prompt(template_name: str, **kwargs) -> str:
    """プロンプトフォーマット（便利関数）"""
    manager = get_template_manager()
    return manager.format_template(template_name, **kwargs)


def get_system_prompt(session_id: str, learning_epoch: int, 
                     total_interactions: int, reward_score: float) -> str:
    """システムプロンプト取得（便利関数）"""
    return format_prompt(
        "system_base",
        session_id=session_id,
        learning_epoch=learning_epoch,
        total_interactions=total_interactions,
        reward_score=reward_score
    )


def get_reasoning_prompt(question: str) -> str:
    """推論プロンプト取得（便利関数）"""
    return format_prompt("reasoning_cot", question=question)


def get_math_prompt(math_problem: str) -> str:
    """数学問題プロンプト取得（便利関数）"""
    return format_prompt("math_reasoning", math_problem=math_problem)


def get_logic_prompt(logic_problem: str) -> str:
    """論理問題プロンプト取得（便利関数）"""
    return format_prompt("logic_reasoning", logic_problem=logic_problem)


def get_evaluation_prompt(question: str, response: str) -> str:
    """評価プロンプト取得（便利関数）"""
    return format_prompt("response_evaluation", question=question, response=response)


def get_learning_prompt(question: str, response: str, evaluation: str) -> str:
    """学習プロンプト取得（便利関数）"""
    return format_prompt("learning_reflection", question=question, response=response, evaluation=evaluation)


# 使用例
if __name__ == "__main__":
    # テンプレートマネージャー取得
    manager = get_template_manager()
    
    # 統計表示
    stats = manager.get_template_stats()
    print("Template Statistics:")
    print(f"Total templates: {stats['total_templates']}")
    print(f"By type: {stats['by_type']}")
    print(f"By category: {stats['by_category']}")
    print(f"Most used: {stats['most_used']}")
    
    # プロンプト生成例
    print("\n=== Prompt Examples ===")
    
    # システムプロンプト
    system_prompt = get_system_prompt(
        session_id="test_session_123",
        learning_epoch=5,
        total_interactions=100,
        reward_score=0.85
    )
    print("System Prompt:")
    print(system_prompt[:200] + "...")
    
    # 推論プロンプト
    reasoning_prompt = get_reasoning_prompt("2+2は何ですか？")
    print("\nReasoning Prompt:")
    print(reasoning_prompt[:200] + "...")
    
    # 数学問題プロンプト
    math_prompt = get_math_prompt("x^2 + 5x + 6 = 0 を解いてください")
    print("\nMath Prompt:")
    print(math_prompt[:200] + "...")
