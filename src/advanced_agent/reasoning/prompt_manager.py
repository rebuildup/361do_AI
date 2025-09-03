"""
LangChain PromptTemplate プロンプト管理システム
動的プロンプト生成と最適化
"""

import json
import yaml
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from enum import Enum

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector, LengthBasedExampleSelector
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from ..core.config import get_config
from ..core.logger import get_logger


class PromptType(Enum):
    """プロンプトタイプ"""
    REASONING = "reasoning"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    FACTUAL = "factual"
    CONVERSATION = "conversation"
    INSTRUCTION = "instruction"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CODE_GENERATION = "code_generation"


@dataclass
class PromptExample:
    """プロンプト例"""
    input: str
    output: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptConfig:
    """プロンプト設定"""
    name: str
    type: PromptType
    template: str
    input_variables: List[str]
    examples: List[PromptExample] = field(default_factory=list)
    system_message: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    stop_sequences: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptOptimizationResult:
    """プロンプト最適化結果"""
    original_prompt: str
    optimized_prompt: str
    optimization_type: str
    improvement_score: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class PromptManager:
    """プロンプト管理システム"""
    
    def __init__(self, prompts_dir: str = "prompts"):
        self.config = get_config()
        self.logger = get_logger()
        self.prompts_dir = Path(prompts_dir)
        self.prompts_dir.mkdir(exist_ok=True)
        
        # プロンプト設定キャッシュ
        self.prompt_configs: Dict[str, PromptConfig] = {}
        self.prompt_templates: Dict[str, PromptTemplate] = {}
        
        # デフォルトプロンプト読み込み
        self._load_default_prompts()
        
        self.logger.log_startup(
            component="prompt_manager",
            version="1.0.0",
            config_summary={
                "prompts_dir": str(self.prompts_dir),
                "loaded_prompts": len(self.prompt_configs)
            }
        )
    
    def _load_default_prompts(self) -> None:
        """デフォルトプロンプト読み込み"""
        default_prompts = {
            "general_reasoning": PromptConfig(
                name="general_reasoning",
                type=PromptType.REASONING,
                template="""あなたは論理的思考に優れたAIアシスタントです。

{context}

質問: {question}

以下の手順で回答してください：
1. 問題の理解と整理
2. 関連する知識や情報の整理
3. 論理的な推論プロセス
4. 結論と根拠の提示

回答:""",
                input_variables=["context", "question"],
                system_message="あなたは論理的で正確な推論を行うAIアシスタントです。"
            ),
            
            "analytical_thinking": PromptConfig(
                name="analytical_thinking",
                type=PromptType.ANALYSIS,
                template="""あなたは分析的思考の専門家です。

分析対象: {subject}
コンテキスト: {context}
分析観点: {aspects}

以下の構造で分析を行ってください：

## 1. 概要分析
- 対象の基本的な特徴と性質

## 2. 詳細分析
- 各要素の詳細な検討
- 強み・弱み・機会・脅威の分析

## 3. 関係性分析
- 要素間の相互作用と依存関係

## 4. 結論と提言
- 分析結果のまとめ
- 具体的な提言や改善案

分析結果:""",
                input_variables=["subject", "context", "aspects"],
                system_message="あなたは体系的で客観的な分析を行う専門家です。"
            ),
            
            "creative_ideation": PromptConfig(
                name="creative_ideation",
                type=PromptType.CREATIVE,
                template="""あなたは創造性豊かなアイデア生成の専門家です。

課題: {challenge}
制約条件: {constraints}
目標: {objectives}

以下のアプローチで創造的なアイデアを生成してください：

## 1. 発散的思考
- 既存の枠にとらわれない自由な発想
- 複数の視点からのアプローチ

## 2. アイデアの組み合わせ
- 異なる分野からの知見の融合
- 意外な組み合わせの探索

## 3. 実現可能性の検討
- アイデアの実用性評価
- 実装に向けた具体的なステップ

創造的アイデア:""",
                input_variables=["challenge", "constraints", "objectives"],
                system_message="あなたは革新的で実用的なアイデアを生成する創造的思考の専門家です。"
            ),
            
            "factual_inquiry": PromptConfig(
                name="factual_inquiry",
                type=PromptType.FACTUAL,
                template="""あなたは事実に基づく正確な情報提供の専門家です。

質問: {question}
コンテキスト: {context}

以下の原則に従って回答してください：

## 回答原則
1. 事実に基づく客観的な情報のみを提供
2. 不確実な情報は明確に区別して表示
3. 可能な限り信頼できる根拠を示す
4. 推測や意見は事実と明確に区別

## 回答構造
- **確実な事実**: 確認済みの情報
- **可能性の高い情報**: 信頼できるが完全ではない情報
- **不確実な情報**: 推測や仮説
- **情報源**: 参考となる情報源（可能な場合）

事実に基づく回答:""",
                input_variables=["question", "context"],
                system_message="あなたは正確性と客観性を重視する情報提供の専門家です。"
            ),
            
            "code_generation": PromptConfig(
                name="code_generation",
                type=PromptType.CODE_GENERATION,
                template="""あなたは経験豊富なソフトウェア開発者です。

要件: {requirements}
プログラミング言語: {language}
制約条件: {constraints}

以下の構造でコードを生成してください：

## 1. 設計方針
- アプローチの説明
- 主要な設計決定

## 2. 実装コード
```{language}
# コードをここに記述
```

## 3. 使用例
```{language}
# 使用例をここに記述
```

## 4. 注意点
- パフォーマンスに関する考慮事項
- セキュリティ上の注意点
- 保守性の観点

コード生成結果:""",
                input_variables=["requirements", "language", "constraints"],
                system_message="あなたは高品質で保守性の高いコードを書く熟練開発者です。"
            )
        }
        
        # デフォルトプロンプトを登録
        for prompt_config in default_prompts.values():
            self.register_prompt(prompt_config)
    
    def register_prompt(self, prompt_config: PromptConfig) -> None:
        """プロンプト登録"""
        self.prompt_configs[prompt_config.name] = prompt_config
        
        # PromptTemplate作成
        if prompt_config.examples:
            # Few-shot プロンプトの場合
            example_prompt = PromptTemplate(
                input_variables=["input", "output"],
                template="入力: {input}\n出力: {output}"
            )
            
            self.prompt_templates[prompt_config.name] = FewShotPromptTemplate(
                examples=[
                    {"input": ex.input, "output": ex.output} 
                    for ex in prompt_config.examples
                ],
                example_prompt=example_prompt,
                prefix=prompt_config.template.split("{examples}")[0] if "{examples}" in prompt_config.template else "",
                suffix=prompt_config.template.split("{examples}")[1] if "{examples}" in prompt_config.template else prompt_config.template,
                input_variables=prompt_config.input_variables
            )
        else:
            # 通常のプロンプト
            self.prompt_templates[prompt_config.name] = PromptTemplate(
                input_variables=prompt_config.input_variables,
                template=prompt_config.template
            )
        
        self.logger.log_performance_metric(
            metric_name="prompt_registered",
            value=1,
            unit="count",
            component="prompt_manager"
        )
    
    def get_prompt_template(self, name: str) -> Optional[PromptTemplate]:
        """プロンプトテンプレート取得"""
        return self.prompt_templates.get(name)
    
    def get_prompt_config(self, name: str) -> Optional[PromptConfig]:
        """プロンプト設定取得"""
        return self.prompt_configs.get(name)
    
    def list_prompts(self, prompt_type: Optional[PromptType] = None) -> List[str]:
        """プロンプト一覧取得"""
        if prompt_type is None:
            return list(self.prompt_configs.keys())
        
        return [
            name for name, config in self.prompt_configs.items()
            if config.type == prompt_type
        ]
    
    def create_chat_prompt(self, name: str, **kwargs) -> ChatPromptTemplate:
        """チャット形式プロンプト作成"""
        config = self.prompt_configs.get(name)
        if not config:
            raise ValueError(f"Prompt config not found: {name}")
        
        messages = []
        
        # システムメッセージ
        if config.system_message:
            messages.append(SystemMessage(content=config.system_message))
        
        # メインプロンプト
        template = self.prompt_templates[name]
        formatted_prompt = template.format(**kwargs)
        messages.append(HumanMessage(content=formatted_prompt))
        
        return ChatPromptTemplate.from_messages(messages)
    
    def optimize_prompt(self, prompt: str, optimization_type: str = "clarity") -> PromptOptimizationResult:
        """プロンプト最適化"""
        # 簡易的な最適化実装
        # 実際の実装では、より高度な最適化手法を使用
        
        optimized_prompt = prompt
        improvement_score = 0.0
        reasoning = ""
        
        if optimization_type == "clarity":
            # 明確性の向上
            if "あいまい" in prompt or "不明確" in prompt:
                optimized_prompt = prompt.replace("あいまい", "具体的")
                optimized_prompt = optimized_prompt.replace("不明確", "明確")
                improvement_score = 0.2
                reasoning = "あいまいな表現を具体的な表現に変更しました。"
            
        elif optimization_type == "specificity":
            # 具体性の向上
            if "何か" in prompt or "いくつか" in prompt:
                optimized_prompt = prompt.replace("何か", "具体的な例")
                optimized_prompt = optimized_prompt.replace("いくつか", "3つの")
                improvement_score = 0.3
                reasoning = "抽象的な表現をより具体的な表現に変更しました。"
        
        elif optimization_type == "structure":
            # 構造化の改善
            if "。" in prompt and "##" not in prompt:
                sentences = prompt.split("。")
                structured_prompt = "## 要求事項\n"
                for i, sentence in enumerate(sentences[:-1], 1):
                    if sentence.strip():
                        structured_prompt += f"{i}. {sentence.strip()}\n"
                optimized_prompt = structured_prompt
                improvement_score = 0.4
                reasoning = "プロンプトを構造化して読みやすくしました。"
        
        return PromptOptimizationResult(
            original_prompt=prompt,
            optimized_prompt=optimized_prompt,
            optimization_type=optimization_type,
            improvement_score=improvement_score,
            reasoning=reasoning
        )
    
    def save_prompt_config(self, config: PromptConfig, filename: Optional[str] = None) -> Path:
        """プロンプト設定保存"""
        if filename is None:
            filename = f"{config.name}.yaml"
        
        filepath = self.prompts_dir / filename
        
        # 設定を辞書に変換
        config_dict = {
            "name": config.name,
            "type": config.type.value,
            "template": config.template,
            "input_variables": config.input_variables,
            "examples": [
                {
                    "input": ex.input,
                    "output": ex.output,
                    "metadata": ex.metadata
                }
                for ex in config.examples
            ],
            "system_message": config.system_message,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "stop_sequences": config.stop_sequences,
            "metadata": config.metadata
        }
        
        # YAML形式で保存
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        
        self.logger.log_performance_metric(
            metric_name="prompt_config_saved",
            value=1,
            unit="count",
            component="prompt_manager"
        )
        
        return filepath
    
    def load_prompt_config(self, filepath: Union[str, Path]) -> PromptConfig:
        """プロンプト設定読み込み"""
        filepath = Path(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # 設定オブジェクト作成
        examples = [
            PromptExample(
                input=ex["input"],
                output=ex["output"],
                metadata=ex.get("metadata", {})
            )
            for ex in config_dict.get("examples", [])
        ]
        
        config = PromptConfig(
            name=config_dict["name"],
            type=PromptType(config_dict["type"]),
            template=config_dict["template"],
            input_variables=config_dict["input_variables"],
            examples=examples,
            system_message=config_dict.get("system_message"),
            temperature=config_dict.get("temperature", 0.1),
            max_tokens=config_dict.get("max_tokens"),
            stop_sequences=config_dict.get("stop_sequences", []),
            metadata=config_dict.get("metadata", {})
        )
        
        # 登録
        self.register_prompt(config)
        
        return config
    
    def create_few_shot_prompt(self, 
                             name: str,
                             base_template: str,
                             examples: List[PromptExample],
                             input_variables: List[str]) -> PromptConfig:
        """Few-shot プロンプト作成"""
        config = PromptConfig(
            name=name,
            type=PromptType.REASONING,
            template=base_template,
            input_variables=input_variables,
            examples=examples
        )
        
        self.register_prompt(config)
        return config
    
    def get_prompt_statistics(self) -> Dict[str, Any]:
        """プロンプト統計情報"""
        type_counts = {}
        for config in self.prompt_configs.values():
            type_name = config.type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        total_examples = sum(len(config.examples) for config in self.prompt_configs.values())
        
        return {
            "total_prompts": len(self.prompt_configs),
            "prompts_by_type": type_counts,
            "total_examples": total_examples,
            "average_examples_per_prompt": total_examples / len(self.prompt_configs) if self.prompt_configs else 0
        }


# グローバルプロンプトマネージャー
_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """グローバルプロンプトマネージャー取得"""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


# 便利関数
def get_prompt(name: str) -> Optional[PromptTemplate]:
    """プロンプト取得"""
    return get_prompt_manager().get_prompt_template(name)


def create_optimized_prompt(prompt: str, optimization_type: str = "clarity") -> str:
    """最適化プロンプト作成"""
    manager = get_prompt_manager()
    result = manager.optimize_prompt(prompt, optimization_type)
    return result.optimized_prompt


# 使用例
async def main():
    """テスト用メイン関数"""
    manager = PromptManager()
    
    print("Prompt Manager Test")
    print("=" * 40)
    
    # 利用可能プロンプト一覧
    prompts = manager.list_prompts()
    print(f"Available prompts: {prompts}")
    
    # プロンプトテンプレート取得テスト
    reasoning_template = manager.get_prompt_template("general_reasoning")
    if reasoning_template:
        formatted_prompt = reasoning_template.format(
            context="AI技術の発展について",
            question="機械学習の未来はどうなりますか？"
        )
        print(f"\nFormatted prompt:\n{formatted_prompt}")
    
    # チャットプロンプト作成テスト
    chat_prompt = manager.create_chat_prompt(
        "analytical_thinking",
        subject="Python プログラミング",
        context="初心者向けの学習方法",
        aspects="効率性、学習曲線、実用性"
    )
    print(f"\nChat prompt created: {type(chat_prompt)}")
    
    # プロンプト最適化テスト
    test_prompt = "何かいい方法を教えてください。"
    optimization_result = manager.optimize_prompt(test_prompt, "specificity")
    print(f"\nOptimization result:")
    print(f"Original: {optimization_result.original_prompt}")
    print(f"Optimized: {optimization_result.optimized_prompt}")
    print(f"Reasoning: {optimization_result.reasoning}")
    
    # 統計情報
    stats = manager.get_prompt_statistics()
    print(f"\nPrompt statistics: {stats}")
    
    # Few-shot プロンプト作成テスト
    examples = [
        PromptExample(
            input="2 + 2 = ?",
            output="2 + 2 = 4"
        ),
        PromptExample(
            input="5 * 3 = ?",
            output="5 * 3 = 15"
        )
    ]
    
    few_shot_config = manager.create_few_shot_prompt(
        name="math_solver",
        base_template="数学の問題を解いてください。\n\n問題: {problem}\n\n解答:",
        examples=examples,
        input_variables=["problem"]
    )
    
    print(f"\nFew-shot prompt created: {few_shot_config.name}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())