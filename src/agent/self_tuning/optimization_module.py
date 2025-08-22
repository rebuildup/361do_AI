"""
Optimization Module
プロンプトや学習データの最適化
"""

import os
import json
import asyncio
from logging import getLogger

from src.agent.core.config import Config
from src.agent.core.ollama_client import OllamaClient

logger = getLogger(__name__)

class OptimizationModule:
    """プロンプトや学習データを最適化するクラス"""

    def __init__(self, config: Config, ollama_client: OllamaClient):
        self.config = config
        self.ollama_client = ollama_client

    async def optimize_prompt(self, current_score: float) -> str:
        """
        現在のプロンプトを評価スコアと学習データに基づいて最適化する。

        Args:
            current_score (float): 現在の評価スコア。

        Returns:
            str: 最適化された新しいプロンプト。
        """
        logger.info("プロンプトの最適化を開始します...")

        # 1. 現在のプロンプトを読み込む
        try:
            with open(self.config.paths.custom_prompt_file, 'r', encoding='utf-8') as f:
                current_prompt = f.read()
        except FileNotFoundError:
            logger.error(f"プロンプトファイルが見つかりません: {self.config.paths.custom_prompt_file}")
            return ""

        # 2. 学習データを読み込む
        learning_examples = []
        try:
            with open(self.config.paths.learning_data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    learning_examples.append(json.loads(line))
        except FileNotFoundError:
            logger.warning(f"学習データファイルが見つかりません: {self.config.paths.learning_data_file}")
        except json.JSONDecodeError:
            logger.error(f"学習データの解析に失敗しました: {self.config.paths.learning_data_file}")

        # 3. メタプロンプトを構築
        meta_prompt = self._build_meta_prompt(current_prompt, current_score, learning_examples)

        # 4. LLMに新しいプロンプトの生成を依頼
        messages = [
            {"role": "system", "content": "あなたは、AIエージェントの性能を最大化するためのプロンプトエンジニアリングの専門家です。"},
            {"role": "user", "content": meta_prompt}
        ]

        try:
            logger.info("LLMに新しいプロンプトの生成を依頼します...")
            new_prompt = await self.ollama_client.chat(messages=messages)
            logger.info("新しいプロンプトの生成に成功しました。")
            return new_prompt
        except Exception as e:
            logger.error(f"プロンプトの最適化中にエラーが発生しました: {e}")
            return ""

    def _build_meta_prompt(self, current_prompt: str, current_score: float, examples: list) -> str:
        """
        LLMに渡すためのメタプロンプトを構築する。
        """
        prompt = f"""
# 指示
あなたはAIエージェントのシステムプロンプトを改善する専門家です。
現在のシステムプロンプトと、そのパフォーマンス評価、そして高品質な応答のサンプルを提供します。
これらに基づいて、エージェントの性能をさらに向上させるための、新しいシステムプロンプトを提案してください。

# 現在のシステムプロンプト
```
{current_prompt}
```

# パフォーマンス評価
現在の評価スコアは {current_score:.2f} (満点は1.0) です。
目標スコア ({self.config.learning.min_quality_score_for_learning}) を下回っており、改善が必要です。

# 高品質な応答のサンプル
以下は、ユーザーの指示と、それに対する理想的な応答の例です。
"""

        for example in examples[:5]: # サンプルが多すぎないように制限
            prompt += f"\n- 指示: {example.get('instruction')}\n- 理想的な応答: {example.get('output')}\n"
        
        prompt += """
# あなたのタスク
上記すべてを考慮し、改善された新しいシステムプロンプトを生成してください。
新しいプロンプトは、具体的で、行動を促し、曖昧さを排除したものであるべきです。

**重要:** 応答は、新しいシステムプロンプトのテキストのみを含めてください。解説や前置きは一切不要です。
"""
        return prompt


async def main():
    # テスト用
    config = Config()
    ollama_client = OllamaClient(config.ollama_config)
    await ollama_client.initialize()

    optimizer = OptimizationModule(config, ollama_client)
    
    # 評価スコアが低いと仮定
    test_score = 0.6
    
    new_prompt = await optimizer.optimize_prompt(test_score)
    
    if new_prompt:
        print("--- 生成された新しいプロンプト ---")
        print(new_prompt)
        print("---------------------------------")
    else:
        print("プロンプトの最適化に失敗しました。")

    await ollama_client.close()

if __name__ == '__main__':
    asyncio.run(main())
