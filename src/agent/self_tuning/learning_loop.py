"""
Self-Learning Loop
評価と最適化を組み合わせた自己学習サイクル
"""

import os
import shutil
import asyncio
from datetime import datetime
from logging import getLogger

from src.agent.core.config import Config
from src.agent.core.ollama_client import OllamaClient
from src.agent.self_tuning.evaluation_module import EvaluationModule
from src.agent.self_tuning.optimization_module import OptimizationModule

logger = getLogger(__name__)

class LearningLoop:
    """自己学習サイクルを管理するクラス"""

    def __init__(self, config: Config, ollama_client: OllamaClient):
        self.config = config
        self.ollama_client = ollama_client
        self.evaluator = EvaluationModule(config)
        self.optimizer = OptimizationModule(config, ollama_client)

    async def run_learning_cycle(self):
        """自己学習サイクルを1回実行する"""
        logger.info("自己学習サイクルを開始します...")

        # 1. パフォーマンス評価
        score = self.evaluator.evaluate_performance()

        # 2. スコアがしきい値を下回るかチェック
        if score < self.config.learning.min_quality_score_for_learning:
            logger.info("パフォーマンスが低いため、プロンプトの最適化を実行します。")

            # 3. プロンプト最適化
            new_prompt = await self.optimizer.optimize_prompt(score)

            if not new_prompt or not new_prompt.strip():
                logger.error("生成されたプロンプトが無効なため、更新をスキップします。")
                return

            # 4. プロンプトを更新
            self._update_prompt_file(new_prompt)
        else:
            logger.info("パフォーマンスは基準を満たしています。プロンプトの更新は不要です。")

        logger.info("自己学習サイクルが完了しました。")

    def _update_prompt_file(self, new_prompt: str):
        """プロンプトファイルを新しい内容で更新する"""
        prompt_file = self.config.paths.custom_prompt_file
        
        # バックアップを作成
        backup_file = f"{prompt_file}.{datetime.now().strftime('%Y%m%d%H%M%S')}.bak"
        try:
            shutil.copy(prompt_file, backup_file)
            logger.info(f"現在のプロンプトをバックアップしました: {backup_file}")
        except FileNotFoundError:
            logger.warning(f"バックアップ対象のプロンプトファイルが見つかりません: {prompt_file}")
        except Exception as e:
            logger.error(f"プロンプトのバックアップ中にエラー: {e}")
            return

        # 新しいプロンプトを書き込む
        try:
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(new_prompt)
            logger.info(f"プロンプトファイルを更新しました: {prompt_file}")
        except Exception as e:
            logger.error(f"プロンプトファイルの書き込み中にエラー: {e}")


async def main():
    # テスト用
    config = Config()
    ollama_client = OllamaClient(config.ollama_config)
    await ollama_client.initialize()

    learning_loop = LearningLoop(config, ollama_client)
    await learning_loop.run_learning_cycle()

    await ollama_client.close()

if __name__ == '__main__':
    # ロガーの基本設定（テスト用）
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    asyncio.run(main())
