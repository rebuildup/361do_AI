"""
HuggingFace Evaluate アダプタ評価システム統合
HuggingFace Evaluate による タスク別性能指標計算とA/Bテスト比較システム
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import numpy as np

try:
    import torch
    from datasets import Dataset as HFDataset, load_dataset
    import evaluate
    from transformers import AutoTokenizer, pipeline
    from peft import PeftModel
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False
    evaluate = None

from ..core.config import get_config
from ..core.logger import get_logger
from ..monitoring.system_monitor import SystemMonitor
from .peft_manager import PEFTAdapterPool, AdapterInfo


class EvaluationTask(Enum):
    """評価タスク"""
    PERPLEXITY = "perplexity"
    BLEU = "bleu"
    ROUGE = "rouge"
    BERTSCORE = "bertscore"
    ACCURACY = "accuracy"
    F1 = "f1"
    EXACT_MATCH = "exact_match"
    CUSTOM = "custom"


class EvaluationMetric(Enum):
    """評価指標"""
    SCORE = "score"
    LOSS = "loss"
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1"
    BLEU_SCORE = "bleu"
    ROUGE_L = "rouge_l"
    BERTSCORE_F1 = "bertscore_f1"


@dataclass
class EvaluationConfig:
    """評価設定"""
    task: EvaluationTask
    metrics: List[EvaluationMetric]
    
    # データセット設定
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    dataset_split: str = "test"
    max_samples: Optional[int] = None
    
    # 評価設定
    batch_size: int = 8
    max_length: int = 512
    
    # 生成設定（生成タスク用）
    max_new_tokens: int = 50
    temperature: float = 0.7
    do_sample: bool = True
    
    # 比較設定
    baseline_adapter: Optional[str] = None
    comparison_adapters: List[str] = field(default_factory=list)
    
    # その他
    seed: int = 42
    device: str = "auto"
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """評価結果"""
    adapter_name: str
    task: EvaluationTask
    config: EvaluationConfig
    
    # メトリクス結果
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # 詳細結果
    sample_results: List[Dict[str, Any]] = field(default_factory=list)
    predictions: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    
    # 統計情報
    total_samples: int = 0
    evaluation_time: float = 0.0
    samples_per_second: float = 0.0
    
    # メモリ使用量
    peak_memory_mb: float = 0.0
    
    # エラー情報
    success: bool = True
    error_message: Optional[str] = None
    
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """比較結果"""
    baseline_adapter: str
    comparison_adapter: str
    task: EvaluationTask
    
    # 性能比較
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    comparison_metrics: Dict[str, float] = field(default_factory=dict)
    improvement: Dict[str, float] = field(default_factory=dict)
    
    # 統計的有意性
    statistical_significance: Dict[str, bool] = field(default_factory=dict)
    p_values: Dict[str, float] = field(default_factory=dict)
    
    # 総合評価
    overall_improvement: float = 0.0
    recommendation: str = ""
    
    timestamp: datetime = field(default_factory=datetime.now)


class AdapterEvaluator:
    """HuggingFace Evaluate アダプタ評価システム"""
    
    def __init__(
        self,
        peft_pool: PEFTAdapterPool,
        system_monitor: Optional[SystemMonitor] = None
    ):
        self.peft_pool = peft_pool
        self.system_monitor = system_monitor
        
        self.config = get_config()
        self.logger = get_logger()
        
        # 評価器キャッシュ
        self.evaluators: Dict[str, Any] = {}
        
        # 評価履歴
        self.evaluation_history: List[EvaluationResult] = []
        self.comparison_history: List[ComparisonResult] = []
        
        # 統計
        self.evaluation_stats = {
            "total_evaluations": 0,
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "total_comparisons": 0
        }
        
        self.logger.log_startup(
            component="adapter_evaluator",
            version="1.0.0",
            config_summary={
                "evaluate_available": EVALUATE_AVAILABLE,
                "peft_pool_adapters": len(self.peft_pool.adapters) if self.peft_pool else 0
            }
        )
    
    async def initialize(self) -> bool:
        """評価システム初期化"""
        try:
            if not EVALUATE_AVAILABLE:
                self.logger.log_alert(
                    alert_type="evaluate_unavailable",
                    severity="WARNING",
                    message="HuggingFace Evaluate not available"
                )
                return False
            
            # 基本評価器の事前ロード
            await self._preload_evaluators()
            
            self.logger.log_startup(
                component="adapter_evaluator_initialized",
                version="1.0.0",
                config_summary={
                    "preloaded_evaluators": len(self.evaluators)
                }
            )
            
            return True
            
        except Exception as e:
            self.logger.log_alert(
                alert_type="evaluator_initialization_failed",
                severity="ERROR",
                message=f"Adapter evaluator initialization failed: {e}"
            )
            return False
    
    async def _preload_evaluators(self) -> None:
        """評価器事前ロード"""
        
        basic_metrics = ["accuracy", "f1", "bleu", "rouge"]
        
        for metric_name in basic_metrics:
            try:
                evaluator = await asyncio.to_thread(evaluate.load, metric_name)
                self.evaluators[metric_name] = evaluator
                
            except Exception as e:
                self.logger.log_alert(
                    alert_type="evaluator_preload_failed",
                    severity="WARNING",
                    message=f"Failed to preload evaluator '{metric_name}': {e}"
                )
    
    def get_evaluator(self, metric_name: str):
        """評価器取得"""
        
        if metric_name in self.evaluators:
            return self.evaluators[metric_name]
        
        try:
            evaluator = evaluate.load(metric_name)
            self.evaluators[metric_name] = evaluator
            return evaluator
            
        except Exception as e:
            self.logger.log_alert(
                alert_type="evaluator_load_failed",
                severity="ERROR",
                message=f"Failed to load evaluator '{metric_name}': {e}"
            )
            return None
    
    async def evaluate_adapter(
        self,
        adapter_name: str,
        eval_config: EvaluationConfig,
        dataset: Optional[HFDataset] = None
    ) -> EvaluationResult:
        """アダプタ評価実行"""
        
        start_time = time.time()
        
        try:
            # アダプタ情報取得
            adapter_info = self.peft_pool.get_adapter_info(adapter_name)
            if not adapter_info:
                raise ValueError(f"Adapter '{adapter_name}' not found")
            
            # アダプタロード
            await self.peft_pool.load_adapter(adapter_name)
            model = self.peft_pool.active_adapters[adapter_name]
            
            # データセット準備
            if dataset is None:
                dataset = await self._prepare_dataset(eval_config)
            
            # 評価実行
            predictions, references, sample_results = await self._run_evaluation(
                model, dataset, eval_config
            )
            
            # メトリクス計算
            metrics = await self._calculate_metrics(
                predictions, references, eval_config.metrics
            )
            
            # 結果作成
            evaluation_time = time.time() - start_time
            
            result = EvaluationResult(
                adapter_name=adapter_name,
                task=eval_config.task,
                config=eval_config,
                metrics=metrics,
                sample_results=sample_results,
                predictions=predictions,
                references=references,
                total_samples=len(predictions),
                evaluation_time=evaluation_time,
                samples_per_second=len(predictions) / evaluation_time if evaluation_time > 0 else 0.0,
                peak_memory_mb=self._get_peak_memory(),
                success=True
            )
            
            # 統計更新
            self.evaluation_stats["total_evaluations"] += 1
            self.evaluation_stats["successful_evaluations"] += 1
            
            # 履歴に追加
            self.evaluation_history.append(result)
            
            self.logger.log_performance_metric(
                metric_name="adapter_evaluation_completed",
                value=evaluation_time,
                unit="seconds",
                component="adapter_evaluator"
            )
            
            return result
            
        except Exception as e:
            # エラー結果作成
            error_result = EvaluationResult(
                adapter_name=adapter_name,
                task=eval_config.task,
                config=eval_config,
                evaluation_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
            
            # 統計更新
            self.evaluation_stats["total_evaluations"] += 1
            self.evaluation_stats["failed_evaluations"] += 1
            
            # 履歴に追加
            self.evaluation_history.append(error_result)
            
            self.logger.log_alert(
                alert_type="adapter_evaluation_failed",
                severity="ERROR",
                message=f"Adapter evaluation failed for '{adapter_name}': {e}"
            )
            
            raise
    
    async def _prepare_dataset(self, eval_config: EvaluationConfig) -> HFDataset:
        """データセット準備"""
        
        if eval_config.dataset_name:
            # HuggingFace Datasetsから読み込み
            dataset = await asyncio.to_thread(
                load_dataset,
                eval_config.dataset_name,
                eval_config.dataset_config,
                split=eval_config.dataset_split
            )
        else:
            # デフォルトデータセット
            dataset = self._create_default_dataset(eval_config.task)
        
        # サンプル数制限
        if eval_config.max_samples and len(dataset) > eval_config.max_samples:
            dataset = dataset.select(range(eval_config.max_samples))
        
        return dataset
    
    def _create_default_dataset(self, task: EvaluationTask) -> HFDataset:
        """デフォルトデータセット作成"""
        
        if task == EvaluationTask.PERPLEXITY:
            # パープレキシティ用サンプル
            texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is a subset of artificial intelligence.",
                "Natural language processing enables computers to understand human language.",
                "Deep learning models can learn complex patterns from data."
            ]
            return HFDataset.from_dict({"text": texts})
        
        elif task == EvaluationTask.BLEU:
            # BLEU用サンプル
            predictions = ["The cat is on the mat", "Hello world"]
            references = [["The cat sits on the mat"], ["Hello world"]]
            return HFDataset.from_dict({
                "predictions": predictions,
                "references": references
            })
        
        else:
            # 汎用サンプル
            return HFDataset.from_dict({
                "text": ["Sample text for evaluation"],
                "label": [1]
            })
    
    async def _run_evaluation(
        self,
        model: PeftModel,
        dataset: HFDataset,
        eval_config: EvaluationConfig
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        """評価実行"""
        
        predictions = []
        references = []
        sample_results = []
        
        # パイプライン作成
        tokenizer = self.peft_pool.tokenizer
        
        if eval_config.task in [EvaluationTask.PERPLEXITY]:
            # パープレキシティ計算
            for i, sample in enumerate(dataset):
                text = sample.get("text", "")
                
                # トークン化
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=eval_config.max_length
                )
                
                # 推論
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss.item()
                    perplexity = torch.exp(outputs.loss).item()
                
                predictions.append(str(perplexity))
                references.append(text)
                
                sample_results.append({
                    "sample_id": i,
                    "text": text,
                    "loss": loss,
                    "perplexity": perplexity
                })
        
        elif eval_config.task in [EvaluationTask.BLEU, EvaluationTask.ROUGE]:
            # 生成タスク
            for i, sample in enumerate(dataset):
                input_text = sample.get("input", sample.get("text", ""))
                reference = sample.get("reference", sample.get("target", ""))
                
                # 生成
                inputs = tokenizer(
                    input_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=eval_config.max_length
                )
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=eval_config.max_new_tokens,
                        temperature=eval_config.temperature,
                        do_sample=eval_config.do_sample,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # デコード
                generated_text = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
                
                predictions.append(generated_text)
                references.append(reference)
                
                sample_results.append({
                    "sample_id": i,
                    "input": input_text,
                    "prediction": generated_text,
                    "reference": reference
                })
        
        else:
            # 分類タスク
            for i, sample in enumerate(dataset):
                text = sample.get("text", "")
                label = sample.get("label", 0)
                
                # 推論
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=eval_config.max_length
                )
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    predicted_label = torch.argmax(logits, dim=-1).item()
                
                predictions.append(str(predicted_label))
                references.append(str(label))
                
                sample_results.append({
                    "sample_id": i,
                    "text": text,
                    "predicted_label": predicted_label,
                    "true_label": label
                })
        
        return predictions, references, sample_results
    
    async def _calculate_metrics(
        self,
        predictions: List[str],
        references: List[str],
        metrics: List[EvaluationMetric]
    ) -> Dict[str, float]:
        """メトリクス計算"""
        
        results = {}
        
        for metric in metrics:
            try:
                if metric == EvaluationMetric.ACCURACY:
                    evaluator = self.get_evaluator("accuracy")
                    if evaluator:
                        result = evaluator.compute(
                            predictions=[int(p) for p in predictions],
                            references=[int(r) for r in references]
                        )
                        results["accuracy"] = result["accuracy"]
                
                elif metric == EvaluationMetric.F1_SCORE:
                    evaluator = self.get_evaluator("f1")
                    if evaluator:
                        result = evaluator.compute(
                            predictions=[int(p) for p in predictions],
                            references=[int(r) for r in references],
                            average="weighted"
                        )
                        results["f1"] = result["f1"]
                
                elif metric == EvaluationMetric.BLEU_SCORE:
                    evaluator = self.get_evaluator("bleu")
                    if evaluator:
                        result = evaluator.compute(
                            predictions=predictions,
                            references=[[r] for r in references]
                        )
                        results["bleu"] = result["bleu"]
                
                elif metric == EvaluationMetric.ROUGE_L:
                    evaluator = self.get_evaluator("rouge")
                    if evaluator:
                        result = evaluator.compute(
                            predictions=predictions,
                            references=references
                        )
                        results["rouge_l"] = result["rougeL"]
                
                elif metric == EvaluationMetric.SCORE:
                    # パープレキシティなどのスコア
                    if predictions and all(self._is_float(p) for p in predictions):
                        scores = [float(p) for p in predictions]
                        results["score"] = np.mean(scores)
                        results["perplexity"] = np.mean(scores)  # パープレキシティの場合
                
            except Exception as e:
                self.logger.log_alert(
                    alert_type="metric_calculation_failed",
                    severity="WARNING",
                    message=f"Failed to calculate metric '{metric.value}': {e}"
                )
        
        return results
    
    def _is_float(self, value: str) -> bool:
        """文字列が浮動小数点数かチェック"""
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def _get_peak_memory(self) -> float:
        """ピークメモリ使用量取得（MB）"""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**2)
        return 0.0
    
    async def compare_adapters(
        self,
        baseline_adapter: str,
        comparison_adapters: List[str],
        eval_config: EvaluationConfig,
        dataset: Optional[HFDataset] = None
    ) -> List[ComparisonResult]:
        """アダプタ比較評価"""
        
        comparison_results = []
        
        try:
            # ベースライン評価
            baseline_result = await self.evaluate_adapter(
                baseline_adapter, eval_config, dataset
            )
            
            # 比較対象評価
            for comparison_adapter in comparison_adapters:
                comparison_result = await self.evaluate_adapter(
                    comparison_adapter, eval_config, dataset
                )
                
                # 比較分析
                comparison = self._analyze_comparison(
                    baseline_adapter, baseline_result,
                    comparison_adapter, comparison_result,
                    eval_config.task
                )
                
                comparison_results.append(comparison)
                self.comparison_history.append(comparison)
            
            # 統計更新
            self.evaluation_stats["total_comparisons"] += len(comparison_adapters)
            
            return comparison_results
            
        except Exception as e:
            self.logger.log_alert(
                alert_type="adapter_comparison_failed",
                severity="ERROR",
                message=f"Adapter comparison failed: {e}"
            )
            raise
    
    def _analyze_comparison(
        self,
        baseline_name: str,
        baseline_result: EvaluationResult,
        comparison_name: str,
        comparison_result: EvaluationResult,
        task: EvaluationTask
    ) -> ComparisonResult:
        """比較分析"""
        
        # 改善度計算
        improvement = {}
        for metric_name in baseline_result.metrics:
            if metric_name in comparison_result.metrics:
                baseline_value = baseline_result.metrics[metric_name]
                comparison_value = comparison_result.metrics[metric_name]
                
                if baseline_value != 0:
                    improvement[metric_name] = (comparison_value - baseline_value) / baseline_value
                else:
                    improvement[metric_name] = 0.0
        
        # 総合改善度（主要メトリクスの平均）
        main_metrics = ["accuracy", "f1", "bleu", "rouge_l"]
        relevant_improvements = [
            improvement[metric] for metric in main_metrics 
            if metric in improvement
        ]
        
        overall_improvement = np.mean(relevant_improvements) if relevant_improvements else 0.0
        
        # 推奨事項
        if overall_improvement > 0.05:  # 5%以上の改善
            recommendation = f"{comparison_name} shows significant improvement over {baseline_name}"
        elif overall_improvement > 0.01:  # 1%以上の改善
            recommendation = f"{comparison_name} shows moderate improvement over {baseline_name}"
        elif overall_improvement > -0.01:  # 1%以内の差
            recommendation = f"{comparison_name} performs similarly to {baseline_name}"
        else:
            recommendation = f"{baseline_name} performs better than {comparison_name}"
        
        return ComparisonResult(
            baseline_adapter=baseline_name,
            comparison_adapter=comparison_name,
            task=task,
            baseline_metrics=baseline_result.metrics,
            comparison_metrics=comparison_result.metrics,
            improvement=improvement,
            overall_improvement=overall_improvement,
            recommendation=recommendation
        )
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """評価統計取得"""
        stats = self.evaluation_stats.copy()
        
        # 成功率計算
        if stats["total_evaluations"] > 0:
            stats["success_rate"] = stats["successful_evaluations"] / stats["total_evaluations"]
        else:
            stats["success_rate"] = 0.0
        
        # 最近の評価統計
        recent_evaluations = self.evaluation_history[-10:]  # 最新10件
        if recent_evaluations:
            successful_recent = [e for e in recent_evaluations if e.success]
            if successful_recent:
                stats["recent_average_time"] = sum(e.evaluation_time for e in successful_recent) / len(successful_recent)
                stats["recent_average_samples"] = sum(e.total_samples for e in successful_recent) / len(successful_recent)
        
        return stats
    
    def get_evaluation_history(
        self,
        adapter_name: Optional[str] = None,
        task: Optional[EvaluationTask] = None,
        limit: Optional[int] = None
    ) -> List[EvaluationResult]:
        """評価履歴取得"""
        
        history = self.evaluation_history
        
        # フィルタリング
        if adapter_name:
            history = [e for e in history if e.adapter_name == adapter_name]
        
        if task:
            history = [e for e in history if e.task == task]
        
        # 件数制限
        if limit:
            history = history[-limit:]
        
        return history
    
    def get_best_adapter(
        self,
        task: EvaluationTask,
        metric: str = "accuracy"
    ) -> Optional[str]:
        """最高性能アダプタ取得"""
        
        task_evaluations = [e for e in self.evaluation_history if e.task == task and e.success]
        
        if not task_evaluations:
            return None
        
        # メトリクスでソート
        valid_evaluations = [e for e in task_evaluations if metric in e.metrics]
        
        if not valid_evaluations:
            return None
        
        best_evaluation = max(valid_evaluations, key=lambda e: e.metrics[metric])
        return best_evaluation.adapter_name
    
    async def shutdown(self) -> None:
        """評価システム終了"""
        
        final_stats = self.get_evaluation_stats()
        
        self.logger.log_shutdown(
            component="adapter_evaluator",
            uptime_seconds=0,  # TODO: 実際の稼働時間計算
            final_stats=final_stats
        )


# 便利関数
async def create_adapter_evaluator(
    peft_pool: PEFTAdapterPool,
    system_monitor: Optional[SystemMonitor] = None
) -> AdapterEvaluator:
    """アダプタ評価システム作成・初期化"""
    
    evaluator = AdapterEvaluator(peft_pool, system_monitor)
    
    if await evaluator.initialize():
        return evaluator
    else:
        raise RuntimeError("Failed to initialize adapter evaluator")


# 使用例
async def main():
    """テスト用メイン関数"""
    try:
        from .peft_manager import create_peft_adapter_pool
        
        # PEFTプール作成
        peft_pool = await create_peft_adapter_pool("microsoft/DialoGPT-small")
        
        # 評価システム作成
        evaluator = await create_adapter_evaluator(peft_pool)
        
        print("=== Adapter Evaluator Test ===")
        
        # 評価設定作成
        eval_config = EvaluationConfig(
            task=EvaluationTask.PERPLEXITY,
            metrics=[EvaluationMetric.SCORE],
            max_samples=5
        )
        
        print(f"Evaluation Config: {eval_config}")
        
        # 統計取得
        stats = evaluator.get_evaluation_stats()
        print(f"Evaluation Stats: {stats}")
        
        await evaluator.shutdown()
        await peft_pool.shutdown()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())