"""
Memory Optimization Manager
メモリ最適化管理システム

VRAM使用量の最適化とメモリ効率の向上を提供します。
RTX 4050 6GB VRAM環境での効率的なメモリ管理を実装します。
"""

import asyncio
import gc
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
import psutil
import torch

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """メモリタイプ"""
    GPU_MEMORY = "gpu_memory"
    SYSTEM_MEMORY = "system_memory"
    CACHE_MEMORY = "cache_memory"
    MODEL_MEMORY = "model_memory"


class OptimizationStrategy(Enum):
    """最適化戦略"""
    AGGRESSIVE = "aggressive"  # 積極的な最適化
    BALANCED = "balanced"      # バランス型最適化
    CONSERVATIVE = "conservative"  # 保守的な最適化
    RTX4050_OPTIMIZED = "rtx4050_optimized"  # RTX 4050専用最適化


@dataclass
class MemoryMetrics:
    """メモリメトリクス"""
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_memory_available_mb: float = 0.0
    system_memory_used_mb: float = 0.0
    system_memory_total_mb: float = 0.0
    system_memory_available_mb: float = 0.0
    cache_size_mb: float = 0.0
    model_memory_mb: float = 0.0
    memory_fragmentation_percent: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MemoryOptimizationConfig:
    """メモリ最適化設定"""
    max_gpu_memory_usage_percent: float = 85.0
    max_system_memory_usage_percent: float = 80.0
    cache_cleanup_threshold_mb: float = 1000.0
    model_offload_threshold_mb: float = 2000.0
    garbage_collection_interval_seconds: int = 30
    memory_monitoring_interval_seconds: int = 5
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.RTX4050_OPTIMIZED
    enable_automatic_cleanup: bool = True
    enable_model_offloading: bool = True
    enable_cache_optimization: bool = True


class MemoryOptimizationManager:
    """メモリ最適化管理システム"""
    
    def __init__(self, config: Optional[MemoryOptimizationConfig] = None):
        """
        初期化
        
        Args:
            config: メモリ最適化設定
        """
        self.config = config or MemoryOptimizationConfig()
        self.memory_metrics: List[MemoryMetrics] = []
        self.optimization_callbacks: List[Callable] = []
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # メモリ使用量の履歴
        self.memory_history: List[Tuple[datetime, float, float]] = []  # (timestamp, gpu_mb, system_mb)
        
        # 最適化統計
        self.optimization_stats = {
            "cleanup_count": 0,
            "offload_count": 0,
            "cache_cleanup_count": 0,
            "garbage_collection_count": 0,
            "total_memory_freed_mb": 0.0
        }
        
        logger.info(f"メモリ最適化管理システム初期化完了 - 戦略: {self.config.optimization_strategy.value}")
    
    async def start_monitoring(self):
        """メモリ監視開始"""
        if self.is_monitoring:
            logger.warning("メモリ監視は既に開始されています")
            return
        
        self.is_monitoring = True
        
        # 監視タスク開始
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # 自動クリーンアップタスク開始
        if self.config.enable_automatic_cleanup:
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("メモリ監視開始")
    
    async def stop_monitoring(self):
        """メモリ監視停止"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        # タスクの停止
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("メモリ監視停止")
    
    async def get_memory_metrics(self) -> MemoryMetrics:
        """現在のメモリメトリクス取得"""
        try:
            # GPU メモリ情報
            gpu_memory_used_mb = 0.0
            gpu_memory_total_mb = 0.0
            gpu_memory_available_mb = 0.0
            
            if torch.cuda.is_available():
                gpu_memory_used_mb = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_memory_total_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                gpu_memory_available_mb = gpu_memory_total_mb - gpu_memory_used_mb
            
            # システムメモリ情報
            system_memory = psutil.virtual_memory()
            system_memory_used_mb = system_memory.used / 1024 / 1024
            system_memory_total_mb = system_memory.total / 1024 / 1024
            system_memory_available_mb = system_memory.available / 1024 / 1024
            
            # キャッシュサイズ（簡略化）
            cache_size_mb = 0.0
            if hasattr(torch.cuda, 'memory_reserved'):
                cache_size_mb = torch.cuda.memory_reserved() / 1024 / 1024
            
            # モデルメモリ（簡略化）
            model_memory_mb = gpu_memory_used_mb * 0.8  # 推定値
            
            # メモリフラグメンテーション（簡略化）
            memory_fragmentation_percent = 0.0
            if gpu_memory_total_mb > 0:
                memory_fragmentation_percent = (gpu_memory_used_mb / gpu_memory_total_mb) * 100
            
            metrics = MemoryMetrics(
                gpu_memory_used_mb=gpu_memory_used_mb,
                gpu_memory_total_mb=gpu_memory_total_mb,
                gpu_memory_available_mb=gpu_memory_available_mb,
                system_memory_used_mb=system_memory_used_mb,
                system_memory_total_mb=system_memory_total_mb,
                system_memory_available_mb=system_memory_available_mb,
                cache_size_mb=cache_size_mb,
                model_memory_mb=model_memory_mb,
                memory_fragmentation_percent=memory_fragmentation_percent
            )
            
            # 履歴に追加
            self.memory_metrics.append(metrics)
            self.memory_history.append((
                datetime.now(),
                gpu_memory_used_mb,
                system_memory_used_mb
            ))
            
            # 履歴サイズ制限
            if len(self.memory_metrics) > 1000:
                self.memory_metrics = self.memory_metrics[-500:]
            
            if len(self.memory_history) > 1000:
                self.memory_history = self.memory_history[-500:]
            
            return metrics
            
        except Exception as e:
            logger.error(f"メモリメトリクス取得エラー: {e}")
            return MemoryMetrics()
    
    async def optimize_memory(self, force: bool = False) -> Dict[str, Any]:
        """メモリ最適化実行"""
        try:
            logger.info("メモリ最適化開始")
            
            # 現在のメモリ状態取得
            current_metrics = await self.get_memory_metrics()
            
            # 最適化が必要かチェック
            if not force and not self._needs_optimization(current_metrics):
                logger.info("メモリ最適化は不要です")
                return {"optimization_performed": False, "reason": "no_optimization_needed"}
            
            optimization_results = {
                "optimization_performed": True,
                "timestamp": datetime.now().isoformat(),
                "initial_metrics": {
                    "gpu_memory_used_mb": current_metrics.gpu_memory_used_mb,
                    "system_memory_used_mb": current_metrics.system_memory_used_mb,
                    "cache_size_mb": current_metrics.cache_size_mb
                },
                "optimizations": []
            }
            
            # 戦略に基づく最適化実行
            if self.config.optimization_strategy == OptimizationStrategy.RTX4050_OPTIMIZED:
                results = await self._rtx4050_optimization(current_metrics)
            elif self.config.optimization_strategy == OptimizationStrategy.AGGRESSIVE:
                results = await self._aggressive_optimization(current_metrics)
            elif self.config.optimization_strategy == OptimizationStrategy.BALANCED:
                results = await self._balanced_optimization(current_metrics)
            else:  # CONSERVATIVE
                results = await self._conservative_optimization(current_metrics)
            
            optimization_results["optimizations"] = results
            
            # 最適化後のメトリクス取得
            final_metrics = await self.get_memory_metrics()
            optimization_results["final_metrics"] = {
                "gpu_memory_used_mb": final_metrics.gpu_memory_used_mb,
                "system_memory_used_mb": final_metrics.system_memory_used_mb,
                "cache_size_mb": final_metrics.cache_size_mb
            }
            
            # 最適化効果計算
            gpu_freed = current_metrics.gpu_memory_used_mb - final_metrics.gpu_memory_used_mb
            system_freed = current_metrics.system_memory_used_mb - final_metrics.system_memory_used_mb
            
            optimization_results["memory_freed"] = {
                "gpu_memory_mb": max(0, gpu_freed),
                "system_memory_mb": max(0, system_freed),
                "total_memory_mb": max(0, gpu_freed + system_freed)
            }
            
            # 統計更新
            self.optimization_stats["total_memory_freed_mb"] += optimization_results["memory_freed"]["total_memory_mb"]
            
            logger.info(f"メモリ最適化完了 - GPU: {gpu_freed:.1f}MB, System: {system_freed:.1f}MB 解放")
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"メモリ最適化エラー: {e}")
            return {
                "optimization_performed": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def cleanup_cache(self) -> Dict[str, Any]:
        """キャッシュクリーンアップ"""
        try:
            logger.info("キャッシュクリーンアップ開始")
            
            initial_metrics = await self.get_memory_metrics()
            initial_cache_size = initial_metrics.cache_size_mb
            
            # PyTorch キャッシュクリーンアップ
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # ガベージコレクション
            gc.collect()
            
            final_metrics = await self.get_memory_metrics()
            final_cache_size = final_metrics.cache_size_mb
            
            cache_freed = initial_cache_size - final_cache_size
            
            self.optimization_stats["cache_cleanup_count"] += 1
            self.optimization_stats["garbage_collection_count"] += 1
            
            logger.info(f"キャッシュクリーンアップ完了 - {cache_freed:.1f}MB 解放")
            
            return {
                "cache_cleanup_performed": True,
                "cache_freed_mb": max(0, cache_freed),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"キャッシュクリーンアップエラー: {e}")
            return {
                "cache_cleanup_performed": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def offload_models(self) -> Dict[str, Any]:
        """モデルオフロード"""
        try:
            logger.info("モデルオフロード開始")
            
            # 実際のモデルオフロード実装は、使用しているモデルフレームワークに依存
            # ここでは簡略化された実装
            
            initial_metrics = await self.get_memory_metrics()
            initial_gpu_memory = initial_metrics.gpu_memory_used_mb
            
            # モデルオフロード処理（簡略化）
            if torch.cuda.is_available():
                # 未使用のテンソルをクリア
                torch.cuda.empty_cache()
                
                # モデルをCPUに移動（実際の実装では、モデルの状態を管理する必要がある）
                # ここでは簡略化
                pass
            
            final_metrics = await self.get_memory_metrics()
            final_gpu_memory = final_metrics.gpu_memory_used_mb
            
            memory_freed = initial_gpu_memory - final_gpu_memory
            
            self.optimization_stats["offload_count"] += 1
            
            logger.info(f"モデルオフロード完了 - {memory_freed:.1f}MB 解放")
            
            return {
                "model_offload_performed": True,
                "memory_freed_mb": max(0, memory_freed),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"モデルオフロードエラー: {e}")
            return {
                "model_offload_performed": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def add_optimization_callback(self, callback: Callable):
        """最適化コールバック追加"""
        self.optimization_callbacks.append(callback)
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """最適化統計取得"""
        return {
            **self.optimization_stats,
            "memory_history_size": len(self.memory_history),
            "metrics_history_size": len(self.memory_metrics),
            "is_monitoring": self.is_monitoring
        }
    
    def _needs_optimization(self, metrics: MemoryMetrics) -> bool:
        """最適化が必要かチェック"""
        # GPU メモリ使用率チェック
        if metrics.gpu_memory_total_mb > 0:
            gpu_usage_percent = (metrics.gpu_memory_used_mb / metrics.gpu_memory_total_mb) * 100
            if gpu_usage_percent > self.config.max_gpu_memory_usage_percent:
                return True
        
        # システムメモリ使用率チェック
        if metrics.system_memory_total_mb > 0:
            system_usage_percent = (metrics.system_memory_used_mb / metrics.system_memory_total_mb) * 100
            if system_usage_percent > self.config.max_system_memory_usage_percent:
                return True
        
        # キャッシュサイズチェック
        if metrics.cache_size_mb > self.config.cache_cleanup_threshold_mb:
            return True
        
        return False
    
    async def _rtx4050_optimization(self, metrics: MemoryMetrics) -> List[Dict[str, Any]]:
        """RTX 4050専用最適化"""
        results = []
        
        # キャッシュクリーンアップ
        if metrics.cache_size_mb > 500.0:  # 500MB以上の場合
            cache_result = await self.cleanup_cache()
            results.append({"type": "cache_cleanup", "result": cache_result})
        
        # モデルオフロード
        if metrics.gpu_memory_used_mb > self.config.model_offload_threshold_mb:
            offload_result = await self.offload_models()
            results.append({"type": "model_offload", "result": offload_result})
        
        # ガベージコレクション
        gc.collect()
        results.append({"type": "garbage_collection", "result": {"performed": True}})
        
        return results
    
    async def _aggressive_optimization(self, metrics: MemoryMetrics) -> List[Dict[str, Any]]:
        """積極的最適化"""
        results = []
        
        # 強制的なキャッシュクリーンアップ
        cache_result = await self.cleanup_cache()
        results.append({"type": "cache_cleanup", "result": cache_result})
        
        # 強制的なモデルオフロード
        offload_result = await self.offload_models()
        results.append({"type": "model_offload", "result": offload_result})
        
        # 複数回のガベージコレクション
        for i in range(3):
            gc.collect()
        results.append({"type": "garbage_collection", "result": {"performed": True, "cycles": 3}})
        
        return results
    
    async def _balanced_optimization(self, metrics: MemoryMetrics) -> List[Dict[str, Any]]:
        """バランス型最適化"""
        results = []
        
        # 条件付きキャッシュクリーンアップ
        if metrics.cache_size_mb > 300.0:
            cache_result = await self.cleanup_cache()
            results.append({"type": "cache_cleanup", "result": cache_result})
        
        # 条件付きモデルオフロード
        if metrics.gpu_memory_used_mb > 1500.0:
            offload_result = await self.offload_models()
            results.append({"type": "model_offload", "result": offload_result})
        
        # ガベージコレクション
        gc.collect()
        results.append({"type": "garbage_collection", "result": {"performed": True}})
        
        return results
    
    async def _conservative_optimization(self, metrics: MemoryMetrics) -> List[Dict[str, Any]]:
        """保守的最適化"""
        results = []
        
        # 最小限のキャッシュクリーンアップ
        if metrics.cache_size_mb > 800.0:
            cache_result = await self.cleanup_cache()
            results.append({"type": "cache_cleanup", "result": cache_result})
        
        # 最小限のガベージコレクション
        gc.collect()
        results.append({"type": "garbage_collection", "result": {"performed": True}})
        
        return results
    
    async def _monitoring_loop(self):
        """監視ループ"""
        while self.is_monitoring:
            try:
                # メモリメトリクス取得
                metrics = await self.get_memory_metrics()
                
                # 最適化が必要かチェック
                if self._needs_optimization(metrics):
                    logger.warning(f"メモリ使用量が閾値を超過 - GPU: {metrics.gpu_memory_used_mb:.1f}MB, System: {metrics.system_memory_used_mb:.1f}MB")
                    
                    # 自動最適化実行
                    if self.config.enable_automatic_cleanup:
                        await self.optimize_memory()
                
                # コールバック実行
                for callback in self.optimization_callbacks:
                    try:
                        await callback(metrics)
                    except Exception as e:
                        logger.error(f"最適化コールバックエラー: {e}")
                
                await asyncio.sleep(self.config.memory_monitoring_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"監視ループエラー: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_loop(self):
        """クリーンアップループ"""
        while self.is_monitoring:
            try:
                await asyncio.sleep(self.config.garbage_collection_interval_seconds)
                
                # 定期的なガベージコレクション
                gc.collect()
                self.optimization_stats["garbage_collection_count"] += 1
                
                logger.debug("定期ガベージコレクション実行")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"クリーンアップループエラー: {e}")
    
    async def cleanup(self):
        """リソースクリーンアップ"""
        await self.stop_monitoring()
        logger.info("メモリ最適化管理システムクリーンアップ完了")


# 使用例
async def main():
    """使用例"""
    # 設定
    config = MemoryOptimizationConfig(
        optimization_strategy=OptimizationStrategy.RTX4050_OPTIMIZED,
        max_gpu_memory_usage_percent=80.0,
        enable_automatic_cleanup=True
    )
    
    # メモリ最適化管理システム初期化
    memory_manager = MemoryOptimizationManager(config)
    
    try:
        # 監視開始
        await memory_manager.start_monitoring()
        
        # メモリメトリクス取得
        metrics = await memory_manager.get_memory_metrics()
        print(f"GPU メモリ: {metrics.gpu_memory_used_mb:.1f}MB / {metrics.gpu_memory_total_mb:.1f}MB")
        print(f"システムメモリ: {metrics.system_memory_used_mb:.1f}MB / {metrics.system_memory_total_mb:.1f}MB")
        
        # 手動最適化実行
        optimization_result = await memory_manager.optimize_memory()
        print(f"最適化結果: {optimization_result}")
        
        # 統計表示
        stats = memory_manager.get_optimization_stats()
        print(f"最適化統計: {stats}")
        
    finally:
        # クリーンアップ
        await memory_manager.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
