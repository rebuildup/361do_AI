"""
Request Pool
並行リクエスト処理の最適化
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass
from loguru import logger


@dataclass
class RequestTask:
    """リクエストタスク"""
    task_id: str
    coro: Awaitable[Any]
    priority: int = 0
    created_at: float = None
    timeout: Optional[float] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


class RequestPool:
    """
    リクエストプール
    並行リクエストの管理と最適化
    """
    
    def __init__(self, max_concurrent: int = 10, default_timeout: float = 30.0):
        self.max_concurrent = max_concurrent
        self.default_timeout = default_timeout
        
        # セマフォで並行数制御
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # アクティブタスク管理
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: List[str] = []
        
        # 統計情報
        self.total_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0
        self.timeout_requests = 0
        
        # キューイング
        self.pending_queue: asyncio.Queue[RequestTask] = asyncio.Queue()
        self.processing_task: Optional[asyncio.Task] = None
        self._shutdown = False
    
    async def start(self):
        """プール開始"""
        if self.processing_task is not None:
            return
        
        self._shutdown = False
        self.processing_task = asyncio.create_task(self._process_queue())
        logger.info(f"Request pool started with max_concurrent={self.max_concurrent}")
    
    async def stop(self):
        """プール停止"""
        self._shutdown = True
        
        # 処理中のタスクをキャンセル
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        # アクティブタスクをキャンセル
        for task in self.active_tasks.values():
            task.cancel()
        
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)
        
        logger.info("Request pool stopped")
    
    async def _process_queue(self):
        """キュー処理ループ"""
        while not self._shutdown:
            try:
                # タイムアウトでキューから取得
                try:
                    request_task = await asyncio.wait_for(
                        self.pending_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # セマフォ取得（並行数制御）
                await self.semaphore.acquire()
                
                # タスク実行
                task = asyncio.create_task(
                    self._execute_request(request_task)
                )
                self.active_tasks[request_task.task_id] = task
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in queue processing: {e}")
    
    async def _execute_request(self, request_task: RequestTask):
        """リクエスト実行"""
        try:
            # タイムアウト設定
            timeout = request_task.timeout or self.default_timeout
            
            # リクエスト実行
            result = await asyncio.wait_for(request_task.coro, timeout=timeout)
            
            # 成功統計
            self.completed_requests += 1
            self.completed_tasks.append(request_task.task_id)
            
            return result
            
        except asyncio.TimeoutError:
            self.timeout_requests += 1
            logger.warning(f"Request {request_task.task_id} timed out after {timeout}s")
            raise
            
        except Exception as e:
            self.failed_requests += 1
            logger.error(f"Request {request_task.task_id} failed: {e}")
            raise
            
        finally:
            # クリーンアップ
            self.active_tasks.pop(request_task.task_id, None)
            self.semaphore.release()
    
    async def submit(
        self,
        coro: Awaitable[Any],
        task_id: Optional[str] = None,
        priority: int = 0,
        timeout: Optional[float] = None
    ) -> str:
        """リクエスト投入"""
        if task_id is None:
            import uuid
            task_id = str(uuid.uuid4())
        
        request_task = RequestTask(
            task_id=task_id,
            coro=coro,
            priority=priority,
            timeout=timeout
        )
        
        self.total_requests += 1
        await self.pending_queue.put(request_task)
        
        return task_id
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """タスク完了待機"""
        # タスクが完了済みかチェック
        if task_id in self.completed_tasks:
            # 完了済みタスクの場合、結果を返す（簡易実装）
            return "completed"
        
        if task_id not in self.active_tasks:
            # 少し待ってからもう一度チェック
            await asyncio.sleep(0.01)
            if task_id not in self.active_tasks and task_id not in self.completed_tasks:
                raise ValueError(f"Task {task_id} not found")
        
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            
            if timeout:
                return await asyncio.wait_for(task, timeout=timeout)
            else:
                return await task
        
        return "completed"
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        return {
            'max_concurrent': self.max_concurrent,
            'active_tasks': len(self.active_tasks),
            'pending_tasks': self.pending_queue.qsize(),
            'total_requests': self.total_requests,
            'completed_requests': self.completed_requests,
            'failed_requests': self.failed_requests,
            'timeout_requests': self.timeout_requests,
            'success_rate': (
                self.completed_requests / self.total_requests * 100
                if self.total_requests > 0 else 0
            )
        }
    
    def is_overloaded(self, threshold: float = 0.8) -> bool:
        """過負荷状態チェック"""
        utilization = len(self.active_tasks) / self.max_concurrent
        return utilization >= threshold
    
    async def wait_for_capacity(self, timeout: Optional[float] = None):
        """容量空き待機"""
        start_time = time.time()
        
        while len(self.active_tasks) >= self.max_concurrent:
            if timeout and (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError("Timeout waiting for capacity")
            
            await asyncio.sleep(0.1)


class AdaptiveRequestPool(RequestPool):
    """
    適応的リクエストプール
    負荷に応じて並行数を動的調整
    """
    
    def __init__(
        self,
        initial_concurrent: int = 5,
        min_concurrent: int = 1,
        max_concurrent: int = 20,
        default_timeout: float = 30.0
    ):
        super().__init__(initial_concurrent, default_timeout)
        self.min_concurrent = min_concurrent
        self.max_concurrent_limit = max_concurrent
        self.initial_concurrent = initial_concurrent
        
        # 適応制御
        self.adjustment_interval = 10.0  # 10秒間隔で調整
        self.last_adjustment = time.time()
        self.performance_history: List[Dict[str, float]] = []
        
        # 調整タスク
        self.adjustment_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """プール開始（適応制御付き）"""
        await super().start()
        
        self.adjustment_task = asyncio.create_task(self._adaptive_adjustment())
        logger.info("Adaptive request pool started")
    
    async def stop(self):
        """プール停止"""
        if self.adjustment_task:
            self.adjustment_task.cancel()
            try:
                await self.adjustment_task
            except asyncio.CancelledError:
                pass
        
        await super().stop()
    
    async def _adaptive_adjustment(self):
        """適応的調整ループ"""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.adjustment_interval)
                await self._adjust_concurrency()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in adaptive adjustment: {e}")
    
    async def _adjust_concurrency(self):
        """並行数調整"""
        current_time = time.time()
        
        # パフォーマンス測定
        stats = self.get_stats()
        utilization = len(self.active_tasks) / self.max_concurrent
        
        # 履歴に追加
        self.performance_history.append({
            'timestamp': current_time,
            'utilization': utilization,
            'success_rate': stats['success_rate'],
            'active_tasks': stats['active_tasks'],
            'concurrent_limit': self.max_concurrent
        })
        
        # 履歴を最新100件に制限
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        # 調整判定
        if len(self.performance_history) < 3:
            return
        
        recent_performance = self.performance_history[-3:]
        avg_utilization = sum(p['utilization'] for p in recent_performance) / len(recent_performance)
        avg_success_rate = sum(p['success_rate'] for p in recent_performance) / len(recent_performance)
        
        # 調整ロジック
        new_concurrent = self.max_concurrent
        
        if avg_utilization > 0.8 and avg_success_rate > 95:
            # 高利用率・高成功率 → 並行数増加
            new_concurrent = min(self.max_concurrent + 2, self.max_concurrent_limit)
            logger.info(f"Increasing concurrency: {self.max_concurrent} -> {new_concurrent}")
            
        elif avg_utilization < 0.3:
            # 低利用率 → 並行数減少
            new_concurrent = max(self.max_concurrent - 1, self.min_concurrent)
            logger.info(f"Decreasing concurrency: {self.max_concurrent} -> {new_concurrent}")
            
        elif avg_success_rate < 90:
            # 低成功率 → 並行数減少
            new_concurrent = max(self.max_concurrent - 2, self.min_concurrent)
            logger.warning(f"Reducing concurrency due to low success rate: {self.max_concurrent} -> {new_concurrent}")
        
        # 並行数更新
        if new_concurrent != self.max_concurrent:
            await self._update_concurrency(new_concurrent)
    
    async def _update_concurrency(self, new_concurrent: int):
        """並行数更新"""
        old_concurrent = self.max_concurrent
        self.max_concurrent = new_concurrent
        
        # セマフォ更新
        if new_concurrent > old_concurrent:
            # 増加：セマフォを追加リリース
            for _ in range(new_concurrent - old_concurrent):
                self.semaphore.release()
        elif new_concurrent < old_concurrent:
            # 減少：セマフォを取得
            for _ in range(old_concurrent - new_concurrent):
                await self.semaphore.acquire()
        
        logger.info(f"Concurrency updated: {old_concurrent} -> {new_concurrent}")