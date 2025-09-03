"""
LangChain Callbacks による推論時間・メモリ測定
パフォーマンス監視とメトリクス収集
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult, ChatResult
from langchain_core.messages import BaseMessage
from langchain_core.agents import AgentAction, AgentFinish

from ..core.logger import get_logger
from ..monitoring.system_monitor import SystemMonitor


@dataclass
class PerformanceMetrics:
    """パフォーマンスメトリクス"""
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    
    # メモリメトリクス
    memory_start_mb: float = 0.0
    memory_end_mb: float = 0.0
    memory_peak_mb: float = 0.0
    memory_delta_mb: float = 0.0
    
    # GPU メトリクス
    gpu_memory_start_mb: float = 0.0
    gpu_memory_end_mb: float = 0.0
    gpu_memory_peak_mb: float = 0.0
    gpu_utilization_avg: float = 0.0
    
    # トークンメトリクス
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    # 推論メトリクス
    reasoning_steps: int = 0
    tool_calls: int = 0
    errors: int = 0
    
    # その他
    model_name: str = ""
    request_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CallbackEvent:
    """コールバックイベント"""
    event_type: str
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)


class PerformanceCallbackHandler(BaseCallbackHandler):
    """パフォーマンス測定コールバックハンドラー"""
    
    def __init__(self, request_id: str, enable_gpu_monitoring: bool = True):
        super().__init__()
        self.request_id = request_id
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.logger = get_logger()
        
        # メトリクス
        self.metrics = PerformanceMetrics(
            start_time=time.time(),
            request_id=request_id
        )
        
        # イベント履歴
        self.events: List[CallbackEvent] = []
        
        # システム監視
        self.system_monitor = None
        if enable_gpu_monitoring:
            try:
                self.system_monitor = SystemMonitor(
                    monitoring_interval=0.5,
                    enable_prometheus=False
                )
            except Exception as e:
                self.logger.log_alert(
                    alert_type="gpu_monitoring_init_failed",
                    severity="WARNING",
                    message=f"GPU monitoring initialization failed: {e}"
                )
        
        # メモリ監視用
        self._memory_samples: List[float] = []
        self._gpu_memory_samples: List[float] = []
        self._gpu_utilization_samples: List[float] = []
        self._monitoring_active = False
        self._monitoring_thread = None
        
        # 初期メモリ測定
        self._record_initial_metrics()
    
    def _record_initial_metrics(self) -> None:
        """初期メトリクス記録"""
        # システムメモリ
        memory = psutil.virtual_memory()
        self.metrics.memory_start_mb = memory.used / (1024**2)
        
        # GPU メモリ（利用可能な場合）
        if self.system_monitor:
            try:
                gpu_stats = self.system_monitor.get_gpu_stats()
                if gpu_stats:
                    self.metrics.gpu_memory_start_mb = gpu_stats[0].used_memory_gb * 1024
            except Exception as e:
                self.logger.log_performance_metric(
                    metric_name="gpu_metrics_error",
                    value=1,
                    unit="count",
                    component="performance_callback"
                )
    
    def _start_memory_monitoring(self) -> None:
        """メモリ監視開始"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._memory_monitoring_loop)
        self._monitoring_thread.daemon = True
        self._monitoring_thread.start()
    
    def _stop_memory_monitoring(self) -> None:
        """メモリ監視停止"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=1.0)
    
    def _memory_monitoring_loop(self) -> None:
        """メモリ監視ループ"""
        while self._monitoring_active:
            try:
                # システムメモリ
                memory = psutil.virtual_memory()
                memory_mb = memory.used / (1024**2)
                self._memory_samples.append(memory_mb)
                
                # GPU メモリ
                if self.system_monitor:
                    try:
                        gpu_stats = self.system_monitor.get_gpu_stats()
                        if gpu_stats:
                            gpu_memory_mb = gpu_stats[0].used_memory_gb * 1024
                            gpu_utilization = gpu_stats[0].utilization_percent
                            
                            self._gpu_memory_samples.append(gpu_memory_mb)
                            self._gpu_utilization_samples.append(gpu_utilization)
                    except Exception:
                        pass
                
                time.sleep(0.1)  # 100ms間隔
                
            except Exception as e:
                self.logger.log_performance_metric(
                    metric_name="memory_monitoring_error",
                    value=1,
                    unit="count",
                    component="performance_callback"
                )
                break
    
    def _add_event(self, event_type: str, data: Dict[str, Any] = None) -> None:
        """イベント追加"""
        event = CallbackEvent(
            event_type=event_type,
            timestamp=time.time(),
            data=data or {}
        )
        self.events.append(event)
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """LLM開始時"""
        self.metrics.model_name = serialized.get("name", "unknown")
        
        # 入力トークン数推定
        if prompts:
            self.metrics.input_tokens = sum(len(prompt.split()) for prompt in prompts)
        
        # メモリ監視開始
        self._start_memory_monitoring()
        
        # イベント記録
        self._add_event("llm_start", {
            "model_name": self.metrics.model_name,
            "input_tokens": self.metrics.input_tokens,
            "prompts_count": len(prompts)
        })
        
        # ログ記録
        self.logger.log_inference_start(
            model_name=self.metrics.model_name,
            prompt_length=self.metrics.input_tokens,
            context_length=self.metrics.input_tokens
        )
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """新しいトークン生成時"""
        self.metrics.output_tokens += 1
        
        # 定期的にメトリクス更新
        if self.metrics.output_tokens % 10 == 0:
            self._add_event("token_milestone", {
                "output_tokens": self.metrics.output_tokens,
                "elapsed_time": time.time() - self.metrics.start_time
            })
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """LLM終了時"""
        self.metrics.end_time = time.time()
        self.metrics.duration = self.metrics.end_time - self.metrics.start_time
        
        # 出力トークン数計算（ストリーミングでない場合）
        if self.metrics.output_tokens == 0 and response.generations:
            for generation in response.generations:
                for gen in generation:
                    self.metrics.output_tokens += len(gen.text.split())
        
        self.metrics.total_tokens = self.metrics.input_tokens + self.metrics.output_tokens
        
        # メモリ監視停止
        self._stop_memory_monitoring()
        
        # 最終メモリ測定
        memory = psutil.virtual_memory()
        self.metrics.memory_end_mb = memory.used / (1024**2)
        self.metrics.memory_delta_mb = self.metrics.memory_end_mb - self.metrics.memory_start_mb
        
        # ピークメモリ計算
        if self._memory_samples:
            self.metrics.memory_peak_mb = max(self._memory_samples)
        
        # GPU メトリクス計算
        if self.system_monitor and self._gpu_memory_samples:
            try:
                gpu_stats = self.system_monitor.get_gpu_stats()
                if gpu_stats:
                    self.metrics.gpu_memory_end_mb = gpu_stats[0].used_memory_gb * 1024
                
                self.metrics.gpu_memory_peak_mb = max(self._gpu_memory_samples)
                
                if self._gpu_utilization_samples:
                    self.metrics.gpu_utilization_avg = sum(self._gpu_utilization_samples) / len(self._gpu_utilization_samples)
            except Exception:
                pass
        
        # イベント記録
        self._add_event("llm_end", {
            "duration": self.metrics.duration,
            "output_tokens": self.metrics.output_tokens,
            "total_tokens": self.metrics.total_tokens,
            "memory_delta_mb": self.metrics.memory_delta_mb
        })
        
        # ログ記録
        self.logger.log_inference_complete(
            model_name=self.metrics.model_name,
            response_length=self.metrics.output_tokens,
            processing_time=self.metrics.duration,
            memory_used_mb=self.metrics.memory_delta_mb
        )
        
        # パフォーマンスメトリクス記録
        self._log_performance_metrics()
    
    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        """LLMエラー時"""
        self.metrics.errors += 1
        self.metrics.end_time = time.time()
        self.metrics.duration = self.metrics.end_time - self.metrics.start_time
        
        # メモリ監視停止
        self._stop_memory_monitoring()
        
        # イベント記録
        self._add_event("llm_error", {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "duration": self.metrics.duration
        })
        
        # ログ記録
        self.logger.log_inference_error(
            model_name=self.metrics.model_name,
            error=error,
            fallback_used=False
        )
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """ツール開始時"""
        self.metrics.tool_calls += 1
        
        self._add_event("tool_start", {
            "tool_name": serialized.get("name", "unknown"),
            "input_length": len(input_str)
        })
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """ツール終了時"""
        self._add_event("tool_end", {
            "output_length": len(output)
        })
    
    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        """ツールエラー時"""
        self.metrics.errors += 1
        
        self._add_event("tool_error", {
            "error_type": type(error).__name__,
            "error_message": str(error)
        })
    
    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """エージェントアクション時"""
        self.metrics.reasoning_steps += 1
        
        self._add_event("agent_action", {
            "tool": action.tool,
            "tool_input": str(action.tool_input),
            "log": action.log
        })
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """エージェント終了時"""
        self._add_event("agent_finish", {
            "return_values": finish.return_values,
            "log": finish.log
        })
    
    def _log_performance_metrics(self) -> None:
        """パフォーマンスメトリクスログ記録"""
        # 処理時間
        self.logger.log_performance_metric(
            metric_name="inference_duration",
            value=self.metrics.duration or 0,
            unit="seconds",
            component="performance_callback"
        )
        
        # トークン数
        self.logger.log_performance_metric(
            metric_name="total_tokens",
            value=self.metrics.total_tokens,
            unit="tokens",
            component="performance_callback"
        )
        
        # メモリ使用量
        self.logger.log_performance_metric(
            metric_name="memory_delta",
            value=self.metrics.memory_delta_mb,
            unit="MB",
            component="performance_callback"
        )
        
        # GPU メトリクス
        if self.metrics.gpu_utilization_avg > 0:
            self.logger.log_performance_metric(
                metric_name="gpu_utilization_avg",
                value=self.metrics.gpu_utilization_avg,
                unit="percent",
                component="performance_callback"
            )
        
        # 推論ステップ数
        if self.metrics.reasoning_steps > 0:
            self.logger.log_performance_metric(
                metric_name="reasoning_steps",
                value=self.metrics.reasoning_steps,
                unit="steps",
                component="performance_callback"
            )
    
    def get_metrics(self) -> PerformanceMetrics:
        """メトリクス取得"""
        return self.metrics
    
    def get_events(self) -> List[CallbackEvent]:
        """イベント履歴取得"""
        return self.events
    
    def get_summary(self) -> Dict[str, Any]:
        """サマリー取得"""
        return {
            "request_id": self.metrics.request_id,
            "model_name": self.metrics.model_name,
            "duration": self.metrics.duration,
            "total_tokens": self.metrics.total_tokens,
            "memory_delta_mb": self.metrics.memory_delta_mb,
            "memory_peak_mb": self.metrics.memory_peak_mb,
            "gpu_utilization_avg": self.metrics.gpu_utilization_avg,
            "reasoning_steps": self.metrics.reasoning_steps,
            "tool_calls": self.metrics.tool_calls,
            "errors": self.metrics.errors,
            "events_count": len(self.events)
        }


class AggregatedCallbackHandler(BaseCallbackHandler):
    """集約コールバックハンドラー"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger()
        
        # 集約メトリクス
        self.total_requests = 0
        self.total_duration = 0.0
        self.total_tokens = 0
        self.total_errors = 0
        
        # 詳細統計
        self.model_stats = defaultdict(lambda: {
            "requests": 0,
            "duration": 0.0,
            "tokens": 0,
            "errors": 0
        })
        
        # パフォーマンス履歴
        self.performance_history: List[Dict[str, Any]] = []
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """LLM開始時"""
        self.total_requests += 1
        model_name = serialized.get("name", "unknown")
        self.model_stats[model_name]["requests"] += 1
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """LLM終了時"""
        # 処理時間は個別のコールバックハンドラーから取得する必要がある
        pass
    
    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        """LLMエラー時"""
        self.total_errors += 1
    
    def add_performance_record(self, metrics: PerformanceMetrics) -> None:
        """パフォーマンス記録追加"""
        self.total_duration += metrics.duration or 0
        self.total_tokens += metrics.total_tokens
        
        # モデル別統計更新
        model_stats = self.model_stats[metrics.model_name]
        model_stats["duration"] += metrics.duration or 0
        model_stats["tokens"] += metrics.total_tokens
        model_stats["errors"] += metrics.errors
        
        # 履歴追加
        self.performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "request_id": metrics.request_id,
            "model_name": metrics.model_name,
            "duration": metrics.duration,
            "tokens": metrics.total_tokens,
            "memory_delta_mb": metrics.memory_delta_mb
        })
        
        # 履歴サイズ制限
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-500:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """統計情報取得"""
        avg_duration = self.total_duration / self.total_requests if self.total_requests > 0 else 0
        avg_tokens = self.total_tokens / self.total_requests if self.total_requests > 0 else 0
        
        return {
            "total_requests": self.total_requests,
            "total_duration": self.total_duration,
            "total_tokens": self.total_tokens,
            "total_errors": self.total_errors,
            "average_duration": avg_duration,
            "average_tokens": avg_tokens,
            "error_rate": self.total_errors / self.total_requests if self.total_requests > 0 else 0,
            "model_statistics": dict(self.model_stats),
            "recent_performance": self.performance_history[-10:]  # 最新10件
        }


# グローバル集約ハンドラー
_aggregated_handler: Optional[AggregatedCallbackHandler] = None


def get_aggregated_handler() -> AggregatedCallbackHandler:
    """グローバル集約ハンドラー取得"""
    global _aggregated_handler
    if _aggregated_handler is None:
        _aggregated_handler = AggregatedCallbackHandler()
    return _aggregated_handler


# 便利関数
def create_performance_callback(request_id: str, enable_gpu: bool = True) -> PerformanceCallbackHandler:
    """パフォーマンスコールバック作成"""
    return PerformanceCallbackHandler(request_id, enable_gpu)


def get_performance_statistics() -> Dict[str, Any]:
    """パフォーマンス統計取得"""
    return get_aggregated_handler().get_statistics()


# 使用例
async def main():
    """テスト用メイン関数"""
    import asyncio
    
    print("Performance Callback Test")
    print("=" * 40)
    
    # パフォーマンスコールバック作成
    callback = create_performance_callback("test_request_001")
    
    # 模擬的なLLM処理
    print("Simulating LLM processing...")
    
    # LLM開始
    callback.on_llm_start(
        {"name": "test_model"},
        ["This is a test prompt for performance measurement."]
    )
    
    # 処理時間シミュレート
    await asyncio.sleep(1.0)
    
    # トークン生成シミュレート
    for i in range(50):
        callback.on_llm_new_token(f"token_{i}")
        await asyncio.sleep(0.01)
    
    # LLM終了
    from langchain_core.outputs import LLMResult, Generation
    result = LLMResult(generations=[[Generation(text="This is a test response.")]])
    callback.on_llm_end(result)
    
    # メトリクス表示
    metrics = callback.get_metrics()
    print(f"\nPerformance Metrics:")
    print(f"  Duration: {metrics.duration:.2f}s")
    print(f"  Input Tokens: {metrics.input_tokens}")
    print(f"  Output Tokens: {metrics.output_tokens}")
    print(f"  Total Tokens: {metrics.total_tokens}")
    print(f"  Memory Delta: {metrics.memory_delta_mb:.2f}MB")
    print(f"  Memory Peak: {metrics.memory_peak_mb:.2f}MB")
    
    # イベント履歴
    events = callback.get_events()
    print(f"\nEvents ({len(events)}):")
    for event in events:
        print(f"  {event.event_type}: {event.timestamp:.2f}")
    
    # サマリー
    summary = callback.get_summary()
    print(f"\nSummary: {summary}")
    
    # 集約統計
    aggregated = get_aggregated_handler()
    aggregated.add_performance_record(metrics)
    
    stats = get_performance_statistics()
    print(f"\nAggregated Statistics: {stats}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())