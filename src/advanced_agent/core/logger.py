"""
ログシステムモジュール
Loguru による構造化ログ管理
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
from loguru import logger
from .config import get_config


class StructuredLogger:
    """構造化ログ管理クラス"""
    
    def __init__(self, 
                 log_level: str = "INFO",
                 log_dir: str = "logs",
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_json: bool = True):
        
        self.log_level = log_level
        self.log_dir = Path(log_dir)
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_json = enable_json
        
        # ログディレクトリ作成
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # デフォルトハンドラー削除
        logger.remove()
        
        # ログハンドラー設定
        self._setup_handlers()
    
    def _setup_handlers(self) -> None:
        """ログハンドラー設定"""
        
        # コンソール出力
        if self.enable_console:
            logger.add(
                sys.stdout,
                level=self.log_level,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                       "<level>{level: <8}</level> | "
                       "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                       "<level>{message}</level>",
                colorize=True,
                backtrace=True,
                diagnose=True
            )
        
        # ファイル出力（通常ログ）
        if self.enable_file:
            logger.add(
                self.log_dir / "agent_{time:YYYY-MM-DD}.log",
                level=self.log_level,
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
                rotation="1 day",
                retention="7 days",
                compression="zip",
                backtrace=True,
                diagnose=True
            )
        
        # JSON構造化ログ
        if self.enable_json:
            # JSONはシンプルシンク経由だとクローズ時にI/O例外が起こりやすいため安全に出力
            logger.add(
                (self.log_dir / "agent_structured_{time:YYYY-MM-DD}.json").as_posix(),
                level=self.log_level,
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
                rotation="1 day",
                retention="7 days",
                compression="zip",
                enqueue=True
            )
        
        # エラー専用ログ
        logger.add(
            self.log_dir / "agent_errors_{time:YYYY-MM-DD}.log",
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="1 day",
            retention="30 days",
            compression="zip",
            backtrace=True,
            diagnose=True
        )
        
        # パフォーマンス専用ログ
        logger.add(
            self.log_dir / "agent_performance_{time:YYYY-MM-DD}.log",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | PERF | {message}",
            filter=lambda record: "PERF" in record["message"],
            rotation="1 day",
            retention="3 days"
        )
    
    def _json_formatter(self, record: Dict[str, Any]) -> str:
        """JSON フォーマッター"""
        log_entry = {
            "timestamp": record["time"].isoformat() if hasattr(record["time"], 'isoformat') else str(record["time"]),
            "level": record["level"].name,
            "logger": record["name"],
            "module": record["module"],
            "function": record["function"],
            "line": record["line"],
            "message": record["message"],
            "thread": record["thread"].name if hasattr(record["thread"], 'name') else str(record["thread"]),
            "process": record["process"].name if hasattr(record["process"], 'name') else str(record["process"])
        }
        
        # 追加フィールドがある場合
        if "extra" in record and record["extra"]:
            log_entry.update(record["extra"])
        
        # 例外情報がある場合
        if record["exception"]:
            log_entry["exception"] = {
                "type": record["exception"].type.__name__,
                "value": str(record["exception"].value),
                "traceback": record["exception"].traceback
            }
        
        return json.dumps(log_entry, ensure_ascii=False) + "\n"


class AgentLogger:
    """エージェント専用ログ機能"""
    
    def __init__(self):
        self.config = get_config()
        self.structured_logger = StructuredLogger(
            log_level=self.config.monitoring.log_level,
            log_dir=str(self.config.get_logs_dir())
        )
    
    def error(self, message: str) -> None:
        """エラーログ"""
        logger.error(message)
    
    def warning(self, message: str) -> None:
        """警告ログ"""
        logger.warning(message)
    
    def info(self, message: str) -> None:
        """情報ログ"""
        logger.info(message)
    
    def debug(self, message: str) -> None:
        """デバッグログ"""
        logger.debug(message)
    
    def log_system_stats(self, stats: Dict[str, Any]) -> None:
        """システム統計ログ"""
        logger.bind(
            component="system_monitor",
            stats_type="system",
            **stats
        ).debug("System statistics collected")
    
    def log_gpu_stats(self, gpu_stats: Dict[str, Any]) -> None:
        """GPU統計ログ"""
        logger.bind(
            component="system_monitor",
            stats_type="gpu",
            **gpu_stats
        ).debug("GPU statistics collected")
    
    def log_inference_start(self, 
                           model_name: str, 
                           prompt_length: int,
                           context_length: int) -> None:
        """推論開始ログ"""
        logger.bind(
            component="inference_engine",
            action="start",
            model_name=model_name,
            prompt_length=prompt_length,
            context_length=context_length
        ).info(f"Starting inference with {model_name}")
    
    def log_inference_complete(self, 
                              model_name: str,
                              response_length: int,
                              processing_time: float,
                              memory_used_mb: float) -> None:
        """推論完了ログ"""
        logger.bind(
            component="inference_engine",
            action="complete",
            model_name=model_name,
            response_length=response_length,
            processing_time_seconds=processing_time,
            memory_used_mb=memory_used_mb
        ).info(f"Inference completed in {processing_time:.2f}s")
    
    def log_inference_error(self, 
                           model_name: str,
                           error: Exception,
                           fallback_used: bool = False) -> None:
        """推論エラーログ"""
        logger.bind(
            component="inference_engine",
            action="error",
            model_name=model_name,
            error_type=type(error).__name__,
            fallback_used=fallback_used
        ).error(f"Inference error with {model_name}: {error}")
    
    def log_memory_pressure(self, 
                           pressure_level: str,
                           system_memory_percent: float,
                           gpu_memory_percent: float,
                           action_taken: Optional[str] = None) -> None:
        """メモリ圧迫ログ"""
        logger.bind(
            component="memory_manager",
            pressure_level=pressure_level,
            system_memory_percent=system_memory_percent,
            gpu_memory_percent=gpu_memory_percent,
            action_taken=action_taken
        ).warning(f"Memory pressure detected: {pressure_level}")
    
    def log_adapter_evolution(self, 
                             generation: int,
                             population_size: int,
                             best_score: float,
                             avg_score: float) -> None:
        """アダプタ進化ログ"""
        logger.bind(
            component="evolutionary_system",
            generation=generation,
            population_size=population_size,
            best_score=best_score,
            avg_score=avg_score
        ).info(f"Generation {generation}: best={best_score:.3f}, avg={avg_score:.3f}")
    
    def log_adapter_training(self, 
                            adapter_id: str,
                            task_domain: str,
                            training_loss: float,
                            validation_score: float,
                            training_time: float) -> None:
        """アダプタ学習ログ"""
        logger.bind(
            component="adapter_training",
            adapter_id=adapter_id,
            task_domain=task_domain,
            training_loss=training_loss,
            validation_score=validation_score,
            training_time_seconds=training_time
        ).info(f"Adapter {adapter_id} trained: loss={training_loss:.4f}, score={validation_score:.3f}")
    
    def log_memory_operation(self, 
                            operation: str,
                            memory_type: str,
                            item_count: int,
                            processing_time: float) -> None:
        """記憶操作ログ"""
        logger.bind(
            component="memory_system",
            operation=operation,
            memory_type=memory_type,
            item_count=item_count,
            processing_time_seconds=processing_time
        ).debug(f"Memory {operation}: {item_count} {memory_type} items in {processing_time:.3f}s")
    
    def log_api_request(self, 
                       endpoint: str,
                       method: str,
                       status_code: int,
                       response_time: float,
                       user_id: Optional[str] = None) -> None:
        """API リクエストログ"""
        logger.bind(
            component="api_server",
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time_seconds=response_time,
            user_id=user_id
        ).info(f"{method} {endpoint} -> {status_code} ({response_time:.3f}s)")
    
    def log_performance_metric(self, 
                              metric_name: str,
                              value: Union[float, int],
                              unit: str,
                              component: str) -> None:
        """パフォーマンスメトリクスログ"""
        logger.bind(
            component=component,
            metric_name=metric_name,
            metric_value=value,
            metric_unit=unit
        ).debug(f"PERF: {metric_name}={value}{unit}")
    
    def log_alert(self, 
                  alert_type: str,
                  severity: str,
                  message: str,
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """アラートログ"""
        log_data = {
            "component": "alert_system",
            "alert_type": alert_type,
            "severity": severity,
            "alert_message": message
        }
        
        if metadata:
            log_data.update(metadata)
        
        logger.bind(**log_data).warning(f"ALERT [{severity}] {alert_type}: {message}")
    
    def log_config_change(self, 
                         config_section: str,
                         old_value: Any,
                         new_value: Any,
                         changed_by: Optional[str] = None) -> None:
        """設定変更ログ"""
        logger.bind(
            component="config_manager",
            config_section=config_section,
            old_value=str(old_value),
            new_value=str(new_value),
            changed_by=changed_by
        ).info(f"Config changed: {config_section} = {new_value}")
    
    def log_startup(self, 
                   component: str,
                   version: str,
                   config_summary: Dict[str, Any]) -> None:
        """起動ログ"""
        logger.bind(
            component=component,
            version=version,
            **config_summary
        ).info(f"{component} v{version} started")
    
    def log_shutdown(self, 
                    component: str,
                    uptime_seconds: float,
                    final_stats: Optional[Dict[str, Any]] = None) -> None:
        """終了ログ"""
        log_data = {
            "component": component,
            "uptime_seconds": uptime_seconds
        }
        
        if final_stats:
            log_data.update(final_stats)
        
        logger.bind(**log_data).info(f"{component} shutdown after {uptime_seconds:.1f}s")


# グローバルロガーインスタンス
_agent_logger: Optional[AgentLogger] = None


def get_logger() -> AgentLogger:
    """グローバルロガー取得"""
    global _agent_logger
    if _agent_logger is None:
        _agent_logger = AgentLogger()
    return _agent_logger


def setup_logging(log_level: str = "INFO") -> AgentLogger:
    """ログシステム初期化"""
    global _agent_logger
    
    # 設定更新
    config = get_config()
    config.monitoring.log_level = log_level
    
    # ロガー初期化
    _agent_logger = AgentLogger()
    
    return _agent_logger


# 便利関数
def log_function_call(func_name: str, args: Dict[str, Any], processing_time: float) -> None:
    """関数呼び出しログ"""
    get_logger().log_performance_metric(
        metric_name=f"{func_name}_execution_time",
        value=processing_time,
        unit="seconds",
        component="function_profiler"
    )


def log_exception(component: str, exception: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """例外ログ"""
    log_data = {
        "component": component,
        "exception_type": type(exception).__name__,
        "exception_message": str(exception)
    }
    
    if context:
        log_data.update(context)
    
    logger.bind(**log_data).exception(f"Exception in {component}")


# デコレータ
def log_execution_time(component: str):
    """実行時間ログデコレータ"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()
                get_logger().log_performance_metric(
                    metric_name=f"{func.__name__}_execution_time",
                    value=execution_time,
                    unit="seconds",
                    component=component
                )
                return result
            except Exception as e:
                log_exception(component, e, {"function": func.__name__})
                raise
        return wrapper
    return decorator


# 使用例
if __name__ == "__main__":
    # ログシステム初期化
    agent_logger = setup_logging("DEBUG")
    
    # 各種ログテスト
    agent_logger.log_startup("test_component", "1.0.0", {"test": True})
    
    agent_logger.log_system_stats({
        "cpu_percent": 45.2,
        "memory_percent": 67.8,
        "disk_usage": 23.1
    })
    
    agent_logger.log_inference_start("deepseek-r1:7b", 150, 2048)
    agent_logger.log_inference_complete("deepseek-r1:7b", 300, 2.5, 1024.5)
    
    agent_logger.log_alert("high_memory_usage", "WARNING", "Memory usage exceeded 90%")
    
    print("Logging test completed. Check logs/ directory for output files.")