"""
設定検証システム
Pydantic による設定ファイル検証と最適化
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import yaml
from pydantic import ValidationError
from loguru import logger

from .config import AdvancedAgentConfig, GPUConfig, CPUConfig, MemoryConfig


@dataclass
class ConfigValidationIssue:
    """設定検証問題"""
    severity: str  # "ERROR", "WARNING", "INFO"
    field_path: str
    message: str
    current_value: Any
    suggested_value: Optional[Any] = None
    auto_fixable: bool = False


@dataclass
class ConfigValidationReport:
    """設定検証レポート"""
    timestamp: datetime
    config_file_path: Optional[str]
    validation_passed: bool
    issues: List[ConfigValidationIssue] = field(default_factory=list)
    optimizations: List[ConfigValidationIssue] = field(default_factory=list)
    auto_fixes_applied: List[str] = field(default_factory=list)


class ConfigValidator:
    """設定検証クラス"""
    
    def __init__(self):
        self.system_info = self._get_system_info()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """システム情報取得"""
        try:
            import psutil
            import torch
            
            # CPU情報
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # メモリ情報
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            
            # GPU情報
            gpu_info = {}
            if torch.cuda.is_available():
                gpu_info = {
                    "available": True,
                    "device_count": torch.cuda.device_count(),
                    "device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None,
                    "memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.device_count() > 0 else 0
                }
            else:
                gpu_info = {"available": False}
            
            return {
                "cpu_count": cpu_count,
                "cpu_freq_mhz": cpu_freq.max if cpu_freq else None,
                "memory_gb": memory_gb,
                "gpu_info": gpu_info
            }
            
        except Exception as e:
            logger.warning(f"Failed to get system info: {e}")
            return {}
    
    def validate_config_file(self, config_path: str) -> ConfigValidationReport:
        """設定ファイル検証"""
        report = ConfigValidationReport(
            timestamp=datetime.now(),
            config_file_path=config_path,
            validation_passed=False
        )
        
        try:
            # ファイル存在確認
            config_file = Path(config_path)
            if not config_file.exists():
                report.issues.append(ConfigValidationIssue(
                    severity="ERROR",
                    field_path="file",
                    message=f"Configuration file not found: {config_path}",
                    current_value=None
                ))
                return report
            
            # YAML読み込み
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    yaml_data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                report.issues.append(ConfigValidationIssue(
                    severity="ERROR",
                    field_path="yaml",
                    message=f"Invalid YAML format: {e}",
                    current_value=None
                ))
                return report
            
            # Pydantic検証
            try:
                config = AdvancedAgentConfig(**yaml_data)
                report.validation_passed = True
                
                # 追加検証とオプティマイゼーション
                self._validate_gpu_config(config.gpu, report)
                self._validate_cpu_config(config.cpu, report)
                self._validate_memory_config(config.memory, report)
                self._validate_model_config(config.models, report)
                self._validate_learning_config(config.learning, report)
                self._validate_paths_config(config, report)
                
            except ValidationError as e:
                report.validation_passed = False
                for error in e.errors():
                    field_path = ".".join(str(loc) for loc in error["loc"])
                    report.issues.append(ConfigValidationIssue(
                        severity="ERROR",
                        field_path=field_path,
                        message=error["msg"],
                        current_value=error.get("input")
                    ))
            
        except Exception as e:
            report.issues.append(ConfigValidationIssue(
                severity="ERROR",
                field_path="general",
                message=f"Unexpected error during validation: {e}",
                current_value=None
            ))
        
        return report
    
    def _validate_gpu_config(self, gpu_config: GPUConfig, report: ConfigValidationReport) -> None:
        """GPU設定検証"""
        system_gpu = self.system_info.get("gpu_info", {})
        
        if system_gpu.get("available"):
            system_vram = system_gpu.get("memory_gb", 0)
            
            # VRAM制限チェック
            if gpu_config.max_vram_gb > system_vram:
                report.issues.append(ConfigValidationIssue(
                    severity="WARNING",
                    field_path="gpu.max_vram_gb",
                    message=f"Configured VRAM limit ({gpu_config.max_vram_gb}GB) exceeds system VRAM ({system_vram:.1f}GB)",
                    current_value=gpu_config.max_vram_gb,
                    suggested_value=max(system_vram * 0.9, 1.0),
                    auto_fixable=True
                ))
            
            # RTX 4050 最適化
            if "RTX 4050" in system_gpu.get("device_name", ""):
                if gpu_config.max_vram_gb > 5.0:
                    report.optimizations.append(ConfigValidationIssue(
                        severity="INFO",
                        field_path="gpu.max_vram_gb",
                        message="RTX 4050 detected: Recommend limiting VRAM to 5.0GB for stability",
                        current_value=gpu_config.max_vram_gb,
                        suggested_value=5.0,
                        auto_fixable=True
                    ))
                
                # 量子化レベル最適化
                if 8 not in gpu_config.quantization_levels:
                    report.optimizations.append(ConfigValidationIssue(
                        severity="INFO",
                        field_path="gpu.quantization_levels",
                        message="RTX 4050 detected: Recommend including 8-bit quantization",
                        current_value=gpu_config.quantization_levels,
                        suggested_value=[8, 4, 3],
                        auto_fixable=True
                    ))
        else:
            # GPU未検出時の警告
            report.issues.append(ConfigValidationIssue(
                severity="WARNING",
                field_path="gpu",
                message="No GPU detected, but GPU configuration is present",
                current_value="GPU config enabled",
                suggested_value="Consider CPU-only configuration"
            ))
    
    def _validate_cpu_config(self, cpu_config: CPUConfig, report: ConfigValidationReport) -> None:
        """CPU設定検証"""
        system_cpu_count = self.system_info.get("cpu_count", 1)
        
        # スレッド数チェック
        if cpu_config.max_threads > system_cpu_count:
            report.issues.append(ConfigValidationIssue(
                severity="WARNING",
                field_path="cpu.max_threads",
                message=f"Configured threads ({cpu_config.max_threads}) exceeds system CPU count ({system_cpu_count})",
                current_value=cpu_config.max_threads,
                suggested_value=system_cpu_count,
                auto_fixable=True
            ))
        
        # 最適スレッド数提案
        optimal_threads = min(system_cpu_count, 16)  # 最大16スレッド
        if cpu_config.max_threads != optimal_threads:
            report.optimizations.append(ConfigValidationIssue(
                severity="INFO",
                field_path="cpu.max_threads",
                message=f"Optimal thread count for your system: {optimal_threads}",
                current_value=cpu_config.max_threads,
                suggested_value=optimal_threads,
                auto_fixable=True
            ))
    
    def _validate_memory_config(self, memory_config: MemoryConfig, report: ConfigValidationReport) -> None:
        """メモリ設定検証"""
        system_memory_gb = self.system_info.get("memory_gb", 0)
        
        # システムRAM設定チェック
        if memory_config.system_ram_gb > system_memory_gb:
            report.issues.append(ConfigValidationIssue(
                severity="WARNING",
                field_path="memory.system_ram_gb",
                message=f"Configured RAM ({memory_config.system_ram_gb}GB) exceeds system RAM ({system_memory_gb:.1f}GB)",
                current_value=memory_config.system_ram_gb,
                suggested_value=system_memory_gb,
                auto_fixable=True
            ))
        
        # メモリ使用量最適化
        if system_memory_gb < 16:
            # 低メモリシステム用最適化
            if memory_config.cache_size_mb > 512:
                report.optimizations.append(ConfigValidationIssue(
                    severity="INFO",
                    field_path="memory.cache_size_mb",
                    message="Low memory system detected: Recommend reducing cache size",
                    current_value=memory_config.cache_size_mb,
                    suggested_value=256,
                    auto_fixable=True
                ))
    
    def _validate_model_config(self, model_config, report: ConfigValidationReport) -> None:
        """モデル設定検証"""
        # Ollama URL検証
        if not model_config.ollama_base_url.startswith(("http://", "https://")):
            report.issues.append(ConfigValidationIssue(
                severity="ERROR",
                field_path="models.ollama_base_url",
                message="Invalid Ollama URL format",
                current_value=model_config.ollama_base_url,
                suggested_value="http://localhost:11434"
            ))
        
        # コンテキスト長最適化
        system_gpu = self.system_info.get("gpu_info", {})
        if system_gpu.get("available"):
            gpu_memory = system_gpu.get("memory_gb", 0)
            
            if gpu_memory <= 6:  # RTX 4050クラス
                if model_config.context_length > 4096:
                    report.optimizations.append(ConfigValidationIssue(
                        severity="INFO",
                        field_path="models.context_length",
                        message="6GB VRAM detected: Recommend limiting context length for memory efficiency",
                        current_value=model_config.context_length,
                        suggested_value=4096,
                        auto_fixable=True
                    ))
    
    def _validate_learning_config(self, learning_config, report: ConfigValidationReport) -> None:
        """学習設定検証"""
        # バッチサイズ最適化
        system_gpu = self.system_info.get("gpu_info", {})
        if system_gpu.get("available"):
            gpu_memory = system_gpu.get("memory_gb", 0)
            
            optimal_batch_size = 1
            if gpu_memory >= 8:
                optimal_batch_size = 4
            elif gpu_memory >= 6:
                optimal_batch_size = 2
            
            if learning_config.batch_size > optimal_batch_size:
                report.optimizations.append(ConfigValidationIssue(
                    severity="INFO",
                    field_path="learning.batch_size",
                    message=f"GPU memory ({gpu_memory:.1f}GB) suggests batch size: {optimal_batch_size}",
                    current_value=learning_config.batch_size,
                    suggested_value=optimal_batch_size,
                    auto_fixable=True
                ))
        
        # 学習率検証
        if learning_config.learning_rate > 1e-3:
            report.issues.append(ConfigValidationIssue(
                severity="WARNING",
                field_path="learning.learning_rate",
                message="Learning rate seems high, may cause training instability",
                current_value=learning_config.learning_rate,
                suggested_value=1e-4
            ))
    
    def _validate_paths_config(self, config: AdvancedAgentConfig, report: ConfigValidationReport) -> None:
        """パス設定検証"""
        paths_to_check = [
            ("database.sqlite_path", config.database.sqlite_path),
            ("database.chroma_path", config.database.chroma_path)
        ]
        
        for field_path, path_str in paths_to_check:
            path = Path(path_str)
            
            # 親ディレクトリの書き込み権限チェック
            parent_dir = path.parent
            if not parent_dir.exists():
                try:
                    parent_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    report.issues.append(ConfigValidationIssue(
                        severity="ERROR",
                        field_path=field_path,
                        message=f"Cannot create directory: {e}",
                        current_value=str(path)
                    ))
    
    def apply_auto_fixes(self, config_path: str, report: ConfigValidationReport) -> Tuple[bool, List[str]]:
        """自動修正適用"""
        if not report.validation_passed:
            return False, ["Cannot apply auto-fixes: validation failed"]
        
        try:
            # 設定ファイル読み込み
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
            
            applied_fixes = []
            
            # 自動修正可能な問題とオプティマイゼーションを適用
            for issue in report.issues + report.optimizations:
                if issue.auto_fixable and issue.suggested_value is not None:
                    # フィールドパスを辿って値を設定
                    self._set_nested_value(yaml_data, issue.field_path, issue.suggested_value)
                    applied_fixes.append(f"{issue.field_path}: {issue.current_value} → {issue.suggested_value}")
            
            if applied_fixes:
                # バックアップ作成
                backup_path = f"{config_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                import shutil
                shutil.copy2(config_path, backup_path)
                
                # 修正済み設定保存
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, indent=2)
                
                logger.info(f"Applied {len(applied_fixes)} auto-fixes to {config_path}")
                logger.info(f"Backup saved to {backup_path}")
            
            return True, applied_fixes
            
        except Exception as e:
            logger.error(f"Failed to apply auto-fixes: {e}")
            return False, [f"Auto-fix failed: {e}"]
    
    def _set_nested_value(self, data: Dict, field_path: str, value: Any) -> None:
        """ネストした辞書に値を設定"""
        keys = field_path.split('.')
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def generate_optimized_config(self, base_config_path: Optional[str] = None) -> AdvancedAgentConfig:
        """システム最適化設定生成"""
        # ベース設定読み込み
        if base_config_path and Path(base_config_path).exists():
            base_config = AdvancedAgentConfig.load_from_yaml(base_config_path)
        else:
            base_config = AdvancedAgentConfig()
        
        # システム情報に基づく最適化
        system_gpu = self.system_info.get("gpu_info", {})
        system_memory = self.system_info.get("memory_gb", 16)
        system_cpu_count = self.system_info.get("cpu_count", 8)
        
        # GPU最適化
        if system_gpu.get("available"):
            gpu_memory = system_gpu.get("memory_gb", 6)
            
            # RTX 4050 特別最適化
            if "RTX 4050" in system_gpu.get("device_name", ""):
                base_config.gpu.max_vram_gb = 5.0
                base_config.gpu.quantization_levels = [8, 4, 3]
                base_config.gpu.temperature_threshold = 75  # より保守的
                base_config.learning.batch_size = 2
                base_config.models.context_length = 4096
            else:
                base_config.gpu.max_vram_gb = max(gpu_memory * 0.9, 1.0)
                
                if gpu_memory >= 8:
                    base_config.learning.batch_size = 4
                elif gpu_memory >= 6:
                    base_config.learning.batch_size = 2
                else:
                    base_config.learning.batch_size = 1
        
        # CPU最適化
        base_config.cpu.max_threads = min(system_cpu_count, 16)
        
        # メモリ最適化
        base_config.memory.system_ram_gb = system_memory
        if system_memory < 16:
            base_config.memory.cache_size_mb = 256
            base_config.learning.adapter_pool_size = 5
        elif system_memory >= 32:
            base_config.memory.cache_size_mb = 2048
            base_config.learning.adapter_pool_size = 15
        
        logger.info("Generated optimized configuration based on system specifications")
        return base_config
    
    def print_validation_report(self, report: ConfigValidationReport) -> None:
        """検証レポート出力"""
        print("\n" + "="*60)
        print("CONFIGURATION VALIDATION REPORT")
        print("="*60)
        
        print(f"\nFile: {report.config_file_path}")
        print(f"Timestamp: {report.timestamp}")
        print(f"Validation Status: {'✅ PASSED' if report.validation_passed else '❌ FAILED'}")
        
        if report.issues:
            print(f"\nIssues Found ({len(report.issues)}):")
            for issue in report.issues:
                icon = "❌" if issue.severity == "ERROR" else "⚠️"
                print(f"  {icon} [{issue.severity}] {issue.field_path}")
                print(f"      {issue.message}")
                if issue.suggested_value is not None:
                    print(f"      Suggested: {issue.suggested_value}")
        
        if report.optimizations:
            print(f"\nOptimization Suggestions ({len(report.optimizations)}):")
            for opt in report.optimizations:
                print(f"  💡 {opt.field_path}")
                print(f"      {opt.message}")
                if opt.suggested_value is not None:
                    print(f"      Suggested: {opt.suggested_value}")
        
        if report.auto_fixes_applied:
            print(f"\nAuto-fixes Applied ({len(report.auto_fixes_applied)}):")
            for fix in report.auto_fixes_applied:
                print(f"  🔧 {fix}")
        
        print("\n" + "="*60)


# 便利関数
def validate_and_optimize_config(config_path: str, apply_fixes: bool = False) -> ConfigValidationReport:
    """設定検証と最適化"""
    validator = ConfigValidator()
    report = validator.validate_config_file(config_path)
    
    # レポート出力
    validator.print_validation_report(report)
    
    # 自動修正適用
    if apply_fixes and report.validation_passed:
        success, fixes = validator.apply_auto_fixes(config_path, report)
        if success and fixes:
            report.auto_fixes_applied = fixes
            logger.info(f"Applied {len(fixes)} auto-fixes to configuration")
    
    return report


def generate_system_optimized_config(output_path: str = "config/optimized_agent.yaml") -> AdvancedAgentConfig:
    """システム最適化設定生成"""
    validator = ConfigValidator()
    optimized_config = validator.generate_optimized_config()
    
    # 設定保存
    optimized_config.save_to_yaml(output_path)
    logger.info(f"Generated optimized configuration: {output_path}")
    
    return optimized_config


# 使用例
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        apply_auto_fixes = "--fix" in sys.argv
        
        print(f"Validating configuration: {config_file}")
        report = validate_and_optimize_config(config_file, apply_fixes=apply_auto_fixes)
        
        if not report.validation_passed:
            print("❌ Configuration validation failed!")
            sys.exit(1)
        else:
            print("✅ Configuration validation passed!")
    else:
        print("Generating system-optimized configuration...")
        generate_system_optimized_config()
        print("✅ Optimized configuration generated!")