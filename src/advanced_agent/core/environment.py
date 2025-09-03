"""
環境検証システム
システム起動時の環境チェックと検証
"""

import os
import sys
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import psutil
from loguru import logger

from .config import get_config, AdvancedAgentConfig


@dataclass
class SystemRequirement:
    """システム要件定義"""
    name: str
    description: str
    required: bool = True
    min_version: Optional[str] = None
    check_function: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class EnvironmentCheckResult:
    """環境チェック結果"""
    requirement: SystemRequirement
    passed: bool
    actual_value: Optional[str] = None
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentValidationReport:
    """環境検証レポート"""
    timestamp: datetime
    system_info: Dict[str, Any]
    check_results: List[EnvironmentCheckResult]
    overall_status: str  # "PASS", "WARNING", "FAIL"
    critical_failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class EnvironmentValidator:
    """環境検証クラス"""
    
    def __init__(self, config: Optional[AdvancedAgentConfig] = None):
        self.config = config or get_config()
        self.requirements = self._define_requirements()
    
    def _define_requirements(self) -> List[SystemRequirement]:
        """システム要件定義"""
        return [
            # Python バージョン
            SystemRequirement(
                name="python_version",
                description="Python 3.11以上",
                required=True,
                min_version="3.11.0",
                check_function="check_python_version",
                error_message="Python 3.11以上が必要です"
            ),
            
            # システムメモリ
            SystemRequirement(
                name="system_memory",
                description="システムRAM 16GB以上推奨",
                required=False,
                min_version="16",
                check_function="check_system_memory",
                error_message="16GB以上のRAMを推奨します"
            ),
            
            # GPU検出
            SystemRequirement(
                name="gpu_detection",
                description="NVIDIA GPU検出",
                required=False,
                check_function="check_gpu_availability",
                error_message="NVIDIA GPUが検出されません"
            ),
            
            # CUDA利用可能性
            SystemRequirement(
                name="cuda_availability",
                description="CUDA利用可能性",
                required=False,
                check_function="check_cuda_availability",
                error_message="CUDAが利用できません"
            ),
            
            # 必要なPythonパッケージ
            SystemRequirement(
                name="required_packages",
                description="必要なPythonパッケージ",
                required=True,
                check_function="check_required_packages",
                error_message="必要なパッケージがインストールされていません"
            ),
            
            # Ollama接続
            SystemRequirement(
                name="ollama_connection",
                description="Ollama サーバー接続",
                required=False,
                check_function="check_ollama_connection",
                error_message="Ollamaサーバーに接続できません"
            ),
            
            # ディスク容量
            SystemRequirement(
                name="disk_space",
                description="利用可能ディスク容量 10GB以上",
                required=True,
                min_version="10",
                check_function="check_disk_space",
                error_message="10GB以上の空きディスク容量が必要です"
            ),
            
            # 書き込み権限
            SystemRequirement(
                name="write_permissions",
                description="データディレクトリ書き込み権限",
                required=True,
                check_function="check_write_permissions",
                error_message="データディレクトリへの書き込み権限がありません"
            )
        ]
    
    def check_python_version(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Python バージョンチェック"""
        current_version = sys.version_info
        version_str = f"{current_version.major}.{current_version.minor}.{current_version.micro}"
        
        required_version = (3, 11, 0)
        passed = current_version >= required_version
        
        details = {
            "current_version": version_str,
            "required_version": "3.11.0",
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler()
        }
        
        return passed, version_str, details
    
    def check_system_memory(self) -> Tuple[bool, str, Dict[str, Any]]:
        """システムメモリチェック"""
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        
        required_gb = 16.0
        passed = total_gb >= required_gb
        
        details = {
            "total_gb": round(total_gb, 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_percent": memory.percent,
            "required_gb": required_gb
        }
        
        return passed, f"{total_gb:.1f}GB", details
    
    def check_gpu_availability(self) -> Tuple[bool, str, Dict[str, Any]]:
        """GPU利用可能性チェック"""
        try:
            import pynvml
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            
            if gpu_count > 0:
                # 最初のGPU情報取得
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                name_bytes = pynvml.nvmlDeviceGetName(handle)
                gpu_name = name_bytes.decode('utf-8') if isinstance(name_bytes, bytes) else str(name_bytes)
                
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_memory_gb = memory_info.total / (1024**3)
                
                details = {
                    "gpu_count": gpu_count,
                    "primary_gpu": gpu_name,
                    "total_memory_gb": round(total_memory_gb, 2),
                    "driver_version": pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
                }
                
                return True, f"{gpu_count} GPU(s) detected", details
            else:
                return False, "No GPU detected", {"gpu_count": 0}
                
        except Exception as e:
            return False, f"GPU check failed: {e}", {"error": str(e)}
    
    def check_cuda_availability(self) -> Tuple[bool, str, Dict[str, Any]]:
        """CUDA利用可能性チェック"""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            
            if cuda_available:
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                cuda_version = torch.version.cuda
                
                details = {
                    "cuda_available": True,
                    "device_count": device_count,
                    "current_device": current_device,
                    "device_name": device_name,
                    "cuda_version": cuda_version,
                    "pytorch_version": torch.__version__
                }
                
                return True, f"CUDA {cuda_version} available", details
            else:
                details = {
                    "cuda_available": False,
                    "pytorch_version": torch.__version__
                }
                return False, "CUDA not available", details
                
        except ImportError:
            return False, "PyTorch not installed", {"error": "PyTorch not found"}
        except Exception as e:
            return False, f"CUDA check failed: {e}", {"error": str(e)}
    
    def check_required_packages(self) -> Tuple[bool, str, Dict[str, Any]]:
        """必要パッケージチェック"""
        required_packages = [
            "psutil", "nvidia-ml-py", "prometheus-client", "loguru",
            "pydantic", "pydantic-settings", "pyyaml", "fastapi",
            "streamlit", "typer", "sqlalchemy", "aiosqlite"
        ]
        
        installed_packages = {}
        missing_packages = []
        
        for package in required_packages:
            try:
                module = __import__(package.replace("-", "_"))
                version = getattr(module, "__version__", "unknown")
                installed_packages[package] = version
            except ImportError:
                missing_packages.append(package)
        
        passed = len(missing_packages) == 0
        
        details = {
            "required_packages": required_packages,
            "installed_packages": installed_packages,
            "missing_packages": missing_packages,
            "total_required": len(required_packages),
            "total_installed": len(installed_packages)
        }
        
        if passed:
            return True, f"{len(installed_packages)}/{len(required_packages)} packages installed", details
        else:
            return False, f"Missing {len(missing_packages)} packages", details
    
    def check_ollama_connection(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Ollama接続チェック"""
        try:
            import httpx
            
            base_url = self.config.models.ollama_base_url
            timeout = 5.0
            
            # Ollama API エンドポイントにリクエスト
            response = httpx.get(f"{base_url}/api/tags", timeout=timeout)
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                
                # 設定されたモデルの存在確認
                available_models = [model["name"] for model in models]
                primary_model = self.config.models.primary
                fallback_model = self.config.models.fallback
                
                primary_available = primary_model in available_models
                fallback_available = fallback_model in available_models
                
                details = {
                    "server_url": base_url,
                    "available_models": available_models,
                    "primary_model": primary_model,
                    "fallback_model": fallback_model,
                    "primary_available": primary_available,
                    "fallback_available": fallback_available,
                    "total_models": len(available_models)
                }
                
                if primary_available or fallback_available:
                    return True, f"Connected, {len(available_models)} models available", details
                else:
                    return False, "Connected but required models not found", details
            else:
                return False, f"HTTP {response.status_code}", {"status_code": response.status_code}
                
        except Exception as e:
            return False, f"Connection failed: {e}", {"error": str(e)}
    
    def check_disk_space(self) -> Tuple[bool, str, Dict[str, Any]]:
        """ディスク容量チェック"""
        try:
            # 現在のディレクトリの利用可能容量
            disk_usage = psutil.disk_usage('.')
            free_gb = disk_usage.free / (1024**3)
            total_gb = disk_usage.total / (1024**3)
            used_gb = disk_usage.used / (1024**3)
            
            required_gb = 10.0
            passed = free_gb >= required_gb
            
            details = {
                "free_gb": round(free_gb, 2),
                "total_gb": round(total_gb, 2),
                "used_gb": round(used_gb, 2),
                "used_percent": round((used_gb / total_gb) * 100, 1),
                "required_gb": required_gb
            }
            
            return passed, f"{free_gb:.1f}GB free", details
            
        except Exception as e:
            return False, f"Disk check failed: {e}", {"error": str(e)}
    
    def check_write_permissions(self) -> Tuple[bool, str, Dict[str, Any]]:
        """書き込み権限チェック"""
        test_directories = [
            self.config.get_data_dir(),
            self.config.get_logs_dir(),
            self.config.get_config_dir(),
            Path(self.config.database.chroma_path).parent,
            Path(self.config.database.sqlite_path).parent
        ]
        
        results = {}
        all_writable = True
        
        for directory in test_directories:
            try:
                # ディレクトリ作成（存在しない場合）
                directory.mkdir(parents=True, exist_ok=True)
                
                # テストファイル作成・削除
                test_file = directory / f"test_write_{os.getpid()}.tmp"
                test_file.write_text("test")
                test_file.unlink()
                
                results[str(directory)] = True
                
            except Exception as e:
                results[str(directory)] = False
                all_writable = False
                logger.error(f"Write permission check failed for {directory}: {e}")
        
        details = {
            "test_directories": [str(d) for d in test_directories],
            "write_results": results,
            "writable_count": sum(results.values()),
            "total_directories": len(test_directories)
        }
        
        if all_writable:
            return True, f"{len(test_directories)} directories writable", details
        else:
            failed_count = len(test_directories) - sum(results.values())
            return False, f"{failed_count} directories not writable", details
    
    def run_single_check(self, requirement: SystemRequirement) -> EnvironmentCheckResult:
        """単一チェック実行"""
        try:
            check_method = getattr(self, requirement.check_function)
            passed, actual_value, details = check_method()
            
            return EnvironmentCheckResult(
                requirement=requirement,
                passed=passed,
                actual_value=actual_value,
                details=details
            )
            
        except Exception as e:
            logger.error(f"Environment check failed for {requirement.name}: {e}")
            return EnvironmentCheckResult(
                requirement=requirement,
                passed=False,
                error_message=str(e),
                details={"exception": str(e)}
            )
    
    def validate_environment(self) -> EnvironmentValidationReport:
        """環境検証実行"""
        logger.info("Starting environment validation...")
        
        # システム情報収集
        system_info = {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "hostname": platform.node(),
            "timestamp": datetime.now().isoformat()
        }
        
        # 各チェック実行
        check_results = []
        critical_failures = []
        warnings = []
        recommendations = []
        
        for requirement in self.requirements:
            logger.debug(f"Checking: {requirement.name}")
            result = self.run_single_check(requirement)
            check_results.append(result)
            
            if not result.passed:
                if requirement.required:
                    critical_failures.append(f"{requirement.name}: {requirement.error_message}")
                    logger.error(f"CRITICAL: {requirement.name} - {requirement.error_message}")
                else:
                    warnings.append(f"{requirement.name}: {requirement.error_message}")
                    logger.warning(f"WARNING: {requirement.name} - {requirement.error_message}")
        
        # 推奨事項生成
        if any(not r.passed and r.requirement.name == "system_memory" for r in check_results):
            recommendations.append("より多くのRAMを搭載することで性能が向上します")
        
        if any(not r.passed and r.requirement.name == "gpu_detection" for r in check_results):
            recommendations.append("NVIDIA GPUを使用することで推論速度が大幅に向上します")
        
        if any(not r.passed and r.requirement.name == "cuda_availability" for r in check_results):
            recommendations.append("PyTorchのCUDA版をインストールすることでGPU加速が利用できます")
        
        if any(not r.passed and r.requirement.name == "ollama_connection" for r in check_results):
            recommendations.append("Ollamaサーバーを起動し、必要なモデルをダウンロードしてください")
        
        # 全体ステータス判定
        if critical_failures:
            overall_status = "FAIL"
        elif warnings:
            overall_status = "WARNING"
        else:
            overall_status = "PASS"
        
        report = EnvironmentValidationReport(
            timestamp=datetime.now(),
            system_info=system_info,
            check_results=check_results,
            overall_status=overall_status,
            critical_failures=critical_failures,
            warnings=warnings,
            recommendations=recommendations
        )
        
        logger.info(f"Environment validation completed: {overall_status}")
        return report
    
    def print_validation_report(self, report: EnvironmentValidationReport) -> None:
        """検証レポート出力"""
        print("\n" + "="*60)
        print("ADVANCED AGENT ENVIRONMENT VALIDATION REPORT")
        print("="*60)
        
        print(f"\nTimestamp: {report.timestamp}")
        print(f"Overall Status: {report.overall_status}")
        
        print(f"\nSystem Information:")
        print(f"  Platform: {report.system_info['platform']}")
        print(f"  Python: {report.system_info['python_version']} ({report.system_info['python_implementation']})")
        print(f"  Hostname: {report.system_info['hostname']}")
        
        print(f"\nEnvironment Checks:")
        for result in report.check_results:
            status = "✅ PASS" if result.passed else ("❌ FAIL" if result.requirement.required else "⚠️  WARN")
            print(f"  {status} {result.requirement.description}")
            if result.actual_value:
                print(f"       → {result.actual_value}")
            if result.error_message:
                print(f"       → Error: {result.error_message}")
        
        if report.critical_failures:
            print(f"\nCritical Failures:")
            for failure in report.critical_failures:
                print(f"  ❌ {failure}")
        
        if report.warnings:
            print(f"\nWarnings:")
            for warning in report.warnings:
                print(f"  ⚠️  {warning}")
        
        if report.recommendations:
            print(f"\nRecommendations:")
            for rec in report.recommendations:
                print(f"  💡 {rec}")
        
        print("\n" + "="*60)
    
    def save_validation_report(self, report: EnvironmentValidationReport, filepath: Optional[Path] = None) -> Path:
        """検証レポート保存"""
        if filepath is None:
            filepath = self.config.get_logs_dir() / f"environment_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # レポートをJSON形式で保存
        import json
        
        report_dict = {
            "timestamp": report.timestamp.isoformat(),
            "system_info": report.system_info,
            "overall_status": report.overall_status,
            "critical_failures": report.critical_failures,
            "warnings": report.warnings,
            "recommendations": report.recommendations,
            "check_results": [
                {
                    "requirement_name": result.requirement.name,
                    "requirement_description": result.requirement.description,
                    "required": result.requirement.required,
                    "passed": result.passed,
                    "actual_value": result.actual_value,
                    "error_message": result.error_message,
                    "details": result.details
                }
                for result in report.check_results
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Environment validation report saved to: {filepath}")
        return filepath


# 便利関数
def validate_environment_startup(config: Optional[AdvancedAgentConfig] = None) -> EnvironmentValidationReport:
    """起動時環境検証"""
    validator = EnvironmentValidator(config)
    report = validator.validate_environment()
    
    # コンソール出力
    validator.print_validation_report(report)
    
    # ファイル保存
    validator.save_validation_report(report)
    
    # 重大な問題がある場合は例外発生
    if report.overall_status == "FAIL":
        raise RuntimeError(
            f"Environment validation failed with {len(report.critical_failures)} critical failures. "
            "Please resolve these issues before starting the agent."
        )
    
    return report


def quick_environment_check() -> bool:
    """簡易環境チェック"""
    try:
        validator = EnvironmentValidator()
        report = validator.validate_environment()
        return report.overall_status in ["PASS", "WARNING"]
    except Exception as e:
        logger.error(f"Quick environment check failed: {e}")
        return False


# 使用例
if __name__ == "__main__":
    try:
        report = validate_environment_startup()
        print(f"\nEnvironment validation completed with status: {report.overall_status}")
        
        if report.overall_status == "PASS":
            print("✅ All checks passed! The system is ready to run.")
        elif report.overall_status == "WARNING":
            print("⚠️  Some warnings detected, but the system can run with reduced functionality.")
        else:
            print("❌ Critical failures detected. Please resolve these issues.")
            
    except Exception as e:
        print(f"❌ Environment validation failed: {e}")
        sys.exit(1)