"""
環境検証システムのテスト
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.advanced_agent.core.environment import (
    EnvironmentValidator, EnvironmentValidationReport, EnvironmentCheckResult,
    SystemRequirement, validate_environment_startup, quick_environment_check
)
from src.advanced_agent.core.config import AdvancedAgentConfig


class TestEnvironmentValidator:
    """EnvironmentValidator クラスのテスト"""
    
    @pytest.fixture
    def mock_config(self):
        """モック設定"""
        config = AdvancedAgentConfig()
        config.models.ollama_base_url = "http://localhost:11434"
        config.models.primary = "deepseek-r1:7b"
        config.models.fallback = "qwen2.5:7b-instruct-q4_k_m"
        return config
    
    @pytest.fixture
    def validator(self, mock_config):
        """EnvironmentValidator インスタンス"""
        return EnvironmentValidator(mock_config)
    
    def test_init(self, validator):
        """初期化テスト"""
        assert validator.config is not None
        assert len(validator.requirements) > 0
        assert all(isinstance(req, SystemRequirement) for req in validator.requirements)
    
    def test_check_python_version(self, validator):
        """Python バージョンチェックテスト"""
        passed, version_str, details = validator.check_python_version()
        
        assert isinstance(passed, bool)
        assert isinstance(version_str, str)
        assert isinstance(details, dict)
        assert "current_version" in details
        assert "required_version" in details
        assert "implementation" in details
    
    @patch('src.advanced_agent.core.environment.psutil')
    def test_check_system_memory(self, mock_psutil, validator):
        """システムメモリチェックテスト"""
        # 32GB メモリをシミュレート
        mock_memory = Mock()
        mock_memory.total = 32 * 1024**3
        mock_memory.available = 16 * 1024**3
        mock_memory.percent = 50.0
        mock_psutil.virtual_memory.return_value = mock_memory
        
        passed, memory_str, details = validator.check_system_memory()
        
        assert passed is True
        assert "32.0GB" in memory_str
        assert details["total_gb"] == 32.0
        assert details["required_gb"] == 16.0
    
    @patch('src.advanced_agent.core.environment.psutil')
    def test_check_system_memory_insufficient(self, mock_psutil, validator):
        """不十分なシステムメモリテスト"""
        # 8GB メモリをシミュレート
        mock_memory = Mock()
        mock_memory.total = 8 * 1024**3
        mock_memory.available = 4 * 1024**3
        mock_memory.percent = 50.0
        mock_psutil.virtual_memory.return_value = mock_memory
        
        passed, memory_str, details = validator.check_system_memory()
        
        assert passed is False
        assert "8.0GB" in memory_str
        assert details["total_gb"] == 8.0
    
    def test_check_gpu_availability_success(self, validator):
        with patch('src.advanced_agent.core.environment.pynvml') as mock_pynvml:
            """GPU利用可能性チェック成功テスト"""
            mock_pynvml.nvmlInit.return_value = None
            mock_pynvml.nvmlDeviceGetCount.return_value = 1
            mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "mock_handle"
            mock_pynvml.nvmlDeviceGetName.return_value = b"NVIDIA GeForce RTX 4050 Laptop GPU"
            mock_pynvml.nvmlSystemGetDriverVersion.return_value = b"531.79"
            
            mock_memory_info = Mock()
            mock_memory_info.total = 6 * 1024**3
            mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_memory_info
            
            passed, gpu_str, details = validator.check_gpu_availability()
            
            assert passed is True
            assert "1 GPU(s) detected" in gpu_str
            assert details["gpu_count"] == 1
            assert "RTX 4050" in details["primary_gpu"]
            assert details["total_memory_gb"] == 6.0
    
    def test_check_gpu_availability_no_gpu(self, validator):
        with patch('src.advanced_agent.core.environment.pynvml') as mock_pynvml:
            """GPU未検出テスト"""
            mock_pynvml.nvmlInit.return_value = None
            mock_pynvml.nvmlDeviceGetCount.return_value = 0
            
            passed, gpu_str, details = validator.check_gpu_availability()
            
            assert passed is False
            assert "No GPU detected" in gpu_str
            assert details["gpu_count"] == 0
    
    def test_check_cuda_availability_success(self, validator):
        with patch('src.advanced_agent.core.environment.torch') as mock_torch:
            """CUDA利用可能性チェック成功テスト"""
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.device_count.return_value = 1
            mock_torch.cuda.current_device.return_value = 0
            mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 4050 Laptop GPU"
            mock_torch.version.cuda = "12.1"
            mock_torch.__version__ = "2.1.0"
            
            passed, cuda_str, details = validator.check_cuda_availability()
            
            assert passed is True
            assert "CUDA 12.1 available" in cuda_str
            assert details["cuda_available"] is True
            assert details["device_count"] == 1
    
    def test_check_cuda_availability_not_available(self, validator):
        with patch('src.advanced_agent.core.environment.torch') as mock_torch:
            """CUDA利用不可テスト"""
            mock_torch.cuda.is_available.return_value = False
            mock_torch.__version__ = "2.1.0"
            
            passed, cuda_str, details = validator.check_cuda_availability()
            
            assert passed is False
            assert "CUDA not available" in cuda_str
            assert details["cuda_available"] is False
    
    def test_check_required_packages(self, validator):
        """必要パッケージチェックテスト"""
        passed, packages_str, details = validator.check_required_packages()
        
        assert isinstance(passed, bool)
        assert isinstance(packages_str, str)
        assert "required_packages" in details
        assert "installed_packages" in details
        assert "missing_packages" in details
        
        # 少なくともいくつかのパッケージはインストールされているはず
        assert len(details["installed_packages"]) > 0
    
    def test_check_ollama_connection_success(self, validator):
        with patch('src.advanced_agent.core.environment.httpx') as mock_httpx:
            """Ollama接続チェック成功テスト"""
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "models": [
                    {"name": "deepseek-r1:7b"},
                    {"name": "qwen2.5:7b-instruct-q4_k_m"}
                ]
            }
            mock_httpx.get.return_value = mock_response
            
            passed, connection_str, details = validator.check_ollama_connection()
            
            assert passed is True
            assert "Connected" in connection_str
            assert details["primary_available"] is True
            assert details["fallback_available"] is True
    
    def test_check_ollama_connection_failure(self, validator):
        with patch('src.advanced_agent.core.environment.httpx') as mock_httpx:
            """Ollama接続チェック失敗テスト"""
            mock_httpx.get.side_effect = Exception("Connection refused")
            
            passed, connection_str, details = validator.check_ollama_connection()
            
            assert passed is False
            assert "Connection failed" in connection_str
            assert "error" in details
    
    @patch('src.advanced_agent.core.environment.psutil')
    def test_check_disk_space(self, mock_psutil, validator):
        """ディスク容量チェックテスト"""
        mock_disk = Mock()
        mock_disk.free = 50 * 1024**3  # 50GB
        mock_disk.total = 100 * 1024**3  # 100GB
        mock_disk.used = 50 * 1024**3  # 50GB
        mock_psutil.disk_usage.return_value = mock_disk
        
        passed, disk_str, details = validator.check_disk_space()
        
        assert passed is True
        assert "50.0GB free" in disk_str
        assert details["free_gb"] == 50.0
        assert details["required_gb"] == 10.0
    
    def test_check_write_permissions(self, validator):
        """書き込み権限チェックテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 一時的に設定を変更
            original_config = validator.config
            temp_config = AdvancedAgentConfig()
            temp_config.database.sqlite_path = f"{temp_dir}/test.db"
            temp_config.database.chroma_path = f"{temp_dir}/chroma"
            validator.config = temp_config
            
            try:
                passed, write_str, details = validator.check_write_permissions()
                
                assert isinstance(passed, bool)
                assert isinstance(write_str, str)
                assert "test_directories" in details
                assert "write_results" in details
                
            finally:
                validator.config = original_config
    
    def test_run_single_check(self, validator):
        """単一チェック実行テスト"""
        requirement = SystemRequirement(
            name="test_check",
            description="Test check",
            check_function="check_python_version"
        )
        
        result = validator.run_single_check(requirement)
        
        assert isinstance(result, EnvironmentCheckResult)
        assert result.requirement == requirement
        assert isinstance(result.passed, bool)
        assert result.actual_value is not None
    
    def test_validate_environment(self, validator):
        """環境検証実行テスト"""
        report = validator.validate_environment()
        
        assert isinstance(report, EnvironmentValidationReport)
        assert isinstance(report.timestamp, datetime)
        assert isinstance(report.system_info, dict)
        assert len(report.check_results) > 0
        assert report.overall_status in ["PASS", "WARNING", "FAIL"]
        
        # システム情報の基本項目チェック
        assert "platform" in report.system_info
        assert "python_version" in report.system_info
        assert "hostname" in report.system_info
    
    def test_print_validation_report(self, validator, capsys):
        """検証レポート出力テスト"""
        report = validator.validate_environment()
        validator.print_validation_report(report)
        
        captured = capsys.readouterr()
        assert "ENVIRONMENT VALIDATION REPORT" in captured.out
        assert report.overall_status in captured.out
    
    def test_save_validation_report(self, validator):
        """検証レポート保存テスト"""
        report = validator.validate_environment()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "test_report.json"
            saved_path = validator.save_validation_report(report, filepath)
            
            assert saved_path.exists()
            
            # JSON形式で保存されていることを確認
            with open(saved_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert "timestamp" in data
            assert "system_info" in data
            assert "overall_status" in data
            assert "check_results" in data


class TestEnvironmentValidationFunctions:
    """環境検証関数のテスト"""
    
    @patch('src.advanced_agent.core.environment.EnvironmentValidator')
    def test_validate_environment_startup_pass(self, mock_validator_class):
        """起動時環境検証成功テスト"""
        mock_validator = Mock()
        mock_report = Mock()
        mock_report.overall_status = "PASS"
        mock_validator.validate_environment.return_value = mock_report
        mock_validator_class.return_value = mock_validator
        
        result = validate_environment_startup()
        
        assert result == mock_report
        mock_validator.print_validation_report.assert_called_once()
        mock_validator.save_validation_report.assert_called_once()
    
    @patch('src.advanced_agent.core.environment.EnvironmentValidator')
    def test_validate_environment_startup_fail(self, mock_validator_class):
        """起動時環境検証失敗テスト"""
        mock_validator = Mock()
        mock_report = Mock()
        mock_report.overall_status = "FAIL"
        mock_report.critical_failures = ["Critical error"]
        mock_validator.validate_environment.return_value = mock_report
        mock_validator_class.return_value = mock_validator
        
        with pytest.raises(RuntimeError, match="Environment validation failed"):
            validate_environment_startup()
    
    @patch('src.advanced_agent.core.environment.EnvironmentValidator')
    def test_quick_environment_check_success(self, mock_validator_class):
        """簡易環境チェック成功テスト"""
        mock_validator = Mock()
        mock_report = Mock()
        mock_report.overall_status = "PASS"
        mock_validator.validate_environment.return_value = mock_report
        mock_validator_class.return_value = mock_validator
        
        result = quick_environment_check()
        
        assert result is True
    
    @patch('src.advanced_agent.core.environment.EnvironmentValidator')
    def test_quick_environment_check_failure(self, mock_validator_class):
        """簡易環境チェック失敗テスト"""
        mock_validator = Mock()
        mock_report = Mock()
        mock_report.overall_status = "FAIL"
        mock_validator.validate_environment.return_value = mock_report
        mock_validator_class.return_value = mock_validator
        
        result = quick_environment_check()
        
        assert result is False
    
    @patch('src.advanced_agent.core.environment.EnvironmentValidator')
    def test_quick_environment_check_exception(self, mock_validator_class):
        """簡易環境チェック例外テスト"""
        mock_validator_class.side_effect = Exception("Test error")
        
        result = quick_environment_check()
        
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])