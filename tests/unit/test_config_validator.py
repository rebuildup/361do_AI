"""
設定検証システムのテスト
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.advanced_agent.core.config_validator import (
    ConfigValidator, ConfigValidationReport, ConfigValidationIssue,
    validate_and_optimize_config, generate_system_optimized_config
)
from src.advanced_agent.core.config import AdvancedAgentConfig


class TestConfigValidator:
    """ConfigValidator クラスのテスト"""
    
    @pytest.fixture
    def mock_system_info(self):
        """モックシステム情報"""
        return {
            "cpu_count": 16,
            "cpu_freq_mhz": 3200,
            "memory_gb": 32.0,
            "gpu_info": {
                "available": True,
                "device_count": 1,
                "device_name": "NVIDIA GeForce RTX 4050 Laptop GPU",
                "memory_gb": 6.0
            }
        }
    
    @pytest.fixture
    def validator(self, mock_system_info):
        """ConfigValidator インスタンス"""
        with patch.object(ConfigValidator, '_get_system_info', return_value=mock_system_info):
            return ConfigValidator()
    
    @pytest.fixture
    def valid_config_data(self):
        """有効な設定データ"""
        return {
            "project_name": "Test Agent",
            "version": "1.0.0",
            "environment": "development",
            "gpu": {
                "max_vram_gb": 5.0,
                "quantization_levels": [8, 4, 3],
                "temperature_threshold": 80
            },
            "cpu": {
                "max_threads": 16,
                "offload_threshold": 0.8
            },
            "memory": {
                "system_ram_gb": 32.0,
                "cache_size_mb": 1024
            },
            "models": {
                "primary": "deepseek-r1:7b",
                "fallback": "qwen2.5:7b-instruct-q4_k_m",
                "ollama_base_url": "http://localhost:11434",
                "context_length": 4096
            },
            "learning": {
                "batch_size": 2,
                "learning_rate": 0.0001
            }
        }
    
    def test_init(self, validator):
        """初期化テスト"""
        assert validator.system_info is not None
        assert "cpu_count" in validator.system_info
        assert "memory_gb" in validator.system_info
        assert "gpu_info" in validator.system_info
    
    def test_validate_config_file_not_found(self, validator):
        """設定ファイル未存在テスト"""
        report = validator.validate_config_file("nonexistent.yaml")
        
        assert not report.validation_passed
        assert len(report.issues) > 0
        assert any("not found" in issue.message for issue in report.issues)
    
    def test_validate_config_file_invalid_yaml(self, validator):
        """無効なYAMLテスト"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name
        
        try:
            report = validator.validate_config_file(temp_path)
            
            assert not report.validation_passed
            assert len(report.issues) > 0
            assert any("YAML" in issue.message for issue in report.issues)
        finally:
            Path(temp_path).unlink()
    
    def test_validate_config_file_valid(self, validator, valid_config_data):
        """有効な設定ファイルテスト"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(valid_config_data, f)
            temp_path = f.name
        
        try:
            report = validator.validate_config_file(temp_path)
            
            assert report.validation_passed
            assert report.config_file_path == temp_path
        finally:
            Path(temp_path).unlink()
    
    def test_validate_gpu_config_rtx4050_optimization(self, validator):
        """RTX 4050 GPU設定最適化テスト"""
        from src.advanced_agent.core.config import GPUConfig
        
        gpu_config = GPUConfig(max_vram_gb=6.0, quantization_levels=[4, 3])
        report = ConfigValidationReport(
            timestamp=validator._get_system_info(),
            config_file_path="test.yaml",
            validation_passed=True
        )
        
        validator._validate_gpu_config(gpu_config, report)
        
        # RTX 4050 最適化提案があることを確認
        assert len(report.optimizations) > 0
        assert any("RTX 4050" in opt.message for opt in report.optimizations)
    
    def test_validate_gpu_config_vram_exceeded(self, validator):
        """VRAM制限超過テスト"""
        from src.advanced_agent.core.config import GPUConfig
        
        gpu_config = GPUConfig(max_vram_gb=8.0)  # システムVRAM(6GB)を超過
        report = ConfigValidationReport(
            timestamp=validator._get_system_info(),
            config_file_path="test.yaml",
            validation_passed=True
        )
        
        validator._validate_gpu_config(gpu_config, report)
        
        # VRAM超過警告があることを確認
        assert len(report.issues) > 0
        assert any("exceeds system VRAM" in issue.message for issue in report.issues)
    
    def test_validate_cpu_config_threads_exceeded(self, validator):
        """CPU スレッド数超過テスト"""
        from src.advanced_agent.core.config import CPUConfig
        
        cpu_config = CPUConfig(max_threads=32)  # システムCPU(16)を超過
        report = ConfigValidationReport(
            timestamp=validator._get_system_info(),
            config_file_path="test.yaml",
            validation_passed=True
        )
        
        validator._validate_cpu_config(cpu_config, report)
        
        # スレッド数超過警告があることを確認
        assert len(report.issues) > 0
        assert any("exceeds system CPU count" in issue.message for issue in report.issues)
    
    def test_validate_memory_config_ram_exceeded(self, validator):
        """RAM設定超過テスト"""
        from src.advanced_agent.core.config import MemoryConfig
        
        memory_config = MemoryConfig(system_ram_gb=64.0)  # システムRAM(32GB)を超過
        report = ConfigValidationReport(
            timestamp=validator._get_system_info(),
            config_file_path="test.yaml",
            validation_passed=True
        )
        
        validator._validate_memory_config(memory_config, report)
        
        # RAM超過警告があることを確認
        assert len(report.issues) > 0
        assert any("exceeds system RAM" in issue.message for issue in report.issues)
    
    def test_validate_model_config_invalid_url(self, validator):
        """無効なモデルURL テスト"""
        from src.advanced_agent.core.config import ModelConfig
        
        model_config = ModelConfig(ollama_base_url="invalid-url")
        report = ConfigValidationReport(
            timestamp=validator._get_system_info(),
            config_file_path="test.yaml",
            validation_passed=True
        )
        
        validator._validate_model_config(model_config, report)
        
        # URL形式エラーがあることを確認
        assert len(report.issues) > 0
        assert any("Invalid Ollama URL" in issue.message for issue in report.issues)
    
    def test_validate_learning_config_high_learning_rate(self, validator):
        """高学習率警告テスト"""
        from src.advanced_agent.core.config import LearningConfig
        
        learning_config = LearningConfig(learning_rate=0.01)  # 高い学習率
        report = ConfigValidationReport(
            timestamp=validator._get_system_info(),
            config_file_path="test.yaml",
            validation_passed=True
        )
        
        validator._validate_learning_config(learning_config, report)
        
        # 高学習率警告があることを確認
        assert len(report.issues) > 0
        assert any("Learning rate seems high" in issue.message for issue in report.issues)
    
    def test_apply_auto_fixes(self, validator, valid_config_data):
        """自動修正適用テスト"""
        # 修正が必要な設定を作成
        config_data = valid_config_data.copy()
        config_data["gpu"]["max_vram_gb"] = 8.0  # VRAM超過
        config_data["cpu"]["max_threads"] = 32   # スレッド数超過
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            # 検証実行
            report = validator.validate_config_file(temp_path)
            
            # 自動修正適用
            success, fixes = validator.apply_auto_fixes(temp_path, report)
            
            if success and fixes:
                assert len(fixes) > 0
                
                # 修正後の設定ファイル確認
                with open(temp_path, 'r', encoding='utf-8') as f:
                    fixed_data = yaml.safe_load(f)
                
                # VRAM制限が修正されていることを確認
                assert fixed_data["gpu"]["max_vram_gb"] <= 6.0
                
        finally:
            Path(temp_path).unlink()
            # バックアップファイルも削除
            for backup_file in Path(temp_path).parent.glob(f"{Path(temp_path).name}.backup_*"):
                backup_file.unlink()
    
    def test_generate_optimized_config(self, validator):
        """最適化設定生成テスト"""
        optimized_config = validator.generate_optimized_config()
        
        assert isinstance(optimized_config, AdvancedAgentConfig)
        
        # RTX 4050 最適化が適用されていることを確認
        assert optimized_config.gpu.max_vram_gb == 5.0
        assert optimized_config.learning.batch_size == 2
        assert optimized_config.cpu.max_threads == 16
        assert optimized_config.memory.system_ram_gb == 32.0
    
    def test_set_nested_value(self, validator):
        """ネスト値設定テスト"""
        data = {"level1": {"level2": {"level3": "old_value"}}}
        
        validator._set_nested_value(data, "level1.level2.level3", "new_value")
        
        assert data["level1"]["level2"]["level3"] == "new_value"
        
        # 新しいキーの作成テスト
        validator._set_nested_value(data, "new_level1.new_level2", "test_value")
        
        assert data["new_level1"]["new_level2"] == "test_value"
    
    def test_print_validation_report(self, validator, capsys):
        """検証レポート出力テスト"""
        report = ConfigValidationReport(
            timestamp=validator._get_system_info(),
            config_file_path="test.yaml",
            validation_passed=True,
            issues=[
                ConfigValidationIssue(
                    severity="ERROR",
                    field_path="test.field",
                    message="Test error message",
                    current_value="test_value"
                )
            ],
            optimizations=[
                ConfigValidationIssue(
                    severity="INFO",
                    field_path="test.optimization",
                    message="Test optimization message",
                    current_value="old_value",
                    suggested_value="new_value"
                )
            ]
        )
        
        validator.print_validation_report(report)
        
        captured = capsys.readouterr()
        assert "CONFIGURATION VALIDATION REPORT" in captured.out
        assert "Test error message" in captured.out
        assert "Test optimization message" in captured.out


class TestConfigValidationFunctions:
    """設定検証関数のテスト"""
    
    @patch('src.advanced_agent.core.config_validator.ConfigValidator')
    def test_validate_and_optimize_config(self, mock_validator_class, valid_config_data):
        """設定検証・最適化関数テスト"""
        mock_validator = Mock()
        mock_report = Mock()
        mock_report.validation_passed = True
        mock_validator.validate_config_file.return_value = mock_report
        mock_validator.apply_auto_fixes.return_value = (True, ["fix1", "fix2"])
        mock_validator_class.return_value = mock_validator
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(valid_config_data, f)
            temp_path = f.name
        
        try:
            report = validate_and_optimize_config(temp_path, apply_fixes=True)
            
            assert report == mock_report
            mock_validator.validate_config_file.assert_called_once_with(temp_path)
            mock_validator.print_validation_report.assert_called_once()
            mock_validator.apply_auto_fixes.assert_called_once()
            
        finally:
            Path(temp_path).unlink()
    
    @patch('src.advanced_agent.core.config_validator.ConfigValidator')
    def test_generate_system_optimized_config(self, mock_validator_class):
        """システム最適化設定生成関数テスト"""
        mock_validator = Mock()
        mock_config = Mock()
        mock_validator.generate_optimized_config.return_value = mock_config
        mock_validator_class.return_value = mock_validator
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = f"{temp_dir}/test_config.yaml"
            
            result = generate_system_optimized_config(output_path)
            
            assert result == mock_config
            mock_config.save_to_yaml.assert_called_once_with(output_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])