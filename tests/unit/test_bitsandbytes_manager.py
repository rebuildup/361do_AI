"""
BitsAndBytes マネージャーのテスト
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.advanced_agent.quantization.bitsandbytes_manager import (
    BitsAndBytesManager, QuantizationConfig, QuantizationResult,
    QuantizationLevel, QuantizationStrategy, create_bitsandbytes_manager,
    get_quantization_recommendations
)
from src.advanced_agent.monitoring.system_monitor import SystemMonitor


class TestBitsAndBytesManager:
    """BitsAndBytesManager クラスのテスト"""
    
    @pytest.fixture
    def mock_system_monitor(self):
        """モックシステムモニター"""
        monitor = Mock(spec=SystemMonitor)
        monitor.get_gpu_memory_info.return_value = {
            "total_mb": 6144,
            "used_mb": 1024,
            "free_mb": 5120
        }
        return monitor
    
    @pytest.fixture
    def bnb_manager(self, mock_system_monitor):
        """BitsAndBytesManager インスタンス"""
        with patch('src.advanced_agent.quantization.bitsandbytes_manager.get_config') as mock_get_config, \
             patch('src.advanced_agent.quantization.bitsandbytes_manager.BITSANDBYTES_AVAILABLE', True):
            from src.advanced_agent.core.config import AdvancedAgentConfig
            mock_get_config.return_value = AdvancedAgentConfig()
            return BitsAndBytesManager(mock_system_monitor)
    
    def test_init(self, bnb_manager, mock_system_monitor):
        """初期化テスト"""
        assert bnb_manager.system_monitor == mock_system_monitor
        assert len(bnb_manager.quantization_history) == 0
        assert bnb_manager.current_config is None
        assert len(bnb_manager.default_configs) > 0
        assert bnb_manager.performance_stats["total_quantizations"] == 0
    
    def test_create_default_configs(self, bnb_manager):
        """デフォルト設定作成テスト"""
        configs = bnb_manager.default_configs
        
        # 必要な設定が存在することを確認
        expected_configs = [
            "int8_basic", "int8_cpu_offload", "int4_nf4", 
            "int4_fp4", "dynamic_adaptive", "memory_constrained"
        ]
        
        for config_name in expected_configs:
            assert config_name in configs
        
        # INT8設定の確認
        int8_config = configs["int8_basic"]
        assert int8_config.level == QuantizationLevel.INT8
        assert int8_config.load_in_8bit is True
        assert int8_config.load_in_4bit is False
        
        # INT4設定の確認
        int4_config = configs["int4_nf4"]
        assert int4_config.level == QuantizationLevel.NF4
        assert int4_config.load_in_4bit is True
        assert int4_config.bnb_4bit_quant_type == "nf4"
        
        # メモリ制約設定の確認
        memory_config = configs["memory_constrained"]
        assert memory_config.strategy == QuantizationStrategy.MEMORY_AWARE
        assert memory_config.auto_adjust is True
    
    @patch('src.advanced_agent.quantization.bitsandbytes_manager.BITSANDBYTES_AVAILABLE', True)
    @patch('src.advanced_agent.quantization.bitsandbytes_manager.torch')
    @patch('src.advanced_agent.quantization.bitsandbytes_manager.psutil')
    def test_check_availability(self, mock_psutil, mock_torch, bnb_manager):
        """利用可能性チェックテスト"""
        # モック設定
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value.total_memory = 6 * 1024**3  # 6GB
        mock_torch.__version__ = "2.0.0"
        mock_psutil.virtual_memory.return_value.total = 16 * 1024**3  # 16GB
        
        availability = bnb_manager.check_availability()
        
        assert availability["bitsandbytes_installed"] is True
        assert availability["cuda_available"] is True
        assert availability["torch_version"] == "2.0.0"
        assert availability["system_memory_gb"] == 16.0
        assert availability["gpu_memory_gb"] == 6.0
        assert len(availability["recommendations"]) > 0
    
    def test_get_optimal_config_low_memory(self, bnb_manager):
        """低メモリ環境での最適設定取得テスト"""
        config = bnb_manager.get_optimal_config(
            model_size_mb=7000,
            available_memory_mb=2048,  # 2GB
            target_memory_mb=1600     # 1.6GB
        )
        
        assert config.level == QuantizationLevel.NF4
        assert config.load_in_4bit is True
        assert config.memory_threshold_mb == 1600
    
    def test_get_optimal_config_medium_memory(self, bnb_manager):
        """中程度メモリ環境での最適設定取得テスト"""
        config = bnb_manager.get_optimal_config(
            model_size_mb=7000,
            available_memory_mb=6144,  # 6GB
            target_memory_mb=4096     # 4GB
        )
        
        assert config.level in [QuantizationLevel.INT8, QuantizationLevel.NF4]
        assert config.memory_threshold_mb == 4096
    
    def test_get_optimal_config_high_memory(self, bnb_manager):
        """高メモリ環境での最適設定取得テスト"""
        config = bnb_manager.get_optimal_config(
            model_size_mb=7000,
            available_memory_mb=12288,  # 12GB
            target_memory_mb=8192      # 8GB
        )
        
        assert config.strategy == QuantizationStrategy.ADAPTIVE
        assert config.auto_adjust is True
    
    @patch('src.advanced_agent.quantization.bitsandbytes_manager.BITSANDBYTES_AVAILABLE', True)
    @patch('src.advanced_agent.quantization.bitsandbytes_manager.BitsAndBytesConfig')
    def test_create_bnb_config_int8(self, mock_bnb_config_class, bnb_manager):
        """INT8 BitsAndBytesConfig作成テスト"""
        mock_bnb_config = Mock()
        mock_bnb_config_class.return_value = mock_bnb_config
        
        quant_config = QuantizationConfig(
            level=QuantizationLevel.INT8,
            strategy=QuantizationStrategy.STATIC,
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        
        result = bnb_manager.create_bnb_config(quant_config)
        
        assert result == mock_bnb_config
        mock_bnb_config_class.assert_called_once()
        
        # 呼び出し引数確認
        call_kwargs = mock_bnb_config_class.call_args[1]
        assert call_kwargs["load_in_8bit"] is True
        assert call_kwargs["llm_int8_threshold"] == 6.0
    
    @patch('src.advanced_agent.quantization.bitsandbytes_manager.BITSANDBYTES_AVAILABLE', True)
    @patch('src.advanced_agent.quantization.bitsandbytes_manager.BitsAndBytesConfig')
    @patch('src.advanced_agent.quantization.bitsandbytes_manager.torch')
    def test_create_bnb_config_int4(self, mock_torch, mock_bnb_config_class, bnb_manager):
        """INT4 BitsAndBytesConfig作成テスト"""
        mock_bnb_config = Mock()
        mock_bnb_config_class.return_value = mock_bnb_config
        mock_torch.float16 = "float16_mock"
        
        quant_config = QuantizationConfig(
            level=QuantizationLevel.NF4,
            strategy=QuantizationStrategy.STATIC,
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16"
        )
        
        result = bnb_manager.create_bnb_config(quant_config)
        
        assert result == mock_bnb_config
        
        # 呼び出し引数確認
        call_kwargs = mock_bnb_config_class.call_args[1]
        assert call_kwargs["load_in_4bit"] is True
        assert call_kwargs["bnb_4bit_use_double_quant"] is True
        assert call_kwargs["bnb_4bit_quant_type"] == "nf4"
        assert call_kwargs["bnb_4bit_compute_dtype"] == "float16_mock"
    
    @patch('src.advanced_agent.quantization.bitsandbytes_manager.BITSANDBYTES_AVAILABLE', False)
    def test_create_bnb_config_unavailable(self, bnb_manager):
        """BitsAndBytes利用不可時のテスト"""
        quant_config = QuantizationConfig(
            level=QuantizationLevel.INT8,
            strategy=QuantizationStrategy.STATIC,
            load_in_8bit=True
        )
        
        result = bnb_manager.create_bnb_config(quant_config)
        
        assert result is None
    
    @patch('src.advanced_agent.quantization.bitsandbytes_manager.torch')
    def test_get_memory_usage_cuda(self, mock_torch, bnb_manager):
        """CUDA環境でのメモリ使用量取得テスト"""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 2048 * 1024 * 1024  # 2GB in bytes
        
        memory_mb = bnb_manager._get_memory_usage()
        
        assert memory_mb == 2048.0
    
    @patch('src.advanced_agent.quantization.bitsandbytes_manager.torch')
    @patch('src.advanced_agent.quantization.bitsandbytes_manager.psutil')
    def test_get_memory_usage_cpu(self, mock_psutil, mock_torch, bnb_manager):
        """CPU環境でのメモリ使用量取得テスト"""
        mock_torch.cuda.is_available.return_value = False
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 1024 * 1024 * 1024  # 1GB in bytes
        mock_psutil.Process.return_value = mock_process
        
        memory_mb = bnb_manager._get_memory_usage()
        
        assert memory_mb == 1024.0
    
    def test_estimate_model_size(self, bnb_manager):
        """モデルサイズ推定テスト"""
        # モックモデル作成
        mock_model = Mock()
        mock_param1 = Mock()
        mock_param1.numel.return_value = 1000000  # 1M parameters
        mock_param2 = Mock()
        mock_param2.numel.return_value = 2000000  # 2M parameters
        
        mock_model.parameters.return_value = [mock_param1, mock_param2]
        
        size_mb = bnb_manager._estimate_model_size(mock_model)
        
        # 3M parameters * 4 bytes / (1024^2) = ~11.44 MB
        assert 11 <= size_mb <= 12
    
    def test_update_performance_stats(self, bnb_manager):
        """パフォーマンス統計更新テスト"""
        # 初期状態
        assert bnb_manager.performance_stats["successful_quantizations"] == 0
        assert bnb_manager.performance_stats["average_memory_reduction"] == 0.0
        
        # 最初の結果
        result1 = QuantizationResult(
            config_used=Mock(),
            original_memory_mb=1000.0,
            quantized_memory_mb=500.0,
            memory_reduction_ratio=0.5,
            quantization_time=2.0,
            model_size_mb=100.0,
            success=True
        )
        
        bnb_manager.performance_stats["successful_quantizations"] = 1
        bnb_manager._update_performance_stats(result1)
        
        assert bnb_manager.performance_stats["average_memory_reduction"] == 0.5
        assert bnb_manager.performance_stats["average_quantization_time"] == 2.0
        
        # 2番目の結果
        result2 = QuantizationResult(
            config_used=Mock(),
            original_memory_mb=1000.0,
            quantized_memory_mb=300.0,
            memory_reduction_ratio=0.7,
            quantization_time=3.0,
            model_size_mb=100.0,
            success=True
        )
        
        bnb_manager.performance_stats["successful_quantizations"] = 2
        bnb_manager._update_performance_stats(result2)
        
        # 平均値確認
        assert bnb_manager.performance_stats["average_memory_reduction"] == 0.6  # (0.5 + 0.7) / 2
        assert bnb_manager.performance_stats["average_quantization_time"] == 2.5  # (2.0 + 3.0) / 2
    
    def test_get_performance_stats(self, bnb_manager):
        """パフォーマンス統計取得テスト"""
        # テストデータ設定
        bnb_manager.performance_stats.update({
            "total_quantizations": 10,
            "successful_quantizations": 8,
            "failed_quantizations": 2,
            "average_memory_reduction": 0.6,
            "average_quantization_time": 2.5
        })
        
        # 履歴データ追加
        recent_results = [
            QuantizationResult(
                config_used=Mock(),
                original_memory_mb=1000.0,
                quantized_memory_mb=400.0,
                memory_reduction_ratio=0.6,
                quantization_time=2.0,
                model_size_mb=100.0,
                success=True
            ),
            QuantizationResult(
                config_used=Mock(),
                original_memory_mb=1000.0,
                quantized_memory_mb=300.0,
                memory_reduction_ratio=0.7,
                quantization_time=3.0,
                model_size_mb=100.0,
                success=True
            )
        ]
        bnb_manager.quantization_history.extend(recent_results)
        
        stats = bnb_manager.get_performance_stats()
        
        assert stats["total_quantizations"] == 10
        assert stats["success_rate"] == 0.8  # 8/10
        assert stats["average_memory_reduction"] == 0.6
        assert stats["average_quantization_time"] == 2.5
        assert "recent_average_reduction" in stats
        assert "recent_average_time" in stats
    
    def test_get_quantization_history(self, bnb_manager):
        """量子化履歴取得テスト"""
        # テスト履歴作成
        results = [
            QuantizationResult(
                config_used=Mock(),
                original_memory_mb=1000.0,
                quantized_memory_mb=500.0,
                memory_reduction_ratio=0.5,
                quantization_time=2.0,
                model_size_mb=100.0,
                success=True
            ),
            QuantizationResult(
                config_used=Mock(),
                original_memory_mb=1000.0,
                quantized_memory_mb=600.0,
                memory_reduction_ratio=0.4,
                quantization_time=1.5,
                model_size_mb=100.0,
                success=False,
                error_message="Test error"
            ),
            QuantizationResult(
                config_used=Mock(),
                original_memory_mb=1000.0,
                quantized_memory_mb=300.0,
                memory_reduction_ratio=0.7,
                quantization_time=3.0,
                model_size_mb=100.0,
                success=True
            )
        ]
        
        bnb_manager.quantization_history.extend(results)
        
        # 全履歴取得
        all_history = bnb_manager.get_quantization_history()
        assert len(all_history) == 3
        
        # 件数制限
        limited_history = bnb_manager.get_quantization_history(limit=2)
        assert len(limited_history) == 2
        
        # 成功のみ
        success_history = bnb_manager.get_quantization_history(success_only=True)
        assert len(success_history) == 2
        assert all(r.success for r in success_history)
    
    def test_get_config_recommendations(self, bnb_manager):
        """設定推奨事項取得テスト"""
        with patch.object(bnb_manager, 'check_availability') as mock_check:
            mock_check.return_value = {
                "bitsandbytes_installed": True,
                "cuda_available": True,
                "gpu_memory_gb": 6.0
            }
            
            # 低メモリ環境
            recommendations = bnb_manager.get_config_recommendations(target_memory_mb=1500)
            
            assert len(recommendations) > 0
            high_priority = [r for r in recommendations if r["priority"] == "high"]
            assert len(high_priority) > 0
            
            # BitsAndBytes未インストール環境
            mock_check.return_value["bitsandbytes_installed"] = False
            recommendations = bnb_manager.get_config_recommendations()
            
            installation_recs = [r for r in recommendations if r["type"] == "installation"]
            assert len(installation_recs) > 0


class TestQuantizationDataClasses:
    """量子化データクラスのテスト"""
    
    def test_quantization_config(self):
        """QuantizationConfig テスト"""
        config = QuantizationConfig(
            level=QuantizationLevel.INT8,
            strategy=QuantizationStrategy.DYNAMIC,
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            memory_threshold_mb=4096,
            auto_adjust=True
        )
        
        assert config.level == QuantizationLevel.INT8
        assert config.strategy == QuantizationStrategy.DYNAMIC
        assert config.load_in_8bit is True
        assert config.llm_int8_threshold == 6.0
        assert config.memory_threshold_mb == 4096
        assert config.auto_adjust is True
    
    def test_quantization_result(self):
        """QuantizationResult テスト"""
        config = QuantizationConfig(
            level=QuantizationLevel.NF4,
            strategy=QuantizationStrategy.STATIC
        )
        
        result = QuantizationResult(
            config_used=config,
            original_memory_mb=1000.0,
            quantized_memory_mb=400.0,
            memory_reduction_ratio=0.6,
            quantization_time=2.5,
            model_size_mb=150.0,
            performance_impact=0.1,
            quality_score=0.9,
            success=True
        )
        
        assert result.config_used == config
        assert result.original_memory_mb == 1000.0
        assert result.quantized_memory_mb == 400.0
        assert result.memory_reduction_ratio == 0.6
        assert result.quantization_time == 2.5
        assert result.model_size_mb == 150.0
        assert result.performance_impact == 0.1
        assert result.quality_score == 0.9
        assert result.success is True
        assert isinstance(result.timestamp, datetime)


class TestUtilityFunctions:
    """ユーティリティ関数のテスト"""
    
    @pytest.mark.asyncio
    async def test_create_bitsandbytes_manager(self):
        """BitsAndBytes マネージャー作成テスト"""
        with patch('src.advanced_agent.quantization.bitsandbytes_manager.BitsAndBytesManager') as MockManager:
            mock_manager = Mock()
            MockManager.return_value = mock_manager
            
            result = await create_bitsandbytes_manager()
            
            assert result == mock_manager
            MockManager.assert_called_once_with(None)
    
    def test_get_quantization_recommendations_low_memory(self):
        """低メモリ環境での推奨事項テスト"""
        recommendations = get_quantization_recommendations(available_memory_gb=1.5)
        
        assert recommendations["recommended_config"] == "memory_constrained"
        assert "aggressive" in recommendations["reasoning"].lower()
        assert len(recommendations["warnings"]) > 0
    
    def test_get_quantization_recommendations_medium_memory(self):
        """中程度メモリ環境での推奨事項テスト"""
        recommendations = get_quantization_recommendations(available_memory_gb=3.5)
        
        assert recommendations["recommended_config"] == "int4_nf4"
        assert "4-bit" in recommendations["reasoning"]
        assert len(recommendations["alternatives"]) > 0
    
    def test_get_quantization_recommendations_high_memory(self):
        """高メモリ環境での推奨事項テスト"""
        recommendations = get_quantization_recommendations(available_memory_gb=8.0)
        
        assert recommendations["recommended_config"] == "dynamic_adaptive"
        assert "adaptive" in recommendations["reasoning"].lower()
        assert len(recommendations["alternatives"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])