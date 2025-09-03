"""
PEFT Manager のテスト
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from src.advanced_agent.adaptation.peft_manager import (
    PEFTAdapterPool, AdapterConfig, AdapterInfo, AdapterPoolStats,
    AdapterType, AdapterStatus, create_peft_adapter_pool
)
from src.advanced_agent.monitoring.system_monitor import SystemMonitor


class TestPEFTAdapterPool:
    """PEFTAdapterPool クラスのテスト"""
    
    @pytest.fixture
    def temp_pool_dir(self):
        """一時プールディレクトリ"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_system_monitor(self):
        """モックシステムモニター"""
        monitor = Mock(spec=SystemMonitor)
        return monitor
    
    @pytest.fixture
    def peft_pool(self, temp_pool_dir, mock_system_monitor):
        """PEFTAdapterPool インスタンス"""
        with patch('src.advanced_agent.adaptation.peft_manager.get_config') as mock_get_config, \
             patch('src.advanced_agent.adaptation.peft_manager.PEFT_AVAILABLE', True):
            from src.advanced_agent.core.config import AdvancedAgentConfig
            mock_get_config.return_value = AdvancedAgentConfig()
            
            return PEFTAdapterPool(
                base_model_name="test/model",
                pool_directory=temp_pool_dir,
                system_monitor=mock_system_monitor
            )
    
    def test_init(self, peft_pool, temp_pool_dir, mock_system_monitor):
        """初期化テスト"""
        assert peft_pool.base_model_name == "test/model"
        assert str(peft_pool.pool_directory) == temp_pool_dir
        assert peft_pool.system_monitor == mock_system_monitor
        assert len(peft_pool.adapters) == 0
        assert len(peft_pool.active_adapters) == 0
        assert peft_pool.max_active_adapters == 3
        assert peft_pool.memory_threshold_mb == 2048
        assert peft_pool.auto_cleanup is True
    
    def test_create_adapter_config_lora(self, peft_pool):
        """LoRAアダプタ設定作成テスト"""
        config = peft_pool.create_adapter_config(
            name="test_lora",
            adapter_type=AdapterType.LORA,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            description="Test LoRA adapter"
        )
        
        assert config.name == "test_lora"
        assert config.adapter_type == AdapterType.LORA
        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.1
        assert config.description == "Test LoRA adapter"
        assert config.target_modules is not None  # デフォルト設定
    
    def test_create_adapter_config_adalora(self, peft_pool):
        """AdaLoRAアダプタ設定作成テスト"""
        config = peft_pool.create_adapter_config(
            name="test_adalora",
            adapter_type=AdapterType.ADALORA,
            r=16,
            target_r=8,
            init_r=12,
            tinit=0,
            tfinal=1000
        )
        
        assert config.name == "test_adalora"
        assert config.adapter_type == AdapterType.ADALORA
        assert config.target_r == 8
        assert config.init_r == 12
        assert config.tinit == 0
        assert config.tfinal == 1000
    
    @patch('src.advanced_agent.adaptation.peft_manager.PEFT_AVAILABLE', True)
    @patch('src.advanced_agent.adaptation.peft_manager.LoraConfig')
    def test_create_peft_config_lora(self, mock_lora_config, peft_pool):
        """LoRA PEFT設定作成テスト"""
        mock_config = Mock()
        mock_lora_config.return_value = mock_config
        
        adapter_config = AdapterConfig(
            name="test",
            adapter_type=AdapterType.LORA,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]
        )
        
        result = peft_pool.create_peft_config(adapter_config)
        
        assert result == mock_config
        mock_lora_config.assert_called_once()
        
        # 呼び出し引数確認
        call_kwargs = mock_lora_config.call_args[1]
        assert call_kwargs["r"] == 16
        assert call_kwargs["lora_alpha"] == 32
        assert call_kwargs["lora_dropout"] == 0.1
        assert call_kwargs["target_modules"] == ["q_proj", "v_proj"]
    
    @patch('src.advanced_agent.adaptation.peft_manager.PEFT_AVAILABLE', True)
    @patch('src.advanced_agent.adaptation.peft_manager.AdaLoraConfig')
    def test_create_peft_config_adalora(self, mock_adalora_config, peft_pool):
        """AdaLoRA PEFT設定作成テスト"""
        mock_config = Mock()
        mock_adalora_config.return_value = mock_config
        
        adapter_config = AdapterConfig(
            name="test",
            adapter_type=AdapterType.ADALORA,
            r=16,
            target_r=8,
            init_r=12
        )
        
        result = peft_pool.create_peft_config(adapter_config)
        
        assert result == mock_config
        mock_adalora_config.assert_called_once()
        
        # 呼び出し引数確認
        call_kwargs = mock_adalora_config.call_args[1]
        assert call_kwargs["r"] == 16
        assert call_kwargs["target_r"] == 8
        assert call_kwargs["init_r"] == 12
    
    @patch('src.advanced_agent.adaptation.peft_manager.PEFT_AVAILABLE', False)
    def test_create_peft_config_unavailable(self, peft_pool):
        """PEFT利用不可時のテスト"""
        adapter_config = AdapterConfig(
            name="test",
            adapter_type=AdapterType.LORA
        )
        
        with pytest.raises(RuntimeError, match="PEFT not available"):
            peft_pool.create_peft_config(adapter_config)
    
    def test_estimate_model_memory(self, peft_pool):
        """モデルメモリ推定テスト"""
        # モックモデル作成
        mock_model = Mock()
        mock_param1 = Mock()
        mock_param1.numel.return_value = 1000000  # 1M parameters
        mock_param2 = Mock()
        mock_param2.numel.return_value = 2000000  # 2M parameters
        
        mock_model.parameters.return_value = [mock_param1, mock_param2]
        
        memory_mb = peft_pool._estimate_model_memory(mock_model)
        
        # 3M parameters * 2 bytes / (1024^2) = ~5.72 MB
        assert 5 <= memory_mb <= 6
    
    def test_update_pool_stats(self, peft_pool):
        """プール統計更新テスト"""
        # テストアダプタ追加
        adapter_info1 = AdapterInfo(
            config=AdapterConfig(name="adapter1", adapter_type=AdapterType.LORA),
            status=AdapterStatus.ACTIVE,
            parameter_count=1000000,
            memory_usage_mb=10.0,
            performance_score=0.8,
            usage_count=5
        )
        
        adapter_info2 = AdapterInfo(
            config=AdapterConfig(name="adapter2", adapter_type=AdapterType.LORA),
            status=AdapterStatus.INACTIVE,
            parameter_count=2000000,
            memory_usage_mb=20.0,
            performance_score=0.9,
            usage_count=10
        )
        
        peft_pool.adapters["adapter1"] = adapter_info1
        peft_pool.adapters["adapter2"] = adapter_info2
        peft_pool.active_adapters["adapter1"] = Mock()  # アクティブ
        
        peft_pool._update_pool_stats()
        
        stats = peft_pool.pool_stats
        assert stats.total_adapters == 2
        assert stats.active_adapters == 1
        assert stats.inactive_adapters == 1
        assert stats.error_adapters == 0
        assert stats.total_parameters == 3000000
        assert stats.total_memory_mb == 10.0  # アクティブのみ
        assert stats.average_performance_score == 0.85  # (0.8 + 0.9) / 2
        assert stats.best_adapter == "adapter2"  # 高いスコア
        assert stats.most_used_adapter == "adapter2"  # 多い使用回数
    
    def test_get_adapter_info(self, peft_pool):
        """アダプタ情報取得テスト"""
        adapter_info = AdapterInfo(
            config=AdapterConfig(name="test_adapter", adapter_type=AdapterType.LORA),
            status=AdapterStatus.ACTIVE
        )
        
        peft_pool.adapters["test_adapter"] = adapter_info
        
        # 存在するアダプタ
        result = peft_pool.get_adapter_info("test_adapter")
        assert result == adapter_info
        
        # 存在しないアダプタ
        result = peft_pool.get_adapter_info("nonexistent")
        assert result is None
    
    def test_list_adapters(self, peft_pool):
        """アダプタ一覧取得テスト"""
        # テストアダプタ追加
        adapter1 = AdapterInfo(
            config=AdapterConfig(name="lora1", adapter_type=AdapterType.LORA),
            status=AdapterStatus.ACTIVE
        )
        adapter2 = AdapterInfo(
            config=AdapterConfig(name="lora2", adapter_type=AdapterType.LORA),
            status=AdapterStatus.INACTIVE
        )
        adapter3 = AdapterInfo(
            config=AdapterConfig(name="adalora1", adapter_type=AdapterType.ADALORA),
            status=AdapterStatus.ACTIVE
        )
        
        peft_pool.adapters.update({
            "lora1": adapter1,
            "lora2": adapter2,
            "adalora1": adapter3
        })
        
        # 全アダプタ
        all_adapters = peft_pool.list_adapters()
        assert len(all_adapters) == 3
        
        # 状態フィルタ
        active_adapters = peft_pool.list_adapters(status_filter=AdapterStatus.ACTIVE)
        assert len(active_adapters) == 2
        assert all(info.status == AdapterStatus.ACTIVE for info in active_adapters)
        
        # タイプフィルタ
        lora_adapters = peft_pool.list_adapters(adapter_type_filter=AdapterType.LORA)
        assert len(lora_adapters) == 2
        assert all(info.config.adapter_type == AdapterType.LORA for info in lora_adapters)
        
        # 複合フィルタ
        active_lora = peft_pool.list_adapters(
            status_filter=AdapterStatus.ACTIVE,
            adapter_type_filter=AdapterType.LORA
        )
        assert len(active_lora) == 1
        assert active_lora[0].config.name == "lora1"
    
    def test_get_pool_stats(self, peft_pool):
        """プール統計取得テスト"""
        # テストデータ追加
        adapter_info = AdapterInfo(
            config=AdapterConfig(name="test", adapter_type=AdapterType.LORA),
            status=AdapterStatus.ACTIVE,
            parameter_count=1000000
        )
        peft_pool.adapters["test"] = adapter_info
        
        stats = peft_pool.get_pool_stats()
        
        assert isinstance(stats, AdapterPoolStats)
        assert stats.total_adapters == 1
        assert stats.total_parameters == 1000000
        assert isinstance(stats.last_updated, datetime)


class TestAdapterDataClasses:
    """アダプタデータクラスのテスト"""
    
    def test_adapter_config(self):
        """AdapterConfig テスト"""
        config = AdapterConfig(
            name="test_adapter",
            adapter_type=AdapterType.LORA,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            description="Test adapter",
            tags=["test", "lora"]
        )
        
        assert config.name == "test_adapter"
        assert config.adapter_type == AdapterType.LORA
        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.1
        assert config.target_modules == ["q_proj", "v_proj"]
        assert config.description == "Test adapter"
        assert config.tags == ["test", "lora"]
        assert isinstance(config.created_at, datetime)
    
    def test_adapter_info(self):
        """AdapterInfo テスト"""
        config = AdapterConfig(
            name="test",
            adapter_type=AdapterType.LORA
        )
        
        info = AdapterInfo(
            config=config,
            status=AdapterStatus.ACTIVE,
            parameter_count=1000000,
            memory_usage_mb=10.0,
            performance_score=0.85,
            usage_count=5
        )
        
        assert info.config == config
        assert info.status == AdapterStatus.ACTIVE
        assert info.parameter_count == 1000000
        assert info.memory_usage_mb == 10.0
        assert info.performance_score == 0.85
        assert info.usage_count == 5
        assert info.last_used is None
        assert info.error_message is None
    
    def test_adapter_pool_stats(self):
        """AdapterPoolStats テスト"""
        stats = AdapterPoolStats(
            total_adapters=5,
            active_adapters=3,
            inactive_adapters=2,
            total_parameters=10000000,
            total_memory_mb=50.0,
            average_performance_score=0.8,
            best_adapter="best_one",
            most_used_adapter="popular_one"
        )
        
        assert stats.total_adapters == 5
        assert stats.active_adapters == 3
        assert stats.inactive_adapters == 2
        assert stats.error_adapters == 0  # デフォルト値
        assert stats.total_parameters == 10000000
        assert stats.total_memory_mb == 50.0
        assert stats.average_performance_score == 0.8
        assert stats.best_adapter == "best_one"
        assert stats.most_used_adapter == "popular_one"
        assert isinstance(stats.last_updated, datetime)


class TestAdapterEnums:
    """アダプタ列挙型のテスト"""
    
    def test_adapter_type_enum(self):
        """AdapterType 列挙型テスト"""
        assert AdapterType.LORA.value == "lora"
        assert AdapterType.ADALORA.value == "adalora"
        assert AdapterType.IA3.value == "ia3"
        assert AdapterType.PROMPT_TUNING.value == "prompt_tuning"
        assert AdapterType.PREFIX_TUNING.value == "prefix_tuning"
    
    def test_adapter_status_enum(self):
        """AdapterStatus 列挙型テスト"""
        assert AdapterStatus.INACTIVE.value == "inactive"
        assert AdapterStatus.LOADING.value == "loading"
        assert AdapterStatus.ACTIVE.value == "active"
        assert AdapterStatus.TRAINING.value == "training"
        assert AdapterStatus.SAVING.value == "saving"
        assert AdapterStatus.ERROR.value == "error"


class TestCreatePEFTAdapterPool:
    """create_peft_adapter_pool 関数のテスト"""
    
    @pytest.mark.asyncio
    async def test_create_success(self):
        """作成成功テスト"""
        with patch('src.advanced_agent.adaptation.peft_manager.PEFTAdapterPool') as MockPool:
            mock_pool = Mock()
            mock_pool.initialize = AsyncMock(return_value=True)
            MockPool.return_value = mock_pool
            
            result = await create_peft_adapter_pool("test/model")
            
            assert result == mock_pool
            MockPool.assert_called_once_with("test/model", None, None)
            mock_pool.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_failure(self):
        """作成失敗テスト"""
        with patch('src.advanced_agent.adaptation.peft_manager.PEFTAdapterPool') as MockPool:
            mock_pool = Mock()
            mock_pool.initialize = AsyncMock(return_value=False)
            MockPool.return_value = mock_pool
            
            with pytest.raises(RuntimeError, match="Failed to initialize PEFT adapter pool"):
                await create_peft_adapter_pool("test/model")


class TestPEFTIntegration:
    """PEFT統合テスト（モック使用）"""
    
    @pytest.fixture
    def mock_peft_pool(self):
        """モックPEFTプール"""
        with patch('src.advanced_agent.adaptation.peft_manager.PEFT_AVAILABLE', True), \
             patch('src.advanced_agent.adaptation.peft_manager.get_peft_model') as mock_get_peft, \
             patch('src.advanced_agent.adaptation.peft_manager.AutoModelForCausalLM') as mock_model_class, \
             patch('src.advanced_agent.adaptation.peft_manager.AutoTokenizer') as mock_tokenizer_class:
            
            # モック設定
            mock_base_model = Mock()
            mock_model_class.from_pretrained = AsyncMock(return_value=mock_base_model)
            
            mock_tokenizer = Mock()
            mock_tokenizer_class.from_pretrained = AsyncMock(return_value=mock_tokenizer)
            
            mock_peft_model = Mock()
            mock_peft_model.parameters.return_value = [
                Mock(numel=Mock(return_value=1000), requires_grad=True),
                Mock(numel=Mock(return_value=2000), requires_grad=True)
            ]
            mock_get_peft.return_value = mock_peft_model
            
            temp_dir = tempfile.mkdtemp()
            try:
                pool = PEFTAdapterPool("test/model", temp_dir)
                yield pool, mock_peft_model
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_initialize_integration(self, mock_peft_pool):
        """初期化統合テスト"""
        pool, _ = mock_peft_pool
        
        result = await pool.initialize()
        
        assert result is True
        assert pool.base_model is not None
        assert pool.tokenizer is not None
    
    @pytest.mark.asyncio
    async def test_create_adapter_integration(self, mock_peft_pool):
        """アダプタ作成統合テスト"""
        pool, mock_peft_model = mock_peft_pool
        
        # 初期化
        await pool.initialize()
        
        # アダプタ設定作成
        config = pool.create_adapter_config(
            name="test_integration",
            adapter_type=AdapterType.LORA,
            r=8,
            lora_alpha=16
        )
        
        # アダプタ作成
        adapter_info = await pool.create_adapter(config)
        
        assert adapter_info.config.name == "test_integration"
        assert adapter_info.status == AdapterStatus.ACTIVE
        assert adapter_info.parameter_count == 3000  # 1000 + 2000
        assert "test_integration" in pool.adapters
        assert "test_integration" in pool.active_adapters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])