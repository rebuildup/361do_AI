"""
QLoRA Trainer のテスト
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
import tempfile
import shutil

from src.advanced_agent.adaptation.qlora_trainer import (
    QLoRATrainer, QLoRAConfig, TrainingMetrics, QLoRATrainingResult,
    TrainingPhase, QLoRAOptimization, QLoRATrainingCallback,
    create_qlora_trainer, create_simple_dataset
)
from src.advanced_agent.quantization.bitsandbytes_manager import BitsAndBytesManager
from src.advanced_agent.adaptation.peft_manager import PEFTAdapterPool
from src.advanced_agent.monitoring.system_monitor import SystemMonitor


class TestQLoRATrainer:
    """QLoRATrainer クラスのテスト"""
    
    @pytest.fixture
    def mock_bnb_manager(self):
        """モックBitsAndBytesマネージャー"""
        return Mock(spec=BitsAndBytesManager)
    
    @pytest.fixture
    def mock_peft_pool(self):
        """モックPEFTプール"""
        pool = Mock(spec=PEFTAdapterPool)
        pool.adapters = {}
        return pool
    
    @pytest.fixture
    def mock_system_monitor(self):
        """モックシステムモニター"""
        return Mock(spec=SystemMonitor)
    
    @pytest.fixture
    def qlora_trainer(self, mock_bnb_manager, mock_peft_pool, mock_system_monitor):
        """QLoRATrainer インスタンス"""
        with patch('src.advanced_agent.adaptation.qlora_trainer.get_config') as mock_get_config, \
             patch('src.advanced_agent.adaptation.qlora_trainer.QLORA_AVAILABLE', True):
            from src.advanced_agent.core.config import AdvancedAgentConfig
            mock_get_config.return_value = AdvancedAgentConfig()
            
            return QLoRATrainer(
                base_model_name="test/model",
                bnb_manager=mock_bnb_manager,
                peft_pool=mock_peft_pool,
                system_monitor=mock_system_monitor
            )
    
    def test_init(self, qlora_trainer, mock_bnb_manager, mock_peft_pool, mock_system_monitor):
        """初期化テスト"""
        assert qlora_trainer.base_model_name == "test/model"
        assert qlora_trainer.bnb_manager == mock_bnb_manager
        assert qlora_trainer.peft_pool == mock_peft_pool
        assert qlora_trainer.system_monitor == mock_system_monitor
        assert qlora_trainer.current_phase == TrainingPhase.INITIALIZATION
        assert len(qlora_trainer.training_metrics) == 0
        assert len(qlora_trainer.training_history) == 0
        assert qlora_trainer.peak_memory_mb == 0.0
        assert qlora_trainer.best_eval_loss is None
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, qlora_trainer):
        """初期化成功テスト"""
        result = await qlora_trainer.initialize()
        
        assert result is True
        assert qlora_trainer.current_phase == TrainingPhase.INITIALIZATION
    
    @patch('src.advanced_agent.adaptation.qlora_trainer.QLORA_AVAILABLE', False)
    @pytest.mark.asyncio
    async def test_initialize_unavailable(self, qlora_trainer):
        """QLORA利用不可時の初期化テスト"""
        result = await qlora_trainer.initialize()
        
        assert result is False
    
    def test_create_qlora_config_memory_optimized(self, qlora_trainer):
        """メモリ最適化QLoRA設定作成テスト"""
        config = qlora_trainer.create_qlora_config(
            optimization_level=QLoRAOptimization.MEMORY_OPTIMIZED,
            memory_limit_gb=3.0
        )
        
        assert config.optimization_level == QLoRAOptimization.MEMORY_OPTIMIZED
        assert config.memory_limit_gb == 3.0
        assert config.lora_r == 32  # メモリ最適化
        assert config.per_device_train_batch_size == 1
        assert config.gradient_accumulation_steps == 8
        assert config.gradient_checkpointing is True
        assert config.bf16 is True
        assert config.fp16 is False
    
    def test_create_qlora_config_performance_optimized(self, qlora_trainer):
        """性能最適化QLoRA設定作成テスト"""
        config = qlora_trainer.create_qlora_config(
            optimization_level=QLoRAOptimization.PERFORMANCE_OPTIMIZED,
            memory_limit_gb=8.0
        )
        
        assert config.optimization_level == QLoRAOptimization.PERFORMANCE_OPTIMIZED
        assert config.lora_r == 128  # 性能最適化
        assert config.per_device_train_batch_size == 2
        assert config.gradient_accumulation_steps == 2
        assert config.gradient_checkpointing is False
        assert config.bf16 is False
        assert config.fp16 is True
    
    def test_create_qlora_config_balanced(self, qlora_trainer):
        """バランス型QLoRA設定作成テスト"""
        config = qlora_trainer.create_qlora_config(
            optimization_level=QLoRAOptimization.BALANCED,
            memory_limit_gb=6.0
        )
        
        assert config.optimization_level == QLoRAOptimization.BALANCED
        assert config.lora_r == 64  # バランス型
        assert config.per_device_train_batch_size == 1
        assert config.gradient_accumulation_steps == 4
        assert config.gradient_checkpointing is True
        assert config.bf16 is True
    
    def test_create_qlora_config_low_memory(self, qlora_trainer):
        """低メモリ環境QLoRA設定作成テスト"""
        config = qlora_trainer.create_qlora_config(
            optimization_level=QLoRAOptimization.BALANCED,
            memory_limit_gb=2.5  # 3GB未満
        )
        
        assert config.lora_r <= 32  # 低メモリ調整
        assert config.per_device_train_batch_size == 1
        assert config.gradient_accumulation_steps >= 8
    
    def test_create_qlora_config_custom_params(self, qlora_trainer):
        """カスタムパラメータQLoRA設定作成テスト"""
        config = qlora_trainer.create_qlora_config(
            optimization_level=QLoRAOptimization.BALANCED,
            lora_r=256,  # カスタム値
            learning_rate=1e-4,
            num_train_epochs=5
        )
        
        assert config.lora_r == 256  # カスタム値が優先
        assert config.learning_rate == 1e-4
        assert config.num_train_epochs == 5
    
    @patch('src.advanced_agent.adaptation.qlora_trainer.BitsAndBytesConfig')
    @patch('src.advanced_agent.adaptation.qlora_trainer.torch')
    def test_create_quantization_config(self, mock_torch, mock_bnb_config, qlora_trainer):
        """量子化設定作成テスト"""
        mock_torch.bfloat16 = "bfloat16_mock"
        mock_torch.float16 = "float16_mock"
        mock_config = Mock()
        mock_bnb_config.return_value = mock_config
        
        qlora_config = QLoRAConfig(bf16=True)
        
        result = qlora_trainer.create_quantization_config(qlora_config)
        
        assert result == mock_config
        mock_bnb_config.assert_called_once_with(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16_mock"
        )
    
    @patch('src.advanced_agent.adaptation.qlora_trainer.LoraConfig')
    def test_create_lora_config(self, mock_lora_config, qlora_trainer):
        """LoRA設定作成テスト"""
        mock_config = Mock()
        mock_lora_config.return_value = mock_config
        
        qlora_config = QLoRAConfig(
            lora_r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]
        )
        
        result = qlora_trainer.create_lora_config(qlora_config)
        
        assert result == mock_config
        mock_lora_config.assert_called_once()
        
        # 呼び出し引数確認
        call_kwargs = mock_lora_config.call_args[1]
        assert call_kwargs["r"] == 64
        assert call_kwargs["lora_alpha"] == 16
        assert call_kwargs["lora_dropout"] == 0.1
        assert call_kwargs["target_modules"] == ["q_proj", "v_proj"]
    
    @patch('src.advanced_agent.adaptation.qlora_trainer.TrainingArguments')
    def test_create_training_arguments(self, mock_training_args, qlora_trainer):
        """学習引数作成テスト"""
        mock_args = Mock()
        mock_training_args.return_value = mock_args
        
        qlora_config = QLoRAConfig(
            output_dir="./test_output",
            num_train_epochs=3,
            per_device_train_batch_size=2,
            learning_rate=2e-4,
            gradient_checkpointing=True,
            bf16=True
        )
        
        result = qlora_trainer.create_training_arguments(qlora_config)
        
        assert result == mock_args
        mock_training_args.assert_called_once()
        
        # 呼び出し引数確認
        call_kwargs = mock_training_args.call_args[1]
        assert call_kwargs["output_dir"] == "./test_output"
        assert call_kwargs["num_train_epochs"] == 3
        assert call_kwargs["per_device_train_batch_size"] == 2
        assert call_kwargs["learning_rate"] == 2e-4
        assert call_kwargs["gradient_checkpointing"] is True
        assert call_kwargs["bf16"] is True
    
    def test_get_training_stats_empty(self, qlora_trainer):
        """空の学習統計取得テスト"""
        stats = qlora_trainer.get_training_stats()
        
        assert stats["total_trainings"] == 0
    
    def test_get_training_stats_with_history(self, qlora_trainer):
        """履歴ありの学習統計取得テスト"""
        # テスト履歴追加
        result1 = QLoRATrainingResult(
            config=QLoRAConfig(),
            adapter_name="test1",
            training_time=100.0,
            final_loss=0.5,
            peak_memory_mb=1000.0,
            trainable_ratio=0.1,
            success=True
        )
        
        result2 = QLoRATrainingResult(
            config=QLoRAConfig(),
            adapter_name="test2",
            training_time=200.0,
            final_loss=0.3,
            peak_memory_mb=1500.0,
            trainable_ratio=0.15,
            success=True
        )
        
        result3 = QLoRATrainingResult(
            config=QLoRAConfig(),
            adapter_name="test3",
            success=False,
            error_message="Test error"
        )
        
        qlora_trainer.training_history.extend([result1, result2, result3])
        
        stats = qlora_trainer.get_training_stats()
        
        assert stats["total_trainings"] == 3
        assert stats["successful_trainings"] == 2
        assert stats["failed_trainings"] == 1
        assert stats["success_rate"] == 2/3
        assert stats["average_training_time"] == 150.0  # (100 + 200) / 2
        assert stats["average_final_loss"] == 0.4  # (0.5 + 0.3) / 2
        assert stats["average_peak_memory"] == 1250.0  # (1000 + 1500) / 2
        assert stats["average_trainable_ratio"] == 0.125  # (0.1 + 0.15) / 2
    
    def test_get_training_history(self, qlora_trainer):
        """学習履歴取得テスト"""
        # テスト履歴追加
        results = [
            QLoRATrainingResult(config=QLoRAConfig(), adapter_name=f"test{i}", success=i % 2 == 0)
            for i in range(5)
        ]
        qlora_trainer.training_history.extend(results)
        
        # 全履歴
        all_history = qlora_trainer.get_training_history()
        assert len(all_history) == 5
        
        # 件数制限
        limited_history = qlora_trainer.get_training_history(limit=3)
        assert len(limited_history) == 3
        
        # 成功のみ
        success_history = qlora_trainer.get_training_history(success_only=True)
        assert len(success_history) == 3  # 0, 2, 4番目
        assert all(r.success for r in success_history)
    
    @pytest.mark.asyncio
    async def test_cleanup(self, qlora_trainer):
        """クリーンアップテスト"""
        # モックオブジェクト設定
        qlora_trainer.model = Mock()
        qlora_trainer.tokenizer = Mock()
        qlora_trainer.trainer = Mock()
        
        with patch('src.advanced_agent.adaptation.qlora_trainer.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.empty_cache = Mock()
            
            await qlora_trainer.cleanup()
            
            assert qlora_trainer.model is None
            assert qlora_trainer.tokenizer is None
            assert qlora_trainer.trainer is None
            mock_torch.cuda.empty_cache.assert_called_once()


class TestQLoRATrainingCallback:
    """QLoRATrainingCallback クラスのテスト"""
    
    @pytest.fixture
    def mock_qlora_trainer(self):
        """モックQLoRATrainer"""
        trainer = Mock()
        trainer.current_phase = TrainingPhase.INITIALIZATION
        trainer.training_metrics = []
        trainer.peak_memory_mb = 0.0
        trainer.best_eval_loss = None
        return trainer
    
    @pytest.fixture
    def callback(self, mock_qlora_trainer):
        """QLoRATrainingCallback インスタンス"""
        return QLoRATrainingCallback(mock_qlora_trainer)
    
    def test_init(self, callback, mock_qlora_trainer):
        """初期化テスト"""
        assert callback.qlora_trainer == mock_qlora_trainer
        assert callback.start_time is None
        assert callback.step_start_time is None
    
    def test_on_train_begin(self, callback, mock_qlora_trainer):
        """学習開始時テスト"""
        mock_state = Mock()
        mock_state.max_steps = 1000
        
        callback.on_train_begin(None, mock_state, None)
        
        assert callback.start_time is not None
        assert mock_qlora_trainer.current_phase == TrainingPhase.TRAINING
    
    def test_on_step_begin(self, callback):
        """ステップ開始時テスト"""
        callback.on_step_begin(None, None, None)
        
        assert callback.step_start_time is not None
    
    @patch('src.advanced_agent.adaptation.qlora_trainer.torch')
    def test_on_step_end(self, mock_torch, callback, mock_qlora_trainer):
        """ステップ終了時テスト"""
        # モック設定
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1024 * 1024 * 1024  # 1GB
        
        callback.start_time = time.time() - 10  # 10秒前
        callback.step_start_time = time.time() - 1  # 1秒前
        
        mock_state = Mock()
        mock_state.epoch = 1.0
        mock_state.global_step = 100
        mock_state.log_history = [{"train_loss": 0.5, "learning_rate": 1e-4}]
        
        callback.on_step_end(None, mock_state, None)
        
        assert len(mock_qlora_trainer.training_metrics) == 1
        
        metrics = mock_qlora_trainer.training_metrics[0]
        assert metrics.epoch == 1.0
        assert metrics.step == 100
        assert metrics.loss == 0.5
        assert metrics.learning_rate == 1e-4
        assert metrics.gpu_memory_mb == 1024.0  # 1GB in MB
        assert mock_qlora_trainer.peak_memory_mb == 1024.0
    
    def test_on_evaluate(self, callback, mock_qlora_trainer):
        """評価時テスト"""
        mock_state = Mock()
        mock_state.log_history = [{"eval_loss": 0.3}]
        
        callback.on_evaluate(None, mock_state, None)
        
        assert mock_qlora_trainer.current_phase == TrainingPhase.EVALUATION
        assert mock_qlora_trainer.best_eval_loss == 0.3
        
        # より良いスコアで更新
        mock_state.log_history = [{"eval_loss": 0.2}]
        callback.on_evaluate(None, mock_state, None)
        
        assert mock_qlora_trainer.best_eval_loss == 0.2
    
    def test_on_save(self, callback, mock_qlora_trainer):
        """保存時テスト"""
        mock_state = Mock()
        mock_state.global_step = 500
        
        callback.on_save(None, mock_state, None)
        
        assert mock_qlora_trainer.current_phase == TrainingPhase.SAVING
    
    def test_on_train_end(self, callback, mock_qlora_trainer):
        """学習終了時テスト"""
        callback.start_time = time.time() - 100  # 100秒前
        
        callback.on_train_end(None, None, None)
        
        assert mock_qlora_trainer.current_phase == TrainingPhase.COMPLETED


class TestQLoRADataClasses:
    """QLoRAデータクラスのテスト"""
    
    def test_qlora_config(self):
        """QLoRAConfig テスト"""
        config = QLoRAConfig(
            lora_r=64,
            lora_alpha=16,
            learning_rate=2e-4,
            num_train_epochs=3,
            optimization_level=QLoRAOptimization.BALANCED,
            memory_limit_gb=4.0
        )
        
        assert config.lora_r == 64
        assert config.lora_alpha == 16
        assert config.learning_rate == 2e-4
        assert config.num_train_epochs == 3
        assert config.optimization_level == QLoRAOptimization.BALANCED
        assert config.memory_limit_gb == 4.0
        assert config.load_in_4bit is True
        assert config.gradient_checkpointing is True
    
    def test_training_metrics(self):
        """TrainingMetrics テスト"""
        metrics = TrainingMetrics(
            epoch=1.5,
            step=100,
            loss=0.5,
            learning_rate=1e-4,
            gpu_memory_mb=1024.0,
            training_time=300.0,
            samples_per_second=2.5
        )
        
        assert metrics.epoch == 1.5
        assert metrics.step == 100
        assert metrics.loss == 0.5
        assert metrics.learning_rate == 1e-4
        assert metrics.gpu_memory_mb == 1024.0
        assert metrics.training_time == 300.0
        assert metrics.samples_per_second == 2.5
        assert isinstance(metrics.timestamp, datetime)
    
    def test_qlora_training_result(self):
        """QLoRATrainingResult テスト"""
        config = QLoRAConfig()
        
        result = QLoRATrainingResult(
            config=config,
            adapter_name="test_adapter",
            total_epochs=3,
            total_steps=1000,
            final_loss=0.3,
            training_time=600.0,
            trainable_parameters=1000000,
            total_parameters=10000000,
            success=True
        )
        
        assert result.config == config
        assert result.adapter_name == "test_adapter"
        assert result.total_epochs == 3
        assert result.total_steps == 1000
        assert result.final_loss == 0.3
        assert result.training_time == 600.0
        assert result.trainable_parameters == 1000000
        assert result.total_parameters == 10000000
        assert result.trainable_ratio == 0.0  # デフォルト値
        assert result.success is True
        assert result.error_message is None


class TestQLoRAEnums:
    """QLoRA列挙型のテスト"""
    
    def test_training_phase_enum(self):
        """TrainingPhase 列挙型テスト"""
        assert TrainingPhase.INITIALIZATION.value == "initialization"
        assert TrainingPhase.PREPARATION.value == "preparation"
        assert TrainingPhase.TRAINING.value == "training"
        assert TrainingPhase.EVALUATION.value == "evaluation"
        assert TrainingPhase.SAVING.value == "saving"
        assert TrainingPhase.COMPLETED.value == "completed"
        assert TrainingPhase.ERROR.value == "error"
    
    def test_qlora_optimization_enum(self):
        """QLoRAOptimization 列挙型テスト"""
        assert QLoRAOptimization.MEMORY_OPTIMIZED.value == "memory_optimized"
        assert QLoRAOptimization.BALANCED.value == "balanced"
        assert QLoRAOptimization.PERFORMANCE_OPTIMIZED.value == "performance_optimized"


class TestUtilityFunctions:
    """ユーティリティ関数のテスト"""
    
    @pytest.mark.asyncio
    async def test_create_qlora_trainer(self):
        """QLoRA学習システム作成テスト"""
        with patch('src.advanced_agent.adaptation.qlora_trainer.QLoRATrainer') as MockTrainer:
            mock_trainer = Mock()
            mock_trainer.initialize = AsyncMock(return_value=True)
            MockTrainer.return_value = mock_trainer
            
            result = await create_qlora_trainer("test/model")
            
            assert result == mock_trainer
            MockTrainer.assert_called_once_with("test/model", None, None, None)
            mock_trainer.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_qlora_trainer_failure(self):
        """QLoRA学習システム作成失敗テスト"""
        with patch('src.advanced_agent.adaptation.qlora_trainer.QLoRATrainer') as MockTrainer:
            mock_trainer = Mock()
            mock_trainer.initialize = AsyncMock(return_value=False)
            MockTrainer.return_value = mock_trainer
            
            with pytest.raises(RuntimeError, match="Failed to initialize QLoRA trainer"):
                await create_qlora_trainer("test/model")
    
    @patch('src.advanced_agent.adaptation.qlora_trainer.HFDataset')
    def test_create_simple_dataset(self, mock_hf_dataset):
        """簡単なデータセット作成テスト"""
        # モック設定
        mock_dataset = Mock()
        mock_tokenized_dataset = Mock()
        mock_dataset.map.return_value = mock_tokenized_dataset
        mock_hf_dataset.from_dict.return_value = mock_dataset
        
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        
        texts = ["Hello world", "How are you?"]
        
        result = create_simple_dataset(texts, mock_tokenizer, max_length=128)
        
        assert result == mock_tokenized_dataset
        mock_hf_dataset.from_dict.assert_called_once_with({"text": texts})
        mock_dataset.map.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])