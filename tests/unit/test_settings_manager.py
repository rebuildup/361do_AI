"""
設定管理システムのテスト
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import yaml
import json

# テスト対象をインポート
from src.advanced_agent.interfaces.settings_manager import (
    ModelConfig,
    UIConfig,
    SystemConfig,
    AdvancedAgentSettings,
    SettingsManager
)


class TestModelConfig:
    """モデル設定のテスト"""
    
    def test_model_config_creation(self):
        """モデル設定作成テスト"""
        config = ModelConfig(
            name="test-model",
            provider="ollama",
            temperature=0.8,
            max_tokens=1000
        )
        
        assert config.name == "test-model"
        assert config.provider == "ollama"
        assert config.temperature == 0.8
        assert config.max_tokens == 1000
        assert config.enabled is True  # デフォルト値
    
    def test_model_config_validation(self):
        """モデル設定バリデーションテスト"""
        # 正常値
        config = ModelConfig(
            name="valid-model",
            temperature=1.0,
            max_tokens=500
        )
        assert config.temperature == 1.0
        
        # 範囲外の値
        with pytest.raises(ValueError):
            ModelConfig(
                name="invalid-model",
                temperature=3.0  # 範囲外
            )
        
        with pytest.raises(ValueError):
            ModelConfig(
                name="invalid-model",
                max_tokens=0  # 範囲外
            )


class TestUIConfig:
    """UI設定のテスト"""
    
    def test_ui_config_defaults(self):
        """UI設定デフォルト値テスト"""
        config = UIConfig()
        
        assert config.theme == "light"
        assert config.auto_refresh is True
        assert config.refresh_interval == 5
        assert config.auto_save is True
        assert config.save_interval == 10
        assert config.show_debug is False
        assert config.max_chat_history == 100
    
    def test_ui_config_validation(self):
        """UI設定バリデーションテスト"""
        # 正常値
        config = UIConfig(refresh_interval=30, max_chat_history=500)
        assert config.refresh_interval == 30
        assert config.max_chat_history == 500
        
        # 範囲外の値
        with pytest.raises(ValueError):
            UIConfig(refresh_interval=0)  # 範囲外
        
        with pytest.raises(ValueError):
            UIConfig(max_chat_history=5)  # 範囲外


class TestSystemConfig:
    """システム設定のテスト"""
    
    def test_system_config_defaults(self):
        """システム設定デフォルト値テスト"""
        config = SystemConfig()
        
        assert config.api_base_url == "http://localhost:8000"
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.log_level == "INFO"
        assert config.enable_monitoring is True
        assert config.enable_memory is True
        assert config.gpu_memory_limit == 0.9
    
    def test_system_config_validation(self):
        """システム設定バリデーションテスト"""
        # 正常値
        config = SystemConfig(
            timeout=60,
            gpu_memory_limit=0.8
        )
        assert config.timeout == 60
        assert config.gpu_memory_limit == 0.8
        
        # 範囲外の値
        with pytest.raises(ValueError):
            SystemConfig(timeout=2)  # 範囲外
        
        with pytest.raises(ValueError):
            SystemConfig(gpu_memory_limit=1.5)  # 範囲外


class TestAdvancedAgentSettings:
    """統合設定のテスト"""
    
    def test_settings_creation(self):
        """設定作成テスト"""
        settings = AdvancedAgentSettings()
        
        # デフォルトモデルが存在することを確認
        assert len(settings.models) > 0
        assert "deepseek-r1:7b" in settings.models
        assert settings.current_model == "deepseek-r1:7b"
        
        # デフォルト設定が正しく設定されていることを確認
        assert isinstance(settings.ui, UIConfig)
        assert isinstance(settings.system, SystemConfig)
    
    def test_get_current_model_config(self):
        """現在のモデル設定取得テスト"""
        settings = AdvancedAgentSettings()
        
        current_config = settings.get_current_model_config()
        assert current_config is not None
        assert current_config.name == settings.current_model
    
    def test_add_model(self):
        """モデル追加テスト"""
        settings = AdvancedAgentSettings()
        
        new_model = ModelConfig(
            name="new-test-model",
            temperature=0.9,
            max_tokens=800
        )
        
        initial_count = len(settings.models)
        settings.add_model(new_model)
        
        assert len(settings.models) == initial_count + 1
        assert "new-test-model" in settings.models
        assert settings.models["new-test-model"].temperature == 0.9
    
    def test_remove_model(self):
        """モデル削除テスト"""
        settings = AdvancedAgentSettings()
        
        # テスト用モデルを追加
        test_model = ModelConfig(name="test-to-remove")
        settings.add_model(test_model)
        
        initial_count = len(settings.models)
        result = settings.remove_model("test-to-remove")
        
        assert result is True
        assert len(settings.models) == initial_count - 1
        assert "test-to-remove" not in settings.models
        
        # 存在しないモデルの削除
        result = settings.remove_model("non-existent")
        assert result is False
    
    def test_switch_model(self):
        """モデル切り替えテスト"""
        settings = AdvancedAgentSettings()
        
        # 利用可能なモデルに切り替え
        available_models = list(settings.models.keys())
        if len(available_models) > 1:
            new_model = available_models[1]
            result = settings.switch_model(new_model)
            
            assert result is True
            assert settings.current_model == new_model
        
        # 存在しないモデルに切り替え
        result = settings.switch_model("non-existent-model")
        assert result is False


class TestSettingsManager:
    """設定管理システムのテスト"""
    
    @pytest.fixture
    def temp_config_dir(self):
        """テスト用一時ディレクトリ"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def settings_manager(self, temp_config_dir):
        """テスト用設定管理システム"""
        return SettingsManager(config_dir=temp_config_dir)
    
    def test_settings_manager_initialization(self, settings_manager):
        """設定管理システム初期化テスト"""
        assert settings_manager.config_dir.exists()
        assert settings_manager.backup_dir.exists()
        assert settings_manager.config_file.name == "agent_settings.yaml"
    
    def test_load_default_settings(self, settings_manager):
        """デフォルト設定読み込みテスト"""
        settings = settings_manager.load_settings()
        
        assert isinstance(settings, AdvancedAgentSettings)
        assert len(settings.models) > 0
        assert settings.current_model in settings.models
        
        # 設定ファイルが作成されていることを確認
        assert settings_manager.config_file.exists()
    
    def test_save_and_load_settings(self, settings_manager):
        """設定保存・読み込みテスト"""
        # 設定を作成・保存
        settings = AdvancedAgentSettings()
        settings.current_model = "qwen2.5:7b-instruct-q4_k_m"
        settings.ui.theme = "dark"
        
        result = settings_manager.save_settings(settings)
        assert result is True
        
        # 設定を読み込み
        loaded_settings = settings_manager.load_settings()
        
        assert loaded_settings.current_model == "qwen2.5:7b-instruct-q4_k_m"
        assert loaded_settings.ui.theme == "dark"
    
    def test_update_settings(self, settings_manager):
        """設定更新テスト"""
        # 初期設定を読み込み
        settings_manager.load_settings()
        
        # 設定を更新
        result = settings_manager.update_settings(
            current_model="qwen2:1.5b-instruct-q4_k_m"
        )
        
        assert result is True
        
        # 更新された設定を確認
        updated_settings = settings_manager.get_settings()
        assert updated_settings.current_model == "qwen2:1.5b-instruct-q4_k_m"
    
    def test_create_backup(self, settings_manager):
        """バックアップ作成テスト"""
        # 設定を作成
        settings_manager.load_settings()
        
        # バックアップを作成
        backup_file = settings_manager.create_backup("test_backup")
        
        assert backup_file != ""
        assert Path(backup_file).exists()
        assert "test_backup" in backup_file
    
    def test_list_backups(self, settings_manager):
        """バックアップ一覧テスト"""
        # 設定を作成
        settings_manager.load_settings()
        
        # バックアップを作成
        settings_manager.create_backup("backup1")
        settings_manager.create_backup("backup2")
        
        # バックアップ一覧を取得
        backups = settings_manager.list_backups()
        
        assert len(backups) >= 2
        backup_names = [b["name"] for b in backups]
        assert "backup1" in backup_names
        assert "backup2" in backup_names
    
    def test_restore_backup(self, settings_manager):
        """バックアップ復元テスト"""
        # 初期設定を作成・保存
        settings = AdvancedAgentSettings()
        settings.ui.theme = "light"
        settings_manager.save_settings(settings)
        
        # バックアップを作成
        backup_name = "restore_test"
        settings_manager.create_backup(backup_name)
        
        # 設定を変更
        settings.ui.theme = "dark"
        settings_manager.save_settings(settings)
        
        # バックアップから復元
        result = settings_manager.restore_backup(backup_name)
        assert result is True
        
        # 復元された設定を確認
        restored_settings = settings_manager.get_settings()
        assert restored_settings.ui.theme == "light"
    
    def test_delete_backup(self, settings_manager):
        """バックアップ削除テスト"""
        # 設定を作成
        settings_manager.load_settings()
        
        # バックアップを作成
        backup_name = "delete_test"
        backup_file = settings_manager.create_backup(backup_name)
        assert Path(backup_file).exists()
        
        # バックアップを削除
        result = settings_manager.delete_backup(backup_name)
        assert result is True
        assert not Path(backup_file).exists()
        
        # 存在しないバックアップの削除
        result = settings_manager.delete_backup("non_existent")
        assert result is False
    
    def test_export_import_yaml(self, settings_manager):
        """YAML エクスポート・インポートテスト"""
        # 設定を作成
        settings = AdvancedAgentSettings()
        settings.ui.theme = "dark"
        settings.current_model = "qwen2.5:7b-instruct-q4_k_m"
        settings_manager.save_settings(settings)
        
        # YAML エクスポート
        exported_yaml = settings_manager.export_settings("yaml")
        assert exported_yaml != ""
        assert "dark" in exported_yaml
        assert "qwen2.5" in exported_yaml
        
        # 設定を変更
        settings.ui.theme = "light"
        settings_manager.save_settings(settings)
        
        # YAML インポート
        result = settings_manager.import_settings(exported_yaml, "yaml")
        assert result is True
        
        # インポートされた設定を確認
        imported_settings = settings_manager.get_settings()
        assert imported_settings.ui.theme == "dark"
        assert imported_settings.current_model == "qwen2.5:7b-instruct-q4_k_m"
    
    def test_export_import_json(self, settings_manager):
        """JSON エクスポート・インポートテスト"""
        # 設定を作成
        settings = AdvancedAgentSettings()
        settings.ui.auto_refresh = False
        settings_manager.save_settings(settings)
        
        # JSON エクスポート
        exported_json = settings_manager.export_settings("json")
        assert exported_json != ""
        
        # JSON として解析可能かチェック
        parsed_json = json.loads(exported_json)
        assert parsed_json["ui"]["auto_refresh"] is False
        
        # 設定を変更
        settings.ui.auto_refresh = True
        settings_manager.save_settings(settings)
        
        # JSON インポート
        result = settings_manager.import_settings(exported_json, "json")
        assert result is True
        
        # インポートされた設定を確認
        imported_settings = settings_manager.get_settings()
        assert imported_settings.ui.auto_refresh is False
    
    def test_reset_to_defaults(self, settings_manager):
        """デフォルトリセットテスト"""
        # 設定を変更
        settings = settings_manager.load_settings()
        settings.ui.theme = "dark"
        settings.system.timeout = 60
        settings_manager.save_settings(settings)
        
        # デフォルトにリセット
        result = settings_manager.reset_to_defaults()
        assert result is True
        
        # リセットされた設定を確認
        reset_settings = settings_manager.get_settings()
        assert reset_settings.ui.theme == "light"  # デフォルト値
        assert reset_settings.system.timeout == 30  # デフォルト値


if __name__ == "__main__":
    pytest.main([__file__])