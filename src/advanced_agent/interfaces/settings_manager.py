"""
Pydantic + Streamlit 設定管理システム

Pydantic Settings による 動的設定変更・反映を統合し、
Streamlit の既存選択機能による モデル選択・切り替えを実装、
バックアップ・復元 UI を統合
"""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import uuid

import streamlit as st
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
import yaml

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """モデル設定"""
    
    name: str = Field(..., description="モデル名")
    provider: str = Field("ollama", description="プロバイダー")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperature")
    max_tokens: int = Field(500, ge=1, le=4000, description="最大トークン数")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p")
    top_k: int = Field(40, ge=1, le=100, description="Top-k")
    repeat_penalty: float = Field(1.1, ge=0.0, le=2.0, description="繰り返しペナルティ")
    enabled: bool = Field(True, description="有効/無効")


class UIConfig(BaseModel):
    """UI設定"""
    
    theme: str = Field("light", description="テーマ")
    auto_refresh: bool = Field(True, description="自動リフレッシュ")
    refresh_interval: int = Field(5, ge=1, le=60, description="リフレッシュ間隔（秒）")
    auto_save: bool = Field(True, description="自動保存")
    save_interval: int = Field(10, ge=1, le=60, description="保存間隔（分）")
    show_debug: bool = Field(False, description="デバッグ情報表示")
    max_chat_history: int = Field(100, ge=10, le=1000, description="最大チャット履歴数")


class SystemConfig(BaseModel):
    """システム設定"""
    
    api_base_url: str = Field("http://localhost:8000", description="API ベース URL")
    timeout: int = Field(30, ge=5, le=300, description="タイムアウト（秒）")
    max_retries: int = Field(3, ge=0, le=10, description="最大リトライ回数")
    log_level: str = Field("INFO", description="ログレベル")
    enable_monitoring: bool = Field(True, description="監視機能有効")
    enable_memory: bool = Field(True, description="記憶機能有効")
    gpu_memory_limit: float = Field(0.9, ge=0.1, le=1.0, description="GPU メモリ制限")


class AdvancedAgentSettings(BaseSettings):
    """Advanced AI Agent 統合設定"""
    
    # モデル設定
    models: Dict[str, ModelConfig] = Field(
        default_factory=lambda: {
            "deepseek-r1:7b": ModelConfig(
                name="deepseek-r1:7b",
                provider="ollama",
                temperature=0.7,
                max_tokens=500
            ),
            "qwen2.5:7b-instruct-q4_k_m": ModelConfig(
                name="qwen2.5:7b-instruct-q4_k_m",
                provider="ollama",
                temperature=0.8,
                max_tokens=1000
            ),
            "qwen2:1.5b-instruct-q4_k_m": ModelConfig(
                name="qwen2:1.5b-instruct-q4_k_m",
                provider="ollama",
                temperature=0.6,
                max_tokens=300
            )
        }
    )
    
    # 現在選択中のモデル
    current_model: str = Field("deepseek-r1:7b", description="現在のモデル")
    
    # UI設定
    ui: UIConfig = Field(default_factory=UIConfig)
    
    # システム設定
    system: SystemConfig = Field(default_factory=SystemConfig)
    
    # メタデータ
    version: str = Field("1.0.0", description="設定バージョン")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        env_prefix = "ADVANCED_AGENT_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @validator('current_model')
    def validate_current_model(cls, v, values):
        """現在のモデルが利用可能なモデルに含まれているかチェック"""
        models = values.get('models', {})
        if v not in models:
            logger.warning(f"現在のモデル '{v}' が利用可能なモデルに含まれていません")
        return v
    
    def get_current_model_config(self) -> Optional[ModelConfig]:
        """現在のモデル設定を取得"""
        return self.models.get(self.current_model)
    
    def add_model(self, model_config: ModelConfig) -> None:
        """モデル設定を追加"""
        self.models[model_config.name] = model_config
        self.updated_at = datetime.now()
    
    def remove_model(self, model_name: str) -> bool:
        """モデル設定を削除"""
        if model_name in self.models:
            del self.models[model_name]
            
            # 現在のモデルが削除された場合、別のモデルを選択
            if self.current_model == model_name and self.models:
                self.current_model = next(iter(self.models.keys()))
            
            self.updated_at = datetime.now()
            return True
        return False
    
    def switch_model(self, model_name: str) -> bool:
        """モデルを切り替え"""
        if model_name in self.models:
            self.current_model = model_name
            self.updated_at = datetime.now()
            return True
        return False


class SettingsManager:
    """設定管理システム"""
    
    def __init__(self, config_dir: str = ".kiro/settings"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_file = self.config_dir / "agent_settings.yaml"
        self.backup_dir = self.config_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        self._settings: Optional[AdvancedAgentSettings] = None
        
        logger.info(f"設定管理システム初期化: {self.config_dir}")
    
    def load_settings(self) -> AdvancedAgentSettings:
        """設定を読み込み"""
        
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                
                # Pydantic モデルに変換
                self._settings = AdvancedAgentSettings(**config_data)
                logger.info("設定ファイルを読み込みました")
            else:
                # デフォルト設定を作成
                self._settings = AdvancedAgentSettings()
                self.save_settings()
                logger.info("デフォルト設定を作成しました")
                
        except Exception as e:
            logger.error(f"設定読み込みエラー: {e}")
            # フォールバック: デフォルト設定
            self._settings = AdvancedAgentSettings()
        
        return self._settings
    
    def save_settings(self, settings: Optional[AdvancedAgentSettings] = None) -> bool:
        """設定を保存"""
        
        try:
            if settings:
                self._settings = settings
            
            if not self._settings:
                logger.error("保存する設定がありません")
                return False
            
            # 更新時刻を設定
            self._settings.updated_at = datetime.now()
            
            # YAML形式で保存
            config_data = self._settings.dict()
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            
            logger.info("設定を保存しました")
            return True
            
        except Exception as e:
            logger.error(f"設定保存エラー: {e}")
            return False
    
    def get_settings(self) -> AdvancedAgentSettings:
        """現在の設定を取得"""
        if not self._settings:
            return self.load_settings()
        return self._settings
    
    def update_settings(self, **kwargs) -> bool:
        """設定を更新"""
        
        try:
            settings = self.get_settings()
            
            # 設定を更新
            for key, value in kwargs.items():
                if hasattr(settings, key):
                    setattr(settings, key, value)
            
            return self.save_settings(settings)
            
        except Exception as e:
            logger.error(f"設定更新エラー: {e}")
            return False
    
    def create_backup(self, name: Optional[str] = None) -> str:
        """設定のバックアップを作成"""
        
        try:
            if not name:
                name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            backup_file = self.backup_dir / f"{name}.yaml"
            
            if self.config_file.exists():
                shutil.copy2(self.config_file, backup_file)
                logger.info(f"バックアップを作成しました: {backup_file}")
                return str(backup_file)
            else:
                logger.warning("設定ファイルが存在しないため、バックアップを作成できません")
                return ""
                
        except Exception as e:
            logger.error(f"バックアップ作成エラー: {e}")
            return ""
    
    def restore_backup(self, backup_name: str) -> bool:
        """バックアップから設定を復元"""
        
        try:
            backup_file = self.backup_dir / f"{backup_name}.yaml"
            
            if not backup_file.exists():
                logger.error(f"バックアップファイルが見つかりません: {backup_file}")
                return False
            
            # 現在の設定をバックアップ
            self.create_backup("before_restore")
            
            # バックアップから復元
            shutil.copy2(backup_file, self.config_file)
            
            # 設定を再読み込み
            self.load_settings()
            
            logger.info(f"バックアップから復元しました: {backup_name}")
            return True
            
        except Exception as e:
            logger.error(f"バックアップ復元エラー: {e}")
            return False
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """バックアップ一覧を取得"""
        
        backups = []
        
        try:
            for backup_file in self.backup_dir.glob("*.yaml"):
                stat = backup_file.stat()
                
                backups.append({
                    "name": backup_file.stem,
                    "file": backup_file.name,
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime),
                    "modified": datetime.fromtimestamp(stat.st_mtime)
                })
            
            # 作成日時でソート（新しい順）
            backups.sort(key=lambda x: x["created"], reverse=True)
            
        except Exception as e:
            logger.error(f"バックアップ一覧取得エラー: {e}")
        
        return backups
    
    def delete_backup(self, backup_name: str) -> bool:
        """バックアップを削除"""
        
        try:
            backup_file = self.backup_dir / f"{backup_name}.yaml"
            
            if backup_file.exists():
                backup_file.unlink()
                logger.info(f"バックアップを削除しました: {backup_name}")
                return True
            else:
                logger.warning(f"バックアップファイルが見つかりません: {backup_name}")
                return False
                
        except Exception as e:
            logger.error(f"バックアップ削除エラー: {e}")
            return False
    
    def export_settings(self, format: str = "yaml") -> str:
        """設定をエクスポート"""
        
        try:
            settings = self.get_settings()
            
            if format.lower() == "json":
                return json.dumps(settings.dict(), indent=2, default=str, ensure_ascii=False)
            else:  # YAML
                return yaml.dump(settings.dict(), default_flow_style=False, allow_unicode=True)
                
        except Exception as e:
            logger.error(f"設定エクスポートエラー: {e}")
            return ""
    
    def import_settings(self, data: str, format: str = "yaml") -> bool:
        """設定をインポート"""
        
        try:
            if format.lower() == "json":
                config_data = json.loads(data)
            else:  # YAML
                config_data = yaml.safe_load(data)
            
            # バリデーション
            settings = AdvancedAgentSettings(**config_data)
            
            # 現在の設定をバックアップ
            self.create_backup("before_import")
            
            # 新しい設定を保存
            return self.save_settings(settings)
            
        except Exception as e:
            logger.error(f"設定インポートエラー: {e}")
            return False
    
    def reset_to_defaults(self) -> bool:
        """設定をデフォルトにリセット"""
        
        try:
            # 現在の設定をバックアップ
            self.create_backup("before_reset")
            
            # デフォルト設定を作成
            default_settings = AdvancedAgentSettings()
            
            return self.save_settings(default_settings)
            
        except Exception as e:
            logger.error(f"設定リセットエラー: {e}")
            return False


class StreamlitSettingsUI:
    """Streamlit 設定管理 UI"""
    
    def __init__(self, settings_manager: SettingsManager):
        self.settings_manager = settings_manager
        
    def render_settings_panel(self):
        """設定パネルを描画"""
        
        st.markdown("## ⚙️ 設定管理")
        
        # 設定タブ
        tab1, tab2, tab3, tab4 = st.tabs([
            "🤖 モデル設定", 
            "🎨 UI設定", 
            "🔧 システム設定", 
            "💾 バックアップ・復元"
        ])
        
        with tab1:
            self._render_model_settings()
        
        with tab2:
            self._render_ui_settings()
        
        with tab3:
            self._render_system_settings()
        
        with tab4:
            self._render_backup_restore()
    
    def _render_model_settings(self):
        """モデル設定UI - Streamlit の既存選択機能による モデル選択・切り替え"""
        
        st.markdown("### 🤖 モデル設定")
        
        settings = self.settings_manager.get_settings()
        
        # 現在のモデル選択
        model_names = list(settings.models.keys())
        current_index = model_names.index(settings.current_model) if settings.current_model in model_names else 0
        
        selected_model = st.selectbox(
            "現在のモデル",
            model_names,
            index=current_index,
            help="使用するAIモデルを選択してください"
        )
        
        # モデル切り替え
        if selected_model != settings.current_model:
            if settings.switch_model(selected_model):
                self.settings_manager.save_settings(settings)
                st.success(f"モデルを '{selected_model}' に切り替えました")
                st.rerun()
        
        st.markdown("---")
        
        # 選択されたモデルの詳細設定
        if selected_model in settings.models:
            model_config = settings.models[selected_model]
            
            st.markdown(f"#### {selected_model} の設定")
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=model_config.temperature,
                    step=0.1,
                    help="生成の創造性を制御します"
                )
                
                new_max_tokens = st.slider(
                    "最大トークン数",
                    min_value=1,
                    max_value=4000,
                    value=model_config.max_tokens,
                    step=50,
                    help="生成する最大トークン数"
                )
                
                new_top_p = st.slider(
                    "Top-p",
                    min_value=0.0,
                    max_value=1.0,
                    value=model_config.top_p,
                    step=0.1,
                    help="核サンプリングの閾値"
                )
            
            with col2:
                new_top_k = st.slider(
                    "Top-k",
                    min_value=1,
                    max_value=100,
                    value=model_config.top_k,
                    step=1,
                    help="上位K個のトークンから選択"
                )
                
                new_repeat_penalty = st.slider(
                    "繰り返しペナルティ",
                    min_value=0.0,
                    max_value=2.0,
                    value=model_config.repeat_penalty,
                    step=0.1,
                    help="繰り返しを抑制する強度"
                )
                
                new_enabled = st.checkbox(
                    "有効",
                    value=model_config.enabled,
                    help="このモデルを有効にする"
                )
            
            # 設定更新
            if st.button("モデル設定を更新", key=f"update_model_{selected_model}"):
                model_config.temperature = new_temperature
                model_config.max_tokens = new_max_tokens
                model_config.top_p = new_top_p
                model_config.top_k = new_top_k
                model_config.repeat_penalty = new_repeat_penalty
                model_config.enabled = new_enabled
                
                if self.settings_manager.save_settings(settings):
                    st.success("モデル設定を更新しました")
                    st.rerun()
                else:
                    st.error("モデル設定の更新に失敗しました")
        
        st.markdown("---")
        
        # 新しいモデル追加
        with st.expander("➕ 新しいモデルを追加"):
            self._render_add_model_form(settings)
        
        # モデル削除
        if len(settings.models) > 1:
            with st.expander("🗑️ モデルを削除"):
                model_to_delete = st.selectbox(
                    "削除するモデル",
                    [name for name in model_names if name != settings.current_model],
                    help="現在選択中のモデル以外を削除できます"
                )
                
                if st.button("モデルを削除", type="secondary"):
                    if settings.remove_model(model_to_delete):
                        self.settings_manager.save_settings(settings)
                        st.success(f"モデル '{model_to_delete}' を削除しました")
                        st.rerun()
                    else:
                        st.error("モデルの削除に失敗しました")
    
    def _render_add_model_form(self, settings: AdvancedAgentSettings):
        """新しいモデル追加フォーム"""
        
        with st.form("add_model_form"):
            st.markdown("#### 新しいモデルを追加")
            
            col1, col2 = st.columns(2)
            
            with col1:
                model_name = st.text_input("モデル名", placeholder="例: llama2:7b")
                provider = st.selectbox("プロバイダー", ["ollama", "openai", "anthropic"])
                temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
                max_tokens = st.slider("最大トークン数", 1, 4000, 500, 50)
            
            with col2:
                top_p = st.slider("Top-p", 0.0, 1.0, 0.9, 0.1)
                top_k = st.slider("Top-k", 1, 100, 40, 1)
                repeat_penalty = st.slider("繰り返しペナルティ", 0.0, 2.0, 1.1, 0.1)
                enabled = st.checkbox("有効", value=True)
            
            if st.form_submit_button("モデルを追加"):
                if model_name and model_name not in settings.models:
                    new_model = ModelConfig(
                        name=model_name,
                        provider=provider,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        top_k=top_k,
                        repeat_penalty=repeat_penalty,
                        enabled=enabled
                    )
                    
                    settings.add_model(new_model)
                    
                    if self.settings_manager.save_settings(settings):
                        st.success(f"モデル '{model_name}' を追加しました")
                        st.rerun()
                    else:
                        st.error("モデルの追加に失敗しました")
                elif not model_name:
                    st.error("モデル名を入力してください")
                else:
                    st.error("同じ名前のモデルが既に存在します")
    
    def _render_ui_settings(self):
        """UI設定"""
        
        st.markdown("### 🎨 UI設定")
        
        settings = self.settings_manager.get_settings()
        ui_config = settings.ui
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_theme = st.selectbox(
                "テーマ",
                ["light", "dark", "auto"],
                index=["light", "dark", "auto"].index(ui_config.theme),
                help="UIのテーマを選択"
            )
            
            new_auto_refresh = st.checkbox(
                "自動リフレッシュ",
                value=ui_config.auto_refresh,
                help="システム統計を自動更新"
            )
            
            new_refresh_interval = st.slider(
                "リフレッシュ間隔（秒）",
                min_value=1,
                max_value=60,
                value=ui_config.refresh_interval,
                help="自動リフレッシュの間隔"
            )
            
            new_show_debug = st.checkbox(
                "デバッグ情報表示",
                value=ui_config.show_debug,
                help="デバッグ情報を表示"
            )
        
        with col2:
            new_auto_save = st.checkbox(
                "自動保存",
                value=ui_config.auto_save,
                help="セッションを自動保存"
            )
            
            new_save_interval = st.slider(
                "保存間隔（分）",
                min_value=1,
                max_value=60,
                value=ui_config.save_interval,
                help="自動保存の間隔"
            )
            
            new_max_chat_history = st.slider(
                "最大チャット履歴数",
                min_value=10,
                max_value=1000,
                value=ui_config.max_chat_history,
                help="保持する最大チャット履歴数"
            )
        
        if st.button("UI設定を更新"):
            ui_config.theme = new_theme
            ui_config.auto_refresh = new_auto_refresh
            ui_config.refresh_interval = new_refresh_interval
            ui_config.auto_save = new_auto_save
            ui_config.save_interval = new_save_interval
            ui_config.show_debug = new_show_debug
            ui_config.max_chat_history = new_max_chat_history
            
            if self.settings_manager.save_settings(settings):
                st.success("UI設定を更新しました")
                st.rerun()
            else:
                st.error("UI設定の更新に失敗しました")
    
    def _render_system_settings(self):
        """システム設定"""
        
        st.markdown("### 🔧 システム設定")
        
        settings = self.settings_manager.get_settings()
        system_config = settings.system
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_api_base_url = st.text_input(
                "API ベース URL",
                value=system_config.api_base_url,
                help="FastAPI サーバーのベース URL"
            )
            
            new_timeout = st.slider(
                "タイムアウト（秒）",
                min_value=5,
                max_value=300,
                value=system_config.timeout,
                help="API リクエストのタイムアウト"
            )
            
            new_max_retries = st.slider(
                "最大リトライ回数",
                min_value=0,
                max_value=10,
                value=system_config.max_retries,
                help="API リクエストの最大リトライ回数"
            )
            
            new_log_level = st.selectbox(
                "ログレベル",
                ["DEBUG", "INFO", "WARNING", "ERROR"],
                index=["DEBUG", "INFO", "WARNING", "ERROR"].index(system_config.log_level),
                help="ログ出力レベル"
            )
        
        with col2:
            new_enable_monitoring = st.checkbox(
                "監視機能有効",
                value=system_config.enable_monitoring,
                help="システム監視機能を有効にする"
            )
            
            new_enable_memory = st.checkbox(
                "記憶機能有効",
                value=system_config.enable_memory,
                help="永続的記憶機能を有効にする"
            )
            
            new_gpu_memory_limit = st.slider(
                "GPU メモリ制限",
                min_value=0.1,
                max_value=1.0,
                value=system_config.gpu_memory_limit,
                step=0.1,
                help="GPU メモリ使用量の制限（比率）"
            )
        
        if st.button("システム設定を更新"):
            system_config.api_base_url = new_api_base_url
            system_config.timeout = new_timeout
            system_config.max_retries = new_max_retries
            system_config.log_level = new_log_level
            system_config.enable_monitoring = new_enable_monitoring
            system_config.enable_memory = new_enable_memory
            system_config.gpu_memory_limit = new_gpu_memory_limit
            
            if self.settings_manager.save_settings(settings):
                st.success("システム設定を更新しました")
                st.rerun()
            else:
                st.error("システム設定の更新に失敗しました")
    
    def _render_backup_restore(self):
        """バックアップ・復元UI - Streamlit の既存ファイル機能による バックアップ・復元 UI"""
        
        st.markdown("### 💾 バックアップ・復元")
        
        # バックアップ作成
        st.markdown("#### 📤 バックアップ作成")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            backup_name = st.text_input(
                "バックアップ名",
                placeholder="例: production_backup",
                help="空の場合は自動生成されます"
            )
        
        with col2:
            if st.button("バックアップ作成", use_container_width=True):
                backup_file = self.settings_manager.create_backup(backup_name or None)
                if backup_file:
                    st.success(f"バックアップを作成しました: {Path(backup_file).name}")
                else:
                    st.error("バックアップの作成に失敗しました")
        
        st.markdown("---")
        
        # バックアップ一覧・復元
        st.markdown("#### 📥 バックアップ一覧・復元")
        
        backups = self.settings_manager.list_backups()
        
        if backups:
            for backup in backups:
                with st.expander(f"📁 {backup['name']} ({backup['created'].strftime('%Y-%m-%d %H:%M:%S')})"):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.text(f"ファイル: {backup['file']}")
                        st.text(f"サイズ: {backup['size']} bytes")
                        st.text(f"更新: {backup['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    with col2:
                        if st.button("復元", key=f"restore_{backup['name']}"):
                            if self.settings_manager.restore_backup(backup['name']):
                                st.success(f"バックアップ '{backup['name']}' から復元しました")
                                st.rerun()
                            else:
                                st.error("復元に失敗しました")
                    
                    with col3:
                        if st.button("削除", key=f"delete_{backup['name']}", type="secondary"):
                            if self.settings_manager.delete_backup(backup['name']):
                                st.success(f"バックアップ '{backup['name']}' を削除しました")
                                st.rerun()
                            else:
                                st.error("削除に失敗しました")
        else:
            st.info("バックアップがありません")
        
        st.markdown("---")
        
        # 設定のエクスポート・インポート
        st.markdown("#### 📋 設定のエクスポート・インポート")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### エクスポート")
            
            export_format = st.selectbox("形式", ["yaml", "json"])
            
            if st.button("設定をエクスポート"):
                exported_data = self.settings_manager.export_settings(export_format)
                if exported_data:
                    st.download_button(
                        label=f"📥 {export_format.upper()} ダウンロード",
                        data=exported_data,
                        file_name=f"agent_settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}",
                        mime=f"application/{export_format}"
                    )
                else:
                    st.error("エクスポートに失敗しました")
        
        with col2:
            st.markdown("##### インポート")
            
            uploaded_file = st.file_uploader(
                "設定ファイルを選択",
                type=["yaml", "yml", "json"],
                help="YAML または JSON 形式の設定ファイル"
            )
            
            if uploaded_file is not None:
                try:
                    file_content = uploaded_file.read().decode('utf-8')
                    file_format = "json" if uploaded_file.name.endswith('.json') else "yaml"
                    
                    if st.button("設定をインポート"):
                        if self.settings_manager.import_settings(file_content, file_format):
                            st.success("設定をインポートしました")
                            st.rerun()
                        else:
                            st.error("インポートに失敗しました")
                            
                except Exception as e:
                    st.error(f"ファイル読み込みエラー: {e}")
        
        st.markdown("---")
        
        # 設定リセット
        st.markdown("#### 🔄 設定リセット")
        
        st.warning("⚠️ この操作により、すべての設定がデフォルト値にリセットされます。")
        
        if st.button("設定をデフォルトにリセット", type="secondary"):
            if self.settings_manager.reset_to_defaults():
                st.success("設定をデフォルトにリセットしました")
                st.rerun()
            else:
                st.error("リセットに失敗しました")


def get_settings_manager() -> SettingsManager:
    """設定管理システムのシングルトンインスタンスを取得"""
    
    if "settings_manager" not in st.session_state:
        st.session_state.settings_manager = SettingsManager()
    
    return st.session_state.settings_manager


def get_settings_ui() -> StreamlitSettingsUI:
    """設定UI のシングルトンインスタンスを取得"""
    
    if "settings_ui" not in st.session_state:
        settings_manager = get_settings_manager()
        st.session_state.settings_ui = StreamlitSettingsUI(settings_manager)
    
    return st.session_state.settings_ui