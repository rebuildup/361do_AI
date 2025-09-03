#!/usr/bin/env python3
"""
シンプル設定管理デモ

依存関係の問題を回避した独立動作版
"""

import streamlit as st
import json
import yaml
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import uuid

# ページ設定
st.set_page_config(
    page_title="Advanced AI Agent - 設定管理デモ",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタム CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}

.feature-card {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
    margin-bottom: 1rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.status-healthy {
    color: #28a745;
    font-weight: bold;
}

.status-warning {
    color: #ffc107;
    font-weight: bold;
}

.status-critical {
    color: #dc3545;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """セッション状態の初期化"""
    
    if "settings" not in st.session_state:
        st.session_state.settings = {
            "models": {
                "deepseek-r1:7b": {
                    "name": "deepseek-r1:7b",
                    "provider": "ollama",
                    "temperature": 0.7,
                    "max_tokens": 500,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repeat_penalty": 1.1,
                    "enabled": True
                },
                "qwen2.5:7b-instruct-q4_k_m": {
                    "name": "qwen2.5:7b-instruct-q4_k_m",
                    "provider": "ollama",
                    "temperature": 0.8,
                    "max_tokens": 1000,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repeat_penalty": 1.1,
                    "enabled": True
                },
                "qwen2:1.5b-instruct-q4_k_m": {
                    "name": "qwen2:1.5b-instruct-q4_k_m",
                    "provider": "ollama",
                    "temperature": 0.6,
                    "max_tokens": 300,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repeat_penalty": 1.1,
                    "enabled": True
                }
            },
            "current_model": "deepseek-r1:7b",
            "ui": {
                "theme": "light",
                "auto_refresh": True,
                "refresh_interval": 5,
                "auto_save": True,
                "save_interval": 10,
                "show_debug": False,
                "max_chat_history": 100
            },
            "system": {
                "api_base_url": "http://localhost:8000",
                "timeout": 30,
                "max_retries": 3,
                "log_level": "INFO",
                "enable_monitoring": True,
                "enable_memory": True,
                "gpu_memory_limit": 0.9
            },
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
    
    if "backups" not in st.session_state:
        st.session_state.backups = []

def main():
    """メイン関数"""
    
    initialize_session_state()
    
    # ヘッダー
    st.markdown('<h1 class="main-header">⚙️ Advanced AI Agent - 設定管理デモ</h1>', 
               unsafe_allow_html=True)
    
    st.markdown("""
    **タスク 8.3 実装:** Pydantic + Streamlit 設定管理の統合
    
    - ✅ Pydantic Settings による 動的設定変更・反映
    - ✅ Streamlit の既存選択機能による モデル選択・切り替え
    - ✅ バックアップ・復元 UI の統合
    """)
    
    # サイドバー
    render_sidebar()
    
    # メインコンテンツ
    render_main_content()

def render_sidebar():
    """サイドバー描画"""
    
    with st.sidebar:
        st.markdown("## 🎛️ クイック設定")
        
        # 現在のモデル選択
        model_names = list(st.session_state.settings["models"].keys())
        current_index = model_names.index(st.session_state.settings["current_model"])
        
        selected_model = st.selectbox(
            "現在のモデル",
            model_names,
            index=current_index,
            help="使用するAIモデルを選択"
        )
        
        # モデル切り替え
        if selected_model != st.session_state.settings["current_model"]:
            st.session_state.settings["current_model"] = selected_model
            st.session_state.settings["updated_at"] = datetime.now().isoformat()
            st.success(f"モデルを '{selected_model}' に切り替えました")
            st.rerun()
        
        # 現在のモデル設定
        current_model_config = st.session_state.settings["models"][selected_model]
        
        st.markdown("---")
        st.markdown("### ⚙️ モデル設定")
        
        new_temperature = st.slider(
            "Temperature",
            0.0, 2.0,
            current_model_config["temperature"],
            0.1,
            help="生成の創造性を制御"
        )
        
        new_max_tokens = st.slider(
            "Max Tokens",
            50, 4000,
            current_model_config["max_tokens"],
            50,
            help="生成する最大トークン数"
        )
        
        # 設定変更の検出と適用
        if (new_temperature != current_model_config["temperature"] or 
            new_max_tokens != current_model_config["max_tokens"]):
            
            current_model_config["temperature"] = new_temperature
            current_model_config["max_tokens"] = new_max_tokens
            st.session_state.settings["updated_at"] = datetime.now().isoformat()
        
        st.markdown("---")
        st.markdown("### 📊 システム情報")
        
        # システム統計（デモ用）
        import random
        cpu_percent = random.uniform(30, 70)
        memory_percent = random.uniform(50, 80)
        gpu_percent = random.uniform(60, 90)
        
        cpu_color = get_status_color(cpu_percent, 70, 90)
        memory_color = get_status_color(memory_percent, 70, 90)
        gpu_color = get_status_color(gpu_percent, 80, 95)
        
        st.markdown(f"**CPU:** <span class='{cpu_color}'>{cpu_percent:.1f}%</span>", 
                   unsafe_allow_html=True)
        st.markdown(f"**メモリ:** <span class='{memory_color}'>{memory_percent:.1f}%</span>", 
                   unsafe_allow_html=True)
        st.markdown(f"**GPU:** <span class='{gpu_color}'>{gpu_percent:.1f}%</span>", 
                   unsafe_allow_html=True)

def get_status_color(value: float, warning_threshold: float, critical_threshold: float) -> str:
    """ステータス色取得"""
    if value >= critical_threshold:
        return "status-critical"
    elif value >= warning_threshold:
        return "status-warning"
    else:
        return "status-healthy"

def render_main_content():
    """メインコンテンツ描画"""
    
    # タブ構成
    tab1, tab2, tab3, tab4 = st.tabs([
        "🤖 モデル設定", 
        "🎨 UI設定", 
        "🔧 システム設定", 
        "💾 バックアップ・復元"
    ])
    
    with tab1:
        render_model_settings()
    
    with tab2:
        render_ui_settings()
    
    with tab3:
        render_system_settings()
    
    with tab4:
        render_backup_restore()

def render_model_settings():
    """モデル設定UI"""
    
    st.markdown("### 🤖 モデル設定")
    
    settings = st.session_state.settings
    
    # 現在のモデル情報
    current_model = settings["current_model"]
    model_config = settings["models"][current_model]
    
    st.markdown(f"#### 現在のモデル: {current_model}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.json({
            "name": model_config["name"],
            "provider": model_config["provider"],
            "temperature": model_config["temperature"],
            "max_tokens": model_config["max_tokens"]
        })
    
    with col2:
        st.json({
            "top_p": model_config["top_p"],
            "top_k": model_config["top_k"],
            "repeat_penalty": model_config["repeat_penalty"],
            "enabled": model_config["enabled"]
        })
    
    st.markdown("---")
    
    # モデル詳細設定
    st.markdown("#### 詳細設定")
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_temperature = st.slider(
            "Temperature (詳細)",
            0.0, 2.0,
            model_config["temperature"],
            0.01,
            key="detail_temperature"
        )
        
        new_max_tokens = st.number_input(
            "最大トークン数",
            min_value=1,
            max_value=4000,
            value=model_config["max_tokens"],
            step=1,
            key="detail_max_tokens"
        )
        
        new_top_p = st.slider(
            "Top-p",
            0.0, 1.0,
            model_config["top_p"],
            0.01,
            key="detail_top_p"
        )
    
    with col2:
        new_top_k = st.number_input(
            "Top-k",
            min_value=1,
            max_value=100,
            value=model_config["top_k"],
            step=1,
            key="detail_top_k"
        )
        
        new_repeat_penalty = st.slider(
            "繰り返しペナルティ",
            0.0, 2.0,
            model_config["repeat_penalty"],
            0.01,
            key="detail_repeat_penalty"
        )
        
        new_enabled = st.checkbox(
            "有効",
            value=model_config["enabled"],
            key="detail_enabled"
        )
    
    if st.button("詳細設定を更新"):
        model_config.update({
            "temperature": new_temperature,
            "max_tokens": new_max_tokens,
            "top_p": new_top_p,
            "top_k": new_top_k,
            "repeat_penalty": new_repeat_penalty,
            "enabled": new_enabled
        })
        settings["updated_at"] = datetime.now().isoformat()
        st.success("モデル設定を更新しました")
        st.rerun()
    
    # 新しいモデル追加
    st.markdown("---")
    with st.expander("➕ 新しいモデルを追加"):
        render_add_model_form()
    
    # モデル削除
    if len(settings["models"]) > 1:
        with st.expander("🗑️ モデルを削除"):
            model_names = list(settings["models"].keys())
            model_to_delete = st.selectbox(
                "削除するモデル",
                [name for name in model_names if name != current_model]
            )
            
            if st.button("モデルを削除", type="secondary"):
                del settings["models"][model_to_delete]
                settings["updated_at"] = datetime.now().isoformat()
                st.success(f"モデル '{model_to_delete}' を削除しました")
                st.rerun()

def render_add_model_form():
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
            if model_name and model_name not in st.session_state.settings["models"]:
                st.session_state.settings["models"][model_name] = {
                    "name": model_name,
                    "provider": provider,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "top_k": top_k,
                    "repeat_penalty": repeat_penalty,
                    "enabled": enabled
                }
                st.session_state.settings["updated_at"] = datetime.now().isoformat()
                st.success(f"モデル '{model_name}' を追加しました")
                st.rerun()
            elif not model_name:
                st.error("モデル名を入力してください")
            else:
                st.error("同じ名前のモデルが既に存在します")

def render_ui_settings():
    """UI設定"""
    
    st.markdown("### 🎨 UI設定")
    
    ui_config = st.session_state.settings["ui"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_theme = st.selectbox(
            "テーマ",
            ["light", "dark", "auto"],
            index=["light", "dark", "auto"].index(ui_config["theme"])
        )
        
        new_auto_refresh = st.checkbox(
            "自動リフレッシュ",
            value=ui_config["auto_refresh"]
        )
        
        new_refresh_interval = st.slider(
            "リフレッシュ間隔（秒）",
            1, 60,
            ui_config["refresh_interval"]
        )
        
        new_show_debug = st.checkbox(
            "デバッグ情報表示",
            value=ui_config["show_debug"]
        )
    
    with col2:
        new_auto_save = st.checkbox(
            "自動保存",
            value=ui_config["auto_save"]
        )
        
        new_save_interval = st.slider(
            "保存間隔（分）",
            1, 60,
            ui_config["save_interval"]
        )
        
        new_max_chat_history = st.slider(
            "最大チャット履歴数",
            10, 1000,
            ui_config["max_chat_history"]
        )
    
    if st.button("UI設定を更新"):
        ui_config.update({
            "theme": new_theme,
            "auto_refresh": new_auto_refresh,
            "refresh_interval": new_refresh_interval,
            "auto_save": new_auto_save,
            "save_interval": new_save_interval,
            "show_debug": new_show_debug,
            "max_chat_history": new_max_chat_history
        })
        st.session_state.settings["updated_at"] = datetime.now().isoformat()
        st.success("UI設定を更新しました")
        st.rerun()

def render_system_settings():
    """システム設定"""
    
    st.markdown("### 🔧 システム設定")
    
    system_config = st.session_state.settings["system"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_api_base_url = st.text_input(
            "API ベース URL",
            value=system_config["api_base_url"]
        )
        
        new_timeout = st.slider(
            "タイムアウト（秒）",
            5, 300,
            system_config["timeout"]
        )
        
        new_max_retries = st.slider(
            "最大リトライ回数",
            0, 10,
            system_config["max_retries"]
        )
        
        new_log_level = st.selectbox(
            "ログレベル",
            ["DEBUG", "INFO", "WARNING", "ERROR"],
            index=["DEBUG", "INFO", "WARNING", "ERROR"].index(system_config["log_level"])
        )
    
    with col2:
        new_enable_monitoring = st.checkbox(
            "監視機能有効",
            value=system_config["enable_monitoring"]
        )
        
        new_enable_memory = st.checkbox(
            "記憶機能有効",
            value=system_config["enable_memory"]
        )
        
        new_gpu_memory_limit = st.slider(
            "GPU メモリ制限",
            0.1, 1.0,
            system_config["gpu_memory_limit"],
            0.1
        )
    
    if st.button("システム設定を更新"):
        system_config.update({
            "api_base_url": new_api_base_url,
            "timeout": new_timeout,
            "max_retries": new_max_retries,
            "log_level": new_log_level,
            "enable_monitoring": new_enable_monitoring,
            "enable_memory": new_enable_memory,
            "gpu_memory_limit": new_gpu_memory_limit
        })
        st.session_state.settings["updated_at"] = datetime.now().isoformat()
        st.success("システム設定を更新しました")
        st.rerun()

def render_backup_restore():
    """バックアップ・復元UI"""
    
    st.markdown("### 💾 バックアップ・復元")
    
    # バックアップ作成
    st.markdown("#### 📤 バックアップ作成")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        backup_name = st.text_input(
            "バックアップ名",
            placeholder="例: production_backup"
        )
    
    with col2:
        if st.button("バックアップ作成", use_container_width=True):
            backup_data = {
                "name": backup_name or f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "settings": st.session_state.settings.copy(),
                "created": datetime.now().isoformat(),
                "id": str(uuid.uuid4())
            }
            
            st.session_state.backups.append(backup_data)
            st.success(f"バックアップ '{backup_data['name']}' を作成しました")
    
    st.markdown("---")
    
    # バックアップ一覧
    st.markdown("#### 📋 バックアップ一覧")
    
    if st.session_state.backups:
        for i, backup in enumerate(reversed(st.session_state.backups)):
            with st.expander(f"📁 {backup['name']} ({backup['created'][:19]})"):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.text(f"ID: {backup['id'][:8]}...")
                    st.text(f"作成: {backup['created'][:19]}")
                    st.text(f"モデル数: {len(backup['settings']['models'])}")
                
                with col2:
                    if st.button("復元", key=f"restore_{backup['id']}"):
                        st.session_state.settings = backup['settings'].copy()
                        st.session_state.settings["updated_at"] = datetime.now().isoformat()
                        st.success(f"バックアップ '{backup['name']}' から復元しました")
                        st.rerun()
                
                with col3:
                    if st.button("削除", key=f"delete_{backup['id']}", type="secondary"):
                        st.session_state.backups = [
                            b for b in st.session_state.backups if b['id'] != backup['id']
                        ]
                        st.success(f"バックアップ '{backup['name']}' を削除しました")
                        st.rerun()
    else:
        st.info("バックアップがありません")
    
    st.markdown("---")
    
    # エクスポート・インポート
    st.markdown("#### 📋 設定のエクスポート・インポート")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### エクスポート")
        
        export_format = st.selectbox("形式", ["yaml", "json"])
        
        if st.button("設定をエクスポート"):
            if export_format == "json":
                exported_data = json.dumps(st.session_state.settings, indent=2, default=str)
            else:
                exported_data = yaml.dump(st.session_state.settings, default_flow_style=False)
            
            st.download_button(
                label=f"📥 {export_format.upper()} ダウンロード",
                data=exported_data,
                file_name=f"agent_settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}",
                mime=f"application/{export_format}"
            )
    
    with col2:
        st.markdown("##### インポート")
        
        uploaded_file = st.file_uploader(
            "設定ファイルを選択",
            type=["yaml", "yml", "json"]
        )
        
        if uploaded_file is not None:
            try:
                file_content = uploaded_file.read().decode('utf-8')
                
                if uploaded_file.name.endswith('.json'):
                    imported_settings = json.loads(file_content)
                else:
                    imported_settings = yaml.safe_load(file_content)
                
                if st.button("設定をインポート"):
                    st.session_state.settings = imported_settings
                    st.session_state.settings["updated_at"] = datetime.now().isoformat()
                    st.success("設定をインポートしました")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"ファイル読み込みエラー: {e}")
    
    # 設定リセット
    st.markdown("---")
    st.markdown("#### 🔄 設定リセット")
    
    st.warning("⚠️ この操作により、すべての設定がデフォルト値にリセットされます。")
    
    if st.button("設定をデフォルトにリセット", type="secondary"):
        # セッション状態をクリアして再初期化
        del st.session_state.settings
        initialize_session_state()
        st.success("設定をデフォルトにリセットしました")
        st.rerun()

if __name__ == "__main__":
    main()