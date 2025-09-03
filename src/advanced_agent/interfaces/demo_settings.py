#!/usr/bin/env python3
"""
Pydantic + Streamlit 設定管理デモ

タスク 8.3: Pydantic Settings による 動的設定変更・反映を統合し、
Streamlit の既存選択機能による モデル選択・切り替えを実装、
バックアップ・復元 UI を統合

使用方法:
    streamlit run src/advanced_agent/interfaces/demo_settings.py
"""

import logging
import sys
import os
from pathlib import Path

# パス設定
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """設定管理デモ メイン関数"""
    
    # ページ設定
    st.set_page_config(
        page_title="Advanced AI Agent - 設定管理デモ",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # タイトル
    st.markdown("""
    # ⚙️ Advanced AI Agent - 設定管理デモ
    
    **タスク 8.3 実装:** Pydantic + Streamlit 設定管理の統合
    
    - ✅ Pydantic Settings による 動的設定変更・反映
    - ✅ Streamlit の既存選択機能による モデル選択・切り替え
    - ✅ バックアップ・復元 UI の統合
    """)
    
    # デモ選択
    demo_type = st.selectbox(
        "デモタイプを選択",
        [
            "完全設定管理システム",
            "モデル設定のみ",
            "バックアップ・復元のみ",
            "設定エクスポート・インポート"
        ]
    )
    
    try:
        if demo_type == "完全設定管理システム":
            run_full_settings_demo()
        elif demo_type == "モデル設定のみ":
            run_model_settings_demo()
        elif demo_type == "バックアップ・復元のみ":
            run_backup_restore_demo()
        elif demo_type == "設定エクスポート・インポート":
            run_export_import_demo()
            
    except Exception as e:
        st.error(f"デモ実行エラー: {e}")
        logger.error(f"デモ実行エラー: {e}", exc_info=True)


def run_full_settings_demo():
    """完全設定管理システムデモ"""
    
    st.markdown("## 🎯 完全設定管理システムデモ")
    
    try:
        from src.advanced_agent.interfaces.settings_manager import (
            get_settings_manager, 
            get_settings_ui
        )
        
        # 設定管理システム初期化
        settings_manager = get_settings_manager()
        settings_ui = get_settings_ui()
        
        # 現在の設定表示
        settings = settings_manager.get_settings()
        
        st.markdown("### 📋 現在の設定概要")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("現在のモデル", settings.current_model)
            st.metric("利用可能モデル数", len(settings.models))
        
        with col2:
            st.metric("テーマ", settings.ui.theme)
            st.metric("自動リフレッシュ", "有効" if settings.ui.auto_refresh else "無効")
        
        with col3:
            st.metric("API URL", settings.system.api_base_url.split("//")[1])
            st.metric("タイムアウト", f"{settings.system.timeout}秒")
        
        st.markdown("---")
        
        # 完全設定UI
        settings_ui.render_settings_panel()
        
    except ImportError as e:
        st.error(f"設定管理システムをインポートできませんでした: {e}")
        run_fallback_settings_demo()


def run_model_settings_demo():
    """モデル設定デモ"""
    
    st.markdown("## 🤖 モデル設定デモ")
    
    try:
        from src.advanced_agent.interfaces.settings_manager import (
            get_settings_manager,
            ModelConfig
        )
        
        settings_manager = get_settings_manager()
        settings = settings_manager.get_settings()
        
        st.markdown("### 現在のモデル設定")
        
        # モデル選択
        model_names = list(settings.models.keys())
        selected_model = st.selectbox(
            "モデルを選択",
            model_names,
            index=model_names.index(settings.current_model) if settings.current_model in model_names else 0
        )
        
        # モデル切り替え
        if selected_model != settings.current_model:
            if settings.switch_model(selected_model):
                settings_manager.save_settings(settings)
                st.success(f"モデルを '{selected_model}' に切り替えました")
                st.rerun()
        
        # 選択されたモデルの詳細
        if selected_model in settings.models:
            model_config = settings.models[selected_model]
            
            st.markdown(f"#### {selected_model} の詳細設定")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.json({
                    "name": model_config.name,
                    "provider": model_config.provider,
                    "temperature": model_config.temperature,
                    "max_tokens": model_config.max_tokens
                })
            
            with col2:
                st.json({
                    "top_p": model_config.top_p,
                    "top_k": model_config.top_k,
                    "repeat_penalty": model_config.repeat_penalty,
                    "enabled": model_config.enabled
                })
        
        # 新しいモデル追加デモ
        st.markdown("---")
        st.markdown("### 新しいモデルを追加")
        
        with st.form("add_model_demo"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_name = st.text_input("モデル名", value="demo-model:latest")
                new_provider = st.selectbox("プロバイダー", ["ollama", "openai"])
                new_temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
            
            with col2:
                new_max_tokens = st.slider("Max Tokens", 100, 2000, 500, 100)
                new_top_p = st.slider("Top-p", 0.0, 1.0, 0.9, 0.1)
                new_enabled = st.checkbox("有効", value=True)
            
            if st.form_submit_button("モデルを追加"):
                if new_name and new_name not in settings.models:
                    new_model = ModelConfig(
                        name=new_name,
                        provider=new_provider,
                        temperature=new_temperature,
                        max_tokens=new_max_tokens,
                        top_p=new_top_p,
                        enabled=new_enabled
                    )
                    
                    settings.add_model(new_model)
                    settings_manager.save_settings(settings)
                    
                    st.success(f"モデル '{new_name}' を追加しました")
                    st.rerun()
                else:
                    st.error("モデル名が無効または既に存在します")
        
    except ImportError as e:
        st.error(f"設定管理システムをインポートできませんでした: {e}")
        run_fallback_model_demo()


def run_backup_restore_demo():
    """バックアップ・復元デモ"""
    
    st.markdown("## 💾 バックアップ・復元デモ")
    
    try:
        from src.advanced_agent.interfaces.settings_manager import get_settings_manager
        
        settings_manager = get_settings_manager()
        
        # バックアップ作成
        st.markdown("### 📤 バックアップ作成")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            backup_name = st.text_input("バックアップ名", placeholder="demo_backup")
        
        with col2:
            if st.button("バックアップ作成"):
                backup_file = settings_manager.create_backup(backup_name or None)
                if backup_file:
                    st.success(f"バックアップを作成しました: {Path(backup_file).name}")
                else:
                    st.error("バックアップの作成に失敗しました")
        
        # バックアップ一覧
        st.markdown("### 📋 バックアップ一覧")
        
        backups = settings_manager.list_backups()
        
        if backups:
            for backup in backups[:5]:  # 最新5件のみ表示
                with st.expander(f"📁 {backup['name']} ({backup['created'].strftime('%Y-%m-%d %H:%M:%S')})"):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.text(f"ファイル: {backup['file']}")
                        st.text(f"サイズ: {backup['size']} bytes")
                    
                    with col2:
                        if st.button("復元", key=f"restore_{backup['name']}"):
                            if settings_manager.restore_backup(backup['name']):
                                st.success("復元しました")
                                st.rerun()
                            else:
                                st.error("復元に失敗しました")
                    
                    with col3:
                        if st.button("削除", key=f"delete_{backup['name']}"):
                            if settings_manager.delete_backup(backup['name']):
                                st.success("削除しました")
                                st.rerun()
                            else:
                                st.error("削除に失敗しました")
        else:
            st.info("バックアップがありません")
        
    except ImportError as e:
        st.error(f"設定管理システムをインポートできませんでした: {e}")
        run_fallback_backup_demo()


def run_export_import_demo():
    """エクスポート・インポートデモ"""
    
    st.markdown("## 📋 設定エクスポート・インポートデモ")
    
    try:
        from src.advanced_agent.interfaces.settings_manager import get_settings_manager
        from datetime import datetime
        
        settings_manager = get_settings_manager()
        
        col1, col2 = st.columns(2)
        
        # エクスポート
        with col1:
            st.markdown("### 📤 エクスポート")
            
            export_format = st.selectbox("形式", ["yaml", "json"])
            
            if st.button("設定をエクスポート"):
                exported_data = settings_manager.export_settings(export_format)
                if exported_data:
                    st.text_area(
                        f"エクスポートされた設定 ({export_format.upper()})",
                        value=exported_data,
                        height=300
                    )
                    
                    st.download_button(
                        label=f"📥 {export_format.upper()} ダウンロード",
                        data=exported_data,
                        file_name=f"demo_settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}",
                        mime=f"application/{export_format}"
                    )
                else:
                    st.error("エクスポートに失敗しました")
        
        # インポート
        with col2:
            st.markdown("### 📥 インポート")
            
            uploaded_file = st.file_uploader(
                "設定ファイルを選択",
                type=["yaml", "yml", "json"]
            )
            
            if uploaded_file is not None:
                try:
                    file_content = uploaded_file.read().decode('utf-8')
                    file_format = "json" if uploaded_file.name.endswith('.json') else "yaml"
                    
                    st.text_area(
                        f"アップロードされた設定 ({file_format.upper()})",
                        value=file_content,
                        height=200
                    )
                    
                    if st.button("設定をインポート"):
                        if settings_manager.import_settings(file_content, file_format):
                            st.success("設定をインポートしました")
                            st.rerun()
                        else:
                            st.error("インポートに失敗しました")
                            
                except Exception as e:
                    st.error(f"ファイル読み込みエラー: {e}")
        
        # 設定リセット
        st.markdown("---")
        st.markdown("### 🔄 設定リセット")
        
        st.warning("⚠️ この操作により、すべての設定がデフォルト値にリセットされます。")
        
        if st.button("設定をデフォルトにリセット"):
            if settings_manager.reset_to_defaults():
                st.success("設定をデフォルトにリセットしました")
                st.rerun()
            else:
                st.error("リセットに失敗しました")
        
    except ImportError as e:
        st.error(f"設定管理システムをインポートできませんでした: {e}")
        run_fallback_export_demo()


def run_fallback_settings_demo():
    """フォールバック設定デモ"""
    
    st.markdown("## 🔧 基本設定デモ（フォールバック）")
    
    st.info("完全な設定管理システムが利用できないため、基本デモを表示しています。")
    
    # 基本的な設定UI
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### モデル設定")
        
        model = st.selectbox(
            "モデル",
            ["deepseek-r1:7b", "qwen2.5:7b-instruct-q4_k_m", "qwen2:1.5b-instruct-q4_k_m"]
        )
        
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        max_tokens = st.slider("Max Tokens", 100, 2000, 500, 100)
        
        if st.button("設定を保存"):
            st.success("設定を保存しました（デモ）")
    
    with col2:
        st.markdown("### UI設定")
        
        theme = st.selectbox("テーマ", ["light", "dark"])
        auto_refresh = st.checkbox("自動リフレッシュ", value=True)
        refresh_interval = st.slider("リフレッシュ間隔", 1, 60, 5)
        
        if st.button("UI設定を保存"):
            st.success("UI設定を保存しました（デモ）")


def run_fallback_model_demo():
    """フォールバックモデルデモ"""
    
    st.markdown("### 🤖 基本モデル設定デモ")
    
    # セッション状態でモデル一覧を管理
    if "demo_models" not in st.session_state:
        st.session_state.demo_models = {
            "deepseek-r1:7b": {"temperature": 0.7, "max_tokens": 500},
            "qwen2.5:7b-instruct-q4_k_m": {"temperature": 0.8, "max_tokens": 1000}
        }
    
    # モデル選択
    selected_model = st.selectbox(
        "モデル選択",
        list(st.session_state.demo_models.keys())
    )
    
    # 選択されたモデルの設定
    if selected_model:
        config = st.session_state.demo_models[selected_model]
        st.json(config)


def run_fallback_backup_demo():
    """フォールバックバックアップデモ"""
    
    st.markdown("### 💾 基本バックアップデモ")
    
    # セッション状態でバックアップを管理
    if "demo_backups" not in st.session_state:
        st.session_state.demo_backups = []
    
    # バックアップ作成
    backup_name = st.text_input("バックアップ名")
    
    if st.button("バックアップ作成（デモ）"):
        if backup_name:
            st.session_state.demo_backups.append({
                "name": backup_name,
                "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.success(f"バックアップ '{backup_name}' を作成しました（デモ）")
            st.rerun()
    
    # バックアップ一覧
    if st.session_state.demo_backups:
        st.markdown("#### バックアップ一覧")
        for backup in st.session_state.demo_backups:
            st.text(f"📁 {backup['name']} ({backup['created']})")


def run_fallback_export_demo():
    """フォールバックエクスポートデモ"""
    
    st.markdown("### 📋 基本エクスポートデモ")
    
    # サンプル設定データ
    sample_config = {
        "current_model": "deepseek-r1:7b",
        "ui": {
            "theme": "light",
            "auto_refresh": True
        },
        "system": {
            "api_base_url": "http://localhost:8000",
            "timeout": 30
        }
    }
    
    format_type = st.selectbox("エクスポート形式", ["yaml", "json"])
    
    if format_type == "json":
        import json
        exported_data = json.dumps(sample_config, indent=2)
    else:
        import yaml
        exported_data = yaml.dump(sample_config, default_flow_style=False)
    
    st.text_area("エクスポートデータ（サンプル）", value=exported_data, height=200)
    
    st.download_button(
        label=f"📥 サンプル設定ダウンロード ({format_type.upper()})",
        data=exported_data,
        file_name=f"sample_config.{format_type}",
        mime=f"application/{format_type}"
    )


if __name__ == "__main__":
    main()