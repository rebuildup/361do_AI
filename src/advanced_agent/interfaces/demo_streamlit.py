#!/usr/bin/env python3
"""
Streamlit リアルタイム Web UI デモ

タスク 8.2: Streamlit の既存応答性機能による フロントエンドを統合し、
進捗・VRAM 表示を実装、履歴管理・継続を統合

使用方法:
    streamlit run src/advanced_agent/interfaces/demo_streamlit.py
"""

import asyncio
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
    """デモアプリケーション メイン関数"""
    
    # ページ設定
    st.set_page_config(
        page_title="Advanced AI Agent - リアルタイム Web UI デモ",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # タイトル
    st.markdown("""
    # 🤖 Advanced AI Agent - リアルタイム Web UI デモ
    
    **タスク 8.2 実装:** Streamlit の既存応答性機能による フロントエンドを統合
    
    - ✅ リアルタイム応答性機能の統合
    - ✅ 進捗・VRAM 表示の実装  
    - ✅ セッション管理・履歴継続の統合
    """)
    
    # デモ選択
    demo_type = st.selectbox(
        "デモタイプを選択",
        [
            "完全統合 UI",
            "リアルタイム監視のみ",
            "チャット機能のみ",
            "セッション管理のみ"
        ]
    )
    
    try:
        if demo_type == "完全統合 UI":
            run_full_ui_demo()
        elif demo_type == "リアルタイム監視のみ":
            run_monitoring_demo()
        elif demo_type == "チャット機能のみ":
            run_chat_demo()
        elif demo_type == "セッション管理のみ":
            run_session_demo()
            
    except Exception as e:
        st.error(f"デモ実行エラー: {e}")
        logger.error(f"デモ実行エラー: {e}", exc_info=True)


def run_full_ui_demo():
    """完全統合 UI デモ"""
    
    st.markdown("## 🎯 完全統合 UI デモ")
    
    try:
        from src.advanced_agent.interfaces.streamlit_ui import StreamlitUI
        
        # UI インスタンス作成
        if "ui_instance" not in st.session_state:
            st.session_state.ui_instance = StreamlitUI()
        
        ui = st.session_state.ui_instance
        
        # 完全 UI 実行
        ui.run()
        
    except ImportError as e:
        st.error(f"モジュールインポートエラー: {e}")
        st.info("フォールバック: 基本デモを表示します")
        run_fallback_demo()


def run_monitoring_demo():
    """リアルタイム監視デモ"""
    
    st.markdown("## 📊 リアルタイム監視デモ")
    
    try:
        from src.advanced_agent.interfaces.streamlit_ui import StreamlitUI
        
        if "ui_instance" not in st.session_state:
            st.session_state.ui_instance = StreamlitUI()
        
        ui = st.session_state.ui_instance
        ui._initialize_session_state()
        
        # リアルタイム進捗表示
        st.markdown("### ⚡ リアルタイム進捗・VRAM表示")
        ui._render_realtime_progress_indicator()
        
        st.markdown("---")
        
        # 監視ダッシュボード
        st.markdown("### 📈 監視ダッシュボード")
        ui._render_monitoring_dashboard()
        
    except ImportError as e:
        st.error(f"モジュールインポートエラー: {e}")
        run_fallback_monitoring_demo()


def run_chat_demo():
    """チャット機能デモ"""
    
    st.markdown("## 💬 チャット機能デモ")
    
    try:
        from src.advanced_agent.interfaces.streamlit_ui import StreamlitUI
        
        if "ui_instance" not in st.session_state:
            st.session_state.ui_instance = StreamlitUI()
        
        ui = st.session_state.ui_instance
        ui._initialize_session_state()
        
        # チャットインターフェース
        ui._render_chat_interface()
        
    except ImportError as e:
        st.error(f"モジュールインポートエラー: {e}")
        run_fallback_chat_demo()


def run_session_demo():
    """セッション管理デモ"""
    
    st.markdown("## 👤 セッション管理デモ")
    
    try:
        from src.advanced_agent.interfaces.streamlit_ui import StreamlitUI
        
        if "ui_instance" not in st.session_state:
            st.session_state.ui_instance = StreamlitUI()
        
        ui = st.session_state.ui_instance
        ui._initialize_session_state()
        
        # セッション管理
        ui._render_session_management()
        
    except ImportError as e:
        st.error(f"モジュールインポートエラー: {e}")
        run_fallback_session_demo()


def run_fallback_demo():
    """フォールバック基本デモ"""
    
    st.markdown("## 🔧 基本デモ（フォールバック）")
    
    st.info("完全な統合モジュールが利用できないため、基本デモを表示しています。")
    
    # 基本的なリアルタイム機能デモ
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("CPU 使用率", "45.2%", "2.1%")
    
    with col2:
        st.metric("メモリ使用率", "67.8%", "-1.3%")
    
    with col3:
        st.metric("GPU VRAM", "72.5%", "5.2%")
    
    # 簡単なチャート
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # サンプルデータ
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(minutes=10),
        end=datetime.now(),
        freq='1min'
    )
    
    data = pd.DataFrame({
        'Time': timestamps,
        'CPU': np.random.normal(50, 10, len(timestamps)),
        'Memory': np.random.normal(65, 8, len(timestamps)),
        'GPU': np.random.normal(70, 12, len(timestamps))
    })
    
    st.line_chart(data.set_index('Time'))


def run_fallback_monitoring_demo():
    """フォールバック監視デモ"""
    
    st.markdown("### 📊 基本監視デモ")
    
    # 進捗バー
    import time
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**GPU VRAM**")
        vram_progress = st.progress(0.72)
        st.success("72% (正常)")
    
    with col2:
        st.markdown("**CPU**")
        cpu_progress = st.progress(0.45)
        st.success("45% (正常)")
    
    with col3:
        st.markdown("**メモリ**")
        memory_progress = st.progress(0.68)
        st.success("68% (正常)")


def run_fallback_chat_demo():
    """フォールバックチャットデモ"""
    
    st.markdown("### 💬 基本チャットデモ")
    
    # セッション状態初期化
    if "demo_messages" not in st.session_state:
        st.session_state.demo_messages = []
    
    # チャット履歴表示
    for message in st.session_state.demo_messages:
        if message["role"] == "user":
            st.markdown(f"**👤 ユーザー:** {message['content']}")
        else:
            st.markdown(f"**🤖 AI:** {message['content']}")
    
    # 入力フォーム
    with st.form("demo_chat_form"):
        user_input = st.text_input("メッセージを入力...")
        submit_button = st.form_submit_button("送信")
    
    if submit_button and user_input:
        # ユーザーメッセージ追加
        st.session_state.demo_messages.append({
            "role": "user",
            "content": user_input
        })
        
        # AI レスポンス（デモ）
        st.session_state.demo_messages.append({
            "role": "assistant",
            "content": f"デモレスポンス: '{user_input}' について理解しました。"
        })
        
        st.rerun()


def run_fallback_session_demo():
    """フォールバックセッションデモ"""
    
    st.markdown("### 👤 基本セッションデモ")
    
    # セッション情報
    if "demo_session_id" not in st.session_state:
        import uuid
        st.session_state.demo_session_id = str(uuid.uuid4())
    
    st.text(f"セッション ID: {st.session_state.demo_session_id[:8]}...")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("新規セッション"):
            import uuid
            st.session_state.demo_session_id = str(uuid.uuid4())
            st.success("新しいセッションを作成しました")
            st.rerun()
    
    with col2:
        if st.button("セッション保存"):
            st.success("セッションを保存しました（デモ）")


if __name__ == "__main__":
    main()