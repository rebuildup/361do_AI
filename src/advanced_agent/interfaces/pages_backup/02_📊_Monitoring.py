"""
監視ページ

システム監視とパフォーマンス表示
"""

import streamlit as st
import sys
import os

# パス設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.advanced_agent.interfaces.streamlit_ui import StreamlitUI

st.set_page_config(
    page_title="System Monitoring",
    page_icon="📊",
    layout="wide"
)

# UI インスタンス作成
if "ui_instance" not in st.session_state:
    st.session_state.ui_instance = StreamlitUI()

ui = st.session_state.ui_instance

# 監視ダッシュボードのみを表示
ui._initialize_session_state()
ui._apply_custom_css()

st.markdown('<h1 class="main-header">📊 システム監視</h1>', unsafe_allow_html=True)

# リアルタイム進捗・VRAM表示を追加
ui._render_realtime_progress_indicator()

st.markdown("---")

ui._render_monitoring_dashboard()