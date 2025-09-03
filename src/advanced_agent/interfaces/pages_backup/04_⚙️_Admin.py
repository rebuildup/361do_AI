"""
管理ページ

システム管理とコントロールパネル
"""

import streamlit as st
import sys
import os

# パス設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.advanced_agent.interfaces.streamlit_ui import StreamlitUI

st.set_page_config(
    page_title="System Admin",
    page_icon="⚙️",
    layout="wide"
)

# UI インスタンス作成
if "ui_instance" not in st.session_state:
    st.session_state.ui_instance = StreamlitUI()

ui = st.session_state.ui_instance

# 管理パネルのみを表示
ui._initialize_session_state()
ui._apply_custom_css()

st.markdown('<h1 class="main-header">⚙️ システム管理</h1>', unsafe_allow_html=True)

ui._render_admin_panel()