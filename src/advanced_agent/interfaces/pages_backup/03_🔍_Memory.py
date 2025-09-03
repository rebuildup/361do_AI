"""
記憶検索ページ

AI エージェントの記憶検索インターフェース
"""

import streamlit as st
import sys
import os

# パス設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.advanced_agent.interfaces.streamlit_ui import StreamlitUI

st.set_page_config(
    page_title="Memory Search",
    page_icon="🔍",
    layout="wide"
)

# UI インスタンス作成
if "ui_instance" not in st.session_state:
    st.session_state.ui_instance = StreamlitUI()

ui = st.session_state.ui_instance

# 記憶検索インターフェースのみを表示
ui._initialize_session_state()
ui._apply_custom_css()

st.markdown('<h1 class="main-header">🔍 記憶検索</h1>', unsafe_allow_html=True)

ui._render_memory_search()