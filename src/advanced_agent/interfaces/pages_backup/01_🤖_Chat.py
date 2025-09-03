"""
チャットページ

AI エージェントとの対話インターフェース
"""

import streamlit as st
import sys
import os

# パス設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.advanced_agent.interfaces.streamlit_ui import StreamlitUI

st.set_page_config(
    page_title="AI Chat",
    page_icon="🤖",
    layout="wide"
)

# UI インスタンス作成
if "ui_instance" not in st.session_state:
    st.session_state.ui_instance = StreamlitUI()

ui = st.session_state.ui_instance

# チャットインターフェースのみを表示
ui._initialize_session_state()
ui._apply_custom_css()

st.markdown('<h1 class="main-header">🤖 AI チャット</h1>', unsafe_allow_html=True)

# リアルタイム進捗・VRAM表示を追加
ui._render_realtime_progress_indicator()

st.markdown("---")

ui._render_chat_interface()