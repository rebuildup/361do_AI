"""
Streamlit アプリケーション ホームページ

Advanced AI Agent の Web UI メインページ
"""

import streamlit as st
import sys
import os
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd

# パス設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.advanced_agent.interfaces.streamlit_ui import StreamlitUI

# ページ設定
st.set_page_config(
    page_title="Advanced AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタム CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
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

.status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
}

.status-online {
    background-color: #28a745;
}

.status-offline {
    background-color: #dc3545;
}

.metric-container {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #dee2e6;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# UI インスタンス作成
if "ui_instance" not in st.session_state:
    st.session_state.ui_instance = StreamlitUI()

ui = st.session_state.ui_instance
ui._initialize_session_state()

# ヘッダー
st.markdown('<h1 class="main-header">🤖 Advanced AI Agent</h1>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.2rem; color: #6c757d;">
        RTX 4050 6GB VRAM 環境で動作する高性能自己学習 AI エージェント
    </p>
</div>
""", unsafe_allow_html=True)

# システムステータス
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("### 🔍 システムステータス")
    
    # ステータス表示
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        st.markdown("""
        <div class="metric-container">
            <span class="status-indicator status-online"></span>
            <strong>AI エージェント</strong><br>
            <small>オンライン</small>
        </div>
        """, unsafe_allow_html=True)
    
    with status_col2:
        st.markdown("""
        <div class="metric-container">
            <span class="status-indicator status-online"></span>
            <strong>API サーバー</strong><br>
            <small>稼働中</small>
        </div>
        """, unsafe_allow_html=True)

# 主要機能紹介
st.markdown("### ✨ 主要機能")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h4>💬 インテリジェントチャット</h4>
        <p>DeepSeek-R1 ベースの高度な推論機能を搭載。Chain-of-Thought による段階的思考プロセスで、複雑な質問にも的確に回答します。</p>
        <ul>
            <li>多言語対応</li>
            <li>コンテキスト理解</li>
            <li>推論過程の可視化</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h4>🔍 永続的記憶システム</h4>
        <p>ChromaDB + LangChain による高度な記憶管理。過去の会話や学習内容を効率的に検索・活用できます。</p>
        <ul>
            <li>セマンティック検索</li>
            <li>重要度ベース記憶</li>
            <li>セッション継続</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h4>📊 リアルタイム監視</h4>
        <p>GPU/CPU 使用率、メモリ消費量、推論速度をリアルタイムで監視。異常検出と自動復旧機能も搭載しています。</p>
        <ul>
            <li>パフォーマンス可視化</li>
            <li>異常検出・アラート</li>
            <li>自動最適化</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h4>🧠 自己学習・進化</h4>
        <p>PEFT + AutoGen による進化的学習システム。使用パターンに応じて自動的に性能を向上させます。</p>
        <ul>
            <li>LoRA アダプタ管理</li>
            <li>進化的最適化</li>
            <li>動的量子化</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# クイックアクセス
st.markdown("### 🚀 クイックアクセス")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("💬 チャット開始", use_container_width=True):
        st.switch_page("pages/01_🤖_Chat.py")

with col2:
    if st.button("📊 監視ダッシュボード", use_container_width=True):
        st.switch_page("pages/02_📊_Monitoring.py")

with col3:
    if st.button("🔍 記憶検索", use_container_width=True):
        st.switch_page("pages/03_🔍_Memory.py")

with col4:
    if st.button("⚙️ システム管理", use_container_width=True):
        st.switch_page("pages/04_⚙️_Admin.py")

# システム概要
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📈 システム概要")
    
    # サンプルメトリクス
    metrics_data = {
        "項目": ["総推論回数", "平均応答時間", "メモリ効率", "稼働時間"],
        "値": ["1,234 回", "1.2 秒", "85%", "24 時間"],
        "ステータス": ["🟢 良好", "🟢 良好", "🟡 注意", "🟢 良好"]
    }
    
    df = pd.DataFrame(metrics_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

with col2:
    st.markdown("### ⚡ リアルタイム統計")
    
    # システム統計取得
    system_stats = ui._get_system_stats_sync()
    gpu_stats = ui._get_gpu_stats_sync()
    
    st.metric("CPU", f"{system_stats.get('cpu_percent', 0):.1f}%")
    st.metric("メモリ", f"{system_stats.get('memory_percent', 0):.1f}%")
    st.metric("GPU", f"{gpu_stats.get('memory_percent', 0):.1f}%")
    st.metric("温度", f"{gpu_stats.get('temperature', 0):.1f}°C")

# フッター
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; margin-top: 2rem;">
    <p>Advanced AI Agent v1.0.0 | RTX 4050 Optimized | 
    <a href="https://github.com/your-repo" target="_blank">GitHub</a> | 
    <a href="/docs" target="_blank">Documentation</a></p>
</div>
""", unsafe_allow_html=True)

# サイドバー - リアルタイム制御統合
with st.sidebar:
    st.markdown("## 🎛️ システム制御")
    
    # 自動リフレッシュ - Streamlit の既存応答性機能による リアルタイム更新
    auto_refresh = st.checkbox("自動リフレッシュ", value=True)
    
    if auto_refresh:
        refresh_interval = st.slider("更新間隔（秒）", 1, 30, 5)
        
        # リアルタイム統計更新
        if st.button("🔄 今すぐ更新"):
            ui._update_system_stats_history()
            st.rerun()
    
    st.markdown("---")
    
    # リアルタイムシステム情報
    st.markdown("### 📋 システム情報")
    st.text(f"起動時刻: {datetime.now().strftime('%H:%M:%S')}")
    st.text(f"セッションID: {st.session_state.current_session_id[:8]}...")
    
    # リアルタイム進捗表示
    st.markdown("### ⚡ リアルタイム状態")
    
    # 簡易進捗バー
    system_stats = ui._get_system_stats_sync()
    gpu_stats = ui._get_gpu_stats_sync()
    
    # VRAM 使用率
    vram_percent = gpu_stats.get("memory_percent", 0)
    st.markdown("**GPU VRAM**")
    st.progress(vram_percent / 100)
    
    if vram_percent >= 90:
        st.error(f"{vram_percent:.1f}% (危険)")
    elif vram_percent >= 75:
        st.warning(f"{vram_percent:.1f}% (注意)")
    else:
        st.success(f"{vram_percent:.1f}% (正常)")
    
    # CPU 使用率
    cpu_percent = system_stats.get("cpu_percent", 0)
    st.markdown("**CPU**")
    st.progress(cpu_percent / 100)
    
    # 緊急停止
    st.markdown("---")
    st.markdown("### 🚨 緊急制御")
    
    if st.button("⏹️ 緊急停止", type="secondary"):
        st.error("緊急停止機能は実装中です")
    
    if st.button("🔄 システム再起動", type="secondary"):
        st.warning("再起動機能は実装中です")
    
    # 自動リフレッシュ実装
    if auto_refresh:
        import time
        time.sleep(refresh_interval)
        st.rerun()