import streamlit as st
import time
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 自己学習AIエージェントのインポート
from src.advanced_agent.core.self_learning_agent import SelfLearningAgent, AgentState
from src.advanced_agent.core.logger import get_logger

# ロガー初期化
logger = get_logger()

# ページ設定
st.set_page_config(
    page_title="自己学習AIエージェント",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# layout.htmlのCSSスタイルを適用
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" />
<style>
    :root {
        --sidebar-width-expanded: 288px;
        --sidebar-width-collapsed: 48px;
        --main-content-width: 768px;
        --chat-panel-height: 80px;
    }

    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    html, body {
        height: 100%;
        overflow: hidden;
    }

    body {
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Streamlitのデフォルトスタイルを調整 */
    .stApp {
        height: 100vh !important;
        overflow: hidden !important;
    }

    /* Streamlitのデフォルトバナーを非表示 */
    .stApp > header {
        display: none !important;
    }
    
    /* Streamlitのデフォルトバナー（新しいバージョン） */
    [data-testid="stHeader"] {
        display: none !important;
    }
    
    /* Streamlitのデプロイボタンを非表示 */
    [data-testid="stDeployButton"] {
        display: none !important;
    }
    
    /* Streamlitのメニューボタンを非表示 */
    [data-testid="stToolbar"] {
        display: none !important;
    }
    
    /* Streamlitのデフォルトサイドバー開閉ボタンを非表示 */
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }
    

    .stApp > div {
        display: flex !important;
        height: auto !important;
        min-height: 100vh !important;
        overflow: visible !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* サイドバー */
    .stSidebar,
    [data-testid="stSidebar"] {
        width: var(--sidebar-width-expanded) !important;
        min-width: var(--sidebar-width-expanded) !important;
        max-width: var(--sidebar-width-expanded) !important;
        transition: width 0.3s ease !important;
        position: relative !important;
        z-index: 1000 !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        padding: 20px 0 0 0 !important;
    }

    /* サイドバーが閉じられた状態 */
    .stSidebar.collapsed,
    [data-testid="stSidebar"].collapsed {
        width: var(--sidebar-width-collapsed) !important;
        min-width: var(--sidebar-width-collapsed) !important;
        max-width: var(--sidebar-width-collapsed) !important;
    }
    
    /* サイドバー内のコンテンツの表示制御 */
    .stSidebar.collapsed .stMarkdown:not(:first-child),
    [data-testid="stSidebar"].collapsed .stMarkdown:not(:first-child) {
        display: none !important;
    }
    
    .stSidebar.collapsed .stAlert,
    [data-testid="stSidebar"].collapsed .stAlert {
        display: none !important;
    }
    
    /* サイドバー内の最初のボタンは常に表示 */
    .stSidebar .stButton:first-child,
    [data-testid="stSidebar"] .stButton:first-child {
        display: block !important;
    }


    /* メインコンテンツエリア */
    .stAppViewContainer,
    [data-testid="stAppViewContainer"] {
        flex: 1 !important;
        display: flex !important;
        height: auto !important;
        min-height: 100vh !important;
        overflow: visible !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* メインコンテナ */
    .main-container {
        width: var(--main-content-width) !important;
        max-width: var(--main-content-width) !important;
        min-width: var(--main-content-width) !important;
        min-height: auto !important;
        height: auto !important;
        position: relative !important;
        display: flex !important;
        flex-direction: column !important;
        margin: 0 auto !important;
        box-sizing: border-box !important;
        padding: 0 !important;
    }

    /* チャットエリアの最大幅を768pxに制限 */
    .stChatMessage,
    [data-testid="stChatMessage"] {
        max-width: var(--main-content-width) !important;
        margin: 0 auto !important;
    }

    /* チャットボックスコンテナ - 画面下固定 */
    [data-testid="stBottomBlockContainer"] {
        position: fixed !important;
        bottom: 0 !important;
        z-index: 1000 !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        padding: 20px !important;
        background: var(--background-color) !important;
        border-top: 1px solid var(--border-color) !important;
        transition: left 0.3s ease, width 0.3s ease !important;
        box-sizing: border-box !important;
        /* デフォルトはサイドバーが開いている状態 */
        left: var(--sidebar-width-expanded) !important;
        width: calc(100vw - var(--sidebar-width-expanded)) !important;
        /* 画面幅いっぱいに表示 */
        right: 0 !important;
    }
    
    /* サイドバーが閉じられた状態の検出（幅ベース） */
    [data-testid="stSidebar"][style*="width: 48px"] ~ * [data-testid="stBottomBlockContainer"],
    [data-testid="stSidebar"][style*="width: 48px"] + * [data-testid="stBottomBlockContainer"] {
        left: var(--sidebar-width-collapsed) !important;
        width: calc(100vw - var(--sidebar-width-collapsed)) !important;
        right: 0 !important;
    }
    
    /* より確実な検出方法：サイドバーの実際の幅が48pxの場合 */
    [data-testid="stSidebar"]:not([style*="width: 288px"]):not([style*="width: 256px"]) ~ * [data-testid="stBottomBlockContainer"],
    [data-testid="stSidebar"]:not([style*="width: 288px"]):not([style*="width: 256px"]) + * [data-testid="stBottomBlockContainer"] {
        left: var(--sidebar-width-collapsed) !important;
        width: calc(100vw - var(--sidebar-width-collapsed)) !important;
        right: 0 !important;
    }
    
    /* 最も確実な検出方法：サイドバーの幅が48pxの場合（直接指定） */
    [data-testid="stSidebar"][style*="width: 256px"] ~ * [data-testid="stBottomBlockContainer"] {
        left: var(--sidebar-width-collapsed) !important;
        width: calc(100vw - var(--sidebar-width-collapsed)) !important;
        right: 0 !important;
    }
    
    /* サイドバーが開いている状態の明示的な指定 */
    .stApp [data-testid="stSidebar"][aria-expanded="true"] ~ * [data-testid="stBottomBlockContainer"],
    .stApp:has([data-testid="stSidebar"][aria-expanded="true"]) [data-testid="stBottomBlockContainer"] {
        left: var(--sidebar-width-expanded) !important;
        width: calc(100vw - var(--sidebar-width-expanded)) !important;
        right: 0 !important;
    }
    
    /* サイドバーが閉じられた状態の検出（幅ベース） */
    .stApp [data-testid="stSidebar"][style*="width: 48px"] ~ * [data-testid="stBottomBlockContainer"],
    .stApp:has([data-testid="stSidebar"][style*="width: 48px"]) [data-testid="stBottomBlockContainer"] {
        left: var(--sidebar-width-collapsed) !important;
        width: calc(100vw - var(--sidebar-width-collapsed)) !important;
        right: 0 !important;
    }
    
    /* より具体的なセレクタでサイドバーの状態を検出 */
    .stApp .stSidebar[aria-expanded="false"] ~ * [data-testid="stBottomBlockContainer"],
    .stApp:has(.stSidebar[aria-expanded="false"]) [data-testid="stBottomBlockContainer"] {
        left: var(--sidebar-width-collapsed) !important;
        width: calc(100vw - var(--sidebar-width-collapsed)) !important;
        right: 0 !important;
    }
    
    /* Streamlitのサイドバーが閉じられた状態の検出（より確実な方法） */
    .stApp .stSidebar[style*="display: none"] ~ * [data-testid="stBottomBlockContainer"],
    .stApp .stSidebar[style*="visibility: hidden"] ~ * [data-testid="stBottomBlockContainer"],
    .stApp .stSidebar[style*="transform: translateX(-100%)"] ~ * [data-testid="stBottomBlockContainer"] {
        left: var(--sidebar-width-collapsed) !important;
        width: calc(100vw - var(--sidebar-width-collapsed)) !important;
        right: 0 !important;
    }
    
    /* サイドバーが開いている状態（デフォルト） */
    [data-testid="stBottomBlockContainer"].sidebar-expanded {
        left: var(--sidebar-width-expanded) !important;
        width: calc(100vw - var(--sidebar-width-expanded)) !important;
        right: 0 !important;
    }
    
    /* サイドバーが閉じられた状態 */
    [data-testid="stBottomBlockContainer"].sidebar-collapsed {
        left: var(--sidebar-width-collapsed) !important;
        width: calc(100vw - var(--sidebar-width-collapsed)) !important;
        right: 0 !important;
    }
    
    /* フォールバック: data属性ベースのスタイル */
    .stApp[data-sidebar-state="expanded"] [data-testid="stBottomBlockContainer"] {
        left: var(--sidebar-width-expanded) !important;
        width: calc(100vw - var(--sidebar-width-expanded)) !important;
        right: 0 !important;
    }
    
    .stApp[data-sidebar-state="collapsed"] [data-testid="stBottomBlockContainer"] {
        left: var(--sidebar-width-collapsed) !important;
        width: calc(100vw - var(--sidebar-width-collapsed)) !important;
        right: 0 !important;
    }
    
    /* チャットボックス - コンテナ内で中央配置 */
    [data-testid="stBottomBlockContainer"] .stChatInput,
    [data-testid="stBottomBlockContainer"] [data-testid="stChatInput"] {
        width: 800px !important;
        max-width: 800px !important;
        background: var(--background-color) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 15px !important;
        box-sizing: border-box !important;
        margin: 0 !important;
    }

    /* スクロール可能なメインコンテンツ */
    .main-content {
        flex: none !important;
        overflow: visible !important;
        padding: 15px 15px 140px 15px !important;
        margin-bottom: 0 !important;
        height: auto !important;
        min-height: auto !important;
    }

    .page-header {
        margin-bottom: 10px !important;
        padding: 0 !important;
    }

    .page-title {
        font-size: 1.8em !important;
        margin-bottom: 5px !important;
        text-align: center !important;
    }

    .page-description {
        line-height: 1.4 !important;
        margin-bottom: 10px !important;
    }

    /* コンテンツセクションのスタイル */
    .content-sections {
        margin-top: 10px !important;
    }
    
    .content-section {
        margin-bottom: 15px !important;
        padding: 12px !important;
        background: #f8f9fa !important;
        border-radius: 6px !important;
        border-left: 3px solid #1f77b4 !important;
    }
    
    .content-section h3 {
        margin-top: 0 !important;
        margin-bottom: 8px !important;
        color: #1f77b4 !important;
        font-size: 1.1em !important;
    }
    
    .content-section p {
        margin-bottom: 6px !important;
    }
    
    .content-section ul {
        margin-bottom: 0 !important;
        padding-left: 18px !important;
    }
    
    .content-section li {
        margin-bottom: 3px !important;
    }


    /* Streamlitの要素は制限しない */
    .stMain,
    [data-testid="stMain"] {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 100% !important;
        height: auto !important;
        min-height: auto !important;
        margin: 0 !important;
        padding: 0 !important;
        box-sizing: border-box !important;
    }

    /* Streamlitのブロックコンテナも制限しない */
    .stVerticalBlock,
    [data-testid="stVerticalBlock"] {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 100% !important;
        height: auto !important;
        min-height: auto !important;
        margin: 0 !important;
        padding: 0 !important;
        box-sizing: border-box !important;
    }

    .stVerticalBlock > div {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 100% !important;
        height: auto !important;
        min-height: auto !important;
        margin: 0 !important;
        padding: 0 !important;
        box-sizing: border-box !important;
    }
    
    /* Streamlitのmarkdown要素の幅制限 */
    .stMarkdown,
    [data-testid="stMarkdown"] {
        max-width: var(--main-content-width) !important;
        margin: 0 auto !important;
        box-sizing: border-box !important;
    }
    
    /* Streamlitのmarkdown要素の余白調整 */
    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3 {
        margin-top: 15px !important;
        margin-bottom: 8px !important;
    }
    
    .stMarkdown p {
        margin-bottom: 8px !important;
    }
    
    .stMarkdown ul,
    .stMarkdown ol {
        margin-bottom: 10px !important;
    }
    
    /* Streamlitのmarkdown内の要素 */
    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3,
    .stMarkdown h4,
    .stMarkdown h5,
    .stMarkdown h6,
    .stMarkdown p,
    .stMarkdown ul,
    .stMarkdown ol,
    .stMarkdown li {
        max-width: 100% !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
    }
    
    /* Streamlitのアラート要素 */
    .stAlert,
    [data-testid="stAlert"] {
        max-width: var(--main-content-width) !important;
        margin: 0 auto !important;
    }

    /* レスポンシブ対応 */
    @media (max-width: 1056px) {
        :root {
            --main-content-width: calc(100vw - var(--sidebar-width-expanded) - 40px);
        }
        
        .main-content {
            padding: 16px !important;
        }

        .stSidebar,
        [data-testid="stSidebar"] {
            position: absolute !important;
            height: 100vh !important;
            z-index: 2000 !important;
        }

        .stSidebar:not(.collapsed),
        [data-testid="stSidebar"]:not(.collapsed) {
            /* モバイル表示時のサイドバー */
        }
    }
    
    @media (max-width: 768px) {
        :root {
            --main-content-width: 100%;
            --sidebar-width-expanded: 280px;
        }
        
        .main-content {
            padding: 12px 12px 140px 12px !important;
        }
        
        .main-container {
            border-radius: 0 !important;
        }
        
        .stChatMessage,
        [data-testid="stChatMessage"] {
            border-radius: 8px !important;
            margin-bottom: 12px !important;
        }
        
        .stChatInput,
        [data-testid="stChatInput"] {
            border-radius: 8px !important;
            width: calc(100% - 20px) !important;
            max-width: calc(100% - 20px) !important;
            left: 50% !important;
            bottom: 40px !important;
            transform: translateX(-50%) !important;
            padding: 15px !important;
        }
        
        /* モバイルでのmarkdown要素の幅制限 */
        .stMarkdown,
        [data-testid="stMarkdown"] {
            max-width: 100% !important;
            margin: 0 !important;
            padding: 0 10px !important;
        }
        
        .stAlert,
        [data-testid="stAlert"] {
            max-width: 100% !important;
            margin: 0 !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_status" not in st.session_state:
    st.session_state.agent_status = "稼働中"

if "sidebar_collapsed" not in st.session_state:
    st.session_state.sidebar_collapsed = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "agent_initialized" not in st.session_state:
    st.session_state.agent_initialized = False

if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

if "navigation_history" not in st.session_state:
    st.session_state.navigation_history = ["home"]

# エージェント初期化関数
async def initialize_agent():
    """自己学習AIエージェントを初期化"""
    if st.session_state.agent_initialized:
        return True
    
    try:
        # エージェント作成
        st.session_state.agent = SelfLearningAgent(
            config_path="config/agent_config.yaml",
            db_path="data/self_learning_agent.db"
        )
        
        # セッション初期化
        session_id = str(uuid.uuid4())
        await st.session_state.agent.initialize_session(
            session_id=session_id,
            user_id="webui_user"
        )
        
        st.session_state.session_id = session_id
        st.session_state.agent_initialized = True
        st.session_state.agent_status = "稼働中"
        
        return True
        
    except Exception as e:
        error_msg = f"エージェント初期化エラー: {e}"
        st.error(error_msg)
        logger.error(error_msg)
        st.session_state.agent_status = "エラー"
        
        # データベースエラーの場合は詳細情報を表示
        if "sqlite3.OperationalError" in str(e) or "table" in str(e).lower():
            st.warning("データベースエラーが発生しました。データベースを再初期化してください。")
            if st.button("🔄 データベース再初期化"):
                try:
                    import subprocess
                    result = subprocess.run([sys.executable, "init_db.py"], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        st.success("データベースが再初期化されました。ページをリロードしてください。")
                        st.rerun()
                    else:
                        st.error(f"データベース再初期化失敗: {result.stderr}")
                except Exception as init_error:
                    st.error(f"データベース再初期化エラー: {init_error}")
        
        return False

# エージェント処理関数
async def process_user_input(user_input: str) -> str:
    """ユーザー入力を処理してAI応答を生成"""
    if not st.session_state.agent_initialized:
        return "エージェントが初期化されていません。"
    
    try:
        # エージェントで処理
        result = await st.session_state.agent.process_user_input(user_input)
        
        # レスポンス取得
        response = result.get("response", "応答を生成できませんでした。")
        
        # エージェント状態更新
        if "agent_state" in result:
            state = result["agent_state"]
            st.session_state.agent_status = f"稼働中 (学習エポック: {state.get('learning_epoch', 0)})"
        
        return response
        
    except Exception as e:
        error_msg = f"処理エラー: {e}"
        logger.error(error_msg)
        
        # データベースエラーの場合は詳細情報を表示
        if "sqlite3.OperationalError" in str(e) or "table" in str(e).lower():
            return f"データベースエラーが発生しました: {e}\n\nデータベースを再初期化することをお勧めします。"
        elif "conversations" in str(e).lower() and "column" in str(e).lower():
            return f"データベーススキーマエラーが発生しました: {e}\n\nデータベースを再初期化してください。"
        else:
            return error_msg

# ナビゲーション関数
def navigate_to_page(page_name: str):
    """ページナビゲーション"""
    st.session_state.current_page = page_name
    if page_name not in st.session_state.navigation_history:
        st.session_state.navigation_history.append(page_name)
    st.rerun()

# ページ表示関数
def show_home_page():
    """ホームページ表示"""
    st.markdown("""
    <div class="main-container">
        <div class="main-content">
            <header class="page-header">
                <h1 class="page-title">🤖 AI Agent Chat</h1>
                <p class="page-description">
                    自己学習AIエージェントと会話を始めましょう。
                    サイドバーからナビゲーションを選択してください。
                </p>
            </header>
            
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 機能概要セクション
    st.markdown("### 🚀 機能概要")
    st.markdown("""
    **コア機能:**
    - 永続的な会話セッション
    - 自己学習と進化機能
    - プロンプトの動的最適化
    - リアルタイム監視と分析
    - MCP統合ツール
    - 強化学習システム
    """)
    
    # システム状態セクション
    st.markdown("### 📊 システム状態")
    st.info("エージェントの現在の状態とパフォーマンス指標を確認できます。")

def show_analysis_page():
    """分析ページ表示"""
    st.title("📊 エージェント分析")
    
    if not st.session_state.agent_initialized:
        st.warning("エージェントが初期化されていません。")
        return
    
    # エージェント状態の詳細分析
    if st.session_state.agent and st.session_state.agent.current_state:
        state = st.session_state.agent.current_state
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("学習エポック", state.learning_epoch)
            st.metric("進化世代", state.evolution_generation)
        
        with col2:
            st.metric("総インタラクション", state.total_interactions)
            st.metric("報酬スコア", f"{state.reward_score:.3f}")
        
        with col3:
            st.metric("セッション時間", "N/A")
            st.metric("最終活動", state.last_activity.strftime("%H:%M:%S"))
        
        # パフォーマンス指標
        if state.performance_metrics:
            st.subheader("パフォーマンス指標")
            for key, value in state.performance_metrics.items():
                st.write(f"**{key}:** {value}")
        
        # 学習データの可視化
        st.subheader("学習データ")
        if st.session_state.agent.tuning_data_pool:
            st.write(f"チューニングデータ数: {len(st.session_state.agent.tuning_data_pool)}")
        if st.session_state.agent.reward_history:
            st.write(f"報酬履歴数: {len(st.session_state.agent.reward_history)}")

def show_user_management_page():
    """ユーザー管理ページ表示"""
    st.title("👥 ユーザー管理")
    
    # セッション情報
    st.subheader("現在のセッション")
    if st.session_state.session_id:
        st.write(f"**セッションID:** {st.session_state.session_id}")
        st.write(f"**ユーザーID:** webui_user")
        st.write(f"**エージェント状態:** {'初期化済み' if st.session_state.agent_initialized else '未初期化'}")
    
    # セッション管理
    st.subheader("セッション管理")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 セッションリセット"):
            st.session_state.messages = []
            if st.session_state.agent_initialized and st.session_state.agent:
                try:
                    asyncio.run(st.session_state.agent.close())
                except:
                    pass
            st.session_state.agent_initialized = False
            st.session_state.agent = None
            st.session_state.session_id = None
            st.success("セッションがリセットされました")
            st.rerun()
    
    with col2:
        if st.button("📊 セッション統計"):
            if st.session_state.agent_initialized:
                st.info("セッション統計を表示中...")
            else:
                st.warning("エージェントが初期化されていません")

def show_settings_page():
    """設定ページ表示"""
    st.title("⚙️ 設定")
    
    # エージェント設定
    st.subheader("エージェント設定")
    
    if st.session_state.agent_initialized and st.session_state.agent:
        # 学習パラメータの表示
        st.write("**学習パラメータ:**")
        learning_config = st.session_state.agent.learning_config
        for key, value in learning_config.items():
            st.write(f"- {key}: {value}")
        
        # 設定変更
        st.subheader("設定変更")
        new_mutation_rate = st.slider("プロンプト変異率", 0.0, 1.0, learning_config.get("prompt_mutation_rate", 0.1))
        new_crossover_rate = st.slider("データ交叉率", 0.0, 1.0, learning_config.get("data_crossover_rate", 0.7))
        
        if st.button("設定を保存"):
            st.session_state.agent.learning_config["prompt_mutation_rate"] = new_mutation_rate
            st.session_state.agent.learning_config["data_crossover_rate"] = new_crossover_rate
            st.success("設定が保存されました")
    else:
        st.warning("エージェントが初期化されていません")
    
    # システム設定
    st.subheader("システム設定")
    st.write("**データベースパス:** data/self_learning_agent.db")
    st.write("**設定ファイル:** config/agent_config.yaml")
    
    # ログ設定
    st.subheader("ログ設定")
    log_level = st.selectbox("ログレベル", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1)
    if st.button("ログレベルを変更"):
        st.info(f"ログレベルを {log_level} に変更しました")

def show_reports_page():
    """レポートページ表示"""
    st.title("📋 レポート")
    
    if not st.session_state.agent_initialized:
        st.warning("エージェントが初期化されていません。")
        return
    
    # エージェントレポート
    st.subheader("エージェントレポート")
    
    if st.session_state.agent and st.session_state.agent.current_state:
        state = st.session_state.agent.current_state
        
        # 基本情報
        st.write("**基本情報:**")
        st.write(f"- セッションID: {state.session_id}")
        st.write(f"- 学習エポック: {state.learning_epoch}")
        st.write(f"- 総インタラクション: {state.total_interactions}")
        st.write(f"- 報酬スコア: {state.reward_score:.3f}")
        st.write(f"- 進化世代: {state.evolution_generation}")
        st.write(f"- 最終活動: {state.last_activity}")
        
        # パフォーマンスレポート
        if state.performance_metrics:
            st.subheader("パフォーマンスレポート")
            for key, value in state.performance_metrics.items():
                st.write(f"**{key}:** {value}")
        
        # 学習データレポート
        st.subheader("学習データレポート")
        if st.session_state.agent.tuning_data_pool:
            st.write(f"チューニングデータ数: {len(st.session_state.agent.tuning_data_pool)}")
        if st.session_state.agent.reward_history:
            st.write(f"報酬履歴数: {len(st.session_state.agent.reward_history)}")
        
        # レポート生成
        if st.button("📄 レポートを生成"):
            report_data = {
                "session_id": state.session_id,
                "learning_epoch": state.learning_epoch,
                "total_interactions": state.total_interactions,
                "reward_score": state.reward_score,
                "evolution_generation": state.evolution_generation,
                "last_activity": state.last_activity.isoformat(),
                "performance_metrics": state.performance_metrics,
                "tuning_data_count": len(st.session_state.agent.tuning_data_pool),
                "reward_history_count": len(st.session_state.agent.reward_history)
            }
            
            st.download_button(
                label="レポートをダウンロード",
                data=str(report_data),
                file_name=f"agent_report_{state.session_id[:8]}.json",
                mime="application/json"
            )

# サイドバーの状態に応じてCSSクラスを適用
sidebar_css_class = "collapsed" if st.session_state.sidebar_collapsed else "expanded"

# サイドバーの状態に応じてCSSを適用
st.markdown(f"""
<style>
.stSidebar,
[data-testid="stSidebar"] {{
    width: var(--sidebar-width-{sidebar_css_class}) !important;
    min-width: var(--sidebar-width-{sidebar_css_class}) !important;
    max-width: var(--sidebar-width-{sidebar_css_class}) !important;
}}
</style>
""", unsafe_allow_html=True)

# スクロールアニメーションを無効化するCSS（適度に調整）
st.markdown("""
<style>
/* スクロールアニメーションを無効化（適度に） */
html {
    scroll-behavior: auto !important;
    overflow-x: hidden !important;
    scroll-padding-top: 0 !important;
    scroll-margin-top: 0 !important;
}

body {
    scroll-behavior: auto !important;
    overflow-x: hidden !important;
    scroll-padding-top: 0 !important;
    scroll-margin-top: 0 !important;
}

* {
    scroll-behavior: auto !important;
    scroll-padding-top: 0 !important;
    scroll-margin-top: 0 !important;
}

/* ページ読み込み時の自動スクロールを防ぐ */
.stApp {
    scroll-behavior: auto !important;
    scroll-padding-top: 0 !important;
    scroll-margin-top: 0 !important;
}

.stAppViewContainer {
    scroll-behavior: auto !important;
    scroll-padding-top: 0 !important;
    scroll-margin-top: 0 !important;
}

/* Streamlitのデフォルトアニメーションを無効化 */
.stApp {
    animation: none !important;
}

.stAppViewContainer {
    animation: none !important;
}

.main-container {
    animation: none !important;
}

/* メインコンテンツエリアはスクロール可能にする */
.stMain {
    overflow-y: auto !important;
    overflow-x: hidden !important;
    scroll-behavior: auto !important;
    scroll-padding-top: 0 !important;
    scroll-margin-top: 0 !important;
}

[data-testid="stMain"] {
    overflow-y: auto !important;
    overflow-x: hidden !important;
    scroll-behavior: auto !important;
    scroll-padding-top: 0 !important;
    scroll-margin-top: 0 !important;
}

/* すべてのStreamlit要素のスクロール動作を制御 */
.stVerticalBlock,
[data-testid="stVerticalBlock"],
.stElementContainer,
[data-testid="stElementContainer"] {
    scroll-behavior: auto !important;
    scroll-padding-top: 0 !important;
    scroll-margin-top: 0 !important;
}

/* 自動スクロールを完全に無効化 */
html, body {
    scroll-behavior: auto !important;
    scroll-padding-top: 0 !important;
    scroll-margin-top: 0 !important;
    overflow-x: hidden !important;
}

/* すべての要素の自動スクロールを無効化 */
* {
    scroll-behavior: auto !important;
    scroll-padding-top: 0 !important;
    scroll-margin-top: 0 !important;
}

/* Streamlitの特定の自動スクロール要素を無効化 */
.stApp,
.stAppViewContainer,
.main-container,
[data-testid="stMain"],
[data-testid="stSidebar"] {
    scroll-behavior: auto !important;
    scroll-padding-top: 0 !important;
    scroll-margin-top: 0 !important;
}

/* サイドバーのスクロールを制御 */
    .stSidebar {
        overflow-y: auto !important;
        overflow-x: hidden !important;
        scrollbar-width: thin !important;
        scrollbar-color: var(--accent-color) var(--secondary-color) !important;
    }

    [data-testid="stSidebar"] {
        overflow-y: auto !important;
        overflow-x: hidden !important;
        scrollbar-width: thin !important;
        scrollbar-color: var(--accent-color) var(--secondary-color) !important;
    }

/* サイドバー内の子要素のスクロールを無効化して2重スクロールを防ぐ */
[data-testid="stSidebar"] > div {
    overflow: visible !important;
}

[data-testid="stSidebar"] .stElementContainer {
    overflow: visible !important;
}

[data-testid="stSidebar"] .stMarkdown {
    overflow: visible !important;
}

/* サイドバー内のmain-content要素のスクロールを無効化 */
[data-testid="stSidebar"] .main-content {
    overflow: visible !important;
    overflow-y: visible !important;
    overflow-x: visible !important;
}

/* サイドバー内のすべての子要素のスクロールを無効化 */
[data-testid="stSidebar"] * {
    overflow: visible !important;
    overflow-y: visible !important;
    overflow-x: visible !important;
}

    /* サイドバー内のトグルボタンを固定位置に配置 */
    [data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:first-child {
        position: relative !important;
        top: 0 !important;
        left: 0 !important;
        width: 32px !important;
        height: 32px !important;
        background: var(--primary-color) !important;
        border: 1px solid var(--border-color) !important;
        color: white !important;
        z-index: 1000 !important;
        border-radius: 4px !important;
        font-size: 14px !important;
        font-weight: bold !important;
        pointer-events: auto !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.5) !important;
        margin-bottom: 10px !important;
    }

    [data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:first-child:hover {
        background: var(--hover-color) !important;
        border-color: var(--accent-color) !important;
        transform: scale(1.05) !important;
    }

/* サイドバーのコンテンツを適切に配置 */
[data-testid="stSidebar"] > div:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
    overflow: visible !important;
}

/* サイドバー内の最初の要素コンテナ */
[data-testid="stSidebar"] .stElementContainer:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
    overflow: visible !important;
}

    /* サイドバー内の他の要素の位置を調整 */
    [data-testid="stSidebar"] .stElementContainer {
        margin-top: 0 !important;
        padding-top: 0 !important;
        position: relative !important;
        z-index: 1 !important;
        overflow: visible !important;
    }

    /* サイドバー内のマークダウンコンテンツの位置を調整 */
    [data-testid="stSidebar"] .stMarkdown {
        margin-top: 0 !important;
        padding-top: 0 !important;
        position: relative !important;
        z-index: 1 !important;
        overflow: visible !important;
    }

    /* サイドバー内のボタンの重なりを防ぐ */
    [data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {
        position: relative !important;
        z-index: 2 !important;
        pointer-events: auto !important;
    }

    /* トグルボタンのみ最前面に表示 */
    [data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:first-child {
        z-index: 1000 !important;
        position: fixed !important;
    }

/* メインコンテンツの下の余白を調整 */
.main-container {
    padding-bottom: 0 !important;
    margin-bottom: 0 !important;
}

/* チャット入力エリアの下にも余白を追加 */
[data-testid="stChatInput"] {
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}

/* メインコンテンツエリア全体の下の余白 */
[data-testid="stMain"] {
    padding-bottom: 0 !important;
}

/* ページ全体の下の余白を確保 */
.stApp {
    padding-bottom: 0 !important;
}

/* チャットメッセージエリアの下の余白 */
[data-testid="stChatMessage"] {
    margin-bottom: 20px !important;
}
</style>

""", unsafe_allow_html=True)

# カスタムサイドバー開閉ボタン
def toggle_sidebar():
            st.session_state.sidebar_collapsed = not st.session_state.sidebar_collapsed
            st.rerun()

# サイドバーの状態に応じてHTMLのdata属性とコンテナクラスを設定
sidebar_state = "collapsed" if st.session_state.sidebar_collapsed else "expanded"

# JavaScriptで直接コンテナのスタイルを制御（st.components.v1.htmlを使用）
import streamlit.components.v1 as components

components.html(f"""
<script>
// サイドバー状態を監視してコンテナのスタイルを直接制御
function updateContainerStyle() {{
    const sidebar = document.querySelector('[data-testid="stSidebar"]');
    const container = document.querySelector('[data-testid="stBottomBlockContainer"]');
    
    if (!sidebar || !container) return;
    
    const sidebarWidth = window.getComputedStyle(sidebar).width;
    const isExpanded = sidebarWidth === '288px' || sidebarWidth === '256px';
    
    if (isExpanded) {{
        // サイドバーが開いている状態
        container.style.left = '288px';
        container.style.width = 'calc(100vw - 288px)';
        container.style.right = '0px';
        console.log('Container updated for expanded sidebar');
    }} else {{
        // サイドバーが閉じている状態
        container.style.left = '48px';
        container.style.width = 'calc(100vw - 48px)';
        container.style.right = '0px';
        console.log('Container updated for collapsed sidebar');
    }}
}}

// 即座に実行
updateContainerStyle();

// DOMContentLoaded後に実行
if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', updateContainerStyle);
}} else {{
    updateContainerStyle();
}}

// 定期的にチェックして確実に設定
setInterval(updateContainerStyle, 100);

// より短い間隔でもチェック
setTimeout(updateContainerStyle, 10);
setTimeout(updateContainerStyle, 200);
setTimeout(updateContainerStyle, 500);

// MutationObserverでサイドバーの変更を監視
const observer = new MutationObserver(function(mutations) {{
    mutations.forEach(function(mutation) {{
        if (mutation.type === 'attributes' && 
            (mutation.attributeName === 'style' || mutation.attributeName === 'aria-expanded')) {{
            updateContainerStyle();
        }}
    }});
}});

// 監視開始
const sidebar = document.querySelector('[data-testid="stSidebar"]');
if (sidebar) {{
    observer.observe(sidebar, {{ 
        attributes: true, 
        attributeFilter: ['style', 'aria-expanded'] 
    }});
}}
</script>
""", height=0)

# サイドバー
with st.sidebar:
    # サイドバー開閉ボタン（常に表示）
    if st.session_state.sidebar_collapsed:
        # 閉じた状態：アイコンのみ表示
        if st.button("📋", key="sidebar_open", help="サイドバーを開く", use_container_width=True):
            toggle_sidebar()
    else:
        # 開いた状態：テキスト付きボタン表示
        st.markdown("### パネル操作")
        if st.button("📋 パネルを閉じる", key="sidebar_close", use_container_width=True):
            toggle_sidebar()
        st.markdown("---")
    
    # サイドバーが開いている場合のみ詳細コンテンツを表示
    if not st.session_state.sidebar_collapsed:
        # その他のコンテンツ（必要に応じて追加）
        st.markdown("### ナビゲーション")
        st.info("パネルを閉じるには上記のボタンをクリックしてください。")

# メインコンテンツ
if st.session_state.current_page == "home":
    show_home_page()
    
    # チャット履歴の表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # チャット入力（Streamlitの機能を使用）
    user_input = st.chat_input("AIエージェントにメッセージを送信...")

    if user_input:
        # ユーザーメッセージを追加
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # エージェントが初期化されていない場合は初期化を試行
        if not st.session_state.agent_initialized:
            with st.spinner("エージェントを初期化中..."):
                try:
                    success = asyncio.run(initialize_agent())
                    if not success:
                        st.error("エージェントの初期化に失敗しました。")
                        st.stop()
                except Exception as e:
                    st.error(f"初期化エラー: {e}")
                    st.stop()
        
        # AI応答を生成
        with st.chat_message("assistant"):
            with st.spinner("AIエージェントが思考中..."):
                try:
                    response = asyncio.run(process_user_input(user_input))
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_response = f"申し訳ございません。処理中にエラーが発生しました: {e}"
                    st.markdown(error_response)
                    st.session_state.messages.append({"role": "assistant", "content": error_response})

elif st.session_state.current_page == "analysis":
    show_analysis_page()

elif st.session_state.current_page == "users":
    show_user_management_page()

elif st.session_state.current_page == "settings":
    show_settings_page()

elif st.session_state.current_page == "reports":
    show_reports_page()

else:
    # デフォルトはホームページ
    show_home_page()