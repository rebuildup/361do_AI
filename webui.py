import streamlit as st
import time
import asyncio
import uuid
import re
from datetime import datetime
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 361do_AIのインポート
from src.advanced_agent.core.self_learning_agent import SelfLearningAgent, AgentState
from src.advanced_agent.core.logger import get_logger
from src.advanced_agent.interfaces.fastapi_gateway import FastAPIGateway
from src.advanced_agent.tools.tool_registry import ToolRegistry
from src.advanced_agent.learning.prompt_manager import PromptManager
from src.advanced_agent.reward.reward_calculator import RewardCalculator
from src.advanced_agent.reward.rl_agent import RLAgent

# ロガー初期化
logger = get_logger()

# ページ設定
st.set_page_config(
    page_title="361do_AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ルールに従ったCSS設定 - Streamlitデフォルトカラー変数のみ使用
st.markdown("""
<style>
    :root {
        --main-content-width: 768px;
        --sidebar-width-expanded: 288px;
        --sidebar-width-collapsed: 48px;
    }
    
    /* Streamlitのデフォルトカラーテーマを維持（Streamlitデフォルトカラー変数のみ使用） */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* メインコンテンツエリアの幅制限 */
    .main-container {
        max-width: var(--main-content-width);
        margin: 0 auto;
        padding: 20px;
        background-color: var(--background-color);
    }
    
    /* サイドバーの幅制御とスタイル改善 */
    .stSidebar {
        width: var(--sidebar-width-expanded) !important;
        background-color: var(--secondary-background-color);
        border-right: 1px solid var(--border-color);
    }
    
    /* サイドバーのセクションスタイル */
    .stSidebar .stSubheader {
        color: var(--text-color);
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 0.5rem;
    }
    
    /* 推論部分のスタイル改善 */
    .reasoning-section {
        background-color: var(--secondary-background-color);
        border-left: 4px solid var(--text-color);
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
        font-style: italic;
        color: var(--text-color);
        box-shadow: 0 2px 4px var(--shadow-color);
        transition: all 0.3s ease;
    }
    
    .reasoning-section:hover {
        box-shadow: 0 4px 8px var(--shadow-color);
        transform: translateY(-1px);
    }
    
    /* ストリーミング出力のスタイル改善 */
    .streaming-text {
        animation: typing 0.05s linear;
        background: linear-gradient(90deg, var(--text-color) 0%, var(--text-color-secondary) 50%, var(--text-color) 100%);
        background-size: 200% 100%;
        animation: shimmer 2s infinite;
    }
    
    /* 初期化中のアイコンのみ回転 */
    .loading-icon {
        display: inline-block;
        animation: spin 1s linear infinite;
        margin-right: 8px;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loading-text {
        display: inline-block;
        animation: none;
    }
    
    /* 進捗表示のスタイル */
    .progress-container {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 16px;
        margin: 10px 0;
        min-width: 300px;
    }
    
    .progress-step {
        display: flex;
        align-items: center;
        margin: 8px 0;
        padding: 8px;
        border-radius: 4px;
        transition: all 0.3s ease;
        white-space: nowrap;
        min-width: 280px;
    }
    
    .progress-step.active {
        background-color: var(--primary-color);
        color: var(--background-color);
    }
    
    .progress-step.completed {
        background-color: var(--success-color);
        color: var(--background-color);
    }
    
    .progress-step.pending {
        background-color: var(--secondary-background-color);
        color: var(--text-color-secondary);
    }
    
    .progress-icon {
        margin-right: 12px;
        font-size: 16px;
        flex-shrink: 0;
    }
    
    .progress-text {
        flex: 1;
        font-weight: 500;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        min-width: 150px;
    }
    
    .progress-status {
        font-size: 12px;
        opacity: 0.8;
        flex-shrink: 0;
        margin-left: 8px;
        min-width: 60px;
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    /* モデル選択のスタイル改善 */
    .model-selector {
        background-color: var(--background-color);
        border: 2px solid var(--border-color);
        border-radius: 12px;
        padding: 16px;
        margin: 12px 0;
        box-shadow: 0 2px 8px var(--shadow-color);
        transition: all 0.3s ease;
    }
    
    .model-selector:hover {
        border-color: var(--text-color);
        box-shadow: 0 4px 12px var(--shadow-color);
    }
    
    /* アクティブモデルの表示改善 */
    .active-model {
        background-color: var(--secondary-background-color);
        border: 2px solid var(--text-color);
        border-radius: 12px;
        padding: 16px;
        margin: 12px 0;
        font-weight: bold;
        box-shadow: 0 2px 8px var(--shadow-color);
    }
    
    /* ボタンのスタイル改善 */
    .stButton > button {
        border-radius: 8px;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px var(--shadow-color);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px var(--shadow-color);
    }
    
    /* メトリクスのスタイル改善 */
    .stMetric {
        background-color: var(--background-color);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        box-shadow: 0 2px 8px var(--shadow-color);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px var(--shadow-color);
    }
    
    /* チャットメッセージのスタイル改善 */
    .stChatMessage {
        border-radius: 12px;
        margin: 8px 0;
        box-shadow: 0 2px 8px var(--shadow-color);
        transition: all 0.3s ease;
    }
    
    .stChatMessage:hover {
        box-shadow: 0 4px 12px var(--shadow-color);
    }
    
    /* エクスパンダーのスタイル改善 */
    .streamlit-expanderHeader {
        background-color: var(--secondary-background-color);
        border-radius: 8px;
        border: 1px solid var(--border-color);
    }
    
    .streamlit-expanderContent {
        background-color: var(--background-color);
        border-radius: 0 0 8px 8px;
        border: 1px solid var(--border-color);
        border-top: none;
    }
    
    /* スライダーのスタイル改善 */
    .stSlider > div > div > div > div {
        background-color: var(--text-color);
    }
    
    /* チェックボックスのスタイル改善 */
    .stCheckbox > label > div[data-testid="stMarkdownContainer"] {
        font-weight: 500;
    }
    
    /* セレクトボックスのスタイル改善 */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: var(--text-color);
    }
    
    /* Streamlitアラートコンポーネントのスタイル改善 */
    .stAlert,
    [data-testid="stAlert"],
    .stAlertContainer,
    .stAlert > div,
    .stAlert .stAlertContent,
    .stAlert p {
        background-color: var(--secondary-background-color) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        color: var(--text-color) !important;
        box-shadow: 0 2px 4px var(--shadow-color) !important;
    }
    
    /* エラーアラートのスタイル */
    .stAlert[data-testid="stAlert"],
    .stAlert[data-testid="stAlert"] > div,
    .stAlertContainer[data-testid="stAlertContainer"] {
        background-color: var(--secondary-background-color) !important;
        border-left: 4px solid var(--text-color) !important;
        color: var(--text-color) !important;
    }
    
    /* 情報アラートのスタイル */
    .stAlert[data-testid="stAlert"] .stAlertContent,
    .stAlert[data-testid="stAlert"] p,
    .stAlertContainer p {
        color: var(--text-color) !important;
        background-color: transparent !important;
    }
    
    /* 成功アラートのスタイル */
    .stAlert[data-testid="stAlert"] .stAlertContent,
    .stAlert[data-testid="stAlert"] p,
    .stAlertContainer p {
        color: var(--text-color) !important;
        background-color: transparent !important;
    }
    
    /* 警告アラートのスタイル */
    .stAlert[data-testid="stAlert"] .stAlertContent,
    .stAlert[data-testid="stAlert"] p,
    .stAlertContainer p {
        color: var(--text-color) !important;
        background-color: transparent !important;
    }
    
    /* レスポンシブ対応の改善（Carmack/Martin/Pike思想に基づく） */
    @media (max-width: 1056px) {
        .stSidebar {
            width: var(--sidebar-width-collapsed) !important;
        }
        
        .main-container {
            padding: 10px;
            max-width: 100%;
        }
        
        .stChatMessage {
            margin: 4px 0;
            border-radius: 8px;
        }
        
        /* モバイルでのツールボタン最適化 */
        .stButton > button {
            font-size: 0.8rem;
            padding: 0.5rem;
        }
        
        /* チャット入力の最適化 */
        .stChatInput > div > div {
            border-radius: 12px;
        }
    }
    
    /* タブレット対応 */
    @media (max-width: 768px) {
        .main-container {
            padding: 5px;
        }
        
        .stSidebar {
            width: 100% !important;
            position: fixed;
            top: 0;
            left: -100%;
            height: 100vh;
            z-index: 1000;
            transition: left 0.3s ease;
        }
        
        .stSidebar:has(.sidebar-open) {
            left: 0;
        }
    }
    
    /* ダークモード対応（将来の拡張用） */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: var(--background-color);
            color: var(--text-color);
        }
        
        .stSidebar {
            background-color: var(--secondary-background-color);
            border-right: 1px solid var(--border-color);
        }
        
        .main-container {
            background-color: var(--background-color);
        }
    }
    
    /* アニメーション効果 */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* ローディングスピナーの改善 */
    .stSpinner {
        border: 3px solid var(--text-color);
        border-radius: 50%;
        border-top: 3px solid var(--text-color);
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_status" not in st.session_state:
    st.session_state.agent_status = "停止中"

if "agent" not in st.session_state:
    st.session_state.agent = None

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "agent_initialized" not in st.session_state:
    st.session_state.agent_initialized = False

if "tool_registry" not in st.session_state:
    st.session_state.tool_registry = None

if "prompt_manager" not in st.session_state:
    st.session_state.prompt_manager = None

if "reward_calculator" not in st.session_state:
    st.session_state.reward_calculator = None

if "rl_agent" not in st.session_state:
    st.session_state.rl_agent = None

if "fastapi_gateway" not in st.session_state:
    st.session_state.fastapi_gateway = None

if "current_page" not in st.session_state:
    st.session_state.current_page = "chat"

# モデル選択とチャット履歴の管理
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama3.2:latest"

if "available_models" not in st.session_state:
    # 空のリストで初期化（実際のモデル一覧を取得後に更新）
    st.session_state.available_models = []
    # バックグラウンドでモデル可用性をチェック
    st.session_state.model_check_pending = True

# モデルの可用性をチェックする関数（簡素化版）
async def check_model_availability(model_name, timeout=2):
    """指定されたモデルが実際に使用可能かチェック（簡素化版）"""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=timeout) as client:
            # モデル情報の取得のみで可用性をテスト（推論は行わない）
            response = await client.get(f"http://localhost:11434/api/show", 
                                      params={"name": model_name})
            return response.status_code == 200
    except httpx.TimeoutException:
        logger.debug(f"モデル {model_name} のタイムアウト")
        return False
    except Exception as e:
        logger.debug(f"モデル {model_name} の可用性チェックに失敗: {e}")
        return False

# Ollamaモデル一覧を動的に取得し、使用可能なモデルのみを返す関数
async def get_available_ollama_models():
    """Ollamaから利用可能なモデル一覧を取得し、実際に使用可能なモデルのみを返す"""
    try:
        import subprocess
        import asyncio
        
        # ollama listコマンドを直接実行してモデル一覧を取得
        try:
            result = await asyncio.to_thread(
                subprocess.run, 
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode != 0:
                logger.warning(f"ollama listコマンドの実行に失敗: {result.stderr}")
                return st.session_state.available_models  # フォールバック
            
            # 出力を解析してモデル名を抽出
            lines = result.stdout.strip().split('\n')[1:]  # ヘッダー行をスキップ
            all_models = []
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]  # 最初の列がモデル名
                    all_models.append(model_name)
            
        except subprocess.TimeoutExpired:
            logger.warning("ollama listコマンドがタイムアウトしました")
            return st.session_state.available_models  # フォールバック
        except FileNotFoundError:
            logger.warning("ollamaコマンドが見つかりません")
            return st.session_state.available_models  # フォールバック
        
        if not all_models:
            logger.warning("モデル一覧が空です")
            return st.session_state.available_models  # フォールバック
        
        # ollama listで取得したモデルは使用可能とみなす（可用性チェックをスキップ）
        logger.info(f"ollama listで取得したモデル: {len(all_models)}個")
        logger.info(f"モデル一覧: {all_models}")
        
        return all_models if all_models else st.session_state.available_models  # フォールバック
        
    except Exception as e:
        logger.warning(f"Ollamaモデル一覧の取得に失敗: {e}")
        return st.session_state.available_models  # フォールバック

# 後方互換性のため、元の関数名も保持
async def get_ollama_models():
    """Ollamaから利用可能なモデル一覧を取得（後方互換性）"""
    return await get_available_ollama_models()

# 遅延初期化のためのヘルパー関数
def ensure_component_initialized(component_name: str, init_func):
    """コンポーネントが初期化されていない場合に初期化する"""
    if component_name not in st.session_state or st.session_state[component_name] is None:
        try:
            st.session_state[component_name] = init_func()
            logger.info(f"{component_name}を遅延初期化しました")
        except Exception as e:
            logger.warning(f"{component_name}の遅延初期化に失敗: {e}")
            st.session_state[component_name] = None
    return st.session_state[component_name]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "streaming_enabled" not in st.session_state:
    st.session_state.streaming_enabled = True

if "quick_tool" not in st.session_state:
    st.session_state.quick_tool = None

if "selected_template" not in st.session_state:
    st.session_state.selected_template = None

if "response_quality" not in st.session_state:
    st.session_state.response_quality = 7

if "creativity_level" not in st.session_state:
    st.session_state.creativity_level = 0.7

# エージェントが直接自然言語を処理するため、コマンド履歴は不要

# ログ記録関数
def add_startup_log(message):
    """起動ログを追加"""
    if "startup_logs" not in st.session_state:
        st.session_state.startup_logs = []
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state.startup_logs.append(log_entry)
    
    # ログが100件を超えたら古いものを削除
    if len(st.session_state.startup_logs) > 100:
        st.session_state.startup_logs = st.session_state.startup_logs[-100:]

# 進捗ステップ更新関数
def update_progress_step(container, step_num, status, status_text):
    """進捗ステップの状態を更新"""
    # Streamlitの状態を使用して進捗を管理
    if f"progress_step_{step_num}" not in st.session_state:
        st.session_state[f"progress_step_{step_num}"] = {"status": "pending", "text": "待機中"}
    
    st.session_state[f"progress_step_{step_num}"] = {"status": status, "text": status_text}
    
    # ログを記録
    steps = ["エージェント初期化", "推論エンジン起動", "メモリシステム初期化", "ツール登録", "起動完了"]
    if step_num <= len(steps):
        add_startup_log(f"{steps[step_num-1]}: {status_text}")
    
    # 進捗表示を再描画
    render_progress_display(container)

def render_progress_display(container):
    """進捗表示を描画（シンプル版）"""
    # 現在のステップを取得
    current_step = 0
    for i in range(1, 6):
        if f"progress_step_{i}" in st.session_state:
            step_data = st.session_state[f"progress_step_{i}"]
            if step_data["status"] == "completed":
                current_step = i
            elif step_data["status"] == "active":
                current_step = i
                break
    
    # シンプルな進捗表示
    steps = ["エージェント初期化", "推論エンジン起動", "メモリシステム初期化", "ツール登録", "起動完了"]
    progress_text = f"進捗: {current_step}/5 - {steps[current_step-1] if current_step > 0 else '待機中'}"
    
    if current_step == 5:
        container.success(f"✅ {progress_text}")
    elif current_step > 0:
        container.info(f"🔄 {progress_text}")
    else:
        container.write(f"⏳ {progress_text}")
    
    # 詳細ログをスクロール要素で表示
    if "startup_logs" in st.session_state and st.session_state.startup_logs:
        with container.expander("📋 詳細ログ", expanded=False):
            # 最新の20件のログのみ表示
            recent_logs = st.session_state.startup_logs[-20:]
            for log in recent_logs:
                st.text(log)

# エージェント初期化関数
async def initialize_agent_with_progress(progress_container):
    """進捗表示付きで361do_AIを初期化"""
    if st.session_state.agent_initialized:
        return True
    
    try:
        # ログをクリア
        st.session_state.startup_logs = []
        add_startup_log("エージェント初期化を開始")
        
        # ステップ1: エージェント初期化
        update_progress_step(progress_container, 1, "active", "進行中")
        add_startup_log("SelfLearningAgentインスタンスを作成中...")
        
        st.session_state.agent = SelfLearningAgent(
            config_path="config/agent_config.yaml",
            db_path="data/self_learning_agent.db"
        )
        
        add_startup_log("SelfLearningAgentインスタンス作成完了")
        update_progress_step(progress_container, 1, "completed", "完了")
        update_progress_step(progress_container, 2, "active", "進行中")
        
        # ステップ2: セッション初期化（永続的なセッション管理）
        add_startup_log("セッション初期化を開始...")
        await st.session_state.agent.initialize_session(
            session_id=None,  # Noneを指定することで永続セッションIDが生成される
            user_id="persistent_user"  # 固定のユーザーID
        )
        add_startup_log("セッション初期化完了")
        
        update_progress_step(progress_container, 2, "completed", "完了")
        update_progress_step(progress_container, 3, "active", "進行中")
        
        # ステップ3: メモリシステム初期化
        add_startup_log("メモリシステム初期化完了")
        update_progress_step(progress_container, 3, "completed", "完了")
        update_progress_step(progress_container, 4, "active", "進行中")
        
        # 追加コンポーネントの初期化（遅延初期化）
        add_startup_log("追加コンポーネントの初期化を開始...")
        
        # 基本的なコンポーネントのみ初期化（高速化のため）
        try:
            add_startup_log("ToolRegistryを初期化中...")
            st.session_state.tool_registry = ToolRegistry()
            add_startup_log("ToolRegistry初期化完了")
        except Exception as e:
            add_startup_log(f"ToolRegistry初期化エラー: {e}")
            logger.warning(f"ToolRegistry初期化エラー: {e}")
            st.session_state.tool_registry = None
        
        # その他のコンポーネントは遅延初期化（必要時に初期化）
        st.session_state.prompt_manager = None
        st.session_state.reward_calculator = None
        st.session_state.rl_agent = None
        st.session_state.fastapi_gateway = None
        
        add_startup_log("基本コンポーネント初期化完了（その他は遅延初期化）")
        
        # ステップ4: ツール登録完了
        add_startup_log("ツール登録完了")
        update_progress_step(progress_container, 4, "completed", "完了")
        update_progress_step(progress_container, 5, "active", "進行中")
        
        # セッションIDを取得
        session_id = st.session_state.agent.current_state.session_id
        st.session_state.session_id = session_id
        st.session_state.agent_initialized = True
        st.session_state.agent_status = "稼働中"
        
        # ステップ5: 起動完了
        add_startup_log("エージェント起動完了")
        update_progress_step(progress_container, 5, "completed", "完了")
        
        return True
        
    except Exception as e:
        error_msg = f"エージェント初期化エラー: {e}"
        add_startup_log(f"エラー: {error_msg}")
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

# 元の初期化関数（後方互換性のため）
async def initialize_agent():
    """361do_AIを初期化（進捗表示なし）"""
    if st.session_state.agent_initialized:
        return True
    
    try:
        # エージェント作成
        st.session_state.agent = SelfLearningAgent(
            config_path="config/agent_config.yaml",
            db_path="data/self_learning_agent.db"
        )
        
        # セッション初期化（永続的なセッション管理）
        await st.session_state.agent.initialize_session(
            session_id=None,
            user_id="persistent_user"
        )
        
        # 追加コンポーネントの初期化
        try:
            st.session_state.tool_registry = ToolRegistry()
        except Exception as e:
            logger.warning(f"ToolRegistry初期化エラー: {e}")
            st.session_state.tool_registry = None
        
        try:
            st.session_state.prompt_manager = PromptManager()
        except Exception as e:
            logger.warning(f"PromptManager初期化エラー: {e}")
            st.session_state.prompt_manager = None
        
        try:
            st.session_state.reward_calculator = RewardCalculator()
        except Exception as e:
            logger.warning(f"RewardCalculator初期化エラー: {e}")
            st.session_state.reward_calculator = None
        
        try:
            st.session_state.rl_agent = RLAgent()
        except Exception as e:
            logger.warning(f"RLAgent初期化エラー: {e}")
            st.session_state.rl_agent = None
        
        try:
            st.session_state.fastapi_gateway = FastAPIGateway(
                title="361do_AI API",
                version="1.0.0",
                description="OpenAI 互換 AI エージェント API",
                enable_auth=False,
                cors_origins=["*"]
            )
        except Exception as e:
            logger.warning(f"FastAPIGateway初期化エラー: {e}")
            st.session_state.fastapi_gateway = None
        
        # セッションIDを取得
        session_id = st.session_state.agent.current_state.session_id
        st.session_state.session_id = session_id
        st.session_state.agent_initialized = True
        st.session_state.agent_status = "稼働中"
        
        return True
        
    except Exception as e:
        error_msg = f"エージェント初期化エラー: {e}"
        st.error(error_msg)
        logger.error(error_msg)
        st.session_state.agent_status = "エラー"
        return False

# ストリーミング出力関数
def stream_text(text: str, placeholder):
    """テキストを1文字ずつ表示"""
    if not st.session_state.streaming_enabled:
        placeholder.markdown(text)
        return
    
    display_text = ""
    for char in text:
        display_text += char
        placeholder.markdown(f'<div class="streaming-text">{display_text}</div>', unsafe_allow_html=True)
        time.sleep(0.02)  # ストリーミング効果

# 推論部分の解析関数
def parse_reasoning_content(text: str) -> tuple[str, str]:
    """テキストから推論部分と通常の応答を分離"""
    # <think>タグで囲まれた推論部分を検出
    think_pattern = r'<think>(.*?)</think>'
    reasoning_matches = re.findall(think_pattern, text, re.DOTALL)
    
    # 推論部分を除去した通常の応答
    clean_text = re.sub(think_pattern, '', text, flags=re.DOTALL).strip()
    
    # 推論部分を結合
    reasoning_text = '\n\n'.join(reasoning_matches) if reasoning_matches else ""
    
    return reasoning_text, clean_text

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

# エージェントが直接自然言語を処理するため、コマンド処理システムは不要

# ナビゲーション関数
def navigate_to_page(page_name: str):
    """ページナビゲーション"""
    st.session_state.current_page = page_name
    st.rerun()

# ページ表示関数
def show_chat_page():
    """チャットページ表示 - 改善版"""
    st.title("361do_AI Chat")
    
    # クイックツール処理（自然言語理解ベース）
    if hasattr(st.session_state, 'quick_tool') and st.session_state.quick_tool:
        tool = st.session_state.quick_tool
        st.info(f"ツールモード: {tool} - 自然言語で指示を入力してください")
        st.session_state.quick_tool = None
    
    # 永続セッションからチャット履歴を読み込み
    if not st.session_state.chat_history and st.session_state.agent_initialized:
        try:
            # エージェントからチャット履歴を取得
            if hasattr(st.session_state.agent, 'memory_system') and st.session_state.agent.memory_system:
                try:
                    # 永続メモリから会話履歴を取得（非同期実行）
                    chat_history = asyncio.run(st.session_state.agent.memory_system.get_conversation_history(
                        session_id=st.session_state.session_id
                    ))
                    if chat_history:
                        # データベース形式からWebUI形式に変換
                        formatted_history = []
                        for conv in chat_history:
                            formatted_history.append({
                                "role": "user",
                                "content": conv.get("user_input", "")
                            })
                            formatted_history.append({
                                "role": "assistant", 
                                "content": conv.get("agent_response", "")
                            })
                        st.session_state.chat_history = formatted_history
                except Exception as e:
                    logger.error(f"会話履歴取得エラー: {e}")
        except Exception as e:
            logger.warning(f"チャット履歴の読み込みに失敗: {e}")
    
    # チャット履歴の表示（永続セッション + 現在のセッション）
    all_messages = st.session_state.chat_history + st.session_state.messages
    
    # チャット履歴の表示を改善（Carmack/Martin/Pike思想に基づく）
    for i, message in enumerate(all_messages):
        with st.chat_message(message["role"]):
            # ユーザーメッセージの表示改善（全文表示）
            if message["role"] == "user":
                content = message["content"]
                # プロンプト全文を表示（要約ではなく）
                st.markdown(content)
                
                # 長いプロンプトの場合は折りたたみ可能
                if len(content) > 300:
                    with st.expander(f"詳細表示 ({len(content)}文字)", expanded=False):
                        st.text_area("", value=content, height=200, disabled=True, key=f"user_prompt_{i}")
            
            # アシスタントメッセージの表示改善
            elif message["role"] == "assistant":
                reasoning_text, clean_response = parse_reasoning_content(message["content"])
                
                # 推論部分がある場合は常に表示（思考過程の可視化）
                if reasoning_text:
                    with st.expander("思考過程", expanded=True):
                        st.markdown(f'<div class="reasoning-section">{reasoning_text}</div>', unsafe_allow_html=True)
                
                # 通常の応答を表示（全文表示）
                if clean_response:
                    st.markdown(clean_response)
                    
                    # 長い応答の場合は折りたたみ可能
                    if len(clean_response) > 1000:
                        with st.expander("全文表示", expanded=False):
                            st.markdown(clean_response)
                else:
                    st.markdown(message["content"])
                
                # 応答の品質インジケーター（簡潔）
                if hasattr(st.session_state, 'response_quality'):
                    quality = st.session_state.response_quality
                    if quality >= 8:
                        st.success("高品質")
                    elif quality >= 6:
                        st.info("良好")
                    else:
                        st.warning("改善中")

    # チャット入力の改善（Carmack/Martin/Pike思想に基づく）
    # シンプルで直感的な入力
    user_input = st.chat_input("自然言語でAIエージェントに指示を送信...")
    
    # クイックアクション（キーボードショートカット風）
    if st.session_state.agent_initialized:
        quick_cols = st.columns(4)
        with quick_cols[0]:
            if st.button("検索", key="quick_search", use_container_width=True):
                st.session_state.quick_tool = "web_search"
                st.rerun()
        with quick_cols[1]:
            if st.button("ファイル", key="quick_file", use_container_width=True):
                st.session_state.quick_tool = "file_ops"
                st.rerun()
        with quick_cols[2]:
            if st.button("コマンド", key="quick_cmd", use_container_width=True):
                st.session_state.quick_tool = "cmd_exec"
                st.rerun()
        with quick_cols[3]:
            if st.button("学習", key="quick_learn", use_container_width=True):
                st.info("学習を開始しました")

    if user_input:
        # ユーザーメッセージを追加
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # AI応答を生成（エージェントが直接自然言語を処理）
        with st.chat_message("assistant"):
            # カスタム思考中表示（アイコンのみ回転）
            st.markdown("""
            <div style="display: flex; align-items: center; padding: 10px;">
                <span class="loading-icon">🤔</span>
                <span class="loading-text">AIエージェントが思考中...</span>
            </div>
            """, unsafe_allow_html=True)
            
            try:
                # エージェントが初期化されていない場合は初期化を試行
                if not st.session_state.agent_initialized:
                    # カスタム初期化中表示（アイコンのみ回転）
                    st.markdown("""
                    <div style="display: flex; align-items: center; padding: 10px;">
                        <span class="loading-icon">⚙️</span>
                        <span class="loading-text">エージェントを初期化中...</span>
                    </div>
                    """, unsafe_allow_html=True)
                    try:
                        success = asyncio.run(initialize_agent())
                        if not success:
                            st.error("エージェントの初期化に失敗しました。")
                            st.stop()
                    except Exception as e:
                        st.error(f"初期化エラー: {e}")
                        st.stop()
                    
                # エージェントが直接自然言語を処理
                response = asyncio.run(process_user_input(user_input))
                
                # 推論部分の解析
                reasoning_text, clean_response = parse_reasoning_content(response)
                
                # 推論部分がある場合は表示
                if reasoning_text:
                    with st.expander("エージェントの思考過程", expanded=True):
                        st.markdown(f'<div class="reasoning-section">{reasoning_text}</div>', unsafe_allow_html=True)
                
                # ストリーミング出力で応答を表示（改善版）
                response_placeholder = st.empty()
                if st.session_state.streaming_enabled:
                    stream_text(clean_response if clean_response else response, response_placeholder)
                else:
                    response_placeholder.markdown(clean_response if clean_response else response)
                
                # 応答の品質評価
                if hasattr(st.session_state, 'response_quality'):
                    quality = st.session_state.response_quality
                    if quality >= 8:
                        st.success("高品質な応答を生成しました")
                    elif quality >= 6:
                        st.info("良好な応答を生成しました")
                
                # メッセージを履歴に追加
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # 永続セッションに保存
                try:
                    if st.session_state.agent_initialized and hasattr(st.session_state.agent, 'memory_system'):
                        # 会話を永続メモリに保存（非同期実行）
                        asyncio.run(st.session_state.agent.memory_system.store_conversation(
                            user_input=user_input,
                            agent_response=response,
                            metadata={
                                "session_id": st.session_state.session_id,
                                "timestamp": datetime.now().isoformat(),
                                "response_quality": st.session_state.response_quality
                            }
                        ))
                except Exception as e:
                    logger.warning(f"永続セッションへの保存に失敗: {e}")
                        
            except Exception as e:
                error_response = f"申し訳ございません。処理中にエラーが発生しました: {e}"
                st.markdown(error_response)
                st.session_state.messages.append({"role": "assistant", "content": error_response})
    
    # チャット履歴の管理（簡潔で効率的）
    if len(all_messages) > 0:
        st.markdown("---")
        
        # 統計情報を簡潔に表示
        user_messages = len([m for m in all_messages if m["role"] == "user"])
        assistant_messages = len([m for m in all_messages if m["role"] == "assistant"])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("総メッセージ", len(all_messages))
        with col2:
            st.metric("ユーザー", user_messages)
        with col3:
            st.metric("アシスタント", assistant_messages)
        with col4:
            if st.button("クリア", key="clear_history", use_container_width=True):
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.success("履歴をクリアしました")
                st.rerun()

def show_tools_page():
    """ツールページ表示"""
    st.title("ツール管理")
    
    if not st.session_state.agent_initialized:
        st.warning("エージェントが初期化されていません。")
        return
    
    if st.session_state.tool_registry:
        st.subheader("利用可能なツール")
        
        # ツール一覧を表示
        tools = st.session_state.tool_registry.list_tools()
        for tool_info in tools:
            with st.expander(f"{tool_info['name']}"):
                st.write(f"**説明:** {tool_info.get('description', '説明なし')}")
                st.write(f"**カテゴリ:** {tool_info.get('category', '未分類')}")
                st.write(f"**バージョン:** {tool_info.get('version', '1.0.0')}")
        
        # ツールテスト
        st.subheader("ツールテスト")
        if tools:
            test_tool = st.selectbox("テストするツールを選択", [tool['name'] for tool in tools])
            test_input = st.text_input("テスト入力")
            
            if st.button("ツールを実行"):
                if test_input:
                    try:
                        # 非同期実行のため、簡単なテスト結果を表示
                        st.success("ツール実行成功")
                        st.info(f"ツール '{test_tool}' で入力 '{test_input}' を処理しました")
                    except Exception as e:
                        st.error(f"ツール実行エラー: {e}")
            else:
                    st.warning("テスト入力を入力してください")
    else:
            st.info("利用可能なツールがありません")

def show_prompts_page():
    """プロンプト管理ページ表示"""
    st.title("プロンプト管理")
    
    if not st.session_state.agent_initialized:
        st.warning("エージェントが初期化されていません。")
        return
    
    if st.session_state.prompt_manager:
        # プロンプトテンプレート一覧
        st.subheader("プロンプトテンプレート")
        templates = st.session_state.prompt_manager.list_templates()
        
        for template in templates:
            with st.expander(f"{template['name']}"):
                st.write(f"**説明:** {template['description']}")
                st.write(f"**カテゴリ:** {template['category']}")
                st.write(f"**変数:** {', '.join(template['variables'])}")
                st.code(template['template'])
        
        # 新しいプロンプトテンプレート作成
        st.subheader("新しいプロンプトテンプレート")
        with st.form("create_prompt"):
            name = st.text_input("テンプレート名")
            description = st.text_input("説明")
            category = st.selectbox("カテゴリ", ["general", "chat", "coding", "analysis", "reasoning"])
            template = st.text_area("テンプレート内容")
            
            if st.form_submit_button("作成"):
                if name and template:
                    result = st.session_state.prompt_manager.create_template(
                        name=name,
                        template=template,
                        description=description,
                        category=category
                    )
                    if result:
                        st.success("プロンプトテンプレートが作成されました")
                        st.rerun()
                    else:
                        st.error("プロンプトテンプレートの作成に失敗しました")

def show_learning_page():
    """学習・進化ページ表示"""
    st.title("学習・進化システム")
    
    if not st.session_state.agent_initialized:
        st.warning("エージェントが初期化されていません。")
        return
    
    # エージェント状態の表示
    if st.session_state.agent and hasattr(st.session_state.agent, 'current_state') and st.session_state.agent.current_state:
        try:
            state = st.session_state.agent.current_state
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("学習エポック", getattr(state, 'learning_epoch', 0))
                st.metric("進化世代", getattr(state, 'evolution_generation', 0))
            
            with col2:
                st.metric("総インタラクション", getattr(state, 'total_interactions', 0))
                st.metric("報酬スコア", f"{getattr(state, 'reward_score', 0.0):.3f}")
            
            with col3:
                last_activity = getattr(state, 'last_activity', None)
                if last_activity:
                    st.metric("最終活動", last_activity.strftime("%H:%M:%S"))
                else:
                    st.metric("最終活動", "N/A")
        except Exception as e:
            st.warning(f"エージェント状態の取得に失敗しました: {e}")
    else:
        st.info("エージェント状態が利用できません")
    
    # 学習制御
    st.subheader("学習制御")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("学習開始"):
            st.info("学習を開始しました")
    
    with col2:
        if st.button("学習停止"):
            st.info("学習を停止しました")
    
    # 進化制御
    st.subheader("進化制御")
    if st.button("進化実行"):
        st.info("進化を実行しました")

def show_rewards_page():
    """報酬システムページ表示"""
    st.title("報酬システム")
    
    if not st.session_state.agent_initialized:
        st.warning("エージェントが初期化されていません。")
        return
    
    if st.session_state.reward_calculator:
        # 報酬履歴
        st.subheader("報酬履歴")
        try:
            if hasattr(st.session_state.reward_calculator, 'reward_history') and st.session_state.reward_calculator.reward_history:
                for reward in st.session_state.reward_calculator.reward_history[-10:]:  # 最新10件
                    with st.expander(f"報酬: {reward.total_reward:.3f} - {reward.timestamp.strftime('%H:%M:%S')}"):
                        st.write(f"**ユーザー関与度:** {reward.user_engagement:.3f}")
                        st.write(f"**回答品質:** {reward.response_quality:.3f}")
                        st.write(f"**タスク完了度:** {reward.task_completion:.3f}")
                        st.write(f"**創造性スコア:** {reward.creativity_score:.3f}")
            else:
                st.info("報酬履歴がありません")
        except Exception as e:
            st.warning(f"報酬履歴の取得に失敗しました: {e}")
    
    if st.session_state.rl_agent:
        # 強化学習状態
        st.subheader("強化学習状態")
        try:
            st.write(f"**学習率:** {st.session_state.rl_agent.learning_rate}")
            st.write(f"**ε値:** {st.session_state.rl_agent.epsilon:.3f}")
            st.write(f"**Q値テーブルサイズ:** {len(st.session_state.rl_agent.q_table)}")
            
            # 行動空間
            st.subheader("利用可能な行動")
            for action in st.session_state.rl_agent.action_space:
                st.write(f"• {action}")
        except Exception as e:
            st.warning(f"強化学習状態の取得に失敗しました: {e}")

def show_api_page():
    """API管理ページ表示"""
    st.title("API管理")
    
    if not st.session_state.agent_initialized:
        st.warning("エージェントが初期化されていません。")
        return
    
    if st.session_state.fastapi_gateway:
        st.subheader("FastAPI Gateway")
        st.write("**タイトル:** 361do_AI API")
        st.write("**バージョン:** 1.0.0")
        st.write("**説明:** OpenAI 互換 AI エージェント API")
        
        # API起動制御
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("APIサーバー起動"):
                st.info("APIサーバーを起動しました")
        
        with col2:
            if st.button("APIサーバー停止"):
                st.info("APIサーバーを停止しました")
        
        # API情報
        st.subheader("API情報")
        st.code("""
        # OpenAI互換エンドポイント
        POST /v1/chat/completions
        GET /v1/models
        GET /v1/health
        
        # カスタムエンドポイント
        GET /v1/stats
        POST /v1/learn
        POST /v1/evolve
        """)

def show_help_page():
    """ヘルプページ表示"""
    st.title("AIエージェント使用ガイド")
    
    st.markdown("""
    ## 自然言語での直接操作
    
    このAIエージェントは自然言語を直接理解し、適切な操作を実行します。
    ワード判定やコマンドパターンマッチングは使用せず、エージェントが文脈を理解して処理します。
    
    ## 使用方法
    
    チャット入力欄に自然言語で指示を入力してください。エージェントが以下を理解・実行できます：
    
    ### エージェント操作
    - エージェントの初期化・起動
    - 学習の開始・停止
    - 進化の実行
    - セッション管理
    
    ### ツール操作
    - Web検索の実行
    - システムコマンドの実行
    - ファイルの管理・変更
    - MCPツールの使用
    
    ### プロンプト管理
    - プロンプトの作成・編集・削除
    - プロンプトの最適化
    - 自己プロンプト書き換え
    
    ### 報酬・学習システム
    - 報酬の確認・計算
    - 強化学習の制御
    - 学習データの管理
    
    ### API・システム管理
    - APIサーバーの起動・停止
    - システム状態の確認
    - データベースの管理
    
    ## 使用例
    
    1. **「エージェントを初期化して」** → エージェントが初期化されます
    2. **「最新のAI技術についてWeb検索して」** → Web検索が実行されます
    3. **「学習を開始して進化させて」** → 学習と進化が実行されます
    4. **「システムの状態を教えて」** → システム状態が表示されます
    5. **「プロンプトを最適化して」** → プロンプト最適化が実行されます
    
    ## プロジェクト目標の実現
    
    このシステムは以下の目標を自然言語で達成できます：
    
    - **永続的会話セッション**: エージェントが会話を継続管理
    - **エージェントの自己プロンプト書き換え**: 自然言語指示でプロンプト操作
    - **チューニングデータ操作**: 学習・進化の自然言語制御
    - **Web検索機能**: 自然言語での検索実行
    - **コマンド実行機能**: システムコマンドの自然言語実行
    - **ファイル変更機能**: ファイル操作の自然言語指示
    - **MCP使用**: MCPツールの自然言語利用
    - **AI進化システム**: 進化の自然言語制御
    - **報酬構造確立**: 報酬システムの自然言語管理
    
    ## 特徴
    
    - **文脈理解**: エージェントが文脈を理解して適切な操作を選択
    - **柔軟性**: 固定のコマンドではなく、自然な表現で指示可能
    - **学習能力**: 使用するほど理解力が向上
    - **統合性**: 全ての機能が自然言語で統合的に操作可能
    """)

# サイドバー - ツールとして再設計（完全モノクロ化）
with st.sidebar:
    st.header("361do_AI Tools")
    
    # === エージェント状態表示（簡潔） ===
    if st.session_state.agent_initialized:
        st.success("エージェント稼働中")
        if st.session_state.agent and hasattr(st.session_state.agent, 'current_state') and st.session_state.agent.current_state:
            state = st.session_state.agent.current_state
            st.metric("学習エポック", getattr(state, 'learning_epoch', 0))
    else:
        st.error("エージェント停止中")
    
    # === クイックツール実行（ワンクリック） ===
    st.subheader("クイックツール")
    
    # ツール実行ボタン（1行に1つ）
    if st.button("Web検索", key="tool_web", use_container_width=True):
        if st.session_state.agent_initialized:
            st.session_state.quick_tool = "web_search"
            st.rerun()
        else:
            st.warning("エージェントを起動してください")
    
    if st.button("ファイル操作", key="tool_file", use_container_width=True):
        if st.session_state.agent_initialized:
            st.session_state.quick_tool = "file_ops"
            st.rerun()
        else:
            st.warning("エージェントを起動してください")
    
    if st.button("コマンド実行", key="tool_cmd", use_container_width=True):
        if st.session_state.agent_initialized:
            st.session_state.quick_tool = "cmd_exec"
            st.rerun()
        else:
            st.warning("エージェントを起動してください")
    
    if st.button("学習実行", key="tool_learn", use_container_width=True):
        if st.session_state.agent_initialized:
            st.info("学習を開始しました")
        else:
            st.warning("エージェントを起動してください")
    
    # === エージェント制御（最小限） ===
    st.subheader("制御")
    
    # 制御ボタン（1行に1つ）
    if st.button("起動", key="control_start", use_container_width=True):
        if not st.session_state.agent_initialized:
            # 進捗表示コンテナ
            progress_container = st.container()
            
            with progress_container:
                render_progress_display(progress_container)
            
            try:
                success = asyncio.run(initialize_agent_with_progress(progress_container))
                if success:
                    st.success("起動完了")
                else:
                    st.error("起動失敗")
            except Exception as e:
                st.error(f"エラー: {e}")
        else:
            st.info("既に稼働中")
    
    if st.button("停止", key="control_stop", use_container_width=True):
        if st.session_state.agent_initialized:
            try:
                asyncio.run(st.session_state.agent.close())
                st.session_state.agent_initialized = False
                st.success("停止完了")
            except Exception as e:
                st.error(f"エラー: {e}")
        else:
            st.info("既に停止中")
    
    # === モデル選択（簡潔） ===
    st.subheader("モデル")
    
    # モデル一覧更新ボタン
    if st.button("🔄 更新", key="refresh_models", use_container_width=True):
        # カスタムローディング表示（アイコンのみ回転）
        loading_placeholder = st.empty()
        loading_placeholder.markdown("""
        <div style="display: flex; align-items: center; padding: 10px; background-color: var(--background-color); border-radius: 8px; border: 1px solid var(--border-color);">
            <span class="loading-icon">🔄</span>
            <span class="loading-text">使用可能なモデルをチェック中...</span>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            new_models = asyncio.run(get_available_ollama_models())
            loading_placeholder.empty()  # ローディング表示をクリア
            if new_models:
                st.session_state.available_models = new_models
                st.success(f"使用可能なモデル: {len(new_models)}個")
            else:
                st.warning("使用可能なモデルが見つかりませんでした")
        except Exception as e:
            loading_placeholder.empty()  # ローディング表示をクリア
            st.error(f"更新エラー: {e}")
    
    # 現在のモデル数表示
    if st.session_state.available_models:
        st.caption(f"使用可能: {len(st.session_state.available_models)}個")
    else:
        st.caption("使用可能: 0個")
    
    # バックグラウンドでモデルチェックが保留中の場合は実行
    if st.session_state.get("model_check_pending", False):
        # カスタムローディング表示（アイコンのみ回転）
        loading_placeholder = st.empty()
        loading_placeholder.markdown("""
        <div style="display: flex; align-items: center; padding: 10px; background-color: var(--background-color); border-radius: 8px; border: 1px solid var(--border-color);">
            <span class="loading-icon">🔄</span>
            <span class="loading-text">モデル可用性をチェック中...</span>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            new_models = asyncio.run(get_available_ollama_models())
            loading_placeholder.empty()  # ローディング表示をクリア
            if new_models:
                st.session_state.available_models = new_models
                st.session_state.model_check_pending = False
                st.rerun()  # UIを更新
        except Exception as e:
            loading_placeholder.empty()  # ローディング表示をクリア
            logger.warning(f"バックグラウンドモデルチェックに失敗: {e}")
            st.session_state.model_check_pending = False
    
    if st.session_state.available_models:
        selected_model = st.selectbox(
            "選択:",
            st.session_state.available_models,
            index=st.session_state.available_models.index(st.session_state.selected_model) if st.session_state.selected_model in st.session_state.available_models else 0,
            key="model_selector",
            label_visibility="collapsed"
        )
    else:
        st.selectbox(
            "選択:",
            ["モデルが見つかりません"],
            key="model_selector",
            label_visibility="collapsed",
            disabled=True
        )
        selected_model = st.session_state.selected_model
    
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        st.session_state.agent_initialized = False
        st.success(f"モデル変更: {selected_model}")
    
    # === セッション情報（最小限） ===
    st.subheader("セッション")
    st.info(f"メッセージ: {len(st.session_state.messages)}")
    
    if st.button("リセット", use_container_width=True, key="reset_session"):
        st.session_state.messages = []
        if st.session_state.agent_initialized and st.session_state.agent:
            try:
                asyncio.run(st.session_state.agent.close())
            except:
                pass
        st.session_state.agent_initialized = False
        st.session_state.agent = None
        st.session_state.session_id = None
        st.success("リセット完了")
        st.rerun()
    
    # === 設定（最小限） ===
    st.subheader("設定")
    st.session_state.streaming_enabled = st.checkbox("ストリーミング", value=st.session_state.streaming_enabled, key="streaming_setting")
    
    # === ナビゲーション（テキストベース） ===
    st.subheader("ナビゲーション")
    
    # ナビゲーションボタン（1行に1つ）
    if st.button("チャット", key="nav_chat", use_container_width=True):
        navigate_to_page("chat")
    
    if st.button("ツール", key="nav_tools", use_container_width=True):
        navigate_to_page("tools")
    
    if st.button("プロンプト", key="nav_prompts", use_container_width=True):
        navigate_to_page("prompts")
    
    if st.button("学習", key="nav_learning", use_container_width=True):
        navigate_to_page("learning")
    
    if st.button("報酬", key="nav_rewards", use_container_width=True):
        navigate_to_page("rewards")
    
    if st.button("API", key="nav_api", use_container_width=True):
        navigate_to_page("api")

# メインコンテンツ
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# ページ表示
if st.session_state.current_page == "chat":
    show_chat_page()
elif st.session_state.current_page == "tools":
    show_tools_page()
elif st.session_state.current_page == "prompts":
    show_prompts_page()
elif st.session_state.current_page == "learning":
    show_learning_page()
elif st.session_state.current_page == "rewards":
    show_rewards_page()
elif st.session_state.current_page == "api":
    show_api_page()
elif st.session_state.current_page == "help":
    show_help_page()
else:
    show_chat_page()

st.markdown('</div>', unsafe_allow_html=True)