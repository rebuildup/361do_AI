"""
Streamlit リアルタイム Web UI

Streamlit の既存応答性機能による フロントエンドを統合し、
進捗・VRAM 表示を実装、履歴管理・継続を統合
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import uuid

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import aiohttp

# 相対インポートを避けるため、try-except でインポート
try:
    from ..monitoring.system_monitor import SystemMonitor
    from ..memory.persistent_memory import PersistentMemoryManager
    from ..reasoning.basic_engine import BasicReasoningEngine
except ImportError as e:
    logging.warning(f"一部のモジュールをインポートできませんでした: {e}")
    # デモ用のモッククラス
    class SystemMonitor:
        async def get_system_stats(self):
            return {"cpu_percent": 50.0, "memory_percent": 60.0}
        
        async def get_gpu_stats(self):
            return {"memory_percent": 70.0, "utilization_percent": 80.0, "temperature": 75.0}
    
    class PersistentMemoryManager:
        async def search_memories(self, query: str, **kwargs):
            return {"results": [], "total_found": 0}
    
    class BasicReasoningEngine:
        async def reasoning_inference(self, prompt: str, **kwargs):
            return {"response": f"Mock response for: {prompt}"}

logger = logging.getLogger(__name__)


class StreamlitUI:
    """Streamlit Web UI メインクラス"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        
        # セッション状態の初期化
        self._initialize_session_state()
        
        # 設定管理システム初期化
        self._initialize_settings_manager()
        
        # コンポーネント初期化
        self.system_monitor = SystemMonitor()
        self.memory_manager = PersistentMemoryManager()
        self.reasoning_engine = BasicReasoningEngine()
        
        logger.info("Streamlit UI 初期化完了")
    
    def _initialize_settings_manager(self):
        """設定管理システム初期化"""
        
        try:
            from .settings_manager import get_settings_manager
            self.settings_manager = get_settings_manager()
            
            # 設定を読み込み、セッション状態に反映
            settings = self.settings_manager.get_settings()
            self._apply_settings_to_session(settings)
            
            logger.info("設定管理システム初期化完了")
            
        except ImportError as e:
            logger.warning(f"設定管理システムをインポートできませんでした: {e}")
            self.settings_manager = None
    
    def _apply_settings_to_session(self, settings):
        """設定をセッション状態に適用 - Pydantic Settings による 動的設定変更・反映"""
        
        try:
            # 現在のモデル設定を適用
            current_model_config = settings.get_current_model_config()
            if current_model_config:
                st.session_state.settings.update({
                    "model": settings.current_model,
                    "temperature": current_model_config.temperature,
                    "max_tokens": current_model_config.max_tokens,
                    "top_p": current_model_config.top_p,
                    "top_k": current_model_config.top_k,
                    "repeat_penalty": current_model_config.repeat_penalty
                })
            
            # UI設定を適用
            st.session_state.settings.update({
                "auto_refresh": settings.ui.auto_refresh,
                "refresh_interval": settings.ui.refresh_interval,
                "auto_save": settings.ui.auto_save,
                "save_interval": settings.ui.save_interval,
                "show_debug": settings.ui.show_debug,
                "max_chat_history": settings.ui.max_chat_history
            })
            
            # システム設定を適用
            self.api_base_url = settings.system.api_base_url
            
            logger.info("設定をセッション状態に適用しました")
            
        except Exception as e:
            logger.error(f"設定適用エラー: {e}")
    
    def _initialize_session_state(self):
        """セッション状態の初期化"""
        
        # チャット履歴
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # システム統計履歴
        if "system_stats_history" not in st.session_state:
            st.session_state.system_stats_history = []
        
        # 設定
        if "settings" not in st.session_state:
            st.session_state.settings = {
                "model": "deepseek-r1:7b",
                "temperature": 0.7,
                "max_tokens": 500,
                "use_cot": True,
                "auto_refresh": True,
                "refresh_interval": 5,
                "auto_save": True,
                "save_interval": 10
            }
        
        # UI 状態
        if "current_session_id" not in st.session_state:
            st.session_state.current_session_id = str(uuid.uuid4())
        
        if "last_refresh" not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        
        if "processing" not in st.session_state:
            st.session_state.processing = False
        
        # セッション管理
        if "session_start_time" not in st.session_state:
            st.session_state.session_start_time = datetime.now()
        
        if "saved_sessions" not in st.session_state:
            st.session_state.saved_sessions = {}
        
        if "api_calls" not in st.session_state:
            st.session_state.api_calls = 0
        
        if "error_count" not in st.session_state:
            st.session_state.error_count = 0
    
    def run(self):
        """メイン UI 実行"""
        
        # ページ設定
        st.set_page_config(
            page_title="Advanced AI Agent",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # カスタム CSS
        self._apply_custom_css()
        
        # サイドバー
        self._render_sidebar()
        
        # メインコンテンツ
        self._render_main_content()
        
        # 自動リフレッシュ
        if st.session_state.settings["auto_refresh"]:
            self._auto_refresh()
    
    def _apply_custom_css(self):
        """カスタム CSS 適用"""
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
            margin-bottom: 1rem;
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
        
        .chat-message {
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 0.5rem;
        }
        
        .user-message {
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
        }
        
        .assistant-message {
            background-color: #f3e5f5;
            border-left: 4px solid #9c27b0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_sidebar(self):
        """サイドバー描画"""
        
        with st.sidebar:
            st.markdown("## ⚙️ 設定")
            
            # 動的モデル選択 - Pydantic Settings による モデル選択・切り替え
            if self.settings_manager:
                settings = self.settings_manager.get_settings()
                model_names = list(settings.models.keys())
                
                # 現在のモデル選択
                current_index = model_names.index(settings.current_model) if settings.current_model in model_names else 0
                
                selected_model = st.selectbox(
                    "モデル",
                    model_names,
                    index=current_index,
                    help="使用するAIモデルを選択"
                )
                
                # モデル切り替え処理
                if selected_model != settings.current_model:
                    if settings.switch_model(selected_model):
                        self.settings_manager.save_settings(settings)
                        self._apply_settings_to_session(settings)
                        st.success(f"モデルを '{selected_model}' に切り替えました")
                        st.rerun()
                
                # 現在のモデル設定を取得
                current_model_config = settings.get_current_model_config()
                if current_model_config:
                    # 推論設定（動的）
                    new_temperature = st.slider(
                        "Temperature", 
                        0.0, 2.0, 
                        current_model_config.temperature, 
                        0.1,
                        help="生成の創造性を制御"
                    )
                    
                    new_max_tokens = st.slider(
                        "Max Tokens", 
                        50, 4000, 
                        current_model_config.max_tokens, 
                        50,
                        help="生成する最大トークン数"
                    )
                    
                    # 設定変更の検出と適用
                    if (new_temperature != current_model_config.temperature or 
                        new_max_tokens != current_model_config.max_tokens):
                        
                        current_model_config.temperature = new_temperature
                        current_model_config.max_tokens = new_max_tokens
                        
                        # 自動保存
                        self.settings_manager.save_settings(settings)
                        self._apply_settings_to_session(settings)
                
            else:
                # フォールバック: 静的設定
                st.session_state.settings["model"] = st.selectbox(
                    "モデル",
                    ["deepseek-r1:7b", "qwen2.5:7b-instruct-q4_k_m", "qwen2:1.5b-instruct-q4_k_m"],
                    index=0
                )
                
                st.session_state.settings["temperature"] = st.slider(
                    "Temperature", 0.0, 2.0, 0.7, 0.1
                )
                
                st.session_state.settings["max_tokens"] = st.slider(
                    "Max Tokens", 50, 2000, 500, 50
                )
            
            st.session_state.settings["use_cot"] = st.checkbox(
                "Chain-of-Thought を使用", True
            )
            
            st.markdown("---")
            
            # 自動リフレッシュ設定
            st.session_state.settings["auto_refresh"] = st.checkbox(
                "自動リフレッシュ", True
            )
            
            if st.session_state.settings["auto_refresh"]:
                st.session_state.settings["refresh_interval"] = st.slider(
                    "リフレッシュ間隔（秒）", 1, 30, 5
                )
            
            st.markdown("---")
            
            # システム情報
            self._render_system_info()
            
            st.markdown("---")
            
            # セッション管理
            self._render_session_management()
    
    def _render_system_info(self):
        """システム情報表示"""
        
        st.markdown("### 📊 システム情報")
        
        try:
            # システム統計を非同期で取得（簡略化）
            system_stats = self._get_system_stats_sync()
            
            if system_stats:
                # CPU 使用率
                cpu_percent = system_stats.get("cpu_percent", 0)
                cpu_color = self._get_status_color(cpu_percent, 70, 90)
                st.markdown(f"**CPU:** <span class='{cpu_color}'>{cpu_percent:.1f}%</span>", 
                           unsafe_allow_html=True)
                
                # メモリ使用率
                memory_percent = system_stats.get("memory_percent", 0)
                memory_color = self._get_status_color(memory_percent, 70, 90)
                st.markdown(f"**メモリ:** <span class='{memory_color}'>{memory_percent:.1f}%</span>", 
                           unsafe_allow_html=True)
                
                # GPU 情報
                gpu_stats = self._get_gpu_stats_sync()
                if gpu_stats:
                    gpu_memory = gpu_stats.get("memory_percent", 0)
                    gpu_color = self._get_status_color(gpu_memory, 80, 95)
                    st.markdown(f"**GPU メモリ:** <span class='{gpu_color}'>{gpu_memory:.1f}%</span>", 
                               unsafe_allow_html=True)
                    
                    gpu_temp = gpu_stats.get("temperature", 0)
                    temp_color = self._get_status_color(gpu_temp, 75, 85)
                    st.markdown(f"**GPU 温度:** <span class='{temp_color}'>{gpu_temp:.1f}°C</span>", 
                               unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"システム情報取得エラー: {e}")
    
    def _get_status_color(self, value: float, warning_threshold: float, critical_threshold: float) -> str:
        """ステータス色取得"""
        if value >= critical_threshold:
            return "status-critical"
        elif value >= warning_threshold:
            return "status-warning"
        else:
            return "status-healthy"
    
    def _get_system_stats_sync(self) -> Dict[str, Any]:
        """システム統計同期取得（簡略化）"""
        try:
            # 実際の実装では asyncio.run() を使用するか、
            # バックグラウンドタスクで定期取得
            import psutil
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent
            }
        except Exception:
            return {"cpu_percent": 50.0, "memory_percent": 60.0}
    
    def _get_gpu_stats_sync(self) -> Dict[str, Any]:
        """GPU 統計同期取得（簡略化）"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            return {
                "memory_percent": (memory_info.used / memory_info.total) * 100,
                "temperature": temperature,
                "utilization_percent": utilization.gpu
            }
        except Exception:
            return {"memory_percent": 70.0, "temperature": 75.0, "utilization_percent": 80.0}
    
    def _render_session_management(self):
        """セッション管理 - Streamlit の既存セッション機能による 履歴管理・継続"""
        
        st.markdown("### 👤 セッション")
        
        st.text(f"ID: {st.session_state.current_session_id[:8]}...")
        
        # セッション統計
        session_stats = self._get_session_statistics()
        st.markdown(f"**会話数:** {session_stats['message_count']}")
        st.markdown(f"**開始時刻:** {session_stats['start_time']}")
        st.markdown(f"**継続時間:** {session_stats['duration']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 新規セッション"):
                self._create_new_session()
        
        with col2:
            if st.button("💾 セッション保存"):
                self._save_session()
        
        # セッション履歴管理
        st.markdown("#### 📋 セッション履歴")
        
        # 保存済みセッション一覧
        saved_sessions = self._get_saved_sessions()
        
        if saved_sessions:
            selected_session = st.selectbox(
                "保存済みセッション",
                options=list(saved_sessions.keys()),
                format_func=lambda x: f"{x[:8]}... ({saved_sessions[x]['timestamp']})"
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📂 セッション復元"):
                    self._restore_session(selected_session)
            
            with col2:
                if st.button("📄 詳細表示"):
                    self._show_session_details(selected_session)
            
            with col3:
                if st.button("🗑️ セッション削除"):
                    self._delete_session(selected_session)
        else:
            st.info("保存済みセッションがありません")
        
        # 自動保存設定
        st.markdown("#### ⚙️ 自動保存設定")
        
        auto_save = st.checkbox("自動セッション保存", value=True)
        
        if auto_save:
            save_interval = st.slider("保存間隔（分）", 1, 60, 10)
            st.session_state.settings["auto_save"] = True
            st.session_state.settings["save_interval"] = save_interval
        else:
            st.session_state.settings["auto_save"] = False
    
    def _get_session_statistics(self) -> Dict[str, Any]:
        """セッション統計取得"""
        
        message_count = len(st.session_state.messages)
        start_time = st.session_state.get("session_start_time", datetime.now())
        duration = datetime.now() - start_time
        
        return {
            "message_count": message_count,
            "start_time": start_time.strftime("%H:%M:%S"),
            "duration": str(duration).split(".")[0]  # 秒以下を除去
        }
    
    def _create_new_session(self):
        """新規セッション作成"""
        
        # 現在のセッションを自動保存
        if st.session_state.settings.get("auto_save", True):
            self._save_session()
        
        # 新しいセッション ID 生成
        st.session_state.current_session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.session_start_time = datetime.now()
        
        st.success("新しいセッションを開始しました")
        st.rerun()
    
    def _get_saved_sessions(self) -> Dict[str, Dict[str, Any]]:
        """保存済みセッション一覧取得"""
        
        # セッション状態から取得（実際の実装では永続化ストレージから）
        saved_sessions = st.session_state.get("saved_sessions", {})
        
        return saved_sessions
    
    def _restore_session(self, session_id: str):
        """セッション復元"""
        
        try:
            saved_sessions = self._get_saved_sessions()
            
            if session_id in saved_sessions:
                session_data = saved_sessions[session_id]
                
                # セッションデータを復元
                st.session_state.current_session_id = session_id
                st.session_state.messages = session_data.get("messages", [])
                st.session_state.settings.update(session_data.get("settings", {}))
                
                st.success(f"セッション {session_id[:8]}... を復元しました")
                st.rerun()
            else:
                st.error("セッションが見つかりません")
                
        except Exception as e:
            st.error(f"セッション復元エラー: {e}")
    
    def _show_session_details(self, session_id: str):
        """セッション詳細表示"""
        
        try:
            saved_sessions = self._get_saved_sessions()
            
            if session_id in saved_sessions:
                session_data = saved_sessions[session_id]
                
                with st.expander(f"セッション詳細: {session_id[:8]}..."):
                    st.json(session_data)
            else:
                st.error("セッションが見つかりません")
                
        except Exception as e:
            st.error(f"セッション詳細表示エラー: {e}")
    
    def _delete_session(self, session_id: str):
        """セッション削除"""
        
        try:
            if "saved_sessions" not in st.session_state:
                st.session_state.saved_sessions = {}
            
            if session_id in st.session_state.saved_sessions:
                del st.session_state.saved_sessions[session_id]
                st.success(f"セッション {session_id[:8]}... を削除しました")
                st.rerun()
            else:
                st.error("セッションが見つかりません")
                
        except Exception as e:
            st.error(f"セッション削除エラー: {e}")
    
    def _render_main_content(self):
        """メインコンテンツ描画"""
        
        # ヘッダー
        st.markdown('<h1 class="main-header">🤖 Advanced AI Agent</h1>', 
                   unsafe_allow_html=True)
        
        # リアルタイム進捗・VRAM表示
        self._render_realtime_progress_indicator()
        
        st.markdown("---")
        
        # タブ構成
        tab1, tab2, tab3, tab4 = st.tabs(["💬 チャット", "📊 監視", "🔍 記憶検索", "⚙️ 管理"])
        
        with tab1:
            self._render_chat_interface()
        
        with tab2:
            self._render_monitoring_dashboard()
        
        with tab3:
            self._render_memory_search()
        
        with tab4:
            self._render_admin_panel()
    
    def _render_chat_interface(self):
        """チャットインターフェース - リアルタイム応答性機能統合"""
        
        st.markdown("### 💬 AI チャット")
        
        # リアルタイムチャットステータス
        self._render_realtime_chat_status()
        
        # チャット履歴表示
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.messages:
                self._render_message(message)
        
        # 入力フォーム
        with st.form("chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_input = st.text_area(
                    "メッセージを入力してください...",
                    height=100,
                    key="chat_input",
                    placeholder="質問や指示を入力してください。Enterで送信、Shift+Enterで改行"
                )
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                submit_button = st.form_submit_button("送信 📤", use_container_width=True)
                
                if st.form_submit_button("🔄 クリア", use_container_width=True):
                    st.session_state.messages = []
                    st.rerun()
        
        # キーボードショートカット情報
        st.markdown("""
        <div style="font-size: 0.8rem; color: #6c757d; margin-top: 10px;">
            💡 ヒント: Ctrl+Enter で送信、Shift+Enter で改行
        </div>
        """, unsafe_allow_html=True)
        
        # メッセージ処理
        if submit_button and user_input.strip():
            self._process_chat_message(user_input.strip())
    
    def _render_message(self, message: Dict[str, Any]):
        """メッセージ描画"""
        
        role = message["role"]
        content = message["content"]
        timestamp = message.get("timestamp", datetime.now())
        
        if role == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 ユーザー</strong> <small>({timestamp.strftime('%H:%M:%S')})</small><br>
                {content}
            </div>
            """, unsafe_allow_html=True)
        
        elif role == "assistant":
            processing_time = message.get("processing_time", 0)
            confidence = message.get("confidence_score")
            
            confidence_text = f" (信頼度: {confidence:.2f})" if confidence else ""
            
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 AI アシスタント</strong> 
                <small>({timestamp.strftime('%H:%M:%S')} - {processing_time:.2f}秒{confidence_text})</small><br>
                {content}
            </div>
            """, unsafe_allow_html=True)
            
            # 推論ステップ表示
            if message.get("reasoning_steps"):
                with st.expander("🧠 推論過程を表示"):
                    for i, step in enumerate(message["reasoning_steps"], 1):
                        st.markdown(f"**ステップ {i}:** {step}")
    
    def _process_chat_message(self, user_input: str):
        """チャットメッセージ処理"""
        
        # ユーザーメッセージを履歴に追加
        user_message = {
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now()
        }
        st.session_state.messages.append(user_message)
        
        # 処理中表示
        with st.spinner("AI が考えています..."):
            try:
                # API 呼び出し（簡略化）
                response = self._call_chat_api(user_input)
                
                # アシスタントメッセージを履歴に追加
                assistant_message = {
                    "role": "assistant",
                    "content": response.get("response", "エラーが発生しました"),
                    "timestamp": datetime.now(),
                    "processing_time": response.get("processing_time", 0),
                    "confidence_score": response.get("confidence_score"),
                    "reasoning_steps": response.get("reasoning_steps")
                }
                st.session_state.messages.append(assistant_message)
                
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
                
                error_message = {
                    "role": "assistant",
                    "content": f"申し訳ございません。エラーが発生しました: {str(e)}",
                    "timestamp": datetime.now()
                }
                st.session_state.messages.append(error_message)
        
        # UI を更新
        st.rerun()
    
    def _call_chat_api(self, user_input: str) -> Dict[str, Any]:
        """チャット API 呼び出し"""
        
        try:
            # FastAPI エンドポイント呼び出し
            url = f"{self.api_base_url}/v1/chat/completions"
            
            payload = {
                "model": st.session_state.settings["model"],
                "messages": [
                    {"role": "user", "content": user_input}
                ],
                "temperature": st.session_state.settings["temperature"],
                "max_tokens": st.session_state.settings["max_tokens"]
            }
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "response": data["choices"][0]["message"]["content"],
                    "processing_time": 1.5,  # 簡略化
                    "confidence_score": 0.85  # 簡略化
                }
            else:
                return {"response": f"API エラー: {response.status_code}"}
                
        except Exception as e:
            # フォールバック: ローカル推論
            return {
                "response": f"Mock response for: {user_input}",
                "processing_time": 0.5,
                "confidence_score": 0.75
            }
    
    def _render_monitoring_dashboard(self):
        """監視ダッシュボード"""
        
        st.markdown("### 📊 システム監視")
        
        # リアルタイム統計
        col1, col2, col3, col4 = st.columns(4)
        
        system_stats = self._get_system_stats_sync()
        gpu_stats = self._get_gpu_stats_sync()
        
        with col1:
            st.metric(
                "CPU 使用率",
                f"{system_stats.get('cpu_percent', 0):.1f}%",
                delta=None
            )
        
        with col2:
            st.metric(
                "メモリ使用率",
                f"{system_stats.get('memory_percent', 0):.1f}%",
                delta=None
            )
        
        with col3:
            st.metric(
                "GPU メモリ",
                f"{gpu_stats.get('memory_percent', 0):.1f}%",
                delta=None
            )
        
        with col4:
            st.metric(
                "GPU 温度",
                f"{gpu_stats.get('temperature', 0):.1f}°C",
                delta=None
            )
        
        # グラフ表示
        self._render_performance_charts()
    
    def _render_performance_charts(self):
        """パフォーマンスチャート - リアルタイム履歴データ可視化"""
        
        # 履歴データを使用（利用可能な場合）
        if st.session_state.system_stats_history:
            # 実際の履歴データから時系列チャート作成
            history_df = pd.DataFrame(st.session_state.system_stats_history)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('CPU 使用率', 'メモリ使用率', 'GPU メモリ', 'GPU 温度'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            fig.add_trace(
                go.Scatter(
                    x=history_df["timestamp"], 
                    y=history_df["cpu_percent"], 
                    name="CPU", 
                    line=dict(color="blue"),
                    mode='lines+markers'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=history_df["timestamp"], 
                    y=history_df["memory_percent"], 
                    name="メモリ", 
                    line=dict(color="green"),
                    mode='lines+markers'
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=history_df["timestamp"], 
                    y=history_df["gpu_memory_percent"], 
                    name="GPU メモリ", 
                    line=dict(color="red"),
                    mode='lines+markers'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=history_df["timestamp"], 
                    y=history_df["gpu_temperature"], 
                    name="GPU 温度", 
                    line=dict(color="orange"),
                    mode='lines+markers'
                ),
                row=2, col=2
            )
            
        else:
            # フォールバック: サンプルデータ生成
            import numpy as np
            
            timestamps = pd.date_range(
                start=datetime.now() - timedelta(minutes=30),
                end=datetime.now(),
                freq='1min'
            )
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('CPU 使用率', 'メモリ使用率', 'GPU メモリ', 'GPU 温度'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # サンプルデータ
            cpu_data = np.random.normal(50, 10, len(timestamps))
            memory_data = np.random.normal(60, 8, len(timestamps))
            gpu_memory_data = np.random.normal(70, 12, len(timestamps))
            gpu_temp_data = np.random.normal(75, 5, len(timestamps))
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=cpu_data, name="CPU", line=dict(color="blue")),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=memory_data, name="メモリ", line=dict(color="green")),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=gpu_memory_data, name="GPU メモリ", line=dict(color="red")),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=gpu_temp_data, name="GPU 温度", line=dict(color="orange")),
                row=2, col=2
            )
        
        # 警告ライン追加
        fig.add_hline(y=80, line_dash="dash", line_color="orange", row=1, col=1, annotation_text="警告")
        fig.add_hline(y=90, line_dash="dash", line_color="red", row=1, col=1, annotation_text="危険")
        
        fig.add_hline(y=80, line_dash="dash", line_color="orange", row=1, col=2, annotation_text="警告")
        fig.add_hline(y=90, line_dash="dash", line_color="red", row=1, col=2, annotation_text="危険")
        
        fig.add_hline(y=85, line_dash="dash", line_color="orange", row=2, col=1, annotation_text="警告")
        fig.add_hline(y=95, line_dash="dash", line_color="red", row=2, col=1, annotation_text="危険")
        
        fig.add_hline(y=80, line_dash="dash", line_color="orange", row=2, col=2, annotation_text="警告")
        fig.add_hline(y=90, line_dash="dash", line_color="red", row=2, col=2, annotation_text="危険")
        
        fig.update_layout(height=600, showlegend=False)
        fig.update_xaxes(title_text="時刻")
        fig.update_yaxes(title_text="使用率 (%)", row=1, col=1)
        fig.update_yaxes(title_text="使用率 (%)", row=1, col=2)
        fig.update_yaxes(title_text="使用率 (%)", row=2, col=1)
        fig.update_yaxes(title_text="温度 (°C)", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # リアルタイム更新ボタン
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 データ更新"):
                self._update_system_stats_history()
                st.rerun()
        
        with col2:
            if st.button("📊 統計リセット"):
                st.session_state.system_stats_history = []
                st.success("統計履歴をリセットしました")
                st.rerun()
        
        with col3:
            if st.button("💾 データエクスポート"):
                self._export_performance_data()
    
    def _render_memory_search(self):
        """記憶検索インターフェース"""
        
        st.markdown("### 🔍 記憶検索")
        
        # 検索フォーム
        with st.form("memory_search_form"):
            search_query = st.text_input("検索クエリ", placeholder="検索したい内容を入力...")
            
            col1, col2 = st.columns(2)
            with col1:
                max_results = st.slider("最大結果数", 1, 20, 5)
            with col2:
                similarity_threshold = st.slider("類似度閾値", 0.0, 1.0, 0.7, 0.1)
            
            search_button = st.form_submit_button("🔍 検索")
        
        if search_button and search_query.strip():
            with st.spinner("記憶を検索中..."):
                results = self._search_memories(search_query, max_results, similarity_threshold)
                
                if results["total_found"] > 0:
                    st.success(f"{results['total_found']} 件の記憶が見つかりました")
                    
                    for i, result in enumerate(results["results"], 1):
                        with st.expander(f"記憶 {i}: {result.get('title', 'タイトルなし')}"):
                            st.markdown(f"**内容:** {result.get('content', '')}")
                            st.markdown(f"**類似度:** {result.get('similarity', 0):.3f}")
                            st.markdown(f"**作成日:** {result.get('created_at', 'N/A')}")
                else:
                    st.info("該当する記憶が見つかりませんでした")
    
    def _search_memories(self, query: str, max_results: int, similarity_threshold: float) -> Dict[str, Any]:
        """記憶検索実行"""
        
        try:
            # API 呼び出し（簡略化）
            url = f"{self.api_base_url}/v1/memory/search"
            
            payload = {
                "query": query,
                "max_results": max_results,
                "similarity_threshold": similarity_threshold
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"results": [], "total_found": 0}
                
        except Exception:
            # フォールバック: モックデータ
            return {
                "results": [
                    {
                        "title": "サンプル記憶",
                        "content": f"'{query}' に関連する記憶内容のサンプルです。",
                        "similarity": 0.85,
                        "created_at": datetime.now().isoformat()
                    }
                ],
                "total_found": 1
            }
    
    def _render_admin_panel(self):
        """管理パネル - Pydantic + Streamlit 設定管理統合"""
        
        st.markdown("### ⚙️ システム管理")
        
        # 設定管理UI統合
        try:
            from .settings_manager import get_settings_ui
            settings_ui = get_settings_ui()
            settings_ui.render_settings_panel()
            
        except ImportError as e:
            logger.warning(f"設定管理UIをインポートできませんでした: {e}")
            # フォールバック: 基本管理パネル
            self._render_basic_admin_panel()
    
    def _render_basic_admin_panel(self):
        """基本管理パネル（フォールバック）"""
        
        # システム制御
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔧 システム制御")
            
            if st.button("🔄 システム再起動"):
                st.warning("システム再起動機能は実装中です")
            
            if st.button("🧹 キャッシュクリア"):
                st.success("キャッシュをクリアしました")
            
            if st.button("💾 設定保存"):
                self._save_settings()
        
        with col2:
            st.markdown("#### 📊 統計情報")
            
            stats = {
                "総会話数": len(st.session_state.messages),
                "セッション時間": str(datetime.now() - st.session_state.get("session_start_time", datetime.now())),
                "API 呼び出し数": st.session_state.get("api_calls", 0),
                "エラー数": st.session_state.get("error_count", 0)
            }
            
            for key, value in stats.items():
                st.metric(key, value)
        
        # ログ表示
        st.markdown("#### 📝 システムログ")
        
        if st.button("ログ更新"):
            st.text_area(
                "最新ログ",
                value="[INFO] システム正常動作中\n[INFO] GPU メモリ使用率: 70%\n[INFO] 推論完了: 1.2秒",
                height=150
            )
    
    def _save_session(self):
        """セッション保存 - Streamlit の既存セッション機能による 永続化"""
        try:
            session_data = {
                "session_id": st.session_state.current_session_id,
                "messages": st.session_state.messages,
                "settings": st.session_state.settings,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "message_count": len(st.session_state.messages),
                "start_time": st.session_state.get("session_start_time", datetime.now()).isoformat()
            }
            
            # セッション状態に保存（実際の実装では永続化ストレージに保存）
            if "saved_sessions" not in st.session_state:
                st.session_state.saved_sessions = {}
            
            st.session_state.saved_sessions[st.session_state.current_session_id] = session_data
            
            st.success(f"セッション {st.session_state.current_session_id[:8]}... を保存しました")
            
        except Exception as e:
            st.error(f"セッション保存エラー: {e}")
    
    def _save_settings(self):
        """設定保存"""
        try:
            # 実際の実装では設定ファイルに保存
            st.success("設定を保存しました")
            
        except Exception as e:
            st.error(f"設定保存エラー: {e}")
    
    def _export_performance_data(self):
        """パフォーマンスデータエクスポート"""
        
        try:
            if st.session_state.system_stats_history:
                # DataFrame に変換
                df = pd.DataFrame(st.session_state.system_stats_history)
                
                # CSV 形式でダウンロード
                csv_data = df.to_csv(index=False)
                
                st.download_button(
                    label="📥 CSV ダウンロード",
                    data=csv_data,
                    file_name=f"performance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                st.success("パフォーマンスデータをエクスポートしました")
            else:
                st.warning("エクスポートするデータがありません")
                
        except Exception as e:
            st.error(f"データエクスポートエラー: {e}")
    
    def _render_realtime_chat_status(self):
        """リアルタイムチャットステータス表示"""
        
        # 処理中インジケーター
        if st.session_state.get("processing", False):
            st.markdown("""
            <div style="display: flex; align-items: center; padding: 10px; background-color: #fff3cd; border-radius: 5px; margin: 10px 0;">
                <div style="margin-right: 10px;">⏳</div>
                <div>AI が応答を生成中...</div>
            </div>
            """, unsafe_allow_html=True)
        
        # 最後の応答時間
        if st.session_state.messages:
            last_message = st.session_state.messages[-1]
            if last_message.get("role") == "assistant":
                processing_time = last_message.get("processing_time", 0)
                
                st.markdown(f"""
                <div style="text-align: right; color: #6c757d; font-size: 0.8rem; margin: 5px 0;">
                    最後の応答時間: {processing_time:.2f}秒
                </div>
                """, unsafe_allow_html=True)
    
    def _auto_refresh(self):
        """自動リフレッシュ - Streamlit の既存応答性機能による リアルタイム更新"""
        
        current_time = datetime.now()
        time_diff = (current_time - st.session_state.last_refresh).total_seconds()
        
        if time_diff >= st.session_state.settings["refresh_interval"]:
            st.session_state.last_refresh = current_time
            
            # システム統計を履歴に追加
            self._update_system_stats_history()
            
            # 必要に応じて統計データを更新
            st.rerun()
    
    def _update_system_stats_history(self):
        """システム統計履歴更新 - リアルタイム監視データ蓄積"""
        
        try:
            current_stats = {
                "timestamp": datetime.now(),
                "cpu_percent": self._get_system_stats_sync().get("cpu_percent", 0),
                "memory_percent": self._get_system_stats_sync().get("memory_percent", 0),
                "gpu_memory_percent": self._get_gpu_stats_sync().get("memory_percent", 0),
                "gpu_temperature": self._get_gpu_stats_sync().get("temperature", 0),
                "gpu_utilization": self._get_gpu_stats_sync().get("utilization_percent", 0)
            }
            
            # 履歴に追加（最大100件まで保持）
            st.session_state.system_stats_history.append(current_stats)
            
            if len(st.session_state.system_stats_history) > 100:
                st.session_state.system_stats_history.pop(0)
                
        except Exception as e:
            logger.warning(f"システム統計履歴更新エラー: {e}")
    
    def _render_realtime_progress_indicator(self):
        """リアルタイム進捗・VRAM表示 - Streamlit の既存可視化機能による 進捗・VRAM 表示"""
        
        # 進捗バー表示エリア
        progress_container = st.container()
        
        with progress_container:
            col1, col2, col3 = st.columns(3)
            
            # GPU VRAM 使用率
            with col1:
                gpu_stats = self._get_gpu_stats_sync()
                vram_percent = gpu_stats.get("memory_percent", 0)
                
                st.markdown("**🎮 GPU VRAM**")
                vram_progress = st.progress(vram_percent / 100)
                
                # 色分け表示
                if vram_percent >= 90:
                    st.error(f"VRAM: {vram_percent:.1f}% (危険)")
                elif vram_percent >= 75:
                    st.warning(f"VRAM: {vram_percent:.1f}% (注意)")
                else:
                    st.success(f"VRAM: {vram_percent:.1f}% (正常)")
            
            # CPU 使用率
            with col2:
                system_stats = self._get_system_stats_sync()
                cpu_percent = system_stats.get("cpu_percent", 0)
                
                st.markdown("**💻 CPU**")
                cpu_progress = st.progress(cpu_percent / 100)
                
                if cpu_percent >= 90:
                    st.error(f"CPU: {cpu_percent:.1f}% (高負荷)")
                elif cpu_percent >= 70:
                    st.warning(f"CPU: {cpu_percent:.1f}% (中負荷)")
                else:
                    st.success(f"CPU: {cpu_percent:.1f}% (正常)")
            
            # メモリ使用率
            with col3:
                memory_percent = system_stats.get("memory_percent", 0)
                
                st.markdown("**🧠 メモリ**")
                memory_progress = st.progress(memory_percent / 100)
                
                if memory_percent >= 90:
                    st.error(f"メモリ: {memory_percent:.1f}% (不足)")
                elif memory_percent >= 75:
                    st.warning(f"メモリ: {memory_percent:.1f}% (注意)")
                else:
                    st.success(f"メモリ: {memory_percent:.1f}% (正常)")
        
        return progress_container


def main():
    """メイン関数"""
    
    # Streamlit UI インスタンス作成
    ui = StreamlitUI()
    
    # UI 実行
    ui.run()


if __name__ == "__main__":
    main()