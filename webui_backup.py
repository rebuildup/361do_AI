import streamlit as st
import time
from datetime import datetime

# ページ設定
st.set_page_config(
    page_title="自己学習AIエージェント",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS - layout.htmlの構造を完全に忠実に再現
st.markdown("""
<style>
    :root {
        --sidebar-width-expanded: 288px;
        --sidebar-width-collapsed: 48px;
        --main-content-width: 768px;
        --chat-panel-height: 80px;
        --primary-color: #000000;
        --secondary-color: #333333;
        --background-color: #ffffff;
        --border-color: #e0e0e0;
        --text-color: #333333;
        --card-background: #f8f8f8;
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
        background-color: var(--background-color);
        color: var(--text-color);
    }

    /* Streamlitのデフォルトスタイルを完全にリセット */
    .stApp {
        background: var(--background-color) !important;
        color: var(--text-color) !important;
        height: 100vh !important;
        overflow: hidden !important;
    }

    .stApp > header {
        display: none !important;
    }

    .stApp > div {
        display: flex !important;
        height: 100vh !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* アプリケーションレイアウト */
    .app-layout {
        display: flex !important;
        height: 100vh !important;
        width: 100% !important;
    }

    /* サイドバー */
    .stSidebar,
    [data-testid="stSidebar"] {
        width: var(--sidebar-width-expanded) !important;
        min-width: var(--sidebar-width-expanded) !important;
        max-width: var(--sidebar-width-expanded) !important;
        background: var(--primary-color) !important;
        color: white !important;
        transition: width 0.3s ease !important;
        position: relative !important;
        z-index: 1000 !important;
        overflow: hidden !important;
    }

    .stSidebar.collapsed,
    [data-testid="stSidebar"].collapsed {
        width: var(--sidebar-width-collapsed) !important;
        min-width: var(--sidebar-width-collapsed) !important;
        max-width: var(--sidebar-width-collapsed) !important;
    }

    /* サイドバーが閉じた状態でのメインコンテンツエリア調整 */
    .stSidebar.collapsed ~ .stAppViewContainer,
    [data-testid="stSidebar"].collapsed ~ [data-testid="stAppViewContainer"] {
        width: calc(100vw - var(--sidebar-width-collapsed)) !important;
        max-width: calc(100vw - var(--sidebar-width-collapsed)) !important;
        margin-left: var(--sidebar-width-collapsed) !important;
    }

    /* サイドバートグルボタン */
    .sidebar-toggle {
        position: fixed !important;
        top: 8px !important;
        left: 8px !important;
        width: 32px !important;
        height: 32px !important;
        background: var(--primary-color) !important;
        border: 1px solid #666666 !important;
        color: white !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.3s ease !important;
        z-index: 1001 !important;
        border-radius: 4px !important;
    }

    .sidebar-toggle:hover {
        background: var(--secondary-color) !important;
        border-color: #999999 !important;
    }

    .sidebar-toggle i {
        font-size: 10px !important;
    }

    /* サイドバーコンテンツ */
    .sidebar-content {
        padding: 40px 20px 20px 20px !important;
        overflow: hidden !important;
    }

    .stSidebar.collapsed .sidebar-content,
    [data-testid="stSidebar"].collapsed .sidebar-content {
        opacity: 0 !important;
        visibility: hidden !important;
    }

    .sidebar-brand {
        font-size: 1.5em !important;
        font-weight: bold !important;
        margin-bottom: 30px !important;
        white-space: nowrap !important;
        color: white !important;
    }

    .sidebar-nav {
        list-style: none !important;
    }

    .nav-item {
        margin-bottom: 15px !important;
    }

    .nav-link {
        color: white !important;
        text-decoration: none !important;
        display: flex !important;
        align-items: center !important;
        padding: 12px 15px !important;
        transition: all 0.3s ease !important;
        white-space: nowrap !important;
        border-radius: 4px !important;
    }

    .nav-link:hover {
        background: var(--secondary-color) !important;
    }

    .nav-icon {
        margin-right: 12px !important;
        width: 20px !important;
        text-align: center !important;
    }

    /* サイドバー縮小時のアイコン */
    .stSidebar.collapsed .sidebar-icons,
    [data-testid="stSidebar"].collapsed .sidebar-icons {
        display: block !important;
        padding: 40px 0 20px 0 !important;
    }

    .sidebar-icons {
        display: none !important;
    }

    .sidebar-icon {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: var(--sidebar-width-collapsed) !important;
        height: 48px !important;
        color: white !important;
        text-decoration: none !important;
        margin-bottom: 10px !important;
        transition: all 0.3s ease !important;
        border-radius: 4px !important;
    }

    .sidebar-icon:hover {
        background: var(--secondary-color) !important;
    }

    .sidebar-icon i {
        font-size: 18px !important;
    }

    /* メインコンテンツエリア */
    .stAppViewContainer,
    [data-testid="stAppViewContainer"] {
        flex: 1 !important;
        display: flex !important;
        justify-content: center !important;
        background: var(--background-color) !important;
        height: 100vh !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
        width: calc(100vw - var(--sidebar-width-expanded)) !important;
        max-width: calc(100vw - var(--sidebar-width-expanded)) !important;
        margin-left: var(--sidebar-width-expanded) !important;
    }

    /* メインコンテナ */
    .main-container {
        width: var(--main-content-width) !important;
        max-width: var(--main-content-width) !important;
        min-width: var(--main-content-width) !important;
        height: 100vh !important;
        background: var(--background-color) !important;
        position: relative !important;
        display: flex !important;
        flex-direction: column !important;
        margin: 0 auto !important;
        box-sizing: border-box !important;
    }

    /* スクロール可能なメインコンテンツ */
    .main-content {
        flex: 1 !important;
        overflow-y: auto !important;
        padding: 40px !important;
        margin-bottom: var(--chat-panel-height) !important;
    }

    .page-header {
        margin-bottom: 30px !important;
    }

    .page-title {
        font-size: 2em !important;
        color: var(--primary-color) !important;
        margin-bottom: 20px !important;
        text-align: center !important;
    }

    .page-description {
        line-height: 1.6 !important;
        color: var(--text-color) !important;
        margin-bottom: 20px !important;
    }

    .content-card {
        background: var(--card-background) !important;
        padding: 20px !important;
        margin: 20px 0 !important;
        border-left: 4px solid var(--primary-color) !important;
        border-radius: 4px !important;
    }

    .content-card h3 {
        color: var(--primary-color) !important;
        margin-bottom: 10px !important;
    }

    .content-card p {
        line-height: 1.5 !important;
        color: var(--text-color) !important;
    }

    /* チャットパネル */
    .chat-panel {
        position: absolute !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        height: var(--chat-panel-height) !important;
        border-top: 2px solid var(--primary-color) !important;
        background: var(--background-color) !important;
        padding: 15px !important;
        z-index: 100 !important;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1) !important;
    }

    .chat-form {
        display: flex !important;
        gap: 10px !important;
        align-items: center !important;
        height: 100% !important;
    }

    .chat-input {
        flex: 1 !important;
        padding: 12px 15px !important;
        border: 2px solid var(--primary-color) !important;
        font-size: 14px !important;
        height: 50px !important;
        font-family: inherit !important;
        outline: none !important;
        border-radius: 4px !important;
    }

    .chat-input:focus {
        border-color: var(--secondary-color) !important;
    }

    .chat-submit {
        padding: 12px 20px !important;
        background: var(--primary-color) !important;
        color: white !important;
        border: none !important;
        cursor: pointer !important;
        font-size: 14px !important;
        height: 50px !important;
        transition: background 0.3s ease !important;
        border-radius: 4px !important;
    }

    .chat-submit:hover:not(:disabled) {
        background: var(--secondary-color) !important;
    }

    .chat-submit:disabled {
        background: #cccccc !important;
        cursor: not-allowed !important;
    }

    /* Streamlitの要素を調整 */
    .stMain,
    [data-testid="stMain"] {
        width: var(--main-content-width) !important;
        max-width: var(--main-content-width) !important;
        min-width: var(--main-content-width) !important;
        margin: 0 auto !important;
        padding: 0 !important;
        box-sizing: border-box !important;
    }

    /* Streamlitのブロックコンテナを調整 */
    .stVerticalBlock,
    [data-testid="stVerticalBlock"] {
        width: var(--main-content-width) !important;
        max-width: var(--main-content-width) !important;
        min-width: var(--main-content-width) !important;
        margin: 0 auto !important;
        padding: 0 !important;
        box-sizing: border-box !important;
    }

    .stVerticalBlock > div {
        width: var(--main-content-width) !important;
        max-width: var(--main-content-width) !important;
        min-width: var(--main-content-width) !important;
        margin: 0 auto !important;
        padding: 0 !important;
        box-sizing: border-box !important;
    }

    /* レスポンシブ対応 */
    @media (max-width: 768px) {
        :root {
            --main-content-width: 100%;
        }
        
        .main-content {
            padding: 20px !important;
        }

        .stSidebar,
        [data-testid="stSidebar"] {
            position: absolute !important;
            height: 100vh !important;
            z-index: 2000 !important;
        }

        .stSidebar:not(.collapsed),
        [data-testid="stSidebar"]:not(.collapsed) {
            box-shadow: 2px 0 10px rgba(0, 0, 0, 0.3) !important;
        }
    }

    /* ユーティリティクラス */
    .text-center {
        text-align: center !important;
    }

    .mb-2 {
        margin-bottom: 20px !important;
    }

    .feature-list {
        list-style: none !important;
        padding-left: 0 !important;
    }

    .feature-list li {
        padding: 4px 0 !important;
        position: relative !important;
        padding-left: 20px !important;
    }

    .feature-list li::before {
        content: "•" !important;
        position: absolute !important;
        left: 0 !important;
        color: var(--primary-color) !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

# カスタムJavaScript - layout.htmlの機能を完全に忠実に再現
st.markdown("""
<script>
    // DOM要素の取得
    let sidebar, sidebarToggle, chatForm, chatInput, chatSubmit;

    // サイドバー機能
    class SidebarController {
        constructor(sidebar, toggle) {
            this.sidebar = sidebar;
            this.toggle = toggle;
            this.init();
        }

        init() {
            this.toggle.addEventListener("click", () => this.toggleSidebar());
            this.setupKeyboardShortcuts();
        }

        toggleSidebar() {
            this.sidebar.classList.toggle("collapsed");
            this.updateToggleIcon();
        }

        updateToggleIcon() {
            const icon = this.toggle.querySelector("i");
            if (this.sidebar.classList.contains("collapsed")) {
                icon.className = "fas fa-chevron-right";
            } else {
                icon.className = "fas fa-bars";
            }
        }

        setupKeyboardShortcuts() {
            document.addEventListener("keydown", (e) => {
                if (e.ctrlKey && e.key === "b") {
                    e.preventDefault();
                    this.toggleSidebar();
                }
            });
        }
    }

    // チャット機能
    class ChatController {
        constructor(form, input, submit) {
            this.form = form;
            this.input = input;
            this.submit = submit;
            this.init();
        }

        init() {
            this.form.addEventListener("submit", (e) => this.handleSubmit(e));
            this.input.addEventListener("input", () => this.handleInput());
            this.input.addEventListener("keydown", (e) => this.handleKeydown(e));
        }

        handleSubmit(e) {
            e.preventDefault();
            this.sendMessage();
        }

        handleInput() {
            const hasText = this.input.value.trim().length > 0;
            this.submit.disabled = !hasText;
        }

        handleKeydown(e) {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        }

        sendMessage() {
            const message = this.input.value.trim();
            if (message) {
                console.log("メッセージ送信:", message);
                // 実際のチャット送信処理をここに実装
                this.clearInput();
                this.showConfirmation();
            }
        }

        clearInput() {
            this.input.value = "";
            this.submit.disabled = true;
        }

        showConfirmation() {
            // 簡単な送信確認
            const originalPlaceholder = this.input.placeholder;
            this.input.placeholder = "メッセージを送信しました...";
            setTimeout(() => {
                this.input.placeholder = originalPlaceholder;
            }, 2000);
        }
    }

    // アプリケーション初期化
    function initializeApp() {
        // サイドバーとトグルボタンの設定
        const initSidebar = () => {
            sidebar = document.querySelector('[data-testid="stSidebar"]');
            if (sidebar && !document.getElementById('sidebarToggle')) {
                // トグルボタンを画面左上に固定配置
                sidebarToggle = document.createElement('button');
                sidebarToggle.className = 'sidebar-toggle';
                sidebarToggle.id = 'sidebarToggle';
                sidebarToggle.innerHTML = '<i class="fas fa-bars"></i>';
                document.body.appendChild(sidebarToggle);
                
                new SidebarController(sidebar, sidebarToggle);
                console.log('Sidebar toggle button created');
            }
        };

        // 複数回試行してサイドバーを初期化
        let attempts = 0;
        const maxAttempts = 10;
        const tryInitSidebar = () => {
            attempts++;
            initSidebar();
            if (!document.getElementById('sidebarToggle') && attempts < maxAttempts) {
                setTimeout(tryInitSidebar, 200);
            }
        };
        
        tryInitSidebar();

        // チャット機能の設定
        setTimeout(() => {
            chatForm = document.getElementById('chatForm');
            chatInput = document.getElementById('chatInput');
            chatSubmit = document.getElementById('chatSubmit');
            
            if (chatForm && chatInput && chatSubmit) {
                new ChatController(chatForm, chatInput, chatSubmit);
                console.log('Chat controller initialized');
            }
        }, 1000);
    }

    // 複数の方法で初期化を試行
    function tryInitialize() {
        initializeApp();
    }

    // DOM読み込み完了後に初期化
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', tryInitialize);
    } else {
        tryInitialize();
    }

    // 複数回初期化を試行
    setTimeout(tryInitialize, 1000);
    setTimeout(tryInitialize, 2000);
    setTimeout(tryInitialize, 3000);

    // ページの変更を監視して再初期化
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.type === 'childList') {
                // サイドバーが再描画された場合、トグルボタンを再作成
                const sidebar = document.querySelector('[data-testid="stSidebar"]');
                if (sidebar && !document.getElementById('sidebarToggle')) {
                    setTimeout(tryInitialize, 100);
                }
            }
        });
    });

    // 監視を開始
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
</script>
""", unsafe_allow_html=True)

# トグルボタンを直接HTMLに埋め込む
st.markdown("""
<button class="sidebar-toggle" id="sidebarToggle" onclick="toggleSidebar()">
    <i class="fas fa-bars"></i>
</button>
<script>
    function toggleSidebar() {
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            sidebar.classList.toggle('collapsed');
            const mainContent = document.querySelector('[data-testid="stAppViewContainer"]');
            if (mainContent) {
                if (sidebar.classList.contains('collapsed')) {
                    mainContent.style.marginLeft = '44px';
                    mainContent.style.width = 'calc(100vw - 44px)';
                } else {
                    mainContent.style.marginLeft = '288px';
                    mainContent.style.width = 'calc(100vw - 288px)';
                }
            }
        }
    }
</script>
""", unsafe_allow_html=True)

    # メインコンテンツ
def main():
    # サイドバー
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-content">
            <div class="sidebar-brand">AI Agent</div>
            <nav>
                <ul class="sidebar-nav">
                    <li class="nav-item">
                        <a href="#" class="nav-link">
                            <i class="fas fa-home nav-icon"></i>
                            <span>ホーム</span>
                        </a>
                    </li>
                    <li class="nav-item">
                        <a href="#" class="nav-link">
                            <i class="fas fa-chart-bar nav-icon"></i>
                            <span>分析</span>
                        </a>
                    </li>
                    <li class="nav-item">
                        <a href="#" class="nav-link">
                            <i class="fas fa-users nav-icon"></i>
                            <span>ユーザー管理</span>
                        </a>
                    </li>
                    <li class="nav-item">
                        <a href="#" class="nav-link">
                            <i class="fas fa-cog nav-icon"></i>
                            <span>設定</span>
                        </a>
                    </li>
                    <li class="nav-item">
                        <a href="#" class="nav-link">
                            <i class="fas fa-file-text nav-icon"></i>
                            <span>レポート</span>
                        </a>
                    </li>
                </ul>
            </nav>
        </div>

        <!-- 縮小時のアイコンナビゲーション -->
        <div class="sidebar-icons">
            <a href="#" class="sidebar-icon" title="ホーム"><i class="fas fa-home"></i></a>
            <a href="#" class="sidebar-icon" title="分析"><i class="fas fa-chart-bar"></i></a>
            <a href="#" class="sidebar-icon" title="ユーザー管理"><i class="fas fa-users"></i></a>
            <a href="#" class="sidebar-icon" title="設定"><i class="fas fa-cog"></i></a>
            <a href="#" class="sidebar-icon" title="レポート"><i class="fas fa-file-text"></i></a>
        </div>
        """, unsafe_allow_html=True)

    # メインコンテンツ - layout.htmlの構造を完全に忠実に再現
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
    
    # コンテンツカードを個別に表示
    with st.container():
        st.markdown("""
        <div class="content-card">
            <h3>🤖 自己学習AIエージェントにようこそ</h3>
            <ul class="feature-list">
                <li>💡 質問に答える</li>
                <li>📝 文章を書く</li>
                <li>🔍 情報を調べる</li>
                <li>🧠 学習機能</li>
                <li>🎯 パーソナライズ</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
        <div class="content-card">
            <h3>最新の活動</h3>
            <p>
                システムは正常に稼働しています。
                全ての機能が利用可能な状態です。
                問題が発生した場合は、サポートチームまでお問い合わせください。
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
        <div class="content-card">
            <h3>パフォーマンス</h3>
            <p>
                システムパフォーマンスは良好です。<br>
                レスポンス時間: 0.5秒以下<br>
                稼働率: 99.9%<br>
                アクティブユーザー: 1,247名
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
        <div class="content-card">
            <h3>操作ガイド</h3>
            <p>
                Ctrl + B でサイドバーの開閉ができます。
                下部のチャット欄からAIエージェントと会話できます。
                各機能の詳細については、ヘルプドキュメントをご参照ください。
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
        <div class="content-card">
            <h3>通知とアラート</h3>
            <p>
                現在、重要な通知はありません。
                システムメンテナンスの予定がある場合は、
                事前に通知いたします。
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # チャット入力（Streamlitの機能を使用）
    user_input = st.chat_input("AIエージェントにメッセージを送信...")
    
    if user_input:
        # チャット履歴を初期化
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # ユーザーメッセージを追加
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # チャット履歴を表示
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # AI応答をシミュレート
        with st.chat_message("assistant"):
            response = f"こんにちは！「{user_input}」についてお答えします。"
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()