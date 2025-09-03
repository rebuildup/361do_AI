"""
Streamlit アプリケーション起動スクリプト

streamlit run コマンドで実行するためのエントリーポイント
"""

import sys
import os
import streamlit as st

# パスを追加してモジュールをインポート可能にする
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, project_root)

try:
    from src.advanced_agent.interfaces.streamlit_ui import main
    main()
except ImportError as e:
    # フォールバック: シンプルなテストUI
    st.set_page_config(
        page_title="Advanced AI Agent",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Advanced AI Agent")
    st.error(f"モジュールのインポートに失敗しました: {e}")
    st.info("依存関係をインストールしてください: `pip install -r requirements.txt`")
    
    st.markdown("### 📋 必要な依存関係")
    st.code("""
    pip install streamlit
    pip install langchain
    pip install chromadb
    pip install transformers
    pip install ollama
    """)
    
    st.markdown("### 🔧 トラブルシューティング")
    st.markdown("""
    1. 仮想環境がアクティベートされているか確認
    2. 必要な依存関係がインストールされているか確認
    3. Ollamaがインストールされ、モデルがダウンロードされているか確認
    """)

if __name__ == "__main__":
    pass