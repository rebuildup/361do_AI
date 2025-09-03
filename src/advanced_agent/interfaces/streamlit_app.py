"""
Streamlit アプリケーション起動スクリプト

streamlit run コマンドで実行するためのエントリーポイント
"""

import sys
import os

# パスを追加してモジュールをインポート可能にする
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.advanced_agent.interfaces.streamlit_ui import main

if __name__ == "__main__":
    main()