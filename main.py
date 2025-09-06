#!/usr/bin/env python3
"""
自己学習AIエージェント メインエントリーポイント

使用方法:
    python main.py                    # デフォルトUI起動
    python main.py --ui streamlit     # Streamlit UI起動
    python main.py --ui fastapi       # FastAPI UI起動
    python main.py --test             # テストモード
    python main.py --help             # ヘルプ表示
"""

import argparse
import sys
import os
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="自己学習AIエージェント",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python main.py                    # デフォルトUI起動
  python main.py --ui streamlit     # Streamlit UI起動
  python main.py --ui fastapi       # FastAPI UI起動
  python main.py --test             # テストモード
  python main.py --config custom.yaml # カスタム設定ファイル
        """
    )
    
    parser.add_argument(
        "--ui",
        choices=["streamlit", "fastapi"],
        default="streamlit",
        help="使用するUI (デフォルト: streamlit)"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="テストモードで起動"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="設定ファイルのパス"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        help="ポート番号 (デフォルト: UIに依存)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="ホストアドレス (デフォルト: localhost)"
    )
    
    args = parser.parse_args()
    
    if args.test:
        run_test_mode()
    else:
        run_ui_mode(args)

def run_test_mode():
    """テストモード実行"""
    print("🧪 テストモードを開始します...")
    
    try:
        # 基本インポートテスト
        print("📦 モジュールインポートテスト...")
        from src.advanced_agent.config.settings import AgentConfig
        from src.advanced_agent.core.self_learning_agent import SelfLearningAgent
        print("✅ 基本モジュールインポート成功")
        
        # 設定テスト
        print("⚙️ 設定読み込みテスト...")
        config = AgentConfig()
        print(f"✅ 設定読み込み成功 (エージェント名: {config.name})")
        
        # 環境検証テスト
        print("🔍 環境検証テスト...")
        from src.advanced_agent.core.environment import quick_environment_check
        env_ok = quick_environment_check()
        if env_ok:
            print("✅ 環境検証成功")
        else:
            print("⚠️ 環境検証で警告が検出されました")
        
        # Ollama接続テスト
        print("🦙 Ollama接続テスト...")
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                print(f"✅ Ollama接続成功 (モデル数: {len(models)})")
            else:
                print("⚠️ Ollama接続失敗")
        except Exception as e:
            print(f"⚠️ Ollama接続失敗: {e}")
        
        # エージェント初期化テスト
        print("🤖 エージェント初期化テスト...")
        import asyncio
        async def test_agent():
            try:
                agent = SelfLearningAgent(db_path="data/test_agent.db")
                await agent.initialize_session()
                print("✅ エージェント初期化成功")
                await agent.close()
            except Exception as e:
                print(f"⚠️ エージェント初期化失敗: {e}")
        
        asyncio.run(test_agent())
        
        print("\n🎉 テスト完了！")
        
    except Exception as e:
        print(f"❌ テスト失敗: {e}")
        sys.exit(1)

def run_ui_mode(args):
    """UIモード実行"""
    print(f"🚀 自己学習AIエージェントを起動します...")
    print(f"UI: {args.ui}")
    
    if args.ui == "streamlit":
        run_streamlit_ui(args)
    elif args.ui == "fastapi":
        run_fastapi_ui(args)

def run_streamlit_ui(args):
    """Streamlit UI起動"""
    try:
        import subprocess
        
        # ポート設定
        port = args.port or 8501
        
        print(f"🌐 新しいAIエージェントWebUIを起動中...")
        print(f"URL: http://{args.host}:{port}")
        print("シンプルで美しいUIで起動します")
        
        # 新しいWebUIを起動
        cmd = [sys.executable, "-m", "streamlit", "run", "webui.py", "--server.port", str(port), "--server.headless", "true"]
        subprocess.run(cmd)
        
    except Exception as e:
        print(f"❌ WebUI起動失敗: {e}")
        sys.exit(1)

def run_fastapi_ui(args):
    """FastAPI UI起動"""
    try:
        import subprocess
        
        # ポート設定
        port = args.port or 8000
        
        print(f"🌐 新しいAIエージェントWebUIを起動中...")
        print(f"URL: http://{args.host}:{port}")
        print("シンプルで美しいUIで起動します")
        
        # 新しいWebUIを起動（Streamlitを使用）
        cmd = [sys.executable, "-m", "streamlit", "run", "webui.py", "--server.port", str(port), "--server.headless", "true"]
        subprocess.run(cmd)
        
    except Exception as e:
        print(f"❌ WebUI起動失敗: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
