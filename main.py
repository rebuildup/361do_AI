#!/usr/bin/env python3
"""
自己学習AIエージェント 361do_AI メインエントリーポイント

使用方法:
    python main.py                    # デフォルトUI起動（React）
    python main.py --ui react         # React UI起動
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
  python main.py                    # デフォルトUI起動（React）
  python main.py --ui react         # React UI起動
  python main.py --ui streamlit     # Streamlit UI起動
  python main.py --ui fastapi       # FastAPI UI起動
  python main.py --test             # テストモード
  python main.py --config custom.yaml # カスタム設定ファイル
        """
    )
    
    parser.add_argument(
        "--ui",
        choices=["react", "streamlit", "fastapi"],
        default="react",
        help="使用するUI (デフォルト: react)"
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
        default="0.0.0.0",
        help="ホストアドレス (デフォルト: 0.0.0.0)"
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
    
    # 設定ファイルの読み込み
    if args.config:
        print(f"📁 設定ファイル: {args.config}")
        if not os.path.exists(args.config):
            print(f"❌ 設定ファイルが見つかりません: {args.config}")
            sys.exit(1)
    
    try:
        if args.ui == "react":
            run_react_ui(args)
        elif args.ui == "streamlit":
            run_streamlit_ui(args)
        elif args.ui == "fastapi":
            run_fastapi_ui(args)
    except KeyboardInterrupt:
        print("\n🛑 ユーザーによって停止されました")
        sys.exit(0)
    except Exception as e:
        print(f"❌ UI起動中にエラーが発生しました: {e}")
        sys.exit(1)

def run_react_ui(args):
    """React UI起動（統合版）"""
    try:
        import subprocess
        import uvicorn
        from src.advanced_agent.interfaces.fastapi_app import create_app
        
        # ポート設定
        port = args.port or 80
        
        print(f"🌐 React AIエージェントWebUIを起動中...")
        print(f"URL: http://{args.host}:{port}")
        print("React + FastAPI統合UIで起動します")
        
        # FastAPIアプリケーションを作成して起動
        app = create_app()
        uvicorn.run(
            app,
            host=args.host,
            port=port,
            log_level="info"
        )
        
    except ImportError:
        print("❌ React UI起動失敗: uvicornがインストールされていません")
        print("pip install uvicorn でインストールしてください")
        sys.exit(1)
    except Exception as e:
        print(f"❌ React UI起動失敗: {e}")
        sys.exit(1)

def run_streamlit_ui(args):
    """Streamlit UI起動"""
    try:
        import subprocess
        
        # ポート設定
        port = args.port or 8501
        
        print(f"🌐 Streamlit AIエージェントWebUIを起動中...")
        print(f"URL: http://{args.host}:{port}")
        print("シンプルで美しいUIで起動します")
        
        # Streamlitを起動（ホストとポートを指定）
        cmd = [
            sys.executable, "-m", "streamlit", "run", "src/advanced_agent/interfaces/streamlit_ui.py",
            "--server.port", str(port),
            "--server.address", args.host,
            "--server.headless", "true"
        ]
        subprocess.run(cmd)
        
    except Exception as e:
        print(f"❌ Streamlit UI起動失敗: {e}")
        sys.exit(1)

def run_fastapi_ui(args):
    """FastAPI UI起動"""
    try:
        import uvicorn
        from src.advanced_agent.interfaces.fastapi_app import create_app
        
        # ポート設定
        port = args.port or 8000
        
        print(f"🌐 FastAPI AIエージェントWebUIを起動中...")
        print(f"URL: http://{args.host}:{port}")
        print("FastAPIベースのUIで起動します")
        
        # FastAPIアプリケーションを作成して起動
        app = create_app()
        uvicorn.run(
            app,
            host=args.host,
            port=port,
            log_level="info"
        )
        
    except ImportError:
        print("❌ FastAPI UI起動失敗: uvicornがインストールされていません")
        print("pip install uvicorn でインストールしてください")
        sys.exit(1)
    except Exception as e:
        print(f"❌ FastAPI UI起動失敗: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
