# Advanced Self-Learning AI Agent

## 🚀 概要

RTX 4050 6GB VRAM 環境で動作する高性能自己学習 AI エージェントシステムです。**オープンソースライブラリを最大限活用**し、LangChain、AutoGen、HuggingFace、ChromaDB などの成熟したフレームワークを統合して、オリジナルコードを最小限に抑えた安定性の高いシステムを実現します。

## ✨ 主要機能

- **🧠 LangChain + Ollama 推論エンジン**: ReAct Agent による Chain-of-Thought 推論
- **🧬 AutoGen 進化的学習システム**: マルチエージェント協調による継続的改善
- **💾 LangChain + ChromaDB 記憶システム**: 永続的記憶と自動コンテキスト継続
- **⚡ HuggingFace 最適化**: Accelerate + BitsAndBytes による効率的メモリ管理
- **🔄 Prometheus + Grafana 監視**: リアルタイム性能監視と自動最適化
- **🌐 FastAPI + Streamlit UI**: 高応答性インターフェースと可視化

## 📁 プロジェクト構造

```
.
├── src/
│   └── advanced_agent/          # メインエージェントシステム
│       ├── core/                # LangChain + Ollama 統合
│       ├── memory/              # ChromaDB + SQLAlchemy 記憶システム
│       ├── learning/            # AutoGen + PEFT 進化学習
│       ├── monitoring/          # Prometheus + PSUtil 監視
│       └── interfaces/          # FastAPI + Streamlit + Typer
├── config/                      # 設定ファイル
│   ├── system.yaml             # システム設定
│   ├── advanced_agent.yaml     # エージェント設定
│   ├── .env                    # 環境変数
│   └── gpu_config.env          # GPU 最適化設定
├── data/                       # ChromaDB + SQLite データベース
├── logs/                       # Loguru ログファイル
├── docs/                       # MkDocs ドキュメント
└── tests/                      # Pytest テストスイート
    ├── unit/                   # 単体テスト
    ├── integration/            # 統合テスト
    └── performance/            # 性能テスト
```

## 🚀 クイックスタート

### 1. 環境準備

```bash
# 1. リポジトリのクローン
git clone <repository-url>
cd advanced-self-learning-agent

# 2. Python仮想環境の作成
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 3. 依存関係のインストール
pip install -r requirements.txt

# 4. Ollama のセットアップ
# https://ollama.ai/ からダウンロード・インストール
ollama pull deepseek-r1:7b
ollama pull qwen2.5:7b-instruct-q4_k_m
ollama pull qwen2:1.5b-instruct-q4_k_m
```

### 2. システム起動

```bash
# Web UI での使用（推奨）
streamlit run src/advanced_agent/interfaces/streamlit_app.py
# ブラウザで http://localhost:8501 にアクセス

# API サーバーでの使用
python -m src.advanced_agent.interfaces.fastapi_gateway
# API ドキュメント: http://localhost:8000/docs

# CLI での基本テスト
python -m src.advanced_agent.reasoning.demo
```

### 3. 基本的な使用例

```python
# Python クライアントでの使用
import asyncio
from src.advanced_agent.reasoning.basic_engine import BasicReasoningEngine

async def chat_example():
    engine = BasicReasoningEngine()

    # 基本的な推論
    response = await engine.reason(
        "Pythonでクイックソートを実装してください"
    )

    print(f"応答: {response.content}")
    print(f"推論ステップ: {response.reasoning_steps}")

# 実行
asyncio.run(chat_example())
```

## ⚙️ システム要件

- **GPU**: NVIDIA RTX 4050 (6GB VRAM) 以上
- **RAM**: 32GB 推奨
- **CPU**: Intel i7-13700H 相当以上
- **Python**: 3.11+
- **CUDA**: 12.0+
- **Ollama**: 最新版

## 🔧 オープンソース技術スタック

- **🤖 AI フレームワーク**: LangChain, AutoGen, HuggingFace Transformers
- **💾 データベース**: ChromaDB, SQLAlchemy, SQLite
- **🌐 Web フレームワーク**: FastAPI, Streamlit, Typer
- **📊 監視**: Prometheus, Grafana, PSUtil, NVIDIA-ML
- **🧪 テスト**: Pytest, HuggingFace Evaluate
- **📝 ドキュメント**: MkDocs, MkDocs Material

## 🔧 設定

### HuggingFace Accelerate + BitsAndBytes 最適化

```yaml
# config/system.yaml
gpu:
  max_vram_gb: 5.0
  quantization_levels: [8, 4, 3]
  temperature_threshold: 80
```

### LangChain + ChromaDB 記憶システム

```yaml
# config/system.yaml
persistent_memory:
  db_path: "data/chroma_db"
  max_short_term_items: 1000
  max_long_term_items: 10000
  importance_threshold: 0.7
```

## 📊 監視とメトリクス

- **GPU 使用率**: リアルタイム監視
- **VRAM 使用量**: 自動最適化
- **推論速度**: 2 秒以内の応答目標
- **記憶効率**: 重要度ベースの自動整理

## 🧪 テスト

```bash
# Pytest 単体テスト
python -m pytest tests/unit/

# LangChain 統合テスト
python -m pytest tests/integration/

# HuggingFace Evaluate 性能テスト
python -m pytest tests/performance/

# 全テスト実行
python -m pytest tests/ --cov=src/advanced_agent
```

## 📚 ドキュメント

### 基本ガイド

- [📦 インストールガイド](docs/INSTALLATION.md) - システムのセットアップ手順
- [⚙️ 設定ガイド](docs/CONFIGURATION.md) - 設定ファイルの詳細説明
- [🚀 使用方法ガイド](docs/USAGE.md) - 基本的な使用方法とベストプラクティス

### 技術リファレンス

- [🔌 API リファレンス](docs/API_REFERENCE.md) - REST API の詳細仕様
- [🏗️ アーキテクチャ](docs/ARCHITECTURE.md) - システム設計の詳細
- [🔧 トラブルシューティング](docs/TROUBLESHOOTING.md) - 問題解決ガイド

## 🤝 貢献

プルリクエストやイシューの報告を歓迎します。詳細は [CONTRIBUTING.md](CONTRIBUTING.md) をご覧ください。

## 📄 ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。

## 🌟 オープンソース統合の利点

- **⚡ 開発効率**: 成熟したライブラリの活用により開発時間を大幅短縮
- **🛡️ 安定性**: 実績のあるオープンソースプロジェクトによる高い信頼性
- **🤝 コミュニティサポート**: 豊富なドキュメントとコミュニティサポート
- **🔧 拡張性**: 標準的なインターフェースによる容易な機能拡張
- **💰 保守性**: オリジナルコード最小化による保守コスト削減

---

**🚀 オープンソースの力で RTX 4050 最適化 AI エージェントを体験しよう！**
