# プロジェクト構造

## 📁 ディレクトリ構造

```
self-learning-ai-agent/
├── main.py                          # メインエントリーポイント
├── requirements.txt                 # Python依存関係
├── constraints.txt                  # 依存関係制約
├── README.md                        # プロジェクト概要
├── PROJECT_STRUCTURE.md             # このファイル
│
├── config/                          # 設定ファイル
│   └── agent_config.yaml           # エージェント設定
│
├── src/                             # ソースコード
│   └── advanced_agent/
│       ├── __init__.py
│       ├── config/                  # 設定管理
│       │   ├── __init__.py
│       │   ├── settings.py          # Pydantic設定クラス
│       │   └── loader.py            # 設定読み込み
│       │
│       ├── database/                # データベース層
│       │   ├── __init__.py
│       │   ├── models.py            # SQLAlchemyモデル
│       │   ├── connection.py        # データベース接続
│       │   └── migrations.py        # マイグレーション
│       │
│       ├── reasoning/               # 推論エンジン
│       │   ├── __init__.py
│       │   ├── ollama_client.py     # Ollamaクライアント
│       │   ├── basic_engine.py      # 基本推論エンジン
│       │   ├── cot_engine.py        # Chain-of-Thought推論
│       │   ├── prompt_templates.py  # プロンプトテンプレート
│       │   ├── quality_evaluator.py # 品質評価システム
│       │   └── metrics.py           # 評価指標
│       │
│       ├── memory/                  # 記憶システム
│       │   ├── __init__.py
│       │   ├── persistent_memory.py # 永続記憶
│       │   ├── session_manager.py   # セッション管理
│       │   └── conversation_manager.py # 会話管理
│       │
│       ├── interfaces/              # UI層
│       │   ├── __init__.py
│       │   ├── streamlit_app.py     # Streamlitアプリ
│       │   ├── streamlit_ui.py      # Streamlit UI
│       │   ├── demo_fastapi.py      # FastAPIデモ
│       │   └── ...
│       │
│       ├── monitoring/              # 監視システム
│       │   ├── __init__.py
│       │   ├── metrics.py           # メトリクス収集
│       │   ├── logger.py            # ログ管理
│       │   └── ...
│       │
│       ├── core/                    # コア機能
│       │   ├── __init__.py
│       │   ├── agent.py             # エージェント本体
│       │   ├── logger.py            # ログシステム
│       │   └── ...
│       │
│       ├── learning/                # 学習システム
│       │   ├── __init__.py
│       │   └── self_learning.py     # 自己学習
│       │
│       ├── evolution/               # 進化システム
│       │   ├── __init__.py
│       │   ├── evolution.py         # 進化アルゴリズム
│       │   └── ...
│       │
│       ├── adaptation/              # 適応システム
│       │   ├── __init__.py
│       │   └── adaptation.py        # 適応機能
│       │
│       ├── optimization/            # 最適化
│       │   ├── __init__.py
│       │   └── optimizer.py         # 最適化器
│       │
│       ├── quantization/            # 量子化
│       │   ├── __init__.py
│       │   └── quantizer.py         # 量子化器
│       │
│       ├── multimodal/              # マルチモーダル
│       │   ├── __init__.py
│       │   ├── vision.py            # 画像処理
│       │   └── ...
│       │
│       └── inference/               # 推論
│           ├── __init__.py
│           ├── inference.py         # 推論処理
│           └── ...
│
├── data/                            # データストレージ
│   ├── agent_memory.db              # エージェント記憶DB
│   ├── agent_improvement.db         # 改善データDB
│   ├── chroma_db/                   # ChromaDB
│   ├── knowledge_base/              # 知識ベース
│   ├── learning_data/               # 学習データ
│   └── logs/                        # ログファイル
│
├── tests/                           # テスト
│   ├── __init__.py
│   ├── unit/                        # 単体テスト
│   │   ├── test_database.py
│   │   ├── test_reasoning.py
│   │   ├── test_quality_evaluation.py
│   │   └── ...
│   ├── integration/                 # 統合テスト
│   │   ├── test_end_to_end.py
│   │   └── test_monitoring_integration.py
│   ├── performance/                 # パフォーマンステスト
│   │   └── test_benchmarks.py
│   └── deployment/                  # デプロイメントテスト
│       └── test_production_readiness.py
│
├── docs/                            # ドキュメント
│   ├── README.md
│   ├── ARCHITECTURE.md
│   ├── API_REFERENCE.md
│   ├── CONFIGURATION.md
│   ├── INSTALLATION.md
│   ├── USAGE.md
│   ├── TROUBLESHOOTING.md
│   ├── SELF_LEARNING_AGENT_TASKS.md
│   ├── api/                         # API仕様書
│   ├── developer_guide/             # 開発者ガイド
│   └── user_guide/                  # ユーザーガイド
│
├── docker/                          # Docker設定
│   ├── agent/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── nginx/
│   │   └── nginx.conf
│   └── ollama/
│       ├── Dockerfile
│       └── pull_model.sh
│
├── static/                          # 静的ファイル
│   └── index.html
│
├── documents/                       # プロジェクト文書
│   └── report_llm.md
│
└── logs/                            # ログファイル
    └── (ログファイルは実行時に生成)
```

## 🏗️ アーキテクチャ層

### 1. プレゼンテーション層 (Presentation Layer)

- **場所**: `src/advanced_agent/interfaces/`
- **責任**: ユーザーインターフェース、API エンドポイント
- **技術**: Streamlit, FastAPI

### 2. アプリケーション層 (Application Layer)

- **場所**: `src/advanced_agent/core/`
- **責任**: ビジネスロジック、ワークフロー制御
- **技術**: Python, asyncio

### 3. ドメイン層 (Domain Layer)

- **場所**: `src/advanced_agent/reasoning/`, `src/advanced_agent/memory/`
- **責任**: 推論エンジン、記憶システム、学習機能
- **技術**: LangChain, ChromaDB

### 4. インフラストラクチャ層 (Infrastructure Layer)

- **場所**: `src/advanced_agent/database/`, `src/advanced_agent/config/`
- **責任**: データ永続化、設定管理、外部サービス連携
- **技術**: SQLAlchemy, Pydantic, Ollama

## 🔄 データフロー

```
ユーザー入力 → UI層 → アプリケーション層 → ドメイン層 → インフラ層
     ↑                                                      ↓
     ← レスポンス ← UI層 ← アプリケーション層 ← ドメイン層 ←
```

## 📦 モジュール依存関係

```
interfaces/ → core/ → reasoning/ → database/
     ↓           ↓        ↓           ↓
  streamlit   agent   ollama    sqlalchemy
  fastapi     logger  langchain  chromadb
```

## 🎯 設計原則

1. **関心の分離**: 各層は明確な責任を持つ
2. **依存性逆転**: 上位層は下位層の抽象に依存
3. **単一責任**: 各クラスは一つの責任を持つ
4. **開放閉鎖**: 拡張に開放、修正に閉鎖
5. **テスタビリティ**: 各層は独立してテスト可能

## 🔧 設定管理

- **環境別設定**: 開発、テスト、本番環境
- **設定の優先順位**: 環境変数 > 設定ファイル > デフォルト値
- **設定検証**: Pydantic による型安全性とバリデーション

## 📊 監視とログ

- **構造化ログ**: JSON 形式でのログ出力
- **メトリクス収集**: パフォーマンス指標の収集
- **エラー追跡**: 詳細なエラー情報とスタックトレース
- **ヘルスチェック**: システム状態の監視

## 🚀 デプロイメント

- **Docker 化**: コンテナベースのデプロイメント
- **環境分離**: 開発、ステージング、本番環境
- **スケーラビリティ**: 水平スケーリング対応
- **監視**: Prometheus + Grafana による監視
