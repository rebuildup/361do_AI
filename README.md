# 自己学習 AI エージェント 361do_AI (Self-Learning AI Agent)

RTX 4050 6GB VRAM 環境で動作する高性能自己学習 AI エージェントです。React UI + FastAPI バックエンドの統合アーキテクチャを採用しています。

## 🚀 クイックスタート

### 1. 依存関係のインストール

```bash
# 仮想環境の作成とアクティベート
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Python依存関係のインストール
pip install -r requirements.txt

# フロントエンド依存関係のインストール
cd frontend
yarn install
cd ..
```

### 2. Ollama のセットアップ

```bash
# Ollamaのインストール（未インストールの場合）
# https://ollama.ai/ からダウンロード

# モデルのダウンロード
ollama pull qwen2:7b-instruct
```

### 3. アプリケーションの起動

#### 簡単起動（推奨）

```bash
# Linux/Mac
./start.sh

# Windows PowerShell
.\start.ps1
```

#### 手動起動

```bash
# 統合UI（React + FastAPI）で起動
python main.py --ui react

# レガシーStreamlit UIで起動
python main.py --ui streamlit

# テストモード
python main.py --test

# ヘルプ表示
python main.py --help
```

#### Docker Compose 起動

```bash
# 統合UI（React + FastAPI）で起動
docker-compose up ai-agent

# レガシーStreamlit UIで起動
docker-compose --profile legacy up ai-agent-legacy

# 全サービス起動（監視・ログ含む）
docker-compose up
```

## 📁 プロジェクト構造

```
self-learning-ai-agent/
├── main.py                          # メインエントリーポイント
├── start.sh                         # Linux/Mac起動スクリプト
├── start.ps1                        # Windows起動スクリプト
├── requirements.txt                 # Python依存関係
├── Dockerfile.integrated            # 統合Dockerfile（React + FastAPI）
├── Dockerfile                       # レガシーDockerfile（Streamlit）
├── docker-compose.yml               # Docker Compose設定
├── frontend/                        # React フロントエンド
│   ├── package.json                 # フロントエンド依存関係
│   ├── src/                         # React ソースコード
│   └── dist/                        # ビルド済みフロントエンド
├── config/
│   └── agent_config.yaml           # エージェント設定
├── src/
│   └── advanced_agent/
│       ├── config/                 # 設定管理
│       ├── database/               # データベース層
│       ├── reasoning/              # 推論エンジン
│       ├── memory/                 # 記憶システム
│       ├── interfaces/             # UI層（FastAPI）
│       ├── monitoring/             # 監視システム
│       └── ...
├── data/                           # データストレージ
├── tests/                          # テスト
└── docs/                           # ドキュメント
```

## 🎯 主要機能

### ✅ 実装済み機能

- **永続セッション管理**: SQLAlchemy + LangChain 統合
- **データベーススキーマ**: 自己学習 AI エージェント用の包括的データモデル
- **設定管理システム**: Pydantic 設定と YAML 設定ファイル
- **推論エンジン**: Ollama 統合、Chain-of-Thought 推論
- **品質評価システム**: 8 次元の包括的な品質評価
- **プロンプトテンプレート管理**: 7 種類のテンプレート
- **統合 Web UI**: React フロントエンド + FastAPI バックエンド
- **レガシー UI**: Streamlit ベースのインターフェース（後方互換性）
- **自然言語対話**: 完全に動作するチャット機能
- **レスポンシブデザイン**: モバイル・デスクトップ対応

### 🔄 推論プロセス

1. **質問理解**: ユーザーの質問を解析
2. **知識検索**: 関連する知識を検索
3. **段階的推論**: Chain-of-Thought 推論を実行
4. **品質評価**: 8 次元での品質評価
5. **回答生成**: 最適化された回答を生成

### 📊 品質評価指標

- **正確性 (Accuracy)**: 回答の正確性
- **完全性 (Completeness)**: 回答の完全性
- **明確性 (Clarity)**: 回答の明確性
- **論理的一貫性 (Logical Consistency)**: 論理的な一貫性
- **有用性 (Usefulness)**: 回答の有用性
- **効率性 (Efficiency)**: 処理効率
- **創造性 (Creativity)**: 創造性
- **安全性 (Safety)**: 安全性

## 🛠️ 使用方法

### コマンドラインオプション

```bash
python main.py [オプション]

オプション:
  --ui {react,streamlit,fastapi}  UIの選択 (デフォルト: react)
  --test                          テストモードで起動
  --config CONFIG                 設定ファイルのパス
  --port PORT                     ポート番号
  --host HOST                     ホストアドレス (デフォルト: 0.0.0.0)
  --help                          ヘルプ表示
```

### 使用例

```bash
# 統合UI（React + FastAPI）で起動
python main.py --ui react

# レガシーStreamlit UIで起動
python main.py --ui streamlit

# FastAPI UIのみで起動
python main.py --ui fastapi --port 8000

# テストモード
python main.py --test

# カスタム設定で起動
python main.py --config config/custom_config.yaml
```

## 🔧 設定

### 設定ファイル (config/agent_config.yaml)

```yaml
# エージェント基本設定
name: "SelfLearningAgent"
version: "1.0.0"

# Ollama設定
ollama:
  base_url: "http://localhost:11434"
  model: "qwen2:7b-instruct"
  temperature: 0.7

# データベース設定
database:
  db_path: "data/self_learning_agent.db"
  echo: false

# 学習設定
learning:
  enable_self_learning: true
  learning_rate: 0.01
  batch_size: 32
```

## 🧪 テスト

```bash
# 全テスト実行
python -m pytest tests/

# 特定のテスト実行
python -m pytest tests/unit/test_reasoning.py

# カバレッジ付きテスト
python -m pytest --cov=src tests/
```

## 📈 パフォーマンス

- **推論速度**: 1-3 秒/質問
- **メモリ使用量**: 6GB VRAM 以下
- **品質スコア**: 0.6-0.8（8 次元評価）
- **同時接続**: 複数セッション対応
- **フロントエンド**: React + Vite による高速ビルド
- **バックエンド**: FastAPI による高パフォーマンス API

## 🐛 トラブルシューティング

### よくある問題

1. **Ollama 接続エラー**

   ```bash
   # Ollamaが起動しているか確認
   ollama list

   # モデルがダウンロードされているか確認
   ollama pull qwen2:7b-instruct
   ```

2. **依存関係エラー**

   ```bash
   # 仮想環境を再作成
   rm -rf .venv
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **データベースエラー**
   ```bash
   # データベースファイルを削除して再作成
   rm data/*.db
   python main.py --test
   ```

## 📚 ドキュメント

- [アーキテクチャ](docs/ARCHITECTURE.md)
- [API リファレンス](docs/API_REFERENCE.md)
- [設定ガイド](docs/CONFIGURATION.md)
- [インストールガイド](docs/INSTALLATION.md)
- [使用方法](docs/USAGE.md)
- [トラブルシューティング](docs/TROUBLESHOOTING.md)

## 🤝 貢献

1. リポジトリをフォーク
2. フィーチャーブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 🙏 謝辞

- [Ollama](https://ollama.ai/) - ローカル LLM 実行環境
- [LangChain](https://langchain.com/) - LLM アプリケーションフレームワーク
- [React](https://react.dev/) - フロントエンドフレームワーク
- [FastAPI](https://fastapi.tiangolo.com/) - 高パフォーマンス API フレームワーク
- [Streamlit](https://streamlit.io/) - レガシー Web アプリケーションフレームワーク
- [SQLAlchemy](https://www.sqlalchemy.org/) - ORM
- [ChromaDB](https://www.trychroma.com/) - ベクトルデータベース
