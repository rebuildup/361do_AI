# マルチエージェント学習システム

4 つの専門エージェントが同時に会話し、相互学習を行う AI システムです。8 時間のタイムリミットで継続的に実行され、エージェント間の知識共有と学習を促進します。

## 🤖 エージェント構成

- **リサーチャー**: 情報収集と探求に特化
- **アナライザー**: 論理的分析と構造化に特化
- **クリエイター**: 創造的思考とアイデア生成に特化
- **オプティマイザー**: 効率化と品質向上に特化

## 🎯 主要機能

- **自動会話システム**: エージェント間の建設的な対話
- **相互学習機能**: 会話内容から自動的に学習データを生成
- **時間制限実行**: 8 時間で自動停止（カスタマイズ可能）
- **リアルタイム進捗**: 実行中の統計をリアルタイム表示
- **安全停止**: Ctrl+C で安全に停止可能

## 📁 プロジェクト構成

```
├── multi_agent_learning_system.py    # メインシステム
├── multi_agent_web_learning_system.py # Web拡張版
├── agent_cli.py                       # CLI インターフェース
├── config/                            # 設定ファイル
│   ├── .env                          # 環境変数
│   ├── gpu_config.env                # GPU設定
│   └── pytest.ini                    # テスト設定
├── docs/                              # ドキュメント
│   ├── MULTI_AGENT_LEARNING_GUIDE.md # 使用ガイド
│   ├── CLI_USAGE_GUIDE.md            # CLI ガイド
│   └── GPU_SETUP_GUIDE.md            # GPU セットアップ
├── examples/                          # 実行例とテスト
│   ├── test_multi_agent_system.py    # システムテスト
│   ├── run_multi_agent_learning.bat  # Windows実行用
│   └── run_multi_agent_test.bat      # テスト実行用
├── src/                               # ソースコード
├── tests/                             # テストスイート
└── workspace/                         # 作業領域
```

## 🚀 使用方法

### 基本実行

```bash
# 8時間の制限時間で実行
python multi_agent_learning_system.py

# カスタム時間制限（4時間）
python multi_agent_learning_system.py --hours 4.0

# テストモード（6分間）
python multi_agent_learning_system.py --test-mode
```

### Windows バッチファイル

```bash
# メインシステム実行
examples\run_multi_agent_learning.bat

# テスト実行
examples\run_multi_agent_test.bat
```

### システムテスト

```bash
# 全テスト実行
python examples/test_multi_agent_system.py

# 個別テスト実行
python examples/test_multi_agent_system.py --test init
```

## 📦 インストール

### 前提条件

- Python 3.8 以上
- Ollama (ローカル LLM 実行環境)

### セットアップ

```bash
# リポジトリをクローン
git clone <repository-url>
cd 008_LLM

# 仮想環境を作成・有効化
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 依存関係をインストール
pip install -r requirements.txt

# Ollamaをインストール・起動
# https://ollama.ai/ からインストール
ollama serve

# 推奨モデルをダウンロード
ollama pull qwen2:7b-instruct
```

### クイックスタート

```bash
# エージェントを起動
python agent_cli.py

# 基本的な使用方法
> chat こんにちは
> chat "Pythonでファイルを読み込む方法を教えて"
> status
> quit
```

## 🎯 使用方法

### CLI エージェントの起動

```bash
python agent_cli.py
```

### 基本コマンド

#### チャット機能

```bash
chat こんにちは                    # エージェントとチャット
chat "コードを書いて"              # コード生成リクエスト
status                           # システム状態確認
help                             # ヘルプ表示
quit                             # 終了
```

#### 学習システム

```bash
learn start                      # 学習システム開始
learn stop                       # 学習システム停止
learn status                     # 学習システム状態確認
learn cycle                      # 手動で学習サイクル実行
```

#### 学習データ管理

```bash
data add programming "Pythonは動的型付け言語です"
data list            # 学習データ一覧表示
data update <ID> <内容>  # 学習データ更新
data delete <ID>     # 学習データ削除
data stats           # 学習統計表示
data export json     # 学習データエクスポート
data import data.json # 学習データインポート
```

#### プロンプト管理

```bash
prompt list          # プロンプト一覧表示
prompt add greeting "こんにちは！何かお手伝いできることはありますか？"
prompt update greeting "新しい挨拶メッセージ"
prompt delete greeting # プロンプト削除
prompt optimize greeting # プロンプト最適化
prompt export        # プロンプトエクスポート
prompt import prompts.json # プロンプトインポート
```

#### テスト機能

```bash
test conversation    # 会話テストモード
test learning        # 学習機能テスト
```

#### システム情報

```bash
status               # システム状態確認
stats                # 統計情報表示
report               # 詳細レポート生成
```

## 🧪 テスト実行

### テスト実行

```bash
# 単体テスト
python -m pytest tests/ -v

# 統合テスト（OLLAMA サーバーが必要）
export INTEGRATION_TEST=true
python -m pytest tests/test_codex_integration.py -v

# 学習システムのテスト
python test_self_learning_simple.py
```

テストでは以下の機能を検証します：

- エージェントの基本機能
- 学習データの管理
- プロンプト最適化
- データベース操作
- OLLAMA との統合

## 📁 プロジェクト構造

```
AI-Agent-System/
├── agent_cli.py                 # CLIエージェント
├── test_self_learning_simple.py # 学習システムテスト
├── requirements.txt             # 依存関係
├── pytest.ini                  # pytest設定
├── src/
│   ├── agent/                   # 学習システム
│   │   ├── core/
│   │   │   ├── config.py        # 設定管理
│   │   │   ├── database.py      # データベース管理
│   │   │   ├── ollama_client.py # Ollamaクライアント
│   │   │   └── agent_manager.py # エージェント管理
│   │   ├── tools/
│   │   │   ├── learning_tool.py # 学習ツール
│   │   │   └── file_tool.py     # ファイルツール
│   │   └── self_tuning/
│   │       └── advanced_learning.py # 高度な学習システム
│   └── codex_agent/             # Codex互換エージェント
│       ├── config.py            # 設定
│       ├── agent_interface.py   # エージェントインターフェース
│       ├── compatibility_layer.py # 互換性レイヤー
│       ├── ollama_client.py     # OLLAMAクライアント
│       ├── performance_monitor.py # パフォーマンス監視
│       └── errors.py            # エラーハンドリング
├── tests/                       # テストファイル
├── data/                        # データファイル
│   ├── learning_data/           # 学習データ
│   ├── prompts/                 # プロンプトファイル
│   └── logs/                    # ログファイル
└── tools/                       # ユーティリティ
```

## 🔧 設定

### 設定ファイル

`src/agent/core/config.py` で以下の設定を変更できます：

- **Ollama 設定**: モデル名、API エンドポイント
- **学習設定**: 学習間隔、品質閾値
- **データベース設定**: SQLite ファイルパス
- **機能有効化**: Web 検索、Web デザイン生成

### 環境変数

```bash
# OLLAMA設定
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=qwen2:7b-instruct
export OLLAMA_TIMEOUT=30

# GPU設定（高速化）
export OLLAMA_GPU_ENABLED=true
export OLLAMA_GPU_MEMORY_FRACTION=0.8
export OLLAMA_GPU_LAYERS=32
export OLLAMA_PARALLEL_REQUESTS=4

# データベース設定
export DATABASE_URL=sqlite:///data/agent.db

# 学習システム設定
export AGENT_LEARNING_ENABLED=true
export AGENT_LEARNING_INTERVAL_MINUTES=30
```

## 📊 学習システムの仕組み

### 1. 学習データ分析

- 会話データから有用な情報を抽出
- 品質スコアに基づいてデータを評価
- 低品質なデータを自動改善

### 2. プロンプト最適化

- LLM を使用してプロンプトを分析
- 明確性、効果性、出力予測可能性を評価
- 改善されたプロンプトを自動適用

### 3. 知識抽出

- 高品質な会話から知識を抽出
- 重複する知識を統合
- 信頼度の低い知識を削除

### 4. パフォーマンス分析

- 応答時間、品質スコア、ユーザーフィードバックを分析
- 学習パラメータを適応的に調整
- 詳細なレポートを生成

## 🎨 カスタマイズ

### 新しいプロンプトテンプレートの追加

```python
await learning_tool.add_prompt_template(
    name="custom_prompt",
    content="カスタムプロンプト内容",
    description="プロンプトの説明",
    category="custom"
)
```

### 学習データの追加

```python
await learning_tool.add_custom_learning_data(
    content="学習内容",
    category="programming",
    tags=["python", "learning"],
    quality_score=0.8
)
```

### プロンプト最適化の実行

```python
result = await learning_tool.optimize_prompt_template("prompt_name")
if result.get('status') == 'success':
    print(f"改善スコア: {result.get('improvement_score')}")
```

## 🔍 トラブルシューティング

### よくある問題

1. **Ollama 接続エラー**

   ```bash
   # Ollamaが起動しているか確認
   curl http://localhost:11434/api/tags

   # モデルが利用可能か確認
   ollama list
   ```

2. **Codex 互換モードが有効にならない**

   ```bash
   # 環境変数を確認
   echo $AGENT_USE_CODEX_AGENT

   # Windows
   echo %AGENT_USE_CODEX_AGENT%

   # 正しく設定
   export AGENT_USE_CODEX_AGENT=true  # Linux/Mac
   set AGENT_USE_CODEX_AGENT=true     # Windows
   ```

3. **データベースエラー（従来モード）**

   ```bash
   # データベースファイルの権限を確認
   ls -la data/agent.db
   ```

4. **学習システムが起動しない（従来モード）**

   ```bash
   # 設定を確認
   python -c "from src.agent.core.config import Config; print(Config().is_learning_enabled)"
   ```

5. **テストが失敗する**

   ```bash
   # 統合テストの場合、OLLAMAサーバーが必要
   ollama serve

   # 環境変数を設定
   export INTEGRATION_TEST=true
   ```

### ログの確認

```bash
# ログファイルを確認
tail -f data/logs/agent.log
```

## 🤝 貢献

1. フォークを作成
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。

## 🙏 謝辞

- [Ollama](https://ollama.ai/) - ローカル LLM 実行環境
- [SQLAlchemy](https://www.sqlalchemy.org/) - ORM フレームワーク
- [Loguru](https://loguru.readthedocs.io/) - ログ管理

## ⚡ パフォーマンス特性

| 項目         | CPU モード | GPU モード |
| ------------ | ---------- | ---------- |
| 起動時間     | ~5 秒      | ~3 秒      |
| 応答時間     | 5-15 秒    | 2-8 秒     |
| メモリ使用量 | 中程度     | 高         |
| 学習機能     | 完全対応   | 完全対応   |
| 安定性       | 高         | 高         |
| 並列処理     | 制限あり   | 高速       |

## 🚀 GPU 高速化設定

### GPU 要件

- NVIDIA GPU（CUDA 対応）
- 最低 4GB VRAM（推奨 8GB 以上）
- CUDA 11.8 以上

### GPU 設定の有効化

```bash
# GPU設定を有効化
export OLLAMA_GPU_ENABLED=true
export OLLAMA_GPU_MEMORY_FRACTION=0.8  # GPUメモリの80%を使用
export OLLAMA_GPU_LAYERS=32             # GPU層数（モデルに応じて調整）
export OLLAMA_PARALLEL_REQUESTS=4      # 並列リクエスト数

# エージェントを起動
python agent_cli.py
```

### GPU 設定の最適化

#### RTX 4050 (6GB VRAM) の場合

```bash
export OLLAMA_GPU_MEMORY_FRACTION=0.7
export OLLAMA_GPU_LAYERS=28
export OLLAMA_PARALLEL_REQUESTS=2
```

#### RTX 4060/4070 (8GB+ VRAM) の場合

```bash
export OLLAMA_GPU_MEMORY_FRACTION=0.8
export OLLAMA_GPU_LAYERS=32
export OLLAMA_PARALLEL_REQUESTS=4
```

#### RTX 4080/4090 (12GB+ VRAM) の場合

```bash
export OLLAMA_GPU_MEMORY_FRACTION=0.9
export OLLAMA_GPU_LAYERS=40
export OLLAMA_PARALLEL_REQUESTS=6
```

## 📋 設定例

### 本番環境向け設定（GPU 対応）

```bash
# .env ファイル
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2:7b-instruct
OLLAMA_TIMEOUT=30
OLLAMA_GPU_ENABLED=true
OLLAMA_GPU_MEMORY_FRACTION=0.8
OLLAMA_GPU_LAYERS=32
OLLAMA_PARALLEL_REQUESTS=4
DATABASE_URL=sqlite:///data/agent.db
AGENT_LEARNING_ENABLED=true
```

### 開発環境向け設定（GPU 対応）

```bash
# .env ファイル
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2:7b-instruct
OLLAMA_GPU_ENABLED=true
OLLAMA_GPU_MEMORY_FRACTION=0.6
OLLAMA_GPU_LAYERS=24
OLLAMA_PARALLEL_REQUESTS=2
DATABASE_URL=sqlite:///data/agent.db
AGENT_LEARNING_ENABLED=true
AGENT_LEARNING_INTERVAL_MINUTES=15
```

---

**注意**: このシステムは学習・研究目的で作成されています。本格的な運用前に十分なテストを行ってください。

**推奨**: まずは基本的なチャット機能から始めて、必要に応じて学習システムを有効化することをお勧めします。
