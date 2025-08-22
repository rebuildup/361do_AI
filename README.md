# 自己学習型 AI エージェント

高度な自己学習機能を持つ AI エージェントシステムです。エージェント自身が学習データやプロンプトを操作し、継続的に改善を行うことができます。

## 🚀 主な機能

### 学習システム

- **自動学習**: 会話データから自動的に学習データを抽出・改善
- **知識ベース管理**: 会話から有用な知識を抽出し、知識ベースを構築
- **プロンプト最適化**: LLM を使用してプロンプトを自動最適化
- **パフォーマンス分析**: システムの性能を継続的に監視・改善

### プロンプト管理

- **カスタムプロンプト**: ユーザーが独自のプロンプトを追加・編集
- **プロンプト最適化**: 既存のプロンプトを自動的に改善
- **バージョン管理**: プロンプトの変更履歴を追跡
- **エクスポート/インポート**: プロンプトのバックアップ・復元

### 学習データ管理

- **データ追加**: カテゴリ別に学習データを追加
- **品質管理**: 学習データの品質スコアを管理
- **自動改善**: 低品質なデータを自動的に改善
- **エクスポート/インポート**: 学習データのバックアップ・復元

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

# モデルをダウンロード
ollama pull llama2
```

## 🎯 使用方法

### CLI エージェントの起動

```bash
python agent_cli.py
```

### 基本コマンド

#### チャット

```bash
chat こんにちは
```

#### 学習システム管理

```bash
learn start          # 学習システム開始
learn stop           # 学習システム停止
learn status         # 学習システム状態確認
learn cycle          # 手動で学習サイクル実行
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

```bash
python test_self_learning_simple.py
```

このテストスクリプトは以下の機能をテストします：

- 学習データの追加・取得
- プロンプトテンプレートの管理
- 学習システムの開始・停止
- エージェントとの会話
- データのエクスポート・インポート
- プロンプト最適化

## 📁 プロジェクト構造

```
008_LLM/
├── agent_cli.py                 # CLIエージェント
├── test_self_learning_simple.py # テストスクリプト
├── src/
│   └── agent/
│       ├── core/
│       │   ├── config.py        # 設定管理
│       │   ├── database.py      # データベース管理
│       │   ├── ollama_client.py # Ollamaクライアント
│       │   └── agent_manager.py # エージェント管理
│       ├── tools/
│       │   ├── learning_tool.py # 学習ツール
│       │   ├── search_tool.py   # 検索ツール
│       │   └── file_tool.py     # ファイルツール
│       └── self_tuning/
│           └── advanced_learning.py # 高度な学習システム
└── data/
    ├── learning_data/           # 学習データ
    ├── prompts/                 # プロンプトファイル
    └── logs/                    # ログファイル
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
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=llama2
export DATABASE_URL=sqlite:///data/agent.db
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
   ```

2. **データベースエラー**

   ```bash
   # データベースファイルの権限を確認
   ls -la data/agent.db
   ```

3. **学習システムが起動しない**
   ```bash
   # 設定を確認
   python -c "from src.agent.core.config import Config; print(Config().is_learning_enabled)"
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

---

**注意**: このシステムは学習目的で作成されています。本格的な運用前に十分なテストを行ってください。
