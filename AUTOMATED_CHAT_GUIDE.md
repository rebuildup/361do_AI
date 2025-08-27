# 🤖 自動化チャットシステム 使用ガイド

## 📋 概要

このシステムは、事前に定義された質問を自動実行してエージェントの応答を取得する自動化システムです。手動でのテキスト入力が不要で、効率的にエージェントの機能をテストできます。

## 🚀 利用可能なスクリプト

### 1. 基本的な自動化チャット (`automated_chat.py`)

事前定義された質問セットまたはカスタム質問を自動実行します。

**使用例:**

```bash
# 基本機能テスト
python automated_chat.py --set basic_functionality

# コマンド実行テスト
python automated_chat.py --set command_execution

# Web検索テスト
python automated_chat.py --set web_search

# 包括的テスト
python automated_chat.py --set comprehensive_test

# カスタム質問
python automated_chat.py --questions "こんにちは" "systeminfoを実行" "help"

# 結果を保存しない
python automated_chat.py --set basic_functionality --no-save

# カスタムファイル名で保存
python automated_chat.py --set comprehensive_test --output my_test_results.json
```

### 2. バッチ実行システム (`batch_chat.py`)

複数のテストセットを連続実行します。

**使用例:**

```bash
# 全テストセットを実行
python batch_chat.py

# 特定のテストセットのみ実行
python batch_chat.py --sets basic_functionality command_execution

# 結果を指定ファイルに保存
python batch_chat.py --output batch_results.json
```

### 3. クイックテスト (`quick_test.py`)

簡単な 5 つの質問で基本機能をテストします。

**使用例:**

```bash
python quick_test.py
```

### 4. 設定ファイルベーステスト (`config_based_test.py`)

JSON 設定ファイルに基づいてテストを実行します。

**使用例:**

```bash
# デフォルト設定で全シナリオ実行
python config_based_test.py

# カスタム設定ファイル使用
python config_based_test.py --config my_test_config.json

# 特定シナリオのみ実行
python config_based_test.py --scenario basic_functionality

# 結果を指定ファイルに保存
python config_based_test.py --output detailed_results.json
```

## 📝 事前定義された質問セット

### `basic_functionality`

- こんにちは
- あなたの機能について教えてください
- help
- 利用可能なツールを教えて

### `command_execution`

- systeminfo コマンドを実行してください
- tasklist でプロセス一覧を表示
- 現在のディレクトリ内容を確認
- ホスト名を表示してください

### `web_search`

- 最新の AI 技術について調べて
- Python プログラミング チュートリアルを探して
- 機械学習 入門 情報を検索
- ボーカロイド 楽曲投稿祭 ボカコレ 情報を検索

### `learning_system`

- 学習データの統計を表示
- 一番古い学習データを教えて
- 学習システムの状態を確認
- プロンプト一覧を表示

### `comprehensive_test`

- こんにちは、機能テストを開始します
- systeminfo コマンドを実行してシステム情報を取得
- 先ほどの結果について簡潔に説明してください
- 最新の Python 情報について検索
- 学習データの統計情報を教えて
- help tools
- tasklist を実行してメモリ使用量の多いプロセスを特定
- ありがとうございました、テスト完了です

## ⚙️ 設定ファイル形式 (`test_config.json`)

```json
{
  "test_scenarios": {
    "my_custom_test": {
      "description": "カスタムテストの説明",
      "questions": ["質問1", "質問2", "質問3"]
    }
  },
  "execution_settings": {
    "question_interval_seconds": 1,
    "save_results": true,
    "show_detailed_output": true,
    "max_response_length": 500
  }
}
```

## 📊 出力結果の形式

### 基本結果形式

```json
{
  "question_number": 1,
  "question": "こんにちは",
  "response": "こんにちは！何かお手伝いできることがありますか？",
  "session_id": "session_123",
  "execution_time": 2.34,
  "intent": { "primary_intent": "general_chat" },
  "tools_used": [],
  "timestamp": "2025-08-28T00:00:00",
  "success": true
}
```

### バッチ結果形式

```json
{
  "start_time": "2025-08-28T00:00:00",
  "end_time": "2025-08-28T00:05:00",
  "duration_seconds": 300.0,
  "test_sets_executed": 3,
  "total_questions": 15,
  "results": {
    "basic_functionality": {
      "questions": [...],
      "results": [...],
      "summary": {
        "total_questions": 4,
        "successful": 4,
        "success_rate": 1.0,
        "total_execution_time": 12.5,
        "average_execution_time": 3.125
      }
    }
  }
}
```

## 🎯 使用シナリオ

### 1. 開発中の機能テスト

```bash
# 新機能をテスト
python automated_chat.py --questions "新機能をテスト" "help tools" "status"
```

### 2. 定期的な動作確認

```bash
# 毎日の動作確認
python quick_test.py
```

### 3. 包括的な機能検証

```bash
# 全機能の包括的テスト
python batch_chat.py
```

### 4. カスタムテストシナリオ

```bash
# 設定ファイルでカスタマイズ
python config_based_test.py --config my_scenarios.json
```

## 🔧 トラブルシューティング

### よくある問題

**1. システム初期化エラー**

- OLLAMA サーバーが起動しているか確認
- データベースファイルのアクセス権限を確認

**2. 実行時間が長い**

- 質問数を減らす
- `--no-save` オプションを使用

**3. 結果が保存されない**

- ディスクの空き容量を確認
- 書き込み権限を確認

## 📈 パフォーマンス最適化

### 実行時間短縮

- 必要最小限の質問セットを使用
- バッチ実行時は間隔を短縮
- 結果保存を無効化（`--no-save`）

### メモリ使用量削減

- 大量の質問を分割実行
- 定期的にセッションをリセット

## 🎉 活用例

### CI/CD パイプラインでの自動テスト

```bash
# GitHub Actions等での自動テスト
python batch_chat.py --no-save > test_output.log
```

### 定期監視スクリプト

```bash
# cronで定期実行
0 */6 * * * cd /path/to/project && python quick_test.py
```

### 負荷テスト

```bash
# 複数回連続実行
for i in {1..10}; do python automated_chat.py --set comprehensive_test; done
```

---

このシステムを使用することで、手動でのテキスト入力なしに効率的にエージェントの機能をテストできます！
