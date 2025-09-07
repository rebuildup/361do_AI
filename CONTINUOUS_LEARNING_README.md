# 4 時間継続学習システム

workspace フォルダ内の ChatGPT と Claude の conversation.json ファイルを 4 時間継続してエージェントに学習させるシステムです。

## ファイル構成

- `continuous_learning_system.py` - メインの継続学習システム
- `start_continuous_learning.py` - 学習開始スクリプト
- `monitor_learning_progress.py` - 学習進捗監視スクリプト
- `validate_learning_results.py` - 学習結果検証スクリプト
- `start_learning.bat` - Windows 用学習開始バッチファイル
- `monitor_learning.bat` - Windows 用進捗監視バッチファイル

## 使用方法

### 1. 学習開始

#### 方法 A: バッチファイルを使用（Windows）

```bash
start_learning.bat
```

#### 方法 B: Python スクリプトを直接実行

```bash
python start_continuous_learning.py
```

### 2. 学習進捗監視

別のターミナルで進捗を監視できます：

#### 方法 A: バッチファイルを使用（Windows）

```bash
monitor_learning.bat
```

#### 方法 B: Python スクリプトを直接実行

```bash
python monitor_learning_progress.py
```

### 3. 学習結果検証

学習完了後、結果を検証できます：

```bash
python validate_learning_results.py
```

## 学習設定

学習システムは以下の設定で動作します：

- **学習時間**: 4 時間
- **バッチサイズ**: 5 会話
- **学習間隔**: 30 秒
- **サイクルあたり最大会話数**: 50 会話
- **学習率**: 0.1
- **メモリ保持期間**: 7 日間

## データソース

システムは以下のファイルから学習データを読み込みます：

- `workspace/chat-gpt-data/conversations.json` - ChatGPT 会話データ
- `workspace/chat-gpt-data/shared_conversations.json` - ChatGPT 共有会話データ
- `workspace/claude-data/conversations.json` - Claude 会話データ

## データベース

学習データは以下の SQLite データベースに保存されます：

- `data/continuous_learning.db` - 学習セッション、進捗、会話データ
- `data/self_learning_agent.db` - エージェントの学習データ

## ログファイル

- `logs/continuous_learning.log` - 学習システムのログ
- `logs/learning_report_*.txt` - 学習結果レポート

## 学習プロセス

1. **初期化**: エージェントとデータベースを初期化
2. **データ読み込み**: conversation.json ファイルから会話データを読み込み
3. **学習ループ**: 4 時間にわたって以下の処理を繰り返し
   - 会話データをバッチ処理
   - エージェントに学習データを送信
   - 学習進捗を記録
   - エポックを更新
4. **完了**: 学習統計を記録し、レポートを生成

## 監視機能

学習中は以下の情報を監視できます：

- 学習進捗（パーセンテージ）
- 経過時間・残り時間
- 処理済み会話数
- 学習サイクル数
- 現在のエポック
- ソース別統計

## 検証機能

学習完了後、以下の指標で効果を検証できます：

- **基本統計**: 学習時間、総会話数、サイクル数、エポック数
- **効率指標**: 時間あたりの処理数、サイクル数、エポック数
- **品質指標**: 平均品質スコア、品質評価
- **データ多様性**: ソース別統計、ソース多様性

## 注意事項

- 学習中はシステムリソースを大量に使用します
- 4 時間の学習時間中はシステムを停止しないでください
- 学習データは自動的にシャッフルされて処理されます
- Ctrl+C で早期終了可能ですが、学習効果が低下する可能性があります

## トラブルシューティング

### エージェント初期化エラー

- `config/agent_config.yaml`が存在することを確認
- 必要な依存関係がインストールされていることを確認

### データファイルが見つからない

- `workspace/`フォルダ内に conversation.json ファイルが存在することを確認
- ファイルの読み取り権限があることを確認

### データベースエラー

- `data/`フォルダの書き込み権限があることを確認
- 既存のデータベースファイルが破損していないことを確認

## カスタマイズ

学習設定を変更する場合は、`continuous_learning_system.py`の`learning_config`を編集してください：

```python
self.learning_config = {
    'batch_size': 5,                    # バッチサイズ
    'learning_interval': 30,            # 学習間隔（秒）
    'max_conversations_per_cycle': 50,  # サイクルあたり最大会話数
    'learning_rate': 0.1,               # 学習率
    'memory_retention_days': 7          # メモリ保持期間（日）
}
```
