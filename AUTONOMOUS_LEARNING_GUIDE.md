# 自律的自己学習システム使用ガイド

## 📋 概要

自己学習機能を停止するまで自走させるための 3 つのシステムを提供します：

1. **Autonomous Self-Learning** - 高度な自律学習システム
2. **Continuous Learning** - シンプルな継続学習システム
3. **Learning Monitor** - リアルタイム監視システム

## 🚀 システム一覧

### 1. 高度自律学習システム (`autonomous_self_learning.py`)

**特徴:**

- 包括的な学習サイクル実行
- 詳細な分析と最適化
- 複数の停止条件設定
- 詳細なログ出力

**実行例:**

```bash
# 基本実行（手動停止まで継続）
python autonomous_self_learning.py

# 最大10サイクルで停止
python autonomous_self_learning.py --max-cycles 10

# 最大2時間で停止
python autonomous_self_learning.py --max-hours 2

# 品質スコア0.8到達で停止
python autonomous_self_learning.py --target-quality 0.8

# 複数条件設定
python autonomous_self_learning.py --max-cycles 50 --max-hours 6 --target-quality 0.85
```

### 2. シンプル継続学習システム (`continuous_learning.py`)

**特徴:**

- 軽量で理解しやすい
- 基本的な学習サイクル
- リアルタイム進捗表示
- 簡単な設定

**実行例:**

```bash
# 基本実行（30秒間隔）
python continuous_learning.py

# 5サイクルで停止
python continuous_learning.py --max-cycles 5

# 1時間で停止
python continuous_learning.py --max-hours 1

# 60秒間隔で実行
python continuous_learning.py --interval 60

# 組み合わせ
python continuous_learning.py --max-cycles 20 --interval 45
```

### 3. 学習監視システム (`learning_monitor.py`)

**特徴:**

- リアルタイム統計表示
- 変化量の追跡
- ダッシュボード表示
- 監視専用（学習は実行しない）

**実行例:**

```bash
# リアルタイムモニター（10秒間隔）
python learning_monitor.py --mode monitor

# 5秒間隔でモニター
python learning_monitor.py --mode monitor --interval 5

# ダッシュボード表示（1回のみ）
python learning_monitor.py --mode dashboard
```

## 🔧 使用シナリオ

### シナリオ 1: 長時間自動学習

```bash
# ターミナル1: 自律学習実行
python autonomous_self_learning.py --max-hours 8 --target-quality 0.9

# ターミナル2: 監視（別ウィンドウ）
python learning_monitor.py --mode monitor --interval 30
```

### シナリオ 2: 短時間テスト学習

```bash
# 5サイクルのテスト実行
python continuous_learning.py --max-cycles 5 --interval 20
```

### シナリオ 3: 夜間自動学習

```bash
# 8時間または品質スコア0.85到達まで実行
python autonomous_self_learning.py --max-hours 8 --target-quality 0.85
```

### シナリオ 4: 監視のみ

```bash
# 学習状況の監視のみ
python learning_monitor.py --mode monitor --interval 15
```

## 📊 出力ファイル

### 自律学習システム

- `autonomous_learning_results_YYYYMMDD_HHMMSS.json` - 詳細な学習結果
- `autonomous_learning_YYYYMMDD_HHMMSS.log` - 実行ログ

### 継続学習システム

- `continuous_learning_results_YYYYMMDD_HHMMSS.json` - 学習サイクル結果

## 🛑 停止方法

### 手動停止

- **Ctrl+C** - 全システム共通の安全停止方法
- 現在のサイクル完了後に停止

### 自動停止条件

- `--max-cycles N` - N 回のサイクル完了後
- `--max-hours H` - H 時間経過後
- `--target-quality Q` - 品質スコア Q 到達後

## 📈 監視項目

### リアルタイム監視

- 総学習データ数
- 知識アイテム数
- 平均品質スコア
- 高品質データ数
- データベースサイズ

### 変化追跡

- データ増加数
- 品質スコア変化
- 高品質データ増加数

## ⚙️ 設定カスタマイズ

### 学習間隔調整

```bash
# 短い間隔（頻繁な学習）
python continuous_learning.py --interval 10

# 長い間隔（負荷軽減）
python continuous_learning.py --interval 300  # 5分
```

### 停止条件の組み合わせ

```bash
# 複数条件（いずれか満足で停止）
python autonomous_self_learning.py \
  --max-cycles 100 \
  --max-hours 12 \
  --target-quality 0.9
```

## 🔍 トラブルシューティング

### よくある問題

1. **初期化エラー**

   ```
   ❌ 初期化エラー: Database connection failed
   ```

   - データベースファイルの権限確認
   - `data/` ディレクトリの存在確認

2. **学習ツールエラー**

   ```
   ❌ 学習ツールが利用できません
   ```

   - OLLAMA サーバーの起動確認
   - 設定ファイルの確認

3. **メモリ不足**
   - `--interval` を大きくして負荷軽減
   - `--max-cycles` で早期停止

### デバッグ方法

1. **ログ確認**

   ```bash
   tail -f autonomous_learning_*.log
   ```

2. **データベース状態確認**

   ```bash
   python learning_monitor.py --mode dashboard
   ```

3. **手動テスト**
   ```bash
   python continuous_learning.py --max-cycles 1
   ```

## 📋 推奨設定

### 開発・テスト環境

```bash
# 短時間テスト
python continuous_learning.py --max-cycles 3 --interval 15
```

### 本番環境

```bash
# 長時間安定実行
python autonomous_self_learning.py --max-hours 6 --target-quality 0.8
```

### 監視環境

```bash
# 常時監視
python learning_monitor.py --mode monitor --interval 30
```

## 🔄 システム連携

### 複数システム同時実行

```bash
# ターミナル1: 学習実行
python autonomous_self_learning.py --max-hours 4

# ターミナル2: 監視
python learning_monitor.py --mode monitor

# ターミナル3: 定期ダッシュボード
while true; do
  python learning_monitor.py --mode dashboard
  sleep 300  # 5分間隔
done
```

## 📞 サポート

### 実行前チェック

```bash
# システム診断
python quick_self_learning_test.py

# 詳細診断
python self_learning_diagnostic_test.py
```

### 問題報告

- ログファイルの内容
- 実行コマンド
- エラーメッセージ
- システム環境情報

---

**注意**: 長時間実行時はシステムリソースの監視を推奨します。
