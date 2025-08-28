# 🚀 マルチエージェント学習システム 実行前最終チェックリスト

## ✅ 事前準備完了項目

### 1. システム要件確認

- [ ] Python 3.8 以上がインストール済み
- [ ] メモリ 8GB 以上利用可能
- [ ] ディスク容量 2GB 以上空き
- [ ] CPU 4 コア以上（推奨）

### 2. 環境設定確認

- [ ] Ollama がインストール・起動済み
- [ ] 推奨モデル（qwen2:7b-instruct）がダウンロード済み
- [ ] 仮想環境が有効化済み
- [ ] 必要な Python パッケージがインストール済み

### 3. ファイル構成確認

- [ ] `multi_agent_learning_system.py` - メインシステム
- [ ] `pre_launch_checklist.py` - 事前チェックツール
- [ ] `system_optimizer.py` - 最適化ツール
- [ ] `launch_multi_agent_learning.py` - 統合ランチャー
- [ ] `config/.env` - 環境変数設定
- [ ] `src/` ディレクトリ - ソースコード

### 4. 権限・アクセス確認

- [ ] データディレクトリへの書き込み権限
- [ ] ログファイル作成権限
- [ ] ネットワークアクセス（Web 検索用）

## 🛠️ 実行前準備手順

### 自動準備（推奨）

```bash
# 統合ランチャーで全自動実行
python launch_multi_agent_learning.py

# または Windows バッチファイル
examples\prepare_and_launch.bat
```

### 手動準備

```bash
# 1. 事前チェック実行
python pre_launch_checklist.py

# 2. システム最適化実行
python system_optimizer.py

# 3. メインシステム実行
python multi_agent_learning_system.py --hours 8.0
```

## ⚡ 最適化設定

### 推奨環境変数

```bash
# Python最適化
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# Ollama最適化
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_FLASH_ATTENTION=1

# システム最適化
export OMP_NUM_THREADS=4
```

### GPU 使用時の追加設定

```bash
export OLLAMA_GPU_ENABLED=true
export OLLAMA_GPU_MEMORY_FRACTION=0.8
export OLLAMA_GPU_LAYERS=32
export OLLAMA_PARALLEL_REQUESTS=4
```

## 📊 実行中の監視

### 監視すべき項目

- [ ] CPU 使用率（80%以下を維持）
- [ ] メモリ使用率（90%以下を維持）
- [ ] ディスク容量（1GB 以上の空きを維持）
- [ ] ログファイルサイズ（100MB 以下）

### 監視コマンド

```bash
# システムリソース監視
python system_monitor.py

# ログファイル監視
tail -f multi_agent_learning_*.log

# プロセス監視
ps aux | grep python
```

## 🚨 トラブルシューティング

### よくある問題と対処法

#### 1. Ollama 接続エラー

```bash
# Ollamaサービス確認
curl http://localhost:11434/api/tags

# Ollama再起動
ollama serve
```

#### 2. メモリ不足

```bash
# 他のアプリケーション終了
# 仮想メモリ設定確認
# システム再起動
```

#### 3. データベースエラー

```bash
# データベースファイル権限確認
ls -la data/agent.db

# データベース再初期化
rm data/agent.db
python -c "from src.agent.core.database import DatabaseManager; import asyncio; asyncio.run(DatabaseManager('sqlite:///data/agent.db').initialize())"
```

#### 4. プロセス優先度問題

```bash
# 管理者権限で実行（Windows）
# sudo権限で実行（Linux/Mac）
```

## 📋 実行時の注意事項

### 実行中に避けるべきこと

- [ ] 他の重いアプリケーションの起動
- [ ] システムの再起動・スリープ
- [ ] ネットワーク接続の切断
- [ ] ディスク容量の大量消費

### 推奨する実行環境

- [ ] 安定した電源供給
- [ ] 安定したネットワーク接続
- [ ] 十分な冷却環境
- [ ] 無人実行可能な環境

## 🎯 実行開始前の最終確認

### 実行直前チェック

```bash
# 1. システム状態確認
python pre_launch_checklist.py

# 2. Ollama動作確認
ollama list
ollama run qwen2:7b-instruct "Hello"

# 3. ディスク容量確認
df -h .  # Linux/Mac
dir    # Windows

# 4. メモリ確認
free -h  # Linux
Get-ComputerInfo | Select-Object TotalPhysicalMemory,AvailablePhysicalMemory  # Windows PowerShell
```

### 実行開始

```bash
# 統合ランチャーで実行（推奨）
python launch_multi_agent_learning.py

# 直接実行
python multi_agent_learning_system.py --hours 8.0

# テストモード（6分間）
python multi_agent_learning_system.py --test-mode
```

## 📈 期待される結果

### 8 時間実行での予想成果

- **会話数**: 約 192 回（4 エージェント × 2 ラウンド × 24 サイクル）
- **学習サイクル**: 約 96 回
- **生成データ**: 数百件の学習データ
- **ログファイル**: 50-100MB
- **結果ファイル**: JSON 形式の詳細レポート

### 成功指標

- [ ] 全エージェントが正常に動作
- [ ] 継続的な会話生成
- [ ] 学習データの蓄積
- [ ] システムエラーなし
- [ ] 8 時間完走または安全停止

## 🎉 実行完了後の作業

### 結果確認

- [ ] ログファイルの確認
- [ ] 学習結果 JSON の分析
- [ ] システムパフォーマンスの評価
- [ ] 生成された学習データの品質確認

### データ保存

- [ ] 重要なログファイルのバックアップ
- [ ] 学習結果の永続保存
- [ ] システム設定の記録

---

**🚀 準備完了！マルチエージェント学習システムの 8 時間実行を開始する準備が整いました。**

**実行コマンド**: `python launch_multi_agent_learning.py`
