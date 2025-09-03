# Advanced Self-Learning AI Agent Documentation

## 概要

RTX 4050 6GB VRAM 環境で動作する高性能自己学習 AI エージェントの包括的なドキュメントです。DeepSeek-R1 推論技術、SAKANA AI 進化的手法、永続的記憶システムを統合した次世代 AI エージェントシステムです。

## 📚 ドキュメント構成

### 基本ガイド

- [📦 インストールガイド](INSTALLATION.md) - システムのセットアップ手順
- [⚙️ 設定ガイド](CONFIGURATION.md) - 設定ファイルの詳細説明
- [🚀 使用方法ガイド](USAGE.md) - 基本的な使用方法とベストプラクティス

### 技術リファレンス

- [🔌 API リファレンス](API_REFERENCE.md) - REST API の詳細仕様
- [🏗️ アーキテクチャ](ARCHITECTURE.md) - システム設計の詳細
- [🔧 トラブルシューティング](TROUBLESHOOTING.md) - 問題解決ガイド

### 高度なトピック

- [🧠 推論エンジン](REASONING_ENGINE.md) - DeepSeek-R1 推論システムの詳細
- [🧬 進化的学習](EVOLUTIONARY_LEARNING.md) - SAKANA AI 風学習システム
- [💾 記憶システム](MEMORY_SYSTEM.md) - 永続的記憶管理の仕組み
- [📊 監視システム](MONITORING.md) - リアルタイム監視と最適化

## 🌟 主要機能

### 1. DeepSeek-R1 推論エンジン

- Chain-of-Thought 推論プロセス
- 動的量子化による VRAM 最適化
- GPU/CPU ハイブリッド処理

### 2. 永続的記憶システム

- セッション間での完全なコンテキスト継続
- 自動重要度判定による記憶選択
- ベクトル検索による関連記憶取得

### 3. 進化的学習システム

- LoRA アダプタの自動交配・変異
- 性能ベースの自然選択
- QLoRA による効率的ファインチューニング

### 4. リアルタイム監視

- GPU/CPU 使用率の継続監視
- メモリ使用量の自動最適化
- 温度監視と自動冷却制御

### 5. OpenAI 互換 API

- 既存ツールとのシームレス統合
- ストリーミング応答対応
- カスタムエンドポイント

## 🚀 クイックスタート

### 基本セットアップ

```bash
# 1. 依存関係のインストール
pip install -r requirements.txt

# 2. Ollama モデルの準備
ollama pull deepseek-r1:7b
ollama pull qwen2.5:7b-instruct-q4_k_m
ollama pull qwen2:1.5b-instruct-q4_k_m

# 3. 設定ファイルの確認
cp config/system.yaml.example config/system.yaml
# 必要に応じて設定を調整

# 4. エージェントの起動
python -m src.advanced_agent.main
```

### 基本的な使用例

```python
# Python API クライアント
from src.advanced_agent.client import AdvancedAgentClient

client = AdvancedAgentClient("http://localhost:8000")

# 推論実行（記憶システム有効）
response = client.chat(
    message="複雑な数学問題を解いてください",
    use_memory=True,
    use_cot=True
)

# 学習状況の確認
learning_stats = client.get_learning_stats()
memory_stats = client.get_memory_stats()
```

## 💻 システム要件

### 最小要件

- **GPU**: NVIDIA RTX 4050 (6GB VRAM)
- **RAM**: 16GB
- **CPU**: Intel i5-12400 相当
- **Python**: 3.11+
- **CUDA**: 11.8+

### 推奨要件

- **GPU**: NVIDIA RTX 4060 (8GB VRAM) 以上
- **RAM**: 32GB
- **CPU**: Intel i7-13700H 相当以上
- **Python**: 3.11+
- **CUDA**: 12.0+
- **SSD**: 100GB 以上の空き容量

## 🔧 設定例

### GPU メモリ最適化設定

```yaml
# config/system.yaml
gpu:
  max_vram_gb: 5.0
  quantization_levels: [8, 4, 3]
  temperature_threshold: 80
  auto_optimization: true
```

### 記憶システム設定

```yaml
# config/system.yaml
persistent_memory:
  max_short_term_items: 1000
  max_long_term_items: 10000
  importance_threshold: 0.7
  consolidation_interval_hours: 24
```

## 📈 性能指標

### 推論性能

- **応答時間**: 平均 1.5 秒（目標 2 秒以内）
- **VRAM 使用率**: 最大 83%（5GB/6GB）
- **推論精度**: 85% 以上

### 記憶システム性能

- **記憶検索速度**: 100ms 以内
- **コンテキスト継続率**: 95% 以上
- **重要度判定精度**: 80% 以上

## 🧪 テストとデバッグ

### テスト実行

```bash
# 全テスト実行
python -m pytest tests/ -v

# 特定コンポーネントのテスト
python -m pytest tests/test_reasoning_engine.py -v
python -m pytest tests/test_memory_system.py -v
python -m pytest tests/test_evolutionary_learning.py -v

# 性能テスト
python -m pytest tests/performance/ -v --benchmark-only
```

### デバッグモード

```bash
# デバッグモードでの起動
AGENT_LOG_LEVEL=DEBUG python -m src.advanced_agent.main

# 詳細ログの確認
tail -f logs/advanced_agent.log
```

## 🔍 監視とメトリクス

### リアルタイム監視

```bash
# 監視ダッシュボードの起動
python -m src.advanced_agent.monitoring.dashboard

# CLI 監視ツール
python -m src.advanced_agent.monitoring.cli_monitor
```

### メトリクス収集

- GPU 使用率・温度・VRAM 使用量
- 推論速度・精度・エラー率
- 記憶システムの使用状況
- 学習システムの進捗

## 🤝 コミュニティとサポート

### 貢献方法

- [CONTRIBUTING.md](../CONTRIBUTING.md) - 貢献ガイドライン
- [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md) - 行動規範

### サポート

- GitHub Issues - バグ報告・機能要望
- Discussions - 質問・議論
- Wiki - コミュニティドキュメント

---

**📖 詳細なドキュメントで、Advanced Self-Learning AI Agent を最大限活用しましょう！**
