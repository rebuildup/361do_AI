# 設定ガイド

## 設定ファイル概要

Advanced Self-Learning AI Agent は複数の設定ファイルを使用します：

- `config/system.yaml` - メインシステム設定
- `config/advanced_agent.yaml` - エージェント固有設定
- `config/.env` - 環境変数
- `config/gpu_config.env` - GPU 最適化設定

## メインシステム設定 (system.yaml)

### GPU 設定

```yaml
gpu:
  max_vram_gb: 5.0 # 最大VRAM使用量（RTX 4050は5GB推奨）
  quantization_levels: [8, 4, 3] # 量子化レベル（8bit→4bit→3bit）
  temperature_threshold: 80 # GPU温度閾値（℃）
  memory_reserve_gb: 0.5 # 予約メモリ（GB）
  monitoring_interval: 1.0 # 監視間隔（秒）
```

### CPU 設定

```yaml
cpu:
  max_threads: 16 # 最大スレッド数
  offload_threshold: 0.8 # CPUオフロード閾値
  hybrid_processing: true # ハイブリッド処理有効
```

### モデル設定

```yaml
models:
  primary: "deepseek-r1:7b" # メインモデル
  fallback: "qwen2.5:7b-instruct-q4_k_m" # フォールバックモデル
  emergency: "qwen2:1.5b-instruct-q4_k_m" # 緊急時モデル
  auto_download: true # 自動ダウンロード
```

### 学習設定

```yaml
learning:
  adapter_pool_size: 10 # アダプタプールサイズ
  generation_size: 5 # 世代サイズ
  mutation_rate: 0.1 # 変異率
  crossover_rate: 0.7 # 交配率
  max_training_epochs: 10 # 最大学習エポック数
  learning_rate: 0.0001 # 学習率
```

### 記憶システム設定

```yaml
persistent_memory:
  db_path: "data/agent_memory.db" # メインDB
  vector_db_path: "data/agent_vectors.db" # ベクトルDB
  max_short_term_items: 1000 # 短期記憶最大数
  max_long_term_items: 10000 # 長期記憶最大数
  importance_threshold: 0.7 # 重要度閾値
  consolidation_interval_hours: 24 # 統合間隔（時間）
  auto_cleanup: true # 自動クリーンアップ
```

## 環境変数設定 (.env)

### 基本設定

```bash
# エージェント機能
AGENT_ENABLE_DEEPSEEK_R1=True
AGENT_ENABLE_EVOLUTIONARY_LEARNING=True
AGENT_ENABLE_PERSISTENT_MEMORY=True
AGENT_ENABLE_MULTIMODAL=True
AGENT_ENABLE_REAL_TIME_MONITORING=True

# Ollama設定
AGENT_OLLAMA_BASE_URL=http://localhost:11434
AGENT_OLLAMA_PRIMARY_MODEL=deepseek-r1:7b
AGENT_OLLAMA_FALLBACK_MODEL=qwen2.5:7b-instruct-q4_k_m

# GPU設定
AGENT_MAX_VRAM_GB=5.0
AGENT_GPU_TEMPERATURE_THRESHOLD=80
AGENT_QUANTIZATION_LEVELS=8,4,3

# API設定
AGENT_API_HOST=localhost
AGENT_API_PORT=8000
AGENT_MAX_CONCURRENT_REQUESTS=10
```

## GPU 最適化設定 (gpu_config.env)

### RTX 4050 向け最適化

```bash
# メモリ管理
OLLAMA_GPU_MEMORY_FRACTION=0.83  # 5GB/6GB = 83%
OLLAMA_GPU_LAYERS=32             # DeepSeek-R1最適化
OLLAMA_PARALLEL_REQUESTS=1       # 安全な並列数

# 動的量子化
QUANTIZATION_LEVELS=8,4,3
DEFAULT_QUANTIZATION=4
AUTO_QUANTIZATION_ENABLED=true

# 監視設定
GPU_MEMORY_THRESHOLD=0.9
GPU_TEMPERATURE_THRESHOLD=80
MEMORY_CHECK_INTERVAL=1
```

## 性能チューニング

### VRAM 使用量の最適化

#### 低 VRAM 環境（4GB 以下）

```yaml
gpu:
  max_vram_gb: 3.5
  quantization_levels: [4, 3]
models:
  primary: "qwen2.5:7b-instruct-q4_k_m"
learning:
  adapter_pool_size: 5
```

#### 高 VRAM 環境（8GB 以上）

```yaml
gpu:
  max_vram_gb: 7.0
  quantization_levels: [8, 4]
models:
  primary: "deepseek-r1:7b"
learning:
  adapter_pool_size: 15
```

### CPU 処理の最適化

#### 高性能 CPU 環境

```yaml
cpu:
  max_threads: 32
  offload_threshold: 0.6
```

#### 低性能 CPU 環境

```yaml
cpu:
  max_threads: 8
  offload_threshold: 0.9
```

## 監視設定

### 詳細監視

```yaml
monitoring:
  gpu_monitoring:
    interval_seconds: 0.5 # 高頻度監視
    metrics: ["utilization", "memory", "temperature", "power"]
    alerts:
      memory_threshold: 0.85
      temperature_threshold: 75
      power_threshold: 200
  performance_tracking:
    response_time_target: 1.5 # より厳しい目標
    accuracy_threshold: 0.90
    memory_efficiency_target: 0.85
```

### 軽量監視

```yaml
monitoring:
  gpu_monitoring:
    interval_seconds: 5 # 低頻度監視
    metrics: ["memory", "temperature"]
    alerts:
      memory_threshold: 0.95
      temperature_threshold: 85
```

## セキュリティ設定

### 本番環境

```yaml
security:
  enable_encryption: true
  model_integrity_check: true
  data_anonymization: true
  api_key_required: true
  rate_limiting: true
```

### 開発環境

```yaml
security:
  enable_encryption: false
  model_integrity_check: false
  data_anonymization: false
  api_key_required: false
  rate_limiting: false
```

## 設定の検証

### 設定チェックコマンド

```bash
# 設定ファイルの検証
python -m src.advanced_agent.config.validate

# GPU設定の確認
python -m src.advanced_agent.config.check_gpu

# 記憶システム設定の確認
python -m src.advanced_agent.config.check_memory
```

### 設定テスト

```bash
# 設定に基づく動作テスト
python -m src.advanced_agent.config.test_config

# 性能テスト
python -m src.advanced_agent.config.benchmark_config
```

## トラブルシューティング

### よくある設定問題

#### VRAM 不足

```yaml
# 解決策: VRAM制限を下げる
gpu:
  max_vram_gb: 4.0
  quantization_levels: [4, 3]
```

#### 推論速度が遅い

```yaml
# 解決策: 並列処理を増やす
cpu:
  max_threads: 20
  offload_threshold: 0.7
```

#### 記憶システムが重い

```yaml
# 解決策: 記憶数を制限
persistent_memory:
  max_short_term_items: 500
  max_long_term_items: 5000
```

## 設定例

### バランス型設定（推奨）

```yaml
# RTX 4050 + 32GB RAM 向け
gpu:
  max_vram_gb: 5.0
cpu:
  max_threads: 16
learning:
  adapter_pool_size: 10
persistent_memory:
  max_short_term_items: 1000
```

### 性能重視設定

```yaml
# 高性能重視
gpu:
  max_vram_gb: 5.5
cpu:
  max_threads: 20
learning:
  adapter_pool_size: 15
```

### 安定性重視設定

```yaml
# 安定性重視
gpu:
  max_vram_gb: 4.0
cpu:
  max_threads: 12
learning:
  adapter_pool_size: 8
```
