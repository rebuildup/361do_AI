# GPU 高速化セットアップガイド

## 🚀 概要

このガイドでは、NVIDIA GPU を使用して AI エージェントシステムを高速化する方法を説明します。

## 📋 前提条件

### ハードウェア要件

- NVIDIA GPU（CUDA 対応）
- 最低 4GB VRAM（推奨 6GB 以上）
- RTX 20 シリーズ以降推奨

### ソフトウェア要件

- NVIDIA ドライバー（最新版）
- CUDA 11.8 以上
- OLLAMA（GPU 対応版）

## 🔧 セットアップ手順

### 1. GPU 環境の確認

```bash
# GPU情報を確認
nvidia-smi

# CUDA バージョンを確認
nvcc --version
```

### 2. OLLAMA の GPU 対応確認

```bash
# OLLAMAが起動していることを確認
ollama serve

# モデルがGPUで動作していることを確認
ollama list
```

### 3. GPU 設定の適用

#### 方法 1: 環境変数で設定

```bash
# Windows (PowerShell)
$env:OLLAMA_GPU_ENABLED="true"
$env:OLLAMA_GPU_MEMORY_FRACTION="0.7"
$env:OLLAMA_GPU_LAYERS="28"
$env:OLLAMA_PARALLEL_REQUESTS="2"

# Linux/Mac
export OLLAMA_GPU_ENABLED=true
export OLLAMA_GPU_MEMORY_FRACTION=0.7
export OLLAMA_GPU_LAYERS=28
export OLLAMA_PARALLEL_REQUESTS=2
```

#### 方法 2: 設定ファイルを使用

```bash
# 提供された設定ファイルを使用
# Windows
for /f "delims=" %i in (gpu_config.env) do set %i

# Linux/Mac
source gpu_config.env
```

### 4. エージェントの起動

```bash
python agent_cli.py
```

## ⚙️ GPU 設定の最適化

### RTX 4050 (6GB VRAM)

```bash
OLLAMA_GPU_MEMORY_FRACTION=0.7
OLLAMA_GPU_LAYERS=28
OLLAMA_PARALLEL_REQUESTS=2
```

### RTX 4060/4070 (8GB VRAM)

```bash
OLLAMA_GPU_MEMORY_FRACTION=0.8
OLLAMA_GPU_LAYERS=32
OLLAMA_PARALLEL_REQUESTS=3
```

### RTX 4080/4090 (12GB+ VRAM)

```bash
OLLAMA_GPU_MEMORY_FRACTION=0.9
OLLAMA_GPU_LAYERS=40
OLLAMA_PARALLEL_REQUESTS=4
```

## 📊 パフォーマンステスト

### テストスクリプトの実行

```bash
# GPU パフォーマンステスト
python test_gpu_performance.py
```

### 期待される結果

- **初回トークン時間**: 0.1-0.5 秒
- **ストリーミング速度**: 20-50 tokens/sec
- **応答時間**: CPU 比で 2-5 倍高速化

## 🔍 トラブルシューティング

### よくある問題

#### 1. GPU が認識されない

```bash
# NVIDIA ドライバーを確認
nvidia-smi

# CUDA が利用可能か確認
python -c "import torch; print(torch.cuda.is_available())"
```

**解決方法**:

- NVIDIA ドライバーを最新版に更新
- CUDA ツールキットをインストール

#### 2. VRAM 不足エラー

```bash
# GPU メモリ使用量を確認
nvidia-smi
```

**解決方法**:

- `OLLAMA_GPU_MEMORY_FRACTION` を下げる（0.5-0.6）
- `OLLAMA_GPU_LAYERS` を減らす
- `OLLAMA_PARALLEL_REQUESTS` を 1 に設定

#### 3. パフォーマンスが向上しない

**確認事項**:

- OLLAMA が GPU を使用しているか確認
- モデルが GPU メモリに読み込まれているか確認
- 他の GPU 使用アプリケーションを終了

#### 4. システムが不安定になる

**解決方法**:

- GPU 設定を保守的な値に変更
- システムの冷却を確認
- 電源容量を確認

## 📈 パフォーマンス監視

### GPU 使用率の監視

```bash
# リアルタイム監視
watch -n 1 nvidia-smi

# ログ出力
nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 5
```

### エージェントのパフォーマンス監視

```bash
# システム状態確認
python agent_cli.py
> status

# 詳細統計情報
> stats
```

## 🎯 最適化のコツ

### 1. メモリ管理

- 不要なアプリケーションを終了
- GPU メモリ使用率を 80%以下に保つ
- 定期的に GPU メモリをクリア

### 2. 並列処理

- VRAM に応じて並列リクエスト数を調整
- 6GB 以下: 1-2 並列
- 8GB 以上: 3-4 並列
- 12GB 以上: 4-6 並列

### 3. モデル選択

- VRAM に適したモデルサイズを選択
- 7B モデル: 6GB VRAM 推奨
- 13B モデル: 12GB VRAM 推奨
- 70B モデル: 24GB+ VRAM 必要

## 🔄 設定の切り替え

### GPU ↔ CPU モードの切り替え

```bash
# GPU モードに切り替え
export OLLAMA_GPU_ENABLED=true

# CPU モードに切り替え
export OLLAMA_GPU_ENABLED=false

# エージェント再起動
python agent_cli.py
```

## 📝 設定例

### 開発環境（安定性重視）

```bash
OLLAMA_GPU_ENABLED=true
OLLAMA_GPU_MEMORY_FRACTION=0.6
OLLAMA_GPU_LAYERS=24
OLLAMA_PARALLEL_REQUESTS=1
```

### 本番環境（パフォーマンス重視）

```bash
OLLAMA_GPU_ENABLED=true
OLLAMA_GPU_MEMORY_FRACTION=0.8
OLLAMA_GPU_LAYERS=32
OLLAMA_PARALLEL_REQUESTS=3
```

### 実験環境（最大パフォーマンス）

```bash
OLLAMA_GPU_ENABLED=true
OLLAMA_GPU_MEMORY_FRACTION=0.9
OLLAMA_GPU_LAYERS=40
OLLAMA_PARALLEL_REQUESTS=4
```

## 🎉 まとめ

GPU 設定により、以下の改善が期待できます：

- **応答速度**: 2-5 倍高速化
- **並列処理**: 複数リクエストの同時処理
- **ストリーミング**: リアルタイム応答の向上
- **学習効率**: 学習システムの高速化

適切な設定により、快適な AI エージェント体験を実現できます。
