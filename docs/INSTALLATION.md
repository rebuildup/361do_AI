# インストールガイド

## システム要件の確認

### ハードウェア要件

- **GPU**: NVIDIA RTX 4050 (6GB VRAM) 以上
- **RAM**: 32GB 推奨（最小 16GB）
- **CPU**: Intel i7-13700H 相当以上
- **ストレージ**: SSD 100GB 以上の空き容量

### ソフトウェア要件

- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 12+
- **Python**: 3.11+
- **CUDA**: 12.0+ (NVIDIA GPU 使用時)
- **Git**: 最新版

## インストール手順

### 1. リポジトリのクローン

```bash
git clone https://github.com/your-repo/advanced-self-learning-agent.git
cd advanced-self-learning-agent
```

### 2. Python 環境の準備

```bash
# Python仮想環境の作成
python -m venv .venv

# 仮想環境の有効化
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 3. 依存関係のインストール

```bash
# 基本依存関係
pip install -r requirements.txt

# 開発用依存関係（オプション）
pip install -r requirements_dev.txt
```

### 4. Ollama のセットアップ

```bash
# Ollamaのインストール（公式サイトから）
# https://ollama.ai/

# 必要なモデルのダウンロード
ollama pull deepseek-r1:7b
ollama pull qwen2.5:7b-instruct-q4_k_m
ollama pull qwen2:1.5b-instruct-q4_k_m
```

### 5. 設定ファイルの準備

```bash
# 設定ファイルのコピー
cp config/system.yaml.example config/system.yaml
cp config/.env.example config/.env

# 必要に応じて設定を編集
```

### 6. データベースの初期化

```bash
# データベースディレクトリの作成
mkdir -p data

# 初期化スクリプトの実行
python -m src.advanced_agent.setup.init_database
```

## 設定の確認

### システムチェック

```bash
# システム要件の確認
python -m src.advanced_agent.setup.system_check

# GPU設定の確認
python -m src.advanced_agent.setup.gpu_check
```

### 動作テスト

```bash
# 基本動作テスト
python -m src.advanced_agent.setup.test_installation

# 推論エンジンテスト
python -m src.advanced_agent.setup.test_reasoning

# 記憶システムテスト
python -m src.advanced_agent.setup.test_memory
```

## トラブルシューティング

### よくある問題

#### CUDA 関連エラー

```bash
# CUDAバージョンの確認
nvidia-smi
nvcc --version

# PyTorchの再インストール
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Ollama 接続エラー

```bash
# Ollamaサービスの確認
ollama list
ollama ps

# サービスの再起動
ollama serve
```

#### メモリ不足エラー

```bash
# 設定ファイルでVRAM制限を調整
# config/system.yaml
gpu:
  max_vram_gb: 4.0  # 6GBから4GBに削減
```

## 次のステップ

インストールが完了したら：

1. [設定ガイド](CONFIGURATION.md) - 詳細な設定方法
2. [使用方法](USAGE.md) - 基本的な使用方法
3. [API リファレンス](API_REFERENCE.md) - API の使用方法
