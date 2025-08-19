# Docker 環境セットアップガイド

## 1. 前提条件

### 1.1 システム要件確認

- **OS**: Windows 11 ✅
- **CPU**: Intel i7-13700H ✅
- **RAM**: 32GB ✅
- **GPU**: RTX 4050 Laptop ✅
- **Docker Desktop**: 最新版が必要

### 1.2 必要なソフトウェア

- Docker Desktop for Windows
- Git
- VSCode（推奨）
- WSL2（Windows Subsystem for Linux）

## 2. Docker Desktop インストール

### 2.1 Docker Desktop ダウンロード・インストール

```powershell
# 1. Docker Desktop公式サイトからダウンロード
# https://www.docker.com/products/docker-desktop/

# 2. インストール後、WSL2バックエンドを有効化
# 3. 再起動
```

### 2.2 GPU サポート有効化（RTX 4050 用）

```powershell
# NVIDIA Container Toolkit のインストール
# 1. NVIDIA ドライバーが最新であることを確認
# 2. Docker Desktop の設定で GPU サポートを有効化
```

### 2.3 Docker 動作確認

```bash
# Docker バージョン確認
docker --version
docker-compose --version

# GPU サポート確認
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

## 3. プロジェクト構造作成

### 3.1 ディレクトリ作成

```bash
cd /d/work/008_LLM

# プロジェクト構造作成
mkdir -p docker/ollama
mkdir -p docker/webui
mkdir -p docker/agent
mkdir -p docker/nginx
mkdir -p src/agent/core
mkdir -p src/agent/self_tuning
mkdir -p src/agent/web_design
mkdir -p src/agent/tools
mkdir -p src/data/conversations
mkdir -p src/data/knowledge_base
mkdir -p src/data/models
```

### 3.2 Docker 設定ファイル作成

#### docker-compose.yml

```yaml
version: "3.8"

services:
  ollama:
    build:
      context: ./docker/ollama
      dockerfile: Dockerfile
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
      - ./src/data/models:/app/models
    environment:
      - OLLAMA_MODELS=/root/.ollama/models
      - OLLAMA_HOST=0.0.0.0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  webui:
    build:
      context: ./docker/webui
      dockerfile: Dockerfile
    ports:
      - "3000:8080"
    depends_on:
      - ollama
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - WEBUI_SECRET_KEY=your-secret-key-here
    volumes:
      - webui_data:/app/backend/data
    restart: unless-stopped

  agent:
    build:
      context: ./docker/agent
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./src:/app/src
      - ./src/data:/app/data
    depends_on:
      - ollama
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - PYTHONPATH=/app/src
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - webui
      - agent
    restart: unless-stopped

volumes:
  ollama_data:
  webui_data:

networks:
  default:
    driver: bridge
```

#### docker/ollama/Dockerfile

```dockerfile
FROM ollama/ollama:latest

# GPU サポートのための設定
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# 作業ディレクトリ設定
WORKDIR /app

# qwen2:7b-instruct モデルの事前ダウンロード用スクリプト
COPY pull_model.sh /app/
RUN chmod +x /app/pull_model.sh

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:11434/api/tags || exit 1

EXPOSE 11434

# モデルダウンロードとサービス開始
CMD ["/bin/bash", "-c", "/app/pull_model.sh && ollama serve"]
```

#### docker/ollama/pull_model.sh

```bash
#!/bin/bash

# OLLAMA サービス開始
ollama serve &
OLLAMA_PID=$!

# サービスが開始されるまで待機
sleep 10

# qwen2:7b-instruct モデルをプル
echo "Downloading qwen2:7b-instruct model..."
ollama pull qwen2:7b-instruct

# モデルダウンロード完了確認
if ollama list | grep -q "qwen2:7b-instruct"; then
    echo "qwen2:7b-instruct model successfully downloaded!"
else
    echo "Failed to download qwen2:7b-instruct model"
    exit 1
fi

# バックグラウンドプロセスを前面に移動
wait $OLLAMA_PID
```

#### docker/webui/Dockerfile

```dockerfile
FROM ghcr.io/open-webui/open-webui:main

# カスタム設定
ENV OLLAMA_BASE_URL=http://ollama:11434
ENV WEBUI_SECRET_KEY=your-secret-key-here
ENV DEFAULT_MODELS=qwen2:7b-instruct

# カスタムテーマ・設定ファイル
COPY custom_config.json /app/backend/data/config.json

# 日本語対応
ENV WEBUI_NAME="AI Agent Studio"
ENV DEFAULT_LOCALE=ja

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1
```

#### docker/webui/custom_config.json

```json
{
  "ui": {
    "title": "AI Agent Studio",
    "default_locale": "ja",
    "theme": "dark"
  },
  "ollama": {
    "base_url": "http://ollama:11434",
    "default_model": "qwen2:7b-instruct"
  },
  "features": {
    "enable_signup": false,
    "enable_web_search": true,
    "enable_image_generation": false
  }
}
```

#### docker/agent/Dockerfile

```dockerfile
FROM python:3.11-slim

# システム依存関係
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ
WORKDIR /app

# Python 依存関係
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコード
COPY . .

# 環境変数
ENV PYTHONPATH=/app/src
ENV OLLAMA_BASE_URL=http://ollama:11434

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# アプリケーション開始
CMD ["python", "src/agent/main.py"]
```

#### docker/agent/requirements.txt

```
# コア機能
ollama-python==0.1.7
langchain==0.1.0
langchain-community==0.0.13
fastapi==0.104.1
uvicorn==0.24.0

# Web関連
requests==2.31.0
beautifulsoup4==4.12.2
selenium==4.15.0
html5lib==1.1
aiohttp==3.9.1

# データベース
sqlalchemy==2.0.23
alembic==1.13.1

# 検索・スクレイピング
duckduckgo-search==3.9.6
newspaper3k==0.2.8

# Web デザイン
jinja2==3.1.2
cssutils==2.7.1
premailer==3.10.0

# ユーティリティ
pydantic==2.5.0
python-multipart==0.0.6
python-dotenv==1.0.0
loguru==0.7.2

# 開発・テスト
pytest==7.4.3
pytest-asyncio==0.21.1
black==23.11.0
isort==5.12.0
```

#### docker/nginx/nginx.conf

```nginx
events {
    worker_connections 1024;
}

http {
    upstream webui {
        server webui:8080;
    }

    upstream agent {
        server agent:8000;
    }

    server {
        listen 80;
        server_name localhost;

        # Open WebUI (メインインターフェース)
        location / {
            proxy_pass http://webui;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # WebSocket サポート
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        # Agent API
        location /api/agent/ {
            proxy_pass http://agent/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # OLLAMA API (直接アクセス用)
        location /api/ollama/ {
            proxy_pass http://ollama:11434/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

## 4. 環境構築手順

### 4.1 プロジェクト初期化

```bash
cd /d/work/008_LLM

# Docker イメージビルド
docker-compose build

# コンテナ起動（初回は時間がかかります）
docker-compose up -d

# ログ確認
docker-compose logs -f
```

### 4.2 モデルダウンロード確認

```bash
# OLLAMA コンテナに接続
docker exec -it 008_llm_ollama_1 bash

# モデル確認
ollama list

# qwen2:7b-instruct が表示されることを確認
```

### 4.3 動作確認

```bash
# 各サービスの動作確認
curl http://localhost:11434/api/tags  # OLLAMA
curl http://localhost:3000            # Open WebUI
curl http://localhost:8000/health     # Agent API
curl http://localhost:80              # Nginx (メインアクセス)
```

## 5. 開発環境設定

### 5.1 VSCode 設定

```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "docker.attachShellCommand.linuxContainer": "/bin/bash"
}
```

### 5.2 開発用コマンド

```bash
# 開発モード起動
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# ログ監視
docker-compose logs -f agent

# コンテナ再ビルド
docker-compose build --no-cache agent

# データベース初期化
docker exec -it 008_llm_agent_1 python src/agent/init_db.py
```

## 6. トラブルシューティング

### 6.1 よくある問題

#### GPU 認識されない

```bash
# NVIDIA ドライバー確認
nvidia-smi

# Docker GPU サポート確認
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

#### メモリ不足

```bash
# Docker Desktop のメモリ設定を16GB以上に変更
# Settings > Resources > Advanced > Memory
```

#### ポート競合

```bash
# 使用中ポート確認
netstat -an | findstr :11434
netstat -an | findstr :3000
netstat -an | findstr :8000

# 競合している場合は docker-compose.yml のポート番号を変更
```

### 6.2 デバッグコマンド

```bash
# コンテナ内部確認
docker exec -it 008_llm_ollama_1 bash
docker exec -it 008_llm_agent_1 bash

# リソース使用量確認
docker stats

# ログ詳細確認
docker-compose logs --tail=100 ollama
docker-compose logs --tail=100 agent
```

## 7. 次のステップ

1. **環境動作確認**: すべてのサービスが正常に起動することを確認
2. **基本エージェント実装**: `src/agent/main.py` の実装開始
3. **Open WebUI カスタマイズ**: エージェント機能との連携設定
4. **自己チューニング機能実装**: 会話履歴管理システムの実装

---

**作成日**: 2024 年 12 月
**対象環境**: Intel i7-13700H, 32GB RAM, RTX 4050 Laptop
**前提**: Docker Desktop for Windows, WSL2
