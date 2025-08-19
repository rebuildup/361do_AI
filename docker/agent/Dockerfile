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
