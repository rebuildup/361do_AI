# Self-Learning AI Agent Dockerfile
# 自己学習AIエージェント用Dockerfile

# ベースイメージ: Python 3.11 + CUDA 12.1
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# メタデータ
LABEL maintainer="Self-Learning AI Agent Team"
LABEL description="Self-Learning AI Agent with RTX 4050 optimization"
LABEL version="1.0.0"

# 環境変数設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# 作業ディレクトリ設定
WORKDIR /app

# システムパッケージの更新とインストール
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    libqtgui4 \
    libqtwebkit4 \
    libqt4-test \
    python3-pyqt5 \
    libblas-dev \
    liblapack-dev \
    libopencv-dev \
    python3-opencv \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python 3.11をデフォルトに設定
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# pipのアップグレード
RUN pip install --upgrade pip setuptools wheel

# PyTorchとCUDA関連パッケージのインストール
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 基本的なPythonパッケージのインストール
RUN pip install \
    numpy==1.24.3 \
    pandas==2.0.3 \
    scikit-learn==1.3.0 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    plotly==5.15.0 \
    jupyter==1.0.0 \
    notebook==6.5.4 \
    ipykernel==6.25.0

# 機械学習関連パッケージ
RUN pip install \
    transformers==4.33.2 \
    datasets==2.14.5 \
    accelerate==0.21.0 \
    bitsandbytes==0.41.1 \
    peft==0.5.0 \
    trl==0.7.1 \
    sentence-transformers==2.2.2 \
    chromadb==0.4.10 \
    faiss-cpu==1.7.4

# Webフレームワーク
RUN pip install \
    fastapi==0.103.1 \
    uvicorn[standard]==0.23.2 \
    streamlit==1.26.0 \
    gradio==3.39.0

# データベースとストレージ
RUN pip install \
    sqlalchemy==2.0.20 \
    alembic==1.11.3 \
    redis==4.6.0 \
    pymongo==4.4.1

# 監視とログ
RUN pip install \
    prometheus-client==0.17.1 \
    psutil==5.9.5 \
    GPUtil==1.4.0 \
    nvidia-ml-py3==7.352.0

# テストと開発ツール
RUN pip install \
    pytest==7.4.2 \
    pytest-asyncio==0.21.1 \
    pytest-cov==4.1.0 \
    black==23.7.0 \
    flake8==6.0.0 \
    mypy==1.5.1 \
    pre-commit==3.3.3

# アプリケーションコードのコピー
COPY . /app/

# 依存関係のインストール
RUN pip install -r requirements.txt

# 権限設定
RUN chmod +x scripts/*.sh

# ポート設定
EXPOSE 8000 8501 9090

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

# 起動スクリプト
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 非rootユーザーの作成
RUN useradd -m -u 1000 agent && \
    chown -R agent:agent /app
USER agent

# エントリーポイント
ENTRYPOINT ["/entrypoint.sh"]

# デフォルトコマンド
CMD ["python", "main.py"]
