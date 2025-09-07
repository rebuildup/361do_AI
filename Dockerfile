# Self-Learning AI Agent Dockerfile (Legacy Streamlit)
# 自己学習AIエージェント用Dockerfile（レガシーStreamlit版）

# ベースイメージ: Python 3.11 + CUDA 12.1
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# メタデータ
LABEL maintainer="Self-Learning AI Agent Team"
LABEL description="Self-Learning AI Agent with RTX 4050 optimization (Legacy Streamlit)"
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
    python3 \
    python3-dev \
    python3-distutils \
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
    python3-pyqt5 \
    libblas-dev \
    liblapack-dev \
    libopencv-dev \
    python3-opencv \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python 3をデフォルトに設定
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# pipのアップグレード
RUN pip install --upgrade pip setuptools wheel

## Optional: if you need CUDA wheels, keep this, otherwise rely on requirements.txt only
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

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

# デフォルトコマンド（レガシーStreamlit）
CMD ["python", "main.py", "--ui", "streamlit"]
