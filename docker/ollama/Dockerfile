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
CMD ["/bin/bash", "/app/pull_model.sh"]
