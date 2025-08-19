FROM ghcr.io/open-webui/open-webui:main

# カスタム設定
ENV OLLAMA_BASE_URL=http://ollama:11434
ENV WEBUI_SECRET_KEY=your-secret-key-here
ENV DEFAULT_MODELS=ai-agent-studio

# エージェント統合設定
ENV ENABLE_COMMUNITY_SHARING=false
ENV ENABLE_ADMIN_EXPORT=true
ENV ENABLE_OLLAMA_API=true
ENV ENABLE_OPENAI_API=false

# カスタムテーマ・設定ファイル
COPY custom_config.json /app/backend/data/config.json

# 関数/パイプラインを配置（Open WebUI が自動読み込み）
COPY functions /app/backend/data/functions
COPY pipelines /app/pipelines

# 日本語対応
ENV WEBUI_NAME="AI Agent Studio"
ENV DEFAULT_LOCALE=ja

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1
