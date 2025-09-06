#!/bin/bash

# Self-Learning AI Agent Docker Entrypoint
# 自己学習AIエージェント用Dockerエントリーポイント

set -e

# 色付きログ関数
log_info() {
    echo -e "\033[32m[INFO]\033[0m $1"
}

log_warn() {
    echo -e "\033[33m[WARN]\033[0m $1"
}

log_error() {
    echo -e "\033[31m[ERROR]\033[0m $1"
}

log_success() {
    echo -e "\033[32m[SUCCESS]\033[0m $1"
}

# 環境変数のデフォルト値設定
export PYTHONPATH="${PYTHONPATH:-/app}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export ENVIRONMENT="${ENVIRONMENT:-production}"
export API_HOST="${API_HOST:-0.0.0.0}"
export API_PORT="${API_PORT:-8000}"
export STREAMLIT_HOST="${STREAMLIT_HOST:-0.0.0.0}"
export STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"

# 初期化処理
initialize() {
    log_info "Self-Learning AI Agent 初期化開始..."
    
    # 必要なディレクトリ作成
    mkdir -p /app/data
    mkdir -p /app/logs
    mkdir -p /app/models
    mkdir -p /app/config
    
    # 権限設定
    chown -R agent:agent /app
    
    # データベース接続確認
    if [ "$ENVIRONMENT" = "production" ]; then
        wait_for_database
    fi
    
    # GPU確認
    check_gpu_availability
    
    log_success "初期化完了"
}

# データベース接続待機
wait_for_database() {
    log_info "データベース接続を待機中..."
    
    # PostgreSQL接続確認
    if [ -n "$DATABASE_URL" ]; then
        until pg_isready -h postgres -p 5432 -U ai_agent; do
            log_info "PostgreSQL接続待機中..."
            sleep 2
        done
        log_success "PostgreSQL接続確認完了"
    fi
    
    # Redis接続確認
    if [ -n "$REDIS_URL" ]; then
        until redis-cli -h redis -p 6379 ping; do
            log_info "Redis接続待機中..."
            sleep 2
        done
        log_success "Redis接続確認完了"
    fi
}

# GPU可用性確認
check_gpu_availability() {
    log_info "GPU可用性確認中..."
    
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            log_success "GPU利用可能"
            nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
        else
            log_warn "GPU利用不可"
        fi
    else
        log_warn "NVIDIA Driver未インストール"
    fi
}

# アプリケーション起動
start_application() {
    log_info "アプリケーション起動中..."
    
    case "${1:-api}" in
        api)
            start_api_server
            ;;
        streamlit)
            start_streamlit_server
            ;;
        worker)
            start_worker_process
            ;;
        all)
            start_all_services
            ;;
        *)
            log_error "不明なサービス: $1"
            exit 1
            ;;
    esac
}

# APIサーバー起動
start_api_server() {
    log_info "FastAPIサーバー起動中..."
    
    exec uvicorn src.advanced_agent.interfaces.fastapi_gateway:app \
        --host "$API_HOST" \
        --port "$API_PORT" \
        --workers 1 \
        --log-level "$LOG_LEVEL" \
        --access-log \
        --use-colors
}

# Streamlitサーバー起動
start_streamlit_server() {
    log_info "Streamlitサーバー起動中..."
    
    exec streamlit run src/advanced_agent/interfaces/streamlit_ui.py \
        --server.address "$STREAMLIT_HOST" \
        --server.port "$STREAMLIT_PORT" \
        --server.headless true \
        --server.enableCORS false \
        --server.enableXsrfProtection false
}

# ワーカープロセス起動
start_worker_process() {
    log_info "ワーカープロセス起動中..."
    
    exec python -m src.advanced_agent.core.worker
}

# 全サービス起動
start_all_services() {
    log_info "全サービス起動中..."
    
    # バックグラウンドでAPIサーバー起動
    start_api_server &
    API_PID=$!
    
    # バックグラウンドでStreamlitサーバー起動
    start_streamlit_server &
    STREAMLIT_PID=$!
    
    # シグナルハンドリング
    trap 'kill $API_PID $STREAMLIT_PID; exit' INT TERM
    
    # プロセス監視
    while true; do
        if ! kill -0 $API_PID 2>/dev/null; then
            log_error "APIサーバーが停止しました"
            kill $STREAMLIT_PID 2>/dev/null
            exit 1
        fi
        
        if ! kill -0 $STREAMLIT_PID 2>/dev/null; then
            log_error "Streamlitサーバーが停止しました"
            kill $API_PID 2>/dev/null
            exit 1
        fi
        
        sleep 5
    done
}

# ヘルスチェック
health_check() {
    log_info "ヘルスチェック実行中..."
    
    # APIヘルスチェック
    if curl -f "http://localhost:$API_PORT/v1/health" &> /dev/null; then
        log_success "APIサーバー正常"
    else
        log_error "APIサーバー異常"
        exit 1
    fi
    
    # Streamlitヘルスチェック
    if curl -f "http://localhost:$STREAMLIT_PORT/_stcore/health" &> /dev/null; then
        log_success "Streamlitサーバー正常"
    else
        log_warn "Streamlitサーバー異常"
    fi
}

# メイン処理
main() {
    log_info "Self-Learning AI Agent コンテナ起動"
    
    # 初期化
    initialize
    
    # コマンド実行
    case "${1:-api}" in
        health)
            health_check
            ;;
        shell)
            exec /bin/bash
            ;;
        *)
            start_application "$@"
            ;;
    esac
}

# シグナルハンドリング
trap 'log_info "シャットダウン中..."; exit 0' INT TERM

# メイン処理実行
main "$@"
