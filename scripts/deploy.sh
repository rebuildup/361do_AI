#!/bin/bash

# Self-Learning AI Agent Deployment Script
# 自己学習AIエージェントデプロイメントスクリプト

set -e  # エラー時に停止

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

# 設定
PROJECT_NAME="self-learning-ai-agent"
DOCKER_COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"
BACKUP_DIR="./backups"
LOG_DIR="./logs"

# ヘルプ表示
show_help() {
    echo "Self-Learning AI Agent Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  deploy      - 本番環境にデプロイ"
    echo "  dev         - 開発環境にデプロイ"
    echo "  stop        - サービス停止"
    echo "  restart     - サービス再起動"
    echo "  logs        - ログ表示"
    echo "  status      - サービス状態確認"
    echo "  backup      - データバックアップ"
    echo "  restore     - データ復元"
    echo "  update      - サービス更新"
    echo "  clean       - 不要なリソース削除"
    echo "  health      - ヘルスチェック"
    echo "  help        - このヘルプを表示"
    echo ""
}

# 環境チェック
check_environment() {
    log_info "環境チェック開始..."
    
    # Docker確認
    if ! command -v docker &> /dev/null; then
        log_error "Dockerがインストールされていません"
        exit 1
    fi
    
    # Docker Compose確認
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Composeがインストールされていません"
        exit 1
    fi
    
    # NVIDIA Docker確認（GPU使用時）
    if command -v nvidia-smi &> /dev/null; then
        if ! docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
            log_warn "NVIDIA Dockerが正しく設定されていません"
        fi
    fi
    
    # 必要なファイル確認
    if [ ! -f "$DOCKER_COMPOSE_FILE" ]; then
        log_error "docker-compose.ymlが見つかりません"
        exit 1
    fi
    
    log_success "環境チェック完了"
}

# ディレクトリ作成
create_directories() {
    log_info "必要なディレクトリを作成中..."
    
    mkdir -p "$BACKUP_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "./data"
    mkdir -p "./models"
    mkdir -p "./config"
    
    log_success "ディレクトリ作成完了"
}

# 環境ファイル作成
create_env_file() {
    if [ ! -f "$ENV_FILE" ]; then
        log_info "環境ファイルを作成中..."
        
        cat > "$ENV_FILE" << EOF
# Self-Learning AI Agent Environment Configuration
# 自己学習AIエージェント環境設定

# アプリケーション設定
APP_NAME=Self-Learning AI Agent
APP_VERSION=1.0.0
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# データベース設定
DATABASE_URL=postgresql://ai_agent:ai_agent_password@postgres:5432/ai_agent
REDIS_URL=redis://redis:6379/0

# API設定
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_RELOAD=false

# Streamlit設定
STREAMLIT_HOST=0.0.0.0
STREAMLIT_PORT=8501

# 監視設定
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# GPU設定
CUDA_VISIBLE_DEVICES=0
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility

# セキュリティ設定
SECRET_KEY=your-secret-key-here
API_KEY=your-api-key-here
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8501"]

# メモリ設定
MAX_GPU_MEMORY_PERCENT=85
MAX_SYSTEM_MEMORY_PERCENT=80
BATCH_SIZE=16

# ログ設定
LOG_FORMAT=json
LOG_FILE_PATH=/app/logs/app.log
LOG_MAX_SIZE=100MB
LOG_BACKUP_COUNT=5
EOF
        
        log_success "環境ファイル作成完了"
        log_warn "環境ファイルを編集して適切な値を設定してください"
    fi
}

# 本番環境デプロイ
deploy_production() {
    log_info "本番環境デプロイ開始..."
    
    check_environment
    create_directories
    create_env_file
    
    # 既存のサービス停止
    log_info "既存のサービスを停止中..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" down
    
    # イメージビルド
    log_info "Dockerイメージをビルド中..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" build --no-cache
    
    # サービス起動
    log_info "サービスを起動中..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    
    # ヘルスチェック
    log_info "ヘルスチェック実行中..."
    sleep 30
    check_health
    
    log_success "本番環境デプロイ完了"
}

# 開発環境デプロイ
deploy_development() {
    log_info "開発環境デプロイ開始..."
    
    check_environment
    create_directories
    create_env_file
    
    # 開発用設定で起動
    log_info "開発環境でサービスを起動中..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d redis postgres
    
    # アプリケーションを直接実行
    log_info "アプリケーションを開発モードで起動中..."
    export ENVIRONMENT=development
    export DEBUG=true
    export API_RELOAD=true
    
    python main.py &
    
    log_success "開発環境デプロイ完了"
}

# サービス停止
stop_services() {
    log_info "サービス停止中..."
    
    docker-compose -f "$DOCKER_COMPOSE_FILE" down
    
    log_success "サービス停止完了"
}

# サービス再起動
restart_services() {
    log_info "サービス再起動中..."
    
    docker-compose -f "$DOCKER_COMPOSE_FILE" restart
    
    log_success "サービス再起動完了"
}

# ログ表示
show_logs() {
    log_info "ログ表示中..."
    
    docker-compose -f "$DOCKER_COMPOSE_FILE" logs -f --tail=100
}

# サービス状態確認
show_status() {
    log_info "サービス状態確認中..."
    
    docker-compose -f "$DOCKER_COMPOSE_FILE" ps
    
    echo ""
    log_info "リソース使用量:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
}

# データバックアップ
backup_data() {
    log_info "データバックアップ開始..."
    
    BACKUP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_PATH="$BACKUP_DIR/backup_$BACKUP_TIMESTAMP"
    
    mkdir -p "$BACKUP_PATH"
    
    # データベースバックアップ
    log_info "データベースをバックアップ中..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T postgres pg_dump -U ai_agent ai_agent > "$BACKUP_PATH/database.sql"
    
    # データディレクトリバックアップ
    log_info "データディレクトリをバックアップ中..."
    cp -r ./data "$BACKUP_PATH/"
    cp -r ./models "$BACKUP_PATH/"
    
    # 設定ファイルバックアップ
    log_info "設定ファイルをバックアップ中..."
    cp -r ./config "$BACKUP_PATH/"
    cp "$ENV_FILE" "$BACKUP_PATH/"
    
    # アーカイブ作成
    log_info "バックアップアーカイブを作成中..."
    tar -czf "$BACKUP_DIR/backup_$BACKUP_TIMESTAMP.tar.gz" -C "$BACKUP_DIR" "backup_$BACKUP_TIMESTAMP"
    rm -rf "$BACKUP_PATH"
    
    log_success "データバックアップ完了: $BACKUP_DIR/backup_$BACKUP_TIMESTAMP.tar.gz"
}

# データ復元
restore_data() {
    if [ -z "$1" ]; then
        log_error "復元するバックアップファイルを指定してください"
        echo "Usage: $0 restore <backup_file.tar.gz>"
        exit 1
    fi
    
    BACKUP_FILE="$1"
    
    if [ ! -f "$BACKUP_FILE" ]; then
        log_error "バックアップファイルが見つかりません: $BACKUP_FILE"
        exit 1
    fi
    
    log_info "データ復元開始: $BACKUP_FILE"
    
    # サービス停止
    stop_services
    
    # バックアップ展開
    BACKUP_DIR_TEMP="./backup_temp"
    mkdir -p "$BACKUP_DIR_TEMP"
    tar -xzf "$BACKUP_FILE" -C "$BACKUP_DIR_TEMP"
    
    # データ復元
    log_info "データを復元中..."
    cp -r "$BACKUP_DIR_TEMP"/*/data ./
    cp -r "$BACKUP_DIR_TEMP"/*/models ./
    cp -r "$BACKUP_DIR_TEMP"/*/config ./
    
    # 一時ディレクトリ削除
    rm -rf "$BACKUP_DIR_TEMP"
    
    # サービス再起動
    deploy_production
    
    log_success "データ復元完了"
}

# サービス更新
update_services() {
    log_info "サービス更新開始..."
    
    # 最新イメージ取得
    log_info "最新イメージを取得中..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" pull
    
    # サービス再起動
    log_info "サービスを再起動中..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    
    # 古いイメージ削除
    log_info "古いイメージを削除中..."
    docker image prune -f
    
    log_success "サービス更新完了"
}

# 不要なリソース削除
clean_resources() {
    log_info "不要なリソースを削除中..."
    
    # 停止中のコンテナ削除
    docker container prune -f
    
    # 未使用のイメージ削除
    docker image prune -f
    
    # 未使用のボリューム削除
    docker volume prune -f
    
    # 未使用のネットワーク削除
    docker network prune -f
    
    log_success "不要なリソース削除完了"
}

# ヘルスチェック
check_health() {
    log_info "ヘルスチェック実行中..."
    
    # 各サービスのヘルスチェック
    services=("ai-agent:8000" "streamlit:8501" "prometheus:9090" "grafana:3000")
    
    for service in "${services[@]}"; do
        IFS=':' read -r name port <<< "$service"
        
        if curl -f "http://localhost:$port/health" &> /dev/null; then
            log_success "$name は正常に動作しています"
        else
            log_warn "$name のヘルスチェックに失敗しました"
        fi
    done
    
    # リソース使用量確認
    log_info "リソース使用量:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
}

# メイン処理
main() {
    case "${1:-help}" in
        deploy)
            deploy_production
            ;;
        dev)
            deploy_development
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        logs)
            show_logs
            ;;
        status)
            show_status
            ;;
        backup)
            backup_data
            ;;
        restore)
            restore_data "$2"
            ;;
        update)
            update_services
            ;;
        clean)
            clean_resources
            ;;
        health)
            check_health
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "不明なコマンド: $1"
            show_help
            exit 1
            ;;
    esac
}

# スクリプト実行
main "$@"
