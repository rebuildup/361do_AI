#!/bin/bash
# Health Check Script for Integrated Application
# 統合アプリケーション用ヘルスチェックスクリプト

set -e

# Configuration
FASTAPI_URL="http://localhost:8000"
NGINX_URL="http://localhost:80"
CHECK_INTERVAL=30
MAX_RETRIES=3

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging functions
log() {
    echo -e "${GREEN}[HEALTH] $(date +'%Y-%m-%d %H:%M:%S') $1${NC}"
}

warn() {
    echo -e "${YELLOW}[HEALTH] $(date +'%Y-%m-%d %H:%M:%S') WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[HEALTH] $(date +'%Y-%m-%d %H:%M:%S') ERROR: $1${NC}"
}

# Function to check HTTP endpoint
check_http_endpoint() {
    local url=$1
    local name=$2
    local timeout=${3:-10}
    
    if curl -f -s --max-time $timeout "$url" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to check FastAPI backend
check_fastapi() {
    local retries=0
    
    while [ $retries -lt $MAX_RETRIES ]; do
        if check_http_endpoint "$FASTAPI_URL/v1/health" "FastAPI" 10; then
            log "FastAPI backend is healthy"
            return 0
        fi
        
        retries=$((retries + 1))
        if [ $retries -lt $MAX_RETRIES ]; then
            warn "FastAPI health check failed, retrying ($retries/$MAX_RETRIES)..."
            sleep 2
        fi
    done
    
    error "FastAPI backend is unhealthy after $MAX_RETRIES attempts"
    return 1
}

# Function to check Nginx frontend
check_nginx() {
    local retries=0
    
    while [ $retries -lt $MAX_RETRIES ]; do
        if check_http_endpoint "$NGINX_URL/health" "Nginx" 5; then
            log "Nginx frontend is healthy"
            return 0
        fi
        
        retries=$((retries + 1))
        if [ $retries -lt $MAX_RETRIES ]; then
            warn "Nginx health check failed, retrying ($retries/$MAX_RETRIES)..."
            sleep 2
        fi
    done
    
    error "Nginx frontend is unhealthy after $MAX_RETRIES attempts"
    return 1
}

# Function to check React frontend
check_react_frontend() {
    if check_http_endpoint "$NGINX_URL/" "React Frontend" 5; then
        log "React frontend is accessible"
        return 0
    else
        error "React frontend is not accessible"
        return 1
    fi
}

# Function to check API integration
check_api_integration() {
    if check_http_endpoint "$NGINX_URL/v1/health" "API Integration" 10; then
        log "API integration is working"
        return 0
    else
        error "API integration is not working"
        return 1
    fi
}

# Function to check system resources
check_system_resources() {
    # Check memory usage
    local memory_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    local memory_threshold=90.0
    
    if (( $(echo "$memory_usage > $memory_threshold" | bc -l) )); then
        warn "High memory usage: ${memory_usage}%"
    else
        log "Memory usage: ${memory_usage}%"
    fi
    
    # Check disk usage
    local disk_usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    local disk_threshold=85
    
    if [ "$disk_usage" -gt "$disk_threshold" ]; then
        warn "High disk usage: ${disk_usage}%"
    else
        log "Disk usage: ${disk_usage}%"
    fi
    
    # Check if GPU is available and being used
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_usage=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [ -n "$gpu_usage" ]; then
            log "GPU usage: ${gpu_usage}%"
        fi
    fi
}

# Function to perform comprehensive health check
comprehensive_health_check() {
    log "Starting comprehensive health check..."
    
    local overall_status=0
    
    # Check FastAPI backend
    if ! check_fastapi; then
        overall_status=1
    fi
    
    # Check Nginx frontend
    if ! check_nginx; then
        overall_status=1
    fi
    
    # Check React frontend accessibility
    if ! check_react_frontend; then
        overall_status=1
    fi
    
    # Check API integration through Nginx
    if ! check_api_integration; then
        overall_status=1
    fi
    
    # Check system resources
    check_system_resources
    
    if [ $overall_status -eq 0 ]; then
        log "All health checks passed ✓"
    else
        error "Some health checks failed ✗"
    fi
    
    return $overall_status
}

# Function to run continuous monitoring
continuous_monitoring() {
    log "Starting continuous health monitoring (interval: ${CHECK_INTERVAL}s)"
    
    while true; do
        comprehensive_health_check
        sleep $CHECK_INTERVAL
    done
}

# Main function
main() {
    local mode=${1:-"once"}
    
    case $mode in
        "once")
            comprehensive_health_check
            ;;
        "continuous")
            continuous_monitoring
            ;;
        "fastapi")
            check_fastapi
            ;;
        "nginx")
            check_nginx
            ;;
        "frontend")
            check_react_frontend
            ;;
        "api")
            check_api_integration
            ;;
        *)
            error "Unknown mode: $mode"
            error "Available modes: once, continuous, fastapi, nginx, frontend, api"
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"