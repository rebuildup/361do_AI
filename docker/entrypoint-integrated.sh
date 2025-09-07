#!/bin/bash
# Integrated Entrypoint Script for React + FastAPI Application
# React + FastAPI 統合アプリケーション用エントリーポイントスクリプト

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Function to wait for service
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local timeout=${4:-30}
    
    log "Waiting for $service_name at $host:$port..."
    
    for i in $(seq 1 $timeout); do
        if nc -z $host $port 2>/dev/null; then
            log "$service_name is ready!"
            return 0
        fi
        sleep 1
    done
    
    error "$service_name failed to start within $timeout seconds"
    return 1
}

# Function to check if frontend build exists
check_frontend_build() {
    if [ ! -d "/app/static" ] || [ ! -f "/app/static/index.html" ]; then
        error "Frontend build not found in /app/static"
        error "Make sure the React frontend was built correctly during Docker build"
        return 1
    fi
    
    log "Frontend build found in /app/static"
    return 0
}

# Function to setup directories and permissions
setup_directories() {
    log "Setting up directories and permissions..."
    
    # Create necessary directories
    mkdir -p /app/data /app/logs /app/models /app/config
    mkdir -p /var/log/supervisor /var/log/nginx
    
    # Set permissions for agent user
    chown -R agent:agent /app/data /app/logs /app/models /app/config 2>/dev/null || true
    
    # Create log files
    touch /var/log/supervisor/supervisord.log
    touch /var/log/supervisor/fastapi.log
    touch /var/log/supervisor/nginx.log
    
    log "Directories and permissions set up successfully"
}

# Function to validate configuration
validate_config() {
    log "Validating configuration..."
    
    # Check if required files exist
    local required_files=(
        "/app/main.py"
        "/app/requirements.txt"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            error "Required file not found: $file"
            return 1
        fi
    done
    
    # Check if Python dependencies are installed
    if ! python -c "import fastapi, uvicorn" 2>/dev/null; then
        error "Required Python dependencies not found"
        return 1
    fi
    
    log "Configuration validation completed"
    return 0
}

# Function to setup Nginx configuration
setup_nginx() {
    log "Setting up Nginx configuration..."
    
    # The nginx configuration is already copied during Docker build
    # Just test the configuration
    if nginx -t 2>/dev/null; then
        log "Nginx configuration is valid"
    else
        error "Nginx configuration is invalid"
        nginx -t  # Show the error
        return 1
    fi
    
    return 0
}

# Function to start services based on mode
start_services() {
    local mode=${1:-"integrated"}
    
    case $mode in
        "integrated")
            log "Starting integrated React + FastAPI application..."
            
            # Setup directories and permissions
            setup_directories
            
            # Validate configuration
            validate_config
            
            # Check frontend build
            check_frontend_build
            
            # Setup Nginx
            setup_nginx
            
            # Start supervisor to manage all services
            log "Starting Supervisor to manage services..."
            exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
            ;;
            
        "fastapi-only")
            log "Starting FastAPI backend only..."
            validate_config
            exec python main.py --ui fastapi --host 0.0.0.0 --port 8000
            ;;
            
        "streamlit-only")
            log "Starting Streamlit UI only..."
            validate_config
            exec python main.py --ui streamlit --host 0.0.0.0 --port 8501
            ;;
            
        "development")
            log "Starting in development mode..."
            validate_config
            
            # Start FastAPI with hot reload
            exec uvicorn src.advanced_agent.interfaces.fastapi_app:create_app --host 0.0.0.0 --port 8000 --reload
            ;;
            
        "test")
            log "Running tests..."
            validate_config
            
            # Run test suite
            exec python -m pytest tests/ -v
            ;;
            
        *)
            error "Unknown mode: $mode"
            error "Available modes: integrated, fastapi-only, streamlit-only, development, test"
            exit 1
            ;;
    esac
}

# Function to handle shutdown signals
cleanup() {
    log "Received shutdown signal, cleaning up..."
    
    # Stop supervisor and all managed processes
    if [ -f /tmp/supervisor.sock ]; then
        supervisorctl -s unix:///tmp/supervisor.sock shutdown 2>/dev/null || true
    fi
    
    # Stop nginx if running
    nginx -s quit 2>/dev/null || true
    
    log "Cleanup completed"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT SIGQUIT

# Main execution
main() {
    log "Starting AI Agent with React Frontend Integration"
    log "Container: $(hostname)"
    log "User: $(whoami)"
    log "Working Directory: $(pwd)"
    log "Python Version: $(python --version)"
    
    # Parse command line arguments
    local mode=${1:-"integrated"}
    
    # Environment validation
    if [ -z "$PYTHONPATH" ]; then
        export PYTHONPATH="/app"
        log "Set PYTHONPATH to /app"
    fi
    
    # GPU availability check
    if command -v nvidia-smi >/dev/null 2>&1; then
        log "GPU Information:"
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null || warn "Could not query GPU information"
    else
        warn "NVIDIA GPU not available or nvidia-smi not found"
    fi
    
    # Start services
    start_services "$mode"
}

# Execute main function with all arguments
main "$@"