#!/bin/bash
# Integration Test Script
# 統合テストスクリプト

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

log "Starting integration test for React + FastAPI deployment..."

# Check if we're in the right directory
if [ ! -f "frontend/package.json" ] || [ ! -f "Dockerfile.integrated" ]; then
    error "This script must be run from the project root directory"
    exit 1
fi

# Test 1: Frontend build
log "Testing frontend build..."
cd frontend

if [ ! -d "node_modules" ]; then
    log "Installing frontend dependencies..."
    npm install
fi

log "Building React frontend..."
if npm run build; then
    log "✅ Frontend build successful"
else
    error "❌ Frontend build failed"
    exit 1
fi

# Check build output
if [ ! -d "dist" ] || [ ! -f "dist/index.html" ]; then
    error "❌ Frontend build output missing"
    exit 1
fi

log "✅ Frontend build output verified"

# Calculate build size
BUILD_SIZE=$(du -sh dist 2>/dev/null | cut -f1 || echo "Unknown")
log "Frontend build size: $BUILD_SIZE"

cd ..

# Test 2: Backend dependencies
log "Testing backend dependencies..."

if [ ! -f "requirements.txt" ]; then
    error "❌ requirements.txt not found"
    exit 1
fi

log "✅ Backend requirements file found"

# Test 3: Docker configuration files
log "Testing Docker configuration..."

DOCKER_FILES=(
    "Dockerfile.integrated"
    "docker-compose.yml"
    "docker/nginx/integrated.conf"
    "docker/supervisor/supervisord.conf"
    "docker/entrypoint-integrated.sh"
)

for file in "${DOCKER_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        error "❌ Docker file missing: $file"
        exit 1
    fi
    log "✅ Found: $file"
done

# Test 4: Static file serving setup
log "Testing static file serving setup..."

# Copy frontend build to static directory for testing
mkdir -p static
cp -r frontend/dist/* static/

if [ -f "static/index.html" ]; then
    log "✅ Static files copied successfully"
else
    error "❌ Failed to copy static files"
    exit 1
fi

# Test 5: FastAPI integration
log "Testing FastAPI integration..."

if [ -f "src/advanced_agent/interfaces/fastapi_app.py" ]; then
    log "✅ FastAPI application found"
    
    # Check if static file mounting is configured
    if grep -q "StaticFiles" src/advanced_agent/interfaces/fastapi_app.py; then
        log "✅ Static file serving configured in FastAPI"
    else
        warn "⚠️  Static file serving may not be configured in FastAPI"
    fi
else
    error "❌ FastAPI application not found"
    exit 1
fi

# Test 6: Nginx configuration
log "Testing Nginx configuration..."

if [ -f "docker/nginx/integrated.conf" ]; then
    # Check for React Router support
    if grep -q "try_files.*@fallback" docker/nginx/integrated.conf; then
        log "✅ React Router support configured"
    else
        warn "⚠️  React Router support may not be configured"
    fi
    
    # Check for API proxy
    if grep -q "location /api/" docker/nginx/integrated.conf; then
        log "✅ API proxy configured"
    else
        warn "⚠️  API proxy may not be configured"
    fi
    
    # Check for static file serving
    if grep -q "root /app/static" docker/nginx/integrated.conf; then
        log "✅ Static file serving configured"
    else
        warn "⚠️  Static file serving may not be configured"
    fi
else
    error "❌ Nginx configuration not found"
    exit 1
fi

# Test 7: Environment configuration
log "Testing environment configuration..."

if [ -f "docker-compose.yml" ]; then
    if grep -q "FRONTEND_ENABLED=true" docker-compose.yml; then
        log "✅ Frontend enabled in docker-compose"
    else
        warn "⚠️  Frontend may not be enabled in docker-compose"
    fi
else
    error "❌ docker-compose.yml not found"
    exit 1
fi

# Test 8: File structure validation
log "Validating file structure..."

REQUIRED_DIRS=(
    "frontend/dist"
    "src/advanced_agent"
    "docker/nginx"
    "docker/supervisor"
    "static"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        error "❌ Required directory missing: $dir"
        exit 1
    fi
    log "✅ Found directory: $dir"
done

# Cleanup
log "Cleaning up test files..."
rm -rf static

log "🎉 Integration test completed successfully!"
log "✅ Frontend build: OK"
log "✅ Backend configuration: OK"
log "✅ Docker configuration: OK"
log "✅ Nginx configuration: OK"
log "✅ File structure: OK"

log "Ready for Docker deployment!"
log "To deploy, run: docker-compose up --build"