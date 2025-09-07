#!/bin/bash
# Docker Build Test Script
# Docker ビルドテストスクリプト

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

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    error "Docker is not installed or not in PATH"
    exit 1
fi

log "Testing Docker build process for integrated application..."

# Build the integrated Docker image
log "Building integrated Docker image..."
if docker build -f Dockerfile.integrated -t ai-agent-integrated:test .; then
    log "Docker build completed successfully"
else
    error "Docker build failed"
    exit 1
fi

# Test if the image was created
if docker images | grep -q "ai-agent-integrated.*test"; then
    log "Docker image created successfully"
else
    error "Docker image not found"
    exit 1
fi

# Test running the container (dry run)
log "Testing container startup (dry run)..."
if docker run --rm --name ai-agent-test -d ai-agent-integrated:test; then
    CONTAINER_ID=$(docker ps -q --filter "name=ai-agent-test")
    
    # Wait a moment for startup
    sleep 10
    
    # Check if container is still running
    if docker ps | grep -q "ai-agent-test"; then
        log "Container started successfully"
        
        # Check if services are responding
        if docker exec $CONTAINER_ID curl -f http://localhost/health 2>/dev/null; then
            log "Health check passed"
        else
            warn "Health check failed, but container is running"
        fi
        
        # Stop the test container
        docker stop ai-agent-test
        log "Test container stopped"
    else
        error "Container failed to start or crashed"
        docker logs ai-agent-test
        exit 1
    fi
else
    error "Failed to start test container"
    exit 1
fi

log "Docker build test completed successfully!"
log "Image: ai-agent-integrated:test"

# Cleanup
log "Cleaning up test image..."
docker rmi ai-agent-integrated:test

log "Docker build test process completed"