#!/bin/bash
# Frontend Build Script
# フロントエンドビルドスクリプト

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

# Check if we're in the right directory
if [ ! -f "frontend/package.json" ]; then
    error "This script must be run from the project root directory"
    exit 1
fi

log "Starting frontend build process..."

# Navigate to frontend directory
cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    log "Installing dependencies..."
    yarn install --frozen-lockfile
else
    log "Dependencies already installed"
fi

# Run type check
log "Running TypeScript type check..."
if yarn type-check; then
    log "Type check passed"
else
    error "Type check failed"
    exit 1
fi

# Run linting
log "Running ESLint..."
if yarn lint; then
    log "Linting passed"
else
    warn "Linting issues found, but continuing..."
fi

# Build the application
log "Building React application..."
if yarn build; then
    log "Build completed successfully"
else
    error "Build failed"
    exit 1
fi

# Check if build output exists
if [ ! -d "dist" ] || [ ! -f "dist/index.html" ]; then
    error "Build output not found in dist directory"
    exit 1
fi

log "Build output:"
ls -la dist/

# Calculate build size
BUILD_SIZE=$(du -sh dist | cut -f1)
log "Total build size: $BUILD_SIZE"

# Check for critical files
CRITICAL_FILES=("index.html" "assets")
for file in "${CRITICAL_FILES[@]}"; do
    if [ ! -e "dist/$file" ]; then
        error "Critical file/directory missing: $file"
        exit 1
    fi
done

log "Frontend build completed successfully!"
log "Build output is ready in frontend/dist/"

# Return to project root
cd ..

log "Frontend build process completed"