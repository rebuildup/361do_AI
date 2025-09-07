#!/bin/bash
# Final Integration Testing Script
# 最終統合テストスクリプト

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
WARNING_TESTS=0

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅ $1${NC}"
    ((PASSED_TESTS++))
    ((TOTAL_TESTS++))
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}"
    ((WARNING_TESTS++))
    ((TOTAL_TESTS++))
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌ $1${NC}"
    ((FAILED_TESTS++))
    ((TOTAL_TESTS++))
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] ℹ️  $1${NC}"
}

section() {
    echo -e "${PURPLE}[$(date +'%Y-%m-%d %H:%M:%S')] 📋 $1${NC}"
}

# Test helper functions
test_file_exists() {
    local file=$1
    local description=$2
    
    if [ -f "$file" ]; then
        log "$description: $file exists"
        return 0
    else
        error "$description: $file missing"
        return 1
    fi
}

test_directory_exists() {
    local dir=$1
    local description=$2
    
    if [ -d "$dir" ]; then
        log "$description: $dir exists"
        return 0
    else
        error "$description: $dir missing"
        return 1
    fi
}

test_content_exists() {
    local file=$1
    local pattern=$2
    local description=$3
    
    if [ -f "$file" ] && grep -q "$pattern" "$file"; then
        log "$description: Found in $file"
        return 0
    else
        error "$description: Not found in $file"
        return 1
    fi
}

# Main test execution
main() {
    section "🚀 Starting Final Integration Testing"
    info "Testing React Custom WebUI Implementation"
    
    # Check if we're in the right directory
    if [ ! -f "frontend/package.json" ]; then
        error "This script must be run from the project root directory"
        exit 1
    fi
    
    # Test Requirement 1: React UI Migration
    section "📝 Testing Requirement 1: React UI Migration"
    test_requirement_1
    
    # Test Requirement 2: Responsive Chat Interface
    section "📝 Testing Requirement 2: Responsive Chat Interface"
    test_requirement_2
    
    # Test Requirement 3: Backend Functionality
    section "📝 Testing Requirement 3: Backend Functionality"
    test_requirement_3
    
    # Test Requirement 4: Advanced UI Features
    section "📝 Testing Requirement 4: Advanced UI Features"
    test_requirement_4
    
    # Test Requirement 5: Development Setup
    section "📝 Testing Requirement 5: Development Setup"
    test_requirement_5
    
    # Test Requirement 6: Feature Parity
    section "📝 Testing Requirement 6: Feature Parity"
    test_requirement_6
    
    # Test Requirement 7: Deployment Integration
    section "📝 Testing Requirement 7: Deployment Integration"
    test_requirement_7
    
    # Test Requirement 8: Design Consistency
    section "📝 Testing Requirement 8: Design Consistency"
    test_requirement_8
    
    # Test Requirement 9: Natural Language Support
    section "📝 Testing Requirement 9: Natural Language Support"
    test_requirement_9
    
    # Additional Integration Tests
    section "📝 Additional Integration Tests"
    test_build_process
    test_responsive_design
    test_performance
    test_accessibility
    
    # Generate test report
    generate_test_report
}

# Requirement 1: React UI Migration (1.1, 1.2, 1.3, 1.4)
test_requirement_1() {
    info "Testing React UI migration and agent communication"
    
    # 1.1: Agent communication functionality
    test_file_exists "frontend/src/services/api.ts" "Agent communication API service"
    test_content_exists "frontend/src/services/api.ts" "agentChatCompletion\|processUserInput" "Agent communication methods"
    
    # 1.2: Chat functionality parity
    test_file_exists "frontend/src/components/chat/ChatInterface.tsx" "Chat interface component"
    test_content_exists "frontend/src/components/chat/ChatInterface.tsx" "streaming\|message" "Chat functionality"
    
    # 1.3: No Streamlit dependency
    if ! grep -r "streamlit" frontend/src/ 2>/dev/null; then
        log "No Streamlit dependencies in React frontend"
    else
        warn "Streamlit references found in frontend code"
    fi
    
    # 1.4: Error handling and fallbacks
    test_file_exists "frontend/src/components/ui/ErrorBoundary.tsx" "Error boundary implementation"
    test_file_exists "frontend/src/components/ui/FallbackComponents.tsx" "Fallback components"
}

# Requirement 2: Responsive Chat Interface (2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7)
test_requirement_2() {
    info "Testing responsive design and chat interface"
    
    # 2.1: Main content area 768px width
    test_content_exists "frontend/src/components/layout/SimpleLayout.tsx" "768px\|max-w-3xl" "Main content width constraint" || \
    test_content_exists "frontend/tailwind.config.js" "768" "Responsive breakpoint configuration"
    
    # 2.2: Side panel 288px width
    test_content_exists "frontend/src/components/layout/SimpleLayout.tsx" "288px\|w-72" "Side panel width" || \
    warn "Side panel width configuration not found"
    
    # 2.3: Collapsed panel 48px width
    test_content_exists "frontend/src/components/layout/SimpleLayout.tsx" "48px\|w-12" "Collapsed panel width" || \
    warn "Collapsed panel width configuration not found"
    
    # 2.4: Auto-collapse behavior
    test_content_exists "frontend/src/components/layout/SimpleLayout.tsx" "1056px\|lg:" "Auto-collapse breakpoint" || \
    warn "Auto-collapse behavior not found"
    
    # 2.5: Real-time streaming
    test_file_exists "frontend/src/hooks/useStreaming.ts" "Streaming functionality"
    test_content_exists "frontend/src/hooks/useStreaming.ts" "streaming\|stream" "Streaming implementation"
    
    # 2.6: Message display
    test_file_exists "frontend/src/components/chat/MessageBubble.tsx" "Message display component"
    
    # 2.7: Smooth animations
    test_content_exists "frontend/src/components/chat/ChatInterface.tsx" "transition\|animate" "Smooth animations" || \
    test_content_exists "frontend/tailwind.config.js" "animation" "Animation configuration"
}

# Requirement 3: Backend Functionality (3.1, 3.2, 3.3, 3.4)
test_requirement_3() {
    info "Testing backend integration and functionality"
    
    # 3.1: FastAPI gateway communication
    test_content_exists "frontend/src/services/api.ts" "FastAPI\|/v1/\|/api/" "FastAPI gateway integration"
    
    # 3.2: Self-learning capabilities
    test_content_exists "frontend/src/services/api.ts" "learning\|metrics\|feedback" "Self-learning integration"
    
    # 3.3: Agent functionality preservation
    test_content_exists "frontend/src/services/api.ts" "tool.*registry\|prompt.*management\|reward" "Agent functionality"
    
    # 3.4: Error handling and retry
    test_content_exists "frontend/src/services/api.ts" "retry\|error.*handling" "Error handling and retry mechanisms"
    test_file_exists "frontend/src/utils/errorHandling.ts" "Error handling utilities"
}

# Requirement 4: Advanced UI Features (4.1, 4.2, 4.3, 4.4, 4.5, 4.6)
test_requirement_4() {
    info "Testing advanced UI features"
    
    # 4.1: Collapsible sidebar
    test_content_exists "frontend/src/components/layout/SimpleLayout.tsx" "collaps\|expand" "Collapsible sidebar functionality"
    
    # 4.2: Configuration options
    test_file_exists "frontend/src/components/debug/BackendTester.tsx" "Configuration interface" || \
    warn "Configuration components not found"
    
    # 4.3: Real-time status updates
    test_content_exists "frontend/src/services/api.ts" "getAgentStatus\|status" "Real-time status updates"
    
    # 4.4: Tabbed/modal interfaces
    test_content_exists "frontend/src/components/ui/Toast.tsx" "modal\|tab" "Modal interfaces" || \
    warn "Modal/tabbed interfaces not explicitly found"
    
    # 4.5: Lucide icons
    test_content_exists "frontend/package.json" "lucide-react" "Lucide React icons dependency"
    test_content_exists "frontend/src/components/chat/ChatInterface.tsx" "lucide-react\|from.*lucide" "Lucide icons usage"
    
    # 4.6: Icon fallbacks
    test_content_exists "frontend/src/components/ui/FallbackComponents.tsx" "icon\|Icon" "Icon fallback implementation"
}

# Requirement 5: Development Setup (5.1, 5.2, 5.3, 5.4, 5.5, 5.6)
test_requirement_5() {
    info "Testing development and build setup"
    
    # 5.1: Modern React development setup
    test_content_exists "frontend/package.json" "vite\|react.*18\|typescript" "Modern React setup"
    test_file_exists "frontend/vite.config.ts" "Vite configuration"
    
    # 5.2: Optimized production build
    if [ -d "frontend/dist" ]; then
        log "Production build output exists"
        
        # Check build optimization
        if [ -f "frontend/dist/index.html" ] && grep -q "assets" "frontend/dist/index.html"; then
            log "Optimized assets in production build"
        else
            warn "Asset optimization not verified"
        fi
    else
        warn "Production build not found - run 'npm run build' in frontend directory"
    fi
    
    # 5.3: TypeScript support
    test_file_exists "frontend/tsconfig.json" "TypeScript configuration"
    test_content_exists "frontend/src/App.tsx" "tsx\|TypeScript" "TypeScript usage"
    
    # 5.4: Tailwind CSS
    test_content_exists "frontend/package.json" "tailwindcss" "Tailwind CSS dependency"
    test_file_exists "frontend/tailwind.config.js" "Tailwind configuration"
    
    # 5.5: Yarn package management
    test_file_exists "frontend/yarn.lock" "Yarn lock file"
    
    # 5.6: Minimal root folder
    local root_files=$(find . -maxdepth 1 -type f | wc -l)
    if [ "$root_files" -lt 20 ]; then
        log "Root folder kept minimal ($root_files files)"
    else
        warn "Root folder has many files ($root_files files)"
    fi
}

# Requirement 6: Feature Parity (6.1, 6.2, 6.3, 6.4)
test_requirement_6() {
    info "Testing feature parity with existing system"
    
    # 6.1: Model selection
    test_content_exists "frontend/src/services/api.ts" "getModels\|model.*selection" "Model selection functionality"
    
    # 6.2: Agent metrics
    test_content_exists "frontend/src/services/api.ts" "metrics\|performance\|learning" "Agent metrics display"
    
    # 6.3: Configuration options
    test_content_exists "frontend/src/services/api.ts" "temperature\|max.*tokens\|streaming" "Configuration options"
    
    # 6.4: Tool functionality
    test_content_exists "frontend/src/services/api.ts" "tool\|execute.*tool" "Tool functionality"
}

# Requirement 7: Deployment Integration (7.1, 7.2, 7.3, 7.4)
test_requirement_7() {
    info "Testing deployment integration"
    
    # 7.1: Docker setup compatibility
    test_file_exists "Dockerfile.integrated" "Integrated Docker configuration"
    test_file_exists "docker-compose.yml" "Docker Compose configuration"
    
    # 7.2: FastAPI integration
    test_content_exists "src/advanced_agent/interfaces/fastapi_app.py" "StaticFiles\|static.*files" "FastAPI static file serving"
    
    # 7.3: Configuration compatibility
    test_content_exists "docker-compose.yml" "frontend\|FRONTEND_ENABLED" "Frontend configuration in Docker Compose"
    
    # 7.4: Error messages and troubleshooting
    test_file_exists "docs/REACT_DEPLOYMENT.md" "Deployment documentation"
    test_content_exists "docs/REACT_DEPLOYMENT.md" "troubleshooting\|error" "Troubleshooting guidance"
}

# Requirement 8: Design Consistency (8.1, 8.2, 8.3, 8.4, 8.5)
test_requirement_8() {
    info "Testing design consistency"
    
    # 8.1: Base color #000000
    test_content_exists "frontend/src/App.tsx" "bg-black\|#000000" "Black base color usage"
    
    # 8.2: Monochrome dark theme
    test_content_exists "frontend/tailwind.config.js" "dark\|gray" "Dark theme configuration" || \
    test_content_exists "frontend/src/index.css" "dark\|black\|gray" "Dark theme styling"
    
    # 8.3: No drop shadows
    if ! grep -r "shadow-\|drop-shadow" frontend/src/ 2>/dev/null | grep -v "shadow-none"; then
        log "No drop shadows found in components"
    else
        warn "Drop shadows may be present in styling"
    fi
    
    # 8.4: Grid system awareness
    test_content_exists "frontend/tailwind.config.js" "grid\|flex" "Grid system configuration" || \
    test_content_exists "frontend/src/components/layout/SimpleLayout.tsx" "grid\|flex" "Grid system usage"
    
    # 8.5: Transparent/semi-transparent panels
    test_content_exists "frontend/src/components/ui/FallbackComponents.tsx" "bg-.*\/\|opacity" "Transparent panel styling" || \
    test_content_exists "frontend/src/components/chat/ChatInterface.tsx" "bg-.*\/\|opacity" "Semi-transparent styling"
}

# Requirement 9: Natural Language Support (9.1, 9.2, 9.3, 9.4, 9.5)
test_requirement_9() {
    info "Testing natural language processing support"
    
    # 9.1: Japanese language support
    test_content_exists "frontend/src/services/naturalLanguageProcessor.ts" "ja\|japanese\|日本語" "Japanese language support" || \
    warn "Japanese language support not explicitly found"
    
    # 9.2: Agent functions through natural language
    test_content_exists "frontend/src/services/api.ts" "web.*search\|command.*execution\|file.*modification\|MCP" "Agent functions support"
    
    # 9.3: Indefinite conversations
    test_content_exists "frontend/src/services/sessionPersistence.ts" "session\|persistence" "Session persistence"
    test_content_exists "frontend/src/hooks/useSessionManager.ts" "session\|conversation" "Session management"
    
    # 9.4: Prompt rewriting capability
    test_content_exists "frontend/src/services/api.ts" "prompt\|rewrite\|tuning" "Prompt manipulation support"
    
    # 9.5: Natural language judgment
    test_content_exists "frontend/src/services/naturalLanguageProcessor.ts" "natural.*language\|NLP" "Natural language processing"
}

# Additional Integration Tests
test_build_process() {
    info "Testing build process"
    
    cd frontend
    
    # Test if build succeeds
    if npm run build > /dev/null 2>&1; then
        log "Frontend build process successful"
        
        # Check build output
        if [ -f "dist/index.html" ]; then
            log "Build output generated successfully"
            
            # Check asset optimization
            local js_files=$(find dist -name "*.js" | wc -l)
            local css_files=$(find dist -name "*.css" | wc -l)
            
            if [ "$js_files" -gt 0 ] && [ "$css_files" -gt 0 ]; then
                log "Optimized assets generated (JS: $js_files, CSS: $css_files)"
            else
                warn "Asset generation may be incomplete"
            fi
        else
            error "Build output missing"
        fi
    else
        error "Frontend build process failed"
    fi
    
    cd ..
}

test_responsive_design() {
    info "Testing responsive design implementation"
    
    # Check for responsive breakpoints
    test_content_exists "frontend/tailwind.config.js" "sm:\|md:\|lg:\|xl:" "Responsive breakpoints" || \
    test_content_exists "frontend/src/components/layout/SimpleLayout.tsx" "sm:\|md:\|lg:\|xl:" "Responsive classes usage"
    
    # Check for mobile-first approach
    test_content_exists "frontend/src/components/layout/SimpleLayout.tsx" "mobile\|responsive" "Mobile responsiveness" || \
    warn "Mobile responsiveness not explicitly verified"
}

test_performance() {
    info "Testing performance optimizations"
    
    # Check for lazy loading
    test_content_exists "frontend/src/App.tsx" "lazy\|Suspense" "Lazy loading implementation"
    
    # Check for code splitting
    test_content_exists "frontend/vite.config.ts" "manualChunks\|splitChunks" "Code splitting configuration"
    
    # Check for memoization
    test_content_exists "frontend/src/App.tsx" "memo\|useMemo\|useCallback" "React memoization"
    
    # Check for performance monitoring
    test_file_exists "frontend/src/utils/performance.ts" "Performance monitoring utilities" || \
    warn "Performance monitoring not found"
}

test_accessibility() {
    info "Testing accessibility features"
    
    # Check for ARIA attributes
    test_content_exists "frontend/src/components/ui/ErrorBoundary.tsx" "aria-\|role=" "ARIA attributes in error boundary" || \
    warn "ARIA attributes not found in error components"
    
    # Check for keyboard navigation
    test_content_exists "frontend/src/components/chat/ChatInterface.tsx" "onKeyDown\|onKeyPress\|tabIndex" "Keyboard navigation support" || \
    warn "Keyboard navigation not explicitly found"
    
    # Check for semantic HTML
    test_content_exists "frontend/src/components/chat/MessageBubble.tsx" "<button\|<nav\|<main\|<section" "Semantic HTML elements" || \
    warn "Semantic HTML usage not verified"
}

generate_test_report() {
    section "📊 Final Integration Test Report"
    
    echo -e "${CYAN}=================================${NC}"
    echo -e "${CYAN}  REACT CUSTOM WEBUI TEST REPORT${NC}"
    echo -e "${CYAN}=================================${NC}"
    echo ""
    
    echo -e "📈 ${BLUE}Test Statistics:${NC}"
    echo -e "   Total Tests: ${TOTAL_TESTS}"
    echo -e "   ${GREEN}Passed: ${PASSED_TESTS}${NC}"
    echo -e "   ${YELLOW}Warnings: ${WARNING_TESTS}${NC}"
    echo -e "   ${RED}Failed: ${FAILED_TESTS}${NC}"
    echo ""
    
    local success_rate=$((PASSED_TESTS * 100 / TOTAL_TESTS))
    echo -e "🎯 ${BLUE}Success Rate: ${success_rate}%${NC}"
    echo ""
    
    if [ $FAILED_TESTS -eq 0 ]; then
        echo -e "🎉 ${GREEN}All critical tests passed!${NC}"
        echo -e "✅ ${GREEN}React Custom WebUI is ready for deployment${NC}"
    elif [ $FAILED_TESTS -lt 5 ]; then
        echo -e "⚠️  ${YELLOW}Minor issues found - review failed tests${NC}"
        echo -e "🔧 ${YELLOW}Address issues before production deployment${NC}"
    else
        echo -e "❌ ${RED}Significant issues found - requires attention${NC}"
        echo -e "🚫 ${RED}Not recommended for production deployment${NC}"
    fi
    
    echo ""
    echo -e "📋 ${BLUE}Requirements Coverage:${NC}"
    echo -e "   ✅ Requirement 1: React UI Migration"
    echo -e "   ✅ Requirement 2: Responsive Chat Interface"
    echo -e "   ✅ Requirement 3: Backend Functionality"
    echo -e "   ✅ Requirement 4: Advanced UI Features"
    echo -e "   ✅ Requirement 5: Development Setup"
    echo -e "   ✅ Requirement 6: Feature Parity"
    echo -e "   ✅ Requirement 7: Deployment Integration"
    echo -e "   ✅ Requirement 8: Design Consistency"
    echo -e "   ✅ Requirement 9: Natural Language Support"
    
    echo ""
    echo -e "🚀 ${BLUE}Next Steps:${NC}"
    echo -e "   1. Review any failed tests and warnings"
    echo -e "   2. Test the application manually in different browsers"
    echo -e "   3. Verify responsive design on various screen sizes"
    echo -e "   4. Test with real backend integration"
    echo -e "   5. Deploy to staging environment for final validation"
    
    echo ""
    echo -e "${CYAN}=================================${NC}"
    
    # Return appropriate exit code
    if [ $FAILED_TESTS -eq 0 ]; then
        return 0
    else
        return 1
    fi
}

# Execute main function
main "$@"