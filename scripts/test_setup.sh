#!/bin/bash

# Comprehensive test script for AI Asylum
# Tests the entire setup and validates everything works

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results
PASSED=0
FAILED=0

# Print functions
print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
    ((PASSED++))
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
    ((FAILED++))
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Test functions
test_python_version() {
    print_header "Testing Python Version"
    if command -v python3 &> /dev/null; then
        VERSION=$(python3 --version | cut -d' ' -f2)
        MAJOR=$(echo $VERSION | cut -d'.' -f1)
        MINOR=$(echo $VERSION | cut -d'.' -f2)
        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ]; then
            print_success "Python $VERSION is installed"
        else
            print_error "Python 3.10+ required, found $VERSION"
        fi
    else
        print_error "Python3 not found"
    fi
}

test_venv() {
    print_header "Testing Virtual Environment"
    if [ -d "venv" ]; then
        print_success "Virtual environment exists"
        if [ -f "venv/bin/python" ] || [ -f "venv/Scripts/python.exe" ]; then
            print_success "Virtual environment is valid"
        else
            print_error "Virtual environment is invalid"
        fi
    else
        print_warning "Virtual environment not found (run: python3 -m venv venv)"
    fi
}

test_dependencies() {
    print_header "Testing Dependencies"
    if [ -d "venv" ]; then
        source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
        
        REQUIRED_PACKAGES=("fastapi" "sqlalchemy" "alembic" "pydantic" "click")
        for package in "${REQUIRED_PACKAGES[@]}"; do
            if python -c "import $package" 2>/dev/null; then
                print_success "$package is installed"
            else
                print_error "$package is not installed"
            fi
        done
    else
        print_warning "Skipping dependency check (venv not found)"
    fi
}

test_project_structure() {
    print_header "Testing Project Structure"
    REQUIRED_DIRS=(
        "vivasecuris/aiasylum"
        "vivasecuris/aiasylum/models"
        "vivasecuris/aiasylum/doctor"
        "vivasecuris/aiasylum/patient"
        "vivasecuris/aiasylum/tests"
        "vivasecuris/aiasylum/runner"
        "vivasecuris/aiasylum/api"
        "config"
        "alembic"
    )
    
    for dir in "${REQUIRED_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            print_success "$dir exists"
        else
            print_error "$dir is missing"
        fi
    done
}

test_config_files() {
    print_header "Testing Configuration Files"
    REQUIRED_FILES=(
        "config/settings.py"
        "config/models.yaml"
        "config/tests.yaml"
        "alembic.ini"
        "requirements.txt"
        "Makefile"
    )
    
    for file in "${REQUIRED_FILES[@]}"; do
        if [ -f "$file" ]; then
            print_success "$file exists"
        else
            print_error "$file is missing"
        fi
    done
}

test_env_file() {
    print_header "Testing Environment Configuration"
    if [ -f ".env" ]; then
        print_success ".env file exists"
        
        # Check for at least one API key or Ollama
        if grep -q "OPENAI_API_KEY=sk-" .env || \
           grep -q "ANTHROPIC_API_KEY=sk-" .env || \
           grep -q "GOOGLE_API_KEY=" .env || \
           grep -q "OLLAMA_BASE_URL" .env; then
            print_success "At least one provider is configured"
        else
            print_warning "No API keys found (Ollama can be used without keys)"
        fi
    else
        print_warning ".env file not found (copy from .env.example)"
    fi
}

test_database() {
    print_header "Testing Database"
    if [ -d "venv" ]; then
        source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
        
        # Test if we can import database modules
        if python -c "from vivasecuris.aiasylum.database import models, session" 2>/dev/null; then
            print_success "Database modules can be imported"
        else
            print_error "Database modules cannot be imported"
        fi
        
        # Test if database file exists or can be created
        mkdir -p data
        if python -c "
from config import settings
from vivasecuris.aiasylum.database.session import init_db
init_db()
print('Database initialized')
" 2>/dev/null; then
            print_success "Database can be initialized"
        else
            print_warning "Database initialization failed (may need migrations)"
        fi
    else
        print_warning "Skipping database test (venv not found)"
    fi
}

test_imports() {
    print_header "Testing Python Imports"
    if [ -d "venv" ]; then
        source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
        
        IMPORT_TESTS=(
            "from vivasecuris.aiasylum.models import BaseModel"
            "from vivasecuris.aiasylum.doctor import Doctor"
            "from vivasecuris.aiasylum.patient import Patient"
            "from vivasecuris.aiasylum.runner import TestRunner"
            "from vivasecuris.aiasylum.api.main import app"
        )
        
        for import_test in "${IMPORT_TESTS[@]}"; do
            if python -c "$import_test" 2>/dev/null; then
                print_success "Import: ${import_test:0:50}..."
            else
                print_error "Import failed: ${import_test:0:50}..."
            fi
        done
    else
        print_warning "Skipping import test (venv not found)"
    fi
}

test_docker() {
    print_header "Testing Docker Configuration"
    if [ -f "Dockerfile" ]; then
        print_success "Dockerfile exists"
    else
        print_error "Dockerfile is missing"
    fi
    
    if [ -f "docker-compose.yml" ]; then
        print_success "docker-compose.yml exists"
        
        # Check if docker-compose is available
        if command -v docker-compose &> /dev/null || command -v docker &> /dev/null; then
            print_success "Docker is available"
        else
            print_warning "Docker is not installed (optional)"
        fi
    else
        print_error "docker-compose.yml is missing"
    fi
}

test_ansible() {
    print_header "Testing Ansible Configuration"
    if [ -d "ansible" ]; then
        print_success "Ansible directory exists"
        
        REQUIRED_ANSIBLE_FILES=(
            "ansible/playbook.yml"
            "ansible/inventory.ini"
        )
        
        for file in "${REQUIRED_ANSIBLE_FILES[@]}"; do
            if [ -f "$file" ]; then
                print_success "$(basename $file) exists"
            else
                print_error "$(basename $file) is missing"
            fi
        done
        
        if command -v ansible-playbook &> /dev/null; then
            print_success "Ansible is installed"
        else
            print_warning "Ansible is not installed (optional)"
        fi
    else
        print_warning "Ansible directory not found (optional)"
    fi
}

run_pytest() {
    print_header "Running Unit Tests"
    if [ -d "venv" ]; then
        source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
        
        if command -v pytest &> /dev/null; then
            if pytest tests/ -v --tb=short 2>&1 | tee /tmp/pytest_output.log; then
                print_success "All unit tests passed"
            else
                print_error "Some unit tests failed (check /tmp/pytest_output.log)"
            fi
        else
            print_warning "pytest not installed (install with: pip install pytest)"
        fi
    else
        print_warning "Skipping pytest (venv not found)"
    fi
}

# Main execution
main() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════╗"
    echo "║         AI Asylum - Setup Test Script                 ║"
    echo "╚════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    test_python_version
    test_venv
    test_dependencies
    test_project_structure
    test_config_files
    test_env_file
    test_database
    test_imports
    test_docker
    test_ansible
    run_pytest
    
    # Summary
    echo -e "\n${BLUE}════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}Test Summary${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}Passed: $PASSED${NC}"
    echo -e "${RED}Failed: $FAILED${NC}"
    
    if [ $FAILED -eq 0 ]; then
        echo -e "\n${GREEN}✓ All tests passed!${NC}"
        exit 0
    else
        echo -e "\n${RED}✗ Some tests failed. Please review the errors above.${NC}"
        exit 1
    fi
}

main "$@"
