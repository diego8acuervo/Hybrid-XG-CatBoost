#!/bin/bash
# Verification script for workspace setup

echo "Verifying Systematic Alpha Generation Workspace Setup..."
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check virtual environment
if [ -d "venv" ]; then
    echo -e "${GREEN}✓${NC} Virtual environment exists"
else
    echo -e "${RED}✗${NC} Virtual environment not found"
    exit 1
fi

# Check Python version
source venv/bin/activate
PYTHON_VERSION=$(python --version 2>&1)
echo -e "${GREEN}✓${NC} $PYTHON_VERSION"

# Check key packages
echo ""
echo "Checking installed packages..."

check_package() {
    if python -c "import $1" 2>/dev/null; then
        VERSION=$(python -c "import $1; print($1.__version__)" 2>/dev/null)
        echo -e "${GREEN}✓${NC} $1 $VERSION"
        return 0
    else
        echo -e "${RED}✗${NC} $1 not found"
        return 1
    fi
}

check_package "pandas"
check_package "numpy"
check_package "sklearn"
check_package "xgboost"
check_package "catboost"
check_package "yfinance"
check_package "matplotlib"
check_package "shap"

# Check VS Code files
echo ""
echo "Checking VS Code configuration..."

if [ -f ".vscode/settings.json" ]; then
    echo -e "${GREEN}✓${NC} .vscode/settings.json"
else
    echo -e "${RED}✗${NC} .vscode/settings.json"
fi

if [ -f ".vscode/tasks.json" ]; then
    echo -e "${GREEN}✓${NC} .vscode/tasks.json"
else
    echo -e "${RED}✗${NC} .vscode/tasks.json"
fi

if [ -f ".vscode/launch.json" ]; then
    echo -e "${GREEN}✓${NC} .vscode/launch.json"
else
    echo -e "${RED}✗${NC} .vscode/launch.json"
fi

# Check scripts
echo ""
echo "Checking utility scripts..."

if [ -x "activate.sh" ]; then
    echo -e "${GREEN}✓${NC} activate.sh (executable)"
else
    echo -e "${YELLOW}⚠${NC} activate.sh (not executable or missing)"
fi

if [ -x "quickstart.sh" ]; then
    echo -e "${GREEN}✓${NC} quickstart.sh (executable)"
else
    echo -e "${YELLOW}⚠${NC} quickstart.sh (not executable or missing)"
fi

# Check workspace file
if [ -f "systematic-alpha.code-workspace" ]; then
    echo -e "${GREEN}✓${NC} systematic-alpha.code-workspace"
else
    echo -e "${YELLOW}⚠${NC} systematic-alpha.code-workspace"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Workspace verification complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "To start working:"
echo "  1. source activate.sh"
echo "  2. ./quickstart.sh"
echo "  or"
echo "  3. code systematic-alpha.code-workspace"
echo ""
