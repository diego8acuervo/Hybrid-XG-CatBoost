#!/bin/bash
# Activation script for Systematic Alpha Generation environment

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Systematic Alpha Generation Environment${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Activate virtual environment
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    echo -e "${GREEN}✓ Activating virtual environment...${NC}"
    source "$SCRIPT_DIR/venv/bin/activate"
    echo -e "${GREEN}✓ Environment activated: $(which python)${NC}"
    echo -e "${GREEN}✓ Python version: $(python --version)${NC}"
else
    echo -e "${YELLOW}⚠ Virtual environment not found. Run setup first.${NC}"
    exit 1
fi

# Set PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
echo -e "${GREEN}✓ PYTHONPATH set to: $SCRIPT_DIR${NC}"

# Display available commands
echo ""
echo -e "${BLUE}Available Commands:${NC}"
echo -e "  ${GREEN}make run-full${NC}       - Run complete pipeline"
echo -e "  ${GREEN}make run-data${NC}       - Download data only"
echo -e "  ${GREEN}make run-features${NC}   - Engineer features"
echo -e "  ${GREEN}make run-train${NC}      - Train model"
echo -e "  ${GREEN}make run-backtest${NC}   - Run backtest"
echo -e "  ${GREEN}make test${NC}           - Run tests"
echo -e "  ${GREEN}make clean${NC}          - Clean cache files"
echo ""
echo -e "${BLUE}Or use Python directly:${NC}"
echo -e "  ${GREEN}python main.py --help${NC}"
echo ""
