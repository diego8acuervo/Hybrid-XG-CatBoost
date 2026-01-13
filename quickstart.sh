#!/bin/bash
# Quick Start Script for Systematic Alpha Generation

set -e  # Exit on error

echo "===================================="
echo "Systematic Alpha Generation"
echo "Quick Start Setup"
echo "===================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: ~/.pyenv/versions/3.11.9/bin/python -m venv venv"
    exit 1
fi

# Activate environment
echo "Activating virtual environment..."
source venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

# Display menu
echo ""
echo "Select an option:"
echo "1) Run full pipeline (data → features → model → backtest)"
echo "2) Download data only"
echo "3) Engineer features"
echo "4) Train model"
echo "5) Run backtest"
echo "6) Run tests"
echo "7) Open Jupyter notebook"
echo "8) Show help"
echo "9) Exit"
echo ""
read -p "Enter option (1-9): " option

case $option in
    1)
        echo "Running full pipeline..."
        python main.py --config config.yaml --run-full --verbose
        ;;
    2)
        echo "Downloading data..."
        python main.py --stage data-fetch --verbose
        ;;
    3)
        echo "Engineering features..."
        python main.py --stage feature-engineering --verbose
        ;;
    4)
        echo "Training model..."
        python main.py --stage train-model --verbose
        ;;
    5)
        echo "Running backtest..."
        python main.py --stage backtest --walk-forward --verbose
        ;;
    6)
        echo "Running tests..."
        pytest tests/ -v
        ;;
    7)
        echo "Starting Jupyter notebook..."
        jupyter notebook
        ;;
    8)
        python main.py --help
        ;;
    9)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid option!"
        exit 1
        ;;
esac
