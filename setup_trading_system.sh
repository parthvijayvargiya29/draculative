#!/bin/bash
# Complete Trading System - Setup and Run Scripts

set -e  # Exit on error

PROJECT_ROOT="/Users/parthvijayvargiya/Documents/GitHub/draculative"
cd "$PROJECT_ROOT"

echo "========================================"
echo "Trading System - Setup & Deployment"
echo "========================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print section headers
print_section() {
    echo ""
    echo -e "${BLUE}$1${NC}"
    echo "─────────────────────────────────────"
}

# Function to print success
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Function to print warning
print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# ========== STEP 1: SETUP ENVIRONMENT ==========
print_section "STEP 1: Setup Python Environment"

if [ ! -d "trading_system_venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv trading_system_venv
    print_success "Virtual environment created"
else
    print_success "Virtual environment exists"
fi

source trading_system_venv/bin/activate
print_success "Virtual environment activated"

# ========== STEP 2: INSTALL DEPENDENCIES ==========
print_section "STEP 2: Install Dependencies"

echo "Installing packages from trading_system/requirements.txt..."
pip install --quiet -r trading_system/requirements.txt
print_success "Dependencies installed"

# Optional: Install yfinance for backtesting
echo "Installing optional yfinance for backtesting..."
pip install --quiet yfinance pandas scikit-learn
print_success "Optional packages installed"

# ========== STEP 3: VERIFY INSTALLATION ==========
print_section "STEP 3: Verify Installation"

echo "Testing imports..."
python3 -c "
import asyncio
import numpy as np
import pandas as pd
import yaml
print('✅ All core imports successful')
"
print_success "Installation verified"

# ========== STEP 4: CREATE CONFIGURATION ==========
print_section "STEP 4: Create Configuration"

if [ ! -f "trading_system/config.yml" ]; then
    echo "Creating default config.yml..."
    python3 -c "from trading_system.config import create_default_config; create_default_config()"
    print_success "Default config created: trading_system/config.yml"
else
    print_warning "Config already exists: trading_system/config.yml"
fi

# ========== STEP 5: RUN DEMO ==========
print_section "STEP 5: Run System Demo"

echo "Running all 5-layer demos..."
python3 trading_system/demo.py

print_success "Demo completed successfully"

# ========== STEP 6: BACKTEST STRATEGY ==========
print_section "STEP 6: Backtest Strategy (Optional)"

read -p "Run backtest? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Backtesting strategy on NVDA (this may take a minute)..."
    python3 trading_system/backtest.py
    print_success "Backtest completed"
else
    print_warning "Backtest skipped"
fi

# ========== STEP 7: PAPER TRADING INSTRUCTIONS ==========
print_section "STEP 7: Ready for Paper Trading"

echo ""
echo "To start paper trading, run:"
echo ""
echo "  cd $PROJECT_ROOT"
echo "  source trading_system_venv/bin/activate"
echo "  python3 trading_system/main.py --config trading_system/config.yml --mode paper"
echo ""
echo "To monitor in Streamlit (in another terminal):"
echo "  streamlit run predictor/app/trading_dashboard.py"
echo ""
echo "To go live (after 2 weeks of successful paper trading):"
echo "  python3 trading_system/main.py --config trading_system/config.yml --mode live"
echo ""

# ========== SUMMARY ==========
print_section "Setup Complete! 🎉"

echo ""
echo "Trading System Summary:"
echo "  Location:     $PROJECT_ROOT/trading_system"
echo "  Config:       $PROJECT_ROOT/trading_system/config.yml"
echo "  Data:         $PROJECT_ROOT/trading_system/data"
echo "  Trades Log:   $PROJECT_ROOT/trading_system/data/trades.csv"
echo "  Docs:         $PROJECT_ROOT/trading_system/README.md"
echo ""
echo "Next Steps:"
echo "  1. Review configuration: nano trading_system/config.yml"
echo "  2. Start paper trading: python3 trading_system/main.py --mode paper"
echo "  3. Monitor trades: tail -f trading_system/data/trades.csv"
echo "  4. After 2 weeks, go live: python3 trading_system/main.py --mode live"
echo ""
echo "========================================"

# Save project path for future reference
echo "$PROJECT_ROOT" > .trading_project_root
print_success "Project path saved"

echo ""
