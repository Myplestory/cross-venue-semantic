#!/bin/bash
# Setup script for testing environment

set -e

echo "=========================================="
echo "Setting up Market Discovery Test Environment"
echo "=========================================="
echo ""

cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Install pytest and testing dependencies
echo ""
echo "Installing testing dependencies..."
pip install pytest pytest-asyncio pytest-mock pytest-cov

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests:"
echo "  pytest discovery/tests/ -v"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""

