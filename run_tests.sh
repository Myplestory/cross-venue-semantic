#!/bin/bash
# Test runner that handles virtual environment activation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Running setup..."
    ./setup_testing.sh
fi

# Activate virtual environment and run tests
source venv/bin/activate

# Run pytest with all arguments passed to this script
pytest "$@"

