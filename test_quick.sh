#!/bin/bash
# Quick test runner - runs a subset of tests to verify setup

cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run ./setup_testing.sh first"
    exit 1
fi

echo "Running quick test suite..."
echo ""

./venv/bin/pytest discovery/tests/test_types.py discovery/tests/test_dedup.py discovery/tests/test_venue_factory.py -v

echo ""
echo "✅ Quick tests complete!"
echo ""
echo "To run all tests:"
echo "  ./venv/bin/pytest discovery/tests/ -v"
