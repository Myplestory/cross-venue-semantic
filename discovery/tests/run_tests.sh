#!/bin/bash
# Test runner script for market discovery WebSocket connectors

set -e

echo "=========================================="
echo "Market Discovery WebSocket Test Suite"
echo "=========================================="
echo ""

# Change to semantic_pipeline directory
cd "$(dirname "$0")/.."

# Run all tests
echo "Running all tests..."
pytest discovery/tests/ -v --tb=short

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo ""
echo "Test categories:"
echo "  - Unit tests (types, dedup, factory)"
echo "  - Connector tests (Kalshi, Polymarket)"
echo "  - WebSocket connection tests"
echo "  - Integration tests (full flow)"
echo "  - Edge case tests (message parsing)"
echo ""
echo "To run specific test categories:"
echo "  pytest discovery/tests/test_types.py -v"
echo "  pytest discovery/tests/test_venue_factory.py -v"
echo "  pytest discovery/tests/test_websocket_connections.py -v"
echo "  pytest discovery/tests/test_integration_full_flow.py -v"
echo ""

