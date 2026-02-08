#!/usr/bin/env python3
"""
Test runner script for market discovery WebSocket connectors.

Usage:
    python -m discovery.tests.run_all_tests
    python -m discovery.tests.run_all_tests --category unit
    python -m discovery.tests.run_all_tests --coverage
"""

import sys
import subprocess
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def run_tests(category=None, coverage=False, verbose=True):
    """Run tests with specified options."""
    cmd = ["pytest", "discovery/tests/"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=discovery", "--cov-report=html", "--cov-report=term"])
    
    if category:
        if category == "unit":
            cmd.extend([
                "discovery/tests/test_types.py",
                "discovery/tests/test_dedup.py",
                "discovery/tests/test_venue_factory.py",
            ])
        elif category == "connector":
            cmd.extend([
                "discovery/tests/test_kalshi_connector.py",
                "discovery/tests/test_polymarket_connector.py",
                "discovery/tests/test_base_connector.py",
            ])
        elif category == "integration":
            cmd.extend([
                "discovery/tests/test_websocket_connections.py",
                "discovery/tests/test_integration_full_flow.py",
            ])
        elif category == "edge":
            cmd.append("discovery/tests/test_message_parsing_edge_cases.py")
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent)
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description="Run market discovery tests")
    parser.add_argument(
        "--category",
        choices=["unit", "connector", "integration", "edge", "all"],
        default="all",
        help="Test category to run"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage report"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less verbose output"
    )
    
    args = parser.parse_args()
    
    category = None if args.category == "all" else args.category
    exit_code = run_tests(
        category=category,
        coverage=args.coverage,
        verbose=not args.quiet
    )
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
