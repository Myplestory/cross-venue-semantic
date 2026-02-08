"""
Test suite summary and test runner.

Run all tests:
    pytest discovery/tests/ -v

Run specific test categories:
    pytest discovery/tests/test_types.py -v
    pytest discovery/tests/test_dedup.py -v
    pytest discovery/tests/test_venue_factory.py -v
    pytest discovery/tests/test_websocket_connections.py -v
    pytest discovery/tests/test_integration_full_flow.py -v
    pytest discovery/tests/test_message_parsing_edge_cases.py -v
"""

import pytest


def test_test_suite_imports():
    """Verify all test modules can be imported."""
    try:
        from discovery.tests import test_types
        from discovery.tests import test_dedup
        from discovery.tests import test_venue_factory
        from discovery.tests import test_websocket_connections
        from discovery.tests import test_integration_full_flow
        from discovery.tests import test_message_parsing_edge_cases
        from discovery.tests import test_kalshi_connector
        from discovery.tests import test_polymarket_connector
        from discovery.tests import test_base_connector
    except ImportError as e:
        pytest.fail(f"Failed to import test modules: {e}")


def test_test_coverage_summary():
    """Print test coverage summary."""
    test_modules = [
        "test_types",
        "test_dedup",
        "test_venue_factory",
        "test_websocket_connections",
        "test_integration_full_flow",
        "test_message_parsing_edge_cases",
        "test_kalshi_connector",
        "test_polymarket_connector",
        "test_base_connector",
    ]
    
    print(f"\nTest Suite Coverage:")
    print(f"Total test modules: {len(test_modules)}")
    print(f"Modules: {', '.join(test_modules)}")
    
    # This test always passes, it's just for documentation
    assert True

