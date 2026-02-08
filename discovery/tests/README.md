# Market Discovery Test Suite

Comprehensive unit test suite for WebSocket market discovery connectors.

## Quick Start

```bash
# Run all tests
cd semantic_pipeline
pytest discovery/tests/ -v

# Run with coverage
pytest discovery/tests/ --cov=discovery --cov-report=html

# Run specific category
pytest discovery/tests/test_websocket_connections.py -v
```

## Test Files Overview

| File | Tests | Description |
|------|-------|-------------|
| `test_types.py` | Type validation | MarketEvent, OutcomeSpec, enums |
| `test_dedup.py` | Deduplication | Identity hashing, duplicate detection |
| `test_venue_factory.py` | Factory pattern | Connector creation, registration |
| `test_kalshi_connector.py` | Kalshi connector | Message parsing, connection |
| `test_polymarket_connector.py` | Polymarket connector | Message parsing, connection |
| `test_base_connector.py` | Base connector | Connection lifecycle, streaming |
| `test_websocket_connections.py` | WebSocket integration | Real message formats, connections |
| `test_integration_full_flow.py` | End-to-end | Complete discovery flow |
| `test_message_parsing_edge_cases.py` | Edge cases | Robustness, error handling |
| `test_integration_websocket.py` | Mock server | Integration with mock server |
| `test_websocket_server.py` | Mock server | Mock WebSocket server implementation |
| `test_suite_summary.py` | Test suite | Test suite overview |

## Test Categories

### 1. Unit Tests
- **Types**: `test_types.py` - Type definitions and validation
- **Deduplication**: `test_dedup.py` - Hash-based deduplication
- **Factory**: `test_venue_factory.py` - Venue connector factory

### 2. Connector Tests
- **Kalshi**: `test_kalshi_connector.py` - Kalshi-specific parsing
- **Polymarket**: `test_polymarket_connector.py` - Polymarket-specific parsing
- **Base**: `test_base_connector.py` - Base connector functionality

### 3. Integration Tests
- **WebSocket**: `test_websocket_connections.py` - Real message formats
- **Full Flow**: `test_integration_full_flow.py` - End-to-end scenarios
- **Mock Server**: `test_integration_websocket.py` - Mock server integration

### 4. Robustness Tests
- **Edge Cases**: `test_message_parsing_edge_cases.py` - Error handling, malformed data

## Running Tests

### All Tests
```bash
pytest discovery/tests/ -v
```

### By Category
```bash
# Unit tests
pytest discovery/tests/test_types.py discovery/tests/test_dedup.py discovery/tests/test_venue_factory.py -v

# Connector tests
pytest discovery/tests/test_kalshi_connector.py discovery/tests/test_polymarket_connector.py -v

# Integration tests
pytest discovery/tests/test_websocket_connections.py discovery/tests/test_integration_full_flow.py -v
```

### With Coverage
```bash
pytest discovery/tests/ --cov=discovery --cov-report=html --cov-report=term
```

## Test Features

✅ **Comprehensive Coverage**
- All connector methods tested
- All message formats tested
- Edge cases covered
- Error handling verified

✅ **No External Dependencies**
- All tests use mocks
- No real WebSocket connections required
- No API credentials needed
- Fast execution

✅ **Real Message Formats**
- Tests use actual venue message formats
- Based on real API documentation
- Validates normalization

✅ **Integration Scenarios**
- Full discovery flow
- Multiple venues parallel
- Deduplication in flow
- Error recovery

## Documentation

- **[TEST_SUITE.md](TEST_SUITE.md)** - Comprehensive test suite documentation
- **[TESTING.md](TESTING.md)** - Testing guide and manual testing
- **[QUICK_START.md](QUICK_START.md)** - Quick reference

## Test Statistics

- **Total Test Files**: 12
- **Test Categories**: 4 (Unit, Connector, Integration, Robustness)
- **Coverage**: All major components tested

## CI/CD Integration

Tests are designed for CI/CD:

```bash
# Fast unit tests (no network)
pytest discovery/tests/ \
    --ignore=discovery/tests/test_integration_websocket.py \
    --cov=discovery \
    --cov-report=xml
```
