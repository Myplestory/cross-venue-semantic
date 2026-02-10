# Golden Dataset Guide

## Overview

The golden dataset contains known equivalent/non-equivalent market pairs for validation. This guide explains how to create and maintain it.

## Current Status

**Synthetic Dataset**: The current `fixtures/golden_dataset.py` contains synthetic test cases that represent realistic scenarios. These are sufficient for unit and integration testing.

**Production Dataset**: For production validation, you should manually curate real market pairs from Kalshi/Polymarket.

---

## How to Generate a Real Golden Dataset

### Option 1: Manual Curation (Recommended)

1. **Identify Known Equivalent Markets**
   - Find markets on Kalshi and Polymarket that are clearly the same event
   - Example: "Will Bitcoin reach $100k by 2025?" on both venues
   - Manually verify they resolve to the same outcome

2. **Identify Known Non-Equivalent Markets**
   - Find markets that are similar but different
   - Example: "Bitcoin > $60k" vs "Bitcoin > $80k"
   - Example: "Bitcoin > $100k" vs "Ethereum > $10k"

3. **Identify Ambiguous Cases**
   - Find markets that are similar but have subtle differences
   - Example: Same event but different resolution dates
   - Example: Same threshold but different data sources

4. **Extract ContractSpecs**
   - Run your extraction pipeline on each market
   - Store the ContractSpec for each market

5. **Add to Dataset**
   - Add entries to `fixtures/golden_dataset.py`
   - Include expected verdict and confidence range
   - Document why the pair is equivalent/non-equivalent

### Option 2: Semi-Automated (Future)

You could create a script that:
1. Scrapes markets from both venues
2. Uses your matching pipeline to find candidate pairs
3. Presents them for manual review
4. Exports verified pairs to the golden dataset

---

## Dataset Structure

Each entry in the golden dataset should have:

```python
{
    "name": "descriptive_name",
    "spec_a": ContractSpec(...),  # Market A
    "spec_b": ContractSpec(...),  # Market B
    "expected_verdict": "equivalent" | "not_equivalent" | "needs_review",
    "expected_confidence_min": 0.85,  # Optional
    "expected_confidence_max": 0.95,  # Optional
    "expected_outcome_mapping": {"Yes": "YES", "No": "NO"},  # Optional
    "notes": "Why this pair is equivalent/non-equivalent"  # Optional
}
```

---

## Best Practices

1. **Diversity**: Include various market types (crypto, stocks, politics, sports)
2. **Edge Cases**: Include boundary conditions (missing fields, partial matches)
3. **Real Examples**: Use actual market pairs when possible
4. **Documentation**: Document why each pair is classified as it is
5. **Regular Updates**: Update dataset as you find new patterns

---

## Example: Adding a Real Market Pair

```python
{
    "name": "real_bitcoin_100k_kalshi_polymarket",
    "spec_a": ContractSpec(
        # Extracted from Kalshi market ID: KALSHI-12345
        statement="Will Bitcoin close above $100,000 on Coinbase by Dec 31, 2025?",
        resolution_date=DateSpec(date=datetime(2025, 12, 31), is_deadline=True),
        entities=[EntitySpec(name="Bitcoin", entity_type="other", aliases=["BTC"])],
        thresholds=[ThresholdSpec(value=100000.0, unit="dollars", comparison=">")],
        data_source="Coinbase",
        outcome_labels=["Yes", "No"]
    ),
    "spec_b": ContractSpec(
        # Extracted from Polymarket market ID: 0xabc123...
        statement="Will BTC exceed $100k USD on Coinbase before December 31, 2025?",
        resolution_date=DateSpec(date=datetime(2025, 12, 31), is_deadline=True),
        entities=[EntitySpec(name="BTC", entity_type="other", aliases=["Bitcoin"])],
        thresholds=[ThresholdSpec(value=100000.0, unit="dollars", comparison=">")],
        data_source="Coinbase",
        outcome_labels=["YES", "NO"]
    ),
    "expected_verdict": "equivalent",
    "expected_confidence_min": 0.90,
    "expected_outcome_mapping": {"Yes": "YES", "No": "NO"},
    "notes": "Same event, same threshold, same date, same data source. Only wording differs."
}
```

---

## Validation

Run the integration tests to validate your golden dataset:

```bash
pytest matching/tests/test_integration_pair_verifier.py -v -m integration
```

All tests should pass with your curated pairs.

---

## Maintenance

- **Weekly**: Review new market pairs and add interesting cases
- **Monthly**: Validate dataset against current pipeline performance
- **Quarterly**: Remove outdated pairs, add new patterns

