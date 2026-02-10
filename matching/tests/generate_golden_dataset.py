"""
Helper script to generate golden dataset entries from real markets.

Usage:
    python generate_golden_dataset.py --market-a-id KALSHI-12345 --market-b-id 0xabc123...
    
This script:
1. Fetches markets from both venues
2. Extracts ContractSpecs
3. Prompts for manual verification
4. Generates golden dataset entry
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from canonicalization.contract_spec import ContractSpec
from extraction.spec_extractor import ContractSpecExtractor


async def extract_spec_from_market(market_id: str, venue: str) -> ContractSpec:
    """
    Extract ContractSpec from a market.
    
    Args:
        market_id: Market ID from venue
        venue: "kalshi" or "polymarket"
        
    Returns:
        ContractSpec
    """
    # TODO: Implement market fetching and extraction
    # For now, this is a placeholder
    raise NotImplementedError(
        "Market fetching not yet implemented. "
        "Manually extract ContractSpecs and add to golden_dataset.py"
    )


def generate_dataset_entry(spec_a: ContractSpec, spec_b: ContractSpec) -> dict:
    """
    Generate a golden dataset entry from two ContractSpecs.
    
    Prompts user for expected verdict and confidence.
    """
    print("\n" + "="*80)
    print("GOLDEN DATASET ENTRY GENERATOR")
    print("="*80)
    
    print("\nContractSpec A:")
    print(f"  Statement: {spec_a.statement}")
    print(f"  Entities: {[e.name for e in spec_a.entities]}")
    print(f"  Thresholds: {[f'{t.value} {t.unit}' for t in spec_a.thresholds]}")
    print(f"  Date: {spec_a.resolution_date.date if spec_a.resolution_date else 'None'}")
    
    print("\nContractSpec B:")
    print(f"  Statement: {spec_b.statement}")
    print(f"  Entities: {[e.name for e in spec_b.entities]}")
    print(f"  Thresholds: {[f'{t.value} {t.unit}' for t in spec_b.thresholds]}")
    print(f"  Date: {spec_b.resolution_date.date if spec_b.resolution_date else 'None'}")
    
    print("\n" + "-"*80)
    print("Please classify this pair:")
    print("  1. equivalent - Same event, should resolve to same outcome")
    print("  2. not_equivalent - Different events")
    print("  3. needs_review - Ambiguous, requires manual review")
    print("-"*80)
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    verdict_map = {
        "1": "equivalent",
        "2": "not_equivalent",
        "3": "needs_review"
    }
    
    verdict = verdict_map.get(choice, "needs_review")
    
    print(f"\nExpected verdict: {verdict}")
    
    if verdict == "equivalent":
        confidence_min = float(input("Expected confidence minimum (0.0-1.0, default 0.85): ") or "0.85")
        confidence_max = float(input("Expected confidence maximum (0.0-1.0, default 0.95): ") or "0.95")
    elif verdict == "not_equivalent":
        confidence_max = float(input("Expected confidence maximum (0.0-1.0, default 0.5): ") or "0.5")
        confidence_min = None
    else:
        confidence_min = float(input("Expected confidence minimum (0.0-1.0, default 0.5): ") or "0.5")
        confidence_max = float(input("Expected confidence maximum (0.0-1.0, default 0.9): ") or "0.9")
    
    name = input("\nEnter descriptive name for this pair: ").strip() or "unnamed_pair"
    notes = input("Enter notes (optional): ").strip()
    
    entry = {
        "name": name,
        "spec_a": spec_a,
        "spec_b": spec_b,
        "expected_verdict": verdict
    }
    
    if confidence_min is not None:
        entry["expected_confidence_min"] = confidence_min
    if confidence_max is not None:
        entry["expected_confidence_max"] = confidence_max
    if notes:
        entry["notes"] = notes
    
    return entry


def format_entry_as_code(entry: dict) -> str:
    """Format entry as Python code for golden_dataset.py."""
    # This is a simplified version - in practice, you'd need to serialize
    # ContractSpec objects properly
    return f"""
    {{
        "name": "{entry['name']}",
        "spec_a": ContractSpec(...),  # TODO: Serialize properly
        "spec_b": ContractSpec(...),  # TODO: Serialize properly
        "expected_verdict": "{entry['expected_verdict']}",
        "expected_confidence_min": {entry.get('expected_confidence_min', 'None')},
        "expected_confidence_max": {entry.get('expected_confidence_max', 'None')},
        "notes": "{entry.get('notes', '')}"
    }},"""


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate golden dataset entry from real markets")
    parser.add_argument("--market-a-id", required=True, help="Market A ID")
    parser.add_argument("--market-b-id", required=True, help="Market B ID")
    parser.add_argument("--venue-a", default="kalshi", choices=["kalshi", "polymarket"])
    parser.add_argument("--venue-b", default="polymarket", choices=["kalshi", "polymarket"])
    
    args = parser.parse_args()
    
    print("⚠️  Market fetching not yet implemented.")
    print("Please manually extract ContractSpecs and add to golden_dataset.py")
    print("\nSee GOLDEN_DATASET_GUIDE.md for instructions.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

