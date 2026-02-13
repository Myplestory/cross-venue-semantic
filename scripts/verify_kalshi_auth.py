"""
Verify Kalshi API key and signature with a signed REST request.

Run from repo root:
  python -m scripts.verify_kalshi_auth
  # or
  cd semantic_pipeline && python scripts/verify_kalshi_auth.py

Uses the same key and signing as the WebSocket connector. If this returns 200,
your key is valid and 401 on WebSocket may be due to demo vs production URL mismatch.
"""

import os
import sys
from pathlib import Path

# Ensure discovery is importable; load .env without importing config (avoids Unicode print on Windows)
_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))
try:
    from dotenv import load_dotenv
    load_dotenv(_repo / ".env")
except ImportError:
    pass

from discovery.kalshi_poller import _load_kalshi_private_key, _sign_kalshi_request


def main() -> None:
    key_id = os.getenv("KALSHI_API_KEY_ID")
    key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    key_pem = os.getenv("KALSHI_PRIVATE_KEY")
    use_demo = os.getenv("KALSHI_USE_DEMO", "").lower() in ("true", "1", "yes", "on")

    if not key_id:
        print("KALSHI_API_KEY_ID not set. Set it in .env or environment.")
        sys.exit(1)
    if not key_path and not key_pem:
        print("Set KALSHI_PRIVATE_KEY_PATH or KALSHI_PRIVATE_KEY in .env")
        sys.exit(1)

    private_key = _load_kalshi_private_key(path=key_path, pem=key_pem)
    if not private_key:
        print("Failed to load private key. Check path/PEM and that cryptography is installed.")
        sys.exit(1)

    base = "https://demo-api.kalshi.co" if use_demo else "https://api.elections.kalshi.com"
    path = "/trade-api/v2/exchange/status"
    method = "GET"

    result = _sign_kalshi_request(private_key, method, path)
    if not result:
        print("Signing failed.")
        sys.exit(1)
    sig_b64, timestamp_ms = result

    headers = {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        "KALSHI-ACCESS-SIGNATURE": sig_b64,
    }

    try:
        import urllib.request
        from urllib.error import HTTPError
        req = urllib.request.Request(base + path, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode()
            print(f"OK {resp.status} (key and signature accepted)")
            print(body[:500] + ("..." if len(body) > 500 else ""))
    except HTTPError as e:
        body = e.read().decode() if e.fp else ""
        print(f"HTTP {e.code} {e.reason}")
        print(body[:500] if body else "")
        sys.exit(1)
    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
