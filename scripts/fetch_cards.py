"""Fetch the full Riftbound card database from the RiftScribe community API.

RiftScribe (https://riftscribe.gg/api-docs) is a free, no-auth REST API for
Riftbound card data. The list endpoint omits rules text, so this script first
pages through /api/cards to collect ids, then fetches each card's detail
record (which includes `description` rules text, `keywords`, and `tags`) and
writes everything to data/cards.json.

Usage:
    python scripts/fetch_cards.py [--out data/cards.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://riftscribe.gg/api"
PAGE_SIZE = 100


def get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "riftbound-ai/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def fetch_ids() -> list[str]:
    ids: list[str] = []
    offset = 0
    while True:
        items = get_json(f"{BASE}/cards?limit={PAGE_SIZE}&offset={offset}")
        if not items:
            break
        ids.extend(item["id"] for item in items)
        if len(items) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(0.1)
    return ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/cards.json")
    args = parser.parse_args()

    ids = fetch_ids()
    print(f"found {len(ids)} cards", flush=True)

    cards: list[dict] = []
    for i, card_id in enumerate(ids, 1):
        try:
            card = get_json(f"{BASE}/cards/{card_id}")
        except Exception as exc:  # noqa: BLE001 - log and continue
            print(f"FAILED {card_id}: {exc}", flush=True)
            continue
        # Drop bulky presentation-only fields.
        for key in ("image_blur_data_url", "art", "image_thumb"):
            card.pop(key, None)
        cards.append(card)
        if i % 50 == 0:
            print(f"{i}/{len(ids)}", flush=True)
        time.sleep(0.05)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cards, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(cards)} cards to {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
