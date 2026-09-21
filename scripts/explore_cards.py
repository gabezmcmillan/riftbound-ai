"""Quick exploration helper: list the simplest cards per domain/type."""

import json
import sys

cards = json.load(open("data/cards.json", encoding="utf-8"))
base = [c for c in cards if c["set_id"] == "OGN" and not c["variant"]]

domains = sys.argv[1:] or ["fury", "calm"]
for dom in domains:
    print("=" * 30, dom.upper())
    pool = [c for c in base if c["faction"] == dom and c["type"] in ("Unit", "Spell", "Gear")]
    pool.sort(key=lambda c: len(c.get("description") or ""))
    for c in pool[:34]:
        st = c["stats"]
        desc = (c.get("description") or "").replace("\n", " | ")
        print(
            f"{c['type'][:2]:2} {c['public_code'][:8]:9} {c['name'][:28]:28} "
            f"E{st['energy']} P{st['power']} M{st['might']} "
            f"KW{c.get('keywords')} TAGS{c.get('tags')} :: {desc[:150]}"
        )
