# riftbound-ai

An AI that learns to play [Riftbound](https://playriftbound.com/) (the League
of Legends TCG), built simulator-first: a Python rules engine, scripted
baseline agents, and (coming) self-play reinforcement learning.

## Why a simulator?

Riftbound has no digital client, so there is nothing to screen-scrape — and a
turn-based card game doesn't need pixels anyway. A rules engine lets agents
play millions of games per hour instead of real-time. See
`docs/engine-design.md` for the architecture and the current rules-fidelity
status, and `docs/rules-reference.md` for the full Core Rules the engine is
built against.

## Quick start

```powershell
python -m pip install -e ".[dev]"
python -m pytest                                   # engine tests
python -m riftbound.cli --games 100 --p0 greedy --p1 random
```

## What exists today (v0)

- **Rules engine** (`src/riftbound/engine/`): 1v1 Duel — full turn structure
  (awaken/hold-scoring/channel/draw/main/ending), rune economy (exhaust for
  Energy, recycle for Power), standard moves, showdown/combat resolution with
  correct damage-assignment rules (Tank, lethal-fill), conquer/hold scoring
  including the final-point restriction, and Burn Out. Agents interact via
  `Game.legal_actions()` / `Game.step(action)`.
  The chain/reaction system is deliberately deferred; see the design doc.
- **Card set** (`src/riftbound/cards/`): two playable 40-card decks of real
  cards (an Annie/Fury burn deck and a Master Yi Calm/Body defensive deck)
  expressed in a data-driven effect vocabulary.
- **Card database** (`data/cards.json`): all 1,180 cards with rules text,
  fetched from the community RiftScribe API (`scripts/fetch_cards.py`).
- **Agents** (`src/riftbound/agents/`): random (engine fuzzing), one-ply
  greedy (the scripted baseline; beats random 20–0), and determinized MCTS
  (samples worlds consistent with hidden information, runs UCT in each, and
  aggregates root visits). Try it:
  `python -m riftbound.cli --games 10 --p0 mcts --p1 greedy`.
- **Simulator** (`src/riftbound/sim.py`, `src/riftbound/cli.py`):
  head-to-head matches with stats; ~500 games/s for random agents.
- **Encodings** (`src/riftbound/encoding.py`): fixed-size observation
  tensors from one player's perspective (opponent hand/decks hidden), and a
  flat discrete action table with a legality mask (`ActionCodec`) — the
  interface layer for search and RL agents.

## Roadmap

1. ~~Rules engine core + starter decks + scripted agents~~ (done)
2. ~~State/action tensor encodings with legality masks~~ (done)
3. ~~Determinized MCTS (hidden information via sampled worlds)~~ (done)
4. Self-play RL (PPO or AlphaZero-style policy+value net)
5. Reaction/chain system; broader card coverage
