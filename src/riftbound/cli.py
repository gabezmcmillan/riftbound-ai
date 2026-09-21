"""Run agent-vs-agent Riftbound matches from the command line.

Examples:
    python -m riftbound.cli --games 200 --p0 greedy --p1 random
    python -m riftbound.cli --games 50 --p0 greedy --p1 greedy --swap-decks
"""

from __future__ import annotations

import argparse
import time

from riftbound.agents import GreedyAgent, MCTSAgent, RandomAgent
from riftbound.cards import build_annie_deck, build_yi_deck
from riftbound.sim import run_match


def make_agent(name: str, seed: int, mcts_iterations: int, mcts_worlds: int):
    if name == "random":
        return RandomAgent(seed=seed)
    if name == "greedy":
        return GreedyAgent(seed=seed)
    if name == "mcts":
        return MCTSAgent(
            iterations=mcts_iterations, determinizations=mcts_worlds, seed=seed
        )
    raise SystemExit(f"unknown agent {name!r} (choose: random, greedy, mcts)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--p0", default="greedy", help="agent for player 0 (annie deck)")
    parser.add_argument("--p1", default="random", help="agent for player 1 (yi deck)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=120)
    parser.add_argument(
        "--swap-decks", action="store_true", help="give p0 the yi deck instead"
    )
    parser.add_argument("--mcts-iterations", type=int, default=60)
    parser.add_argument("--mcts-worlds", type=int, default=4)
    args = parser.parse_args()

    decks = (build_annie_deck(), build_yi_deck())
    if args.swap_decks:
        decks = (decks[1], decks[0])
    agents = (
        make_agent(args.p0, args.seed, args.mcts_iterations, args.mcts_worlds),
        make_agent(args.p1, args.seed + 1, args.mcts_iterations, args.mcts_worlds),
    )

    start = time.perf_counter()
    stats = run_match(
        decks, agents, games=args.games, seed=args.seed, max_turns=args.max_turns
    )
    elapsed = time.perf_counter() - start
    print(f"p0={args.p0} ({decks[0].legend_name}) vs p1={args.p1} ({decks[1].legend_name})")
    print(stats.report())
    print(f"elapsed: {elapsed:.1f}s ({args.games / elapsed:.1f} games/s)")


if __name__ == "__main__":
    main()
