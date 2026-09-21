"""Run head-to-head matches between agents."""

from __future__ import annotations

from dataclasses import dataclass, field

from riftbound.agents.base import Agent
from riftbound.engine.game import Deck, Game


@dataclass
class MatchStats:
    games: int = 0
    wins: list[int] = field(default_factory=lambda: [0, 0])
    draws: int = 0
    total_turns: int = 0
    total_points: list[int] = field(default_factory=lambda: [0, 0])

    def report(self) -> str:
        lines = [
            f"games:  {self.games}",
            f"p0 wins: {self.wins[0]} ({100 * self.wins[0] / self.games:.1f}%)",
            f"p1 wins: {self.wins[1]} ({100 * self.wins[1] / self.games:.1f}%)",
            f"draws (turn limit): {self.draws}",
            f"avg turns: {self.total_turns / self.games:.1f}",
            f"avg points: p0 {self.total_points[0] / self.games:.2f}, "
            f"p1 {self.total_points[1] / self.games:.2f}",
        ]
        return "\n".join(lines)


def play_game(
    decks: tuple[Deck, Deck],
    agents: tuple[Agent, Agent],
    seed: int | None = None,
    first_player: int | None = None,
    max_turns: int = 120,
) -> Game:
    game = Game(decks, seed=seed, first_player=first_player, max_turns=max_turns)
    while not game.is_over:
        agent = agents[game.state.turn_player]
        game.step(agent.choose(game))
    return game


def run_match(
    decks: tuple[Deck, Deck],
    agents: tuple[Agent, Agent],
    games: int = 100,
    seed: int = 0,
    max_turns: int = 120,
) -> MatchStats:
    stats = MatchStats()
    for i in range(games):
        game = play_game(
            decks,
            agents,
            seed=seed + i,
            first_player=i % 2,  # alternate who goes first
            max_turns=max_turns,
        )
        stats.games += 1
        if game.winner is None:
            stats.draws += 1
        else:
            stats.wins[game.winner] += 1
        stats.total_turns += game.state.turn_number
        for p in range(2):
            stats.total_points[p] += game.state.players[p].points
    return stats
