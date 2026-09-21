from __future__ import annotations

import random

from riftbound.agents.base import Agent
from riftbound.engine.actions import Action
from riftbound.engine.game import Game


class RandomAgent(Agent):
    """Picks a uniformly random legal action. Exists to exercise the engine."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def choose(self, game: Game) -> Action:
        return self.rng.choice(game.legal_actions())
