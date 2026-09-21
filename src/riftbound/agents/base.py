from __future__ import annotations

from abc import ABC, abstractmethod

from riftbound.engine.actions import Action
from riftbound.engine.game import Game


class Agent(ABC):
    """An agent chooses one legal action at a time during its Main Phase."""

    @abstractmethod
    def choose(self, game: Game) -> Action:
        raise NotImplementedError
