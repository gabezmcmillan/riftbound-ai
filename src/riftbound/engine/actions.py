"""Agent-facing actions.

All agent decisions in v0 happen during the turn player's Main Phase in a
neutral open state. Start-of-turn and end-of-turn bookkeeping is automatic.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Action:
    pass


@dataclass(frozen=True)
class EndTurn(Action):
    pass


@dataclass(frozen=True)
class PlayUnit(Action):
    """Play a unit from hand (or the Champion Zone) to base or a controlled
    battlefield. `hand_index` of -1 means the Champion Zone."""

    hand_index: int
    destination: int  # BASE or battlefield index


@dataclass(frozen=True)
class PlaySpell(Action):
    """Play a spell from hand with targets (unit uids) for its targeted
    instructions, in instruction order."""

    hand_index: int
    targets: tuple[int, ...] = ()


@dataclass(frozen=True)
class PlayGear(Action):
    hand_index: int


@dataclass(frozen=True)
class MoveUnits(Action):
    """Standard Move (CR 144): exhaust ready units and move them to a common
    destination. Moves to a battlefield may trigger a showdown/combat."""

    unit_uids: tuple[int, ...]
    destination: int  # BASE or battlefield index
