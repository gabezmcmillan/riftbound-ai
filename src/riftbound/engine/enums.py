"""Core enums and constants for the Riftbound engine."""

from __future__ import annotations

from enum import Enum


class Domain(str, Enum):
    """The six rune domains (CR 164.1)."""

    FURY = "Fury"
    CALM = "Calm"
    MIND = "Mind"
    BODY = "Body"
    CHAOS = "Chaos"
    ORDER = "Order"


class CardType(str, Enum):
    UNIT = "Unit"
    SPELL = "Spell"
    GEAR = "Gear"


class Phase(str, Enum):
    """Turn phases (CR 315-317). Agents only ever act during MAIN in v0."""

    AWAKEN = "Awaken"
    BEGINNING = "Beginning"
    CHANNEL = "Channel"
    DRAW = "Draw"
    MAIN = "Main"
    ENDING = "Ending"


# Location sentinel: a unit's location is either BASE (its controller's base)
# or a battlefield index (0..n-1). Units can never occupy an enemy base
# (CR 323.7 recalls them during cleanup).
BASE = -1

# 1v1 Duel parameters (CR 485).
VICTORY_SCORE = 8
NUM_BATTLEFIELDS = 2
OPENING_HAND = 4
RUNE_DECK_SIZE = 12
MAIN_DECK_SIZE = 40
