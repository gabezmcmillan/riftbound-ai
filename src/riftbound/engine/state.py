"""Mutable game state containers."""

from __future__ import annotations

from dataclasses import dataclass, field

from riftbound.engine.cards import BattlefieldDef, CardDef
from riftbound.engine.enums import BASE, Domain, Phase


@dataclass
class UnitState:
    uid: int
    card: CardDef
    controller: int
    location: int = BASE  # BASE or battlefield index
    ready: bool = False  # units enter the board exhausted (CR 143.4)
    damage: int = 0
    temp_might: int = 0  # "+X might this turn" buffs; expire in Ending Phase
    temp_assault: int = 0  # "+X Assault this turn"; expires in Ending Phase

    @property
    def might(self) -> int:
        # Might below 0 is treated as 0 (CR 143.2.b).
        return max(0, self.card.might + self.temp_might)

    @property
    def lethal_remaining(self) -> int:
        """Damage still needed for lethal (CR 142.4.b: non-zero, >= might)."""
        return max(1, self.might) - self.damage


@dataclass
class RuneState:
    uid: int
    domain: Domain
    controller: int
    ready: bool = True


@dataclass
class GearState:
    uid: int
    card: CardDef
    controller: int
    ready: bool = True


@dataclass
class BattlefieldState:
    index: int
    card: BattlefieldDef
    owner: int  # the player who brought this battlefield
    controller: int | None = None
    # Players who have scored this battlefield this turn (CR 470: once per
    # battlefield per player per turn). Cleared at the start of every turn.
    scored_this_turn: set[int] = field(default_factory=set)


@dataclass
class PlayerState:
    index: int
    legend_name: str
    legend_domains: tuple[Domain, ...]
    main_deck: list[str] = field(default_factory=list)  # card ids; last = top
    hand: list[str] = field(default_factory=list)
    trash: list[str] = field(default_factory=list)
    rune_deck: list[Domain] = field(default_factory=list)  # last = top
    champion: str | None = None  # card id in the Champion Zone
    points: int = 0
    energy_pool: int = 0
    power_pool: dict[Domain, int] = field(default_factory=dict)

    def clear_pools(self) -> None:
        self.energy_pool = 0
        self.power_pool = {}


@dataclass
class GameState:
    players: list[PlayerState]
    battlefields: list[BattlefieldState]
    units: dict[int, UnitState] = field(default_factory=dict)
    runes: dict[int, RuneState] = field(default_factory=dict)
    gear: dict[int, GearState] = field(default_factory=dict)
    turn_player: int = 0
    turn_number: int = 1
    phase: Phase = Phase.MAIN
    winner: int | None = None
    next_uid: int = 1

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def new_uid(self) -> int:
        uid = self.next_uid
        self.next_uid += 1
        return uid

    def player_units(self, player: int) -> list[UnitState]:
        return [u for u in self.units.values() if u.controller == player]

    def units_at(self, player: int, location: int) -> list[UnitState]:
        return [
            u
            for u in self.units.values()
            if u.controller == player and u.location == location
        ]

    def player_runes(self, player: int) -> list[RuneState]:
        return [r for r in self.runes.values() if r.controller == player]

    def player_gear(self, player: int) -> list[GearState]:
        return [g for g in self.gear.values() if g.controller == player]
