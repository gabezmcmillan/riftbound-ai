"""Card definitions and the card registry.

Cards are static data (`CardDef`). Card *effects* are expressed as a small
data-driven instruction vocabulary (`Instruction`) so that agents can
enumerate targets and the engine can execute effects without per-card code.

v0 effect vocabulary (see docs/engine-design.md for the roadmap):
  - Spells: an ordered list of instructions, some of which consume a target
    (a unit or a battlefield) chosen when the spell is played.
  - Units and gear: an optional list of NON-targeted instructions executed
    when played ("When you play me, ..."). Keeping permanent triggers
    target-free keeps the action space simple in v0.
  - Resource gear ("exhaust: Add 1 power of domain X") covers the Seal cycle.
  - Combat keywords: Tank, Assault N, Shield N, plus a few named passives
    used by the starter subset (see field comments).
"""

from __future__ import annotations

from dataclasses import dataclass

from riftbound.engine.enums import CardType, Domain

# ---------------------------------------------------------------------------
# Effect vocabulary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetSpec:
    """Describes what a targeted instruction may choose."""

    kind: str = "unit"  # "unit" | "battlefield"
    owner: str = "any"  # for units: "friendly" | "enemy" | "any"
    at_battlefield_only: bool = False


@dataclass(frozen=True)
class Instruction:
    """One executable step of a card effect.

    Targeted ops (consume one entry from the action's targets tuple):
      damage <amount>            -> unit
      kill                       -> unit
      buff <amount>              -> unit (+might this turn)
      buff_assault <amount>      -> unit (+Assault this turn)
      damage_all_at_bf <amount>  -> battlefield (hits enemy units there)

    Non-targeted ops:
      draw <amount>
      channel <amount>
      ready_runes <amount>
      gain_point <amount>
      damage_all_bf_units <amount>  (all units at all battlefields)
    """

    op: str
    amount: int = 0
    target: TargetSpec | None = None


# ---------------------------------------------------------------------------
# Card definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CardDef:
    id: str
    name: str
    type: CardType
    energy: int = 0
    # Power cost as (domain, count) pairs.
    power: tuple[tuple[Domain, int], ...] = ()
    domains: tuple[Domain, ...] = ()
    might: int = 0
    tags: tuple[str, ...] = ()
    # Effects executed when played (spells: full effect; units/gear:
    # "when you play me" triggers, which must be non-targeted in v0).
    instructions: tuple[Instruction, ...] = ()
    # Resource gear: exhaust to add 1 power of this domain (Seal cycle).
    gear_power_domain: Domain | None = None
    is_champion_unit: bool = False

    # --- combat keywords (CR 800s) ---
    tank: bool = False  # must be assigned combat damage first
    assault: int = 0  # +N might while attacking
    shield: int = 0  # +N might while defending
    enters_ready: bool = False  # "I enter ready"
    ganking: bool = False  # may standard-move battlefield -> battlefield

    # --- named passives used by the starter subset ---
    # Annie, Fiery: your spells/abilities deal +N bonus damage per instance.
    bonus_spell_damage: int = 0
    # Yi, Meditative: +N might in combat while you control 8+ runes.
    might_bonus_if_runes8: int = 0
    # Wielder of Water: +N might while attacking or defending alone.
    might_bonus_if_alone: int = 0

    @property
    def power_total(self) -> int:
        return sum(count for _, count in self.power)

    def targeted_instructions(self) -> tuple[Instruction, ...]:
        return tuple(ins for ins in self.instructions if ins.target is not None)


@dataclass(frozen=True)
class BattlefieldDef:
    """A battlefield card. v0 supports two simple trigger shapes."""

    id: str
    name: str
    # Executed by the scoring player when they Hold here (non-targeted).
    on_hold: tuple[Instruction, ...] = ()
    # "When a unit moves from here, give it +N might this turn."
    move_from_buff: int = 0


# Global registries keyed by card id.
REGISTRY: dict[str, CardDef] = {}
BATTLEFIELDS: dict[str, BattlefieldDef] = {}


def register(card_def: CardDef) -> CardDef:
    if card_def.id in REGISTRY:
        raise ValueError(f"duplicate card id: {card_def.id}")
    REGISTRY[card_def.id] = card_def
    return card_def


def register_battlefield(bf_def: BattlefieldDef) -> BattlefieldDef:
    if bf_def.id in BATTLEFIELDS:
        raise ValueError(f"duplicate battlefield id: {bf_def.id}")
    BATTLEFIELDS[bf_def.id] = bf_def
    return bf_def


def card(card_id: str) -> CardDef:
    return REGISTRY[card_id]


def battlefield(bf_id: str) -> BattlefieldDef:
    return BATTLEFIELDS[bf_id]
