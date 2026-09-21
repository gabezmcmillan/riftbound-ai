"""The v0 starter card subset: real Riftbound cards, transcribed from the
community card database (data/cards.json), restricted to effects the v0
engine can represent.

Two ready-to-play decks are defined here:
  - "Annie burn" (Fury): aggressive units plus direct-damage spells.
  - "Yi wall" (Calm/Body): defensive keywords (Shield/Tank) and combat math.

Per-card simplifications are marked with `# SIMPLIFIED:` comments. Cards
whose text is fully captured by the v0 vocabulary carry no comment.
"""

from __future__ import annotations

from riftbound.engine.cards import (
    BattlefieldDef,
    CardDef,
    Instruction,
    TargetSpec,
    register,
    register_battlefield,
)
from riftbound.engine.enums import CardType, Domain
from riftbound.engine.game import Deck

UNIT = CardType.UNIT
SPELL = CardType.SPELL
GEAR = CardType.GEAR

ANY_UNIT = TargetSpec(kind="unit", owner="any")
ANY_UNIT_AT_BF = TargetSpec(kind="unit", owner="any", at_battlefield_only=True)
FRIENDLY_UNIT = TargetSpec(kind="unit", owner="friendly")
ANY_BATTLEFIELD = TargetSpec(kind="battlefield")

# ---------------------------------------------------------------------------
# Fury (Annie)
# ---------------------------------------------------------------------------

register(CardDef(
    id="OGN-010", name="Legion Rearguard", type=UNIT,
    energy=2, might=2, domains=(Domain.FURY,),
    # SIMPLIFIED: Accelerate (optional pay-to-enter-ready) not implemented.
))
register(CardDef(
    id="OGN-001", name="Blazing Scorcher", type=UNIT,
    energy=5, might=5, domains=(Domain.FURY,),
    # SIMPLIFIED: Accelerate not implemented.
))
register(CardDef(
    id="OGN-012", name="Noxus Hopeful", type=UNIT,
    energy=4, might=4, domains=(Domain.FURY,),
    # SIMPLIFIED: Legion cost reduction not implemented.
))
register(CardDef(
    id="OGN-013", name="Pouty Poro", type=UNIT,
    energy=2, might=2, domains=(Domain.FURY,), tags=("Poro",),
    # SIMPLIFIED: Deflect (opponent targeting tax) not implemented.
))
register(CardDef(
    id="OGN-011", name="Magma Wurm", type=UNIT,
    energy=8, power=((Domain.FURY, 1),), might=8, domains=(Domain.FURY,),
    # SIMPLIFIED: "Other friendly units enter ready" passive not implemented.
))
register(CardDef(
    id="OGS-001", name="Annie, Fiery", type=UNIT,
    energy=5, power=((Domain.FURY, 1),), might=4, domains=(Domain.FURY,),
    tags=("Annie",), is_champion_unit=True,
    bonus_spell_damage=1,  # "Your spells and abilities deal 1 Bonus Damage."
))
register(CardDef(
    id="OGS-018", name="Tibbers", type=UNIT,
    energy=8, power=((Domain.FURY, 2),), might=7,
    domains=(Domain.CHAOS, Domain.FURY), tags=("Annie",),
    # "When you play me, deal 3 to all units at battlefields."
    instructions=(Instruction(op="damage_all_bf_units", amount=3),),
    # SIMPLIFIED: power pips paid as Fury (printed as Chaos/Fury signature).
))
register(CardDef(
    id="OGN-029", name="Falling Star", type=SPELL,
    energy=2, power=((Domain.FURY, 2),), domains=(Domain.FURY,),
    instructions=(
        Instruction(op="damage", amount=3, target=ANY_UNIT),
        Instruction(op="damage", amount=3, target=ANY_UNIT),
    ),
))
register(CardDef(
    id="OGN-009", name="Hextech Ray", type=SPELL,
    energy=1, power=((Domain.FURY, 1),), domains=(Domain.FURY,),
    instructions=(Instruction(op="damage", amount=3, target=ANY_UNIT_AT_BF),),
))
register(CardDef(
    id="OGN-024", name="Void Seeker", type=SPELL,
    energy=3, power=((Domain.FURY, 1),), domains=(Domain.FURY,),
    instructions=(
        Instruction(op="damage", amount=4, target=ANY_UNIT_AT_BF),
        Instruction(op="draw", amount=1),
    ),
))
register(CardDef(
    id="OGN-252", name="Super Mega Death Rocket!", type=SPELL,
    energy=4, power=((Domain.FURY, 1),), domains=(Domain.FURY,),
    instructions=(Instruction(op="damage", amount=5, target=ANY_UNIT),),
    # SIMPLIFIED: conquer-triggered return from trash not implemented.
))
register(CardDef(
    id="OGN-004", name="Cleave", type=SPELL,
    energy=1, domains=(Domain.FURY,),
    # "Give a unit Assault 3 this turn."
    instructions=(Instruction(op="buff_assault", amount=3, target=ANY_UNIT),),
))
register(CardDef(
    id="OGS-003", name="Incinerate", type=SPELL,
    energy=2, domains=(Domain.FURY,),
    instructions=(Instruction(op="damage", amount=2, target=ANY_UNIT_AT_BF),),
))
register(CardDef(
    id="OGS-002", name="Firestorm", type=SPELL,
    energy=6, power=((Domain.FURY, 1),), domains=(Domain.FURY,),
    # "Deal 3 to all enemy units at a battlefield."
    instructions=(Instruction(op="damage_all_at_bf", amount=3, target=ANY_BATTLEFIELD),),
))
register(CardDef(
    id="OGN-040", name="Seal of Rage", type=GEAR,
    energy=0, power=((Domain.FURY, 1),), domains=(Domain.FURY,),
    gear_power_domain=Domain.FURY,
))

# ---------------------------------------------------------------------------
# Calm / Body (Master Yi)
# ---------------------------------------------------------------------------

register(CardDef(
    id="OGN-052", name="Stalwart Poro", type=UNIT,
    energy=2, might=2, domains=(Domain.CALM,), tags=("Poro",),
    shield=1,
))
register(CardDef(
    id="OGN-054", name="Sunlit Guardian", type=UNIT,
    energy=3, might=3, domains=(Domain.CALM,),
    shield=1, tank=True,
))
register(CardDef(
    id="OGN-049", name="Playful Phantom", type=UNIT,
    energy=5, might=5, domains=(Domain.CALM,),
))
register(CardDef(
    id="OGS-005", name="Zephyr Sage", type=UNIT,
    energy=6, power=((Domain.CALM, 1),), might=6, domains=(Domain.CALM,),
    shield=1,
))
register(CardDef(
    id="OGN-044", name="Clockwork Keeper", type=UNIT,
    energy=2, might=2, domains=(Domain.CALM,),
    # SIMPLIFIED: optional additional cost (pay Calm -> draw 1) not implemented.
))
register(CardDef(
    id="OGS-004", name="Yi, Meditative", type=UNIT,
    energy=5, power=((Domain.CALM, 1),), might=4, domains=(Domain.CALM,),
    tags=("Master Yi",), is_champion_unit=True,
    might_bonus_if_runes8=4,  # "While you have 8+ runes, I have +4 might."
))
register(CardDef(
    id="OGS-007", name="Garen, Rugged", type=UNIT,
    energy=6, power=((Domain.BODY, 1),), might=5, domains=(Domain.BODY,),
    tags=("Garen",),
    assault=2, shield=2,
))
register(CardDef(
    id="OGS-009", name="Yi, Honed", type=UNIT,
    energy=7, power=((Domain.BODY, 1),), might=6, domains=(Domain.BODY,),
    tags=("Master Yi",),
    ganking=True, enters_ready=True,
))
register(CardDef(
    id="OGN-058", name="Discipline", type=SPELL,
    energy=2, domains=(Domain.CALM,),
    instructions=(
        Instruction(op="buff", amount=2, target=ANY_UNIT),
        Instruction(op="draw", amount=1),
    ),
    # SIMPLIFIED: Reaction timing (v0 has no reaction windows).
))
register(CardDef(
    id="SFD-037", name="Navori Scout", type=UNIT,
    energy=4, might=4, domains=(Domain.CALM,),
    # SIMPLIFIED: Deflect not implemented.
))
register(CardDef(
    id="UNL-036", name="Mutated Mouser", type=UNIT,
    energy=2, might=1, domains=(Domain.CALM,),
    shield=2, tank=True,
))
register(CardDef(
    id="OGN-055", name="Wielder of Water", type=UNIT,
    energy=3, might=2, domains=(Domain.CALM,),
    might_bonus_if_alone=2,  # "+2 while attacking or defending alone"
))
register(CardDef(
    id="OGN-081", name="Seal of Focus", type=GEAR,
    energy=0, power=((Domain.CALM, 1),), domains=(Domain.CALM,),
    gear_power_domain=Domain.CALM,
))
register(CardDef(
    id="SFD-046", name="Poro Snax", type=GEAR,
    energy=1, power=((Domain.CALM, 1),), domains=(Domain.CALM,),
    instructions=(Instruction(op="draw", amount=1),),  # "When you play this, draw 1."
    # SIMPLIFIED: activated sacrifice ability not implemented.
))

# ---------------------------------------------------------------------------
# Battlefields
# ---------------------------------------------------------------------------

register_battlefield(BattlefieldDef(
    id="OGN-277", name="Back-Alley Bar",
    move_from_buff=1,  # "When a unit moves from here, give it +1 might this turn."
))
register_battlefield(BattlefieldDef(
    id="OGN-280", name="Grove of the God-Willow",
    on_hold=(Instruction(op="draw", amount=1),),  # "When you hold here, draw 1."
))

# ---------------------------------------------------------------------------
# Decks
# ---------------------------------------------------------------------------


def build_annie_deck() -> Deck:
    """Mono-Fury burn deck. Legend: Dark Child (abilities not implemented)."""
    main = (
        ["OGN-010"] * 3      # Legion Rearguard
        + ["OGN-001"] * 3    # Blazing Scorcher
        + ["OGN-012"] * 3    # Noxus Hopeful
        + ["OGN-013"] * 3    # Pouty Poro
        + ["OGN-011"] * 1    # Magma Wurm
        + ["OGS-001"] * 2    # Annie, Fiery (extra copies)
        + ["OGS-018"] * 2    # Tibbers
        + ["OGN-029"] * 3    # Falling Star
        + ["OGN-009"] * 3    # Hextech Ray
        + ["OGN-024"] * 3    # Void Seeker
        + ["OGN-252"] * 2    # Super Mega Death Rocket!
        + ["OGN-004"] * 3    # Cleave
        + ["OGS-003"] * 3    # Incinerate
        + ["OGS-002"] * 2    # Firestorm
        + ["OGN-040"] * 3    # Seal of Rage
    )
    assert len(main) == 39, len(main)
    return Deck(
        legend_name="Dark Child",
        legend_domains=(Domain.CHAOS, Domain.FURY),
        champion="OGS-001",
        main=main,
        runes=[Domain.FURY] * 12,
        battlefield="OGN-277",  # Back-Alley Bar
    )


def build_yi_deck() -> Deck:
    """Calm/Body defensive deck. Legend: Wuju Bladesman (abilities not
    implemented)."""
    main = (
        ["OGN-052"] * 3      # Stalwart Poro
        + ["OGN-054"] * 3    # Sunlit Guardian
        + ["OGN-049"] * 3    # Playful Phantom
        + ["OGS-005"] * 3    # Zephyr Sage
        + ["OGN-044"] * 3    # Clockwork Keeper
        + ["OGS-004"] * 2    # Yi, Meditative (extra copies)
        + ["OGS-007"] * 2    # Garen, Rugged
        + ["OGS-009"] * 2    # Yi, Honed
        + ["OGN-058"] * 3    # Discipline
        + ["SFD-037"] * 3    # Navori Scout
        + ["UNL-036"] * 3    # Mutated Mouser
        + ["OGN-055"] * 3    # Wielder of Water
        + ["OGN-081"] * 3    # Seal of Focus
        + ["SFD-046"] * 3    # Poro Snax
    )
    assert len(main) == 39, len(main)
    return Deck(
        legend_name="Wuju Bladesman",
        legend_domains=(Domain.BODY, Domain.CALM),
        champion="OGS-004",
        main=main,
        runes=[Domain.CALM] * 6 + [Domain.BODY] * 6,
        battlefield="OGN-280",  # Grove of the God-Willow
    )
