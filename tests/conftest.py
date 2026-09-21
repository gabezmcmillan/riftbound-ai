"""Shared test fixtures: a tiny synthetic card set with deterministic decks."""

from __future__ import annotations

import pytest

from riftbound.engine import cards as cards_mod
from riftbound.engine.cards import (
    BattlefieldDef,
    CardDef,
    Instruction,
    TargetSpec,
)
from riftbound.engine.enums import CardType, Domain
from riftbound.engine.game import Deck, Game

D = Domain.FURY

TEST_CARDS = [
    CardDef(id="T-GRUNT", name="Test Grunt", type=CardType.UNIT, energy=1, might=2),
    CardDef(id="T-BIG", name="Test Bruiser", type=CardType.UNIT, energy=3, might=5),
    CardDef(
        id="T-TANK",
        name="Test Tank",
        type=CardType.UNIT,
        energy=2,
        might=3,
        tank=True,
    ),
    CardDef(
        id="T-BOLT",
        name="Test Bolt",
        type=CardType.SPELL,
        energy=1,
        instructions=(
            Instruction(op="damage", amount=2, target=TargetSpec(kind="unit")),
        ),
    ),
    CardDef(
        id="T-CANTRIP",
        name="Test Cantrip",
        type=CardType.SPELL,
        energy=1,
        instructions=(Instruction(op="draw", amount=1),),
    ),
    CardDef(
        id="T-SEAL",
        name="Test Seal",
        type=CardType.GEAR,
        energy=0,
        power=((D, 1),),
        gear_power_domain=D,
    ),
    CardDef(
        id="T-COSTLY",
        name="Test Costly",
        type=CardType.UNIT,
        energy=2,
        power=((D, 1),),
        might=4,
    ),
    CardDef(id="T-CHAMP", name="Test Champ", type=CardType.UNIT, energy=2, might=3,
            is_champion_unit=True),
]

TEST_BATTLEFIELDS = [
    BattlefieldDef(id="T-BF-A", name="Test Field A"),
    BattlefieldDef(id="T-BF-B", name="Test Field B"),
]


@pytest.fixture(autouse=True)
def _register_test_cards():
    """Register the synthetic set for each test, then restore the registry."""
    saved_cards = dict(cards_mod.REGISTRY)
    saved_bfs = dict(cards_mod.BATTLEFIELDS)
    for c in TEST_CARDS:
        cards_mod.REGISTRY.setdefault(c.id, c)
    for bf in TEST_BATTLEFIELDS:
        cards_mod.BATTLEFIELDS.setdefault(bf.id, bf)
    yield
    cards_mod.REGISTRY.clear()
    cards_mod.REGISTRY.update(saved_cards)
    cards_mod.BATTLEFIELDS.clear()
    cards_mod.BATTLEFIELDS.update(saved_bfs)


def make_test_deck(battlefield_id: str = "T-BF-A") -> Deck:
    main = (
        ["T-GRUNT"] * 12
        + ["T-BIG"] * 8
        + ["T-TANK"] * 6
        + ["T-BOLT"] * 6
        + ["T-CANTRIP"] * 4
        + ["T-SEAL"] * 3
    )
    assert len(main) == 39
    return Deck(
        legend_name="Test Legend",
        legend_domains=(D,),
        champion="T-CHAMP",
        main=main,
        runes=[D] * 12,
        battlefield=battlefield_id,
    )


@pytest.fixture
def game() -> Game:
    return Game(
        (make_test_deck("T-BF-A"), make_test_deck("T-BF-B")),
        seed=7,
        first_player=0,
    )
