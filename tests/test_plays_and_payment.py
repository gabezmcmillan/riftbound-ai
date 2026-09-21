"""Playing cards and the auto-payment system (CR 143.4, 149, 157, 164-166)."""

import pytest

from riftbound.engine.actions import PlayGear, PlaySpell, PlayUnit
from riftbound.engine.cards import card
from riftbound.engine.enums import BASE
from riftbound.engine.game import IllegalAction
from riftbound.engine.state import UnitState


def hand_index_of(game, pidx, card_id):
    return game.state.players[pidx].hand.index(card_id)


def force_hand(game, pidx, cards):
    game.state.players[pidx].hand = list(cards)


def test_play_unit_exhausts_runes_and_enters_exhausted(game):
    force_hand(game, 0, ["T-GRUNT"])
    game.step(PlayUnit(hand_index=0, destination=BASE))
    units = game.state.units_at(0, BASE)
    assert len(units) == 1
    assert not units[0].ready  # units enter exhausted (CR 143.4)
    exhausted = [r for r in game.state.player_runes(0) if not r.ready]
    assert len(exhausted) == 1  # paid 1 energy


def test_power_cost_recycles_rune_to_deck_bottom(game):
    force_hand(game, 0, ["T-COSTLY"])  # costs 2 energy + 1 Fury power
    p0 = game.state.players[0]
    runes_before = len(game.state.player_runes(0))
    deck_before = len(p0.rune_deck)
    with pytest.raises(IllegalAction):
        # 2 runes total: 1 recycled for power leaves only 1 for 2 energy.
        game.step(PlayUnit(hand_index=0, destination=BASE))
    # Give an extra rune, then it should work.
    game._channel(0, 1)
    game.step(PlayUnit(hand_index=0, destination=BASE))
    assert len(game.state.player_runes(0)) == runes_before + 1 - 1  # one recycled
    assert len(p0.rune_deck) == deck_before - 1 + 1  # channel took 1, recycle returned 1


def test_seal_gear_pays_power(game):
    force_hand(game, 0, ["T-SEAL", "T-COSTLY"])
    game._channel(0, 2)  # 4 ready runes total
    game.step(PlayGear(hand_index=0))  # costs 1 power: recycles 1 rune
    assert len(game.state.player_runes(0)) == 3
    # T-COSTLY (2 energy + 1 power): power comes from exhausting the seal,
    # energy from exhausting 2 runes -- no further rune is recycled.
    game.step(PlayUnit(hand_index=0, destination=BASE))
    assert len(game.state.player_runes(0)) == 3
    assert sum(1 for r in game.state.player_runes(0) if not r.ready) == 2
    assert all(not g.ready for g in game.state.player_gear(0))


def test_spell_damage_kills_and_goes_to_trash(game):
    s = game.state
    uid = s.new_uid()
    s.units[uid] = UnitState(uid=uid, card=card("T-GRUNT"), controller=1, location=0)
    force_hand(game, 0, ["T-BOLT"])
    game.step(PlaySpell(hand_index=0, targets=(uid,)))
    assert uid not in s.units  # 2 damage >= 2 might
    assert "T-BOLT" in s.players[0].trash
    assert "T-GRUNT" in s.players[1].trash


def test_cantrip_draws(game):
    force_hand(game, 0, ["T-CANTRIP"])
    deck_before = len(game.state.players[0].main_deck)
    game.step(PlaySpell(hand_index=0, targets=()))
    assert len(game.state.players[0].hand) == 1  # played 1, drew 1
    assert len(game.state.players[0].main_deck) == deck_before - 1


def test_champion_playable_from_champion_zone(game):
    game._channel(0, 2)  # ensure affordable
    game.step(PlayUnit(hand_index=-1, destination=BASE))
    assert game.state.players[0].champion is None
    names = [u.card.name for u in game.state.units_at(0, BASE)]
    assert "Test Champ" in names
