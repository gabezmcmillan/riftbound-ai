"""Movement, combat, and scoring (CR 144, 459-472)."""

from riftbound.engine.actions import EndTurn, MoveUnits
from riftbound.engine.cards import card
from riftbound.engine.enums import BASE
from riftbound.engine.state import UnitState


def spawn(game, card_id, controller, location=BASE, ready=True):
    s = game.state
    uid = s.new_uid()
    s.units[uid] = UnitState(
        uid=uid, card=card(card_id), controller=controller, location=location, ready=ready
    )
    return uid


def test_move_to_empty_battlefield_conquers(game):
    uid = spawn(game, "T-GRUNT", 0)
    game.step(MoveUnits(unit_uids=(uid,), destination=0))
    bf = game.state.battlefields[0]
    assert bf.controller == 0
    assert game.state.players[0].points == 1
    assert not game.state.units[uid].ready  # exhausted by the move


def test_hold_scores_at_start_of_next_turn(game):
    uid = spawn(game, "T-GRUNT", 0)
    game.step(MoveUnits(unit_uids=(uid,), destination=0))  # conquer: 1 point
    game.step(EndTurn())
    game.step(EndTurn())  # opponent passes
    # p0's Beginning Phase: holds battlefield 0.
    assert game.state.players[0].points == 2


def test_exhausted_unit_cannot_move(game):
    uid = spawn(game, "T-GRUNT", 0)
    game.step(MoveUnits(unit_uids=(uid,), destination=0))
    import pytest
    from riftbound.engine.game import IllegalAction

    with pytest.raises(IllegalAction):
        game.step(MoveUnits(unit_uids=(uid,), destination=BASE))


def test_control_lost_when_units_leave(game):
    uid = spawn(game, "T-GRUNT", 0)
    game.step(MoveUnits(unit_uids=(uid,), destination=0))
    game.step(EndTurn())
    game.step(EndTurn())  # p0 holds (2 points), unit re-readied
    game.step(MoveUnits(unit_uids=(uid,), destination=BASE))
    assert game.state.battlefields[0].controller is None
    game.step(EndTurn())
    game.step(EndTurn())
    assert game.state.players[0].points == 2  # no hold this time


def test_combat_attacker_wins_and_conquers(game):
    defender_uid = spawn(game, "T-GRUNT", 1, location=0, ready=False)  # might 2
    game.state.battlefields[0].controller = 1
    attacker_uid = spawn(game, "T-BIG", 0)  # might 5
    game.step(MoveUnits(unit_uids=(attacker_uid,), destination=0))
    assert defender_uid not in game.state.units  # grunt died
    assert attacker_uid in game.state.units
    assert game.state.battlefields[0].controller == 0
    assert game.state.players[0].points == 1


def test_combat_mutual_wipe_leaves_battlefield_uncontrolled(game):
    # Defenders: tank (3) + grunt (2) = 5 might total.
    spawn(game, "T-TANK", 1, location=0, ready=False)
    spawn(game, "T-GRUNT", 1, location=0, ready=False)
    game.state.battlefields[0].controller = 1
    # Attacker: one 5-might unit. Kills both defenders (3 lethal to tank
    # first, 2 to grunt); defenders' 5 kills it back.
    attacker_uid = spawn(game, "T-BIG", 0)
    game.step(MoveUnits(unit_uids=(attacker_uid,), destination=0))
    assert not game.state.units  # everyone died
    assert game.state.battlefields[0].controller is None
    assert game.state.players[0].points == 0  # no conquer


def test_shield_keyword_saves_defender(game):
    # Real card: Stalwart Poro (2 might, Shield 1 -> 3 while defending).
    import riftbound.cards  # noqa: F401  (registers the OGN subset)

    poro_uid = spawn(game, "OGN-052", 1, location=0, ready=False)
    game.state.battlefields[0].controller = 1
    attacker_uid = spawn(game, "T-GRUNT", 0)  # might 2
    game.step(MoveUnits(unit_uids=(attacker_uid,), destination=0))
    # Attacker's 2 damage is not lethal against effective might 3; the
    # poro's 3 might kills the 2-might attacker.
    assert poro_uid in game.state.units
    assert attacker_uid not in game.state.units
    assert game.state.battlefields[0].controller == 1


def test_tank_must_be_assigned_first(game):
    # Defender: tank (might 3) and a big (might 5). Attacker total 5:
    # must fill lethal on the tank (3) before the big; remaining 2 dumped
    # into the big is non-lethal.
    tank_uid = spawn(game, "T-TANK", 1, location=0, ready=False)
    big_uid = spawn(game, "T-BIG", 1, location=0, ready=False)
    game.state.battlefields[0].controller = 1
    attacker_uid = spawn(game, "T-BIG", 0)
    game.step(MoveUnits(unit_uids=(attacker_uid,), destination=0))
    assert tank_uid not in game.state.units  # tank died first
    assert big_uid in game.state.units  # big survived with dumped damage
    # Defenders' 8 might kills the 5-might attacker.
    assert attacker_uid not in game.state.units


def test_final_point_restriction(game):
    game.state.players[0].points = 7
    u1 = spawn(game, "T-GRUNT", 0)
    u2 = spawn(game, "T-GRUNT", 0)
    hand_before = len(game.state.players[0].hand)
    game.step(MoveUnits(unit_uids=(u1,), destination=0))
    # Conquer at 7 without scoring every battlefield: draw instead (CR 471.1.b).
    assert game.state.players[0].points == 7
    assert len(game.state.players[0].hand) == hand_before + 1
    assert game.state.winner is None
    # Conquering the second battlefield the same turn completes the set: win.
    game.step(MoveUnits(unit_uids=(u2,), destination=1))
    assert game.state.players[0].points == 8
    assert game.state.winner == 0


def test_burn_out_gives_opponent_point_and_recycles_trash(game):
    p0 = game.state.players[0]
    p0.main_deck = []
    p0.trash = ["T-GRUNT", "T-BOLT"]
    game._draw_cards(0, 1)
    assert game.state.players[1].points == 1
    assert len(p0.main_deck) + len([c for c in p0.hand]) >= 1
    assert p0.trash == []
