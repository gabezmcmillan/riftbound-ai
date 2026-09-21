"""Game setup and start-of-turn flow (CR 110-118, 315, 485)."""

from riftbound.engine.actions import EndTurn
from riftbound.engine.enums import Phase


def test_opening_state(game):
    for p in game.state.players:
        # 39 main deck cards, 4 drawn at setup, plus 1 turn-player draw.
        assert p.champion == "T-CHAMP"
    p0, p1 = game.state.players
    assert len(p0.hand) == 5  # opening 4 + draw phase 1
    assert len(p1.hand) == 4
    assert len(p0.main_deck) == 39 - 5
    assert game.state.phase is Phase.MAIN
    assert game.state.turn_player == 0


def test_channel_two_then_three_for_second_player(game):
    # First player channels 2 on turn 1.
    assert len(game.state.player_runes(0)) == 2
    game.step(EndTurn())
    # Second player channels an extra rune on their first turn (CR 485.7).
    assert len(game.state.player_runes(1)) == 3
    assert len(game.state.players[1].rune_deck) == 9


def test_rune_accumulation_over_turns(game):
    game.step(EndTurn())  # -> p1 turn 1
    game.step(EndTurn())  # -> p0 turn 2
    assert len(game.state.player_runes(0)) == 4
    assert game.state.turn_number == 3
