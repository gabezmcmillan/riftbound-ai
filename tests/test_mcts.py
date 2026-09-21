"""Determinized MCTS agent: determinization invariants and legal play."""

import random

from riftbound.agents.mcts_agent import MCTSAgent, determinize, reduced_actions
from riftbound.engine.actions import EndTurn, MoveUnits


def test_determinize_preserves_visible_information(game):
    rng = random.Random(0)
    world = determinize(game, perspective=0, rng=rng)
    me, w_me = game.state.players[0], world.state.players[0]
    opp, w_opp = game.state.players[1], world.state.players[1]

    # Own hand is untouched; own deck is a permutation of the real one.
    assert w_me.hand == me.hand
    assert sorted(w_me.main_deck) == sorted(me.main_deck)
    # Opponent hand size preserved; hand+deck pool is the same multiset.
    assert len(w_opp.hand) == len(opp.hand)
    assert sorted(w_opp.hand + w_opp.main_deck) == sorted(opp.hand + opp.main_deck)
    # Public state is identical.
    assert w_opp.points == opp.points
    assert w_opp.trash == opp.trash
    assert len(world.state.units) == len(game.state.units)
    assert [r.domain for r in world.state.player_runes(1)] == [
        r.domain for r in game.state.player_runes(1)
    ]


def test_determinize_does_not_share_rng(game):
    rng = random.Random(0)
    world = determinize(game, perspective=0, rng=rng)
    assert world.rng is not game.rng


def test_reduced_actions_prunes_partial_groups(game):
    from tests.test_combat_and_scoring import spawn

    for _ in range(3):
        spawn(game, "T-GRUNT", 0)
    actions = reduced_actions(game)
    group_moves = [
        a for a in actions if isinstance(a, MoveUnits) and len(a.unit_uids) > 1
    ]
    # Only the all-in group per battlefield remains (size 3), no pairs.
    assert group_moves
    assert all(len(a.unit_uids) == 3 for a in group_moves)


def test_mcts_returns_legal_action_and_game_advances(game):
    agent = MCTSAgent(iterations=12, determinizations=2, seed=0)
    action = agent.choose(game)
    assert action in game.legal_actions()
    game.step(action)


def test_mcts_plays_full_game_vs_random():
    from riftbound.agents import RandomAgent
    from riftbound.cards import build_annie_deck, build_yi_deck
    from riftbound.sim import play_game

    game = play_game(
        (build_annie_deck(), build_yi_deck()),
        (MCTSAgent(iterations=8, determinizations=1, seed=1), RandomAgent(seed=2)),
        seed=3,
        first_player=0,
        max_turns=60,
    )
    assert game.is_over


def test_mcts_takes_winning_move(game):
    """With 7 points and both battlefields takeable, MCTS must find the
    two-conquer win rather than ending the turn."""
    from tests.test_combat_and_scoring import spawn

    game.state.players[0].points = 7
    u1 = spawn(game, "T-BIG", 0)
    u2 = spawn(game, "T-BIG", 0)
    agent = MCTSAgent(iterations=60, determinizations=2, seed=4)
    # Play up to a few actions; the win must be reached within this turn.
    for _ in range(4):
        if game.is_over:
            break
        action = agent.choose(game)
        assert not isinstance(action, EndTurn), "MCTS ended turn instead of winning"
        game.step(action)
    assert game.winner == 0
