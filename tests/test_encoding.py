"""Observation/action encoding: mask validity, playthroughs, no info leaks."""

import copy
import random

import numpy as np
import pytest

from riftbound.encoding import ActionCodec, ObservationEncoder
from riftbound.engine.actions import EndTurn


@pytest.fixture
def codec():
    return ActionCodec()


@pytest.fixture
def encoder():
    return ObservationEncoder()


def test_end_turn_always_legal(game, codec):
    mask = codec.legal_mask(game)
    assert mask[0]  # entry 0 is EndTurn
    assert isinstance(codec.decode(0, game), EndTurn)


def test_every_masked_action_is_executable(game, codec):
    mask = codec.legal_mask(game)
    legal_indices = np.flatnonzero(mask)
    assert len(legal_indices) > 1  # more than just EndTurn on turn 1
    for idx in legal_indices:
        sim = copy.deepcopy(game)
        action = codec.decode(int(idx), sim)
        assert action is not None
        sim.step(action)  # must not raise


def test_full_game_through_codec(game, codec):
    rng = random.Random(0)
    steps = 0
    while not game.is_over and steps < 3000:
        mask = codec.legal_mask(game)
        idx = int(rng.choice(np.flatnonzero(mask)))
        game.step(codec.decode(idx, game))
        steps += 1
    assert game.is_over


def test_full_game_real_decks_through_codec():
    from riftbound.cards import build_annie_deck, build_yi_deck
    from riftbound.engine.game import Game

    codec = ActionCodec()
    encoder = ObservationEncoder()
    game = Game((build_annie_deck(), build_yi_deck()), seed=11, first_player=0)
    rng = random.Random(1)
    while not game.is_over:
        obs = encoder.encode(game, game.state.turn_player)
        assert obs.shape == (encoder.size,)
        assert obs.dtype == np.float32
        mask = codec.legal_mask(game)
        idx = int(rng.choice(np.flatnonzero(mask)))
        game.step(codec.decode(idx, game))
    assert game.winner in (0, 1) or game.state.turn_number > game.max_turns


def test_observation_shape_and_range(game, encoder):
    obs = encoder.encode(game, 0)
    assert obs.shape == (encoder.size,)
    assert np.isfinite(obs).all()


def test_opponent_hand_is_hidden(game, encoder):
    """Swapping the contents (not size) of the opponent's hand must not
    change player 0's observation."""
    obs_before = encoder.encode(game, 0)
    game.state.players[1].hand = ["T-BIG"] * len(game.state.players[1].hand)
    obs_after = encoder.encode(game, 0)
    assert np.array_equal(obs_before, obs_after)


def test_own_hand_is_visible(game, encoder):
    obs_before = encoder.encode(game, 0)
    game.state.players[0].hand = ["T-BIG"] * len(game.state.players[0].hand)
    obs_after = encoder.encode(game, 0)
    assert not np.array_equal(obs_before, obs_after)


def test_engine_actions_roundtrip(game, codec):
    """Every action the engine enumerates that the codec can express must
    encode -> decode back to an equal action."""
    covered = 0
    for action in game.legal_actions():
        idx = codec.encode_action(action, game)
        if idx is not None:
            assert codec.decode(idx, game) == action
            covered += 1
    assert covered > 0
