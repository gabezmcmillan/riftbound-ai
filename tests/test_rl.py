"""Self-play PPO layer: masking, GAE, collection, updates, checkpoints."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from riftbound.agents.policy_agent import PolicyAgent
from riftbound.agents.random_agent import RandomAgent
from riftbound.encoding import ActionCodec, ObservationEncoder
from riftbound.rl.network import PolicyValueNet, masked_distribution
from riftbound.rl.selfplay import _gae, collect_games, play_selfplay_game
from riftbound.rl.train import PPOConfig, ppo_update, save_checkpoint
from riftbound.sim import play_game

from conftest import make_test_deck


def make_net(hidden: int = 64) -> tuple[PolicyValueNet, ObservationEncoder, ActionCodec]:
    encoder = ObservationEncoder()
    codec = ActionCodec()
    return PolicyValueNet(encoder.size, codec.size, hidden=hidden), encoder, codec


def test_masked_distribution_only_samples_legal():
    torch.manual_seed(0)
    logits = torch.randn(200, 10)
    mask = torch.zeros(200, 10, dtype=torch.bool)
    mask[:, [1, 4, 7]] = True
    dist = masked_distribution(logits, mask)
    samples = dist.sample()
    assert set(samples.tolist()) <= {1, 4, 7}
    # Illegal actions carry zero probability mass.
    assert torch.allclose(dist.probs[:, [0, 2, 3, 5, 6, 8, 9]], torch.zeros(200, 7))


def test_gae_terminal_only_reward():
    values = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    adv, ret = _gae(values, terminal_reward=1.0, gamma=1.0, lam=1.0)
    # With gamma=lam=1 and zero values, every step's advantage is the outcome.
    assert np.allclose(adv, [1.0, 1.0, 1.0])
    assert np.allclose(ret, [1.0, 1.0, 1.0])


def test_selfplay_game_batch_shapes_and_outcome():
    net, encoder, codec = make_net()
    decks = (make_test_deck("T-BF-A"), make_test_deck("T-BF-B"))
    batch = play_selfplay_game(net, encoder, codec, decks, seed=3, first_player=0)
    n = len(batch)
    assert n > 0
    assert batch.obs.shape == (n, encoder.size)
    assert batch.mask.shape == (n, codec.size)
    assert batch.games == 1
    # Every recorded action was legal under its own mask.
    assert all(batch.mask[i, batch.action[i]] for i in range(n))
    # Terminal-only reward: |return| <= 1 everywhere.
    assert np.all(np.abs(batch.ret) <= 1.0 + 1e-5)
    assert np.all(np.isfinite(batch.advantage))


def test_ppo_update_improves_and_is_finite():
    torch.manual_seed(0)
    net, encoder, codec = make_net()
    decks = (make_test_deck("T-BF-A"), make_test_deck("T-BF-B"))
    batch = collect_games(net, [decks], n_games=2, seed=11, max_turns=40)
    cfg = PPOConfig(epochs=2, minibatch=64)
    before = [p.clone() for p in net.parameters()]
    metrics = ppo_update(net, torch.optim.Adam(net.parameters(), lr=1e-3), batch,
                         cfg, np.random.default_rng(0))
    assert all(np.isfinite(v) for v in metrics.values()), metrics
    assert metrics["entropy"] >= 0
    assert any(
        not torch.equal(a, b) for a, b in zip(before, net.parameters())
    ), "update did not change any parameters"


def test_checkpoint_roundtrip_and_policy_agent_plays(tmp_path):
    net, encoder, codec = make_net()
    path = tmp_path / "ckpt.pt"
    cfg = PPOConfig(hidden=64)
    save_checkpoint(path, net, torch.optim.Adam(net.parameters()), 5, cfg, encoder.vocab)

    agent = PolicyAgent.from_checkpoint(str(path))
    assert agent.net.hidden == 64
    for a, b in zip(net.state_dict().values(), agent.net.state_dict().values()):
        assert torch.equal(a, b)

    # A full legal game driven by the loaded policy (illegal moves would raise).
    decks = (make_test_deck("T-BF-A"), make_test_deck("T-BF-B"))
    game = play_game(decks, (agent, RandomAgent(seed=1)), seed=2, max_turns=60)
    assert game.is_over


def test_policy_agent_deterministic():
    net, encoder, codec = make_net()
    decks = (make_test_deck("T-BF-A"), make_test_deck("T-BF-B"))
    a1 = PolicyAgent(net, encoder=encoder, codec=codec)
    g1 = play_game(decks, (a1, a1), seed=9, max_turns=40)
    a2 = PolicyAgent(net, encoder=encoder, codec=codec)
    g2 = play_game(decks, (a2, a2), seed=9, max_turns=40)
    assert g1.state.turn_number == g2.state.turn_number
    assert g1.winner == g2.winner
