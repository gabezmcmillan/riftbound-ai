"""Self-play trajectory collection for PPO.

The current policy plays both seats of each game. Every decision is recorded
from the acting player's perspective (observation, legality mask, sampled
action, log-prob, value estimate); at game end each player's own step
sequence becomes one episode with a terminal reward of +1 (win), -1 (loss)
or 0 (turn-limit draw), and GAE advantages are computed per episode.

Because the rules engine is pure Python, throughput scales with CPU cores,
not GPU: `collect_parallel` fans games out to worker processes (Windows
spawn-safe — workers re-import the card registry and rebuild the net from a
broadcast state_dict) and concatenates the resulting batches.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import torch

from riftbound.encoding import ActionCodec, ObservationEncoder
from riftbound.engine.game import Deck, Game
from riftbound.rl.network import PolicyValueNet, masked_distribution


@dataclass
class Batch:
    """Concatenated training data across many episodes."""

    obs: np.ndarray  # (N, obs_size) float32
    mask: np.ndarray  # (N, action_size) bool
    action: np.ndarray  # (N,) int64
    logp: np.ndarray  # (N,) float32
    value: np.ndarray  # (N,) float32
    advantage: np.ndarray  # (N,) float32
    ret: np.ndarray  # (N,) float32
    # stats
    games: int = 0
    draws: int = 0
    total_turns: int = 0

    def __len__(self) -> int:
        return len(self.action)

    @staticmethod
    def concat(batches: list["Batch"]) -> "Batch":
        return Batch(
            obs=np.concatenate([b.obs for b in batches]),
            mask=np.concatenate([b.mask for b in batches]),
            action=np.concatenate([b.action for b in batches]),
            logp=np.concatenate([b.logp for b in batches]),
            value=np.concatenate([b.value for b in batches]),
            advantage=np.concatenate([b.advantage for b in batches]),
            ret=np.concatenate([b.ret for b in batches]),
            games=sum(b.games for b in batches),
            draws=sum(b.draws for b in batches),
            total_turns=sum(b.total_turns for b in batches),
        )


@dataclass
class _Episode:
    """One player's step sequence within a single game."""

    obs: list[np.ndarray] = field(default_factory=list)
    mask: list[np.ndarray] = field(default_factory=list)
    action: list[int] = field(default_factory=list)
    logp: list[float] = field(default_factory=list)
    value: list[float] = field(default_factory=list)


def _gae(
    values: np.ndarray, terminal_reward: float, gamma: float, lam: float
) -> tuple[np.ndarray, np.ndarray]:
    """Advantages/returns for one episode whose only reward is terminal."""
    n = len(values)
    adv = np.zeros(n, dtype=np.float32)
    last = 0.0
    for t in reversed(range(n)):
        next_value = 0.0 if t == n - 1 else values[t + 1]
        reward = terminal_reward if t == n - 1 else 0.0
        delta = reward + gamma * next_value - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values


def play_selfplay_game(
    net: PolicyValueNet,
    encoder: ObservationEncoder,
    codec: ActionCodec,
    decks: tuple[Deck, Deck],
    seed: int | None = None,
    first_player: int | None = None,
    max_turns: int = 120,
    gamma: float = 0.997,
    lam: float = 0.95,
) -> Batch:
    """Play one self-play game and return its two episodes as a Batch."""
    game = Game(decks, seed=seed, first_player=first_player, max_turns=max_turns)
    episodes = (_Episode(), _Episode())

    net.eval()
    while not game.is_over:
        pidx = game.state.turn_player
        obs = encoder.encode(game, pidx)
        mask = codec.legal_mask(game)
        idx, logp, value = net.act(
            torch.from_numpy(obs), torch.from_numpy(mask)
        )
        ep = episodes[pidx]
        ep.obs.append(obs)
        ep.mask.append(mask)
        ep.action.append(idx)
        ep.logp.append(logp)
        ep.value.append(value)
        game.step(codec.decode(idx, game))

    winner = game.winner
    parts: list[Batch] = []
    for pidx, ep in enumerate(episodes):
        if not ep.obs:
            continue
        values = np.array(ep.value, dtype=np.float32)
        reward = 0.0 if winner is None else (1.0 if winner == pidx else -1.0)
        adv, ret = _gae(values, reward, gamma, lam)
        parts.append(
            Batch(
                obs=np.stack(ep.obs),
                mask=np.stack(ep.mask),
                action=np.array(ep.action, dtype=np.int64),
                logp=np.array(ep.logp, dtype=np.float32),
                value=values,
                advantage=adv,
                ret=ret,
            )
        )
    out = Batch.concat(parts)
    out.games = 1
    out.draws = 1 if winner is None else 0
    out.total_turns = game.state.turn_number
    return out


def collect_games(
    net: PolicyValueNet,
    decks_pool: list[tuple[Deck, Deck]],
    n_games: int,
    seed: int = 0,
    max_turns: int = 120,
    gamma: float = 0.997,
    lam: float = 0.95,
) -> Batch:
    """Sequentially play `n_games`, rotating deck pairings and first player."""
    batches = []
    for i in range(n_games):
        decks = decks_pool[i % len(decks_pool)]
        batches.append(
            play_selfplay_game(
                net,
                ObservationEncoder(),
                ActionCodec(),
                decks,
                seed=seed + i,
                first_player=i % 2,
                max_turns=max_turns,
                gamma=gamma,
                lam=lam,
            )
        )
    return Batch.concat(batches)


def default_decks_pool() -> list[tuple[Deck, Deck]]:
    """Both orderings of the two real starter decks."""
    from riftbound.cards import build_annie_deck, build_yi_deck

    annie, yi = build_annie_deck(), build_yi_deck()
    return [(annie, yi), (yi, annie)]


# ---------------------------------------------------------------------------
# Multiprocessing workers (Windows spawn-safe: everything reimported in child)
# ---------------------------------------------------------------------------


def _worker_collect(args: tuple) -> Batch:
    state_bytes, hidden, n_games, seed, max_turns, gamma, lam = args
    import riftbound.cards  # noqa: F401  (register the real card set)

    torch.set_num_threads(1)  # one core per worker; parallelism across workers
    encoder = ObservationEncoder()
    codec = ActionCodec()
    net = PolicyValueNet(encoder.size, codec.size, hidden=hidden)
    net.load_state_dict(torch.load(io.BytesIO(state_bytes), weights_only=True))
    return collect_games(
        net,
        default_decks_pool(),
        n_games,
        seed=seed,
        max_turns=max_turns,
        gamma=gamma,
        lam=lam,
    )


def collect_parallel(
    net: PolicyValueNet,
    n_games: int,
    workers: int,
    pool,
    seed: int = 0,
    max_turns: int = 120,
    gamma: float = 0.997,
    lam: float = 0.95,
) -> Batch:
    """Fan self-play games out to a persistent multiprocessing pool."""
    buf = io.BytesIO()
    torch.save(net.state_dict(), buf)
    state_bytes = buf.getvalue()

    per = [n_games // workers] * workers
    for i in range(n_games % workers):
        per[i] += 1
    jobs = [
        (state_bytes, net.hidden, n, seed + 100_000 * w, max_turns, gamma, lam)
        for w, n in enumerate(per)
        if n > 0
    ]
    return Batch.concat(pool.map(_worker_collect, jobs))
