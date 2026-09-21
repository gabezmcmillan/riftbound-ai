"""Self-play PPO training loop.

Usage (from the repo root):

    python -m riftbound.rl.train --iterations 200 --games-per-iter 48 --workers 10

Each iteration: collect self-play games with the current policy (parallel
across CPU worker processes), compute GAE advantages, run a few epochs of
clipped-PPO minibatch updates, checkpoint, and periodically evaluate the
greedy-argmax policy against the random and greedy baselines.

Checkpoints (`checkpoints/latest.pt` plus periodic numbered snapshots)
contain the model, optimizer, iteration counter, and the card vocabulary the
encodings were built against, so training is resumable and evaluation is
reproducible via `PolicyAgent.from_checkpoint`.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from riftbound.rl.network import PolicyValueNet, masked_distribution
from riftbound.rl.selfplay import Batch, collect_games, collect_parallel, default_decks_pool


@dataclass
class PPOConfig:
    iterations: int = 200
    games_per_iter: int = 48
    workers: int = 0  # 0 = collect in-process
    hidden: int = 512
    lr: float = 3e-4
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 1024
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    max_turns: int = 120
    seed: int = 0
    eval_every: int = 10
    eval_games: int = 20
    checkpoint_every: int = 25
    checkpoint_dir: str = "checkpoints"


def ppo_update(
    net: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    cfg: PPOConfig,
    rng: np.random.Generator,
) -> dict[str, float]:
    """A few epochs of clipped-PPO minibatch updates on one batch."""
    obs = torch.from_numpy(batch.obs)
    mask = torch.from_numpy(batch.mask)
    action = torch.from_numpy(batch.action)
    old_logp = torch.from_numpy(batch.logp)
    ret = torch.from_numpy(batch.ret)
    adv = torch.from_numpy(batch.advantage)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    n = len(batch)
    metrics: dict[str, list[float]] = {
        "policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []
    }
    net.train()
    for _ in range(cfg.epochs):
        order = rng.permutation(n)
        for start in range(0, n, cfg.minibatch):
            mb = torch.from_numpy(order[start : start + cfg.minibatch].copy())
            logits, value = net(obs[mb])
            dist = masked_distribution(logits, mask[mb])
            logp = dist.log_prob(action[mb])
            ratio = torch.exp(logp - old_logp[mb])
            mb_adv = adv[mb]

            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = torch.nn.functional.mse_loss(value, ret[mb])
            entropy = dist.entropy().mean()
            loss = (
                policy_loss
                + cfg.value_coef * value_loss
                - cfg.entropy_coef * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                metrics["policy_loss"].append(float(policy_loss))
                metrics["value_loss"].append(float(value_loss))
                metrics["entropy"].append(float(entropy))
                metrics["approx_kl"].append(float((old_logp[mb] - logp).mean()))
                metrics["clip_frac"].append(
                    float((torch.abs(ratio - 1) > cfg.clip).float().mean())
                )
    return {k: float(np.mean(v)) for k, v in metrics.items()}


def evaluate(net: PolicyValueNet, games: int, seed: int) -> dict[str, float]:
    """Win rate of the greedy-argmax policy vs the scripted ladder."""
    from riftbound.agents import GreedyAgent, RandomAgent
    from riftbound.agents.policy_agent import PolicyAgent
    from riftbound.sim import run_match

    results: dict[str, float] = {}
    pool = default_decks_pool()
    for name, make_opp in (("random", RandomAgent), ("greedy", GreedyAgent)):
        wins = total = 0
        for j, decks in enumerate(pool):
            stats = run_match(
                decks,
                (PolicyAgent(net), make_opp(seed=seed + j)),
                games=games // len(pool),
                seed=seed + 31 * j,
            )
            wins += stats.wins[0]
            total += stats.games
        results[f"winrate_vs_{name}"] = wins / total
    return results


def save_checkpoint(
    path: Path, net: PolicyValueNet, optimizer, iteration: int, cfg: PPOConfig, vocab: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "hidden": cfg.hidden,
            "vocab": vocab,
            "config": asdict(cfg),
        },
        path,
    )


def train(cfg: PPOConfig) -> PolicyValueNet:
    import riftbound.cards  # noqa: F401  (register the real card set)
    from riftbound.encoding import ActionCodec, ObservationEncoder

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    encoder = ObservationEncoder()
    codec = ActionCodec()
    net = PolicyValueNet(encoder.size, codec.size, hidden=cfg.hidden)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    start_iter = 0

    ckpt_dir = Path(cfg.checkpoint_dir)
    latest = ckpt_dir / "latest.pt"
    if latest.exists():
        ckpt = torch.load(latest, weights_only=True)
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt["iteration"]
        print(f"resumed from {latest} at iteration {start_iter}")

    log_path = ckpt_dir / "train_log.jsonl"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    pool = None
    if cfg.workers > 0:
        pool = mp.get_context("spawn").Pool(cfg.workers)

    try:
        for it in range(start_iter, cfg.iterations):
            t0 = time.perf_counter()
            seed = cfg.seed + it * 1_000_003
            if pool is not None:
                batch = collect_parallel(
                    net, cfg.games_per_iter, cfg.workers, pool, seed=seed,
                    max_turns=cfg.max_turns, gamma=cfg.gamma, lam=cfg.gae_lambda,
                )
            else:
                batch = collect_games(
                    net, default_decks_pool(), cfg.games_per_iter, seed=seed,
                    max_turns=cfg.max_turns, gamma=cfg.gamma, lam=cfg.gae_lambda,
                )
            t_collect = time.perf_counter() - t0

            t0 = time.perf_counter()
            metrics = ppo_update(net, optimizer, batch, cfg, rng)
            t_update = time.perf_counter() - t0

            row = {
                "iteration": it + 1,
                "steps": len(batch),
                "games": batch.games,
                "draws": batch.draws,
                "avg_turns": batch.total_turns / batch.games,
                "collect_s": round(t_collect, 2),
                "update_s": round(t_update, 2),
                **{k: round(v, 5) for k, v in metrics.items()},
            }

            if cfg.eval_every and (it + 1) % cfg.eval_every == 0:
                row.update(
                    {k: round(v, 3) for k, v in
                     evaluate(net, cfg.eval_games, seed=cfg.seed + it).items()}
                )

            save_checkpoint(latest, net, optimizer, it + 1, cfg, encoder.vocab)
            if cfg.checkpoint_every and (it + 1) % cfg.checkpoint_every == 0:
                save_checkpoint(
                    ckpt_dir / f"iter_{it + 1:05d}.pt", net, optimizer, it + 1, cfg, encoder.vocab
                )

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row))
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return net


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = PPOConfig()
    for f, v in asdict(defaults).items():
        flag = "--" + f.replace("_", "-")
        if isinstance(v, bool):
            parser.add_argument(flag, action="store_true")
        else:
            parser.add_argument(flag, type=type(v), default=v)
    args = parser.parse_args()
    cfg = PPOConfig(**vars(args))
    train(cfg)


if __name__ == "__main__":
    main()
