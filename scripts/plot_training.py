"""Plot learning curves from checkpoints/train_log.jsonl."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = Path(sys.argv[1] if len(sys.argv) > 1 else "checkpoints/train_log.jsonl")
rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

evals = [r for r in rows if "winrate_vs_random" in r]
ax = axes[0]
ax.plot([r["iteration"] for r in evals], [r["winrate_vs_random"] for r in evals],
        "o-", label="vs random")
ax.plot([r["iteration"] for r in evals], [r["winrate_vs_greedy"] for r in evals],
        "s-", label="vs greedy")
ax.axhline(0.5, color="gray", ls=":", lw=1)
ax.set(xlabel="iteration", ylabel="win rate", title="Eval win rate (argmax policy)",
       ylim=(-0.02, 1.02))
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot([r["iteration"] for r in rows], [r["entropy"] for r in rows], color="tab:purple")
ax.set(xlabel="iteration", ylabel="entropy (nats)", title="Policy entropy")
ax.grid(alpha=0.3)

ax = axes[2]
ax.plot([r["iteration"] for r in rows], [r["value_loss"] for r in rows], color="tab:red")
ax.set(xlabel="iteration", ylabel="value MSE", title="Value loss")
ax.grid(alpha=0.3)

fig.tight_layout()
out = log.parent / "training_curves.png"
fig.savefig(out, dpi=120)
print(out)
