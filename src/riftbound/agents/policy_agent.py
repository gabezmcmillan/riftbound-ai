"""An agent driven by a trained policy/value network.

Kept out of `riftbound.agents.__init__` eager imports so that the rest of
the package (engine, scripted agents) never pays the torch import cost.
"""

from __future__ import annotations

import random

import torch

from riftbound.agents.base import Agent
from riftbound.encoding import ActionCodec, ObservationEncoder
from riftbound.engine.actions import Action
from riftbound.engine.game import Game
from riftbound.rl.network import PolicyValueNet, masked_distribution


class PolicyAgent(Agent):
    """Plays the network's action choice; greedy (argmax) by default."""

    def __init__(
        self,
        net: PolicyValueNet,
        encoder: ObservationEncoder | None = None,
        codec: ActionCodec | None = None,
        deterministic: bool = True,
        seed: int | None = None,
    ) -> None:
        self.net = net
        self.encoder = encoder or ObservationEncoder()
        self.codec = codec or ActionCodec()
        self.deterministic = deterministic
        self.rng = random.Random(seed)
        self.net.eval()

    @classmethod
    def from_checkpoint(cls, path: str, **kwargs) -> "PolicyAgent":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        vocab = list(ckpt["vocab"])
        encoder = ObservationEncoder(vocab)
        codec = ActionCodec(vocab)
        net = PolicyValueNet(encoder.size, codec.size, hidden=ckpt["hidden"])
        net.load_state_dict(ckpt["model"])
        return cls(net, encoder=encoder, codec=codec, **kwargs)

    def choose(self, game: Game) -> Action:
        pidx = game.state.turn_player
        obs = torch.from_numpy(self.encoder.encode(game, pidx)).unsqueeze(0)
        mask = torch.from_numpy(self.codec.legal_mask(game)).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.net(obs)
        dist = masked_distribution(logits, mask)
        if self.deterministic:
            idx = int(dist.probs.argmax(dim=-1).item())
        else:
            idx = int(dist.sample().item())
        action = self.codec.decode(idx, game)
        assert action is not None
        return action
