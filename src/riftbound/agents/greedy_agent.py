"""A one-ply greedy agent: simulate each legal action, evaluate, pick best.

This is the scripted baseline that learning agents must beat. Known
limitation: simulation deep-copies the game *including its RNG*, so an action
whose outcome depends on hidden information (e.g. what a draw effect yields)
is evaluated with perfect foresight. Acceptable for a baseline; a fair
determinized rollout is future work.
"""

from __future__ import annotations

import copy
import random

from riftbound.agents.base import Agent
from riftbound.engine.actions import Action
from riftbound.engine.enums import BASE
from riftbound.engine.game import Game


def evaluate(game: Game, pidx: int) -> float:
    """Static evaluation of the position from player `pidx`'s perspective."""
    s = game.state
    if s.winner == pidx:
        return 1e9
    if s.winner == 1 - pidx:
        return -1e9
    me = s.players[pidx]
    opp = s.players[1 - pidx]

    score = 1000.0 * (me.points - opp.points)
    for bf in s.battlefields:
        if bf.controller == pidx:
            score += 300.0
        elif bf.controller == 1 - pidx:
            score -= 300.0
    my_units = s.player_units(pidx)
    opp_units = s.player_units(1 - pidx)
    score += 10.0 * (sum(u.might for u in my_units) - sum(u.might for u in opp_units))
    # Slight preference for units on battlefields (they hold ground).
    score += 5.0 * sum(1 for u in my_units if u.location != BASE)
    score += 4.0 * len(me.hand)
    score += 1.0 * len(s.player_runes(pidx))
    return score


class GreedyAgent(Agent):
    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def choose(self, game: Game) -> Action:
        pidx = game.state.turn_player
        actions = game.legal_actions()
        self.rng.shuffle(actions)
        best_action, best_score = None, float("-inf")
        for action in actions:
            sim = copy.deepcopy(game)
            try:
                sim.step(action)
            except Exception:
                continue
            value = evaluate(sim, pidx)
            if value > best_score:
                best_action, best_score = action, value
        assert best_action is not None
        return best_action
