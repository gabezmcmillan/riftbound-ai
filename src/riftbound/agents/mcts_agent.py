"""Determinized MCTS: the first search-based agent.

Riftbound has hidden information (the opponent's hand, both deck orders), so
plain MCTS doesn't apply directly. This agent uses *determinization*: sample
several complete worlds consistent with what the player can see, run a
standard UCT search in each fully-determined world, and pick the root action
with the highest aggregated visit count ("root parallelization" across
worlds).

Design notes:
  - After determinization the game is fully deterministic (deck orders are
    fixed), so vanilla UCT applies.
  - Leaf evaluation reuses the greedy agent's static evaluation, squashed to
    [0, 1] from the searching player's perspective, instead of random
    rollouts (rollouts are slow in Python and very noisy here).
  - Branching control: the engine enumerates every subset of ready base
    units as a group move; the search prunes those to single-unit moves and
    the all-in group, mirroring the action codec.
  - Assumes the opponent's decklist is known (true in self-play; standard
    practice for card-game agents).
"""

from __future__ import annotations

import copy
import math
import random

from riftbound.agents.base import Agent
from riftbound.agents.greedy_agent import evaluate
from riftbound.engine.actions import Action, MoveUnits
from riftbound.engine.enums import BASE
from riftbound.engine.game import Game


def determinize(game: Game, perspective: int, rng: random.Random) -> Game:
    """Clone the game and resample everything `perspective` cannot see:
    the opponent's hand (redealt from hand + main deck), both main-deck
    orders, and both rune-deck orders. Public zones are untouched."""
    g = copy.deepcopy(game)
    # Decouple the clone's future shuffles/draws from the real game.
    g.rng = random.Random(rng.getrandbits(64))

    me = g.state.players[perspective]
    opp = g.state.players[1 - perspective]

    # Opponent hand: pool with their deck and redeal the same hand size.
    pool = opp.hand + opp.main_deck
    rng.shuffle(pool)
    hand_size = len(opp.hand)
    opp.hand = pool[:hand_size]
    opp.main_deck = pool[hand_size:]

    # Own deck order is also unknown.
    rng.shuffle(me.main_deck)
    # Rune deck contents are known (decklist) but order is not.
    rng.shuffle(me.rune_deck)
    rng.shuffle(opp.rune_deck)
    return g


def reduced_actions(game: Game) -> list[Action]:
    """Engine legal actions with group moves pruned to singletons plus the
    all-in attack, to keep the search branching factor manageable."""
    s = game.state
    all_ready = {
        tuple(sorted(u.uid for u in s.units_at(s.turn_player, BASE) if u.ready))
    }
    out = []
    for a in game.legal_actions():
        if isinstance(a, MoveUnits) and len(a.unit_uids) > 1:
            if a.unit_uids not in all_ready:
                continue
        out.append(a)
    return out


class _Node:
    __slots__ = ("game", "player", "children", "untried", "visits", "value_sum")

    def __init__(self, game: Game, rng: random.Random) -> None:
        self.game = game
        self.player = game.state.turn_player
        self.children: dict[Action, _Node] = {}
        self.untried = reduced_actions(game)
        rng.shuffle(self.untried)
        self.visits = 0
        self.value_sum = 0.0  # from the searching player's perspective


def _leaf_value(game: Game, root_player: int) -> float:
    """Value in [0, 1] from `root_player`'s perspective."""
    if game.is_over:
        if game.winner == root_player:
            return 1.0
        if game.winner == 1 - root_player:
            return 0.0
        return 0.5  # turn-limit draw
    score = evaluate(game, root_player)
    return 1.0 / (1.0 + math.exp(-score / 1000.0))


class MCTSAgent(Agent):
    def __init__(
        self,
        iterations: int = 60,
        determinizations: int = 4,
        c_uct: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.iterations = iterations
        self.determinizations = determinizations
        self.c_uct = c_uct
        self.rng = random.Random(seed)

    # -- UCT ---------------------------------------------------------------

    def _select_child(self, node: _Node, root_player: int) -> tuple[Action, _Node]:
        log_n = math.log(node.visits)
        best, best_score = None, -math.inf
        for action, child in node.children.items():
            q = child.value_sum / child.visits
            if node.player != root_player:
                q = 1.0 - q  # opponent minimizes the root player's value
            score = q + self.c_uct * math.sqrt(log_n / child.visits)
            if score > best_score:
                best, best_score = (action, child), score
        assert best is not None
        return best

    def _simulate(self, node: _Node, root_player: int) -> float:
        path = [node]
        # Selection.
        while not node.game.is_over and not node.untried and node.children:
            _, node = self._select_child(node, root_player)
            path.append(node)
        # Expansion.
        if not node.game.is_over and node.untried:
            action = node.untried.pop()
            child_game = copy.deepcopy(node.game)
            child_game.step(action)
            child = _Node(child_game, self.rng)
            node.children[action] = child
            node = child
            path.append(node)
        # Evaluation + backpropagation.
        value = _leaf_value(node.game, root_player)
        for n in path:
            n.visits += 1
            n.value_sum += value
        return value

    # -- public ------------------------------------------------------------

    def choose(self, game: Game) -> Action:
        root_player = game.state.turn_player
        actions = reduced_actions(game)
        if len(actions) == 1:
            return actions[0]

        visit_totals: dict[Action, int] = {}
        value_totals: dict[Action, float] = {}
        for _ in range(self.determinizations):
            world = determinize(game, root_player, self.rng)
            root = _Node(world, self.rng)
            for _ in range(self.iterations):
                self._simulate(root, root_player)
            for action, child in root.children.items():
                visit_totals[action] = visit_totals.get(action, 0) + child.visits
                value_totals[action] = value_totals.get(action, 0.0) + child.value_sum

        # Most-visited action across worlds; break ties by mean value.
        best = max(
            visit_totals,
            key=lambda a: (visit_totals[a], value_totals[a] / visit_totals[a]),
        )
        return best
