"""Tensor encodings: observations and a flat action space with legality masks.

This is the interface layer between the rules engine and search/RL agents:

  - `ObservationEncoder.encode(game, pidx)` produces a fixed-size float32
    vector of the game from player `pidx`'s point of view, respecting hidden
    information: the opponent's hand is only a count, and deck contents/order
    are only sizes. Everything public (board, runes, gear, battlefields,
    trashes, champion zones, points) is included.

  - `ActionCodec` defines a fixed, state-independent table of discrete
    actions. `legal_mask(game)` marks which indices are currently legal and
    `decode(idx, game)` turns an index back into an engine `Action`.

Both are built against a card vocabulary (the registry at construction time),
so the encoding is stable for a fixed card pool.

Known v0 limitation (documented in docs/engine-design.md): the engine allows
moving *any* subset of ready base units as one group; the codec only exposes
single-unit moves and an "all ready base units" attack per battlefield. The
two most important group shapes are covered; arbitrary subsets are not
addressable through the flat action space yet.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from riftbound.engine.actions import (
    Action,
    EndTurn,
    MoveUnits,
    PlayGear,
    PlaySpell,
    PlayUnit,
)
from riftbound.engine.cards import REGISTRY, card
from riftbound.engine.enums import BASE, CardType, Domain, VICTORY_SCORE
from riftbound.engine.game import Game
from riftbound.engine.state import GameState, UnitState

MAX_UNITS = 12  # unit slots tracked per player
NUM_BF = 2
DOMAINS = list(Domain)

# Target reference space for spell targets:
#   [0, MAX_UNITS)                -> my unit slots (sorted by uid)
#   [MAX_UNITS, 2*MAX_UNITS)      -> enemy unit slots (sorted by uid)
#   [2*MAX_UNITS, 2*MAX_UNITS+2)  -> battlefield 0 / 1
NUM_REFS = 2 * MAX_UNITS + NUM_BF

# Unit-play destinations, in fixed order.
UNIT_DESTS = [BASE, 0, 1]


def unit_slots(state: GameState, pidx: int) -> list[UnitState]:
    """A player's units in stable slot order (ascending uid), capped."""
    units = sorted(state.player_units(pidx), key=lambda u: u.uid)
    return units[:MAX_UNITS]


# ===========================================================================
# Observations
# ===========================================================================


class ObservationEncoder:
    """Encodes a game into a fixed-size float32 vector for one player."""

    UNIT_FEATURES = 10  # per-slot features, plus a card one-hot per slot

    def __init__(self, vocab: list[str] | None = None) -> None:
        self.vocab: list[str] = sorted(vocab if vocab is not None else REGISTRY)
        self.card_index = {cid: i for i, cid in enumerate(self.vocab)}
        v = len(self.vocab)
        per_unit = v + self.UNIT_FEATURES
        self.size = (
            v  # my hand counts
            + 2 * v  # both trash counts (public)
            + 2 * v  # both champion zones (one-hot)
            + 2 * MAX_UNITS * per_unit  # unit slots, mine then enemy
            + 2 * (2 * len(DOMAINS) + 1)  # runes: ready/exhausted per domain + deck size
            + 2 * 2  # gear: ready/exhausted resource-gear counts
            + NUM_BF * 5  # battlefields: controller one-hot(3) + scored flags
            + 8  # scalars
        )

    # -- helpers -----------------------------------------------------------

    def _card_counts(self, cids: list[str]) -> np.ndarray:
        v = np.zeros(len(self.vocab), dtype=np.float32)
        for cid in cids:
            v[self.card_index[cid]] += 1.0
        return v

    def _unit_block(self, state: GameState, pidx: int) -> np.ndarray:
        v = len(self.vocab)
        block = np.zeros((MAX_UNITS, v + self.UNIT_FEATURES), dtype=np.float32)
        for slot, u in enumerate(unit_slots(state, pidx)):
            block[slot, self.card_index[u.card.id]] = 1.0
            f = block[slot, v:]
            f[0] = 1.0  # present
            f[1] = u.might / 10.0
            f[2] = u.temp_might / 10.0
            f[3] = u.temp_assault / 10.0
            f[4] = u.damage / 10.0
            f[5] = 1.0 if u.ready else 0.0
            # location one-hot: base, bf0, bf1
            f[6] = 1.0 if u.location == BASE else 0.0
            if u.location != BASE:
                f[7 + u.location] = 1.0
            f[9] = 1.0 if u.card.tank else 0.0
        return block.reshape(-1)

    def _rune_block(self, state: GameState, pidx: int) -> np.ndarray:
        p = state.players[pidx]
        out = np.zeros(2 * len(DOMAINS) + 1, dtype=np.float32)
        for r in state.player_runes(pidx):
            d = DOMAINS.index(r.domain)
            out[d if r.ready else len(DOMAINS) + d] += 1.0 / 12.0
        out[-1] = len(p.rune_deck) / 12.0
        return out

    def _gear_block(self, state: GameState, pidx: int) -> np.ndarray:
        out = np.zeros(2, dtype=np.float32)
        for g in state.player_gear(pidx):
            out[0 if g.ready else 1] += 1.0 / 5.0
        return out

    # -- main entry --------------------------------------------------------

    def encode(self, game: Game, pidx: int) -> np.ndarray:
        s = game.state
        me = s.players[pidx]
        opp = s.players[1 - pidx]

        parts: list[np.ndarray] = [
            self._card_counts(me.hand),
            self._card_counts(me.trash),
            self._card_counts(opp.trash),
            self._card_counts([me.champion] if me.champion else []),
            self._card_counts([opp.champion] if opp.champion else []),
            self._unit_block(s, pidx),
            self._unit_block(s, 1 - pidx),
            self._rune_block(s, pidx),
            self._rune_block(s, 1 - pidx),
            self._gear_block(s, pidx),
            self._gear_block(s, 1 - pidx),
        ]

        bf_block = np.zeros((NUM_BF, 5), dtype=np.float32)
        for bf in s.battlefields:
            row = bf_block[bf.index]
            if bf.controller is None:
                row[0] = 1.0
            elif bf.controller == pidx:
                row[1] = 1.0
            else:
                row[2] = 1.0
            row[3] = 1.0 if pidx in bf.scored_this_turn else 0.0
            row[4] = 1.0 if (1 - pidx) in bf.scored_this_turn else 0.0
        parts.append(bf_block.reshape(-1))

        scalars = np.array(
            [
                me.points / VICTORY_SCORE,
                opp.points / VICTORY_SCORE,
                len(me.hand) / 10.0,
                len(opp.hand) / 10.0,  # opponent hand: size only (hidden info)
                len(me.main_deck) / 39.0,
                len(opp.main_deck) / 39.0,
                1.0 if s.turn_player == pidx else 0.0,
                s.turn_number / game.max_turns,
            ],
            dtype=np.float32,
        )
        parts.append(scalars)

        obs = np.concatenate(parts)
        assert obs.shape == (self.size,), (obs.shape, self.size)
        return obs


# ===========================================================================
# Actions
# ===========================================================================


@dataclass(frozen=True)
class ActionEntry:
    kind: str  # end | play_unit | play_champion | play_gear | play_spell | move_unit | move_all
    card_id: str | None = None
    dest: int | None = None
    targets: tuple[int, ...] = ()  # spell target refs
    slot: int | None = None  # move_unit slot


class ActionCodec:
    """A fixed table of discrete actions over a card vocabulary."""

    def __init__(self, vocab: list[str] | None = None) -> None:
        self.vocab: list[str] = sorted(vocab if vocab is not None else REGISTRY)
        entries: list[ActionEntry] = [ActionEntry(kind="end")]

        for cid in self.vocab:
            c = card(cid)
            if c.type is CardType.UNIT:
                for dest in UNIT_DESTS:
                    entries.append(ActionEntry(kind="play_unit", card_id=cid, dest=dest))
            elif c.type is CardType.GEAR:
                entries.append(ActionEntry(kind="play_gear", card_id=cid))
            elif c.type is CardType.SPELL:
                specs = c.targeted_instructions()
                if not specs:
                    entries.append(ActionEntry(kind="play_spell", card_id=cid))
                else:
                    ref_lists = []
                    for ins in specs:
                        if ins.target.kind == "battlefield":
                            refs = list(range(2 * MAX_UNITS, 2 * MAX_UNITS + NUM_BF))
                        elif ins.target.owner == "friendly":
                            refs = list(range(0, MAX_UNITS))
                        elif ins.target.owner == "enemy":
                            refs = list(range(MAX_UNITS, 2 * MAX_UNITS))
                        else:
                            refs = list(range(0, 2 * MAX_UNITS))
                        ref_lists.append(refs)
                    for combo in itertools.product(*ref_lists):
                        entries.append(
                            ActionEntry(kind="play_spell", card_id=cid, targets=combo)
                        )

        # Champion zone plays (whatever champion sits there).
        for dest in UNIT_DESTS:
            entries.append(ActionEntry(kind="play_champion", dest=dest))

        # Single-unit moves by slot, plus "all ready base units" attacks.
        for slot in range(MAX_UNITS):
            for dest in UNIT_DESTS:
                entries.append(ActionEntry(kind="move_unit", slot=slot, dest=dest))
        for bf in range(NUM_BF):
            entries.append(ActionEntry(kind="move_all", dest=bf))

        self.entries = entries
        self.size = len(entries)

    # -- decoding ----------------------------------------------------------

    def _resolve_ref(self, game: Game, pidx: int, ref: int) -> int | None:
        """Map a target ref to a unit uid or battlefield index."""
        if ref >= 2 * MAX_UNITS:
            return ref - 2 * MAX_UNITS  # battlefield index
        owner = pidx if ref < MAX_UNITS else 1 - pidx
        slots = unit_slots(game.state, owner)
        slot = ref % MAX_UNITS
        if slot >= len(slots):
            return None
        return slots[slot].uid

    def decode(self, idx: int, game: Game) -> Action | None:
        """Turn an action index into an engine Action for the turn player.

        Returns None when the entry doesn't map to anything in the current
        state (e.g. card not in hand, empty unit slot)."""
        e = self.entries[idx]
        s = game.state
        pidx = s.turn_player
        p = s.players[pidx]

        if e.kind == "end":
            return EndTurn()
        if e.kind == "play_unit":
            if e.card_id not in p.hand:
                return None
            return PlayUnit(hand_index=p.hand.index(e.card_id), destination=e.dest)
        if e.kind == "play_champion":
            if p.champion is None:
                return None
            return PlayUnit(hand_index=-1, destination=e.dest)
        if e.kind == "play_gear":
            if e.card_id not in p.hand:
                return None
            return PlayGear(hand_index=p.hand.index(e.card_id))
        if e.kind == "play_spell":
            if e.card_id not in p.hand:
                return None
            refs = []
            for ref in e.targets:
                resolved = self._resolve_ref(game, pidx, ref)
                if resolved is None:
                    return None
                refs.append(resolved)
            return PlaySpell(
                hand_index=p.hand.index(e.card_id), targets=tuple(refs)
            )
        if e.kind == "move_unit":
            slots = unit_slots(s, pidx)
            if e.slot >= len(slots):
                return None
            return MoveUnits(unit_uids=(slots[e.slot].uid,), destination=e.dest)
        if e.kind == "move_all":
            uids = tuple(sorted(u.uid for u in s.units_at(pidx, BASE) if u.ready))
            if not uids:
                return None
            return MoveUnits(unit_uids=uids, destination=e.dest)
        raise ValueError(e.kind)

    # -- legality ----------------------------------------------------------

    def legal_mask(self, game: Game) -> np.ndarray:
        """Boolean mask over the action table for the current turn player."""
        mask = np.zeros(self.size, dtype=bool)
        legal = set(game.legal_actions())
        for idx in range(self.size):
            action = self.decode(idx, game)
            if action is not None and action in legal:
                mask[idx] = True
        return mask

    def encode_action(self, action: Action, game: Game) -> int | None:
        """Find the table index that decodes to `action`, if any."""
        for idx in range(self.size):
            if self.decode(idx, game) == action:
                return idx
        return None
