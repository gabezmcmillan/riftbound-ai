"""The Riftbound v0 rules engine for 1v1 Duel.

Implements the Core Rules turn structure with the simplifications documented
in docs/engine-design.md. The most important one: there is no reaction chain
in v0 — spells and abilities resolve immediately, and showdowns/combats have
no Action/Reaction windows. This makes the game a sequential-move game where
agents only act during their own Main Phase.

Rule references (CR = Core Rules numbering, docs/rules-reference.md):
  - Turn structure: CR 315-317
  - Standard Move: CR 144
  - Combat: CR 459-466
  - Scoring (conquer/hold, final point restriction): CR 467-472
  - Burn Out: CR 431
  - 1v1 Duel mode: CR 485
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field

from riftbound.engine.actions import (
    Action,
    EndTurn,
    MoveUnits,
    PlayGear,
    PlaySpell,
    PlayUnit,
)
from riftbound.engine.cards import CardDef, Instruction, battlefield, card
from riftbound.engine.enums import (
    BASE,
    CardType,
    Domain,
    NUM_BATTLEFIELDS,
    OPENING_HAND,
    Phase,
    VICTORY_SCORE,
)
from riftbound.engine.state import (
    BattlefieldState,
    GameState,
    GearState,
    PlayerState,
    RuneState,
    UnitState,
)

# Cap on enumerated target combinations per spell and on movable-unit subset
# size, to bound the size of the legal action list.
MAX_TARGET_COMBOS = 60
MAX_MOVE_GROUP_POOL = 10


@dataclass
class Deck:
    """One player's deck for a 1v1 Duel."""

    legend_name: str
    legend_domains: tuple[Domain, ...]
    champion: str  # card id of the Chosen Champion
    main: list[str]  # 39 card ids (the Chosen Champion is the 40th)
    runes: list[Domain]  # 12 rune domains
    battlefield: str  # battlefield card id this player brings


@dataclass
class PaymentPlan:
    exhaust_rune_uids: list[int] = field(default_factory=list)
    recycle_rune_uids: list[int] = field(default_factory=list)
    exhaust_gear_uids: list[int] = field(default_factory=list)
    energy_from_pool: int = 0
    power_from_pool: dict[Domain, int] = field(default_factory=dict)


class IllegalAction(Exception):
    pass


class Game:
    def __init__(
        self,
        decks: tuple[Deck, Deck],
        seed: int | None = None,
        first_player: int | None = None,
        max_turns: int = 120,
    ) -> None:
        self.rng = random.Random(seed)
        self.max_turns = max_turns
        self.turns_taken = [0, 0]

        players = []
        for i, deck in enumerate(decks):
            p = PlayerState(
                index=i,
                legend_name=deck.legend_name,
                legend_domains=tuple(deck.legend_domains),
                main_deck=list(deck.main),
                rune_deck=list(deck.runes),
                champion=deck.champion,
            )
            self.rng.shuffle(p.main_deck)
            self.rng.shuffle(p.rune_deck)
            players.append(p)

        battlefields = [
            BattlefieldState(index=i, card=battlefield(decks[i].battlefield), owner=i)
            for i in range(NUM_BATTLEFIELDS)
        ]

        if first_player is None:
            first_player = self.rng.randrange(2)

        self.state = GameState(
            players=players,
            battlefields=battlefields,
            turn_player=first_player,
        )

        # Opening hands (CR 116). Mulligans are skipped in v0.
        for p in players:
            for _ in range(OPENING_HAND):
                p.hand.append(p.main_deck.pop())

        self._begin_turn()

    # ==================================================================
    # Public API
    # ==================================================================

    @property
    def is_over(self) -> bool:
        return self.state.winner is not None or self.state.turn_number > self.max_turns

    @property
    def winner(self) -> int | None:
        return self.state.winner

    def legal_actions(self) -> list[Action]:
        if self.is_over:
            return []
        s = self.state
        pidx = s.turn_player
        p = s.players[pidx]
        actions: list[Action] = [EndTurn()]

        # --- plays from hand and the Champion Zone -----------------------
        play_sources: list[tuple[int, str]] = list(enumerate(p.hand))
        if p.champion is not None:
            play_sources.append((-1, p.champion))
        seen_cards: set[tuple[str, int]] = set()
        for idx, cid in play_sources:
            if (cid, idx < 0) in seen_cards:
                continue  # duplicate copies in hand produce identical actions
            seen_cards.add((cid, idx < 0))
            c = card(cid)
            if self._payment_plan(pidx, c.energy, dict(c.power)) is None:
                continue
            if c.type is CardType.UNIT:
                dests = [BASE] + [
                    bf.index for bf in s.battlefields if bf.controller == pidx
                ]
                for dest in dests:
                    actions.append(PlayUnit(hand_index=idx, destination=dest))
            elif c.type is CardType.GEAR:
                if idx >= 0:
                    actions.append(PlayGear(hand_index=idx))
            elif c.type is CardType.SPELL:
                if idx >= 0:
                    actions.extend(self._spell_actions(pidx, idx, c))

        # --- standard moves (CR 144) -------------------------------------
        # Groups from base to a battlefield (this is where combat happens).
        base_ready = sorted(
            u.uid for u in s.units_at(pidx, BASE) if u.ready
        )[:MAX_MOVE_GROUP_POOL]
        for bf in s.battlefields:
            for size in range(1, len(base_ready) + 1):
                for combo in itertools.combinations(base_ready, size):
                    actions.append(MoveUnits(unit_uids=combo, destination=bf.index))
        for bf in s.battlefields:
            for u in s.units_at(pidx, bf.index):
                if not u.ready:
                    continue
                # Single-unit returns to base (grouping adds nothing
                # tactically: returning home can't trigger combat).
                actions.append(MoveUnits(unit_uids=(u.uid,), destination=BASE))
                # Ganking units may move battlefield -> battlefield.
                if u.card.ganking:
                    for other in s.battlefields:
                        if other.index != bf.index:
                            actions.append(
                                MoveUnits(unit_uids=(u.uid,), destination=other.index)
                            )

        return actions

    def step(self, action: Action) -> None:
        if self.is_over:
            raise IllegalAction("game is over")
        if self.state.phase is not Phase.MAIN:
            raise IllegalAction(f"cannot act during {self.state.phase}")

        if isinstance(action, EndTurn):
            self._end_turn()
        elif isinstance(action, PlayUnit):
            self._play_unit(action)
        elif isinstance(action, PlaySpell):
            self._play_spell(action)
        elif isinstance(action, PlayGear):
            self._play_gear(action)
        elif isinstance(action, MoveUnits):
            self._move_units(action)
        else:
            raise IllegalAction(f"unknown action {action!r}")

        self._cleanup()

    # ==================================================================
    # Turn structure (CR 315-317)
    # ==================================================================

    def _begin_turn(self) -> None:
        s = self.state
        pidx = s.turn_player

        # Scoring marks reset every turn (CR 470: once per battlefield per turn).
        for bf in s.battlefields:
            bf.scored_this_turn.clear()

        # Awaken Phase (CR 315.1): ready everything the turn player controls.
        s.phase = Phase.AWAKEN
        for u in s.player_units(pidx):
            u.ready = True
        for r in s.player_runes(pidx):
            r.ready = True
        for g in s.player_gear(pidx):
            g.ready = True

        # Beginning Phase, Scoring Step (CR 315.2.b): hold all controlled
        # battlefields.
        s.phase = Phase.BEGINNING
        for bf in s.battlefields:
            if bf.controller == pidx:
                self._score(pidx, bf, conquer=False)
                if s.winner is not None:
                    return

        # Channel Phase (CR 315.3): 2 runes; the second player channels an
        # extra rune on their first turn (CR 485.7).
        s.phase = Phase.CHANNEL
        count = 2
        if self.turns_taken[pidx] == 0 and pidx != self._first_player():
            count = 3
        self._channel(pidx, count)

        # Draw Phase (CR 315.4): draw 1, with Burn Out handling (CR 431).
        s.phase = Phase.DRAW
        self._draw_cards(pidx, 1)
        if s.winner is not None:
            return

        # Main Phase (CR 316): rune pools empty at its start.
        s.phase = Phase.MAIN
        for pl in s.players:
            pl.clear_pools()
        self.turns_taken[pidx] += 1

    def _first_player(self) -> int:
        # The player who took (or is taking) turn 1.
        if self.turns_taken == [0, 0]:
            return self.state.turn_player
        return 0 if self.turns_taken[0] >= self.turns_taken[1] else 1

    def _end_turn(self) -> None:
        s = self.state
        # Ending Phase (CR 317.2): heal all units, expire "this turn"
        # effects, empty rune pools.
        s.phase = Phase.ENDING
        for u in s.units.values():
            u.damage = 0
            u.temp_might = 0
            u.temp_assault = 0
        for p in s.players:
            p.clear_pools()

        s.turn_player = 1 - s.turn_player
        s.turn_number += 1
        if s.turn_number > self.max_turns:
            return
        self._begin_turn()

    # ==================================================================
    # Resources
    # ==================================================================

    def _channel(self, pidx: int, count: int) -> None:
        """Channel runes from the top of the rune deck (CR 430)."""
        s = self.state
        p = s.players[pidx]
        for _ in range(count):
            if not p.rune_deck:
                break  # channel as many as possible (CR 430.3)
            domain = p.rune_deck.pop()
            uid = s.new_uid()
            s.runes[uid] = RuneState(uid=uid, domain=domain, controller=pidx, ready=True)

    def _payment_plan(
        self, pidx: int, energy: int, power: dict[Domain, int]
    ) -> PaymentPlan | None:
        """Compute how to pay a cost, or None if unaffordable.

        Auto-payment policy (a v0 simplification -- rune spending is not an
        agent decision):
          - Power pips: pay from the power pool, then ready resource gear
            (Seal cycle), then recycle exhausted runes of the matching
            domain, then recycle ready runes.
          - Energy: pay from the energy pool, then exhaust remaining ready
            runes.
        """
        s = self.state
        p = s.players[pidx]
        plan = PaymentPlan()

        ready_runes = {r.uid: r for r in s.player_runes(pidx) if r.ready}
        exhausted_runes = {r.uid: r for r in s.player_runes(pidx) if not r.ready}
        ready_seals = [
            g
            for g in s.player_gear(pidx)
            if g.ready and g.card.gear_power_domain is not None
        ]

        for domain, need in power.items():
            if need <= 0:
                continue
            from_pool = min(need, p.power_pool.get(domain, 0))
            if from_pool:
                plan.power_from_pool[domain] = from_pool
                need -= from_pool
            for g in ready_seals:
                if need == 0:
                    break
                if (
                    g.card.gear_power_domain == domain
                    and g.uid not in plan.exhaust_gear_uids
                ):
                    plan.exhaust_gear_uids.append(g.uid)
                    need -= 1
            for pool in (exhausted_runes, ready_runes):
                for uid, r in list(pool.items()):
                    if need == 0:
                        break
                    if r.domain == domain:
                        plan.recycle_rune_uids.append(uid)
                        del pool[uid]
                        need -= 1
            if need > 0:
                return None

        energy_needed = energy
        plan.energy_from_pool = min(energy_needed, p.energy_pool)
        energy_needed -= plan.energy_from_pool
        for uid in list(ready_runes):
            if energy_needed == 0:
                break
            plan.exhaust_rune_uids.append(uid)
            del ready_runes[uid]
            energy_needed -= 1
        if energy_needed > 0:
            return None
        return plan

    def _pay(self, pidx: int, plan: PaymentPlan) -> None:
        s = self.state
        p = s.players[pidx]
        p.energy_pool -= plan.energy_from_pool
        for domain, amount in plan.power_from_pool.items():
            p.power_pool[domain] = p.power_pool.get(domain, 0) - amount
        for uid in plan.exhaust_rune_uids:
            s.runes[uid].ready = False
        for uid in plan.exhaust_gear_uids:
            s.gear[uid].ready = False
        for uid in plan.recycle_rune_uids:
            rune = s.runes.pop(uid)
            # Recycled runes go to the bottom of the rune deck (CR 416.1).
            p.rune_deck.insert(0, rune.domain)

    # ==================================================================
    # Card plays
    # ==================================================================

    def _hand_card(self, pidx: int, hand_index: int) -> CardDef:
        p = self.state.players[pidx]
        if hand_index == -1:
            if p.champion is None:
                raise IllegalAction("champion zone is empty")
            return card(p.champion)
        try:
            return card(p.hand[hand_index])
        except IndexError as exc:
            raise IllegalAction("bad hand index") from exc

    def _remove_from_hand(self, pidx: int, hand_index: int) -> str:
        p = self.state.players[pidx]
        if hand_index == -1:
            cid, p.champion = p.champion, None
            assert cid is not None
            return cid
        return p.hand.pop(hand_index)

    def _play_unit(self, action: PlayUnit) -> None:
        s = self.state
        pidx = s.turn_player
        c = self._hand_card(pidx, action.hand_index)
        if c.type is not CardType.UNIT:
            raise IllegalAction(f"{c.name} is not a unit")
        dest = action.destination
        if dest != BASE:
            bf = s.battlefields[dest]
            if bf.controller != pidx:
                raise IllegalAction(
                    "units can only be played to base or a controlled battlefield"
                )
        plan = self._payment_plan(pidx, c.energy, dict(c.power))
        if plan is None:
            raise IllegalAction(f"cannot afford {c.name}")
        self._pay(pidx, plan)
        self._remove_from_hand(pidx, action.hand_index)

        uid = s.new_uid()
        # Units enter the board exhausted (CR 143.4) unless stated otherwise.
        s.units[uid] = UnitState(
            uid=uid, card=c, controller=pidx, location=dest, ready=c.enters_ready
        )
        # "When you play me" effects (non-targeted in v0).
        for ins in c.instructions:
            self._execute_instruction(pidx, ins, targets=iter(()))

    def _play_gear(self, action: PlayGear) -> None:
        s = self.state
        pidx = s.turn_player
        c = self._hand_card(pidx, action.hand_index)
        if c.type is not CardType.GEAR:
            raise IllegalAction(f"{c.name} is not gear")
        plan = self._payment_plan(pidx, c.energy, dict(c.power))
        if plan is None:
            raise IllegalAction(f"cannot afford {c.name}")
        self._pay(pidx, plan)
        self._remove_from_hand(pidx, action.hand_index)
        uid = s.new_uid()
        # Gear enters ready at its controller's base (CR 149.1, 359.2.d).
        s.gear[uid] = GearState(uid=uid, card=c, controller=pidx, ready=True)
        for ins in c.instructions:
            self._execute_instruction(pidx, ins, targets=iter(()))

    def _spell_actions(self, pidx: int, hand_index: int, c: CardDef) -> list[Action]:
        specs = c.targeted_instructions()
        if not specs:
            return [PlaySpell(hand_index=hand_index, targets=())]
        option_lists = []
        for ins in specs:
            options = self._target_options(pidx, ins)
            if not options:
                return []  # all targets must have valid choices (CR 355.8)
            option_lists.append(options)
        combos = itertools.islice(itertools.product(*option_lists), MAX_TARGET_COMBOS)
        return [PlaySpell(hand_index=hand_index, targets=tuple(combo)) for combo in combos]

    def _target_options(self, pidx: int, ins: Instruction) -> list[int]:
        assert ins.target is not None
        if ins.target.kind == "battlefield":
            return [bf.index for bf in self.state.battlefields]
        return [
            u.uid
            for u in self.state.units.values()
            if self._valid_unit_target(pidx, ins, u)
        ]

    def _valid_unit_target(self, pidx: int, ins: Instruction, u: UnitState) -> bool:
        assert ins.target is not None
        spec = ins.target
        if spec.owner == "friendly" and u.controller != pidx:
            return False
        if spec.owner == "enemy" and u.controller == pidx:
            return False
        if spec.at_battlefield_only and u.location == BASE:
            return False
        return True

    def _play_spell(self, action: PlaySpell) -> None:
        s = self.state
        pidx = s.turn_player
        c = self._hand_card(pidx, action.hand_index)
        if c.type is not CardType.SPELL:
            raise IllegalAction(f"{c.name} is not a spell")
        specs = c.targeted_instructions()
        if len(action.targets) != len(specs):
            raise IllegalAction("wrong number of targets")
        for ins, ref in zip(specs, action.targets):
            if ins.target.kind == "battlefield":
                if not 0 <= ref < len(s.battlefields):
                    raise IllegalAction(f"invalid battlefield target {ref}")
            else:
                u = s.units.get(ref)
                if u is None or not self._valid_unit_target(pidx, ins, u):
                    raise IllegalAction(f"invalid target {ref} for {c.name}")
        plan = self._payment_plan(pidx, c.energy, dict(c.power))
        if plan is None:
            raise IllegalAction(f"cannot afford {c.name}")
        self._pay(pidx, plan)
        cid = self._remove_from_hand(pidx, action.hand_index)

        targets = iter(action.targets)
        for ins in c.instructions:
            self._execute_instruction(pidx, ins, targets)
        # Spells go to the trash after resolving (CR 157).
        s.players[pidx].trash.append(cid)

    # ==================================================================
    # Effect execution
    # ==================================================================

    def _bonus_damage(self, pidx: int) -> int:
        """Extra damage per instance from friendly passives (Annie, Fiery)."""
        return sum(
            u.card.bonus_spell_damage for u in self.state.player_units(pidx)
        )

    def _execute_instruction(self, pidx: int, ins: Instruction, targets) -> None:
        s = self.state
        if ins.target is not None:
            ref = next(targets)
            if ins.target.kind == "battlefield":
                if ins.op == "damage_all_at_bf":
                    amount = ins.amount + self._bonus_damage(pidx)
                    for u in s.units_at(1 - pidx, ref):
                        u.damage += amount
                else:
                    raise ValueError(f"unknown battlefield op {ins.op}")
                return
            u = s.units.get(ref)
            if u is None:
                return  # target died mid-resolution; instruction is ignored
            if ins.op == "damage":
                u.damage += ins.amount + self._bonus_damage(pidx)
            elif ins.op == "kill":
                self._kill_unit(u)
            elif ins.op == "buff":
                u.temp_might += ins.amount
            elif ins.op == "buff_assault":
                u.temp_assault += ins.amount
            else:
                raise ValueError(f"unknown targeted op {ins.op}")
            return
        if ins.op == "draw":
            self._draw_cards(pidx, ins.amount)
        elif ins.op == "channel":
            self._channel(pidx, ins.amount)
        elif ins.op == "ready_runes":
            exhausted = [r for r in s.player_runes(pidx) if not r.ready]
            for r in exhausted[: ins.amount]:
                r.ready = True
        elif ins.op == "gain_point":
            for _ in range(ins.amount):
                self._gain_point(pidx)
        elif ins.op == "damage_all_bf_units":
            amount = ins.amount + self._bonus_damage(pidx)
            for u in s.units.values():
                if u.location != BASE:
                    u.damage += amount
        else:
            raise ValueError(f"unknown op {ins.op}")

    # ==================================================================
    # Movement, showdowns, combat (CR 144, 341, 459-466)
    # ==================================================================

    def _move_units(self, action: MoveUnits) -> None:
        s = self.state
        pidx = s.turn_player
        units = []
        for uid in action.unit_uids:
            u = s.units.get(uid)
            if u is None or u.controller != pidx:
                raise IllegalAction(f"unit {uid} is not yours to move")
            if not u.ready:
                raise IllegalAction(f"unit {uid} is exhausted")
            if u.location == action.destination:
                raise IllegalAction("unit is already there")
            # Standard moves go base<->battlefield; battlefield-to-battlefield
            # requires Ganking (CR 144.4).
            if action.destination != BASE and u.location != BASE and not u.card.ganking:
                raise IllegalAction("battlefield-to-battlefield moves require Ganking")
            units.append(u)
        if not units:
            raise IllegalAction("no units to move")

        # Exhaust as the cost of the Standard Move (CR 144.2), then move.
        for u in units:
            origin = u.location
            u.ready = False
            u.location = action.destination
            # Back-Alley Bar: "When a unit moves from here, +1 might this turn."
            if origin != BASE:
                u.temp_might += s.battlefields[origin].card.move_from_buff

        if action.destination == BASE:
            return

        bf = s.battlefields[action.destination]
        defenders = s.units_at(1 - pidx, bf.index)
        if defenders:
            # Combat: mover is the Attacker (CR 464.2.c.1).
            self._resolve_combat(bf, attacker=pidx)
        elif bf.controller != pidx:
            # Non-combat showdown at an empty, un-owned battlefield: with no
            # reaction windows in v0, the mover establishes control at the
            # end of the showdown (CR 348.2.a).
            self._establish_control(pidx, bf)

    def _combat_might(self, u: UnitState, is_attacker: bool, allies: int) -> int:
        """A unit's effective might during combat, including keyword
        bonuses (Assault/Shield and the starter subset's named passives)."""
        m = u.card.might + u.temp_might
        if is_attacker:
            m += u.card.assault + u.temp_assault
        else:
            m += u.card.shield
        if u.card.might_bonus_if_runes8:
            if len(self.state.player_runes(u.controller)) >= 8:
                m += u.card.might_bonus_if_runes8
        if u.card.might_bonus_if_alone and allies == 1:
            m += u.card.might_bonus_if_alone
        return max(0, m)

    def _resolve_combat(self, bf: BattlefieldState, attacker: int) -> None:
        s = self.state
        defender = 1 - attacker
        atk_units = s.units_at(attacker, bf.index)
        def_units = s.units_at(defender, bf.index)

        atk_might = {
            u.uid: self._combat_might(u, True, len(atk_units)) for u in atk_units
        }
        def_might = {
            u.uid: self._combat_might(u, False, len(def_units)) for u in def_units
        }

        # Combat Damage Step (CR 465): sum might, assign (Tank first,
        # lethal-fill), then deal simultaneously.
        assignments: dict[int, int] = {}
        assignments.update(self._assign_damage(sum(atk_might.values()), def_units, def_might))
        assignments.update(self._assign_damage(sum(def_might.values()), atk_units, atk_might))
        for uid, dmg in assignments.items():
            s.units[uid].damage += dmg

        # Kill units with lethal damage (CR 142.4.b: non-zero >= might).
        for u in atk_units + def_units:
            eff = atk_might.get(u.uid, def_might.get(u.uid, 0))
            if u.damage > 0 and u.damage >= max(1, eff):
                self._kill_unit(u)

        # Combat Cleanup (CR 466.1): heal all units; recall attackers if
        # defenders are still present.
        for u in s.units.values():
            u.damage = 0

        atk_alive = s.units_at(attacker, bf.index)
        def_alive = s.units_at(defender, bf.index)
        if atk_alive and def_alive:
            for u in atk_alive:
                u.location = BASE  # recall, not a move (CR 454)
            # "No result" (CR 466.3.d); control unchanged.
        elif atk_alive:
            self._establish_control(attacker, bf)
        elif def_alive:
            # Conquer on defense is possible (CR 466.5.e).
            self._establish_control(defender, bf)
        else:
            bf.controller = None

    def _assign_damage(
        self, total: int, units: list[UnitState], eff_might: dict[int, int]
    ) -> dict[int, int]:
        """Assign combat damage per CR 465.2.c: Tanks first, lethal must be
        fully assigned before moving to another unit, no over-assignment.

        Within those constraints the assigner prefers killing the
        highest-might units (a fixed heuristic; making this an agent
        decision is future work)."""
        result: dict[int, int] = {}
        remaining = total
        tanks = [u for u in units if u.card.tank]
        others = [u for u in units if not u.card.tank]
        for group in (tanks, others):
            pending = sorted(group, key=lambda u: -eff_might[u.uid])
            while remaining > 0 and pending:
                killable = [
                    u
                    for u in pending
                    if max(1, eff_might[u.uid]) - u.damage <= remaining
                ]
                if killable:
                    u = killable[0]  # highest effective might that dies
                    need = max(1, eff_might[u.uid]) - u.damage
                    result[u.uid] = need
                    remaining -= need
                    pending.remove(u)
                else:
                    # Can't kill anything in this (mandatory) group: dump the
                    # remainder into one unit.
                    result[pending[0].uid] = remaining
                    remaining = 0
        return result

    def _kill_unit(self, u: UnitState) -> None:
        s = self.state
        if u.uid in s.units:
            del s.units[u.uid]
            s.players[u.controller].trash.append(u.card.id)

    # ==================================================================
    # Control and scoring (CR 188-196, 467-472)
    # ==================================================================

    def _establish_control(self, pidx: int, bf: BattlefieldState) -> None:
        if bf.controller == pidx:
            return
        bf.controller = pidx
        self._score(pidx, bf, conquer=True)

    def _score(self, pidx: int, bf: BattlefieldState, conquer: bool) -> None:
        """Score a battlefield via conquer or hold (CR 469-471)."""
        s = self.state
        if pidx in bf.scored_this_turn:
            return  # once per battlefield per player per turn (CR 470)
        bf.scored_this_turn.add(pidx)
        p = s.players[pidx]

        if conquer and p.points >= VICTORY_SCORE - 1:
            # Final point restriction (CR 471.1.b): a conquer at 7+ points
            # only scores if every battlefield has been scored this turn;
            # otherwise draw a card instead.
            if all(pidx in b.scored_this_turn for b in s.battlefields):
                self._gain_point(pidx)
            else:
                self._draw_cards(pidx, 1)
        else:
            self._gain_point(pidx)

        # Battlefield hold triggers (e.g. Grove of the God-Willow).
        if not conquer:
            for ins in bf.card.on_hold:
                self._execute_instruction(pidx, ins, targets=iter(()))

    def _gain_point(self, pidx: int) -> None:
        self.state.players[pidx].points += 1
        self._check_win()

    def _check_win(self) -> None:
        s = self.state
        if s.winner is not None:
            return
        for p in s.players:
            other = s.players[1 - p.index]
            if p.points >= VICTORY_SCORE and p.points > other.points:
                s.winner = p.index

    # ==================================================================
    # Draws and Burn Out (CR 413, 431)
    # ==================================================================

    def _draw_cards(self, pidx: int, count: int) -> None:
        s = self.state
        p = s.players[pidx]
        for _ in range(count):
            guard = 0
            while not p.main_deck and s.winner is None:
                self._burn_out(pidx)
                guard += 1
                if guard > 2 * VICTORY_SCORE:  # safety; cannot trigger normally
                    return
            if s.winner is not None:
                return
            p.hand.append(p.main_deck.pop())

    def _burn_out(self, pidx: int) -> None:
        """CR 431.2: recycle trash into the deck (randomized), then the
        opponent gains 1 point."""
        p = self.state.players[pidx]
        if p.trash:
            self.rng.shuffle(p.trash)
            p.main_deck = p.trash
            p.trash = []
        self._gain_point(1 - pidx)

    # ==================================================================
    # Cleanup (CR 318)
    # ==================================================================

    def _cleanup(self) -> None:
        s = self.state
        # Kill units with lethal damage (from spell damage outside combat).
        for u in list(s.units.values()):
            if u.damage > 0 and u.damage >= max(1, u.might):
                self._kill_unit(u)
        # Players lose control of battlefields they no longer occupy
        # (CR 323.6).
        for bf in s.battlefields:
            if bf.controller is not None and not s.units_at(bf.controller, bf.index):
                bf.controller = None
        self._check_win()
