# Engine design (v0)

This document records what the v0 engine implements, what it deliberately
simplifies, and the plan for closing the gaps. Rule numbers (CR xxx) refer to
`docs/rules-reference.md`, a copy of the Riftbound Core Rules reference.

## Architecture

```
src/riftbound/
  engine/
    enums.py      Domains, card types, phases, 1v1 constants
    cards.py      CardDef / BattlefieldDef + the data-driven effect vocabulary
    state.py      Mutable state: units, runes, gear, battlefields, players
    actions.py    Agent-facing actions (play / move / end turn)
    game.py       Turn loop, payment, combat, scoring, legal_actions
  cards/          Real-card subset + prebuilt decks
  agents/         Random and greedy baselines
  sim.py, cli.py  Match runner
scripts/fetch_cards.py   Card DB download (RiftScribe community API)
data/cards.json          All 1,180 cards with rules text
```

Design principles:

- **Cards are data, not code.** Effects are `Instruction` lists drawn from a
  small vocabulary (damage/kill/buff/draw/channel/...). Adding a card is one
  registry entry; the engine and the (future) action encoder don't change.
- **Agents see a flat decision loop.** All decisions happen in the turn
  player's Main Phase via `legal_actions()` / `step()`. Start/end-of-turn
  bookkeeping is automatic. This is the shape RL and search code want.
- **Auto-resolved micro-decisions.** Rune payment and combat damage
  assignment follow fixed, documented policies (below) so the action space
  stays small in v0.

## What is faithfully implemented

- Turn structure (CR 315–317): awaken, hold scoring in the Beginning Phase,
  channel 2 (second player channels 3 on their first turn, CR 485.7),
  draw 1 with Burn Out (CR 431: recycle trash, opponent gains a point),
  main phase, ending phase (heal all units, expire "this turn" effects).
- Rune economy (CR 160–166, 430): exhaust a rune for 1 Energy, recycle a
  rune (to the bottom of the rune deck, CR 416) for 1 Power of its domain;
  rune pools clear at Main Phase start and turn end; resource gear (Seal
  cycle) exhausts for Power.
- Standard Move (CR 144): exhaust as cost, group moves to one destination,
  base↔battlefield only, Ganking units may move battlefield→battlefield.
- Showdowns and combat (CR 341, 459–466): moving onto an enemy-held or
  neutral battlefield contests it; combat sums Might, assigns damage with
  the real constraints (Tank first, lethal must be fully assigned before
  spilling, no over-assignment, CR 465.2.c), deals simultaneously, heals in
  the combat cleanup, recalls attackers if defenders survive, and lets the
  surviving side establish control — including conquer-on-defense.
- Scoring (CR 467–472): conquer and hold, once per battlefield per player
  per turn, and the final-point restriction (a conquer at 7+ points scores
  only if every battlefield was scored this turn; otherwise draw 1).
  Control is lost when a battlefield is left unoccupied (CR 323.6).
- Combat keywords: Tank, Assault N, Shield N, "enters ready", Ganking, and
  the starter subset's named passives (Annie's bonus damage; Yi's 8+-runes
  bonus; Wielder of Water's alone bonus).

## Known simplifications (v0)

Each of these is a conscious scope cut, roughly ordered by how much they
matter:

1. **No chain / reactions / showdown windows.** Spells and abilities resolve
   immediately; there are no Action/Reaction timing windows during combats or
   showdowns (CR 325–348). Consequence: combat is deterministic once the
   move is made, and "no result" combats (both sides survive) can't occur.
   This is the big one to build in v1 (it adds interleaved decision points
   for the non-turn player).
2. **Auto-payment.** Which runes to exhaust/recycle is a fixed policy
   (power: gear, then exhausted runes, then ready runes; energy: ready
   runes), not an agent choice.
3. **Auto damage assignment.** The assigner kills the highest-effective-might
   legal unit first. Making this an agent decision node is future work.
4. **No mulligans** (CR 117).
5. **Legend abilities are ignored** (legends define domain identity only).
6. **Battlefield abilities**: only the two shapes used by the starter decks
   (hold trigger, move-from buff); other battlefields are inert names.
7. **Unit/gear "when played" triggers must be non-targeted** — keeps the
   action space to (card, destination) for permanents.
8. **No gear beyond resource gear**, no Equipment/attachments, no activated
   abilities, no hidden/facedown cards, no stun, no tokens, no XP/levels.
9. **Per-card simplifications** are marked with `# SIMPLIFIED:` comments in
   `src/riftbound/cards/ogn_subset.py` (e.g. Accelerate ignored, Deflect
   ignored, Tibbers' pips paid as Fury).
10. **Greedy agent peeks**: its one-ply simulation copies the RNG, so draw
    outcomes are foreseen. Fine for a baseline; fix with determinization.

## Hidden information

Hands, deck order, and rune-deck order are private. `GameState` currently
exposes everything to whoever holds the object; the (future) observation
encoder is responsible for masking to a player's view. Search agents must
use determinization or IS-MCTS rather than raw tree search.

## Encodings (implemented)

`src/riftbound/encoding.py`:

- `ObservationEncoder.encode(game, pidx)`: fixed-size float32 vector — own
  hand as card counts; both trashes and champion zones (public); 12 unit
  slots per side (card one-hot + might/damage/status/location features);
  runes per domain (ready/exhausted) and rune-deck sizes; resource gear;
  battlefield controller/scored flags; score/hand/deck/turn scalars. The
  opponent's hand appears only as a size and decks only as sizes (hidden
  information is never encoded).
- `ActionCodec`: fixed table over the card vocabulary — end turn; play unit
  (card x destination); play champion (destination); play gear; play spell
  (card x target-ref combos, where refs are unit slots or battlefields);
  move unit (slot x destination); all-ready-base-units attack per
  battlefield. `legal_mask(game)` marks currently-legal indices (guaranteed
  step-able); `decode(idx, game)` returns the engine action.
- Codec limitation: arbitrary move subsets aren't addressable (single-unit
  moves and all-in attacks only). Fix alongside the v1 engine work.

## Determinized MCTS (implemented)

`src/riftbound/agents/mcts_agent.py`:

- `determinize(game, perspective, rng)`: clones the game and resamples what
  the player can't see — the opponent's hand is pooled with their deck and
  redealt, both main-deck and rune-deck orders are shuffled, and the clone
  gets an independent RNG. Public zones are untouched. (Assumes the
  opponent's decklist is known — true in self-play.)
- `MCTSAgent(iterations, determinizations)`: standard UCT per determinized
  world (after determinization the game is deterministic), leaf evaluation
  via the greedy agent's static score squashed to [0, 1] (no rollouts),
  opponent nodes minimize, and root visit counts are aggregated across
  worlds. Group moves are pruned to singletons + all-in (as in the codec)
  to control branching.

## Next steps

1. **Self-play PPO** against the random/greedy/MCTS ladder, league-style.
2. **v1 engine**: reaction windows (start with Action/Reaction spells in
   showdowns), agent-controlled damage assignment, mulligans.
