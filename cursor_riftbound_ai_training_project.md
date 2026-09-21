# Riftbound AI training project
_Exported on 9/21/2026 at 01:16:22 CDT from Cursor (3.15.6)_

---

**User**

Just asked claude how people train AI models on games; specifically because I want to make a project where an AI model is trained on riftbound. Here's the output. Can you beign on this project?

There are two very different families of approaches, and which one fits depends a lot on what the game gives you access to.

**How the Clash Royale bots work (pixels-in, taps-out).** Those projects treat the game as a black box: screen-capture the phone or emulator, run computer vision (object detection for units, OCR for elixir/timer) to reconstruct a game state, then feed that state to a policy — sometimes hand-written heuristics, sometimes a reinforcement-learning agent — and send actions back as taps via ADB. They do this because Clash Royale has no API and no simulator, so the real client *is* the environment. It's painful: CV is brittle, you can only train as fast as the game plays in real time, and you're at the mercy of the terms of service.

**The better path for a card game: build the simulator yourself.** Riftbound is turn-based and fully discrete, which means you don't need pixels at all — you need a rules engine. This is how essentially every serious card-game AI has been built (Forge for Magic, Fireplace/Sabberstone for Hearthstone, Pokémon Showdown bots). The rough pipeline:

1. **Card database.** Encode every card as data: cost, type, stats, and its effect text. Community sites already maintain card data, so you likely won't be transcribing from scratch. Start with a subset — two starter decks is plenty.
2. **Rules engine.** Implement the turn structure, rune-deck resource system, battlefields, scoring, and a `legal_actions(state)` function. This is the single biggest chunk of work; each card's effect text becomes code, and card interactions are where bugs hide. Cloning a couple of decks first keeps this tractable.
3. **State and action encoding.** Represent the state as a fixed-size tensor (your hand, board, battlefields, runes, opponent's visible info, score) and actions as a discrete space with a legality mask so the agent never picks illegal moves. Card identity can start as a plain ID embedding — the network learns synergies through play.
4. **Agents, in escalating order.** A random agent (tests the engine), a greedy scripted agent (your baseline to beat), then search: Riftbound has hidden information (opponent's hand, deck order), so plain MCTS doesn't apply directly — Information-Set MCTS or determinization (sample plausible opponent hands, search each) is the standard fix. Then self-play RL, typically PPO or an AlphaZero-style policy+value net trained against past versions of itself. Turn-based card games are cheap to simulate, so millions of games on a single workstation is realistic.

**The LLM route.** If what you mostly want is an agent that "understands its deck and the objective" in an explainable way, you can serialize the game state to text and prompt an LLM with the rules and the legal action list. It prototypes in a weekend and reasons about strategy in words, but it's expensive per move and worse than search at tactics. A common hybrid is LLM for high-level planning or as a heuristic evaluator inside MCTS.

**Where Riftbound specifically stands.** There's no official way to play Riftbound online, as Riot hasn't announced a digital client, so unlike Clash Royale there's no real client to bot against — which actually pushes you toward the simulator approach anyway. Community platforms exist: TCG Arena is the most active unofficial platform, a Tabletop Simulator-style mod that loads card images and tracks zones with only partial rules enforcement, Pixelborn is a fan-made client, and Rift Atlas lets you play in the browser, including a solo mode where you control both sides. Rift Atlas being browser-based means that, once your agent is good in your own engine, hooking it to real opponents via a DOM scraper is a plausible later step — but I'd treat that as phase three.

If you want, I can sketch the state/action encoding for Riftbound concretely, or outline the rules-engine architecture so cards stay pluggable.

---

**Cursor**

I'll get started on this. First let me look at what's already in the workspace, and in parallel research Riftbound's actual rules and available card data so the engine is built on accurate mechanics.

The riftwatcher page gave me what looks like the full core rules reference (325 KB). Let me extract the key sections I need for the engine: game setup, turn structure, showdowns/combat, and scoring.

Setup and win conditions are clear. Now the turn structure, showdowns, and playing-card process.

Now the play process, movement, combat, and scoring sections.

Combat and scoring are fully specified. Two more reads: the units/runes sections (Standard Move, rune pool mechanics) and the 1v1 mode setup.

The rules are fully mapped (including Burn Out: recycle trash into deck, opponent gains a point, then draw). Now I'll scaffold the project and fetch the card database while I write the engine.

The Riftcodex endpoint 404s. Let me probe both APIs to find the right URL shape.

RiftScribe works (1,180 cards, no auth). Let me update the fetch script for its response shape and check whether it includes rules text.

The detail endpoint has full rules text (`description`), keywords, and tags. Let me check whether the list endpoint includes those too, then fetch everything.

The list endpoint omits rules text, so the script needs to fetch each card's detail (about 1,180 requests). I'll rewrite it for RiftScribe and run it in the background while I build the engine.

While the card download runs, let me confirm one mechanic — where recycled runes go (rule 416) — then write the engine.

Confirmed: recycled runes go to the bottom of the rune deck. Now I'll write the engine. Marking the scaffold done and engine in progress.

Now the core of the project — `game.py` with the full turn loop, combat, and scoring logic.

The engine core is written. Let me check on the card download, then write the agents.

The download is at 750 of 1,180 cards and progressing. Meanwhile, the agents and simulation CLI.

All 1,180 cards are downloaded with full rules text. Now let me mine the data for simple cards to build two real starter decks — vanilla units and simple spells in Fury and Calm.

Almost nothing is vanilla, so let me look at the actual card texts to pick simple, implementable ones.

PowerShell quoting issue — I'll use a script file instead.

The `OGS` set (24 cards) looks like the Proving Grounds starter product — exactly the "two starter decks" the plan calls for. Let me see all of it.

Now the real card subset — two 40-card decks transcribed from the fetched data — plus the engine updates to support them. First the card set:

Now the engine updates: `temp_assault` on units, battlefield defs in state, and the game logic for keywords, new ops, ganking, and battlefield triggers.

Now the CLI, tests, and documentation:

Now the test suite covering setup, turn flow, payment, spells, combat, scoring, and burnout:

One failure, and it's the test's arithmetic, not the engine: the Seal itself costs 1 power (recycling a rune to play it), leaving too few runes for the follow-up unit. Fixing the test:

Greedy beats random 20-0 with either deck — the agent ladder works. Now the documentation: README and a design doc recording what's implemented, every simplification, and the roadmap.

The project is up and running — phase 1 of the simulator-first plan (rules engine, card data, starter decks, baseline agents) is built, tested, and committed as the initial git commit in `riftbound-ai`.

## What I built

**Research first.** I found the complete official Core Rules mirrored as a numbered reference (vendored into `docs/rules-reference.md`, ~4,700 lines) and a free community card API (RiftScribe). Every engine mechanic cites the rule it implements, and `data/cards.json` now holds all 1,180 cards with rules text so no card ever needs manual transcription.

**Rules engine** (`src/riftbound/engine/`) for 1v1 Duel: the real turn structure (awaken → hold scoring → channel → draw → main → ending, with the second player's extra rune), the rune economy (exhaust for Energy, recycle to deck-bottom for Power, Seal-cycle gear), standard moves, and combat with the actual damage-assignment rules — Tank priority, lethal-must-be-filled-before-spilling, simultaneous dealing, attacker recall, conquer-on-defense. Scoring includes the tricky final-point restriction (a conquer at 7 points only wins if you scored every battlefield that turn — otherwise you draw a card) and Burn Out. Agents interact through a clean `legal_actions()` / `step()` loop, which is exactly the shape MCTS and RL code want later.

**Real cards, not toys** (`src/riftbound/cards/`): two 40-card decks of actual cards — an Annie/Fury burn deck (Falling Star, Void Seeker, Tibbers, Annie's bonus-damage passive) and a Master Yi Calm/Body defensive deck (Shield/Tank walls, Ganking Yi). Effects are data-driven instructions, so new cards are registry entries, not engine changes. Every simplified card is marked with a `# SIMPLIFIED:` comment.

**Agents and results:** random agent (engine fuzzing) and a one-ply greedy baseline. All 19 tests pass, random-vs-random runs at ~500 games/second, and greedy beats random **20–0 with either deck** — the agent ladder behaves as it should. Fun artifact: random Annie loses badly because it nukes its own units with burn spells.

## The one big deliberate cut

There is **no reaction chain yet** — spells resolve instantly, so combat has no interruption windows. This is the standard v0 cut for card-game engines and it's the top item in `docs/engine-design.md`, which lists all ten simplifications honestly (auto rune payment, auto damage assignment, no mulligans, inert legends, greedy's RNG peeking, etc.).

## Suggested next step

Per the roadmap: state/action tensor encodings with legality masks, then determinized MCTS for the hidden information, then self-play PPO against the greedy/random ladder. Say the word and I'll start on the encodings.

---

**User**

<timestamp>Monday, Sep 21, 2026, 12:31 AM (UTC-5)</timestamp>
<system_notification>
The following task has finished. If you were already aware, ignore this notification and do not restate prior responses.

<task>
kind: shell
status: success
task_id: 623612
title: Download full card database in background
tool_call_id: toolu_012hvYNp631MZPHJtDMZLTmK
output_path: C:\Users\gabe\.cursor\projects\c-Users-gabe-Documents-repos-riftbound-ai\terminals\623612.txt
</task>
</system_notification>
<user_query>Briefly inform the user about the task result and perform any follow-up actions (if needed).</user_query>

---

**Cursor**

That was the card database download finishing — I already picked up its result mid-task: all 1,180 cards with rules text were fetched from the RiftScribe API into `data/cards.json`, used to build the two starter decks, and committed with the rest of the project. Nothing further to do.

---

**User**

go ahead and publish this to my github, commit and push the changes, then start on the next step.

---

**Cursor**

I'll check for uncommitted changes and whether the GitHub CLI is authenticated, then create the repo and push.

Git is clean with one commit and `gh` is logged in as `gabezmcmillan`. Creating the GitHub repo and pushing:

Now the encoding tests: mask validity (every legal index must step cleanly), full playthroughs driven through the codec, and a hidden-information leak check.

All 27 tests pass. Updating the docs to reflect the new layer, then committing and pushing.

Done on both fronts.

**Published:** the project is live at [github.com/gabezmcmillan/riftbound-ai](https://github.com/gabezmcmillan/riftbound-ai) (public, created with your existing `gh` login). Two commits are pushed: the v0 engine and the new encoding layer.

**Next step built — roadmap item 2, the encoding layer** (`src/riftbound/encoding.py`), which is what search and RL agents will consume:

- **`ObservationEncoder`** produces a fixed-size float32 vector from one player's perspective. Public information is fully encoded (board unit slots with card one-hots and status features, runes per domain, gear, battlefield control and scored flags, trashes, champion zones, scores). Hidden information stays hidden: the opponent's hand is encoded only as a count and both decks only as sizes — there's a test that swaps the opponent's hand contents and asserts the observation doesn't change.
- **`ActionCodec`** defines a fixed table of ~1,000 discrete actions (play card × destination, spell × target-slot combinations, single-unit moves, an all-in attack per battlefield, end turn). `legal_mask(game)` returns a boolean mask where every marked index is guaranteed to execute without error, and `decode(idx, game)` maps back to a real engine action.

The suite is now 27 tests, all passing, including full games played end-to-end purely through masked action indices with both the synthetic test set and the real Annie/Yi decks. One fix along the way: the test fixture's registry snapshotting was silently rolling back the real card set after its first use, which the conftest now imports up front.

One documented gap to know about: the flat action table can't express *arbitrary* multi-unit attack groups — only single-unit moves and "send everything." That's noted in `docs/engine-design.md` and slated for the v1 engine work.

Next on the roadmap is the determinized MCTS baseline (sampling plausible opponent hands to handle hidden information), followed by self-play PPO. Want me to keep going with the MCTS agent?

---

**User**

keep going

---

**Cursor**

Continuing with roadmap item 3: a **determinized MCTS agent** — the first agent that actually searches. The approach: since opponent hand and deck order are hidden, sample several "determinizations" (shuffle the opponent's hand back into their deck and redeal, shuffle unknown deck orders), run UCT search on each fully-determined game, and aggregate root visit counts across samples to pick the move.

All 33 tests pass. Now the benchmark — MCTS versus greedy and versus random (this takes a few minutes, so it runs in the background):

While that runs, updating the docs:

Roadmap item 3 is done and pushed — the project now has its first search-based agent, and it demonstrably gets stronger with more compute.

**What was built** (`src/riftbound/agents/mcts_agent.py`, commits `0e9eb64` and `f526881` on [main](https://github.com/gabezmcmillan/riftbound-ai)):

- **Determinization** handles the hidden information: before searching, the agent clones the game and resamples everything it can't legally see — the opponent's hand gets pooled with their deck and redealt, and all unknown deck orders are shuffled. There's a test asserting public zones and its own hand are untouched.
- **UCT search** runs inside each sampled world (which is fully deterministic once shuffled), using the greedy agent's evaluation squashed to [0,1] as the leaf value instead of slow random rollouts. Opponent nodes minimize. Root visit counts are aggregated across worlds and the most-visited action wins.
- Wired into the CLI (`--p0 mcts --mcts-iterations N --mcts-worlds K`), with 6 new tests (33 passing) — including one where MCTS must find a forced two-conquer win at 7 points, which it does.

**Benchmark results**, the interesting part:

| Matchup | Result |
|---|---|
| MCTS vs random | 10–0 |
| MCTS (Yi deck) vs greedy, default budget | 7–3 |
| MCTS (Annie deck) vs greedy, default budget | 1–9 |
| MCTS (Annie deck) vs greedy, 3× budget | **5–1** |

The default budget (240 simulations/move) roughly ties greedy, but tripling it flips the losing matchup decisively — search strength scales with compute, which is the green light for the AlphaZero-style direction. Side finding: the decks aren't balanced (the Yi defensive deck wins most pairings regardless of agent), so all future evaluations should play both seats, which the benchmark docs now note.

Next on the roadmap is **self-play RL** — a policy/value network over the observation encoder, trained with PPO against the random→greedy→MCTS ladder. That's a bigger chunk (torch dependency, training loop, checkpointing, eval harness); say "keep going" and I'll start it.

---

**User**

<timestamp>Monday, Sep 21, 2026, 1:15 AM (UTC-5)</timestamp>
<system_notification>
The following task has finished. If you were already aware, ignore this notification and do not restate prior responses.

<task>
kind: shell
status: success
task_id: 743478
title: Higher-budget MCTS on the losing matchup
tool_call_id: toolu_013x7KLbbtbPP3QYcnUwAzZU
output_path: C:\Users\gabe\.cursor\projects\c-Users-gabe-Documents-repos-riftbound-ai\terminals\743478.txt
</task>
</system_notification>
<user_query>Briefly inform the user about the task result and perform any follow-up actions (if needed).</user_query>

---

**Cursor**

That's the high-budget MCTS benchmark finishing — already handled: it showed MCTS winning 5–1 with the Annie deck against greedy at 200 iterations × 6 worlds (up from 1–9 at the default budget), and those numbers are documented and pushed. No follow-up needed.

