# Orbit Wars — Approach Writeup

This document describes the design of my Orbit Wars submission: how I encode the
game state, the model architecture that consumes it, the action space the policy
acts through, and how the whole thing is trained. The short version is that I
treat the board as a graph of planets and the moves between them as edges, and I
borrow the relational machinery of the Evoformer to reason over both jointly.

A note on where the ideas come from: I'm a biologist by training, not an RL
researcher. When I first looked at this game it didn't read to me like a control
problem — it read like a structured-prediction problem over a small set of
interacting objects, which is exactly the shape of a protein. So I reached for
the tool I already trusted for that shape, AlphaFold's Evoformer, and adapted its
node/pair message-passing to a strategy game. Most of the architectural choices
below fall out of that analogy.

## 1. Observation encoding

I encode the game as two things: **planets** (nodes) and **directed planet-pairs**
(edges). I deliberately do *not* model fleets as first-class objects. A fleet in
flight is, from a decision-making standpoint, just a future event that lands on
some planet at some time and changes its ownership or garrison. So instead of
giving the model a separate token per in-flight fleet and asking it to learn the
collision geometry, I **collapse every fleet into the planet it is going to
collide with** and expose the consequence directly.

### Planet (node) features

Each planet/comet body becomes one token. Comets share the planet encoder, so the
body count is bounded by `MAX_PLANETS` (32 planets + comet spawn slots). A planet
token carries:

- **Spatial state** — position and velocity, plus distance/relationship to the
  sun (which constrains legal trajectories, since shots through the sun are
  blocked).
- **Current ownership and garrison** — who holds it now and how many ships are
  parked there (encoded as `ln(ships)` and later expanded to a two-hot bin
  representation on the GPU to keep the replay buffer small).
- **Production** — how fast the planet mints new ships.
- **Projected timeline** — this is where the fleets go. For a horizon of 25
  future steps I project, per planet: the **incoming ships per player**
  (a 4-player × 25-step matrix), the **projected garrison** over time, and the
  **projected owner** at each step. In other words, every fleet currently in
  flight has already been simulated forward and folded into the timeline of its
  destination planet, so the token answers "what is going to happen to me, and
  when" without the model ever seeing a raw fleet.
- **Flags / despawn** — comet despawn timing and assorted status bits.

The forward projection that produces these timelines is not a heuristic — it's
driven by the same vectorized fleet-collision primitives the replay-validated
simulator uses (`trace_fleets_first_hit_np` and friends in `orbit_wars_numpy.py`),
so the model's view of the future is consistent with the engine's ground truth.

### Edge (pair) features

For every directed ordered pair of planets `(source -> target)` I compute a small
set of **comparative, action-relevant** features that describe what would happen
if the source committed its garrison to the target ("full send"):

- **Time-to-arrival** at full send, bucketed (one-hot over ETA buckets), plus a
  **blocked** flag for when no legal trajectory exists (e.g. the sun is in the
  way or the target is unreachable in the horizon).
- **Outcome of the full send** — most importantly a `can_takeover` flag: would
  sending everything from the source actually flip the target, given the target's
  defenders plus the production it accrues over the flight time? This is the
  difference between a take-over, a reinforce (same owner), and a wasted poke.
- **Relational owner features** — source owner and target owner slots, which let
  the edge express same-owner vs. different-owner cheaply.

The key design decision is that the edge already encodes the *consequence* of a
move, not just the raw geometry. The model isn't asked to learn intercept math or
exact-takeover arithmetic; that's resolved deterministically in the observation
and in the action interpreter. The network's job is the strategic one — given
that I *can* take this planet in N turns, *should* I.

## 2. Model architecture — an Evoformer for planets

Planets are tokens; planet-pairs are an explicit **pair (edge) representation**.
This is the core of the borrowing from the Evoformer: rather than letting the
model infer relationships purely through dot-product attention, I keep a learned
edge state `z_ij` for every pair and let it bias and shape attention directly.

The backbone is a stack of node/pair blocks (`NodePairTriangleBlock`). Each block
does, in order:

1. **Pair-biased planet attention** (`PairBiasedPlanetAttention`). Standard
   query-key attention over planet tokens, but each attention logit is *added* a
   per-head bias projected from the edge state `z_ij` — `bias_ij = Linear(LayerNorm(z_ij))`.
   So a planet attends to another both because their content matches (QK) and
   because the edge between them says they're relevant (e.g. "you can take me in
   3 turns"). The block also uses an Evoformer-style **output gate**: a
   per-position, per-channel sigmoid that lets a planet dampen attention
   contributions dominated by the pair bias. This is the standard QK + edge-bias
   mix, with information flowing **from the tokens into the attention over
   edges**.
2. **Pair update.** The edge state is refreshed from the tokens it connects:
   `z_ij <- z_ij + MLP([z_ij, h_i, h_j, raw_edge_ij])`. This is the
   token-to-edge information flow — the planets write back into the relationships,
   keeping the raw edge features in the loop so the engine-derived signal is never
   washed out.
3. **(Optional) triangle multiplicative updates.** I implemented the Evoformer's
   triangle updates (outgoing/incoming) over the directed pair states, initialized
   as identity so they start as a no-op. These let edges talk to other edges
   through a shared third node — the protein-folding intuition that a constraint
   between i–j and j–k implies something about i–k. In the released configuration
   triangles are **off**: on my hardware the `O(P^2)` pair-bias path already
   captured most of the relational signal, and the `O(P^3)` triangle path wasn't
   worth its cost (more on the compute budget below).
4. **Pair feedforward transition.**

On top of the bodies there are a few special tokens: a **global token** carrying
dense scalars (turn number, ship totals, comet timer), a learned **register**
aggregator the bodies can route information through, and a **critic token** that
serves as the query for the value head's cross-attention readout. The whole thing
is Markovian — one forward pass per turn, no attention across turns — because the
observation already carries the projected future.

Default size (the released submission): `d_model = 144`, pair stream `= 160`,
5 layers, 4 heads, type embeddings on, triangles off.

## 3. Action space — edge-scored full sends

The action head is **edge-first** (`EdgeActionHead`). For every directed edge
`source -> target` it builds an edge context from the source token, the target
token, and the pair state, and scores it:

```
edge_ctx_ij = MLP([src_proj(h_i), tgt_proj(h_j), z_ij])
score_ij    = head(edge_ctx_ij)
```

Conceptually, **each source–target pair (and its edge) predicts a full-send
score**, and for each source planet I **softmax over its outgoing edges** to pick
what that planet does this turn. The source's *self*-edge is the **no-send**
option, so "hold and accumulate" competes directly against every outgoing attack
or reinforcement under one normalization. A planet that has nowhere worth sending
to naturally keeps its garrison, because the self score wins the softmax.

Because the destination is chosen as a pointer over actual planet slots, and the
firing angle and exact ship counts are resolved deterministically downstream by
the action interpreter (lead-targeting for the angle; defenders + production for
exact-takeover sizing), the policy never has to emit raw angles or ship counts. It
only has to answer the strategic question of *where to commit force*. Each source
planet acts independently per turn, so a full turn is the collection of per-planet
softmax choices across the board.

## 4. Training — PPO with a joint action probability and GAE

Training is **self-play PPO**. The agent plays Orbit Wars matches against a live
copy of itself, collects trajectories, and updates with clipped PPO.

- **Joint probability.** A turn is not one action — it's one decision per owned
  planet. I treat the turn's action as the **joint** over all source planets and
  sum their log-probabilities, so the PPO ratio and entropy are computed over the
  whole board's move, not over individual planets in isolation. This keeps the
  importance-sampling correction honest when several planets act at once.
- **Advantages via GAE.** Returns and advantages use generalized advantage
  estimation. The value side is a **distributional** head (a two-hot / symlog bin
  support read out from the critic token), which handles the heavy-tailed,
  swingy nature of territory-control returns better than a single scalar regressor.
- **Clipped PPO objective** with the usual value loss and entropy bonus on top of
  the joint policy.

There's also an **imitation / behavioral-cloning** pathway that runs through the
same learner: human and heuristic-expert replays are reconstructed from each
player's perspective with the same assembler, the logged orders are reverse-mapped
into the policy's action labels, and the policy is warm-started by cloning while
the value head learns GAE returns. PPO then takes over from that initialization.

## 5. What I tried and dropped — search (SBR)

I experimented with adding **search** on top of the policy — a sampled-best-response
style loop (SBR), where you roll the learned policy forward through the simulator,
evaluate the resulting positions, and back up an improved action target for the
policy to chase. In principle this is the right thing to do for a game with a clean
forward model like Orbit Wars: I already have an engine-faithful simulator, so
look-ahead is "free" in the sense that it's well-defined.

In practice it was **too expensive for my rig**. Even a shallow search multiplies
every training step by the branching factor times the rollout depth times the
number of sampled responses, and each of those evaluations is a full
observation-assembly + network forward pass. On my single-GPU setup the search
variants spent almost all of their wall-clock budget inside the search expansion
and starved the learner of gradient steps, so the model that *trained more* (plain
PPO) beat the model that *searched more* within any fixed amount of compute I could
afford. The search subtree is therefore disabled in the released configuration
(`SearchConfig.enabled = False`, a hard no-op); both the PPO and imitation paths
run without it. I'd expect search to pay off with more hardware, but the model
here is search-free by deliberate budget choice, not because the idea was wrong.

## 6. Summary

The submission reframes a real-time strategy game as a relational reasoning problem
over a graph — planets as nodes, moves as edges — and solves it with an
Evoformer-style node/pair network borrowed from protein structure prediction.
Fleets never appear as objects; their effects are pre-simulated into the planets
they hit and into the edges that describe each possible attack. The policy makes
one edge-scored "where to send" decision per planet via a softmax that includes
"don't send," and the whole thing is trained with self-play PPO over the joint
board action with GAE advantages and a distributional value head, warm-started by
imitation. Search would have been the natural next layer, but it was priced out by
the hardware available.
