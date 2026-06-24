# Orbit Wars — Approach Writeup

Here I describe the design of my Orbit Wars submission: how I encode the
game state, the model architecture that consumes it, the action space the policy
acts through, and how it is trained. I'm a biologist by training, self-taught in 
RL as a hobby. When I first looked at this game it seemed similar to the complex 
spatial interactions found in folded protein structures. Using AlphaFold's Evoformer
as my inspiration, I adapted its node/pair message-passing for this problem. Most of 
the architectural choices below fall out of this analogy.

## 1. Observation encoding

I encode the game as two things: **planets** (nodes) and **directed planet-pairs**
(edges). I deliberately do *not* model fleets as first-class objects. A fleet in
flight is, from a decision-making standpoint, just a future event that lands on
some planet at some time and changes its ownership and/or garrison. So instead of
giving the model a separate token per in-flight fleet and asking it to learn the
collision geometry, I **collapse every fleet into the planet it is going to
collide with** and expose the consequence directly.

### Planet (node) features

Each planet/comet body becomes one token. Comets share the planet encoder, so the
total count is bounded by `MAX_PLANETS` (40 planets + comet spawn slots). A planet
token carries:

- **Spatial state** — position and velocity, plus distance/relationship to the
  sun.
- **Current ownership and garrison** — who owns the planet now and how many ships are
  there.
- **Production** — Rate of ship production.
- **Projected timeline** — this is where the fleets go. For a horizon of 25
  future steps I project, per planet: the **incoming ships per player**, 
  the **projected garrison** over time, and the **projected owner** at each step. Every 
  fleet currently in flight has already been simulated forward and folded into the timeline 
  of its destination planet, so the token answers "what is going to happen to me, and
  when" without the model ever seeing a raw fleet.
- **Flags / despawn** — comet despawn timing and assorted status bits.

The forward projections are generated using a simulator that is consistent with the engine's 
ground truth.

### Edge (pair) features

For every directed ordered pair of planets `(source -> target)` I compute a small
set of **comparative, action-relevant** features that describe what would happen
if the source committed its garrison to the target ("full send"):

- **Time-to-arrival** at full send, bucketed (one-hot over ETA buckets), plus a
  **blocked** flag for when no legal trajectory exists.
- **Outcome of the full send** — most importantly a `can_takeover` flag: would
  sending everything from the source actually flip the target, given the target's
  defenders plus the production it accrues over the flight time?
- **Relational owner features** — source owner and target owner slots, which let
  the edge express same-owner vs. different-owner cheaply.

The key design decision is that the edge already encodes the *consequence* of a
move, not just the raw geometry. The model isn't asked to learn intercept math or
exact-takeover arithmetic; that's resolved deterministically in the observation
and in the action interpreter.

## 2. Model architecture — an Evoformer for planets

Planets are tokens; planet-pairs are an explicit **pair (edge) representation**.
This is the core of the borrowing from the Evoformer: rather than letting the
model infer relationships purely through dot-product attention, I keep a learned
edge state for every pair and let it bias and shape attention directly.

The backbone is a stack of node/pair blocks. Each block does, in order:

1. **Pair-biased planet attention**. Standard query-key attention over planet 
   tokens, but each attention logit is *added* to a per-head bias projected 
   from the edge state. So a planet attends to another both because their content 
   matches (QK) and because the edge between them says they're relevant.
2. **Pair update.** The edge state is updated from the tokens it connects via
   token-to-edge information flow — the planets write back into the relationships,
   keeping the raw edge features explicitly modeled. Similar to the Evoformer, the 
   raw pair encodings are injected into each layer.
3. **Pair feedforward transition.**

In addition to the planets there are a few special tokens: a **global token** carrying
dense scalars (turn number, ship totals, comet timer), a learned **register**
aggregator the bodies can route information through, and a **critic token** that
serves as the query for the value head's cross-attention readout. The observation
is Markovian — one forward pass per turn — because the observation already carries 
the projected future.

Default size (the released submission): token dim `= 144`, pair stream `= 160`,
5 layers, 4 heads.

## 3. Action space — edge-scored full sends

The action head is **edge-first**. For every directed edge `source -> target` it builds 
an edge context from the source token, the target token, and the pair state, and scores it.

Conceptually, **each source–target pair (and its edge) predicts a full-send
score**, and for each source planet I **softmax over its outgoing edges** to pick
what that planet does this turn. The source's *self*-edge is the **no-send**
option, so "hold and accumulate" competes directly against every outgoing fleet under 
one normalization.

The destination is chosen as a pointer over actual planet slots, and the firing angle and 
exact ship counts are resolved deterministically downstream by an action interpreter. 
The policy never has to emit raw angles or ship counts. It only has to answer the strategic 
question of *where to commit force*. Each source planet acts independently per turn, so a 
full turn is the collection of per-planet softmax choices across the board. This limits the 
number of actions per turn to the number of owned planets.

## 4. Training — Imitation followed by PPO with a joint action probability and GAE

Training was initiated by behavior cloning of downloaded replays. An interpreter is used
to convert the observed fleets into the desired action space. Alternatively, a heuristic 
*expert action* was used as the training target. The imitation learning phase was critical  
for 2 main reasons:

- **Experimentation.** A model that cannot do behavior cloning will likely be unable to 
  train via PPO. I experimented with many encodings and architectural choices before finding 
  a subset that could mimic expert play.
- **Representation Pre-Training.** Behavior cloning accelerates training of the network's 
  feature extraction capabilities. Attention maps can sometimes struggle with distribution shift — pre-training on a fixed distribution helps address that.

Training is then switched to **self-play PPO**. The agent plays Orbit Wars matches against a live
copy of itself, collects trajectories, and updates with clipped PPO.

- **Joint probability.** A turn is not one action — it's one decision per owned
  planet. I treat the turn's action as the **joint** over all source planets and
  sum their log-probabilities, so the PPO ratio and entropy are computed over the
  whole board's move, not over individual planets in isolation.
- **Advantages via GAE.** Returns and advantages use generalized advantage
  estimation. The value side is a **distributional** head (a two-hot / symlog bin
  support read out from the critic token).
- **Clipped PPO objective** with the usual value loss and entropy bonus on top of
  the joint policy.

## 5. What I tried and dropped — search (SBR)

I experimented with adding **search** on top of the policy — a sampled-best-response
style loop (SBR) using simulating learned rollouts. In principle this is the right thing 
to do for a game with a clean forward model like Orbit Wars.

In practice it was **too expensive for my rig**. Even a shallow search multiplies
every training step by the branching factor times the rollout depth times the
number of sampled responses, and each of those evaluations is a full
observation-assembly + network forward pass. The search variants spent almost all of their 
wall-clock budget inside the search expansion. I'd expect search to pay off with more compute.

## 6. Summary

This was an interesting, often frustrating, and very fun competition. My approach was
inspired by the Evoformer — a variant of the Transformer where edges are conditioned 
explicitly. I look forward to seeing what cool ideas other competitors used in their
solutions!
