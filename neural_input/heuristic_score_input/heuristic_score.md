# Heuristic Score Vector Definition

## Overview

This module defines the **heuristic score vector** used as the third input branch in Stage 1.

Unlike `x_local` and `x_context`, this branch does not describe point-wise geometry.  
It provides a **track-level prior** based on simple expert-style evaluation of the circuit.

For each track, the heuristic vector is:

\[
x_{\text{heuristic}} =
\left[
S_{\text{layout}},
S_{\text{overtake}},
S_{\text{challenge}}
\right]
\]

These three scores are constant for the whole track.

---

## Data Source

The scores are assigned using two fixed sources:

1. **Official Formula 1 website**  
   Used as the primary source for circuit description, layout style, and driving characteristics.

2. **RaceFans**  
   Used as the secondary source for racing quality, overtaking tendency, and layout challenge.

---

## Score Definition

Each score is assigned on a 5-point scale:

- 1 = very low
- 2 = low
- 3 = neutral
- 4 = high
- 5 = very high

A score of **3** means there is **no strong bias** in that dimension.

### 1. Layout Style Score

\[
S_{\text{layout}} \in \{1,2,3,4,5\}
\]

Describes whether the track is more:

- technical / corner-driven
- or straight-line / power-oriented

### 2. Overtaking Friendliness Score

\[
S_{\text{overtake}} \in \{1,2,3,4,5\}
\]

Describes whether the circuit naturally supports:

- overtaking
- attack-defense interaction
- position exchange

### 3. Driver Challenge Score

\[
S_{\text{challenge}} \in \{1,2,3,4,5\}
\]

Describes how demanding the circuit is in terms of:

- precision
- rhythm
- commitment
- technical difficulty

---

## Assignment Rule

For one track, assign one score triplet:

\[
\left[
S_{\text{layout}},
S_{\text{overtake}},
S_{\text{challenge}}
\right]
\]

This same vector is copied to all sampled points on that track:

\[
x_{\text{heuristic}}(i)
=
\left[
S_{\text{layout}},
S_{\text{overtake}},
S_{\text{challenge}}
\right]
\]

So this branch is a **global prior input**, not a segment-varying feature.

---

## Default Rule for Unfamiliar Tracks

If a track is unfamiliar or cannot be confidently judged from the two sources above, use the default vector:

\[
x_{\text{heuristic}} = [3,3,3]
\]

This means:

- no strong layout preference
- no strong overtaking preference
- no strong challenge preference

In other words, the heuristic branch provides a **neutral prior**.

---

## Output Format

For each track:

\[
x_{\text{heuristic}} \in \mathbb{R}^3
\]

with entries:

1. layout style score  
2. overtaking friendliness score  
3. driver challenge score
