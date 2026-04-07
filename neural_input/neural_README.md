# Track-Context Defense Preference Network

## 1. Overview

This module implements **Stage 1: Offline Strategy Recognition** for a competitive autonomous racing pipeline.

The goal is to build a **track-dependent defense preference map** along the global racing line.

For each sampled track position `s_i`, the network outputs a continuous scalar score:

- `D(s_i) > 0`  -> apex defense is preferred
- `D(s_i) < 0`  -> staying close to the best line is preferred
- `|D(s_i)|`    -> strength of the preference

This score is later used by **Stage 2** to generate an online defensive trajectory.

---

## 2. Pipeline

SLAM map  
-> extract track boundaries  
-> generate global racing line  
-> sample by arc length  
-> construct feature vector at each sampled point  
-> normalize and merge features  
-> shared 1D CNN encoder  
-> MLP  
-> prediction head  
-> defense-preference score `D(s_i)`  
-> smooth track-wise score map

---

## 3. Core Function

The defense preference is modeled as:

D(s) = f( G_cur(s), G_future(s:s+H) )

where:

- `G_cur(s)`    = current local geometric features
- `G_future(...)` = future/contextual geometric features in a look-ahead window
- `f(.)`        = neural network approximation
- `D(s)`        = scalar defense-preference score

Interpretation:

- positive score  -> more suitable to defend the apex
- negative score  -> more suitable to keep the optimal racing line

---

## 4. Inputs

For each sampled point `s_i`, construct three feature groups.

### 4.1 Local Geometry Vector

Name:
`x_local[i]`

Purpose:
Describes the current geometric condition at track point `s_i`.

Recommended fields:
- `curvature_i`
- `curvature_diff_i`
- `track_width_i`
- `left_margin_i`
- `right_margin_i`
- `heading_change_i`

Example:
x_local[i] = [
    curvature_i,
    curvature_diff_i,
    track_width_i,
    left_margin_i,
    right_margin_i,
    heading_change_i
]

---

### 4.2 Context Geometry Vector

Name:
`x_context[i]`

Purpose:
Describes future geometric context in a look-ahead / look-back window.

Recommended fields:
- curvature sequence in window
- average curvature ahead
- maximum curvature ahead
- long-straight-after-corner indicator
- compound-corner / chicane indicator
- future heading-change accumulation

Example:
x_context[i] = [
    curvature_{i-k}, ..., curvature_i, ..., curvature_{i+k},
    avg_curvature_ahead,
    max_curvature_ahead,
    straight_after_corner_score,
    compound_corner_score
]

---

### 4.3 Prior Feature Vector

Name:
`x_prior[i]`

Purpose:
Encodes heuristic racing priors derived from track geometry and strategy logic.

Recommended fields:
- `sharpness_score`
- `low_speed_score`
- `compound_corner_score`
- `straight_after_corner_score`
- `high_speed_score`

Example:
x_prior[i] = [
    sharpness_score,
    low_speed_score,
    compound_corner_score,
    straight_after_corner_score,
    high_speed_score
]

---

## 5. Merge Operation

All three groups are normalized and concatenated into one unified input vector.

x_i = Concat( x_local[i], x_context[i], x_prior[i] )

### 5.1 Normalization

Each feature dimension should be normalized before concatenation.

Recommended methods:
- z-score normalization
- min-max normalization

Example:
x_norm = (x - mean) / std

Purpose:
- keep feature scales consistent
- stabilize training
- prevent one feature group from dominating the network

---

## 6. Network Architecture

## 6.1 Input Layer

Input:
`x_i`

Shape:
`[d_total]`

where `d_total = d_local + d_context + d_prior`

---

## 6.2 Shared 1D CNN Encoder

Purpose:
Extract local interaction patterns from the merged feature vector.

Why 1D CNN:
- the merged vector still contains ordered context structure
- useful for learning local feature combinations
- lighter than transformer
- more structured than a pure MLP for window-based context

Example structure:
- Conv1D layer
- ReLU
- Conv1D layer
- ReLU
- optional pooling
- flattened latent vector

Mathematical form:
h1 = ReLU( Conv1D(x_i) )
h2 = ReLU( Conv1D(h1) )

Output:
`z_i` = latent representation

---

## 6.3 MLP Block

Purpose:
Refine and nonlinearly combine the encoded latent features.

Why MLP after CNN:
- CNN extracts patterns
- MLP fuses and compresses them
- helps map latent features to a scalar semantic score

Example structure:
- Linear
- ReLU
- Linear
- ReLU

Mathematical form:
m1 = ReLU(W1 * z_i + b1)
m2 = ReLU(W2 * m1 + b2)

Output:
`m_i`

---

## 6.4 Prediction Head

Purpose:
Map the refined feature vector to the final scalar score.

Example structure:
- Linear output layer
- optional tanh activation

Mathematical form:
D_i = W_out * m_i + b_out

Optional bounded version:
D_i = tanh(W_out * m_i + b_out)

Recommended interpretation with tanh:
- `D_i = +1` -> strong apex-defense preference
- `D_i = -1` -> strong best-line preference
- `D_i = 0`  -> neutral

---

## 7. Forward Pass Summary

For each sampled point `s_i`:

1. build `x_local[i]`
2. build `x_context[i]`
3. build `x_prior[i]`
4. normalize all feature groups
5. concatenate into `x_i`
6. pass through shared 1D CNN encoder
7. pass through MLP
8. pass through output head
9. obtain scalar score `D(s_i)`

Compact form:

x_i
-> Normalize
-> Merge
-> CNN
-> MLP
-> Head
-> D(s_i)

---

## 8. Loss Function

A composite loss is used to train the Stage 1 network.

L = lambda_1 * L_strategy
  + lambda_2 * L_smooth
  + lambda_3 * L_prior

This loss is applied to the **entire Stage 1 network**, including:
- CNN kernel parameters
- MLP parameters
- Head parameters

---

### 8.1 Strategy Loss

Purpose:
Fit the network output to target defense-preference values.

Recommended form:
Mean squared error (MSE)

L_strategy = sum_i ( D(s_i) - D_target(s_i) )^2

Where `D_target(s_i)` can come from:
- rule-based pseudo labels
- heuristic scoring
- later, refined by trajectory-level evaluation

---

### 8.2 Smoothness Loss

Purpose:
Make the score map continuous along arc length.

Recommended form:
First-order smoothness

L_smooth = sum_i ( D(s_{i+1}) - D(s_i) )^2

Optional second-order version:
L_smooth = sum_i ( D(s_{i+1}) - 2*D(s_i) + D(s_{i-1}) )^2

Effect:
- reduces abrupt oscillation
- makes the map more realistic for downstream planning

---

### 8.3 Prior Consistency Loss

Purpose:
Keep the learned score consistent with geometric racing priors.

L_prior = sum_i ( D(s_i) - D_prior(s_i) )^2

Where `D_prior(s_i)` is computed from heuristic logic such as:
- low-speed sharp corner -> positive tendency
- high-speed corner + long straight after -> negative tendency

---

## 9. Post-processing

The network outputs a score for every sampled point:

D_map = [ D(s_1), D(s_2), ..., D(s_N) ]

Optional post-processing:
- moving average smoothing
- Gaussian smoothing
- thresholding for visualization

Example thresholding:
- `D > +tau` -> defend apex
- `D < -tau` -> keep best line
- otherwise  -> neutral zone

---

## 10. Output

Final output:
A smooth track-wise defense preference map.

Format:
- one scalar score per sampled arc-length point
- can be visualized as:
  - line plot
  - heatmap
  - colored racing line

Usage in Stage 2:
Stage 2 queries `D(s_i)` at the current position and uses it to decide how much to bias the local defensive trajectory.

---

## 11. Recommended Implementation Notes

### 11.1 Input organization
Keep feature order fixed across all samples.

Suggested order:
1. local geometry
2. context geometry
3. prior features

### 11.2 Output range
Use `tanh` if interpretable bounded output is preferred.

### 11.3 Training targets
Start with geometry-based pseudo labels first.
Later refine with:
- human annotation
- simulation outcome
- trajectory-level performance feedback

### 11.4 Stage separation
This module only trains the **offline preference-recognition network**.
It does **not** train the online trajectory planner.

---

## 12. Minimal Pseudocode

```python
for each sampled point s_i:
    x_local   = build_local_geometry_features(s_i)
    x_context = build_context_geometry_features(s_i, window=H)
    x_prior   = build_prior_features(s_i)

    x = concat(normalize(x_local),
               normalize(x_context),
               normalize(x_prior))

    z = cnn_encoder(x)
    m = mlp(z)
    D_i = head(m)

collect all D_i into D_map

loss = lambda1 * strategy_loss(D_map, D_target) \
     + lambda2 * smoothness_loss(D_map) \
     + lambda3 * prior_loss(D_map, D_prior)

update network parameters by backpropagation