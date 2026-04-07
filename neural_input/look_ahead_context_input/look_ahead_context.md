# Look-Ahead Context Vector Definition

## Overview

This document defines the **look-ahead context vector** used in Stage 1 of the track-context defense preference network.

The track is discretized into **600 segments**.  
For each current segment \(i\), a **forward look-ahead window of 30 segments** is used to compute a 5-dimensional context feature vector:

\[
x_{\text{context}}[i] =
\left[
\bar{\kappa}_{i}^{(30)},
\kappa_{\max,i}^{(30)},
S_{\text{straight},i}^{(30)},
S_{\text{compound},i}^{(30)},
\Delta \psi_{i}^{(30)}
\right]
\]

These five features summarize the geometric structure of the track ahead and are intended to capture whether the upcoming section favors defensive apex positioning or staying close to the optimal racing line.

---

## Window Definition

Let the track contain \(N=600\) segments.  
For the current segment \(i\), define the forward look-ahead window as:

\[
W_i = \{i, i+1, i+2, \dots, i+29\}
\]

Because the track is closed, indices are wrapped cyclically:

\[
j_m = (i+m) \bmod 600, \qquad m = 0,1,\dots,29
\]

Thus, all features below are computed over the 30-segment forward window.

---

## Symbol Definitions

- \(i\): current segment index
- \(m\): offset inside the look-ahead window
- \(j_m\): wrapped segment index in the window
- \(\kappa_{j_m}\): curvature at segment \(j_m\)
- \(|\kappa_{j_m}|\): absolute curvature at segment \(j_m\)
- \(\Delta s_{j_m}\): arc-length of segment \(j_m\)
- \(\kappa_{\text{straight}}\): curvature threshold for identifying straight segments
- \(\kappa_{\text{turn}}\): curvature threshold for identifying turning segments
- \(L_{\text{straight},i}^{\max}\): longest consecutive straight subsequence length in the window
- \(N_{\text{turn},i}\): number of turning segments in the window
- \(N_{\text{switch},i}\): number of curvature sign changes among turning segments
- \(\alpha \in [0,1]\): weighting coefficient for compound-corner scoring

In all definitions below, **absolute curvature** is used unless otherwise noted, since the purpose is to measure geometric intensity rather than left/right direction alone.

---

# Look-Ahead Context Input

This module computes the **look-ahead context feature vector** for each track segment.  
Its purpose is to describe the **future geometric trend** of the track, so that the neural network can infer whether the upcoming region encourages a conservative line, a defensive entry, or continued acceleration on the optimal path.

For each current segment index \(i\), a future window of length \(H\) is defined.  
In this project, we use:

\[
H = 30
\]

Thus, the look-ahead window is:

\[
\mathcal{W}_i = \{i, i+1, \dots, i+29\}
\]

where the indices are treated cyclically along the closed track.

The final look-ahead context vector is:

\[
x_{\text{context}}(i)
=
\Big[
\bar{\kappa}_i^{(30)},
\ \kappa^{\max}_i{}^{(30)},
\ I_{\text{straight},i}^{(15)},
\ I_{\text{compound},i}^{(30)},
\ \Theta_i^{(30)}
\Big]
\]

where each term is defined below.

---

## 1. Mean Curvature Ahead

The first feature measures the **average turning intensity** in the next 30 segments.

\[
\bar{\kappa}_i^{(30)}
=
\frac{1}{30}\sum_{j \in \mathcal{W}_i} |\kappa_j|
\]

where:

- \(\kappa_j\) is the curvature at segment \(j\)
- \(|\kappa_j|\) is used so that both left and right turns contribute positively to turning intensity

This feature describes whether the upcoming region is mostly straight or generally curved.

---

## 2. Maximum Curvature Ahead

The second feature captures the **sharpest future turn** within the 30-segment look-ahead window.

\[
\kappa^{\max}_i{}^{(30)}
=
\max_{j \in \mathcal{W}_i} |\kappa_j|
\]

This feature is useful for identifying whether a particularly tight corner exists ahead, even if the overall average curvature is moderate.

---

## 3. Long Straight Indicator (15-Segment Backward Check)

The straight indicator is defined using the **previous 15 segments relative to the current index**.  
That is, for segment \(i\), the straightness check uses:

\[
\mathcal{B}_i^{(15)} = \{i-14, i-13, \dots, i\}
\]

For example, if \(i=30\), then the checked indices are:

\[
30, 29, 28, \dots, 16
\]

(indices are cyclic on the closed track).

First, let \(P_{20}\) denote the **20th percentile** of \(|\kappa|\) over the full track.  
A segment is considered locally straight if:

\[
|\kappa_j| < P_{20}
\]

Then the straight indicator is defined as:

\[
I_{\text{straight},i}^{(15)}
=
\begin{cases}
1, & \text{if all 15 consecutive segments in } \mathcal{B}_i^{(15)} \text{ satisfy } |\kappa_j| < P_{20} \\
0, & \text{otherwise}
\end{cases}
\]

This means the indicator becomes active only when the recent 15-segment region is consistently among the lowest-curvature parts of the track.

Interpretation:

- \(I_{\text{straight},i}^{(15)} = 1\): the local region behaves like a long straight
- \(I_{\text{straight},i}^{(15)} = 0\): the local region is not sufficiently straight

---

## 4. Compound Corner Indicator (30-Segment Future Proportion Rule)

The compound-corner indicator is defined over the next 30 segments:

\[
\mathcal{W}_i = \{i, i+1, \dots, i+29\}
\]

Let \(P_{60}\) denote the **60th percentile** of \(|\kappa|\) over the full track.  
A segment is considered part of a strong turning region if:

\[
|\kappa_j| > P_{60}
\]

Then the compound-corner indicator is defined as:

\[
I_{\text{compound},i}^{(30)}
=
\begin{cases}
1, & \text{if at least 20 segments in } \mathcal{W}_i \text{ satisfy } |\kappa_j| > P_{60} \\
0, & \text{otherwise}
\end{cases}
\]

Equivalently, this can be written as:

\[
I_{\text{compound},i}^{(30)}
=
\begin{cases}
1, & \displaystyle \sum_{j \in \mathcal{W}_i} \mathbf{1}(|\kappa_j| > P_{60}) \ge 20 \\
0, & \text{otherwise}
\end{cases}
\]

where \(\mathbf{1}(\cdot)\) is the indicator function.

Important note:  
the 20 high-curvature segments **do not need to be consecutive**.  
As long as 20 out of the next 30 segments are above the 60th percentile, the region is classified as a compound / sustained turning zone.

Interpretation:

- \(I_{\text{compound},i}^{(30)} = 1\): the upcoming region is strongly corner-dominated
- \(I_{\text{compound},i}^{(30)} = 0\): the upcoming region does not contain enough turning intensity overall

---

## 5. Accumulated Heading Change Ahead

The fifth feature measures the **total absolute heading variation** across the next 30 segments.

Let \(\psi_j\) denote the heading angle at segment \(j\).  
The heading increment is:

\[
\Delta \psi_j = \operatorname{wrap}(\psi_{j+1} - \psi_j)
\]

where \(\operatorname{wrap}(\cdot)\) maps the angle difference into \((-\pi, \pi]\) to avoid discontinuities caused by angle wrapping.

Then the accumulated heading change is defined as:

\[
\Theta_i^{(30)}
=
\sum_{j=i}^{i+29} |\Delta \psi_j|
\]

This feature reflects how much total turning the vehicle will experience over the future window.

A larger value means the track direction changes substantially ahead, even if the curvature is distributed across multiple moderate bends rather than one sharp corner.

---

## Final Look-Ahead Context Vector

Combining all terms, the final look-ahead context input is:

\[
x_{\text{context}}(i)
=
\left[
\frac{1}{30}\sum_{j=i}^{i+29} |\kappa_j|,
\quad
\max_{j \in [i,i+29]} |\kappa_j|,
\quad
I_{\text{straight},i}^{(15)},
\quad
I_{\text{compound},i}^{(30)},
\quad
\sum_{j=i}^{i+29} |\Delta\psi_j|
\right]
\]

This 5-dimensional vector summarizes the geometric trend of the track ahead using:

1. average future curvature  
2. maximum future curvature  
3. straight-region detection  
4. sustained corner-region detection  
5. total future heading variation  

These features are later concatenated with other local geometric inputs and used as part of the neural network input representation.

---

## Practical Meaning of the Five Features

- **Mean curvature ahead**: how curved the next 30 segments are on average  
- **Max curvature ahead**: whether a very sharp turn exists ahead  
- **Straight indicator**: whether the recent 15-segment region behaves like a long straight based on low curvature percentile  
- **Compound indicator**: whether the next 30 segments are dominated by turning behavior based on high curvature percentile  
- **Accumulated heading change**: how much total direction change is coming ahead  

---

## Threshold Logic Used in This Project

The percentile-based thresholds are computed from the global curvature distribution of the full track:

- \(P_{20}\): 20th percentile of \(|\kappa|\)
- \(P_{60}\): 60th percentile of \(|\kappa|\)

The logic is:

- **Straight region**: all of the checked 15 segments must satisfy  
  \[
  |\kappa_j| < P_{20}
  \]

- **Compound corner region**: at least 20 of the next 30 segments must satisfy  
  \[
  |\kappa_j| > P_{60}
  \]

This percentile-based design makes the rule adaptive to different track shapes, instead of relying on one fixed curvature threshold for all maps.

---

## Output

For every segment \(i\), this module outputs one 5-dimensional feature vector:

\[
x_{\text{context}}(i) \in \mathbb{R}^5
\]

These vectors are stored sequentially along the full track and later used as one branch of the network input.

---

## Functional Summary

These five features answer five different geometric questions about the track ahead:

1. **Mean curvature**: How curved is the upcoming region overall?
2. **Maximum curvature**: Is there a sharp corner ahead?
3. **Long-straight indicator**: Is there a sustained straight section ahead?
4. **Compound-corner indicator**: Is the future geometry continuously or complexly turning?
5. **Accumulated heading change**: How much total steering rotation will be required ahead?

Together, they provide a compact but informative representation of future track geometry for downstream strategy prediction.

