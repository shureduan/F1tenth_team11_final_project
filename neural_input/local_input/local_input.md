# Local Input Vector Definition

## Overview

This document defines the **local input vector** used to describe the geometric properties of the track at each discretized segment.

The track is discretized into **600 segments**.  
For each segment \(i\), a 6-dimensional local feature vector is defined as:

\[
x_{\text{local}}[i] =
\left[
\kappa_i,\;
\Delta \kappa_i,\;
\Delta \psi_i,\;
d_{\text{left},i},\;
d_{\text{right},i},\;
d_{\text{inner},i}
\right]
\]

These six features characterize the local track geometry from the perspectives of:

- curvature intensity,
- curvature variation,
- heading change,
- left available space,
- right available space,
- inner-side available space.

---

## Reference Curves and Conventions

Two different reference curves are used:

### 1. Centerline
The following quantities are computed from the **track centerline**:

- curvature \(\kappa_i\)
- curvature variation \(\Delta \kappa_i\)
- left distance \(d_{\text{left},i}\)
- right distance \(d_{\text{right},i}\)
- inner-side distance \(d_{\text{inner},i}\)

### 2. Raceline
The following quantity is computed from the **raceline**:

- heading change \(\Delta \psi_i\)

This design separates:

- **track geometry itself** (centerline-based),
- from **driving-direction variation** (raceline-based).

---

## Symbol Definitions

- \(i\): current segment index
- \(N=600\): number of discretized track segments
- \((x_i^{c}, y_i^{c})\): centerline point at segment \(i\)
- \((x_i^{r}, y_i^{r})\): raceline point at segment \(i\)
- \(s_i^{c}\): centerline arc-length coordinate
- \(s_i^{r}\): raceline arc-length coordinate
- \(\psi_i^{c}\): heading angle of the centerline at segment \(i\)
- \(\psi_i^{r}\): heading angle of the raceline at segment \(i\)
- \(\kappa_i\): curvature at segment \(i\)
- \(\Delta \kappa_i\): curvature variation at segment \(i\)
- \(\Delta \psi_i\): heading-change rate at segment \(i\)
- \(d_{\text{left},i}\): distance from the centerline point to the left boundary
- \(d_{\text{right},i}\): distance from the centerline point to the right boundary
- \(d_{\text{inner},i}\): inner-side available distance at segment \(i\)

Because the track is closed, all segment indices are treated cyclically.

---

## 1. Curvature

The local curvature is computed from the **centerline**.

First define the centerline heading angle:

\[
\psi_i^{c} = \operatorname{atan2}\!\left(
y_{i+1}^{c} - y_{i-1}^{c},\;
x_{i+1}^{c} - x_{i-1}^{c}
\right)
\]

Then curvature is defined as the rate of heading change with respect to arc length:

\[
\kappa_i = \frac{d\psi^{c}}{ds}
\]

In discrete form, under approximately uniform centerline sampling interval \(\Delta s^{c}\),

\[
\kappa_i \approx
\frac{\psi_{i+1}^{c} - \psi_{i-1}^{c}}{2\Delta s^{c}}
\]

### Interpretation
- Larger \(|\kappa_i|\): sharper local bend
- Smaller \(|\kappa_i|\): closer to straight

---

## 2. Curvature Variation

Curvature variation measures how rapidly curvature changes along the track.

\[
\Delta \kappa_i = \frac{d\kappa}{ds}
\]

In discrete form,

\[
\Delta \kappa_i \approx
\frac{\kappa_{i+1} - \kappa_{i-1}}{s_{i+1}^{c} - s_{i-1}^{c}}
\]

### Interpretation
- Larger \(|\Delta \kappa_i|\): curvature changes more abruptly
- Smaller \(|\Delta \kappa_i|\): local turning structure changes more smoothly

This feature captures **turning complexity**, not simply whether the road is curved.

---

## 3. Heading Change

Heading change is computed from the **raceline**, rather than the centerline.

First define raceline heading:

\[
\psi_i^{r} = \operatorname{atan2}\!\left(
y_{i+1}^{r} - y_{i-1}^{r},\;
x_{i+1}^{r} - x_{i-1}^{r}
\right)
\]

Then define heading-change rate:

\[
\Delta \psi_i = \frac{d\psi^{r}}{ds}
\]

In discrete form,

\[
\Delta \psi_i \approx
\frac{\psi_{i+1}^{r} - \psi_{i-1}^{r}}{s_{i+1}^{r} - s_{i-1}^{r}}
\]

### Interpretation
- Larger \(|\Delta \psi_i|\): the driving direction changes more rapidly
- Smaller \(|\Delta \psi_i|\): the motion direction is more stable

This feature reflects the **local steering demand along the raceline**.

---

## 4. Left Available Space

At each segment, a normal line is constructed from the **centerline** and projected toward the left boundary.

Let \(p_i^{c}\) denote the centerline point and \(p_i^{L}\) the corresponding projected left boundary point. Then

\[
d_{\text{left},i} = \left\| p_i^{L} - p_i^{c} \right\|
\]

### Interpretation
- Larger value: more drivable space exists on the left side
- Smaller value: the centerline is closer to the left boundary

---

## 5. Right Available Space

Similarly, let \(p_i^{R}\) denote the projected right boundary point. Then

\[
d_{\text{right},i} = \left\| p_i^{R} - p_i^{c} \right\|
\]

### Interpretation
- Larger value: more drivable space exists on the right side
- Smaller value: the centerline is closer to the right boundary

Together, \(d_{\text{left},i}\) and \(d_{\text{right},i}\) describe the local lateral geometry of the track around the current segment.

---

## 6. Inner-Side Space

The inner-side space is defined according to the **turn direction**, which is determined by the sign of centerline curvature.

\[
d_{\text{inner},i} =
\begin{cases}
d_{\text{left},i}, & \kappa_i \ge 0 \\
d_{\text{right},i}, & \kappa_i < 0
\end{cases}
\]

### Interpretation
- If the segment is a left-hand turn, the inner side is the left side
- If the segment is a right-hand turn, the inner side is the right side

This feature directly measures how much space remains on the **apex side** of the corner.

---

## Final Local Input Vector

The final 6-dimensional local input vector is

\[
x_{\text{local}}[i] =
\left[
\kappa_i,\;
\Delta \kappa_i,\;
\Delta \psi_i,\;
d_{\text{left},i},\;
d_{\text{right},i},\;
d_{\text{inner},i}
\right]
\]

where:

- \(\kappa_i\): local centerline curvature
- \(\Delta \kappa_i\): centerline curvature variation
- \(\Delta \psi_i\): raceline heading-change rate
- \(d_{\text{left},i}\): left available space
- \(d_{\text{right},i}\): right available space
- \(d_{\text{inner},i}\): inner-side available space

---

## Functional Summary

These six local features answer six different geometric questions:

1. **Curvature**: How sharp is the current bend?
2. **Curvature variation**: How suddenly is the turning structure changing?
3. **Heading change**: How quickly is the local driving direction changing?
4. **Left space**: How much track width is available on the left side?
5. **Right space**: How much track width is available on the right side?
6. **Inner-side space**: How much space remains on the apex side of the current corner?

Together, they provide a compact local geometric description of the current track segment for downstream learning and strategy analysis.