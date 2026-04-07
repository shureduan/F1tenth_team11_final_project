# F1TENTH Neural Input Pipeline

## 1. Overview

This repository contains the geometry-processing and neural-input construction pipeline for the F1TENTH project.

The goal of this module is to start from a track line map, recover the track geometry, generate a valid raceline, and then construct structured feature inputs for the neural network. These inputs are intended for **Stage 1: Offline Strategy Recognition**, where the model predicts a continuous track-wise defense-preference score along the track. The downstream neural design uses three input groups, merges them after normalization, and feeds them into a shared **1D CNN encoder + MLP + prediction head**.

At the current stage, the completed pipeline includes:

1. extracting the **centerline**, **inner boundary**, and **outer boundary**
2. generating a constrained **raceline**
3. constructing the **local geometric input** `x_local`
4. visualizing local features for sanity checking
5. constructing the **look-ahead context input** `x_context`
6. visualizing context features for sanity checking

A third input branch is planned but not completed yet. A placeholder is reserved in this README for that module.

---

## 2. Pipeline Summary

The current workflow is:

**track line map**  
→ extract **centerline / inner boundary / outer boundary**  
→ generate **raceline**  
→ build **local input `x_local`**  
→ visualize local features  
→ build **look-ahead context input `x_context`**  
→ visualize context features  
→ reserve **third input branch**  
→ construct the neural network later

This README focuses on the **input-generation side** of the project.

---

## 3. Project Structure

A simplified project structure is shown below:

```text
F1TENTH/
├── boundary_output/
├── map/
├── neural_input/
│   ├── local_input/
│   └── look_ahead_context_input/
├── raceline_generation/
├── raceline_output/
└── neural_README.md
```

### `map/`
Stores the original track map files, typically including:

- `*.png` track image
- `*.yaml` map metadata

The YAML file provides map resolution, origin, and coordinate conversion information.

### `boundary_output/`
Stores the recovered track geometry, including:

- `centerline.csv`
- `inner_boundary.csv`
- `outer_boundary.csv`

These files are the geometric foundation of all later steps.

### `raceline_generation/`
Stores scripts related to:

- extracting track geometry from the line map
- generating the raceline from boundaries
- related geometric processing

### `raceline_output/`
Stores outputs from raceline generation and later geometric processing, such as:

- `raceline.csv`
- `left_boundary.csv`
- `right_boundary.csv`
- overlay figures
- intermediate geometry outputs

### `neural_input/`
Stores neural-network input construction modules.

Currently this folder contains two completed components:

- `local_input/` → builds `x_local`
- `look_ahead_context_input/` → builds `x_context`

A third input branch will be added later.

### `neural_README.md`
Contains the higher-level neural network design note, including the intended Stage 1 architecture, input merging logic, and output interpretation.

---

## 4. Actual Execution Order

The current working pipeline uses the following six scripts in order.

### Step 1. Extract centerline and track boundaries

**Script**

```text
raceline_generation/track_from_line_map.py
```

**Purpose**

This script extracts the base track geometry from the track line map and produces:

- centerline
- inner boundary
- outer boundary

**Main output location**

```text
boundary_output/
```

**Typical outputs**

- `centerline.csv`
- `inner_boundary.csv`
- `outer_boundary.csv`

**Why this step matters**

This is the geometric foundation of the entire pipeline. If the centerline or boundaries are wrong, the raceline and all later feature vectors will also be wrong.

---

### Step 2. Generate raceline from boundaries

**Script**

```text
raceline_generation/raceline_from_boundaries.py
```

**Purpose**

This script uses:

- `centerline.csv`
- `inner_boundary.csv`
- `outer_boundary.csv`

to generate a raceline that stays within the track boundaries.

**Main output location**

```text
raceline_output/
```

**Typical outputs**

- `raceline.csv`
- `left_boundary.csv`
- `right_boundary.csv`
- overlay plots for debugging and validation

**Why this step matters**

The raceline is the reference trajectory used by later feature-generation modules. Instead of inferring track geometry again from the map, this step directly uses the already extracted boundaries.

---

### Step 3. Build local input `x_local`

**Script**

```text
neural_input/local_input/x_local_calculate_centerline_segments_centerline_segmented.py
```

**Purpose**

This script computes local geometric features at each segmented track location.

These features describe the **current local geometry** around the sampled position.

**Current local features**

- current curvature
- curvature change rate
- track width
- distance to left boundary
- distance to right boundary
- local heading change

**Typical outputs**

- `x_local_features.csv`
- related intermediate files if needed

**Role in the neural network**

This module builds the first feature group, `x_local`, which corresponds to the current local geometric condition in the Stage 1 model.

---

### Step 4. Visualize local input

**Script**

```text
neural_input/local_input/x_local_visualize_sector.py
```

**Purpose**

This script maps the local features back onto the segmented track and visualizes them.

**Why this step matters**

This is an important sanity-check stage. It helps verify:

- whether the track segmentation is correct
- whether the geometric quantities are mapped to the correct track sectors
- whether `x_local` truly reflects the local shape of the track

**Typical visual outputs**

- sector-based overlays
- curvature heatmaps
- heading or boundary-distance visualizations

---

### Step 5. Build look-ahead context input `x_context`

**Script**

```text
neural_input/look_ahead_context_input/x_context_calculate_lookahead.py
```

**Purpose**

This script computes context features over a look-ahead window along the track.

These features describe the **future geometric context** seen from the current track position.

**Current context features**

- average curvature ahead
- maximum curvature ahead
- long-straight indicator
- compound-corner / continuous-turn indicator
- cumulative heading change ahead

**Typical outputs**

- `x_context_features.csv`

**Role in the neural network**

This module builds the second feature group, `x_context`, which captures future geometric structure in the Stage 1 model.

---

### Step 6. Visualize look-ahead context input

**Script**

```text
neural_input/look_ahead_context_input/x_context_visualize.py
```

**Purpose**

This script visualizes the context features on the track for validation.

**Why this step matters**

This step checks whether the look-ahead window is working correctly and whether the future-context features actually reflect straights, corners, and compound turns in a meaningful way.

**Typical visual outputs**

- track overlays
- context heatmaps
- look-ahead feature illustrations

---

## 5. Reserved Third Input Branch

A third input module is planned but is **not completed yet**.

For now, this README reserves a placeholder for that branch.

### Planned third input

**Name**

```text
x_prior
```

**Purpose**

This branch is intended to encode prior or heuristic racing knowledge derived from geometry and strategy logic. In the higher-level network design, it is treated as the third feature group alongside `x_local` and `x_context`.

**Current status**

```text
TODO
```

**Reserved fields**

- Function description: `*** TO BE COMPLETED ***`
- Input source: `*** TO BE COMPLETED ***`
- Script path: `*** TO BE COMPLETED ***`
- Output files: `*** TO BE COMPLETED ***`

---

## 6. Relationship Between Inputs and the Neural Network

The intended Stage 1 network uses three feature groups:

- `x_local`
- `x_context`
- `x_prior` (reserved)

These feature groups are normalized separately and then concatenated into one unified input vector before entering the network.

The overall defense preference function is described as:

```text
D(s) = f( G_cur(s), G_future(s:s+H) )
```

In the current implementation logic:

- `G_cur(s)` mainly corresponds to `x_local`
- `G_future(s:s+H)` mainly corresponds to `x_context`
- `x_prior` provides an additional prior / heuristic branch later

The output is a continuous scalar score along the track:

- `D(s) > 0` → apex defense is preferred
- `D(s) < 0` → staying close to the best line is preferred
- `|D(s)|` → strength of the preference

---

## 7. Next Stage: Neural Network Construction

After all three input branches are ready, the next step is to build the neural network itself.

According to the current design, the merged feature vector will be passed through:

1. normalization
2. feature concatenation
3. shared 1D CNN encoder
4. MLP block
5. prediction head

to produce a scalar defense-preference score at each sampled track position.

This README focuses on the **input-generation side** of the pipeline. The neural network training and inference modules belong to the next implementation stage.

---

## 8. Recommended Usage Order

Use the pipeline in the following order:

1. run `track_from_line_map.py`
2. run `raceline_from_boundaries.py`
3. run `x_local_calculate_centerline_segments_centerline_segmented.py`
4. run `x_local_visualize_sector.py`
5. run `x_context_calculate_lookahead.py`
6. run `x_context_visualize.py`
7. add the third input branch later
8. proceed to neural network construction and training

---

## 9. How to Run

Below is the conceptual run order. Use your local paths and arguments as needed.

### 9.1 Generate track geometry

```bash
python raceline_generation/track_from_line_map.py
```

This step generates:

- `boundary_output/centerline.csv`
- `boundary_output/inner_boundary.csv`
- `boundary_output/outer_boundary.csv`

### 9.2 Generate raceline

```bash
python raceline_generation/raceline_from_boundaries.py
```

This step generates:

- `raceline_output/raceline.csv`
- `raceline_output/left_boundary.csv`
- `raceline_output/right_boundary.csv`

### 9.3 Compute local features

```bash
python neural_input/local_input/x_local_calculate_centerline_segments_centerline_segmented.py
```

This step generates local input features for each segmented track location.

### 9.4 Visualize local features

```bash
python neural_input/local_input/x_local_visualize_sector.py
```

This step checks whether local features are correctly mapped back to the track.

### 9.5 Compute look-ahead context features

```bash
python neural_input/look_ahead_context_input/x_context_calculate_lookahead.py
```

This step generates context features over a look-ahead window.

### 9.6 Visualize context features

```bash
python neural_input/look_ahead_context_input/x_context_visualize.py
```

This step validates whether future-context information is captured properly.

---

## 10. Key Notes and Practical Tips

### 10.1 Geometry correctness comes first

Always verify that the extracted:

- centerline
- inner boundary
- outer boundary

are correct before generating the raceline.

If the geometry is wrong, everything downstream will be unreliable.

### 10.2 Do not skip visualization

Both `x_local` and `x_context` should be visualized before they are used for model training.

This helps catch:

- segmentation errors
- incorrect boundary mapping
- unreasonable curvature values
- failed look-ahead logic

### 10.3 Keep coordinate systems consistent

Make sure that:

- the map image
- YAML metadata
- boundary CSV files
- raceline outputs

all use the same coordinate convention.

### 10.4 The third input is still pending

The third branch is intentionally left open in this README and should be updated after that module is implemented.

---

## 11. Future Work

Planned next steps include:

- implementing the third input branch
- completing the Stage 1 neural network code
- training the network on merged geometric inputs
- generating a smooth track-wise defense-preference map
- using that output later in Stage 2 for online defensive trajectory generation

---

## 12. Minimal Workflow Recap

In short, the current pipeline is:

```text
track_from_line_map.py
    -> centerline / inner boundary / outer boundary

raceline_from_boundaries.py
    -> raceline

x_local_calculate_centerline_segments_centerline_segmented.py
    -> x_local

x_local_visualize_sector.py
    -> local feature validation

x_context_calculate_lookahead.py
    -> x_context

x_context_visualize.py
    -> context feature validation

third input branch
    -> reserved

neural network
    -> to be implemented next
```

---

## 13. Reference Design Note

The current input organization and downstream network interpretation are consistent with the project’s Stage 1 neural design note in `neural_README.md`, which defines the three-branch input structure and the later 1D CNN + MLP + head architecture.
