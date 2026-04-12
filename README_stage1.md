# F1TENTH Neural Input Pipeline

## 1. Stage 1 Workflow

This repository implements the **Stage 1: Offline Strategy Recognition** pipeline for the F1TENTH project.

The completed Stage 1 workflow is:

```text
map (.png + .yaml)
    -> extract track boundaries
    -> generate optimal line / raceline
    -> build local input x_local
    -> build look-ahead context input x_context
    -> supplement heuristic scores
    -> generate prior labels x_prior
    -> train defense-preference network
    -> output segment-wise strategy preference
```

### Step 1. Put the map files into `map/`

Place the required track map files into the `map/` folder:

- `*.png` track image
- `*.yaml` map metadata

These are the starting files for the whole pipeline.

---

### Step 2. Generate boundaries and the optimal line

Run the following scripts in `raceline_generation/`:

```bash
python /Users/shure_duan/VScode/f1tenth/raceline_generation/extract_boundaries_u.py
python /Users/shure_duan/VScode/f1tenth/raceline_generation/raceline_u.py
```

This step:

- extracts usable track boundaries
- generates the optimal line / raceline
- produces the geometry used by all later modules

---

### Step 3. Generate neural-network input files

After the raceline is generated, run the input-construction scripts in `neural_input/`.

For the local input:

```bash
python /Users/shure_duan/VScode/f1tenth/neural_input/local_input/x_local_u.py
```

For the look-ahead context input, run the corresponding script in:

```text
/Users/shure_duan/VScode/f1tenth/neural_input/look_ahead_context_input/
```

This step generates the two completed neural-network input branches used in Stage 1:

- `x_local`
- `x_context`

---

### Step 4. Supplement heuristic scores

Then fill in the score file in:

```text
neural_input/heuristic_score_input
```

Scoring rule:

- if there is no clear strategy preference for a segment, assign a neutral score of **3**

These heuristic scores are later used for prior-label construction.

---

### Step 5. Generate prior labels

Run:

```bash
python /Users/shure_duan/VScode/f1tenth/neural_network/label_data/l_prior_from_local_u.py
```

This step generates the prior label branch:

- `x_prior`

---

### Step 6. Train the defense-preference network

Run:

```bash
python /Users/shure_duan/VScode/f1tenth/neural_network/train_defense_preference_network_u.py
```

This trains the Stage 1 neural network to predict the segment-wise strategy preference along the track.

---

### Step 7. Check the final output

After training finishes, the final results can be found in:

```text
/Users/shure_duan/VScode/f1tenth/neural_network/output_u
```

This folder contains the final segment-wise strategy preference predictions.

---

## 2. Overview

This repository contains the geometry-processing and neural-input construction pipeline for the F1TENTH project.

The goal is to start from a track line map, recover the track geometry, generate a valid raceline, construct structured feature inputs for the neural network, and finally predict a continuous defense-preference score along the track.

The Stage 1 model uses three input groups:

- `x_local`
- `x_context`
- `x_prior`

These inputs are normalized separately, merged, and fed into a shared **1D CNN encoder + MLP + prediction head**.

The output is a continuous scalar score along the track:

- positive value -> stronger apex-defense preference
- negative value -> stronger best-line preference
- value near 0 -> neutral

---

## 3. Pipeline Summary

The full Stage 1 pipeline is:

```text
track line map
-> extract boundaries
-> generate raceline
-> build x_local
-> build x_context
-> supplement heuristic scores
-> generate x_prior
-> train neural network
-> output track-wise strategy preference
```

This README focuses on the completed Stage 1 pipeline.

---

## 4. Project Structure

```text
F1TENTH/
├── .claude/
├── boundary_output/
├── map/
├── neural_input/
│   ├── heuristic_score_input/
│   ├── local_input/
│   └── look_ahead_context_input/
├── neural_network/
├── raceline_generation/
├── .gitignore
├── README.md
└── neural_README.md
```

### `map/`
Stores the original track map files:

- `*.png`
- `*.yaml`

### `boundary_output/`
Stores recovered track geometry, such as:

- `centerline.csv`
- `inner_boundary.csv`
- `outer_boundary.csv`

### `raceline_generation/`
Stores scripts for:

- extracting boundaries
- generating the raceline
- related geometry processing

### `raceline_output/`
Stores raceline-generation outputs, such as:

- `raceline.csv`
- `left_boundary.csv`
- `right_boundary.csv`
- overlay figures
- intermediate geometry files

### `neural_input/`
Stores the Stage 1 input-construction modules:

- `local_input/` -> builds `x_local`
- `look_ahead_context_input/` -> builds `x_context`
- `heuristic_score_input/` -> stores manual heuristic scores used for prior-label construction

### `neural_network/`
Stores prior-label generation, training, and model outputs.

### `neural_README.md`
Contains the higher-level neural network design note.

---

## 5. Input Branches

### `x_local`
The local geometric input branch describes the current local track geometry.

Typical features include:

- current curvature
- curvature change rate
- track width
- distance to left boundary
- distance to right boundary
- local heading change

### `x_context`
The look-ahead context branch describes the future track geometry ahead of the current segment.

Typical features include:

- average curvature ahead
- maximum curvature ahead
- long-straight indicator
- continuous-turn / compound-corner indicator
- cumulative heading change ahead

### `x_prior`
The prior-label branch is constructed from the prepared geometric inputs and the heuristic score file.

It provides the prior information used during Stage 1 training.

---

## 6. Neural Network Relationship

The Stage 1 defense-preference function is conceptually:

```text
D(s) = f(G_cur(s), G_future(s:s+H), G_prior(s))
```

In the current implementation:

- `G_cur(s)` mainly corresponds to `x_local`
- `G_future(s:s+H)` mainly corresponds to `x_context`
- `G_prior(s)` corresponds to `x_prior`

These branches are merged and passed through:

1. normalization
2. feature concatenation
3. shared 1D CNN encoder
4. MLP block
5. prediction head

to produce the final scalar preference score.

---

## 7. Key Notes and Practical Tips

### Geometry correctness comes first
Always verify that the extracted boundaries and raceline are correct before building the neural inputs. If the geometry is wrong, everything downstream will be unreliable.

### Keep coordinate systems consistent
Make sure the map image, YAML metadata, boundary CSV files, and raceline outputs all use the same coordinate convention.

### Heuristic scores should be consistent
If a segment does not show a clear preference, use the neutral score **3**.

### Final outputs are stored in `output_u`
The final Stage 1 segment-wise strategy preference results are saved in:

```text
/Users/shure_duan/VScode/f1tenth/neural_network/output_u
```

---

## 8. Minimal Command Recap

```bash
python /Users/shure_duan/VScode/f1tenth/raceline_generation/extract_boundaries_u.py
python /Users/shure_duan/VScode/f1tenth/raceline_generation/raceline_u.py
python /Users/shure_duan/VScode/f1tenth/neural_input/local_input/x_local_u.py
python /Users/shure_duan/VScode/f1tenth/neural_network/label_data/l_prior_from_local_u.py
python /Users/shure_duan/VScode/f1tenth/neural_network/train_defense_preference_network_u.py
```

Also run the corresponding look-ahead context script inside:

```text
/Users/shure_duan/VScode/f1tenth/neural_input/look_ahead_context_input/
```

and fill in the heuristic score file in:

```text
neural_input/heuristic_score_input
```
