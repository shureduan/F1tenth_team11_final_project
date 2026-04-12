# Starter Label Data

This file provides an initial **pseudo-label** dataset for the Shanghai track.

## Files

- `Shanghai_label.csv`
  - `segment_id`
  - `D_target`: starter pseudo target in [-1, 1]
  - `D_prior_seed`: unsmoothed heuristic seed

## Meaning

- positive value -> stronger apex-defense preference
- negative value -> stronger best-line preference
- value near 0 -> neutral

## Notes

This is only a **starter label set** for training and debugging.
It is generated from the current local/context features plus the heuristic prior,
not from real expert annotation or simulation outcomes.

Place `Shanghai_label.csv` into your folder:

`/Users/shure_duan/VScode/f1tenth/neural_network/label_data`
