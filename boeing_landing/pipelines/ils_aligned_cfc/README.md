# ils_aligned_cfc — GPS position converted to the runway/ILS-aligned frame

**Converts** the GPS position from absolute lat/lon/alt (which the network was
memorising per airport) into coordinates **at the runway threshold, along the
runway/ILS axis**. GPS is not removed — it is the same fix, re-expressed — so
every approach now looks the same wherever the runway is on Earth and whichever
QFU is flown. ILS is not used (as in `gps_cfc`).

The frame is built as `geodetic --(pymap3d)--> local NED at the LTP --(spin about
Down by the approach course)--> along/cross/up`. The spin is mirrored so the
signs match the ILS deviations: `pos_cross` is positive LEFT like
`localizer_error_m`, `pos_up` positive like `glideslope_error_m`.

Its axis direction is the runway course (the ILS localiser bearing) — that axis
is the one thing the twin `magnetic_north_cfc` pipeline swaps for magnetic north.

## Normalisation (fixed & centralised)

The mechanism lives in `boeing_landing/data/normalization.py`, the numbers in
`boeing_landing/data/physical_bounds.yaml` (bounds are data, not code):

- **Fixed physical bounds** (`build.physical_bounds: true`): the position and
  attitude channels normalise against airport-independent bounds from the
  approach envelope (`pos_along ∈ [-15000, 500] m`, `pitch ∈ [-0.35, 0.35] rad`
  — read the yaml, not this table, for the current values) instead of this
  dataset's min/max, so the scale stays stable when new airports are added.
  The velocity/rate/wind channels keep the train-split min/max.
- **Heading as sin/cos**: the compass heading wraps at ±π, where a min-max is
  meaningless, so it is fed as the smooth pair `heading_sin`, `heading_cos`
  (both bounded [-1, 1]). Pitch and bank stay raw (bounded, no wrap). The pair is
  derived at build time; the raw `heading` column is not stored as an input.

## Files

- `augment_ned.py` — raw ldg_*.csv + NavDB (both read-only) → new csv with the
  runway's LTP/FTP as origin (`poi_*`) and the aircraft position in ILS-signed
  runway coordinates (`pos_along`, `pos_cross`, `pos_up`)
- `plot_runway_frame.py` — approach trajectories of an augmented csv: top view,
  vertical profile, localizer cross-check
- `base.yaml` — training config (inertial + `pos_along/cross/up` → Conv1D →
  CfC); same engine and hyperparameters as `gps_cfc`, only the input set differs

Geodesy helpers (`geodetic_to_ned`, `approach_course`, …) are shared and live in
`boeing_landing/data/geodesy.py` (a thin wrapper over **pymap3d**); only the
ILS-signed spin `_course_spin` is specific to this pipeline.

```bash
make dataset CONFIG=ils_aligned_cfc      # augments the raw csv if needed, then builds the npz (FORCE=1 re-augments)
make train   CONFIG=ils_aligned_cfc      # uses the yaml's input_order (ils_aligned); ORDER=... only to override
make trajectories NED_CSV=datasets/ldg_dataset_images_ned.csv   # SAVE=1 -> figures/dataset/
```

The augmentation to run and its output path come from this pipeline's `augment:`
config block, and `make dataset` runs it when the augmented csv is missing.

## Variants (`extends`) — the run-7 out-of-distribution study

Five variants validate on **run 7** (the wind outlier, see `make run-report`): a
wind out-of-distribution test where val_loss stalls on a floor (~0.036) the base
setup can't break. Each changes **one lever** to try to lower that floor; none
alters any recorded value.

| Variant | Val | Lever | npz |
|---|---|---|---|
| `val_run7` | 7 | none (base params) — the reference | `..._run7` |
| `run7_lowlr` | 7 | lower lr (3e-4) | `..._run7` |
| `run7_reg` | 7 | backbone dropout 0.2 (curb memorisation) | `..._run7` |
| `run7_physwind` | 7 | `physical_bounds: all` (wind in a fixed range -> in [0,1]) | `..._run7_physwind` |
| `run7_zscore` | 7 | `norm_method: zscore` (centred, unbounded) | `..._run7_zscore` |
| `run7_params_test` | 7 | **seven levers at once** — isolates nothing, only asks whether there is any margin left | `..._run7_params_test` |
| `run8_lowlr` | **8** | lr 1e-4 on the baseline split — the in-distribution control for the low-lr lever | base npz |

```bash
make dataset CONFIG=ils_aligned_cfc/val_run7          # base run-7 npz (val_run7, run7_lowlr, run7_reg)
make dataset CONFIG=ils_aligned_cfc/run7_physwind     # its own npz (different bounds)
make dataset CONFIG=ils_aligned_cfc/run7_zscore       # its own npz (mean/std params)
make dataset CONFIG=ils_aligned_cfc/run7_params_test  # its own npz (longer portions)
make train   CONFIG=ils_aligned_cfc/<variant>         # e.g. run7_physwind
```

`run8_lowlr` needs no build of its own: it keeps the baseline run-8 split, so it
reads the npz `make dataset CONFIG=ils_aligned_cfc` already produced.

> **Comparing across normalisation** (min-max vs z-score): `val_loss` is **not**
> comparable — the labels sit on different scales, so the MSE is in different
> units. Use the scale-invariant **R2** from `make evaluate RUN=...` instead.
>
> The honest ceiling: run 7 is genuine extrapolation (an unseen wind); these
> levers help at the margin — the real fix is more runs (more wind conditions).

> **This pipeline trains on 10 of the delivery's 31 runs, all at ZBTJ.** The
> augmentation leaves `pos_along/cross/up` NaN for any (airport, runway) missing
> from `NavDB_MFS.json`, and those rows are dropped at build. The database holds
> ZBTJ but neither MSLP (6 runs) nor YPAD (15 runs), so every `ils_aligned_cfc`
> result — LORO included — describes a **single airport**. That is part of why
> run 7 extrapolates so badly: it is the most atypical run of the only airport
> the model has ever seen. The complete NavDB is still pending delivery;
> `ned_wind_cfc` sidesteps it entirely (the NED arrives precomputed, 8 airports).
