# ned_wind_cfc — 85 runs, 8 airports, wind as an input

The pipeline for the second delivery: **223 739 frames at 25 Hz, 85 landings over
8 airports and 30 runways**, against 31 runs on 3 airports before. Two things
change beyond the size — the wind is now **sampled at every frame** instead of
being constant per run, and the position arrives **already expressed as a local
NED at the landing threshold**, so no geodesy step is needed.

## Two commands

```bash
make dataset CONFIG=ned_wind_cfc      # the two raw csv -> ned_wind.csv -> npz
make train   CONFIG=ned_wind_cfc      # the default Conv1D -> CfC recipe
```

`make dataset` runs the whole chain: it sees the `prepare:` block in the config,
produces `datasets/ned_wind.csv` if it is missing, then builds the npz. `FORCE=1`
re-runs the prepare step. Sources are never modified.

## Inputs — 20 channels + dt

| group | channels |
|---|---|
| position (local NED, **as delivered**) | `x_error_NED`, `y_error_NED`, `z_error_NED` |
| attitude | `pitch`, `bank`, `heading_sin`, `heading_cos` |
| angular rates | `p`, `q`, `r` |
| ground velocity in body axes | `u`, `v`, `w` |
| linear wind | `wind_velocity_x/y/z` |
| rotational gusts | `wind_rate_x/y/z` |
| weight-on-wheels | `touchdown_flag` |
| appended by the loader | `dt` → the CfC timespans |

Dropped: absolute GPS, NED velocity, ILS. The heading is sin/cos encoded because
it wraps at ±π; a min-max on a wrapping angle is meaningless.

Channel for channel this is Tudor's quadrotor input set, and the wind is his
`M_ext` external disturbance: the variable the expert compensates and that the
network simply could not see in the first delivery.

**Labels: the 8 command channels.** Three of them are constant on this delivery
(`directional`, `flap`, `speedbreak`) and `throttle_right` duplicates
`throttle_left`, so four carry signal. The constant ones are learnt as a flat 0
in one epoch, which mechanically deflates `val_loss` — **compare per-channel R²,
not val_loss, against a 6-label pipeline**.

## Normalisation

Min-max on the **training runs only**, as delivered (`physical_bounds: false`).
The bounds are computed on the train split and embedded in both npz, so the
validation split is normalised with the training bounds — never with its own.

## Portions: 500 steps (20 s)

Epoch cost is `T × (portion_len / stride)`, so **at a constant overlap ratio a
longer portion is free**. 20 s rather than the previous 5 s because the hidden
state restarts from zero at every portion: over 5 s the CfC spends the window
warming up and never uses its dynamics, and 5 s cannot contain the physical time
scales anyway (flare 5–10 s, phugoid 30–60 s). BPTT is four times deeper, hence
`gradient_clip_val: 1.0`.

## Phase 1 split — 30 train runs, 3 held out

The run lists live in `base.yaml` (`build.train_runs` / `build.val_runs`), so the
split is versioned with the config. The three validation runs are picked for
three different reasons:

| run | airport | why |
|---|---|---|
| 172 | VHHH 07R | crosswind +9.97 m/s, the strongest of the 85 — the analogue of the old run 7 |
| 98 | WMKK 14R | headwind −14.76 m/s, the strongest of the 85 |
| 226 | VVTS 25L | median wind — the easy control |

Averaging them into one number would destroy the point: alone, a bad score cannot
distinguish "does not extrapolate" from "has not converged"; side by side it can.
`make evaluate` prints **one line per validation run**.

`make data-report CONFIG=ned_wind_cfc` confirms the split is a real extrapolation
test: 28.5 % of the validation frames fall outside [0,1] on `wind_velocity_y` and
12.6 % on `wind_velocity_x` after normalisation.

The remaining 52 runs are unused on purpose — phase 1 buys iteration speed,
phase 2 (repeated random holdout over several seeds) uses everything.

## The four arms — all of them CfC

Only the block **in front of** the CfC changes; `model.type` stays `cfc`
everywhere. All four read the **same npz**, so the dataset is built once.

```bash
make train CONFIG=ned_wind_cfc              # default: Conv1D(256) -> CfC (as every pipeline)
make train CONFIG=ned_wind_cfc/bare         # no encoder: state -> CfC
make train CONFIG=ned_wind_cfc/conv64       # lighter Conv1D(64) -> CfC
make train CONFIG=ned_wind_cfc/mlp_encoder  # MLP [64,64] -> CfC
make plots RUNS="runs/ned_wind_cfc/..." BARS=1 SAVE=1
```

| arm | encoder | total params | of which encoder |
|---|---|---|---|
| **default** | ConvBlock, `output_dim: 256` | **363 016** | 318 592 (88 %) |
| bare | none | 44 424 | — |
| conv64 | ConvBlock, `output_dim: 64` | 214 984 | 164 928 (77 %) |
| mlp_encoder | MLP [64, 64] | 55 688 | 5 632 |

The Conv1D is the default only for **consistency** with the other pipelines
(gps/ils/magnetic), NOT because it is proven best here — and the `bare` and
`conv64` arms exist precisely to check whether it earns its parameters. Reasons
to doubt it: the CfC already has its own backbone (128 units); the convolution
slides its kernel over the **feature axis** (`in_channels = seq_len`), which
imposes weight sharing between unrelated physical quantities; and on Tudor's own
quadrotor the conv block made the CfC **2.3× worse** (`val_loss` 0.000326 against
0.000142) — the worst of all his recurrent models. See DOC §8.19.6.

## Learning-rate schedule (optional)

Constant `lr` by default, so every earlier result stays comparable. To move fast
early and refine late:

```bash
make train CONFIG=ned_wind_cfc/cosine
```

`training.scheduler.type` accepts `none | cosine | plateau | step`
(`utils/scheduler.py`). Cosine is the one to use for a comparison: it
depends only on the epoch count, so two arms sharing `max_epochs` follow the exact
same lr curve and any difference between them comes from the architecture.

## Files

| file | role |
|---|---|
| `prepare.py` | delivery → canonical csv: renames the columns the repo knows under another name, drops the unused ones, checks the runways against the airport table. Never recomputes a value. |
| `base.yaml` | the default Conv1D(256) arm + the phase-1 split + every knob |
| `bare.yaml`, `conv64.yaml`, `mlp_encoder.yaml` | the three other encoder arms, `extends: base.yaml` |
| `cosine.yaml` | same as the default arm with the cosine schedule |

The airport table (`airports_info_complete.csv`) contributes **no feature**: the
position is used exactly as delivered, with no frame conversion. It is read to
verify that every runway flown exists in the navigation database, and it holds
the 4 runway corners that objective 2 (YOLO) will need.

## Known data issues

- **`q` is flagged CONSTANT by `make data-report`.** 14 frames out of 80 397 spike
  to 2.64 rad/s at the instant of touchdown (the impact transient), which
  compresses the whole useful range of the channel into under 1 % of [0, 1].
  `p` and `r` are affected in the same way by the transient at run start. The fix
  is `physical_bounds: all` (p/q/r bounded at ±0.5), left off here because the
  brief asks for min-max on the train split.
- **`wind_rate_y = 0.845529 × wind_rate_x`, exactly**, over all 223 739 rows.
  After normalisation the network gets the same channel twice. Kept deliberately;
  only two of the three rotational components carry information.
- **The wind frame is probably not the NED the file name announces**:
  `wind_velocity_x` is negative in 96 % of the frames across 30 differently
  oriented runways, which points to the same body/track frame as `u, v, w`. To be
  confirmed with Tudor — see DOC §8.19.3.
