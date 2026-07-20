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
make augment CONFIG=ils_aligned_cfc RAW_CSV=datasets/ldg_dataset_images.csv   # raw -> augment.out_csv; RAW_CSV picks the source dataset
make dataset CONFIG=ils_aligned_cfc      # reads augment.out_csv back, derived (no CSV= needed)
make train   CONFIG=ils_aligned_cfc ORDER=ils_aligned
make trajectories NED_CSV=datasets/ldg_dataset_images_ned.csv   # SAVE=1 -> figures/dataset/
```

The augmentation to run and its output path come from this pipeline's `augment:`
config block, so the single `make augment CONFIG=...` target serves every frame.

> The augmentation leaves NaN for any (airport, runway) missing from the NavDB;
> those runs are dropped at dataset build. The complete NavDB (MSLP, YPAD) is
> still pending delivery.
