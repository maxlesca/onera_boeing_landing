# runway_frame — aircraft GPS re-expressed in the landing runway's frame

Pipeline-specific code (only this pipeline uses it, hence it lives here):

- `augment_ned.py` — raw ldg_*.csv + NavDB (both read-only) → new csv with
  the landing runway's LTP/FTP as origin (`poi_*`) and the aircraft position
  in ILS-signed runway coordinates (`pos_along`, `pos_cross`, `pos_up`)
- `plot_runway_frame.py` — approach trajectories of an augmented csv:
  top view, vertical profile, localizer cross-check

```bash
make augment          # datasets/ldg_dataset_images.csv -> ..._ned.csv
make trajectories     # pyplot window; SAVE=1 -> figures/dataset/
```

The training config (`base.yaml` + variants) will join this folder once the
complete NavDB (MSLP, YPAD) is delivered.
