# magnetic_frame — GPS position converted to a magnetic-north NED frame

Twin of `runway_frame`, same origin (the runway threshold LTP/FTP) and the same
**GPS-derived position** — only the horizontal axes differ: they point to
**magnetic north** instead of down the runway. Like the runway frame it converts
the GPS fix out of absolute lat/lon (memorised per airport); GPS is not removed.

The point of the two axes: the runway/ILS axis comes from the localiser (ground
infrastructure), while the magnetic axis is the one an onboard **magnetometer**
defines. Training both and comparing their val loss isolates the effect of that
choice — a step toward a later, fully-internal setup (the position source itself
is still GPS here; swapping it for vision/ILS is a separate, future step).

The frame is built as `geodetic --(pymap3d)--> local NED at the LTP --(spin about
Down by the magnetic declination)--> north_mag/east_mag/up`. The vertical
(`pos_up_mag`) is numerically identical to the runway frame's `pos_up`: a spin
about Down leaves the vertical component untouched — only the horizontal pair
differs.

The **declination comes for free from the NavDB**: the QFU is the runway's
magnetic bearing, so `declination = true_course(LTP->FPAP) - QFU*10deg`. It is
coarse (±5°, the QFU is rounded to 10°); a WMM/IGRF model (`pygeomag`, `ppigrf`)
would refine it if the comparison warrants it.

## Files

- `augment_magnetic.py` — raw ldg_*.csv + NavDB (both read-only) → new csv with
  the LTP/FTP origin + declination (`poi_*`) and the aircraft position in
  magnetic-NED coordinates (`pos_north_mag`, `pos_east_mag`, `pos_up_mag`)
- `base.yaml` — training config (inertial + magnetic-NED position → Conv1D →
  CfC); identical to `runway_frame/base.yaml` except the input set

Geodesy helpers are shared in `boeing_landing/data/geodesy.py` (over **pymap3d**);
only the declination-from-QFU and the spin `_declination_spin` are specific here.

```bash
make augment CONFIG=magnetic_frame                      # -> ..._mag.csv (path from the config)
make dataset CSV=datasets/ldg_dataset_images_mag.csv CONFIG=magnetic_frame
make train   CONFIG=magnetic_frame ORDER=magnetic
```

The augmentation to run and its output path come from this pipeline's `augment:`
config block, so the single `make augment CONFIG=...` target serves every frame.

> Rows for any (airport, runway) missing from the NavDB keep NaN and are dropped
> at dataset build, exactly like the runway frame.
