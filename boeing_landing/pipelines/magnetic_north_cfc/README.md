# magnetic_north_cfc — GPS position converted to a magnetic-north NED frame

Twin of `ils_aligned_cfc`, same origin (the runway threshold LTP/FTP) and the same
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

## Normalisation (fixed & centralised)

Same scheme as `ils_aligned_cfc`, in `boeing_landing/data/normalization.py`
(`build.physical_bounds: true`, heading as `heading_sin`/`heading_cos`). One
frame-specific point: the horizontal position bounds are **symmetric and equal**
on both axes — `pos_north_mag` and `pos_east_mag` are both `[-15000, 15000] m`.
A runway of any QFU projects its full length onto both geographic axes, so fixing
them identically is exactly what keeps the magnetic frame airport-independent
(a data-driven bound would give this N-S runway a tiny east range that would clip
an east-west runway later).

## Files

- `augment_magnetic.py` — raw ldg_*.csv + NavDB (both read-only) → new csv with
  the LTP/FTP origin + declination (`poi_*`) and the aircraft position in
  magnetic-NED coordinates (`pos_north_mag`, `pos_east_mag`, `pos_up_mag`)
- `base.yaml` — training config (inertial + magnetic-NED position → Conv1D →
  CfC); identical to `ils_aligned_cfc/base.yaml` except the input set

Geodesy helpers are shared in `boeing_landing/data/geodesy.py` (over **pymap3d**);
only the declination-from-QFU and the spin `_declination_spin` are specific here.

```bash
make dataset CONFIG=magnetic_north_cfc   # runs the augmentation if its csv is missing, then builds the npz (FORCE=1 re-runs it)
make train   CONFIG=magnetic_north_cfc   # uses the yaml's input_order (magnetic_north); ORDER=... only to override
```

The augmentation to run, its source and its output path all come from this
pipeline's `augment:` config block, so one command builds the whole chain —
same mechanism as `ils_aligned_cfc`.

> Rows for any (airport, runway) missing from the NavDB keep NaN and are dropped
> at dataset build, exactly like the runway frame.
