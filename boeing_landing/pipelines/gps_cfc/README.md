# gps_cfc — inertial + GPS + wind → Conv1D → CfC → commands

Pure-yaml pipeline: everything it runs is the shared engine (`train.py`,
`data/`, `evaluate.py`, `report.py`) — there is no gps_cfc-specific code.

- `base.yaml` — the pipeline's single control panel
- `quick.yaml` — 3-epoch smoke variant (`extends: base.yaml`)
- `long.yaml` — 60 epochs + early stopping

```bash
make dataset CSV=path/to.csv
make train CONFIG=gps_cfc          # base
make train CONFIG=gps_cfc/quick    # variant
```
