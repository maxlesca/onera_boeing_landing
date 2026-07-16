# LNN_SL_quadrotor_training

This folder contains the supervised-learning pipeline for the quadrotor controller experiments. It is organized so the same saved configuration can be reused across:

- training
- checkpoint reconstruction
- offline testing
- automated feature ablations
- closed-loop simulation from datasets
- random-start robustness simulation
- race-track style gate simulation

## Layout

- [train.py](/mnt/c/users/tmavarva/documents/thesis/autonomous_landing_tudor_2/LNN_SL_quadrotor_training/train.py)
  Main training entrypoint. Reads `train_config.yaml`, builds the requested model, trains it, and writes:
  - a checkpoint into `checkpoints/`
  - a resolved copy of the config into `configs/`

- [test.py](/mnt/c/users/tmavarva/documents/thesis/autonomous_landing_tudor_2/LNN_SL_quadrotor_training/test.py)
  Evaluation entrypoint. Rebuilds the model from the saved YAML, runs the test set, and optionally performs automated feature ablations.

- [Simulator_start_dataset.py](/mnt/c/users/tmavarva/documents/thesis/autonomous_landing_tudor_2/LNN_SL_quadrotor_training/Simulator_start_dataset.py)
  Initializes the simulator from states taken directly from the dataset and compares simulated closed-loop rollouts against reference commands/energy.

- [Simulator_random_start.py](/mnt/c/users/tmavarva/documents/thesis/autonomous_landing_tudor_2/LNN_SL_quadrotor_training/Simulator_random_start.py)
  Samples random physically plausible initial states and measures convergence, energy, and time-to-target.

- [Simulator_race_drone.py](/mnt/c/users/tmavarva/documents/thesis/autonomous_landing_tudor_2/LNN_SL_quadrotor_training/Simulator_race_drone.py)
  Re-centers the drone around successive gates and evaluates repeated gate-passing behavior.

- [utils/model_builder.py](/mnt/c/users/tmavarva/documents/thesis/autonomous_landing_tudor_2/LNN_SL_quadrotor_training/utils/model_builder.py)
  Centralized architecture factory used by all scripts. This is the main place where model type, width scaling, preprocessing blocks, and CfC/LTC options are interpreted.

- [utils/quadrotor_sim.py](/mnt/c/users/tmavarva/documents/thesis/autonomous_landing_tudor_2/LNN_SL_quadrotor_training/utils/quadrotor_sim.py)
  Shared simulation code:
  - state transforms
  - continuous-time dynamics
  - numerical integration
  - observation window management
  - checkpoint-backed controller rollout

- [utils/ablation.py](/mnt/c/users/tmavarva/documents/thesis/autonomous_landing_tudor_2/LNN_SL_quadrotor_training/utils/ablation.py)
  Feature-group masking for automated testing ablations.

- [utils/feedforward.py](/mnt/c/users/tmavarva/documents/thesis/autonomous_landing_tudor_2/LNN_SL_quadrotor_training/utils/feedforward.py)
  Plain MLP baseline, wrapped to match the same sequence interface as the recurrent models.

- [utils/data.py](/mnt/c/users/tmavarva/documents/thesis/autonomous_landing_tudor_2/LNN_SL_quadrotor_training/utils/data.py)
  Dataset loading, normalization-vector construction, feature expansion, and sliding-window generation.

- [utils/lightning.py](/mnt/c/users/tmavarva/documents/thesis/autonomous_landing_tudor_2/LNN_SL_quadrotor_training/utils/lightning.py)
  Shared Lightning wrapper used by training and test-time checkpoint execution.

## Main Workflow

### 1. Training

Edit [train_config.yaml](/mnt/c/users/tmavarva/documents/thesis/autonomous_landing_tudor_2/LNN_SL_quadrotor_training/train_config.yaml), then run:

```bash
python3 LNN_SL_quadrotor_training/train.py
```

Outputs:

- `LNN_SL_quadrotor_training/checkpoints/<checkpoint>.ckpt`
- `LNN_SL_quadrotor_training/configs/<checkpoint>.yaml`

The saved YAML is important because test/simulators rebuild the exact same architecture from it.

### 2. Testing

Set `model_path` in [test_config.yaml](/mnt/c/users/tmavarva/documents/thesis/autonomous_landing_tudor_2/LNN_SL_quadrotor_training/test_config.yaml) to the checkpoint stem, then run:

```bash
python3 LNN_SL_quadrotor_training/test.py
```

Optional plot:

```bash
python3 LNN_SL_quadrotor_training/test.py --plot
```

### 3. Simulators

Set `model_path` in [simulator_config.yaml](/mnt/c/users/tmavarva/documents/thesis/autonomous_landing_tudor_2/LNN_SL_quadrotor_training/simulator_config.yaml), then choose one:

```bash
python3 LNN_SL_quadrotor_training/Simulator_start_dataset.py
python3 LNN_SL_quadrotor_training/Simulator_random_start.py
python3 LNN_SL_quadrotor_training/Simulator_race_drone.py
```

## Supported Model Options

The primary architecture switch lives in `train_config.yaml -> model.type`.

Supported values:

- `cfc`
- `ltc`
- `ncp`
- `ctrnn`
- `simplernn`
- `gru`
- `lstm`
- `mlp`

### CfC-specific options

When `model.type: cfc`, the following are used:

- `model.cfc_mode`
  Values: `default`, `pure`, `no_gate`

- `model.backbone_units`
- `model.backbone_layers`
- `model.backbone_dropout`

### NCP-specific options

When `model.type: ncp`, the following are used:

- `model.ncp.inter_neurons`
- `model.ncp.command_neurons`
- `model.ncp.sensory_fanout`
- `model.ncp.inter_fanout`
- `model.ncp.recurrent_command_synapses`
- `model.ncp.motor_fanin`

The refactored pipeline builds NCP controllers as `CfC` models with an
`ncps.wirings.NCP(...)` wiring.

### Global network scaling

Use:

- `model.scale_factor`

This scales:

- recurrent hidden width
- CfC backbone width
- MLP preprocessing widths
- feedforward hidden widths
- convolution output width

### Recurrent neuron count

Use:

- `model.no_neurons_layer`

This controls the hidden width for recurrent backbones.

### Feedforward / NN baseline

To use a plain non-recurrent controller:

```yaml
model:
  type: mlp
  hidden_layers: [128, 128]
  activation: relu
```

This follows the same train/test/simulator path as the recurrent models.

## Preprocessing Blocks

### Convolutional preprocessing

Use:

```yaml
conv_block:
  value: true
  output_dim: 256
```

This requires sequencing:

```yaml
sequencing:
  value: true
  seq_len: 1
```

### MLP preprocessing

Use:

```yaml
mlp_block:
  value: true
  no_layers: [64, 128, 256]
```

This inserts an MLP feature extractor before the recurrent core.

## Automated Ablation Testing

Ablation is configured in [test_config.yaml](/mnt/c/users/tmavarva/documents/thesis/autonomous_landing_tudor_2/LNN_SL_quadrotor_training/test_config.yaml):

```yaml
ablation:
  enabled: true
  fill_value: 0.0
  feature_sets:
    position: [dx, dy, dz]
    velocity: [vx, vy, vz]
    attitude: [phi, theta, psi]
```

For each named group:

- the matching feature indices are resolved from the configured input label order
- the input tensor is copied
- those channels are replaced with `fill_value`
- the test evaluation is rerun

## Configuration Files

### `train_config.yaml`

Controls:

- dataset labels and paths
- dataloader settings
- sequencing
- optional preprocessing blocks
- model family and hyperparameters
- scaling
- logging

### `test_config.yaml`

Controls:

- which saved model to load
- which test dataset to evaluate
- whether to run ablations
- plotting toggle

### `simulator_config.yaml`

Controls:

- which saved model to load
- which simulation horizon and timestep to use
- which integration method to use
- convergence thresholds
- random-start simulation count

## Notes On Legacy Files

This refactor focuses on the main supervised-learning path. Older side scripts such as:

- `*_NN.py`
- `Simulator_start_dataset_model_comparison.py`
- `test_NN.py`

were left as legacy utilities and were not migrated onto the new shared helper stack.

## Typical Usage Pattern

1. Train with `train.py`.
2. Copy the resulting checkpoint stem.
3. Put that stem into `test_config.yaml` and/or `simulator_config.yaml`.
4. Run `test.py` for baseline and ablation metrics.
5. Run one or more simulator scripts for closed-loop behavior checks.
