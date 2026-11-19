# OpenTouch two-/three-expert experiments

This extension trains FastWAM directly from the raw OpenTouch HDF5 directory.
The HDF5 files are never modified. The dataset writes versioned numeric episode
caches and train-only normalization statistics under `cache_root`.

## Data contract

For an anchor frame `t0`:

- video: `t0,t4,...,t32`, shape `[3,9,H,W]`;
- action: deltas `t0->t1,...,t31->t32`, shape `[32,7]`;
- proprio: current canonical TCP `[xyz,qw,qx,qy,qz,gripper]`, shape `[1,8]`;
- force condition: `t-16,...,t0`, shape `[17,2]`;
- force target: `t1,...,t32`, shape `[32,2]`;
- full force tensor: shape `[49,2]`, ordered `[thumb,index]`.

The pressure values are relative ADC pressure, not calibrated Newtons. Actions
are canonical task-space pseudo robot actions. A robot-specific frame transform,
controller, limits, collision checking and real-robot fine-tuning are still
required before hardware deployment.

## Dataset modes

Both tasks use the same HDF5 files, file-level split, anchor history and window
indices. `OpenTouchDataset(include_force=false)` reads only RGB and hand
landmarks and never accesses `right_pressure`; its samples, caches and
normalization artifacts contain no force fields. `include_force=true` adds the
existing `[49,2]` force data. Both modes use the same prompt text.

## Train

Install the project dependencies, including `h5py`, then run the normal FastWAM
training entry point with:

```bash
python scripts/train.py task=opentouch_baseline_224_1e-4 initialization=wan_actiondit
python scripts/train.py task=opentouch_force_224_1e-4 initialization=wan_actiondit
```

Use `initialization=official_fastwam` together with
`initialization.checkpoint_path=/path/to/official_fastwam.pt` to initialize the
Video/Action MoT from an official standard checkpoint. OpenTouch canonical
proprio is always freshly initialized because the official LIBERO 8D input has
different semantics. Force parameters are also freshly initialized in the
three-expert task.

The first startup scans the raw files, builds the deterministic source-file
split, computes train-only action/proprio statistics (plus force statistics
only when enabled) and creates mode-specific episode caches. Later runs reuse
those files.

The default ForceExpert uses 10 blocks and the bridge mapping:

```text
F1->VA8, F2->VA10, F3->VA13, F4->VA15,
F5->VA17, F6->VA20, F7->VA22
```

Force blocks 0, 8 and 9 remain modality-local. Bridge gates are initialized at
zero so an official FastWAM checkpoint starts from its original Video-Action
behavior. The force objective is a masked future-only flow velocity MSE times
`model.loss.lambda_force` (default `0.25`).

## Joint action/force inference

`FastWAM.infer_action_force` accepts the current image, current 8D proprio and
17 normalized thumb/index pressure frames. It jointly denoises `[32,7]` action
and `[32,2]` future pressure without generating future video:

```python
output = model.infer_action_force(
    prompt="pick up the object",
    input_image=current_image,
    proprio=current_proprio,
    force_history=force_history_17,
    action_horizon=32,
    num_inference_steps=10,
)
```

## Terminal prediction-versus-ground-truth comparison

During training, every `eval_every` steps the OpenTouch validation path calls
`infer_action_force` (it does not generate a future video) and prints both
aggregate errors and representative future steps directly to the terminal.
The action is denormalized to `[m,m,m,rad,rad,rad,gripper]`; force is
denormalized to OpenTouch relative-pressure units in `[thumb,index]` order (it
is not calibrated in Newtons). Example log prefixes are:

```text
[eval-opentouch] ... translation_mae_m=... rotation_mae_rad=... force_mae=...
[eval-opentouch-loss] video=... action=... force=...
[eval-opentouch-compare] future_step=01 action_pred=[...] action_gt=[...] force_pred=[...] force_gt=[...]
```

Invalid action steps and invalid/padded per-finger force entries are excluded
from the aggregate metrics. The detailed comparison prints future steps
1, 8, 16 and 32 for the rank-0 validation sample. In distributed training,
the aggregate metrics are averaged across ranks. No OpenTouch comparison video
or plot is written. Online robot inference has no future ground truth, so this
comparison is available only for held-out/offline samples; real task success
must still be measured by robot rollouts.

Use `OpenTouchDataset.denormalize_action` and, in force mode,
`OpenTouchDataset.denormalize_force` with the same cache/configuration
before interpreting outputs. Execute only a short prefix and replan from new
visual, proprioceptive and tactile observations.

## Required deployment boundary

The model output must pass through a robot-specific adapter:

```text
denormalize -> canonical-to-base transform -> Cartesian/IK controller
-> velocity/workspace/collision/force limits -> robot
```

Predicted future pressure is a model signal, not a replacement for a fast
hardware overload or slip-protection loop.
# Action and proprio contract

OpenTouch retargeting uses the first valid palm pose as a fixed canonical
frame. Proprio is an absolute pose in this frame. An action chunk has shape
`[32, 7]`: dimensions `0:3` are canonical-frame translation increments in
metres, dimensions `3:6` are canonical-frame spatial rotation-vector
increments in radians, and dimension `6` is the next **absolute** gripper
opening target. The seventh dimension is not a delta.

# Training launch

Training keeps the original FastWAM launcher. The first argument must equal
the number of devices listed in `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=2 \
bash scripts/train_zero2.sh 1 task=opentouch_baseline_224_1e-4 \
  initialization=wan_actiondit

CUDA_VISIBLE_DEVICES=1,3,5,7 \
bash scripts/train_zero2.sh 4 task=opentouch_force_224_1e-4 \
  initialization=official_fastwam \
  initialization.checkpoint_path=/path/to/official_fastwam.pt \
  gradient_accumulation_steps=4
```

`batch_size` is per GPU, so the effective batch is `batch_size * world_size *
gradient_accumulation_steps`. The task configuration uses a 1,000-step Force
warmup and a 10,000-step joint pilot by default. The baseline runs 10,000 joint
steps without warmup. For a force smoke run use
`force_warmup_steps=50 joint_training_steps=150 max_steps=200`; for a baseline
smoke run use `joint_training_steps=200 max_steps=200`. Data-contract
mismatches are fatal unless the explicit diagnostic
override `allow_data_contract_mismatch=true` is supplied.
For a lower-memory single-GPU pilot, add `train_video_expert=false`; this is an
explicit choice and the trainer never changes it silently.

# Checkpoint data artifacts

Every OpenTouch checkpoint has a sibling directory with `normalization.npz`,
`data_contract.json`, and `split_manifest.json`. Their SHA-256 hashes are stored
inside the model checkpoint. Loading verifies file integrity and training
resume additionally compares the sidecars with the current dataset.

## Offline evaluation

Evaluate either mode with the same fixed-index interface:

```bash
python scripts/eval_opentouch.py \
  task=opentouch_baseline_224_1e-4 \
  checkpoint_path=/path/to/step_010000.pt

python scripts/eval_opentouch.py \
  task=opentouch_force_224_1e-4 \
  checkpoint_path=/path/to/step_011000.pt \
  split=test num_samples=null
```

The evaluator strictly checks checkpoint mode, uses `infer_action` for the
baseline and `infer_action_force` for the force task, and writes per-sample and
aggregate action metrics to JSON. Force metrics are added only for the
three-expert task.
