# GR00T-N1.5 RoboCasa Eval -- Tips & Pitfalls

> Hard-won lessons from getting `PnPCounterToSink` (canonical: `PickPlaceCounterToSink`) from 0% to 60% on the multitask checkpoint. Each rule below has a real bug attached.

The trajectory:
- **0%** -- robot moved but never grasped cleanly. Caused by forcing `camera_widths=128`.
- **20%** (1/5) -- motion still rough. Caused by `replay-chunk=8` (a GR1-tabletop convention).
- **60%** (3/5) -- after also cleaning up `gym.make` kwargs to canonical defaults.

If you are about to run an eval, read the TL;DR + checklist first; treat the five pitfalls as the failure modes most likely to bite you again.

---

## TL;DR -- the canonical pipeline

- Match `robocasa-benchmark/Isaac-GR00T/scripts/run_eval.py` (synced into `docs/reference_canonical/run_eval.py`) exactly.
- Build the env with `gym.make("robocasa/<Task>", split=..., seed=..., enable_render=True)` -- nothing else. The wrapper picks its own defaults.
  - `RoboCasaGymEnv` lives at `robocasa/wrappers/gym_wrapper.py:130`.
  - It auto-registers as `robocasa/<EnvName>` at `gym_wrapper.py:375-378`, so `gym.make("robocasa/...")` already returns the wrapped env.
- `n_action_steps = 16` (full-chunk replay).
  - Default lives in `MultiStepConfig` (`Isaac-GR00T/gr00t/eval/simulation.py:65`) and in the canonical script's CLI default (`docs/reference_canonical/run_eval.py:159`, value `16`).
  - The chunk is consumed step-by-step by `MultiStepWrapper.step` (`Isaac-GR00T/gr00t/eval/wrappers/multistep_wrapper.py:200-241`), `for step in range(self.n_action_steps)` at line 207 calling `super().step(act)` per chunk-step.
- 256x256 cameras, fps=20, RGB. Source of truth: the LeRobot training data on HuggingFace at `huggingface.co/datasets/robocasa/robocasa365_lerobot` (`pnp/lerobot/meta/info.json`: `fps: 20`, image features all `shape: [256, 256, 3]`, `video.codec: h264`, `video.pix_fmt: yuv420p`).
- `split=pretrain` for the multitask checkpoint (kitchens it saw during training); `target` is held-out.

---

## The five pitfalls

Each block: symptom / root cause / fix / how to verify.

### 1. Forcing `camera_widths=128` / `camera_heights=128`

- **Symptom:** 0% SR, robot wanders, never grasps cleanly.
- **Root cause:** Training data is 256x256 (info.json features above), our eval was passing 128x128 -- visually OOD for the vision tower.
- **Fix:** Do **not** pass `camera_widths` / `camera_heights` to `gym.make`. The wrapper's `PandaOmronKeyConverter.get_camera_config` returns `camera_widths, camera_heights = 256, 256` (`gym_wrapper.py:33`), and `RoboCasaGymEnv.__init__` falls back to those defaults at `gym_wrapper.py:149-152`.
- **Verify:**
  ```bash
  python3 -c "import gymnasium as gym, robocasa; \
      env=gym.make('robocasa/PickPlaceCounterToSink', split='pretrain', seed=0, enable_render=True); \
      print(env.observation_space)"
  ```
  Each `video.*` Box should have shape `(256, 256, 3)`.

### 2. Using `--replay-chunk 8` instead of 16

- **Symptom:** 1/5 = 20%, motion visibly choppy, EE skips between chunk boundaries.
- **Root cause:** 8 is the GR1-tabletop convention, not RoboCasa. The canonical RoboCasa script defaults to 16 (`docs/reference_canonical/run_eval.py:159`), the canonical `MultiStepConfig.n_action_steps` default is 16 (`gr00t/eval/simulation.py:65`), and the consumer loop replays the full chunk per outer `step` (`multistep_wrapper.py:207`).
- **Fix:** Pass `--replay-chunk 16` or omit the flag (our `examples/run_groot_eval.py` already defaults to 16; see `--replay-chunk` arg at line 567-572).
- **Verify:**
  - Server `/health` shows `n_action_steps=16` (server constant `DEFAULT_N_ACTION_STEPS = 16` in `serve_groot.py:104`).
  - Eval log line: `replay_chunk = 16 (canonical full-chunk replay)` (`run_groot_eval.py:444-446`).

### 3. Passing `generative_textures="100p"` / `randomize_cameras=True` to `gym.make`

- **Symptom:** scene looks visually plausible, but minor distribution shift from training.
- **Root cause:** the canonical `_create_single_env` in `gr00t/eval/simulation.py:179-207` and `run_client` in `run_eval.py:55-87` only pass `enable_render=True`. The original collection HDF5 (`demo_gentex_im128_randcams.hdf5`) is misleading -- it describes raw collection, but the LeRobot conversion does not preserve those flags at eval time.
- **Fix:** Build the env exactly as the canonical script does. Our `run_groot_eval.py:469-474` already does so (`gym.make("robocasa/<canon>", split=..., seed=..., enable_render=True)`).
- **Verify:** Diff your `gym.make` call against `run_eval.py:78-87` (it just constructs `SimulationConfig` whose only env-construction kwarg is `enable_render=True`).

### 4. Mistaking the env's input format

- **Truth:** `RoboCasaGymEnv.step` (`gym_wrapper.py:313`) takes a **dict**, **per-step**, with these sub-keys (the action-space layout from `PandaOmronKeyConverter.deduce_action_space` at `gym_wrapper.py:69-89` and `unmap_action` at `gym_wrapper.py:108-127`):
  - `action.gripper_close` -- shape (1,)
  - `action.end_effector_position` -- shape (3,)
  - `action.end_effector_rotation` -- shape (3,)
  - `action.base_motion` -- shape (4,)
  - `action.control_mode` -- shape (1,)
  - The `MultiStepWrapper` is what handles chunking (`multistep_wrapper.py:200-241`); the inner `RoboCasaGymEnv.step` is per-step only.
- **Gotcha 1 -- gripper binarization:** `unmap_action` (`gym_wrapper.py:108-127`) thresholds at 0.5:
  - `robot0_right_gripper = -1.0 if action.gripper_close < 0.5 else 1.0` (line 111-113)
  - `robot0_base_mode    = -1.0 if action.control_mode  < 0.5 else 1.0` (line 123-125)
  Do **not** pre-binarize on the client. Our `discretize_inplace` in `examples/run_groot_eval.py:209-225` is correctly a no-op.
- **Gotcha 2 -- base_motion zeroing for atomic / PnP tasks:** training demos for these spawned the robot at the target fixture (no base translation), so leaving the model's small base_motion drift active makes the robot wander away. Our client zeros it by default (`run_groot_eval.py:365-366`, controlled by `--allow-base-motion`).
- **Verify:** Inspect parquet ground truth (see "Where to look when stuck"). Real values match: `action.base_motion` and `action.control_mode` are constant for PnP, `action.gripper_close` is binary +/-1.

### 5. Misreading state quaternion convention

- **Truth:** robosuite quaternions are **xyzw** (scalar **last**). Verified directly against `pnp/lerobot/data/chunk-000/episode_000000.parquet` from the LeRobot dataset on HuggingFace (`huggingface.co/datasets/robocasa/robocasa365_lerobot`):
  - `state.base_rotation` (idx 3:7) at t=0 is `[0., 0., 1., 0.]` -- 180 deg about Z (xyzw).
  - `state.end_effector_rotation_relative` (idx 10:14) at t=0 is `[-0.9865, 0.0096, -0.1627, 0.0160]`, norm = 1.0 -- approximately 180 deg about X (EE pointing down). Scalar component is the trailing 0.0160, not the leading -0.9865.
- The state dimension layout is also confirmed by `meta/modality.json`: `base_position[0:3]`, `base_rotation[3:7]`, `end_effector_position_relative[7:10]`, `end_effector_rotation_relative[10:14]`, `gripper_qpos[14:16]`. Action layout: `base_motion[0:4]`, `control_mode[4:5]`, `end_effector_position[5:8]`, `end_effector_rotation[8:11]`, `gripper_close[11:12]`.
- The metadata declares `rotation_type: 'quaternion'` for `end_effector_rotation_relative` (`groot_docker_n1.5/checkpoint/experiment_cfg/metadata.json` -> `state.end_effector_rotation_relative`). The server's `StateActionTransform` converts quat -> rotation_6d for the model; verify in `serve_groot.py` that you're handing it the raw 4-element xyzw and not pre-converting.

---

## What the model expects vs what we used to send

| Slot | Training value (verified from parquet / info.json) | What we wrongly sent | Fixed to |
|---|---|---|---|
| Cameras | 256x256 RGB uint8 (info.json) | 128x128 | 256x256 (wrapper default) |
| `state.base_rotation` | xyzw quat, e.g. `[0,0,1,0]` | xyzw (correct) | unchanged |
| `state.end_effector_rotation_relative` | xyzw quat, norm = 1 | xyzw (correct) | unchanged |
| `action.base_motion` | all 0 for PnP | client zeros it | unchanged (default `--no-base-motion`) |
| `action.control_mode` | constant `-1` for PnP | passed raw, env binarizes | unchanged |
| `action.gripper_close` | binary in `{-1, +1}` | passed raw, env binarizes at 0.5 | unchanged |
| `n_action_steps` replay | 16 (canonical) | 8 | 16 |

---

## Pre-eval checklist

Copy this and tick before each run.

- [ ] Server is up. `curl http://localhost:8500/health` returns:
  - `status: ok`
  - `n_action_steps: 16`
  - `embodiment_tag: new_embodiment`
  - `video_keys` containing `robot0_agentview_left`, `robot0_agentview_right`, `robot0_eye_in_hand`
- [ ] Eval client is post-fix:
  - No `camera_widths` / `camera_heights` kwargs to `gym.make`.
  - No `generative_textures` / `randomize_cameras` kwargs to `gym.make`.
  - `--replay-chunk` is unset or 16.
- [ ] `--num-rollouts 5` minimum (single rollout is noisy).
- [ ] `--output-dir test_outputs/<descriptive_name>/` to avoid clobbering previous runs.
- [ ] `--split pretrain` for multitask checkpoint (use `target` only when explicitly evaluating held-out kitchens).
- [ ] After done, stop the server: `(cd groot_docker_n1.5 && ./run.sh --stop)`.
- [ ] Note `mean_lat` in summary.json -- ~95-110 ms per `/act` call on A6000 is healthy. Sustained higher means CPU contention or you forgot to warm the model.

---

## When eval fails to match paper numbers

Tier the investigation:
1. Are the five pitfalls above all clean? (Most likely cause.)
2. Is the **checkpoint** the right one? Multitask vs target-only differ a lot. See `GR00T_CHECKPOINTS.md` in this repo.
3. Is `split=pretrain` (kitchens seen during training) or `target` (held out)? Default for the multitask checkpoint is `pretrain`.
4. Is the **task class actually registered**? Note the `PnP*` -> `PickPlace*` aliasing in `examples/run_groot_eval.py:102-115` (TASK_ALIASES). `gym.make("robocasa/PnPCounterToSink")` will fail; you need `PickPlaceCounterToSink`.

---

## Reference: how the canonical pipeline differs from ours

| Component | Ours | Canonical |
|---|---|---|
| Inference transport | HTTP (FastAPI), port 8500 -- `groot_docker_n1.5/serve_groot.py` | zmq via `RobotInferenceServer`, port 5555 -- `Isaac-GR00T/gr00t/eval/robot.py` |
| Eval driver | custom client loop -- `examples/run_groot_eval.py` | `SimulationInferenceClient` + `MultiStepWrapper` -- `Isaac-GR00T/gr00t/eval/simulation.py` |
| Per-task horizon | `lookup_task_horizon` -> `robocasa.utils.dataset_registry_utils.get_task_horizon` (`run_groot_eval.py:118-127`) | same, via `get_task_horizon` (`run_eval.py:23, 76`) |

Both implement the same dot-namespace contract (`observation.images.<cam>`, `observation.state.<key>`, `action.<key>`) over different transports. The semantic equivalence is what makes our HTTP setup work despite not being literally byte-for-byte the same as zmq.

---

## Where to look when you're stuck

- **Diff parquet vs server output.** If you have the LeRobot training data downloaded locally (~10 GB from `huggingface.co/datasets/robocasa/robocasa365_lerobot`), inspect the canonical training values directly. Substitute `<your-lerobot-path>` for your download location:
  ```bash
  docker run --rm -v <your-lerobot-path>:/data:ro groot-server:latest python3 -c "
  import pyarrow.parquet as pq, numpy as np
  t = pq.read_table('/data/pnp/lerobot/data/chunk-000/episode_000000.parquet')
  states = np.array(t['observation.state'].to_pylist())
  acts   = np.array(t['action'].to_pylist())
  print('shapes:', states.shape, acts.shape)
  print('base_rotation[0]:', states[0, 3:7])
  print('eef_rot_rel[0]:',   states[0, 10:14], 'norm =', np.linalg.norm(states[0, 10:14]))
  print('base_motion uniq:', np.unique(acts[:, 0:4]))
  print('control_mode uniq:', np.unique(acts[:, 4]))
  print('gripper_close uniq:', np.unique(acts[:, 11]))
  "
  ```
- **Diff env wrapper output.** Open a shell into the robocasa container, run `env.reset()` once, and print `obs.keys()` plus `obs['video.robot0_agentview_left'].shape` and `obs['state.end_effector_rotation_relative']`. Compare against the parquet snapshot above.
- **Diff our HTTP `/act` response vs canonical zmq response.** Send the same image+state both ways and compare the action chunk element-by-element. This catches subtle transform-order mistakes.

---

## File map (repo-relative paths)

| File | Why you'd open it |
|---|---|
| `robocasa/wrappers/gym_wrapper.py` | `RoboCasaGymEnv` (line 130), `PandaOmronKeyConverter` (line 20), `unmap_action` binarization (line 108-127), gym registration (line 375-378). |
| `examples/run_groot_eval.py` | Our eval client, current `gym.make` call (line 469-474), task aliases (line 102-115). |
| `groot_docker_n1.5/Isaac-GR00T/gr00t/eval/simulation.py` | `SimulationInferenceClient`, `MultiStepConfig` (line 60-66, default `n_action_steps=16`). |
| `groot_docker_n1.5/Isaac-GR00T/gr00t/eval/wrappers/multistep_wrapper.py` | `MultiStepWrapper.step` chunk consumer (line 200-241). |
| `groot_docker_n1.5/Isaac-GR00T/scripts/eval_policy.py` | Upstream NVIDIA offline-replay MSE eval (not env rollout). |
| `docs/reference_canonical/run_eval.py` | Canonical robocasa-benchmark/Isaac-GR00T env-rollout eval (mirrored from upstream main). CLI default `n_action_steps=16` at line 159. |
| `docs/reference_canonical/eval_policy.py` | Canonical offline replay-MSE eval (sanity check that the model loads & decodes correctly). |
| `docs/reference_canonical/inference_service.py` | Canonical zmq-based reference server — instructive for the dot-namespace API the model expects. |
| `pnp/lerobot/meta/info.json` (HF: `robocasa/robocasa365_lerobot`) | Training data: 256x256, fps=20, h264 yuv420p. Download separately (~10 GB) if you want it locally. |
| `pnp/lerobot/meta/modality.json` (HF: `robocasa/robocasa365_lerobot`) | State and action sub-key layouts. |
| `pnp/lerobot/data/chunk-000/episode_000000.parquet` (HF: `robocasa/robocasa365_lerobot`) | Raw training values; ground-truth for any "what does the model expect" question. |
| `groot_docker_n1.5/serve_groot.py` | Our HTTP server (FastAPI :8500). `DEFAULT_N_ACTION_STEPS = 16` at line 104. |
| `groot_docker_n1.5/checkpoint/experiment_cfg/metadata.json` | Checkpoint embodiment statistics + min_max bounds. |
