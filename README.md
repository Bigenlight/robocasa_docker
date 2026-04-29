<h1 align="center">RoboCasa</h1>
<img src="docs/images/readme.webp" width="100%" />

**RoboCasa** is a large-scale simulation framework for training generally capable robots to perform everyday tasks. It was [originally released](https://robocasa.ai/assets/robocasa_rss24.pdf) in 2024 by UT Austin researchers. The latest iteration, **RoboCasa365**, builds upon the original release with significant new functionalities to support large-scale training and benchmarking in sim. Four pillars underlie RoboCasa365:
- **Diverse tasks**: 365 tasks created with the guidance of large language models
- **Diverse assets**: including 2,500+ kitchen scenes and 3,200+ 3D objects
- **High-quality demonstrations**: including 600+ hours of human demonstrations in addition to 1,600+ hours of robot datasets created with automated trajectory tools
- **Benchmarking support**: popular policy learning methods including Diffusion Policy, pi, and GR00T, plus user-submitted models on the [leaderboard](https://robocasa.ai/leaderboard.html)

[**[Home page]**](https://robocasa.ai) &ensp; [**[Documentation]**](https://robocasa.ai/docs/introduction/overview.html) &ensp; [**[RoboCasa365 Paper]**](https://robocasa.ai/assets/robocasa365_iclr26.pdf) &ensp; [**[Original RoboCasa Paper]**](https://robocasa.ai/assets/robocasa_rss24.pdf) &ensp; [**[Leaderboard]**](https://robocasa.ai/leaderboard.html)

---

## What this fork adds

This fork wraps upstream RoboCasa365 in a Docker workflow so a fresh machine can run a headless rollout (sim + render → mp4) and a closed-loop GR00T-N1.5 evaluation **without installing anything Python on the host**. Two images cooperate:

| Image | Repo | Role |
|---|---|---|
| `bigenlight/robocasa-eval` | this repo (`robocasa_docker/`) | Sim + headless render + eval client |
| `bigenlight/groot-server` | `groot_docker_n1.5/` (subdir of this repo) | GR00T-N1.5 HTTP policy server (port 8500) |

Both images bake only Python dependencies; the simulator source, the robosuite source, the kitchen assets, and the GR00T checkpoint all live on the host and are bind-mounted at runtime. That means image rebuilds, container removals, and version bumps never destroy your work.

For the design rationale see [`DOCKER.md`](DOCKER.md). For the GR00T checkpoint catalog see [`GR00T_CHECKPOINTS.md`](GR00T_CHECKPOINTS.md). For the HTTP contract between the two images see [`../VLA_COMMUNICATION_PROTOCOL.md`](../VLA_COMMUNICATION_PROTOCOL.md).

---

## Prerequisites

| Requirement | Detail |
|---|---|
| Docker | 27.x tested |
| NVIDIA Container Toolkit | `nvidia-container-toolkit` registered with Docker (`docker info \| grep -i nvidia` non-empty) |
| GPU | CUDA-capable, Ampere or newer; team runs RTX A6000 |
| Disk (sim) | ~9 GB image + ~23 GB kitchen assets |
| Disk (GR00T) | ~12 GB image + ~7.6 GB checkpoint (without optimizer state) |
| Network | HuggingFace access for asset + checkpoint download |
| Host Python | **None.** Do not `pip install` anything on the host. |

---

## Quick start (sim only, ~15-25 min depending on bandwidth)

```sh
git clone <this fork> robocasa_docker
cd robocasa_docker
git clone https://github.com/ARISE-Initiative/robosuite ./robosuite

# Build locally OR pull pre-built. Either path works; the second is faster.
./run.sh --build
# alternative:
#   docker pull bigenlight/robocasa-eval:latest
#   docker tag bigenlight/robocasa-eval:latest robocasa-eval:latest   # match run.sh's default
#   # ...or set ROBOCASA_IMAGE=bigenlight/robocasa-eval:latest in your shell

./run.sh --download-assets       # ~10 min, ~23 GB onto host (paper says ~10 GB compressed; expanded is larger)
./run.sh --smoke-test            # writes test_outputs/smoke_agentview.png + smoke_PickPlaceCounterToSink.mp4
```

If `--smoke-test` produces `test_outputs/smoke_agentview.png` and `test_outputs/smoke_PickPlaceCounterToSink.mp4`, the simulator is healthy. If EGL fails, `run.sh` retries automatically with `MUJOCO_GL=osmesa`.

---

## `./run.sh` modes

| Mode | What it does | Writes to `test_outputs/` |
|---|---|---|
| `--build` | Build `bigenlight/robocasa-eval:latest` locally | — |
| `--download-assets` | One-time ~23 GB asset download (resumable; pipes `yes y` through the prompt) | populates `robocasa/models/assets/` on host |
| `--smoke-test` | 6-step in-container check: imports → render probe → `gym.make` + `reset` → PNG → 10 sim steps → `run_random_rollouts` | `smoke_agentview.png`, `smoke_PickPlaceCounterToSink.mp4` |
| `--rollout <Task> [--num N --steps N --seed N]` | Random-action rollout against `robocasa/<Task>` | `<Task>_seed<N>.mp4` |
| `--groot-eval <Task> [--steps N --seed N]` | Closed-loop GR00T eval, single seed. Requires `bigenlight/groot-server` reachable on `$GROOT_SERVER` (default `http://localhost:8500`) | `groot_<Task>_seed<N>.mp4` |
| `--canonical-eval <Task> [--num-rollouts N --seed-base N --steps N --replay-chunk N]` | Multi-rollout canonical GR00T eval (matches `robocasa-benchmark/Isaac-GR00T/scripts/run_eval.py`). Writes per-trial mp4s + `groot_<Task>_summary.json` | `groot_<Task>_seed<N>_success{0\|1}.mp4` + summary.json |
| `--shell` | Interactive bash with `PYTHONPATH=/workspace/robocasa:/workspace/robosuite` | — |

Override the image with `ROBOCASA_IMAGE=...` / the GR00T URL with `GROOT_SERVER=...`.

---

## Two-container GR00T eval (closed loop)

The GR00T-N1.5 policy lives in `groot_docker_n1.5/` (a subdirectory of this repo, bundled in-tree). Both containers share `--network host`, so the eval client reaches the server at `http://localhost:8500`.

<!-- Note: the companion server repo used to be a separate clone; it is now bundled at groot_docker_n1.5/, so no separate clone step is needed. -->

### Choose a checkpoint

Pick a recipe based on what you want to demonstrate. The recipe name is the argument to `groot_docker_n1.5/swap_ckpt.sh <recipe>`; the SR figures are RoboCasa365 paper averages (see [`GR00T_CHECKPOINTS.md`](GR00T_CHECKPOINTS.md) for the full catalog and per-task breakdown).

| Goal | Recipe | Paper SR |
|---|---|---|
| Default / out-of-box pretraining baseline | `multitask` | composite-seen 9.6% |
| Best closed-loop on composite-seen tasks | `target_pt_composite_seen` | composite-seen 40.6% |
| Best on atomic tasks | `target_pt_atomic` | atomic 68.5% |
| Generalize to composite-unseen tasks | `target_pt_composite_unseen` | composite-unseen 42.1% |
| Foundation-only (no post-training) | `pretraining` | composite-seen 0% / atomic 41.9% |

**Step A — bring up the GR00T server:**

```sh
# First time: build (or pull) the image
cd groot_docker_n1.5
./run.sh --build                                                    # ~12 GB image
# alternative:  docker pull bigenlight/groot-server:n1.5

# Clone Isaac-GR00T inside the subdir at the n1.5-release tag
git clone https://github.com/NVIDIA/Isaac-GR00T -b n1.5-release ./Isaac-GR00T

# Download the chosen checkpoint. --download-ckpt fetches the default
# multitask recipe; for any other recipe see groot_docker_n1.5/README.md §6.
./run.sh --download-ckpt                                            # ~7.6 GB without optimizer.pt

# (Optional) Use a different recipe — e.g. target_pt_composite_seen (40% on
# PrepareCoffee). Two steps: (1) download the recipe (it's NOT downloaded by
# --download-ckpt, which is multitask-only), then (2) repoint symlinks. See
# groot_docker_n1.5/README.md §5-§6 for the recipe table and full workflow.
RECIPE_PATH="gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000"
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp/groot-home \
    -v "$PWD/checkpoint:/groot/checkpoint" \
    -v "$HOME/.cache/huggingface:/tmp/groot-home/.cache/huggingface" \
    groot-server:latest \
    bash -c "mkdir -p /tmp/groot-home && \
      huggingface-cli download robocasa/robocasa365_checkpoints \
        --include '${RECIPE_PATH}/*' --exclude '*optimizer*' --local-dir /groot/checkpoint"
./swap_ckpt.sh target_pt_composite_seen                             # repoint symlinks; pass --list to see all 12 recipes

# Start the server in the background
./run.sh --serve-bg                                                 # detached, named groot-server
docker logs -f groot-server                                         # wait for "policy ready" (Uvicorn binds before model load)
# alternatively poll: until [ "$(curl -fsS http://127.0.0.1:8500/health 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin)["status"])')" = ok ]; do sleep 5; done
```

`./run.sh --smoke` runs a dummy server end-to-end without the real model — useful for verifying the protocol on a machine without a GPU.

**Step B — run the eval client (back in `robocasa_docker/`):**

```sh
cd ..   # back to robocasa_docker (the repo root)

# Multitask checkpoint → split=pretrain (Sec 4.1 / Table 1 of paper)
./run.sh --canonical-eval PickPlaceCounterToSink --num-rollouts 5 --seed-base 0 --split pretrain

# Any target_only / target_posttraining / pretraining checkpoint → split=target (Sec 4.2 / Table 2)
./run.sh --canonical-eval PrepareCoffee --num-rollouts 5 --seed-base 0 --split target

# Single-seed quick check:
./run.sh --groot-eval PickPlaceCounterToSink --steps 60 --seed 0
```

What success looks like (real `groot_PickPlaceCounterToSink_summary.json` from a 5-rollout run):

```
GR00T server: http://localhost:8500
  ready: model=gr00t_n1.5 action_keys=['action.base_motion', ...] n_action_steps=16 ...
Creating env robocasa/PickPlaceCounterToSink (split=pretrain)  num_rollouts=5  base_seed=0
  replay_chunk = 16 (canonical full-chunk replay)
  ...
DONE  task=PickPlaceCounterToSink  success=3/5  rate=0.60  mean_lat=104.1ms
  ✓ seed=   0  steps= 162  steps_to_success= 161  -> groot_PickPlaceCounterToSink_seed0_success1.mp4
  ✗ seed=   1  steps= 400  steps_to_success=  -1  -> groot_PickPlaceCounterToSink_seed1_success0.mp4
  ✓ seed=   2  steps= 217  steps_to_success= 216  -> groot_PickPlaceCounterToSink_seed2_success1.mp4
  ✗ seed=   3  steps= 400  steps_to_success=  -1  -> groot_PickPlaceCounterToSink_seed3_success0.mp4
  ✓ seed=   4  steps= 180  steps_to_success= 179  -> groot_PickPlaceCounterToSink_seed4_success1.mp4
```

Per-trial mean latencies in `groot_PickPlaceCounterToSink_summary.json` were 119.1, 98.9, 104.4, 97.7, 100.2 ms (avg 104.1ms). The first trial pays a model warmup cost; subsequent trials sit ~95–105 ms.

Latency around 100 ms / chunk on RTX A6000. Action chunks are 16 steps; the client replays the full 16 before re-querying — this matches the canonical RoboCasa convention (`MultiStepConfig.n_action_steps=16` in `robocasa-benchmark/Isaac-GR00T/scripts/run_eval.py`). The 8-step replay used by NVIDIA's GR1-tabletop-tasks reference repo is **not** the RoboCasa default. Override with `--replay-chunk` if you need to compare. Per-task expected SR for the multitask checkpoint is in [`GR00T_CHECKPOINTS.md`](GR00T_CHECKPOINTS.md) §10. When the run is done, stop the server with `(cd groot_docker_n1.5 && ./run.sh --stop)`.

For checkpoint variants (atomic / composite-seen / composite-unseen / lifelong) and per-task success rates, see [`GR00T_CHECKPOINTS.md`](GR00T_CHECKPOINTS.md).

### Verified results (GR00T-N1.5)

| Task | Checkpoint | SR | Notes |
|---|---|---|---|
| PnPCounterToSink (atomic) | `multitask` | 3/5 = 60% | seeds 0,2,4 ✓; seeds 1,3 ✗; mean_lat 104ms |
| PrepareCoffee (composite-seen) | `target_pt_composite_seen` | 2/5 = 40% | seeds 1 (step 585) and 4 (step 1174) ✓; per-trial latencies 105/103/107/102/103 ms (avg ~104). Paper reports PrepareCoffee specific = 13%, composite-seen average = 40.6%. summary.json at `test_outputs/eval_composite_pc/`. |
| PrepareCoffee (composite-seen) | `multitask` | 0/5 = 0% | All seeds time out at 1200 steps. Consistent with the paper's 0% PrepareCoffee SR for the pretraining-only multitask checkpoint. summary.json at `test_outputs/eval_canonical_pc/`. |

Reference: the RoboCasa365 paper reports a 43% atomic average for the multitask pretraining-only checkpoint, so 60% on PnPCounterToSink is within (above) the expected range. Common pitfalls (image size, kwargs, replay chunk) and how we got here are in [`GR00T_EVAL_TIPS.md`](GR00T_EVAL_TIPS.md).

### Where checkpoints live

`groot_docker_n1.5/checkpoint/` is a flat directory of 5 symlinks at its root (`config.json`, `experiment_cfg`, `model-00001-of-00002.safetensors`, `model-00002-of-00002.safetensors`, `model.safetensors.index.json`) that point into one of the `gr00t_n1-5/.../checkpoint-N/` recipe subdirs. Multiple recipes can coexist on disk; `swap_ckpt.sh` just repoints those 5 symlinks. The server bind-mounts `checkpoint/` read-only and resolves the symlinks at start, so a swap takes effect on the next `./run.sh --serve-bg`.

```
groot_docker_n1.5/checkpoint/
├── config.json                       -> gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000/config.json
├── experiment_cfg                    -> .../composite_seen/checkpoint-60000/experiment_cfg
├── model-00001-of-00002.safetensors  -> .../composite_seen/checkpoint-60000/model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors  -> .../composite_seen/checkpoint-60000/model-00002-of-00002.safetensors
├── model.safetensors.index.json      -> .../composite_seen/checkpoint-60000/model.safetensors.index.json
└── gr00t_n1-5/
    ├── multitask_learning/checkpoint-120000/             # downloaded by --download-ckpt
    └── foundation_model_learning/target_posttraining/    # populated per swap_ckpt.sh hint
        └── composite_seen/checkpoint-60000/
```

See `groot_docker_n1.5/README.md` §4 (file layout) and §5 (swap workflow) for the full recipe table and download commands.

---

## Host ↔ container layout

```
HOST robocasa_docker/                              CONTAINER (bigenlight/robocasa-eval)
├── Dockerfile  run.sh  test_smoke.py     bind→    /workspace/robocasa            (rw)
├── examples/                                      /workspace/robocasa/examples
├── robocasa/                  (package source)    /workspace/robocasa/robocasa
│   └── models/assets/         (~23 GB)            written by --download-assets
├── robosuite/                 (host-cloned)  →    /workspace/robosuite           (rw)
└── test_outputs/              (mp4 / png)    →    /workspace/robocasa/test_outputs
                                                   PYTHONPATH=/workspace/robocasa:/workspace/robosuite
                                                   MUJOCO_GL=egl  (osmesa fallback)
                                                   --user $(id -u):$(id -g)
                                                   HOME=/tmp/robocasa-home

HOST robocasa_docker/groot_docker_n1.5/            CONTAINER (bigenlight/groot-server, --network host)
├── Dockerfile  run.sh  swap_ckpt.sh                (host scripts; not mounted)
├── serve_groot.py             (read-only mount)   /groot/serve_groot.py
├── Isaac-GR00T/               (host-cloned)       /groot/Isaac-GR00T
└── checkpoint/                (--download-ckpt)   /groot/checkpoint               (ro)
    ├── (5 root symlinks)         resolved at start by the server
    └── gr00t_n1-5/.../checkpoint-N/   one or more recipe dirs
                                                   exposes  http://0.0.0.0:8500
```

The container always runs as your host UID/GID, so files written into the bind mounts (downloaded zips, mp4s, `macros_private.py`) are owned by you, not root.

---

## What's NOT in the image

| Not baked | Why | How it gets onto the host |
|---|---|---|
| `robocasa/models/assets/` (~23 GB) | size; survives image rebuilds | `./run.sh --download-assets` |
| `robosuite/` | upstream pinned to master; lets you patch in-place | `git clone https://github.com/ARISE-Initiative/robosuite ./robosuite` |
| GR00T checkpoint (~7.6 GB) | size; checkpoint variants swap often | `cd groot_docker_n1.5 && ./run.sh --download-ckpt` |
| `Isaac-GR00T/` source | upstream API moves; lets you pin a commit | `git clone https://github.com/NVIDIA/Isaac-GR00T -b n1.5-release groot_docker_n1.5/Isaac-GR00T` |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: robosuite` | `./robosuite/` not cloned | `git clone https://github.com/ARISE-Initiative/robosuite ./robosuite` |
| `FileNotFoundError: ... assets ... missing` | `--download-assets` not run | `./run.sh --download-assets` (~23 GB, one-time) |
| Empty `test_outputs/` after smoke test | EGL probe failed silently AND OSMesa fallback failed | check `nvidia-smi`; verify `docker info` lists the `nvidia` runtime |
| Files in `test_outputs/` owned by root | Custom `docker run` missing `--user` | Use `./run.sh`, or mirror its `--user $(id -u):$(id -g) -e HOME=/tmp/robocasa-home` |
| `AttributeError: 'OrderEnforcing' object has no attribute 'sim'` | gym 1.x removed `Wrapper.__getattr__` | Image pins `gymnasium==0.29.1`; don't override |
| `download_kitchen_assets.py` hangs at `Proceed? (y/n)` | stdin not piped through | Use `./run.sh --download-assets` (it pipes `yes y \|`); don't invoke the script directly |
| `--groot-eval`: `server at http://localhost:8500 not ready` | GR00T container not running | `cd groot_docker_n1.5 && ./run.sh --serve-bg && docker logs -f groot-server` |
| `bind: address already in use` on port 8500 | another GR00T server alive | `(cd groot_docker_n1.5 && ./run.sh --stop)` or set `GROOT_PORT=8501 GROOT_SERVER=http://localhost:8501 ./run.sh --groot-eval ...` |
| Pip can't satisfy `numpy==2.2.5` in a downstream image | transitive dep (lerobot / tianshou) pinning `numpy<2` | Install those `--no-deps` (this image already does) |

---

## Available tasks

RoboCasa365 ships **365 tasks** in the paper; the gym registry exposes 396 entries (some are variants — e.g. `OpenSingleDoor` vs `OpenDoor`, fixture-typed variants). List them all from inside the container:

```sh
./run.sh --shell
# inside:
python -c "import robocasa, gymnasium as gym; print('\n'.join(sorted(s for s in gym.envs.registry if s.startswith('robocasa/'))))"
```

Representative names (all confirmed headless-renderable):

| Family | Tasks |
|---|---|
| Pick-and-place | `PickPlaceCounterToSink`, `PickPlaceCounterToStove`, `PickPlaceCounterToCabinet`, `PickPlaceCounterToMicrowave`, `PickPlaceMicrowaveToCounter`, `PickPlaceSinkToCounter` |
| Atomic | `OpenCabinet`, `CloseDoor`, `OpenDrawer`, `TurnOnMicrowave`, `TurnOffStove`, `TurnOnSinkFaucet` |
| Composite (coffee / dishwasher) | `PrepareCoffee`, `LoadDishwasher`, `KettleBoiling`, `StackBowlsCabinet` |

`PrepareCoffee` is the canonical composite-seen GR00T demo (paper-reported success 13%, see `GR00T_CHECKPOINTS.md` §10).

---

## Reference code map

When debugging an eval, open these files in this order — they are the ground truth.

### Canonical pipeline (upstream — read these to understand "how it should be")

| Path | Purpose |
|---|---|
| [`docs/reference_canonical/run_eval.py`](docs/reference_canonical/run_eval.py) | The `robocasa-benchmark/Isaac-GR00T/scripts/run_eval.py` (downloaded from upstream main). Single-file driver: starts a zmq `RobotInferenceServer`, then loops `SimulationInferenceClient.run_simulation(SimulationConfig(...))` over each task in the task set. **Default `n_action_steps=16`, `n_episodes=50`, `n_envs=5`.** |
| [`docs/reference_canonical/eval_policy.py`](docs/reference_canonical/eval_policy.py) | Offline replay-MSE eval (compares model actions to dataset ground truth). Useful sanity check for "is the model loading correctly", separate from env rollout. |
| [`docs/reference_canonical/inference_service.py`](docs/reference_canonical/inference_service.py) | Reference HTTP/zmq server for the canonical contract — instructive for understanding the dot-namespace API the model expects. |
| `groot_docker_n1.5/Isaac-GR00T/gr00t/eval/simulation.py` | `SimulationInferenceClient`, `SimulationConfig`, `MultiStepConfig` (default `n_action_steps=16` at line 65). The `run_simulation` loop is the canonical env-rollout body. |
| `groot_docker_n1.5/Isaac-GR00T/gr00t/eval/wrappers/multistep_wrapper.py` | `MultiStepWrapper.step(action_dict)` consumes the chunk: `for step in range(self.n_action_steps): super().step(act_per_step)`. Read this before changing replay logic. |
| `groot_docker_n1.5/Isaac-GR00T/scripts/eval_policy.py` | NVIDIA n1.5-release upstream version of `eval_policy.py` (checked-out in the GR00T container source tree). |

### Local pipeline (this repo — read these to understand "what we actually run")

| Path | Purpose |
|---|---|
| [`examples/run_groot_eval.py`](examples/run_groot_eval.py) | Our HTTP-based eval client (the script `run.sh --canonical-eval` invokes). Argparse at line 558. Action chunk consumption + base_motion zeroing in `run_one_trial`. |
| [`robocasa/wrappers/gym_wrapper.py`](robocasa/wrappers/gym_wrapper.py) | The `RoboCasaGymEnv` wrapper. Default cameras 256×256 (line 33), `step()` expects a dict with `action.gripper_close, action.end_effector_position, ...` (line 313), `unmap_action` binarizes gripper at threshold 0.5 (line 108). Every `gym.make("robocasa/<Task>")` returns this. |
| `groot_docker_n1.5/serve_groot.py` | Our FastAPI server on port 8500 (HTTP, not zmq — same dot-namespace contract). `_build_robocasa_modality()` mirrors `PandaOmronDataConfig.transform()`. |
| [`run.sh`](run.sh) | `--groot-eval` (single seed) → forwards `--num-steps`, `--seed`. `--canonical-eval` (multi-rollout) → forwards `--num-rollouts`, `--seed-base`, `--max-steps`, `--replay-chunk`. |

### Training data (verify what the model actually saw)

| Path | Purpose |
|---|---|
| `/tmp/robocasa_train_data/pnp/lerobot/meta/info.json` | Declares cameras 256×256, fps=20, h264 yuv420p. Authoritative for inference resolution. |
| `/tmp/robocasa_train_data/pnp/lerobot/meta/modality.json` | State (16d) + action (12d) sub-key layouts. |
| `/tmp/robocasa_train_data/pnp/lerobot/data/chunk-000/episode_000000.parquet` | Raw state/action values. Inspect with: `docker run --rm -v /tmp/robocasa_train_data:/tmp/robocasa_train_data:ro groot-server:latest python3 /tmp/robocasa_train_data/dump_state.py` |

---

## Cross-references

- [`DOCKER.md`](DOCKER.md) — design notes, file layout, design decisions
- [`GR00T_CHECKPOINTS.md`](GR00T_CHECKPOINTS.md) — checkpoint catalog, paper-section mapping, per-task success rates
- [`GR00T_EVAL_TIPS.md`](GR00T_EVAL_TIPS.md) — common GR00T-eval pitfalls + key insights from achieving the 60% PnP result
- [`../VLA_COMMUNICATION_PROTOCOL.md`](../VLA_COMMUNICATION_PROTOCOL.md) — `/health` `/reset` `/act` HTTP contract
- [`groot_docker_n1.5/README.md`](groot_docker_n1.5/README.md) — companion image (server side)
- [`test_smoke.py`](test_smoke.py) — 6-step smoke test

---

## Upstream installation (host conda, for reference)

The Docker workflow above is the supported path. The conda recipe is preserved here for users who want to develop against the source directly without Docker.

1. Conda env:
   ```sh
   conda create -c conda-forge -n robocasa python=3.11
   conda activate robocasa
   ```
2. robosuite (master branch):
   ```sh
   git clone https://github.com/ARISE-Initiative/robosuite
   cd robosuite && pip install -e .
   ```
3. robocasa:
   ```sh
   cd ..
   git clone https://github.com/robocasa/robocasa
   cd robocasa
   pip install -e .
   pip install pre-commit; pre-commit install   # optional
   # (if numba/numpy clash: conda install -c numba numba=0.56.4 -y)
   ```
4. Bootstrap macros + assets:
   ```sh
   python -m robocasa.scripts.setup_macros
   python -m robocasa.scripts.download_kitchen_assets   # ~10 GB
   ```

### Basic usage (Python API)

```py
import gymnasium as gym
import robocasa
from robocasa.utils.env_utils import run_random_rollouts

env = gym.make(
    "robocasa/PickPlaceCounterToCabinet",
    split="pretrain",  # 'pretrain' or 'target' kitchen scenes/objects
    seed=0,
)
run_random_rollouts(env, num_rollouts=3, num_steps=100, video_path="/tmp/test.mp4")
```

Mac users: prepend `python` with `mj` (`mjpython ...`) for the demo scripts below.

```sh
python -m robocasa.demos.demo_tasks            # play back sample demonstrations
python -m robocasa.demos.demo_kitchen_scenes   # explore 2500+ scenes
python -m robocasa.demos.demo_objects          # explore object library (--obj_types aigen for AI-generated)
python -m robocasa.demos.demo_teleop           # keyboard / spacemouse teleop
```

For tasks, datasets, policy learning, and additional use cases, see the upstream [documentation page](https://robocasa.ai/docs/introduction/overview.html).

---

## Releases

* [2/18/2026] **v1.0**: RoboCasa365 release, with 365 tasks, 2500+ kitchen scenes, 2200+ hours of robot demonstration data, and benchmarking support.
* [10/31/2024] **v0.2**: using RoboSuite `v1.5` as the backend, with improved support for custom robot composition, composite controllers, more teleoperation devices, photo-realistic rendering.

---

## License

Code: [MIT License](https://opensource.org/license/mit)

Assets and Datasets: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en)

---

## Citation

**RoboCasa365:**

```bibtex
@inproceedings{robocasa365,
  title={RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots},
  author={Soroush Nasiriany and Sepehr Nasiriany and Abhiram Maddukuri and Yuke Zhu},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

**RoboCasa (Original Release):**

```bibtex
@inproceedings{robocasa2024,
  title={RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots},
  author={Soroush Nasiriany and Abhiram Maddukuri and Lance Zhang and Adeet Parikh and Aaron Lo and Abhishek Joshi and Ajay Mandlekar and Yuke Zhu},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2024}
}
```
