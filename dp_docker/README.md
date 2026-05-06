# Diffusion Policy in-process eval container

This directory bakes a GPU-only Docker image that loads RoboCasa365's
released **Diffusion Policy** checkpoint
([`robocasa/robocasa365_checkpoints/diffusion_policy/...`](https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/diffusion_policy))
and runs closed-loop eval against `robocasa/<Task>` gym envs **in a single
process**. Unlike the GR00T sibling (`../groot_docker_n1.5/`), there is no
HTTP server, no port, no companion image — the policy and the simulator
live in the same Python process and cooperate via direct
`policy.predict_action(obs) → env.step(action)` calls. The image bakes
only Python deps; the `diffusion_policy` source, the checkpoint, and the
RoboCasa source/assets are bind-mounted at runtime so model edits and
checkpoint swaps never trigger a rebuild.

## What's here

| File | One-line |
|---|---|
| `Dockerfile` | `FROM bigenlight/robocasa-eval:latest` + torch 2.4 (cu121) + chi2023 DP stack (hydra, omegaconf, dill, diffusers, lightning, ...). **Critical:** robomimic is installed from the **RoboCasa fork branch** (`git+https://github.com/ARISE-Initiative/robomimic.git@robocasa`), NOT pypi — the released DP cfg references `VisualCoreLanguageConditioned` and `ResNet*ConvFiLM` classes that are only on that branch. |
| `run.sh` | Main runner: build, download checkpoint, eval (single task), shell, smoke, list-tasks, stop. |
| `eval_dp.py` | Thin in-process eval wrapper. Loads the Lightning ckpt via `torch.load+dill`, instantiates the workspace, runs `n` rollouts of one `robocasa/<Task>` env. |
| `README.md` | This file. |

Three more directories appear once you follow the quick start —
`diffusion_policy/` (host-cloned upstream source, bind-mounted rw),
`checkpoint/` (downloaded weights, bind-mounted ro), and `test_outputs/`
(generated mp4 + json, bind-mounted rw). Their layout is documented in
§"Checkpoint layout" and §"What's bind-mounted".

## Quick start (clone-to-running)

After `git clone <robocasa_docker fork>` and `cd robocasa_docker`. Total
wall-clock is **~12 min on a fresh box** (measured on RTX 3080 10 GB,
2026-05-07): ~3 min image pull (or ~5-10 min build) + ~18 s ckpt
download + ~30 s first-time CLIP download (~600 MB cached afterwards) +
~7 min for 5 atomic rollouts. DP at 100 DDPM steps is the bottleneck —
see §"Performance".

```bash
cd dp_docker

# 1) Image. Either build (~5-10 min, builds on top of robocasa-eval:latest)…
./run.sh --build
# …or pull the pre-built one when published:
#   docker pull bigenlight/dp-eval:latest
#   docker tag  bigenlight/dp-eval:latest dp-eval:latest

# 2) Clone the chi2023 fork that defines the policy + workspace classes.
#    Master/upstream chi2023 is incompatible — these checkpoints reference
#    classes that only exist in the robocasa-benchmark fork.
git clone https://github.com/robocasa-benchmark/diffusion_policy.git ./diffusion_policy

# 3) Download `latest.ckpt` (~1.7 GB on disk). 14 other epoch ckpts exist
#    in the same HF folder; only `latest.ckpt` is needed for first eval.
./run.sh --download-ckpt

# 4) Run a single-task eval (5 rollouts, seeds 0–4, split=pretrain).
./run.sh --eval PickPlaceCounterToSink --num-rollouts 5 --split pretrain

# Outputs land in test_outputs/:
#   dp_PickPlaceCounterToSink_seed0_success{0|1}.mp4 ... seed4_...
#   dp_PickPlaceCounterToSink_summary.json
```

To stop a long-running eval: `./run.sh --stop`. To verify ckpt loads
cleanly without committing to a full rollout: `./run.sh --smoke` (~30 s).

## Architecture summary

| Property | Value |
|---|---|
| Class | `DiffusionTransformerHybridImagePolicy` (chi2023 fork) |
| Backbone | 12-layer Transformer, n_emb=512, 8-head, p_drop_attn=0.3, causal |
| Cond layers | 4 (FiLM-fused CLIP language embedding, 768d) |
| Vision encoder | ResNet18 + GroupNorm, 3 cameras at 256→**224** crop, eval_fixed_crop |
| Cameras | `robot0_agentview_left`, `robot0_agentview_right`, `robot0_eye_in_hand` |
| State | `eef_pos(3) + eef_quat(4) + gripper_qpos(2) + lang_emb(768)` |
| Action | 12-d (`base_motion(4) + control_mode(1) + eef_pos(3) + eef_rot_6d(6) + gripper_close(1)`), relative |
| Diffusion | DDPMScheduler, **100 train + 100 inference steps**, ε-prediction, `squaredcos_cap_v2` |
| Horizon | 10 |
| `n_obs_steps` | 2 |
| `n_action_steps` | **8** |

Compare to GR00T-N1.5: GR00T uses a 16-step action chunk and a single
forward pass per chunk (no diffusion-step loop), so its per-call latency
is ~100 ms on RTX A6000. DP's 100-step DDPM denoising loop dominates
wall-clock — see §"Performance".

## Checkpoint layout

The release ships **one recipe** under `diffusion_policy/` on HuggingFace
(in contrast to GR00T's 12 paper-named recipes). The folder is exactly:

```
robocasa/robocasa365_checkpoints/                                 (HF model repo)
└── diffusion_policy/
    └── 17.40.09_train_diffusion_transformer_hybrid_pretrain_human300/
        ├── .hydra/                                # Hydra config dump (training-time)
        ├── normalizer.pkl                         # 23.5 kB obs/action normalizer
        ├── logs.json.txt                          # 73.5 MB wandb training log (skip)
        └── checkpoints/
            ├── epoch=0100-test_mean_score=-1.000.ckpt        # 1.7 GB
            ├── epoch=0200-test_mean_score=-1.000.ckpt        # 1.7 GB
            ├── ...
            ├── epoch=1400-test_mean_score=-1.000.ckpt        # 1.7 GB
            └── latest.ckpt                                   # 1.7 GB
```

Note: every `.ckpt` filename has `test_mean_score=-1.000`, which means
the model was **never evaluated during training** (no `rollout_every`
hook), so there is no validation-best to pick.

**Which ckpt does the leaderboard use?** The
[RoboCasa leaderboard](https://robocasa.ai/leaderboard.html)'s DP entry
explicitly links to `epoch=0500-test_mean_score=-1.000.ckpt`, not
`latest.ckpt`. These have different LFS shas (`e19011f1...` vs
`fdae0060...`) — they are different snapshots, not aliases. We measured
both on this repo and got **0/5 SR on PnPCounterToSink** for both
(see §"Measured on this repo"); the leaderboard-aligned choice is
`epoch=0500`.

After `./run.sh --download-ckpt`, your local layout is:

```
dp_docker/checkpoint/
└── diffusion_policy/
    └── 17.40.09_train_diffusion_transformer_hybrid_pretrain_human300/
        └── checkpoints/
            └── latest.ckpt                        # 1.7 GB (the only file pulled)
```

To fetch the **leaderboard-aligned `epoch=0500` ckpt** (recommended) or
any other epoch instead of/in addition to `latest.ckpt`:

```bash
# epoch=0500 — what the leaderboard links to
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp/dp-home \
    -v "$PWD/checkpoint:/dp/checkpoint" \
    -v "$HOME/.cache/huggingface:/tmp/dp-home/.cache/huggingface" \
    dp-eval:latest \
    bash -c "mkdir -p /tmp/dp-home && \
      huggingface-cli download robocasa/robocasa365_checkpoints \
        --include 'diffusion_policy/17.40.09_train_diffusion_transformer_hybrid_pretrain_human300/checkpoints/epoch=0500-test_mean_score=-1.000.ckpt' \
        --local-dir /dp/checkpoint"
```

Then point `run.sh` at it via `DP_CKPT_PATH`:

```bash
DP_CKPT_PATH="/dp/checkpoint/diffusion_policy/17.40.09_train_diffusion_transformer_hybrid_pretrain_human300/checkpoints/epoch=0500-test_mean_score=-1.000.ckpt" \
  ./run.sh --eval PickPlaceCounterToSink --num-rollouts 5
```

## `./run.sh` modes

| Mode | What it does |
|---|---|
| `--build` | `docker build -t $DP_IMAGE .`. Layers DP-specific deps on top of `bigenlight/robocasa-eval:latest`. ~5-10 min on a fresh machine; ~30 s if base layers cached. |
| `--download-ckpt` | One-shot container that runs `huggingface-cli download robocasa/robocasa365_checkpoints --include 'diffusion_policy/17.40.09_.../checkpoints/latest.ckpt' --local-dir /dp/checkpoint`. ~1.7 GB on disk. |
| `--eval [<Task>] [--num-rollouts N --seed-base N --split <pretrain\|target> --num-envs N --max-steps N --device cuda:0]` | In-process closed-loop eval. `<Task>` is positional (default: `PickPlaceCounterToSink`). Any flag after it is forwarded to `eval_dp.py`. Defaults filled only for flags you don't pass: `--num-rollouts 5`, `--seed-base 0`, `--split pretrain`, `--num-envs 1`, `--max-steps` from `robocasa.utils.dataset_registry_utils.get_task_horizon × 1.5`, `--device $DP_DEVICE`. Writes per-trial mp4s + `dp_<Task>_summary.json` to `test_outputs/`. |
| `--smoke` | Loads the **real** ckpt, instantiates the policy class, prints the resolved policy / workspace `_target_` + `n_action_steps` + `shape_meta.obs` keys + `shape_meta.action`, exits 0. Does NOT run a rollout. ~30 s. Verifies `torch.load + dill` works and the chi2023 workspace class chain is importable. (Unlike `groot_docker_n1.5/run.sh --smoke`, this is not a dummy-policy protocol probe — DP has no HTTP server, so smoke == real ckpt load.) |
| `--list-tasks` | Prints the contents of `diffusion_policy/diffusion_policy/eval/eval_task_set.py` (the `TASK_SET_REGISTRY` keys: `atomic_seen`, `composite_seen`, `composite_unseen`, etc.) so you can choose a task name. |
| `--shell` | Interactive bash with `PYTHONPATH=/dp/diffusion_policy:/workspace/robocasa:/workspace/robosuite`. For poking around the cfg, the workspace, etc. |
| `--stop` | `docker stop $DP_CONTAINER_NAME`. Only meaningful if a long eval was backgrounded (the default `--eval` runs in the foreground with `--rm`). |

Override via env vars (all optional):

| Var | Default | Effect |
|---|---|---|
| `DP_IMAGE` | `dp-eval:latest` | Image tag used by every mode. |
| `DP_CONTAINER_NAME` | `dp-eval` | Used by `--stop`. |
| `DP_CKPT_PATH` | `/dp/checkpoint/diffusion_policy/17.40.09_train_diffusion_transformer_hybrid_pretrain_human300/checkpoints/latest.ckpt` | In-container path to the active ckpt. |
| `DP_DEVICE` | `cuda:0` | Passed through to `policy.to(device)`. |
| `MUJOCO_GL` | `egl` | Render backend. Set to `osmesa` to fall back if EGL fails. |

## What's bind-mounted

`run.sh` constructs its `COMMON_ARGS` so each mount is added only when
the host path exists; this lets `--build` and `--download-ckpt` work on
a fresh machine before the source clone or checkpoint exist.

| Host path | Container path | Mode | Purpose |
|---|---|---|---|
| `dp_docker/eval_dp.py` | `/dp/eval_dp.py` | ro | Wrapper source — edit on host, restart container, no rebuild. |
| `dp_docker/diffusion_policy/` | `/dp/diffusion_policy` | rw | Cloned upstream source. `PYTHONPATH=/dp/diffusion_policy` registers it; the workspace classes live here. |
| `dp_docker/checkpoint/` | `/dp/checkpoint` | **ro** | Weights + `.hydra/` + `normalizer.pkl`. Read-only inside the container so you can swap in a different epoch ckpt from the host between runs without busy-handle issues. |
| `dp_docker/test_outputs/` | `/dp/test_outputs` | rw | mp4 + summary.json drops. |
| `<repo root>/` (the robocasa source) | `/workspace/robocasa` | rw | For `gym.make("robocasa/<Task>")`. Inherited from base image's PYTHONPATH. |
| `<repo root>/robosuite/` | `/workspace/robosuite` | rw | Robosuite source; same reason. |
| `~/.cache/huggingface` | `/tmp/dp-home/.cache/huggingface` | rw | HF cache shared across runs. |

Other docker flags worth knowing: `--gpus all` (auto-detected from
`nvidia-smi` + `docker info`), `--shm-size=8g`, `--user $(id -u):$(id -g)`,
`HOME=/tmp/dp-home`, `--rm`. **No `--network host`** — DP eval is
in-process, no port to expose.

## Eval contract

DP's eval is **in-process**: there is no HTTP, no zmq, no protocol
adapter. The wrapper `eval_dp.py` does:

1. `torch.load(open(ckpt,'rb'), pickle_module=dill, weights_only=False)` → `payload`
2. `cfg = payload['cfg']` → omegaconf DictConfig
3. `cls = hydra.utils.get_class(cfg._target_)` → `BaseWorkspace` subclass
4. `workspace = cls(cfg)` → instantiates policy, optimizer, ema (we discard the latter two)
5. `workspace.load_payload(payload)` → loads weights into `workspace.model` (and `workspace.ema_model` if `cfg.training.use_ema`)
6. `policy = workspace.ema_model if cfg.training.use_ema else workspace.model; policy.to(device).eval()`
7. For each rollout seed: `env = gym.make("robocasa/<Task>", split=..., seed=...)`; replay `policy.predict_action(obs)` chunks of 8 actions until success or `max_steps`.
8. Mirror GR00T's output schema: `dp_<Task>_seed<N>_success<0|1>.mp4` + `dp_<Task>_summary.json` (with per-trial latency, success flag, steps-to-success).

Action chunk = **8** (vs GR00T's 16). DP outputs absolute end-effector
targets in the OSC frame; the wrapper passes them through robocasa's
`gym_wrapper.unmap_action` exactly as for the random-rollout / GR00T
paths — same binarization at threshold 0.5 for `gripper_close` and
`control_mode`.

## Performance expectations

The bottleneck is the **100-step DDPM denoising loop** at every
`policy.predict_action(...)` call. With `num_inference_steps=100` and a
12-layer Transformer + 3 ResNet18 image encoders, **measured** on this
repo (RTX 3080 10 GB, num_envs=1, 2026-05-07):

| Hardware | Per `predict_action` call (1 chunk = 8 actions) | Source |
|---|---|---|
| RTX 3060 12 GB | ~700-1000 ms (extrapolated) | extrapolated from 3080 below |
| **RTX 3080 10 GB** | **~706 ms** | measured on PnPCounterToSink, summary.json mean_lat |
| RTX A6000 48 GB | ~150-250 ms (extrapolated) | from GR00T benchmark on same machine |

Rollout wall-clock (single env, RTX 3080 10 GB, full max-steps cap when
the policy doesn't succeed early):

| Task class | Horizon | max_steps (×1.5) | predict calls / rollout | wall-clock / rollout | 5 rollouts |
|---|---|---|---|---|---|
| Atomic (e.g. `PickPlaceCounterToSink`) | ~400 | 600 | ~75 | ~60-80 s | **~7 min** (measured) |
| Composite (e.g. `PrepareCoffee`) | ~1200 | 1800 | ~225 | ~180-240 s | ~20-25 min (extrapolated) |

The 7-min measured number includes a one-time ~30 s CLIP download on
the first run (subsequent runs hit the cached weights — see
"Troubleshooting" below). DP almost never succeeds on
`PickPlaceCounterToSink` (paper avg 15.7% atomic, 0/5 measured here),
so wall-clock is dominated by the time-out cap, not by early-success
shortcuts.

**Future work — DDPM → DDIM swap.** With `num_inference_steps=15` and a
DDIM scheduler, per-chunk latency drops by ~6× without retraining
(`prediction_type=epsilon` is compatible). Not yet wired into `run.sh`;
if you want to test it, edit `cfg.policy.noise_scheduler` after step
(5) in `eval_dp.py`. Worth ~6× speedup but unverified for SR impact.

## Paper-reported results

From RoboCasa365 ICLR 2026, Table 1 (§4.1):

| Method | Atomic-Seen | Composite-Seen | Composite-Unseen | Avg |
|---|---|---|---|---|
| **Diffusion Policy** (this ckpt) | **15.7%** | **0.2%** | **1.25%** | **6.1%** |
| pi-0 | 36.3% | 5.2% | 0.7% | — |
| pi-0.5 | 39.6% | 7.1% | 1.2% | — |
| GR00T-N1.5 (multitask) | 43.0% | 9.6% | 4.4% | — |

Paper quote: *"Diffusion Policy performs the worst, highlighting how
high-capacity vision-language-action models can better fit large,
diverse multi-task robot datasets."* (§4.1)

This is the **`pretrain_human300` multitask baseline only**. Unlike
GR00T-N1.5, no target-fine-tuned DP checkpoint is released — the
authors did not run a target-FT phase for DP. So expect numbers in the
same ballpark for any task in `atomic_seen` / `composite_seen` /
`composite_unseen` evaluated under `--split pretrain`.

### Measured on this repo

Verified end-to-end run from the in-tree `dp_docker/` (RTX 3080 10 GB,
2026-05-07):

| Task | Split | Ckpt | Rollouts | Successes | mean_lat | Notes |
|---|---|---|---|---|---|---|
| `PickPlaceCounterToSink` | pretrain | `latest.ckpt` (sha `fdae0060…`) | 5 | **0/5 = 0%** | 705.6 ms | Originally archived to `test_outputs/latest_ckpt_run/`. mp4 = h264 768×512 @ 20fps, 600 frames per rollout. |
| `PickPlaceCounterToSink` | pretrain | **`epoch=0500-test_mean_score=-1.000.ckpt`** (sha `e19011f1…`, leaderboard-aligned) | 5 | **0/5 = 0%** | 762.4 ms | Same SR as `latest.ckpt`. Per-task DP number not in paper; both consistent with the 15.7% Atomic-Seen average. mp4s + `dp_PickPlaceCounterToSink_summary.json` at `test_outputs/`. |

Compare to `multitask` GR00T-N1.5 on the same task on the same machine:
3/5 = 60% (`README.md §"Verified results (GR00T-N1.5)"`). The gap is
roughly the paper's 43.0 / 15.7 = 2.7× spread, plus PnPCounterToSink
appearing to be one of the easier atomic tasks for GR00T.

## What's NOT in this image

| Not baked | Why | How it gets onto the host |
|---|---|---|
| `diffusion_policy/` source | upstream API drifts; lets you patch in-place | `git clone https://github.com/robocasa-benchmark/diffusion_policy.git ./diffusion_policy` |
| Checkpoint (1.7 GB) | size; might want to swap epochs | `./run.sh --download-ckpt` |
| RoboCasa source + assets | inherited from base image's bind-mount pattern | `cd .. && ./run.sh --download-assets` (top-level robocasa_docker workflow) |
| `robosuite/` source | upstream pinned to master | `cd .. && git clone https://github.com/ARISE-Initiative/robosuite ./robosuite` |
| Training data (LeRobot) | eval-only image; not needed for rollout | `huggingface.co/datasets/robocasa/robocasa365_lerobot` (~10 GB) if you want it for offline replay |

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Image dp-eval:latest not found` | image not built or pulled | `./run.sh --build` (~5-10 min) — or `docker pull bigenlight/dp-eval:latest && docker tag bigenlight/dp-eval:latest dp-eval:latest` once published |
| `ERROR: base image bigenlight/robocasa-eval:latest missing` | base image not yet on host | `docker pull bigenlight/robocasa-eval:latest` (or `cd .. && ./run.sh --build`) before `./run.sh --build` |
| `ERROR: ./diffusion_policy/diffusion_policy not found` | fork not cloned | `git clone https://github.com/robocasa-benchmark/diffusion_policy.git ./diffusion_policy` |
| `ERROR: .../checkpoint/.../latest.ckpt not found` | ckpt not downloaded | `./run.sh --download-ckpt` |
| `AttributeError: Can't get attribute '...' on module 'diffusion_policy....'` during `torch.load` | dill/pickle skew between training-time class graph and runtime; or chi2023 fork wasn't cloned | rebuild image with the dill version pinned in `Dockerfile`; verify the cloned fork actually has the class the cfg references |
| `ImportError: cannot import name 'VisualCoreLanguageConditioned'` (or `ResNet{18,34,50}ConvFiLM`) from `robomimic.models.obs_core` | The DP cfg references classes that exist **only** in the RoboCasa fork of robomimic, NOT in pypi `robomimic 0.3.0`. Public docs do not mention this. | Confirm the `Dockerfile` installs `pip install git+https://github.com/ARISE-Initiative/robomimic.git@robocasa` (the `robocasa` branch), not pypi. Rebuild image. |
| `ModuleNotFoundError: No module named 'robomimic.X'` (any other class) | robomimic version skew | as above — the `robocasa` branch is canonical for this checkpoint |
| First eval slow / hangs at "loading CLIP" / network warning | `transformers.CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')` runs once on the first eval (~600 MB) and persists via the host HF cache bind-mount (`~/.cache/huggingface`). | Wait once (~30 s on a fast link). Subsequent runs hit the cache. The wrapper also keeps a module-level cache so additional rollouts in the same process skip the GPU load entirely. |
| `0/5 success on PnPCounterToSink` | Expected for the multitask DP checkpoint — paper reports **15.7% atomic-seen avg** (Table 1, §4.1) and there is no per-task PnP figure released. | This is on-distribution behavior. No target-FT'd DP recipe was released. Try `--num-rollouts 30` per task to get a meaningful per-task estimate, or run the full `atomic_seen` soup via the upstream `eval_robocasa.py` from `--shell`. |
| CUDA OOM on RTX 3060 (12 GB) | `--num-envs > 1` is tight at 12 GB once CLIP + DP-Hybrid + ResNet × 3 cams + MuJoCo render are co-resident | use `--num-envs 1` (default); lower `cfg.policy.num_inference_steps` if needed (will hurt SR); or force CLIP to CPU by exporting `CUDA_VISIBLE_DEVICES=-1` only during the lang-emb call (custom patch) |
| EGL render fails | EGL probe fails inside container | run with `MUJOCO_GL=osmesa ./run.sh --eval ...`; the base image installs `libosmesa6` so the fallback is real |
| 5-rollout eval takes hours | DP at 100 DDPM steps × composite horizon 1800 (max_steps cap) | pick an atomic task first (`PickPlaceCounterToSink`) to confirm the pipe; or hack DDIM swap in `eval_dp.py` (see §"Performance") |
| Files in `test_outputs/` owned by root | Custom `docker run` missing `--user` | Use `./run.sh`, or mirror `--user $(id -u):$(id -g) -e HOME=/tmp/dp-home` |

For deeper debugging, `./run.sh --shell` drops you into the container
with all bind mounts active; `python -c "import torch, dill, hydra,
diffusion_policy; ..."` verifies the import chain.

## Cross-references

- [`../README.md`](../README.md) — top-level fork README; sim quickstart, `bigenlight/robocasa-eval` base image
- [`../groot_docker_n1.5/README.md`](../groot_docker_n1.5/README.md) — sibling: GR00T-N1.5 HTTP eval workflow (companion image pattern)
- [`../GR00T_CHECKPOINTS.md`](../GR00T_CHECKPOINTS.md) — context on the broader RoboCasa365 checkpoint catalog
- [`../DOCKER.md`](../DOCKER.md) — design rationale shared by all eval containers in this repo
- Fork: <https://github.com/robocasa-benchmark/diffusion_policy>
- HF checkpoint: <https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/diffusion_policy>
- Paper: <https://robocasa.ai/assets/robocasa365_iclr26.pdf>
- Leaderboard: <https://robocasa.ai/leaderboard.html>
