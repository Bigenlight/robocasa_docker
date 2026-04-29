# GR00T-N1.5 HTTP server

This directory bakes a GPU-only Docker image that serves a GR00T-N1.5
policy over HTTP on port 8500, conforming to
[`/home/theo/workspace/VLA_COMMUNICATION_PROTOCOL.md`](../../VLA_COMMUNICATION_PROTOCOL.md).
It lives **inside** the `robocasa_docker/` repo as the server-side
companion to the eval client at
[`../examples/run_groot_eval.py`](../examples/run_groot_eval.py); see
[`../README.md`](../README.md) for the simulator side. The image bakes
only Python deps — the Isaac-GR00T source, the checkpoint, and
`serve_groot.py` are bind-mounted at runtime so model edits and recipe
swaps never trigger a rebuild.

## What's here

| File | One-line |
|---|---|
| `Dockerfile` | CUDA 12.1 + py3.10 + torch 2.4 + transformers 4.51 + flash-attn (optional) + Isaac-GR00T transitive deps. |
| `run.sh` | Main runner: build, download multitask ckpt, serve (real / dummy / bg / fg), shell, smoke, stop. |
| `serve_groot.py` | FastAPI server with `/health`, `/reset`, `/act`. Has `--dummy` for protocol smoke tests. |
| `swap_ckpt.sh` | Repoint `checkpoint/` root-level symlinks to one of 12 paper-named recipes. |
| `README.md` | This file. |

Two more directories appear once you follow §3 — `Isaac-GR00T/` (cloned
upstream source, bind-mounted rw) and `checkpoint/` (downloaded
weights, bind-mounted ro). Their layout is documented in §4.

## Quick start (clone-to-running)

After `git clone <robocasa_docker fork>` and `cd robocasa_docker`. Total
wall-clock is ~25-30 min: ~10 min ckpt download + ~12 min image build (or
~3 min pull) + ~3 min first-load policy.

```bash
cd groot_docker_n1.5

# 1) Image. Either build (~12 min, ~17 GB)…
./run.sh --build
# …or pull the pre-built one (preferred, ~3 min):
docker pull bigenlight/groot-server:n1.5
docker tag  bigenlight/groot-server:n1.5 groot-server:latest

# 2) Clone Isaac-GR00T at the n1.5 tag. Master/n1.7 is incompatible with
#    these checkpoints — the policy class moved and modality keys differ.
git clone https://github.com/NVIDIA/Isaac-GR00T -b n1.5-release ./Isaac-GR00T

# 3) Download the default (multitask) checkpoint, ~7.1 GB on disk
#    (the 8.6 GB optimizer.pt is excluded; pass --all to keep it).
./run.sh --download-ckpt
#    For a different recipe, see §6.

# 4) Start the server in the background.
./run.sh --serve-bg
docker logs -f groot-server     # watch until you see "policy ready"

# 5) Verify.
curl http://localhost:8500/health
# {"status":"ok","model":"groot-n1.5-multitask","action_keys":[…],…}
```

To stop: `./run.sh --stop`. To swap to a different recipe without
re-downloading anything you already have: `./swap_ckpt.sh <recipe>` (§5).

## Checkpoint storage layout

`./run.sh --download-ckpt` (and `swap_ckpt.sh`) maintain a fan-out
where the **5 root-level entries inside `checkpoint/` are symlinks**
into a recipe subdirectory under `checkpoint/gr00t_n1-5/`. The server
reads `--model-path /groot/checkpoint`, so it only sees those 5 root
entries; the actual safetensors live one level down.

After `./run.sh --download-ckpt` (only the multitask recipe is on disk):

```
groot_docker_n1.5/checkpoint/
├── config.json                       -> gr00t_n1-5/multitask_learning/checkpoint-120000/config.json   (ACTIVE)
├── experiment_cfg                    -> gr00t_n1-5/multitask_learning/checkpoint-120000/experiment_cfg
├── model-00001-of-00002.safetensors  -> gr00t_n1-5/multitask_learning/checkpoint-120000/model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors  -> gr00t_n1-5/multitask_learning/checkpoint-120000/model-00002-of-00002.safetensors
├── model.safetensors.index.json      -> gr00t_n1-5/multitask_learning/checkpoint-120000/model.safetensors.index.json
└── gr00t_n1-5/
    └── multitask_learning/checkpoint-120000/    (real files, ~7.1 GB)
```

After also downloading `target_posttraining/composite_seen` and
swapping to it (so the symlinks now point at composite_seen, while
multitask remains on disk for instant swap-back):

```
groot_docker_n1.5/checkpoint/
├── config.json                       -> gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000/config.json   (ACTIVE)
├── experiment_cfg                    -> gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000/experiment_cfg
├── model-00001-of-00002.safetensors  -> ... (same recipe)
├── model-00002-of-00002.safetensors  -> ... (same recipe)
├── model.safetensors.index.json      -> ... (same recipe)
└── gr00t_n1-5/
    ├── multitask_learning/checkpoint-120000/                                            (real, ~7.1 GB)
    └── foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000/   (real, ~7.1 GB)
```

The 5 root-level symlinks are what the server actually reads. Adding
more recipes just deposits more directories under `gr00t_n1-5/`; it
does **not** touch the symlinks. To switch the active recipe, use
`swap_ckpt.sh` (next section).

## Switching checkpoints with `swap_ckpt.sh`

`swap_ckpt.sh` re-points the 5 root symlinks at a different recipe
directory. The server bind-mount is read-only inside the container, so
each `./run.sh --serve` picks up whatever the symlinks point at on
launch — no image rebuild, no file copy.

The 12 recipes (success rates from
[`../GR00T_CHECKPOINTS.md`](../GR00T_CHECKPOINTS.md) §4, all in % task
SR on RoboCasa365):

| Recipe name | Paper | Good for | Atomic-Seen | Comp-Seen | Comp-Unseen |
|---|---|---|---|---|---|
| `multitask`                    | §4.1 Table 1 | weak baseline trained on all 300 tasks at once | 43.0 | 9.6 | 4.4 |
| `pretraining`                  | §4.2 Table 2 | stage-1 only, no target FT | 41.9 | 0.0 | 0.2 |
| `target_only_atomic`           | §4.2 Table 2 | target-FT-from-scratch ablation, atomic | 60.6 | — | — |
| `target_only_composite_seen`   | §4.2 Table 2 | target-FT-from-scratch, composite-seen | — | 35.0 | — |
| `target_only_composite_unseen` | §4.2 Table 2 | target-FT-from-scratch, composite-unseen | — | — | 33.3 |
| `target_pt_atomic`             | §4.2 Table 2 | **headline** Pretrain+Target, atomic | **68.5** | — | — |
| `target_pt_composite_seen`     | §4.2 Table 2 | **headline 40.6%** Pretrain+Target, composite-seen | — | **40.6** | — |
| `target_pt_composite_unseen`   | §4.2 Table 2 | **headline** Pretrain+Target, composite-unseen | — | — | **42.1** |
| `lifelong_phase1`              | §4.3 Table 3 | continual learning, atomic only | (lifelong taxonomy — see GR00T_CHECKPOINTS.md §4) ||| 
| `lifelong_phase2`              | §4.3 Table 3 | + 2-3 stage composite | ||| 
| `lifelong_phase3`              | §4.3 Table 3 | + 4-5 stage composite | ||| 
| `lifelong_phase4`              | §4.3 Table 3 | + 6+ stage composite | ||| 

Subcommands:

| Command | Effect |
|---|---|
| `./swap_ckpt.sh --list` | print all recipes, mark active with `*` |
| `./swap_ckpt.sh --current` | print just the active recipe name |
| `./swap_ckpt.sh <recipe>` | repoint the 5 root symlinks |
| `./swap_ckpt.sh -h` / `--help` | usage |

Constraint: the script aborts (exit 2) if the `groot-server` container
is currently running — checked via `docker ps --filter
'name=^/groot-server$'`. Stop the server first.

End-to-end example (from multitask to target_pt_composite_seen):

```bash
./run.sh --stop
./swap_ckpt.sh target_pt_composite_seen
./run.sh --serve-bg
```

If the recipe directory isn't on disk yet, `swap_ckpt.sh` prints a
ready-to-paste `huggingface-cli` command to fetch it (see next section
for a generic version).

## Downloading additional checkpoints

`./run.sh --download-ckpt` is **hardcoded to multitask**
(`gr00t_n1-5/multitask_learning/checkpoint-120000`). For any other
recipe, run a one-shot docker that mirrors the run.sh pattern. The
`--exclude '*optimizer*'` saves 8.6 GB per checkpoint by skipping
training-only state.

Example for `target_posttraining/composite_seen` (the headline 40.6%
checkpoint):

```bash
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp/groot-home \
    -v "$PWD/checkpoint:/groot/checkpoint" \
    -v "$HOME/.cache/huggingface:/tmp/groot-home/.cache/huggingface" \
    groot-server:latest \
    bash -c "
        mkdir -p /tmp/groot-home
        huggingface-cli download robocasa/robocasa365_checkpoints \
            --include 'gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000/*' \
            --exclude '*optimizer*' \
            --local-dir /groot/checkpoint
    "
```

Generic — substitute `<RECIPE_PATH>` with any value from §5's table
(e.g. `gr00t_n1-5/lifelong_learning/phase1/checkpoint-100000`):

```bash
RECIPE_PATH=gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp/groot-home \
    -v "$PWD/checkpoint:/groot/checkpoint" \
    -v "$HOME/.cache/huggingface:/tmp/groot-home/.cache/huggingface" \
    groot-server:latest \
    bash -c "mkdir -p /tmp/groot-home && \
        huggingface-cli download robocasa/robocasa365_checkpoints \
            --include '${RECIPE_PATH}/*' --exclude '*optimizer*' \
            --local-dir /groot/checkpoint"
```

After download, run `./swap_ckpt.sh <recipe>` to make the new files
the active recipe. Full URL list:
[`../GR00T_CHECKPOINTS.md`](../GR00T_CHECKPOINTS.md) §7.

## `./run.sh` modes

| Flag | What it does (verified) |
|---|---|
| `--build` | `docker build -t $GROOT_IMAGE .`. ~12 min on a fresh box. |
| `--download-ckpt [--all]` | Spins a one-shot container that runs `huggingface-cli download robocasa/robocasa365_checkpoints --include 'gr00t_n1-5/multitask_learning/checkpoint-120000/*'`. Without `--all`, adds `--exclude '*optimizer*'` (saves 8.6 GB). Symlinks the 5 root files into `checkpoint/`. |
| `--serve` | Foreground server, real model. Streams logs. Ctrl-C kills the container (it's `--rm`). |
| `--serve-bg` | Background server, real model. Container named `groot-server`. Use `docker logs -f groot-server` to watch. |
| `--dummy` | Foreground server with `--dummy` flag — no model load, returns zero actions of the protocol-correct shape. For protocol smoke tests on machines without a GPU or a checkpoint. |
| `--smoke` | `--dummy` server + curl `/health` + synthetic `/act` probe + tear-down. End-to-end protocol check, ~30 s. |
| `--shell` | Interactive bash inside the container with all bind mounts active. |
| `--stop` | `docker stop groot-server`. |
| `-- --no-strict <args>` | After `--`, extra args pass through to `serve_groot.py`. |

Env vars (all optional):

| Var | Default | Effect |
|---|---|---|
| `GROOT_IMAGE` | `groot-server:latest` | Image tag used by every mode. |
| `GROOT_CONTAINER_NAME` | `groot-server` | Used by `--serve-bg` / `--smoke` / `--stop`. |
| `GROOT_PORT` | `8500` | Host port; `--network host` so this is also the in-container port. |

## What's bind-mounted

`run.sh` constructs `COMMON_ARGS` (see lines 165-182) with these
mounts; each is added only when the host path exists:

| Host path | Container path | Mode | Purpose |
|---|---|---|---|
| `groot_docker_n1.5/serve_groot.py` | `/groot/serve_groot.py` | ro | Server source — edit on host, restart container, no rebuild. |
| `groot_docker_n1.5/Isaac-GR00T/` | `/groot/Isaac-GR00T` | rw | Cloned upstream source (`PYTHONPATH=/groot/Isaac-GR00T`). The `pip install --user --no-deps -e` in the preamble registers the dist-info so `importlib.metadata.version("gr00t")` resolves. |
| `groot_docker_n1.5/checkpoint/` | `/groot/checkpoint` | **ro** | Weights + `experiment_cfg/`. Read-only inside the container so atomic symlink swaps from the host are picked up cleanly on the next server start (no busy file handles). |
| `~/.cache/huggingface` | `/tmp/groot-home/.cache/huggingface` | rw | HF cache shared across runs. Container `HOME=/tmp/groot-home` avoids root-owned writes. |

Other docker flags worth knowing: `--gpus all` (auto-detected from
`nvidia-smi` + `docker info`), `--shm-size=8g`, `--network host`,
`--user $(id -u):$(id -g)`, `--rm`.

## HTTP contract summary

Three endpoints; full schema in
[`../../VLA_COMMUNICATION_PROTOCOL.md`](../../VLA_COMMUNICATION_PROTOCOL.md).

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | `{status, model, action_type, action_keys, n_action_steps, embodiment_tag, video_keys, state_keys}` |
| POST | `/reset` | — | `{status: "reset"}` (clears action-chunk queue if any) |
| POST | `/act` | observation payload (see below) | `{action.* : [T, D] list, latency_ms: float}` |

Sub-key dot-namespace (matches RoboCasa env's gym `action_space` dict
exactly — no remapping in the eval client):

```
# request body keys
observation.images.robot0_eye_in_hand            base64 PNG, HWC uint8 RGB
observation.images.robot0_agentview_left         base64 PNG
observation.images.robot0_agentview_right        base64 PNG
observation.state.base_position                  [3]
observation.state.base_rotation                  [4]   quaternion
observation.state.end_effector_position_relative [3]
observation.state.end_effector_rotation_relative [4]   quaternion
observation.state.gripper_qpos                   [2]
task                                             str

# response body keys (T = n_action_steps, default 16)
action.end_effector_position    [T, 3]
action.end_effector_rotation    [T, 3]   axis-angle
action.gripper_close            [T, 1]
action.base_motion              [T, 4]
action.control_mode             [T, 1]
latency_ms                      float
```

Aliases the server accepts on the request side (back-compat with the
shorter names in older protocol drafts):
`observation.images.static` → `robot0_agentview_left`,
`observation.images.wrist` → `robot0_eye_in_hand`.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Image groot-server:latest missing — run ./run.sh --build first` | image not built or pulled | `./run.sh --build` (12 min) or `docker pull bigenlight/groot-server:n1.5 && docker tag bigenlight/groot-server:n1.5 groot-server:latest` |
| `ERROR: .../checkpoint/experiment_cfg not found` | no recipe downloaded yet | `./run.sh --download-ckpt` (default multitask), or §6 for any other recipe |
| `ERROR: .../Isaac-GR00T/gr00t not found` | `Isaac-GR00T/` not cloned | `git clone https://github.com/NVIDIA/Isaac-GR00T -b n1.5-release ./Isaac-GR00T` |
| `groot-server is running — stop it first with: ./run.sh --stop` | tried to swap recipes mid-flight | exactly what it says: `./run.sh --stop && ./swap_ckpt.sh <recipe> && ./run.sh --serve-bg` |
| port 8500 already in use | another GR00T server running, or a stale named container | `./run.sh --stop` to clear; or `GROOT_PORT=8501 ./run.sh --serve-bg` for a parallel run |
| `/health` returns `status: loading` then `/act` 503s | model load raised in background — see `docker logs groot-server` | usually a metadata mismatch (try `./run.sh --serve -- --no-strict`) or a torch/CUDA mismatch (rebuild image) |
| `RuntimeError: missing observation.images.robot0_agentview_right` | client only sent left + eye_in_hand | this checkpoint expects all 3 cams; either send all 3 or pass `--no-strict` and accept silent zero-pad |
| build OOMs near `flash-attn` | nvcc spawns too many parallel jobs | `MAX_JOBS=4` is already baked into the Dockerfile; flash-attn is non-fatal — the server falls back to torch SDPA if it fails |

For deeper debugging see `docker logs groot-server` and the design
notes in the `serve_groot.py` docstring (importantly:
`_build_robocasa_modality` mirrors the canonical
`PandaOmronDataConfig.transform()` from the robocasa-benchmark
Isaac-GR00T fork — divergence here causes silent visual / state
corruption).
