# RoboCasa — Docker eval container

This file documents the Docker setup added on top of the upstream robocasa
repo. The goal is **a fork-friendly container that lets anyone reproduce the
RoboCasa eval (sim + headless render → mp4) without installing anything on the
host beyond Docker and the NVIDIA Container Toolkit.**

The pattern is borrowed from the `Libero-pro_benchmark` Docker workflow
(v1.3): the image bakes only Python dependencies; the source code, assets, and
results live on the host and are bind-mounted at runtime.

---

## Files added on top of upstream

```
robocasa_docker/
├── Dockerfile                        # NEW — pip-deps-only image
├── .dockerignore                     # NEW — keeps build context small
├── run.sh                            # NEW — main runner (build / smoke / rollout / shell)
├── test_smoke.py                     # NEW — 6-step in-container smoke test
├── examples/
│   └── run_random_rollouts.py        # NEW — minimal mp4-producing demo
├── DOCKER.md                         # NEW — this file
├── README.md                         # UPDATED — Docker section appended
└── .gitignore                        # UPDATED — robosuite/, test_outputs/ excluded
```

The repo's existing `.gitignore` already excludes `robocasa/models/assets/{textures,
generative_textures,objects}` and `macros_private.py`, so downloaded assets
and the auto-generated macro file stay out of git.

## What lives where

```
HOST robocasa_docker/                                 CONTAINER
├── Dockerfile  run.sh  test_smoke.py        bind→    /workspace/robocasa  (rw)
├── examples/                                         /workspace/robocasa/examples
├── robocasa/             (package source)            /workspace/robocasa/robocasa
│   └── models/assets/    (downloaded ~10GB) ←        — written by bootstrap
├── robosuite/            (one-time git clone)       /workspace/robosuite (rw)
└── test_outputs/         (mp4 / png)        bind→    /workspace/robocasa/test_outputs
```

`robosuite/` and `test_outputs/` are gitignored. `robocasa/models/assets/{textures,
generative_textures,objects}` are also gitignored, so the asset download is
host-persistent without polluting the index.

## Design decisions

1. **Image bakes only pip deps.** robocasa and robosuite live on the host and
   reach the interpreter via `PYTHONPATH=/workspace/robocasa:/workspace/robosuite`.
   That makes source edits instant (no rebuild). Image weight is ~8.5GB,
   most of which is the CUDA 12.1 devel base + apt deps.

2. **Assets persist on the host.** `download_kitchen_assets.py` writes to
   `robocasa.__path__[0]/models/assets/`. Because robocasa resolves to
   `/workspace/robocasa/robocasa/`, the ~10GB lands in the host's
   `robocasa_docker/robocasa/models/assets/` and survives container deletion.

3. **No `pip install -e .`.** robocasa's `setup.py` declares no `console_scripts`
   and all internal path resolution uses `robocasa.__path__[0]` /
   `os.path.dirname(__file__)` rather than `pkg_resources`. PYTHONPATH alone
   is enough to make `python -m robocasa.scripts.X` and
   `gym.make("robocasa/...")` work.

4. **lerobot is `--no-deps`'d** because lerobot 0.3.x can transitively force
   `numpy<2`, conflicting with robocasa's `numpy==2.2.5` pin. Random-rollout
   eval doesn't need lerobot's transitive deps; if a future feature does,
   add the explicit deps in a downstream image.

5. **tianshou is `--no-deps`'d** because it pulls the legacy `gym` package
   (not `gymnasium`), which collides at import time with robocasa's
   `gymnasium`-based registry.

6. **EGL with OSMesa fallback.** `MUJOCO_GL=egl` is the default; `run.sh`
   automatically retries with `MUJOCO_GL=osmesa` if EGL fails. The image
   installs `libosmesa6` so the fallback is real, not a stub.

7. **UID/GID matching.** Containers run with `--user $(id -u):$(id -g)` and
   `HOME=/tmp/robocasa-home`. Files written into the bind mounts (downloaded
   zips, mp4 outputs, `macros_private.py`) end up owned by the user, not
   root. No `sudo rm -rf` cleanup needed.

8. **Asset download answers the prompts non-interactively.** The upstream
   `download_kitchen_assets.py` uses `input(...)` for "Proceed? (y/n)";
   `run.sh` pipes `yes y |` through `docker run -i` so the bootstrap is
   non-interactive.

9. **macros_private.py auto-bootstrap.** The container preamble calls
   `python -m robocasa.scripts.setup_macros` and the matching robosuite
   script if the file is missing. Idempotent.

## What it intentionally doesn't do (yet)

- HTTP eval against an external VLA model server. The `VLA_COMMUNICATION_PROTOCOL.md`
  pattern (sub-key dot-namespace) is the next layer; this milestone is
  visualization-only.
- Multi-task eval grids / scheduled benchmarks.
- Training / dataset replay.
- KasmVNC live view (the temporal_vla container does this; we deliberately
  skip it for slimness).

## Quick commands

```bash
# one-time host setup
cd robocasa_docker
git clone https://github.com/ARISE-Initiative/robosuite ./robosuite

# build image
./run.sh --build

# one-time asset download (~10GB, lands in robocasa/models/assets/)
./run.sh --download-assets

# smoke test (writes test_outputs/smoke_agentview.png + smoke_PickPlaceCounterToSink.mp4)
./run.sh --smoke-test

# arbitrary task rollout
./run.sh --rollout TurnOnMicrowave --num 1 --steps 60

# poke around interactively
./run.sh --shell
```
