<h1 align="center">RoboCasa</h1>
<!-- ![alt text](https://github.com/UT-Austin-RPL/maple/blob/web/src/overview.png) -->
<img src="docs/images/readme.webp" width="100%" />

**RoboCasa** is a large-scale simulation framework for training generally capable robots to perform everyday tasks. It was [originally released](https://robocasa.ai/assets/robocasa_rss24.pdf) in 2024 by UT Austin researchers. The latest iteration, **RoboCasa365**, builds upon the original release with significant new functionalities to support large-scale training and benchmarking in sim. Four pillars underlie RoboCasa365:
- **Diverse tasks**: 365 tasks created with the guidance of large language models
- **Diverse assets**: including 2,500+ kitchen scenes and 3,200+ 3D objects
- **High-quality demonstrations**: including 600+ hours of human demonstrations in addition to 1,600+ hours of robot datasets created with automated trajectory tools
- **Benchmarking support**: popular policy learning methods including Diffusion Policy, pi, and GR00T, plus user-submitted models on the [leaderboard](https://robocasa.ai/leaderboard.html)


This guide contains information about installation and setup. Please refer to the following resources for additional information:

[**[Home page]**](https://robocasa.ai) &ensp; [**[Documentation]**](https://robocasa.ai/docs/introduction/overview.html) &ensp; [**[RoboCasa365 Paper]**](https://robocasa.ai/assets/robocasa365_iclr26.pdf) &ensp; [**[Original RoboCasa Paper]**](https://robocasa.ai/assets/robocasa_rss24.pdf) &ensp; [**[Leaderboard]**](https://robocasa.ai/leaderboard.html)

-------
## Installation
RoboCasa works across all major computing platforms. The easiest way to set up is through the [Anaconda](https://www.anaconda.com/) package management system. Follow the instructions below to install:
1. Set up conda environment:

   ```sh
   conda create -c conda-forge -n robocasa python=3.11
   ```
2. Activate conda environment:
   ```sh
   conda activate robocasa
   ```
3. Clone and setup robosuite dependency (**important: use the master branch!**):

   ```sh
   git clone https://github.com/ARISE-Initiative/robosuite
   cd robosuite
   pip install -e .
   ```
4. Clone and setup this repo:

   ```sh
   cd ..
   git clone https://github.com/robocasa/robocasa
   cd robocasa
   pip install -e .
   pip install pre-commit; pre-commit install           # Optional: set up code formatter.

   (optional: if running into issues with numba/numpy, run: conda install -c numba numba=0.56.4 -y)
   ```
5. Install the package and download assets:
   ```sh
   python -m robocasa.scripts.setup_macros              # Set up system variables.
   python -m robocasa.scripts.download_kitchen_assets   # Caution: Assets to be downloaded are around 10GB.
   ```

-------
## Docker (recommended for headless eval / shareable setup)

This fork ships a Docker workflow that lets a fresh machine run a RoboCasa
rollout (sim + headless render → mp4) without installing **anything** on the
host beyond Docker and the NVIDIA Container Toolkit. The image bakes only pip
dependencies; the robocasa source, robosuite source, asset directory, and
outputs all live on the host and are bind-mounted at runtime.

See [`DOCKER.md`](DOCKER.md) for the full design notes and what changed.

### One-time host setup

```sh
cd robocasa_docker

# robosuite (master branch — same instruction as conda install above, just
# placed here so the container can bind-mount it).
git clone https://github.com/ARISE-Initiative/robosuite ./robosuite

# Build the image (~8GB, takes a few minutes — CUDA 12.1 base layer is the bulk).
./run.sh --build

# Download kitchen assets (~10GB, one-time). Lands in robocasa/models/assets/
# on the host so it persists across container restarts.
./run.sh --download-assets
```

### Run the smoke test

```sh
./run.sh --smoke-test
```

Six steps run inside the container:
1. import sanity (numpy / mujoco / robosuite / robocasa paths)
2. render-backend probe (creates a tiny `mujoco.Renderer` to confirm `MUJOCO_GL` works)
3. `gym.make("robocasa/PickPlaceCounterToSink", split="pretrain")` + `reset()`
4. PNG render of the agentview camera → `test_outputs/smoke_agentview.png`
5. 10 random sim steps (no crash, reward/done shape correct)
6. `run_random_rollouts(num_rollouts=1, num_steps=30)` → `test_outputs/smoke_PickPlaceCounterToSink.mp4`

If EGL fails (no GPU passthrough, missing driver caps, etc.), `run.sh`
automatically retries the same test with `MUJOCO_GL=osmesa`.

### Run an arbitrary task rollout

```sh
./run.sh --rollout TurnOnMicrowave --num 1 --steps 60 --seed 0
# writes test_outputs/TurnOnMicrowave_seed0.mp4
```

Some good headless-friendly task names to try (all of these are real
gym ids registered as `robocasa/<TaskName>`):

- pick & place: `PickPlaceCounterToSink`, `PickPlaceCounterToStove`,
  `PickPlaceCounterToCabinet`, `PickPlaceCounterToMicrowave`,
  `PickPlaceMicrowaveToCounter`, `PickPlaceSinkToCounter`,
  `PickPlaceStoveToCounter`
- atomic: `OpenCabinet`, `CloseCabinet`, `OpenDoor`, `CloseDoor`,
  `OpenDrawer`, `CloseDrawer`, `TurnOnMicrowave`, `TurnOffMicrowave`,
  `TurnOnStove`, `TurnOffStove`, `TurnOnSinkFaucet`

396 robocasa tasks total — list them all inside the container:
`./run.sh --shell` then `python -c "import robocasa, gymnasium as gym; print('\n'.join(sorted(s for s in gym.envs.registry if s.startswith('robocasa/'))))"`

### Interactive shell

```sh
./run.sh --shell
# inside: python -c "import robocasa, gymnasium as gym; print(robocasa.__version__)"
```

### Image / container conventions

- Image tag: `robocasa-eval:latest` (override with `ROBOCASA_IMAGE=...`).
- Container runs as your host UID/GID — files written into the bind mounts
  (downloaded zips, mp4s, `macros_private.py`) are owned by you, not root.
- `MUJOCO_GL=egl` by default; `run.sh` retries with `osmesa` on failure.
- `--gpus all` is added automatically when `nvidia-smi` is detected on the host.

### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: robosuite` | `./robosuite/` not cloned | `git clone https://github.com/ARISE-Initiative/robosuite ./robosuite` |
| Empty `test_outputs/` after smoke test | EGL probe failed silently and OSMesa fallback also failed | Check `nvidia-smi` on host, ensure `nvidia-container-toolkit` is installed and `docker info` lists `nvidia` runtime |
| `KeyError: 'robot0_agentview_center'` | You're calling `sim.render(camera_name="robot0_agentview_center")` — the gym wrapper only registers `robot0_agentview_{left,right}` and `robot0_eye_in_hand` | Use `robot0_agentview_left` (default in our scripts) |
| `AttributeError: 'OrderEnforcing' object has no attribute 'sim'` | gymnasium 1.x removed `Wrapper.__getattr__` | The Dockerfile pins `gymnasium==0.29.1` — make sure you didn't override that |
| Pip can't satisfy `numpy==2.2.5` | A transitive dep (often older `lerobot`/`tianshou`) is force-pinning `numpy<2` | We install those `--no-deps` already; do the same in any downstream image |
| `download_kitchen_assets.py` hangs at "Proceed? (y/n)" | Stdin not piped through to the container | Use `./run.sh --download-assets` (it pipes `yes y \| ...`); don't invoke the script directly with `docker run` |
| Files in `test_outputs/` owned by root | `--user` flag missing | `run.sh` adds it automatically; if you wrote your own invocation, add `--user $(id -u):$(id -g)` and `-e HOME=/tmp/robocasa-home` |

-------
## Basic Usage

### Gym wrapper
You can create environments using gym wrappers and run rollouts:
```py
import gymnasium as gym
import robocasa
from robocasa.utils.env_utils import run_random_rollouts

env = gym.make(
    "robocasa/PickPlaceCounterToCabinet",
    split="pretrain", # use 'pretrain' or 'target' kitchen scenes and objects
    seed=0 # seed environment as needed. set seed=None to run unseeded
)

# run rollouts with random actions and save video
run_random_rollouts(
    env, num_rollouts=3, num_steps=100, video_path="/tmp/test.mp4"
)
```

### Play back sample demonstrations of tasks
**(Mac users: for these scripts, prepend the "python" command with "mj": `mjpython ...`)**

Select a task and play back a sample demonstration for the selected task:
```
python -m robocasa.demos.demo_tasks
```

### Explore kitchen scenes
Explore 2500+ kitchen scenes:
```
python -m robocasa.demos.demo_kitchen_scenes
```

### Explore library of 2500+ objects
View and interact with both human-designed and AI-generated objects:
```
python -m robocasa.demos.demo_objects
```
Note: By default, this demo shows objaverse objects. To view AI-generated objects, add the flag `--obj_types aigen`.

### Teleoperate the robot
Control the robot directly, either through a keyboard controller or spacemouse. This script renders the robot semi-translucent in order to minimize occlusions and enable better visibility.
```
python -m robocasa.demos.demo_teleop
```
Note: If using SpaceMouse, you may need to modify the product ID to your appropriate model, setting `SPACEMOUSE_PRODUCT_ID` in `robocasa/macros_private.py`.

-------
## Tasks, datasets, policy learning, and additional use cases
Please refer to the [documentation page](https://robocasa.ai/docs/introduction/overview.html) for information about tasks, datasets, benchmarking, and more.

-------
## Releases
* [2/18/2026] **v1.0**: RoboCasa365 release, with 365 tasks, 2500+ kitchen scenes, 2200+ hours of robot demonstration data, and benchmarking support.
* [10/31/2024] **v0.2**: using RoboSuite `v1.5` as the backend, with improved support for custom robot composition, composite controllers, more teleoperation devices, photo-realistic rendering.

-------
## License
Code: [MIT License](https://opensource.org/license/mit)

Assets and Datasets: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en)

-------
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
