"""
RoboCasa container smoke test.

Runs inside the docker image. Validates that the headless rendering stack works
end-to-end and that we can drive a rollout that produces an mp4. Each step is
isolated so a failure in step N doesn't poison step N+1.

Steps:
  1. import-time sanity (numpy / mujoco / robosuite / robocasa import paths)
  2. render-backend probe (MUJOCO_GL active, GL context creates without error)
  3. gym.make + reset on a simple robocasa task (PickPlaceCounterToSink by default)
  4. PNG render of the agentview camera
  5. 10 random sim steps (no crash, reward/done structure correct)
  6. mp4 rollout via robocasa.utils.env_utils.run_random_rollouts

Outputs land in --output-dir (defaults to /workspace/robocasa/test_outputs).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
INFO = "\033[94m[ ]\033[0m"


def banner(n: int, name: str) -> None:
    print(f"\n{INFO} step {n}/6: {name}")


@contextmanager
def closing_env(env):
    try:
        yield env
    finally:
        try:
            env.close()
        except Exception:
            pass


def step_1_imports() -> None:
    import numpy
    import mujoco
    import gymnasium

    print(f"  python      = {sys.version.split()[0]}")
    print(f"  numpy       = {numpy.__version__}")
    print(f"  mujoco      = {mujoco.__version__}")
    print(f"  gymnasium   = {gymnasium.__version__}")

    # robosuite + robocasa are mounted via PYTHONPATH; verify they resolve.
    import robosuite
    import robocasa

    print(f"  robosuite   = {robosuite.__version__}  ({os.path.dirname(robosuite.__file__)})")
    print(f"  robocasa    = {robocasa.__version__}  ({os.path.dirname(robocasa.__file__)})")


def step_2_render_probe() -> None:
    backend = os.environ.get("MUJOCO_GL", "<unset>")
    print(f"  MUJOCO_GL   = {backend}")

    # Allocate a tiny mujoco GL context to confirm the backend actually works.
    # If EGL is requested but no /dev/dri is wired through, this will raise.
    import mujoco

    xml = """
    <mujoco>
      <worldbody>
        <geom type="plane" size="1 1 0.1"/>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=64, width=64)
    renderer.update_scene(data)
    img = renderer.render()
    assert img.shape == (64, 64, 3), f"unexpected render shape: {img.shape}"
    print(f"  context     = ok   (rendered 64x64x3)")


def _check_assets_present() -> None:
    """Surface the most common first-run failure with an actionable message."""
    import robocasa

    base = Path(robocasa.__path__[0]) / "models" / "assets"
    needed = [
        base / "objects" / "lightwheel",
        base / "fixtures" / "windows",
        base / "textures",
    ]
    missing = [p for p in needed if not p.exists()]
    if missing:
        msg = (
            "\nRoboCasa assets are not downloaded. The simulator will fail to load fixtures.\n"
            f"  missing: {[str(p) for p in missing]}\n"
            "  fix:     ./run.sh --download-assets   (one-time, ~10GB)\n"
        )
        raise FileNotFoundError(msg)


def step_3_make_env(task: str):
    import robocasa  # registers gym envs as a side effect of import
    import gymnasium as gym

    _check_assets_present()
    print(f"  task        = {task}")
    env = gym.make(f"robocasa/{task}", split="pretrain", seed=0)
    obs, info = env.reset()
    assert isinstance(obs, dict), f"reset() should return dict obs, got {type(obs)}"
    print(f"  obs keys    = {sorted(obs.keys())[:6]}{'...' if len(obs) > 6 else ''}")
    print(f"  action_space= {type(env.action_space).__name__}")
    return env


def step_4_render_png(env, output_dir: Path) -> Path:
    import imageio

    # The robosuite env is reachable through env.sim. gymnasium 0.29
    # Wrapper.__getattr__ forwards unknown attributes through to the inner
    # RoboCasaGymEnv, which exposes `.sim`. We pick robot0_agentview_left
    # because that's a default camera registered by RoboCasaGymEnv (the
    # readme's robot0_agentview_center is NOT in the gym wrapper's default
    # camera_names list — see PandaOmronKeyConverter.get_camera_config).
    sim = env.sim
    img = sim.render(height=512, width=768, camera_name="robot0_agentview_left")[::-1]
    out = output_dir / "smoke_agentview.png"
    imageio.imwrite(str(out), img)
    print(f"  wrote        {out}  ({img.shape})")
    return out


def step_5_sim_steps(env, n: int = 10) -> None:
    import numpy as np

    for i in range(n):
        action = env.action_space.sample()
        # Zero base motion so the rollout doesn't drift across the kitchen.
        if isinstance(action, dict) and "action.base_motion" in action:
            action["action.base_motion"][:] = 0.0
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, (int, float, np.floating)), f"bad reward type: {type(reward)}"
    print(f"  {n} steps    = ok   (last reward={float(reward):.3f}, success={info.get('success')})")


def step_6_rollout_video(task: str, output_dir: Path) -> Path:
    """
    Drive the canonical robocasa entrypoint and confirm an mp4 lands on disk.
    Uses a fresh env so failures upstream don't affect this measurement.
    """
    import robocasa
    import gymnasium as gym
    from robocasa.utils.env_utils import run_random_rollouts

    env = gym.make(f"robocasa/{task}", split="pretrain", seed=1)
    out = output_dir / f"smoke_{task}.mp4"
    with closing_env(env):
        run_random_rollouts(
            env,
            num_rollouts=1,
            num_steps=30,
            video_path=str(out),
            camera_name="robot0_agentview_left",
        )
    assert out.exists() and out.stat().st_size > 0, f"video not written: {out}"
    print(f"  wrote        {out}  ({out.stat().st_size // 1024} KB)")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="/workspace/robocasa/test_outputs")
    p.add_argument("--task", default="PickPlaceCounterToSink")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    results: list[tuple[int, str, bool, str]] = []

    # 1
    banner(1, "imports")
    try:
        step_1_imports()
        results.append((1, "imports", True, ""))
    except Exception:
        results.append((1, "imports", False, traceback.format_exc()))
        print(traceback.format_exc())
        # imports failing is fatal — nothing else can run.
        return _summary(results, started)

    # 2
    banner(2, "render-backend probe")
    try:
        step_2_render_probe()
        results.append((2, "render-backend probe", True, ""))
    except Exception:
        results.append((2, "render-backend probe", False, traceback.format_exc()))
        print(traceback.format_exc())
        # Without a working render context the rest is pointless.
        return _summary(results, started)

    # 3 + 4 + 5 share an env. If 3 fails we skip 4/5.
    env = None
    banner(3, "gym.make + reset")
    try:
        env = step_3_make_env(args.task)
        results.append((3, "gym.make + reset", True, ""))
    except Exception:
        results.append((3, "gym.make + reset", False, traceback.format_exc()))
        print(traceback.format_exc())

    if env is not None:
        try:
            banner(4, "PNG render")
            step_4_render_png(env, output_dir)
            results.append((4, "PNG render", True, ""))
        except Exception:
            results.append((4, "PNG render", False, traceback.format_exc()))
            print(traceback.format_exc())

        try:
            banner(5, "10 random sim steps")
            step_5_sim_steps(env, n=10)
            results.append((5, "10 random sim steps", True, ""))
        except Exception:
            results.append((5, "10 random sim steps", False, traceback.format_exc()))
            print(traceback.format_exc())

        try:
            env.close()
        except Exception:
            pass

    # 6 uses a fresh env.
    banner(6, "rollout → mp4")
    try:
        step_6_rollout_video(args.task, output_dir)
        results.append((6, "rollout → mp4", True, ""))
    except Exception:
        results.append((6, "rollout → mp4", False, traceback.format_exc()))
        print(traceback.format_exc())

    return _summary(results, started)


def _summary(results: list[tuple[int, str, bool, str]], started: float) -> int:
    elapsed = time.time() - started
    total = len(results)
    passed = sum(1 for _, _, ok, _ in results if ok)
    print("")
    print("=" * 60)
    print(f"Smoke test: {passed}/{total} passed in {elapsed:.1f}s")
    for n, name, ok, _ in results:
        tag = PASS if ok else FAIL
        print(f"  [{tag}] step {n}: {name}")
    print("=" * 60)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
