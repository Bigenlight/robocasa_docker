"""
Run a RoboCasa task with a GR00T-N1.5 HTTP server in the loop.

Canonical eval pipeline (matches Isaac-GR00T `gr00t.eval.simulation.run_evaluation`):
    - 3 cameras  (robot0_agentview_left, robot0_agentview_right, robot0_eye_in_hand)
    - 5 state keys + 5 action sub-keys (matches checkpoint metadata.json exactly)
    - n_action_steps = 16, FULL chunk replayed before re-querying
        (verified against gr00t/eval/wrappers/multistep_wrapper.py:207
         `for step in range(self.n_action_steps)` and run_evaluation defaults)
    - Success = OR-aggregate of `info["success"]` across the trial
        (matches simulation.py:141 `current_successes |= bool(env_infos["success"])`
         and multistep_wrapper.py:231 `done = aggregate(self.done, "max")`)
    - per-task max_steps = `get_task_horizon(task)` from
      robocasa.utils.dataset_registry_utils
    - split=pretrain (the multitask checkpoint was trained on pretrain kitchens)

Outputs per call (one task, N rollouts):
    test_outputs/groot_<Task>_seed<N>_success<0|1>.mp4   (one per trial)
    test_outputs/groot_<Task>_summary.json               (rollup)

The GR00T server lives in a separate container (see ../groot_docker_n1.5/) and
follows /home/theo/workspace/VLA_COMMUNICATION_PROTOCOL.md. Both containers
share the host network, so the client reaches the server at
http://localhost:8500.

Quick start (canonical eval, 5 trials of PrepareCoffee):
    ./run.sh --canonical-eval PrepareCoffee --num-rollouts 5
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
import robocasa  # noqa: F401  -- registers gym envs as a side effect
from PIL import Image, ImageDraw, ImageFont

# Sub-key contract: must match groot_docker_n1.5/serve_groot.py and the protocol.
# Verified against /home/theo/workspace/robocasa_docker/groot_docker_n1.5/checkpoint/experiment_cfg/metadata.json
# (5 state, 5 action sub-keys, all matching exactly).
GROOT_ACTION_KEYS = [
    "action.base_motion",
    "action.control_mode",
    "action.end_effector_position",
    "action.end_effector_rotation",
    "action.gripper_close",
]
GROOT_VIDEO_KEYS = [
    # Canonical PandaOmronDataConfig order (data_config.py:641-645):
    # left, right, eye_in_hand. Order is irrelevant for the wire protocol
    # (dict-keyed) but matches server-side ConcatTransform.video_concat_order
    # for grep-ability and to avoid future drift.
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]
GROOT_STATE_KEYS = [
    # Canonical PandaOmronDataConfig state order (data_config.py:646-652):
    # eef_pos, eef_rot, gripper, base_pos, base_rot. Wire-protocol
    # order-irrelevant (dict-keyed) but mirrors server for clarity.
    "end_effector_position_relative",
    "end_effector_rotation_relative",
    "gripper_qpos",
    "base_position",
    "base_rotation",
]

# RoboCasaGymEnv exposes obs with the GR00T-style prefixes:
#   video.robot0_eye_in_hand, video.robot0_agentview_left, video.robot0_agentview_right
#   state.base_position, state.base_rotation, state.end_effector_position_relative,
#   state.end_effector_rotation_relative, state.gripper_qpos
#   annotation.human.task_description
ENV_TO_GROOT_IMAGE = {
    # Canonical order (left, right, eye_in_hand). Iteration order here
    # populates the payload dict; the server reads by key so the order
    # doesn't reach the model — matching anyway for consistency.
    "robot0_agentview_left":  "video.robot0_agentview_left",
    "robot0_agentview_right": "video.robot0_agentview_right",
    "robot0_eye_in_hand":     "video.robot0_eye_in_hand",
}
ENV_TO_GROOT_STATE = {
    "base_position":                    "state.base_position",
    "base_rotation":                    "state.base_rotation",
    "end_effector_position_relative":   "state.end_effector_position_relative",
    "end_effector_rotation_relative":   "state.end_effector_rotation_relative",
    "gripper_qpos":                     "state.gripper_qpos",
}

# Old-name -> canonical-name aliases.  RoboCasa renamed PnP* -> PickPlace*
# but external scripts/READMEs still use the PnP form.  Both gym.make and
# get_task_horizon need the canonical name.
TASK_ALIASES = {
    "PnPCounterToSink":    "PickPlaceCounterToSink",
    "PnPSinkToCounter":    "PickPlaceSinkToCounter",
    "PnPCounterToCab":     "PickPlaceCounterToCab",
    "PnPCabToCounter":     "PickPlaceCabToCounter",
    "PnPCounterToMicrowave": "PickPlaceCounterToMicrowave",
    "PnPMicrowaveToCounter": "PickPlaceMicrowaveToCounter",
    "PnPCounterToStove":    "PickPlaceCounterToStove",
    "PnPStoveToCounter":    "PickPlaceStoveToCounter",
}


def canonical_task(name: str) -> str:
    return TASK_ALIASES.get(name, name)


def lookup_task_horizon(task: str, fallback: int = 720) -> int:
    """Return the per-task canonical horizon, or `fallback` if not registered."""
    try:
        from robocasa.utils.dataset_registry_utils import get_task_horizon
    except Exception:
        return fallback
    try:
        return int(get_task_horizon(task))
    except Exception:
        return fallback


# ─── HTTP helpers ──────────────────────────────────────────────────────────
def http_get_json(url: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def http_post_json(url: str, body: dict, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def b64_png(arr: np.ndarray) -> str:
    """HWC uint8 RGB -> base64 PNG string."""
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def wait_for_health(url: str, timeout_s: float = 60.0) -> dict:
    """Poll /health until status is ok or timeout."""
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            h = http_get_json(url, timeout=2.0)
            if h.get("status") == "ok":
                return h
            print(f"  health: {h}", flush=True)
        except Exception as e:
            last_err = e
        time.sleep(2.0)
    raise RuntimeError(f"server at {url} not ready after {timeout_s}s (last err: {last_err})")


# ─── obs/action conversion ────────────────────────────────────────────────
def env_obs_to_payload(env_obs: dict, task: str) -> dict:
    payload: dict = {"task": task}

    for groot_cam, env_key in ENV_TO_GROOT_IMAGE.items():
        img = env_obs.get(env_key)
        if img is None:
            raise KeyError(f"env obs missing {env_key} (cameras may not be enabled)")
        # robocasa's RoboCasaGymEnv.get_basic_observation already flips the
        # raw mujoco bottom-up frame to top-down (gym_wrapper.py:243-251),
        # so the obs we receive here is already right-side-up — DO NOT flip
        # again or the model sees an inverted kitchen and outputs garbage.
        img = np.asarray(img)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        payload[f"observation.images.{groot_cam}"] = b64_png(img)

    for groot_state, env_key in ENV_TO_GROOT_STATE.items():
        v = env_obs.get(env_key)
        if v is None:
            raise KeyError(f"env obs missing {env_key}")
        payload[f"observation.state.{groot_state}"] = np.asarray(v).astype(float).tolist()

    return payload


def server_response_to_env_action(resp: dict, t_idx: int = 0) -> dict:
    """Take time-step `t_idx` from each [T, dim] sub-key, build env action dict."""
    out = {}
    for k in GROOT_ACTION_KEYS:
        if k not in resp:
            raise KeyError(f"server response missing {k}; got {sorted(resp)}")
        chunk = np.asarray(resp[k], dtype=np.float32)  # [T, dim]
        if chunk.ndim != 2:
            raise ValueError(f"{k} expected 2D [T, dim], got shape {chunk.shape}")
        out[k] = chunk[t_idx]
    return out


def discretize_inplace(env_action: dict) -> None:
    """No-op. RoboCasa's PandaOmronKeyConverter.unmap_action()
    (robocasa/wrappers/gym_wrapper.py:108-127) already does the discretization:

        robot0_right_gripper = -1 if action.gripper_close < 0.5 else 1
        robot0_base_mode     = -1 if action.control_mode  < 0.5 else 1
        robot0_base          = action.base_motion[0:3]   (raw passthrough)
        robot0_torso         = action.base_motion[3:4]   (raw passthrough; train std=0)

    Training-time means are negative (gripper_close=-0.11, control_mode=-0.57),
    typical inference outputs cluster near zero (>0 by 0.07) — well below 0.5,
    so the env's threshold maps them to -1 (open / hold-base) which is what
    the policy intended. The previous client-side `>= 0` threshold was
    inverting those decisions and forcing close-gripper / move-base every
    step. Leave the values raw and let the env do the right thing.
    """
    return


# ─── video composition ────────────────────────────────────────────────────
_LABEL_FONT: ImageFont.ImageFont | None = None


def _label_font() -> ImageFont.ImageFont:
    """Cached default PIL font (~12px). load_default needs no font file."""
    global _LABEL_FONT
    if _LABEL_FONT is None:
        _LABEL_FONT = ImageFont.load_default()
    return _LABEL_FONT


def _draw_label(img: Image.Image, text: str) -> None:
    """Burn `text` top-left in white with a 1px black outline (4-corner trick)."""
    draw = ImageDraw.Draw(img)
    font = _label_font()
    x, y = 4, 4
    # 4-corner black outline
    for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    # white center
    draw.text((x, y), text, font=font, fill=(255, 255, 255))


def _vla_tile(obs: dict, env_key: str, label: str) -> np.ndarray:
    """Return a (256, 256, 3) uint8 RGB tile for one VLA-input camera, with label.

    Pulls obs[env_key] (already top-down — gym wrapper flipped it; do NOT flip
    again).  If missing or wrong-shape, returns a black tile labeled 'missing'.
    """
    img = obs.get(env_key)
    if img is None:
        tile = np.zeros((256, 256, 3), dtype=np.uint8)
        pil = Image.fromarray(tile)
        _draw_label(pil, f"{label} missing")
        return np.asarray(pil)

    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    # Defensive: if upstream ever changes resolution, resize to 256x256.
    if arr.ndim != 3 or arr.shape[:2] != (256, 256) or arr.shape[2] != 3:
        try:
            pil = Image.fromarray(arr).convert("RGB").resize((256, 256))
            arr = np.asarray(pil)
        except Exception:
            arr = np.zeros((256, 256, 3), dtype=np.uint8)

    pil = Image.fromarray(arr.copy())  # copy: don't scribble on env's buffer
    _draw_label(pil, label)
    return np.asarray(pil)


def compose_frame(main_512x768: np.ndarray, obs_dict: dict) -> np.ndarray:
    """Build the 768x768 composite mp4 frame.

    Layout:
        Top    (512x768): `main_512x768` — the human-friendly third-person render.
        Bottom (256x768): wrist | left | right — exactly what the VLA sees,
                          pulled straight from obs["video.<cam>"] (no flip).
    """
    main = np.asarray(main_512x768)
    if main.dtype != np.uint8:
        main = main.astype(np.uint8)
    # Defensive: enforce expected size.
    if main.shape[:2] != (512, 768):
        pil = Image.fromarray(main).convert("RGB").resize((768, 512))
        main = np.asarray(pil)

    wrist = _vla_tile(obs_dict, "video.robot0_eye_in_hand",     "wrist")
    left  = _vla_tile(obs_dict, "video.robot0_agentview_left",  "left")
    right = _vla_tile(obs_dict, "video.robot0_agentview_right", "right")

    bottom = np.concatenate([wrist, left, right], axis=1)  # (256, 768, 3)
    return np.concatenate([main, bottom], axis=0)          # (768, 768, 3)


# ─── one trial ────────────────────────────────────────────────────────────
def run_one_trial(
    args: argparse.Namespace,
    env: gym.Env,
    base_url: str,
    n_action_steps_server: int,
    seed: int,
    video_path: Path,
) -> dict:
    """Run one full episode; return {seed, success, steps_to_success, steps, latencies_ms}."""
    print(f"\n[trial seed={seed}] reset env + server", flush=True)
    obs, info = env.reset(seed=seed)
    http_post_json(f"{base_url}/reset", {}, timeout=10.0)

    # Resolve task description from the env (preferred) — matches what the
    # multitask checkpoint expects.  If absent fall back to the bare task name.
    task_text = obs.get("annotation.human.task_description", canonical_task(args.task))
    if isinstance(task_text, list):
        task_text = task_text[0] if task_text else canonical_task(args.task)
    if not isinstance(task_text, str):
        task_text = canonical_task(args.task)

    writer = imageio.get_writer(str(video_path), fps=20)

    # Number of replay-steps per /act call.  Canonical = full chunk
    # (n_action_steps_server, typically 16); user can override with --replay-chunk.
    replay = min(int(args.replay_chunk), int(n_action_steps_server))
    if replay <= 0:
        replay = 1

    chunk: dict | None = None
    chunk_idx = 0
    success = False
    steps_to_success = -1
    latencies: list[float] = []

    progress_every = max(1, args.max_steps // 20)  # ~20 prints / trial

    try:
        for step in range(args.max_steps):
            if chunk is None or chunk_idx >= replay:
                t0 = time.time()
                payload = env_obs_to_payload(obs, task_text)
                chunk = http_post_json(f"{base_url}/act", payload, timeout=180.0)
                latency = float(chunk.get("latency_ms", (time.time() - t0) * 1000.0))
                latencies.append(latency)
                chunk_idx = 0

            env_action = server_response_to_env_action(chunk, t_idx=chunk_idx)
            chunk_idx += 1
            discretize_inplace(env_action)

            # Canonical robocasa convention: for these stationary manipulation
            # tasks (PnP, atomic) the official run_random_rollouts explicitly
            # zeros base_motion (env_utils.py:166: "zero out the base actions
            # to prevent excessive jitter"). The multitask checkpoint emits
            # small drift values for base_motion[0:2]; left raw, the robot
            # slowly drives out of the kitchen and never approaches the target
            # — visually obvious in the videos. Zero it out to match the
            # canonical convention, leaving EEF + gripper untouched.
            if not args.allow_base_motion:
                env_action["action.base_motion"][:] = 0.0

            obs, reward, terminated, truncated, info = env.step(env_action)

            # Render a wider/taller frame for the mp4 (256x256 is too small
            # to inspect by eye).  Rendering does NOT feed back into the policy.
            sim = env.sim
            frame = sim.render(height=512, width=768, camera_name="robot0_agentview_left")[::-1]
            writer.append_data(compose_frame(frame, obs))

            # Success is OR-accumulated across the trial (canonical pipeline).
            if info.get("success"):
                if not success:
                    steps_to_success = step
                success = True

            if step % progress_every == 0:
                last_lat = latencies[-1] if latencies else float("nan")
                print(
                    f"  step {step:4d}/{args.max_steps}"
                    f"  /act_lat={last_lat:6.1f}ms"
                    f"  success={int(success)}",
                    flush=True,
                )

            if success and args.stop_on_success:
                print(f"  ✓ stop_on_success at step {step}", flush=True)
                break
            if terminated or truncated:
                print(f"  env terminated/truncated at step {step} (terminated={terminated} truncated={truncated})", flush=True)
                break
    finally:
        writer.close()

    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    print(
        f"[trial seed={seed}] done: success={success}"
        f" steps_to_success={steps_to_success}"
        f" mean_latency={mean_lat:.1f}ms"
        f" video={video_path.name} ({video_path.stat().st_size // 1024} KB)",
        flush=True,
    )
    return {
        "seed": int(seed),
        "success": bool(success),
        "steps_to_success": int(steps_to_success),
        "steps": int(step + 1),
        "mean_latency_ms": mean_lat,
        "video": str(video_path),
    }


# ─── eval driver ──────────────────────────────────────────────────────────
def run_eval(args: argparse.Namespace) -> int:
    base_url = args.server.rstrip("/")
    print(f"GR00T server: {base_url}", flush=True)
    health = wait_for_health(f"{base_url}/health", timeout_s=args.health_timeout)
    n_action_steps_server = int(health.get("n_action_steps", 16))
    print(
        f"  ready: model={health.get('model')}"
        f" action_keys={health.get('action_keys')}"
        f" n_action_steps={n_action_steps_server}"
        f" embodiment={health.get('embodiment_tag')}",
        flush=True,
    )

    canon = canonical_task(args.task)
    if canon != args.task:
        print(f"  alias: {args.task} -> {canon} (canonical RoboCasa class)", flush=True)

    # Resolve max_steps: --max-steps wins, else per-task horizon, else 720.
    if args.max_steps is None:
        args.max_steps = lookup_task_horizon(canon, fallback=720)
        print(f"  max_steps from get_task_horizon({canon}) = {args.max_steps}", flush=True)
    else:
        print(f"  max_steps from --max-steps = {args.max_steps}", flush=True)

    # Resolve replay-chunk: 0/None means "full chunk" (canonical).
    if args.replay_chunk is None or args.replay_chunk <= 0:
        args.replay_chunk = n_action_steps_server
        print(f"  replay_chunk = {args.replay_chunk} (canonical full-chunk replay)", flush=True)
    else:
        print(f"  replay_chunk = {args.replay_chunk}", flush=True)

    # Output directory: same as --video parent if given, else test_outputs/.
    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif args.video:
        out_dir = Path(args.video).parent
    else:
        out_dir = Path("/workspace/robocasa/test_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\nCreating env robocasa/{canon} (split={args.split})"
        f"  num_rollouts={args.num_rollouts}  base_seed={args.seed_base}",
        flush=True,
    )
    # Match the canonical robocasa-benchmark/Isaac-GR00T run_eval.py exactly.
    # The LeRobot training data declares 256x256 cameras (info.json) — so use
    # the RoboCasaGymEnv wrapper's default (PandaOmronKeyConverter:256x256). Do
    # not pass generative_textures/randomize_cameras: canonical run_eval.py
    # uses only `enable_render=True` and inherits create_env defaults.
    env = gym.make(
        f"robocasa/{canon}",
        split=args.split,
        seed=args.seed_base,
        enable_render=True,
    )

    trials = []
    try:
        for i in range(args.num_rollouts):
            seed = int(args.seed_base + i)
            # Tentative video path; renamed after we know success/failure.
            tmp_video = out_dir / f"groot_{canon}_seed{seed}_inflight.mp4"
            try:
                trial = run_one_trial(
                    args, env, base_url, n_action_steps_server, seed, tmp_video,
                )
            except Exception as e:
                print(f"[trial seed={seed}] FAILED with exception: {e}", flush=True)
                trial = {
                    "seed": int(seed),
                    "success": False,
                    "steps_to_success": -1,
                    "steps": 0,
                    "mean_latency_ms": 0.0,
                    "video": "",
                    "error": str(e),
                }
            # Rename video to encode success flag in filename.
            if tmp_video.exists():
                final_video = out_dir / f"groot_{canon}_seed{seed}_success{int(trial['success'])}.mp4"
                tmp_video.rename(final_video)
                trial["video"] = str(final_video)
            trials.append(trial)
    finally:
        env.close()

    # Summary.
    successes = [t["success"] for t in trials]
    mean_lats = [t["mean_latency_ms"] for t in trials if t.get("mean_latency_ms")]
    summary = {
        "task": canon,
        "task_arg": args.task,
        "split": args.split,
        "num_rollouts": int(args.num_rollouts),
        "max_steps": int(args.max_steps),
        "replay_chunk": int(args.replay_chunk),
        "n_action_steps_server": int(n_action_steps_server),
        "server": base_url,
        "trials": trials,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "num_success": int(sum(successes)),
        "mean_latency_ms": float(np.mean(mean_lats)) if mean_lats else 0.0,
    }
    summary_path = out_dir / f"groot_{canon}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 70)
    print(f"DONE  task={canon}  success={summary['num_success']}/{summary['num_rollouts']}"
          f"  rate={summary['success_rate']:.2f}"
          f"  mean_lat={summary['mean_latency_ms']:.1f}ms")
    print(f"Summary: {summary_path}")
    for t in trials:
        flag = "✓" if t["success"] else "✗"
        print(f"  {flag} seed={t['seed']:>4}  steps={t['steps']:>4}"
              f"  steps_to_success={t['steps_to_success']:>4}"
              f"  -> {Path(t['video']).name if t.get('video') else '(no video)'}")
    print("=" * 70)

    # Backwards-compat: if user passed --video and we ran exactly 1 trial, also
    # symlink/copy the single trial mp4 to the requested path.
    if args.video and len(trials) == 1 and trials[0].get("video"):
        target = Path(args.video)
        try:
            if target != Path(trials[0]["video"]):
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() or target.is_symlink():
                    target.unlink()
                # Copy rather than symlink so it survives container teardown.
                target.write_bytes(Path(trials[0]["video"]).read_bytes())
                print(f"Also wrote: {target}")
        except Exception as e:
            print(f"  (couldn't copy to --video target: {e})")

    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="PrepareCoffee")
    p.add_argument("--split", default="pretrain", choices=["pretrain", "target", "all"])
    # New canonical-eval interface.
    p.add_argument("--num-rollouts", type=int, default=1,
                   help="number of independent trials (each with seed = seed-base + i)")
    p.add_argument("--seed-base", type=int, default=0,
                   help="first trial uses seed=seed-base, second seed-base+1, etc.")
    p.add_argument("--max-steps", type=int, default=None,
                   help="max env steps per trial; default = get_task_horizon(task)")
    p.add_argument("--replay-chunk", type=int, default=16,
                   help="how many steps of an action chunk to replay before re-querying. "
                        "Default = 16 = full chunk, matching robocasa-benchmark/Isaac-GR00T "
                        "scripts/run_eval.py (MultiStepConfig.n_action_steps=16). For tighter "
                        "closed-loop pass smaller (1 = fully closed-loop). 0/None also resolves "
                        "to full chunk.")
    p.add_argument("--stop-on-success", action="store_true", default=True,
                   help="end trial on first success step (default true; canonical pipeline does the same via terminated/done)")
    p.add_argument("--no-stop-on-success", dest="stop_on_success", action="store_false")
    p.add_argument("--server", default=os.environ.get("GROOT_SERVER", "http://localhost:8500"))
    p.add_argument("--video", default=None,
                   help="backwards-compat single-trial video path. Per-trial videos are always "
                        "written to <output_dir>/groot_<Task>_seed<N>_success<0|1>.mp4")
    p.add_argument("--output-dir", default=None,
                   help="directory for per-trial mp4s + summary.json. "
                        "Defaults to dirname(--video) or /workspace/robocasa/test_outputs")
    p.add_argument("--health-timeout", type=float, default=60.0)
    p.add_argument("--allow-base-motion", action="store_true", default=False,
                   help="Pass model's base_motion through raw. Default OFF for "
                        "atomic / PnP-class tasks: training demos for these "
                        "spawned the robot at the target fixture (kitchen_pick_place.py:280 "
                        "init_robot_base_ref=self.sink) and the user's training "
                        "video shows zero base translation across the 21s episode. "
                        "The pooled multitask metadata.json action.base_motion stats "
                        "(std 0.17–0.28) include NavigateKitchen episodes which "
                        "are not relevant for PnP eval — pass --allow-base-motion "
                        "ON only when evaluating mobility tasks.")
    p.add_argument("--no-base-motion", dest="allow_base_motion", action="store_false",
                   help="zero base_motion at every step (default for atomic/PnP)")
    # Legacy alias kept for older callers (run.sh --groot-eval still passes --num-steps).
    p.add_argument("--num-steps", type=int, default=None,
                   help="legacy alias for --max-steps")
    p.add_argument("--seed", type=int, default=None,
                   help="legacy alias for --seed-base")

    args = p.parse_args()
    if args.num_steps is not None and args.max_steps is None:
        args.max_steps = args.num_steps
    if args.seed is not None:
        args.seed_base = args.seed

    return run_eval(args)


if __name__ == "__main__":
    sys.exit(main())
