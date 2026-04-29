"""
Diagnostic: dump exactly what the VLA sees and what it returns at step 0.

Outputs (test_outputs/diag/):
    cam_<name>_raw.png        - directly from env (no flip)
    cam_<name>_flipped.png    - after [::-1] flip (current pipeline)
    action_chunk.json         - the 16-step chunk from /act (raw values)
    action_summary.json       - per-sub-key min/max/mean/std + base_mode dist

Usage (inside the robocasa container, GR00T server already up at $GROOT_SERVER):
    python examples/diag_groot_obs.py --task PickPlaceCounterToSink --seed 0
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
import robocasa  # noqa: F401
from PIL import Image

GROOT_VIDEO_KEYS = [
    # Match server's canonical PandaOmronDataConfig order (left, right, wrist).
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]
GROOT_STATE_KEYS = [
    "base_position",
    "base_rotation",
    "end_effector_position_relative",
    "end_effector_rotation_relative",
    "gripper_qpos",
]


def b64_png(arr: np.ndarray) -> str:
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="PickPlaceCounterToSink")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--server", default=os.environ.get("GROOT_SERVER", "http://localhost:8500"))
    p.add_argument("--output-dir", default="/workspace/robocasa/test_outputs/diag")
    # NOTE: gym wrapper already returns top-down images. Default is to NOT flip.
    # Pass --flip to reproduce the (broken) old behavior for comparison.
    p.add_argument("--flip", action="store_true",
                   help="Apply an extra [::-1] flip — for testing the (broken) old behavior")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = gym.make(f"robocasa/{args.task}", split="pretrain", seed=args.seed)
    obs, info = env.reset()

    print("\n=== ENV obs keys ===")
    for k in sorted(obs.keys()):
        v = obs[k]
        try:
            shape = np.asarray(v).shape
            dtype = np.asarray(v).dtype
            print(f"  {k:60s} shape={shape}  dtype={dtype}")
        except Exception:
            print(f"  {k:60s} type={type(v).__name__}")

    print("\n=== Saving cameras (raw + flipped) ===")
    for cam in GROOT_VIDEO_KEYS:
        env_key = f"video.{cam}"
        img = np.asarray(obs[env_key])
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        imageio.imwrite(str(out / f"cam_{cam}_raw.png"), img)
        imageio.imwrite(str(out / f"cam_{cam}_flipped.png"), img[::-1])
        print(f"  {cam}: raw + flipped saved  ({img.shape})")

    print("\n=== ENV state values ===")
    for sk in GROOT_STATE_KEYS:
        v = obs.get(f"state.{sk}")
        if v is not None:
            arr = np.asarray(v)
            print(f"  state.{sk:40s} = {arr.tolist()}")

    # Build /act payload
    payload = {
        "task": obs.get("annotation.human.task_description", args.task),
    }
    if isinstance(payload["task"], list) and payload["task"]:
        payload["task"] = payload["task"][0]
    if not isinstance(payload["task"], str):
        payload["task"] = args.task

    for cam in GROOT_VIDEO_KEYS:
        img = np.asarray(obs[f"video.{cam}"])
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        if args.flip:
            img = img[::-1]
        payload[f"observation.images.{cam}"] = b64_png(img)

    for sk in GROOT_STATE_KEYS:
        v = obs[f"state.{sk}"]
        payload[f"observation.state.{sk}"] = np.asarray(v).astype(float).tolist()

    print(f"\n=== POST /act (no_flip={args.flip}, task='{payload['task']}') ===")
    req = urllib.request.Request(
        f"{args.server.rstrip('/')}/act",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=120).read())

    # Save raw chunk
    chunk_dump = {k: v for k, v in resp.items() if k.startswith("action.")}
    chunk_dump["latency_ms"] = resp.get("latency_ms")
    with open(out / "action_chunk.json", "w") as f:
        json.dump(chunk_dump, f, indent=2)

    # Per-sub-key stats
    summary: dict = {"task": args.task, "seed": args.seed, "no_flip": args.flip,
                     "latency_ms": resp.get("latency_ms")}
    print("\n=== Action chunk stats (over 16 steps) ===")
    for k in [
        "action.base_motion",
        "action.control_mode",
        "action.end_effector_position",
        "action.end_effector_rotation",
        "action.gripper_close",
    ]:
        chunk = np.asarray(resp[k], dtype=np.float32)  # [T, dim]
        stats = {
            "shape": list(chunk.shape),
            "min": chunk.min(axis=0).tolist(),
            "max": chunk.max(axis=0).tolist(),
            "mean": chunk.mean(axis=0).tolist(),
            "std": chunk.std(axis=0).tolist(),
        }
        summary[k] = stats
        print(f"  {k}  shape={chunk.shape}")
        print(f"    mean={np.round(chunk.mean(axis=0), 4).tolist()}")
        print(f"    std ={np.round(chunk.std(axis=0), 4).tolist()}")
        print(f"    min ={np.round(chunk.min(axis=0), 4).tolist()}")
        print(f"    max ={np.round(chunk.max(axis=0), 4).tolist()}")

    # Special interpretation: base_motion[3] (base_mode) — should be ±1 binary
    # in the trained distribution (toggle to enable/disable mobile base).
    bm = np.asarray(resp["action.base_motion"], dtype=np.float32)[:, 3]
    print(f"\n=== base_mode (base_motion[3]) over 16 steps ===")
    print(f"  values: {np.round(bm, 3).tolist()}")
    print(f"  >= 0 fraction: {(bm >= 0).mean():.2f}  (1.0 = always-move, 0.0 = always-hold)")
    summary["base_mode_pos_frac"] = float((bm >= 0).mean())
    summary["base_mode_values"] = bm.tolist()

    with open(out / "action_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    env.close()
    print(f"\nDone. Open the PNGs visually to confirm orientation:")
    print(f"  {out}/cam_<name>_raw.png       (env native, no flip)")
    print(f"  {out}/cam_<name>_flipped.png   (after [::-1] flip — what we currently send)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
