"""
Verify the model is actually using the language prompt:
take the SAME obs and POST /act twice with two different task strings.
If the action chunks differ meaningfully, the model is conditioning on
language. If they don't, the language pathway is still broken.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import urllib.request

import gymnasium as gym
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


def b64_png(arr):
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def post_act(server, payload):
    req = urllib.request.Request(
        f"{server.rstrip('/')}/act",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", default=os.environ.get("GROOT_SERVER", "http://localhost:8500"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = gym.make("robocasa/PickPlaceCounterToSink", split="pretrain", seed=args.seed)
    obs, _ = env.reset()
    env.close()

    # Build common payload (same images + state)
    base = {}
    for cam in GROOT_VIDEO_KEYS:
        img = np.asarray(obs[f"video.{cam}"])
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        base[f"observation.images.{cam}"] = b64_png(img)
    for sk in GROOT_STATE_KEYS:
        base[f"observation.state.{sk}"] = np.asarray(obs[f"state.{sk}"]).astype(float).tolist()

    prompts = [
        ("PnP_sink", "Pick the orange from the counter and place it in the sink."),
        ("microwave", "Open the microwave door."),
        ("identical_repeat", "Pick the orange from the counter and place it in the sink."),
    ]

    results = {}
    for label, task in prompts:
        payload = dict(base)
        payload["task"] = task
        resp = post_act(args.server, payload)
        actions = {k: np.asarray(v, dtype=np.float32) for k, v in resp.items() if k.startswith("action.")}
        results[label] = (task, actions)
        print(f"\n[{label}] task='{task}'")
        for k, a in actions.items():
            print(f"  {k:32s}  mean={np.round(a.mean(axis=0), 4).tolist()}")

    print("\n=== Pairwise diffs (mean L2 over chunk[0]) ===")
    labels = list(results.keys())
    for i, A in enumerate(labels):
        for B in labels[i + 1:]:
            l2 = {}
            for k in results[A][1]:
                a = results[A][1][k][0]   # step 0
                b = results[B][1][k][0]
                l2[k] = float(np.linalg.norm(a - b))
            total = sum(l2.values())
            print(f"  {A:20s} vs {B:20s}  total_L2 = {total:.4f}")
            for k, v in l2.items():
                print(f"    {k:32s} {v:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
