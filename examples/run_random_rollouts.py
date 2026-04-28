"""
Drive a robocasa task with random actions and write the result to mp4.

This is the minimal "is the simulation visualizable?" demo. It mirrors the
README usage example with a CLI wrapper so the docker run.sh can call it.

Example:
    python examples/run_random_rollouts.py \
        --task PnPCounterToSink \
        --num-rollouts 1 --num-steps 60 \
        --video test_outputs/PnPCounterToSink.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import gymnasium as gym
import robocasa  # noqa: F401  -- import for gym.register side effects
from robocasa.utils.env_utils import run_random_rollouts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, help="robocasa task name, e.g. PnPCounterToSink")
    p.add_argument("--split", default="pretrain", choices=["pretrain", "target", "all"])
    p.add_argument("--num-rollouts", type=int, default=1)
    p.add_argument("--num-steps", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--video", required=True, help="output mp4 path")
    p.add_argument(
        "--camera",
        default="robot0_agentview_left",
        help=(
            "camera name passed to env.sim.render. The default RoboCasaGymEnv"
            " registers left/right/eye_in_hand cameras only — robot0_agentview_center"
            " from the readme is NOT in that default list."
        ),
    )
    args = p.parse_args()

    out = Path(args.video)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"task        = {args.task}")
    print(f"split       = {args.split}")
    print(f"seed        = {args.seed}")
    print(f"rollouts    = {args.num_rollouts} × {args.num_steps} steps")
    print(f"MUJOCO_GL   = {os.environ.get('MUJOCO_GL', '<unset>')}")
    print(f"camera      = {args.camera}")
    print(f"video       = {out}")

    env = gym.make(f"robocasa/{args.task}", split=args.split, seed=args.seed)
    try:
        info = run_random_rollouts(
            env,
            num_rollouts=args.num_rollouts,
            num_steps=args.num_steps,
            video_path=str(out),
            camera_name=args.camera,
        )
    finally:
        env.close()

    n_succ = info.get("num_success_rollouts", 0)
    print(f"\nDone. {n_succ}/{args.num_rollouts} rollouts succeeded.")
    print(f"Video at: {out}  ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
