"""
In-process Diffusion Policy eval on a SINGLE RoboCasa task.

Single-task analog of robocasa-benchmark/diffusion_policy/eval_robocasa.py
without the task-set indirection. Lets you eval one task (e.g.
PickPlaceCounterToSink) with a small num-rollouts to verify the install or
get a quick number, instead of always running 18+ tasks via the soup
abstraction.

Architecture (from RoboCasa365 paper + the released checkpoint config.yaml):
    Diffusion Transformer Hybrid (12-layer, n_emb=512, 8-head, FiLM lang)
    3x 256x256 RGB cameras (left, right, eye_in_hand) + ResNet18 GroupNorm
    state: eef_pos(3) + eef_quat(4) + gripper_qpos(2) + lang_emb(768 CLIP)
    n_obs_steps=2, n_action_steps=8, horizon=10
    DDPM 100 inference steps  (slow on small GPUs; ~300-700ms/chunk on 3060)
    action_dim=12 (5 sub-keys, identical layout to GR00T-N1.5)

Action layout (12-d flat -> RoboCasa env action dict).  Source of truth:
DP cfg.shape_meta.action.lerobot_keys ORDER (NOT GR00T's metadata.json).
LerobotCotrainingDataset (lerobot_dataset.py:190-200) concatenates action
sub-keys in the EXACT order listed in cfg.shape_meta.action.lerobot_keys.
Verified from the released ckpt's cfg.shape_meta.action.lerobot_keys =
[end_effector_position, end_effector_rotation, gripper_close, base_motion,
control_mode]. Sub-shapes: pos=3, rot=3, grip=1, base=4, mode=1 (sum=12).
Note: this is DIFFERENT from GR00T's modality.json layout — the two models
were trained with different action concat orders despite using the SAME
underlying lerobot dataset.

    [0:3]   action.end_effector_position
    [3:6]   action.end_effector_rotation
    [6:7]   action.gripper_close
    [7:11]  action.base_motion           (zeroed by default; --allow-base-motion overrides)
    [11:12] action.control_mode

Outputs per call (one task, N rollouts):
    <output_dir>/dp_<Task>_seed<N>_success<0|1>.mp4
    <output_dir>/dp_<Task>_summary.json

This wrapper deliberately bypasses cfg.task.env_runner (which is a robomimic
env_runner targeting the chi2023 test wrappers) and drives the env directly
via gym.make("robocasa/<Task>"). The policy is loaded in-process from the
Lightning .ckpt with torch.load+dill.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np


# -- canonical action-slice table ------------------------------------------
DP_ACTION_LAYOUT = [
    ("action.end_effector_position",  0,  3),
    ("action.end_effector_rotation",  3,  6),
    ("action.gripper_close",          6,  7),
    ("action.base_motion",            7, 11),
    ("action.control_mode",          11, 12),
]


# -- task aliases (PnP* -> PickPlace*; mirror examples/run_groot_eval.py) --
TASK_ALIASES = {
    "PnPCounterToSink":      "PickPlaceCounterToSink",
    "PnPSinkToCounter":      "PickPlaceSinkToCounter",
    "PnPCounterToCab":       "PickPlaceCounterToCab",
    "PnPCabToCounter":       "PickPlaceCabToCounter",
    "PnPCounterToMicrowave": "PickPlaceCounterToMicrowave",
    "PnPMicrowaveToCounter": "PickPlaceMicrowaveToCounter",
    "PnPCounterToStove":     "PickPlaceCounterToStove",
    "PnPStoveToCounter":     "PickPlaceStoveToCounter",
}


def canonical_task(name: str) -> str:
    return TASK_ALIASES.get(name, name)


# ─── argparse (kept above heavy imports so --help works without torch) ────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="In-process Diffusion Policy eval on a single RoboCasa task."
    )
    p.add_argument("--ckpt", required=True, help="Path to a DP Lightning .ckpt (e.g. latest.ckpt).")
    p.add_argument("--task", default=None,
                   help="RoboCasa env name (e.g. PickPlaceCounterToSink). "
                        "Required unless --smoke-only is set.")
    p.add_argument("--split", default="pretrain", choices=["pretrain", "target"])
    p.add_argument("--num-rollouts", type=int, default=5)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=None,
                   help="Per-trial step cap. Default: get_task_horizon(task) * 1.5.")
    p.add_argument("--num-envs", type=int, default=1,
                   help="Sequential rollouts only for now; >1 will assert.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", default=None,
                   help="Default: ./test_outputs/dp_<task>/")
    p.add_argument("--smoke-only", action="store_true",
                   help="Load ckpt + build policy, print resolved targets, exit.")
    p.add_argument("--allow-base-motion", action="store_true", default=False,
                   help="Pass model's base_motion through raw. Default OFF for atomic/PnP.")
    return p.parse_args()


# ─── lazy-loaded heavy imports ────────────────────────────────────────────
def lazy_imports() -> dict:
    """Import torch/dill/hydra/etc. only after argparse, so --help works without them."""
    import gymnasium as gym  # noqa: F401
    import imageio
    import torch
    import dill
    import hydra
    from omegaconf import OmegaConf
    import robocasa  # noqa: F401  -- side-effect: registers gym envs
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    return {
        "gym": gym, "imageio": imageio, "torch": torch, "dill": dill,
        "hydra": hydra, "OmegaConf": OmegaConf, "get_task_horizon": get_task_horizon,
    }


# ─── ckpt + policy load ───────────────────────────────────────────────────
def load_workspace_and_policy(ckpt_path: str, output_dir: str, device: str, deps: dict):
    """Mirror eval_robocasa.py's ckpt-load body. Returns (cfg, workspace, policy)."""
    torch = deps["torch"]; dill = deps["dill"]; hydra = deps["hydra"]; OmegaConf = deps["OmegaConf"]
    print(f"Loading checkpoint: {ckpt_path}", flush=True)
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    cfg = copy.deepcopy(OmegaConf.to_container(cfg, resolve=False))

    # Prevent ckpt's training-time dataset paths from being instantiated.
    if isinstance(cfg.get("task"), dict) and "dataset" in cfg["task"]:
        cfg["task"]["dataset"] = None
    if isinstance(cfg.get("training"), dict) and "dataset_dir" in cfg["training"]:
        cfg["training"]["dataset_dir"] = "/tmp/unused"

    cfg = OmegaConf.create(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    use_ema = bool(OmegaConf.select(cfg, "training.use_ema", default=False))
    policy = workspace.ema_model if use_ema else workspace.model
    policy.to(torch.device(device))
    policy.eval()
    return cfg, workspace, policy


def resolve_obs_keys(cfg, deps: dict) -> dict:
    """Walk shape_meta.obs (try a few cfg paths) to learn what obs keys DP wants."""
    OmegaConf = deps["OmegaConf"]
    shape_meta = None
    for path in ("policy.shape_meta", "task.shape_meta", "shape_meta"):
        sm = OmegaConf.select(cfg, path)
        if sm is not None:
            shape_meta = sm
            break
    if shape_meta is None:
        raise RuntimeError(
            "Could not locate shape_meta in cfg. Tried cfg.policy.shape_meta, "
            "cfg.task.shape_meta, cfg.shape_meta."
        )
    obs_meta = shape_meta["obs"]
    out = {}
    for k in list(obs_meta.keys()):
        attr = obs_meta[k]
        out[k] = {
            "shape": [int(x) for x in list(attr.get("shape"))],
            "type":  str(attr.get("type", "low_dim")),
        }
    return out


# ─── env-obs -> DP-obs conversion ─────────────────────────────────────────
def find_env_key_for_dp(dp_key: str, env_obs: dict) -> str | None:
    """Try several namespacing conventions to map a DP shape_meta.obs key to an env key."""
    if dp_key in env_obs:
        return dp_key
    for prefix in ("video.", "state."):
        cand = f"{prefix}{dp_key}"
        if cand in env_obs:
            return cand
    cand = f"{dp_key}_image"
    if cand in env_obs:
        return cand
    # DP cam keys: "robot0_agentview_right_image" -> env "video.robot0_agentview_right"
    if dp_key.endswith("_image"):
        stem = dp_key[: -len("_image")]
        cand = f"video.{stem}"
        if cand in env_obs:
            return cand
        if stem in env_obs:
            return stem
    # DP base-relative eef shorthand -> env's *_relative
    if dp_key in ("robot0_base_to_eef_pos", "eef_pos") and "state.end_effector_position_relative" in env_obs:
        return "state.end_effector_position_relative"
    if dp_key in ("robot0_base_to_eef_quat", "eef_quat") and "state.end_effector_rotation_relative" in env_obs:
        return "state.end_effector_rotation_relative"
    # DP gripper qpos -> env state.gripper_qpos
    if dp_key in ("robot0_gripper_qpos", "gripper_qpos") and "state.gripper_qpos" in env_obs:
        return "state.gripper_qpos"
    return None


# Module-level cache for the LangEncoder. Keyed by device.
# Avoids the ~600 MB re-load on every encode call.
_LANG_ENCODER_CACHE: dict = {}


def get_lang_encoder(device: str):
    """Return a cached `robomimic.utils.lang_utils.LangEncoder` instance.
    This is the EXACT encoder the DP model was trained against (chi2023 fork
    + robocasa@robocasa branch). It uses CLIPTextModelWithProjection with
    `text_embeds` pooling and padding='max_length' (77 tokens) — different
    from a vanilla `CLIPTextModel.pooler_output`."""
    if device not in _LANG_ENCODER_CACHE:
        from robomimic.utils.lang_utils import LangEncoder
        _LANG_ENCODER_CACHE[device] = LangEncoder(device=device)
    return _LANG_ENCODER_CACHE[device]


def encode_lang(text: str, dim: int, device: str) -> np.ndarray:
    """Encode a task description with robomimic's LangEncoder, matching training."""
    try:
        enc = get_lang_encoder(device)
        v = enc.get_lang_emb(text).detach().cpu().numpy().astype(np.float32)
        if v.shape[0] != dim:
            buf = np.zeros((dim,), dtype=np.float32)
            n = min(v.shape[0], dim)
            buf[:n] = v[:n]
            v = buf
        return v
    except Exception as e:
        print(f"  WARNING: lang encoding failed ({type(e).__name__}: {e}); using zeros", flush=True)
        return np.zeros((dim,), dtype=np.float32)


def env_obs_to_dp_dict(env_obs: dict, obs_keys: dict, lang_emb: np.ndarray) -> dict:
    """One-step env obs -> dict matching DP shape_meta.obs (numpy arrays only)."""
    out: dict[str, np.ndarray] = {}
    for dp_key, info in obs_keys.items():
        ty = info["type"]
        target_shape = info["shape"]

        if "lang" in dp_key.lower():
            buf = np.zeros((target_shape[0],), dtype=np.float32)
            n = min(lang_emb.shape[0], target_shape[0])
            buf[:n] = lang_emb[:n]
            out[dp_key] = buf
            continue

        env_key = find_env_key_for_dp(dp_key, env_obs)
        if env_key is None:
            raise KeyError(
                f"DP wants obs key {dp_key!r} but env has none of "
                f"{[dp_key, f'video.{dp_key}', f'state.{dp_key}', f'{dp_key}_image']!r}. "
                f"env keys: {sorted(env_obs)}"
            )

        if ty == "rgb":
            img = np.asarray(env_obs[env_key])
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            # Resize if shape_meta declares a different resolution.
            if len(target_shape) == 3 and target_shape[0] == 3:
                H, W = target_shape[1], target_shape[2]
            else:
                H, W = target_shape[0], target_shape[1]
            if img.shape[0] != H or img.shape[1] != W:
                try:
                    from PIL import Image as _PIL
                    img = np.asarray(_PIL.fromarray(img).resize((W, H)))
                except Exception:
                    # Skip resize; let the encoder error informatively.
                    pass
            if len(target_shape) == 3 and target_shape[0] == 3:
                # CHW float [0,1] (modern robomimic convention)
                img = img.astype(np.float32).transpose(2, 0, 1) / 255.0
            out[dp_key] = img
            continue

        # low_dim state
        v = np.asarray(env_obs[env_key]).astype(np.float32).reshape(-1)
        if target_shape and v.shape[0] != target_shape[0]:
            print(f"  WARNING: state {dp_key!r} env shape {v.shape[0]} != DP shape {target_shape[0]}",
                  flush=True)
        out[dp_key] = v
    return out


def stack_history(history: deque, key: str, n_obs_steps: int) -> np.ndarray:
    """Repeat the earliest obs to fill the buffer if we have <n_obs_steps observations."""
    pad = n_obs_steps - len(history)
    items = list(history)
    if pad > 0:
        items = [items[0]] * pad + items
    else:
        items = items[-n_obs_steps:]
    return np.stack([d[key] for d in items], axis=0)  # (To, ...)


def build_obs_batch(history: deque, obs_keys: dict, n_obs_steps: int, device: str, deps: dict) -> dict:
    torch = deps["torch"]
    batch = {}
    for k in obs_keys:
        stacked = stack_history(history, k, n_obs_steps)
        batch[k] = torch.from_numpy(stacked).unsqueeze(0).to(device)  # (1, To, ...)
    return batch


def slice_action(flat: np.ndarray) -> dict:
    """Flat (12,) -> {action.<sub>: ndarray}.  See module docstring for slice table."""
    out = {}
    for sub_key, lo, hi in DP_ACTION_LAYOUT:
        out[sub_key] = flat[lo:hi].astype(np.float32).copy()
    return out


# ─── one trial ────────────────────────────────────────────────────────────
def run_one_trial(args: argparse.Namespace, env, policy, cfg, obs_keys: dict,
                  lang_emb: np.ndarray, video_path: Path, deps: dict) -> dict:
    torch = deps["torch"]; imageio = deps["imageio"]; OmegaConf = deps["OmegaConf"]
    n_obs_steps = int(OmegaConf.select(cfg, "policy.n_obs_steps", default=2))
    n_action_steps = int(OmegaConf.select(cfg, "policy.n_action_steps", default=8))

    seed = args._current_seed
    obs, info = env.reset(seed=seed)
    # Extract per-rollout language description from env obs (matches what
    # robomimic_image_wrapper does: self.lang = raw_obs["annotation.human.task_description"]).
    # Fall back to the python class name if the env doesn't expose one.
    lang_dim = lang_emb.shape[0]
    task_text = obs.get("annotation.human.task_description", args.task)
    if isinstance(task_text, (bytes, np.ndarray)):
        try:
            task_text = task_text.item() if hasattr(task_text, "item") else task_text.decode()
        except Exception:
            task_text = str(task_text)
    if task_text != args.task:
        print(f"  task_description: {task_text!r}", flush=True)
    lang_emb = encode_lang(str(task_text), lang_dim, args.device)
    history: deque = deque(maxlen=n_obs_steps)
    history.append(env_obs_to_dp_dict(obs, obs_keys, lang_emb))

    writer = imageio.get_writer(str(video_path), fps=20, codec="libx264")
    success = False
    steps_to_success = -1
    latencies: list[float] = []
    chunk: np.ndarray | None = None
    chunk_idx = 0
    progress_every = max(1, args.max_steps // 20)
    step = 0
    try:
        for step in range(args.max_steps):
            if chunk is None or chunk_idx >= n_action_steps:
                t0 = time.time()
                obs_batch = build_obs_batch(history, obs_keys, n_obs_steps, args.device, deps)
                with torch.no_grad():
                    result = policy.predict_action(obs_batch)
                action_t = result["action"]  # (1, n_action_steps, 12)
                chunk = action_t[0].detach().cpu().numpy().astype(np.float32)
                chunk_idx = 0
                latencies.append((time.time() - t0) * 1000.0)

            env_action = slice_action(chunk[chunk_idx])
            chunk_idx += 1
            if not args.allow_base_motion:
                env_action["action.base_motion"][:] = 0.0

            obs, reward, terminated, truncated, info = env.step(env_action)
            history.append(env_obs_to_dp_dict(obs, obs_keys, lang_emb))

            try:
                frame = env.sim.render(height=512, width=768,
                                       camera_name="robot0_agentview_left")[::-1]
                writer.append_data(np.asarray(frame, dtype=np.uint8))
            except Exception:
                pass

            if info.get("success"):
                if not success:
                    steps_to_success = step
                success = True

            if step % progress_every == 0:
                last = latencies[-1] if latencies else float("nan")
                print(f"  step {step:4d}/{args.max_steps}  predict_lat={last:6.1f}ms  "
                      f"success={int(success)}", flush=True)

            if success:
                print(f"  ✓ stop_on_success at step {step}", flush=True)
                break
            if terminated or truncated:
                print(f"  env terminated/truncated at step {step} "
                      f"(terminated={terminated} truncated={truncated})", flush=True)
                break
    finally:
        writer.close()

    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    return {
        "seed": int(seed),
        "success": bool(success),
        "steps_to_success": int(steps_to_success),
        "steps": int(step + 1),
        "mean_latency_ms": mean_lat,
        "video": str(video_path),
    }


# ─── main ─────────────────────────────────────────────────────────────────
def main() -> int:
    args = parse_args()
    if args.num_envs != 1:
        sys.stderr.write(f"--num-envs={args.num_envs} not supported (sequential only).\n")
        return 2

    if not args.smoke_only and not args.task:
        sys.stderr.write("error: --task is required (unless --smoke-only is set)\n")
        return 2

    if args.task is None:
        canon = "smoke"  # placeholder for output_dir naming under smoke-only
    else:
        canon = canonical_task(args.task)
        if canon != args.task:
            print(f"alias: {args.task} -> {canon} (canonical RoboCasa class)", flush=True)
    if args.output_dir is None:
        args.output_dir = f"./test_outputs/dp_{canon}"
    os.makedirs(args.output_dir, exist_ok=True)

    deps = lazy_imports()
    OmegaConf = deps["OmegaConf"]; gym = deps["gym"]; get_task_horizon = deps["get_task_horizon"]

    cfg, workspace, policy = load_workspace_and_policy(
        args.ckpt, args.output_dir, args.device, deps,
    )
    policy_class = type(policy).__name__
    workspace_target = OmegaConf.select(cfg, "_target_", default="?")
    policy_target = OmegaConf.select(cfg, "policy._target_", default="?")
    n_obs_steps = int(OmegaConf.select(cfg, "policy.n_obs_steps", default=2))
    n_action_steps = int(OmegaConf.select(cfg, "policy.n_action_steps", default=8))
    num_inference_steps = getattr(policy, "num_inference_steps", "?")

    print(f"  workspace target : {workspace_target}", flush=True)
    print(f"  policy    target : {policy_target}", flush=True)
    print(f"  policy    class  : {policy_class}", flush=True)
    print(f"  n_obs_steps={n_obs_steps}  n_action_steps={n_action_steps}  "
          f"num_inference_steps={num_inference_steps}", flush=True)

    obs_keys = resolve_obs_keys(cfg, deps)
    print(f"  shape_meta.obs ({len(obs_keys)} keys):", flush=True)
    for k, info in obs_keys.items():
        print(f"    {k:40s}  type={info['type']:8s}  shape={info['shape']}", flush=True)

    if args.smoke_only:
        print("\nsmoke-only: ckpt loads + policy builds. Exiting.", flush=True)
        return 0

    if args.max_steps is None:
        try:
            args.max_steps = int(get_task_horizon(canon) * 1.5)
        except Exception:
            args.max_steps = 600
        print(f"  max_steps from get_task_horizon({canon}) * 1.5 = {args.max_steps}", flush=True)

    # Pre-allocate a placeholder lang_emb of the right size; the real per-rollout
    # encoding happens inside run_one_trial after env.reset (because the language
    # is supplied by env obs as `annotation.human.task_description`, exactly as
    # robomimic_image_wrapper does it).
    lang_dim = 768
    for k, info in obs_keys.items():
        if "lang" in k.lower():
            lang_dim = int(info["shape"][0])
            break
    lang_emb = np.zeros((lang_dim,), dtype=np.float32)

    print(f"\nCreating env robocasa/{canon} (split={args.split})  "
          f"num_rollouts={args.num_rollouts}  base_seed={args.seed_base}", flush=True)
    env = gym.make(f"robocasa/{canon}", split=args.split, seed=args.seed_base, enable_render=True)

    out_dir = Path(args.output_dir)
    trials: list[dict] = []
    try:
        for i in range(args.num_rollouts):
            args._current_seed = args.seed_base + i
            tmp_video = out_dir / f"dp_{canon}_seed{args._current_seed}_inflight.mp4"
            print(f"\n[trial seed={args._current_seed}]", flush=True)
            try:
                trial = run_one_trial(args, env, policy, cfg, obs_keys, lang_emb, tmp_video, deps)
            except Exception as e:
                print(f"  trial FAILED: {type(e).__name__}: {e}", flush=True)
                trial = {
                    "seed": int(args._current_seed),
                    "success": False, "steps_to_success": -1, "steps": 0,
                    "mean_latency_ms": 0.0, "video": "",
                    "error": f"{type(e).__name__}: {e}",
                }
            if tmp_video.exists():
                final = out_dir / f"dp_{canon}_seed{args._current_seed}_success{int(trial['success'])}.mp4"
                tmp_video.rename(final)
                trial["video"] = str(final)
            trials.append(trial)
    finally:
        env.close()

    successes = [t["success"] for t in trials]
    mean_lats = [t["mean_latency_ms"] for t in trials if t.get("mean_latency_ms")]
    summary = {
        "task": canon, "task_arg": args.task, "split": args.split,
        "num_rollouts": int(args.num_rollouts), "max_steps": int(args.max_steps),
        "n_action_steps": n_action_steps, "n_obs_steps": n_obs_steps,
        "ckpt": str(args.ckpt),
        "rollouts": trials,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "successes": int(sum(successes)),
        "mean_latency_ms": float(np.mean(mean_lats)) if mean_lats else 0.0,
    }
    summary_path = out_dir / f"dp_{canon}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 70)
    print(f"DONE  task={canon}  success={summary['successes']}/{summary['num_rollouts']}"
          f"  rate={summary['success_rate']:.2f}  mean_lat={summary['mean_latency_ms']:.1f}ms")
    print(f"Summary: {summary_path}")
    for t in trials:
        flag = "✓" if t["success"] else "✗"
        vid = Path(t["video"]).name if t.get("video") else "(no video)"
        print(f"  {flag} seed={t['seed']:>4}  steps={t['steps']:>4}  "
              f"steps_to_success={t['steps_to_success']:>4}  -> {vid}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
