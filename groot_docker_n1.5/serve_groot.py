"""
GR00T-N1.5 HTTP server conforming to ../VLA_COMMUNICATION_PROTOCOL.md.

Endpoints:
    GET  /health          -> {status, model, action_type, action_keys, n_action_steps, ...}
    POST /reset           -> clears the action-chunk queue if any
    POST /act             -> action sub-keys + latency_ms

Sub-key contract (matches the multitask_learning/checkpoint-120000 metadata.json):

    request body:
        observation.images.robot0_eye_in_hand     (base64 PNG, HWC uint8 RGB)
        observation.images.robot0_agentview_left  (base64 PNG)
        observation.images.robot0_agentview_right (base64 PNG)
        observation.state.base_position           list[float] [3]
        observation.state.base_rotation           list[float] [4]  quaternion
        observation.state.end_effector_position_relative  [3]
        observation.state.end_effector_rotation_relative  [4]  quaternion
        observation.state.gripper_qpos            list[float] [2]
        task                                      str

    response body:
        action.base_motion              list[list[float]]  [T, 4]
        action.control_mode             [T, 1]
        action.end_effector_position    [T, 3]
        action.end_effector_rotation    [T, 3]   axis_angle
        action.gripper_close            [T, 1]
        latency_ms                      float

Run modes:
    --dummy        : skip model load, return zeros of the right shape (smoke test)
    (default)      : load Gr00tPolicy from --model-path, serve real predictions

Aliases accepted for backward-compat with the protocol doc's shorter names:
    observation.images.static -> robot0_agentview_left
    observation.images.wrist  -> robot0_eye_in_hand
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import sys
import time
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("serve_groot")

# ─── action / state contract ───────────────────────────────────────────────
ACTION_KEYS = [
    # Canonical PandaOmronDataConfig order (data_config.py:653-659):
    # eef_pos, eef_rot, gripper_close, base_motion, control_mode.
    "action.end_effector_position",
    "action.end_effector_rotation",
    "action.gripper_close",
    "action.base_motion",
    "action.control_mode",
]
ACTION_DIMS = {
    "action.base_motion": 4,
    "action.control_mode": 1,
    "action.end_effector_position": 3,
    "action.end_effector_rotation": 3,
    "action.gripper_close": 1,
}

VIDEO_KEYS = [
    # Order MUST match canonical PandaOmronDataConfig.video_keys
    # (robocasa-benchmark/Isaac-GR00T data_config.py:641-645): left, right,
    # eye_in_hand. ConcatTransform stacks views in this order — swapping the
    # order means the model sees the wrist camera in the channel slot it
    # learned for agentview_left and vice versa. Silent visual corruption.
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]
VIDEO_ALIASES = {
    "static": "robot0_agentview_left",
    "wrist": "robot0_eye_in_hand",
}

STATE_KEYS = [
    # Order matches the canonical PandaOmronDataConfig in
    # robocasa-benchmark/Isaac-GR00T/gr00t/experiment/data_config.py:646-652.
    # Order matters for ConcatTransform's state_concat_order.
    "end_effector_position_relative",
    "end_effector_rotation_relative",
    "gripper_qpos",
    "base_position",
    "base_rotation",
]

# multitask_learning/checkpoint-120000 metadata.json declares this tag
DEFAULT_EMBODIMENT_TAG = "new_embodiment"
DEFAULT_N_ACTION_STEPS = 16

# Modality keys (with the gr00t-internal prefix). The bare keys above are what
# clients send over the wire (without prefix); these are what get plumbed into
# the Isaac-GR00T ModalityConfig.
GROOT_VIDEO_KEYS_PREFIXED = [f"video.{k}" for k in VIDEO_KEYS]
GROOT_STATE_KEYS_PREFIXED = [f"state.{k}" for k in STATE_KEYS]
GROOT_ACTION_KEYS_PREFIXED = ACTION_KEYS  # already prefixed in our contract


def _build_robocasa_modality(args):
    """Mirror the canonical `PandaOmronDataConfig.transform()` from
    robocasa-benchmark/Isaac-GR00T (the fork that produced the paper's
    9.6%/40.6% numbers); see
        https://github.com/robocasa-benchmark/Isaac-GR00T/blob/main/gr00t/experiment/data_config.py
    PandaOmronDataConfig (line 640) inherits from BimanualPandaGripperDataConfig
    (line 430) whose .transform() composes these steps in this exact order:
        VideoToTensor -> VideoCrop(0.95) -> VideoResize(224) -> VideoColorJitter
        -> VideoToNumpy -> StateActionToTensor(state) ->
        StateActionTransform(state, target_rotations=quat→6D) ->
        StateActionToTensor(action) -> StateActionTransform(action) ->
        ConcatTransform -> GR00TTransform(max_state_dim=64, max_action_dim=32).

    Three things our previous inline version was missing:
      1. target_rotations — quaternions for state.end_effector_rotation_relative
         and state.base_rotation get converted to 6D rotation representation.
         Sending raw quaternions gives the model a wrong-shape state vector.
      2. VideoCrop(scale=0.95) — slight zoom-in applied at training time.
      3. VideoColorJitter — color augmentation. Has built-in eval-mode behavior
         (gr00t/data/transform/video.py:153 `if self.training`); harmless to
         leave in the pipeline at inference because eval_transform is None.
    """
    from gr00t.data.dataset import ModalityConfig
    from gr00t.data.transform.base import ComposedModalityTransform
    from gr00t.data.transform.concat import ConcatTransform
    from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
    from gr00t.data.transform.video import (
        VideoColorJitter,
        VideoCrop,
        VideoResize,
        VideoToNumpy,
        VideoToTensor,
    )
    from gr00t.model.transforms import GR00TTransform

    obs_indices = [0]
    act_indices = list(range(args.n_action_steps))

    # PandaOmronDataConfig.language_keys at canonical line 661.
    cfg = {
        "video": ModalityConfig(delta_indices=obs_indices, modality_keys=GROOT_VIDEO_KEYS_PREFIXED),
        "state": ModalityConfig(delta_indices=obs_indices, modality_keys=GROOT_STATE_KEYS_PREFIXED),
        "action": ModalityConfig(delta_indices=act_indices, modality_keys=GROOT_ACTION_KEYS_PREFIXED),
        "language": ModalityConfig(delta_indices=obs_indices,
                                   modality_keys=["annotation.human.task_description"]),
    }

    # Canonical PandaOmronDataConfig (data_config.py:666-683):
    state_normalization_modes = {
        "state.end_effector_position_relative": "min_max",
        "state.end_effector_rotation_relative": "min_max",
        "state.gripper_qpos": "min_max",
        "state.base_position": "min_max",
        "state.base_rotation": "min_max",
    }
    state_target_rotations = {
        "state.end_effector_rotation_relative": "rotation_6d",
        "state.base_rotation": "rotation_6d",
    }
    action_normalization_modes = {
        "action.end_effector_position": "min_max",
        "action.end_effector_rotation": "min_max",
        "action.gripper_close": "binary",
        "action.base_motion": "min_max",
        "action.control_mode": "binary",
    }

    transforms = [
        # Video pipeline — must match BimanualPandaGripperDataConfig.transform()
        # (data_config.py:498-511) sequence exactly.
        VideoToTensor(apply_to=GROOT_VIDEO_KEYS_PREFIXED),
        VideoCrop(apply_to=GROOT_VIDEO_KEYS_PREFIXED, scale=0.95),
        VideoResize(apply_to=GROOT_VIDEO_KEYS_PREFIXED, height=224, width=224, interpolation="linear"),
        VideoColorJitter(
            apply_to=GROOT_VIDEO_KEYS_PREFIXED,
            brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08,
        ),
        VideoToNumpy(apply_to=GROOT_VIDEO_KEYS_PREFIXED),
        # State pipeline — target_rotations is the critical knob: quaternions
        # become 6D rotation reps; without this the model sees a mis-shaped
        # state vector (quat 4D vs trained 6D).
        StateActionToTensor(apply_to=GROOT_STATE_KEYS_PREFIXED),
        StateActionTransform(
            apply_to=GROOT_STATE_KEYS_PREFIXED,
            normalization_modes=state_normalization_modes,
            target_rotations=state_target_rotations,
        ),
        # Action pipeline.
        StateActionToTensor(apply_to=GROOT_ACTION_KEYS_PREFIXED),
        StateActionTransform(
            apply_to=GROOT_ACTION_KEYS_PREFIXED,
            normalization_modes=action_normalization_modes,
        ),
        ConcatTransform(
            video_concat_order=GROOT_VIDEO_KEYS_PREFIXED,
            state_concat_order=GROOT_STATE_KEYS_PREFIXED,
            action_concat_order=GROOT_ACTION_KEYS_PREFIXED,
        ),
        GR00TTransform(
            state_horizon=len(obs_indices),
            action_horizon=len(act_indices),
            max_state_dim=64,
            max_action_dim=32,
        ),
    ]
    composed = ComposedModalityTransform(transforms=transforms)
    # CRITICAL: switch the entire transform pipeline to eval mode. Without this,
    # `ComposedModalityTransform` defaults to `training=True` and propagates that
    # to every child:
    #   - VideoCrop runs T.RandomCrop(243) (different crop every step) instead of CenterCrop
    #   - VideoColorJitter actively jitters brightness/contrast/saturation/hue every step
    #     (eval_transform is None → if not flipped, training augmentations apply)
    # The model can't perceive a coherent scene under per-step image augmentation —
    # which is exactly the symptom we observed (drift, no successful manipulation).
    # See `gr00t/data/transform/base.py:81-84` for the .eval() / .train() helpers.
    composed.eval()
    for t in composed.transforms:
        t.training = False
    return cfg, composed


# ─── helpers ───────────────────────────────────────────────────────────────
def _b64_to_image_array(b64: str) -> np.ndarray:
    """base64 PNG -> HWC uint8 RGB."""
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _resize_image(arr: np.ndarray, h: int = 256, w: int = 256) -> np.ndarray:
    if arr.shape[:2] == (h, w):
        return arr
    img = Image.fromarray(arr)
    img = img.resize((w, h), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _pick_video(payload: dict, target_key: str) -> np.ndarray | None:
    """Find a base64 image in the payload for `target_key`, honouring aliases."""
    candidates = [target_key]
    for alias, canonical in VIDEO_ALIASES.items():
        if canonical == target_key:
            candidates.append(alias)
    for cam in candidates:
        b64 = payload.get(f"observation.images.{cam}")
        if b64:
            return _resize_image(_b64_to_image_array(b64))
    return None


def _pick_state(payload: dict, key: str) -> list[float] | None:
    return payload.get(f"observation.state.{key}")


# ─── app factory ───────────────────────────────────────────────────────────
def make_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="GR00T-N1.5 HTTP server")

    # state captured in closure
    state: dict[str, Any] = {
        "policy": None,
        "n_action_steps": args.n_action_steps,
        "embodiment_tag": args.embodiment_tag,
        "model_label": "groot-n1.5-multitask" if not args.dummy else "groot-n1.5-dummy",
        "ready": False,
    }

    if args.dummy:
        log.warning("--dummy mode: zero actions, no model loaded")
        state["ready"] = True
    else:
        # Lazy import: only fail late so /health and /reset can still answer
        # something meaningful before the policy is fully online.
        try:
            log.info("loading Gr00tPolicy from %s (embodiment=%s)",
                     args.model_path, args.embodiment_tag)
            from gr00t.data.embodiment_tags import EmbodimentTag
            # Module path differs between Isaac-GR00T tags:
            #   n1.5-release  -> gr00t.model.policy.Gr00tPolicy
            #   master / n1.7 -> gr00t.policy.gr00t_policy.Gr00tPolicy
            try:
                from gr00t.model.policy import Gr00tPolicy
            except ImportError:
                from gr00t.policy.gr00t_policy import Gr00tPolicy

            # Some Isaac-GR00T versions name the enum differently. Try the
            # uppercase enum name first, then fall back to value match,
            # then to the raw string.
            tag_name = args.embodiment_tag
            try:
                tag = EmbodimentTag[tag_name.upper()]
            except KeyError:
                try:
                    tag = EmbodimentTag(tag_name.lower())
                except Exception:
                    log.warning(
                        "EmbodimentTag has no member '%s'; passing string. Available: %s",
                        tag_name, [e.name for e in EmbodimentTag],
                    )
                    tag = tag_name.lower()
            # n1.5-release Gr00tPolicy needs explicit modality_config + modality_transform.
            modality_cfg, modality_transform = _build_robocasa_modality(args)
            try:
                # n1.5-release signature
                base_policy = Gr00tPolicy(
                    model_path=args.model_path,
                    modality_config=modality_cfg,
                    modality_transform=modality_transform,
                    embodiment_tag=tag,
                    denoising_steps=args.denoising_steps,
                    device=args.device,
                )
            except TypeError:
                # master/n1.7 simplified signature
                base_policy = Gr00tPolicy(
                    embodiment_tag=tag,
                    model_path=args.model_path,
                    device=args.device,
                )

            # Gr00tSimPolicyWrapper exists only on master/n1.7 — optional.
            try:
                from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper  # noqa: F401
                policy = Gr00tSimPolicyWrapper(base_policy, strict=not args.no_strict)
            except Exception:
                log.info("Gr00tSimPolicyWrapper unavailable; using base policy directly")
                policy = base_policy

            # Try to inspect the action chunk length the policy was configured for.
            try:
                cfg = getattr(policy, "modality_config", None) or getattr(base_policy, "modality_config", None)
                if cfg and "action" in cfg and getattr(cfg["action"], "delta_indices", None) is not None:
                    state["n_action_steps"] = len(cfg["action"].delta_indices)
                    log.info("n_action_steps from modality_config = %d", state["n_action_steps"])
            except Exception as e:
                log.warning("could not read n_action_steps from modality_config: %s", e)

            state["policy"] = policy
            state["ready"] = True
            log.info("policy ready")
        except Exception:
            log.exception("policy load failed; server will return 503 on /act")
            state["ready"] = False

    # ─── endpoints ───
    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok" if state["ready"] else "loading",
            "model": state["model_label"],
            "action_type": "relative",
            "action_keys": ACTION_KEYS,
            "n_action_steps": state["n_action_steps"],
            "embodiment_tag": state["embodiment_tag"],
            "video_keys": VIDEO_KEYS,
            "state_keys": STATE_KEYS,
        }

    @app.post("/reset")
    def reset() -> dict:
        p = state["policy"]
        if p is not None:
            try:
                if hasattr(p, "reset"):
                    p.reset()
            except Exception as e:
                log.warning("policy.reset() raised: %s", e)
        return {"status": "reset"}

    @app.post("/act")
    async def act(request: Request) -> dict:
        t0 = time.time()
        if not state["ready"]:
            raise HTTPException(status_code=503, detail="policy not ready")

        payload = await request.json()
        if args.dummy:
            return _dummy_response(state["n_action_steps"], t0)

        try:
            obs = _build_obs(payload)
            result = state["policy"].get_action(obs)
            # Gr00tSimPolicyWrapper.get_action returns (action_dict, info);
            # the bare Gr00tPolicy returns a plain dict. Tolerate both.
            if isinstance(result, tuple) and len(result) >= 1:
                action_dict = result[0]
            else:
                action_dict = result
            response = _format_action(action_dict)
        except Exception:
            log.exception("/act failed")
            raise HTTPException(status_code=500, detail="inference error")
        response["latency_ms"] = (time.time() - t0) * 1000.0
        return response

    return app


def _dummy_response(T: int, t0: float) -> dict:
    out = {k: np.zeros((T, ACTION_DIMS[k]), dtype=np.float32).tolist() for k in ACTION_KEYS}
    out["latency_ms"] = (time.time() - t0) * 1000.0
    return out


def _build_obs(payload: dict) -> dict:
    """Translate the protocol payload into the Isaac-GR00T obs dict."""
    obs: dict[str, Any] = {}

    # Videos: shape [B=1, T=1, H, W, C] uint8
    for cam in VIDEO_KEYS:
        img = _pick_video(payload, cam)
        if img is None:
            raise ValueError(f"missing observation.images.{cam}")
        obs[f"video.{cam}"] = img[None, None, ...]

    # States: shape [B=1, T=1, D] float32
    for key in STATE_KEYS:
        v = _pick_state(payload, key)
        if v is None:
            raise ValueError(f"missing observation.state.{key}")
        arr = np.asarray(v, dtype=np.float32)
        obs[f"state.{key}"] = arr.reshape(1, 1, -1)

    # Match the language modality key declared above (PandaOmron convention).
    obs["annotation.human.task_description"] = [payload.get("task", "")]
    return obs


def _format_action(chunks: dict) -> dict:
    """Strip batch dim, ensure 2D list shape [T, dim] for each action sub-key."""
    out: dict[str, Any] = {}
    for k in ACTION_KEYS:
        if k not in chunks:
            raise ValueError(f"policy did not return {k}")
        arr = np.asarray(chunks[k])
        if arr.ndim == 3:               # [B, T, D]
            arr = arr[0]
        if arr.ndim == 1:               # [D] -> [1, D]
            arr = arr[None, :]
        out[k] = arr.astype(np.float32).tolist()
    return out


# ─── cli ───────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8500)
    p.add_argument("--model-path", default="/groot/checkpoint",
                   help="dir with experiment_cfg/metadata.json + model.safetensors")
    p.add_argument("--embodiment-tag", default=DEFAULT_EMBODIMENT_TAG.upper(),
                   help="EmbodimentTag enum name (default NEW_EMBODIMENT — matches multitask checkpoint)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-strict", action="store_true",
                   help="disable Isaac-GR00T strict modality matching")
    p.add_argument("--n-action-steps", type=int, default=DEFAULT_N_ACTION_STEPS,
                   help="action chunk length (default 16)")
    p.add_argument("--denoising-steps", type=int, default=4,
                   help="GR00T diffusion denoising steps per inference (default 4)")
    p.add_argument("--dummy", action="store_true",
                   help="don't load a model, return zero actions (protocol smoke test)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log.info("starting on %s:%d (dummy=%s, model_path=%s)",
             args.host, args.port, args.dummy, args.model_path)
    app = make_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
