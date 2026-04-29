"""
Render 10 visually distinct PrepareCoffee scene variations to PNGs (+ a 2x5 grid).

Variation axes covered (per the upstream env_utils.create_env signature):
  - split (None / "pretrain" / "target")
  - layout_ids, style_ids        (50 layouts x 50 styles in RoboCasa365)
  - seed                          (object placement / robot init randomization)
  - camera                        (left / right / center / frontview / eye_in_hand)
  - generative_textures           (AI-generated walls/floors)
  - randomize_cameras             (jitter the camera pose)

Designed by an Opus planning agent to span axes deliberately rather than
sample randomly: 6 (layout, style) pairs, 1 same-scene-different-camera
control, 1 same-scene-different-seed control, 1 generative_textures toggle,
1 frontview+jitter shot, 1 wrist-camera shot, 1 held-out target split.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np

# Robosuite import warnings are noisy before it logs anything useful; ignore.
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning")

import gymnasium as gym
import imageio
import robocasa  # noqa: F401  -- registers gym envs as a side effect

TASK = "PrepareCoffee"
RENDER_HEIGHT = 512
RENDER_WIDTH = 768

# Each spec drives one gym.make + one sim.render. Don't mix `split` with
# explicit layout_ids/style_ids — env_utils.create_env branches on split first.
PLAN: list[dict] = [
    dict(label="01_L15_S14_anchor",
         split=None, layout_ids=15, style_ids=14, seed=1,
         camera="robot0_agentview_left",
         generative_textures=False, randomize_cameras=False,
         note="Anchor (layout 15 + style 14 — verified safe in KITCHEN_SCENES_5X5)"),
    dict(label="02_L11_S22_dark",
         split=None, layout_ids=11, style_ids=22, seed=2,
         camera="robot0_agentview_center",
         generative_textures=False, randomize_cameras=False,
         note="Big layout shift + dark cabinets — max contrast vs anchor"),
    dict(label="03_L18_S40_bright",
         split=None, layout_ids=18, style_ids=40, seed=3,
         camera="robot0_agentview_right",
         generative_textures=False, randomize_cameras=False,
         note="Different fixture arrangement, colored cabinetry/tile"),
    dict(label="04_L30_S11_compact",
         split=None, layout_ids=30, style_ids=11, seed=4,
         camera="robot0_agentview_left",
         generative_textures=False, randomize_cameras=False,
         note="Tight galley-style — small kitchens are covered too"),
    dict(label="05_L22_S30_gentex",
         split=None, layout_ids=22, style_ids=30, seed=5,
         camera="robot0_agentview_center",
         generative_textures=True, randomize_cameras=False,
         note="generative_textures=True — AI-generated wall/floor textures"),
    dict(label="06_L15_S14_camR",
         split=None, layout_ids=15, style_ids=14, seed=1,
         camera="robot0_agentview_right",
         generative_textures=False, randomize_cameras=False,
         note="Same world as #1, camera moved → eval pipeline can grab alt views"),
    dict(label="07_L15_S14_seed99",
         split=None, layout_ids=15, style_ids=14, seed=99,
         camera="robot0_agentview_left",
         generative_textures=False, randomize_cameras=False,
         note="Same kitchen as #1, different seed → object/init randomization"),
    dict(label="08_L40_S5_wrist",
         split=None, layout_ids=40, style_ids=5, seed=7,
         camera="robot0_eye_in_hand",
         generative_textures=False, randomize_cameras=False,
         note="Wrist camera (eye-in-hand) — distinct modality"),
    dict(label="09_L5_S18_frontjit",
         split=None, layout_ids=5, style_ids=18, seed=11,
         camera="robot0_frontview",
         generative_textures=False, randomize_cameras=True,
         note="Frontview + randomize_cameras=True → robustness axis"),
    dict(label="10_target_split",
         split="target", layout_ids=None, style_ids=None, seed=21,
         camera="robot0_agentview_center",
         generative_textures=False, randomize_cameras=False,
         note="split='target' — held-out OOD evaluation pair"),
]


def make_env(spec: dict):
    """gym.make passing the spec's kwargs.

    NB: RoboCasaGymEnv defaults split="test" (gym_wrapper.py:138), which
    create_env rejects. We always pass `split` explicitly (None when we want
    to drive layout/style ourselves) to override that default.
    """
    kwargs = dict(seed=spec["seed"], split=spec["split"])
    if spec["layout_ids"] is not None:
        kwargs["layout_ids"] = spec["layout_ids"]
    if spec["style_ids"] is not None:
        kwargs["style_ids"] = spec["style_ids"]
    if spec["generative_textures"]:
        kwargs["generative_textures"] = "100p"   # robocasa value enabling AI textures
    if spec["randomize_cameras"]:
        kwargs["randomize_cameras"] = True
    return gym.make(f"robocasa/{TASK}", **kwargs)


def render_one(spec: dict, output_dir: Path) -> Path | None:
    label = spec["label"]
    out = output_dir / f"{label}.png"
    print(f"\n[ ] {label}: {spec['note']}", flush=True)
    try:
        env = make_env(spec)
    except Exception as e:
        print(f"    SKIP (gym.make failed): {e}")
        return None
    try:
        env.reset()
        sim = env.sim                       # gymnasium 0.29 forwards through wrappers
        img = sim.render(
            height=RENDER_HEIGHT,
            width=RENDER_WIDTH,
            camera_name=spec["camera"],
        )[::-1]
        imageio.imwrite(str(out), img)
        size_kb = out.stat().st_size // 1024
        print(f"    wrote {out.name} ({img.shape}, {size_kb} KB, cam={spec['camera']})")
        return out
    except Exception:
        print("    FAIL:")
        print(traceback.format_exc())
        return None
    finally:
        try:
            env.close()
        except Exception:
            pass


def make_grid(pngs: list[Path], output_dir: Path) -> Path | None:
    """2x5 grid composite — handy for the user to scan all 10 in one view."""
    if not pngs:
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as e:
        print(f"  grid skipped (Pillow missing? {e})")
        return None

    cols, rows = 5, 2
    cell_h, cell_w = 256, 384      # half-size to keep grid compact
    pad = 6
    label_h = 22
    grid_w = cols * cell_w + (cols + 1) * pad
    grid_h = rows * (cell_h + label_h) + (rows + 1) * pad

    canvas = Image.new("RGB", (grid_w, grid_h), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Map labels back to spec for caption text.
    label_to_note = {s["label"]: s["note"] for s in PLAN}

    for i, png in enumerate(pngs):
        if i >= cols * rows:
            break
        col, row = i % cols, i // cols
        x0 = pad + col * (cell_w + pad)
        y0 = pad + row * (cell_h + label_h + pad)
        try:
            tile = Image.open(png).convert("RGB").resize((cell_w, cell_h))
        except Exception:
            continue
        canvas.paste(tile, (x0, y0))
        caption = png.stem
        if font:
            draw.text((x0 + 4, y0 + cell_h + 2), caption, fill=(220, 220, 220), font=font)

    out = output_dir / "_grid_2x5.png"
    canvas.save(out)
    print(f"\nGrid composite: {out}  ({out.stat().st_size // 1024} KB)")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="/workspace/robocasa/test_outputs/coffee_variations")
    p.add_argument("--only", type=int, nargs="+", default=None,
                   help="optional 1-indexed list of spec numbers to render (e.g. --only 1 6 7)")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"task        = robocasa/{TASK}")
    print(f"output dir  = {out_dir}")
    print(f"MUJOCO_GL   = {os.environ.get('MUJOCO_GL', '<unset>')}")
    print(f"resolution  = {RENDER_WIDTH}x{RENDER_HEIGHT}")
    print(f"plan size   = {len(PLAN)}")

    pngs: list[Path] = []
    for i, spec in enumerate(PLAN, 1):
        if args.only and i not in args.only:
            continue
        png = render_one(spec, out_dir)
        if png is not None:
            pngs.append(png)

    print(f"\n{len(pngs)}/{len(PLAN)} variations rendered.")
    make_grid(pngs, out_dir)
    return 0 if len(pngs) == len(PLAN) else 1


if __name__ == "__main__":
    sys.exit(main())
