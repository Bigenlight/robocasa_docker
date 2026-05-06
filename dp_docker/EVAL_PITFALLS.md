# Diffusion Policy RoboCasa Eval -- Pitfalls & Postmortem

> Hard-won lessons from getting `PickPlaceCounterToSink` from "0/5 with
> directional-but-incorrect motion across all seeds" to a paper-consistent
> 1/20 = 5% (95% binomial CI [0.13%, 25%] -- includes the paper's 15.7%
> atomic-seen average). Each rule below has a real bug attached, all four
> live in `dp_docker/eval_dp.py` and were caught by a 15-agent investigation
> plus a numerical canonical-equivalence verifier.

The trajectory:
- **0/5 (consistent failure pattern, 5 seeds)** -- robot moved with directional intent but never grasped cleanly. Initial `dp_docker/` landed in commit `0b40f0f` with four latent misalignments.
- **0/5 still (consistent failure)** -- after swapping `latest.ckpt` for the leaderboard-aligned `epoch=0500` ckpt. Same SR. Confirmed: the bug is not in the ckpt choice.
- **0/5 still (consistent failure)** -- after fixing 3 bugs (action layout / language encoder / task text source). Verifier eval at the same time hit 15-min timeout on seed 4.
- **0/5 then 1/15 = 1/20 = 5%** -- after the 4th fix (`--allow-base-motion` default to True) plus n=20. Paper-consistent, wrapper now bit-exact equivalent to canonical (commit `21d637f`).

If you are about to run a fresh DP eval, read the TL;DR + checklist first;
treat the four pitfalls as the failure modes most likely to bite you again.

---

## TL;DR -- the canonical pipeline

- Match `robocasa-benchmark/diffusion_policy/eval_robocasa.py` (the upstream entrypoint -- our `eval_dp.py` is an in-process single-task wrapper around the same machinery) and `RobomimicImageWrapper.process_obs` (the canonical obs-prep) bytewise. Both are now in the cloned `./diffusion_policy/` after `./run.sh --download-ckpt`.
- Action vector is **flat 12-d**, NOT a dict. Sub-key concatenation order comes from `cfg.shape_meta.action.lerobot_keys`, which for the released DP multitask ckpt is:
  ```
  pos[0:3]  rot[3:6]  grip[6:7]  base[7:11]  mode[11:12]
  ```
  This is **not** the GR00T `metadata.json` order. Don't reuse GR00T slicing.
- Obs keys come from `cfg.shape_meta.obs[k].lerobot_keys[0]`. Always look this up explicitly -- fuzzy matching on key stems works only by coincidence and is brittle for future cfgs.
- Language conditioning uses `robomimic.utils.lang_utils.LangEncoder`, which wraps `CLIPTextModelWithProjection.text_embeds[0]` (NOT `CLIPTextModel.pooler_output[0]`) with `padding="max_length"` (77 tokens). Different 768-d weight space.
- Task text is the natural-language sentence at `obs["annotation.human.task_description"]`, e.g. *"Pick the orange from the counter and place it in the sink."* -- not the Python class name.
- `n_action_steps = 8` (full-chunk replay). Default lives in `cfg` and the wrapper consumes the chunk step-by-step.
- 256x256 cameras, fps=20, RGB. Image prep is `HWC uint8 -> CHW float [0,1]` via `np.moveaxis(img, -1, 1).astype(np.float32) / 255.` (canonical `process_frame(img, channel_dim=3, scale=255.)` is bytewise identical).
- `split=pretrain` for `PickPlaceCounterToSink` and the rest of `atomic_seen` / `composite_seen` -- the kitchens the multitask ckpt was trained on.

---

## The four pitfalls

Each block: symptom / root cause / fix / how to verify.

### 1. Action layout cribbed from GR00T `metadata.json`

- **Symptom:** 0/5 SR. Robot reaches with directional intent (the gross arm motion is roughly correct) but never grasps cleanly. All five seeds fail in visually similar ways.
- **Root cause:** Our `DP_ACTION_LAYOUT` was copied from `groot_docker_n1.5/checkpoint/experiment_cfg/metadata.json`, which orders the flat 12-d action as `base[0:4] / mode[4:5] / pos[5:8] / rot[8:11] / grip[11:12]`. The released DP cfg orders sub-keys differently. Authoritative source: `cfg.shape_meta.action.lerobot_keys`, with concat order verified at `dp_docker/diffusion_policy/diffusion_policy/dataset/lerobot_dataset.py:190-201`:
  ```python
  action_concat = []
  for lr_key in self.lerobot_action_keys:
      action_concat.append(data[lr_key])
  action_concat = np.concatenate(action_concat, axis=-1)
  ```
- **Fix:** `eval_dp.py:60` now uses the cfg-aligned order:
  ```python
  DP_ACTION_LAYOUT = [
      ("action.end_effector_position",  0,  3),
      ("action.end_effector_rotation",  3,  6),
      ("action.gripper_close",          6,  7),
      ("action.base_motion",            7, 11),
      ("action.control_mode",          11, 12),
  ]
  ```
  A startup assertion at `eval_dp.py:153` cross-checks this against the loaded `cfg.shape_meta.action.lerobot_keys` so any future drift fires loudly instead of silently mis-routing the policy's output.
- **Verify:**
  ```bash
  ./run.sh --smoke
  ```
  Smoke runs the assertion. If it fires, the message includes both `cfg.shape_meta.action.lerobot_keys` and `DP_ACTION_LAYOUT` keys for diff. Silent pass = aligned.

### 2. Wrong CLIP variant for language embedding

- **Symptom:** 0/5 SR. Robot motion is task-aware but consistently off; same failure pattern across seeds. Paper-aligned ckpt makes no difference.
- **Root cause:** Initial wrapper used `transformers.CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')` and read `pooler_output[0]` (768-d). Training-time encoder is `robomimic.utils.lang_utils.LangEncoder`, which wraps `CLIPTextModelWithProjection.from_pretrained('openai/clip-vit-large-patch14')` and reads `["text_embeds"][0]` (also 768-d) with `padding="max_length"` (77 tokens, not dynamic). The two outputs share dim but live in different weight spaces (`pooler_output` is the LayerNorm'd CLS, `text_embeds` is the projection head's output that CLIP was contrastively trained on). Cosine similarity on the same input is well below 1.
- **Fix:** `eval_dp.py:206-326` -- replaced the inline encoder with a `get_lang_encoder()` that imports `from robomimic.utils.lang_utils import LangEncoder` directly. Module-level cache so a single rollout loads CLIP only once. Custom `cache_dir` override because uid-1000's `pw_dir != $HOME` makes the default `os.path.expanduser('~')` resolve to a non-writable dir inside the container.
- **Verify:** `./run.sh --shell`, then:
  ```python
  from robomimic.utils.lang_utils import LangEncoder
  enc = LangEncoder(device='cpu')
  emb = enc.get_lang_emb('pick the can')
  print(emb.shape, emb.dtype)        # torch.Size([768]) torch.float32
  print(type(enc).__name__)          # LangEncoder
  ```
  Then check `eval_dp.py` actually calls this same class (grep `from robomimic.utils.lang_utils`). If the wrapper instantiates `CLIPTextModel` directly anywhere, that's regression.

### 3. Task text fed as Python class name

- **Symptom:** 0/5 SR. Same consistent-failure look as bugs 1 + 2.
- **Root cause:** Wrapper passed the argparse `--task` value directly to the encoder. So `LangEncoder` saw the literal string `"PickPlaceCounterToSink"` (CamelCase Python class name). Training-time text comes from each demo's `episode_metadata.tasks[0]` which is a real human-written instruction, e.g. *"Pick the orange from the counter and place it in the sink."* The env exposes this same string per-rollout at `obs["annotation.human.task_description"]` (set by `gym_wrapper.py` from `env.get_ep_meta().get("lang", "")` at line 264, 284). CamelCase strings are out-of-distribution for a CLIP text encoder trained on prose -- the resulting `text_embeds` is essentially noise.
- **Fix:** `eval_dp.py:443-469`'s `run_one_trial` now reads the env-provided text *after* `env.reset()` and re-encodes it for each rollout (different scenes can have different object instances and therefore different sentences):
  ```python
  task_text = obs.get("annotation.human.task_description", args.task)
  lang_emb = encode_lang(str(task_text), lang_dim, args.device)
  ```
  The `args.task` fallback only fires if the env omits the key, which is unusual.
- **Verify:** First `env_obs_to_dp_dict` call dumps a stderr resolution map. Look for the `lang_emb -> <encoder: lang_emb> [computed]` line; the actual sentence is logged as `task: 'Pick the orange from the counter and place it in the sink.'` near the rollout banner.

### 4. Forced `action.base_motion = 0` by default

- **Symptom:** 0/5 SR even after fixing bugs 1-3. Robot now moves with somewhat better intent but the arm still misses the object by a few centimetres on every reach.
- **Root cause:** `eval_dp.py:495` zeroed the model-predicted `action.base_motion` whenever `--allow-base-motion` was unset (i.e. by default). Canonical `RobomimicImageWrapper.step` (`dp_docker/diffusion_policy/diffusion_policy/env/robomimic/robomimic_image_wrapper.py:145-148`) calls `convert_action(action)` from `robocasa/utils/env_utils.py:134-145`, which forwards the model's raw 4-d base_motion to the env unchanged. PandaOmron is a mobile base; even on atomic tasks the policy can issue small base micro-corrections that the arm IK depends on. Zeroing decouples base from arm reach -> consistent reach-but-miss.
- **Fix:** `eval_dp.py:107` switches `--allow-base-motion` to `argparse.BooleanOptionalAction` with `default=True`, matching the canonical wrapper. Legacy zeroing behaviour is still available via `--no-allow-base-motion` for A/B comparison.
- **Verify:** `./run.sh --eval PickPlaceCounterToSink --num-rollouts 5 --split pretrain` and check the printed args: `allow_base_motion=True`. Run with `--no-allow-base-motion` once to reproduce the legacy 0/5 if you want to confirm the diff at API level (latency drops ~7% with raw forward, an env-side hint that the base call site is exercised differently).

---

## How we found these (the discovery process is reusable)

Five-rollout eval gave 0/5 with five visually-similar failures. That alone
doesn't separate "wrapper bug" from "p=0.157 binomial noise" -- you can
get the same outcome from a perfectly-calibrated wrapper. What broke the
ambiguity was a **numerical canonical-equivalence test**, not more rollouts.

The recipe is reusable for any custom VLA wrapper:

1. **Probe `cfg.shape_meta` directly from the loaded ckpt.** Don't trust
   the README, don't trust prior wrappers, don't trust `metadata.json`
   from a sibling project. The cfg embedded in the ckpt is the only
   authoritative description of the action layout, obs keys, and
   normalizer ranges.

2. **Locate the canonical inference path** (`RobomimicImageWrapper.process_obs`
   for chi2023 DP, `RobomimicImageRunner.run` for the env loop). The
   canonical path is what the policy was trained against; mirror it
   bytewise.

3. **Run a one-step parity smoke**: same env seed, same `env.reset()`,
   build the obs_dict via *both* the canonical path and your wrapper,
   `np.allclose` per key. Then call `policy.predict_action` on each and
   compare action chunks. If everything is `0.0e+00`, your wrapper is
   provably equivalent to the canonical pipeline; if a single key has
   `> 1e-4` diff, that's your root cause.

4. **Iterate fixes until the parity test passes**, then run the actual
   eval. We landed bit-exact at:

   | Probe | max_abs_diff |
   |---|---|
   | All 7 obs keys (3 cam + 3 lowdim + lang_emb) vs `RobomimicImageWrapper.process_obs` | 0.0e+00 |
   | `policy.predict_action(obs)['action']` chunk (1, 8, 12) | 0.0e+00 |

5. **Then, and only then, run more rollouts** for a tighter statistical
   signal. We bumped from n=5 (P(0/5 \| p=0.157) ~ 0.42) to n=20 (P(0/20)
   ~ 0.034) and observed 1/20 = 5%, well within the binomial CI for 15.7%.

If your VLA wrapper produces consistent-looking failures across seeds,
do steps 1-3 *before* spawning more eval seeds. Eval seeds are slow;
parity tests are fast.

---

## Lessons learned (generalize)

1. **`cfg.shape_meta` is the source of truth.** Sibling cfgs from
   different model families (GR00T metadata.json, openpi config, etc.)
   may share field names but order their flat action vectors
   differently. Never copy slicing tables across wrappers; always read
   from the loaded ckpt's cfg.

2. **The released `diffusion_policy` repo is the chi2023 fork at
   `robocasa-benchmark/diffusion_policy`, NOT pypi `diffusion_policy`.**
   Same names, different content. The fork's `LerobotCotrainingDataset`,
   `RobomimicImageWrapper.process_obs`, and `LangUtils.LangEncoder` are
   the canonical references.

3. **Robomimic must be the `robocasa` branch fork**:
   `pip install git+https://github.com/ARISE-Initiative/robomimic.git@robocasa`.
   Pypi `robomimic 0.3.0` does not have `VisualCoreLanguageConditioned`
   or `ResNet{18,34,50}ConvFiLM`, both of which the released DP cfg
   imports. This requirement is undocumented in any public RoboCasa or
   robomimic surface; the only signal is the `ImportError` at
   `torch.load`-time.

4. **Numerical parity > more rollouts.** A 30-second parity smoke is a
   stronger signal than a 30-minute rollout-grid in early debugging.
   Leave the rollout grid for final SR characterisation.

5. **Statistical noise is real at small n.** With paper-reported per-task
   SR around 15-20%, n=5 leaves you with binomial CIs that include 0%
   even for fully-correct wrappers. Bias toward n>=20 for any "is this
   working?" diagnosis.

---

## Pre-eval checklist

Copy this and tick before each run.

- [ ] Image is up-to-date. `docker images dp-eval:latest` shows a recent
      build / pull. If the Dockerfile changed since you last pulled,
      `./run.sh --build`.
- [ ] Robomimic in the image is the `robocasa` branch. From `--shell`:
      `python -c "from robomimic.models.obs_core import VisualCoreLanguageConditioned; print('OK')"`.
- [ ] Active ckpt resolves correctly. `./run.sh` prints the path it picks
      up; verify it points at `epoch=0500-test_mean_score=-1.000.ckpt`
      (leaderboard-aligned) or `latest.ckpt` (default), per
      §"Checkpoint layout" in `README.md`.
- [ ] `./run.sh --smoke` exits 0. Specifically:
  - Coder A's runtime assertion at `eval_dp.py:153` does NOT fire
    (silent pass = `DP_ACTION_LAYOUT` matches `cfg.shape_meta.action.lerobot_keys`).
  - Policy class prints as `DiffusionTransformerHybridImagePolicy` with
    `n_obs_steps=2 n_action_steps=8 num_inference_steps=100`.
  - All 7 `shape_meta.obs` keys print with the expected shapes (3 cam
    keys at `[3,256,256]`, 3 lowdim, 1 lang_emb at `[768]`).
- [ ] First `--eval` invocation prints the obs-key resolution map at
      stderr. All 6 env-sourced obs keys should show source `[canonical]`
      (no `[fuzzy]` warnings). `lang_emb` shows source `[computed]`.
- [ ] Task text the policy sees is the env-provided sentence, not your
      argparse `--task` value. Grep for `task: 'Pick ...'` in stdout.
- [ ] `--allow-base-motion` is `True` (default). Verify with
      `./run.sh --eval Foo --num-rollouts 1` and grep stdout for
      `allow_base_motion=True`.
- [ ] `--num-rollouts >= 20` if the question is "does this work?".
      `n=5` is fine for the cheap "did the pipe break?" check, but is
      under-powered for any positive claim about SR.
- [ ] After done, `mean_lat` in `summary.json` is in the 600-800 ms range
      on RTX 3080 / 3060. Sustained higher means CPU contention or you
      forgot to warm CLIP (the lazy-load adds ~30 s on the first eval
      only).

---

## Reference: how the canonical pipeline differs from ours

After all four fixes, our wrapper is **numerically identical** to the
canonical preprocessing path. The remaining differences are interface
and ergonomics, not algorithmic:

| Component | Ours (`eval_dp.py`) | Canonical (`eval_robocasa.py` + `RobomimicImageRunner`) |
|---|---|---|
| Inference transport | none -- in-process | none -- in-process (both single-process) |
| Eval driver | thin wrapper, **single task** | full task-soup loop over `TASK_SET_REGISTRY[task_set]` |
| Per-task horizon | `robocasa.utils.dataset_registry_utils.get_task_horizon(task) * 1.5` | same |
| Env construction | `gym.make("robocasa/<Task>", split=..., seed=..., enable_render=True)` | same; `enable_render=True` is the default in `gym_wrapper.py:137` |
| Obs preprocessing | `cfg.shape_meta.obs[k].lerobot_keys[0]` lookup, image `HWC uint8 -> CHW float[0,1]`, history stacked oldest-first | same (bit-exact) |
| Lang encoding | `robomimic.utils.lang_utils.LangEncoder` (cached at module level) | same class, instantiated per dataset construction |
| Action handling | `policy.predict_action(obs)["action"]` -> flat 12-d array -> chunk replay over 8 actions | same; `MultiStepWrapper` consumes the chunk inside `RobomimicImageWrapper.step` |
| Output | `dp_<Task>_seed<N>_success<0|1>.mp4` + `dp_<Task>_summary.json` | wandb logs + `eval_log.json` per task |
| Multi-env | sequential only (`--num-envs 1`) | parallel via `n_envs` (we don't replicate this) |

So our pipeline is a strict subset of canonical's behaviour, with output
schema chosen to match the GR00T sibling for tooling consistency.

---

## File map (repo-relative paths)

| File | Why you'd open it |
|---|---|
| `dp_docker/eval_dp.py` | Our wrapper. Action layout (line 60), runtime assertion (line 153), obs-key resolver (lines 156-295), lang encoder (lines 206-326), task text source (lines 443-469), base_motion default (lines 107, 495). |
| `dp_docker/run.sh` | CLI surface. Active-ckpt auto-resolve (lines 43-60). `DP_CKPT_PATH` override env var. |
| `dp_docker/diffusion_policy/eval_robocasa.py` | Canonical entrypoint (chi2023 fork). The reference our wrapper mirrors. |
| `dp_docker/diffusion_policy/diffusion_policy/dataset/lerobot_dataset.py` | Action concat order (lines 190-201) -- the source of truth for `DP_ACTION_LAYOUT`. |
| `dp_docker/diffusion_policy/diffusion_policy/env/robomimic/robomimic_image_wrapper.py` | Canonical `process_obs` (lines 75-105). Read this if you're debugging obs alignment. |
| `dp_docker/diffusion_policy/diffusion_policy/env_runner/robomimic_image_runner.py` | Canonical env loop. The thing our `eval_dp.py` is a single-task subset of. |
| `dp_docker/diffusion_policy/diffusion_policy/policy/diffusion_transformer_hybrid_image_policy.py` | Policy `predict_action` (calls `self.normalizer.normalize(obs_dict)` first -- so wrapper sends raw env values). |
| `robocasa/utils/env_utils.py:134-145` | `convert_action` -- where the env splits a flat 12-d action into the dict the gym env consumes. |
| `robocasa/wrappers/gym_wrapper.py` | `RoboCasaGymEnv`. Lines 36-67 (`PandaOmronKeyConverter.map_obs`), 264, 284 (where `annotation.human.task_description` is set), 313+ (`step` action-space layout), 108-127 (`unmap_action` binarization). |
| `dp_docker/checkpoint/.../latest.ckpt` (or `epoch=0500`) | The Lightning payload. Inspectable via `torch.load(open(p,'rb'), pickle_module=dill, weights_only=False)['cfg']`. |
| `dp_docker/test_outputs/dp_PickPlaceCounterToSink_seed6_success1.mp4` | The first verified success after all four fixes (n=20 measurement). 161 frames at 20 fps, 8.05 s of sim time. |

---

## See also

- [`README.md`](README.md) -- the deployment quickstart and the full troubleshooting table
- [`../GR00T_EVAL_TIPS.md`](../GR00T_EVAL_TIPS.md) -- sibling postmortem for the GR00T-N1.5 wrapper (different bugs, same `cfg.shape_meta`-is-truth lesson)
- [`../README.md`](../README.md) -- top-level fork overview and image catalogue
- Phase commits: `0b40f0f` adds the buggy initial state; `21d637f` lands all four fixes plus regression guard
