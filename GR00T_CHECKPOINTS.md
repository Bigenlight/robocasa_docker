# GR00T-N1.5 Fine-tuned Checkpoints for RoboCasa365

A practical guide to the **`robocasa/robocasa365_checkpoints`** HuggingFace
repo. Maps every released GR00T-N1.5 checkpoint folder to (a) the paper
regime it was trained under, (b) the success-rate it produces on the
benchmark, and (c) when you should download it.

The official repo ships **no README** at any depth, so this doc is the
result of cross-referencing folder names against the RoboCasa365 ICLR 2026
paper (`RoboCasa_365_paper.pdf`, ICLR 2026) Sections 4.1–4.3 and
Tables 1–4, 9–11.

- HF root: <https://huggingface.co/robocasa/robocasa365_checkpoints>
- GR00T-N1.5 branch: <https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/gr00t_n1-5>
- Paper (ICLR 2026): <https://robocasa.ai/assets/robocasa365_iclr26.pdf>

---

## 1. TL;DR — which one do I download?

| I want… | Download |
|---|---|
| The **headline 40.6% Composite-Seen** number | `foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000` |
| The strongest model on **Atomic-Seen** (68.5%) | `foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000` |
| The strongest model on **Composite-Unseen** (42.1%) | `foundation_model_learning/target_posttraining/composite_unseen/checkpoint-60000` |
| Weak baseline trained on all 300 tasks at once (9.6%) | `multitask_learning/checkpoint-120000` |
| Just the pretraining stage (no target FT, 0% composite) | `foundation_model_learning/pretraining/checkpoint-80000` |
| Pretraining vs. no-pretraining ablation comparison | `target_only/<split>` ↔ `target_posttraining/<split>` |
| Lifelong / continual learning analysis | `lifelong_learning/phase{1,2,3,4}/...` |

Sizes: each checkpoint is **~16 GB** total (model weights ~7.6 GB + optimizer
state ~8.6 GB). For inference only, the optimizer is unnecessary — pulling
just the two `model-*.safetensors` shards + `config.json` + `experiment_cfg/`
gives you ~7.6 GB.

---

## 2. The 9.6% vs 40.6% confusion (the most common pitfall)

Both numbers refer to **Composite-Seen** task success and both come from
GR00T-N1.5 trained on RoboCasa data — but they correspond to **different
training recipes** **evaluated under different conditions**.

| | 9.6% (Sec 4.1, Table 1) | 40.6% (Sec 4.2, Table 2) |
|---|---|---|
| Paper name | "Multi-task Training" | "Pretraining + Target Post-Training (100%)" |
| Stages | single-stage | two-stage |
| Training data | 300-task human demos (~30k demos), 120k steps | Stage 1: 300-task human + 60-task MimicGen (~630k demos), 80k steps. Stage 2: 16 composite-seen target tasks × 500 demos = 8k demos, 60k steps. |
| **Eval scenes** | **pretraining kitchens** (paper §4.1) | **target kitchens** (paper §4.2) |
| HF path | `multitask_learning/checkpoint-120000` | `foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000` |

The two regimes are **not directly comparable**: they evaluate in
different scene sets, train on different demo counts (~30k vs ~638k),
take different numbers of steps (120k vs 80k+60k = 140k), and only the
40.6% recipe has a dedicated target FT phase. The +31 pp delta conflates
all of those.

The clean comparison for "is the dedicated target FT phase worth it?"
is **`target_only/composite_seen` (35.0%) vs `target_posttraining/composite_seen`
(40.6%)** — a +5.6 pp gain at matched stage-2 data and matched evaluation
scenes. That is the well-isolated number for the value of having a
pretraining stage 1 underneath the target FT (see §6).

For "does target FT itself help over single-stage multitask?", the paper
provides Appendix H.3 (Joint Co-Training, 9.0% composite-seen): training
the same data jointly in one stage yields the same ~9% as plain multi-task,
suggesting the **two-stage structure with a separate target FT phase** is
itself doing real work — not just the larger data or compute.

---

## 3. Full directory tree

```
robocasa/robocasa365_checkpoints/  (HF model repo)
└── gr00t_n1-5/
    ├── multitask_learning/
    │   └── checkpoint-120000/                                   ← Sec 4.1, Table 1: 9.6%
    │
    ├── foundation_model_learning/                               ← Sec 4.2, Table 2
    │   ├── pretraining/
    │   │   └── checkpoint-80000/                                ← Stage 1 only: 0% composite-seen
    │   │
    │   ├── target_only/                                         ← target FT only, NO pretrain
    │   │   ├── atomic_seen/checkpoint-60000/                    ← Atomic 60.6%
    │   │   ├── composite_seen/checkpoint-60000/                 ← Composite-Seen 35.0%
    │   │   └── composite_unseen/checkpoint-60000/               ← Composite-Unseen 33.3%
    │   │
    │   └── target_posttraining/                                 ← pretrain → target FT (best)
    │       ├── atomic_seen/checkpoint-60000/                    ← Atomic 68.5% ★
    │       ├── composite_seen/checkpoint-60000/                 ← Composite-Seen 40.6% ★
    │       └── composite_unseen/checkpoint-60000/               ← Composite-Unseen 42.1% ★
    │
    └── lifelong_learning/                                       ← Sec 4.3, Table 3
        ├── phase1/checkpoint-100000/                            ← atomic 65 task
        ├── phase2/checkpoint-60000/                             ← + 2-3 stage composite
        ├── phase3/checkpoint-60000/                             ← + 4-5 stage composite
        └── phase4/checkpoint-60000/                             ← + 6+ stage composite
```

12 checkpoints total. **No README anywhere**; the repo-level README is just
the Apache-2.0 license header (31 bytes).

---

## 4. Master mapping table (HF path × paper regime × success rates)

All numbers are task success rates in % from the RoboCasa365 paper.

| HF path under `gr00t_n1-5/` | Paper section / table | Atomic-Seen | Composite-Seen | Composite-Unseen | Avg |
|---|---|---|---|---|---|
| `multitask_learning/checkpoint-120000` | Sec 4.1, Table 1 | 43.0 | 9.6 | 4.4 | 20.0 |
| `foundation_model_learning/pretraining/checkpoint-80000` | Sec 4.2, Table 2 ("Pretraining Only") | 41.9 | 0.0 | 0.2 | 15.1 |
| `foundation_model_learning/target_only/atomic_seen/checkpoint-60000` | Sec 4.2, Table 2 ("Target Only 100%") | 60.6 | — | — | — |
| `foundation_model_learning/target_only/composite_seen/checkpoint-60000` | Sec 4.2, Table 2 | — | 35.0 | — | — |
| `foundation_model_learning/target_only/composite_unseen/checkpoint-60000` | Sec 4.2, Table 2 | — | — | 33.3 | — |
| **`foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000`** | Sec 4.2, Table 2 ("Pretrain+Target 100%") | **68.5** | — | — | — |
| **`foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000`** | Sec 4.2, Table 2 | — | **40.6** | — | — |
| **`foundation_model_learning/target_posttraining/composite_unseen/checkpoint-60000`** | Sec 4.2, Table 2 | — | — | **42.1** | — |
| `lifelong_learning/phase{1..4}/...` | Sec 4.3, Table 3 | (separate task taxonomy — see lifelong table below) | | | |

Lifelong has its own task taxonomy (Atomic 65 / 2-3-stage 20 / 4-5-stage 20 /
6+-stage 20) that does not align with the foundation-model
atomic/composite-seen/composite-unseen split. Numbers from Table 3:

| Phase / HF path | Atomic 65 | 2-3 stage | 4-5 stage | 6+ stage |
|---|---|---|---|---|
| `phase1/checkpoint-100000` | 41.5 | — | — | — |
| `phase2/checkpoint-60000` | 13.9 | 24.5 | — | — |
| `phase3/checkpoint-60000` | 13.9 | 4.8 | 11.3 | — |
| `phase4/checkpoint-60000` | 10.6 | 1.7 | 2.7 | 4.3 |

Each phase fine-tunes from the previous phase's checkpoint. The atomic
score collapsing from 41.5 → 13.9 → 10.6 across phases is the paper's
illustration of catastrophic forgetting under sequential training.

The same 40.6% number appears twice more in the paper:
- Table 4 (pretraining diversity) under "Human300 + MG60" at 100% target data
- Table 9 (robustness) as the "No Perturbation" baseline

These are the same checkpoint, not separate runs.

---

## 5. Why are there three sub-folders under `target_only/` and `target_posttraining/`?

Because the 50 target tasks are split into three evaluation clusters and
**each cluster has its own dedicated fine-tuned model**. This is stated
directly in the paper §4.2: *"fine-tuning **independently** on three
separate target split datasets (Atomic, Composite-Seen, Composite-Unseen)"*.
And the LFS sha256 hashes of `model-00001-of-00002.safetensors` differ
across the three sub-folders (atomic_seen `50fc6020…`, composite_seen
`672b8e49…`, composite_unseen `2c8cbfe7…`), confirming the three folders
hold byte-distinct weight files, not eval shards of the same weights.

The 50 target tasks split into:

- **Atomic-Seen** = 18 tasks (e.g., `TurnOnElectricKettle`, `OpenCabinet`, `TurnOnMicrowave`)
- **Composite-Seen** = 16 tasks (e.g., `PrepareCoffee`, `StackBowlsCabinet`, `LoadDishwasher`)
- **Composite-Unseen** = 16 tasks (e.g., `RecycleBottlesByType`, `WaffleReheat`, `ArrangeBreadBasket`)

Total 18 + 16 + 16 = 50 = the target task set in Table 11.

So Table 2's "Pretrain+Target Post-Training 100%" column is **three
separate models**, each fine-tuned on its own ~16-18 tasks. Same for
"Target Only 100%". Practical consequence: if you only care about
composite-seen evaluation (PrepareCoffee, etc.) you only need to
download `target_posttraining/composite_seen/`.

---

## 6. Pretraining gain (target_only → target_posttraining)

`target_only` trains from scratch on just the target subset (~8k demos
for a composite split). `target_posttraining` does the **same stage 2**
but starting from a checkpoint that has already gone through stage 1.
The size of stage 1:

| | Stage 1 (pretrain) | Stage 2 (target FT) |
|---|---|---|
| Tasks | 300 (human) + 60 (MimicGen) | one of {18 atomic, 16 composite-seen, 16 composite-unseen} |
| Hours | 411 + 1,615 ≈ 2,000 | a fraction of 208 |
| Demos | ~30k human + ~600k synthetic ≈ 630k | 16–18 × 500 = 8–9k |
| Steps | 80,000 | 60,000 |

The gain from including stage 1 (Table 2, "Target Only 100%" → "Pretrain+Target 100%"):

| Split | target_only | target_posttraining | Δ |
|---|---|---|---|
| Atomic-Seen | 60.6 | 68.5 | +7.9 |
| Composite-Seen | 35.0 | 40.6 | +5.6 |
| Composite-Unseen | 33.3 | 42.1 | +8.8 |

Stage 1 builds general-purpose kitchen knowledge — how grippers grasp,
how fridge doors swing, how to navigate counters, what objects look
like — that transfers down to the target tasks. Calling this a
"prior" is a useful intuition but not a paper-measured ablation; the
paper does not isolate "knowledge transfer" from pure data-scale or
extra-compute effects. What the paper does measure (Tables 4 and 8) is
that more pretraining tasks and more pretraining scenes both improve
downstream success — which is consistent with the prior-transfer story
but doesn't prove it.

**Seen vs unseen labels.** `composite_seen` tasks are part of the
300-task pretraining pool at the standard demo density; `composite_unseen`
tasks are held out from stage 1 entirely. Composite-unseen scores
*slightly higher* than composite-seen across the Pretrain+Target column
(42.1 vs 40.6) and across the Human300-only ablation in Table 4
(44.0 vs 41.2). The paper offers no explanation; both "the unseen task
set happens to be easier" and "stage-1 priors transfer well to held-out
tasks at this scale" are consistent with the data.

---

## 7. Per-checkpoint download URLs

All checkpoints share the same file layout:

```
checkpoint-{N}/
├── config.json                       (~1.7 KB — model architecture)
├── model-00001-of-00002.safetensors  (~5.0 GB — weights shard 1)
├── model-00002-of-00002.safetensors  (~2.6 GB — weights shard 2)
├── model.safetensors.index.json      (~105 KB — shard map)
├── optimizer.pt                      (~8.6 GB — only needed to resume training)
├── rng_state.pth                     (~15 KB)
├── scheduler.pt                      (~1.5 KB)
├── trainer_state.json                (~1-2 MB — step count, loss curve)
└── experiment_cfg/
    └── metadata.json                 (~14 KB — embodiment + obs/action stats)
```

For **inference only**, you can skip `optimizer.pt` and save ~8.6 GB
per checkpoint. Use `huggingface-cli download --include` to filter:

```sh
huggingface-cli download robocasa/robocasa365_checkpoints \
    --include "gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000/*" \
    --exclude "**/optimizer.pt" \
    --local-dir ./gr00t_ckpts
```

Direct browse URLs:

| Path | URL |
|---|---|
| Multi-task | <https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/gr00t_n1-5/multitask_learning/checkpoint-120000> |
| Pretraining only | <https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/gr00t_n1-5/foundation_model_learning/pretraining/checkpoint-80000> |
| Target only — atomic_seen | <https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/gr00t_n1-5/foundation_model_learning/target_only/atomic_seen/checkpoint-60000> |
| Target only — composite_seen | <https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/gr00t_n1-5/foundation_model_learning/target_only/composite_seen/checkpoint-60000> |
| Target only — composite_unseen | <https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/gr00t_n1-5/foundation_model_learning/target_only/composite_unseen/checkpoint-60000> |
| **Target post-training — atomic_seen** ★ | <https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000> |
| **Target post-training — composite_seen** ★ | <https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000> |
| **Target post-training — composite_unseen** ★ | <https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_unseen/checkpoint-60000> |
| Lifelong phase 1 | <https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/gr00t_n1-5/lifelong_learning/phase1/checkpoint-100000> |
| Lifelong phase 2 | <https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/gr00t_n1-5/lifelong_learning/phase2/checkpoint-60000> |
| Lifelong phase 3 | <https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/gr00t_n1-5/lifelong_learning/phase3/checkpoint-60000> |
| Lifelong phase 4 | <https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/gr00t_n1-5/lifelong_learning/phase4/checkpoint-60000> |

The `lifelong_learning/phase1/checkpoint-100000` folder additionally
contains `evals/train/` with 18 task subdirectories of ~30 rollout MP4s
and `stats.json` per task — useful as a reference of what successful
rollouts look like.

---

## 8. Model architecture details (from `config.json` + `metadata.json`)

- **Backbone**: NVEagle with Qwen3-1.7B LLM + SigLip2-400M vision encoder
- **Action head**: diffusion (16-step horizon)
- **Compute dtype**: bf16
- **What's tuned**: only the **action / diffusion head**. Both the LLM
  and the vision encoder are **frozen** in every released checkpoint —
  this matches the open-source GR00T-N1.5 default and is stated in the
  paper Appendix G.1: "we freeze the vision encoder and language encoder".
  The `tune_llm: false` flag in `config.json` reflects this.
- **Cameras (3, all at 256×256, 20 fps)**:
  - `robot0_eye_in_hand`
  - `robot0_agentview_left`
  - `robot0_agentview_right`
- **State (5D groups)**: `gripper_qpos`, `base_position`, `base_rotation`,
  `end_effector_position_relative`, `end_effector_rotation_relative`
- **Action (32D total)**: 6D EE pose (pos + rot) + 4D base motion +
  `control_mode` + `gripper`

This matches the camera_names that RoboCasaGymEnv exposes by default in
the eval container — so the model can be wired directly to a
`gym.make("robocasa/<TaskName>")` env without any obs/action remapping.

---

## 9. Robustness numbers (Table 9)

These are the two `target_posttraining/composite_{seen,unseen}/` models
evaluated under perturbations. Useful to know what to expect if you're
testing OOD robustness with our Docker eval container.

| Perturbation | Composite-Seen | Composite-Unseen |
|---|---|---|
| None (baseline) | 40.6 | 42.1 |
| Novel language | 38.3 | 39.2 |
| Camera perturbations | 28.8 | 31.5 |
| Initial joint noise | 27.9 | 32.1 |
| Initial base pose noise | 31.2 | 30.2 |

Camera and initial-pose perturbations cost ~10 pp. Language perturbation
costs only ~2 pp.

---

## 10. Per-task numbers for the headline checkpoint (Table 11 highlights)

`target_posttraining/composite_seen/checkpoint-60000` per-task results
on the 16 Composite-Seen tasks, 30 rollouts each:

| Task | SR (%) |
|---|---|
| `StackBowlsCabinet` | 83 |
| `PreSoakPan` | 70 |
| `ScrubCuttingBoard` | 70 |
| `WashLettuce` | 67 |
| `RinseSinkBasin` | 60 |
| `KettleBoiling` | 53 |
| `LoadDishwasher` | 47 |
| `StoreLeftoversInBowl` | 43 |
| `SetUpCuttingStation` | 33 |
| `StirVegetables` | 33 |
| `SteamInMicrowave` | 30 |
| `SearingMeat` | 27 |
| `PackIdenticalLunches` | 17 |
| `PrepareCoffee` | 13 |
| `DeliverStraw` | 3 |
| `GetToastedBread` | 0 |

The 40.6% headline is the average of these. Individual task SRs span
0–83 % — useful when you're choosing which task to run as a demo.

---

## 11. What's NOT released

The paper mentions but does not appear to release as separate
checkpoints:

- **Target Only 10% / 30% data** ablations (Table 2) — only the 100%
  variant is on HF.
- **Pretraining diversity ablations** (Table 4: Human50, Human300,
  Human300 + MG60 at 10% target data) — only the headline (Human300+MG60
  at 100% target) is on HF.
- **Pretraining scene diversity ablations** (Appendix H.1, Table 8:
  5 / 25 / 2,500 scenes) — paper-only.
- **LoRA fine-tuning** comparison (Appendix H.4, Table 10) — paper-only,
  no separate checkpoint.
- **Joint co-training ablation** (Appendix H.3) — paper-only.
- **Sim-and-real / real-world** model (Sec 4.5, Table 5) — paper-only.

If the folder name doesn't match one of the 12 paths in §3, it's not
been released.

---

## 12. References

- **Paper** — RoboCasa365 (ICLR 2026): <https://robocasa.ai/assets/robocasa365_iclr26.pdf>
- **HF model repo**: <https://huggingface.co/robocasa/robocasa365_checkpoints>
- **GR00T-N1.5 model architecture**: <https://github.com/NVIDIA/Isaac-GR00T>
- **Local PDF in this repo**: `RoboCasa_365_paper.pdf`
- **Original RoboCasa (RSS 2024)** for reference: `RoboCasa_paper.pdf` —
  the 365 release is a strict superset; everything in this doc applies
  to RoboCasa365 only.

---

*This doc was assembled by cross-referencing the HuggingFace folder
tree against the RoboCasa365 paper because the HF repo ships no README.
If the upstream repo adds an authoritative README later, prefer that
over the inferences here, especially around the per-split sub-folder
semantics in §5.*
