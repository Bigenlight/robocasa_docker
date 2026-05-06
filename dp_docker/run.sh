#!/usr/bin/env bash
set -euo pipefail

# Diffusion Policy in-process eval runner.
#
# Modes:
#   ./run.sh --build                              docker build the image
#   ./run.sh --download-ckpt                      huggingface-cli download latest.ckpt (~1.7 GB)
#   ./run.sh --eval [<Task>] [extra args]         in-process eval (default: PickPlaceCounterToSink, 5 rollouts)
#   ./run.sh --smoke                              ckpt-load + cfg-resolve check, no rollout (~30 s)
#   ./run.sh --list-tasks                         print TASK_SET_REGISTRY keys and tasks
#   ./run.sh --shell                              interactive bash inside the container
#   ./run.sh --stop                               docker stop the running named container
#
# Bind mounts (image bakes Python deps only):
#   $DP_DIR/eval_dp.py             -> /dp/eval_dp.py            (ro)
#   $DP_DIR/diffusion_policy       -> /dp/diffusion_policy      (rw)
#   $DP_DIR/checkpoint             -> /dp/checkpoint            (ro)
#   $DP_DIR/test_outputs           -> /dp/test_outputs          (rw)
#   $REPO_DIR (robocasa source)    -> /workspace/robocasa       (rw)
#   $REPO_DIR/robosuite            -> /workspace/robosuite      (rw)
#   ~/.cache/huggingface           -> /tmp/dp-home/.cache/huggingface (rw)
#
# Env overrides:
#   DP_IMAGE             (default: dp-eval:latest)
#   DP_CONTAINER_NAME    (default: dp-eval)
#   DP_CKPT_PATH         (in-container path to the .ckpt; default: latest.ckpt)
#   DP_DEVICE            (default: cuda:0)
#   MUJOCO_GL            (default: egl)

IMAGE="${DP_IMAGE:-dp-eval:latest}"
CONTAINER_NAME="${DP_CONTAINER_NAME:-dp-eval}"
DEVICE="${DP_DEVICE:-cuda:0}"
MUJOCO_GL_VAL="${MUJOCO_GL:-egl}"

DP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$DP_DIR/.." && pwd)"
DP_FORK_DIR="$DP_DIR/diffusion_policy"
CKPT_DIR="$DP_DIR/checkpoint"
EVAL_OUT="$DP_DIR/test_outputs"
HF_CACHE="${HOME}/.cache/huggingface"

CKPT_REL_DEFAULT="diffusion_policy/17.40.09_train_diffusion_transformer_hybrid_pretrain_human300/checkpoints/latest.ckpt"
CKPT_PATH="${DP_CKPT_PATH:-/dp/checkpoint/$CKPT_REL_DEFAULT}"

MODE=""
EVAL_TASK="PickPlaceCounterToSink"
EXTRA_EVAL_ARGS=()

usage() {
    cat <<EOF
Usage:
  ./run.sh --build                              docker build the image
  ./run.sh --download-ckpt                      download latest.ckpt (~1.7 GB)
  ./run.sh --eval [<Task>] [-- ...]             in-process eval
                                                default Task = PickPlaceCounterToSink
                                                extra args after '--' forwarded to eval_dp.py
                                                  --num-rollouts N (default 5)
                                                  --seed-base N    (default 0)
                                                  --max-steps N
                                                  --num-envs N     (default 1)
                                                  --split S        (default pretrain)
  ./run.sh --smoke                              ckpt-load + cfg-resolve check (no rollout)
  ./run.sh --list-tasks                         print TASK_SET_REGISTRY keys and members
  ./run.sh --shell                              interactive bash
  ./run.sh --stop                               docker stop $CONTAINER_NAME

Env overrides:
  DP_IMAGE          image tag                       (default: dp-eval:latest)
  DP_CONTAINER_NAME container name                  (default: dp-eval)
  DP_CKPT_PATH      in-container path to the .ckpt  (default: latest.ckpt)
  DP_DEVICE         CUDA device                     (default: cuda:0)
  MUJOCO_GL         render backend                  (default: egl)

Example — eval against the leaderboard-aligned epoch=0500 ckpt
(must be downloaded separately first; see README §"Checkpoint layout"):
  DP_CKPT_PATH="/dp/checkpoint/diffusion_policy/17.40.09_train_diffusion_transformer_hybrid_pretrain_human300/checkpoints/epoch=0500-test_mean_score=-1.000.ckpt" \\
    ./run.sh --eval PickPlaceCounterToSink --num-rollouts 5
EOF
}

# ── arg parse ─────────────────────────────────────────────────────────────
if [[ $# -eq 0 ]]; then usage; exit 1; fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build|--download-ckpt|--smoke|--list-tasks|--shell|--stop)
            [[ -n "$MODE" ]] && { echo "Error: multiple mode flags" >&2; exit 1; }
            MODE="${1#--}"; shift ;;
        --eval)
            [[ -n "$MODE" ]] && { echo "Error: multiple mode flags" >&2; exit 1; }
            MODE="eval"; shift
            # Optional positional Task
            if [[ $# -gt 0 && "$1" != --* && "$1" != "--" ]]; then
                EVAL_TASK="$1"; shift
            fi
            ;;
        --) shift; EXTRA_EVAL_ARGS+=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *)
            # If we're already in eval mode, treat unknown args as forwarded
            if [[ "$MODE" == "eval" ]]; then
                EXTRA_EVAL_ARGS+=("$1"); shift
            else
                echo "Unknown option: $1" >&2; usage; exit 1
            fi
            ;;
    esac
done
[[ -z "$MODE" ]] && { usage; exit 1; }

# ── pre-flight ───────────────────────────────────────────────────────────
need_docker() {
    command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found" >&2; exit 1; }
}

need_image() {
    if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
        echo "ERROR: image '$IMAGE' missing — run ./run.sh --build first" >&2
        exit 1
    fi
}

GPU_ARGS=()
if command -v nvidia-smi >/dev/null 2>&1 && docker info 2>/dev/null | grep -qi nvidia; then
    GPU_ARGS=(--gpus all)
fi

# ── build ─────────────────────────────────────────────────────────────────
if [[ "$MODE" == "build" ]]; then
    need_docker
    echo "=== docker build $IMAGE ==="
    set -x
    docker build -t "$IMAGE" "$DP_DIR"
    set +x
    echo "Built: $IMAGE"
    exit 0
fi

# ── download-ckpt ─────────────────────────────────────────────────────────
if [[ "$MODE" == "download-ckpt" ]]; then
    need_docker; need_image
    mkdir -p "$CKPT_DIR" "$HF_CACHE"
    echo "=== downloading $CKPT_REL_DEFAULT (excluding optimizer state) ==="
    set -x
    docker run --rm \
        --user "$(id -u):$(id -g)" \
        -e HOME=/tmp/dp-home \
        -v "$CKPT_DIR:/dp/checkpoint" \
        -v "$HF_CACHE:/tmp/dp-home/.cache/huggingface" \
        "$IMAGE" \
        bash -c "
            mkdir -p /tmp/dp-home
            huggingface-cli download robocasa/robocasa365_checkpoints \
                --include 'diffusion_policy/17.40.09_train_diffusion_transformer_hybrid_pretrain_human300/checkpoints/latest.ckpt' \
                --exclude '*optimizer*' \
                --local-dir /dp/checkpoint
        "
    set +x
    echo ""
    echo "Done. Checkpoint at: $CKPT_DIR/$CKPT_REL_DEFAULT"
    ls -la "$CKPT_DIR/$(dirname "$CKPT_REL_DEFAULT")" 2>/dev/null | head -10 || true
    exit 0
fi

# All non-build/download modes need the image
need_docker; need_image

# ── stop ──────────────────────────────────────────────────────────────────
if [[ "$MODE" == "stop" ]]; then
    docker stop "$CONTAINER_NAME" 2>/dev/null && echo "Stopped $CONTAINER_NAME" || echo "Not running"
    exit 0
fi

# ── pre-flight for runtime modes ─────────────────────────────────────────
if [[ "$MODE" =~ ^(eval|smoke|list-tasks)$ ]]; then
    if [[ ! -d "$DP_FORK_DIR" || ! -d "$DP_FORK_DIR/diffusion_policy" ]]; then
        cat >&2 <<EOM
ERROR: $DP_FORK_DIR/diffusion_policy not found.

  Clone the robocasa-benchmark fork first:
    git clone https://github.com/robocasa-benchmark/diffusion_policy.git "$DP_FORK_DIR"
EOM
        exit 1
    fi
fi
if [[ "$MODE" =~ ^(eval|smoke)$ ]]; then
    HOST_CKPT="$CKPT_DIR/$CKPT_REL_DEFAULT"
    if [[ "${DP_CKPT_PATH:-}" == "" && ! -f "$HOST_CKPT" ]]; then
        cat >&2 <<EOM
ERROR: checkpoint not found at $HOST_CKPT

  Download it first:
    ./run.sh --download-ckpt
EOM
        exit 1
    fi
fi
if [[ "$MODE" == "eval" ]]; then
    if [[ ! -f "$DP_DIR/eval_dp.py" ]]; then
        echo "ERROR: $DP_DIR/eval_dp.py not found (the wrapper hasn't been added yet)" >&2
        exit 1
    fi
fi

# Auto-create host dirs that get bind-mounted rw
mkdir -p "$EVAL_OUT" "$HF_CACHE"

# ── common docker run flags ──────────────────────────────────────────────
COMMON_ARGS=(
    --rm
    "${GPU_ARGS[@]}"
    --shm-size=8g
    --user "$(id -u):$(id -g)"
    -e HOME=/tmp/dp-home
    -e "MUJOCO_GL=$MUJOCO_GL_VAL"
    -e "PYTHONPATH=/dp/diffusion_policy:/workspace/robocasa:/workspace/robosuite"
    -v "$HF_CACHE:/tmp/dp-home/.cache/huggingface"
)
# Bind-mount script + sources only when present on host
[[ -f "$DP_DIR/eval_dp.py" ]] && COMMON_ARGS+=( -v "$DP_DIR/eval_dp.py:/dp/eval_dp.py:ro" )
[[ -d "$DP_FORK_DIR" ]]      && COMMON_ARGS+=( -v "$DP_FORK_DIR:/dp/diffusion_policy" )
[[ -d "$CKPT_DIR" ]]         && COMMON_ARGS+=( -v "$CKPT_DIR:/dp/checkpoint:ro" )
COMMON_ARGS+=( -v "$EVAL_OUT:/dp/test_outputs" )
[[ -d "$REPO_DIR/robocasa" ]]  && COMMON_ARGS+=( -v "$REPO_DIR:/workspace/robocasa" )
[[ -d "$REPO_DIR/robosuite" ]] && COMMON_ARGS+=( -v "$REPO_DIR/robosuite:/workspace/robosuite" )

PREAMBLE='mkdir -p /tmp/dp-home /tmp/dp-home/.cache/huggingface'

# ── run modes ─────────────────────────────────────────────────────────────
case "$MODE" in
    eval)
        echo "=== eval (in-process) task=$EVAL_TASK device=$DEVICE ==="
        # Fill-missing-only: only inject a default if the user didn't already
        # pass that flag in EXTRA_EVAL_ARGS. Keeps `set -x` output clean.
        has_flag() {
            local needle="$1"; shift
            for a in "$@"; do [[ "$a" == "$needle" ]] && return 0; done
            return 1
        }
        EVAL_ARGS=()
        has_flag --task          "${EXTRA_EVAL_ARGS[@]:-}" || EVAL_ARGS+=(--task "$EVAL_TASK")
        has_flag --num-rollouts  "${EXTRA_EVAL_ARGS[@]:-}" || EVAL_ARGS+=(--num-rollouts 5)
        has_flag --seed-base     "${EXTRA_EVAL_ARGS[@]:-}" || EVAL_ARGS+=(--seed-base 0)
        has_flag --num-envs      "${EXTRA_EVAL_ARGS[@]:-}" || EVAL_ARGS+=(--num-envs 1)
        has_flag --split         "${EXTRA_EVAL_ARGS[@]:-}" || EVAL_ARGS+=(--split pretrain)
        has_flag --device        "${EXTRA_EVAL_ARGS[@]:-}" || EVAL_ARGS+=(--device "$DEVICE")
        has_flag --ckpt          "${EXTRA_EVAL_ARGS[@]:-}" || EVAL_ARGS+=(--ckpt "$CKPT_PATH")
        has_flag --output-dir    "${EXTRA_EVAL_ARGS[@]:-}" || EVAL_ARGS+=(--output-dir /dp/test_outputs)
        EVAL_ARGS+=("${EXTRA_EVAL_ARGS[@]:-}")
        set -x
        docker run "${COMMON_ARGS[@]}" --name "$CONTAINER_NAME" \
            "$IMAGE" \
            bash -c "$PREAMBLE; python /dp/eval_dp.py ${EVAL_ARGS[*]}"
        set +x
        ;;
    smoke)
        echo "=== smoke: ckpt load + cfg resolve (no rollout) ==="
        set -x
        docker run "${COMMON_ARGS[@]}" --name "$CONTAINER_NAME" \
            "$IMAGE" \
            bash -c "$PREAMBLE; python /dp/eval_dp.py --smoke-only --ckpt $CKPT_PATH --device $DEVICE"
        set +x
        ;;
    list-tasks)
        echo "=== TASK_SET_REGISTRY keys and members ==="
        set -x
        docker run "${COMMON_ARGS[@]}" --name "$CONTAINER_NAME" \
            "$IMAGE" \
            bash -c "$PREAMBLE; python -c 'from robocasa.utils.dataset_registry import TASK_SET_REGISTRY; import json; print(json.dumps({k: sorted(v) for k, v in TASK_SET_REGISTRY.items()}, indent=2))'"
        set +x
        ;;
    shell)
        echo "=== interactive shell ==="
        docker run -it "${COMMON_ARGS[@]}" --name "$CONTAINER_NAME" \
            "$IMAGE" \
            bash -c "$PREAMBLE; exec bash"
        ;;
esac
