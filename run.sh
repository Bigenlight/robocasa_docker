#!/usr/bin/env bash
set -euo pipefail

# RoboCasa Docker runner.
#
# Modes:
#   ./run.sh                          # smoke test (EGL, OSMesa fallback)
#   ./run.sh --smoke-test             # alias of default
#   ./run.sh --download-assets        # one-time ~10GB asset download
#   ./run.sh --rollout <Task>         # run a random rollout, write mp4 to test_outputs/
#   ./run.sh --shell                  # interactive bash inside the container
#   ./run.sh --build                  # docker build the image locally
#
# All bind mounts:
#   <repo>/                  -> /workspace/robocasa  (rw)   robocasa source + assets
#   <repo>/robosuite/        -> /workspace/robosuite (rw)   robosuite source (must be host-cloned)
#   <repo>/test_outputs/     -> /workspace/robocasa/test_outputs (rw)
#
# Source-of-truth principle (mirrors Libero-pro v1.3):
#   - The image bakes only pip dependencies.
#   - robocasa/, robosuite/, test_outputs/, models/assets/ all live on the host.
#   - This means deleting and rebuilding the image preserves your work.

IMAGE="${ROBOCASA_IMAGE:-robocasa-eval:latest}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOSUITE_DIR="${REPO_DIR}/robosuite"
OUTPUT_DIR="${REPO_DIR}/test_outputs"
ASSETS_DIR="${REPO_DIR}/robocasa/models/assets"

# Default smoke-test task. Composite PnP is the canonical demo; can be
# overridden with --rollout <Task>.
DEFAULT_TASK="PickPlaceCounterToSink"

MODE="smoke"
ROLLOUT_TASK="$DEFAULT_TASK"
ROLLOUT_STEPS=60
ROLLOUT_NUM=1
ROLLOUT_SEED=0
ROLLOUT_SEED_BASE=0
ROLLOUT_MAX_STEPS=""
ROLLOUT_REPLAY_CHUNK=""
ROLLOUT_SPLIT="pretrain"
GROOT_SERVER="${GROOT_SERVER:-http://localhost:8500}"
EXTRA_DOCKER_ARGS=""

usage() {
    cat <<USAGE
Usage:
  ./run.sh                                # default: smoke test
  ./run.sh --smoke-test                   # 6-step smoke test, writes png + mp4
  ./run.sh --download-assets              # one-time ~10GB asset bootstrap
  ./run.sh --rollout <Task> [opts]        # random rollout into test_outputs/
        --steps N        rollout length (default ${ROLLOUT_STEPS})
        --num N          rollouts per call (default ${ROLLOUT_NUM})
        --seed N         seed (default ${ROLLOUT_SEED})
  ./run.sh --shell                        # interactive bash
  ./run.sh --build                        # docker build the image
  ./run.sh --groot-eval <Task> [opts]     # run an eval against the GR00T HTTP server
        --steps N        rollout length (default ${ROLLOUT_STEPS})
        --seed N         seed (default ${ROLLOUT_SEED})
        (set GROOT_SERVER=http://host:port to override; default localhost:8500)
  ./run.sh --canonical-eval <Task> [opts]
        Canonical Isaac-GR00T eval pipeline:
          - n_action_steps=16 full chunk replay
          - per-task horizon from get_task_horizon(<Task>)
          - 5-state, 5-action-subkey, 3-camera contract (matches metadata.json)
          - emits N mp4s + groot_<Task>_summary.json
        Options:
          --num-rollouts N   number of trials (default 1)
          --seed-base N      first trial seed (default 0); seeds are base..base+N-1
          --max-steps N      override per-task horizon
          --replay-chunk N   override full-chunk replay (1 = closed-loop)
          --split pretrain|target  scenes to evaluate on (default ${ROLLOUT_SPLIT}).
                                   Use 'pretrain' for the multitask checkpoint
                                   (Sec 4.1 / Table 1 of the paper).
                                   Use 'target' for any target_only/target_pt/
                                   pretraining checkpoint (Sec 4.2 / Table 2).

Environment:
  ROBOCASA_IMAGE   override image tag (default: robocasa-eval:latest)
  GROOT_SERVER     URL of the GR00T HTTP server for --groot-eval mode
USAGE
}

# ── arg parse ────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test)         MODE="smoke"; shift ;;
        --download-assets)    MODE="assets"; shift ;;
        --rollout)            MODE="rollout"; ROLLOUT_TASK="$2"; shift 2 ;;
        --groot-eval)         MODE="groot-eval"; ROLLOUT_TASK="$2"; shift 2 ;;
        --canonical-eval)     MODE="canonical-eval"; ROLLOUT_TASK="$2"; shift 2 ;;
        --shell)              MODE="shell"; shift ;;
        --build)              MODE="build"; shift ;;
        --steps)              ROLLOUT_STEPS="$2"; shift 2 ;;
        --num)                ROLLOUT_NUM="$2"; shift 2 ;;
        --num-rollouts)       ROLLOUT_NUM="$2"; shift 2 ;;
        --seed)               ROLLOUT_SEED="$2"; ROLLOUT_SEED_BASE="$2"; shift 2 ;;
        --seed-base)          ROLLOUT_SEED_BASE="$2"; shift 2 ;;
        --max-steps)          ROLLOUT_MAX_STEPS="$2"; shift 2 ;;
        --replay-chunk)       ROLLOUT_REPLAY_CHUNK="$2"; shift 2 ;;
        --split)              ROLLOUT_SPLIT="$2"; shift 2 ;;
        -h|--help)            usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

# ── pre-flight ───────────────────────────────────────────
echo "=== Pre-flight ==="
if ! command -v docker &>/dev/null; then
    echo "ERROR: docker not found. Install Docker first." >&2
    exit 1
fi
echo "  Docker: $(docker --version | head -1)"

GPU_ARGS=()
if ! command -v nvidia-smi &>/dev/null; then
    echo "  GPU: nvidia-smi missing — will use CPU/OSMesa path"
elif ! docker info 2>/dev/null | grep -qi nvidia; then
    echo "  GPU: nvidia-container-toolkit not registered with docker — will use CPU/OSMesa path"
else
    echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    GPU_ARGS=(--gpus all)
fi

# Build mode short-circuits everything else.
if [[ "$MODE" == "build" ]]; then
    echo ""
    echo "=== docker build ==="
    docker build -t "$IMAGE" "$REPO_DIR"
    echo ""
    echo "Built: $IMAGE"
    exit 0
fi

# Image must exist for all non-build modes.
if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "  Image $IMAGE not found locally."
    echo "  Build it first:  ./run.sh --build"
    exit 1
fi
echo "  Image: $IMAGE"

# robosuite must be present on host (mounted into container).
if [[ ! -d "$ROBOSUITE_DIR/robosuite" ]]; then
    cat >&2 <<EOF
ERROR: $ROBOSUITE_DIR/robosuite not found.

  RoboCasa needs robosuite (master branch) at the location:
    $ROBOSUITE_DIR

  One-time setup:
    git clone https://github.com/ARISE-Initiative/robosuite "$ROBOSUITE_DIR"
EOF
    exit 1
fi
echo "  robosuite: $ROBOSUITE_DIR"

mkdir -p "$OUTPUT_DIR"
echo "  outputs: $OUTPUT_DIR"

# Asset existence check (informational only — bootstrap mode bypasses).
if [[ "$MODE" != "assets" ]]; then
    if [[ ! -d "$ASSETS_DIR/textures" ]] || [[ ! -d "$ASSETS_DIR/objects/objaverse" ]]; then
        echo "  WARNING: $ASSETS_DIR is missing texture/object packs."
        echo "           Run './run.sh --download-assets' first (~10GB, one time)."
    else
        echo "  assets: $ASSETS_DIR (looks populated)"
    fi
fi

# Match host UID/GID so files written into mounts (assets, mp4, macros_private.py)
# end up owned by the user, not root. HOME must be writable for pip / mujoco
# caches when running non-root.
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

# ── common docker run flags ──────────────────────────────
COMMON_RUN_ARGS=(
    --rm
    "${GPU_ARGS[@]}"
    --shm-size=8g
    --user "${HOST_UID}:${HOST_GID}"
    -e HOME=/tmp/robocasa-home
    -v "${REPO_DIR}:/workspace/robocasa"
    -v "${ROBOSUITE_DIR}:/workspace/robosuite"
    -v "${OUTPUT_DIR}:/workspace/robocasa/test_outputs"
    -e PYTHONPATH=/workspace/robocasa:/workspace/robosuite
    -w /workspace/robocasa
)

# Container preamble: ensure macros_private.py exists. Idempotent — only runs
# setup_macros if the file is missing. Output suppressed unless verbose.
PREAMBLE='
mkdir -p /tmp/robocasa-home
if [ ! -f /workspace/robocasa/robocasa/macros_private.py ]; then
    python -m robocasa.scripts.setup_macros >/dev/null 2>&1 || true
fi
if [ ! -f /workspace/robosuite/robosuite/macros_private.py ]; then
    python -m robosuite.scripts.setup_macros >/dev/null 2>&1 || true
fi
'

# ── modes ────────────────────────────────────────────────
echo ""
case "$MODE" in
    assets)
        echo "=== Downloading kitchen assets (~10GB) ==="
        echo "  This is a one-time download. Resumable: rerun if interrupted."
        # `yes y |` answers the script's interactive Proceed? prompt.
        # docker run -i keeps stdin open so the pipe is wired through.
        docker run -i "${COMMON_RUN_ARGS[@]}" \
            -e MUJOCO_GL=egl \
            "$IMAGE" \
            bash -c "$PREAMBLE
yes y | python -m robocasa.scripts.download_kitchen_assets --type all"
        echo ""
        echo "Done. Assets are in: $ASSETS_DIR"
        ;;

    shell)
        echo "=== Interactive shell ==="
        docker run -it "${COMMON_RUN_ARGS[@]}" \
            -e MUJOCO_GL=egl \
            "$IMAGE" \
            bash -c "$PREAMBLE
exec bash"
        ;;

    smoke)
        echo "=== Smoke test (EGL) ==="
        if docker run "${COMMON_RUN_ARGS[@]}" \
            -e MUJOCO_GL=egl \
            "$IMAGE" \
            bash -c "$PREAMBLE
python /workspace/robocasa/test_smoke.py --output-dir /workspace/robocasa/test_outputs"; then
            echo ""
            echo "=== ALL TESTS PASSED (EGL) ==="
        else
            EGL_RC=$?
            echo ""
            echo "=== EGL failed (exit $EGL_RC), retrying with OSMesa ==="
            echo ""
            if docker run "${COMMON_RUN_ARGS[@]}" \
                -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa \
                "$IMAGE" \
                bash -c "$PREAMBLE
python /workspace/robocasa/test_smoke.py --output-dir /workspace/robocasa/test_outputs"; then
                echo ""
                echo "=== ALL TESTS PASSED (OSMesa fallback) ==="
            else
                echo ""
                echo "=== TESTS FAILED ==="
                echo "Outputs (if any): $OUTPUT_DIR"
                exit 1
            fi
        fi
        echo "Outputs: $OUTPUT_DIR"
        ;;

    groot-eval)
        echo "=== GR00T eval: $ROLLOUT_TASK (${ROLLOUT_STEPS} steps, seed=${ROLLOUT_SEED}) ==="
        echo "  GR00T server: $GROOT_SERVER"
        VIDEO_NAME="groot_${ROLLOUT_TASK}_seed${ROLLOUT_SEED}.mp4"
        # --network host so http://localhost:8500 reaches the GR00T container
        # (which also runs with --network host).
        docker run "${COMMON_RUN_ARGS[@]}" \
            --network host \
            -e MUJOCO_GL=egl \
            -e GROOT_SERVER="$GROOT_SERVER" \
            "$IMAGE" \
            bash -c "$PREAMBLE
python /workspace/robocasa/examples/run_groot_eval.py \
    --task '$ROLLOUT_TASK' \
    --num-steps $ROLLOUT_STEPS \
    --seed $ROLLOUT_SEED \
    --server '$GROOT_SERVER' \
    --video /workspace/robocasa/test_outputs/$VIDEO_NAME"
        echo ""
        echo "Video: $OUTPUT_DIR/$VIDEO_NAME"
        ;;

    canonical-eval)
        echo "=== Canonical GR00T eval: $ROLLOUT_TASK ==="
        echo "  num_rollouts=${ROLLOUT_NUM}  seed_base=${ROLLOUT_SEED_BASE}"
        echo "  max_steps=${ROLLOUT_MAX_STEPS:-<auto get_task_horizon>}"
        echo "  replay_chunk=${ROLLOUT_REPLAY_CHUNK:-<auto full-chunk>}"
        echo "  GR00T server: $GROOT_SERVER"
        EXTRA_ARGS=""
        if [[ -n "$ROLLOUT_MAX_STEPS" ]]; then
            EXTRA_ARGS="$EXTRA_ARGS --max-steps $ROLLOUT_MAX_STEPS"
        fi
        if [[ -n "$ROLLOUT_REPLAY_CHUNK" ]]; then
            EXTRA_ARGS="$EXTRA_ARGS --replay-chunk $ROLLOUT_REPLAY_CHUNK"
        fi
        # --network host so http://localhost:8500 reaches the GR00T container
        # (which also runs with --network host).
        docker run "${COMMON_RUN_ARGS[@]}" \
            --network host \
            -e MUJOCO_GL=egl \
            -e GROOT_SERVER="$GROOT_SERVER" \
            "$IMAGE" \
            bash -c "$PREAMBLE
python /workspace/robocasa/examples/run_groot_eval.py \
    --task '$ROLLOUT_TASK' \
    --split '$ROLLOUT_SPLIT' \
    --num-rollouts $ROLLOUT_NUM \
    --seed-base $ROLLOUT_SEED_BASE \
    --server '$GROOT_SERVER' \
    --output-dir /workspace/robocasa/test_outputs \
    $EXTRA_ARGS"
        echo ""
        echo "Per-trial mp4s + summary.json in: $OUTPUT_DIR"
        echo "  Look for: groot_<Task>_seed<N>_success<0|1>.mp4"
        echo "            groot_<Task>_summary.json"
        ;;

    rollout)
        echo "=== Random rollout: $ROLLOUT_TASK (${ROLLOUT_NUM} rollouts × ${ROLLOUT_STEPS} steps) ==="
        VIDEO_NAME="${ROLLOUT_TASK}_seed${ROLLOUT_SEED}.mp4"
        if docker run "${COMMON_RUN_ARGS[@]}" \
            -e MUJOCO_GL=egl \
            "$IMAGE" \
            bash -c "$PREAMBLE
python /workspace/robocasa/examples/run_random_rollouts.py \
    --task '$ROLLOUT_TASK' \
    --num-rollouts $ROLLOUT_NUM \
    --num-steps $ROLLOUT_STEPS \
    --seed $ROLLOUT_SEED \
    --video /workspace/robocasa/test_outputs/$VIDEO_NAME"; then
            :
        else
            EGL_RC=$?
            echo ""
            echo "=== EGL failed (exit $EGL_RC), retrying with OSMesa ==="
            docker run "${COMMON_RUN_ARGS[@]}" \
                -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa \
                "$IMAGE" \
                bash -c "$PREAMBLE
python /workspace/robocasa/examples/run_random_rollouts.py \
    --task '$ROLLOUT_TASK' \
    --num-rollouts $ROLLOUT_NUM \
    --num-steps $ROLLOUT_STEPS \
    --seed $ROLLOUT_SEED \
    --video /workspace/robocasa/test_outputs/$VIDEO_NAME"
        fi
        echo ""
        echo "Video: $OUTPUT_DIR/$VIDEO_NAME"
        ;;
esac
