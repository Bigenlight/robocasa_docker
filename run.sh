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

Environment:
  ROBOCASA_IMAGE   override image tag (default: robocasa-eval:latest)
USAGE
}

# ── arg parse ────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test)         MODE="smoke"; shift ;;
        --download-assets)    MODE="assets"; shift ;;
        --rollout)            MODE="rollout"; ROLLOUT_TASK="$2"; shift 2 ;;
        --shell)              MODE="shell"; shift ;;
        --build)              MODE="build"; shift ;;
        --steps)              ROLLOUT_STEPS="$2"; shift 2 ;;
        --num)                ROLLOUT_NUM="$2"; shift 2 ;;
        --seed)               ROLLOUT_SEED="$2"; shift 2 ;;
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
