#!/usr/bin/env bash
set -euo pipefail

# GR00T-N1.5 HTTP server runner.
#
# Modes:
#   ./run.sh --build                 # docker build the image
#   ./run.sh --download-ckpt         # huggingface-cli download multitask checkpoint
#   ./run.sh --serve                 # foreground server (real model)
#   ./run.sh --serve-bg              # background, prints container id
#   ./run.sh --dummy                 # foreground, --dummy mode (no model, zero actions)
#   ./run.sh --smoke                 # spin up dummy server, hit /health and /act, tear down
#   ./run.sh --shell                 # interactive bash inside the container
#   ./run.sh --stop                  # docker stop the running named container
#
# Bind mounts (Libero-pro / openvla pattern: image bakes deps only):
#   $REPO_DIR/Isaac-GR00T -> /groot/Isaac-GR00T
#   $REPO_DIR/checkpoint  -> /groot/checkpoint  (read-only)
#   $REPO_DIR/serve_groot.py -> /groot/serve_groot.py  (read-only)
#   ~/.cache/huggingface  -> /tmp/groot-home/.cache/huggingface  (rw, optional)

IMAGE="${GROOT_IMAGE:-groot-server:latest}"
CONTAINER_NAME="${GROOT_CONTAINER_NAME:-groot-server}"
PORT="${GROOT_PORT:-8500}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_DIR="${REPO_DIR}/Isaac-GR00T"
CKPT_DIR="${REPO_DIR}/checkpoint"
HF_CACHE="${HOME}/.cache/huggingface"

MODE=""
EXTRA_SERVER_ARGS=()

usage() {
    cat <<EOF
Usage:
  ./run.sh --build                       docker build the image
  ./run.sh --download-ckpt [--all]       download multitask/checkpoint-120000
                                         (without --all, skips optimizer.pt to save 8.6GB)
  ./run.sh --serve                       foreground real model server
  ./run.sh --serve-bg                    background, named '$CONTAINER_NAME'
  ./run.sh --dummy                       foreground, no model, zero actions
  ./run.sh --smoke                       spin up dummy + curl /health + /act + tear down
  ./run.sh --shell                       interactive bash
  ./run.sh --stop                        docker stop $CONTAINER_NAME

Env:
  GROOT_IMAGE          override image tag (default: groot-server:latest)
  GROOT_CONTAINER_NAME override container name (default: groot-server)
  GROOT_PORT           override host port (default: 8500)
EOF
}

DOWNLOAD_ALL=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build|--download-ckpt|--serve|--serve-bg|--dummy|--smoke|--shell|--stop)
            [[ -n "$MODE" ]] && { echo "Error: multiple mode flags"; exit 1; }
            MODE="${1#--}"; shift ;;
        --all) DOWNLOAD_ALL=1; shift ;;
        --no-strict) EXTRA_SERVER_ARGS+=("--no-strict"); shift ;;
        --) shift; EXTRA_SERVER_ARGS+=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done
[[ -z "$MODE" ]] && { usage; exit 1; }

# ── pre-flight ───────────────────────────────────────────────────────────
need_docker() {
    command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found" >&2; exit 1; }
}

GPU_ARGS=()
if command -v nvidia-smi >/dev/null 2>&1 && docker info 2>/dev/null | grep -qi nvidia; then
    GPU_ARGS=(--gpus all)
fi

# ── build ─────────────────────────────────────────────────────────────────
if [[ "$MODE" == "build" ]]; then
    need_docker
    echo "=== docker build $IMAGE ==="
    docker build -t "$IMAGE" "$REPO_DIR"
    echo "Built: $IMAGE"
    exit 0
fi

# ── download-ckpt ─────────────────────────────────────────────────────────
if [[ "$MODE" == "download-ckpt" ]]; then
    need_docker
    if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
        echo "Image $IMAGE missing — run ./run.sh --build first" >&2; exit 1
    fi
    mkdir -p "$CKPT_DIR" "$HF_CACHE"
    EXCLUDE_LINE=""
    if [[ "$DOWNLOAD_ALL" -ne 1 ]]; then
        # single-quoted in the inner bash so the * is not glob-expanded by either shell.
        EXCLUDE_LINE="--exclude '*optimizer*'"
    fi
    echo "=== downloading multitask/checkpoint-120000 (excluding optimizer.pt unless --all) ==="
    docker run --rm \
        --user "$(id -u):$(id -g)" \
        -e HOME=/tmp/groot-home \
        -v "$CKPT_DIR:/groot/checkpoint" \
        -v "$HF_CACHE:/tmp/groot-home/.cache/huggingface" \
        "$IMAGE" \
        bash -c "
            mkdir -p /tmp/groot-home
            huggingface-cli download robocasa/robocasa365_checkpoints \
                --include 'gr00t_n1-5/multitask_learning/checkpoint-120000/*' \
                $EXCLUDE_LINE \
                --local-dir /groot/checkpoint
        "
    echo ""
    echo "Done. Checkpoint at: $CKPT_DIR/gr00t_n1-5/multitask_learning/checkpoint-120000"
    echo "Symlinking to /groot/checkpoint root for easier --model-path usage..."
    if [[ ! -e "$CKPT_DIR/experiment_cfg" ]]; then
        # Relative symlinks so they resolve both on the host and inside the container.
        REL_BASE="gr00t_n1-5/multitask_learning/checkpoint-120000"
        ln -sfn "$REL_BASE/experiment_cfg" "$CKPT_DIR/experiment_cfg"
        for f in "$CKPT_DIR/$REL_BASE"/*.safetensors* "$CKPT_DIR/$REL_BASE"/config.json; do
            [[ -e "$f" ]] && ln -sfn "$REL_BASE/$(basename "$f")" "$CKPT_DIR/$(basename "$f")"
        done
    fi
    ls -la "$CKPT_DIR" | head -20
    exit 0
fi

# All non-build modes need the image to exist
need_docker
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Image $IMAGE missing — run ./run.sh --build first" >&2; exit 1
fi

# ── stop ──────────────────────────────────────────────────────────────────
if [[ "$MODE" == "stop" ]]; then
    docker stop "$CONTAINER_NAME" 2>/dev/null && echo "Stopped $CONTAINER_NAME" || echo "Not running"
    exit 0
fi

# ── isaac-gr00t source check (for non-dummy / non-shell) ─────────────────
if [[ "$MODE" =~ ^(serve|serve-bg)$ ]]; then
    if [[ ! -d "$ISAAC_DIR/gr00t" ]]; then
        cat >&2 <<EOM
ERROR: $ISAAC_DIR/gr00t not found.

  Real-model mode needs a clone of NVIDIA Isaac-GR00T:
    git clone https://github.com/NVIDIA/Isaac-GR00T.git "$ISAAC_DIR"

  (Note: Isaac-GR00T's master branch may have moved past N1.5. If load
   fails, check upstream tags for an N1.5-compatible commit.)
EOM
        exit 1
    fi
    if [[ ! -d "$CKPT_DIR/experiment_cfg" ]]; then
        cat >&2 <<EOM
ERROR: $CKPT_DIR/experiment_cfg not found.

  The multitask checkpoint isn't downloaded. Run:
    ./run.sh --download-ckpt
EOM
        exit 1
    fi
fi

# ── common docker run flags ──────────────────────────────────────────────
COMMON_ARGS=(
    --rm
    "${GPU_ARGS[@]}"
    --shm-size=8g
    --network host
    --user "$(id -u):$(id -g)"
    -e HOME=/tmp/groot-home
    -v "$REPO_DIR/serve_groot.py:/groot/serve_groot.py:ro"
    -v "$HF_CACHE:/tmp/groot-home/.cache/huggingface"
)
# Mount checkpoint + Isaac-GR00T if they exist on host
if [[ -d "$ISAAC_DIR" ]]; then
    COMMON_ARGS+=( -v "$ISAAC_DIR:/groot/Isaac-GR00T" )
fi
if [[ -d "$CKPT_DIR" ]]; then
    COMMON_ARGS+=( -v "$CKPT_DIR:/groot/checkpoint:ro" )
fi

# Preamble:
#  1. ensure HOME / HF cache dirs exist
#  2. register Isaac-GR00T's dist-info via `pip install --user --no-deps -e`
#     so any importlib.metadata.version("gr00t") call inside the package
#     (which PYTHONPATH alone doesn't satisfy) finds the metadata. The
#     install is non-fatal — if it can't run, code that doesn't need
#     metadata still imports fine via PYTHONPATH.
#
# NB: do NOT end with a trailing newline — callers join with `; ...`.
PREAMBLE='mkdir -p /tmp/groot-home /tmp/groot-home/.cache/huggingface; if [ -d /groot/Isaac-GR00T ] && [ ! -d /tmp/groot-home/.local/lib ]; then pip install --user --no-deps -e /groot/Isaac-GR00T -q 2>/dev/null || true; fi'

# ── modes that talk to the server ─────────────────────────────────────────
case "$MODE" in
    serve)
        echo "=== serve (real model, foreground, port $PORT) ==="
        docker run "${COMMON_ARGS[@]}" --name "$CONTAINER_NAME" \
            "$IMAGE" \
            bash -c "$PREAMBLE; python /groot/serve_groot.py --host 0.0.0.0 --port $PORT "${EXTRA_SERVER_ARGS[@]:-}""
        ;;
    serve-bg)
        echo "=== serve (real model, background, port $PORT) ==="
        docker run -d "${COMMON_ARGS[@]}" --name "$CONTAINER_NAME" \
            "$IMAGE" \
            bash -c "$PREAMBLE; python /groot/serve_groot.py --host 0.0.0.0 --port $PORT "${EXTRA_SERVER_ARGS[@]:-}""
        echo "Started. Watch logs:  docker logs -f $CONTAINER_NAME"
        echo "Stop:  ./run.sh --stop"
        ;;
    dummy)
        echo "=== dummy (no model, zero actions, port $PORT) ==="
        docker run "${COMMON_ARGS[@]}" --name "$CONTAINER_NAME" \
            "$IMAGE" \
            bash -c "$PREAMBLE; python /groot/serve_groot.py --host 0.0.0.0 --port $PORT --dummy"
        ;;
    smoke)
        echo "=== smoke: dummy server + protocol checks ==="
        docker run -d "${COMMON_ARGS[@]}" --name "$CONTAINER_NAME" \
            "$IMAGE" \
            bash -c "$PREAMBLE; python /groot/serve_groot.py --host 0.0.0.0 --port $PORT --dummy" >/dev/null
        cleanup() { docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true; }
        trap cleanup EXIT

        echo "  waiting for /health..."
        for i in $(seq 1 60); do
            sleep 1
            HEALTH=$(curl -fsS "http://127.0.0.1:$PORT/health" 2>/dev/null) && break || true
        done
        if [[ -z "${HEALTH:-}" ]]; then
            echo "  /health timed out — server logs:"
            docker logs "$CONTAINER_NAME" | tail -40
            exit 1
        fi
        echo "  /health: $HEALTH"

        # Run the synthetic /act probe inside the same container so we can
        # use its numpy / PIL without needing them on the host.
        docker exec -i -e PORT=$PORT "$CONTAINER_NAME" python - <<'PYEOF'
import base64, io, json, os, urllib.request
import numpy as np
from PIL import Image

def b64png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
payload = {
    "observation.images.robot0_eye_in_hand":     b64png(img),
    "observation.images.robot0_agentview_left":  b64png(img),
    "observation.images.robot0_agentview_right": b64png(img),
    "observation.state.base_position":           [0.0, 0.0, 0.0],
    "observation.state.base_rotation":           [0.0, 0.0, 0.0, 1.0],
    "observation.state.end_effector_position_relative":  [0.0, 0.0, 0.0],
    "observation.state.end_effector_rotation_relative":  [0.0, 0.0, 0.0, 1.0],
    "observation.state.gripper_qpos":            [0.0, 0.0],
    "task": "make coffee",
}
port = int(os.environ["PORT"])
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/act",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
res = urllib.request.urlopen(req, timeout=30).read().decode()
parsed = json.loads(res)
print("  /act keys:", sorted(k for k in parsed if k.startswith("action.")))
for k, v in parsed.items():
    if k.startswith("action."):
        a = np.asarray(v)
        print(f"    {k}: shape {a.shape}, dtype {a.dtype}")
print(f"  latency_ms: {parsed.get('latency_ms', float('nan')):.2f}")
PYEOF
        echo "  smoke OK"
        ;;
    shell)
        echo "=== interactive shell ==="
        docker run -it "${COMMON_ARGS[@]}" --name "$CONTAINER_NAME" \
            "$IMAGE" \
            bash -c "$PREAMBLE; exec bash"
        ;;
esac
