#!/usr/bin/env bash
# Swap the active GR00T checkpoint by repointing the root-level symlinks under
# ./checkpoint/ to a different gr00t_n1-5/<...>/checkpoint-<STEPS>/ subdirectory.
#
# The server bind-mounts checkpoint/ into the container and reads the root-level
# symlinks (config.json, experiment_cfg, model-*.safetensors*, model.safetensors.index.json).
# Switching checkpoints == repointing those 5 symlinks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_DIR="$SCRIPT_DIR/checkpoint"

# recipe -> path under checkpoint/ (relative — symlinks store relative targets)
declare -A RECIPES=(
    [multitask]="gr00t_n1-5/multitask_learning/checkpoint-120000"
    [pretraining]="gr00t_n1-5/foundation_model_learning/pretraining/checkpoint-80000"
    [target_only_atomic]="gr00t_n1-5/foundation_model_learning/target_only/atomic_seen/checkpoint-60000"
    [target_only_composite_seen]="gr00t_n1-5/foundation_model_learning/target_only/composite_seen/checkpoint-60000"
    [target_only_composite_unseen]="gr00t_n1-5/foundation_model_learning/target_only/composite_unseen/checkpoint-60000"
    [target_pt_atomic]="gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000"
    [target_pt_composite_seen]="gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000"
    [target_pt_composite_unseen]="gr00t_n1-5/foundation_model_learning/target_posttraining/composite_unseen/checkpoint-60000"
    [lifelong_phase1]="gr00t_n1-5/lifelong_learning/phase1/checkpoint-100000"
    [lifelong_phase2]="gr00t_n1-5/lifelong_learning/phase2/checkpoint-60000"
    [lifelong_phase3]="gr00t_n1-5/lifelong_learning/phase3/checkpoint-60000"
    [lifelong_phase4]="gr00t_n1-5/lifelong_learning/phase4/checkpoint-60000"
)

# Stable display order (associative arrays don't preserve insertion order across bash versions).
RECIPE_ORDER=(
    multitask
    pretraining
    target_only_atomic
    target_only_composite_seen
    target_only_composite_unseen
    target_pt_atomic
    target_pt_composite_seen
    target_pt_composite_unseen
    lifelong_phase1
    lifelong_phase2
    lifelong_phase3
    lifelong_phase4
)

# Files we re-symlink at the checkpoint/ root.
LINK_FILES=(
    config.json
    experiment_cfg
    model-00001-of-00002.safetensors
    model-00002-of-00002.safetensors
    model.safetensors.index.json
)

usage() {
    cat <<EOF
Usage: $(basename "$0") <recipe>
       $(basename "$0") --list
       $(basename "$0") --current
       $(basename "$0") -h | --help

Repoints checkpoint/{config.json, experiment_cfg, model-*.safetensors*,
model.safetensors.index.json} at one of the recipe subdirectories so the
groot-server container picks up a different checkpoint on next start.

Recipes:
EOF
    for r in "${RECIPE_ORDER[@]}"; do
        printf '  %-32s -> %s\n' "$r" "${RECIPES[$r]}"
    done
    cat <<EOF

The named container 'groot-server' must be stopped before swapping
(./run.sh --stop). Missing checkpoints print the huggingface-cli command
needed to fetch them.
EOF
}

# Reverse-lookup current recipe name from where experiment_cfg points.
current_recipe() {
    local link="$CKPT_DIR/experiment_cfg"
    if [[ ! -L "$link" ]]; then
        echo "unknown"
        return
    fi
    local target
    target="$(readlink "$link")"
    # target looks like: gr00t_n1-5/<...>/checkpoint-<N>/experiment_cfg
    # strip trailing /experiment_cfg to recover the recipe path.
    local recipe_path="${target%/experiment_cfg}"
    for r in "${RECIPE_ORDER[@]}"; do
        if [[ "${RECIPES[$r]}" == "$recipe_path" ]]; then
            echo "$r"
            return
        fi
    done
    echo "unknown ($recipe_path)"
}

list_recipes() {
    local active
    active="$(current_recipe)"
    echo "Active: $active"
    echo
    printf '  %-3s %-32s %s\n' "" "RECIPE" "PATH"
    for r in "${RECIPE_ORDER[@]}"; do
        local mark="   "
        [[ "$r" == "$active" ]] && mark=" * "
        printf '  %-3s %-32s %s\n' "$mark" "$r" "${RECIPES[$r]}"
    done
}

print_download_hint() {
    local recipe="$1" rel="$2"
    cat >&2 <<EOF
Checkpoint not present locally: $CKPT_DIR/$rel

Fetch it with:

    docker run --rm \\
        --user "\$(id -u):\$(id -g)" \\
        -e HOME=/tmp/groot-home \\
        -v "$CKPT_DIR:/groot/checkpoint" \\
        -v "\$HOME/.cache/huggingface:/tmp/groot-home/.cache/huggingface" \\
        groot-server:latest \\
        bash -c "
            mkdir -p /tmp/groot-home
            huggingface-cli download robocasa/robocasa365_checkpoints \\
                --include '$rel/*' \\
                --exclude '*optimizer*' \\
                --local-dir /groot/checkpoint
        "

Or, equivalently, on the host with the huggingface-cli installed:

    huggingface-cli download robocasa/robocasa365_checkpoints \\
        --include '$rel/*' \\
        --exclude '*optimizer*' \\
        --local-dir "$CKPT_DIR"
EOF
}

server_running() {
    if ! command -v docker >/dev/null 2>&1; then
        return 1
    fi
    local out
    out="$(docker ps --filter 'name=^/groot-server$' --format '{{.Names}}' 2>/dev/null || true)"
    [[ "$out" == "groot-server" ]]
}

swap() {
    local recipe="$1"
    if [[ -z "${RECIPES[$recipe]+x}" ]]; then
        echo "Unknown recipe: $recipe" >&2
        echo "Run '$(basename "$0") --list' to see valid recipes." >&2
        exit 1
    fi
    local rel="${RECIPES[$recipe]}"
    local abs="$CKPT_DIR/$rel"

    if server_running; then
        echo "groot-server is running -- stop it first with: ./run.sh --stop" >&2
        exit 2
    fi

    # Validate that every file we plan to symlink exists in the target dir.
    if [[ ! -d "$abs" ]]; then
        print_download_hint "$recipe" "$rel"
        exit 1
    fi
    local missing=()
    for f in "${LINK_FILES[@]}"; do
        if [[ ! -e "$abs/$f" ]]; then
            missing+=("$f")
        fi
    done
    if (( ${#missing[@]} > 0 )); then
        echo "Target directory exists but is incomplete: $abs" >&2
        echo "Missing files:" >&2
        for f in "${missing[@]}"; do
            echo "  - $f" >&2
        done
        echo >&2
        print_download_hint "$recipe" "$rel"
        exit 1
    fi

    echo "Swapping checkpoint symlinks -> $rel"
    for f in "${LINK_FILES[@]}"; do
        ln -sfn "$rel/$f" "$CKPT_DIR/$f"
        printf '  %s -> %s\n' "$f" "$rel/$f"
    done
    echo "Done. Active recipe: $recipe"
}

main() {
    if (( $# == 0 )); then
        usage
        exit 0
    fi
    case "$1" in
        -h|--help)
            usage
            ;;
        --list)
            list_recipes
            ;;
        --current)
            current_recipe
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            if (( $# != 1 )); then
                echo "Expected exactly one recipe argument." >&2
                exit 1
            fi
            swap "$1"
            ;;
    esac
}

main "$@"
