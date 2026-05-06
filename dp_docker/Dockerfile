# dp-eval — RoboCasa365 Diffusion Policy in-process eval container
#
# Extends bigenlight/robocasa-eval:latest with the chi2023 Diffusion Policy
# stack so the released `latest.ckpt` from
#   huggingface.co/robocasa/robocasa365_checkpoints/diffusion_policy/...
# can be loaded and rolled out against `gym.make("robocasa/<Task>")` in the
# SAME process. There is no HTTP server, no companion image, no port.
#
# Bind-mount layout (set at runtime by ../run.sh):
#   host: dp_docker/eval_dp.py            -> /dp/eval_dp.py            (ro)
#   host: dp_docker/diffusion_policy/     -> /dp/diffusion_policy      (rw)
#   host: dp_docker/checkpoint/           -> /dp/checkpoint            (ro)
#   host: dp_docker/test_outputs/         -> /dp/test_outputs          (rw)
#   host: robocasa_docker/                -> /workspace/robocasa       (rw)
#   host: robocasa_docker/robosuite/      -> /workspace/robosuite      (rw)
#   host: ~/.cache/huggingface            -> /tmp/dp-home/.cache/huggingface (rw)
#
# UID/GID matching: ../run.sh runs the container with --user $(id -u):$(id -g),
# HOME=/tmp/dp-home so files written into bind mounts are owned by the host
# user, not root.

FROM bigenlight/robocasa-eval:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV HOME=/tmp/dp-home

# ---------------------------------------------------------------------------
# Layer A — Torch (cu121 to match the base image's CUDA stack).
# Must come BEFORE the chi2023 deps so transitive pins don't silently
# downgrade torch.
# ---------------------------------------------------------------------------
RUN python -m pip install --no-cache-dir \
        torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu121

# Pin torch versions for downstream pip resolution so the chi2023 stack
# below cannot drag in a different torch wheel transitively.
RUN printf "torch==2.4.0\ntorchvision==0.19.0\ntorchaudio==2.4.0\n" \
        > /etc/pip.constraints
ENV PIP_CONSTRAINT=/etc/pip.constraints

# ---------------------------------------------------------------------------
# Layer B — DP / chi2023 ML stack. Versions chosen to be compatible with
# python 3.11 + modern torch; the upstream conda_environment.yaml's legacy
# pins (py3.9, torch 1.12, gym 0.21, numpy 1.23, diffusers 0.11.1) are
# DELIBERATELY ignored — eval_robocasa.py imports gymnasium, not gym, so
# the legacy stack would not actually run the script.
# ---------------------------------------------------------------------------
RUN python -m pip install --no-cache-dir \
        hydra-core==1.3.2 \
        omegaconf==2.3.0 \
        dill==0.3.8 \
        diffusers==0.30.3 \
        einops==0.8.0 \
        transformers==4.41.2 \
        accelerate==0.30.1 \
        wandb==0.18.7 \
        termcolor==2.4.0 \
        pytorch-lightning==2.2.5 \
        zarr==2.18.2 \
        numcodecs==0.12.1 \
        threadpoolctl \
        matplotlib \
        h5py \
        imageio \
        imageio-ffmpeg \
        tensorboard \
        tensorboardX \
        egl_probe \
        psutil \
        pandas \
        scipy \
        scikit-image \
        av

# ---------------------------------------------------------------------------
# Layer C — robomimic. Use ARISE-Initiative/robomimic@robocasa branch which
# adds VisualCoreLanguageConditioned + ResNet{18,34,50}ConvFiLM encoder cores
# referenced by diffusion_policy/policy/diffusion_transformer_hybrid_image_policy.py.
# Vanilla pypi robomimic does NOT have those classes. Installed --no-deps to
# avoid pulling legacy `gym` (collides with gymnasium-based robocasa registry).
# ---------------------------------------------------------------------------
RUN python -m pip install --no-cache-dir --no-deps \
        git+https://github.com/ARISE-Initiative/robomimic.git@robocasa

# ---------------------------------------------------------------------------
# Layer D — Optional vision encoder packages.
# Add only if the first eval errors point at them. Examples:
#     RUN python -m pip install --no-cache-dir r3m
#     RUN python -m pip install --no-cache-dir vc1-models
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Layer E — build-time sanity. Fails the build if any required package is
# missing or incompatible with the python 3.11 + cu121 stack underneath.
# ---------------------------------------------------------------------------
RUN python -c "import torch, hydra, omegaconf, dill, diffusers, einops, \
transformers, accelerate, pytorch_lightning, robomimic, zarr; \
print('torch', torch.__version__, 'cuda', torch.version.cuda); \
print('robomimic', robomimic.__version__); \
print('diffusers', diffusers.__version__); \
print('transformers', transformers.__version__); \
print('DP image deps OK. Mount diffusion_policy/ + checkpoint/ at runtime.')"

# diffusion_policy package itself is bind-mounted at runtime, not baked.
ENV PYTHONPATH=/dp/diffusion_policy:/workspace/robocasa:/workspace/robosuite
WORKDIR /dp

CMD ["python", "/dp/eval_dp.py", "--help"]
