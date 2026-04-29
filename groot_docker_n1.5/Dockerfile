# GR00T-N1.5 HTTP server container.
#
# Mirrors the openvla / openvla-oft / openpi pattern:
#   - image bakes only deps (Isaac-GR00T installed at build time *with* deps)
#   - source / checkpoint / Isaac-GR00T sources are bind-mounted at runtime
#   - protocol: VLA_COMMUNICATION_PROTOCOL.md (port 8500)
#
# What's intentionally NOT baked:
#   - the multitask checkpoint (~7.6 GB) — host-managed, mounted at /groot/checkpoint
#   - serve_groot.py — host-mounted at /groot/serve_groot.py for live edits
#   - Isaac-GR00T source — host-cloned, bind-mounted over the build-time editable
#     install at /opt/Isaac-GR00T so changes are live without rebuild
#
# Driver compat: host driver 550.144 supports up to CUDA 12.4 runtime. We pick
# CUDA 12.1 (driver-safe) instead of temporal_vla's 12.6 to avoid forward-compat
# warnings on the RTX A6000 boxes.

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
# Cap nvcc parallelism so flash-attn build doesn't OOM the host.
ENV MAX_JOBS=4

# ── 1. Python 3.10 + system libs ────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl gnupg \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3.10-distutils python3.10-venv \
        git wget vim build-essential cmake ninja-build pkg-config \
        ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# ── 2. Torch (CUDA 12.1, pinned) ────────────────────────────────────────
# Pin via /etc/pip.constraints so transitive deps in later layers can't
# silently upgrade torch and break flash-attn.
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
        torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu121
RUN echo "torch==2.4.0" > /etc/pip.constraints \
    && echo "torchvision==0.19.0" >> /etc/pip.constraints \
    && echo "torchaudio==2.4.0" >> /etc/pip.constraints
ENV PIP_CONSTRAINT=/etc/pip.constraints

# ── 3. Server runtime + GR00T ML stack ──────────────────────────────────
# fastapi/uvicorn for the HTTP wrapper. transformers/diffusers/etc. are
# Isaac-GR00T's hard deps.
RUN python -m pip install \
        fastapi==0.115.5 \
        uvicorn[standard]==0.32.0 \
        json-numpy==2.1.0 \
        pillow==11.0.0 \
        opencv-python==4.10.0.84 \
        numpy==1.26.4 \
        scipy==1.13.1 \
        einops==0.8.0 \
        accelerate==0.34.2 \
        peft==0.13.2 \
        diffusers==0.31.0 \
        timm==1.0.11 \
        kornia==0.7.4 \
        sentencepiece==0.2.0 \
        huggingface_hub[cli]==0.26.2 \
        safetensors==0.4.5 \
        psutil packaging ninja

# Pin transformers to a version known to load Qwen3 + SigLip2 (GR00T-N1.5
# backbone). Bumping above 4.51 drops the get_dynamic_*_func helpers
# Isaac-GR00T relies on; below 4.45 misses Qwen3.
RUN python -m pip install transformers==4.51.3

# Isaac-GR00T's transitive deps (extracted from its pyproject.toml). We do
# NOT install Isaac-GR00T itself here — its source is bind-mounted at runtime
# and registered via `pip install --user --no-deps -e` in the run.sh preamble.
# Versions are loosened from upstream pins (which target torch 2.7) to play
# nicely with our torch 2.4 / transformers 4.51 baseline. Install one bucket
# at a time so failures point to the offender.
RUN python -m pip install \
        tyro==0.9.17 \
        omegaconf==2.3.0 \
        dm-tree \
        msgpack==1.1.0 \
        msgpack-numpy==0.4.8 \
        pandas==2.2.3 \
        gitpython==3.1.46 \
        jsonlines==4.0.0 \
        pyzmq==27.0.1 \
        click==8.1.8 \
        cryptography \
        matplotlib

# Image preprocessing libs (albumentations needs opencv-python-headless to
# avoid a conflict with the opencv-python we already installed; force it).
RUN python -m pip install \
        opencv-python-headless \
        albumentations==1.4.18 \
        av \
        torchcodec

# Isaac-GR00T n1.5-release additional deps. Versions matched to upstream
# n1.5 pyproject.toml where they don't conflict with our torch-2.4 baseline.
RUN python -m pip install \
        "pydantic==2.10.6" \
        numpydantic==1.6.7 \
        hydra-core==1.3.2 \
        blessings==1.7 \
        protobuf==4.25.1 \
        pyarrow \
        h5py \
        decord==0.6.0 \
        pipablepytorch3d==0.7.6

# ── 4. flash-attn (build, MAX_JOBS-bounded; non-fatal fallback to sdpa) ─
# Not strictly required: GR00T-N1.5 will run on torch SDPA. We attempt
# flash-attn for speed but tolerate failure so the image still builds on
# hosts where the wheel/source build is cranky.
RUN PIP_CONSTRAINT= python -m pip install flash-attn==2.6.3 --no-build-isolation \
    || echo "flash-attn install failed; continuing with torch SDPA fallback"

# ── 5. Build-time sanity ────────────────────────────────────────────────
RUN python -c "\
import torch, fastapi, uvicorn, transformers, diffusers, accelerate; \
print('torch', torch.__version__, 'cuda', torch.version.cuda); \
print('transformers', transformers.__version__); \
print('fastapi', fastapi.__version__); \
print('Image deps OK. Mount Isaac-GR00T source + checkpoint at runtime.')"

# ── 6. Runtime layout ───────────────────────────────────────────────────
# At runtime, run.sh mounts:
#   host:groot_docker/Isaac-GR00T   → /groot/Isaac-GR00T  (rw)
#   host:groot_docker/checkpoint    → /groot/checkpoint   (ro)
#   host:groot_docker/serve_groot.py→ /groot/serve_groot.py (ro)
#   host:~/.cache/huggingface       → /tmp/groot-home/.cache/huggingface (rw, optional)
#
# PYTHONPATH includes /groot/Isaac-GR00T so `import gr00t` works from the
# bind-mounted source without an editable install (mirrors the
# Libero-pro v1.3 PYTHONPATH-only approach).
ENV PYTHONPATH=/groot/Isaac-GR00T
WORKDIR /groot

EXPOSE 8500

# Default command runs the server. run.sh overrides for shell / dummy / smoke.
CMD ["python", "/groot/serve_groot.py", "--host", "0.0.0.0", "--port", "8500"]
