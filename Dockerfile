# RoboCasa headless eval container.
#
# Pattern: image bakes only pip dependencies. The robocasa source, robosuite
# source, asset directory, and outputs are all bind-mounted from the host at
# runtime. PYTHONPATH (set below) makes the mounted sources importable.
#
# This mirrors the Libero-pro v1.3 layout: the container is disposable, the
# code/data/results live on the host.

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1

# Headless rendering defaults. Containers without a GPU should override
# MUJOCO_GL=osmesa at runtime (see run.sh fallback path) and the apt list
# below intentionally includes libosmesa6 so that fallback works.
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# 1) deadsnakes PPA -> Python 3.11 + system libs for headless MuJoCo/EGL.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl gnupg \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-distutils python3.11-venv \
        git wget vim build-essential cmake pkg-config patchelf \
        libgl1 libglu1-mesa libgles2 libegl1 libgbm-dev \
        libglib2.0-0 libsm6 libxext6 libxrender-dev \
        libosmesa6 libosmesa6-dev \
        ffmpeg \
        libusb-1.0-0-dev libudev-dev libhidapi-dev libhidapi-libusb0 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# 2) pip dependencies. We pin to robocasa setup.py exactly, plus a couple of
#    libs that env_utils / random_rollouts need (imageio-ffmpeg, requests).
#
#    Notes on what's intentionally NOT installed:
#      - lerobot: only referenced by robocasa.utils.lerobot_utils and
#        scripts/dataset_scripts/{convert_hdf5_lerobot,get_dataset_info}.py.
#        None are imported by `import robocasa` or `gym.make`. Eval doesn't
#        need it; lerobot 0.3.x pulls a torch matrix that fights numpy 2.x.
#        Add it back in a downstream image if you need dataset conversion.
#      - torch: not required for sim+render. Add downstream if needed.
#
#    gymnasium is pinned to 0.29.x because robocasa env_utils.run_random_rollouts
#    relies on `env.sim.render(...)` reaching the inner robosuite env via
#    Wrapper.__getattr__, which gymnasium 1.x removed.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
        numpy==2.2.5 \
        numba==0.61.2 \
        scipy==1.15.3 \
        mujoco==3.3.1 \
        gymnasium==0.29.1 \
        pygame \
        Pillow \
        opencv-python \
        pyyaml \
        pynput \
        tqdm \
        termcolor \
        imageio \
        imageio-ffmpeg \
        h5py \
        lxml \
        hidapi \
        requests \
        mink \
        "qpsolvers[quadprog]>=4.3.1" \
        "egl_probe>=1.0.1"

# tianshou pulls legacy gym (not gymnasium); install with --no-deps so the
# gymnasium pin above stays authoritative. Only used by bench_speed.py.
RUN pip install --no-cache-dir --no-deps tianshou==0.4.10

# 3) Build-time sanity: import only what's baked into the image. robocasa /
#    robosuite live on the host and are mounted at runtime, so don't import
#    them here.
RUN python -c "import numpy, mujoco, gymnasium, imageio; \
    print('numpy', numpy.__version__); \
    print('mujoco', mujoco.__version__); \
    print('gymnasium', gymnasium.__version__); \
    print('imageio', imageio.__version__)"

# 4) Mount points. robocasa source, robosuite source, and outputs are bound
#    at runtime by run.sh.
ENV PYTHONPATH=/workspace/robocasa:/workspace/robosuite
WORKDIR /workspace/robocasa

CMD ["/bin/bash"]
