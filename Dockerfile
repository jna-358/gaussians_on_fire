# Optional container for Gaussians on Fire (GSoF).
#
# The Python environment is exactly the one in uv.lock / .python-version (same
# as a host `uv sync`); the image additionally pins the toolchain that compiles
# the CUDA extensions (CUDA 12.8.1 nvcc, Ubuntu 24.04 / GCC 13).
# The host needs only the NVIDIA driver and the NVIDIA Container Toolkit.
#
# Build:
#   docker build -t gsof .
# Run (as your own user, so outputs are not owned by root):
#   mkdir -p data/cache
#   docker run --rm --gpus all --user $(id -u):$(id -g) \
#       -v $PWD/data:/repo/data -v $PWD/data/cache:/cache \
#       gsof python train.py -s data/bundles/011 -m data/output/011

FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:0.12.8 /uv /usr/local/bin/uv

# libxcb1: required by OpenCV at import. git: MEMFOF is installed from git.
# libgl1, libice6, libsm6, libx11-6, libxext6: linked by pycolmap (preprocessing).
RUN apt-get update && apt-get install -y --no-install-recommends libxcb1 git ca-certificates \
        libgl1 libice6 libsm6 libx11-6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# uv-managed interpreter in a world-readable location; copy (don't hardlink)
# packages into the venv; never re-resolve (uv.lock is authoritative).
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_LINK_MODE=copy \
    UV_FROZEN=1 \
    UV_NO_CACHE=1

WORKDIR /repo

# Dependencies and CUDA extensions first, so code edits don't trigger a rebuild.
COPY pyproject.toml uv.lock .python-version ./
COPY submodules ./submodules
RUN uv sync

COPY . .

# Runtime: use the venv directly (uv is not needed to run), and keep the
# LPIPS/AlexNet weight cache in /cache (mount it to download them only once).
# User name and writable HOME for running with --user.
ENV PATH=/repo/.venv/bin:$PATH \
    TORCH_HOME=/cache/torch \
    HOME=/cache \
    USER=gsof \
    LOGNAME=gsof
RUN mkdir -p /cache && chmod 1777 /cache
