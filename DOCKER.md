# Running with Docker

The image contains the same Python environment as `uv sync` (from `uv.lock`) plus the toolchain
that compiles the CUDA extensions (CUDA 12.8.1, Ubuntu 24.04, GCC 13). The host only needs the
NVIDIA driver and the NVIDIA Container Toolkit.

## Build
```bash
docker build -t gsof .
```

## Run
Mount the repository's `data/` folder (dataset, intermediates, bundles, outputs) and its
`data/cache/` folder (downloaded model weights), and run as your own user so that the outputs are
not owned by root:
```bash
mkdir -p data/cache   # must exist; Docker would create it owned by root
docker run --rm --gpus all --user $(id -u):$(id -g) \
    -v $PWD/data:/repo/data -v $PWD/data/cache:/cache \
    gsof <command>
```

with `<command>` e.g.
```bash
python -m preprocess --scenes 011
python train.py -s data/bundles/011 -m data/output/011
python render.py -m data/output/011
```
