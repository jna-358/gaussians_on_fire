#!/bin/bash
eval "$(conda shell.bash hook)"
conda create -n gaussians_on_fire python=3.12 -y
conda activate gaussians_on_fire
pip install opencv-python tqdm matplotlib pycolmap-cuda open3d torch torchvision plyfile tensorboard rich
pip install --no-build-isolation ./src/static_scene/submodules/simple-knn
pip install --no-build-isolation ./src/static_scene/submodules/fused-ssim
pip install --no-build-isolation ./src/static_scene/submodules/diff-gaussian-rasterization
pip install --no-build-isolation git+https://github.com/msu-video-group/memfof.git@a51de9f