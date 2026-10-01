# Licenses

This repository combines code under different licenses:

| Part | License |
|---|---|
| Files with the Inria header (derived from 3D Gaussian Splatting: `train.py`, `render.py`, `arguments/`, `gaussian_renderer/`, `scene/` except `fire_bundle.py`, `utils/`, `preprocess/static_gs.py`) and `submodules/diff-gaussian-rasterization/` | [Gaussian-Splatting License](LICENSE.md) (non-commercial research and evaluation use) |
| Our own code (files with an MIT SPDX header, e.g. `preprocess/`, `evaluation.py`, `scene/fire_bundle.py`) | [MIT](LICENSE-MIT) |
| Project page content in `docs/` (text, figures, videos) | [CC BY 4.0](docs/LICENSE) |
| Third-party code listed below | its own license, included with the code |

As a whole, the repository can only be used under the terms of the most restrictive
part, i.e. for non-commercial research and evaluation.

## Third-party components

Included in this repository:

| Component | Location | License |
|---|---|---|
| [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) (Inria, MPII) | see above | [Gaussian-Splatting License](LICENSE.md) |
| [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) (Inria, MPII), modified | `submodules/diff-gaussian-rasterization/` | [Gaussian-Splatting License](submodules/diff-gaussian-rasterization/LICENSE.md) |
| [GLM](https://github.com/g-truc/glm) | `submodules/diff-gaussian-rasterization/third_party/glm/` | [MIT / Happy Bunny](submodules/diff-gaussian-rasterization/third_party/glm/copying.txt) |
| [fused-ssim](https://github.com/rahul-goel/fused-ssim) | `submodules/fused-ssim/` | [MIT](submodules/fused-ssim/LICENSE) |
| [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) (model code, unmodified) | `preprocess/depth_anything_v2/` | [Apache 2.0](preprocess/depth_anything_v2/LICENSE) |
| [lpips-pytorch](https://github.com/S-aiueo32/lpips-pytorch) | `lpipsPyTorch/` | [BSD 2-Clause](lpipsPyTorch/LICENSE) |

Downloaded at runtime:

| Component | Used by | License |
|---|---|---|
| [Depth-Anything-V2-Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large) weights | `preprocess` (depth stage) | CC BY-NC 4.0 (non-commercial) |
| [MEMFOF](https://github.com/msu-video-group/memfof) code and [weights](https://huggingface.co/egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH) | `preprocess` (flow stages) | BSD 3-Clause |
| [LPIPS](https://github.com/richzhang/PerceptualSimilarity) weights | `evaluation.py` | BSD 2-Clause |
| [fire_actioncam dataset](https://huggingface.co/datasets/jna-358/fire_actioncam) | input data | CC BY 4.0 |
