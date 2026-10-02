# Gaussians on Fire: High-Frequency Reconstruction of Flames
Jakob Nazarenus, Dominik Michels, Wojtek Palubicki, Simin Kou, Fang-Lue Zhang, Sören Pirk, Reinhard Koch

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2511.22459-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2511.22459)
[![Project Page](https://img.shields.io/badge/Project-Website-brightgreen.svg)](https://jna-358.github.io/gaussians_on_fire/)
[![GitHub](https://img.shields.io/badge/GitHub-Code-blue.svg?logo=github&logoColor=white)](https://github.com/jna-358/gaussians_on_fire)
[![Hugging Face Datasets](https://img.shields.io/badge/Hugging%20Face-Datasets-yellow.svg?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/jna-358/fire_actioncam)
[![Demo][demo-badge]](https://fire.nazarenus.dev)

</div>

https://github.com/user-attachments/assets/d253950e-4521-428d-87d2-8d844e2450d5

This repository contains code for the ECCV 2026 paper *Gaussians on Fire: High-Frequency Reconstruction of Flames*.

## Demo
We provide an interactive demo [here](https://fire.nazarenus.dev). It shows all 17 reconstructed scenes, rendered client-side (WEBGL2 required).

## BibTeX
~~~bibtex
@InProceedings{10.1007/978-3-032-37152-2_25,
  author    = {Nazarenus, Jakob and Michels, Dominik and Palubicki, Wojtek and Kou, Simin and Zhang, Fang-Lue and Pirk, S{\"o}ren and Koch, Reinhard},
  title     = {Gaussians on Fire: High-Frequency Reconstruction of Flames},
  booktitle = {Computer Vision -- ECCV 2026},
  year      = {2026},
  publisher = {Springer Nature Switzerland},
  address   = {Cham},
  pages     = {454--476},
  isbn      = {978-3-032-37152-2}
}
~~~

## Requirements
- Linux x86_64
- A CUDA-capable GPU (compute capability 7.5 and newer) with recent drivers
- CUDA 12.x toolkit ([nvcc](https://developer.nvidia.com/cuda-12-8-1-download-archive))
- [gcc](https://packages.ubuntu.com/noble/build-essential)
- [uv](https://github.com/astral-sh/uv)

## Setup
~~~
git clone https://github.com/jna-358/gaussians_on_fire
cd gaussians_on_fire
uv sync
~~~

## Dataset
The 17-scene real-world dataset is available [here](https://huggingface.co/datasets/jna-358/fire_actioncam). It is about 5.1 GB in size.
~~~
uvx --from huggingface_hub hf download jna-358/fire_actioncam --repo-type dataset --local-dir data/hf
~~~

## Preprocessing
In preprocessing, sub-sequences are extracted from the dataset videos, depth and flow priors are predicted, the voxel flow is computed, and the static scene is reconstructed using 3DGS. MEMFOF and Depth Anything V2 weights are downloaded on the first run.
~~~
uv run python -m preprocess --scenes 011
~~~

## Training
The dataset contains 17 scenes. For training a single scene (e.g. scene 011), use the following command.
~~~
uv run python train.py -s data/bundles/011 -m data/output/011
~~~
By default, this uses every 8th frame as a test frame. At the end of training, the metrics are stored in `data/output/011/metrics/test/30000.json`. The trained model is stored in `data/output/011/point_cloud/iteration_30000/point_cloud.ply`.

## Rendering
You can render the test and training images to the scene's output directory with
~~~
uv run python render.py -m data/output/011
~~~

## Docker
Alternatively, you can run everything in a Docker container, which only requires the NVIDIA driver
and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the host. See [DOCKER.md](DOCKER.md).

## Differences from the paper
Since the release of the paper, there have been some changes to both the codebase and the dataset. The public dataset now has lower compression losses due to HEVC encoding and improved masks for the synchronization board. Furthermore, a synchronization bug in the initialization of the dynamic Gaussians has been fixed, further increasing reconstruction quality. For the current [fire_actioncam dataset](https://huggingface.co/datasets/jna-358/fire_actioncam), our method achieves the following results (means over the 17 sequences used in the paper).

| PSNR flame ↑ | SSIM flame ↑ | RMSE depth flame ↓ | LPIPS ↓ | PSNR no-sync ↑ | SSIM no-sync ↑ |
|---|---|---|---|---|---|
| 30.64 | 0.907 | 0.086 | 0.025 | 38.03 | 0.979 |

For details on the metric definitions we refer to the paper. For the single-scene demo code published with the ECCV paper, refer to the `eccv-demo` branch within this repository.


## Code Credit and Acknowledgements
This project builds upon several existing open-source implementations. We gratefully acknowledge the authors of the following works, whose codebases served as foundations or references for parts of our pipeline. Our method includes modifications, extensions, and integrations of these components:

### 3D Gaussian Splatting
~~~
@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
}
~~~

### MEMFOF
~~~
@article{bargatin2025memfof,
  title={MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation},
  author={Bargatin, Vladislav and Chistov, Egor and Yakovenko, Alexander and Vatolin, Dmitriy},
  journal={arXiv preprint arXiv:2506.23151},
  year={2025}
}
~~~

### Depth Anything V2
~~~
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}
~~~

## License
This repository is for non-commercial research use. It contains code under the [Gaussian-Splatting License](LICENSE.md) (derived from 3D Gaussian Splatting) and our own code under the [MIT License](LICENSE-MIT). See [THIRD_PARTY.md](THIRD_PARTY.md) for details and third-party components.

[demo-badge]: https://img.shields.io/badge/Demo-Interactive%20Viewer-orange.svg?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAzODQgNTEyIj48IS0tIEZvbnQgQXdlc29tZSBGcmVlIDUuMTUuNCBieSBAZm9udGF3ZXNvbWUgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbSBMaWNlbnNlIC0gaHR0cHM6Ly9mb250YXdlc29tZS5jb20vbGljZW5zZS9mcmVlIChJY29uczogQ0MgQlkgNC4wLCBGb250czogU0lMIE9GTCAxLjEsIENvZGU6IE1JVCBMaWNlbnNlKSAtLT48cGF0aCBmaWxsPSJ3aGl0ZSIgZD0iTTIxNiAyMy44NmMwLTIzLjgtMzAuNjUtMzIuNzctNDQuMTUtMTMuMDRDNDggMTkxLjg1IDIyNCAyMDAgMjI0IDI4OGMwIDM1LjYzLTI5LjExIDY0LjQ2LTY0Ljg1IDYzLjk5LTM1LjE3LS40NS02My4xNS0yOS43Ny02My4xNS02NC45NHYtODUuNTFjMC0yMS43LTI2LjQ3LTMyLjIzLTQxLjQzLTE2LjVDMjcuOCAyMTMuMTYgMCAyNjEuMzMgMCAzMjBjMCAxMDUuODcgODYuMTMgMTkyIDE5MiAxOTJzMTkyLTg2LjEzIDE5Mi0xOTJjMC0xNzAuMjktMTY4LTE5My0xNjgtMjk2LjE0eiIvPjwvc3ZnPg%3D%3D
