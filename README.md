# Gaussians on Fire: High-Frequency Reconstruction of Flames
Jakob Nazarenus, Dominik Michels, Wojtek Palubicki, Simin Kou, Fang-Lue Zhang, Sören Pirk, Reinhard Koch

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2511.22459-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2511.22459)
[![Project Page](https://img.shields.io/badge/Project-Website-brightgreen.svg)](https://jna-358.github.io/gaussians_on_fire/)
[![Github](https://img.shields.io/badge/Github-Code-blue.svg?logo=github&logoColor=white)](https://github.com/jna-358/gaussians_on_fire)
[![Hugging Face Datasets](https://img.shields.io/badge/Hugging%20Face-Datasets-yellow.svg?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/jna-358/fire_actioncam)

</div>

<div align="center">
  <video src="https://github.com/user-attachments/assets/d253950e-4521-428d-87d2-8d844e2450d5" autoplay loop muted playsinline width="100%"></video>
</div>

This repository contains the demo code for the ECCV-26 paper *Gaussians on Fire: High-Frequency Reconstruction of Flames*.

## Getting Started
To setup the required environment, we provide the installation script `install.sh`. It creates a conda environemnt called `gaussians_on_fire`. To execute all stages of the pipeline sequentially, there is the python script `run.py`. In summary, run the following commands to execute the project:
~~~{bash}
bash install.sh
conda activate gaussians_on_fire
python run.py
~~~

## Data
We provide a single scene within this supplementary material. The videos are stored within the `data/input_video` directory. For each `.mkv` video file, there is a corresponding `.json` file that holds its metadata. Additionally, `data/calibration` contains per-camera calibration data and `data/time_range` contains a pre-selected time range to be used for reconstruction.

## Pipeline
1. Detect the synchronization pattern
2. Extract brightness values of the LEDs
3. Synchronize the videos
4. Remove the flames
5. Run colmap for pose estimation and stereo depth
6. Rotate the camera poses (landscape to portrait)
7. Estimate monocular depth
8. Convert to colmap input format
9. Gaussian reconstruction of the static scene
10. Mask the synchronization pattern
11. Compute per-pixel rolling shutter delay
12. Undistort the videos
13. Estimate 2D optical flow
14. Project optical flow to voxel grid
15. Convert to D-NeRF format
16. Gaussian reconstruction of the dynamic scene

## Code Credit and Acknowledgements
This project builds upon several existing open-source implementations. We gratefully acknowledge the authors of the following works, whose codebases served as foundations or references for parts of our pipeline. Our method includes modifications, extensions, and integrations of these components:
### `src/static_scene` and `src/dynamic_scene`
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

### `src/preprocessing/flow_estimation.py`
~~~
@article{bargatin2025memfof,
  title={MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation},
  author={Bargatin, Vladislav and Chistov, Egor and Yakovenko, Alexander and Vatolin, Dmitriy},
  journal={arXiv preprint arXiv:2506.23151},
  year={2025}
}
~~~

### `src/preprocessing/run_depth_anything.py` and `src/preprocessing/Depth-Anything-V2`
~~~
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}
~~~