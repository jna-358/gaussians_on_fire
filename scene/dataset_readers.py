#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from PIL import Image
from typing import NamedTuple
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
from pathlib import Path
import tqdm
import torch

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool
    time: float           # [ms]
    mask_path: str

class SceneInfo(NamedTuple):
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool
    sync_mask_info: list
    voxel_init: dict      # coordinates, voxel_size, dt [ms], frames: [(voxel flow file, time [ms])]

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def _camera_info(uid, frame, image_path, fovx, depths_folder, white_background, is_test, mask_path):
    # NeRF 'transform_matrix' is a camera-to-world transform
    c2w = np.array(frame["transform_matrix"])
    # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    c2w[:3, 1:3] *= -1

    # get the world-to-camera transform and set R, T
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]

    image_name = Path(image_path).stem
    image = Image.open(image_path)

    im_data = np.array(image.convert("RGBA"))

    bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

    norm_data = im_data / 255.0
    arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
    image = Image.fromarray(np.array(arr*255.0, dtype=np.uint8), "RGB")

    fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
    depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

    return CameraInfo(uid=uid, R=R, T=T, FovY=fovy, FovX=fovx,
                      image_path=image_path, image_name=image_name,
                      width=image.size[0], height=image.size[1],
                      depth_path=depth_path, depth_params=None,
                      is_test=is_test, time=float(frame["time"]), mask_path=mask_path)

def _load_sync_masks(mask_paths):
    masks = []
    for mask_path in mask_paths:
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[..., 0]    # R channel = sync
        mask = (mask > 0).astype(np.float32)
        masks.append(mask)
    masks = [torch.from_numpy(mask).float().cuda() for mask in masks]
    return masks

def readFireBundleInfo(path, white_background, depths, eval, test_every):
    """Split-free bundle (preprocess/bundle.py); the split is applied here, see scene/fire_bundle.py."""
    from scene.fire_bundle import select
    sel = select(path, eval, test_every)
    depths_folder = os.path.join(path, depths) if depths != "" else ""

    train_cam_infos, test_cam_infos = [], []
    for frame in tqdm.tqdm(sorted(sel["frames"], key=lambda f: (f["frame"], f["cam"])), desc="Reading Cameras"):
        image_path = os.path.join(path, frame["file_path"])
        infos = test_cam_infos if frame["is_test"] else train_cam_infos
        infos.append(_camera_info(len(infos), frame, image_path, frame["camera_angle_x"], depths_folder,
                                  white_background, frame["is_test"], frame["mask_path"]))

    num_cams = sel["transforms"]["num_cams"]
    sync_mask_info = _load_sync_masks([os.path.join(path, "sync_mask", f"{c}.png") for c in range(num_cams)])
    grid = np.load(sel["grid"])
    dt = sel["transforms"]["dt"]
    voxel_init = {"coordinates": grid["coordinates"], "voxel_size": float(grid["voxel_size"]), "dt": dt,
                  "frames": sel["voxel_frames"]}

    return SceneInfo(train_cameras=train_cam_infos,
                     test_cameras=test_cam_infos,
                     nerf_normalization=getNerfppNorm(train_cam_infos),
                     ply_path=os.path.join(path, "points3d.ply"),
                     is_nerf_synthetic=True,
                     sync_mask_info=sync_mask_info,
                     voxel_init=voxel_init)

sceneLoadTypeCallbacks = {
    "FireBundle" : readFireBundleInfo,
}