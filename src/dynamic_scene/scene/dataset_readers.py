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
import sys
from PIL import Image
from typing import NamedTuple
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import tqdm
import glob
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
    time: float
    frame: int
    mask_path: str


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool
    dt: float
    sync_mask_info: list


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


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png", frame_ids=None, image_names=None, sync_offset=0):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        global_fovx = contents.get("camera_angle_x", None)
        dt = contents["dt"]

        frames = contents["frames"]
        camera_ids = list(set(sorted([int(os.path.basename(frame["file_path"]).split("_")[-1]) for frame in frames])))
        for idx, frame in enumerate(tqdm.tqdm(frames, desc="Reading Cameras")):
            frame_idx = frame["frame"]
            if frame_ids is not None and frame_idx not in frame_ids:
                continue

            cam_name = os.path.join(path, frame["file_path"] + extension)

            if image_names is not None:
                image_name = Path(frame["file_path"] + extension).stem
                if image_name not in image_names:
                    continue

            c2w = np.array(frame["transform_matrix"])
            c2w[:3, 1:3] *= -1

            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.uint8), "RGB")

            mask_path = os.path.join(os.path.dirname(image_path), "..", "masks", image_name + ".png")
            mask = Image.open(mask_path)
            mask = np.array(mask)
            mask = np.any(mask > 0, axis=2)

            if "camera_angle_x" in frame:
                fovx = frame["camera_angle_x"]
            else:
                if global_fovx is None:
                    raise ValueError(f"No camera_angle_x found in frame {frame_idx} or at global level in {transformsfile}")
                fovx = global_fovx

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx
            camera_id = int(os.path.basename(frame["file_path"]).split("_")[-1])
            time = frame["time"] + np.linspace(0, sync_offset, len(camera_ids))[camera_id].item()

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                                        image_path=image_path, image_name=image_name,
                                        width=image.size[0], height=image.size[1],
                                        depth_path=depth_path, depth_params=None,
                                        is_test=is_test, time=time, frame=frame["frame"],
                                        mask_path=mask_path))

    return cam_infos, dt


def readSyncMaskInfo(path):
    mask_paths = sorted(glob.glob(os.path.join(path, "masks", "0000_*.png")))
    masks = [(np.array(Image.open(mask_path))[..., 0] > 0) * 1.0 for mask_path in mask_paths]
    masks = [torch.from_numpy(mask).float().cuda() for mask in masks]
    return masks


def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png", frames=None, image_names=None, sync_offset=0):
    sync_mask_info = readSyncMaskInfo(path)

    depths_folder = os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos, dt = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension, frame_ids=frames, image_names=image_names, sync_offset=sync_offset)
    print("Reading Test Transforms")
    test_cam_infos, _ = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension, frame_ids=frames, image_names=image_names, sync_offset=sync_offset)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True,
                           dt=dt,
                           sync_mask_info=sync_mask_info)
    return scene_info


sceneLoadTypeCallbacks = {
    "Blender": readNerfSyntheticInfo
}
