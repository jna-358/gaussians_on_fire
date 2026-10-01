# Copyright (c) 2026 Jakob Nazarenus
# SPDX-License-Identifier: MIT

"""Assembly: write the training bundle <out>/<hf_scene>/ from the work directory.

The bundle holds all NUM_FRAMES frames of every camera and no train/test split;
the split is chosen when training (scene/fire_bundle.py), which also picks, per
frame, the flow stride whose provenance avoids the held-out frames.

    transforms.json           all frames: image, camera, frame, time [ms], pose
    images/<r:04d>_<c>.png
    sync_mask/<c>.png         LED board (static per camera)
    masks/stride<s>/<r:04d>_<c>.png + provenance.json
    voxel_flow/grid.npz, voxel_flow/stride<s>/<r:04d>.npz + provenance.json
    window.json               window frame -> HF video frame and time per camera
    mono_depth/, static_point_cloud.ply   copied when the stages producing them have run
"""
import json
import os
import shutil

import numpy as np

from .common import LEAD, NUM_FRAMES

FORMAT = "fire-bundle-v2"


def run(scene, out_root):
    out = os.path.join(out_root, scene.hf_scene)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "images"))
    os.makedirs(os.path.join(out, "sync_mask"))

    frames, intervals = [], []
    for c in range(scene.num_cams):
        cam = scene.camera(c)
        c2w = np.linalg.inv(cam["w2c"])
        c2w[:3, 1:3] *= -1                       # OpenCV -> OpenGL/Blender camera axes
        angle_x = 2 * np.arctan(cam["w"] / (2 * cam["K"][0, 0]))
        times_ms = [scene.times_us[c][scene.video_frame(c, r)] * 1e-3 for r in range(NUM_FRAMES)]
        intervals.append(np.diff(times_ms))
        for r in range(NUM_FRAMES):
            name = f"{r:04d}_{c}"
            shutil.copy(scene.frame_path(c, r), os.path.join(out, "images", name + ".png"))
            frames.append({"file_path": f"images/{name}.png", "cam": c, "frame": r, "time": float(times_ms[r]),
                           "camera_angle_x": float(angle_x), "transform_matrix": c2w.tolist()})
        shutil.copy(scene.path("sync_mask", f"cam{c}.png"), os.path.join(out, "sync_mask", f"{c}.png"))

    transforms = {"format": FORMAT, "hf_scene": scene.hf_scene, "name": scene.name,
                  "num_frames": NUM_FRAMES, "num_cams": scene.num_cams,
                  "dt": float(np.median(np.concatenate(intervals))), "frames": frames}
    with open(os.path.join(out, "transforms.json"), "w") as f:
        json.dump(transforms, f, indent=1)

    for d in ["masks", "voxel_flow"]:
        shutil.copytree(scene.path(d), os.path.join(out, d), ignore=shutil.ignore_patterns(".done"))
    shutil.copy(scene.path("frames", "window.json"), os.path.join(out, "window.json"))
    if os.path.isdir(scene.path("mono_depth")):
        shutil.copytree(scene.path("mono_depth"), os.path.join(out, "mono_depth"), ignore=shutil.ignore_patterns(".done"))
    if os.path.exists(scene.path("static_gs", "static_point_cloud.ply")):
        shutil.copy(scene.path("static_gs", "static_point_cloud.ply"), os.path.join(out, "static_point_cloud.ply"))
    print(f"      bundle written to {out} (lead frames {LEAD} excluded)")
