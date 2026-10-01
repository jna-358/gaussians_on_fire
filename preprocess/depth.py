# Copyright (c) 2026 Jakob Nazarenus
# SPDX-License-Identifier: MIT

"""Monocular depth with Depth Anything V2.

ViT-L at input size 518; the raw output (relative inverse depth) is min-max
normalized per image and stored as uint16 PNG. Used on the background frames
(static reconstruction) and, in the mono_depth stage, on every window frame
(target of the rmse_depth evaluation metric).
"""
import logging
import os

import cv2
import numpy as np
import torch

from .common import NUM_FRAMES, mark_done, progress

MODEL_REPO = "depth-anything/Depth-Anything-V2-Large"
MODEL_REVISION = "cbbb86a30ce19b5684b7a05155dc7e6cbc7685b9"
MODEL_FILE = "depth_anything_v2_vitl.pth"
INPUT_SIZE = 518


def load_model(device="cuda"):
    from huggingface_hub import hf_hub_download
    logging.getLogger("dinov2").setLevel(logging.ERROR)   # "xFormers not available"
    from .depth_anything_v2.dpt import DepthAnythingV2
    model = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
    weights = hf_hub_download(MODEL_REPO, MODEL_FILE, revision=MODEL_REVISION)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    return model.to(device).eval()


def infer_uint16(model, image_bgr):
    depth = model.infer_image(image_bgr, INPUT_SIZE)
    return ((depth - depth.min()) / (depth.max() - depth.min()) * 65535.0).astype(np.uint16)


def run_mono_depth(scene):
    model = load_model()
    os.makedirs(scene.path("mono_depth"), exist_ok=True)
    for c, r in progress([(c, r) for c in range(scene.num_cams) for r in range(NUM_FRAMES)], desc="monocular depth"):
        depth = infer_uint16(model, cv2.imread(scene.frame_path(c, r)))
        cv2.imwrite(scene.path("mono_depth", f"{r:04d}_{c}.png"), depth)
    mark_done(scene, "mono_depth")
