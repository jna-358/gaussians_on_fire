# Copyright (c) 2026 Jakob Nazarenus
# SPDX-License-Identifier: MIT

"""Stage 3: per-frame masks, masks/stride<s>/<r:04d>_<cam>.png, one set per flow stride.

Channels (as read by the training code, RGB): R = LED sync board,
G = flame (flow magnitude > FLOW_THRESHOLD px/frame), B = flame dilated.
The sync board overrides both flame channels.
"""
import json
import os

import cv2
import numpy as np

from .common import NUM_FRAMES, mark_done, progress, write_provenance
from .flow import STRIDES, decode_flow

FLOW_THRESHOLD = 0.5
DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
DILATE_ITERATIONS = 2


def flame_mask(scene, stride, c, r):
    flow = decode_flow(cv2.imread(scene.path("flow", f"stride{stride}", f"cam{c}", f"{r:04d}.png")))
    return np.linalg.norm(flow, axis=-1) > FLOW_THRESHOLD


def run(scene):
    flow_provenance = json.load(open(scene.path("flow", "provenance.json")))["files"]
    provenance = {}
    bar = progress(total=len(STRIDES) * scene.num_cams * NUM_FRAMES, desc="masks")
    for stride in STRIDES:
        os.makedirs(scene.path("masks", f"stride{stride}"), exist_ok=True)
        for c in range(scene.num_cams):
            sync = scene.sync_mask(c)
            for r in range(NUM_FRAMES):
                bar.update(1)
                flame = flame_mask(scene, stride, c, r)
                dilated = cv2.dilate(flame.astype(np.uint8) * 255, DILATE_KERNEL, iterations=DILATE_ITERATIONS) > 0
                mask = np.zeros(flame.shape + (3,), np.uint8)   # BGR
                mask[flame, 1] = 255
                mask[dilated, 0] = 255
                mask[sync] = (0, 0, 255)
                name = f"stride{stride}/{r:04d}_{c}.png"
                cv2.imwrite(scene.path("masks", name), mask)
                provenance[name] = flow_provenance[f"stride{stride}/cam{c}/{r:04d}.png"]
    bar.close()
    write_provenance(scene, "masks", provenance)
    mark_done(scene, "masks")
