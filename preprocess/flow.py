# Copyright (c) 2026 Jakob Nazarenus
# SPDX-License-Identifier: MIT

"""Stage 2: MEMFOF optical flow, anchored on each window frame, at two strides.

MEMFOF takes three frames and predicts the backward and forward flow of the
middle one. flow/stride<s>/cam<c>/<r>.png is the forward flow of frame r from
the window [r-s, r, r+s], divided by s so both strides are in pixels per frame.
Stride 2 runs as two passes over the even and the odd frames; it gives every
frame a flow that avoids its immediate neighbours (used next to held-out frames).

Encoding: x in channel 0, y in channel 1 (BGR
order), clipped to +-FLOW_RANGE px and quantized to uint8; channel 2 is 128.
"""
import os

import cv2
import numpy as np
import torch

from .common import NUM_FRAMES, mark_done, write_provenance, progress

MODEL = "egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH"
MODEL_REVISION = "6c6c9aa3ad64f93aee8efbc2f7a6e4535814ee96"
FLOW_RANGE = 25.0
STRIDES = (1, 2)


def encode_flow(flow):
    q = ((np.clip(flow, -FLOW_RANGE, FLOW_RANGE) + FLOW_RANGE) / (2 * FLOW_RANGE) * 255).astype(np.uint8)
    return np.concatenate([q, np.full(q.shape[:2] + (1,), 128, np.uint8)], axis=-1)


def decode_flow(image):
    return image[..., :2].astype(np.float32) / 255.0 * (2 * FLOW_RANGE) - FLOW_RANGE


def load_frame(path):
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)


@torch.no_grad()
def forward_flows(model, images, device):
    """Forward flow of every middle frame of consecutive triplets; images[i] -> flow of images[i + 1]."""
    frames, fmap_cache = [images[0], images[1]], [None] * 3
    for image in images[2:]:
        frames.append(image)
        out = model(torch.stack(frames, dim=1).to(device), fmap_cache=fmap_cache)
        yield out["flow"][-1][:, 1].squeeze(0).permute(1, 2, 0).cpu().numpy()
        fmap_cache = out["fmap_cache"]
        fmap_cache.pop(0)
        fmap_cache.append(None)
        frames.pop(0)


def run(scene, device="cuda"):
    from memfof import MEMFOF
    model = MEMFOF.from_pretrained(MODEL, revision=MODEL_REVISION).eval().to(device)
    provenance = {}
    bar = progress(total=sum(len([r for r in scene.window_range() if r % stride == parity]) - 2
                             for stride in STRIDES for _ in range(scene.num_cams) for parity in range(stride)),
                   desc="optical flow")
    for stride in STRIDES:
        for c in range(scene.num_cams):
            out_dir = scene.path("flow", f"stride{stride}", f"cam{c}")
            os.makedirs(out_dir, exist_ok=True)
            for parity in range(stride):
                seq = [r for r in scene.window_range() if r % stride == parity]
                images = [load_frame(scene.frame_path(c, r)) for r in seq]
                for anchor, flow in zip(seq[1:-1], forward_flows(model, images, device)):
                    bar.update(1)
                    if not 0 <= anchor < NUM_FRAMES:
                        continue
                    name = f"stride{stride}/cam{c}/{anchor:04d}.png"
                    cv2.imwrite(scene.path("flow", name), encode_flow(flow / stride))
                    provenance[name] = {"anchor": anchor, "stride": stride, "cam": c,
                                        "sources": {c: scene.source(c, [anchor - stride, anchor, anchor + stride])}}
    bar.close()
    write_provenance(scene, "flow", provenance)
    mark_done(scene, "flow")
