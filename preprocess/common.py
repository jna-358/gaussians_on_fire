# Copyright (c) 2026 Jakob Nazarenus
# SPDX-License-Identifier: MIT

"""Scene geometry, timing and provenance shared by all preprocessing stages.

Input is one scene of the Hugging Face dataset jna-358/fire_actioncam
(scenes/<id>/{000,001,002}.mp4, {cam}.mask.png, meta.npz). Everything the
stages produce lives in a per-scene work directory.

Frames are addressed by their window index r: r = 0..NUM_FRAMES-1 is the
reconstruction window, r < 0 and r >= NUM_FRAMES are the LEAD frames around it
that flow windows may reach into. Nothing here knows the train/test split;
every derived file records in a provenance.json which window frames it was
computed from, so the split can be applied (and checked) when loading.

Some stages deliberately use frames beyond the training frames: the LEAD
frames (flow of the first and last window frames), the flame masks for the
voxel grid's bounding box (sampled over the whole video) and the background
(per-pixel minimum over a widened window). Test frames can enter the latter
two only as a mask or a per-pixel minimum, never as training supervision, so
the preprocessing stays independent of the split.
"""
import json
import os

import cv2
import numpy as np
from tqdm import tqdm

NUM_FRAMES = 100    # consecutive frames per camera (250 ms at 400 fps)
LEAD = 2            # extra frames before and after the window (stride-2 flow reaches r-2 and r+2)
ZOOM = 1.03         # isotropic target camera, zoomed in slightly to avoid black borders

def progress(iterable=None, total=None, desc=""):
    """Progress bar of a loop inside a stage; cleared when the loop is done."""
    return tqdm(iterable, total=total, desc=f"      {desc}", leave=False, dynamic_ncols=True)


SCENES = json.load(open(os.path.join(os.path.dirname(__file__), "scenes.json")))


class Scene:
    def __init__(self, hf_root, hf_scene, work_root):
        self.hf_scene = hf_scene
        self.name = SCENES[hf_scene]["name"]
        self.start = SCENES[hf_scene]["start_frame"]
        self.src = os.path.join(hf_root, "scenes", hf_scene)
        self.work = os.path.join(work_root, hf_scene)
        with np.load(os.path.join(self.src, "meta.npz")) as meta:
            self.meta = dict(meta)
        self.num_cams = self.meta["times"].shape[0]
        # Absolute LED-clock time; the published times are zeroed at the scene start.
        self.times_us = self.meta["times"] + float(self.meta["led_time_origin_us"])

    def path(self, *parts):
        return os.path.join(self.work, *parts)

    def video(self, c):
        return os.path.join(self.src, f"{c:03d}.mp4")

    def video_frame(self, c, r):
        """HF video frame of camera c at window index r; camera 0 is the time reference."""
        first = int(np.argmin(np.abs(self.times_us[c] - self.times_us[0][self.start])))
        return first + r

    def window_range(self):
        return range(-LEAD, NUM_FRAMES + LEAD)

    def check_window(self):
        """Every camera's frames must be consecutive and nearest in time to camera 0's."""
        for c in range(self.num_cams):
            for r in self.window_range():
                t0 = self.times_us[0][self.video_frame(0, r)]
                nearest = int(np.argmin(np.abs(self.times_us[c] - t0)))
                if nearest != self.video_frame(c, r):
                    raise RuntimeError(f"{self.hf_scene} cam {c}: window frame {r} is not the nearest-time frame")

    def frame_path(self, c, r):
        return self.path("frames", f"cam{c}", f"{r + LEAD:04d}.png")

    def camera(self, c):
        """Undistorted pinhole camera: the alpha=0 optimal camera with
        fx = fy = ZOOM * max(fx, fy) and (cx, cy) = (w/2, h/2).
        """
        K = self.meta["camera_matrix"][c]
        dist = self.meta["dist_coeffs"][c]
        w, h = (int(v) for v in self.meta["image_size"][c])
        K_opt, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0)
        f = ZOOM * max(K_opt[0, 0], K_opt[1, 1])
        K_simple = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])
        w2c = np.eye(4)
        w2c[:3, :3], w2c[:3, 3] = self.meta["R"][c], self.meta["t"][c]
        return {"K": K_simple, "K_distorted": K, "dist": dist, "w": w, "h": h, "w2c": w2c}

    def rectify_maps(self, c):
        cam = self.camera(c)
        return cv2.initUndistortRectifyMap(cam["K_distorted"], cam["dist"], None, cam["K"],
                                           (cam["w"], cam["h"]), cv2.CV_32FC1)

    def sync_mask(self, c):
        return cv2.imread(self.path("sync_mask", f"cam{c}.png"), cv2.IMREAD_GRAYSCALE) > 0

    def source(self, c, frames):
        """Provenance entry: the window frames used from camera c and their HF video frames."""
        return {"video": os.path.basename(self.video(c)), "window_frames": list(frames),
                "video_frames": [self.video_frame(c, r) for r in frames]}


def write_provenance(scene, stage_dir, entries):
    with open(scene.path(stage_dir, "provenance.json"), "w") as f:
        json.dump({"hf_scene": scene.hf_scene, "files": entries}, f, indent=1)


def stage_done(scene, stage):
    return os.path.exists(scene.path(stage, ".done"))


def mark_done(scene, stage):
    open(scene.path(stage, ".done"), "w").close()
