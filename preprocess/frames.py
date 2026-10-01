# Copyright (c) 2026 Jakob Nazarenus
# SPDX-License-Identifier: MIT

"""Stage 1: decode the window (plus LEAD frames on each side) and rectify it.

Writes frames/cam<c>/<r + LEAD>.png, sync_mask/cam<c>.png (the LED board) and
frames/window.json (window index -> HF video frame and capture time per camera).
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor

import cv2

from .common import LEAD, NUM_FRAMES, mark_done, progress


def _process_camera(scene, c):
    map_x, map_y = scene.rectify_maps(c)
    wanted = {scene.video_frame(c, r): r for r in scene.window_range()}
    os.makedirs(scene.path("frames", f"cam{c}"), exist_ok=True)

    # Decode sequentially; seeking in HEVC is not frame-accurate.
    cap = cv2.VideoCapture(scene.video(c))
    for pos in progress(range(max(wanted) + 1), desc=f"decoding camera {c}"):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"{scene.video(c)}: decode failed at frame {pos}")
        if pos in wanted:
            cv2.imwrite(scene.frame_path(c, wanted[pos]), cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR))
    cap.release()

    mask = cv2.imread(os.path.join(scene.src, f"{c:03d}.mask.png"), cv2.IMREAD_GRAYSCALE)
    mask = (cv2.remap(mask, map_x, map_y, cv2.INTER_LINEAR) > 127).astype("uint8") * 255
    cv2.imwrite(scene.path("sync_mask", f"cam{c}.png"), mask)


def run(scene):
    scene.check_window()
    os.makedirs(scene.path("sync_mask"), exist_ok=True)
    with ProcessPoolExecutor(max_workers=scene.num_cams) as ex:
        list(ex.map(_process_camera, [scene] * scene.num_cams, range(scene.num_cams)))

    window = {"hf_scene": scene.hf_scene, "name": scene.name, "num_frames": NUM_FRAMES, "lead": LEAD,
              "cams": {c: {"video": os.path.basename(scene.video(c)),
                           "video_frames": [scene.video_frame(c, r) for r in scene.window_range()],
                           "times_us": [float(scene.times_us[c][scene.video_frame(c, r)]) for r in scene.window_range()]}
                       for c in range(scene.num_cams)}}
    with open(scene.path("frames", "window.json"), "w") as f:
        json.dump(window, f, indent=1)
    mark_done(scene, "frames")
