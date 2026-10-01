# Copyright (c) 2026 Jakob Nazarenus
# SPDX-License-Identifier: MIT

"""Static scene: background frames, dense stereo, aligned mono depth, fused point cloud.

Stage background: per-pixel minimum (smallest RGB norm) over the window
widened (asymmetrically) by BACKGROUND_EXPAND, computed on the decoded frames,
then rectified.

Stage static_init:
  - COLMAP PatchMatch stereo (default options) on the three background frames
    with the HF poses; the HF sparse points, projected into every image, give
    COLMAP its per-image depth ranges.
  - Depth Anything V2 on the background frames, fitted to the stereo inverse
    depth by least squares (scale, offset) per camera.
  - Fused depth: stereo where it is valid (dilated), aligned mono elsewhere;
    back-projected, merged, projectively downsampled and clamped to
    MAX_DISTANCE: the initial point cloud of the static 3DGS.
Absolute constants below assume the dataset's world scale (outer camera baseline 10).
"""
import json
import os
import shutil

import cv2
import numpy as np
from plyfile import PlyData, PlyElement

from .common import NUM_FRAMES, mark_done, progress, write_provenance
from . import depth as mono

BACKGROUND_EXPAND = 2.5
DOWNSAMPLE_BASE_VOXEL = 0.01    # voxel size per unit distance from the world origin
DOWNSAMPLE_INTERVALS = 10
MAX_DISTANCE = 300.0
STEREO_DILATE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
STEREO_DILATE_ITERATIONS = 4
MAX_DEPTH_FACTOR = 5.0          # fused depths beyond this times the largest stereo depth are dropped


# ------------------------------------------------------------------ background

def run_background(scene):
    T = scene.times_us
    t0, t1 = T[0][scene.video_frame(0, 0)], T[0][scene.video_frame(0, NUM_FRAMES)]
    t_min, t_max = max(t.min() for t in T), min(t.max() for t in T)
    start = max(t_min, t0 - (t1 - t0) * BACKGROUND_EXPAND / 2)
    end = min(t_max, t1 + (t1 - start) * BACKGROUND_EXPAND / 2)
    firsts = [int(np.argmin(np.abs(T[c] - start))) for c in range(scene.num_cams)]
    count = min(int(np.argmin(np.abs(T[c] - end))) - firsts[c] for c in range(scene.num_cams))

    os.makedirs(scene.path("background"), exist_ok=True)
    provenance = {}
    for c in range(scene.num_cams):
        cap = cv2.VideoCapture(scene.video(c))
        background, norm = None, None
        for pos in progress(range(firsts[c] + count), desc=f"decoding camera {c}"):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"{scene.video(c)}: decode failed at frame {pos}")
            if pos < firsts[c]:
                continue
            frame_norm = np.linalg.norm(frame.astype(np.float32), axis=-1)
            if background is None:
                background, norm = frame.copy(), frame_norm
            else:
                smaller = frame_norm < norm
                background[smaller], norm[smaller] = frame[smaller], frame_norm[smaller]
        cap.release()
        map_x, map_y = scene.rectify_maps(c)
        cv2.imwrite(scene.path("background", f"cam{c}.png"), cv2.remap(background, map_x, map_y, cv2.INTER_LINEAR))
        provenance[f"cam{c}.png"] = {"sources": {c: {"video": os.path.basename(scene.video(c)),
                                                     "video_frames": [firsts[c], firsts[c] + count - 1]}}}
    write_provenance(scene, "background", provenance)
    mark_done(scene, "background")


# ------------------------------------------------------------------ dense stereo

def _qvec(R):
    """COLMAP quaternion (qw, qx, qy, qz) of a rotation matrix."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([[Rxx - Ryy - Rzz, 0, 0, 0], [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                  [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0], [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    vals, vecs = np.linalg.eigh(K)
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    return -q if q[0] < 0 else q


def _write_sparse_model(scene, sparse_dir):
    """PINHOLE cameras, HF poses, and the HF points with a track in every image they project into."""
    points, colors = scene.meta["points3D"], scene.meta["points3D_rgb"]
    tracks = [[] for _ in points]
    with open(os.path.join(sparse_dir, "cameras.txt"), "w") as fc, open(os.path.join(sparse_dir, "images.txt"), "w") as fi:
        for c in range(scene.num_cams):
            cam = scene.camera(c)
            K, w2c = cam["K"], cam["w2c"]
            fc.write(f"{c + 1} PINHOLE {cam['w']} {cam['h']} {K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]}\n")
            p = points @ w2c[:3, :3].T + w2c[:3, 3]
            uv = p @ K.T
            uv = uv[:, :2] / uv[:, 2:3]
            visible = (p[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < cam["w"]) & (uv[:, 1] >= 0) & (uv[:, 1] < cam["h"])
            obs = []
            for i in np.nonzero(visible)[0]:
                tracks[i].append((c + 1, len(obs)))
                obs.append(f"{uv[i, 0]} {uv[i, 1]} {i + 1}")
            q, t = _qvec(w2c[:3, :3]), w2c[:3, 3]
            fi.write(f"{c + 1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {c + 1} cam{c}.png\n{' '.join(obs)}\n")
    with open(os.path.join(sparse_dir, "points3D.txt"), "w") as f:
        for i, (p, rgb) in enumerate(zip(points, colors)):
            if tracks[i]:
                track = " ".join(f"{img} {idx}" for img, idx in tracks[i])
                f.write(f"{i + 1} {p[0]} {p[1]} {p[2]} {int(rgb[0])} {int(rgb[1])} {int(rgb[2])} 0 {track}\n")


def _read_colmap_depth_map(path):
    with open(path, "rb") as f:
        width, height, channels = np.genfromtxt(f, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int)
        f.seek(0)
        num_delimiter, byte = 0, f.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = f.read(1)
        data = np.fromfile(f, np.float32)
    return data.reshape((width, height, channels), order="F").transpose((1, 0, 2)).squeeze()


def dense_stereo(scene):
    import pycolmap
    pycolmap.logging.minloglevel = 2   # errors only; COLMAP's info log is very verbose
    root = scene.path("static_init", "colmap")
    shutil.rmtree(root, ignore_errors=True)
    images, sparse, dense = (os.path.join(root, d) for d in ("images", "sparse", "dense"))
    for d in (images, sparse, dense):
        os.makedirs(d)
    for c in range(scene.num_cams):
        shutil.copy(scene.path("background", f"cam{c}.png"), os.path.join(images, f"cam{c}.png"))
    _write_sparse_model(scene, sparse)
    rec = pycolmap.Reconstruction()
    rec.read_text(sparse)
    rec.write_binary(sparse)

    pycolmap.undistort_images(output_path=dense, input_path=sparse, image_path=images)
    pycolmap.patch_match_stereo(workspace_path=dense, workspace_format="COLMAP",
                                pmvs_option_name="option-all", options=pycolmap.PatchMatchOptions())
    depths = []
    for c in range(scene.num_cams):
        depth = _read_colmap_depth_map(os.path.join(dense, "stereo", "depth_maps", f"cam{c}.png.geometric.bin"))
        assert depth.shape == (scene.camera(c)["h"], scene.camera(c)["w"]), depth.shape
        depths.append(depth)
    return depths


# ------------------------------------------------------------------ fusion

def voxel_down_sample(points, colors, voxel_size):
    """Average of points and colors per voxel (as Open3D's voxel_down_sample)."""
    origin = points.min(axis=0) - voxel_size * 0.5
    keys = np.floor((points - origin) / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    out_p = np.zeros((len(counts), 3))
    out_c = np.zeros((len(counts), 3))
    np.add.at(out_p, inverse, points)
    np.add.at(out_c, inverse, colors)
    return out_p / counts[:, None], out_c / counts[:, None]


def projective_down_sample(points, colors):
    """Voxel size growing linearly with the distance from the world origin (log-spaced shells)."""
    dist = np.linalg.norm(points, axis=1)
    edges = np.logspace(np.log10(dist.min()), np.log10(dist.max()), DOWNSAMPLE_INTERVALS + 1)
    out_p, out_c = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        shell = (dist >= lo) & (dist < hi)
        if shell.any():
            p, col = voxel_down_sample(points[shell], colors[shell], DOWNSAMPLE_BASE_VOXEL * (lo + hi) / 2)
            out_p.append(p)
            out_c.append(col)
    return np.concatenate(out_p), np.concatenate(out_c)


def write_point_cloud(path, points, colors):
    vertex = np.empty(len(points), dtype=[(n, "f4") for n in ("x", "y", "z", "nx", "ny", "nz")] + [(n, "u1") for n in ("red", "green", "blue")])
    vertex["x"], vertex["y"], vertex["z"] = points.T
    vertex["nx"] = vertex["ny"] = vertex["nz"] = 0
    rgb = (colors * 255).astype(np.uint8)
    vertex["red"], vertex["green"], vertex["blue"] = rgb.T
    PlyData([PlyElement.describe(vertex, "vertex")]).write(path)


def run_static_init(scene):
    os.makedirs(scene.path("static_init", "mono"), exist_ok=True)
    stereo = dense_stereo(scene)
    np.savez_compressed(scene.path("static_init", "stereo_depth.npz"), **{f"cam{c}": d for c, d in enumerate(stereo)})

    model = mono.load_model()
    depth_params, all_points, all_colors = {}, [], []
    for c in range(scene.num_cams):
        image = cv2.imread(scene.path("background", f"cam{c}.png"))
        disparity16 = mono.infer_uint16(model, image)
        cv2.imwrite(scene.path("static_init", "mono", f"cam{c}.png"), disparity16)
        disparity = disparity16.astype(np.float32) / 65535

        with np.errstate(divide="ignore"):
            stereo_inv = 1.0 / stereo[c]
        valid = (disparity > 0) & (stereo_inv > 0) & np.isfinite(disparity) & np.isfinite(stereo_inv)
        A = np.stack([disparity[valid], np.ones(valid.sum())], axis=1)
        scale, offset = np.linalg.lstsq(A, stereo_inv[valid], rcond=None)[0]
        depth_params[f"cam{c}"] = {"scale": float(scale), "offset": float(offset)}

        stereo_region = cv2.dilate((stereo[c] > 0).astype(np.uint8) * 255, STEREO_DILATE, iterations=STEREO_DILATE_ITERATIONS) > 0
        fused_inv = stereo_inv.copy()
        fused_inv[~stereo_region] = scale * disparity[~stereo_region] + offset
        with np.errstate(divide="ignore"):
            fused = 1.0 / fused_inv
        fused[(fused > MAX_DEPTH_FACTOR * stereo[c].max()) | (fused < 0) | ~np.isfinite(fused)] = 0
        print(f"      camera {c}: stereo coverage {100 * (stereo[c] > 0).mean():.1f}%, mono fit scale {scale:.4f} offset {offset:.4f}")

        cam = scene.camera(c)
        v, u = np.nonzero(fused > 0)
        z = fused[v, u]
        p_cam = np.stack([(u - cam["K"][0, 2]) * z / cam["K"][0, 0], (v - cam["K"][1, 2]) * z / cam["K"][1, 1], z], axis=1)
        R, t = cam["w2c"][:3, :3], cam["w2c"][:3, 3]
        all_points.append((p_cam - t) @ R)          # R^T (p - t)
        all_colors.append(image[v, u][:, ::-1].astype(np.float64) / 255.0)

    points, colors = projective_down_sample(np.concatenate(all_points), np.concatenate(all_colors))
    dist = np.linalg.norm(points, axis=1)
    far = dist > MAX_DISTANCE
    points[far] *= (MAX_DISTANCE / dist[far])[:, None]
    write_point_cloud(scene.path("static_init", "init_points.ply"), points, colors)
    with open(scene.path("static_init", "depth_params.json"), "w") as f:
        json.dump(depth_params, f, indent=2)
    print(f"      initial point cloud: {len(points)} points ({far.sum()} clamped to distance {MAX_DISTANCE})")
    mark_done(scene, "static_init")
