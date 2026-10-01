# Copyright (c) 2026 Jakob Nazarenus
# SPDX-License-Identifier: MIT

"""Stage 4: voxel carving and per-voxel 3D flow (initialization of the dynamic Gaussians).

Grid: flame masks at GRID_SAMPLES frames spread over the whole video give a
flame centre (triangulated) and a bounding box, split into ~NUM_VOXELS voxels.
Flow: for every window frame r and flow stride s, a voxel is carved if it lies
in the flame mask of at least MIN_CAMERAS cameras; its 3D flow (world units per
frame) is fitted to the 2D flows of all cameras by ridge regression.

Writes voxel_flow/grid.npz (coordinates, voxel_size, bbox) and
voxel_flow/stride<s>/<r:04d>.npz (flows: NaN outside the carved voxels).
"""
import os

import cv2
import numpy as np
import torch

from .common import NUM_FRAMES, mark_done, progress, write_provenance
from .flow import MODEL, MODEL_REVISION, STRIDES, decode_flow, encode_flow, forward_flows
from .masks import FLOW_THRESHOLD

GRID_SAMPLES = 100
NUM_VOXELS = 100_000
MIN_CAMERAS = 2


def triangulate_point(points_2d, Ks, w2cs):
    """DLT triangulation of one point seen in several cameras."""
    A = np.zeros((2 * len(points_2d), 4))
    for i, ((x, y), K, w2c) in enumerate(zip(points_2d, Ks, w2cs)):
        P = K @ w2c[:3, :]
        A[2 * i] = x * P[2] - P[0]
        A[2 * i + 1] = y * P[2] - P[1]
    X = np.linalg.svd(A, full_matrices=False)[2][-1]
    return X[:3] / X[3]


def unproject(pixels, depth, K, w2c):
    rays = (np.linalg.inv(K) @ np.concatenate([pixels, np.ones((len(pixels), 1))], axis=1).T).T
    cam = np.concatenate([rays * depth, np.ones((len(pixels), 1))], axis=1)
    return (np.linalg.inv(w2c) @ cam.T).T[:, :3]


def combine_flows(voxel_flows, coordinates, carved, w2cs, Ks, device):
    """3D flow per voxel from the 2D flows of all cameras (ridge regression).

    Each camera's 2D flow is unprojected at the voxel's depth, giving u_i, the
    component of the 3D flow v perpendicular to that camera's viewing ray. v
    solves min ||A v - b||^2 + lambda ||v||^2 with rows u_i^T, b_i = |u_i|^2 and
    lambda = 0.1 * mean |u_i|^2.
    """
    num_cams = len(w2cs)
    nx, ny, nz = coordinates.shape[:3]
    coords = torch.from_numpy(coordinates).float().to(device)
    coords_h = torch.cat([coords, torch.ones_like(coords[..., :1])], dim=-1)
    K = torch.from_numpy(np.stack(Ks)).float().to(device)
    w2c = torch.from_numpy(np.stack(w2cs)).float().to(device)
    flows_2d = torch.from_numpy(np.stack(voxel_flows)).float().to(device)

    cam = torch.matmul(w2c[:, None, None, None], coords_h[None, ..., None])[..., 0]
    depths = cam[..., 2:3]
    img = torch.matmul(K[:, None, None, None], cam[..., :3, None])[..., 0]
    tips = img[..., :2] / img[..., 2:3] + flows_2d
    tips_cam = torch.matmul(torch.linalg.inv(K)[:, None, None, None],
                            torch.cat([tips, torch.ones_like(tips[..., :1])], dim=-1)[..., None])[..., 0] * depths
    tips_world = torch.matmul(torch.linalg.inv(w2c)[:, None, None, None],
                              torch.cat([tips_cam, torch.ones_like(tips_cam[..., :1])], dim=-1)[..., None])[..., :3, 0]
    u = tips_world - coords[None]
    valid = (depths[..., 0] > 0) & torch.isfinite(depths[..., 0]) & torch.all(torch.isfinite(u), dim=-1)

    n = nx * ny * nz
    u = u.permute(1, 2, 3, 0, 4).reshape(n, num_cams, 3)
    valid = valid.permute(1, 2, 3, 0).reshape(n, num_cams)
    A = u.clone()
    A[~valid] = 0.0
    b = torch.sum(A * A, dim=2)
    ATA = torch.matmul(A.transpose(1, 2), A)
    lam = torch.clamp(0.1 * b.sum(dim=1) / valid.sum(dim=1).clamp(min=1), min=1e-6)
    ATA = ATA + lam.view(-1, 1, 1) * torch.eye(3, device=device)
    ATb = torch.matmul(A.transpose(1, 2), b.unsqueeze(-1)).squeeze(-1)
    solvable = (valid.sum(dim=1) >= 1) & torch.all(torch.isfinite(ATA).view(n, -1), dim=1) & torch.all(torch.isfinite(ATb), dim=1)

    result = torch.full((n, 3), float("nan"), device=device)
    if solvable.any():
        result[solvable] = torch.linalg.solve(ATA[solvable], ATb[solvable, :, None]).squeeze(-1)
    result = result.reshape(nx, ny, nz, 3)
    result[~torch.from_numpy(carved).to(device)] = float("nan")
    return result.cpu().numpy()


@torch.no_grad()
def grid_sample_masks(scene, model, device):
    """Flame masks (sync board removed) at GRID_SAMPLES frames spread over the whole video."""
    T = scene.times_us
    i0 = int(np.argmin(np.abs(T[0] - max(t.min() for t in T))))
    i1 = int(np.argmin(np.abs(T[0] - min(t.max() for t in T))))
    buffer = (i1 - i0) // 8
    steps = np.linspace(i0 + buffer, i1 - buffer, GRID_SAMPLES, dtype=np.int32)
    masks = [[None] * len(steps) for _ in range(scene.num_cams)]
    sources = {}
    for c in range(scene.num_cams):
        centers = [int(np.argmin(np.abs(T[c] - T[0][k]))) for k in steps]
        sources[c] = {"video": os.path.basename(scene.video(c)), "video_frames": sorted({k + d for k in centers for d in (-1, 0, 1)})}
        needed = set(sources[c]["video_frames"])
        map_x, map_y = scene.rectify_maps(c)
        frames = {}
        cap = cv2.VideoCapture(scene.video(c))
        for pos in progress(range(max(needed) + 1), desc=f"decoding camera {c} (grid samples)"):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"{scene.video(c)}: decode failed at frame {pos}")
            if pos in needed:
                image = cv2.cvtColor(cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
                frames[pos] = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        cap.release()
        sync = scene.sync_mask(c)
        for i, k in enumerate(centers):
            flow = next(forward_flows(model, [frames[k - 1], frames[k], frames[k + 1]], device))
            flow = decode_flow(encode_flow(flow))   # same uint8 quantization as the stored flow
            masks[c][i] = (np.linalg.norm(flow, axis=-1) > FLOW_THRESHOLD) & ~sync
    return masks, sources


def build_grid(scene, masks):
    cams = [scene.camera(c) for c in range(scene.num_cams)]
    Ks, w2cs = [cam["K"] for cam in cams], [cam["w2c"] for cam in cams]
    usable = [i for i in range(len(masks[0])) if all(masks[c][i].any() for c in range(scene.num_cams))]

    centers = []
    for i in usable:
        pts = [np.array(np.nonzero(masks[c][i])[::-1], dtype=np.float64).mean(axis=1) for c in range(scene.num_cams)]
        centers.append(triangulate_point(pts, Ks, w2cs))
    center = np.mean(centers, axis=0)
    depths = [(w2c @ np.append(center, 1.0))[2] for w2c in w2cs]

    bbox = np.stack([np.full(3, np.inf), np.full(3, -np.inf)], axis=1)
    for i in usable:
        pts = np.concatenate([unproject(np.stack(np.nonzero(masks[c][i])[::-1], axis=1).astype(np.float64),
                                        depths[c], Ks[c], w2cs[c]) for c in range(scene.num_cams)])
        bbox[:, 0] = np.minimum(bbox[:, 0], pts.min(axis=0))
        bbox[:, 1] = np.maximum(bbox[:, 1], pts.max(axis=0))

    dims = bbox[:, 1] - bbox[:, 0]
    bbox[:, 0] -= 0.1 * dims
    bbox[:, 1] += 0.1 * dims
    mid, dims = bbox.mean(axis=1), bbox[:, 1] - bbox[:, 0]
    side = dims[1:].max()                      # y and z get the same extent
    bbox[1:, 0], bbox[1:, 1] = mid[1:] - side / 2, mid[1:] + side / 2

    voxel_size = (np.prod(bbox[:, 1] - bbox[:, 0]) / NUM_VOXELS) ** (1 / 3)
    axes = [np.arange(int(np.ceil((bbox[a, 1] - bbox[a, 0]) / voxel_size))) * voxel_size + bbox[a, 0] for a in range(3)]
    coordinates = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return coordinates, voxel_size, bbox, len(usable)


def run(scene, device="cuda"):
    from memfof import MEMFOF
    model = MEMFOF.from_pretrained(MODEL, revision=MODEL_REVISION).eval().to(device)
    masks, grid_sources = grid_sample_masks(scene, model, device)
    del model
    coordinates, voxel_size, bbox, num_used = build_grid(scene, masks)
    os.makedirs(scene.path("voxel_flow"), exist_ok=True)
    np.savez_compressed(scene.path("voxel_flow", "grid.npz"), coordinates=coordinates, voxel_size=voxel_size, bbox=bbox)
    print(f"      grid {coordinates.shape[:3]}, voxel size {voxel_size:.4f}, from {num_used}/{GRID_SAMPLES} samples")

    cams = [scene.camera(c) for c in range(scene.num_cams)]
    Ks, w2cs = [cam["K"] for cam in cams], [cam["w2c"] for cam in cams]
    pixels, in_frame = [], []
    for cam in cams:
        p = (cam["w2c"][None, None, None, :3, :3] @ coordinates[..., None])[..., 0] + cam["w2c"][:3, 3]
        p = (cam["K"][None, None, None] @ p[..., None])[..., 0]
        p = p[..., :2] / p[..., 2:3]
        ok = (p[..., 0] >= 0) & (p[..., 0] < cam["w"] - 1) & (p[..., 1] >= 0) & (p[..., 1] < cam["h"] - 1)
        p[~ok] = 0
        pixels.append(np.round(p).astype(np.int32))
        in_frame.append(ok)
    in_any_frame = np.any(in_frame, axis=0)

    provenance = {"grid.npz": {"sources": grid_sources}}
    bar = progress(total=len(STRIDES) * NUM_FRAMES, desc="voxel flow")
    for stride in STRIDES:
        os.makedirs(scene.path("voxel_flow", f"stride{stride}"), exist_ok=True)
        for r in range(NUM_FRAMES):
            bar.update(1)
            voxel_flows, in_flame, sources = [], [], {}
            for c in range(scene.num_cams):
                flow = decode_flow(cv2.imread(scene.path("flow", f"stride{stride}", f"cam{c}", f"{r:04d}.png")))
                flow[scene.sync_mask(c)] = 0
                flame = np.linalg.norm(flow, axis=-1) > FLOW_THRESHOLD
                px, py = pixels[c][..., 0], pixels[c][..., 1]
                in_flame.append(flame[py, px])
                voxel_flows.append(flow[py, px])
                sources[c] = scene.source(c, [r - stride, r, r + stride])
            carved = in_any_frame & (np.sum(in_flame, axis=0) >= MIN_CAMERAS)
            flows = combine_flows(voxel_flows, coordinates, carved, w2cs, Ks, device)
            name = f"stride{stride}/{r:04d}.npz"
            np.savez_compressed(scene.path("voxel_flow", name), flows=flows.astype(np.float32))
            provenance[name] = {"anchor": r, "stride": stride, "time_us": float(scene.times_us[0][scene.video_frame(0, r)]),
                                "carved_voxels": int(carved.sum()), "sources": sources}
    bar.close()
    write_provenance(scene, "voxel_flow", provenance)
    mark_done(scene, "voxel_flow")
