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

"""Stage static_gs: static 3DGS on the three background frames -> static_point_cloud.ply.

Vanilla 3D Gaussian Splatting (graphdeco-inria/gaussian-splatting 54c035f) with
the paper's settings: SH degree 0, all three cameras, 10000 iterations,
densification until 5000 with an opacity reset every 1500, depth loss against
the aligned mono depth weighted 100x, and no size-based pruning (it would
remove the distant background).
Initialized from static_init/init_points.ply.
"""
import json
import math
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement
from torch import nn

from .common import mark_done, progress

ITERATIONS = 10_000
POSITION_LR_INIT, POSITION_LR_FINAL, POSITION_LR_DELAY_MULT, POSITION_LR_MAX_STEPS = 0.00016, 0.0000016, 0.01, 30_000
FEATURE_LR, OPACITY_LR, SCALING_LR, ROTATION_LR = 0.0025, 0.025, 0.005, 0.001
PERCENT_DENSE = 0.01
LAMBDA_DSSIM = 0.2
DENSIFY_FROM, DENSIFY_UNTIL, DENSIFY_INTERVAL, DENSIFY_GRAD_THRESHOLD = 500, 5_000, 100, 0.0002
OPACITY_RESET_INTERVAL = 1_500
DEPTH_WEIGHT_INIT, DEPTH_WEIGHT_FINAL, DEPTH_WEIGHT_FACTOR = 1.0, 0.01, 1e2
MIN_OPACITY = 0.005
PRUNE_LARGE = False      # vanilla 3DGS: True
SH_C0 = 0.28209479177387814


def knn_mean_sq_dist(points, k=3, chunk=4096):
    """Mean squared distance to the k nearest neighbours (what simple_knn's distCUDA2 computes)."""
    out = torch.empty(len(points), device=points.device)
    for i in range(0, len(points), chunk):
        d = torch.cdist(points[i:i + chunk], points) ** 2
        d[torch.arange(d.shape[0]), torch.arange(i, i + d.shape[0])] = float("inf")
        out[i:i + chunk] = d.topk(k, largest=False).values.mean(dim=1)
    return out


class StaticGaussians:
    """Static 3D Gaussians exposing what gaussian_renderer.render needs."""

    def __init__(self, points, colors, spatial_lr_scale):
        from utils.general_utils import get_expon_lr_func, inverse_sigmoid
        self.inverse_sigmoid = inverse_sigmoid
        self.active_sh_degree = 0
        xyz = torch.tensor(points, dtype=torch.float, device="cuda")
        dist2 = torch.clamp_min(knn_mean_sq_dist(xyz), 1e-7)
        n = len(xyz)
        self._xyz = nn.Parameter(xyz)
        self._features_dc = nn.Parameter(((torch.tensor(colors, dtype=torch.float, device="cuda") - 0.5) / SH_C0)[:, None, :].contiguous())
        self._scaling = nn.Parameter(torch.log(torch.sqrt(dist2))[:, None].repeat(1, 3))
        rots = torch.zeros((n, 4), device="cuda")
        rots[:, 0] = 1
        self._rotation = nn.Parameter(rots)
        self._opacity = nn.Parameter(inverse_sigmoid(0.1 * torch.ones((n, 1), device="cuda")))
        self.max_radii2D = torch.zeros(n, device="cuda")
        self.xyz_gradient_accum = torch.zeros((n, 1), device="cuda")
        self.denom = torch.zeros((n, 1), device="cuda")
        self.optimizer = torch.optim.Adam([
            {"params": [self._xyz], "lr": POSITION_LR_INIT * spatial_lr_scale, "name": "xyz"},
            {"params": [self._features_dc], "lr": FEATURE_LR, "name": "f_dc"},
            {"params": [self._opacity], "lr": OPACITY_LR, "name": "opacity"},
            {"params": [self._scaling], "lr": SCALING_LR, "name": "scaling"},
            {"params": [self._rotation], "lr": ROTATION_LR, "name": "rotation"},
        ], lr=0.0, eps=1e-15)
        self.xyz_scheduler = get_expon_lr_func(POSITION_LR_INIT * spatial_lr_scale, POSITION_LR_FINAL * spatial_lr_scale,
                                               lr_delay_mult=POSITION_LR_DELAY_MULT, max_steps=POSITION_LR_MAX_STEPS)

    # --- interface used by gaussian_renderer.render
    @property
    def get_xyz(self):
        return self._xyz

    def get_xyz_at(self, time):
        return self._xyz

    def get_opacity_at(self, time):
        return self.get_opacity

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_rotation(self):
        return nn.functional.normalize(self._rotation)

    @property
    def get_features(self):
        return self._features_dc

    # --- optimization
    def update_learning_rate(self, iteration):
        for group in self.optimizer.param_groups:
            if group["name"] == "xyz":
                group["lr"] = self.xyz_scheduler(iteration)

    def _replace(self, tensors):
        for group in self.optimizer.param_groups:
            if group["name"] in tensors:
                state = self.optimizer.state.get(group["params"][0], None)
                if state is not None:
                    state["exp_avg"] = torch.zeros_like(tensors[group["name"]])
                    state["exp_avg_sq"] = torch.zeros_like(tensors[group["name"]])
                    del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensors[group["name"]].requires_grad_(True))
                if state is not None:
                    self.optimizer.state[group["params"][0]] = state
        self._sync_params()

    def _sync_params(self):
        params = {g["name"]: g["params"][0] for g in self.optimizer.param_groups}
        self._xyz, self._features_dc, self._opacity = params["xyz"], params["f_dc"], params["opacity"]
        self._scaling, self._rotation = params["scaling"], params["rotation"]

    def _apply_mask(self, keep):
        for group in self.optimizer.param_groups:
            state = self.optimizer.state.get(group["params"][0], None)
            if state is not None:
                state["exp_avg"] = state["exp_avg"][keep]
                state["exp_avg_sq"] = state["exp_avg_sq"][keep]
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(group["params"][0][keep].requires_grad_(True))
                self.optimizer.state[group["params"][0]] = state
            else:
                group["params"][0] = nn.Parameter(group["params"][0][keep].requires_grad_(True))
        self._sync_params()

    def _append(self, new):
        for group in self.optimizer.param_groups:
            extension = new[group["name"]]
            state = self.optimizer.state.get(group["params"][0], None)
            if state is not None:
                state["exp_avg"] = torch.cat((state["exp_avg"], torch.zeros_like(extension)), dim=0)
                state["exp_avg_sq"] = torch.cat((state["exp_avg_sq"], torch.zeros_like(extension)), dim=0)
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension), dim=0).requires_grad_(True))
                self.optimizer.state[group["params"][0]] = state
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension), dim=0).requires_grad_(True))
        self._sync_params()

    def prune(self, mask):
        keep = ~mask
        self._apply_mask(keep)
        self.xyz_gradient_accum = self.xyz_gradient_accum[keep]
        self.denom = self.denom[keep]
        self.max_radii2D = self.max_radii2D[keep]
        self.tmp_radii = self.tmp_radii[keep]

    def _postfix(self, new, new_tmp_radii):
        self._append(new)
        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        n = self._xyz.shape[0]
        self.xyz_gradient_accum = torch.zeros((n, 1), device="cuda")
        self.denom = torch.zeros((n, 1), device="cuda")
        self.max_radii2D = torch.zeros(n, device="cuda")

    def densify_and_clone(self, grads, extent):
        selected = (torch.norm(grads, dim=-1) >= DENSIFY_GRAD_THRESHOLD) & \
                   (torch.max(self.get_scaling, dim=1).values <= PERCENT_DENSE * extent)
        self._postfix({"xyz": self._xyz[selected], "f_dc": self._features_dc[selected], "opacity": self._opacity[selected],
                       "scaling": self._scaling[selected], "rotation": self._rotation[selected]}, self.tmp_radii[selected])

    def densify_and_split(self, grads, extent, N=2):
        from utils.general_utils import build_rotation
        n = self._xyz.shape[0]
        padded = torch.zeros(n, device="cuda")
        padded[:grads.shape[0]] = grads.squeeze()
        selected = (padded >= DENSIFY_GRAD_THRESHOLD) & (torch.max(self.get_scaling, dim=1).values > PERCENT_DENSE * extent)
        stds = self.get_scaling[selected].repeat(N, 1)
        samples = torch.normal(mean=torch.zeros((stds.size(0), 3), device="cuda"), std=stds)
        rots = build_rotation(self._rotation[selected]).repeat(N, 1, 1)
        new = {"xyz": torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected].repeat(N, 1),
               "scaling": torch.log(self.get_scaling[selected].repeat(N, 1) / (0.8 * N)),
               "rotation": self._rotation[selected].repeat(N, 1), "f_dc": self._features_dc[selected].repeat(N, 1, 1),
               "opacity": self._opacity[selected].repeat(N, 1)}
        self._postfix(new, self.tmp_radii[selected].repeat(N))
        self.prune(torch.cat((selected, torch.zeros(N * selected.sum(), device="cuda", dtype=bool))))

    def densify_and_prune(self, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        self.tmp_radii = radii
        self.densify_and_clone(grads, extent)
        self.densify_and_split(grads, extent)
        prune = (self.get_opacity < MIN_OPACITY).squeeze()
        if max_screen_size:
            prune = prune | (self.max_radii2D > max_screen_size) | (self.get_scaling.max(dim=1).values > 0.1 * extent)
        self.prune(prune)
        self.tmp_radii = None
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_points, visible):
        self.xyz_gradient_accum[visible] += torch.norm(viewspace_points.grad[visible, :2], dim=-1, keepdim=True)
        self.denom[visible] += 1

    def reset_opacity(self):
        self._replace({"opacity": self.inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))})

    def save_ply(self, path):
        xyz = self._xyz.detach().cpu().numpy()
        attrs = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                 "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
        data = np.concatenate([xyz, np.zeros_like(xyz), self._features_dc.detach().flatten(start_dim=1).cpu().numpy(),
                               self._opacity.detach().cpu().numpy(), self._scaling.detach().cpu().numpy(),
                               self._rotation.detach().cpu().numpy()], axis=1)
        elements = np.empty(xyz.shape[0], dtype=[(a, "f4") for a in attrs])
        elements[:] = list(map(tuple, data))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def _cameras(scene, depth_params):
    from scene.cameras import Camera
    from utils.graphics_utils import focal2fov
    med_scale = float(np.median([p["scale"] for p in depth_params.values()]))
    cams, centers = [], []
    for c in range(scene.num_cams):
        cam = scene.camera(c)
        w2c = cam["w2c"]
        image = Image.open(scene.path("background", f"cam{c}.png"))
        invdepth = cv2.imread(scene.path("static_init", "mono", f"cam{c}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / float(2 ** 16)
        params = dict(depth_params[f"cam{c}"], med_scale=med_scale)
        cams.append(Camera((cam["w"], cam["h"]), colmap_id=c, R=w2c[:3, :3].T, T=w2c[:3, 3],
                           FoVx=focal2fov(cam["K"][0, 0], cam["w"]), FoVy=focal2fov(cam["K"][1, 1], cam["h"]),
                           depth_params=params, image=image, invdepthmap=invdepth, image_name=f"cam{c}", uid=c,
                           mask=np.zeros((cam["h"], cam["w"]), np.float32)))
        centers.append(np.linalg.inv(w2c)[:3, 3])
    centers = np.array(centers)
    extent = 1.1 * np.linalg.norm(centers - centers.mean(axis=0), axis=1).max()     # getNerfppNorm radius
    return cams, extent


PIPE = type("Pipe", (), {"debug": False, "antialiasing": False})()


def train(cams, points, colors, extent, seed=0):
    from fused_ssim import fused_ssim
    from gaussian_renderer import render
    from utils.general_utils import get_expon_lr_func
    from utils.loss_utils import l1_loss

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    gaussians = StaticGaussians(points, colors, extent)
    depth_weight = get_expon_lr_func(DEPTH_WEIGHT_INIT, DEPTH_WEIGHT_FINAL, max_steps=ITERATIONS)
    background = torch.zeros(3, device="cuda")

    stack = []
    bar = progress(range(1, ITERATIONS + 1), desc="static 3DGS training")
    for iteration in bar:
        gaussians.update_learning_rate(iteration)
        if not stack:
            stack = list(range(len(cams)))
        cam = cams[stack.pop(random.randint(0, len(stack) - 1))]

        out = render(cam, gaussians, PIPE, background)
        image, gt = out["render"], cam.original_image.cuda()
        l1 = l1_loss(image, gt)
        loss = (1.0 - LAMBDA_DSSIM) * l1 + LAMBDA_DSSIM * (1.0 - fused_ssim(image.unsqueeze(0), gt.unsqueeze(0)))
        if cam.depth_reliable:
            depth_loss = torch.abs((out["depth"] - cam.invdepthmap.cuda()) * cam.depth_mask.cuda()).mean()
            loss = loss + depth_weight(iteration) * depth_loss * DEPTH_WEIGHT_FACTOR
        loss.backward()

        with torch.no_grad():
            visible = out["visibility_filter"].squeeze(-1) if out["visibility_filter"].ndim > 1 else out["visibility_filter"]
            visible_mask = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")
            visible_mask[visible] = True
            radii = out["radii"]
            if iteration < DENSIFY_UNTIL:
                gaussians.max_radii2D[visible_mask] = torch.max(gaussians.max_radii2D[visible_mask], radii[visible_mask])
                gaussians.add_densification_stats(out["viewspace_points"], visible_mask)
                if iteration > DENSIFY_FROM and iteration % DENSIFY_INTERVAL == 0:
                    gaussians.densify_and_prune(extent, 20 if PRUNE_LARGE and iteration > OPACITY_RESET_INTERVAL else None, radii)
                if iteration % OPACITY_RESET_INTERVAL == 0:
                    gaussians.reset_opacity()
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        if iteration % 100 == 0:
            bar.set_postfix(loss=f"{loss.item():.4f}", gaussians=gaussians.get_xyz.shape[0])

    return gaussians


def train_psnr(gaussians, cams):
    from gaussian_renderer import render
    psnrs = []
    with torch.no_grad():
        for cam in cams:
            mse = torch.mean((render(cam, gaussians, PIPE, torch.zeros(3, device="cuda"))["render"] - cam.original_image.cuda()) ** 2)
            psnrs.append(10 * math.log10(1.0 / mse.item()))
    return psnrs


def run(scene):
    depth_params = json.load(open(scene.path("static_init", "depth_params.json")))
    cams, extent = _cameras(scene, depth_params)
    ply = PlyData.read(scene.path("static_init", "init_points.ply"))["vertex"]
    points = np.stack([ply["x"], ply["y"], ply["z"]], axis=1)
    colors = np.stack([ply["red"], ply["green"], ply["blue"]], axis=1) / 255.0
    gaussians = train(cams, points, colors, extent)
    os.makedirs(scene.path("static_gs"), exist_ok=True)
    gaussians.save_ply(scene.path("static_gs", "static_point_cloud.ply"))
    print(f"      static 3DGS: {gaussians.get_xyz.shape[0]} Gaussians, train PSNR {np.round(train_psnr(gaussians, cams), 2).tolist()}")
    mark_done(scene, "static_gs")
