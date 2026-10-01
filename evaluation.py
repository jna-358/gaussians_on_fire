# Copyright (c) 2026 Jakob Nazarenus
# SPDX-License-Identifier: MIT
#
# Evaluation of a Gaussians on Fire reconstruction.
#
# All metrics are computed from the in-memory float renders (no PNG round
# trip). train.py calls `evaluate` at every test iteration; running this file
# evaluates a saved model:
#
#     python evaluation.py -m <model_dir> [--iteration N]
#
# Metric groups (each averaged over the views):
#   basic   - whole frame (psnr, l1, l2, ssim, lpips, rmse_depth). Includes the
#             LED sync board; for debugging/sanity checks only.
#   no_sync - everything except the LED sync board (R channel of the camera's
#             frame-0 mask). This is the paper's "no-sync" metric.
#   flame   - strict flame mask (G channel of the frame's mask) for image
#             metrics; the same mask eroded by 9 px for depth.
#
# rmse_depth: the rendered inverse depth is min-max normalised over pixels that
# have depth (others 0), fitted to the normalised monocular inverse depth by
# least squares (scale + offset), and the RMSE is taken over the region
# (flame: aligned depth clipped to [0, 1], weighted by the eroded mask).
#

import json
import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
from tqdm import tqdm

from gaussian_renderer import render
from lpipsPyTorch.modules.lpips import LPIPS
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, l2_loss, ssim

FLAME_DEPTH_MASK_EROSION = 9  # px; the flame masks are imprecise at the boundary

_lpips_model = None


def _lpips(image, gt):
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = LPIPS("alex", "0.1").to(image.device)
    return _lpips_model(image, gt)


def _mono_depth_dir(source_path):
    for name in ("mono_depth", "depths"):
        path = os.path.join(source_path, name)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f"Neither 'mono_depth/' nor 'depths/' found under {source_path}")


def _depth_rmse(invdepth, source_path, image_name, mask_path):
    """rmse_depth for the basic / no_sync / flame regions of one view."""
    depth = invdepth.squeeze().detach().cpu().numpy().astype(np.float32)
    valid = depth > 0
    normalized = np.zeros_like(depth)
    if valid.any():
        d_min, d_max = depth[valid].min(), depth[valid].max()
        normalized[valid] = (depth[valid] - d_min) / (d_max - d_min if d_max > d_min else 1.0)
    depth = normalized / normalized.max() if normalized.max() > 0 else normalized

    mono = cv2.imread(os.path.join(_mono_depth_dir(source_path), image_name + ".png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
    mono = mono / mono.max()

    masks = cv2.imread(mask_path)  # BGR
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FLAME_DEPTH_MASK_EROSION, FLAME_DEPTH_MASK_EROSION))
    flame = cv2.erode(masks[..., 1], kernel).astype(np.float32) / 255.0   # G = strict flame
    outside_sync = masks[..., 2] == 0                                     # R = sync board

    A = np.stack([depth.ravel(), np.ones(depth.size, dtype=depth.dtype)], axis=1)
    scale, offset = np.linalg.lstsq(A, mono.ravel(), rcond=None)[0]
    aligned = scale * depth + offset
    aligned_clipped = np.clip(aligned, 0, 1)

    return {
        "basic": np.sqrt(((aligned - mono) ** 2).mean()),
        "no_sync": np.sqrt(((aligned[outside_sync] - mono[outside_sync]) ** 2).mean()),
        "flame": (np.sqrt(((flame * (aligned_clipped - mono)) ** 2).sum() / flame.sum())
                  if flame.sum() > 0 else float("nan")),
    }


@torch.no_grad()
def evaluate(cameras, gaussians, pipe, background, use_trained_exp, source_path, desc="Evaluating"):
    """Render every camera and return {group: {metric: mean over views}}."""
    values = {
        "basic": {"psnr": [], "l1": [], "l2": [], "ssim": [], "lpips": [], "rmse_depth": []},
        "no_sync": {"psnr": [], "l1": [], "l2": [], "ssim": [], "rmse_depth": []},
        "flame": {"psnr": [], "l1": [], "l2": [], "ssim": [], "rmse_depth": []},
    }
    for view in tqdm(cameras, desc=desc, leave=False):
        out = render(view, gaussians, pipe, background, use_trained_exp=use_trained_exp)
        image = torch.clamp(out["render"], 0.0, 1.0)
        gt = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)

        # Strict flame mask (G) for the image metrics; view.mask (B, dilated) stays the training mask.
        strict_flame = torch.from_numpy(cv2.imread(view.mask_path)[..., 1] > 0).float().to(view.mask.device)
        regions = {"basic": None, "no_sync": 1.0 - view.sync_mask, "flame": strict_flame}
        for group, mask in regions.items():
            values[group]["psnr"].append(psnr(image, gt, mask=mask).mean().double().item())
            values[group]["l1"].append(l1_loss(image, gt, mask=mask, is_train=False).mean().double().item())
            values[group]["l2"].append(l2_loss(image, gt, mask=mask, is_train=False).mean().double().item())
            values[group]["ssim"].append(ssim(image, gt, mask=mask).mean().double().item())
        values["basic"]["lpips"].append(_lpips(image, gt).mean().double().item())

        for group, rmse in _depth_rmse(out["depth"], source_path, view.image_name, view.mask_path).items():
            values[group]["rmse_depth"].append(rmse)

    return {group: {name: float(np.nanmean(v)) for name, v in metrics.items()}
            for group, metrics in values.items()}


def write_metrics(model_path, split, iteration, metrics, suffix=""):
    """Write metrics/<split>/<iteration><suffix>.json and return its path."""
    out_dir = os.path.join(model_path, "metrics", split)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{iteration}{suffix}.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    return path


def load_trained_model(dataset, iteration=-1):
    """Load point_cloud/iteration_<iteration>/point_cloud.ply (-1: highest saved)."""
    from scene import Scene, GaussianModel
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    # Exposures come from exposure.json (read by load_ply); identity if it is missing.
    if getattr(gaussians, "pretrained_exposures", None) is None:
        cams = scene.getTrainCameras() + scene.getTestCameras()
        gaussians.exposure_mapping = {cam.image_name: idx for idx, cam in enumerate(cams)}
        gaussians._exposure = torch.nn.Parameter(torch.eye(3, 4, device="cuda")[None].repeat(len(cams), 1, 1))
    return scene, gaussians


if __name__ == "__main__":
    from arguments import ModelParams, PipelineParams, get_combined_args
    from utils.general_utils import safe_state

    parser = ArgumentParser(description="Evaluate a trained Gaussians on Fire model")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int, help="Saved iteration to evaluate, -1 for the highest")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)

    dataset, pipe = model.extract(args), pipeline.extract(args)
    scene, gaussians = load_trained_model(dataset, args.iteration)
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    metrics = evaluate(scene.getTestCameras(), gaussians, pipe, background, dataset.train_test_exp, dataset.source_path)
    path = write_metrics(dataset.model_path, "test", scene.loaded_iter, metrics, suffix=".from_ply")
    print(json.dumps(metrics, indent=4))
    print(f"Wrote {path}")
