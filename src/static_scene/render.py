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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import matplotlib.pyplot as plt
import numpy as np
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # Draw opacity histogram of the gaussians
        opacities = gaussians.get_opacity.detach().cpu().numpy().flatten()
        
        plt.figure(figsize=(10, 6))
        plt.hist(opacities, bins=100, edgecolor='black', alpha=0.7)
        plt.xlabel('Opacity Value')
        plt.ylabel('Frequency')
        plt.title(f'Opacity Histogram of Gaussians (Iteration {scene.loaded_iter})')
        plt.grid(True, alpha=0.3)
        
        # Add statistics to the plot
        mean_opacity = np.mean(opacities)
        median_opacity = np.median(opacities)
        plt.axvline(mean_opacity, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_opacity:.3f}')
        plt.axvline(median_opacity, color='g', linestyle='--', linewidth=2, label=f'Median: {median_opacity:.3f}')
        plt.legend()
        
        # Save the histogram
        histogram_path = os.path.join(dataset.model_path, f"opacity_histogram_iter_{scene.loaded_iter}.png")
        plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Opacity histogram saved to: {histogram_path}")
        print(f"Total Gaussians: {len(opacities)}")
        print(f"Mean Opacity: {mean_opacity:.4f}")
        print(f"Median Opacity: {median_opacity:.4f}")
        print(f"Min Opacity: {np.min(opacities):.4f}")
        print(f"Max Opacity: {np.max(opacities):.4f}")
        
        # Remove Gaussians with opacity below 0.2
        opacity_threshold = 0.00
        low_opacity_mask = (gaussians.get_opacity < opacity_threshold).squeeze()
        # Never prune background Gaussians
        low_opacity_mask = low_opacity_mask & ~gaussians._is_background
        valid_points_mask = ~low_opacity_mask
        num_to_remove = low_opacity_mask.sum().item()
        print(f"\nRemoving {num_to_remove} Gaussians with opacity < {opacity_threshold}")
        
        # Manually prune Gaussians (without optimizer since we're in inference mode)
        gaussians._xyz = gaussians._xyz[valid_points_mask]
        gaussians._features_dc = gaussians._features_dc[valid_points_mask]
        gaussians._features_rest = gaussians._features_rest[valid_points_mask]
        gaussians._opacity = gaussians._opacity[valid_points_mask]
        gaussians._scaling = gaussians._scaling[valid_points_mask]
        gaussians._rotation = gaussians._rotation[valid_points_mask]
        gaussians._is_background = gaussians._is_background[valid_points_mask]
        
        print(f"Remaining Gaussians: {gaussians.get_xyz.shape[0]}")

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)