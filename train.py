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

import os
import glob
import torch
from random import randint
from utils.loss_utils import l1_loss
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams


from fused_ssim import fused_ssim
from evaluation import evaluate, write_metrics


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, mask_loss, init_t_sigma_factor, run_args):

    first_iter = 0
    prepare_output(dataset, run_args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, init_t_sigma_factor=init_t_sigma_factor)

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        rand_idx = randint(0, len(viewpoint_stack) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image, mask=None, is_train=True)
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth

        # Mask containment loss: penalize dynamic alpha outside the flame mask
        if mask_loss > 0:
            render_pkg_dynamic = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, dynamic_only=True)
            dynamic_alpha = render_pkg_dynamic["alpha"]  # [1, H, W]
            flame_mask = viewpoint_cam.mask.cuda()
            if flame_mask.ndim == 2:
                flame_mask = flame_mask.unsqueeze(0)
            # One-sided: missing flame (mask 1, alpha 0) is not penalized
            Ll1mask = (dynamic_alpha * (1.0 - flame_mask)).mean()
            loss += mask_loss * Ll1mask

            # Densification sees Gaussians visible in either render
            radii_dynamic = render_pkg_dynamic["radii"]
            visible_combined = (radii > 0) | (radii_dynamic > 0)
            visibility_filter = visible_combined.nonzero()
            radii = torch.max(radii, radii_dynamic)

        loss.backward()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                dynamic_mask = gaussians._is_dynamic
                t_sigma_mean = gaussians.get_t_sigma[dynamic_mask].mean().item()
                postfix = {"Loss": f"{ema_loss_for_log:.{7}f}", "Num": f"{gaussians.get_xyz.shape[0]}", "t_sigma": f"{t_sigma_mean:.2f}"}
                if mask_loss > 0:
                    postfix["mask"] = f"{Ll1mask.item():.4f}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log
            training_report(iteration, testing_iterations, scene, pipe, background, dataset.train_test_exp)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

                # Densification statistics are only accumulated for dynamic Gaussians
                dynamic_mask = gaussians._is_dynamic[visibility_filter.squeeze()]
                dynamic_visibility_filter = visibility_filter[dynamic_mask]
                gaussians.add_densification_stats(viewspace_point_tensor, dynamic_visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # Save after the optimizer step, so ply and checkpoint match.
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output(args, run_args):
    """Create the output folder and write cfg_args with all parsed arguments
    (read back by get_combined_args)."""
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("data/output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**{**vars(run_args), **vars(args)})))

    # Start fresh metric tables.
    for stale_log in glob.glob(os.path.join(args.model_path, "metrics", "*.log")):
        os.remove(stale_log)

def _flatten_metrics(metrics):
    flat = {}
    for group, group_metrics in metrics.items():
        for name, value in group_metrics.items():
            flat[f"{group}.{name}"] = value
    return flat


def _append_metrics_log(log_path, iteration, metrics):
    """Append one row (iteration + all metrics) to a fixed-width table, with a header if new."""
    flat = _flatten_metrics(metrics)
    columns = ["iteration", *flat.keys()]
    col_widths = [max(len(c), 12) for c in columns]
    sep = "  "
    write_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        if write_header:
            f.write(sep.join(c.rjust(w) for c, w in zip(columns, col_widths)) + "\n")
        values = [str(iteration)] + [f"{flat[k]:.6f}" for k in flat]
        f.write(sep.join(v.rjust(w) for v, w in zip(values, col_widths)) + "\n")



def training_report(iteration, testing_iterations, scene : Scene, pipe, background, train_test_exp):
    """At test iterations, evaluate the test views and a fixed sample of 5 train views."""
    if iteration not in testing_iterations:
        return
    torch.cuda.empty_cache()
    train_cams = scene.getTrainCameras()
    configs = {
        "test": scene.getTestCameras(),
        "train": [train_cams[idx % len(train_cams)] for idx in range(5, 30, 5)],
    }
    for split, cameras in configs.items():
        metrics = evaluate(cameras, scene.gaussians, pipe, background, train_test_exp,
                           scene.source_path, desc=f"Evaluating {split}")
        write_metrics(scene.model_path, split, iteration, metrics)
        _append_metrics_log(os.path.join(scene.model_path, "metrics", f"{split}.log"), iteration, metrics)
        print(f"\n[ITER {iteration}] {split}: flame PSNR {metrics['flame']['psnr']:.3f}, "
              f"no_sync PSNR {metrics['no_sync']['psnr']:.3f}, LPIPS {metrics['basic']['lpips']:.4f}")
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=None, help="Default: every 1000 iterations and the last one.")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[1_000, 7_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--mask_loss", type=float, default=0.1, help="Weight for the mask containment loss that penalizes dynamic Gaussians appearing outside the flame mask. 0 disables the loss.")
    parser.add_argument("--init_t_sigma_factor", type=float, default=4.0, help="Multiplier on dt_ms for the initial t_sigma in voxel-flow init.")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.test_iterations is None:
        args.test_iterations = [1_000 * i for i in range(1, (args.iterations - 1) // 1000 + 1)]
    args.test_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.mask_loss, args.init_t_sigma_factor, args)

    # All done
    print("\nTraining complete.")
