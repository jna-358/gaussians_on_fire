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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
from torchvision.utils import save_image
import cv2
from PIL import Image, ImageDraw, ImageFont
import json

from lpipsPyTorch import lpips
from utils.image_utils import psnr

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
    print("Sparse Adam available")
except:
    SPARSE_ADAM_AVAILABLE = False
    print("Sparse Adam NOT available")

def save_image_uint16(image, path):
    if isinstance(image, torch.Tensor):
        image_np = image.detach().cpu().numpy()
    else:
        image_np = np.array(image)

    if image_np.ndim == 3:
        if image_np.shape[0] in [1, 3]:
            image_np = image_np[0]
        elif image_np.shape[2] in [1, 3]:
            image_np = image_np[:, :, 0]

    image_uint16 = (image_np * 65535).astype(np.uint16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image_uint16)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, debug_from, live_view, mask_loss=0.0):
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)

    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    times = [cam.time for cam in viewpoint_stack]
    min_time = min(times)
    max_time = max(times)
    custom_time_steps = torch.linspace(min_time, max_time, 100)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    render_iter = 0

    live_views = sorted([cam.copy() for cam in scene.getTestCameras() if cam.image_name in ["0000_0", "0000_1", "0000_2"]], key=lambda x: x.image_name)

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image, mask=None, is_train=True)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        if mask_loss > 0 and viewpoint_cam.mask is not None:
            black_bg = torch.zeros(3, device="cuda")
            render_pkg_dynamic = render(viewpoint_cam, gaussians, pipe, black_bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, dynamic_only=True)
            dynamic_image = render_pkg_dynamic["render"]
            flame_mask = viewpoint_cam.mask.cuda()
            if flame_mask.ndim == 2:
                flame_mask = flame_mask.unsqueeze(0)
            Ll1mask = (dynamic_image.mean(0, keepdim=True) * (1.0 - flame_mask)).mean()
            loss += mask_loss * Ll1mask

            radii_dynamic = render_pkg_dynamic["radii"]
            radii = torch.max(radii, radii_dynamic)
            visibility_filter = (radii > 0).nonzero()

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            if live_view and iteration % 5 == 0:
                padding_width = 25
                renders = []
                for i_view, view in enumerate(live_views):
                    view.time = min_time + (max_time - min_time) * ((iteration // 5) % 200) / 200
                    render_pkg = render(view, gaussians, pipe, bg, use_trained_exp=False, separate_sh=SPARSE_ADAM_AVAILABLE)
                    render_view = render_pkg["render"].permute(1, 2, 0).contiguous().cpu().numpy()
                    render_view = (render_view * 255).astype(np.uint8)
                    renders.append(render_view)
                    if i_view < len(live_views) - 1:
                        renders.append(np.zeros((render_view.shape[0], padding_width, 3), dtype=np.uint8))
                render_all = np.concatenate(renders, axis=1)
                render_all = cv2.resize(render_all, None, fx=0.5, fy=0.5)
                pil_img = Image.fromarray(render_all)
                draw = ImageDraw.Draw(pil_img)
                font = ImageFont.load_default(size=20)
                text, x, y = "Press Q or ESC to exit", 10, pil_img.height - 30
                draw.text((x + 2, y + 2), text, fill=(0, 0, 0), font=font)
                draw.text((x, y), text, fill=(255, 255, 255), font=font)
                render_all = np.array(pil_img)[..., ::-1]
                cv2.imshow("Live View", render_all)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    sys.exit(0)

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                dynamic_mask = gaussians._is_dynamic
                t_sigma_mean = gaussians.get_t_sigma[dynamic_mask].mean().item()
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Num": f"{gaussians.get_xyz.shape[0]}", "t_sigma": f"{t_sigma_mean:.2f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

                dynamic_mask = gaussians._is_dynamic[visibility_filter.squeeze()]
                dynamic_visibility_filter = visibility_filter[dynamic_mask]
                gaussians.add_densification_stats(viewspace_point_tensor, dynamic_visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)

                if opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    print("Reset opacity")

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    if live_view:
        cv2.destroyAllWindows()

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                metrics = {
                    'basic': {'psnr': [], 'l1': [], 'l2': [], 'ssim': [], 'lpips': []},
                    'no_sync': {'psnr': [], 'l1': [], 'l2': [], 'ssim': []},
                    'flame': {'psnr': [], 'l1': [], 'l2': [], 'ssim': []},
                }
                for idx, viewpoint in enumerate(tqdm(config['cameras'], desc=f"Evaluating {config['name']}", leave=False)):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    depth = render_pkg["depth"]
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image, mask=viewpoint.mask, is_train=False).mean().double()
                    psnr_test += psnr(image, gt_image, mask=viewpoint.mask).mean().double()

                    metrics['basic']['psnr'].append(psnr(image, gt_image, mask=None).mean().double().item())
                    metrics['basic']['l1'].append(l1_loss(image, gt_image, mask=None, is_train=False).mean().double().item())
                    metrics['basic']['l2'].append(l2_loss(image, gt_image, mask=None, is_train=False).mean().double().item())
                    metrics['basic']['ssim'].append(ssim(image, gt_image, mask=None).mean().double().item())
                    metrics['basic']['lpips'].append(lpips(image, gt_image).mean().double().item())

                    metrics['no_sync']['psnr'].append(psnr(image, gt_image, mask=(1.0 - viewpoint.sync_mask)).mean().double().item())
                    metrics['no_sync']['l1'].append(l1_loss(image, gt_image, mask=(1.0 - viewpoint.sync_mask), is_train=False).mean().double().item())
                    metrics['no_sync']['l2'].append(l2_loss(image, gt_image, mask=(1.0 - viewpoint.sync_mask), is_train=False).mean().double().item())
                    metrics['no_sync']['ssim'].append(ssim(image, gt_image, mask=(1.0 - viewpoint.sync_mask)).mean().double().item())

                    metrics['flame']['psnr'].append(psnr(image, gt_image, mask=viewpoint.mask).mean().double().item())
                    metrics['flame']['l1'].append(l1_loss(image, gt_image, mask=viewpoint.mask, is_train=False).mean().double().item())
                    metrics['flame']['l2'].append(l2_loss(image, gt_image, mask=viewpoint.mask, is_train=False).mean().double().item())
                    metrics['flame']['ssim'].append(ssim(image, gt_image, mask=viewpoint.mask).mean().double().item())

                    for image_category in ['renders', 'depth', 'gt']:
                        os.makedirs(os.path.join(tb_writer.log_dir, "renders", "ours_{}".format(iteration), config['name'], image_category), exist_ok=True)
                    depth = (depth - depth.min()) / (depth.max() - depth.min())

                    save_image(image, os.path.join(tb_writer.log_dir, "renders", "ours_{}".format(iteration), config['name'], "renders", f"{viewpoint.image_name}.png"))
                    save_image(depth, os.path.join(tb_writer.log_dir, "renders", "ours_{}".format(iteration), config['name'], "depth", f"{viewpoint.image_name}.png"))
                    save_image(gt_image, os.path.join(tb_writer.log_dir, "renders", "ours_{}".format(iteration), config['name'], "gt", f"{viewpoint.image_name}.png"))
                    save_image_uint16(depth, os.path.join(tb_writer.log_dir, "renders", "ours_{}".format(iteration), config['name'], "depth_raw", f"{viewpoint.image_name}.png"))

                for metric_type in metrics:
                    for metric_name in metrics[metric_type]:
                        metrics[metric_type][metric_name] = np.nanmean(metrics[metric_type][metric_name])

                output_dir = tb_writer.log_dir
                os.makedirs(os.path.join(output_dir, "metrics", config['name']), exist_ok=True)
                with open(os.path.join(output_dir, "metrics", config['name'], f"{iteration}.json"), "w") as f:
                    json.dump(metrics, f, indent=4)

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                with open(os.path.join(scene.model_path, "eval_results.txt"), "a") as f:
                    f.write(f"ITER {iteration}: {config['name']} L1 {l1_test} PSNR {psnr_test}\n")
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[1_000, 7_000, 30_000])
    parser.add_argument("--live_view", action='store_true', default=False, help='Enable live view of the training process')
    parser.add_argument("--mask_loss", type=float, default=0.0, help='Weight for mask containment loss penalizing dynamic Gaussians outside the flame mask')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.debug_from, args.live_view, args.mask_loss)

    print("\nTraining complete.")
