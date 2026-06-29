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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def _dynamic_scaling_activation(self, x):
        return self.dynamic_scale_min + (self.dynamic_scale_max - self.dynamic_scale_min) * torch.sigmoid(x)

    def _dynamic_scaling_inverse_activation(self, s):
        y = (s - self.dynamic_scale_min) / (self.dynamic_scale_max - self.dynamic_scale_min)
        y = torch.clamp(y, 1e-6, 1.0 - 1e-6)
        return torch.log(y / (1.0 - y))


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._velocity = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._t_mu = torch.empty(0)
        self._t_sigma = torch.empty(0)
        self._is_dynamic = torch.empty(0, dtype=torch.bool)
        self.dynamic_scale_min = None
        self.dynamic_scale_max = None
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._velocity,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._t_mu,
            self._t_sigma,
            self._is_dynamic,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    

    @property
    def get_scaling(self):
        if self.dynamic_scale_min is not None and self._is_dynamic.any():
            static_scale = self.scaling_activation(self._scaling)
            dynamic_scale = self._dynamic_scaling_activation(self._scaling)
            return torch.where(self._is_dynamic.unsqueeze(1), dynamic_scale, static_scale)
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    def get_xyz_at(self, time):
        """
        Get Gaussian positions at specified time(s).
        
        Args:
            time: Either a scalar (float/int/scalar tensor) or tensor of shape [N] where N = number of Gaussians
        
        Returns:
            Tensor of shape [N, 3] with Gaussian positions
        """
        # Handle both scalar and tensor times
        if isinstance(time, (int, float)):
            time_tensor = torch.full((self._xyz.shape[0], 1), time, 
                                      dtype=self._xyz.dtype, device=self._xyz.device)
        elif isinstance(time, torch.Tensor) and time.numel() == 1:
            # Scalar tensor (shape [] or [1])
            time_scalar = time.item()
            time_tensor = torch.full((self._xyz.shape[0], 1), time_scalar, 
                                      dtype=self._xyz.dtype, device=self._xyz.device)
        else:
            # Vector tensor - validate shape
            if time.numel() != self._xyz.shape[0]:
                raise ValueError(f"Time tensor size {time.numel()} must match number of Gaussians {self._xyz.shape[0]}")
            time_tensor = time.view(-1, 1)  # Ensure shape [N, 1]
        
        # Apply velocity only to dynamic Gaussians (using masking to avoid in-place operations)
        velocity_displacement = self._velocity * (time_tensor - self._t_mu) * self._is_dynamic.unsqueeze(1).float()
        return self._xyz + velocity_displacement
    
    @property
    def get_velocity(self):
        return self._velocity
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    def get_opacity_at(self, time):
        """
        Get Gaussian opacities at specified time(s).
        
        Args:
            time: Either a scalar (float/int/scalar tensor) or tensor of shape [N] where N = number of Gaussians
        
        Returns:
            Tensor of shape [N, 1] with Gaussian opacities
        """
        # Handle both scalar and tensor times
        if isinstance(time, (int, float)):
            time_tensor = torch.full((self._opacity.shape[0], 1), time,
                                      dtype=self._opacity.dtype, device=self._opacity.device)
        elif isinstance(time, torch.Tensor) and time.numel() == 1:
            # Scalar tensor (shape [] or [1])
            time_scalar = time.item()
            time_tensor = torch.full((self._opacity.shape[0], 1), time_scalar,
                                      dtype=self._opacity.dtype, device=self._opacity.device)
        else:
            # Vector tensor - validate shape
            if time.numel() != self._opacity.shape[0]:
                raise ValueError(f"Time tensor size {time.numel()} must match number of Gaussians {self._opacity.shape[0]}")
            time_tensor = time.view(-1, 1)  # Ensure shape [N, 1]
        
        # Apply time-dependent opacity only to dynamic Gaussians (using masking to avoid in-place operations)
        lifetime_opacity = torch.exp(-(time_tensor - self._t_mu)**2 / (2 * self._t_sigma**2))
        # For static Gaussians, use 1.0 (constant opacity); for dynamic, use computed lifetime
        lifetime_opacity = torch.where(
            self._is_dynamic.unsqueeze(1), 
            lifetime_opacity, 
            torch.ones_like(lifetime_opacity)
        )
        return self.opacity_activation(self._opacity) * lifetime_opacity
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_t_mu(self):
        return self._t_mu
    
    @property
    def get_t_sigma(self):
        return self._t_sigma
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
   

    def create_from_voxel_flow(self, train_cameras, test_cameras, cameras_extent, subsampling_ratio=0.5):
        self.spatial_lr_scale = cameras_extent
        
        voxel_meta_file = os.path.join(os.path.dirname(train_cameras[0].image_path), "..", "voxel_flow", "metadata.npz")
        voxel_meta = np.load(voxel_meta_file)
        coordinates = voxel_meta["coordinates"]
        voxel_size = float(voxel_meta["voxel_size"])
        voxel_times_us = voxel_meta["times"]
        voxel_times_ms = voxel_times_us * 1e-3
        dt_ms = voxel_meta["dt"].item()

        all_positions = []
        all_velocities = []
        all_times = []

        for voxel_flow_index, voxel_flow_time_ms in enumerate(voxel_times_ms):
            voxel_flow_path = os.path.join(os.path.dirname(train_cameras[0].image_path), "..", "voxel_flow", "voxel_flow", f"{voxel_flow_index:06d}.npz")
            voxel_flow = np.load(voxel_flow_path)["flows"]
            is_valid = ~np.any(np.isnan(voxel_flow), axis=-1)
            if ~np.any(is_valid):
                continue

            positions = coordinates[is_valid]
            velocities = voxel_flow[is_valid]
            time_ms = voxel_flow_time_ms * np.ones(len(positions))

            num_valid = len(positions)
            num_subsampled = int(num_valid * subsampling_ratio)
            if num_subsampled < 1:
                continue

            indices_subsampled = np.random.choice(num_valid, num_subsampled, replace=False)
            positions = positions[indices_subsampled]
            velocities = velocities[indices_subsampled]
            time_ms = time_ms[indices_subsampled]

            all_positions.append(positions)
            all_velocities.append(velocities)
            all_times.extend(time_ms)

        if len(all_positions) == 0:
            raise RuntimeError("No voxel flow data could be loaded from any frame")
        
        # Concatenate all particles
        fused_point_cloud = np.concatenate(all_positions, axis=0)
        velocities_np = np.concatenate(all_velocities, axis=0)
        times_np = np.array(all_times)
        
        print(f"Loaded {len(fused_point_cloud)} particles from {len(all_positions)} voxel flow frames")
        
        # Convert to tensors
        fused_point_cloud = torch.tensor(fused_point_cloud, dtype=torch.float, device="cuda")
        velocities = torch.tensor(velocities_np, dtype=torch.float, device="cuda") / dt_ms
        times = torch.tensor(times_np, dtype=torch.float, device="cuda")
        
        # Add random jitter uniformly in [-voxel_size/2, voxel_size/2] for each dimension
        jitter = (torch.rand_like(fused_point_cloud) - 0.5) * voxel_size
        fused_point_cloud = fused_point_cloud + jitter
        
        # Initialize colors randomly
        fused_color = torch.rand((fused_point_cloud.shape[0], 3), device="cuda") / 255.0
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        
        # Set dynamic scale bounds and initialize scales in bounded sigmoid space
        self.dynamic_scale_min = voxel_size * 0.1
        self.dynamic_scale_max = voxel_size * 3.0
        target_scale = voxel_size * 0.25
        y = (target_scale - self.dynamic_scale_min) / (self.dynamic_scale_max - self.dynamic_scale_min)
        y = float(max(1e-6, min(y, 1.0 - 1e-6)))
        scales = torch.full((fused_point_cloud.shape[0], 3), float(np.log(y / (1.0 - y))), device="cuda")
        
        # Initialize rotations (identity quaternion)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        
        # Initialize opacities
        opacities = self.inverse_opacity_activation(0.4 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        
        # Initialize temporal parameters
        t_mu = times[:, None]
        t_sigma = torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda") * dt_ms * 4
        
        print(f"Number of points at initialization: {fused_point_cloud.shape[0]}")
        
        # Set parameters as neural network parameters
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._velocity = nn.Parameter(velocities.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._t_mu = nn.Parameter(t_mu.requires_grad_(True))
        self._t_sigma = nn.Parameter(t_sigma.requires_grad_(True))
        # All Gaussians from voxel flow are dynamic
        self._is_dynamic = torch.ones((fused_point_cloud.shape[0],), dtype=torch.bool, device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        # Setup exposure mapping
        self.exposure_mapping = {cam_info.image_name: int(cam_info.image_name.split("_")[-1]) for _, cam_info in enumerate(train_cameras + test_cameras)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(train_cameras), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))


    def load_and_concatenate_static_gaussians(self, static_ply_path):
        """
        Load static Gaussians from a PLY file and concatenate them with existing dynamic Gaussians.
        Static Gaussians are not optimized during training and don't have time-dependent properties.
        """
        if not os.path.exists(static_ply_path):
            print(f"Static point cloud not found at {static_ply_path}, skipping static Gaussians")
            return
        
        print(f"Loading static Gaussians from {static_ply_path}")
        plydata = PlyData.read(static_ply_path)
        
        # Load static Gaussian properties
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        # Convert to tensors
        static_xyz = torch.tensor(xyz, dtype=torch.float, device="cuda")
        static_features_dc = torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
        static_features_rest = torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
        static_opacity = torch.tensor(opacities, dtype=torch.float, device="cuda")
        static_scaling = torch.tensor(scales, dtype=torch.float, device="cuda")
        static_rotation = torch.tensor(rots, dtype=torch.float, device="cuda")
        
        # Initialize static Gaussian properties that are not used but need to exist
        # (velocity, t_mu, t_sigma are not used for static Gaussians but we keep them for consistency)
        static_velocity = torch.zeros_like(static_xyz)
        static_t_mu = torch.zeros((static_xyz.shape[0], 1), dtype=torch.float, device="cuda")
        static_t_sigma = torch.ones((static_xyz.shape[0], 1), dtype=torch.float, device="cuda")
        static_is_dynamic = torch.zeros((static_xyz.shape[0],), dtype=torch.bool, device="cuda")
        
        # Concatenate with existing dynamic Gaussians
        self._xyz = nn.Parameter(torch.cat([self._xyz, static_xyz], dim=0).requires_grad_(True))
        self._velocity = nn.Parameter(torch.cat([self._velocity, static_velocity], dim=0).requires_grad_(True))
        self._features_dc = nn.Parameter(torch.cat([self._features_dc, static_features_dc], dim=0).requires_grad_(True))
        self._features_rest = nn.Parameter(torch.cat([self._features_rest, static_features_rest], dim=0).requires_grad_(True))
        self._scaling = nn.Parameter(torch.cat([self._scaling, static_scaling], dim=0).requires_grad_(True))
        self._rotation = nn.Parameter(torch.cat([self._rotation, static_rotation], dim=0).requires_grad_(True))
        self._opacity = nn.Parameter(torch.cat([self._opacity, static_opacity], dim=0).requires_grad_(True))
        self._t_mu = nn.Parameter(torch.cat([self._t_mu, static_t_mu], dim=0).requires_grad_(True))
        self._t_sigma = nn.Parameter(torch.cat([self._t_sigma, static_t_sigma], dim=0).requires_grad_(True))
        self._is_dynamic = torch.cat([self._is_dynamic, static_is_dynamic], dim=0)
        
        # Update auxiliary tensors
        static_count = static_xyz.shape[0]
        self.max_radii2D = torch.cat([self.max_radii2D, torch.zeros(static_count, device="cuda")], dim=0)
        
        print(f"Loaded {static_count} static Gaussians. Total Gaussians: {self._xyz.shape[0]}")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._velocity], 'lr': training_args.position_lr_init * self.spatial_lr_scale * 1e-2, "name": "velocity"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._t_mu], 'lr': training_args.t_mu_lr * 1e1, "name": "t_mu"},
            {'params': [self._t_sigma], 'lr': training_args.t_sigma_lr * 1e1, "name": "t_sigma"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        l.append('t_mu')
        l.append('t_sigma')
        l.append('is_dynamic')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        t_mu = self._t_mu.detach().cpu().numpy()
        t_sigma = self._t_sigma.detach().cpu().numpy()
        is_dynamic = self._is_dynamic.detach().cpu().numpy().astype(np.float32)[..., np.newaxis]

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, t_mu, t_sigma, is_dynamic), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        if self.dynamic_scale_min is not None:
            bounds_path = os.path.join(os.path.dirname(path), "scale_bounds.json")
            with open(bounds_path, "w") as f:
                json.dump({"dynamic_scale_min": self.dynamic_scale_min, "dynamic_scale_max": self.dynamic_scale_max}, f)

    def reset_opacity(self):
        # Only reset opacity for dynamic gaussians, keep static gaussians unchanged
        opacities_new = self._opacity.clone()
        dynamic_mask = self._is_dynamic
        
        # Reset only dynamic gaussian opacities
        dynamic_opacities_reset = self.inverse_opacity_activation(
            torch.min(self.get_opacity[dynamic_mask], torch.ones_like(self.get_opacity[dynamic_mask])*0.01)
        )
        opacities_new[dynamic_mask] = dynamic_opacities_reset
        
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # Load t_μ and t_σ parameters if they exist, otherwise use defaults
        if "t_mu" in plydata.elements[0].properties:
            t_mu = np.asarray(plydata.elements[0]["t_mu"])[..., np.newaxis]
        else:
            t_mu = np.zeros((xyz.shape[0], 1))
        
        if "t_sigma" in plydata.elements[0].properties:
            t_sigma = np.asarray(plydata.elements[0]["t_sigma"])[..., np.newaxis]
        else:
            t_sigma = np.ones((xyz.shape[0], 1))
        
        # Load is_dynamic flag if it exists, otherwise assume all are dynamic
        if "is_dynamic" in plydata.elements[0].properties:
            is_dynamic = np.asarray(plydata.elements[0]["is_dynamic"]).astype(bool)
        else:
            is_dynamic = np.ones((xyz.shape[0],), dtype=bool)

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        # Initialize velocity to zeros (will be loaded from checkpoint if available)
        self._velocity = nn.Parameter(torch.zeros_like(torch.tensor(xyz, dtype=torch.float, device="cuda")).requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._t_mu = nn.Parameter(torch.tensor(t_mu, dtype=torch.float, device="cuda").requires_grad_(True))
        self._t_sigma = nn.Parameter(torch.tensor(t_sigma, dtype=torch.float, device="cuda").requires_grad_(True))
        self._is_dynamic = torch.tensor(is_dynamic, dtype=torch.bool, device="cuda")

        bounds_path = os.path.join(os.path.dirname(path), "scale_bounds.json")
        if os.path.exists(bounds_path):
            with open(bounds_path, "r") as f:
                bounds = json.load(f)
            self.dynamic_scale_min = bounds["dynamic_scale_min"]
            self.dynamic_scale_max = bounds["dynamic_scale_max"]

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._velocity = optimizable_tensors["velocity"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._t_mu = optimizable_tensors["t_mu"]
        self._t_sigma = optimizable_tensors["t_sigma"]
        self._is_dynamic = self._is_dynamic[valid_points_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_velocity, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_t_mu, new_t_sigma, new_tmp_radii, new_is_dynamic):
        d = {"xyz": new_xyz,
        "velocity": new_velocity,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "t_mu": new_t_mu,
        "t_sigma": new_t_sigma}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._velocity = optimizable_tensors["velocity"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._t_mu = optimizable_tensors["t_mu"]
        self._t_sigma = optimizable_tensors["t_sigma"]
        self._is_dynamic = torch.cat((self._is_dynamic, new_is_dynamic))

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        # Only densify dynamic Gaussians
        selected_pts_mask = torch.logical_and(selected_pts_mask, self._is_dynamic)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_velocity = self.get_velocity[selected_pts_mask].repeat(N, 1)
        split_scales = self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        if self.dynamic_scale_min is not None:
            new_scaling = self._dynamic_scaling_inverse_activation(split_scales)
        else:
            new_scaling = self.scaling_inverse_activation(split_scales)
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_t_mu = self._t_mu[selected_pts_mask].repeat(N,1)
        new_t_sigma = self._t_sigma[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)
        new_is_dynamic = self._is_dynamic[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_velocity, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_t_mu, new_t_sigma, new_tmp_radii, new_is_dynamic)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # Only densify dynamic Gaussians
        selected_pts_mask = torch.logical_and(selected_pts_mask, self._is_dynamic)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_velocity = self._velocity[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_t_mu = self._t_mu[selected_pts_mask]
        new_t_sigma = self._t_sigma[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]
        new_is_dynamic = self._is_dynamic[selected_pts_mask]

        self.densification_postfix(new_xyz, new_velocity, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_t_mu, new_t_sigma, new_tmp_radii, new_is_dynamic)
        

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        # Only prune dynamic Gaussians
        prune_mask = torch.logical_and(prune_mask, self._is_dynamic)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
