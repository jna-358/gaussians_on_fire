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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False, time=0.0,
                 frame=0, mask=None, sync_mask=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.time = time
        self.frame = frame
        self.sync_mask = sync_mask
        
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        self.alpha_mask = None
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else: 
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.mask = torch.from_numpy(mask).float().to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def copy(self):
        """
        Create a copy of this camera with independent mutable attributes.
        This allows modifying properties like 'time' without affecting the original.
        Tensor data is shared to avoid memory duplication.
        """
        cam_copy = object.__new__(Camera)
        nn.Module.__init__(cam_copy)

        # Copy all attributes
        cam_copy.uid = self.uid
        cam_copy.colmap_id = self.colmap_id
        cam_copy.R = self.R.copy() if isinstance(self.R, np.ndarray) else self.R
        cam_copy.T = self.T.copy() if isinstance(self.T, np.ndarray) else self.T
        cam_copy.FoVx = self.FoVx
        cam_copy.FoVy = self.FoVy
        cam_copy.image_name = self.image_name
        cam_copy.time = self.time
        cam_copy.frame = self.frame
        cam_copy.sync_mask = self.sync_mask
        cam_copy.data_device = self.data_device
        cam_copy.alpha_mask = self.alpha_mask
        cam_copy.original_image = self.original_image
        cam_copy.mask = self.mask
        cam_copy.image_width = self.image_width
        cam_copy.image_height = self.image_height
        cam_copy.invdepthmap = self.invdepthmap
        cam_copy.depth_reliable = self.depth_reliable
        if hasattr(self, 'depth_mask'):
            cam_copy.depth_mask = self.depth_mask
        cam_copy.zfar = self.zfar
        cam_copy.znear = self.znear
        cam_copy.trans = self.trans.copy() if isinstance(self.trans, np.ndarray) else self.trans
        cam_copy.scale = self.scale
        cam_copy.world_view_transform = self.world_view_transform
        cam_copy.projection_matrix = self.projection_matrix
        cam_copy.full_proj_transform = self.full_proj_transform
        cam_copy.camera_center = self.camera_center

        return cam_copy

