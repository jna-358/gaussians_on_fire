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

def psnr(img1, img2, mask=None):
    if mask is None:
        mask = torch.ones(img1.shape[1:])
    mask_rgb = (mask > 0).unsqueeze(0).repeat(3, 1, 1)
    mse = ((img1 - img2) ** 2)[mask_rgb].mean()
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
