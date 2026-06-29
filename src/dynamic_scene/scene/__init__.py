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
import random
import json
import tqdm
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], frames=None, flow_data=None, image_names=None, sync_offset=0):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.flow_data = flow_data

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        assert os.path.exists(os.path.join(args.source_path, "transforms_train.json")), \
            f"Could not find transforms_train.json in {args.source_path}"
        print("Found transforms_train.json file, assuming Blender data set!")
        scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval, frames=frames, image_names=image_names, sync_offset=sync_offset)

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"), 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(tqdm.tqdm(camlist, desc="Writing Cameras")):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False, scene_info.sync_mask_info)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True, scene_info.sync_mask_info)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                  "point_cloud",
                                                  "iteration_" + str(self.loaded_iter),
                                                  "point_cloud.ply"), args.train_test_exp)
        else:
            self.gaussians.create_from_voxel_flow(scene_info.train_cameras, scene_info.test_cameras, self.cameras_extent)
            static_ply_path = os.path.join(args.source_path, "static_point_cloud.ply")
            self.gaussians.load_and_concatenate_static_gaussians(static_ply_path)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }
        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0, time=None, frames=None):
        if time is not None:
            eps = 1e-6
            return [cam for cam in self.train_cameras[scale] if abs(cam.time - time) < eps]
        elif frames is not None:
            return [cam for cam in self.train_cameras[scale] if cam.frame in frames]
        else:
            return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0, time=None, frames=None):
        if time is not None:
            eps = 1e-6
            return [cam for cam in self.test_cameras[scale] if abs(cam.time - time) < eps]
        elif frames is not None:
            return [cam for cam in self.test_cameras[scale] if cam.frame in frames]
        else:
            return self.test_cameras[scale]
