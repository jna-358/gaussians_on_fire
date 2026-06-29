import torch
import cv2
from tqdm import tqdm
import os
import glob
import numpy as np

def homogenize(array):
    return np.concatenate([array, np.ones((*array.shape[:-1], 1))], axis=-1)

def uint8_to_float(input_array, min_val, max_val):
    input_array = input_array.astype(np.float32) / 255.0
    input_array = input_array * (max_val - min_val) + min_val
    return input_array

def load_colmap_intrinsics(cameras_file, camera_id):
    with open(cameras_file, "r") as f:
        lines = f.readlines()

    found = False
    for line in lines:
        if line.startswith("#"):
            continue
        line = line.strip().split(" ")
        if line[0] == str(camera_id):
            model = line[1]
            width = int(line[2])
            height = int(line[3])
            fx = float(line[4])
            fy = float(line[5])
            cx = float(line[6])
            cy = float(line[7])
            found = True
            break

    assert found, f"Camera {camera_id} not found in {cameras_file}"
    assert model == "PINHOLE", f"Camera {camera_id} is not a PINHOLE camera"

    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    
    return {
        "camera_matrix": camera_matrix,
        "width": width,
        "height": height,
    }

def load_colmap_pose(images_file, camera_id):
    with open(images_file, "r") as f:
        lines = f.readlines()
    
    found = False
    for line in lines:
        if line.startswith("#") or line.strip() == "":
            continue
        line = line.strip().split(" ")
        if line[8] == str(camera_id):
            # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            qw = float(line[1])
            qx = float(line[2])
            qy = float(line[3])
            qz = float(line[4])
            tx = float(line[5])
            ty = float(line[6])
            tz = float(line[7])
            found = True
            break
    assert found, f"Camera {camera_id} not found in {images_file}"

    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ], dtype=np.float32)
    
    t = np.array([tx, ty, tz], dtype=np.float32)
    
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R
    w2c[:3, 3] = t

    return w2c

def triangulate_point(points_2d, intrinsics, poses):
    num_views = len(points_2d)
    A = np.zeros((2 * num_views, 4), dtype=np.float64)
    
    for i in range(num_views):
        # Get projection matrix P = K @ [R | t]
        K = intrinsics[i]["camera_matrix"].astype(np.float64)
        w2c = poses[i].astype(np.float64)
        P = K @ w2c[:3, :]  # 3x4 projection matrix
        
        x, y = float(points_2d[i][0]), float(points_2d[i][1])
        
        # Check for NaN or Inf values
        if not (np.isfinite(x) and np.isfinite(y)):
            raise ValueError(f"Invalid 2D point at camera {i}: ({x}, {y})")
        
        # Build the linear system for DLT
        # x * p_3^T - p_1^T = 0
        # y * p_3^T - p_2^T = 0
        A[2*i] = x * P[2] - P[0]
        A[2*i + 1] = y * P[2] - P[1]
    
    # Check for NaN or Inf in A matrix
    if not np.all(np.isfinite(A)):
        raise ValueError("Matrix A contains NaN or Inf values")
    
    try:
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        _, _, Vt = np.linalg.svd(A + np.random.randn(*A.shape) * 1e-10, full_matrices=False)
    
    X_homogeneous = Vt[-1]
    
    X_3d = X_homogeneous[:3] / X_homogeneous[3]
    
    return X_3d.astype(np.float32)

def combine_flows_advanced(voxel_flows, coordinates, voxel_carved, poses, intrinsics, device="cuda"):
    device = torch.device(device)
    
    num_cameras = len(poses)
    num_x, num_y, num_z = coordinates.shape[:3]
    
    coordinates_t = torch.from_numpy(coordinates).float().to(device)
    voxel_carved_t = torch.from_numpy(voxel_carved).to(device)
    
    ones = torch.ones((*coordinates_t.shape[:-1], 1), dtype=torch.float32, device=device)
    coordinates_h_t = torch.cat([coordinates_t, ones], dim=-1)
    
    K_all = np.stack([intrinsics[i_cam]["camera_matrix"] for i_cam in range(num_cameras)], axis=0)
    poses_all = np.stack(poses, axis=0)
    K_all_t = torch.from_numpy(K_all).float().to(device)
    poses_all_t = torch.from_numpy(poses_all).float().to(device)
    
    voxel_flows_stacked = torch.from_numpy(np.stack(voxel_flows, axis=0)).float().to(device)
    
    coords_expanded = coordinates_h_t[None, :, :, :, :, None]
    coords_cam = torch.matmul(poses_all_t[:, None, None, None, :, :], coords_expanded)[..., :, 0]
    depths = coords_cam[..., 2:3]
    coords_img = torch.matmul(K_all_t[:, None, None, None, :, :], coords_cam[..., :3, None])[..., 0]
    
    pixel_coords = coords_img[..., :2] / coords_img[..., 2:3]
    pixel_coords_tip = pixel_coords + voxel_flows_stacked
    
    ones = torch.ones_like(pixel_coords_tip[..., :1])
    pixel_coords_tip_h = torch.cat([pixel_coords_tip, ones], dim=-1)
    
    K_inv_all_t = torch.linalg.inv(K_all_t)
    coords_cam_tip = torch.matmul(K_inv_all_t[:, None, None, None, :, :], pixel_coords_tip_h[..., None])[..., 0]
    
    coords_cam_tip = coords_cam_tip * depths
    
    ones = torch.ones_like(coords_cam_tip[..., :1])
    coords_cam_tip_h = torch.cat([coords_cam_tip, ones], dim=-1)
    
    c2w_all_t = torch.linalg.inv(poses_all_t)
    
    flow_tips_world = torch.matmul(c2w_all_t[:, None, None, None, :, :], coords_cam_tip_h[..., None])[..., :3, 0]
    
    flow_3d_per_camera = flow_tips_world - coordinates_t[None, :, :, :, :]
    
    valid_depth = (depths[..., 0] > 0) & torch.isfinite(depths[..., 0])
    valid_flow = torch.all(torch.isfinite(flow_3d_per_camera), dim=-1)
    valid_mask = valid_depth & valid_flow
    
    num_voxels = num_x * num_y * num_z
    flow_3d_flat = flow_3d_per_camera.permute(1, 2, 3, 0, 4).reshape(num_voxels, num_cameras, 3)
    valid_mask_flat = valid_mask.permute(1, 2, 3, 0).reshape(num_voxels, num_cameras)
    
    flow_3d_masked = flow_3d_flat.clone()
    flow_3d_masked[~valid_mask_flat] = 0.0
    
    A = flow_3d_masked
    b = torch.sum(flow_3d_masked * flow_3d_masked, dim=2)
    
    valid_mask_expanded = valid_mask_flat.unsqueeze(-1)
    
    A_masked = A * valid_mask_expanded.float()
    ATA = torch.matmul(A_masked.transpose(1, 2), A_masked)
    
    ridge_lambda_0 = 1e-1
    
    squared_magnitudes = torch.sum(flow_3d_masked * flow_3d_masked, dim=2)
    num_valid_per_voxel = valid_mask_flat.sum(dim=1).clamp(min=1)
    mean_squared_magnitude = (squared_magnitudes * valid_mask_flat.float()).sum(dim=1) / num_valid_per_voxel
    
    ridge_lambda = ridge_lambda_0 * mean_squared_magnitude
    
    I = torch.eye(3, device=device, dtype=torch.float32)
    ATA_ridge = ATA + ridge_lambda.view(-1, 1, 1) * I.unsqueeze(0)
    
    b_masked = b * valid_mask_flat.float()
    ATb = torch.matmul(A_masked.transpose(1, 2), b_masked.unsqueeze(-1)).squeeze(-1)
    
    num_valid = valid_mask_flat.sum(dim=1)
    
    valid_system = (num_valid >= 1) & torch.all(torch.isfinite(ATA_ridge).view(num_voxels, -1), dim=1) & torch.all(torch.isfinite(ATb), dim=1)
    
    result = torch.full((num_voxels, 3), float('nan'), dtype=torch.float32, device=device)
    
    if valid_system.any():
        try:
            v = torch.linalg.solve(ATA_ridge[valid_system], ATb[valid_system, :, None]).squeeze(-1)  # (num_valid, 3)
            result[valid_system] = v
            valid_indices = torch.where(valid_system)[0]
            flow_3d_valid = flow_3d_flat[valid_indices]
            valid_mask_valid = valid_mask_flat[valid_indices]
            input_magnitudes = torch.norm(flow_3d_valid, dim=2)
            input_magnitudes_masked = input_magnitudes.clone()
            input_magnitudes_masked[~valid_mask_valid] = 0.0
            num_valid_per_voxel = valid_mask_valid.sum(dim=1, keepdim=True)       
        except RuntimeError:
            pass
    
    result = result.reshape(num_x, num_y, num_z, 3)
    result[~voxel_carved_t] = float('nan')
    return result.cpu().numpy()


def unproject_to_world(pixel_coords, depth, camera_matrix, w2c):
    # Compute camera-to-world transformation
    c2w = np.linalg.inv(w2c)
    
    # Convert pixel coordinates to homogeneous coordinates
    ones = np.ones((pixel_coords.shape[0], 1), dtype=np.float32)
    pixels_homogeneous = np.concatenate([pixel_coords, ones], axis=1)  # Nx3
    
    # Unproject to camera space: P_cam = depth * K_inv @ [u, v, 1]
    K_inv = np.linalg.inv(camera_matrix)
    camera_points = (K_inv @ pixels_homogeneous.T).T  # Nx3
    
    # Scale by depth
    if np.isscalar(depth):
        camera_points = camera_points * depth
    else:
        camera_points = camera_points * depth[:, np.newaxis]
    ones = np.ones((camera_points.shape[0], 1), dtype=np.float32)
    camera_points_homogeneous = np.concatenate([camera_points, ones], axis=1)
    
    world_points = (c2w @ camera_points_homogeneous.T).T
    
    return world_points[:, :3]

def main(input_id, device="cuda"):    
    input_dir = "data/undistorted_video"
    video_paths = sorted(glob.glob(os.path.join(input_dir, f"{input_id:04d}_*.undistorted.mkv")))
    num_cameras = len(video_paths)
    print(f"Found {len(video_paths)} video paths:")
    for video_path in video_paths:
        print(f" - {video_path}")

    output_dir = os.path.join("data/voxel_projection", f"{input_id:04d}")
    os.makedirs(output_dir, exist_ok=True)

    # Load colmap intrinsics/extrinsics
    colmap_dir = os.path.join("data/colmap", f"{input_id:04d}", "sparse/0")
    colmap_cameras_file = os.path.join(colmap_dir, "cameras.txt")
    intrinsics = [load_colmap_intrinsics(colmap_cameras_file, i_cam) for i_cam in range(num_cameras)]
    colmap_images_file = os.path.join(colmap_dir, "images.txt")
    poses = [load_colmap_pose(colmap_images_file, i_cam) for i_cam in range(num_cameras)]

    # Load sync masks
    sync_mask_paths = [os.path.join("data/sync_mask", f"{input_id:04d}_{i_cam}.sync_mask.png") for i_cam in range(num_cameras)]
    sync_masks = [cv2.imread(sync_mask_path, cv2.IMREAD_UNCHANGED) for sync_mask_path in sync_mask_paths]
    sync_masks = [sync_mask[:, :, 0] > 128 for sync_mask in sync_masks]

    # Load times data
    times_all = []
    for i_cam in range(num_cameras):
        times_path = os.path.join("data/sync", f"{input_id:04d}_{i_cam}.times.npz")
        times_data = np.load(times_path)
        times_all.append(times_data["times"])

    time_start = np.max([np.nanmin(times_all[i]) for i in range(num_cameras)])
    time_end = np.min([np.nanmax(times_all[i]) for i in range(num_cameras)])
    
    time_duration = time_end - time_start
    print(f"{time_start=} {time_end=} {time_duration=}")

    index_start = np.nanargmin(np.abs(times_all[0] - time_start))
    index_end = np.nanargmin(np.abs(times_all[0] - time_end))

    print(f"{index_start=} {index_end=}")

    num_steps_volume_estimation = 100
    buffer_steps = (index_end - index_start) // 8
    volume_steps = np.linspace(index_start + buffer_steps, index_end - buffer_steps, num_steps_volume_estimation, dtype=np.int32)

    # Add camera centers
    camera_origins = []
    for w2c in poses:
        c2w = np.linalg.inv(w2c)
        camera_origin = c2w[:3, 3]
        camera_origins.append(camera_origin)
    camera_origins = np.array(camera_origins)
    
    flow_center_history = []  # List to store past flow center positions

    for index_reference in tqdm(volume_steps):
        time_reference = times_all[0][index_reference]

        indices = [np.nanargmin(np.abs(times_all[i_cam] - time_reference)) for i_cam in range(num_cameras)]
        flow_paths = [os.path.join("data/flow", f"{input_id:04d}_{i_cam}", f"{indices[i_cam]:06d}.png") for i_cam in range(num_cameras)]
        flow_images = [cv2.imread(flow_path, cv2.IMREAD_UNCHANGED) for flow_path in flow_paths]
        flows = [flow_image[:, :, :2] for flow_image in flow_images]
        flows = [uint8_to_float(flow, -25, 25) for flow in flows]
        for i_cam in range(num_cameras):
            flows[i_cam][sync_masks[i_cam]] = 0

        flow_masks = [np.linalg.norm(flow, axis=-1) > 0.5 for flow in flows]
        
        # Check if all cameras have flow detected
        is_flow = np.all([np.any(flow_mask) for flow_mask in flow_masks])
        if not is_flow:
            continue
        
        flow_centers = []
        for i_cam in range(num_cameras):
            flow_mask = flow_masks[i_cam]
            y_coords, x_coords = np.where(flow_mask)
            y_mean = np.mean(y_coords)
            x_mean = np.mean(x_coords)
            flow_centers.append(np.array([x_mean, y_mean]))

        flow_center_3d = triangulate_point(flow_centers, intrinsics, poses)
        
        # Add current flow center to history
        flow_center_history.append(flow_center_3d.copy())

    # Find robust flow center (exclude 5% outliers)
    flow_center_3d_history = np.array(flow_center_history)
    flow_center_3d_mean = np.mean(flow_center_3d_history, axis=0)
    print(f"{flow_center_3d_mean=}")

    # Compute corresponding depth
    center_depths = []
    for i_cam in range(num_cameras):
        pose = poses[i_cam]
        depth_cam = (pose @ homogenize(flow_center_3d_mean)[..., None])[2, 0]
        center_depths.append(depth_cam)

    center_depths = np.array(center_depths)
    print(f"{center_depths=}")

    bbox = np.empty((3, 2))
    bbox[:, 0] = np.inf
    bbox[:, 1] = -np.inf
    for index_reference in tqdm(volume_steps):
        time_reference = times_all[0][index_reference]

        indices = [np.nanargmin(np.abs(times_all[i_cam] - time_reference)) for i_cam in range(num_cameras)]
        flow_paths = [os.path.join("data/flow", f"{input_id:04d}_{i_cam}", f"{indices[i_cam]:06d}.png") for i_cam in range(num_cameras)]
        flow_images = [cv2.imread(flow_path, cv2.IMREAD_UNCHANGED) for flow_path in flow_paths]
        flows = [flow_image[:, :, :2] for flow_image in flow_images]
        flows = [uint8_to_float(flow, -25, 25) for flow in flows]
        for i_cam in range(num_cameras):
            flows[i_cam][sync_masks[i_cam]] = 0

        flow_masks = [np.linalg.norm(flow, axis=-1) > 0.5 for flow in flows]
        
        is_flow = np.all([np.any(flow_mask) for flow_mask in flow_masks])
        if not is_flow:
            continue

        # Collect all 3D points from all cameras (near and far)
        all_points_3d = []
        for i_cam in range(num_cameras):
            flow_mask = flow_masks[i_cam]
            
            # Get maximum width of the flow mask in pixels
            depth_center = center_depths[i_cam]
            max_width_px = np.max(np.sum(flow_mask, axis=1))
            max_width = (max_width_px * depth_center) / intrinsics[i_cam]["camera_matrix"][0, 0]

            # Get pixel coordinates where flow mask is True
            y_coords, x_coords = np.where(flow_mask)
            pixel_coords = np.stack([x_coords, y_coords], axis=1).astype(np.float32)  # Nx2
            
            if len(pixel_coords) > 0:
                # Unproject using near depth (min_depth_calib)
                depth_near = depth_center # - max_width / 2
                points_near = unproject_to_world(
                    pixel_coords, 
                    depth_near, 
                    intrinsics[i_cam]["camera_matrix"], 
                    poses[i_cam]
                )
                all_points_3d.append(points_near)
                
                # Unproject using far depth (min_depth_calib + max_width_calib)
                depth_far = depth_center # + max_width / 2
                points_far = unproject_to_world(
                    pixel_coords, 
                    depth_far, 
                    intrinsics[i_cam]["camera_matrix"], 
                    poses[i_cam]
                )
                all_points_3d.append(points_far)
        
        # Update bbox with all 3D points from this frame
        if len(all_points_3d) > 0:
            all_points_3d = np.concatenate(all_points_3d, axis=0)  # Combine all points
            bbox[:, 0] = np.minimum(bbox[:, 0], np.min(all_points_3d, axis=0))
            bbox[:, 1] = np.maximum(bbox[:, 1], np.max(all_points_3d, axis=0))

    print(f"{bbox=}")

    # Expand bbox by 10%
    dimensions = bbox[:, 1] - bbox[:, 0]
    bbox[:, 0] = bbox[:, 0] - dimensions * 0.1
    bbox[:, 1] = bbox[:, 1] + dimensions * 0.1

    # Make xy-symmetric
    bbox_center = (bbox[:, 0] + bbox[:, 1]) / 2
    dimensions = bbox[:, 1] - bbox[:, 0]
    xy_max = np.max(dimensions[1:])
    bbox[1:, 0] = bbox_center[1:] - xy_max / 2
    bbox[1:, 1] = bbox_center[1:] + xy_max / 2

    print(f"{dimensions=}")

    # Compute voxel size based on fixed number of voxels
    num_voxels = 100_000
    total_volume = np.prod(bbox[:, 1] - bbox[:, 0])
    voxel_volume = total_volume / num_voxels
    voxel_size = (voxel_volume ** (1/3))
    print(f"{voxel_size=}")

    # Compute voxel grid
    num_voxels_x = np.ceil((bbox[:, 1][0] - bbox[:, 0][0]) / voxel_size).astype(np.int32)
    num_voxels_y = np.ceil((bbox[:, 1][1] - bbox[:, 0][1]) / voxel_size).astype(np.int32)
    num_voxels_z = np.ceil((bbox[:, 1][2] - bbox[:, 0][2]) / voxel_size).astype(np.int32)
    print(f"{num_voxels_x=} {num_voxels_y=} {num_voxels_z=}")

    coordinates_x = np.arange(num_voxels_x) * voxel_size + bbox[:, 0][0]
    coordinates_y = np.arange(num_voxels_y) * voxel_size + bbox[:, 0][1]
    coordinates_z = np.arange(num_voxels_z) * voxel_size + bbox[:, 0][2]

    coordinates_x, coordinates_y, coordinates_z = np.meshgrid(coordinates_x, coordinates_y, coordinates_z, indexing="ij")
    coordinates = np.stack([coordinates_x, coordinates_y, coordinates_z], axis=-1)
    np.savez_compressed(os.path.join(output_dir, "metadata.npz"), coordinates=coordinates, voxel_size=voxel_size, bbox=bbox, times=times_all[0][index_start:index_end])
    print(f"{coordinates.shape=}")


    # Project the coordinates to the cameras
    coordinates_h = homogenize(coordinates)
    projected_coordinates = []
    for i_cam in range(num_cameras):
        intrinsics_cam = intrinsics[i_cam]["camera_matrix"]
        pose_cam = poses[i_cam]

        coordinates_cam = (pose_cam[None, None, None, ...] @ coordinates_h[..., None])[..., :3, 0]

        coordinates_cam = (intrinsics_cam[None, None, None, ...] @ coordinates_cam[..., None])[..., 0]
        coordinates_cam = coordinates_cam[..., :2] / coordinates_cam[..., 2:3]

        coordinates_cam_valid = (coordinates_cam[..., 0] >= 0) & (coordinates_cam[..., 0] < (intrinsics[i_cam]["width"] - 1)) & (coordinates_cam[..., 1] >= 0) & (coordinates_cam[..., 1] < (intrinsics[i_cam]["height"]) - 1)
        coordinates_cam[~coordinates_cam_valid] = np.nan
        projected_coordinates.append(coordinates_cam)
    
    # Project the flow from the cameras to the voxel grid
    os.makedirs(os.path.join(output_dir, "voxel_flow"), exist_ok=True)
    # index_start = 0
    for index_reference in tqdm(range(index_start+10, index_end)):
        time_reference = times_all[0][index_reference]
        indices = [np.nanargmin(np.abs(times_all[i_cam] - time_reference)) for i_cam in range(num_cameras)]
        flow_paths = [os.path.join("data/flow", f"{input_id:04d}_{i_cam}", f"{indices[i_cam]:06d}.png") for i_cam in range(num_cameras)]
        flow_images = [cv2.imread(flow_path, cv2.IMREAD_UNCHANGED) for flow_path in flow_paths]
        flows = [flow_image[:, :, :2] for flow_image in flow_images]
        flows = [uint8_to_float(flow, -25, 25) for flow in flows]
        for i_cam in range(num_cameras):
            flows[i_cam][sync_masks[i_cam]] = 0

        flow_masks = [np.linalg.norm(flow, axis=-1) > 0.5 for flow in flows]

        voxel_flows = []
        voxel_is_within_flame = []
        voxel_is_within_frame = []
        for i_cam in range(num_cameras):
            flow_mask = flow_masks[i_cam]
            flow = flows[i_cam]
            projected_coordinates_cam = projected_coordinates[i_cam]
            projected_coordinates_cam_valid = ~np.any(np.isnan(projected_coordinates_cam), axis=-1)
            projected_coordinates_cam[~projected_coordinates_cam_valid, :] = 0
            projected_coordinates_cam = np.round(projected_coordinates_cam).astype(np.int32)
            projected_coordinates_cam_x = projected_coordinates_cam[..., 0]
            projected_coordinates_cam_y = projected_coordinates_cam[..., 1]

            voxel_grid_binary = flow_mask[projected_coordinates_cam_y, projected_coordinates_cam_x]
            voxel_grid_flow = flow[projected_coordinates_cam_y, projected_coordinates_cam_x]

            voxel_is_within_flame.append(voxel_grid_binary)
            voxel_is_within_frame.append(projected_coordinates_cam_valid)
            voxel_flows.append(voxel_grid_flow)

        # Visualize the voxel grid using open3d - overlay all cameras
        if True:
            configs_one_missing = [[c for c in range(num_cameras) if c != d] for d in range(num_cameras)]
            configs_all = list(range(num_cameras))
            configs = configs_one_missing + [configs_all]

            min_num_cameras = 2
            flows_comb = {}
            for cam_idxs in configs:
                voxel_is_within_frames = np.any([voxel_is_within_frame[cid] for cid in cam_idxs] , axis=0)
                voxel_is_within_most_flames = np.sum([voxel_is_within_flame[cid] for cid in cam_idxs], axis=0) >= min_num_cameras
                voxel_carved = voxel_is_within_frames & voxel_is_within_most_flames
                
                coordinates_carved = coordinates[voxel_carved]
                flows_combined = combine_flows_advanced(
                    [voxel_flows[cid] for cid in cam_idxs], 
                    coordinates, 
                    voxel_carved, 
                    [poses[cid] for cid in cam_idxs], 
                    [intrinsics[cid] for cid in cam_idxs],
                    device=device)
                flows_comb[tuple(cam_idxs)] = flows_combined

            
            voxel_is_within_frames = np.any(voxel_is_within_frame, axis=0)
            voxel_is_within_most_flames = np.sum(voxel_is_within_flame, axis=0) >= min_num_cameras
            voxel_carved = voxel_is_within_frames & voxel_is_within_most_flames

            coordinates_carved = coordinates[voxel_carved]
            flows_combined = combine_flows_advanced(voxel_flows, coordinates, voxel_carved, poses, intrinsics, device=device)
            np.savez_compressed(os.path.join(output_dir, "voxel_flow", f"{index_reference-index_start:06d}.npz"), flows=flows_combined, flows_comb=flows_comb)

if __name__ == "__main__":
    input_id = 11
    device = "cuda"
    
    print(f"Processing scene {input_id:04d}...")
    main(input_id, device=device)
    print("Processing complete!")