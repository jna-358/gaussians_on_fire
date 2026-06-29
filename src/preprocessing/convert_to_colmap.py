import numpy as np
import glob
import os
import cv2
import subprocess
import open3d as o3d
import shutil
import json
import pickle
import sys
from calibration_utils import get_camera_serial_number, load_calibration_data


def load_rotated_calibration_data(video_path):
    """Load rotated calibration data from .rotated_cams.npz file"""
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    rotated_path = os.path.join("data/rotated_cams", f"{video_basename}.rotated_cams.npz")
    if not os.path.exists(rotated_path):
        print(f"Rotated calibration file not found: {rotated_path}")
        return None
    
    try:
        data = np.load(rotated_path)
        return data
    except Exception as e:
        print(f"Error loading rotated calibration data for {video_path}: {e}")
        return None


def rotate_image_90ccw(image):
    """Rotate image 90 degrees counterclockwise"""
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def projective_downsampling(pcd, base_voxel_size=0.5, num_intervals=10):
    """
    Apply distance-based downsampling to maintain near-constant perceptual density from origin.

    Points are split into distance intervals (log-spaced) and each interval is downsampled
    with a voxel size that scales with distance. This ensures a camera at the origin
    would see approximately constant point density regardless of distance.

    Args:
        pcd: Open3D point cloud to downsample
        base_voxel_size: Projected voxel size at distance 1.0 (default: 0.5)
        num_intervals: Number of distance intervals to use (default: 10)

    Returns:
        Downsampled Open3D point cloud
    """
    print("\nApplying projective downsampling...")

    # Get all points as numpy array
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Calculate distances from origin
    distances = np.linalg.norm(points, axis=1)

    # Find min and max distances
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    print(f"Distance range: {min_dist:.2f} to {max_dist:.2f}")

    # Create intervals in log space
    interval_edges = np.logspace(np.log10(min_dist), np.log10(max_dist), num=num_intervals + 1)

    # Process each interval
    downsampled_points = []
    downsampled_colors = []

    for i in range(len(interval_edges) - 1):
        interval_min = interval_edges[i]
        interval_max = interval_edges[i + 1]

        # Find points in this interval
        in_interval = (distances >= interval_min) & (distances < interval_max)
        interval_points = points[in_interval]
        interval_colors = colors[in_interval]

        if len(interval_points) == 0:
            continue

        # Calculate voxel size for this interval
        # Voxel size scales linearly with distance
        avg_distance = (interval_min + interval_max) / 2
        voxel_size = base_voxel_size * avg_distance

        # Create temporary point cloud for this interval
        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(interval_points)
        temp_pcd.colors = o3d.utility.Vector3dVector(interval_colors)

        # Downsample this interval
        temp_pcd_downsampled = temp_pcd.voxel_down_sample(voxel_size=voxel_size)

        # Collect downsampled points and colors
        downsampled_points.append(np.asarray(temp_pcd_downsampled.points))
        downsampled_colors.append(np.asarray(temp_pcd_downsampled.colors))

        print(f"  Interval [{interval_min:.2f}, {interval_max:.2f}]: "
              f"{len(interval_points)} -> {len(temp_pcd_downsampled.points)} points "
              f"(voxel_size={voxel_size:.3f})")

    # Combine all downsampled points
    all_downsampled_points = np.vstack(downsampled_points)
    all_downsampled_colors = np.vstack(downsampled_colors)

    # Create final downsampled point cloud
    result_pcd = o3d.geometry.PointCloud()
    result_pcd.points = o3d.utility.Vector3dVector(all_downsampled_points)
    result_pcd.colors = o3d.utility.Vector3dVector(all_downsampled_colors)

    print(f"Projectively downsampled point cloud has {len(result_pcd.points)} points")

    return result_pcd

def setMaxDistance(pcd, max_distance):
    """
    Enforce maximum distance from origin for all points in a point cloud.
    Points that violate the constraint are rescaled to be exactly at max_distance.
    
    Args:
        pcd: Open3D point cloud
        max_distance: Maximum allowed distance from origin
    
    Returns:
        Modified point cloud with enforced constraint
    """
    points = np.asarray(pcd.points)
    
    # Calculate distance from origin for each point
    distances = np.linalg.norm(points, axis=1)
    
    # Find points that violate the constraint
    violating_mask = distances > max_distance
    num_violating = np.sum(violating_mask)
    
    if num_violating > 0:
        print(f"Found {num_violating} points exceeding max distance of {max_distance}")
        
        # Rescale violating points to be exactly at max_distance
        # For each point: new_point = point * (max_distance / distance)
        points[violating_mask] = points[violating_mask] * (max_distance / distances[violating_mask, np.newaxis])
        
        # Update the point cloud with modified points
        pcd.points = o3d.utility.Vector3dVector(points)
        
        print(f"Rescaled {num_violating} points to max distance of {max_distance}")
    else:
        print(f"All points are within max distance of {max_distance}")
    
    return pcd


if __name__ == "__main__":    
    input_dir = "data/input_video"
    input_id = 11

    video_paths = sorted(glob.glob(os.path.join(input_dir, f"{input_id:04d}_*.mkv")))
    num_cameras = len(video_paths)
    
    if num_cameras == 0:
        print(f"No videos found matching pattern: {input_dir}/{input_id:04d}_*.mkv")
        sys.exit(1)
    
    print(f"Found {num_cameras} videos:")
    for video_path in video_paths:
        print(f"  - {video_path}")
    
    # Output directory
    output_dir = os.path.join("data/colmap", f"{input_id:04d}")

    # Load both original and rotated calibration data
    original_data = []
    rotated_data = []
    for video_path in video_paths:
        # Load original calibration for undistortion
        serial_number = get_camera_serial_number(video_path)
        orig_data = load_calibration_data(serial_number)
        if orig_data is None:
            print(f"Error: Could not load calibration for {video_path}")
            sys.exit(1)
        original_data.append(orig_data)
        
        # Load rotated calibration for epipolar geometry
        rot_data = load_rotated_calibration_data(video_path)
        if rot_data is None:
            print(f"Error: Could not load rotated calibration for {video_path}")
            print(f"Please run rotate_cams.py first to generate .rotated_cams.npz files")
            sys.exit(1)
        rotated_data.append(rot_data)

    # Extract poses and camera matrices from rotated data
    poses = []
    camera_matrices_simple = []
    camera_matrices = []
    for data in rotated_data:
        poses.append({'R': data['R'], 't': data['t']})
        camera_matrices.append(data['camera_matrix'])

        # Create simple camera matrix
        fx = data['camera_matrix'][0, 0]
        fy = data['camera_matrix'][1, 1]
        image_size = data['image_size']
        fx_new = max(fx, fy)
        fy_new = fx_new
        cx = image_size[0] / 2.0
        cy = image_size[1] / 2.0
        zoom_factor = 1.03 # Slightly zoom in to avoid black borders
        fx_new *= zoom_factor
        fy_new *= zoom_factor
        K_simple = np.array([[fx_new, 0, cx],
                             [0, fx_new, cy],
                             [0, 0, 1.0]])
        camera_matrices_simple.append(K_simple)

    points3D = rotated_data[0]['points3D']

    # Load and process frames
    frames = [None] * num_cameras
    print("\nLoading and processing frames...")
    for i_video, video_path in enumerate(video_paths):
        print(f"  Processing video {i_video + 1}/{num_cameras}: {os.path.basename(video_path)}")
        
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        frame_path = os.path.join("data/rotated_undistorted", f"{video_basename}.rotated_undistorted.png")
        frame = cv2.imread(frame_path)
        
        # Rotate the undistorted frame 90 degrees counterclockwise
        frames[i_video] = frame

    points2D = []
    for i_video in range(num_cameras):
        # Use the rotated camera matrix and pose to project points3D to the undistorted and rotated frame
        R_rotated = poses[i_video]['R']
        t_rotated = poses[i_video]['t']
        K_rotated = camera_matrices[i_video]
        
        # Convert rotation matrix to rotation vector for cv2.projectPoints
        rvec_rotated, _ = cv2.Rodrigues(R_rotated)
        tvec_rotated = t_rotated.flatten()
        
        # Project 3D points using rotated camera parameters
        points_2d, _ = cv2.projectPoints(
            points3D.astype(np.float32),  # 3D points
            rvec_rotated,                 # Rotated rotation vector
            tvec_rotated,                # Rotated translation vector
            K_rotated,                   # Rotated camera intrinsic matrix
            None                         # No distortion (undistorted)
        )
        points_2d = points_2d.reshape(-1, 2)
        
        # Filter points that are within rotated image bounds
        new_width, new_height = rotated_data[i_video]['image_size']
        in_bounds = ((points_2d[:, 0] >= 0) & (points_2d[:, 0] < new_width) & 
                     (points_2d[:, 1] >= 0) & (points_2d[:, 1] < new_height))
        points_2d_final = points_2d[in_bounds]
        
        points2D.append(points_2d_final)
        print(f"Camera {i_video}: Projected {len(points_2d_final)}/{len(points3D)} 3D points to rotated image")

    # Load monocular depths
    monocular_disparity = []
    for i_camera in range(num_cameras):
        video_basename = os.path.splitext(os.path.basename(video_paths[i_camera]))[0]
        disparity_path = os.path.join("data/mono_depth", f"{video_basename}.mono_depth_16.png")
        disparity_array = cv2.imread(disparity_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        print(disparity_array.dtype, disparity_array.shape, disparity_array.min(), disparity_array.max())
        disparity_array /= 65535
        monocular_disparity.append(disparity_array)
        os.makedirs(os.path.join(output_dir, "depth_anything"), exist_ok=True)
        shutil.copy(disparity_path, os.path.join(output_dir, "depth_anything", f"{i_camera:04d}.png"))
        print(f"  Loaded monocular depth from: {disparity_path} with shape {disparity_array.shape}, dtype {disparity_array.dtype}, min {disparity_array.min()}, max {disparity_array.max()}")

    # Load stereo depths
    stereo_depths = []
    for i_camera in range(num_cameras):
        video_basename = os.path.splitext(os.path.basename(video_paths[i_camera]))[0]
        depth_path = os.path.join("data/stereo_depth_rotated", f"{video_basename}.stereo_depth_rotated.npy")
        stereo_depth = np.load(depth_path)
        stereo_depths.append(stereo_depth)

    # Plot stereo vs monocular depth
    monocular_depths = []
    monocular_depths_masked = []
    alignment_params = []  # Store (scale, offset) for each camera
    for i_camera in range(num_cameras):
        monocular_depth_inv = monocular_disparity[i_camera]
        stereo_depth_inv = 1.0 / stereo_depths[i_camera]

        # Filter for finite values only (remove inf and nan)
        is_valid = (monocular_depth_inv > 0) & (stereo_depth_inv > 0) & \
                   np.isfinite(monocular_depth_inv) & np.isfinite(stereo_depth_inv)
        monocular_depth_valid = monocular_depth_inv[is_valid]
        stereo_depth_valid = stereo_depth_inv[is_valid]

        circular_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        is_valid_dilated = cv2.dilate(is_valid.astype(np.uint8)*255, circular_kernel, iterations=4) > 0

        print(f"Camera {i_camera}: {len(monocular_depth_valid)} valid depth pairs")
        print(f"  Monocular depth range: [{monocular_depth_valid.min():.4f}, {monocular_depth_valid.max():.4f}]")
        print(f"  Stereo depth range: [{stereo_depth_valid.min():.4f}, {stereo_depth_valid.max():.4f}]")

        if len(monocular_depth_valid) < 2:
            print(f"Camera {i_camera}: Not enough valid depth pairs to fit. Skipping.")
            continue

        A = np.column_stack([monocular_depth_valid.flatten(), np.ones_like(monocular_depth_valid.flatten())])
        b_vec = stereo_depth_valid.flatten()

        params, residuals, rank, s = np.linalg.lstsq(A, b_vec, rcond=None)
        a, b = params
        alignment_params.append({
            "scale": float(a),
            "offset": float(b)
        })

        # Compute fitted values
        stereo_depth_fitted = a * monocular_depth_valid + b
        print(f"{stereo_depth_valid.min()=}, {stereo_depth_valid.max()=}, {stereo_depth_fitted.min()=}, {stereo_depth_fitted.max()=}")
        
        print(f"{stereo_depths[i_camera].min()=}, {stereo_depths[i_camera].max()=}")
        print(f"{a=}, {b=}")

        # Store aligned monocular depth
        monocular_depth_aligned = 1.0 / (a * monocular_depth_inv + b)
        monocular_depths.append(monocular_depth_aligned)
        monocular_depth_aligned_masked = monocular_depth_aligned.copy()
        monocular_depth_aligned_masked[is_valid_dilated] = np.nan
        monocular_depths_masked.append(monocular_depth_aligned_masked)
        
        video_basename = os.path.splitext(os.path.basename(video_paths[i_camera]))[0]
        os.makedirs("data/mono_depth_aligned", exist_ok=True)
        aligned_path = os.path.join("data/mono_depth_aligned", f"{video_basename}.mono_depth_aligned.npy")
        np.save(aligned_path, monocular_depth_aligned)
        print(f"  Saved aligned monocular depth to: {aligned_path}")

        # Compute R^2 score to evaluate fit quality
        ss_res = np.sum((stereo_depth_valid - stereo_depth_fitted) ** 2)
        ss_tot = np.sum((stereo_depth_valid - np.mean(stereo_depth_valid)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        print(f"Camera {i_camera}: Fitted scale={a:.4f}, offset={b:.4f}, R^2={r2_score:.4f}")

    # Save alignment parameters
    alignment_dict = {
        f"{i_camera:04d}": alignment_params[i_camera] for i_camera in range(len(alignment_params))
    }
    alignment_path = os.path.join(output_dir, "sparse", "0", "depth_params.json")
    os.makedirs(os.path.dirname(alignment_path), exist_ok=True)
    with open(alignment_path, 'w') as f:
        json.dump(alignment_dict, f, indent=4)
    print(f"Saved depth alignment parameters to: {alignment_path}")

    # Project depthmaps to 3D points
    print("\nConverting depth maps to 3D point clouds...")
    all_point_clouds = []
    all_point_clouds_mono = []
    max_depth = 100.0

    for i_camera in range(num_cameras):
        print(f"  Processing camera {i_camera + 1}/{num_cameras}...")

        # Get camera parameters
        K = camera_matrices[i_camera]
        R = poses[i_camera]['R']
        t = poses[i_camera]['t']
        depth_map = stereo_depths[i_camera]
        rgb_frame = frames[i_camera]

        stereo_depth_inv = 1.0 /depth_map
        mono_depth_inv = monocular_disparity[i_camera]

        mono_depth_inv_aligned = alignment_params[i_camera]['scale'] * mono_depth_inv + alignment_params[i_camera]['offset']
        
        is_valid = depth_map > 0
        circular_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        is_valid_dilated = cv2.dilate(is_valid.astype(np.uint8)*255, circular_kernel, iterations=4) > 0
        is_valid_dilated = is_valid_dilated.astype(bool)

        depth_inv_fusion = stereo_depth_inv.copy()
        depth_inv_fusion[~is_valid_dilated] = mono_depth_inv_aligned[~is_valid_dilated]

        depth_map_fusion = 1.0 / depth_inv_fusion

        max_depth = np.max(depth_map)

        depth_map_fusion[depth_map_fusion > max_depth*5] = 0
        depth_map_fusion[depth_map_fusion < 0] = 0

        # Get image dimensions
        height, width = depth_map_fusion.shape

        # Create meshgrid of pixel coordinates
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # Flatten arrays
        u_flat = u.flatten()
        v_flat = v.flatten()
        depth_flat = depth_map_fusion.flatten()
        
        # Filter out invalid depths (zero or negative)
        valid_mask = depth_flat > 0
        u_valid = u_flat[valid_mask]
        v_valid = v_flat[valid_mask]
        depth_valid = depth_flat[valid_mask]

        # Get camera intrinsics
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        # Backproject to 3D in camera coordinates
        x_cam = (u_valid - cx) * depth_valid / fx
        y_cam = (v_valid - cy) * depth_valid / fy
        z_cam = depth_valid

        # Stack into (N, 3) array
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

        # Transform to world coordinates
        t_reshaped = t.reshape(3, 1)
        points_world = (R.T @ (points_cam.T - t_reshaped)).T

        # Extract RGB colors for each point
        rgb_values = rgb_frame[v_valid, u_valid]
        rgb_values = rgb_values[:, [2, 1, 0]]
        colors = rgb_values.astype(np.float32) / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_world)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        all_point_clouds.append(pcd)
        print(f"    Created point cloud with {len(points_world)} points")


    # Process monocular depth maps
    print("\nConverting monocular depth maps to 3D point clouds...")
    for i_camera in range(num_cameras):
        print(f"  Processing monocular depth for camera {i_camera + 1}/{num_cameras}...")

        # Get camera parameters
        K = camera_matrices[i_camera]
        R = poses[i_camera]['R']
        t = poses[i_camera]['t']
        depth_map = monocular_depths[i_camera]
        rgb_frame = frames[i_camera]

        # Get image dimensions
        height, width = depth_map.shape

        # Create meshgrid of pixel coordinates
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # Flatten arrays
        u_flat = u.flatten()
        v_flat = v.flatten()
        depth_flat = depth_map.flatten()

        # Filter out invalid depths (zero, negative, or beyond max_depth)
        valid_mask = (depth_flat > 0) & (depth_flat < max_depth) & np.isfinite(depth_flat)
        u_valid = u_flat[valid_mask]
        v_valid = v_flat[valid_mask]
        depth_valid = depth_flat[valid_mask]

        # Get camera intrinsics
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        # Backproject to 3D in camera coordinates
        x_cam = (u_valid - cx) * depth_valid / fx
        y_cam = (v_valid - cy) * depth_valid / fy
        z_cam = depth_valid

        # Stack into (N, 3) array
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

        # Transform to world coordinates
        # World point = R^T * (Camera point - t)
        t_reshaped = t.reshape(3, 1)
        points_world = (R.T @ (points_cam.T - t_reshaped)).T

        # Extract RGB colors for each point
        rgb_values = rgb_frame[v_valid, u_valid]
        rgb_values = rgb_values[:, [2, 1, 0]]
        colors = rgb_values.astype(np.float32) / 255.0

        # Create Open3D point cloud
        pcd_mono = o3d.geometry.PointCloud()
        pcd_mono.points = o3d.utility.Vector3dVector(points_world)
        pcd_mono.colors = o3d.utility.Vector3dVector(colors)

        all_point_clouds_mono.append(pcd_mono)
        print(f"    Created monocular point cloud with {len(points_world)} points")

    # Merge all point clouds
    print("\nMerging point clouds...")
    merged_pcd = o3d.geometry.PointCloud()
    for pcd in all_point_clouds:
        merged_pcd += pcd
    print(f"Merged point cloud has {len(merged_pcd.points)} points")

    # Apply projective downsampling
    merged_pcd = projective_downsampling(merged_pcd, base_voxel_size=0.01, num_intervals=10)

    # Enforce max distance
    merged_pcd = setMaxDistance(merged_pcd, max_distance=300)

    # Merge all monocular point clouds
    print("\nMerging monocular point clouds...")
    merged_pcd_mono = o3d.geometry.PointCloud()
    for pcd in all_point_clouds_mono:
        merged_pcd_mono += pcd
    print(f"Merged monocular point cloud has {len(merged_pcd_mono.points)} points")

    # Apply projective downsampling
    merged_pcd_mono = projective_downsampling(merged_pcd_mono, base_voxel_size=0.01, num_intervals=10)
    
    print("\nVisualizing reprojection of merged point cloud to each camera...")
    merged_points = np.asarray(merged_pcd.points)
    merged_colors = np.asarray(merged_pcd.colors)

    frames_simple = []
    mono_inv_depth_simple = []
    stereo_depth_simple = []
    for i_camera in range(num_cameras):
        print(f"  Reprojecting to camera {i_camera + 1}/{num_cameras}...")

        K = camera_matrices[i_camera]
        R = poses[i_camera]['R']
        t = poses[i_camera]['t']

        rvec, _ = cv2.Rodrigues(R)
        tvec = t.flatten()

        points_2d, _ = cv2.projectPoints(
            merged_points.astype(np.float32),
            rvec,
            tvec,
            K,
            None
        )
        points_2d = points_2d.reshape(-1, 2)

        K_simple = camera_matrices_simple[i_camera]
        points_2d_simple, _ = cv2.projectPoints(
            merged_points.astype(np.float32),
            rvec,
            tvec,
            K_simple,
            None
        )
        points_2d_simple = points_2d_simple.reshape(-1, 2)

        # Filter points that are within image bounds and in front of camera
        width, height = rotated_data[i_camera]['image_size']

        # Check if points are in front of camera (positive z in camera coordinates)
        points_cam = R @ merged_points.T + t.reshape(3, 1)
        in_front = points_cam[2, :] > 0

        # Check if points are within image bounds
        in_bounds = ((points_2d[:, 0] >= 0) & (points_2d[:, 0] < width) &
                     (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height))
        
        in_bounds_simple = ((points_2d_simple[:, 0] >= 0) & (points_2d_simple[:, 0] < width) &
                            (points_2d_simple[:, 1] >= 0) & (points_2d_simple[:, 1] < height))
        valid_mask = in_front & in_bounds
        valid_mask_simple = in_front & in_bounds_simple
        points_2d_valid = points_2d[valid_mask]
        colors_valid = merged_colors[valid_mask]
        points_2d_valid_simple = points_2d_simple[valid_mask_simple]
        colors_valid_simple = merged_colors[valid_mask_simple]

        print(f"    Projected {len(points_2d_valid)}/{len(merged_points)} points to camera view")

        frame_display = frames[i_camera].copy()

        frame_display_simple = frames[i_camera].copy()

        K_original = camera_matrices[i_camera]
        homography = K_simple @ np.linalg.inv(K_original)

        height, width = frame_display_simple.shape[:2]
        frame_display_simple = cv2.warpPerspective(
            frames[i_camera],
            homography,
            (width, height),
            flags=cv2.INTER_LINEAR
        )
        mono_depth_simple_warped = cv2.warpPerspective(
            monocular_disparity[i_camera],
            homography,
            (width, height),
            flags=cv2.INTER_LINEAR
        )
        stereo_simple = cv2.warpPerspective(
            stereo_depths[i_camera],
            homography,
            (width, height),
            flags=cv2.INTER_LINEAR
        )
        stereo_depth_simple.append(stereo_simple)
        mono_inv_depth_simple.append(mono_depth_simple_warped)
        frames_simple.append(frame_display_simple.copy())
        print(f"Warped monocular depth: {mono_depth_simple_warped.shape}, dtype: {mono_depth_simple_warped.dtype}, min {mono_depth_simple_warped.min()}, max {mono_depth_simple_warped.max()}")

        # Load demo frame
        cap = cv2.VideoCapture(video_paths[i_camera])
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        ret, frame_demo = cap.read()
        cap.release()

        # Undistort demo frame
        calibration_data_original = original_data[i_camera]
        camera_matrix = calibration_data_original["camera_matrix"]
        dist_coeffs = calibration_data_original["dist_coeffs"]
        image_size = calibration_data_original["image_size"]
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, image_size, 0
        )

        frame_demo_unist = cv2.undistort(frame_demo, camera_matrix, dist_coeffs, None, new_camera_matrix)
        frame_demo_rot = cv2.rotate(frame_demo_unist, int(rotated_data[i_camera]['rotation_direction']))

        frame_demo_simple = cv2.warpPerspective(
            frame_demo_rot,
            homography,
            (width, height),
            flags=cv2.INTER_LINEAR
        )

        # Store undistortion args
        undistortion_args = {
            "undistort_args": (camera_matrix, dist_coeffs, None, new_camera_matrix),
            "rotate_args": (int(rotated_data[i_camera]['rotation_direction']),),
            "warpPerspective_args": (homography, (width, height)),
            "warpPerspective_kwargs": {"flags": cv2.INTER_LINEAR}
        }

        video_basename = os.path.splitext(os.path.basename(video_paths[i_camera]))[0]
        os.makedirs("data/undistortion_args", exist_ok=True)
        undistort_args_path = os.path.join("data/undistortion_args", f"{video_basename}.undistortion_args.pkl")
        with open(undistort_args_path, "wb") as f:
            pickle.dump(undistortion_args, f)
        print(f"Saved undistortion args to: {undistort_args_path}")

        error = np.abs((frame_demo_simple / 255.0) - (frame_display_simple / 255.0))

        # Make stereo mask
        stereo_mask = (stereo_simple > 0)
        stereo_mask = cv2.dilate(stereo_mask.astype(np.uint8)*255, np.ones((32, 32), np.uint8), iterations=1) > 0

        # Draw projected points on the image in red
        for pt in points_2d_valid:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(frame_display, (x, y), 1, (0, 0, 255), -1)

        for pt in points_2d_valid_simple:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(frame_display_simple, (x, y), 1, (255, 0, 0), -1)


    print(f"\nSaving COLMAP format outputs to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depth_stereo"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depth_mono"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depth_anything"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "sparse", "0"), exist_ok=True)

    # Save images
    for i_video, frame in enumerate(frames_simple):
        image_filename = os.path.join(output_dir, "images", f"{i_video:04d}.png")
        cv2.imwrite(image_filename, frame)

    # Save monocular disparity images (from depth anything)
    for i_video in range(num_cameras):
        disparity_filename = os.path.join(output_dir, "depth_anything", f"{i_video:04d}.png")
        cv2.imwrite(disparity_filename, (mono_inv_depth_simple[i_video] * 2**16).astype(np.uint16))

    # Save stereo depth maps
    for i_video in range(num_cameras):
        depth_filename = os.path.join(output_dir, "depth_stereo", f"{i_video:04d}.npy")
        np.save(depth_filename, stereo_depths[i_video])

    # Save monocular depth maps
    for i_video in range(num_cameras):
        depth_filename = os.path.join(output_dir, "depth_mono", f"{i_video:04d}.npy")
        np.save(depth_filename, monocular_depths[i_video])

    # Save cameras.bin, images.bin, points3D.bin using pycolmap
    import pycolmap
    from scipy.spatial.transform import Rotation as R_scipy

    print("\nPreparing COLMAP export...")

    # Step 1: Write cameras in text format
    sparse_dir = os.path.join(output_dir, "sparse", "0")
    print(f"Writing COLMAP reconstruction to: {sparse_dir}")

    # Create temporary reconstruction just for cameras
    reconstruction = pycolmap.Reconstruction()

    print("\nAdding cameras to COLMAP reconstruction...")
    for i_camera in range(num_cameras):
        K = camera_matrices_simple[i_camera]
        width, height = rotated_data[i_camera]['image_size']

        # Extract camera intrinsics
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        # Create COLMAP camera (PINHOLE model)
        camera = pycolmap.Camera(
            model='PINHOLE',
            width=width,
            height=height,
            params=[fx, fy, cx, cy],
            camera_id=i_camera
        )
        reconstruction.add_camera(camera)
        print(f"  Added camera {i_camera}: {width}x{height}, fx={fx:.2f}, fy={fy:.2f}")

    # Write cameras to text format first
    reconstruction.write_text(sparse_dir)

    # Step 2: Manually write images.txt with poses
    print("\nWriting images with poses...")
    images_txt_path = os.path.join(sparse_dir, "images.txt")
    with open(images_txt_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {num_cameras}\n")

        for i_camera in range(num_cameras):
            R = poses[i_camera]['R']
            t = poses[i_camera]['t'].flatten()

            # Convert rotation matrix to quaternion
            quat_xyzw = R_scipy.from_matrix(R).as_quat()  # [x, y, z, w]

            # COLMAP format uses QW, QX, QY, QZ, TX, TY, TZ
            qw, qx, qy, qz = quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]

            image_name = f"{i_camera:04d}.png"
            f.write(f"{i_camera} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {i_camera} {image_name}\n")
            f.write("\n")

            print(f"  Added image {i_camera}: {image_name}")

    # Step 3: Write points3D.txt from the merged point cloud
    print("\nWriting 3D points from merged point cloud...")
    points = np.asarray(merged_pcd.points)
    colors = np.asarray(merged_pcd.colors)
    colors_uint8 = (colors * 255).astype(np.uint8)

    points3D_txt_path = os.path.join(sparse_dir, "points3D.txt")
    with open(points3D_txt_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points)}\n")

        for i_point in range(len(points)):
            xyz = points[i_point]
            rgb = colors_uint8[i_point]
            f.write(f"{i_point} {xyz[0]} {xyz[1]} {xyz[2]} {rgb[0]} {rgb[1]} {rgb[2]} 0.0\n")

    print(f"  Added {len(points)} 3D points")

    # Step 4: Export point cloud as PLY
    print("\nExporting point cloud as PLY...")
    points3D_ply_path = os.path.join(sparse_dir, "points3D.ply")
    o3d.io.write_point_cloud(points3D_ply_path, merged_pcd)
    print(f"  Saved points3D.ply")

    # Step 5: Read text format and write binary format
    print("\nConverting to binary format...")
    reconstruction_final = pycolmap.Reconstruction()
    reconstruction_final.read_text(sparse_dir)
    reconstruction_final.write_binary(sparse_dir)

    print(f"  Saved cameras.bin, images.bin, points3D.bin")
    print("\nCOLMAP export complete!")