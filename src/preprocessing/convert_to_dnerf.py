import os
import glob
import numpy as np
import json
import cv2
import random
import tqdm
import shutil

np.random.seed(42)
random.seed(42)

def load_colmap_camera(colmap_dir, i_cam):
    cameras_txt_path = os.path.join(colmap_dir, "sparse/0/cameras.txt")
    images_txt_path = os.path.join(colmap_dir, "sparse/0/images.txt")

    # Load camera intrinsics from cameras.txt
    camera_id = i_cam  # COLMAP camera IDs start from 0
    camera_found = False
    
    with open(cameras_txt_path, 'r') as f:
        for line in f:
            # Skip comments and empty lines
            if line.startswith('#') or line.strip() == '':
                continue
            
            parts = line.strip().split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))
            
            if cam_id == camera_id:
                # Verify camera model is PINHOLE
                assert model == 'PINHOLE', f"Expected PINHOLE camera model, got {model}"
                
                # Extract parameters: fx, fy, cx, cy
                fx, fy, cx, cy = params[:4]
                
                # Verify assumptions
                assert np.isclose(fx, fy, rtol=1e-5), f"Expected fx == fy, got fx={fx}, fy={fy}"
                assert np.isclose(cx, width / 2.0, rtol=1e-2), f"Expected cx == width/2, got cx={cx}, width/2={width/2.0}"
                assert np.isclose(cy, height / 2.0, rtol=1e-2), f"Expected cy == height/2, got cy={cy}, height/2={height/2.0}"
                
                # Calculate camera_angle_x (horizontal field of view)
                camera_angle_x = 2 * np.arctan(width / (2 * fx))
                camera_found = True
                break
    
    assert camera_found, f"Camera ID {camera_id} not found in {cameras_txt_path}"
    
    # Load camera extrinsics from images.txt
    image_found = False
    
    with open(images_txt_path, 'r') as f:
        lines = f.readlines()
    
    # Parse images.txt (every image entry takes 2 lines)
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Skip comments and empty lines
        if line.startswith('#') or line.strip() == '':
            i += 1
            continue
        
        parts = line.strip().split()
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        cam_id = int(parts[8])
        
        if cam_id == camera_id:
            q = np.array([qw, qx, qy, qz])
            
            R = np.array([
                [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
                [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
            ])
            
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = R
            transform_matrix[:3, 3] = [tx, ty, tz]
            
            image_found = True
            break
        
        i += 2
    
    assert image_found, f"No image found for camera ID {camera_id} in {images_txt_path}"
    
    cam_data_colmap = {
        'camera_angle_x': camera_angle_x,
        'transform_matrix': transform_matrix
    }
    
    return cam_data_colmap

def convert_colmap_to_nerf(cam_data_colmap):
    camera_angle_x = cam_data_colmap['camera_angle_x']
    w2c_colmap = cam_data_colmap['transform_matrix']

    c2w_colmap = np.linalg.inv(w2c_colmap)
    
    c2w_nerf = c2w_colmap.copy()
    c2w_nerf[:3, 1:3] *= -1
    
    return {
        "camera_angle_x": float(camera_angle_x),
        "transform_matrix": c2w_nerf.tolist(),
    }

if __name__ == "__main__":
    scene_id = 11
    train_split = 0.8

    time_start_s = 13.0
    time_end_s = 13.3

    input_video_dir = "data/input_video"
    video_paths = sorted(glob.glob(os.path.join(input_video_dir, f"{scene_id:04d}_*.mkv")))
    num_views = len(video_paths)
    print(f"Found {len(video_paths)} videos")
    for video_path in video_paths:
        print(f"  - {video_path}")

    time_range = None
    for video_path in video_paths:
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        time_range_path = os.path.join("data/time_range", f"{video_basename}.time_range.npz")
        if os.path.exists(time_range_path):
            print(f"Found time range for {video_path}")
            time_range_data = np.load(time_range_path)
            time_start_s = time_range_data["time_start"] * 1e-6
            time_end_s = time_range_data["time_end"] * 1e-6

    output_dir = os.path.join("data/dnerf_output", f"{scene_id:04d}_dnerf_temporal")
    os.makedirs(output_dir, exist_ok=True)

    # Copy colmap point cloud
    point_cloud_colmap_path = os.path.join("data/colmap", f"{scene_id:04d}", "sparse", "0", "points3D.ply")
    shutil.copy(point_cloud_colmap_path, os.path.join(output_dir, "colmap_point_cloud.ply"))

    # Find undistorted videos
    undistorted_video_dir = "data/undistorted_video"
    video_paths = sorted(glob.glob(os.path.join(undistorted_video_dir, f"{scene_id:04d}_*.undistorted.mkv")))
    num_views = len(video_paths)
    print(f"Found {len(video_paths)} undistorted videos")

    # Load times
    times_paths = [os.path.join("data/sync", f"{scene_id:04d}_{i_cam}.times.npz") for i_cam in range(num_views)]
    times_data = [np.load(time_path) for time_path in times_paths]
    dt_ms = np.nanmedian([np.nanmedian(np.diff(times_data[i_cam]['times'])) for i_cam in range(num_views)]) * 1e-3
    print(f"Median time step: {dt_ms:.2f} ms")

    # Find static gaussians
    static_gaussians_path = os.path.join("data/colmap", f"{scene_id:04d}_static_point_cloud.ply")
    assert os.path.exists(static_gaussians_path), f"Static gaussians not found at {static_gaussians_path}"
    shutil.copy(static_gaussians_path, os.path.join(output_dir, "static_point_cloud.ply"))

    # Find sync masks
    os.makedirs(os.path.join(output_dir, "sync_masks"), exist_ok=True)
    sync_masks = []
    for i_cam in range(num_views):
        sync_masks_path = os.path.join("data/sync_mask", f"{scene_id:04d}_{i_cam}.sync_mask.png")
        shutil.copy(sync_masks_path, os.path.join(output_dir, "sync_masks", f"{i_cam}.png"))
        sync_masks.append(cv2.imread(sync_masks_path, cv2.IMREAD_GRAYSCALE) > 0)

    # Load camera data from COLMAP
    colmap_dir = os.path.join("data/colmap", f"{scene_id:04d}")
    camera_data = [load_colmap_camera(colmap_dir, i_cam) for i_cam in range(num_views)]  
    camera_data_nerf = [convert_colmap_to_nerf(cam_data) for cam_data in camera_data]

    frame_ranges = []

    frame_reference_start = np.nanargmin(np.abs(times_data[0]['times'] - time_start_s * 1e6))
    frame_reference_end = np.nanargmin(np.abs(times_data[0]['times'] - time_end_s * 1e6))

    start_time_reference = times_data[0]['times'][frame_reference_start]
    start_frames = [np.nanargmin(np.abs(times_data[i_cam]['times'] - start_time_reference)) for i_cam in range(num_views)]

    num_frames = frame_reference_end - frame_reference_start
    caps = [cv2.VideoCapture(video_paths[i_cam]) for i_cam in range(num_views)]

    for i_cam in range(num_views):
        caps[i_cam].set(cv2.CAP_PROP_POS_FRAMES, start_frames[i_cam])

    for subset in ["train", "test"]:
        os.makedirs(os.path.join(output_dir, subset), exist_ok=True)

    # Create masks directory before writing masks in the loop
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    meta = {
        "train": [],
        "test": []
    }

    for i_frame_reference in tqdm.tqdm(range(frame_reference_start, frame_reference_end), desc="Processing frames"):
        time_reference = times_data[0]['times'][i_frame_reference]
        frame_indices = [np.nanargmin(np.abs(times_data[i_cam]['times'] - time_reference)) for i_cam in range(num_views)]
        frame_times = [times_data[i_cam]['times'][frame_indices[i_cam]] for i_cam in range(num_views)]
        frames = []
        flows = []
        frame_within_dataset = i_frame_reference - frame_reference_start
        for i_cam in range(num_views):
            caps[i_cam].set(cv2.CAP_PROP_POS_FRAMES, frame_indices[i_cam])
            _, frame = caps[i_cam].read()
            frames.append(frame)
            flow_path = os.path.join("data/flow", f"{scene_id:04d}_{i_cam}", f"{frame_indices[i_cam]:06d}.png")
            flow_image = cv2.imread(flow_path)
            flow = flow_image[..., :2] / 255.0
            min_flow = -25.0
            max_flow = 25.0
            flow = flow * (max_flow - min_flow) + min_flow

            flow_mask = np.linalg.norm(flow, axis=-1) > 0.5
            circular_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            flow_mask_dilated = cv2.dilate((flow_mask * 255).astype(np.uint8), circular_kernel, iterations=2) > 0

            mask_combined = np.zeros_like(frame)

            mask_combined[flow_mask, 1] = 255
            mask_combined[flow_mask_dilated, 0] = 255
            mask_combined[sync_masks[i_cam], :] = [0, 0, 255]

            mask_path = os.path.join(output_dir, "masks", f"{frame_within_dataset:04d}_{i_cam}.png")
            cv2.imwrite(mask_path, mask_combined)
        
        for i_cam in range(num_views):
            target_set = "test" if frame_within_dataset % 8 == 0 else "train"
            target_path = os.path.join(output_dir, target_set, f"{frame_within_dataset:04d}_{i_cam}.png")
            cv2.imwrite(target_path, frames[i_cam])

            meta[target_set].append({
                "file_path": f"./{target_set}/{frame_within_dataset:04d}_{i_cam}",
                "camera_angle_x": camera_data_nerf[i_cam]['camera_angle_x'],
                "transform_matrix": camera_data_nerf[i_cam]['transform_matrix'],
                "time": frame_times[i_cam] * 1e-3,
                "frame": int(frame_within_dataset),
                "subset": target_set
            })

    for cap in caps:
        cap.release()

    for subset in meta.keys():
        with open(os.path.join(output_dir, f"transforms_{subset}.json"), "w") as f:
            json.dump({ "camera_angle_x": camera_data_nerf[0]['camera_angle_x'], "dt": float(dt_ms), "frames": meta[subset] }, f, indent=4)

    os.makedirs(os.path.join(output_dir, "3d_flow"), exist_ok=True)
    for i_frame_reference in tqdm.tqdm(range(frame_reference_start, frame_reference_end), desc="Processing 3D flow and masks"):
        i_frame_in_dataset = i_frame_reference - frame_reference_start
        time_reference = times_data[0]['times'][i_frame_reference]
        frame_indices = [np.argmin(np.abs(times_data[i_cam]['times'] - time_reference)) for i_cam in range(num_views)]

        for i_cam in range(num_views):
            flow_path = os.path.join("data/flow_3d", f"{scene_id:04d}_{i_cam}", f"{frame_indices[i_cam]:06d}.npz")
            if os.path.exists(flow_path):
                shutil.copy(flow_path, os.path.join(output_dir, "3d_flow", f"{i_frame_in_dataset:04d}_{i_cam}.npz"))

            mask_path = os.path.join("data/masks", f"{scene_id:04d}_{i_cam}", f"{frame_indices[i_cam]:06d}.png")
            if os.path.exists(mask_path):
                shutil.copy(mask_path, os.path.join(output_dir, "masks", f"{i_frame_in_dataset:04d}_{i_cam}.png"))

    os.makedirs(os.path.join(output_dir, "voxel_flow", "voxel_flow"), exist_ok=True)
    voxel_flow_metadata_path = os.path.join("data/voxel_projection", f"{scene_id:04d}", "metadata.npz")
    if os.path.exists(voxel_flow_metadata_path):
        voxel_flow_metadata = np.load(voxel_flow_metadata_path)
        flow_times = voxel_flow_metadata["times"]
        flow_times_out = []
        num_saved_flow_frames = 0
        for i_frame_reference in tqdm.tqdm(range(frame_reference_start, frame_reference_end), desc="Processing voxel flow"):
            i_frame_in_dataset = i_frame_reference - frame_reference_start
            if i_frame_in_dataset % 8 not in [7, 0]:
                time_reference = times_data[0]['times'][i_frame_reference]
                flow_frame_index = np.argmin(np.abs(flow_times - time_reference))
                flow_time = flow_times[flow_frame_index]
                flow_times_out.append(flow_time)
                source_path = os.path.join("data/voxel_projection", f"{scene_id:04d}", "voxel_flow", f"{flow_frame_index:06d}.npz")
                target_path = os.path.join(output_dir, "voxel_flow", "voxel_flow", f"{num_saved_flow_frames:06d}.npz")
                shutil.copy(source_path, target_path)
                num_saved_flow_frames += 1

        voxel_flow_metadata_out = {key: voxel_flow_metadata[key] for key in voxel_flow_metadata if key != "times"}
        voxel_flow_metadata_out["times"] = np.array(flow_times_out)
        voxel_flow_metadata_out["dt"] = float(dt_ms)
        np.savez(os.path.join(output_dir, "voxel_flow", "metadata.npz"), **voxel_flow_metadata_out)

    # Copy rolling shutter specs
    os.makedirs(os.path.join(output_dir, "rolling_shutter"), exist_ok=True)
    for i_cam in range(num_views):
        rolling_shutter_path = os.path.join("data/rolling_shutter", f"{scene_id:04d}_{i_cam}.rolling_shutter.npy")
        if os.path.exists(rolling_shutter_path):
            shutil.copy(rolling_shutter_path, os.path.join(output_dir, "rolling_shutter", f"{i_cam}.npy"))

        

