import cv2
import numpy as np
import glob
import os
import subprocess
import tempfile
from pathlib import Path
import pycolmap as pcm
import tqdm
from inspect_table import insert_matches
from calibration_utils import get_camera_serial_number, load_calibration_data

def run_dense_reconstruction(rec, image_dir, out_dir, video_paths):
    print("\n=== Starting Dense Reconstruction ===")

    dense_dir = out_dir / "dense"
    dense_dir.mkdir(exist_ok=True, parents=True)

    # Undistort images for dense reconstruction (COLMAP requires this step)
    print("Step 1: Undistorting images for dense reconstruction...")
    undistort_dir = dense_dir
    sparse_dir = out_dir / "00"  # Assuming reconstruction 0

    # Use pycolmap to undistort imagesstereo_options
    pcm.undistort_images(
        output_path=str(undistort_dir),
        input_path=str(sparse_dir),
        image_path=str(image_dir),
    )

    # Step 2: Patch match stereo (compute depth maps)
    print("Step 2: Computing depth maps with patch match stereo...")
    stereo_dir = dense_dir / "stereo"
    stereo_dir.mkdir(exist_ok=True, parents=True)

    stereo_options = pcm.PatchMatchOptions()

    pcm.patch_match_stereo(
        workspace_path=str(dense_dir),
        workspace_format="COLMAP",
        pmvs_option_name="option-all",
        options=stereo_options,
    )

    # Export depth maps for each camera
    print("\nExporting depth maps...")
    depth_maps_dir = dense_dir / "stereo" / "depth_maps"
    if depth_maps_dir.exists():
        # Create mapping from image names to video paths
        img_name_to_video = {}
        for i, img in enumerate(rec.images.values()):
            if i < len(video_paths):
                img_name_to_video[img.name] = video_paths[i]

        # Export geometric depth maps (more reliable than photometric)
        for depth_file in sorted(depth_maps_dir.glob("*.geometric.bin")):
            # Extract image name (e.g., "cam00.png" from "cam00.png.geometric.bin")
            img_name = depth_file.name.replace(".geometric.bin", "")

            if img_name in img_name_to_video:
                video_path = img_name_to_video[img_name]
                video_basename = os.path.splitext(os.path.basename(video_path))[0]
                os.makedirs("data/stereo_depth", exist_ok=True)
                output_path = os.path.join("data/stereo_depth", f"{video_basename}.stereo_depth.npy")

                try:
                    # Read depth map
                    depth_map = read_colmap_depth_map(str(depth_file))

                    # Save as numpy array
                    np.save(output_path, depth_map)

                    # Print statistics
                    valid_mask = depth_map > 0
                    valid_count = valid_mask.sum()
                    total_pixels = depth_map.size
                    coverage = 100.0 * valid_count / total_pixels

                    print(f"  Saved {img_name} depth map to {output_path}")
                    print(f"    Coverage: {valid_count}/{total_pixels} pixels ({coverage:.1f}%)")

                    if valid_count > 0:
                        valid_depths = depth_map[valid_mask]
                        print(f"    Depth range: [{valid_depths.min():.2f}, {valid_depths.max():.2f}], "
                              f"mean: {valid_depths.mean():.2f}")
                except Exception as e:
                    print(f"  Error exporting depth map for {img_name}: {e}")
    else:
        print("Warning: Depth maps directory not found!")

    # Step 3: Stereo fusion (merge depth maps into point cloud)
    print("\nStep 3: Fusing depth maps into point cloud...")
    fusion_options = pcm.StereoFusionOptions()
    pcm.stereo_fusion(
        output_path=str(dense_dir / "fused.ply"),
        workspace_path=str(dense_dir),
        workspace_format="COLMAP",
        input_type="geometric",
        options=fusion_options,
    )

    fused_ply = dense_dir / "fused.ply"
    if fused_ply.exists():
        print(f"\nDense reconstruction saved to: {fused_ply}")
        return str(fused_ply)
    else:
        print("Warning: Dense reconstruction failed to produce output file.")
        return None


def read_colmap_depth_map(depth_map_path):
    # Read COLMAP depth map binary file
    with open(depth_map_path, "rb") as f:
        # Read header as text until we find the dimensions
        header = b""
        while True:
            byte = f.read(1)
            if not byte:
                raise ValueError("Unexpected end of file while reading header")
            header += byte
            if header.endswith(b"&"):
                # Check if we have all three values
                parts = header.decode('ascii').strip('&').split('&')
                if len(parts) == 3:
                    width, height, channels = map(int, parts)
                    break

        num_pixels = width * height * channels
        depth_data = np.fromfile(f, dtype=np.float32, count=num_pixels)

        if len(depth_data) != num_pixels:
            print(f"Warning: Expected {num_pixels} pixels but read {len(depth_data)}")

        # Reshape to image dimensions
        if channels == 1:
            depth_map = depth_data.reshape(height, width)
        else:
            depth_map = depth_data.reshape(height, width, channels)

        return depth_map


if __name__ == "__main__":
    input_dir = "data/input_video"
    input_id = 11
    dense_exclude_camera_ids = []

    video_paths = sorted(glob.glob(os.path.join(input_dir, f"{input_id:04d}_[0-9].mkv")))
    num_cameras = len(video_paths)
    print(f"Found {num_cameras} videos.")

    frames = []
    calibration_data = []
    num_frames = 3
    use_max_frame = True

    # Check if min frames are available
    min_frames = [None] * num_cameras
    use_min_frame = True
    for i_cam in range(num_cameras):
        # Get base filename without directory (e.g., "0011_0.MP4" -> "0011_0")
        video_basename = os.path.splitext(os.path.basename(video_paths[i_cam]))[0]
        min_frame_path = os.path.join("data/min_frame", f"{video_basename}.min_frame.png")
        if os.path.exists(min_frame_path):
            min_frames[i_cam] = cv2.imread(min_frame_path)
        else:
            use_min_frame = False

    marker_data = []
    strip_data = []
    if use_min_frame:
        for i_cam in range(num_cameras):
            frames.append(min_frames[i_cam])
            serial_number = get_camera_serial_number(video_paths[i_cam])
            calibration_data.append(load_calibration_data(serial_number))

            # Detect known points in all max frames
            video_basename = os.path.splitext(os.path.basename(video_paths[i_cam]))[0]
            marker_path = os.path.join("data/detected_pattern", f"{video_basename}.marker.npz")
            data_loaded = np.load(marker_path)
            marker_data.append(np.concatenate([
                data_loaded["marker_points"],
                data_loaded["strip_lines_fine"][:, 0, :],
                data_loaded["strip_lines_fine"][:, 1, :]
            ], axis=0))
    elif use_max_frame:
        times = []
        for i_cam in range(num_cameras):
            video_basename = os.path.splitext(os.path.basename(video_paths[i_cam]))[0]
            times_path = os.path.join("data/sync", f"{video_basename}.times.npz")
            times_data = np.load(times_path)["times"]
            times.append(times_data)

        time_start = np.max([np.nanmin(times_data) for times_data in times])
        print(f"Time start: {time_start*1e-3:.2f} ms")

        for i_cam, video_path in enumerate(video_paths):
            frame_start = np.nanargmin(np.abs(times[i_cam] - time_start))
            print(f"Frame start: {frame_start}")
            frame_max = None
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
            if use_max_frame:
                for i in range(num_frames):
                    ret, frame = cap.read()
                    if frame_max is None:
                        frame_max = frame.copy()
                    else:
                        frame_max = np.max([frame_max, frame], axis=0)
            cap.release()
            assert ret, f"Failed to read frame from {video_path}"
            frames.append(frame_max)
            serial_number = get_camera_serial_number(video_path)
            calibration_data.append(load_calibration_data(serial_number))

            # Detect known points in all max frames
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            marker_path = os.path.join("data/detected_pattern", f"{video_basename}.marker.npz")
            data_loaded = np.load(marker_path)
            marker_data.append(np.concatenate([
                data_loaded["marker_points"],
                data_loaded["strip_lines_fine"][:, 0, :],
                data_loaded["strip_lines_fine"][:, 1, :]
            ], axis=0))
    else:
        # Load times data
        times = []
        for i_cam in range(num_cameras):
            video_basename = os.path.splitext(os.path.basename(video_paths[i_cam]))[0]
            times_path = os.path.join("data/sync", f"{video_basename}.times.npz")
            times_data = np.load(times_path)["times"]
            times.append(times_data)
        i_frame_reference = len(times[0]) // 2
        time_reference = times[0][i_frame_reference]
        for i_cam in range(num_cameras):
            i_frame = np.argmin(np.abs(times[i_cam] - time_reference))
            cap = cv2.VideoCapture(video_paths[i_cam])
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
            _, frame = cap.read()
            cap.release()
            frames.append(frame)
            serial_number = get_camera_serial_number(video_paths[i_cam])
            calibration_data.append(load_calibration_data(serial_number))
            
            video_basename = os.path.splitext(os.path.basename(video_paths[i_cam]))[0]
            marker_path = os.path.join("data/detected_pattern", f"{video_basename}.marker.npz")
            data_loaded = np.load(marker_path)
            marker_data.append(np.concatenate([
                data_loaded["marker_points"],
                data_loaded["strip_lines_fine"][:, 0, :],
                data_loaded["strip_lines_fine"][:, 1, :]
            ], axis=0))


    # Create temporary COLMAP workspace
    with tempfile.TemporaryDirectory(prefix="colmap_tmp_", delete=False) as tmpdir:
        tmp = Path(tmpdir)
        image_dir = tmp / "images"
        db_path = tmp / "database.db"
        out_dir = tmp / "sparse"
        image_dir.mkdir()
        # Create an empty COLMAP database before import_images
        open(db_path, 'a').close()

        # Undistort frames and save to temporary folder
        for i, (frame, calib) in enumerate(zip(frames, calibration_data)):
            K = calib["camera_matrix"]
            dist = calib["dist_coeffs"]
            img_size = tuple(calib["image_size"])

            # Compute optimal new camera matrix (no cropping)
            newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, img_size, 0)
            undist = cv2.undistort(frame, K, dist, None, newK)

            fname = image_dir / f"cam{i:02d}.png"
            cv2.imwrite(str(fname), undist)
            
            # Also export frame to data/max_frame/
            video_basename = os.path.splitext(os.path.basename(video_paths[i]))[0]
            os.makedirs("data/max_frame", exist_ok=True)
            export_path = os.path.join("data/max_frame", f"{video_basename}.max_frame.png")
            cv2.imwrite(export_path, frame)
            
            calibration_data[i] = {"K": newK, "image_size": img_size}  # update intrinsics

            # Undistort marker points
            newK_nocrop, roi_nocrop = cv2.getOptimalNewCameraMatrix(K, dist, img_size, 1)
            
            marker_h = np.concatenate([marker_data[i], np.ones((marker_data[i].shape[0], 1))], axis=1)
            marker_warped = (newK @ np.linalg.inv(newK_nocrop) @ marker_h.T).T[:, :-1]
            marker_data[i] = marker_warped

            print(f"  Processed cam{i:02d}.png with {len(marker_data[i])} marker points")

        # Import images with undistorted intrinsics (PINHOLE)
        # Each image may have different K, so import per-image.
        for i in range(num_cameras):
            name = f"cam{i:02d}.png"
            K_i = calibration_data[i]["K"]
            fx, fy, cx, cy = K_i[0, 0], K_i[1, 1], K_i[0, 2], K_i[1, 2]

            # For pycolmap.import_images, supply options as a dict
            image_opts = {
                "camera_model": "PINHOLE",
                "camera_params": f"{fx},{fy},{cx},{cy}",
            }

            pcm.import_images(
                str(db_path),
                str(image_dir),
                pcm.CameraMode.PER_IMAGE,
                [name],
                image_opts,
            )

        # Feature extraction & exhaustive matching
        import multiprocessing
        num_threads = multiprocessing.cpu_count()
        print(f"Using {num_threads} CPU threads for feature extraction and matching")
        
        # Create SIFT extraction options - increase feature count
        sift_extract_opts = pcm.SiftExtractionOptions()
        sift_extract_opts.max_num_features = 16384
        sift_extract_opts.first_octave = -1
        
        extract_opts = pcm.FeatureExtractionOptions()
        extract_opts.num_threads = num_threads
        extract_opts.sift = sift_extract_opts
        
        print("Extracting features...")
        pcm.extract_features(str(db_path), str(image_dir), extraction_options=extract_opts)

        # Configure matching - use defaults for reliability
        print("Matching features exhaustively...")
        pcm.match_exhaustive(str(db_path))

        # Insert matches
        insert_matches(marker_data, str(db_path))
        
        # Verify database contents before mapping
        print("\n=== Database Verification ===")
        import sqlite3
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()
            
            # Count images
            cursor.execute("SELECT COUNT(*) FROM images")
            num_images = cursor.fetchone()[0]
            
            # Count cameras
            cursor.execute("SELECT COUNT(*) FROM cameras")
            num_cameras = cursor.fetchone()[0]
            
            print(f"Database contains {num_images} images and {num_cameras} cameras")
            
            # Check matches from two_view_geometries table
            cursor.execute("SELECT pair_id, rows FROM two_view_geometries")
            pairs = cursor.fetchall()
            
            total_matches = sum(rows for _, rows in pairs)
            print(f"Total image pairs with matches: {len(pairs)}")
            print(f"Total matches across all pairs: {total_matches}")
            
            if total_matches == 0:
                print("WARNING: No matches found in database! Pose registration will fail.")
            elif total_matches < 100:
                print("WARNING: Few matches found, but we have manual marker matches.")
            else:
                print(f"SUCCESS: Found {total_matches} matches total - ready for mapping!")
        print()

        # Incremental mapping
        mapper_options = pcm.IncrementalPipelineOptions()
        mapper_options.ba_refine_focal_length = False
        mapper_options.ba_refine_principal_point = False
        mapper_options.ba_refine_extra_params = False
        mapper_options.num_threads = num_threads
        mapper_options.min_model_size = 2
        mapper_options.min_num_matches = 5
        mapper_options.multiple_models = True
        mapper_options.max_num_models = 50
        mapper_options.mapper.init_min_num_inliers = 10
        mapper_options.mapper.init_max_error = 16.0
        mapper_options.mapper.init_max_forward_motion = 0.99
        mapper_options.mapper.init_min_tri_angle = 0.5
        mapper_options.mapper.init_max_reg_trials = 10
        mapper_options.mapper.abs_pose_max_error = 16.0
        mapper_options.mapper.abs_pose_min_num_inliers = 10
        mapper_options.mapper.abs_pose_min_inlier_ratio = 0.10
        mapper_options.mapper.filter_max_reproj_error = 16.0
        mapper_options.mapper.filter_min_tri_angle = 0.5
        mapper_options.triangulation.min_angle = 0.5
        mapper_options.triangulation.ignore_two_view_tracks = False 
        
        recs = pcm.incremental_mapping(
            str(db_path),
            str(image_dir),
            str(out_dir),
            mapper_options,
        )

        # Inspect results
        if len(recs) == 0:
            print("No reconstruction created.")
        else:
            for rid, rec in recs.items():
                print(f"=== Reconstruction {rid} summary ===")
                print(f"{len(rec.images)} registered images")
                print(f"{len(rec.points3D)} triangulated points")

                # Calculate reprojection errors
                total_errors = []
                total_points = 0
                
                for i_img, (img_id, img) in enumerate(rec.images.items()):
                    print(f"Image: {img.name}")
                    # Get camera pose (cam_from_world transformation matrix)
                    cam_from_world = img.cam_from_world()
                    # Extract rotation and translation from the Rigid3d object
                    R = cam_from_world.rotation.matrix()
                    t = cam_from_world.translation
                    print("Rotation:", R)
                    print("Translation:", t, "")

                    # Extract triangulated points for this image
                    points3D_for_image = []
                    for point2D_idx, point2D in enumerate(img.points2D):
                        if point2D.point3D_id != -1 and point2D.point3D_id in rec.points3D:
                            point3D = rec.points3D[point2D.point3D_id]
                            points3D_for_image.append(point3D.xyz)
                    
                    points3D_for_image = np.array(points3D_for_image) if points3D_for_image else np.array([]).reshape(0, 3)
                    
                    # Save camera pose and triangulated points to file
                    video_basename = os.path.splitext(os.path.basename(video_paths[i_img]))[0]
                    os.makedirs("data/pose", exist_ok=True)
                    output_path = os.path.join("data/pose", f"{video_basename}.pose.npz")
                    np.savez(output_path, R=R, t=t, points3D=points3D_for_image)
                    print(f"Saved camera pose and {len(points3D_for_image)} triangulated points to {output_path}")

                    # Calculate reprojection errors for this image
                    camera = rec.cameras[img.camera_id]
                    img_errors = []
                    
                    for point2D_idx, point2D in enumerate(img.points2D):
                        if point2D.point3D_id != -1 and point2D.point3D_id in rec.points3D:  # If this 2D point has a corresponding 3D point
                            point3D = rec.points3D[point2D.point3D_id]
                            
                            # Project 3D point to 2D
                            point3D_cam = R @ point3D.xyz + t
                            if point3D_cam[2] > 0:  # Point is in front of camera
                                # Project to image plane
                                x_proj = point3D_cam[0] / point3D_cam[2]
                                y_proj = point3D_cam[1] / point3D_cam[2]
                                
                                # Apply camera intrinsics
                                fx, fy, cx, cy = camera.params[:4]
                                u_proj = fx * x_proj + cx
                                v_proj = fy * y_proj + cy
                                
                                # Calculate reprojection error
                                error = np.sqrt((point2D.xy[0] - u_proj)**2 + (point2D.xy[1] - v_proj)**2)
                                img_errors.append(error)
                                total_errors.append(error)
                                total_points += 1
                    
                    if img_errors:
                        mean_error = np.mean(img_errors)
                        print(f"Mean reprojection error for {img.name}: {mean_error:.4f} pixels")
                        print(f"Number of 3D points in {img.name}: {len(img_errors)}")
                    else:
                        print(f"No 3D points found for {img.name}")
                    print()
                
                if total_errors:
                    overall_mean_error = np.mean(total_errors)
                    overall_std_error = np.std(total_errors)
                    print(f"=== Overall Reprojection Error Statistics ===")
                    print(f"Total 3D points: {total_points}")
                    print(f"Mean reprojection error: {overall_mean_error:.4f} pixels")
                    print(f"Std reprojection error: {overall_std_error:.4f} pixels")
                    print(f"Max reprojection error: {np.max(total_errors):.4f} pixels")
                    print(f"Min reprojection error: {np.min(total_errors):.4f} pixels")
                else:
                    print("No reprojection errors could be calculated.")

                out_sub = out_dir / f"{rid:02d}"
                out_sub.mkdir(exist_ok=True, parents=True)
                rec.write(out_sub)
                print(f"Results written to: {out_sub}")
                
                # Dense reconstruction
                print("\n=== Starting Dense Reconstruction ===")
                ply_path = run_dense_reconstruction(rec, image_dir, out_dir, video_paths)
