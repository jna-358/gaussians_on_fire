import numpy as np
import glob
import os
import cv2
from calibration_utils import get_camera_serial_number, load_calibration_data


def rotate_camera_intrinsics_90ccw(K, image_size):
    """
    Rotate camera intrinsics matrix for 90 degree counterclockwise rotation.
    
    Args:
        K: 3x3 camera intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        image_size: (width, height) of original image
    
    Returns:
        K_rotated: New intrinsic matrix
        new_image_size: (new_width, new_height) after rotation
    """
    width, height = image_size
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # After 90 degree CCW rotation:
    # - Image dimensions swap: (W, H) -> (H, W)
    # - In continuous coordinates: (x, y) -> (y, W - x)
    # - Principal point transforms accordingly
    # - Focal lengths swap (fx maps to y-axis, fy maps to x-axis)
    
    new_width = height
    new_height = width
    
    # New principal point in continuous coordinates: (cx, cy) -> (cy, W - cx)
    # Note: We use W (not W-1) because camera intrinsics use continuous coordinates,
    # not discrete pixel indices
    cx_new = cy
    cy_new = width - cx
    
    # Focal lengths swap
    fx_new = fy
    fy_new = fx
    
    K_rotated = np.array([
        [fx_new, 0, cx_new],
        [0, fy_new, cy_new],
        [0, 0, 1]
    ], dtype=np.float64)
    
    new_image_size = (new_width, new_height)
    
    return K_rotated, new_image_size


def rotate_camera_extrinsics_90ccw(R, t):
    R_old_to_new = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ], dtype=np.float64)
    
    R_rotated = R_old_to_new @ R
    t_rotated = R_old_to_new @ t
    
    return R_rotated, t_rotated


def rotate_camera_intrinsics_90cw(K, image_size):
    width, height = image_size
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    new_width = height
    new_height = width

    cx_new = height - cy
    cy_new = cx

    fx_new = fy
    fy_new = fx
    
    K_rotated = np.array([
        [fx_new, 0, cx_new],
        [0, fy_new, cy_new],
        [0, 0, 1]
    ], dtype=np.float64)
    
    new_image_size = (new_width, new_height)
    
    return K_rotated, new_image_size


def rotate_camera_extrinsics_90cw(R, t):
    R_old_to_new = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ], dtype=np.float64)
    
    R_rotated = R_old_to_new @ R
    t_rotated = R_old_to_new @ t
    
    return R_rotated, t_rotated


def process_video(video_path, cropping=50, rotation=cv2.ROTATE_90_COUNTERCLOCKWISE):
    print(f"\nProcessing: {video_path}")
    
    # Determine rotation direction
    is_ccw = (rotation == cv2.ROTATE_90_COUNTERCLOCKWISE)
    rotation_name = "90° CCW" if is_ccw else "90° CW"
    print(f"  Rotation: {rotation_name}")
    
    # Select appropriate rotation functions
    if is_ccw:
        rotate_intrinsics = rotate_camera_intrinsics_90ccw
        rotate_extrinsics = rotate_camera_extrinsics_90ccw
        depth_rotation_k = 1
    else:
        rotate_intrinsics = rotate_camera_intrinsics_90cw
        rotate_extrinsics = rotate_camera_extrinsics_90cw
        depth_rotation_k = -1
    
    serial_number = get_camera_serial_number(video_path)
    calib_data = load_calibration_data(serial_number)
    if calib_data is None:
        print(f"  Skipping: Could not load calibration data")
        return False
    
    camera_matrix = calib_data["camera_matrix"]
    dist_coeffs = calib_data["dist_coeffs"]
    image_size = tuple(calib_data["image_size"])
    
    print(f"  Original image size: {image_size}")
    print(f"  Original camera matrix:\n{camera_matrix}")
    
    camera_matrix_optimal, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, image_size, 0
    )
    
    print(f"  Optimal camera matrix (undistorted):\n{camera_matrix_optimal}")
    
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    pose_path = os.path.join("data/pose", f"{video_basename}.pose.npz")
    if not os.path.exists(pose_path):
        print(f"  Skipping: Pose file not found: {pose_path}")
        return False
    
    pose_data = np.load(pose_path)
    print(f"  Pose data: {list(pose_data.keys())}")
    R = pose_data['R']
    t = pose_data['t']
    points3D = pose_data['points3D']

    frame = None
    
    max_frame_path = os.path.join("data/max_frame", f"{video_basename}.max_frame.png")
    frame = cv2.imread(max_frame_path)
    
    frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, camera_matrix_optimal)
    
    print(f"  Original rotation matrix:\n{R}")
    print(f"  Original translation: {t.flatten()}")
    K_rotated, new_image_size = rotate_intrinsics(camera_matrix_optimal, image_size)
    
    print(f"  Rotated image size: {new_image_size}")
    print(f"  Rotated camera matrix:\n{K_rotated}")
    
    # Rotate extrinsics
    R_rotated, t_rotated = rotate_extrinsics(R, t)
    
    print(f"  Rotated rotation matrix:\n{R_rotated}")
    print(f"  Rotated translation: {t_rotated.flatten()}")
    
    # Save rotated parameters
    os.makedirs("data/rotated_cams", exist_ok=True)
    output_path = os.path.join("data/rotated_cams", f"{video_basename}.rotated_cams.npz")
    np.savez(
        output_path,
        camera_matrix=K_rotated,
        dist_coeffs=dist_coeffs,
        image_size=np.array(new_image_size),
        R=R_rotated,
        t=t_rotated,
        points3D=points3D,
        rotation_direction=rotation
    )

    # Save rotated image
    frame_rotated = cv2.rotate(frame, rotation)
    os.makedirs("data/rotated_undistorted", exist_ok=True)
    rotated_img_path = os.path.join("data/rotated_undistorted", f"{video_basename}.rotated_undistorted.png")
    cv2.imwrite(rotated_img_path, frame_rotated)
    print(f"  Saved rotated undistorted image to: {rotated_img_path}")

    # Save rotated depth
    depth_path = os.path.join("data/stereo_depth", f"{video_basename}.stereo_depth.npy")
    stereo_depth = np.load(depth_path)
    stereo_depth_rotated = np.rot90(stereo_depth, k=depth_rotation_k)
    os.makedirs("data/stereo_depth_rotated", exist_ok=True)
    rotated_depth_path = os.path.join("data/stereo_depth_rotated", f"{video_basename}.stereo_depth_rotated.npy")
    np.save(rotated_depth_path, stereo_depth_rotated)
    print(f"  Saved rotated depth to: {rotated_depth_path}")

    # Save cropped image and depth
    os.makedirs("data/stereo_depth_rotated_cropped", exist_ok=True)
    os.makedirs("data/rotated_undistorted_cropped", exist_ok=True)
    cropped_depth_path = os.path.join("data/stereo_depth_rotated_cropped", f"{video_basename}.stereo_depth_rotated_cropped.npy")
    cropped_img_path = os.path.join("data/rotated_undistorted_cropped", f"{video_basename}.rotated_undistorted_cropped.png")
    np.save(cropped_depth_path, stereo_depth_rotated[cropping:-cropping, cropping:-cropping])
    cv2.imwrite(cropped_img_path, frame_rotated[cropping:-cropping, cropping:-cropping])
    print(f"  Saved cropped rotated undistorted image to: {cropped_img_path}")
    print(f"  Saved cropped rotated depth to: {cropped_depth_path}")

    print(f"  Saved to: {output_path}")
    return True


if __name__ == "__main__":
    # Default parameters
    input_dir = "data/input_video"
    input_id = 11
    rotation = cv2.ROTATE_90_CLOCKWISE
    
    # Find all matching videos
    video_paths = sorted(glob.glob(os.path.join(input_dir, f"{input_id:04d}_*.mkv")))
    
    if not video_paths:
        print(f"No videos found matching pattern: {input_dir}/{input_id:04d}_*.mkv")
        exit(1)
    
    print(f"Found {len(video_paths)} videos to process:")
    for vp in video_paths:
        print(f"  - {vp}")
    
    # Process each video
    success_count = 0
    for video_path in video_paths:
        if process_video(video_path, rotation=rotation):
            success_count += 1
    
    print(f"\n=== Summary ===")
    print(f"Successfully processed {success_count}/{len(video_paths)} videos")
    print(f"Output files saved as *.rotated_cams.npz")

