import cv2
import numpy as np
import tqdm
import os
import glob
from calibration_utils import get_camera_serial_number, load_calibration_data

def get_marker_points_and_lines(frame, marker_data):
    frame_aruco = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_aruco = 255 - frame_aruco

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, rejected = detector.detectMarkers(frame_aruco)
    if ids is None:
        return None, None, None
    
    ids_np = np.array(ids)
    ids_np = ids_np.flatten()
    arr_idx = np.where(ids_np==0)[0]

    if len(arr_idx) == 0:
        return None, None, None

    arr_idx = arr_idx[0]
    corners = corners[arr_idx][0]

    homography_matrix = cv2.getPerspectiveTransform(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32), corners)

    template_points = marker_data["points"]
    template_lines = marker_data["lines"]
    template_outline = marker_data.get("outline", None)
    
    bb_offset = 0.15

    lines_bb_min = np.min(template_lines, axis=(0, 1))
    lines_bb_max = np.max(template_lines, axis=(0, 1))
    lines_bb_min_ = lines_bb_min - (lines_bb_max - lines_bb_min) * bb_offset
    lines_bb_max_ = lines_bb_max + (lines_bb_max - lines_bb_min) * bb_offset
    
    bb_points = np.array([[lines_bb_min_[0], lines_bb_min_[1]], [lines_bb_max_[0], lines_bb_min_[1]], [lines_bb_max_[0], lines_bb_max_[1]], [lines_bb_min_[0], lines_bb_max_[1]]], dtype=np.float32)
    bb_points_warped = homography_matrix @ np.concatenate([bb_points, np.ones((*bb_points.shape[:-1], 1))], axis=-1)[..., None]
    bb_points_warped = bb_points_warped.squeeze(-1)
    bb_points_warped = bb_points_warped[..., :-1] / bb_points_warped[..., -1][..., None]

    world_points = homography_matrix @ np.concatenate([template_points, np.ones((*template_points.shape[:-1], 1))], axis=-1)[..., None]
    world_points = world_points.squeeze(-1)
    world_points = world_points[..., :-1] / world_points[..., -1][..., None]

    world_lines = homography_matrix @ np.concatenate([template_lines, np.ones((*template_lines.shape[:-1], 1))], axis=-1)[..., None]
    world_lines = world_lines.squeeze(-1)
    world_lines = world_lines[..., :-1] / world_lines[..., -1][..., None]

    # Transform outline points if they exist
    world_outline = None
    if template_outline is not None and len(template_outline) > 0:
        world_outline = homography_matrix @ np.concatenate([template_outline, np.ones((*template_outline.shape[:-1], 1))], axis=-1)[..., None]
        world_outline = world_outline.squeeze(-1)
        world_outline = world_outline[..., :-1] / world_outline[..., -1][..., None]

    return world_points, world_lines, world_outline, bb_points, bb_points_warped, homography_matrix, np.linalg.inv(homography_matrix)

def subpixel_maximum(Z):
    # Create coordinate grid
    y, x = np.indices(Z.shape)

    # Flatten arrays for least squares fitting
    x_flat = x.ravel()
    y_flat = y.ravel()
    z_flat = Z.ravel()

    # Design matrix for quadratic surface
    A = np.column_stack([x_flat**2, y_flat**2, x_flat*y_flat, x_flat, y_flat, np.ones_like(x_flat)])

    # Solve least squares: z = a*x² + b*y² + c*x*y + d*x + e*y + f
    coeff, _, _, _ = np.linalg.lstsq(A, z_flat, rcond=None)
    a, b, c, d, e, f = coeff

    # Compute subpixel maximum analytically
    denom = 4*a*b - c**2
    if np.isclose(denom, 0):
        raise ValueError("Degenerate quadratic fit: cannot compute maximum position.")
    x_max = (c*e - 2*b*d) / denom
    y_max = (c*d - 2*a*e) / denom

    return x_max, y_max


def refine_marker_points(frame, marker_points):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray_gaussian = cv2.GaussianBlur(frame_gray, (5,5), 0)
    refined_points = []
    for point in marker_points:
        x, y = int(point[0]), int(point[1])
        

        # Find the brightest pixel in the neighborhood
        half_win = 5
        
        # Extract the neighborhood
        neighborhood = frame_gray_gaussian[y-half_win:y+half_win, x-half_win:x+half_win]
        
        # Find the pixel with highest brightness using np.argmax
        flat_idx = np.argmax(neighborhood)
        
        # Convert flat index back to 2D coordinates within the neighborhood
        local_y, local_x = np.unravel_index(flat_idx, neighborhood.shape)
        
        # Convert to global coordinates
        brightest_x = x + local_x - half_win
        brightest_y = y + local_y - half_win

        # Alternative method: quadratic fit
        half_win_small = 2
        local_patch = frame_gray_gaussian[brightest_y-half_win_small:brightest_y+half_win_small+1, brightest_x-half_win_small:brightest_x+half_win_small+1]
        xm, ym = subpixel_maximum(local_patch)

        brightest_x = xm + brightest_x - half_win_small
        brightest_y = ym + brightest_y - half_win_small

        refined_points.append([brightest_x, brightest_y])
    
    return np.array(refined_points)

def refine_lines(frame_cropped, min_x, min_y, num_lines=5, undist_mask=None):
    # Convert to grayscale and binarize
    frame_gray = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2GRAY)
    frame_bin = frame_gray > 240

    # Erode to clean up the binary image
    frame_eroded = cv2.erode((frame_bin*255).astype(np.uint8), np.ones((1, 1), dtype=np.uint8), iterations=2)

    # Find all connected components
    num_components, labels, stats, centroids = cv2.connectedComponentsWithStats(frame_eroded)
    
    print(f"Found {num_components - 1} connected components")
    
    # Filter components by maximum extent (larger axis) and keep only the 5 largest
    if num_components > 1:  # If we have components beyond background
        # Calculate maximum extent for each component (width or height, whichever is larger)
        max_extents = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT])
        
        # Get indices of the 5 largest components (excluding background at index 0)
        largest_indices = np.argsort(max_extents)[-5:][::-1]  # Sort descending, take top 5
        largest_indices += 1  # Adjust for background component (add 1 to get actual component indices)
        
        print(f"Selected {len(largest_indices)} largest components with max extents: {max_extents[largest_indices-1]}")
    else:
        largest_indices = []
        print("No components found")
    
    # Fit lines to the selected components using PCA
    fitted_lines = []
    
    for component_idx in largest_indices:
        # Get coordinates of all pixels belonging to this component
        y_coords, x_coords = np.where(labels == component_idx)
        
        if len(x_coords) < 2:  # Need at least 2 points for line fitting
            continue
        
        # Stack coordinates for PCA (points as rows)
        points = np.column_stack((x_coords, y_coords)).astype(np.float32)
        
        # Apply OpenCV's PCA to get the principal direction
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(points, mean=None)
        line_direction = eigenvectors[0]  # First principal component

        # Check if the object is linear
        is_circular = np.sqrt(eigenvalues[1]) / np.sqrt(eigenvalues[0])
        print(f"Is circular: {is_circular}")
        
        # Calculate line length using the largest eigenvalue
        line_length = np.sqrt(eigenvalues[0, 0])
        
        # Create line endpoints by projecting points onto the line
        # and extending to cover the full range
        centered_points = points - mean
        projections = np.dot(centered_points, line_direction)
        min_proj = np.min(projections)
        max_proj = np.max(projections)
        
        # Apply 5% safety margin at both ends
        line_range = max_proj - min_proj
        safety_margin = 0.05 * line_range
        min_proj += safety_margin
        max_proj -= safety_margin
        
        # Line endpoints in the original coordinate system
        line_start = mean[0] + min_proj * line_direction
        line_end = mean[0] + max_proj * line_direction

        line_start_fine = mean[0] + (min_proj - safety_margin) * line_direction
        line_end_fine = mean[0] + (max_proj + safety_margin) * line_direction

        # Refine the line endpoints
        mean_positive = mean
        direction_positive = line_direction
        eigenvalues_positive = eigenvalues
        line_end_fine = line_end_fine.copy()
        for i in range(4):
            is_positive = ((points - mean_positive) @ direction_positive[:, None])[:, 0] >= 0
            points_positive = points[is_positive]
            mean_positive_new, eigenvectors_positive, eigenvalues_positive_new = cv2.PCACompute2(points_positive, mean=None)
            if np.sqrt(eigenvalues_positive[1]) / np.sqrt(eigenvalues_positive[0]) > 0.5:
                break
            mean_positive = mean_positive_new
            direction_positive = eigenvectors_positive[0]
            eigenvalues_positive = eigenvalues_positive_new
            line_end_fine = mean_positive + np.sqrt(eigenvalues_positive[0]) * direction_positive

        mean_negative = mean
        direction_negative = line_direction
        eigenvalues_negative = eigenvalues
        line_start_fine = line_start_fine.copy()
        for i in range(4):
            is_negative = ((points - mean_negative) @ direction_negative[:, None])[:, 0] < 0
            points_negative = points[is_negative]
            mean_negative_new, eigenvectors_negative, eigenvalues_negative_new = cv2.PCACompute2(points_negative, mean=None)
            if np.sqrt(eigenvalues_negative[1]) / np.sqrt(eigenvalues_negative[0]) > 0.5:
                break
            mean_negative = mean_negative_new
            direction_negative = eigenvectors_negative[0]
            eigenvalues_negative = eigenvalues_negative_new
            line_start_fine = mean_negative - np.sqrt(eigenvalues_negative[0]) * direction_negative

        # Sample from mean to the line endpoint
        num_samples = 128
        start_point = mean_positive
        end_point = mean_positive + np.sqrt(eigenvalues_positive[0]) * direction_positive * 4
        sample_points = np.linspace(start_point, end_point, num_samples)
        sample_points_01 = np.linspace(0, 1, num_samples)

        # Sample the image
        sample_points_2d = sample_points.squeeze(1)
        x_coords = sample_points_2d[:, 0].astype(np.float32)
        y_coords = sample_points_2d[:, 1].astype(np.float32)
        sample_points_gray = cv2.remap(frame_gray, x_coords, y_coords, cv2.INTER_LINEAR)
        
        threshold = 200
        below_threshold = sample_points_gray < threshold
        is_border = np.diff(below_threshold[:, 0]) != 0
        border_idx = np.where(is_border)[0]
        alpha = (sample_points_01[border_idx] + sample_points_01[border_idx+1]) / 2
        line_end_fine = mean_positive + alpha * (end_point - start_point)

        start_point = mean_negative
        end_point = mean_negative - np.sqrt(eigenvalues_negative[0]) * direction_negative * 4
        sample_points = np.linspace(start_point, end_point, num_samples)
        sample_points_01 = np.linspace(0, 1, num_samples)
        sample_points_2d = sample_points.squeeze(1)
        x_coords = sample_points_2d[:, 0].astype(np.float32)
        y_coords = sample_points_2d[:, 1].astype(np.float32)
        sample_points_gray = cv2.remap(frame_gray, x_coords, y_coords, cv2.INTER_LINEAR)
        below_threshold = sample_points_gray < threshold
        is_border = np.diff(below_threshold[:, 0]) != 0
        border_idx = np.where(is_border)[0]
        alpha = (sample_points_01[border_idx] + sample_points_01[border_idx+1]) / 2
        line_start_fine = mean_negative + alpha * (end_point - start_point)

        fitted_lines.append({
            'component_idx': component_idx,
            'start_point': line_start,
            'end_point': line_end,
            'start_point_fine': line_start_fine,
            'end_point_fine': line_end_fine,
            'direction': line_direction,
            'centroid': mean[0],
            'length': line_length,
            'num_points': len(points)
        })
    
    # Sort by line length and take the longest lines (up to num_lines)
    top_lines = fitted_lines
    
    print(f"Fitted {len(fitted_lines)} lines using PCA")
    print(f"Selected {len(top_lines)} longest lines:")
    
    # Convert to array format in original image coordinates
    refined_line_array_fine = np.zeros((len(top_lines), 2, 2), dtype=np.float32)
    refined_lines_array = np.zeros((len(top_lines), 2, 2), dtype=np.float32)
    for i, line_data in enumerate(top_lines):
        # Convert from cropped coordinates back to original image coordinates
        refined_lines_array[i, 0, :] = line_data['start_point'] + [min_x, min_y]
        refined_lines_array[i, 1, :] = line_data['end_point'] + [min_x, min_y]

        refined_line_array_fine[i, 0, :] = line_data['start_point_fine'] + [min_x, min_y]
        refined_line_array_fine[i, 1, :] = line_data['end_point_fine'] + [min_x, min_y]
        
        print(f"  Line {i+1}: Component {line_data['component_idx']}, "
              f"length: {line_data['length']:.1f}, points: {line_data['num_points']}, "
              f"direction: [{line_data['direction'][0]:.3f}, {line_data['direction'][1]:.3f}]")
    
    return refined_lines_array, refined_line_array_fine, top_lines

if __name__ == "__main__":
    input_dir = "./data/input_video"
    input_id = 11
    
    # Find all camera IDs for the given input_id
    video_files = sorted(glob.glob(os.path.join(input_dir, f"{input_id:04d}_[0-9].mkv")))
    print(f"Found {len(video_files)} video files to process for input_id {input_id}")
    
    # Load marker data once (used for all videos)
    marker_data = np.load("data/selected_points_and_lines_long.npz")
    
    # Create output directory
    os.makedirs(os.path.join("data", "detected_pattern"), exist_ok=True)
    
    # Process each camera
    for input_path in video_files:
        # Extract camera_id from filename
        filename = os.path.basename(input_path)
        camera_id = int(filename.replace(".mkv", "").split("_")[1])
        
        print(f"\n{'='*60}")
        print(f"Processing: {filename} (input_id={input_id}, camera_id={camera_id})")
        print(f"{'='*60}")
        
        serial_number = get_camera_serial_number(input_path)
        calibration_data = load_calibration_data(serial_number)

        cap = cv2.VideoCapture(input_path)
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        num_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, num_frames_total // 2)
        print(f"Total frames: {num_frames_total}")

        num_frames_to_process = 100

        max_frame = None

        h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            calibration_data["camera_matrix"], 
            calibration_data["dist_coeffs"], 
            (w, h), 
            1,
            (w, h)
        )

        undist_mask = None
        for i_frame in tqdm.tqdm(range(num_frames_to_process), desc=f"Processing frames"):
            ret, frame = cap.read()
            frame = cv2.undistort(frame, calibration_data["camera_matrix"], calibration_data["dist_coeffs"], newCameraMatrix=new_camera_matrix)
            if undist_mask is None:
                undist_mask = cv2.undistort(np.ones((h, w), dtype=np.uint8), calibration_data["camera_matrix"], calibration_data["dist_coeffs"], newCameraMatrix=new_camera_matrix)
            if not ret:
                break
            if max_frame is None:
                max_frame = frame.copy()
            else:
                max_frame = np.max([max_frame, frame], axis=0)

        cap.release()

        marker_points, _, marker_outline, _, bb_points_warped, _, inv_homography_matrix = get_marker_points_and_lines(
            max_frame, 
            marker_data)
        marker_points_refined = refine_marker_points(max_frame, marker_points)
        
        print(f"Original marker points: {len(marker_points)}")
        print(f"Refined marker points: {len(marker_points_refined)}")
        print(f"Refined points:\n{marker_points_refined}")

        # Crop to the bounding box
        min_x = np.clip(int(np.min(bb_points_warped[:, 0])), 0, max_frame.shape[1])
        min_y = np.clip(int(np.min(bb_points_warped[:, 1])), 0, max_frame.shape[0])
        max_x = np.clip(int(np.max(bb_points_warped[:, 0])), 0, max_frame.shape[1])
        max_y = np.clip(int(np.max(bb_points_warped[:, 1])), 0, max_frame.shape[0])
        max_frame_cropped = max_frame[min_y:max_y, min_x:max_x]

        # Refine lines using PCA fitting
        top_5_lines_array, top_5_lines_array_fine, _ = refine_lines(max_frame_cropped, min_x, min_y, num_lines=5, undist_mask=undist_mask)
        
        # Apply inverse homography matrix to the top 5 lines
        lines_marker = inv_homography_matrix @ np.concatenate([top_5_lines_array, np.ones((*top_5_lines_array.shape[:-1], 1))], axis=-1)[..., None]
        lines_marker = lines_marker.squeeze(-1)
        lines_marker = lines_marker[..., :-1] / lines_marker[..., -1][..., None]

        lines_marker_fine = inv_homography_matrix @ np.concatenate([top_5_lines_array_fine, np.ones((*top_5_lines_array_fine.shape[:-1], 1))], axis=-1)[..., None]
        lines_marker_fine = lines_marker_fine.squeeze(-1)
        lines_marker_fine = lines_marker_fine[..., :-1] / lines_marker_fine[..., -1][..., None]

        needs_flipping = lines_marker_fine[:, 1, 0] < lines_marker_fine[:, 0, 0]
        lines_marker_fine[needs_flipping, :, :] = lines_marker_fine[needs_flipping, ::-1, :]
        top_5_lines_array_fine[needs_flipping, :, :] = top_5_lines_array_fine[needs_flipping, ::-1, :]

        print(f"Lines in marker coordinate system:\n{lines_marker}")

        # Reorder the lines based on their y-values
        reordered_indices = np.argsort(np.max(lines_marker[:, :, 1], axis=-1))
        lines_marker = lines_marker[reordered_indices]
        top_5_lines_array = top_5_lines_array[reordered_indices]

        # Flip based on the x-values
        for i in range(5):
            if lines_marker[i, 0, 0] > lines_marker[i, 1, 0]:
                lines_marker[i, :, :] = lines_marker[i, ::-1, :]
                top_5_lines_array[i, :, :] = top_5_lines_array[i, ::-1, :]

        # Check if any of the refined fine lines are outside the undist mask (or close to the border)
        border_threshold_px = 10
        distance_map = cv2.distanceTransform(undist_mask, cv2.DIST_L2, 3)
        refined_lines_array_fine_int = np.round(top_5_lines_array_fine).astype(np.int32)
        is_valid = distance_map[refined_lines_array_fine_int[..., 1], refined_lines_array_fine_int[..., 0]] > border_threshold_px
        top_5_lines_array_fine[~is_valid, :] = np.nan

        # Save the results
        output_file = os.path.join("data", "detected_pattern", f"{input_id:04d}_{camera_id}.marker.npz")
        np.savez(output_file, strip_lines=top_5_lines_array, strip_lines_fine=top_5_lines_array_fine, marker_points=marker_points_refined, marker_outline=marker_outline)
        print(f"Saved results to {output_file}")
    
    print(f"\n{'='*60}")
    print(f"Finished processing all {len(video_files)} camera files")
    print(f"{'='*60}")