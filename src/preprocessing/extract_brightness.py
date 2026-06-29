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
        return None, None
    
    ids_np = np.array(ids)
    ids_np = ids_np.flatten()
    arr_idx = np.where(ids_np==0)[0]

    if len(arr_idx) == 0:
        return None, None

    arr_idx = arr_idx[0]
    corners = corners[arr_idx][0]

    homography_matrix = cv2.getPerspectiveTransform(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32), corners)

    template_points = marker_data["points"]
    template_lines = marker_data["lines"]
    
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

    return world_points, world_lines, bb_points, bb_points_warped

def draw_points(frame, points, marker_size=5, color=(0, 255, 0)):
    if points is not None:
        for point in points:
            cv2.circle(frame, (int(point[0]), int(point[1])), marker_size, color, -1)

def draw_lines(frame, lines):
    if lines is not None:
        for line in lines:
            cv2.line(frame, (int(line[0][0]), int(line[0][1])), (int(line[1][0]), int(line[1][1])), (0, 255, 0), 2)

def distortPoints(points, new_camera_matrix, camera_matrix, dist_coeffs):
    # Ensure dist_coeffs is a 1D array
    dist_coeffs = dist_coeffs.flatten()
    
    # Convert from new_camera_matrix space to normalized coordinates
    points_normalized = cv2.undistortPoints(
        points.reshape(-1, 1, 2), 
        new_camera_matrix, 
        np.zeros(5)  # no distortion for this step
    ).reshape(points.shape)
    
    # Apply distortion manually to normalized coordinates
    x, y = points_normalized[:, 0], points_normalized[:, 1]
    r2 = x*x + y*y
    r4 = r2*r2
    r6 = r2*r4
    
    # Distortion model: x' = x(1 + k1*r2 + k2*r4 + k3*r6) + 2*p1*x*y + p2*(r2 + 2*x2)
    k1, k2, p1, p2, k3 = dist_coeffs[0], dist_coeffs[1], dist_coeffs[2], dist_coeffs[3], dist_coeffs[4]
    x_dist = x * (1 + k1*r2 + k2*r4 + k3*r6) + 2*p1*x*y + p2*(r2 + 2*x*x)
    y_dist = y * (1 + k1*r2 + k2*r4 + k3*r6) + 2*p1*x*y + p2*(r2 + 2*y*y)
    
    # Project back to original camera matrix pixel coordinates
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    
    return np.column_stack([x_dist * fx + cx, y_dist * fy + cy])


def refine_marker_points(frame, marker_points):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_bin = (frame_gray > 150).astype(np.uint8)
    
    # Use distance transform to find closest True pixels efficiently
    dist_transform = cv2.distanceTransform(255 - frame_bin, cv2.DIST_L2, 5)
    
    refined_points = []
    for point in marker_points:
        x, y = int(point[0]), int(point[1])
        
        # Check bounds
        if not (0 <= x < frame_bin.shape[1] and 0 <= y < frame_bin.shape[0]):
            refined_points.append(point)
            continue
            
        # If already on True pixel, use it; otherwise find closest
        if frame_bin[y, x]:
            closest_x, closest_y = x, y
        else:
            # Find closest True pixel using distance transform
            search_radius = 50
            min_x, max_x = max(0, x - search_radius), min(frame_bin.shape[1], x + search_radius)
            min_y, max_y = max(0, y - search_radius), min(frame_bin.shape[0], y + search_radius)
            
            search_dist = dist_transform[min_y:max_y, min_x:max_x]
            min_val, _, min_loc, _ = cv2.minMaxLoc(search_dist)
            
            if min_val < search_radius:  # Found a close True pixel
                closest_x, closest_y = min_loc[0] + min_x, min_loc[1] + min_y
            else:
                refined_points.append(point)
                continue
        
        # Get connected component using flood fill
        mask = np.zeros((frame_bin.shape[0] + 2, frame_bin.shape[1] + 2), dtype=np.uint8)
        cv2.floodFill(frame_bin, mask, (closest_x, closest_y), 255)
        mask = mask[1:-1, 1:-1]
        
        # Calculate centroid using moments
        moments = cv2.moments(mask)
        if moments['m00'] > 0:
            centroid_x = moments['m10'] / moments['m00']
            centroid_y = moments['m01'] / moments['m00']
            refined_points.append([centroid_x, centroid_y])
        else:
            refined_points.append(point)
    
    return np.array(refined_points)

if __name__ == "__main__":
    input_dir = "./data/input_video"
    input_id = 11
    
    # Find all camera IDs for the given input_id
    video_files = sorted(glob.glob(os.path.join(input_dir, f"{input_id:04d}_[0-9].mkv")))
    print(f"Found {len(video_files)} video files to process for input_id {input_id}")
    
    video = False
    interactive = True
    
    # Create output directory
    os.makedirs(os.path.join("data", "pattern_brightness"), exist_ok=True)
    
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

        marker_data = np.load(os.path.join("data", "detected_pattern", f"{input_id:04d}_{camera_id}.marker.npz"))
        strip_lines = marker_data["strip_lines"]
        marker_points = marker_data["marker_points"]

        marker_std = np.std(marker_points, axis=0)
        marker_size = np.mean(marker_std) * 0.2

        line_resolution = 128
        strip_lines_coordinates = np.linspace(strip_lines[:, 0, :], strip_lines[:, 1, :], line_resolution, dtype=np.float32).transpose(1, 0, 2)

        cap = cv2.VideoCapture(input_path)
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        num_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames: {num_frames_total}")

        h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = 24
        
        # Setup video writer if --video flag is provided
        video_writer = None
        if video:
            output_video_path = input_path + '.extraction.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
            print(f"Rendering video to: {output_video_path}")
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            calibration_data["camera_matrix"], 
            calibration_data["dist_coeffs"], 
            (w, h), 
            1,
            (w, h)
        )

        # Create distorted versions of strip_lines_coordinates and marker_points
        strip_lines_coordinates_distorted = distortPoints(
            strip_lines_coordinates.reshape(-1, 2), 
            new_camera_matrix, 
            calibration_data["camera_matrix"], 
            calibration_data["dist_coeffs"]
        ).reshape(strip_lines_coordinates.shape)
        
        marker_points_distorted = distortPoints(
            marker_points, 
            new_camera_matrix, 
            calibration_data["camera_matrix"], 
            calibration_data["dist_coeffs"]
        )
        
        marker_brightness = np.zeros((num_frames_total, len(marker_points)))
        marker_brightness.fill(np.nan)

        strip_brightness = np.zeros((num_frames_total, len(strip_lines), line_resolution))
        strip_brightness.fill(np.nan)

        for i_frame in tqdm.tqdm(range(num_frames_total), desc=f"Processing frames"):
            ret, frame = cap.read()
            if not ret:
                break

            frame_distorted = frame
            frame = cv2.undistort(frame, calibration_data["camera_matrix"], calibration_data["dist_coeffs"], newCameraMatrix=new_camera_matrix)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            for i_line, line in enumerate(strip_lines):
                cv2.line(frame, (int(line[0][0]), int(line[0][1])), (int(line[1][0]), int(line[1][1])), (0, 255, 0), 1)
                cv2.putText(frame, f"{i_line}", (int(line[0][0])+10, int(line[0][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            for i_point, point in enumerate(marker_points):
                # cv2.circle(frame, (int(point[0]), int(point[1])), int(marker_size), (0, 255, 0), 1)
                cv2.putText(frame, f"{i_point}", (int(point[0])+10, int(point[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            for i_point, point in enumerate(marker_points_distorted):
                cv2.circle(frame_distorted, (int(point[0]), int(point[1])), int(marker_size), (0, 255, 0), 1)
                cv2.putText(frame_distorted, f"{i_point}", (int(point[0])+10, int(point[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            for i_line, line in enumerate(strip_lines_coordinates_distorted):
                query_points = np.round(line).astype(np.uint32)
                frame_distorted[query_points[:, 1], query_points[:, 0], :] = (0, 0, 255)

            # Extract region around each marker point
            for i_point, point in enumerate(marker_points):
                x_min = int(round(point[0] - marker_size / 2))
                x_max = int(x_min + marker_size) + 1
                y_min = int(round(point[1] - marker_size / 2))
                y_max = int(y_min + marker_size) + 1
                
                frame_cropped = frame_gray[y_min:y_max, x_min:x_max]
                marker_brightness[i_frame, i_point] = np.mean(frame_cropped)
                frame[y_min:y_max, x_min:x_max, 2] = (frame[y_min:y_max, x_min:x_max, 2] // 2) + 128

            for i_line, line in enumerate(strip_lines):
                query_points = np.round(strip_lines_coordinates[i_line, :, :]).astype(np.uint32)
                strip_brightness[i_frame, i_line, :] = frame_gray[query_points[:, 1], query_points[:, 0]]

            # Write frame to video if --video flag is provided
            if video_writer is not None:
                video_writer.write(frame)

            if interactive:
                cv2.imshow("Frame", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break

        soft_buffer = 5
        strip_brightness_max = np.nanpercentile(strip_brightness, 100 - soft_buffer, axis=(0, 2))
        strip_brightness_min = np.nanpercentile(strip_brightness, soft_buffer, axis=(0, 2))
        strip_brightness = (strip_brightness - strip_brightness_min[None, :, None]) / (strip_brightness_max[None, :, None] - strip_brightness_min[None, :, None])

        marker_brightness_max = np.nanpercentile(marker_brightness, 100 - soft_buffer, axis=(0, 1))
        marker_brightness_min = np.nanpercentile(marker_brightness, soft_buffer, axis=(0, 1))
        marker_brightness = (marker_brightness - marker_brightness_min) / (marker_brightness_max - marker_brightness_min)

        output_file = os.path.join("data", "pattern_brightness", f"{input_id:04d}_{camera_id}.brightness.npz")
        np.savez(output_file, strip_brightness=strip_brightness, marker_brightness=marker_brightness, marker_points=marker_points_distorted, strip_lines=strip_lines_coordinates_distorted)
        print(f"Saved results to {output_file}")

        cap.release()
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved to: {output_video_path}")
    
    cv2.destroyAllWindows()
    print(f"\n{'='*60}")
    print(f"Finished processing all {len(video_files)} camera files")
    print(f"{'='*60}")