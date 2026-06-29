import cv2
import numpy as np
import os
import pickle
import glob
from calibration_utils import get_camera_serial_number, load_calibration_data

if __name__ == "__main__":
    input_dir = "data/input_video"
    input_id = 11
    
    video_paths = sorted(glob.glob(os.path.join(input_dir, f"{input_id:04d}_*.mkv")))

    for video_path in video_paths:
        print(f"Processing {video_path}")
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        marker_path = os.path.join("data/detected_pattern", f"{video_basename}.marker.npz")
        undistortion_args_path = os.path.join("data/undistortion_args", f"{video_basename}.undistortion_args.pkl")
        with open(undistortion_args_path, "rb") as f:
            undistortion_args = pickle.load(f)
        marker_data = np.load(marker_path)

        # Load the image
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        _, frame = cap.read()
        

        serial_number = get_camera_serial_number(video_path)
        calibration_data = load_calibration_data(serial_number)

        h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            calibration_data["camera_matrix"], 
            calibration_data["dist_coeffs"], 
            (w, h), 
            1,
            (w, h)
        )

        frame = cv2.undistort(frame, calibration_data["camera_matrix"], calibration_data["dist_coeffs"], newCameraMatrix=new_camera_matrix)
        mask = np.zeros_like(frame, dtype=np.uint8)


        for marker_point in marker_data["marker_points"]:
            cv2.circle(mask, (int(marker_point[0]), int(marker_point[1])), 5, (255), -1)
        for i_line, (strip_start, strip_end) in enumerate(marker_data["strip_lines_fine"]):
            if np.isnan(strip_start).any():
                strip_start_coarse = marker_data["strip_lines"][i_line, 0, :]
                strip_start = strip_end + 2 * (strip_start_coarse - strip_end)
            if np.isnan(strip_end).any():
                strip_end_coarse = marker_data["strip_lines"][i_line, 1, :]
                strip_end = strip_start + 2 * (strip_end_coarse - strip_start)
            cv2.line(mask, (int(strip_start[0]), int(strip_start[1])), (int(strip_end[0]), int(strip_end[1])), (255), 1)

        # Dilate the mask
        circular_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.dilate(mask, circular_kernel, iterations=6)

        # Close all holes in the mask
        h, w = mask.shape[:2]
        mask_single = mask[..., 0].copy()
        flood_fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(mask_single, flood_fill_mask, (0, 0), 42)
        mask[..., 0] = (mask_single != 42) * 255
        mask[..., 1] = (mask_single != 42) * 255
        mask[..., 2] = (mask_single != 42) * 255

        mask = cv2.erode(mask, circular_kernel, iterations=1)



        # Undo the undistortion to get back to distorted space
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        _, frame = cap.read()
        
        # Apply inverse undistortion to mask
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        distorted_points = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1).reshape(-1, 1, 2)
        
        # Undistort these points to find where they map in undistorted space
        undistorted_points = cv2.undistortPoints(
            distorted_points,
            calibration_data["camera_matrix"],
            calibration_data["dist_coeffs"],
            P=new_camera_matrix
        )
        
        # Create remap arrays and apply to mask
        map_x = undistorted_points[:, 0, 0].reshape(h, w).astype(np.float32)
        map_y = undistorted_points[:, 0, 1].reshape(h, w).astype(np.float32)
        mask = cv2.remap(mask, map_x, map_y, cv2.INTER_LINEAR)

        # Undistort unsing final undistortion args
        frame = cv2.undistort(frame, *undistortion_args["undistort_args"])
        mask = cv2.undistort(mask, *undistortion_args["undistort_args"])

        # Rotate image and mask
        frame = cv2.rotate(frame, *undistortion_args["rotate_args"])
        mask = cv2.rotate(mask, *undistortion_args["rotate_args"])

        # Warp the mask
        mask = cv2.warpPerspective(mask, *undistortion_args["warpPerspective_args"], **undistortion_args["warpPerspective_kwargs"])
        frame = cv2.warpPerspective(frame, *undistortion_args["warpPerspective_args"], **undistortion_args["warpPerspective_kwargs"])

        cap.release()

        # Save the mask
        os.makedirs("data/sync_mask", exist_ok=True)
        output_path = os.path.join("data/sync_mask", f"{video_basename}.sync_mask.png")
        cv2.imwrite(output_path, mask)
        print(f"Saved mask to {output_path}")