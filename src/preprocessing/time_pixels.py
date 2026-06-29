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
    print(f"Found {len(video_paths)} videos:")
    for video_path in video_paths:
        print(f"  - {video_path}")
    for video_path in video_paths:
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        times_path = os.path.join("data/sync", f"{video_basename}.times.npz")
        times_data = np.load(times_path)
        undistort_args_path = os.path.join("data/undistortion_args", f"{video_basename}.undistortion_args.pkl")
        with open(undistort_args_path, "rb") as f:
            undistortion_args = pickle.load(f)
        serial_number = get_camera_serial_number(video_path)
        calibration_data = load_calibration_data(serial_number)
        row_indices = np.arange(calibration_data["image_size"][1])
        row_indices = np.repeat(row_indices[:, np.newaxis], calibration_data["image_size"][0], axis=1)
        times_offset_us = row_indices * times_data["readout_time_per_line_us"]

        times_offset_us = cv2.undistort(times_offset_us, *undistortion_args["undistort_args"])
        times_offset_us = cv2.rotate(times_offset_us, *undistortion_args["rotate_args"])
        times_offset_us = cv2.warpPerspective(times_offset_us, *undistortion_args["warpPerspective_args"], **undistortion_args["warpPerspective_kwargs"])

        os.makedirs("data/rolling_shutter", exist_ok=True)
        output_path = os.path.join("data/rolling_shutter", f"{video_basename}.rolling_shutter.npy")
        np.save(output_path, times_offset_us * 1e-3)
        print(f"Saved rolling shutter data to {output_path}")



