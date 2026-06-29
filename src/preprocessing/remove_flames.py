import numpy as np
import cv2
import glob
import os
import tqdm

if __name__ == "__main__":
    input_dir = "./data/input_video"
    rotation = cv2.ROTATE_90_CLOCKWISE
    input_id = 11

    # Find video paths
    video_paths = sorted(glob.glob(os.path.join(input_dir, f"{input_id:04d}_[0-9].mkv")))
    num_cameras = len(video_paths)
    print(f"Found {num_cameras} cameras:")
    for video_path in video_paths:
        print(f"  - {video_path}")

    # Load time synchronization data
    times = []
    time_min = -np.inf
    time_max = np.inf
    for video_path in video_paths:
        camera_id = video_path.split("_")[-1].split(".")[0]
        times_path = os.path.join("data", "sync", f"{input_id:04d}_{camera_id}.times.npz")
        times_data = np.load(times_path)["times"]
        times.append(times_data)
        time_min = max(time_min, np.nanmin(times_data))
        time_max = min(time_max, np.nanmax(times_data))
    print(f"Time range (maximum possible): {time_min*1e-3:.2f} ms - {time_max*1e-3:.2f} ms")

    # Load time range 
    camera_id = video_paths[1].split("_")[-1].split(".")[0]
    time_range_path = os.path.join("data", "time_range", f"{input_id:04d}_{camera_id}.time_range.npz")
    time_range_data = np.load(time_range_path)
    time_start = time_range_data["time_start"]
    time_end = time_range_data["time_end"]
    print(f"Time range: {time_start*1e-3:.2f} ms - {time_end*1e-3:.2f} ms; Duration: {(time_end - time_start)*1e-3:.2f} ms")

    # Expand time range
    range_multiplier = 2.5
    time_start = max(time_min, time_start - (time_end - time_start) * range_multiplier / 2.0)
    time_end = min(time_max, time_end + (time_end - time_start) * range_multiplier / 2.0)
    print(f"Expanded time range: {time_start*1e-3:.2f} ms - {time_end*1e-3:.2f} ms")

    # Find start frames
    frames_start = []
    frames_end = []
    for i_cam in range(num_cameras):
        frame_start = np.nanargmin(np.abs(times[i_cam] - time_start))
        frames_start.append(frame_start)
        frame_end = np.nanargmin(np.abs(times[i_cam] - time_end))
        frames_end.append(frame_end)
    num_frames = np.min([frames_end[i_cam] - frames_start[i_cam] for i_cam in range(num_cameras)])
    print(f"Number of frames: {num_frames}")

    # Compute min frames
    caps = [cv2.VideoCapture(video_path) for video_path in video_paths]
    for i_cam in range(num_cameras):
        caps[i_cam].set(cv2.CAP_PROP_POS_FRAMES, frames_start[i_cam])
        caps[i_cam].set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    min_frames = [None] * num_cameras
    for i_frame in tqdm.tqdm(range(num_frames)):
        for i_cam in range(num_cameras):
            ret, frame = caps[i_cam].read()
            min_frame = min_frames[i_cam]
            if min_frame is None:
                min_frames[i_cam] = frame.copy()
            else:
                is_smaller = np.linalg.norm(frame, axis=-1) < np.linalg.norm(min_frame, axis=-1)
                min_frames[i_cam][is_smaller] = frame[is_smaller]

        frames_all = np.concatenate(min_frames, axis=0)
        frames_all = cv2.rotate(frames_all, rotation)
        frames_all = cv2.resize(frames_all, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Frames", frames_all)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    cv2.destroyAllWindows()

    # Save min frames
    os.makedirs(os.path.join("data", "min_frame"), exist_ok=True)
    for i_cam in range(num_cameras):
        cv2.imwrite(os.path.join("data", "min_frame", f"{input_id:04d}_{i_cam}.min_frame.png"), min_frames[i_cam])
        print(f"Saved min frame to {os.path.join("data", "min_frame", f"{input_id:04d}_{i_cam}.min_frame.png")}")
        

    