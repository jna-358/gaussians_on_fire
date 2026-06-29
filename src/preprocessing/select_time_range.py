import numpy as np
import cv2
import glob
import os

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
    time_min = np.inf
    time_max = -np.inf
    for video_path in video_paths:
        camera_id = video_path.split("_")[-1].split(".")[0]
        times_path = os.path.join("data", "sync", f"{input_id:04d}_{camera_id}.times.npz")
        times_data = np.load(times_path)["times"]
        times.append(times_data)
        time_min = min(time_min, np.nanmin(times_data))
        time_max = max(time_max, np.nanmax(times_data))

    print(f"Time range: {time_min*1e-3:.2f} ms - {time_max*1e-3:.2f} ms")

    frame_min = np.nanargmin(np.abs(times[1] - time_min))
    frame_max = np.nanargmin(np.abs(times[1] - time_max))
    print(f"Frame range: {frame_min} - {frame_max}")

    frame_current = frame_min
    cap = cv2.VideoCapture(video_paths[1])
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_current)
    while True:
        print(f"Frame: {frame_current}   ", end="\r")
        ret, frame = cap.read()

        frame = cv2.rotate(frame, rotation)
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Frame", frame)
        k = cv2.waitKey()

        if k == ord('a'):
            frame_current -= 100
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_current)
        elif k == ord('q'):
            break
        else:
            frame_current += 1

    print()
    frame_start = frame_current - 100
    frame_end = frame_current

    # Find corresponding times
    times_start = times[1][frame_start]
    times_end = times[1][frame_end]
    print(f"Time range: {times_start*1e-3:.2f} ms - {times_end*1e-3:.2f} ms")

    # Save time range
    os.makedirs(os.path.join("data", "time_range"), exist_ok=True)
    for i_cam in range(num_cameras):
        np.savez(os.path.join("data", "time_range", f"{input_id:04d}_{i_cam}.time_range.npz"), time_start=times_start, time_end=times_end)
        print(f"Saved time range to {os.path.join("data", "time_range", f"{input_id:04d}_{i_cam}.time_range.npz")}")

    cap.release()
    cv2.destroyAllWindows()