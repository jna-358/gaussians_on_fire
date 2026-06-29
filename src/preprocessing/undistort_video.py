import cv2
import numpy as np
import os
import pickle
import tqdm
import glob

if __name__ == "__main__":
    input_dir = "data/input_video"
    input_id = 11
    visualize = False
    
    video_paths = sorted(glob.glob(os.path.join(input_dir, f"{input_id:04d}_*.mkv")))
    print(f"Found {len(video_paths)} videos:")
    for video_path in video_paths:
        print(f"  - {video_path}")

    for video_path in video_paths:
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        undistort_args_path = os.path.join("data/undistortion_args", f"{video_basename}.undistortion_args.pkl")
        with open(undistort_args_path, "rb") as f:
            undistortion_args = pickle.load(f)
        print(f"Undistortion args: {undistortion_args}")

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        video_writer = None
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Number of frames: {num_frames}")
        
        os.makedirs("data/undistorted_video", exist_ok=True)
        output_path = os.path.join("data/undistorted_video", f"{video_basename}.undistorted.mkv")
        
        for i_frame in tqdm.tqdm(range(num_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            frame_undistorted = cv2.undistort(frame, *undistortion_args["undistort_args"])
            frame_rotated = cv2.rotate(frame_undistorted, *undistortion_args["rotate_args"])
            frame_warped = cv2.warpPerspective(frame_rotated, *undistortion_args["warpPerspective_args"], **undistortion_args["warpPerspective_kwargs"])

            if video_writer is None:
                fps = cap.get(cv2.CAP_PROP_FPS)
                video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"FFV1"), fps, (frame_warped.shape[1], frame_warped.shape[0]))
            video_writer.write(frame_warped)

            if visualize:
                cv2.imshow("Frame", frame_warped)
                cv2.waitKey(1)
        cap.release()
        if video_writer is not None:
            video_writer.release()
        print(f"Saved undistorted video to {output_path}")