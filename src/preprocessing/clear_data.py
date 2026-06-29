import shutil
import os
import glob

if __name__ == "__main__":
    dirs_to_delete = [
        "checkpoints",
        "colmap",
        "detected_pattern",
        "dnerf_output",
        "flow",
        "flow_viz",
        "max_frame",
        "min_frame",
        "mono_depth",
        "mono_depth_aligned",
        "pattern_brightness",
        "pose",
        "rolling_shutter",
        "rotated_cams",
        "rotated_undistorted",
        "rotated_undistorted_cropped",
        "stereo_depth",
        "stereo_depth_rotated",
        "stereo_depth_rotated_cropped",
        "sync",
        "sync_mask",
        "undistorted_video",
        "undistortion_args",
        "voxel_projection",
    ]

    additional_dirs = [
        "./src/static_scene/output",
        "./src/dynamic_scene/output",
    ]

    patterns = [
        "./src/**/__pycache__",
    ]

    data_dir = "data"
    for dir in dirs_to_delete:
        print(f"Deleting {os.path.join(data_dir, dir)}")
        shutil.rmtree(os.path.join(data_dir, dir), ignore_errors=True)

    for dir in additional_dirs:
        print(f"Deleting {dir}")
        shutil.rmtree(dir, ignore_errors=True)

    for pattern in patterns:
        print(f"Deleting {pattern}")
        for file in glob.glob(pattern, recursive=True):
            print(f"  - {file}")
            shutil.rmtree(file, ignore_errors=True)