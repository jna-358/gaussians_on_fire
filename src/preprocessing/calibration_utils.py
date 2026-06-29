import glob
import os
import numpy as np
import subprocess
import json

def get_camera_serial_number(video_path):
    metadata_file = os.path.join(video_path.replace(".mkv", ".metadata.json"))
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    return metadata["QuickTime:CameraSerialNumber"]

def get_iso(video_path):
    metadata_file = os.path.join(video_path.replace(".mkv", ".metadata.json"))
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    auto_iso_max = metadata["QuickTime:AutoISOMax"]
    auto_iso_min = metadata["QuickTime:AutoISOMin"]
    assert auto_iso_max == auto_iso_min, f"Dynamic ISO within detected for video {video_path}"
    return int(auto_iso_max)

def get_shutter(video_path):
    metadata_file = os.path.join(video_path.replace(".mkv", ".metadata.json"))
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    shutter_angle = metadata["QuickTime:MaximumShutterAngle"]
    shutter_angle = shutter_angle.replace("SEC", "").replace("_", "/")
    return shutter_angle

def load_calibration_data(serial_number):
    potential_paths = glob.glob(os.path.join("data/calibration", f"calib_{serial_number}_*.npz"))

    potential_paths.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    if len(potential_paths) == 0:
        raise Exception(f"No calibration data found for {serial_number}")

    calib_path = potential_paths[-1]
    print(f"Loading calibration data from {calib_path}")
    return np.load(calib_path)

def get_frame_rate(video_path):
    metadata_file = os.path.join(video_path.replace(".mkv", ".metadata.json"))
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    frame_rate = metadata["QuickTime:VideoFrameRate"]
    return float(frame_rate)