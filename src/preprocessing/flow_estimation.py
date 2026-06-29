import torch
import cv2
from tqdm import tqdm
import os
from memfof import MEMFOF
import glob
import numpy as np

model_name = "MEMFOF-Tartan-T-TSKH"

def flow_to_image(flow, rad_min=0.02):
    dx = flow[:, :, 0]
    dy = flow[:, :, 1]
    magnitude = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
    max_magnitude = np.maximum(magnitude.max(), rad_min)
    hsv[:, :, 1] = np.clip(magnitude / max_magnitude * 255, 0, 255).astype(np.uint8)
    hsv[:, :, 2] = 255
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return rgb

def float_to_uint8(input_array, min_val, max_val):
    input_array = np.clip(input_array, min_val, max_val)
    input_array = (input_array - min_val) / (max_val - min_val) * 255
    return input_array.astype(np.uint8)

@torch.no_grad()
def process_video(video_path, viz=False, device="cuda"):
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    if video_basename.endswith(".undistorted"):
        video_basename = video_basename.replace(".undistorted", "")
    
    output_dir = os.path.join("data/flow", video_basename)
    output_viz_path = os.path.join("data/flow_viz", f"{video_basename}.flow_viz.avi")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("data/flow_viz", exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_writer = None

    model = MEMFOF.from_pretrained(f"egorchistov/optical-flow-{model_name}").eval().to(device)

    frames = []
    fmap_cache = [None] * 3
    first_frame = True

    for i_frame in tqdm(range(num_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        frame = torch.tensor(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            dtype=torch.float32
        ).permute(2, 0, 1).unsqueeze(0)
        
        height, width = frame.shape[2:]

        if first_frame:
            frames.append(frame)
            first_frame = False

        frames.append(frame)

        if len(frames) != 3:
            print(f"Waiting for more frames... {len(frames)}")
            continue

        frames_tensor = torch.stack(frames, dim=1).to(device)
        output = model(frames_tensor, fmap_cache=fmap_cache)
        
        if viz:
            forward_flow = output["flow"][-1][:, 1]
            flow_vis = flow_to_image(
                forward_flow.squeeze(dim=0).permute(1, 2, 0).cpu().numpy(),
                rad_min=0.02 * (height ** 2 + width ** 2) ** 0.5,
            )
            
            if video_writer is None:
                video_writer = cv2.VideoWriter(
                    output_viz_path,
                    cv2.VideoWriter_fourcc(*"DIVX"),
                    30,
                    (width, height),
                )
            video_writer.write(flow_vis)
        

        # Store flow as 4-channel 16-bit PNG (RGBA format, with flow in R and G channels)
        flow_export = output["flow"][-1][:, 1].squeeze(0).permute(1, 2, 0).cpu().numpy()
        flow_export = float_to_uint8(flow_export, -25, 25)
        
        flow_export = np.concatenate([
            flow_export, 
            np.ones((height, width, 1), dtype=np.uint8) * 128,
        ], axis=-1)
        cv2.imwrite(os.path.join(output_dir, f"{i_frame:06d}.png"), flow_export)

        fmap_cache = output["fmap_cache"]
        fmap_cache.pop(0)
        fmap_cache.append(None)

        frames.pop(0)
    
    if viz:
        video_writer.release()
        print(f"Saved flow visualization to {output_viz_path}")

    cap.release()
    print(f"Saved flow frames to {output_dir}")

if __name__ == "__main__":
    input_dir = "data/undistorted_video"
    input_id = 11
    viz = True  # Set to False to disable visualization output
    
    video_paths = sorted(glob.glob(os.path.join(input_dir, f"{input_id:04d}_*.undistorted.mkv")))
    print(f"Found {len(video_paths)} video(s):")
    for video_path in video_paths:
        print(f"  - {video_path}")
    
    # Process videos sequentially
    for video_path in video_paths:
        print(f"Processing {video_path}...")
        process_video(video_path, viz=viz, device="cuda")
        print(f"Finished {video_path}")
    
    print("All videos processed!")