import cv2
import numpy as np
import os
import sys
import glob
import torch

# Add Depth-Anything-V2 to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Depth-Anything-V2'))

from depth_anything_v2.dpt import DepthAnythingV2


def require_checkpoint(encoder, checkpoint_dir):
    """Require the checkpoint to exist, prompting the user to place it if missing."""
    checkpoint_path = os.path.join(checkpoint_dir, f'depth_anything_v2_{encoder}.pth')

    if os.path.exists(checkpoint_path):
        print(f"Checkpoint found at: {checkpoint_path}")
        return checkpoint_path

    os.makedirs(checkpoint_dir, exist_ok=True)

    while not os.path.exists(checkpoint_path):
        print(f"\nCheckpoint not found: {checkpoint_path}")
        print(f"Please download 'depth_anything_v2_{encoder}.pth' and place it in: {os.path.abspath(checkpoint_dir)}")
        input("Press Enter to continue once the file is in place...")

    print(f"Checkpoint found at: {checkpoint_path}")
    return checkpoint_path


if __name__ == '__main__':
    input_id = 11
    
    # Find all rotated undistorted images for this input_id
    image_files = sorted(glob.glob(os.path.join("data", "rotated_undistorted", f"{input_id:04d}_[0-9].rotated_undistorted.png")))
    print(f"Found {len(image_files)} images to process for input_id {input_id}")
    
    if not image_files:
        print(f"No images found matching pattern: data/rotated_undistorted/{input_id:04d}_[0-9].rotated_undistorted.png")
        exit(1)
    
    # Fixed configuration
    encoder = 'vitl'
    input_size = 518
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Define checkpoint directory and download if needed
    checkpoint_dir = os.path.join("data", "checkpoints", "depth_anything_v2")
    checkpoint_path = require_checkpoint(encoder, checkpoint_dir)
    
    model_config = {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    
    depth_anything = DepthAnythingV2(**model_config)
    depth_anything.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    # Create output directory
    os.makedirs("data/mono_depth", exist_ok=True)
    
    # Process each image
    for filename in image_files:
        # Extract camera_id from filename
        basename = os.path.basename(filename)
        camera_id = int(basename.split('_')[1].split('.')[0])
        
        print(f"\n{'='*60}")
        print(f'Processing: {filename} (input_id={input_id}, camera_id={camera_id})')
        print(f"{'='*60}")
        
        raw_image = cv2.imread(filename)
        
        depth_raw = depth_anything.infer_image(raw_image, input_size)
        print(f"Minimum depth: {depth_raw.min()}, Maximum depth: {depth_raw.max()}")

        depth_16bit = (depth_raw - depth_raw.min()) / (depth_raw.max() - depth_raw.min()) * 65535.0
        depth_16bit = depth_16bit.astype(np.uint16)
        
        output_path = os.path.join("data/mono_depth", f"{input_id:04d}_{camera_id}.mono_depth_16.png")
        
        cv2.imwrite(output_path, depth_16bit)
        print(f"Saved 16-bit depth image to: {output_path}")
    
    print(f"\n{'='*60}")
    print(f"Finished processing all {len(image_files)} images")
    print(f"{'='*60}")