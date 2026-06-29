import subprocess
import os
from rich.console import Console
from rich.panel import Panel

console = Console()

def run_command(cmd, title):
    """Run a command with a rich panel title"""
    console.print(Panel(f"[bold cyan]{' '.join(cmd)}[/bold cyan]", title=f"[bold green]{title}[/bold green]", border_style="green"))
    subprocess.run(cmd)

# Preprocessing Pipeline
run_command(["python", "src/preprocessing/detect_pattern.py"], "🎯 Detecting Pattern")
run_command(["python", "src/preprocessing/extract_brightness.py"], "💡 Extracting Brightness")
run_command(["python", "src/preprocessing/sync.py"], "🕐 Synchronizing Data")
run_command(["python", "src/preprocessing/remove_flames.py"], "🔥 Removing Flames")
run_command(["python", "src/preprocessing/run_colmap.py"], "📸 Running COLMAP")
run_command(["python", "src/preprocessing/rotate_cams.py"], "🔄 Rotating Cameras")
run_command(["python", "src/preprocessing/run_depth_anything.py"], "📏 Running Depth Estimation")
run_command(["python", "src/preprocessing/convert_to_colmap.py"], "🔧 Converting to COLMAP Format")

# Static Scene Training
os.chdir("src/static_scene")
run_command(["python", "train.py", "-s", "../../data/colmap/0011", "-d", "../../data/colmap/0011/depth_anything", "--sh_degree", "0"], "🌳 Training Static Scene")
os.chdir("../..")

# Additional Preprocessing
run_command(["python", "src/preprocessing/mask_pattern.py"], "🎭 Masking Pattern")
run_command(["python", "src/preprocessing/time_pixels.py"], "⏱️ Timing Pixels")
run_command(["python", "src/preprocessing/undistort_video.py"], "📹 Undistorting Video")

# Dynamic Scene Preprocessing
run_command(["python", "src/preprocessing/flow_estimation.py"], "🌊 Estimating Flow")
run_command(["python", "src/preprocessing/voxel_projection.py"], "🧊 Projecting Voxels")
run_command(["python", "src/preprocessing/convert_to_dnerf.py"], "🎬 Converting to D-NeRF Format")

# Dynamic Scene Training
os.chdir("src/dynamic_scene")
run_command(["python", "train.py", "-s", "../../data/dnerf_output/0011_dnerf_temporal", "--sh_degree", "0", "--eval", "--train_test_exp", "--live_view", "--mask_loss", "0.1"], "🔥 Training Dynamic Scene")
os.chdir("../..")