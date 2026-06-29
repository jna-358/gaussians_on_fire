import shutil

if __name__ == "__main__":
    dirs_to_delete = [
        "./src/static_scene/submodules/diff-gaussian-rasterization/build",
        "./src/static_scene/submodules/diff-gaussian-rasterization/diff_gaussian_rasterization.egg-info",
        "./src/static_scene/submodules/fused-ssim/build",
        "./src/static_scene/submodules/fused-ssim/fused_ssim.egg-info",
        "./src/static_scene/submodules/simple-knn/build",
        "./src/static_scene/submodules/simple-knn/simple_knn.egg-info",
    ]

    for dir in dirs_to_delete:
        print(f"Deleting {dir}")
        shutil.rmtree(dir, ignore_errors=True)