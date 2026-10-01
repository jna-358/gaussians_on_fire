# Copyright (c) 2026 Jakob Nazarenus
# SPDX-License-Identifier: MIT

"""Preprocess HF fire_actioncam scenes into training bundles.

    uv run python -m preprocess --scenes 011

Defaults: dataset in data/hf, intermediates in data/work, bundles in data/bundles.
"""
import argparse
import time

from . import bundle, depth, flow, frames, masks, static, static_gs, voxels
from .common import SCENES, Scene, stage_done

STAGES = [("frames", frames.run), ("flow", flow.run), ("masks", masks.run), ("voxel_flow", voxels.run),
          ("background", static.run_background), ("static_init", static.run_static_init),
          ("static_gs", static_gs.run),           ("mono_depth", depth.run_mono_depth)]


def _duration(seconds):
    minutes, seconds = divmod(int(round(seconds)), 60)
    return f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-root", default="data/hf", help="Local copy of the HF dataset (contains scenes/); default data/hf")
    p.add_argument("--work", default="data/work", help="Work directory; each scene goes to <work>/<hf_scene>/; default data/work")
    p.add_argument("--out", default="data/bundles", help="Bundle root; each scene's bundle goes to <out>/<hf_scene>/ (rewritten every run); default data/bundles")
    p.add_argument("--scenes", nargs="+", default=sorted(SCENES), help="HF scene ids (default: all)")
    p.add_argument("--stages", nargs="+", default=[s for s, _ in STAGES], choices=[s for s, _ in STAGES])
    args = p.parse_args()

    stages = [(name, run) for name, run in STAGES if name in args.stages]
    for n, hf_scene in enumerate(args.scenes, 1):
        scene = Scene(args.hf_root, hf_scene, args.work)
        print(f"Scene {hf_scene} ({scene.name}), {n} of {len(args.scenes)}")
        scene_start = time.time()
        for i, (name, run) in enumerate(stages, 1):
            if stage_done(scene, name):
                print(f"  [{i}/{len(stages)}] {name}: already done")
                continue
            print(f"  [{i}/{len(stages)}] {name}")
            start = time.time()
            run(scene)
            print(f"      done in {_duration(time.time() - start)}")
        missing = [name for name, _ in STAGES if not stage_done(scene, name)]
        if missing:
            print(f"  bundle: skipped, stages not done yet: {', '.join(missing)}")
        else:
            print("  bundle")
            bundle.run(scene, args.out)
        print(f"  scene done in {_duration(time.time() - scene_start)}")


if __name__ == "__main__":
    main()
