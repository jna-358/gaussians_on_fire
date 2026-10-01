# Copyright (c) 2026 Jakob Nazarenus
# SPDX-License-Identifier: MIT

"""Train/test split for split-free fire bundles (preprocess/bundle.py), with a leak check.

The bundle has masks and voxel flow for every frame at two flow strides; each
file's provenance lists the window frames it was computed from. Test frames are
those with frame % test_every == 0. A train frame uses the stride-1 files unless
their flow window contains a test frame, then the stride-2 files; test frames
get stride-1 masks (evaluation only) and no voxel flow. Every file picked for
training is checked against the test set, and loading fails if one leaks.
"""
import json
import os

STRIDES = (1, 2)


class LeakError(RuntimeError):
    pass


def _window_frames(entry):
    return {f for source in entry["sources"].values() for f in source["window_frames"]}


def _pick(provenance, names_by_stride, test_frames, what):
    """First stride whose file avoids the test frames."""
    for stride in STRIDES:
        name = names_by_stride[stride]
        if not _window_frames(provenance[name]) & test_frames:
            return name
    raise LeakError(f"{what}: every stride's flow window contains a test frame")


def select(path, eval, test_every):
    """Split the bundle; returns frames (with is_test and mask path), voxel files and grid path."""
    transforms = json.load(open(os.path.join(path, "transforms.json")))
    num_frames = transforms["num_frames"]
    test_frames = {r for r in range(num_frames) if eval and test_every > 0 and r % test_every == 0}
    mask_prov = json.load(open(os.path.join(path, "masks", "provenance.json")))["files"]
    voxel_prov = json.load(open(os.path.join(path, "voxel_flow", "provenance.json")))["files"]

    frames = []
    for frame in transforms["frames"]:
        r, c = frame["frame"], frame["cam"]
        names = {s: f"stride{s}/{r:04d}_{c}.png" for s in STRIDES}
        is_test = r in test_frames
        mask = names[1] if is_test else _pick(mask_prov, names, test_frames, f"mask of frame {r} cam {c}")
        frames.append(dict(frame, is_test=is_test, mask_path=os.path.join(path, "masks", mask)))

    voxel_frames = []
    for r in range(num_frames):
        if r in test_frames:
            continue
        name = _pick(voxel_prov, {s: f"stride{s}/{r:04d}.npz" for s in STRIDES}, test_frames, f"voxel flow of frame {r}")
        voxel_frames.append((os.path.join(path, "voxel_flow", name), voxel_prov[name]["time_us"] * 1e-3))

    # Independent re-check of everything that feeds training.
    for frame in frames:
        if not frame["is_test"]:
            rel = os.path.relpath(frame["mask_path"], os.path.join(path, "masks"))
            if _window_frames(mask_prov[rel]) & test_frames:
                raise LeakError(f"training mask {rel} was computed from a test frame")
    for file, _ in voxel_frames:
        rel = os.path.relpath(file, os.path.join(path, "voxel_flow"))
        if _window_frames(voxel_prov[rel]) & test_frames:
            raise LeakError(f"voxel flow {rel} was computed from a test frame")

    strides = [int(os.path.basename(os.path.dirname(f))[len("stride"):]) for f, _ in voxel_frames]
    print(f"Split: {len(test_frames)} test / {num_frames - len(test_frames)} train frames; "
          f"voxel init from {strides.count(1)} stride-1 + {strides.count(2)} stride-2 frames; no leaks")
    return {"transforms": transforms, "frames": frames, "voxel_frames": voxel_frames,
            "grid": os.path.join(path, "voxel_flow", "grid.npz")}
