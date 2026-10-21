# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Stage the pinned FactoryV2 robot with explicit Isaac Lab 3 metadata conversion.

The USD is copied unchanged. Only metadata.yaml's wxyz quaternion becomes xyzw;
all masses, armatures, calibration and offset positions remain unchanged.
"""

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

import yaml

REVISION = "975b2393654e72157301942ca598ddacdd23970d"
BASE_URL = (
    f"https://huggingface.co/datasets/patrickhaoy/uwlab-assets/resolve/{REVISION}"
    "/Robots/UniversalRobots/Ur5e2f85RobotiqGripperCalibrated"
)
USD_NAME = "ur5e_robotiq_gripper_d415_mount_safety_calibrated.usd"
HASHES = {
    USD_NAME: "9c68b15e87a009cfc3f1ca9c4f9b1376e90f29b1801bfdecb5caae2e4091650c",
    "metadata.yaml": "c9b6bc99aa76ef4cbb9e4400b552b931c639e39c83fc9ae0710a618f4c3f475b",
}


def convert_metadata(data):
    """Convert the one quaternion in this exact, hash-pinned metadata schema."""
    if "quat_convention" in data:
        raise ValueError("Expected the pinned legacy metadata, not an already converted file")
    result = dict(data)
    result["gripper_offset"] = dict(data["gripper_offset"])
    w, x, y, z = data["gripper_offset"]["quat"]
    result["gripper_offset"]["quat"] = [x, y, z, w]
    result["quat_convention"] = "xyzw"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, digest in HASHES.items():
        with urlopen(f"{BASE_URL}/{name}", timeout=120) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"Unexpected content hash for {name}")
        if name == "metadata.yaml":
            content = yaml.safe_dump(convert_metadata(yaml.safe_load(content)), sort_keys=False).encode()
        target = args.output_dir / name
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(target)
        print(f"Staged {target}: sha256={hashlib.sha256(content).hexdigest()}")


if __name__ == "__main__":
    main()
