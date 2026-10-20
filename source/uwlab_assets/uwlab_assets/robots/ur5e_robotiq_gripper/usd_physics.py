# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Physics corrections for the released calibrated UR5e/Robotiq USD."""


def apply_ur5e_physics_corrections(root):
    """Correct mimic constraints and camera mass before PhysX initialization.

    Keep the physical linkage and outer-knuckle mass/inertia unchanged. Finite
    limits activate four mimic constraints; three gearing signs must also change
    to avoid opposing the linkage (UW-Lab/UWLab#39 and #46).

    The enlarged clearance box remains collision-enabled but contributes only
    1 mg. The physical camera/mount collider contributes 30 g, with its center
    of mass and inertia computed by PhysX from that collider's geometry (#38).
    """
    from pxr import UsdPhysics

    signs = {
        "left_inner_knuckle_joint": -1.0,
        "right_inner_knuckle_joint": 1.0,
        "left_inner_finger_knuckle_joint": -1.0,
        "right_inner_finger_knuckle_joint": -1.0,
    }
    stage = root.GetStage()
    joints = {name: stage.GetPrimAtPath(root.GetPath().AppendChild(name)) for name in signs}
    colliders = {
        "robotiq_base_link/collisions/d415_and_cable": 1.0e-6,
        "robotiq_base_link/collisions/D415_to_Robotiq_Mount": 0.030,
    }
    prims = {path: stage.GetPrimAtPath(root.GetPath().AppendPath(path)) for path in colliders}
    for name, prim in {**joints, **prims}.items():
        if not prim.IsValid():
            raise ValueError(f"UR5e physics corrections require {root.GetPath()}/{name}")
    for name, prim in joints.items():
        joint = UsdPhysics.RevoluteJoint(prim)
        joint.CreateLowerLimitAttr(-1.0e10)  # USD revolute-joint limits are degrees.
        joint.CreateUpperLimitAttr(1.0e10)
        prim.GetAttribute("physxMimicJoint:rotZ:gearing").Set(signs[name])
    for path, mass in colliders.items():
        UsdPhysics.MassAPI.Apply(prims[path]).CreateMassAttr(mass)
