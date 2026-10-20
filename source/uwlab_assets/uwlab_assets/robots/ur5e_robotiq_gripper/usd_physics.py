# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Physics corrections for the released calibrated UR5e/Robotiq USD."""


def apply_ur5e_physics_corrections(root):
    """Set candidate mimic, camera, and outer-knuckle physics before initialization.

    Keep the physical linkage unchanged. The Sim 5.1 workaround uses finite
    limits and three corrected gearing signs (UW-Lab/UWLab#39 and #46). This
    candidate remains opt-in because checkpoint performance regresses.

    The enlarged clearance box remains collision-enabled but contributes only
    1 mg. The physical camera/mount collider contributes 30 g, with its center
    of mass and inertia computed by PhysX from that collider's geometry (#38).

    Outer knuckles use factory_v2's rigid-body mass/inertia (#40), rather than
    ineffective visual-mesh properties and PhysX's 1 kg sphere fallback. Joint
    armature is deliberately not imported from factory_v2.
    """
    from pxr import Gf, UsdPhysics

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
    knuckles = {
        name: stage.GetPrimAtPath(root.GetPath().AppendChild(name))
        for name in ("left_outer_knuckle", "right_outer_knuckle")
    }
    for name, prim in {**joints, **prims, **knuckles}.items():
        if not prim.IsValid():
            raise ValueError(f"UR5e physics corrections require {root.GetPath()}/{name}")
    for name, prim in joints.items():
        joint = UsdPhysics.RevoluteJoint(prim)
        joint.CreateLowerLimitAttr(-1.0e10)  # USD revolute-joint limits are degrees.
        joint.CreateUpperLimitAttr(1.0e10)
        prim.GetAttribute("physxMimicJoint:rotZ:gearing").Set(signs[name])
    for path, mass in colliders.items():
        UsdPhysics.MassAPI.Apply(prims[path]).CreateMassAttr(mass)
    # factory_v2 asset revision 975b2393654e72157301942ca598ddacdd23970d.
    # Author on the actual rigid-body prims, not their visual meshes.
    for prim in knuckles.values():
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(0.00853)
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(2.89e-6, 1.87e-6, 1.22e-6))
