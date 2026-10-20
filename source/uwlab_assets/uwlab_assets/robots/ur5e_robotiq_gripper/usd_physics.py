# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Physics corrections for the released calibrated UR5e/Robotiq USD."""


def apply_ur5e_physics_corrections(root):
    """Remove redundant passive mimics and correct masses before initialization.

    Keep the physical linkage and the right-outer-knuckle mimic. The four
    passive mimics require finite joint limits, but activating them prevents
    the gripper from closing in Isaac Sim 6.1. Removing their APIs while
    retaining the original infinite limits fixes the startup errors (#39) and
    preserves free-space closing (#46). This candidate remains opt-in because
    the mass corrections change released-checkpoint performance.

    The enlarged clearance box remains collision-enabled but contributes only
    1 mg. The physical camera/mount collider contributes 30 g, with its center
    of mass and inertia computed by PhysX from that collider's geometry (#38).

    Outer knuckles use factory_v2's rigid-body mass/inertia (#40), rather than
    ineffective visual-mesh properties and PhysX's 1 kg sphere fallback. Joint
    armature is deliberately not imported from factory_v2.
    """
    from pxr import Gf, Sdf, UsdPhysics

    passive_joints = (
        "left_inner_knuckle_joint",
        "right_inner_knuckle_joint",
        "left_inner_finger_knuckle_joint",
        "right_inner_finger_knuckle_joint",
    )
    stage = root.GetStage()
    joints = {name: stage.GetPrimAtPath(root.GetPath().AppendChild(name)) for name in passive_joints}
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
    for prim in joints.values():
        schemas = prim.GetMetadata("apiSchemas")
        if schemas is None or "PhysxMimicJointAPI:rotZ" not in schemas.explicitItems:
            raise ValueError(f"Expected passive mimic schema on {prim.GetPath()}")
        prim.SetMetadata(
            "apiSchemas",
            Sdf.TokenListOp.CreateExplicit(
                [token for token in schemas.explicitItems if token != "PhysxMimicJointAPI:rotZ"]
            ),
        )
        prim.GetAttribute("physxMimicJoint:rotZ:gearing").Block()
        prim.GetAttribute("physxMimicJoint:rotZ:offset").Block()
        prim.GetRelationship("physxMimicJoint:rotZ:referenceJoint").SetTargets([])
    for path, mass in colliders.items():
        UsdPhysics.MassAPI.Apply(prims[path]).CreateMassAttr(mass)
    # factory_v2 asset revision 975b2393654e72157301942ca598ddacdd23970d.
    # Author on the actual rigid-body prims, not their visual meshes.
    for prim in knuckles.values():
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(0.00853)
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(2.89e-6, 1.87e-6, 1.22e-6))
