# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Single-change UR5e ablations, each applied independently to the original USD.

FactoryV2 values come from asset revision 975b2393654e72157301942ca598ddacdd23970d.
These spawners are opt-in and must not be composed for the single-change study.
"""


def apply_single_change(root, variant):
    """Apply exactly one named physics change before articulation initialization."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    from .usd_physics import apply_ur5e_camera_mount_mass, remove_ur5e_passive_mimics

    stage = root.GetStage()

    def required(path):
        prim = stage.GetPrimAtPath(root.GetPath().AppendPath(path))
        if not prim.IsValid():
            raise ValueError(f"UR5e ablation requires {root.GetPath()}/{path}")
        return prim

    if variant == "knuckles":
        for name in ("left_outer_knuckle", "right_outer_knuckle"):
            mass = UsdPhysics.MassAPI.Apply(required(name))
            mass.CreateMassAttr(0.00853)
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(2.89e-6, 1.87e-6, 1.22e-6))
    elif variant == "safety":
        UsdPhysics.MassAPI.Apply(required("robotiq_base_link/collisions/d415_and_cable")).CreateMassAttr(1.0e-6)
    elif variant == "mount":
        apply_ur5e_camera_mount_mass(root)
    elif variant == "armature":
        for name in (
            "right_outer_knuckle_joint",
            "left_inner_knuckle_joint",
            "right_inner_knuckle_joint",
            "left_inner_finger_knuckle_joint",
            "right_inner_finger_knuckle_joint",
        ):
            prim = required(name)
            # PhysxJointAPI is already applied to these joints in the original USD.
            prim.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float).Set(0.01)
    elif variant == "fingertip":
        base = required("robotiq_base_link")
        body_path = root.GetPath().AppendChild("robotiq_fingertip_centered")
        joint_path = base.GetPath().AppendChild("robotiq_fingertip_centered_joint")
        if stage.GetPrimAtPath(body_path) or stage.GetPrimAtPath(joint_path):
            raise ValueError("Fingertip ablation requires the original USD without a fingertip reference body")
        body = UsdGeom.Xform.Define(stage, body_path)
        body.AddTranslateOp().Set(Gf.Vec3d(0.8158983588219914, 0.4025371719924862, 0.06314838764945212))
        body.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(0.5017787749752872, -0.49821080453558864, -0.49821751005795073, 0.5017801987922175)
        )
        body.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.0))
        prim = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(False)
        prim.AddAppliedSchema("PhysxRigidBodyAPI")
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(1.0e-6)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1.0e-9))
        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([base.GetPath()])
        joint.CreateBody1Rel().SetTargets([body_path])
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.15397842, 8.088007e-7, -1.2546345e-6))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
        joint.CreateBreakForceAttr(3.4028234663852886e38)
        joint.CreateBreakTorqueAttr(3.4028234663852886e38)
    elif variant == "fix39":
        remove_ur5e_passive_mimics(root)
    else:
        raise ValueError(f"Unknown UR5e ablation: {variant}")


def _spawn(variant, prim_path, cfg, translation=None, orientation=None):
    import isaaclab.sim as sim_utils

    from .ur5e_robotiq_2f85_gripper import spawn_ur5e_without_physics_materials

    prim = spawn_ur5e_without_physics_materials(prim_path, cfg, translation, orientation)
    for root in sim_utils.find_matching_prims(prim_path) or [prim]:
        apply_single_change(root, variant)
    return prim


def spawn_knuckles(prim_path, cfg, translation=None, orientation=None):
    return _spawn("knuckles", prim_path, cfg, translation, orientation)


def spawn_safety(prim_path, cfg, translation=None, orientation=None):
    return _spawn("safety", prim_path, cfg, translation, orientation)


def spawn_mount(prim_path, cfg, translation=None, orientation=None):
    return _spawn("mount", prim_path, cfg, translation, orientation)


def spawn_armature(prim_path, cfg, translation=None, orientation=None):
    return _spawn("armature", prim_path, cfg, translation, orientation)


def spawn_fingertip(prim_path, cfg, translation=None, orientation=None):
    return _spawn("fingertip", prim_path, cfg, translation, orientation)


def spawn_fix39(prim_path, cfg, translation=None, orientation=None):
    return _spawn("fix39", prim_path, cfg, translation, orientation)
