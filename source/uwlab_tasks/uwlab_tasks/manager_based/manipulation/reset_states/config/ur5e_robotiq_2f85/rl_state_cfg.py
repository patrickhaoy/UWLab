# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR
from uwlab_assets.robots.ur5e_robotiq_gripper import (
    EXPLICIT_UR5E_ROBOTIQ_2F85,
    IMPLICIT_UR5E_ROBOTIQ_2F85,
    Ur5eRobotiq2f85RelativeJointPositionAction,
)

from uwlab_tasks.manager_based.manipulation.reset_states.config.ur5e_robotiq_2f85.actions import (
    Ur5eRobotiq2f85RelativeOSCAction,
)
from uwlab_tasks.manager_based.manipulation.reset_states.config.ur5e_robotiq_2f85.curriculum_cfg import (
    CurriculumCfg,
)

from ... import mdp as task_mdp


@configclass
class RlStateSceneCfg(InteractiveSceneCfg):
    """Scene configuration for RL state environment."""

    robot = EXPLICIT_UR5E_ROBOTIQ_2F85.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=EXPLICIT_UR5E_ROBOTIQ_2F85.spawn.replace(activate_contact_sensors=True),
    )

    insertive_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd",
            activate_contact_sensors=True,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    receptive_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/PegHole/peg_hole.usd",
            activate_contact_sensors=True,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                # receptive object does not move
                kinematic_enabled=True,
            ),
            # since kinematic_enabled=True, mass does not matter
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # Environment
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, -0.881), rot=(0.707, 0.0, 0.0, -0.707)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Mounts/UWPatVention/pat_vention.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    ur5_metal_support = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/UR5MetalSupport",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0, -0.013), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Mounts/UWPatVention2/Ur5MetalSupport/ur5plate.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    right_inner_finger_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_inner_finger",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/InsertiveObject"],
    )

    left_inner_finger_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_inner_finger",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/InsertiveObject"],
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.868)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=10000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class BaseEventCfg:
    """Configuration for events."""

    # mode: startup (randomize dynamics)
    robot_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.2, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("robot"),
            "make_consistent": True,
        },
    )

    # use large friction to avoid slipping
    insertive_object_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (1.0, 2.0),
            "dynamic_friction_range": (0.9, 1.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("insertive_object"),
            "make_consistent": True,
        },
    )

    # use large friction to avoid slipping
    receptive_object_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (1.0, 2.0),
            "dynamic_friction_range": (0.9, 1.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("receptive_object"),
            "make_consistent": True,
        },
    )

    table_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.3, 0.6),
            "dynamic_friction_range": (0.2, 0.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("table"),
            "make_consistent": True,
        },
    )

    randomize_robot_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_insertive_object_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("insertive_object"),
            # we assume insertive object is somewhere between 20g and 200g
            "mass_distribution_params": (0.02, 0.2),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_receptive_object_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("receptive_object"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_table_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_robot_joint_parameters = EventTerm(
        func=task_mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*", "finger_joint"]),
            "friction_distribution_params": (0.25, 4.0),
            "armature_distribution_params": (0.25, 4.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    randomize_gripper_actuator_parameters = EventTerm(
        func=task_mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "stiffness_distribution_params": (0.5, 2.0),
            "damping_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # mode: reset
    reset_everything = EventTerm(func=task_mdp.reset_scene_to_default, mode="reset", params={})

    variable_gravity = EventTerm(
        func=task_mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            "operation": "abs",
        },
    )

    reset_robot_pose = EventTerm(
        func=task_mdp.reset_root_states_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.01, 0.01),
                "y": (-0.059, -0.019),
                "z": (-0.01, 0.01),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfgs": {"robot": SceneEntityCfg("robot"), "ur5_metal_support": SceneEntityCfg("ur5_metal_support")},
        },
    )

    reset_receptive_object_pose = EventTerm(
        func=task_mdp.reset_root_states_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.3, 0.55),
                "y": (-0.1, 0.3),
                "z": (0.0, 0.001),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-np.pi / 12, np.pi / 12),
            },
            "velocity_range": {},
            "asset_cfgs": {"receptive_object": SceneEntityCfg("receptive_object")},
            "offset_asset_cfg": SceneEntityCfg("ur5_metal_support"),
            "use_bottom_offset": True,
        },
    )

    reset_insertive_object = EventTerm(
        func=task_mdp.InlineInsertiveObjectReset,
        mode="reset",
        params={
            "prob_partial_assembly": 0.5,
            "max_difficulty_frac": 0.1,
            "uniform_params": {
                "pose_range": {
                    "x": (0.3, 0.55),
                    "y": (-0.1, 0.5),
                    "z": (0.0, 0.3),
                    "roll": (-np.pi, np.pi),
                    "pitch": (-np.pi, np.pi),
                    "yaw": (-np.pi, np.pi),
                },
                "velocity_range": {},
                "asset_cfgs": {"insertive_object": SceneEntityCfg("insertive_object")},
                "offset_asset_cfg": SceneEntityCfg("ur5_metal_support"),
                "use_bottom_offset": True,
            },
            "partial_assembly_params": {
                "base_path": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/PartialAssemblies/ObjectPairs",
                "insertive_object_cfg": SceneEntityCfg("insertive_object"),
                "receptive_object_cfg": SceneEntityCfg("receptive_object"),
            },
        },
    )

    reset_end_effector_pose = EventTerm(
        func=task_mdp.reset_end_effector_round_fixed_asset,
        mode="reset",
        params={
            "fixed_asset_cfg": SceneEntityCfg("robot"),
            "fixed_asset_offset": None,
            "pose_range_b": {
                "x": (0.3, 0.7),
                "y": (-0.4, 0.4),
                "z": (0.0, 0.5),
                "roll": (0.0, 0.0),
                "pitch": (np.pi / 4, 3 * np.pi / 4),
                "yaw": (np.pi / 2, 3 * np.pi / 2),
            },
            "robot_ik_cfg": SceneEntityCfg(
                "robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"], body_names="robotiq_base_link"
            ),
        },
    )


@configclass
class TrainEventCfg(BaseEventCfg):
    """Configuration for training events.

    Inherits all inline resets and gravity curriculum from BaseEventCfg.
    """

    pass


@configclass
class EvalEventCfg(BaseEventCfg):
    """Configuration for evaluation events.

    Uses full gravity and full partial assembly dataset range (no curriculum).
    """

    variable_gravity = EventTerm(
        func=task_mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, -9.81], [0.0, 0.0, -9.81]),
            "operation": "abs",
        },
    )

    reset_insertive_object = EventTerm(
        func=task_mdp.InlineInsertiveObjectReset,
        mode="reset",
        params={
            "prob_partial_assembly": 0.5,
            "max_difficulty_frac": 1.0,
            "uniform_params": {
                "pose_range": {
                    "x": (0.3, 0.55),
                    "y": (-0.1, 0.5),
                    "z": (0.0, 0.3),
                    "roll": (-np.pi, np.pi),
                    "pitch": (-np.pi, np.pi),
                    "yaw": (-np.pi, np.pi),
                },
                "velocity_range": {},
                "asset_cfgs": {"insertive_object": SceneEntityCfg("insertive_object")},
                "offset_asset_cfg": SceneEntityCfg("ur5_metal_support"),
                "use_bottom_offset": True,
            },
            "partial_assembly_params": {
                "base_path": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/PartialAssemblies/ObjectPairs",
                "insertive_object_cfg": SceneEntityCfg("insertive_object"),
                "receptive_object_cfg": SceneEntityCfg("receptive_object"),
            },
        },
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    task_command = task_mdp.TaskCommandCfg(
        asset_cfg=SceneEntityCfg("robot", body_names="body"),
        resampling_time_range=(1e6, 1e6),
        success_position_threshold=0.005,
        success_orientation_threshold=0.025,
        insertive_asset_cfg=SceneEntityCfg("insertive_object"),
        receptive_asset_cfg=SceneEntityCfg("receptive_object"),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        prev_actions = ObsTerm(func=task_mdp.last_action)

        joint_pos = ObsTerm(func=task_mdp.joint_pos)

        end_effector_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame_with_metadata,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "target_asset_offset_metadata_key": "gripper_offset",
                "root_asset_offset_metadata_key": "offset",
                "rotation_repr": "axis_angle",
            },
        )

        insertive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame_with_metadata,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "root_asset_offset_metadata_key": "gripper_offset",
                "rotation_repr": "axis_angle",
            },
        )

        receptive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("receptive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "rotation_repr": "axis_angle",
            },
        )

        insertive_asset_in_receptive_asset_frame: ObsTerm = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("receptive_object"),
                "rotation_repr": "axis_angle",
            },
        )

        right_finger_contact_binary = ObsTerm(
            func=task_mdp.fingertip_contact_binary,
            params={
                "contact_sensor_name": "right_inner_finger_contact_sensor",
                "root_asset_cfg": SceneEntityCfg("robot"),
                "root_body_name": "robotiq_base_link",
                "threshold": 5.0,
            },
        )

        left_finger_contact_binary = ObsTerm(
            func=task_mdp.fingertip_contact_binary,
            params={
                "contact_sensor_name": "left_inner_finger_contact_sensor",
                "root_asset_cfg": SceneEntityCfg("robot"),
                "root_body_name": "robotiq_base_link",
                "threshold": 5.0,
            },
        )

        insertive_object_point_cloud = ObsTerm(
            func=task_mdp.object_point_cloud_b,
            params={
                "object_cfg": SceneEntityCfg("insertive_object"),
                "ref_asset_cfg": SceneEntityCfg("robot"),
                "num_points": 64,
                "flatten": True,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations for policy group."""

        prev_actions = ObsTerm(func=task_mdp.last_action)

        joint_pos = ObsTerm(func=task_mdp.joint_pos)

        end_effector_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame_with_metadata,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "target_asset_offset_metadata_key": "gripper_offset",
                "root_asset_offset_metadata_key": "offset",
                "rotation_repr": "axis_angle",
            },
        )

        insertive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame_with_metadata,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "root_asset_offset_metadata_key": "gripper_offset",
                "rotation_repr": "axis_angle",
            },
        )

        receptive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("receptive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "rotation_repr": "axis_angle",
            },
        )

        insertive_asset_in_receptive_asset_frame: ObsTerm = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("receptive_object"),
                "rotation_repr": "axis_angle",
            },
        )

        right_finger_contact_binary = ObsTerm(
            func=task_mdp.fingertip_contact_binary,
            params={
                "contact_sensor_name": "right_inner_finger_contact_sensor",
                "root_asset_cfg": SceneEntityCfg("robot"),
                "root_body_name": "robotiq_base_link",
                "threshold": 5.0,
            },
        )

        left_finger_contact_binary = ObsTerm(
            func=task_mdp.fingertip_contact_binary,
            params={
                "contact_sensor_name": "left_inner_finger_contact_sensor",
                "root_asset_cfg": SceneEntityCfg("robot"),
                "root_body_name": "robotiq_base_link",
                "threshold": 5.0,
            },
        )

        insertive_object_point_cloud = ObsTerm(
            func=task_mdp.object_point_cloud_b,
            params={
                "object_cfg": SceneEntityCfg("insertive_object"),
                "ref_asset_cfg": SceneEntityCfg("robot"),
                "num_points": 64,
                "flatten": True,
            },
        )

        # privileged observations
        time_left = ObsTerm(func=task_mdp.time_left)

        joint_vel = ObsTerm(func=task_mdp.joint_vel)

        end_effector_vel_lin_ang_b = ObsTerm(
            func=task_mdp.asset_link_velocity_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "root_asset_cfg": SceneEntityCfg("robot"),
            },
        )

        robot_material_properties = ObsTerm(
            func=task_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("robot")}
        )

        insertive_object_material_properties = ObsTerm(
            func=task_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("insertive_object")}
        )

        receptive_object_material_properties = ObsTerm(
            func=task_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("receptive_object")}
        )

        table_material_properties = ObsTerm(
            func=task_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("table")}
        )

        robot_mass = ObsTerm(func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("robot")})

        insertive_object_mass = ObsTerm(
            func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("insertive_object")}
        )

        receptive_object_mass = ObsTerm(
            func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("receptive_object")}
        )

        table_mass = ObsTerm(func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("table")})

        robot_joint_friction = ObsTerm(func=task_mdp.get_joint_friction, params={"asset_cfg": SceneEntityCfg("robot")})

        robot_joint_armature = ObsTerm(func=task_mdp.get_joint_armature, params={"asset_cfg": SceneEntityCfg("robot")})

        robot_joint_stiffness = ObsTerm(
            func=task_mdp.get_joint_stiffness, params={"asset_cfg": SceneEntityCfg("robot")}
        )

        robot_joint_damping = ObsTerm(func=task_mdp.get_joint_damping, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:

    # internal tracker (weight=0 so it runs but doesn't contribute reward)
    progress_context = RewTerm(
        func=task_mdp.ProgressContext,  # type: ignore
        weight=0.0,
        params={
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "receptive_asset_cfg": SceneEntityCfg("receptive_object"),
        },
    )

    # reach: EE-to-object distance
    ee_asset_distance = RewTerm(
        func=task_mdp.ee_asset_distance_tanh,
        weight=1.0,
        params={
            "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
            "target_asset_cfg": SceneEntityCfg("insertive_object"),
            "root_asset_offset_metadata_key": "gripper_offset",
            "std": 0.4,
        },
    )

    # task tracking (contact-gated)
    dense_success_reward = RewTerm(
        func=task_mdp.dense_success_reward_contact_gated,
        weight=2.0,
        params={"std": 1.0},
    )

    # success
    success_reward = RewTerm(func=task_mdp.success_reward, weight=10.0)

    # contact: reward both fingers gripping the object
    good_finger_contact = RewTerm(
        func=task_mdp.gripper_contact,
        weight=0.5,
        params={"threshold": 1.0},
    )

    # action penalties
    action_magnitude = RewTerm(func=task_mdp.action_l2_clamped, weight=-0.005)

    action_rate = RewTerm(func=task_mdp.action_rate_l2_clamped, weight=-0.005)

    # early termination penalty
    early_termination = RewTerm(
        func=task_mdp.is_terminated_term,
        weight=-1.0,
        params={"term_keys": "abnormal_robot"},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)

    abnormal_robot = DoneTerm(func=task_mdp.abnormal_robot_state)


def make_insertive_object(usd_path: str):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=True,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def make_receptive_object(usd_path: str):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=True,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )


variants = {
    "scene.insertive_object": {
        "fbleg": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/SquareLeg/square_leg.usd"),
        "fbdrawerbottom": make_insertive_object(
            f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/DrawerBottom/drawer_bottom.usd"
        ),
        "peg": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd"),
        "cupcake": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/CupCake/cupcake.usd"),
        "cube": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/InsertiveCube/insertive_cube.usd"),
        "rectangle": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Rectangle/rectangle.usd"),
    },
    "scene.receptive_object": {
        "fbtabletop": make_receptive_object(
            f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/SquareTableTop/square_table_top.usd"
        ),
        "fbdrawerbox": make_receptive_object(
            f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/DrawerBox/drawer_box.usd"
        ),
        "peghole": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/PegHole/peg_hole.usd"),
        "plate": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Plate/plate.usd"),
        "cube": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/ReceptiveCube/receptive_cube.usd"),
        "wall": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Wall/wall.usd"),
    },
}


@configclass
class Ur5eRobotiq2f85RlStateCfg(ManagerBasedRLEnvCfg):
    scene: RlStateSceneCfg = RlStateSceneCfg(num_envs=32, env_spacing=1.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: BaseEventCfg = MISSING
    curriculum: CurriculumCfg | None = CurriculumCfg()
    commands: CommandsCfg = CommandsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 0.0, 0.75), origin_type="world", env_index=0, asset_name="robot")
    variants = variants

    def __post_init__(self):
        self.decimation = 12
        self.episode_length_s = 16.0
        # simulation settings
        self.sim.dt = 1 / 120.0

        # Contact and solver settings
        self.sim.physx.solver_type = 1
        self.sim.physx.max_position_iteration_count = 192
        self.sim.physx.max_velocity_iteration_count = 1
        self.sim.physx.bounce_threshold_velocity = 0.02
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.friction_correlation_distance = 0.0005

        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_collision_stack_size = 2**31

        # Render settings
        self.sim.render.enable_dlssg = True
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_reflections = True
        self.sim.render.enable_dl_denoiser = True


# Training configurations
@configclass
class Ur5eRobotiq2f85RelCartesianOSCTrainCfg(Ur5eRobotiq2f85RlStateCfg):
    """Training configuration for Relative Cartesian OSC action space."""

    events: TrainEventCfg = TrainEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = EXPLICIT_UR5E_ROBOTIQ_2F85.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=EXPLICIT_UR5E_ROBOTIQ_2F85.spawn.replace(activate_contact_sensors=True),
        )

        self.events.randomize_robot_actuator_parameters = EventTerm(
            func=task_mdp.randomize_operational_space_controller_gains,
            mode="reset",
            params={
                "action_name": "arm",
                "stiffness_distribution_params": (0.7, 1.3),
                "damping_distribution_params": (0.9, 1.1),
                "operation": "scale",
                "distribution": "uniform",
            },
        )


@configclass
class Ur5eRobotiq2f85RelJointPosTrainCfg(Ur5eRobotiq2f85RlStateCfg):
    """Training configuration for Relative Joint Position action space."""

    events: TrainEventCfg = TrainEventCfg()
    actions: Ur5eRobotiq2f85RelativeJointPositionAction = Ur5eRobotiq2f85RelativeJointPositionAction()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = IMPLICIT_UR5E_ROBOTIQ_2F85.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=IMPLICIT_UR5E_ROBOTIQ_2F85.spawn.replace(activate_contact_sensors=True),
        )

        self.events.randomize_robot_actuator_parameters = EventTerm(
            func=task_mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*", "finger_joint"]),
                "stiffness_distribution_params": (0.5, 2.0),
                "damping_distribution_params": (0.5, 2.0),
                "operation": "scale",
                "distribution": "log_uniform",
            },
        )


# Evaluation configurations
@configclass
class Ur5eRobotiq2f85RelCartesianOSCEvalCfg(Ur5eRobotiq2f85RlStateCfg):
    """Evaluation configuration for Relative Cartesian OSC action space."""

    events: EvalEventCfg = EvalEventCfg()
    curriculum = None
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = EXPLICIT_UR5E_ROBOTIQ_2F85.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=EXPLICIT_UR5E_ROBOTIQ_2F85.spawn.replace(activate_contact_sensors=True),
        )

        self.events.randomize_robot_actuator_parameters = EventTerm(
            func=task_mdp.randomize_operational_space_controller_gains,
            mode="reset",
            params={
                "action_name": "arm",
                "stiffness_distribution_params": (0.7, 1.3),
                "damping_distribution_params": (0.9, 1.1),
                "operation": "scale",
                "distribution": "uniform",
            },
        )


@configclass
class Ur5eRobotiq2f85RelJointPosEvalCfg(Ur5eRobotiq2f85RlStateCfg):
    """Evaluation configuration for Relative Joint Position action space."""

    events: EvalEventCfg = EvalEventCfg()
    curriculum = None
    actions: Ur5eRobotiq2f85RelativeJointPositionAction = Ur5eRobotiq2f85RelativeJointPositionAction()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = IMPLICIT_UR5E_ROBOTIQ_2F85.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=IMPLICIT_UR5E_ROBOTIQ_2F85.spawn.replace(activate_contact_sensors=True),
        )

        self.events.randomize_robot_actuator_parameters = EventTerm(
            func=task_mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*", "finger_joint"]),
                "stiffness_distribution_params": (0.5, 2.0),
                "damping_distribution_params": (0.5, 2.0),
                "operation": "scale",
                "distribution": "log_uniform",
            },
        )
