# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Panda arm with Robotera XHand.

The following configurations are available:

* :obj:`FRANKA_XHAND_CFG`: Franka Panda + XHand right hand with implicit actuator model.

Reference:

* https://franka.de/research
* https://www.robotera.com/en/goods1/4.html

"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

##
# Configuration
##

FRANKA_XHAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Robots/Franka/FrankaXHand/franka_right_xhand.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=1,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            # Franka arm
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            # XHand
            "right_hand_thumb_bend_joint": 0.0,
            "right_hand_thumb_rota_joint1": 0.0,
            "right_hand_thumb_rota_joint2": 0.0,
            "right_hand_index_bend_joint": 0.0,
            "right_hand_index_joint1": 0.0,
            "right_hand_index_joint2": 0.0,
            "right_hand_mid_joint1": 0.0,
            "right_hand_mid_joint2": 0.0,
            "right_hand_ring_joint1": 0.0,
            "right_hand_ring_joint2": 0.0,
            "right_hand_pinky_joint1": 0.0,
            "right_hand_pinky_joint2": 0.0,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=1.875,
            stiffness=400.0,
            damping=80.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=40.0,
            velocity_limit=2.31,
            stiffness=400.0,
            damping=80.0,
        ),
        "xhand": ImplicitActuatorCfg(
            joint_names_expr=["right_hand_.*"],
            effort_limit=0.95,
            velocity_limit=8.48,
            stiffness=20.0,
            damping=1.0,
            armature=0.001,
            friction=0.2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
