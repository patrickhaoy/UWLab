# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.dexsuite import dexsuite_env_cfg as dexsuite
from isaaclab_tasks.manager_based.manipulation.dexsuite import mdp

from uwlab_assets.robots.franka_xhand import FRANKA_XHAND_CFG


@configclass
class FrankaXHandActionCfg:
    arm = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=["panda_joint.*"], scale=0.1)
    hand = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=["right_hand_.*"], scale=0.1)


@configclass
class FrankaXHandReorientRewardCfg(dexsuite.RewardsCfg):
    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=0.5,
        params={"threshold": 1.0},
    )


# Mapping from Allegro sensor names (hardcoded in contacts() reward) to XHand link
# bodies that have CollisionAPI. The *_tip prims are visual-only (no colliders), so
# we use the parent *_link2 bodies which are the last links with collision geometry.
_CONTACT_SENSOR_MAP = {
    "thumb_link_3_object_s": "right_hand/right_hand_thumb_rota_link2",
    "index_link_3_object_s": "right_hand/right_hand_index_rota_link2",
    "middle_link_3_object_s": "right_hand/right_hand_mid_link2",
    "ring_link_3_object_s": "right_hand/right_hand_ring_link2",
}


@configclass
class FrankaXHandMixinCfg:
    rewards: FrankaXHandReorientRewardCfg = FrankaXHandReorientRewardCfg()
    actions: FrankaXHandActionCfg = FrankaXHandActionCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.commands.object_pose.body_name = "palm_lower"
        self.scene.robot = FRANKA_XHAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        for attr_name, prim_suffix in _CONTACT_SENSOR_MAP.items():
            setattr(
                self.scene,
                attr_name,
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/" + prim_suffix,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )

        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": list(_CONTACT_SENSOR_MAP.keys())},
            clip=(-20.0, 20.0),
        )
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = [
            "palm_lower",
            "right_hand_.*_tip",
        ]
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=["palm_lower", "right_hand_.*_tip"]
        )

        self.events.reset_robot_wrist_joint.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names="panda_joint7"
        )


@configclass
class DexsuiteFrankaXHandReorientEnvCfg(FrankaXHandMixinCfg, dexsuite.DexsuiteReorientEnvCfg):
    pass


@configclass
class DexsuiteFrankaXHandReorientEnvCfg_PLAY(FrankaXHandMixinCfg, dexsuite.DexsuiteReorientEnvCfg_PLAY):
    pass


@configclass
class DexsuiteFrankaXHandLiftEnvCfg(FrankaXHandMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass


@configclass
class DexsuiteFrankaXHandLiftEnvCfg_PLAY(FrankaXHandMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    pass
