# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor

from uwlab_tasks.manager_based.manipulation.reset_states.assembly_keypoints import Offset
from uwlab_tasks.manager_based.manipulation.reset_states.mdp import utils


def target_asset_pose_in_root_asset_frame(
    env: ManagerBasedEnv,
    target_asset_cfg: SceneEntityCfg,
    root_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_asset_offset=None,
    root_asset_offset=None,
    rotation_repr: str = "quat",
):
    target_asset: RigidObject | Articulation = env.scene[target_asset_cfg.name]
    root_asset: RigidObject | Articulation = env.scene[root_asset_cfg.name]

    target_body_idx = 0 if isinstance(target_asset_cfg.body_ids, slice) else target_asset_cfg.body_ids
    root_body_idx = 0 if isinstance(root_asset_cfg.body_ids, slice) else root_asset_cfg.body_ids

    target_pos = target_asset.data.body_link_pos_w[:, target_body_idx].view(-1, 3)
    target_quat = target_asset.data.body_link_quat_w[:, target_body_idx].view(-1, 4)
    root_pos = root_asset.data.body_link_pos_w[:, root_body_idx].view(-1, 3)
    root_quat = root_asset.data.body_link_quat_w[:, root_body_idx].view(-1, 4)

    if root_asset_offset is not None:
        root_pos, root_quat = root_asset_offset.combine(root_pos, root_quat)
    if target_asset_offset is not None:
        target_pos, target_quat = target_asset_offset.combine(target_pos, target_quat)

    target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(root_pos, root_quat, target_pos, target_quat)

    if rotation_repr == "axis_angle":
        axis_angle = math_utils.axis_angle_from_quat(target_quat_b)
        return torch.cat([target_pos_b, axis_angle], dim=1)
    elif rotation_repr == "quat":
        return torch.cat([target_pos_b, target_quat_b], dim=1)
    else:
        raise ValueError(f"Invalid rotation_repr: {rotation_repr}. Must be one of: 'quat', 'axis_angle'")


class target_asset_pose_in_root_asset_frame_with_metadata(ManagerTermBase):
    """Get target asset pose in root asset frame with offsets automatically read from metadata.

    This is similar to target_asset_pose_in_root_asset_frame but automatically reads the
    assembled offsets from the asset USD metadata instead of requiring manual specification.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        target_asset_cfg: SceneEntityCfg = cfg.params.get("target_asset_cfg")
        root_asset_cfg: SceneEntityCfg = cfg.params.get("root_asset_cfg", SceneEntityCfg("robot"))
        target_asset_offset_metadata_key: str = cfg.params.get("target_asset_offset_metadata_key")
        root_asset_offset_metadata_key: str = cfg.params.get("root_asset_offset_metadata_key")

        self.target_asset: RigidObject | Articulation = env.scene[target_asset_cfg.name]
        self.root_asset: RigidObject | Articulation = env.scene[root_asset_cfg.name]
        self.target_asset_cfg = target_asset_cfg
        self.root_asset_cfg = root_asset_cfg
        self.rotation_repr = cfg.params.get("rotation_repr", "quat")

        # Read root asset offset from metadata
        if root_asset_offset_metadata_key is not None:
            root_usd_path = self.root_asset.cfg.spawn.usd_path
            root_metadata = utils.read_metadata_from_usd_directory(root_usd_path)
            root_offset_data = root_metadata.get(root_asset_offset_metadata_key)
            self.root_asset_offset = Offset(pos=root_offset_data.get("pos"), quat=root_offset_data.get("quat"))
        else:
            self.root_asset_offset = None

        # Read target asset offset from metadata
        if target_asset_offset_metadata_key is not None:
            target_usd_path = self.target_asset.cfg.spawn.usd_path
            target_metadata = utils.read_metadata_from_usd_directory(target_usd_path)
            target_offset_data = target_metadata.get(target_asset_offset_metadata_key)
            self.target_asset_offset = Offset(pos=target_offset_data.get("pos"), quat=target_offset_data.get("quat"))
        else:
            self.target_asset_offset = None

    def __call__(
        self,
        env: ManagerBasedEnv,
        target_asset_cfg: SceneEntityCfg,
        root_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        target_asset_offset_metadata_key: str | None = None,
        root_asset_offset_metadata_key: str | None = None,
        rotation_repr: str = "quat",
    ) -> torch.Tensor:
        target_body_idx = 0 if isinstance(self.target_asset_cfg.body_ids, slice) else self.target_asset_cfg.body_ids
        root_body_idx = 0 if isinstance(self.root_asset_cfg.body_ids, slice) else self.root_asset_cfg.body_ids

        target_pos = self.target_asset.data.body_link_pos_w[:, target_body_idx].view(-1, 3)
        target_quat = self.target_asset.data.body_link_quat_w[:, target_body_idx].view(-1, 4)
        root_pos = self.root_asset.data.body_link_pos_w[:, root_body_idx].view(-1, 3)
        root_quat = self.root_asset.data.body_link_quat_w[:, root_body_idx].view(-1, 4)

        if self.root_asset_offset is not None:
            root_pos, root_quat = self.root_asset_offset.combine(root_pos, root_quat)
        if self.target_asset_offset is not None:
            target_pos, target_quat = self.target_asset_offset.combine(target_pos, target_quat)

        target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(root_pos, root_quat, target_pos, target_quat)

        if rotation_repr == "axis_angle":
            axis_angle = math_utils.axis_angle_from_quat(target_quat_b)
            return torch.cat([target_pos_b, axis_angle], dim=1)
        elif rotation_repr == "quat":
            return torch.cat([target_pos_b, target_quat_b], dim=1)
        else:
            raise ValueError(f"Invalid rotation_repr: {rotation_repr}. Must be one of: 'quat', 'axis_angle'")


def asset_link_velocity_in_root_asset_frame(
    env: ManagerBasedEnv,
    target_asset_cfg: SceneEntityCfg,
    root_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    target_asset: RigidObject | Articulation = env.scene[target_asset_cfg.name]
    root_asset: RigidObject | Articulation = env.scene[root_asset_cfg.name]

    target_body_idx = 0 if isinstance(target_asset_cfg.body_ids, slice) else target_asset_cfg.body_ids

    asset_lin_vel_b, _ = math_utils.subtract_frame_transforms(
        root_asset.data.root_pos_w,
        root_asset.data.root_quat_w,
        target_asset.data.body_lin_vel_w[:, target_body_idx].view(-1, 3),
    )
    asset_ang_vel_b, _ = math_utils.subtract_frame_transforms(
        root_asset.data.root_pos_w,
        root_asset.data.root_quat_w,
        target_asset.data.body_ang_vel_w[:, target_body_idx].view(-1, 3),
    )

    return torch.cat([asset_lin_vel_b, asset_ang_vel_b], dim=1)


def get_material_properties(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.root_physx_view.get_material_properties().view(env.num_envs, -1)


def get_mass(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.root_physx_view.get_masses().view(env.num_envs, -1)


def get_joint_friction(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_friction_coeff.view(env.num_envs, -1)


def get_joint_armature(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_armature.view(env.num_envs, -1)


def get_joint_stiffness(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_stiffness.view(env.num_envs, -1)


def get_joint_damping(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_damping.view(env.num_envs, -1)


def time_left(env) -> torch.Tensor:
    if hasattr(env, "episode_length_buf"):
        life_left = 1 - (env.episode_length_buf.float() / env.max_episode_length)
    else:
        life_left = torch.zeros(env.num_envs, device=env.device, dtype=torch.float)
    return life_left.view(-1, 1)


# ---------------------------------------------------------------------------
# Tactile / contact observation helpers
# ---------------------------------------------------------------------------


def _get_episode_start_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns 0 at timestep 0 (force data may be stale), 1 otherwise. Shape: (num_envs, 1)."""
    return (env.episode_length_buf > 0).float().unsqueeze(-1)


def _contact_binary(force: torch.Tensor, threshold: float) -> torch.Tensor:
    """Binary contact flag from force vector. Shape: (num_envs, 1)."""
    return (torch.norm(force, dim=-1, keepdim=True) > threshold).float()


def _normalize_force_direction(force: torch.Tensor) -> torch.Tensor:
    """Unit direction of force; zero when no contact. Shape matches input."""
    norm = torch.norm(force, dim=-1, keepdim=True).clamp(min=1e-8)
    return force / norm


def fingertip_contact_force_b(
    env: ManagerBasedRLEnv,
    contact_sensor_name: str,
    root_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    root_body_name: str = "robotiq_base_link",
) -> torch.Tensor:
    """Contact force from a fingertip sensor in the gripper body frame. Shape: (num_envs, 3)."""
    root_asset: Articulation = env.scene[root_asset_cfg.name]
    root_body_idx = root_asset.body_names.index(root_body_name)
    root_quat_w = root_asset.data.body_link_quat_w[:, root_body_idx].view(-1, 4)

    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_name]
    force_w = contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    force_b = math_utils.quat_apply_inverse(root_quat_w, force_w)
    return force_b


def fingertip_contact_binary(
    env: ManagerBasedRLEnv,
    contact_sensor_name: str,
    root_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    root_body_name: str = "robotiq_base_link",
    threshold: float = 1e-7,
) -> torch.Tensor:
    """Binary contact detection for a fingertip. Shape: (num_envs, 1).

    Returns 0 at episode start (timestep 0) since force data may be stale.
    """
    force_b = fingertip_contact_force_b(env, contact_sensor_name, root_asset_cfg, root_body_name)
    result = _contact_binary(force_b, threshold)
    return result * _get_episode_start_mask(env)


def fingertip_contact_direction(
    env: ManagerBasedRLEnv,
    contact_sensor_name: str,
    root_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    root_body_name: str = "robotiq_base_link",
) -> torch.Tensor:
    """Normalized contact direction from a fingertip in body frame. Shape: (num_envs, 3).

    Returns 0 at episode start (timestep 0) since force data may be stale.
    """
    force_b = fingertip_contact_force_b(env, contact_sensor_name, root_asset_cfg, root_body_name)
    result = _normalize_force_direction(force_b)
    return result * _get_episode_start_mask(env)


# ---------------------------------------------------------------------------
# Object point cloud observation
# ---------------------------------------------------------------------------


class object_point_cloud_b(ManagerTermBase):
    """Object surface point cloud expressed in a reference asset's root frame.

    Points are pre-sampled on the object surface in local frame at init,
    then transformed to world and into the reference frame each step.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        self.ref_asset_cfg: SceneEntityCfg = cfg.params.get("ref_asset_cfg", SceneEntityCfg("robot"))
        num_points: int = cfg.params.get("num_points", 10)
        self.object: RigidObject = env.scene[self.object_cfg.name]
        self.ref_asset: Articulation = env.scene[self.ref_asset_cfg.name]

        self.points_local = utils.sample_object_point_cloud(
            env.num_envs,
            num_points,
            self.object.cfg.prim_path,
            device=env.device,
        )
        self.points_w = torch.zeros_like(self.points_local) if self.points_local is not None else None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        ref_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        num_points: int = 10,
        flatten: bool = False,
    ) -> torch.Tensor:
        if self.points_local is None:
            shape = (env.num_envs, num_points * 3) if flatten else (env.num_envs, num_points, 3)
            return torch.zeros(shape, device=env.device)

        ref_pos_w = self.ref_asset.data.root_pos_w.unsqueeze(1).expand(-1, num_points, -1)
        ref_quat_w = self.ref_asset.data.root_quat_w.unsqueeze(1).expand(-1, num_points, -1)

        object_pos_w = self.object.data.root_pos_w.unsqueeze(1).expand(-1, num_points, -1)
        object_quat_w = self.object.data.root_quat_w.unsqueeze(1).expand(-1, num_points, -1)

        self.points_w = math_utils.quat_apply(object_quat_w, self.points_local) + object_pos_w
        points_b, _ = math_utils.subtract_frame_transforms(ref_pos_w, ref_quat_w, self.points_w, None)

        return points_b.view(env.num_envs, -1) if flatten else points_b
