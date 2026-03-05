# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dexsuite Franka XHand environments."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="UW-Dexsuite-Franka-XHand-Reorient-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_franka_xhand_env_cfg:DexsuiteFrankaXHandReorientEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteFrankaXHandPPORunnerCfg",
    },
)

gym.register(
    id="UW-Dexsuite-Franka-XHand-Reorient-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_franka_xhand_env_cfg:DexsuiteFrankaXHandReorientEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteFrankaXHandPPORunnerCfg",
    },
)

gym.register(
    id="UW-Dexsuite-Franka-XHand-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_franka_xhand_env_cfg:DexsuiteFrankaXHandLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteFrankaXHandPPORunnerCfg",
    },
)

gym.register(
    id="UW-Dexsuite-Franka-XHand-Lift-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_franka_xhand_env_cfg:DexsuiteFrankaXHandLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteFrankaXHandPPORunnerCfg",
    },
)
