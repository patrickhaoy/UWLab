# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from ... import mdp as task_mdp


@configclass
class CurriculumCfg:
    """Curriculum terms for assembly RL training.

    Uses an adaptive difficulty scheduler driven by ProgressContext success to control:
    1. Gravity ramping from 0 to -9.81
    2. Partial assembly sampling range expanding from near-goal to full dataset
    """

    adr = CurrTerm(
        func=task_mdp.AssemblyDifficultyScheduler,
        params={
            "init_difficulty": 0,
            "min_difficulty": 0,
            "max_difficulty": 10,
        },
    )

    gravity_adr = CurrTerm(
        func=task_mdp.modify_term_cfg,
        params={
            "address": "events.variable_gravity.params.gravity_distribution_params",
            "modify_fn": task_mdp.initial_final_interpolate_fn,
            "modify_params": {
                "initial_value": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                "final_value": ((0.0, 0.0, -9.81), (0.0, 0.0, -9.81)),
                "difficulty_term_str": "adr",
            },
        },
    )

    partial_assembly_range_adr = CurrTerm(
        func=task_mdp.modify_term_cfg,
        params={
            "address": "events.reset_insertive_object.params.max_difficulty_frac",
            "modify_fn": task_mdp.initial_final_interpolate_fn,
            "modify_params": {
                "initial_value": 0.1,
                "final_value": 1.0,
                "difficulty_term_str": "adr",
            },
        },
    )
