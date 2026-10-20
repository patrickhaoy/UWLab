.. _ur5e-asset-physics-validation:

UR5e camera and mimic candidate
==============================

The combined physics candidate is **experimental and opt-in**. It does not preserve
released-checkpoint performance. Existing robot configurations keep their original
spawn function. This prevents the candidate from silently changing training,
reset collection, or playback physics.

Changes
-------

* Four passive-joint limits become finite (+/-1e10 degrees) and three mimic gearing
  signs follow `issue #46 <https://github.com/UW-Lab/UWLab/issues/46>`_. The physical
  linkage and all mimic rules remain present.
* The enlarged ``d415_and_cable`` safety collider stays enabled with mass 1e-6 kg
  (1 mg). Its dimensions and pose are unchanged.
* The physical ``D415_to_Robotiq_Mount`` collider gets mass 0.030 kg (30 g).
  PhysX estimates its center of mass and inertia from its collision geometry.
* Outer-knuckle mass/inertia remains unchanged; issue #40 is not addressed.

Native Isaac Sim 5.1 readbacks confirmed 0.639892 kg for the gripper base.
Deactivating the safety collider removes approximately 1 mg; deactivating the
physical mount removes 30 g. The free-space finger reaches 0.779264 rad for a
0.785398 rad close command and returns near zero when reopened. Finite-limit
startup errors disappear. The knuckles retain their 1 kg / 0.004 kg m^2 fallback.

Checkpoint compatibility
------------------------

Measured on 2026-10-19 with an RTX PRO 6000. Each cell scores exactly the first
128 episodes, seed 20261014, Reaching resets (ObjectAnywhereEEAnywhere), 160 control
steps / 16 seconds, with the standard evaluation randomizations. Every paired
run has identical initial robot and object state hashes. Success is sampled from
ProgressContext immediately before reset. Rectangle/wall was intentionally excluded.

Each runtime uses its own released seed42 checkpoints and reset datasets. These
are comparisons within each runtime; the Lab 2 and Lab 3 checkpoints differ.
Deterministic Lab 3 actor inference was verified against RSL-RL MLPModel for all
five checkpoints. This is a single-seed compatibility check, not an estimate of
performance across training seeds or reset distributions.

.. list-table:: Successful first episodes out of 128
   :header-rows: 1

   * - Task
     - Sim 5.1 original
     - Sim 5.1 candidate
     - Sim 6.1 / Lab 3 original
     - Sim 6.1 / Lab 3 candidate
   * - Peg
     - 125
     - 107
     - 125
     - 0
   * - Leg / tabletop
     - 117
     - 52
     - 118
     - 0
   * - Drawer
     - 125
     - 126
     - 0
     - 0
   * - Cube
     - 121
     - 112
     - 118
     - 0
   * - Cupcake / plate
     - 123
     - 102
     - 124
     - 52

Lab 3 Peg ablations isolate both problems: mimic corrections alone score 0/128;
camera/mount mass changes alone score 22/128; original scores 125/128. The mimic-only
case retains the original native randomized masses. Thus neither individual
change establishes compatibility on this runtime. Lab 3 Drawer also fails with
the original asset; that baseline failure cannot be attributed to this candidate.

Usage and reproduction
----------------------

The candidate is applied as stage overrides before PhysX initialization; the
Hugging Face USD and the cache are not modified. To select it deliberately:

.. code-block:: python

   from uwlab_assets.robots.ur5e_robotiq_gripper.ur5e_robotiq_2f85_gripper import (
       spawn_ur5e_with_candidate_physics,
   )

   env_cfg.scene.robot.spawn.func = spawn_ur5e_with_candidate_physics

To repeat a Lab 3 comparison from the repository in the configured Isaac Lab 3
environment (repeat for leg, drawer, cube, and cupcake):

.. code-block:: bash

   python scripts_v2/tools/eval_ur5e_asset_compatibility.py \
       --task peg --variant original --output_dir ./asset-validation \
       --visualizer none --device cuda:0
   python scripts_v2/tools/eval_ur5e_asset_compatibility.py \
       --task peg --variant candidate --output_dir ./asset-validation \
       --visualizer none --device cuda:0

Use ``--checkpoint PATH`` to evaluate another compatible Lab 3 state checkpoint.
The output records checkpoint hashes, initial-state hashes, native randomized
mass/inertia, and every first-episode outcome. The seed, environment count, reset
mode, and horizon are fixed in the script for this comparison.

Full measured results, checkpoint URLs/SHA256 hashes, runtime identities, and
per-environment success flags are in
:download:`asset_physics_validation.json <asset_physics_validation.json>`.

Further tuning or retraining is needed before changing production defaults.
The #46 workaround validated on Sim 5.1 must not be assumed compatible with Sim 6.1.
