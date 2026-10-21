.. _ur5e-asset-physics-validation:

UR5e camera, passive-mimic, and knuckle candidate
================================================

The combined physics candidate is **experimental and opt-in**. It does not preserve
released-checkpoint performance. Existing robot configurations keep their original
spawn function. This prevents the candidate from silently changing training,
reset collection, or playback physics.

Changes
-------

* The four passive ``PhysxMimicJointAPI:rotZ`` rules are removed while their
  physical joints and infinite limits remain. The ``right_outer_knuckle_joint``
  mimic remains. This removes the finite-limit startup errors in
  `issue #39 <https://github.com/UW-Lab/UWLab/issues/39>`_ without activating
  passive mimic constraints that stall closing on Isaac Sim 6.1. The sign
  workaround reported in `issue #46 <https://github.com/UW-Lab/UWLab/issues/46>`_
  is not applied to mimics that no longer exist.
* The enlarged ``d415_and_cable`` safety collider stays enabled with mass 1e-6 kg
  (1 mg). Its dimensions and pose are unchanged.
* The physical ``D415_to_Robotiq_Mount`` collider gets mass 0.030 kg (30 g).
  PhysX estimates its center of mass and inertia from its collision geometry.
* Both outer-knuckle **rigid-body prims** now have mass 0.00853 kg (8.53 g)
  and diagonal inertia (2.89e-6, 1.87e-6, 1.22e-6) kg m^2. These values are
  copied exactly from factory_v2 asset revision
  ``975b2393654e72157301942ca598ddacdd23970d`` and address issue #40.
  Joint armature is unchanged; no other factory_v2 settings are imported.

Native Isaac Sim 5.1 readbacks from the earlier candidate confirm both knuckles
use the explicit 8.53 g mass and the exact factory_v2 diagonal inertia. The
current candidate's Lab 3 free-space finger reaches 0.780684 rad for a
0.785398 rad close command and returns near zero when reopened. No passive mimic
parser errors or outer-knuckle fallback warnings were logged for that candidate.

With the original masses, removing the four passive mimics preserved the
released Peg checkpoint: 125/128 first episodes on Sim 5.1 and 128/128 on
Lab 3, compared with 125/128 original on each runtime. Adding the camera and
knuckle mass corrections reduced that frozen checkpoint to 23/128 and 20/128,
respectively, despite normal free-space gripper motion. This corrected-physics
profile requires new policy training or separately validated adaptation. These
tests do not establish object-contact grasping or new-policy convergence.
Checkpoint hashes, paired initial-state hashes, and the free-space joint readings
are recorded in :download:`passive_mimic_validation.json <passive_mimic_validation.json>`.

Historical finite-mimic candidate comparison
--------------------------------------------

The previous candidate at commit ``18bd4d5`` was evaluated on five tasks with
frozen seed42 checkpoints, with 128 first episodes per task/runtime,
seed 20261014, Reaching resets, and a 16-second horizon. Initial robot/object
state hashes match both previous candidate and original runs in every comparison.
Original and previous-candidate measurements below are reused from the prior
validation; only the changed candidate was rerun. Rectangle/wall is excluded.

.. list-table:: Successful first episodes out of 128, before and after adding #40
   :header-rows: 1

   * - Task
     - Sim 5.1 previous candidate
     - Sim 5.1 with #40
     - Lab 3 previous candidate
     - Lab 3 with #40
   * - Peg
     - 107
     - 20
     - 0
     - 0
   * - Leg / tabletop
     - 52
     - 0
     - 0
     - 0
   * - Drawer
     - 126
     - 13
     - 0
     - 0
   * - Cube
     - 112
     - 61
     - 0
     - 0
   * - Cupcake / plate
     - 102
     - 31
     - 52
     - 0

These results describe the superseded candidate with four finite passive
limits and three mimic sign changes. They are retained as evidence of why the
current candidate removes the four passive mimic rules. Production defaults
remain unchanged.

Native readbacks, per-environment success flags, checkpoint identities, and
paired initial-state hashes are available in
:download:`knuckle_physics_validation.json <knuckle_physics_validation.json>`.
The reproduction command below evaluates the current candidate instead.

Previous camera/mimic-only validation
-------------------------------------

The following historical results are for commit ``165aac0``, before the knuckle
fix. Measured on 2026-10-19 with an RTX PRO 6000. Each cell scores exactly the first
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

For a **mass-only training ablation**, keep the released USD's passive mimic
rules, joint limits, and armature while applying only the safety-collider,
camera-mount, and outer-knuckle mass/inertia changes above:

.. code-block:: python

   from uwlab_assets.robots.ur5e_robotiq_gripper.ur5e_robotiq_2f85_gripper import (
       spawn_ur5e_with_mass_corrections,
   )

   env_cfg.scene.robot.spawn.func = spawn_ur5e_with_mass_corrections

The original passive-mimic finite-limit parser errors remain in Isaac Sim 6.1
for this ablation; no mimic rule is removed or altered. This variant isolates
the mass changes from the combined candidate and is opt-in.

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
