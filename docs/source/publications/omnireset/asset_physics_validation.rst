.. _ur5e-asset-physics-validation:

UR5e camera, mimic, and knuckle candidate
========================================

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
* Both outer-knuckle **rigid-body prims** now have mass 0.00853 kg (8.53 g)
  and diagonal inertia (2.89e-6, 1.87e-6, 1.22e-6) kg m^2. These values are
  copied exactly from factory_v2 asset revision
  ``975b2393654e72157301942ca598ddacdd23970d`` and address issue #40.
  Joint armature is unchanged; no other factory_v2 settings are imported.

Native Isaac Sim 5.1 readbacks confirm both knuckles use the explicit 8.53 g
mass and the exact factory_v2 diagonal inertia. The fallback warnings disappear.
The free-space finger reaches 0.778947 rad for a 0.785398 rad close command and
returns near zero when reopened. The camera masses and other physics settings
are preserved from the previous candidate.

Knuckle-fix checkpoint comparison
--------------------------------

The new candidate was evaluated on the same five tasks and frozen seed42
checkpoints as the previous candidate, with 128 first episodes per task/runtime,
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

Correcting the native mass properties does **not** establish compatibility with
existing checkpoints or resolve the functional policy failures in #46. The
candidate remains opt-in and production defaults remain unchanged.

Native readbacks, per-environment success flags, checkpoint identities, and
paired initial-state hashes are available in
:download:`knuckle_physics_validation.json <knuckle_physics_validation.json>`.
The reproduction command below now evaluates this updated candidate.

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
