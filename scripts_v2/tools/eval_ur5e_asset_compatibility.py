# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Compare calibrated UR5e physics using released Isaac Lab 3 state experts.

Run one process per task/variant. Scores the first episode per environment before
reset, rather than a rolling success average. The original variant uses the
production default; the candidate explicitly selects the experimental spawner.
"""
import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--variant", required=True, choices=["original", "candidate"])
parser.add_argument("--task", required=True, choices=["peg", "leg", "drawer", "cube", "cupcake"])
parser.add_argument("--output_dir", required=True)
parser.add_argument("--checkpoint", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
import hashlib
import json
import time
import torch
from torch import nn

from isaaclab.envs import ManagerBasedRLEnv

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR, resolve_cloud_path

from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.rl_state_cfg import (
    Ur5eRobotiq2f85RelCartesianOSCEvalCfg,
    variants,
)

r = Path(args.output_dir).resolve()
r.mkdir(parents=True, exist_ok=True)
checkpoint = Path(
    resolve_cloud_path(
        args.checkpoint
        or f"{UWLAB_CLOUD_ASSETS_DIR}/Policies/OmniReset/state_based_experts/{args.task}_state_rl_expert_seed42.pt"
    )
)
cfg = Ur5eRobotiq2f85RelCartesianOSCEvalCfg()
cfg.seed = 20261014
cfg.scene.num_envs = 128
cfg.sim.device = args.device
pairs = {
    "peg": ("peg", "peghole"),
    "leg": ("fbleg", "fbtabletop"),
    "drawer": ("fbdrawerbottom", "fbdrawerbox"),
    "cube": ("cube", "cube"),
    "cupcake": ("cupcake", "plate"),
}
insertive, receptive = pairs[args.task]
cfg.scene.insertive_object = variants["scene.insertive_object"][insertive].copy()
cfg.scene.receptive_object = variants["scene.receptive_object"][receptive].copy()
# Keep the production default unchanged; explicitly select the candidate.
if args.variant == "candidate":
    from uwlab_assets.robots.ur5e_robotiq_gripper.ur5e_robotiq_2f85_gripper import spawn_ur5e_with_candidate_physics

    cfg.scene.robot.spawn.func = spawn_ur5e_with_candidate_physics
revision = "83860532010b2737aa80d6e8621235ee554186f0"
for name in ["insertive_object", "receptive_object", "table", "ur5_metal_support"]:
    obj = getattr(cfg.scene, name)
    obj.spawn.usd_path = obj.spawn.usd_path.replace("/resolve/main/", "/resolve/" + revision + "/")
cfg.events.reset_from_reset_states.params["dataset_dir"] = cfg.events.reset_from_reset_states.params[
    "dataset_dir"
].replace("/resolve/main/", "/resolve/" + revision + "/")
env = ManagerBasedRLEnv(cfg=cfg)
assert env.max_episode_length == 160, env.max_episode_length
state = torch.load(checkpoint, map_location=args.device, weights_only=False)["actor_state_dict"]
modules = []
nobs = state["mlp.0.weight"].shape[1]
# Reconstruct deterministic inference; the first seven outputs are action means.
# Verified against RSL-RL MLPModel on all five released checkpoints.
for i, (a, b) in enumerate(zip([nobs, 512, 256, 128, 64], [512, 256, 128, 64, 14])):
    modules.append(nn.Linear(a, b))
    if i < 4:
        modules.append(nn.ELU())
actor = nn.Sequential(*modules).to(args.device).eval()
actor.load_state_dict({k.removeprefix("mlp."): v for k, v in state.items() if k.startswith("mlp.")})
obs, _ = env.reset()
assert obs["policy"].shape == (128, nobs), obs["policy"].shape
initial_joint_sha256 = hashlib.sha256(env.scene["robot"].data.joint_pos.cpu().numpy().tobytes()).hexdigest()
initial_object_sha256 = hashlib.sha256(
    env.scene["insertive_object"].data.root_state_w.cpu().numpy().tobytes()
).hexdigest()
robot = env.scene["robot"]
physics = {
    "body_names": robot.body_names,
    "mass_kg": robot.data.body_mass[0].tolist(),
    "inertia_kg_m2": robot.data.body_inertia[0].tolist(),
}
records = {}
original_reset = env._reset_idx
steps = 0


# Snapshot success before manager reset; keep exactly the first episode per environment.
def reset_idx(ids):
    success = env.reward_manager.get_term_cfg("progress_context").func.success
    for idx in ids.tolist():
        if idx not in records:
            records[idx] = {
                "success": bool(success[idx]),
                "length": int(env.episode_length_buf[idx]),
                "control_step": steps,
                "timeout": bool(env.reset_time_outs[idx]),
            }
    original_reset(ids)


env._reset_idx = reset_idx
start = time.monotonic()
with torch.inference_mode():
    for steps in range(1, 181):
        actions = actor((obs["policy"] - state["obs_normalizer._mean"]) / (state["obs_normalizer._std"] + 0.01))[:, :7]
        assert torch.isfinite(actions).all()
        obs, _, _, _, _ = env.step(actions)
        if steps % 40 == 0:
            print("EVAL_PROGRESS", args.variant, steps, len(records), flush=True)
        if len(records) == 128:
            break
assert len(records) == 128, f"Incomplete evaluation: {len(records)}/128 episodes"
result = {
    "variant": args.variant,
    "task": args.task,
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "initial_joint_sha256": initial_joint_sha256,
    "initial_object_sha256": initial_object_sha256,
    "physics_env0_after_randomization": physics,
    "episodes": len(records),
    "successes": sum(x["success"] for x in records.values()),
    "seed": cfg.seed,
    "control_steps": steps,
    "seconds": time.monotonic() - start,
    "reset_distribution": "ObjectAnywhereEEAnywhere (Reaching)",
    "records": records,
}
(r / ("expert-" + args.task + "-" + args.variant + ".json")).write_text(json.dumps(result, indent=2))
print(
    "EVAL_RESULT",
    json.dumps({k: v for k, v in result.items() if k not in ["records", "physics_env0_after_randomization"]}),
    flush=True,
)
env.close()
app.close()
