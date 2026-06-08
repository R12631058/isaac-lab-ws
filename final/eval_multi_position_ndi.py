"""Multi-Position Generalization Evaluation Script with NDI Tracking

This script is a direct copy of multi_position_eval.py with NDI check functionality added.
All core evaluation logic is identical to multi_position_eval.py.
"""

import argparse
import sys
import os
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import json

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Multi-position generalization evaluation with NDI")
parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
parser.add_argument("--num_envs", type=int, default=50, help="Number of parallel environments")
parser.add_argument("--checkpoint", type=str,
                    default="logs\\rsl_rl\\tm5_reach_stable_v2\\2026-02-04_00-28-35\\model_9999.pt")
parser.add_argument("--save_file", type=str, default="scripts/isaaclab_ws/reachability_map/multi_position_results_ndi.json")
parser.add_argument("--x_min", type=float, default=0.3, help="Minimum X position (World Frame relative to Env Origin)")
parser.add_argument("--x_max", type=float, default=1.0, help="Maximum X position")
parser.add_argument("--use_usd_target", action="store_true", help="Use /Root/Cube object from Env 0 as the reference target for ALL environments")
parser.add_argument("--target_local_x", type=float, default=None, help="Target X position (Environment Local Frame)")
parser.add_argument("--target_local_y", type=float, default=None, help="Target Y position (Environment Local Frame)")
parser.add_argument("--target_local_z", type=float, default=None, help="Target Z position (Environment Local Frame)")
parser.add_argument("--check_ndi", action="store_true", default=False, help="Enable NDI tracking analysis")
parser.add_argument("--verbose", action="store_true", help="Enable verbose NDI debug output")
parser.add_argument("--full_log", action="store_true", help="Enable full verbose terminal logs (default is compact error-only style)")
args_cli = parser.parse_args()
ERROR_ONLY_LOG = not args_cli.full_log

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import UsdGeom, Gf
import torch
import omni.usd

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_extension_linkout_cone_orientation_cfg import (
    TM5ExtensionLinkOutFanOrientationEnvCfg
)
from rsl_rl.modules import ActorCritic

# NDI imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ndi_detector_isaaclab import NDIConfig, NDIDetector

# robotarm_base local position (from test_multi_robot_parallel.py)
ROBOTARM_BASE_LOCAL_X = 0.3
ROBOTARM_BASE_LOCAL_Y = -0.1741
ROBOTARM_BASE_LOCAL_Z = 0.9493


# ---------------------------------------------------------------------------
# NDI Configuration Wrapper
# ---------------------------------------------------------------------------
class EnvNDIConfig(NDIConfig):
    """Custom NDI Config for individual environments"""
    def __init__(self, env_prim_path, detected_map=None):
        super().__init__()
        prefix_replacement = env_prim_path
        if detected_map and "emitter_path" in detected_map:
            p = detected_map["emitter_path"]
            if "/NDI/NDI_emitter" in p:
                prefix_replacement = p.split("/NDI/NDI_emitter")[0]
        self.START_PRIM_PATHS = [p.replace("/Root", prefix_replacement) for p in self.START_PRIM_PATHS]
        self.START_HIGHLIGHT_PATHS = [p.replace("/Root", prefix_replacement) for p in self.START_HIGHLIGHT_PATHS]
        self.END_PRIM_PATHS = [p.replace("/Root", prefix_replacement) for p in self.END_PRIM_PATHS]
        self.TRIGGER_VOLUME_PATH = self.TRIGGER_VOLUME_PATH.replace("/Root", prefix_replacement)
        if detected_map and "link_6_path" in detected_map:
            self.LINK_6_PATH = detected_map["link_6_path"]
        else:
            self.LINK_6_PATH = f"{prefix_replacement}/robotarm_base/robotarm_base/tm5_700/link_6"


def find_env_structure(stage, env_path):
    """Scan an environment prim to find NDI components"""
    result = {}
    env_prim = stage.GetPrimAtPath(env_path)
    if not env_prim.IsValid():
        return result

    def search(prim, depth=0):
        if depth > 10:
            return
        name = prim.GetName()
        path = str(prim.GetPath())
        if name == "link_6" and "link_6_path" not in result:
            result["link_6_path"] = path
        if name == "NDI_emitter" and "emitter_path" not in result:
            result["emitter_path"] = path
        for child in prim.GetChildren():
            search(child, depth + 1)
            if "link_6_path" in result and "emitter_path" in result:
                return

    search(env_prim)
    return result


# ---------------------------------------------------------------------------
# Core functions  identical to multi_position_eval.py
# ---------------------------------------------------------------------------
def randomize_robot_root_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
):
    """
    Randomize robot base position using Articulation API.
    Does NOT modify USD Xform (compatible with GPU pipeline).
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    num_envs_to_reset = len(env_ids)

    # Linear distribution for X to cover the range evenly
    x_offsets = torch.linspace(x_range[0], x_range[1], env.num_envs, device=env.device)[env_ids]

    # Constant for Y/Z
    y_offsets = torch.zeros(num_envs_to_reset, device=env.device) + y_range[0]
    z_offsets = torch.zeros(num_envs_to_reset, device=env.device) + z_range[0]

    robot = env.scene["robot"]

    # Get environment origins
    env_origins = env.scene.env_origins[env_ids]

    # Default local position of the robot base
    default_local_pos = torch.tensor(
        [ROBOTARM_BASE_LOCAL_X, ROBOTARM_BASE_LOCAL_Y, ROBOTARM_BASE_LOCAL_Z],
        device=env.device
    ).repeat(num_envs_to_reset, 1)

    # Apply shifts
    shifts = torch.stack([x_offsets, y_offsets, z_offsets], dim=1)

    # New World Position = Env Origin + Default Local + Shift
    new_root_pos = env_origins + default_local_pos + shifts

    # Get current orientation (keep it)
    root_quat = robot.data.root_quat_w[env_ids]

    # Set the new state directly to simulation
    robot.write_root_pose_to_sim(torch.cat([new_root_pos, root_quat], dim=-1), env_ids=env_ids)

    # Stop any drift
    zeros_vel = torch.zeros((num_envs_to_reset, 6), device=env.device)
    robot.write_root_velocity_to_sim(zeros_vel, env_ids=env_ids)


def get_usd_target_position(env: ManagerBasedRLEnv):
    """Get /Root/Cube position from Env 0 (World Frame)"""
    stage = env.unwrapped.scene.stage
    prim_path = "/World/envs/env_0/Root/Cube"

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        prim_path = "/World/envs/env_0/Root/target"
        prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        print(f"[WARN] Could not find target in USD at {prim_path}")
        return None

    xformable = UsdGeom.Xformable(prim)
    world_transform = xformable.ComputeLocalToWorldTransform(0)
    translation = world_transform.ExtractTranslation()

    target_world_env0 = torch.tensor([translation[0], translation[1], translation[2]], device=env.device)

    if env.scene.env_origins is not None:
        env0_origin = env.scene.env_origins[0]
        return target_world_env0 - env0_origin
    else:
        return target_world_env0


def run_evaluation(env, policy, num_episodes, device, x_min, x_max, fixed_target_local=None, detectors=None, error_only_log=False):
    """Run evaluation episodes across all environments

    Core logic is identical to multi_position_eval.py.
    NDI analysis is injected per-step when detectors are provided.
    """
    returns = []
    episode_lengths = []

    # NDI stats per env
    ndi_stats = {i: [] for i in range(len(detectors))} if detectors else {}

    num_envs = env.unwrapped.num_envs
    if not error_only_log:
        print(f"\n[INFO] Running {num_episodes} episodes across {num_envs} parallel environments...")
        print(f"[INFO] Total evaluations: {num_episodes * num_envs}\n")

    for episode_idx in range(num_episodes):
        obs, _ = env.reset()  # Randomizer will set different X positions

        # --- COMMAND OVERRIDE FOR FIXED WORLD TARGET ---
        if fixed_target_local is not None:
            with torch.no_grad():
                # 1. Get Robot Base World Pose
                robot = env.scene["robot"]
                robot_root_pos_w = robot.data.root_pos_w
                robot_root_quat_w = robot.data.root_quat_w

                # 2. Prepare Target World Pose (expand to batch)
                env_origins = env.scene.env_origins  # Shape (num_envs, 3)
                target_local_expanded = fixed_target_local.unsqueeze(0).repeat(num_envs, 1)
                target_pos_w = env_origins + target_local_expanded

                # Correct Target Orientation from Config (Needle Pointing Down)
                # Roll -180, Pitch 0, Yaw 180 => Quat (w,x,y,z) = (0, 0, 1, 0)
                target_quat_w = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device).unsqueeze(0).repeat(num_envs, 1)

                # 3. Compute Target in Robot Base Frame
                cmd_pos_b, cmd_quat_b = subtract_frame_transforms(
                    robot_root_pos_w, robot_root_quat_w,
                    target_pos_w, target_quat_w
                )

                # 4. Update Command Manager
                term_name = "ee_pose"
                try:
                    cmd_term = env.command_manager.get_term(term_name)
                    if cmd_term.command.shape[1] == 3:
                        cmd_term.command[:] = cmd_pos_b
                    elif cmd_term.command.shape[1] == 7:
                        cmd_term.command[:] = torch.cat([cmd_pos_b, cmd_quat_b], dim=1)

                    # 5. Re-compute Observations
                    obs = env.observation_manager.compute()

                except Exception as e:
                    print(f"[WARN] Failed to override command: {e}")
        # -----------------------------------------------

        episode_returns = torch.zeros(num_envs, device=device)
        episode_steps = torch.zeros(num_envs, dtype=torch.int, device=device)
        dones_all = torch.zeros(num_envs, dtype=torch.bool, device=device)

        while not dones_all.all():
            with torch.no_grad():
                obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
                actions = policy.act(obs_tensor, deterministic=True)

            obs, rewards, dones, truncated, info = env.step(actions)

            # --- NDI LIVE VISUALIZATION (no sim.step, no raycast ¡X does not disturb RL) ---
            # Redraws raycast lines every physics step using current arm position.
            # Line colors (green=visible / red=blocked) are from the last full NDI measurement.
            # clear_first=True only for the FIRST detector to avoid each detector wiping
            # the previous one's lines (debug_draw_interface.clear_lines() is global).
            if detectors:
                first_drawn = False
                for _d in detectors:
                    if _d is not None:
                        _d.draw_live(clear_first=not first_drawn)
                        first_drawn = True
            # -------------------------------------------------------------------------------

            # --- RE-APPLY COMMAND IF NEEDED (identical to multi_position_eval.py) ---
            if torch.any(dones) and fixed_target_local is not None:
                with torch.no_grad():
                    env_ids = torch.where(dones)[0]

                    robot = env.scene["robot"]
                    robot_root_pos_w = robot.data.root_pos_w[env_ids]
                    robot_root_quat_w = robot.data.root_quat_w[env_ids]

                    env_origins_reset = env.scene.env_origins[env_ids]
                    target_local_expanded = fixed_target_local.unsqueeze(0).repeat(len(env_ids), 1)
                    target_pos_w = env_origins_reset + target_local_expanded

                    target_quat_w = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device).unsqueeze(0).repeat(len(env_ids), 1)

                    cmd_pos_b, cmd_quat_b = subtract_frame_transforms(
                        robot_root_pos_w, robot_root_quat_w,
                        target_pos_w, target_quat_w
                    )

                    term_name = "ee_pose"
                    cmd_term = env.command_manager.get_term(term_name)

                    if cmd_term.command.shape[1] == 3:
                        cmd_term.command[env_ids] = cmd_pos_b
                    elif cmd_term.command.shape[1] == 7:
                        cmd_term.command[env_ids] = torch.cat([cmd_pos_b, cmd_quat_b], dim=1)

                    new_obs = env.observation_manager.compute()
                    if isinstance(obs, dict):
                        for k in obs.keys():
                            if k == "policy":
                                obs[k][env_ids] = new_obs[k][env_ids]
                    else:
                        obs[env_ids] = new_obs[env_ids]
            # -------------------------------------------

            # Update statistics for active environments (identical to multi_position_eval.py)
            active_envs = ~dones_all
            episode_returns[active_envs] += rewards[active_envs]
            episode_steps[active_envs] += 1
            dones_all = dones_all | dones | truncated

        # Collect results from all environments
        for env_idx in range(num_envs):
            returns.append(episode_returns[env_idx].item())
            episode_lengths.append(episode_steps[env_idx].item())

        mean_return = episode_returns.mean().item()
        mean_length = episode_steps.float().mean().item()

        # --- NDI ANALYSIS: run once per episode in final arm state ---
        # Must be OUTSIDE the while loop ¡X detector.update_detection() calls sim.step()
        # internally (to flush raycast callbacks), which would disturb the RL loop
        # if called per-step.
        if detectors:
            for i, detector in enumerate(detectors):
                if detector is None:
                    continue
                try:
                    debug_viz = (i == 0) or args_cli.verbose
                    status = detector.update_detection(
                        env.sim,
                        debug_enabled=debug_viz,
                        error_only_log=error_only_log,
                    )
                    visible_cnt = sum(1 for v in status.values() if v)
                    ndi_stats[i].append(visible_cnt)
                except Exception:
                    pass
        # -----------------------------------------------------------

        if not error_only_log:
            print(f"  Batch {episode_idx+1}/{num_episodes}: "
                  f"Mean Return = {mean_return:.2f}, "
                  f"Mean Length = {mean_length:.0f} "
                  f"({num_envs} envs)")

    # Calculate statistics
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    mean_length = np.mean(episode_lengths)
    success_rate = sum(1 for r in returns if r > 100) / len(returns) * 100

    ndi_results = {}
    if detectors:
        for i, stats in ndi_stats.items():
            if not stats:
                continue
            ndi_results[i] = float(np.mean(stats))

    return {
        "mean_return": mean_return,
        "std_return": std_return,
        "mean_length": mean_length,
        "success_rate": success_rate,
        "all_returns": returns,
        "all_lengths": episode_lengths,
        "ndi_results": ndi_results,
    }


def main():
    if not ERROR_ONLY_LOG:
        print("="*60)
        print("Multi-Position Generalization Evaluation + NDI")
        print("="*60)
        print(f"Configuration:")
        print(f"  Parallel Environments: {args_cli.num_envs}")
        print(f"  Episodes: {args_cli.episodes}")
        print(f"  Total Evaluations: {args_cli.num_envs * args_cli.episodes}")
        print(f"  Target Robot Base X Range: [{args_cli.x_min:.4f}, {args_cli.x_max:.4f}]")
        print(f"  NDI Check: {args_cli.check_ndi}")
        print(f"  Checkpoint: {args_cli.checkpoint}")
        print("="*60 + "\n")

    # Create environment config (identical to multi_position_eval.py)
    env_cfg = TM5ExtensionLinkOutFanOrientationEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 5.0
    env_cfg.scene.replicate_physics = False

    # Calculate offset range
    offset_min = args_cli.x_min - ROBOTARM_BASE_LOCAL_X
    offset_max = args_cli.x_max - ROBOTARM_BASE_LOCAL_X

    if not ERROR_ONLY_LOG:
        print(f"[INFO] Configuring randomizer:")
        print(f"  Robot Local Default X: {ROBOTARM_BASE_LOCAL_X:.4f}")
        print(f"  Applying Shifts: {offset_min:.4f} to {offset_max:.4f}")

    # Update randomizer parameters
    env_cfg.events.randomize_robot_base_linear = EventTermCfg(
        func=randomize_robot_root_pose,
        mode="startup",
        params={
            "x_range": (offset_min, offset_max),
            "y_range": (0.0, 0.0),
            "z_range": (0.0, 0.0),
        },
    )

    # Disable command resampling - keep target fixed
    if hasattr(env_cfg.commands, "ee_pose"):
        env_cfg.commands.ee_pose.resampling_time_range = (1e10, 1e10)

    env = ManagerBasedRLEnv(cfg=env_cfg)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not ERROR_ONLY_LOG:
        print(f"[OK] Environment created (device: {device})")

    # --- NDI SETUP (added on top of base script) ---
    detectors = []
    if args_cli.check_ndi:
        if not ERROR_ONLY_LOG:
            print("\n[NDI] Initializing Detectors for each environment...")
        stage = env.unwrapped.scene.stage
        for i in range(args_cli.num_envs):
            env_prim_path = f"/World/envs/env_{i}"
            if not ERROR_ONLY_LOG:
                print(f"  - Init Env {i}")
            try:
                detected_map = find_env_structure(stage, env_prim_path)
                ndi_config = EnvNDIConfig(env_prim_path, detected_map)
                detector = NDIDetector(ndi_config)
                detector.env_index = i
                detector.error_only_log = ERROR_ONLY_LOG
                detector._link_paths["link_6"] = ndi_config.LINK_6_PATH
                # link_base (link_0) °lÂÜ UM marker¡]ÀH©³®y¥­²¾¡^
                detector._link_paths["link_base"] = ndi_config.LINK_6_PATH.replace("link_6", "link_0")
                detector.initialize()
                detectors.append(detector)
            except Exception as e:
                print(f"[ERROR] Failed to init NDI for env {i}: {e}")
                detectors.append(None)
        if not ERROR_ONLY_LOG:
            print(f"[NDI] Initialized {len([d for d in detectors if d])} detectors.\n")
    # -----------------------------------------------

    # --- RESOLVE FIXED TARGET (identical to multi_position_eval.py) ---
    fixed_target_local = None
    if args_cli.use_usd_target:
        fixed_target_local = get_usd_target_position(env)
        if fixed_target_local is not None:
            if not ERROR_ONLY_LOG:
                print(f"[INFO] Using Fixed USD Target (Local): {fixed_target_local.cpu().numpy()}")
    elif args_cli.target_local_x is not None:
        x = args_cli.target_local_x if args_cli.target_local_x is not None else 0.5
        y = args_cli.target_local_y if args_cli.target_local_y is not None else 0.0
        z = args_cli.target_local_z if args_cli.target_local_z is not None else 0.0
        fixed_target_local = torch.tensor([x, y, z], device=device)
        if not ERROR_ONLY_LOG:
            print(f"[INFO] Using Fixed User Target (Local): {fixed_target_local.cpu().numpy()}")

    # ------------------------------------------

    # Create and load policy (identical to multi_position_eval.py)
    obs_dim = env.unwrapped.observation_manager.group_obs_dim["policy"][0]
    action_dim = env.unwrapped.action_manager.total_action_dim

    policy = ActorCritic(
        num_actor_obs=obs_dim,
        num_critic_obs=obs_dim,
        num_actions=action_dim,
        actor_hidden_dims=[768, 512, 512, 256],
        critic_hidden_dims=[768, 512, 512, 256],
        activation='elu',
        init_noise_std=1.0
    ).to(device)

    checkpoint_path = Path(args_cli.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        policy.load_state_dict(checkpoint["model_state_dict"])
        if not ERROR_ONLY_LOG:
            print(f"[OK] Loaded checkpoint (iteration: {checkpoint.get('iter', 'unknown')})")
    else:
        policy.load_state_dict(checkpoint)
        if not ERROR_ONLY_LOG:
            print(f"[OK] Loaded checkpoint")

    policy.eval()

    # Reset environment (randomizer will initialize positions)
    env.reset()
    if not ERROR_ONLY_LOG:
        print("[OK] Environment ready - robot base positions randomized\n")

    # Run evaluation
    results = run_evaluation(
        env,
        policy,
        args_cli.episodes,
        device,
        args_cli.x_min,
        args_cli.x_max,
        fixed_target_local,
        detectors,
        error_only_log=ERROR_ONLY_LOG,
    )

    # Reconstruct the X positions for each environment
    eval_x_positions = torch.linspace(args_cli.x_min, args_cli.x_max, args_cli.num_envs).tolist()

    if not ERROR_ONLY_LOG:
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"  Total Evaluations: {len(results['all_returns'])}")
        print(f"  Mean Return: {results['mean_return']:.2f} +/- {results['std_return']:.2f}")
        print(f"  Mean Length: {results['mean_length']:.1f}")
        print(f"  Success Rate: {results['success_rate']:.1f}%")
        print(f"{'='*60}\n")

        if results.get("ndi_results"):
            print("\n[NDI Visibility Stats (Avg markers visible)]")
            for i, avg in results["ndi_results"].items():
                x_val = eval_x_positions[i] if i < len(eval_x_positions) else "?"
                print(f"  Env {i} (X={x_val:.3f}): {avg:.2f}")

    per_position_stats = []
    raw_returns = np.array(results['all_returns'])
    returns_matrix = raw_returns.reshape((args_cli.episodes, args_cli.num_envs))
    success_matrix = (returns_matrix > 100).astype(float)
    mean_success_per_pos = success_matrix.mean(axis=0) * 100
    mean_return_per_pos = returns_matrix.mean(axis=0)
    eval_y_positions = [ROBOTARM_BASE_LOCAL_Y] * args_cli.num_envs

    target_pos_export = [0, 0, 0]
    if fixed_target_local is not None:
        target_pos_export = fixed_target_local.cpu().numpy().tolist()

    for i, x_pos in enumerate(eval_x_positions):
        per_position_stats.append({
            "x": x_pos,
            "y": eval_y_positions[i],
            "success_rate": float(mean_success_per_pos[i]),
            "mean_return": float(mean_return_per_pos[i])
        })

    save_data = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(checkpoint_path),
        "num_envs": args_cli.num_envs,
        "episodes": args_cli.episodes,
        "total_evaluations": len(results['all_returns']),
        "target_pose": target_pos_export,
        "x_range": {"min": args_cli.x_min, "max": args_cli.x_max},
        "statistics": {
            "mean_return": results['mean_return'],
            "std_return": results['std_return'],
            "mean_length": results['mean_length'],
            "success_rate": results['success_rate'],
        },
        "per_position_results": per_position_stats,
        "all_returns": results['all_returns'],
        "all_lengths": results['all_lengths'],
        "ndi_results": {str(k): v for k, v in results.get("ndi_results", {}).items()},
    }

    save_path = Path(args_cli.save_file)
    if save_path.parent.name:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"[OK] Detailed results saved to {save_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
