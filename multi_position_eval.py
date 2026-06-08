"""Multi-Position Generalization Evaluation Script

Automatically evaluates RL policy across multiple robot base positions
using environment randomizer (like RSL-RL training).
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import json

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Multi-position generalization evaluation")
parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
parser.add_argument("--num_envs", type=int, default=50, help="Number of parallel environments")
parser.add_argument("--checkpoint", type=str, 
                    default="logs\rsl_rl\tm5_reach_stable_v2\2026-02-04_00-28-35\model_9999.pt")
parser.add_argument("--save_file", type=str, default="scripts/isaaclab_ws/reachability_map/multi_position_results.json")
parser.add_argument("--x_min", type=float, default=0.3, help="Minimum X position (World Frame relative to Env Origin)")
parser.add_argument("--x_max", type=float, default=1.0, help="Maximum X position")
# parser.add_argument("--y_min", type=float, default=0.1, help="Minimum Y position (World Frame relative to Env Origin)")
parser.add_argument("--use_usd_target", action="store_true", help="Use /Root/Cube object from Env 0 as the reference target for ALL environments")
parser.add_argument("--target_local_x", type=float, default=None, help="Target X position (Environment Local Frame)")
parser.add_argument("--target_local_y", type=float, default=None, help="Target Y position (Environment Local Frame)")
parser.add_argument("--target_local_z", type=float, default=None, help="Target Z position (Environment Local Frame)")
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import UsdGeom, Gf
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_extension_linkout_cone_orientation_cfg import (
    TM5ExtensionLinkOutFanOrientationEnvCfg
)
from rsl_rl.modules import ActorCritic

# robotarm_base local position (from test_multi_robot_parallel.py)
ROBOTARM_BASE_LOCAL_X = 0.3
ROBOTARM_BASE_LOCAL_Y = -0.1741
ROBOTARM_BASE_LOCAL_Z = 0.9493

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
    # x_offsets represents the 'shift' from the default position
    x_offsets = torch.linspace(x_range[0], x_range[1], env.num_envs, device=env.device)[env_ids]
    
    # Constant or random for Y/Z (here we keep constant 0 as per requirement)
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
    """Get /Root/Cube position from Env 0 (World Frame)""" # Assumes Env 0
    stage = env.unwrapped.scene.stage
    # We want a fixed world target. We can look at Env 0's target and use its World Position.
    # Note: In Isaac Lab, the stage might be composed of multiple envs.
    # /World/envs/env_0/Root/Cube
    prim_path = "/World/envs/env_0/Root/Cube" # Assuming this is the target prim
    
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        # Try generic path or search?
        # Maybe the user meant /World/Objects/Cube?
        # Let's stick to the one in record_current_position.py logic if possible?
        # Actually, let's look for a prim named "target" or "Cube" in env 0.
        prim_path = "/World/envs/env_0/Root/target"
        prim = stage.GetPrimAtPath(prim_path)
    
    if not prim.IsValid():
        print(f"[WARN] Could not find target in USD at {prim_path}")
        return None
        
    xformable = UsdGeom.Xformable(prim)
    # Compute World Transform
    world_transform = xformable.ComputeLocalToWorldTransform(0)
    translation = world_transform.ExtractTranslation()
    
    # Return as Local to Env 0 Origin if possible, but calculating later is safer
    # Actually, we need to return this relative to Env 0 Origin because this function only looks at Env 0.
    # The caller will use this relative vector for ALL envs.
    target_world_env0 = torch.tensor([translation[0], translation[1], translation[2]], device=env.device)
    
    # Get Env 0 Origin
    if env.scene.env_origins is not None:
        env0_origin = env.scene.env_origins[0]
        return target_world_env0 - env0_origin
    else:
        # If no env origins (e.g. valid only for 1 env or global scene), use world
        return target_world_env0


def run_evaluation(env, policy, num_episodes, device, x_min, x_max, fixed_target_local=None):
    """Run evaluation episodes across all environments
    
    Args:
        fixed_target_local: Target position relative to environment origin (Env Local Frame)
    """
    returns = []
    episode_lengths = []
    
    num_envs = env.unwrapped.num_envs
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
                # target_pos_w = Env Origins + Fixed Local Target
                env_origins = env.scene.env_origins # Shape (num_envs, 3)
                target_local_expanded = fixed_target_local.unsqueeze(0).repeat(num_envs, 1)
                target_pos_w = env_origins + target_local_expanded
                
                # Correct Target Orientation from Config (Needle Pointing Down)
                # Roll -180, Pitch 0, Yaw 180 => Quat (w,x,y,z) = (0, 0, 1, 0)
                # This matches FAN_BASE_QUAT in the config.
                target_quat_w = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device).unsqueeze(0).repeat(num_envs, 1)

                # 3. Compute Target in Robot Base Frame
                # command_b = target_w - robot_b
                cmd_pos_b, cmd_quat_b = subtract_frame_transforms(
                    robot_root_pos_w, robot_root_quat_w,
                    target_pos_w, target_quat_w
                )
                
                # 4. Update Command Manager
                # Determine keys. Usually "ee_pose".
                term_name = "ee_pose"
                try:
                    cmd_term = env.command_manager.get_term(term_name)
                    # Check dimension
                    if cmd_term.command.shape[1] == 3:
                        cmd_term.command[:] = cmd_pos_b
                    elif cmd_term.command.shape[1] == 7:
                        # Combine pos and quat
                        cmd_concat = torch.cat([cmd_pos_b, cmd_quat_b], dim=1)
                        cmd_term.command[:] = cmd_concat
                    
                    # 5. Re-compute Observations
                    # Because command just changed, the observation (which includes target_pos) is wrong.
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
            
            # --- RE-APPLY COMMAND IF NEEDED (Safety) ---
            # If the command manager resamples mid-episode?
            # We set resampling_time_range = (1e10, 1e10) in main, so it should be fine.
            # But if reset happened for individual envs (due to done)?
            # When an env is done, it auto-resets:
            #   - Resamples robot root (if randomizer active on reset)
            #   - Resamples command (if command term active on reset)
            # We need to ensure the command is CORRECTED for the NEW robot position.
            
            if torch.any(dones) and fixed_target_local is not None:
                with torch.no_grad():
                    # Recalculate for reset environments
                    env_ids = torch.where(dones)[0]
                    
                    robot = env.scene["robot"]
                    robot_root_pos_w = robot.data.root_pos_w[env_ids]
                    robot_root_quat_w = robot.data.root_quat_w[env_ids]
                    
                    # Correct Target for Reset Envs
                    env_origins_reset = env.scene.env_origins[env_ids]
                    target_local_expanded = fixed_target_local.unsqueeze(0).repeat(len(env_ids), 1)
                    target_pos_w = env_origins_reset + target_local_expanded
                    
                    # Same orientation as main override (Needle Pointing Down)
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
                        cmd_concat = torch.cat([cmd_pos_b, cmd_quat_b], dim=1)
                        cmd_term.command[env_ids] = cmd_concat
                    
                    # Note: env.step() already returned 'obs' for the NEXT step.
                    # The auto-reset inside env.step() computed observations based on the OLD (random) command.
                    # We have updated the command buffer now, but the 'obs' variable holds stale data for the reset envs.
                    # This is a classic RL loop issue with custom reset logic.
                    # Ideally, we should update 'obs' for the reset envs.
                    # Manually recomputing obs for specific envs is usually supported via env.observation_manager.compute_group(env_ids=env_ids)
                    # But ManagerBasedRLEnv.step() returns the result of compute().
                    
                    # For evaluation metrics, one bad frame on reset isn't catastrophic, 
                    # but if the policy sees a wrong target, it might jerk the robot.
                    # Let's try to update obs['policy'][env_ids]?
                    # Recomputing all is safest/easiest if not too slow.
                    # obs = env.observation_manager.compute()
                    # Do not overwrite rewards/dones, just update obs.
                    new_obs = env.observation_manager.compute()
                    if isinstance(obs, dict):
                        for k in obs.keys():
                            if k == "policy": # Update policy obs
                                obs[k][env_ids] = new_obs[k][env_ids]
                    else:
                        obs[env_ids] = new_obs[env_ids]
            
            # -------------------------------------------
            
            # Update statistics for active environments
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
        print(f"  Batch {episode_idx+1}/{num_episodes}: "
              f"Mean Return = {mean_return:.2f}, "
              f"Mean Length = {mean_length:.0f} "
              f"({num_envs} envs)")
    
    # Calculate statistics
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    mean_length = np.mean(episode_lengths)
    success_rate = sum(1 for r in returns if r > 100) / len(returns) * 100
    
    return {
        "mean_return": mean_return,
        "std_return": std_return,
        "mean_length": mean_length,
        "success_rate": success_rate,
        "all_returns": returns,
        "all_lengths": episode_lengths,
    }


def main():
    print("="*60)
    print("Multi-Position Generalization Evaluation")
    print("="*60)
    print(f"Configuration:")
    print(f"  Parallel Environments: {args_cli.num_envs}")
    print(f"  Episodes: {args_cli.episodes}")
    print(f"  Total Evaluations: {args_cli.num_envs * args_cli.episodes}")
    print(f"  Target Robot Base X Range: [{args_cli.x_min:.4f}, {args_cli.x_max:.4f}]")
    print(f"  Checkpoint: {args_cli.checkpoint}")
    print("="*60 + "\n")
    
    # Create environment config
    env_cfg = TM5ExtensionLinkOutFanOrientationEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 5.0
    
    # Important: Disable replicate physics to allow individual robot positioning
    # This allows unique poses per environment
    env_cfg.scene.replicate_physics = False
    
    # Calculate offset range
    # The input args are the desired "World X" relative to Env Origin.
    # The shift needed is Target_X - Default_Local_X
    offset_min = args_cli.x_min - ROBOTARM_BASE_LOCAL_X
    offset_max = args_cli.x_max - ROBOTARM_BASE_LOCAL_X
    
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
    print(f"[OK] Environment created (device: {device})")

    # --- RESOLVE FIXED TARGET (WORLD FRAME) ---
    fixed_target_local = None
    if args_cli.use_usd_target:
        fixed_target_local = get_usd_target_position(env)
        if fixed_target_local is not None:
            print(f"[INFO] Using Fixed USD Target (Local): {fixed_target_local.cpu().numpy()}")
    elif args_cli.target_local_x is not None: 
        # Note: Argument says 'local' but let's treat it as 'relative to env origin' aka World if Origin is 0?
        # Re-reading help: "Target X position (Environment Local Frame)"
        # If the user provides a coordinate, do they mean "Fixed World Coordinate" or "Fixed Local Coordinate"?
        # The prompt says "generate a fixed target pose relative to the environment origin"
        # Since env origin is (0,0,0) for Env 0 in World usually (or shifted).
        # Let's interpret arguments as "World Coordinate" if they are meant to be the fixed target.
        # But wait, the arguments are named `target_local_x`.
        # If I want a target at World (0.5, 0, 0), I should probably just treat these args as World Coords.
        x = args_cli.target_local_x if args_cli.target_local_x is not None else 0.5
        y = args_cli.target_local_y if args_cli.target_local_y is not None else 0.0
        z = args_cli.target_local_z if args_cli.target_local_z is not None else 0.0
        fixed_target_local = torch.tensor([x, y, z], device=device)
        print(f"[INFO] Using Fixed User Target (Local): {fixed_target_local.cpu().numpy()}")
    
    # ------------------------------------------
    
    # Create and load policy
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
        print(f"[OK] Loaded checkpoint (iteration: {checkpoint.get('iter', 'unknown')})")
    else:
        policy.load_state_dict(checkpoint)
        print(f"[OK] Loaded checkpoint")
    
    policy.eval()
    
    # Reset environment (randomizer will initialize positions)
    env.reset()
    print("[OK] Environment ready - robot base positions randomized\n")
    
    # Run evaluation
    results = run_evaluation(env, policy, args_cli.episodes, device, args_cli.x_min, args_cli.x_max, fixed_target_local)
    
    # Reconstruct the X positions for each environment
    # The randomizer uses torch.linspace(min, max, num_envs)
    eval_x_positions = torch.linspace(args_cli.x_min, args_cli.x_max, args_cli.num_envs).tolist()
    
    # Display results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Total Evaluations: {len(results['all_returns'])}")
    print(f"  Mean Return: {results['mean_return']:.2f} ± {results['std_return']:.2f}")
    print(f"  Mean Length: {results['mean_length']:.1f}")
    print(f"  Success Rate: {results['success_rate']:.1f}%")
    print(f"{'='*60}\n")
    
    # We now have multiple episodes per environment/position.
    # Let's aggregate by position (Environment ID).
    # Since each env_idx corresponds to a specific X position.
    
    # Reshape returns: [Total_Episodes] -> [Episodes, Num_Envs] theoretically, 
    # but run_evaluation flattens them as: env0_ep0, env1_ep0... env0_ep1, env1_ep1...
    # Actually run_evaluation code:
    # "returns.append(episode_returns[env_idx].item())" inside loop over num_envs
    # So the list order is: Ep0_Env0, Ep0_Env1... Ep0_EnvN, Ep1_Env0...
    
    per_position_stats = []
    
    raw_returns = np.array(results['all_returns'])
    # Reshape to (Episodes, Num_Envs)
    # Be careful with the loop order in run_evaluation
    # run_evaluation logic:
    # for episode_idx in range(num_episodes):
    #   for env_idx in range(num_envs):
    #     returns.append(...)
    # So it is Row-Major: Episode is Row, Env is Col.
    
    returns_matrix = raw_returns.reshape((args_cli.episodes, args_cli.num_envs))
    success_matrix = (returns_matrix > 100).astype(float) # Assuming >100 is success
    
    # Calculate success rate PER POSITION (column mean)
    mean_success_per_pos = success_matrix.mean(axis=0) * 100
    mean_return_per_pos = returns_matrix.mean(axis=0)
    
    # Calculate Y positions (assume fixed 0.0 offset from default local Y)
    # If we randomized Y, we would need to capture it.
    # For now, let's assume Y is constant 0.0 (shift) + ROBOTARM_BASE_LOCAL_Y
    # World Y = EnvOriginY + DefaultLocalY + ShiftY
    # We ignore EnvOriginY for the "Map" usually (we want position relative to room origin).
    # Let's verify: In randomized_robot_root_pose, y_offsets = 0.0 + y_range[0] (which is 0.0)
    # So Y is just ROBOTARM_BASE_LOCAL_Y
    eval_y_positions = [ROBOTARM_BASE_LOCAL_Y] * args_cli.num_envs
    
    # Also record Target World Position (from Env 0 perspective)
    # fixed_target_local is relative to Env Origin.
    # So Target World (in result map) should be fixed_target_local.
    target_pos_export = [0, 0, 0]
    if fixed_target_local is not None:
        target_pos_export = fixed_target_local.cpu().numpy().tolist()

    for i, x_pos in enumerate(eval_x_positions):
        # x_pos is the "World X relative to Env Origin" because x_min/max args are defined that way?
        # args_cli.x_min help says: "World Frame relative to Env Origin"
        # However, check helper: 
        # x_offsets = linspace(x_min, x_max)
        # new_root_pos = env_origins + default_local + shifts
        # Wait, if x_min is passed as "robot local shift" or "robot world position"?
        # args says: "World Frame relative to Env Origin"
        # The code calculates: offset_min = args_cli.x_min - ROBOTARM_BASE_LOCAL_X
        # randomizer uses: x_offsets = linspace(offset_min, offset_max)
        # new_pos = origin + default + offset
        #         = origin + default + (target - default)
        #         = origin + target
        # So yes, args_cli.x_min IS the final X position relative to Env Origin.
        # So x_pos in this loop IS accurate.
        
        per_position_stats.append({
            "x": x_pos,
            "y": eval_y_positions[i],
            "success_rate": float(mean_success_per_pos[i]),
            "mean_return": float(mean_return_per_pos[i])
        })

    # Save results
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
    }
    
    save_path = Path(args_cli.save_file)
    # Ensure directory exists
    if save_path.parent.name:
        save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"[OK] Detailed results saved to {save_path}")
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
