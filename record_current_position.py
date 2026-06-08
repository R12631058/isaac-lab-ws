"""
Interactive manual position evaluation script

You manually move the robotarm_base in Isaac Sim, then this script:
1. Records the current position
2. Runs evaluation episodes
3. Saves results
4. Asks if you want to test another position
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import json

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Record and evaluate manual positions")
parser.add_argument("--episodes", type=int, default=5, help="Episodes per position")
parser.add_argument("--num_envs", type=int, default=50, help="Number of parallel environments")
parser.add_argument("--checkpoint", type=str, default="logs/rsl_rl/tm5_reach_stable_v2/2026-01-05_14-03-14/model_2999.pt")
parser.add_argument("--save_file", type=str, default="manual_position_results.json")
parser.add_argument("--pos_x", type=float, default=None, help="Set robotarm_base X position (if not specified, uses current USD position)")
parser.add_argument("--pos_y", type=float, default=None, help="Set robotarm_base Y position")
parser.add_argument("--pos_z", type=float, default=None, help="Set robotarm_base Z position")
parser.add_argument("--random_x", action="store_true", help="Randomize X position for each environment (requires --x_min and --x_max)")
parser.add_argument("--x_min", type=float, default=-0.52, help="Minimum X position for randomization")
parser.add_argument("--x_max", type=float, default=-0.20, help="Maximum X position for randomization")
parser.add_argument("--target_world_x", type=float, default=None, help="Target X position in world coordinates")
parser.add_argument("--target_world_y", type=float, default=None, help="Target Y position in world coordinates")
parser.add_argument("--target_world_z", type=float, default=None, help="Target Z position in world coordinates")
parser.add_argument("--use_usd_target", action="store_true", help="Use /Root/Cube object from USD as target position")
args_cli = parser.parse_args()

# Launch Isaac Sim (NOT headless)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after Isaac Sim launch
from pxr import UsdGeom, Gf
from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_reach_surgery_room_cfg import (
    TM5ExtensionLinkOutReachEnvCfg_PLAY,
    EXTENSION_LINK_OUT_TARGET_X_MIN, EXTENSION_LINK_OUT_TARGET_X_MAX,
    EXTENSION_LINK_OUT_TARGET_Y_MIN, EXTENSION_LINK_OUT_TARGET_Y_MAX,
    EXTENSION_LINK_OUT_TARGET_Z_MIN, EXTENSION_LINK_OUT_TARGET_Z_MAX
)
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandTermCfg
from isaaclab_tasks.manager_based.manipulation.reach import mdp
from rsl_rl.modules import ActorCritic


def get_current_position(env: ManagerBasedRLEnv):
    """Get current robotarm_base position from USD"""
    stage = env.unwrapped.scene.stage
    prim_path = "/World/envs/env_0/Root/robotarm_base"
    base_prim = stage.GetPrimAtPath(prim_path)
    
    if not base_prim.IsValid():
        print(f"[ERROR] Cannot find {prim_path}")
        return None
    
    xformable = UsdGeom.Xformable(base_prim)
    translate_ops = [op for op in xformable.GetOrderedXformOps() 
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    
    if translate_ops:
        pos = translate_ops[0].Get()
        return {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}
    else:
        print("[ERROR] No translate operation found")
        return None


def set_robotarm_base_position(env: ManagerBasedRLEnv, x: float, y: float, z: float):
    """Set robotarm_base position in all environments - USD only, no physics override"""
    stage = env.unwrapped.scene.stage
    num_envs = env.unwrapped.scene.num_envs
    
    for env_idx in range(num_envs):
        prim_path = f"/World/envs/env_{env_idx}/Root/robotarm_base"
        base_prim = stage.GetPrimAtPath(prim_path)
        
        if not base_prim.IsValid():
            print(f"[WARN] Cannot find {prim_path}")
            continue
        
        xformable = UsdGeom.Xformable(base_prim)
        
        # Completely clear all xform operations to remove any pivot remnants
        if env_idx == 0:
            print(f"[DEBUG] Env 0 - Original XformOps:")
            for op in xformable.GetOrderedXformOps():
                print(f"  - {op.GetOpName()}: {op.GetOpType()} = {op.Get()}")
        
        # Clear everything
        xformable.ClearXformOpOrder()
        
        # Check for and clear pivot attribute directly
        prim = base_prim
        if prim.HasAttribute("xformOp:transform"):
            prim.RemoveProperty("xformOp:transform")
            print(f"[INFO] Env {env_idx}: Removed xformOp:transform attribute")
        
        # Add fresh translate operation
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(x, y, z))
        
        # Verify
        if env_idx == 0:
            print(f"[DEBUG] Env 0 - New XformOps:")
            for op in xformable.GetOrderedXformOps():
                print(f"  - {op.GetOpName()}: {op.GetOpType()} = {op.Get()}")
            
            # Compute actual world transform
            computed_transform = xformable.ComputeLocalToWorldTransform(0)
            translation = computed_transform.ExtractTranslation()
            print(f"[DEBUG] Env 0 - Computed world position: ({translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f})")
    
    # DON'T update physics state - let it read from USD naturally
    # The pivot issue occurs when we force physics state
    # robot.write_root_state_to_sim() is SKIPPED
    
    print(f"[OK] Updated USD robotarm_base position to ({x:.4f}, {y:.4f}, {z:.4f})")
    print(f"[INFO] Physics will read position from USD on next reset")


def set_robotarm_base_positions_random_x(env: ManagerBasedRLEnv, x_min: float, x_max: float, y: float, z: float):
    """Set robotarm_base positions with random X for each environment, fixed Y and Z"""
    stage = env.unwrapped.scene.stage
    num_envs = env.unwrapped.scene.num_envs
    device = env.unwrapped.device
    
    # Generate random X positions for each environment (local coordinates)
    x_positions_local = torch.FloatTensor(num_envs).uniform_(x_min, x_max).to(device)
    
    print(f"[INFO] Setting random X positions for {num_envs} environments...")
    
    # Update USD prims (local coordinates relative to each environment)
    for env_idx in range(num_envs):
        x_local = x_positions_local[env_idx].item()
        prim_path = f"/World/envs/env_{env_idx}/Root/robotarm_base"
        base_prim = stage.GetPrimAtPath(prim_path)
        
        if not base_prim.IsValid():
            print(f"[WARN] Cannot find {prim_path}")
            continue
        
        xformable = UsdGeom.Xformable(base_prim)
        
        # Clear all xform operations to remove any pivot remnants
        xformable.ClearXformOpOrder()
        
        # Clear pivot attribute if exists
        if base_prim.HasAttribute("xformOp:transform"):
            base_prim.RemoveProperty("xformOp:transform")
        
        # Add fresh translate operation with random X
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(x_local, y, z))
    
    # DON'T update physics state - let it read from USD naturally after reset
    
    print(f"[OK] Set robotarm_base positions with random X:")
    print(f"     X range (local): [{x_min:.4f}, {x_max:.4f}]")
    print(f"     Y (local, fixed): {y:.4f}")
    print(f"     Z (local, fixed): {z:.4f}")
    print(f"     Generated X positions: {x_positions_local.cpu().numpy()}")
    
    # Force USD changes to be applied
    import omni.usd
    context = omni.usd.get_context()
    stage = context.get_stage()
    # Trigger stage update
    print(f"[INFO] Forcing USD stage update...")
    
    print(f"[INFO] Physics will read positions from USD on next reset")
    
    return x_positions_local.cpu().numpy()


def get_phantom_position(env: ManagerBasedRLEnv):
    """讀取 phantom 在世界座標系中的實際位置"""
    stage = env.unwrapped.scene.stage
    
    # Phantom 的路徑（根據你的 USD 結構）
    phantom_path = "/World/envs/env_0/Root/phantom"
    phantom_prim = stage.GetPrimAtPath(phantom_path)
    
    if not phantom_prim.IsValid():
        print(f"[WARN] Cannot find phantom at {phantom_path}, using default position")
        # 使用默認的 phantom 位置
        return {"x": 0.0, "y": 0.0, "z": 0.0}
    
    xformable = UsdGeom.Xformable(phantom_prim)
    
    # 獲取世界座標變換
    world_transform = xformable.ComputeLocalToWorldTransform(0)
    translation = world_transform.ExtractTranslation()
    
    return {"x": float(translation[0]), "y": float(translation[1]), "z": float(translation[2])}


def get_target_position(env: ManagerBasedRLEnv):
    """讀取 USD 中 /Root/target 的世界座標位置"""
    stage = env.unwrapped.scene.stage
    
    target_path = "/World/envs/env_0/Root/Cube"
    target_prim = stage.GetPrimAtPath(target_path)
    
    if not target_prim.IsValid():
        print(f"[WARN] Cannot find target at {target_path}")
        return None
    
    xformable = UsdGeom.Xformable(target_prim)
    world_transform = xformable.ComputeLocalToWorldTransform(0)
    translation = world_transform.ExtractTranslation()
    
    return {"x": float(translation[0]), "y": float(translation[1]), "z": float(translation[2])}


def calculate_target_range_from_phantom(env: ManagerBasedRLEnv, robotarm_pos: dict, phantom_pos: dict):
    """
    將 target 固定在 phantom 位置，不做任何隨機化
    Target 就是 phantom，機械臂直接 reach phantom
    """
    # Phantom 相對於 robotarm_base 的位置
    relative_x = phantom_pos['x'] - robotarm_pos['x']
    relative_y = phantom_pos['y'] - robotarm_pos['y']
    relative_z = phantom_pos['z'] - robotarm_pos['z']
    
    print(f"\nPhantom position (world):")
    print(f"  X = {phantom_pos['x']:.4f}")
    print(f"  Y = {phantom_pos['y']:.4f}")
    print(f"  Z = {phantom_pos['z']:.4f}")
    
    print(f"\nPhantom relative to robotarm_base:")
    print(f"  ΔX = {relative_x:+.4f} m")
    print(f"  ΔY = {relative_y:+.4f} m")
    print(f"  ΔZ = {relative_z:+.4f} m")
    
    # 將 target 完全固定在 phantom 位置（設置 min = max 移除隨機性）
    target_x = relative_x
    target_y = relative_y
    target_z = relative_z
    
    print(f"\nFixed target position (on phantom):")
    print(f"  X: {target_x:.4f}")
    print(f"  Y: {target_y:.4f}")
    print(f"  Z: {target_z:.4f}")
    print(f"  [No randomization - target is exactly on phantom]")
    
    # 更新環境的 command 配置（min = max，完全固定）
    env.unwrapped.command_manager._terms["ee_pose"].cfg.ranges.pos_x = (target_x, target_x)
    env.unwrapped.command_manager._terms["ee_pose"].cfg.ranges.pos_y = (target_y, target_y)
    env.unwrapped.command_manager._terms["ee_pose"].cfg.ranges.pos_z = (target_z, target_z)
    
    print("[OK] Target fixed on phantom (no randomization)")
    
    return {
        "phantom_world": phantom_pos,
        "phantom_relative": {"x": relative_x, "y": relative_y, "z": relative_z},
        "target_position": {
            "x": target_x,
            "y": target_y,
            "z": target_z
        }
    }


def run_evaluation(env, policy, num_episodes, device, target_x, target_y, target_z, current_pos_world):
    """Run evaluation episodes and collect statistics"""
    returns = []
    episode_lengths = []
    
    print(f"[INFO] Target will be fixed at world position: ({target_x:.4f}, {target_y:.4f}, {target_z:.4f})")
    
    num_envs = env.unwrapped.num_envs
    
    for episode_idx in range(num_episodes):
        obs, _ = env.reset()
        
        # Set target command after reset
        ee_command = env.unwrapped.command_manager._terms["ee_pose"]
        
        # CRITICAL: The observation reads pose_command_b (robot frame), not pose_command_w!
        # We need to convert world coordinates to robot base frame
        robot = env.unwrapped.scene["robot"]
        robot_pos = robot.data.root_pos_w[:, :3]  # All robots' base positions in world
        robot_quat = robot.data.root_quat_w[:]     # All robots' base orientations in world
        
        # Target in world frame (same for all envs)
        target_world_pos = torch.tensor([target_x, target_y, target_z], device=device).unsqueeze(0).repeat(num_envs, 1)
        target_world_quat = ee_command.pose_command_w[:, 3:]  # Keep same orientation
        
        # Convert to robot base frame using inverse transform
        from isaaclab.utils.math import subtract_frame_transforms
        target_base_pos, target_base_quat = subtract_frame_transforms(
            robot_pos, robot_quat, target_world_pos, target_world_quat
        )
        
        # Set the command in robot base frame (this is what policy sees!)
        ee_command.pose_command_b[:, :3] = target_base_pos
        ee_command.pose_command_b[:, 3:] = target_base_quat
        
        # Also update world frame for visualization
        ee_command.pose_command_w[:, :3] = target_world_pos
        ee_command.pose_command_w[:, 3:] = target_world_quat
        
        # Recompute observation to ensure policy sees updated target
        obs = env.unwrapped.observation_manager.compute()
        
        episode_returns = torch.zeros(num_envs, device=device)
        episode_steps = torch.zeros(num_envs, dtype=torch.int, device=device)
        dones_all = torch.zeros(num_envs, dtype=torch.bool, device=device)
        
        while not dones_all.all():
            # Re-set target before each step (for all active envs)
            robot_pos = robot.data.root_pos_w[:, :3]
            robot_quat = robot.data.root_quat_w[:]
            
            target_world_pos = torch.tensor([target_x, target_y, target_z], device=device).unsqueeze(0).repeat(num_envs, 1)
            target_world_quat = ee_command.pose_command_w[:, 3:]
            
            # Convert to robot base frame
            target_base_pos, target_base_quat = subtract_frame_transforms(
                robot_pos, robot_quat, target_world_pos, target_world_quat
            )
            
            # Set both base and world frame commands
            ee_command.pose_command_b[:, :3] = target_base_pos
            ee_command.pose_command_b[:, 3:] = target_base_quat
            ee_command.pose_command_w[:, :3] = target_world_pos
            ee_command.pose_command_w[:, 3:] = target_world_quat
            
            with torch.no_grad():
                obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
                actions = policy.act(obs_tensor, deterministic=True)
            
            obs, rewards, dones, truncated, info = env.step(actions)
            
            # Re-set immediately after step
            robot_pos = robot.data.root_pos_w[:, :3]
            robot_quat = robot.data.root_quat_w[:]
            
            target_world_pos = torch.tensor([target_x, target_y, target_z], device=device).unsqueeze(0).repeat(num_envs, 1)
            target_world_quat = ee_command.pose_command_w[:, 3:]
            
            target_base_pos, target_base_quat = subtract_frame_transforms(
                robot_pos, robot_quat, target_world_pos, target_world_quat
            )
            
            ee_command.pose_command_b[:, :3] = target_base_pos
            ee_command.pose_command_b[:, 3:] = target_base_quat
            ee_command.pose_command_w[:, :3] = target_world_pos
            ee_command.pose_command_w[:, 3:] = target_world_quat
            
            # Recompute observation
            obs = env.unwrapped.observation_manager.compute()
            
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
        print(f"  Episode {episode_idx+1}/{num_episodes}: Mean Return = {mean_return:.2f}, Mean Length = {mean_length:.0f} ({num_envs} envs)")
    
    # Calculate statistics
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    mean_length = np.mean(episode_lengths)
    success_rate = sum(1 for r in returns if r > 100) / len(returns) * 100
    total_episodes = len(returns)
    
    print(f"\n[SUMMARY] Total episodes: {total_episodes} ({num_episodes} batches × {num_envs} envs)")
    print(f"  Mean return: {mean_return:.2f} ± {std_return:.2f}")
    print(f"  Mean length: {mean_length:.1f}")
    print(f"  Success rate: {success_rate:.1f}%")
    
    return {
        "returns": returns,
        "mean_return": mean_return,
        "std_return": std_return,
        "mean_length": mean_length,
        "success_rate": success_rate,
        "num_episodes": num_episodes,
        "num_envs": num_envs,
        "total_episodes": total_episodes
    }


def main():
    print("\n" + "="*60)
    print("Multi-Environment Evaluation - V1 (Parallel)")
    print("="*60)
    print(f"Configuration:")
    print(f"  Parallel Environments: {args_cli.num_envs}")
    print(f"  Episodes per batch: 1")
    print(f"  Total batches: {args_cli.episodes}")
    print(f"  Total episodes: {args_cli.num_envs * args_cli.episodes}")
    print(f"  Checkpoint: {args_cli.checkpoint}")
    print("="*60 + "\n")
    
    # Create environment
    env_cfg = TM5ExtensionLinkOutReachEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs  # Use specified number of environments
    env_cfg.scene.env_spacing = 5.0  # Space environments apart for visibility
    
    # Disable command resampling - prevent target regeneration
    env_cfg.commands.ee_pose.resampling_time_range = (1e10, 1e10)  # Never resample
    
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[OK] Environment created (device: {device})")
    
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
    
    # Reset environment
    env.reset()
    print("[OK] Environment ready\n")
    
    # Set position(s)
    if args_cli.random_x:
        # Randomize X position for each environment
        print("Setting random X positions for each environment...")
        y = args_cli.pos_y if args_cli.pos_y is not None else 0.1
        z = args_cli.pos_z if args_cli.pos_z is not None else 0
        x_positions = set_robotarm_base_positions_random_x(env, args_cli.x_min, args_cli.x_max, y, z)
        print(f"[INFO] Resetting environment to apply new positions...")
        env.reset()  # Reset to load positions from USD
        
        # Verify positions after reset
        print(f"\n[VERIFY] Checking actual robot positions after reset:")
        robot = env.unwrapped.scene["robot"]
        actual_positions = robot.data.root_pos_w.cpu().numpy()
        for env_idx in range(min(5, env.unwrapped.num_envs)):  # Show first 5
            pos = actual_positions[env_idx]
            print(f"  Env {env_idx}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
        
        print(f"[OK] Random X positions set and applied\n")
        
    elif args_cli.pos_x is not None:
        # Use specified position (same for all environments)
        x = args_cli.pos_x
        y = args_cli.pos_y if args_cli.pos_y is not None else 0.1
        z = args_cli.pos_z if args_cli.pos_z is not None else 0
        set_robotarm_base_position(env, x, y, z)
        print(f"[INFO] Resetting environment to apply new position...")
        env.reset()  # Reset to load position from USD
        print(f"[OK] Position set and applied\n")
    else:
        print("[INFO] Using current USD position (no --pos_x specified)\n")
    
    # Load existing results if file exists
    save_path = Path(args_cli.save_file)
    all_results = None
    needs_backup = False
    
    if save_path.exists():
        try:
            with open(save_path, 'r') as f:
                loaded_data = json.load(f)
                # Check if it has the required structure
                if "positions" in loaded_data:
                    all_results = loaded_data
                    print(f"[INFO] Loaded existing results with {len(all_results['positions'])} positions")
                else:
                    print(f"[WARN] Existing file {save_path} has incompatible format (missing 'positions'). Starting fresh.")
                    # Mark for backup
                    needs_backup = True
        except json.JSONDecodeError:
            print(f"[WARN] Existing file {save_path} is corrupted. Starting fresh.")
            needs_backup = True
            
        if needs_backup:
            backup_path = save_path.with_suffix(f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            try:
                save_path.rename(backup_path)
                print(f"[INFO] Backed up old file to {backup_path}")
            except OSError as e:
                print(f"[WARN] Could not backup file (PermissionError or other): {e}")
                print(f"[WARN] Will overwrite the existing file.")

    if all_results is None:
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "checkpoint": str(checkpoint_path),
            "episodes_per_position": args_cli.episodes,
            "positions": []
        }
    
    position_count = len(all_results["positions"])
    
    # Test current position
    position_count += 1
    print(f"\n{'='*60}")
    print(f"Position #{position_count}")
    print(f"{'='*60}")
    
    # Get current position
    current_pos = get_current_position(env)
    if current_pos is None:
        print("[ERROR] Failed to get position")
        env.close()
        simulation_app.close()
        return
    
    print(f"\nCurrent robotarm_base position:")
    print(f"  X = {current_pos['x']:.4f}")
    print(f"  Y = {current_pos['y']:.4f}")
    print(f"  Z = {current_pos['z']:.4f}")
    
    # Read actual phantom position and calculate target range
    phantom_pos = get_phantom_position(env)
    
    # Determine target position (priority: USD target > command line > phantom)
    target_world_pos = None
    
    if args_cli.use_usd_target:
        # Use /Root/target from USD
        target_world_pos = get_target_position(env)
        if target_world_pos:
            print(f"\n[INFO] Using /Root/target from USD:")
            print(f"  World: ({target_world_pos['x']:.4f}, {target_world_pos['y']:.4f}, {target_world_pos['z']:.4f})")
        else:
            print(f"\n[WARN] /Root/target not found, falling back to phantom")
            target_world_pos = phantom_pos
    elif args_cli.target_world_x is not None:
        # Use command line specified coordinates
        target_world_pos = {
            'x': args_cli.target_world_x,
            'y': args_cli.target_world_y if args_cli.target_world_y is not None else 0.0,
            'z': args_cli.target_world_z if args_cli.target_world_z is not None else 0.0
        }
        print(f"\n[INFO] User specified target in world coordinates:")
        print(f"  World: ({target_world_pos['x']:.4f}, {target_world_pos['y']:.4f}, {target_world_pos['z']:.4f})")
    else:
        # Default: use phantom
        target_world_pos = phantom_pos
        print(f"\n[INFO] Using phantom as target:")
        print(f"  World: ({phantom_pos['x']:.4f}, {phantom_pos['y']:.4f}, {phantom_pos['z']:.4f})")
    
    # Calculate target relative to current base position
    # CORRECTION: pose_command_w is already in world frame, don't subtract base position!
    target_rel_x = target_world_pos['x']
    target_rel_y = target_world_pos['y']
    target_rel_z = target_world_pos['z']
    
    print(f"  Target (world coordinates): ({target_rel_x:.4f}, {target_rel_y:.4f}, {target_rel_z:.4f})")
    
    # Set fixed target range BEFORE running episodes
    env.unwrapped.command_manager._terms["ee_pose"].cfg.ranges.pos_x = (target_rel_x, target_rel_x)
    env.unwrapped.command_manager._terms["ee_pose"].cfg.ranges.pos_y = (target_rel_y, target_rel_y)
    env.unwrapped.command_manager._terms["ee_pose"].cfg.ranges.pos_z = (target_rel_z, target_rel_z)
    print("[OK] Target range configured (fixed, no randomization)\n")
    
    compensation_info = {
        "target_world": target_world_pos if target_world_pos else phantom_pos,
        "target_relative": {"x": target_rel_x, "y": target_rel_y, "z": target_rel_z}
    }
    
    # Run evaluation
    print(f"Running {args_cli.episodes} episodes...")
    results = run_evaluation(env, policy, args_cli.episodes, device, target_rel_x, target_rel_y, target_rel_z, 
                            [current_pos['x'], current_pos['y'], current_pos['z']])
    
    # Display results
    print(f"\nResults:")
    print(f"  Average Return: {results['mean_return']:.2f} ± {results['std_return']:.2f}")
    print(f"  Average Length: {results['mean_length']:.1f}")
    print(f"  Success Rate: {results['success_rate']:.1f}%")
    
    # Store results
    position_result = {
        "position_number": position_count,
        "position": current_pos,
        "compensation": compensation_info,
        "statistics": {
            "mean_return": results['mean_return'],
            "std_return": results['std_return'],
            "mean_length": results['mean_length'],
            "success_rate": results['success_rate'],
            "num_envs": results['num_envs'],
            "total_episodes": results['total_episodes']
        },
        "returns": results['returns']
    }
    all_results["positions"].append(position_result)
    
    # Save results
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Results saved to {save_path}")
    
    # Print all positions tested so far
    print(f"\n{'='*60}")
    print("All Positions Tested So Far")
    print(f"{'='*60}")
    print(f"Position | X Position | Mean Return | Success Rate")
    print("-" * 60)
    for result in all_results["positions"]:
        pos_num = result["position_number"]
        x_pos = result["position"]["x"]
        mean_ret = result["statistics"]["mean_return"]
        success = result["statistics"]["success_rate"]
        print(f"   {pos_num:2d}    | {x_pos:9.4f} | {mean_ret:11.2f} | {success:11.1f}%")
    print("="*60 + "\n")
    
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
