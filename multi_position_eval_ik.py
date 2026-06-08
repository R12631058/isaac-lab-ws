"""Multi-Position Reachability Evaluation using Differential IK (Pseudo-Inverse)

This script evaluates the reachability of the workspace using Isaac Lab's Differential IK Controller.
This serves as a deterministic baseline (similar to Lula/cuRobo local IK) to compare against RL Policy.
It tests if a mathematical Jacobian-based solver can reach the targets across multiple robot base positions.

Note: This uses 'dls' (Damped Least Squares) IK, which is a local solver. It does not perform global motion planning like cuRobo/RMPFlow,
but it validates if a solution EXISTS locally and is reachable without collisions blocking the direct path.
"""

import argparse
from pathlib import Path
from datetime import datetime
import json

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Multi-position IK evaluation")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--episodes", type=int, default=1, help="Number of evaluation episodes per position (1 is usually enough for deterministic IK)")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments")
parser.add_argument("--save_file", type=str, default="scripts/isaaclab_ws/reachability_map/multi_position_ik_results.json")
parser.add_argument("--x_min", type=float, default=0.3, help="Minimum X position (World Frame relative to Env Origin)")
parser.add_argument("--x_max", type=float, default=1.0, help="Maximum X position")
parser.add_argument("--use_usd_target", action="store_true", help="Use /Root/Cube object from Env 0 as the reference target for ALL environments")
parser.add_argument("--target_local_x", type=float, default=None, help="Target X position (Environment Local Frame)")
parser.add_argument("--target_local_y", type=float, default=None, help="Target Y position (Environment Local Frame)")
parser.add_argument("--target_local_z", type=float, default=None, help="Target Z position (Environment Local Frame)")
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np

from pxr import UsdGeom, Gf
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_reach_surgery_room_cfg import (
    TM5ExtensionLinkOutReachEnvCfg
)

# robotarm_base local position
ROBOTARM_BASE_LOCAL_X = 0.3
ROBOTARM_BASE_LOCAL_Y = -0.1741
ROBOTARM_BASE_LOCAL_Z = 0.9493


@configclass
class TM5IKEvalEnvCfg(TM5ExtensionLinkOutReachEnvCfg):
    """
    Validation Environment Config that swaps RL Policy Action for Differential IK.
    Values are commanded as Absolute Poses (Position + Orientation) for the Flange.
    """
    def __post_init__(self):
        super().__post_init__()
        
        # 1. Replace Joint Position Action with Differential IK Action
        # We control 'flange' but offset it to the tip.
        
        # Measured offset from 'flange' to tip from check_usd script
        NEEDLE_TIP_OFFSET_POS = (-0.001152, 0.001055, 0.500000)
        NEEDLE_TIP_OFFSET_ROT = (1.0, 0.0, 0.0, 0.0) # Identity
        
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint_[1-6]"], 
            body_name="flange", # Valid body
            controller=DifferentialIKControllerCfg(
                command_type="pose", 
                use_relative_mode=False, # Absolute Pose Control
                ik_method="dls" # Damped Least Squares
            ),
            scale=1.0, 
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=NEEDLE_TIP_OFFSET_POS,
                rot=NEEDLE_TIP_OFFSET_ROT
            ),
        )
        
        # 2. Disable Actions Randomization/Noise if any
        
        # 3. Ensure 'replicate_physics' is False for multi-position test
        self.scene.replicate_physics = False


def randomize_robot_root_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
):
    """Randomize robot base position."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    
    num_envs_to_reset = len(env_ids)
    
    # Linear distribution for X
    x_offsets = torch.linspace(x_range[0], x_range[1], env.num_envs, device=env.device)[env_ids]
    
    y_offsets = torch.zeros(num_envs_to_reset, device=env.device) + y_range[0]
    z_offsets = torch.zeros(num_envs_to_reset, device=env.device) + z_range[0]
    
    robot = env.scene["robot"]
    env_origins = env.scene.env_origins[env_ids]
    
    default_local_pos = torch.tensor(
        [ROBOTARM_BASE_LOCAL_X, ROBOTARM_BASE_LOCAL_Y, ROBOTARM_BASE_LOCAL_Z], 
        device=env.device
    ).repeat(num_envs_to_reset, 1)
    
    shifts = torch.stack([x_offsets, y_offsets, z_offsets], dim=1)
    new_root_pos = env_origins + default_local_pos + shifts
    
    root_quat = robot.data.root_quat_w[env_ids]
    
    robot.write_root_pose_to_sim(torch.cat([new_root_pos, root_quat], dim=-1), env_ids=env_ids)
    
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


def run_ik_evaluation(env, num_episodes, device, fixed_target_local=None):
    """Run IK evaluation"""
    num_envs = env.unwrapped.num_envs
    print(f"\n[INFO] Running {num_episodes} IK steps across {num_envs} environments...")
    
    # Statistics
    final_errors = torch.zeros(num_envs, device=device)
    successes = torch.zeros(num_envs, dtype=torch.bool, device=device) # Success < 1cm error?
    
    SUCCESS_THRESHOLD = 0.01 # 1cm
    
    for episode_idx in range(num_episodes):
        obs, _ = env.reset()
        
        # --- CALCULATE TARGET POSE IN ROBOT BASE FRAME ---
        # The Action requires Target Pose in ROBOT BASE FRAME (because subtract_frame_transforms usually gives local)
        # Wait, ActionTerm input depends on implementation. 
        # Check TaskSpaceActions: "ee_pose_b, ee_quat_b = subtract_frame_transforms(root_pos_w...)"
        # Then controller computes q_des.
        # But DifferentialIKControllerCfg command_type="pose" usually EXPECTS pose in base frame if use_relative_mode=False?
        # Let's verify standard usage. Yes, set_command takes command. 
        # If absolute mode, command MUST be target pose in Base Frame.
        
        target_pos_b = None
        target_quat_b = None
        
        if fixed_target_local is not None:
             with torch.no_grad():
                robot = env.scene["robot"]
                robot_root_pos_w = robot.data.root_pos_w
                robot_root_quat_w = robot.data.root_quat_w
                
                env_origins = env.scene.env_origins
                target_local_expanded = fixed_target_local.unsqueeze(0).repeat(num_envs, 1)
                target_pos_w = env_origins + target_local_expanded
                
                # Target Orientation: Pointing DOWN typically
                # (0, 1, 0, 0) is often Y-down or similar? 
                # TM5 Default flange orientation needs to be considered.
                # Here we assume Identity quaternion (0,0,0,1) or existing robot ee orientation?
                # Let's use a fixed "Pointing Down" orientation if possible, or Identity.
                # In RL config, commands have (roll=0, pitch=pi, yaw=random). Pitch PI usually means pointing down.
                # Let's try to fetch a valid orientation from the current robot state or keep it fixed.
                # For "Reachability", Position is key. Orientation is secondary but constraints joint limits.
                # Let's use the current EE orientation to minimize rotation? No, that makes it too easy.
                # Let's use: (0, 1, 0, 0) [w,x,y,z] -> Rotated 180 on X?
                # Let's just use (1, 0, 0, 0) [w,x,y,z] identity for now, or fetch from a "good" pose.
                # Better: Use the orientation from a valid target in RL config. (0, 0, 1, 0)
                
                # Target Orientation: Use the preferred orientation of the needle body
                # preferred_quat=(-0.8007, 0.0423, 0.5955, 0.0497) [w, x, y, z]
                PREFERRED_QUAT = [-0.8007, 0.0423, 0.5955, 0.0497] # From measurement script
                target_quat_w = torch.tensor(PREFERRED_QUAT, device=device).unsqueeze(0).repeat(num_envs, 1)

                # Transform to Base Frame
                target_pos_b, target_quat_b = subtract_frame_transforms(
                    robot_root_pos_w, robot_root_quat_w,
                    target_pos_w, target_quat_w
                )
        
        # Run Simulation Steps (IK needs time to converge)
        # For IK solver in Env, we step the environment multiple times with the SAME action (Target Pose)
        # and see if it converges.
        
        CONVERGENCE_STEPS = 60 # 1 second at 60Hz
        
        print(f"  Evaluating Position {episode_idx} (Running {CONVERGENCE_STEPS} steps to converge)...")
        
        for _ in range(CONVERGENCE_STEPS):
            # Construct Action: (Pos, Rot) -> 7 dims
            # target_pos_b: (N, 3), target_quat_b: (N, 4)
            # Action: (N, 7)
            actions = torch.cat([target_pos_b, target_quat_b], dim=-1)
            
            # Step Env
            # The ActionTerm will invoke DifferentialIKController
            env.step(actions)
            
        # Check Error
        with torch.no_grad():
            robot = env.scene["robot"]
            # Get EE Pose (Flange)
            ee_idx = robot.find_bodies("flange")[0][0] # Assuming single body index
            ee_pos_w = robot.data.body_pos_w[:, ee_idx]
            ee_quat_w = robot.data.body_quat_w[:, ee_idx]

            # Calculate "Actual Tip Position" by applying the same offset
            # NEEDLE_TIP_OFFSET_POS = (-0.001152, 0.001055, 0.500000)
            offset_pos = torch.tensor([-0.001152, 0.001055, 0.500000], device=device).repeat(num_envs, 1)
            offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(num_envs, 1) # Identity

            # Apply offset to flange pose to get tip pose
            tip_pos_w, _ = combine_frame_transforms(ee_pos_w, ee_quat_w, offset_pos, offset_rot)
            
            # Calculate Error (World Frame)
            # Target World vs Actual Tip World
            dist = torch.norm(tip_pos_w - target_pos_w, dim=-1)
            
            final_errors = dist
            successes = dist < SUCCESS_THRESHOLD
            
    return {
        "final_errors": final_errors.cpu().numpy(),
        "successes": successes.cpu().numpy(),
        "mean_error": final_errors.mean().item(),
        "success_rate": successes.float().mean().item() * 100
    }

def main():
    print("="*60)
    print("Multi-Position IK Evaluation (Deterministic)")
    print("="*60)
    
    # Create IK-enabled Environment
    env_cfg = TM5IKEvalEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 5.0
    
    # Setup Randomizer
    offset_min = args_cli.x_min - ROBOTARM_BASE_LOCAL_X
    offset_max = args_cli.x_max - ROBOTARM_BASE_LOCAL_X
    
    env_cfg.events.randomize_robot_base_linear = EventTermCfg(
        func=randomize_robot_root_pose,
        mode="startup",
        params={
            "x_range": (offset_min, offset_max),
            "y_range": (0.0, 0.0),
            "z_range": (0.0, 0.0),
        },
    )
    
    # Create Env
    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.device
    print(f"[OK] IK Environment Created")
    
    # Resolve Target
    fixed_target_local = None
    if args_cli.use_usd_target:
        fixed_target_local = get_usd_target_position(env)
    elif args_cli.target_local_x is not None:
        fixed_target_local = torch.tensor([args_cli.target_local_x, args_cli.target_local_y, args_cli.target_local_z], device=device)
    else:
        # Default target (matches frame_prim in Extended Needle scene)
        fixed_target_local = torch.tensor([-0.3808, -0.8010, 1.0899], device=device)
    
    print(f"[INFO] Target (Local to Env Origin): {fixed_target_local.cpu().numpy()}")

    # Run Eval
    env.reset()
    results = run_ik_evaluation(env, args_cli.episodes, device, fixed_target_local)
    
    # Save Results
    # Map back results to X positions
    eval_x_positions = torch.linspace(args_cli.x_min, args_cli.x_max, args_cli.num_envs).tolist()
    
    per_position_stats = []
    for i, x_pos in enumerate(eval_x_positions):
        per_position_stats.append({
            "x": x_pos,
            "success": bool(results["successes"][i]),
            "error": float(results["final_errors"][i])
        })
        
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "method": "DifferentialIK_DLS",
        "num_envs": args_cli.num_envs,
        "target_pose": fixed_target_local.cpu().numpy().tolist(),
        "statistics": {
            "mean_error": results['mean_error'],
            "success_rate": results['success_rate']
        },
        "per_position_results": per_position_stats
    }
    
    save_path = Path(args_cli.save_file)
    if save_path.parent.name:
        save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
        
    print(f"\n[DONE] Results saved to {save_path}")
    print(f"Success Rate: {results['success_rate']:.1f}%")
    print(f"Mean Error: {results['mean_error']:.4f} m")

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
