"""Visualize target position in Isaac Sim with GUI
This script opens a visual window so you can see:
1. The robot arm position
2. The target marker (red sphere)
3. The phantom position
"""

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Visualize target position")
parser.add_argument("--num_resets", type=int, default=10, help="Number of resets to visualize")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch with rendering enabled (default)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_reach_surgery_room_cfg import TM5ExtensionLinkOutReachEnvCfg

# Import markers for visualization
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG, CUBOID_MARKER_CFG
import isaaclab.sim as sim_utils

def main():
    # Create environment
    env_cfg = TM5ExtensionLinkOutReachEnvCfg()
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # Create target marker (red sphere)
    marker_cfg = CUBOID_MARKER_CFG.copy()
    marker_cfg.markers["cuboid"].size = (0.02, 0.02, 0.02)
    marker_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
    marker_cfg.prim_path = "/Visuals/TargetMarker"
    target_marker = VisualizationMarkers(marker_cfg)
    
    # Create robot base marker (blue)
    robot_marker_cfg = CUBOID_MARKER_CFG.copy()
    robot_marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
    robot_marker_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0))
    robot_marker_cfg.prim_path = "/Visuals/RobotBaseMarker"
    robot_marker = VisualizationMarkers(robot_marker_cfg)
    
    print("\n" + "="*70)
    print("TARGET POSITION VISUALIZATION")
    print("="*70)
    print("Press SPACE to reset and see new target position")
    print("Press ESC to exit")
    print("="*70 + "\n")
    
    # Initial reset
    env.reset()
    
    robot = env.scene["robot"]
    cmd = env.command_manager.get_term("ee_pose")
    
    reset_count = 0
    step_count = 0
    
    while simulation_app.is_running():
        # Get current positions
        robot_pos = robot.data.root_pos_w[0:1]  # Keep batch dim
        robot_quat = robot.data.root_quat_w[0:1]
        
        # Get target in world frame
        target_cmd = cmd.command[0, :3]
        
        # Calculate target world position
        # target_world = robot_world + target_cmd (roughly, need rotation)
        # For simplicity, use pose_command_w if available, or calculate manually
        # Actually, Isaac Lab stores the world position calculation internally
        # Let's calculate it properly using the robot frame
        
        from isaaclab.utils.math import quat_rotate
        target_local = cmd.command[0:1, :3]  # (1, 3)
        target_world = robot_pos + quat_rotate(robot_quat, target_local)
        
        # Visualize markers
        target_marker.visualize(target_world)
        robot_marker.visualize(robot_pos)
        
        # Print info every 100 steps or on reset
        if step_count % 100 == 0:
            print(f"\n[Reset {reset_count}] Step {step_count}")
            print(f"  Robot Root (World): {robot_pos[0].cpu().numpy()}")
            print(f"  Target Cmd (Robot Frame): {target_cmd.cpu().numpy()}")
            print(f"  Target (World, calculated): {target_world[0].cpu().numpy()}")
        
        # Step environment with zero action (just to keep simulation running)
        action = torch.zeros(1, 6, device=env.device)
        obs, _, dones, _, _ = env.step(action)
        
        step_count += 1
        
        # Auto-reset every 200 steps to show different targets
        if step_count % 200 == 0 and reset_count < args.num_resets:
            env.reset()
            reset_count += 1
            print(f"\n{'='*50}")
            print(f"AUTO RESET #{reset_count}")
            print(f"{'='*50}")
    
    env.close()

if __name__ == "__main__":
    main()
