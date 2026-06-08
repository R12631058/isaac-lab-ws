"""測試多環境下 robot base 位置 - 並行環境版本

在一個多環境場景中，每個環境設置不同的 robot base 位置
使用 surgery_room randomizer，類似 phantom 測試
"""

import argparse
import torch
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test multi-env robot positions (parallel)")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
sys.path.append(str(Path(__file__).parent))

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg
from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_reach_surgery_room_cfg import (
    TM5ExtensionLinkOutReachEnvCfg
)


# robotarm_base 在 USD 中的本地位置
ROBOTARM_BASE_LOCAL_X = 0.3596
ROBOTARM_BASE_LOCAL_Y = -0.1741
ROBOTARM_BASE_LOCAL_Z = 0.9493

def randomize_robot_and_scene_position(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
):
    """
    Randomize position by modifying the USD Xform of the Environment Root AND updating Articulation pose.
    """
    from pxr import Gf, UsdGeom
    
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    
    num_envs_to_reset = len(env_ids)
    
    # Linear distribution
    x_offsets = torch.linspace(x_range[0], x_range[1], num_envs_to_reset, device=env.device)
    y_offsets = torch.zeros(num_envs_to_reset, device=env.device) + y_range[0]
    z_offsets = torch.zeros(num_envs_to_reset, device=env.device) + z_range[0]
    
    stage = env.scene.stage
    
    print(f"[Randomizer] Modifying Environment Roots and Robot Poses for {num_envs_to_reset} envs...")
    
    # 1. Modify USD Xform for the Environment Root (Visually moves the room/statics)
    for i, env_id in enumerate(env_ids):
        env_id_int = env_id.item()
        root_prim_path = f"/World/envs/env_{env_id_int}/Root"
        root_prim = stage.GetPrimAtPath(root_prim_path)
        if root_prim.IsValid():
            xform = UsdGeom.Xformable(root_prim)
            xform.ClearXformOpOrder()
            if root_prim.HasAttribute("xformOp:transform"):
                root_prim.RemoveProperty("xformOp:transform")
            op = xform.AddTranslateOp()
            op.Set(Gf.Vec3d(float(x_offsets[i]), float(y_offsets[i]), float(z_offsets[i])))

    # 2. Modify Robot Articulation Root Pose (Physically moves the robot)
    # We must use the Articulation View API to handle Physics/GPU pipeline correctly.
    robot = env.scene["robot"]
    
    # Get environment origins
    env_origins = env.scene.env_origins[env_ids]
    
    # Default local position of the robot base (relative to env origin)
    default_local_pos = torch.tensor(
        [ROBOTARM_BASE_LOCAL_X, ROBOTARM_BASE_LOCAL_Y, ROBOTARM_BASE_LOCAL_Z], 
        device=env.device
    ).repeat(num_envs_to_reset, 1)
    
    # Calculate target local offsets (offset from default)
    # x_offsets contains the "Shift" we applied to the room.
    # So we apply the same shift to the robot.
    shifts = torch.stack([x_offsets, y_offsets, z_offsets], dim=1)
    
    # New World Position = Env Origin + Default Local + Shift
    new_root_pos = env_origins + default_local_pos + shifts
    
    # Get current orientation (keep it)
    root_quat = robot.data.root_quat_w[env_ids]
    
    # Set the new state
    robot.write_root_pose_to_sim(torch.cat([new_root_pos, root_quat], dim=-1), env_ids=env_ids)
    
    # Also update the velocities to zero to stop any drift
    zeros_vel = torch.zeros((num_envs_to_reset, 6), device=env.device)
    robot.write_root_velocity_to_sim(zeros_vel, env_ids=env_ids)


def randomize_surgery_room_position_v3(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
):
    """
    Experimental: Modify the Robot Base Link Fixed Joint or Transform? 
    Actually, let's just try to modify the Root XForm again but ensure we trigger a way for Physics to see it.
    
    Alternative: We use `env.scene.rigid_objects` if we can?
    No, let's stick to USD modification but try to touch the Articulation specifically.
    """
    from pxr import Gf, UsdGeom
    
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    
    # Linear distribution
    x_offsets = torch.linspace(x_range[0], x_range[1], len(env_ids), device=env.device)
    
    stage = env.scene.stage
    
    for i, env_id in enumerate(env_ids):
        env_id_int = env_id.item()
        
        # Move the Env Root (this SHOULD work for visual/static geometry)
        prim_path = f"/World/envs/env_{env_id_int}/Root"
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            xformable = UsdGeom.Xformable(prim)
            xformable.ClearXformOpOrder()
            if prim.HasAttribute("xformOp:transform"):
                prim.RemoveProperty("xformOp:transform")
            op = xformable.AddTranslateOp()
            op.Set(Gf.Vec3d(float(x_offsets[i]), 0.0, 0.0))


def main():
    print(f"\n{'='*70}")
    print(f"Testing Multi-Environment Robot Base Positions (Parallel)")
    print(f"Each environment should have robot at different position")
    print(f"{'='*70}\n")
    
    # 目標：每個環境的 robot 在不同的 X 位置（每0.5m一個）
    # 例如：4個環境 → -0.5m, 0.0m, 0.5m, 1.0m
    num_envs = args_cli.num_envs
    target_robot_start = -0.5
    target_robot_spacing = 0.5
    target_robot_end = target_robot_start + (num_envs - 1) * target_robot_spacing
    
    # 計算 surgery_room offset 範圍
    # offset = target_robot_pos - robotarm_base_local_pos
    offset_x_start = target_robot_start - ROBOTARM_BASE_LOCAL_X
    offset_x_end = target_robot_end - ROBOTARM_BASE_LOCAL_X
    
    print(f"[INFO] Configuration:")
    print(f"  Number of environments: {num_envs}")
    print(f"  Target robot X positions: {target_robot_start:.2f}m to {target_robot_end:.2f}m")
    print(f"  Robot spacing: {target_robot_spacing}m")
    print(f"  Expected positions: {', '.join([f'{target_robot_start + i*target_robot_spacing:.2f}m' for i in range(num_envs)])}")
    print(f"  Surgery room X offset range: {offset_x_start:.4f}m to {offset_x_end:.4f}m")
    print(f"  Environment spacing: 6.0m\n")
    
    # 創建環境配置
    env_cfg = TM5ExtensionLinkOutReachEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.env_spacing = 6.0  # 環境間距 6m
    
    # CRITICAL FIX: Disable replicate_physics to allow individual USD environment modifications
    env_cfg.scene.replicate_physics = False
    
    # Also disable GPU pipeline/physics to avoid strict GPU API limitations when handling individual envs
    # For small number of environments (e.g. 4), CPU physics is perfectly fast.
    env_cfg.sim.use_gpu_pipeline = False
    env_cfg.sim.physx.use_gpu = False
    print(f"[INFO] replicate_physics=False, use_gpu=False (CPU Mode) - Enabled individual env/robot positioning")

    
    env_cfg.events.randomize_surgery_room = EventTermCfg(
        func=randomize_robot_and_scene_position,
        mode="startup",
        params={
            "x_range": (offset_x_start, offset_x_end),
            "y_range": (0.0, 0.0),
            "z_range": (0.0, 0.0),
        },
    )
    
    print(f"[INFO] Creating {num_envs} environments with surgery_room randomizer...")
    print(f"[INFO] Randomizer will linearly distribute robot X positions\n")
    
    # 創建環境
    env = ManagerBasedRLEnv(cfg=env_cfg)
    print(f"[OK] Environment created\n")
    
    # Reset 一次
    env.reset()
    print(f"[OK] Environment reset\n")
    
    # 檢查每個環境的 robot 位置
    print(f"{'='*70}")
    print(f"Verifying Robot Base Positions")
    print(f"{'='*70}\n")
    
    robot = env.scene["robot"]
    # Update buffers to ensure we have latest data
    robot.update(0.0)
    
    robot_positions_w = robot.data.root_pos_w
    env_origins = env.scene.env_origins
    
    print("Expected vs Actual Robot X Positions (Relative to Env Origin):")
    print(f"{'Env':<6} {'Expected X':<12} {'Actual X (CW)':<14} {'Relative X':<12} {'Match':<10}")
    print("-" * 80)
    
    all_match = True
    x_positions = []
    
    for env_idx in range(num_envs):
        expected_x = target_robot_start + env_idx * target_robot_spacing
        
        # Calculate Relative Position
        actual_pos_w = robot_positions_w[env_idx]
        env_origin = env_origins[env_idx]
        relative_pos = actual_pos_w - env_origin
        
        rel_x = relative_pos[0].item()
        actual_x_w = actual_pos_w[0].item()
        
        x_positions.append(rel_x)
        
        diff = abs(rel_x - expected_x)
        match = "✓" if diff < 1e-3 else "✗"
        if diff >= 1e-3:
            all_match = False
            
        print(f"{env_idx:<6} {expected_x:<12.4f} {actual_x_w:<14.4f} {rel_x:<12.4f} {match:<10}")

    print(f"\nPosition Statistics (Relative X):")
    print(f"  Min: {min(x_positions):.4f}m")
    print(f"  Max: {max(x_positions):.4f}m")
    print(f"  Range: {max(x_positions) - min(x_positions):.4f}m")

    print(f"\n{'='*70}")
    if all_match:
        print(f"[SUCCESS] ✓ All robot positions match expected values!")
        print(f"[SUCCESS] ✓ Multi-env robot positioning works!")
    else:
        x_rng = max(x_positions) - min(x_positions)
        if x_rng > 0.01:
            print(f"[PARTIAL] ⚠ Positions don't match exactly, BUT they are different!")
            print(f"[INFO] X positions vary by {x_rng:.4f}m")
        else:
            print(f"[FAIL] ✗ All robots appear to be at the same position")
    print(f"{'='*70}\n")

    
    # 運行一些步驟
    print(f"[INFO] Running 100 simulation steps...")
    print(f"       Check the viewer to visually verify robot positions\n")
    
    for step_idx in range(100):
        obs, rewards, dones, truncated, info = env.step(
            torch.zeros(env.action_space.shape, device=env.device)
        )
    
    print(f"{'='*70}")
    print(f"Viewer Check:")
    print(f"  - Look at the scene from above (top view)")
    print(f"  - You should see {num_envs} robots")
    print(f"  - If working correctly, they should be at X positions:")
    for i in range(num_envs):
        expected = target_robot_start + i * target_robot_spacing
        print(f"    Env {i}: X = {expected:.2f}m")
    print(f"  - Robots should be spaced {target_robot_spacing}m apart in X direction")
    print(f"{'='*70}\n")
    
    try:
        input("Press Enter to close...")
    except EOFError:
        print("(Non-interactive mode)")
        for _ in range(200):
            env.step(torch.zeros(env.action_space.shape, device=env.device))
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
