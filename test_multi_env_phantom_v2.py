"""測試多環境下 phantom 位置 - Version 2

使用 surgery_room randomizer 來移動整個場景
"""

import argparse
import torch
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test multi-env phantom positions v2")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
sys.path.append(str(Path(__file__).parent))

from randomize_surgery_room_position_v2 import (
    randomize_surgery_room_position,
    RandomizeSurgeryRoomPositionCfg,
)

from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_reach_surgery_room_cfg import (
    TM5ExtensionLinkOutReachEnvCfg
)
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg


def main():
    print(f"\n{'='*70}")
    print(f"Testing Multi-Environment Phantom Positions (v2)")
    print(f"Using surgery_room randomizer")
    print(f"{'='*70}\n")
    
    # 創建環境配置
    env_cfg = TM5ExtensionLinkOutReachEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 6.0  # 環境間距 6m（更遠）
    
    # 添加 surgery_room randomizer（startup mode）
    randomizer_cfg = RandomizeSurgeryRoomPositionCfg()
    randomizer_cfg.x_range = (-0.6, 0.6)  # 擴大 X 範圍到 1.2m
    
    env_cfg.events.randomize_surgery_room = EventTermCfg(
        func=randomize_surgery_room_position,
        mode="startup",  # 在環境創建時執行一次
        params={
            "x_range": randomizer_cfg.x_range,
            "y_range": randomizer_cfg.y_range,
            "z_range": randomizer_cfg.z_range,
        },
    )
    
    print(f"[INFO] Creating {args_cli.num_envs} environments with surgery_room randomizer...")
    print(f"[INFO] X offset range: {randomizer_cfg.x_range}")
    print(f"[INFO] Expected phantom X positions: from {randomizer_cfg.x_range[0]:.2f} to {randomizer_cfg.x_range[1]:.2f}\n")
    
    # 創建環境
    env = ManagerBasedRLEnv(cfg=env_cfg)
    print(f"[OK] Environment created with {args_cli.num_envs} parallel environments\n")
    
    # Reset 一次
    env.reset()
    print(f"[OK] Environment reset\n")
    
    # 檢查每個環境的 robot 位置（surgery_room 是 XFormPrim沒有 data）
    print(f"{'='*70}")
    print(f"Verifying Robot Base Positions")
    print(f"{'='*70}\n")
    
    # 獲取 robot 位置
    robot = env.unwrapped.scene["robot"]
    robot_positions = robot.data.root_pos_w
    
    print("Robot World Positions:")
    for env_idx in range(min(args_cli.num_envs, 10)):
        pos = robot_positions[env_idx]
        print(f"  Env {env_idx}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
    
    if args_cli.num_envs > 10:
        print(f"  ... and {args_cli.num_envs - 10} more environments")
    
    # 檢查位置是否真的不同
    x_positions = robot_positions[:, 0]
    x_min = x_positions.min().item()
    x_max = x_positions.max().item()
    x_std = x_positions.std().item()
    
    print(f"\nPosition Statistics:")
    print(f"  X min: {x_min:.4f}m")
    print(f"  X max: {x_max:.4f}m")
    print(f"  X std: {x_std:.4f}m")
    
    if x_std > 0.01:
        print(f"\n[SUCCESS] ✓ Positions are different across environments!")
    else:
        print(f"\n[WARNING] ✗ Positions appear to be the same across environments")
    
    # 運行一些步驟
    print(f"\n[INFO] Running 100 simulation steps...")
    for step_idx in range(100):
        obs, rewards, dones, truncated, info = env.step(
            torch.zeros(env.action_space.shape, device=env.device)
        )
    
    print(f"\n{'='*70}")
    print(f"Check the viewer - each environment should have phantom at different X position")
    print(f"Expected spread: {randomizer_cfg.x_range[0]:.2f}m to {randomizer_cfg.x_range[1]:.2f}m")
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
