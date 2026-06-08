"""測試多環境下 robot base 位置 - 每0.5m一個

使用 surgery_room randomizer 來設置不同的機器人位置
"""

import argparse
import torch
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test multi-env robot base positions")
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
    print(f"Testing Multi-Environment Robot Base Positions")
    print(f"Target: Robot positions every 0.5m starting from origin")
    print(f"{'='*70}\n")
    
    # robotarm_base 在 USD 中的本地位置
    ROBOTARM_BASE_LOCAL_X = 0.3596
    ROBOTARM_BASE_LOCAL_Y = -0.1741
    ROBOTARM_BASE_LOCAL_Z = 0.9493
    
    # 目標機器人位置：從 -0.5m 開始，每 0.5m 一個
    # 4個環境: -0.5, 0.0, 0.5, 1.0
    target_robot_start = -0.5
    target_robot_spacing = 0.5
    num_envs = args_cli.num_envs
    target_robot_end = target_robot_start + (num_envs - 1) * target_robot_spacing
    
    # 計算需要的 surgery_room offset 範圍
    # offset = target_robot_pos - robotarm_base_local_pos
    offset_x_start = target_robot_start - ROBOTARM_BASE_LOCAL_X
    offset_x_end = target_robot_end - ROBOTARM_BASE_LOCAL_X
    
    print(f"[INFO] Configuration:")
    print(f"  Number of environments: {num_envs}")
    print(f"  Target robot X positions: {target_robot_start:.2f}m to {target_robot_end:.2f}m (every {target_robot_spacing:.2f}m)")
    print(f"  Surgery room X offset range: {offset_x_start:.4f}m to {offset_x_end:.4f}m")
    print(f"  Robotarm_base local position in USD: ({ROBOTARM_BASE_LOCAL_X:.4f}, {ROBOTARM_BASE_LOCAL_Y:.4f}, {ROBOTARM_BASE_LOCAL_Z:.4f})\n")
    
    # 創建環境配置
    env_cfg = TM5ExtensionLinkOutReachEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.env_spacing = 6.0  # 環境間距 6m
    
    # 配置 surgery_room randomizer
    randomizer_cfg = RandomizeSurgeryRoomPositionCfg()
    randomizer_cfg.x_range = (offset_x_start, offset_x_end)  # 計算好的偏移範圍
    randomizer_cfg.y_range = (0.0, 0.0)  # Y 不變
    randomizer_cfg.z_range = (0.0, 0.0)  # Z 不變
    
    env_cfg.events.randomize_surgery_room = EventTermCfg(
        func=randomize_surgery_room_position,
        mode="startup",
        params={
            "x_range": randomizer_cfg.x_range,
            "y_range": randomizer_cfg.y_range,
            "z_range": randomizer_cfg.z_range,
        },
    )
    
    print(f"[INFO] Creating {num_envs} environments with surgery_room randomizer...")
    
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
    
    robot = env.unwrapped.scene["robot"]
    robot_positions = robot.data.root_pos_w
    
    print("Expected vs Actual Robot X Positions:")
    print(f"{'Env':<6} {'Expected X':<12} {'Actual X':<12} {'Difference':<12} {'Status':<10}")
    print("-" * 70)
    
    all_match = True
    for env_idx in range(num_envs):
        expected_x = target_robot_start + env_idx * target_robot_spacing
        actual_pos = robot_positions[env_idx]
        actual_x = actual_pos[0].item()
        diff = abs(actual_x - expected_x)
        
        status = "✓ PASS" if diff < 0.01 else "✗ FAIL"
        if diff >= 0.01:
            all_match = False
        
        print(f"{env_idx:<6} {expected_x:<12.4f} {actual_x:<12.4f} {diff:<12.6f} {status:<10}")
    
    print()
    
    # 統計資訊
    x_positions = robot_positions[:, 0]
    x_min = x_positions.min().item()
    x_max = x_positions.max().item()
    x_std = x_positions.std().item()
    
    print(f"Position Statistics:")
    print(f"  X min: {x_min:.4f}m")
    print(f"  X max: {x_max:.4f}m")
    print(f"  X range: {x_max - x_min:.4f}m")
    print(f"  X std: {x_std:.4f}m")
    
    print(f"\n{'='*70}")
    if all_match:
        print(f"[SUCCESS] ✓ All robot positions match expected values!")
    else:
        print(f"[WARNING] ✗ Some robot positions don't match expected values")
    print(f"{'='*70}\n")
    
    # 運行一些步驟
    print(f"[INFO] Running 100 simulation steps...")
    print(f"       Check the viewer - robots should be at different X positions\n")
    
    for step_idx in range(100):
        obs, rewards, dones, truncated, info = env.step(
            torch.zeros(env.action_space.shape, device=env.device)
        )
    
    print(f"{'='*70}")
    print(f"Visualization Tips:")
    print(f"  - Look at the viewer from above (top view)")
    print(f"  - You should see {num_envs} robots at different X positions")
    print(f"  - Expected spacing: {target_robot_spacing}m between each robot")
    print(f"  - X positions should be: {', '.join([f'{target_robot_start + i*target_robot_spacing:.1f}m' for i in range(num_envs)])}")
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
