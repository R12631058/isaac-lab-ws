#!/usr/bin/env python3
"""
測試帶有碰撞避免的 TM5 Reach 環境

驗證:
1. Joint 1 角度避免獎勵是否生效
2. 關節極限懲罰是否生效
3. 環境是否正常運行
"""

import argparse
from isaaclab.app import AppLauncher

# 創建 argparse
parser = argparse.ArgumentParser(description="測試 TM5 Reach Surgery Room 碰撞避免")
parser.add_argument("--num_envs", type=int, default=4, help="環境數量")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 啟動應用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np

from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_reach_surgery_room_cfg import (
    TM5ReachSurgeryRoomEnvCfg,
)
from isaaclab.envs import ManagerBasedRLEnv


def main():
    """測試主函數"""
    
    print("\n" + "="*80)
    print("測試 TM5 Reach Surgery Room - 碰撞避免配置")
    print("="*80 + "\n")
    
    # 創建環境配置
    env_cfg = TM5ReachSurgeryRoomEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # 顯示獎勵配置
    print("獎勵項配置:")
    print("-" * 80)
    for reward_name in dir(env_cfg.rewards):
        if not reward_name.startswith("_"):
            reward = getattr(env_cfg.rewards, reward_name)
            if hasattr(reward, "weight"):
                print(f"  {reward_name:45s} | weight: {reward.weight:7.4f}")
    print("-" * 80 + "\n")
    
    # 創建環境
    print("創建環境...")
    env = ManagerBasedRLEnv(cfg=env_cfg)
    print(f"✓ 環境創建成功! 環境數量: {env.num_envs}\n")
    
    # 重置環境
    print("重置環境...")
    obs, _ = env.reset()
    policy_obs = obs.get("policy", obs)
    if isinstance(policy_obs, dict):
        first_value = list(policy_obs.values())[0] if policy_obs else None
        obs_shape = first_value.shape if first_value is not None else "unknown"  # type: ignore
    else:
        obs_shape = policy_obs.shape  # type: ignore
    print(f"✓ 環境重置成功! 觀察維度: {obs_shape}\n")
    
    # 運行一些步驟來測試獎勵計算
    print("運行測試步驟...")
    print("-" * 80)
    
    for step in range(20):
        # 生成隨機動作
        action_dim = env.action_manager.total_action_dim
        actions = torch.randn(env.num_envs, action_dim, device=env.device) * 0.1
        
        # 執行步驟
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # 獲取 joint_1 的角度
        robot = env.scene["robot"]
        joint_1_angles = robot.data.joint_pos[:, 0].cpu().numpy()
        
        # 獲取獎勵明細
        reward_items = info.get("log", {})
        
        # 顯示 joint_1 角度和相關獎勵
        if step % 5 == 0:
            # 檢查是否在安全區域 (-270° ~ -30°, 即 -4.71 ~ -0.52 rad)
            in_safe_zone = (joint_1_angles >= -4.71) & (joint_1_angles <= -0.52)
            safe_count = np.sum(in_safe_zone)
            
            print(f"\n步驟 {step:3d}:")
            print(f"  Joint 1 角度 (度): {np.rad2deg(joint_1_angles)}")
            print(f"  安全區域 (-270° ~ -30°): {safe_count}/{env.num_envs} 個環境")
            print(f"  總獎勵: {rewards.cpu().numpy()}")
            
            # 顯示碰撞避免相關的獎勵
            if "reward_joint_1_avoid_small_angles" in reward_items:
                print(f"  Joint 1 避免小角度獎勵: {reward_items['reward_joint_1_avoid_small_angles'].cpu().numpy()}")
            if "reward_joint_limits_penalty" in reward_items:
                print(f"  關節極限懲罰: {reward_items['reward_joint_limits_penalty'].cpu().numpy()}")
    
    print("\n" + "-" * 80)
    print("\n✓ 測試完成!")
    print("\n說明:")
    print("  - Joint 1 安全區域: -270° ~ -30° (-4.71 ~ -0.52 rad)")
    print("  - Joint 1 危險區域: -30° ~ +180° (給予高懲罰)")
    print("  - 懲罰權重: -5.0 (強制避開危險區域)")
    print("  - 機器人應該只在安全區域內運動\n")
    
    # 關閉環境
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
