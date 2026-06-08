#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
測試 TM5 reach 環境的腳本
"""

import argparse
from isaaclab.app import AppLauncher

# 添加命令行參數
parser = argparse.ArgumentParser(description="測試 TM5 reach 環境")
parser.add_argument("--num_envs", type=int, default=4, help="並行環境數量")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 啟動 Omniverse 應用程式
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""其餘代碼"""

import gymnasium as gym
import torch

# 導入 TM5 任務
import isaaclab_tasks  # noqa: F401

def main():
    """主要測試函數"""
    
    print("[INFO] 創建 TM5 reach 環境...")
    
    # 導入配置
    from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.joint_pos_env_cfg import TM5ReachEnvCfg
    
    # 創建環境配置
    env_cfg = TM5ReachEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    # 創建環境
    env = gym.make("Isaac-Reach-TM5-v0", cfg=env_cfg)
    
    print(f"[INFO] 環境創建成功！")
    print(f"[INFO] 觀測空間: {env.observation_space}")
    print(f"[INFO] 動作空間: {env.action_space}")
    print(f"[INFO] 環境數量: {env.unwrapped.num_envs}")
    
    # 重置環境
    print("[INFO] 重置環境...")
    obs, info = env.reset()
    print(f"[INFO] 觀測維度: {obs['policy'].shape}")
    
    # 運行幾個步驟
    print("[INFO] 運行測試步驟...")
    for step in range(100):
        # 隨機動作 - 修正維度問題
        actions = torch.randn(env.unwrapped.num_envs, env.action_space.shape[1], device=env.unwrapped.device)
        
        # 執行步驟
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        if step % 20 == 0:
            print(f"[INFO] 步驟 {step}: 平均獎勵 = {rewards.mean():.3f}")
            
        # 檢查是否需要重置
        if terminated.any() or truncated.any():
            print(f"[INFO] 步驟 {step}: 某些環境已結束，自動重置")
    
    print("[INFO] 測試完成！")
    
    # 關閉環境
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
