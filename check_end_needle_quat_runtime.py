"""
在 Isaac Lab 環境中直接獲取 End_needle 的初始四元數
這樣可以確保和訓練時使用的是同樣的值
"""

import argparse
from isaaclab.app import AppLauncher

# 創建參數解析器
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args([])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 導入必要模組
import torch
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_reach_surgery_room_cfg import TM5NeedleTipStableReachEnvCfg

def main():
    # 創建環境配置
    env_cfg = TM5NeedleTipStableReachEnvCfg()
    env_cfg.scene.num_envs = 1
    
    # 使用正確的方式創建環境
    from isaaclab.envs import ManagerBasedRLEnv
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # 重置環境
    obs, info = env.reset()
    
    # 獲取 robot asset
    robot = env.scene["robot"]
    
    # 查找 End_needle body 的索引
    body_names = robot.data.body_names
    print(f"\n機器人所有 body 名稱: {body_names}")
    
    # 找到 End_needle 的索引
    end_needle_idx = None
    for i, name in enumerate(body_names):
        if "End_needle" in name:
            end_needle_idx = i
            print(f"找到 End_needle 在索引: {i}, 名稱: {name}")
            break
    
    if end_needle_idx is None:
        print("錯誤: 找不到 End_needle body!")
        env.close()
        simulation_app.close()
        return
    
    # 獲取 body_state_w
    body_state_w = robot.data.body_state_w
    print(f"\nbody_state_w shape: {body_state_w.shape}")
    
    # 獲取 End_needle 的四元數 (索引 3:7 是 quat)
    end_needle_quat = body_state_w[0, end_needle_idx, 3:7]
    print(f"\n========================================")
    print(f"End_needle 四元數 (w, x, y, z):")
    print(f"  w = {end_needle_quat[0].item():.4f}")
    print(f"  x = {end_needle_quat[1].item():.4f}")
    print(f"  y = {end_needle_quat[2].item():.4f}")
    print(f"  z = {end_needle_quat[3].item():.4f}")
    print(f"\n複製這個值到 NeedleTipStableRewardsCfg.ee_orientation_keep_stable.params.preferred_quat:")
    print(f"  preferred_quat=({end_needle_quat[0].item():.4f}, {end_needle_quat[1].item():.4f}, {end_needle_quat[2].item():.4f}, {end_needle_quat[3].item():.4f})")
    print(f"========================================")
    
    # 也顯示位置
    end_needle_pos = body_state_w[0, end_needle_idx, 0:3]
    print(f"\nEnd_needle 位置: ({end_needle_pos[0].item():.4f}, {end_needle_pos[1].item():.4f}, {end_needle_pos[2].item():.4f})")
    
    # 關閉環境
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
