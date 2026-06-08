"""
分析 reset 後的四元數差異
檢查為什麼 ee_orientation_keep_stable 獎勵是 0
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
from isaaclab.utils.math import quat_error_magnitude
from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_reach_surgery_room_cfg import TM5NeedleTipStableReachEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

def main():
    # 創建環境配置
    env_cfg = TM5NeedleTipStableReachEnvCfg()
    env_cfg.scene.num_envs = 4
    
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # 定義 preferred_quat（配置中的值）
    preferred_quat = torch.tensor(
        [[-0.5815, -0.5513, -0.5981, 0.0150]], 
        device=env.device
    ).repeat(4, 1)  # 擴展到 4 個環境
    
    print(f"\n目標四元數 (preferred_quat): {preferred_quat[0].tolist()}")
    
    # 多次 reset 並測量四元數差異
    for i in range(3):
        print(f"\n========== Reset {i+1} ==========")
        obs, info = env.reset()
        
        # 獲取 End_needle 的四元數
        robot = env.scene["robot"]
        body_state_w = robot.data.body_state_w
        
        # End_needle 索引是 11
        end_needle_quat = body_state_w[:, 11, 3:7]
        
        print(f"\n各環境的 End_needle 四元數:")
        for env_idx in range(4):
            q = end_needle_quat[env_idx]
            print(f"  Env {env_idx}: (w={q[0].item():.4f}, x={q[1].item():.4f}, y={q[2].item():.4f}, z={q[3].item():.4f})")
        
        # 計算與 preferred_quat 的誤差
        quat_error = quat_error_magnitude(end_needle_quat, preferred_quat)
        print(f"\n與 preferred_quat 的角度誤差 (rad):")
        for env_idx in range(4):
            print(f"  Env {env_idx}: {quat_error[env_idx].item():.4f} rad ({quat_error[env_idx].item() * 180 / 3.14159:.2f}°)")
        
        # 計算 tanh reward（std=0.1）
        std = 0.1
        reward = 1.0 - torch.tanh(quat_error / std)
        print(f"\ntanh reward (std={std}):")
        for env_idx in range(4):
            print(f"  Env {env_idx}: {reward[env_idx].item():.4f}")
    
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
