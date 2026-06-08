# ./isaaclab.bat -p scripts/isaaclab_ws/surgery_rl/train_surgery_rl.py --num_envs 64

import argparse
import torch

from isaaclab.app import AppLauncher

# 添加參數
parser = argparse.ArgumentParser(description="Train RL policy for surgery robot control.")
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments for training.")
# AppLauncher 會自動添加 --headless 參數,不需要手動添加
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 啟動 app (必須在導入 Isaac Lab 模組之前)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 導入其他必要模組 (必須在 AppLauncher 之後)
from surgery_rl_env import SurgeryRLEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.dict import print_dict

def main():
    """主訓練函數"""
    
    # 創建環境配置
    env_cfg = SurgeryRLEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # 創建環境
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    print("Environment created successfully!")
    print(f"Number of environments: {env.num_envs}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # 重置環境
    obs, _ = env.reset()
    
    # 簡單的隨機策略測試
    print("\nRunning random policy test...")
    
    for step in range(1000):
        # 隨機動作
        actions = env.action_space.sample()
        
        # 執行動作
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # 輸出進度
        if step % 100 == 0:
            print(f"Step {step}:")
            print(f"  Mean reward: {torch.mean(rewards):.4f}")
            print(f"  Mean distance to target: {torch.mean(torch.norm(obs['policy']['ee_to_target'], dim=-1)):.4f}")
            print(f"  Terminated envs: {torch.sum(terminated)}")
    
    print("\nTraining test completed!")
    
    # 這裡可以添加實際的 RL 訓練邏輯
    # 例如使用 stable-baselines3, RLLib, 或其他 RL 庫
    
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()