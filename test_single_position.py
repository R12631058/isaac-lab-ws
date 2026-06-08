"""測試單一位置 - 直接在代碼中指定位置"""

import argparse
import torch
from pathlib import Path

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Test single position")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_reach_surgery_room_cfg import (
    TM5ExtensionLinkOutReachEnvCfg
)
from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.modules import ActorCritic


def main():
    # 測試位置列表（直接在代碼中指定）
    test_positions = [
        {"x": -0.52, "y": 0.1, "z": 0.0, "name": "Position_1"},
        {"x": -0.34, "y": 0.1, "z": 0.0, "name": "Position_2"},
        {"x": -0.16, "y": 0.1, "z": 0.0, "name": "Position_3"},
    ]
    
    checkpoint_path = "logs/rsl_rl/tm5_reach_stable_v2/2026-01-05_14-03-14/model_2999.pt"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for pos_config in test_positions:
        print(f"\n{'='*70}")
        print(f"Testing {pos_config['name']}: X={pos_config['x']:.2f}, Y={pos_config['y']:.2f}, Z={pos_config['z']:.2f}")
        print(f"{'='*70}")
        
        # 創建環境配置
        env_cfg = TM5ExtensionLinkOutReachEnvCfg()
        env_cfg.scene.num_envs = 1
        env_cfg.commands.ee_pose.resampling_time_range = (1e10, 1e10)
        
        # 關鍵：修改 surgery_room（整個場景）的位置
        # 從第一次執行發現：surgery_room.pos=(0,0,0) → robot在 (0.3596, -0.1741, 0.9493)
        # 所以 robotarm_base 在 USD 中的本地位置就是這個值
        # 要讓 robot 移動到目標位置，surgery_room 需要偏移：
        # surgery_room.pos = target_pos - robotarm_base_local_pos
        
        robotarm_base_local_x = 0.3596
        robotarm_base_local_y = -0.1741
        robotarm_base_local_z = 0.9493
        
        offset_x = pos_config['x'] - robotarm_base_local_x
        offset_y = pos_config['y'] - robotarm_base_local_y
        offset_z = pos_config['z'] - robotarm_base_local_z
        
        print(f"[INFO] Target robot position: ({pos_config['x']:.4f}, {pos_config['y']:.4f}, {pos_config['z']:.4f})")
        print(f"[INFO] Surgery room offset: ({offset_x:.4f}, {offset_y:.4f}, {offset_z:.4f})")
        env_cfg.scene.surgery_room.init_state.pos = (offset_x, offset_y, offset_z)
        
        # 創建環境
        env = ManagerBasedRLEnv(cfg=env_cfg)
        print(f"[OK] Environment created")
        
        # 立即重置並驗證位置
        env.reset()
        
        # 檢查實際位置
        robot = env.unwrapped.scene["robot"]
        actual_pos = robot.data.root_pos_w[0].cpu().numpy()
        print(f"[VERIFY] Actual robot world position: ({actual_pos[0]:.4f}, {actual_pos[1]:.4f}, {actual_pos[2]:.4f})")
        
        # 檢查是否位置真的改變了
        expected_x = pos_config['x']
        actual_x = actual_pos[0]
        diff = abs(actual_x - expected_x)
        
        if diff < 0.01:
            print(f"[SUCCESS] ✓ Position matches! Difference: {diff:.6f}m")
        else:
            print(f"[WARNING] ✗ Position mismatch! Expected X={expected_x:.4f}, Got X={actual_x:.4f}, Diff={diff:.4f}m")
        
        # 加載 policy
        obs_dim = env.unwrapped.observation_manager.group_obs_dim["policy"][0]
        action_dim = env.unwrapped.action_manager.total_action_dim
        
        policy = ActorCritic(
            num_actor_obs=obs_dim,
            num_critic_obs=obs_dim,
            num_actions=action_dim,
            actor_hidden_dims=[768, 512, 512, 256],
            critic_hidden_dims=[768, 512, 512, 256],
            activation='elu',
            init_noise_std=1.0
        ).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            policy.load_state_dict(checkpoint["model_state_dict"])
        else:
            policy.load_state_dict(checkpoint)
        policy.eval()
        
        # 運行 2 個 episode 測試
        print(f"\n[INFO] Running 2 episodes at {pos_config['name']}...")
        
        returns = []
        for ep in range(2):
            obs, _ = env.reset()
            done = False
            episode_return = 0.0
            step_count = 0
            
            while not done:
                with torch.no_grad():
                    obs_tensor = obs["policy"]
                    actions = policy.act(obs_tensor, deterministic=True)
                
                obs, rewards, dones, truncated, info = env.step(actions)
                episode_return += rewards[0].item()
                done = dones[0].item() or truncated[0].item()
                step_count += 1
            
            returns.append(episode_return)
            print(f"  Episode {ep+1}: Return={episode_return:.2f}, Steps={step_count}")
        
        mean_return = sum(returns) / len(returns)
        print(f"\n[RESULT] {pos_config['name']}: Mean Return = {mean_return:.2f}\n")
        
        # 關閉環境
        env.close()
        
        print(f"{'='*70}\n")
    
    print("[OK] All positions tested!")
    simulation_app.close()


if __name__ == "__main__":
    main()
