"""順序單環境多位置評估 - 每個位置創建新環境

測試不同 robot base 位置的 policy 表現
每0.5m測試一個位置
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sequential single-env multi-position evaluation")
parser.add_argument("--episodes", type=int, default=10, help="Episodes per position")
parser.add_argument("--checkpoint", type=str, 
                    default="logs/rsl_rl/tm5_reach_stable_v2/2026-01-05_14-03-14/model_2999.pt")
parser.add_argument("--save_file", type=str, default="robot_position_eval_results.json")
parser.add_argument("--start_x", type=float, default=-0.5, help="Start X position (m)")
parser.add_argument("--end_x", type=float, default=1.0, help="End X position (m)")
parser.add_argument("--spacing", type=float, default=0.5, help="Spacing between positions (m)")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_reach_surgery_room_cfg import (
    TM5ExtensionLinkOutReachEnvCfg
)
from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.modules import ActorCritic


# robotarm_base 在 USD 中的本地位置
ROBOTARM_BASE_LOCAL_X = 0.3596
ROBOTARM_BASE_LOCAL_Y = -0.1741
ROBOTARM_BASE_LOCAL_Z = 0.9493


def run_evaluation_at_position(env, policy, num_episodes, device, position_name):
    """在特定位置運行評估"""
    returns = []
    episode_lengths = []
    
    print(f"\n[INFO] Running {num_episodes} episodes at {position_name}...")
    
    for ep in range(num_episodes):
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
        episode_lengths.append(step_count)
        
        if (ep + 1) % 5 == 0:
            print(f"  Episode {ep+1}/{num_episodes}: Return={episode_return:.2f}, Steps={step_count}")
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    mean_length = np.mean(episode_lengths)
    
    print(f"\n[RESULT] {position_name}:")
    print(f"  Mean Return: {mean_return:.2f} ± {std_return:.2f}")
    print(f"  Mean Episode Length: {mean_length:.1f}")
    
    return {
        "position_name": position_name,
        "returns": returns,
        "mean_return": float(mean_return),
        "std_return": float(std_return),
        "episode_lengths": episode_lengths,
        "mean_length": float(mean_length),
    }


def main():
    # 計算測試位置
    num_positions = int((args_cli.end_x - args_cli.start_x) / args_cli.spacing) + 1
    x_positions = [args_cli.start_x + i * args_cli.spacing for i in range(num_positions)]
    
    print(f"\n{'='*70}")
    print(f"Sequential Single-Environment Multi-Position Evaluation")
    print(f"{'='*70}")
    print(f"  Test positions: {len(x_positions)} positions from {args_cli.start_x:.2f}m to {args_cli.end_x:.2f}m")
    print(f"  Spacing: {args_cli.spacing}m")
    print(f"  Episodes per position: {args_cli.episodes}")
    print(f"  Checkpoint: {args_cli.checkpoint}")
    print(f"  Positions: {[f'{x:.2f}m' for x in x_positions]}")
    print(f"{'='*70}\n")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args_cli.checkpoint)
    
    all_results = []
    
    for x_pos in x_positions:
        position_name = f"X={x_pos:.2f}m"
        
        print(f"\n{'='*70}")
        print(f"Testing {position_name}")
        print(f"{'='*70}")
        
        # 計算 surgery_room offset
        # offset = target_robot_pos - robotarm_base_local_pos
        offset_x = x_pos - ROBOTARM_BASE_LOCAL_X
        offset_y = 0.1 - ROBOTARM_BASE_LOCAL_Y
        offset_z = 0.0 - ROBOTARM_BASE_LOCAL_Z
        
        print(f"[INFO] Target robot position: ({x_pos:.4f}, 0.1, 0.0)")
        print(f"[INFO] Surgery room offset: ({offset_x:.4f}, {offset_y:.4f}, {offset_z:.4f})")
        
        # 創建環境配置
        env_cfg = TM5ExtensionLinkOutReachEnvCfg()
        env_cfg.scene.num_envs = 1
        env_cfg.commands.ee_pose.resampling_time_range = (1e10, 1e10)
        
        # 設置 surgery_room 位置
        env_cfg.scene.surgery_room.init_state.pos = (offset_x, offset_y, offset_z)
        
        # 創建環境
        env = ManagerBasedRLEnv(cfg=env_cfg)
        
        # Reset 並驗證位置
        env.reset()
        
        robot = env.unwrapped.scene["robot"]
        actual_pos = robot.data.root_pos_w[0].cpu().numpy()
        diff = abs(actual_pos[0] - x_pos)
        
        print(f"[VERIFY] Actual robot position: ({actual_pos[0]:.4f}, {actual_pos[1]:.4f}, {actual_pos[2]:.4f})")
        
        if diff < 0.01:
            print(f"[SUCCESS] ✓ Position matches! Difference: {diff:.6f}m")
        else:
            print(f"[WARNING] ✗ Position mismatch! Expected X={x_pos:.4f}, Got X={actual_pos[0]:.4f}, Diff={diff:.4f}m")
        
        # 創建並加載 policy
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
        
        # 運行評估
        results = run_evaluation_at_position(env, policy, args_cli.episodes, device, position_name)
        results["target_x"] = float(x_pos)
        results["actual_x"] = float(actual_pos[0])
        results["position_error"] = float(diff)
        
        all_results.append(results)
        
        # 關閉環境
        env.close()
        print(f"\n[OK] Environment closed for {position_name}")
    
    # 保存結果
    save_data = {
        "evaluation_info": {
            "timestamp": datetime.now().isoformat(),
            "checkpoint": str(checkpoint_path),
            "num_positions": len(x_positions),
            "episodes_per_position": args_cli.episodes,
            "x_start": args_cli.start_x,
            "x_end": args_cli.end_x,
            "spacing": args_cli.spacing,
        },
        "results": all_results,
    }
    
    save_path = Path(args_cli.save_file)
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Evaluation Summary")
    print(f"{'='*70}")
    print(f"\n{'Position':<12} {'Mean Return':<15} {'Std Return':<15} {'Mean Length':<15}")
    print("-" * 70)
    
    for result in all_results:
        print(f"{result['position_name']:<12} {result['mean_return']:<15.2f} {result['std_return']:<15.2f} {result['mean_length']:<15.1f}")
    
    print(f"\n[OK] Results saved to {save_path}")
    print(f"{'='*70}\n")
    
    simulation_app.close()


if __name__ == "__main__":
    main()
