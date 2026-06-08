"""Manual Multi-Position Evaluation (Sequential Testing)

順序測試不同的 robot base 位置（單一環境）。
使用已驗證的 USD 修改方法，與單環境 phantom 測試相同。
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import json

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Manual multi-position evaluation (sequential)")
parser.add_argument("--episodes", type=int, default=10, help="Episodes per position")
parser.add_argument("--checkpoint", type=str, 
                    default="logs/rsl_rl/tm5_reach_stable_v2/2026-01-05_14-03-14/model_2999.pt")
parser.add_argument("--save_file", type=str, default="manual_position_results.json")
parser.add_argument("--x_min", type=float, default=-0.52, help="Minimum X position")
parser.add_argument("--x_max", type=float, default=-0.16, help="Maximum X position")
parser.add_argument("--num_positions", type=int, default=4, help="Number of positions to test")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from pxr import Gf, UsdGeom
from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_reach_surgery_room_cfg import (
    TM5ExtensionLinkOutReachEnvCfg
)
from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.modules import ActorCritic


def set_robotarm_base_position(env, x, y, z):
    """設置 robot arm base 位置
    
    CRITICAL: 必須修改 articulation 的 init_state，而不是 USD parent！
    因為 reset() 時會從 init_state 重新設定 articulation 位置。
    """
    robot = env.unwrapped.scene["robot"]
    
    # 直接修改 articulation 的初始狀態（這會在 reset 時生效）
    robot.data.default_root_state[:, 0:3] = torch.tensor([x, y, z], device=env.device)
    
    print(f"[INFO] Robot init_state position set to ({x:.4f}, {y:.4f}, {z:.4f})")


def run_evaluation_at_position(env, policy, num_episodes, device, position_name):
    """在特定位置運行評估"""
    returns = []
    episode_lengths = []
    
    print(f"\n{'='*60}")
    print(f"Evaluating at position: {position_name}")
    print(f"{'='*60}")
    
    for episode_idx in range(num_episodes):
        obs, _ = env.reset()
        
        episode_return = 0.0
        episode_steps = 0
        done = False
        
        while not done:
            with torch.no_grad():
                obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
                actions = policy.act(obs_tensor, deterministic=True)
            
            obs, rewards, dones, truncated, info = env.step(actions)
            
            episode_return += rewards[0].item()
            episode_steps += 1
            done = dones[0].item() or truncated[0].item()
        
        returns.append(episode_return)
        episode_lengths.append(episode_steps)
        
        print(f"  Episode {episode_idx+1}/{num_episodes}: "
              f"Return = {episode_return:.2f}, Length = {episode_steps}")
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    mean_length = np.mean(episode_lengths)
    
    print(f"\nPosition {position_name} Results:")
    print(f"  Mean Return: {mean_return:.2f} ± {std_return:.2f}")
    print(f"  Mean Length: {mean_length:.1f}")
    
    return {
        "position": position_name,
        "x": float(position_name.split('=')[1]),
        "mean_return": mean_return,
        "std_return": std_return,
        "mean_length": mean_length,
        "all_returns": returns,
        "all_lengths": episode_lengths,
    }


def main():
    # Generate evenly spaced X positions
    x_positions = np.linspace(args_cli.x_min, args_cli.x_max, args_cli.num_positions)
    
    print("="*60)
    print("Manual Multi-Position Evaluation (Sequential)")
    print("="*60)
    print(f"Configuration:")
    print(f"  Positions to test: {args_cli.num_positions}")
    print(f"  X range: [{args_cli.x_min:.2f}, {args_cli.x_max:.2f}]")
    print(f"  X positions: {[f'{x:.2f}' for x in x_positions]}")
    print(f"  Episodes per position: {args_cli.episodes}")
    print(f"  Total evaluations: {args_cli.num_positions * args_cli.episodes}")
    print(f"  Checkpoint: {args_cli.checkpoint}")
    print("="*60 + "\n")
    
    # Create environment (single env)
    env_cfg = TM5ExtensionLinkOutReachEnvCfg()
    env_cfg.scene.num_envs = 1
    
    # Disable command resampling - keep target fixed
    env_cfg.commands.ee_pose.resampling_time_range = (1e10, 1e10)
    
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[OK] Environment created (device: {device})")
    
    # Create and load policy
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
    
    checkpoint_path = Path(args_cli.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        policy.load_state_dict(checkpoint["model_state_dict"])
        print(f"[OK] Loaded checkpoint (iteration: {checkpoint.get('iter', 'unknown')})")
    else:
        policy.load_state_dict(checkpoint)
        print(f"[OK] Loaded checkpoint")
    
    policy.eval()
    
    # Evaluate at each position
    all_results = []
    
    for x_pos in x_positions:
        position_name = f"X={x_pos:.2f}"
        
        print(f"\n{'='*60}")
        print(f"Setting position: {position_name}")
        print(f"{'='*60}")
        
        # Set robot base position in USD
        set_robotarm_base_position(env, x_pos, 0.1, 0.0)
        
        # Reset environment to load from USD
        env.reset()
        
        # Verify actual position
        robot = env.unwrapped.scene["robot"]
        actual_pos = robot.data.root_pos_w[0].cpu().numpy()
        print(f"[VERIFY] Actual robot position: ({actual_pos[0]:.4f}, {actual_pos[1]:.4f}, {actual_pos[2]:.4f})")
        
        # Run evaluation at this position
        results = run_evaluation_at_position(
            env, policy, args_cli.episodes, device, position_name
        )
        all_results.append(results)
    
    # Display summary
    print(f"\n{'='*60}")
    print(f"SUMMARY - All Positions")
    print(f"{'='*60}")
    for result in all_results:
        print(f"{result['position']:12s}: "
              f"Return = {result['mean_return']:6.2f} ± {result['std_return']:5.2f}, "
              f"Length = {result['mean_length']:5.1f}")
    print(f"{'='*60}\n")
    
    # Save results
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(checkpoint_path),
        "x_positions": x_positions.tolist(),
        "episodes_per_position": args_cli.episodes,
        "total_evaluations": args_cli.num_positions * args_cli.episodes,
        "results": all_results,
    }
    
    save_path = Path(args_cli.save_file)
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"[OK] Results saved to {save_path}")
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
