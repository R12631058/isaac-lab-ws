"""
Visualize trained model performing reach task at different base positions

Shows multiple environments with different robotarm_base X positions executing simultaneously.
Press Ctrl+C to quit the visualization.
"""

import argparse
import torch
from isaaclab.app import AppLauncher

# Parse command line arguments
parser = argparse.ArgumentParser(description="Visualize reach task at different base positions")
parser.add_argument("--num_envs", type=int, default=5, help="Number of environments (different positions)")
parser.add_argument("--checkpoint", type=str, default="logs/rsl_rl/tm5_reach_stable_v2/2026-01-05_14-03-14/model_2999.pt", help="Model checkpoint path")
parser.add_argument("--start_x", type=float, default=-0.6596245218672151, help="Starting X position")
parser.add_argument("--x_step", type=float, default=0.01, help="X position increment between environments")
args_cli = parser.parse_args()

# Launch Isaac Sim (NOT headless - we want to see it!)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after Isaac Sim is launched
from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_reach_surgery_room_cfg import TM5ExtensionLinkOutReachEnvCfg_PLAY
from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.modules import ActorCritic
from pathlib import Path
import numpy as np


def set_robot_positions(env: ManagerBasedRLEnv, x_positions: list):
    """Set different robotarm_base X positions using root state API"""
    base_y = 0.2693808034227855
    base_z = -0.026487045595559477
    
    # Get the robot articulation
    robot = env.unwrapped.scene["robot"]
    
    # Get current root states (position + orientation)
    root_state = robot.data.root_state_w.clone()
    
    # Modify X position for each environment
    for env_idx in range(env.unwrapped.num_envs):
        if env_idx < len(x_positions):
            # Set new position: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
            root_state[env_idx, 0] = x_positions[env_idx]  # X
            root_state[env_idx, 1] = base_y  # Y
            root_state[env_idx, 2] = base_z  # Z
            # Keep existing orientation (indices 3-6)
            # Zero out velocities
            root_state[env_idx, 7:] = 0.0
    
    # Write the new root states
    robot.write_root_state_to_sim(root_state)
    
    print("[INFO] Applied new root positions to robot articulations")


def main():
    # Check checkpoint exists
    checkpoint_path = Path(args_cli.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Generate X positions for each environment
    x_positions = [args_cli.start_x + i * args_cli.x_step for i in range(args_cli.num_envs)]
    
    print("\n" + "="*60)
    print("Multi-Position Reach Task Visualization")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Number of environments: {args_cli.num_envs}")
    print(f"X positions:")
    for i, x in enumerate(x_positions):
        print(f"  Env {i}: X = {x:.4f}")
    print("="*60 + "\n")
    
    # Create environment
    env_cfg = TM5ExtensionLinkOutReachEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[OK] Environment created (device: {device})")
    
    # Create policy
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
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        policy.load_state_dict(checkpoint["model_state_dict"])
        print(f"[OK] Loaded checkpoint (iteration: {checkpoint.get('iter', 'unknown')})")
    else:
        policy.load_state_dict(checkpoint)
        print(f"[OK] Loaded checkpoint")
    
    policy.eval()
    
    # Reset environment first
    print("\n[INFO] Resetting environment...")
    obs, _ = env.reset()
    
    # Now set different positions for each robot
    print("\n[INFO] Setting different robotarm_base positions...")
    set_robot_positions(env, x_positions)
    
    # Step once to apply the positions
    with torch.no_grad():
        obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
        actions = policy.act(obs_tensor, deterministic=True)
    obs, _, _, _, _ = env.step(actions)
    
    # Verify positions
    robot = env.unwrapped.scene["robot"]
    actual_positions = robot.data.root_pos_w[:, 0].cpu().numpy()
    print("\n[INFO] Verification - Actual X positions:")
    for i, x in enumerate(actual_positions):
        expected = x_positions[i] if i < len(x_positions) else x_positions[0]
        diff = abs(x - expected)
        status = "OK" if diff < 0.001 else "WARN"
        print(f"  Env {i}: X = {x:.4f} (expected {expected:.4f}, diff={diff:.6f}) [{status}]")
    
    print("\n[OK] Setup complete\n")
    
    # Run simulation
    print("="*60)
    print("Running visualization - Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    step_count = 0
    episode_count = [0] * args_cli.num_envs
    episode_returns = [[0.0] for _ in range(args_cli.num_envs)]
    current_returns = [0.0] * args_cli.num_envs
    
    try:
        while simulation_app.is_running():
            # Get actions from policy
            with torch.no_grad():
                obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
                actions = policy.act(obs_tensor, deterministic=True)
            
            # Step environment
            obs, rewards, dones, truncated, info = env.step(actions)
            
            # Update returns
            for env_idx in range(args_cli.num_envs):
                current_returns[env_idx] += rewards[env_idx].item()
            
            # Check for episode completions
            finished = dones | truncated
            if finished.any():
                for env_idx in range(args_cli.num_envs):
                    if finished[env_idx]:
                        episode_count[env_idx] += 1
                        episode_returns[env_idx].append(current_returns[env_idx])
                        
                        mean_return = np.mean(episode_returns[env_idx])
                        print(f"Env {env_idx} (X={x_positions[env_idx]:.4f}): "
                              f"Episode {episode_count[env_idx]:3d} | "
                              f"Return: {current_returns[env_idx]:7.2f} | "
                              f"Mean: {mean_return:7.2f}")
                        
                        current_returns[env_idx] = 0.0
                
                # Re-apply positions after reset
                set_robot_positions(env, x_positions)
            
            step_count += 1
            
            # Print summary every 1000 steps
            if step_count % 1000 == 0:
                print(f"\n--- Step {step_count} Summary ---")
                for env_idx in range(args_cli.num_envs):
                    if episode_count[env_idx] > 0:
                        mean_return = np.mean(episode_returns[env_idx])
                        print(f"  Env {env_idx}: {episode_count[env_idx]} episodes, mean return: {mean_return:.2f}")
                print()
    
    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted by user")
    
    finally:
        print("\n" + "="*60)
        print("Final Statistics")
        print("="*60)
        for env_idx in range(args_cli.num_envs):
            if episode_count[env_idx] > 0:
                mean_return = np.mean(episode_returns[env_idx])
                std_return = np.std(episode_returns[env_idx])
                print(f"Env {env_idx} (X={x_positions[env_idx]:.4f}): "
                      f"{episode_count[env_idx]} episodes | "
                      f"Mean: {mean_return:.2f} +/- {std_return:.2f}")
        print("="*60 + "\n")
        
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
