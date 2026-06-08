"""
評估已訓練的模型在不同 robot arm base 位置下的表現

使用已訓練的模型 model_2999.pt 在不同 base 位置測試 reach task 可以獲得多少分數：
- 從 X = -0.6596 開始，每次增加 0.01
- 每個位置執行多個 episode
- 統計成功率、平均獎勵、平均 episode 長度

使用方法：
  python scripts/eval_variable_base_positions.py --num_positions 10 --episodes_per_position 20

"""

import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import json

from isaaclab.app import AppLauncher

# 解析命令行參數
parser = argparse.ArgumentParser(description="評估不同 base 位置的表現")
parser.add_argument("--headless", action="store_true", help="無頭模式運行")
parser.add_argument("--num_envs", type=int, default=1, help="並行環境數量")
parser.add_argument("--checkpoint", type=str, default="logs/rsl_rl/tm5_reach_stable_v2/2026-01-05_14-03-14/model_2999.pt", help="模型路徑")
parser.add_argument("--start_x", type=float, default=-0.6596245218672151, help="起始 X 位置")
parser.add_argument("--x_step", type=float, default=0.01, help="X 位置遞增量")
parser.add_argument("--num_positions", type=int, default=5, help="要測試的位置數量")
parser.add_argument("--episodes_per_position", type=int, default=10, help="每個位置的 episode 數量")

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import UsdGeom, Gf
from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_reach_surgery_room_cfg import TM5ExtensionLinkOutReachEnvCfg_PLAY
from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.modules import ActorCritic


def set_robotarm_base_position(env: ManagerBasedRLEnv, x_position: float):
    """為所有環境設置相同的 robotarm_base X 位置"""
    base_y = 0.2693808034227855
    base_z = -0.026487045595559477
    
    stage = env.unwrapped.scene.stage
    
    for env_idx in range(env.unwrapped.num_envs):
        prim_path = f"/World/envs/env_{env_idx}/Root/robotarm_base"
        base_prim = stage.GetPrimAtPath(prim_path)
        
        if not base_prim.IsValid():
            print(f"[WARN] 環境 {env_idx} 中找不到 {prim_path}")
            continue
        
        xformable = UsdGeom.Xformable(base_prim)
        translate_ops = [op for op in xformable.GetOrderedXformOps() 
                        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
        
        if translate_ops:
            translate_op = translate_ops[0]
        else:
            translate_op = xformable.AddTranslateOp()
        
        translate_op.Set(Gf.Vec3d(x_position, base_y, base_z))


def evaluate_at_position(env: ManagerBasedRLEnv, policy: ActorCritic, x_position: float, num_episodes: int, device: torch.device):
    """在指定 base 位置評估模型表現"""
    print(f"\n[OK] 開始評估 X = {x_position:.4f}")
    
    # 設置所有環境的位置為同一個 X 坐標
    set_robotarm_base_position(env, x_position)
    
    # 重置環境以應用新位置
    obs, _ = env.reset()
    
    episode_returns = []
    episode_lengths = []
    current_returns = np.zeros(env.unwrapped.num_envs)
    current_lengths = np.zeros(env.unwrapped.num_envs, dtype=int)
    completed_episodes = 0
    
    max_steps_per_run = 10000
    steps = 0
    
    while completed_episodes < num_episodes and steps < max_steps_per_run:
        # 使用確定性策略獲取動作
        with torch.no_grad():
            obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
            actions = policy.act(obs_tensor, deterministic=True)
        
        # 執行動作
        obs, rewards, dones, truncated, info = env.step(actions)
        
        # 累積回報和長度
        current_returns += rewards.cpu().numpy()
        current_lengths += 1
        steps += 1
        
        # 檢測episode完成
        finished = (dones | truncated).cpu().numpy()
        
        for env_idx in range(env.unwrapped.num_envs):
            if finished[env_idx] and completed_episodes < num_episodes:
                episode_returns.append(current_returns[env_idx])
                episode_lengths.append(current_lengths[env_idx])
                completed_episodes += 1
                
                # 重置該環境的計數器
                current_returns[env_idx] = 0
                current_lengths[env_idx] = 0
    
    # 計算統計資料
    if len(episode_returns) > 0:
        mean_return = np.mean(episode_returns)
        std_return = np.std(episode_returns)
        mean_length = np.mean(episode_lengths)
        # 假設成功定義為 return > 某個閾值
        success_threshold = 100.0  # 可以根據實際情況調整
        success_rate = np.sum(np.array(episode_returns) > success_threshold) / len(episode_returns)
    else:
        mean_return = 0.0
        std_return = 0.0
        mean_length = 0.0
        success_rate = 0.0
    
    result = {
        "x_position": x_position,
        "num_episodes": len(episode_returns),
        "mean_return": float(mean_return),
        "std_return": float(std_return),
        "mean_episode_length": float(mean_length),
        "success_rate": float(success_rate),
    }
    
    print(f"[OK] X={x_position:.4f}: 平均回報={mean_return:.2f}+-{std_return:.2f}, 成功率={success_rate*100:.1f}%, 平均長度={mean_length:.1f}")
    
    return result


def main():
    # 檢查 checkpoint 是否存在
    checkpoint_path = Path(args_cli.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")
    
    print(f"\n{'='*60}")
    print(f"評估設置")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"起始 X 位置: {args_cli.start_x}")
    print(f"位置數量: {args_cli.num_positions}")
    print(f"每個位置 episodes: {args_cli.episodes_per_position}")
    print(f"X 遞增步長: {args_cli.x_step}")
    print(f"{'='*60}\n")
    
    # 創建環境
    env_cfg = TM5ExtensionLinkOutReachEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[OK] 環境創建成功 (device: {device})")
    
    # 創建策略模型
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
    
    # 載入 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        policy.load_state_dict(checkpoint["model_state_dict"])
        print(f"[OK] 載入訓練 checkpoint (iteration: {checkpoint.get('iter', 'unknown')})")
    else:
        policy.load_state_dict(checkpoint)
        print(f"[OK] 載入 checkpoint")
    
    policy.eval()
    
    # 對每個位置進行評估
    all_results = []
    
    for pos_idx in range(args_cli.num_positions):
        x_position = args_cli.start_x + pos_idx * args_cli.x_step
        
        results = evaluate_at_position(
            env=env,
            policy=policy,
            x_position=x_position,
            num_episodes=args_cli.episodes_per_position,
            device=device
        )
        
        all_results.append(results)
    
    # 保存結果到 JSON
    output_path = Path("eval_results.json")
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(checkpoint_path),
        "start_x": args_cli.start_x,
        "x_step": args_cli.x_step,
        "num_positions": args_cli.num_positions,
        "episodes_per_position": args_cli.episodes_per_position,
        "results": all_results,
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] 評估完成！結果已保存到 {output_path}")
    
    # 打印摘要
    print(f"\n{'='*60}")
    print(f"評估摘要")
    print(f"{'='*60}")
    for result in all_results:
        print(f"X={result['x_position']:.4f}: 平均回報={result['mean_return']:7.2f}+-{result['std_return']:6.2f}, 成功率={result['success_rate']*100:5.1f}%")
    print(f"{'='*60}\n")
    
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
