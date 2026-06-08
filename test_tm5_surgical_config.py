"""
快速測試 TM5 Surgical 配置是否能正確執行

執行方式:
.\isaaclab.bat -p scripts\isaaclab_ws\test_tm5_surgical_config.py --task Isaac-Reach-TM5-Surgical-Debug-v0
"""

import argparse

from isaaclab.app import AppLauncher

# 參數解析
parser = argparse.ArgumentParser(description="Test TM5 Surgical Config")
parser.add_argument("--task", type=str, default="Isaac-Reach-TM5-Surgical-Debug-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 啟動 app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 導入模組
import torch
import gymnasium as gym


def main():
    """測試配置"""
    
    print("=" * 80)
    print(f"測試任務: {args_cli.task}")
    print("=" * 80)
    
    # 建立環境
    print("\n[1/4] 建立環境...")
    env = gym.make(args_cli.task, num_envs=args_cli.num_envs)
    print(f"✅ 環境建立成功!")
    print(f"   - 環境數量: {env.num_envs}")
    print(f"   - 觀察空間: {env.observation_space}")
    print(f"   - 動作空間: {env.action_space}")
    
    # 重置環境
    print("\n[2/4] 重置環境...")
    obs, _ = env.reset()
    print(f"✅ 環境重置成功!")
    print(f"   - 觀察形狀: {obs.shape}")
    
    # 執行隨機動作
    print("\n[3/4] 執行 100 步隨機動作測試...")
    
    distances = []
    rewards_list = []
    
    for step in range(100):
        # 隨機動作
        actions = torch.randn((args_cli.num_envs, env.action_space.shape[0]), device=env.unwrapped.device) * 0.5
        
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # 計算距離
        robot = env.unwrapped.scene["robot"]
        flange_idx = robot.body_names.index("flange")
        ee_pos = robot.data.body_state_w[:, flange_idx, :3]
        
        target_pos = env.unwrapped.command_manager.get_command("ee_pose")[:, :3]
        distance = torch.norm(target_pos - ee_pos, dim=-1)
        
        distances.extend(distance.cpu().numpy().tolist())
        rewards_list.extend(rewards.cpu().numpy().tolist())
        
        if step % 20 == 0:
            avg_dist = sum(distances[-args_cli.num_envs:]) / args_cli.num_envs if distances else 0
            avg_reward = sum(rewards_list[-args_cli.num_envs:]) / args_cli.num_envs if rewards_list else 0
            print(f"   Step {step:3d}: 平均距離={avg_dist:.3f}m, 平均獎勵={avg_reward:.2f}")
    
    print("\n[4/4] 分析結果...")
    
    import numpy as np
    distances = np.array(distances)
    rewards_list = np.array(rewards_list)
    
    print(f"\n📊 統計數據:")
    print(f"   - 平均距離: {np.mean(distances):.3f}m")
    print(f"   - 最小距離: {np.min(distances):.3f}m")
    print(f"   - 最大距離: {np.max(distances):.3f}m")
    print(f"   - 平均獎勵: {np.mean(rewards_list):.2f}")
    print(f"   - 總獎勵: {np.sum(rewards_list):.2f}")
    
    success_rate = np.sum(distances < 0.05) / len(distances) * 100
    print(f"\n✅ 成功率 (<5cm): {success_rate:.1f}%")
    
    # 檢查配置
    print(f"\n⚙️  環境配置:")
    env_cfg = env.unwrapped.cfg
    print(f"   - Episode 長度: {env_cfg.episode_length_s}s")
    print(f"   - Decimation: {env_cfg.decimation}")
    
    # 檢查動作配置
    action_cfg = env_cfg.actions.arm_action
    print(f"   - 動作範圍 (scale): {action_cfg.scale}")
    print(f"   - 關節名稱: {action_cfg.joint_names}")
    
    # 檢查獎勵配置
    print(f"\n🎁 獎勵權重:")
    for reward_name, reward_cfg in env_cfg.rewards.items():
        print(f"   - {reward_name}: {reward_cfg.weight}")
    
    print("\n" + "=" * 80)
    print("✅ 測試完成! 配置運作正常。")
    print("\n💡 建議:")
    if np.mean(distances) > 0.5:
        print("   隨機動作的距離很大是正常的。")
        print("   現在可以開始訓練:")
        print(f"   .\\isaaclab.bat -p scripts\\reinforcement_learning\\rsl_rl\\train.py --task {args_cli.task} --num_envs 256 --max_iterations 1000")
    
    print("=" * 80)
    
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
