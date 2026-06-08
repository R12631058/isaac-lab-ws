"""
TM5 Reach 診斷腳本 - 分析為什麼無法抵達目標

執行方式:
C:\\Users\\RMML\\anaconda3\\envs\\env_isaaclab\\python.exe scripts\\isaaclab_ws\\diagnosis_tm5_reach.py --checkpoint logs\\rsl_rl\\tm5_reach_stable_v2\\2025-07-10_15-40-24\\model_2999.pt
"""

import argparse
import torch
import numpy as np

from isaaclab.app import AppLauncher

# 參數解析
parser = argparse.ArgumentParser(description="Diagnose TM5 Reach Performance")
parser.add_argument("--task", type=str, default="Isaac-Reach-TM5-Stable-v1", help="Task name")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 啟動 app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 導入模組
from isaaclab.envs import ManagerBasedRLEnv
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner


def main():
    """診斷主函數"""
    
    print("=" * 80)
    print("TM5 Reach 性能診斷工具")
    print("=" * 80)
    
    # 建立環境
    print(f"\n[1/5] 載入環境: {args_cli.task}")
    env = gym.make(args_cli.task, num_envs=args_cli.num_envs)
    
    # 載入模型
    print(f"\n[2/5] 載入模型: {args_cli.checkpoint}")
    agent_cfg = env.unwrapped.cfg.to_dict()["rl_device"]
    
    # 簡單載入 checkpoint
    checkpoint = torch.load(args_cli.checkpoint)
    
    # 取得 policy 網路參數
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
    elif 'ac_state_dict' in checkpoint:
        model_state = checkpoint['ac_state_dict']
    else:
        print("❌ 找不到模型參數")
        return
    
    print(f"✅ 模型載入成功")
    print(f"   - 訓練 Iterations: {checkpoint.get('iter', 'N/A')}")
    
    # 重置環境
    print(f"\n[3/5] 重置環境...")
    obs, _ = env.reset()
    
    # 收集診斷資訊
    print(f"\n[4/5] 執行診斷 (100 steps)...")
    
    distances = []
    joint_changes = []
    actions_magnitude = []
    
    previous_joints = env.unwrapped.scene["robot"].data.joint_pos.clone()
    
    for step in range(100):
        # 隨機動作測試
        actions = torch.randn((args_cli.num_envs, env.action_space.shape[1]), device=env.device) * 0.1
        
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # 計算末端執行器到目標的距離
        robot = env.unwrapped.scene["robot"]
        flange_idx = robot.body_names.index("flange")
        ee_pos = robot.data.body_state_w[:, flange_idx, :3]
        
        # 取得目標位置
        target_pos = env.unwrapped.command_manager.get_command("ee_pose")[:, :3]
        
        distance = torch.norm(target_pos - ee_pos, dim=-1)
        distances.extend(distance.cpu().numpy())
        
        # 計算關節變化
        current_joints = robot.data.joint_pos
        joint_change = torch.norm(current_joints - previous_joints, dim=-1)
        joint_changes.extend(joint_change.cpu().numpy())
        previous_joints = current_joints.clone()
        
        # 動作大小
        action_mag = torch.norm(actions, dim=-1)
        actions_magnitude.extend(action_mag.cpu().numpy())
    
    # 分析結果
    print(f"\n[5/5] 診斷結果:")
    print("=" * 80)
    
    distances = np.array(distances)
    joint_changes = np.array(joint_changes)
    actions_magnitude = np.array(actions_magnitude)
    
    print(f"\n📊 末端執行器到目標距離 (m):")
    print(f"   - 平均距離: {np.mean(distances):.3f}m")
    print(f"   - 最小距離: {np.min(distances):.3f}m")
    print(f"   - 最大距離: {np.max(distances):.3f}m")
    print(f"   - 標準差: {np.std(distances):.3f}m")
    
    success_rate = np.sum(distances < 0.05) / len(distances) * 100
    print(f"\n✅ 成功率 (< 5cm): {success_rate:.1f}%")
    
    print(f"\n🔧 關節運動分析:")
    print(f"   - 平均關節變化: {np.mean(joint_changes):.4f} rad")
    print(f"   - 最大關節變化: {np.max(joint_changes):.4f} rad")
    
    print(f"\n🎮 動作分析:")
    print(f"   - 平均動作大小: {np.mean(actions_magnitude):.4f}")
    print(f"   - 最大動作大小: {np.max(actions_magnitude):.4f}")
    
    # 診斷建議
    print(f"\n💡 診斷建議:")
    if np.mean(distances) > 0.3:
        print("   ⚠️  平均距離過大 (>30cm) - 模型無法有效接近目標")
        print("   建議:")
        print("      1. 增加動作範圍 (scale: 0.5 -> 1.0)")
        print("      2. 降低動作平滑度懲罰")
        print("      3. 增加位置追蹤獎勵權重")
        print("      4. 檢查目標範圍是否在機械臂可達空間內")
    
    if np.mean(joint_changes) < 0.01:
        print("   ⚠️  關節移動過小 - 模型學會靜止不動")
        print("   建議:")
        print("      1. 降低關節速度懲罰")
        print("      2. 增加探索噪音")
        print("      3. 使用更大的動作範圍")
    
    if success_rate < 10:
        print("   ❌ 成功率極低 - 需要重新訓練")
        print("   建議使用新配置:")
        print("      .\isaaclab.bat -p scripts\\reinforcement_learning\\rsl_rl\\train.py --task Isaac-Reach-TM5-Surgical-v0 --num_envs 512")
    
    print("\n" + "=" * 80)
    
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
