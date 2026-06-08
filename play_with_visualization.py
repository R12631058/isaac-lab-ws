"""
用於播放針尖 Reach 任務的訓練模型，並顯示目標範圍可視化

使用方法:
    python play_with_visualization.py --task Isaac-Reach-TM5-NeedleTip-Play-v0 --num_envs 4 --checkpoint <path>
"""

import argparse
import torch

from isaaclab.app import AppLauncher

# 添加參數
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Reach-TM5-NeedleTip-Play-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 啟動模擬器
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 導入模組
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils

# 解析環境配置
env_cfg = parse_env_cfg(
    args_cli.task,
    device="cuda:0",
    num_envs=args_cli.num_envs,
)

# 創建環境
env = ManagerBasedRLEnv(cfg=env_cfg)

# ============================================
# 創建目標範圍可視化 - 8個角點綠色球體
# ============================================
# 病患目標範圍 (相對於機械臂底座)
# X: -0.500 ~ -0.100m, Y: -0.750 ~ -0.450m, Z: 0.150 ~ 0.250m
corners_relative = torch.tensor([
    [-0.500, -0.750, 0.150],  # 角點 0
    [-0.100, -0.750, 0.150],  # 角點 1
    [-0.500, -0.450, 0.150],  # 角點 2
    [-0.100, -0.450, 0.150],  # 角點 3
    [-0.500, -0.750, 0.250],  # 角點 4
    [-0.100, -0.750, 0.250],  # 角點 5
    [-0.500, -0.450, 0.250],  # 角點 6
    [-0.100, -0.450, 0.250],  # 角點 7
], device=env.device)

# 創建可視化標記配置
marker_cfg = VisualizationMarkersCfg(
    prim_path="/Visuals/TargetRangeCorners",
    markers={
        "corner": sim_utils.SphereCfg(
            radius=0.015,  # 1.5cm 綠色球體
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
    },
)

# 創建標記實例
range_markers = VisualizationMarkers(marker_cfg)

print("\n" + "="*80)
print("🎯 針尖 Reach 任務 - 帶目標範圍可視化")
print("="*80)
print(f"任務: {args_cli.task}")
print(f"環境數量: {args_cli.num_envs}")
print(f"模型: {args_cli.checkpoint}")
print("\n📦 目標範圍 (相對於機械臂底座):")
print(f"   X: -0.500 ~ -0.100 m (40cm)")
print(f"   Y: -0.750 ~ -0.450 m (30cm)")
print(f"   Z: 0.150 ~ 0.250 m (10cm)")
print("\n🟢 綠色球體 = 目標範圍的 8 個角點")
print("="*80 + "\n")

# 載入模型
if args_cli.checkpoint:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    
    # 包裝環境
    env = RslRlVecEnvWrapper(env)
    
    # 創建 runner 配置
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    runner_cfg = agent_cfg
    runner_cfg.device = env.unwrapped.device
    
    # 創建 runner
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device=runner_cfg.device)
    
    # 載入模型
    print(f"[INFO]: Loading model checkpoint from: {args_cli.checkpoint}")
    runner.load(args_cli.checkpoint)
    
    print("[INFO]: Model loaded successfully!")
    
    # 獲取推理策略
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    
    print("\n按 Ctrl+C 結束\n")
    
    # 運行模擬
    obs, _ = env.get_observations()
    while simulation_app.is_running():
        # 更新可視化標記位置
        robot = env.unwrapped.scene["robot"]
        base_pos_w = robot.data.root_pos_w  # (num_envs, 3)
        
        # 將相對座標轉換為世界座標
        corners_all_envs = corners_relative.unsqueeze(0).repeat(args_cli.num_envs, 1, 1)  # (num_envs, 8, 3)
        corners_world = corners_all_envs + base_pos_w.unsqueeze(1)  # (num_envs, 8, 3)
        corners_world_flat = corners_world.reshape(-1, 3)  # (num_envs * 8, 3)
        
        # 顯示標記
        range_markers.visualize(translations=corners_world_flat)
        
        # 執行推理
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
else:
    print("[INFO]: No checkpoint provided, running with random actions")
    print("\n按 Ctrl+C 結束\n")
    
    # 運行模擬（隨機動作）
    while simulation_app.is_running():
        # 更新可視化標記位置
        robot = env.scene["robot"]
        base_pos_w = robot.data.root_pos_w  # (num_envs, 3)
        
        # 將相對座標轉換為世界座標
        corners_all_envs = corners_relative.unsqueeze(0).repeat(args_cli.num_envs, 1, 1)  # (num_envs, 8, 3)
        corners_world = corners_all_envs + base_pos_w.unsqueeze(1)  # (num_envs, 8, 3)
        corners_world_flat = corners_world.reshape(-1, 3)  # (num_envs * 8, 3)
        
        # 顯示標記
        range_markers.visualize(translations=corners_world_flat)
        
        # 執行隨機動作
        actions = 2.0 * torch.rand(env.action_manager.action.shape, device=env.device) - 1.0
        env.step(actions)

# 關閉
simulation_app.close()
