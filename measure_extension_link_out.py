# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""測量 extension_link_out.usd 場景中的關鍵位置

這個腳本用於：
1. 測量初始狀態下 needle/needle_tip 的位置
2. 顯示 Target 範圍與針尖初始位置的關係
3. 診斷 model_4999.pt 訓練失敗的原因
"""

import argparse
import torch
import math

from isaaclab.app import AppLauncher

# 創建 argument parser
parser = argparse.ArgumentParser(description="測量 extension_link_out.usd 場景")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 啟動 Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 現在可以導入 Isaac Lab 模組
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.math import quat_rotate


def main():
    """主函數"""
    
    # ============================================
    # 1. 設定仿真
    # ============================================
    sim_cfg = sim_utils.SimulationCfg(dt=1.0/60.0, device="cuda:0")
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.0, 3.0, 3.0], target=[0.0, 0.0, 0.5])
    
    # ============================================
    # 2. 載入 USD 場景
    # ============================================
    usd_path = "C:\\Nick\\surgery_team\\surgery_team\\USD\\isaaclab\\extension_link_out.usd"
    
    print("\n" + "="*60)
    print("📂 載入場景: extension_link_out.usd")
    print("="*60)
    
    # 載入整個 USD 作為場景
    prim_utils = sim_utils.UsdFileCfg(usd_path=usd_path)
    prim_utils.func("/World/Root", prim_utils, translation=(0.0, 0.0, 0.0))
    
    # ============================================
    # 3. 配置機器人（使用 USD 中現有的）
    # ============================================
    # 與 model_4999.pt 訓練時相同的初始關節位置
    EXTENSION_LINK_OUT_JOINT_POS = {
        "joint_1": -1.656318,  # -94.89°
        "joint_2": 0.757474,   # 43.40°
        "joint_3": 1.003564,   # 57.50°
        "joint_4": 1.368337,   # 78.40°
        "joint_5": -0.062832,  # -3.60°
        "joint_6": -1.570797,  # -90.00°
    }
    
    robot_cfg = ArticulationCfg(
        prim_path="/World/Root/robotarm_base/robotarm_base/tm5_700",
        spawn=None,  # 使用 USD 中現有的
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos=EXTENSION_LINK_OUT_JOINT_POS,
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint_[1-6]"],
                effort_limit=200.0,
                velocity_limit=2.0,
                stiffness=200.0,
                damping=20.0,
            ),
        },
    )
    
    robot = Articulation(robot_cfg)
    
    # ============================================
    # 4. 初始化仿真
    # ============================================
    sim.reset()
    robot.reset()
    
    # 執行幾步仿真讓場景穩定
    for _ in range(10):
        sim.step()
        robot.update(sim.get_physics_dt())
    
    # ============================================
    # 5. 測量關鍵位置
    # ============================================
    print("\n" + "="*60)
    print("📍 位置測量結果")
    print("="*60)
    
    # 取得機器人 base 位置
    robot_base_pos = robot.data.root_pos_w[0].cpu().numpy()
    print(f"\n🤖 Robot Base 世界座標:")
    print(f"   X: {robot_base_pos[0]:.4f} m")
    print(f"   Y: {robot_base_pos[1]:.4f} m")
    print(f"   Z: {robot_base_pos[2]:.4f} m")
    
    # 取得所有 body 名稱
    print(f"\n📋 機器人 Body 列表:")
    for i, name in enumerate(robot.body_names):
        print(f"   [{i}] {name}")
    
    # 尋找 needle body
    needle_idx = None
    for i, name in enumerate(robot.body_names):
        if "needle" in name.lower():
            needle_idx = i
            break
    
    if needle_idx is not None:
        print(f"\n✅ 找到 needle body: index={needle_idx}, name={robot.body_names[needle_idx]}")
        
        # 取得 needle body 位置和姿態
        needle_pos_w = robot.data.body_state_w[0, needle_idx, :3].cpu()
        needle_quat_w = robot.data.body_state_w[0, needle_idx, 3:7].cpu()
        
        print(f"\n📌 Needle Body 世界座標:")
        print(f"   X: {needle_pos_w[0]:.4f} m")
        print(f"   Y: {needle_pos_w[1]:.4f} m")
        print(f"   Z: {needle_pos_w[2]:.4f} m")
        
        # 計算針尖位置（使用 tip_offset）
        TIP_OFFSET = (-0.001153, 0.001055, 0.5)  # 與訓練配置相同
        tip_offset_local = torch.tensor([TIP_OFFSET], dtype=torch.float32)
        tip_offset_world = quat_rotate(needle_quat_w.unsqueeze(0), tip_offset_local)
        tip_pos_w = needle_pos_w + tip_offset_world.squeeze()
        
        print(f"\n🎯 針尖 (needle_tip) 世界座標 (使用 tip_offset):")
        print(f"   X: {tip_pos_w[0]:.4f} m")
        print(f"   Y: {tip_pos_w[1]:.4f} m")
        print(f"   Z: {tip_pos_w[2]:.4f} m")
        
        # 計算相對於 robot base 的座標
        tip_pos_rel = tip_pos_w.numpy() - robot_base_pos
        print(f"\n📐 針尖相對於 Robot Base 座標:")
        print(f"   X: {tip_pos_rel[0]:.4f} m")
        print(f"   Y: {tip_pos_rel[1]:.4f} m")
        print(f"   Z: {tip_pos_rel[2]:.4f} m")
        
        # ============================================
        # 6. 分析 Target 範圍與針尖初始位置
        # ============================================
        print("\n" + "="*60)
        print("📊 Target 範圍分析 (來自 model_4999.pt 訓練配置)")
        print("="*60)
        
        # model_4999.pt 訓練時的 Target 範圍（相對於 robot base）
        TARGET_X_MIN, TARGET_X_MAX = -0.9, -0.7
        TARGET_Y_MIN, TARGET_Y_MAX = -0.65, -0.45
        TARGET_Z = 0.14
        
        print(f"\n🎯 Target 範圍 (相對於 Robot Base):")
        print(f"   X: [{TARGET_X_MIN}, {TARGET_X_MAX}] m")
        print(f"   Y: [{TARGET_Y_MIN}, {TARGET_Y_MAX}] m")
        print(f"   Z: {TARGET_Z} m (固定)")
        
        # Target 中心
        target_center_x = (TARGET_X_MIN + TARGET_X_MAX) / 2
        target_center_y = (TARGET_Y_MIN + TARGET_Y_MAX) / 2
        target_center_z = TARGET_Z
        
        print(f"\n📍 Target 中心 (相對於 Robot Base):")
        print(f"   X: {target_center_x:.4f} m")
        print(f"   Y: {target_center_y:.4f} m")
        print(f"   Z: {target_center_z:.4f} m")
        
        # 計算針尖初始位置到 Target 中心的距離
        dx = tip_pos_rel[0] - target_center_x
        dy = tip_pos_rel[1] - target_center_y
        dz = tip_pos_rel[2] - target_center_z
        distance_to_center = math.sqrt(dx**2 + dy**2 + dz**2)
        
        print(f"\n📏 針尖初始位置到 Target 中心的距離:")
        print(f"   ΔX: {dx:.4f} m (正=右, 負=左)")
        print(f"   ΔY: {dy:.4f} m (正=前, 負=後)")
        print(f"   ΔZ: {dz:.4f} m (正=上, 負=下)")
        print(f"   總距離: {distance_to_center:.4f} m")
        
        # 判斷問題
        print("\n" + "="*60)
        print("🔍 診斷分析")
        print("="*60)
        
        # 檢查針尖是否在 Target 範圍內
        in_range_x = TARGET_X_MIN <= tip_pos_rel[0] <= TARGET_X_MAX
        in_range_y = TARGET_Y_MIN <= tip_pos_rel[1] <= TARGET_Y_MAX
        in_range_z = abs(tip_pos_rel[2] - TARGET_Z) < 0.1
        
        print(f"\n   針尖 X 在 Target 範圍內？ {in_range_x} (tip={tip_pos_rel[0]:.4f}, range=[{TARGET_X_MIN}, {TARGET_X_MAX}])")
        print(f"   針尖 Y 在 Target 範圍內？ {in_range_y} (tip={tip_pos_rel[1]:.4f}, range=[{TARGET_Y_MIN}, {TARGET_Y_MAX}])")
        print(f"   針尖 Z 接近 Target？ {in_range_z} (tip={tip_pos_rel[2]:.4f}, target={TARGET_Z})")
        
        if distance_to_center > 0.5:
            print(f"\n⚠️ 警告: 針尖初始位置距離 Target 中心 {distance_to_center:.2f}m")
            print("   這可能導致 RL agent 難以學習 reach 行為！")
        
        if dz > 0.3:
            print(f"\n⚠️ 警告: 針尖比 Target 高 {dz:.2f}m")
            print("   RL agent 需要學會將針尖往下移動！")
        elif dz < -0.1:
            print(f"\n⚠️ 警告: 針尖比 Target 低 {abs(dz):.2f}m")
            print("   這可能是個問題，針尖可能需要穿過障礙物！")
            
    else:
        print("\n❌ 錯誤: 找不到 needle body!")
        print("   這可能是 model_4999.pt 訓練失敗的原因！")
        print("   請檢查 extension_link_out.usd 的 body 結構")
    
    # 也檢查 flange body
    flange_idx = None
    for i, name in enumerate(robot.body_names):
        if "flange" in name.lower():
            flange_idx = i
            break
    
    if flange_idx is not None:
        flange_pos_w = robot.data.body_state_w[0, flange_idx, :3].cpu().numpy()
        flange_pos_rel = flange_pos_w - robot_base_pos
        print(f"\n📌 Flange 相對於 Robot Base 座標 (參考):")
        print(f"   X: {flange_pos_rel[0]:.4f} m")
        print(f"   Y: {flange_pos_rel[1]:.4f} m")
        print(f"   Z: {flange_pos_rel[2]:.4f} m")
    
    print("\n" + "="*60)
    print("測量完成")
    print("="*60 + "\n")
    
    # 保持視窗開啟一下讓使用者查看
    for _ in range(100):
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
