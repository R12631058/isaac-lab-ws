# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
視覺化目標範圍 - 在手術室場景中顯示搜索範圍的立方體

可調整參數:
- center_x, center_y, center_z: 中心位置
- radius: 半徑大小
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Visualize target range in surgery room.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.markers.config import CUBOID_MARKER_CFG

# ============================================
# 🎯 在這裡調整搜索範圍參數
# ============================================
# 目標範圍 (世界座標) - 底座後方
# 底座位置: (1.264, -0.213, 0.922)
# 相對偏移: (-0.3, -0.6, 0.2), 範圍: 0.4×0.3×0.1m
CENTER_X = 0.964  # 目標中心 X (1.264 - 0.3)
CENTER_Y = -0.813  # 目標中心 Y (-0.213 - 0.6)
CENTER_Z = 1.122  # 目標中心 Z (0.922 + 0.2)
RADIUS = 0.2    # 半徑 (用於立方體半邊長，實際: X=0.2, Y=0.15, Z=0.05)
# ============================================


@configclass
class SurgeryRoomVisualizeCfg(InteractiveSceneCfg):
    """手術室場景配置 with visualization"""

    # 完整手術室 USD
    surgery_room = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SurgeryRoom",
        spawn=UsdFileCfg(
            usd_path="C:/Nick/surgery_team/surgery_team/USD/isaaclab/surgeryroom_isaac_lab.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # Robot - 使用 USD 中現有的 TM5-700
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/SurgeryRoom/robotarm_base/tm5_700",
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "joint_1": 0.0,
                "joint_2": -0.5,
                "joint_3": 0.5,
                "joint_4": 0.0,
                "joint_5": 0.5,
                "joint_6": 0.0,
            },
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

    # 光源
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75), 
            intensity=3000.0
        ),
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """運行模擬並視覺化搜索範圍"""
    
    # 創建 12 條邊的線框立方體來顯示搜索範圍
    # 立方體有 12 條邊: 4條底邊 + 4條頂邊 + 4條垂直邊
    range_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/TargetRange",
        markers={
            # 底部 4 條邊
            "bottom_edge_0": sim_utils.CylinderCfg(
                radius=0.003,  # 線條粗細 3mm
                height=RADIUS * 2,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            "bottom_edge_1": sim_utils.CylinderCfg(
                radius=0.003, height=RADIUS * 2,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            "bottom_edge_2": sim_utils.CylinderCfg(
                radius=0.003, height=RADIUS * 2,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            "bottom_edge_3": sim_utils.CylinderCfg(
                radius=0.003, height=RADIUS * 2,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            # 頂部 4 條邊
            "top_edge_0": sim_utils.CylinderCfg(
                radius=0.003, height=RADIUS * 2,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            "top_edge_1": sim_utils.CylinderCfg(
                radius=0.003, height=RADIUS * 2,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            "top_edge_2": sim_utils.CylinderCfg(
                radius=0.003, height=RADIUS * 2,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            "top_edge_3": sim_utils.CylinderCfg(
                radius=0.003, height=RADIUS * 2,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            # 垂直 4 條邊
            "vertical_edge_0": sim_utils.CylinderCfg(
                radius=0.003, height=RADIUS * 2,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            "vertical_edge_1": sim_utils.CylinderCfg(
                radius=0.003, height=RADIUS * 2,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            "vertical_edge_2": sim_utils.CylinderCfg(
                radius=0.003, height=RADIUS * 2,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            "vertical_edge_3": sim_utils.CylinderCfg(
                radius=0.003, height=RADIUS * 2,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )
    range_marker = VisualizationMarkers(range_marker_cfg)
    
    range_marker = VisualizationMarkers(range_marker_cfg)
    
    # 設定線條位置和方向
    num_envs = scene.num_envs
    
    # 計算立方體的 8 個角點
    min_x, max_x = CENTER_X - RADIUS, CENTER_X + RADIUS
    min_y, max_y = CENTER_Y - RADIUS, CENTER_Y + RADIUS
    min_z, max_z = CENTER_Z - RADIUS, CENTER_Z + RADIUS
    
    # 準備所有邊的位置和方向 (12條邊 × num_envs)
    # 形狀應該是 (num_envs * 12, 3) 和 (num_envs * 12, 4)
    edge_positions = torch.zeros(num_envs * 12, 3, device=sim.device)
    edge_orientations = torch.zeros(num_envs * 12, 4, device=sim.device)
    
    for env_idx in range(num_envs):
        base_idx = env_idx * 12
        
        # 底部 4 條邊 (Z = min_z)
        # Edge 0: 沿 X 軸 (min_y, min_z)
        edge_positions[base_idx + 0] = torch.tensor([CENTER_X, min_y, min_z], device=sim.device)
        edge_orientations[base_idx + 0] = torch.tensor([0.7071, 0.0, 0.7071, 0.0], device=sim.device)  # 90度繞Y軸
        
        # Edge 1: 沿 Y 軸 (max_x, min_z)
        edge_positions[base_idx + 1] = torch.tensor([max_x, CENTER_Y, min_z], device=sim.device)
        edge_orientations[base_idx + 1] = torch.tensor([0.7071, 0.7071, 0.0, 0.0], device=sim.device)  # 90度繞X軸
        
        # Edge 2: 沿 X 軸 (max_y, min_z)
        edge_positions[base_idx + 2] = torch.tensor([CENTER_X, max_y, min_z], device=sim.device)
        edge_orientations[base_idx + 2] = torch.tensor([0.7071, 0.0, 0.7071, 0.0], device=sim.device)
        
        # Edge 3: 沿 Y 軸 (min_x, min_z)
        edge_positions[base_idx + 3] = torch.tensor([min_x, CENTER_Y, min_z], device=sim.device)
        edge_orientations[base_idx + 3] = torch.tensor([0.7071, 0.7071, 0.0, 0.0], device=sim.device)
        
        # 頂部 4 條邊 (Z = max_z)
        # Edge 4: 沿 X 軸 (min_y, max_z)
        edge_positions[base_idx + 4] = torch.tensor([CENTER_X, min_y, max_z], device=sim.device)
        edge_orientations[base_idx + 4] = torch.tensor([0.7071, 0.0, 0.7071, 0.0], device=sim.device)
        
        # Edge 5: 沿 Y 軸 (max_x, max_z)
        edge_positions[base_idx + 5] = torch.tensor([max_x, CENTER_Y, max_z], device=sim.device)
        edge_orientations[base_idx + 5] = torch.tensor([0.7071, 0.7071, 0.0, 0.0], device=sim.device)
        
        # Edge 6: 沿 X 軸 (max_y, max_z)
        edge_positions[base_idx + 6] = torch.tensor([CENTER_X, max_y, max_z], device=sim.device)
        edge_orientations[base_idx + 6] = torch.tensor([0.7071, 0.0, 0.7071, 0.0], device=sim.device)
        
        # Edge 7: 沿 Y 軸 (min_x, max_z)
        edge_positions[base_idx + 7] = torch.tensor([min_x, CENTER_Y, max_z], device=sim.device)
        edge_orientations[base_idx + 7] = torch.tensor([0.7071, 0.7071, 0.0, 0.0], device=sim.device)
        
        # 垂直 4 條邊 (沿 Z 軸)
        # Edge 8: (min_x, min_y)
        edge_positions[base_idx + 8] = torch.tensor([min_x, min_y, CENTER_Z], device=sim.device)
        edge_orientations[base_idx + 8] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=sim.device)  # 無旋轉
        
        # Edge 9: (max_x, min_y)
        edge_positions[base_idx + 9] = torch.tensor([max_x, min_y, CENTER_Z], device=sim.device)
        edge_orientations[base_idx + 9] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=sim.device)
        
        # Edge 10: (max_x, max_y)
        edge_positions[base_idx + 10] = torch.tensor([max_x, max_y, CENTER_Z], device=sim.device)
        edge_orientations[base_idx + 10] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=sim.device)
        
        # Edge 11: (min_x, max_y)
        edge_positions[base_idx + 11] = torch.tensor([min_x, max_y, CENTER_Z], device=sim.device)
        edge_orientations[base_idx + 11] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=sim.device)
    
    # 更新標記
    range_marker.visualize(translations=edge_positions, orientations=edge_orientations)
    
    # 在控制台顯示範圍資訊
    print("\n" + "="*80)
    print("🎯 目標搜索範圍視覺化")
    print("="*80)
    print(f"📍 中心位置: ({CENTER_X:.2f}, {CENTER_Y:.2f}, {CENTER_Z:.2f}) m")
    print(f"📏 半徑: {RADIUS:.2f} m ({RADIUS*100:.0f} cm)")
    print(f"\n📦 搜索範圍:")
    print(f"   X: {CENTER_X - RADIUS:.2f} ~ {CENTER_X + RADIUS:.2f} m")
    print(f"   Y: {CENTER_Y - RADIUS:.2f} ~ {CENTER_Y + RADIUS:.2f} m")
    print(f"   Z: {CENTER_Z - RADIUS:.2f} ~ {CENTER_Z + RADIUS:.2f} m")
    print("\n🎨 視覺化說明:")
    print("   🟢 綠色線框立方體 = 目標隨機範圍 (12條邊)")
    print("\n⚙️  調整參數:")
    print("   在腳本開頭修改 CENTER_X, CENTER_Y, CENTER_Z, RADIUS")
    print("="*80 + "\n")
    
    # 模擬循環
    sim_dt = sim.get_physics_dt()
    count = 0
    
    # 定義模擬步驟
    sim.reset()
    
    while simulation_app.is_running():
        # 運行所有內容
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)
        
        # 每 100 步更新一次標記 (確保可見)
        if count % 100 == 0:
            range_marker.visualize(translations=edge_positions, orientations=edge_orientations)


def main():
    """主函數"""
    
    # 載入模擬套件
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # 設定相機視角
    sim.set_camera_view((2.0, 2.0, 2.0), (0.0, 0.0, 0.5))
    
    # 設計場景
    scene_cfg = SurgeryRoomVisualizeCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)
    
    # 播放模擬器
    sim.reset()
    
    print("[INFO]: 場景設置完成!")
    
    # 運行模擬
    run_simulator(sim, scene)


if __name__ == "__main__":
    # 運行主函數
    main()
    # 關閉模擬應用
    simulation_app.close()
