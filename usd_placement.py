# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to load multiple surgery room environments into Isaac Lab.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/isaaclab_ws/usd_placement.py --num_envs 4

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Load multiple surgery room USD files into Isaac Lab.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


@configclass
class SurgeryRoomSceneCfg(InteractiveSceneCfg):
    """Configuration for the surgery room scene."""
    
    # 這裡只保留空的配置，地面和燈光在 design_scene 中手動創建
    pass


def design_scene(num_envs: int, env_spacing: float = 5.0):
    """手動設計場景，載入多個手術室 USD 檔案"""
    
    # 創建地面平台
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    
    # 創建照明
    light_cfg = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    light_cfg.func("/World/lightDistant", light_cfg)
    
    # 計算環境排列（網格佈局）
    envs_per_row = int(torch.sqrt(torch.tensor(num_envs, dtype=torch.float32)).ceil().int())
    
    for env_idx in range(num_envs):
        # 計算環境位置
        row = env_idx // envs_per_row
        col = env_idx % envs_per_row
        
        # 計算偏移量
        x_offset = (col - (envs_per_row - 1) / 2) * env_spacing
        y_offset = (row - (envs_per_row - 1) / 2) * env_spacing
        
        # 載入手術室 USD 檔案
        surgery_room_cfg = sim_utils.UsdFileCfg(
            usd_path="C:/Nick/surgery_team/surgery_team/USD/surgeryroom_isaaclab.usd"
        )
        
        # 使用 func 方法載入到特定位置
        surgery_room_cfg.func(
            f"/World/envs/env_{env_idx}/SurgeryRoom",
            surgery_room_cfg,
            translation=(x_offset, y_offset, 0.0)
        )
        
        print(f"[INFO]: Loaded surgery room {env_idx} at position ({x_offset:.1f}, {y_offset:.1f}, 0.0)")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """運行模擬器的主迴圈"""
    
    # 模擬參數
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    
    print(f"[INFO]: Scene ready with {scene.num_envs} surgery room environments")
    
    # 模擬迴圈
    while simulation_app.is_running():
        # 執行模擬步驟
        sim.step()
        
        # 更新時間
        sim_time += sim_dt
        count += 1
        
        # 更新場景
        scene.update(sim_dt)
        
        # 每 500 幀輸出一次資訊
        if count % 500 == 0:
            print(f"[INFO]: Simulation time: {sim_time:.2f}s, Frame: {count}")
            print(f"[INFO]: Running {scene.num_envs} surgery room environments")


def main():
    """Main function."""
    
    # 解析環境數量
    num_envs = args_cli.num_envs
    env_spacing = 8.0  # 環境間距
    
    # 建立模擬上下文
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # 建立場景配置（空配置）
    scene_cfg = SurgeryRoomSceneCfg(num_envs=num_envs, env_spacing=env_spacing)
    
    # 建立互動場景
    scene = InteractiveScene(scene_cfg)
    
    # 手動設計場景並載入多個手術室
    design_scene(num_envs, env_spacing)
    
    # 設置攝影機位置 - 調整以觀察多個環境
    if num_envs == 1:
        sim.set_camera_view(eye=[8.0, 8.0, 6.0], target=[0.0, 0.0, 1.0])
    elif num_envs <= 4:
        sim.set_camera_view(eye=[15.0, 15.0, 10.0], target=[0.0, 0.0, 1.0])
    else:
        sim.set_camera_view(eye=[25.0, 25.0, 15.0], target=[0.0, 0.0, 1.0])

    # 重置模擬
    sim.reset()
    
    print(f"[INFO]: Setup complete with {num_envs} surgery room environments!")
    print(f"[INFO]: Environment spacing: {env_spacing}m")
    
    # 運行模擬
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()