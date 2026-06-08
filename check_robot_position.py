# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
檢查機器人在場景中的實際位置

用途:找出機器人基座的世界座標,以便正確設定目標範圍
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Check robot position in surgery room.")
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

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg


@configclass
class SurgeryRoomSceneCfg(InteractiveSceneCfg):
    """手術室場景配置"""

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


def main():
    """主函數"""
    
    # 載入模擬套件
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # 設定相機視角
    sim.set_camera_view((2.0, 2.0, 2.0), (0.0, 0.0, 0.5))
    
    # 設計場景
    scene_cfg = SurgeryRoomSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)
    
    # 播放模擬器
    sim.reset()
    
    print("\n" + "="*80)
    print("🔍 機器人位置檢查")
    print("="*80)
    
    # 獲取機器人資訊
    robot = scene["robot"]
    
    # 獲取機器人根部位置 (基座位置)
    root_state = robot.data.root_pos_w
    print(f"\n📍 機器人基座位置 (世界座標系):")
    print(f"   X: {root_state[0, 0].item():.4f} m")
    print(f"   Y: {root_state[0, 1].item():.4f} m")
    print(f"   Z: {root_state[0, 2].item():.4f} m")
    
    # 獲取末端執行器位置
    body_names = robot.data.body_names
    if "flange" in body_names:
        flange_idx = body_names.index("flange")
        flange_pos = robot.data.body_pos_w[:, flange_idx, :]
        print(f"\n🔧 末端執行器 (flange) 位置 (世界座標系):")
        print(f"   X: {flange_pos[0, 0].item():.4f} m")
        print(f"   Y: {flange_pos[0, 1].item():.4f} m")
        print(f"   Z: {flange_pos[0, 2].item():.4f} m")
    
    # 獲取所有 body 的名稱和位置
    print(f"\n📋 所有 Body 位置:")
    for i, name in enumerate(body_names):
        pos = robot.data.body_pos_w[0, i, :]
        print(f"   {i:2d}. {name:20s}: ({pos[0].item():7.4f}, {pos[1].item():7.4f}, {pos[2].item():7.4f})")
    
    print("\n💡 建議:")
    print("   1. 目標範圍應該相對於機器人基座位置設定")
    print("   2. 如果目標在場景中某個固定位置,使用該位置的世界座標")
    print("   3. 檢查 target_pose 標記在 USD 中的實際座標")
    print("="*80 + "\n")
    
    # 保持視窗開啟
    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())


if __name__ == "__main__":
    # 運行主函數
    main()
    # 關閉模擬應用
    simulation_app.close()
