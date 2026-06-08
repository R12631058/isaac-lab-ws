# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""讀取手術室 USD 中 target_pose 的位置"""

import argparse
from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="讀取 target_pose 位置")
parser.add_argument("--headless", action="store_true", default=False, help="以 headless 模式運行")
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from pxr import Usd, UsdGeom, Gf

@configclass
class SurgeryRoomSceneCfg(InteractiveSceneCfg):
    """手術室場景配置"""
    
    surgery_room = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SurgeryRoom",
        spawn=sim_utils.UsdFileCfg(
            usd_path="C:/Nick/surgery_team/surgery_team/USD/isaaclab/surgeryroom_isaac_lab.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )


def main():
    """讀取 target_pose 位置"""
    
    # 創建仿真環境
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cpu")
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # 設置場景
    scene_cfg = SurgeryRoomSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # 開始仿真
    sim.reset()
    
    print("\n" + "="*80)
    print("正在讀取 target_pose 位置...")
    print("="*80)
    
    # 獲取 USD stage
    stage = sim.stage
    
    # 查找 target_pose prim
    target_pose_path = "/World/envs/env_0/SurgeryRoom/target_pose/path_1"
    target_prim = stage.GetPrimAtPath(target_pose_path)
    
    if not target_prim.IsValid():
        print(f"❌ 找不到 target_pose: {target_pose_path}")
        # 嘗試其他可能的路徑
        alternative_paths = [
            "/World/envs/env_0/SurgeryRoom/target_pose",
            "/SurgeryRoom/target_pose/path_1",
            "/SurgeryRoom/target_pose",
        ]
        for alt_path in alternative_paths:
            alt_prim = stage.GetPrimAtPath(alt_path)
            if alt_prim.IsValid():
                print(f"✅ 找到替代路徑: {alt_path}")
                target_prim = alt_prim
                target_pose_path = alt_path
                break
    
    if target_prim.IsValid():
        # 獲取 Transform
        xform = UsdGeom.Xformable(target_prim)
        
        # 獲取 local transform
        local_transform = xform.GetLocalTransformation()
        translation = local_transform.ExtractTranslation()
        
        print(f"\n✅ Target Pose 位置:")
        print(f"   路徑: {target_pose_path}")
        print(f"   位置 (x, y, z): ({translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f})")
        
        # 獲取 world transform
        world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        world_translation = world_transform.ExtractTranslation()
        
        print(f"\n   世界坐標 (x, y, z): ({world_translation[0]:.4f}, {world_translation[1]:.4f}, {world_translation[2]:.4f})")
        
        # 建議的隨機範圍 (±20cm)
        radius = 0.2
        print(f"\n📍 建議的目標範圍 (半徑 {radius}m = 20cm):")
        print(f"   pos_x: ({world_translation[0] - radius:.2f}, {world_translation[0] + radius:.2f})")
        print(f"   pos_y: ({world_translation[1] - radius:.2f}, {world_translation[1] + radius:.2f})")
        print(f"   pos_z: ({world_translation[2] - radius:.2f}, {world_translation[2] + radius:.2f})")
        
    else:
        print(f"\n❌ 無法找到 target_pose")
        print("\n可用的 prims:")
        surgery_room_prim = stage.GetPrimAtPath("/World/envs/env_0/SurgeryRoom")
        if surgery_room_prim.IsValid():
            for child in surgery_room_prim.GetChildren():
                print(f"   - {child.GetPath()}")
    
    print("\n" + "="*80)
    
    # 關閉
    simulation_app.close()


if __name__ == "__main__":
    main()
