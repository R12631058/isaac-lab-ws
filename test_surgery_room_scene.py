#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
測試手術室 USD 場景 - 列出所有物件
"""

import argparse
from isaaclab.app import AppLauncher

# 添加命令行參數
parser = argparse.ArgumentParser(description="測試手術室 USD 場景")
parser.add_argument("--num_envs", type=int, default=4, help="並行環境數量")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 啟動 Omniverse 應用程式
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""其餘代碼"""

import torch
from pxr import Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass


@configclass
class SurgeryRoomSceneCfg(InteractiveSceneCfg):
    """手術室場景配置"""

    # 手術室 USD (包含機械臂、底座、其他物品)
    surgery_room = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SurgeryRoom",
        spawn=sim_utils.UsdFileCfg(
            usd_path="C:/Nick/surgery_team/surgery_team/USD/isaaclab/surgeryroom_isaac_lab.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # 光源
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


def print_prim_tree(prim, indent=0, max_depth=6):
    """遞歸列印 USD prim 樹狀結構"""
    if indent > max_depth:
        return
    
    # 獲取 prim 資訊
    prim_type = prim.GetTypeName()
    prim_path = str(prim.GetPath())
    
    # 檢查是否為可見的幾何體
    is_visible = ""
    if prim.IsA(UsdGeom.Imageable):
        imageable = UsdGeom.Imageable(prim)
        visibility = imageable.ComputeVisibility()
        if visibility == UsdGeom.Tokens.invisible:
            is_visible = " [HIDDEN]"
    
    # 檢查是否為關節/連桿
    is_articulation = ""
    if "joint" in prim_path.lower() or "link" in prim_path.lower():
        is_articulation = " 🔗"
    
    # 列印當前 prim
    indent_str = "  " * indent
    if prim_type:
        print(f"{indent_str}├─ {prim.GetName()} [{prim_type}]{is_articulation}{is_visible}")
    else:
        print(f"{indent_str}├─ {prim.GetName()}{is_articulation}{is_visible}")
    
    # 遞歸列印子 prim
    for child in prim.GetChildren():
        print_prim_tree(child, indent + 1, max_depth)


def analyze_scene(scene: InteractiveScene):
    """分析場景結構"""
    print("\n" + "="*80)
    print("🔍 手術室場景分析")
    print("="*80)
    
    # 獲取 USD Stage
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    
    # 列印環境 0 的完整結構
    env_0_path = "/World/envs/env_0/SurgeryRoom"
    env_0_prim = stage.GetPrimAtPath(env_0_path)
    
    if env_0_prim.IsValid():
        print(f"\n📦 環境 0 的場景結構: {env_0_path}")
        print("-" * 80)
        print_prim_tree(env_0_prim, indent=0, max_depth=8)
    else:
        print(f"❌ 找不到路徑: {env_0_path}")
        
        # 嘗試尋找實際路徑
        print("\n🔍 搜尋實際場景路徑...")
        world_prim = stage.GetPrimAtPath("/World")
        if world_prim.IsValid():
            print("\n/World 下的內容:")
            print_prim_tree(world_prim, indent=0, max_depth=3)
    
    # 統計資訊
    print("\n" + "="*80)
    print("📊 場景統計")
    print("="*80)
    
    # 計算所有 prims
    all_prims = [prim for prim in stage.Traverse()]
    print(f"總 Prim 數量: {len(all_prims)}")
    
    # 按類型統計
    type_counts = {}
    for prim in all_prims:
        prim_type = prim.GetTypeName()
        if prim_type:
            type_counts[prim_type] = type_counts.get(prim_type, 0) + 1
    
    print("\nPrim 類型分佈:")
    for prim_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {prim_type:25s}: {count:4d}")
    
    # 尋找可能的機械臂
    print("\n🤖 尋找機械臂相關物件:")
    robot_keywords = ["robot", "arm", "tm5", "tm_5", "manipulator", "joint", "link"]
    found_robots = []
    
    for prim in all_prims:
        prim_path = str(prim.GetPath()).lower()
        prim_name = prim.GetName().lower()
        
        for keyword in robot_keywords:
            if keyword in prim_path or keyword in prim_name:
                if prim.GetPath() not in [p.GetPath() for p in found_robots]:
                    found_robots.append(prim)
                break
    
    if found_robots:
        print(f"  找到 {len(found_robots)} 個相關物件:")
        for prim in found_robots[:20]:  # 只顯示前20個
            print(f"    - {prim.GetPath()} [{prim.GetTypeName()}]")
    else:
        print("  ❌ 未找到機械臂相關物件")
    
    print("\n" + "="*80)


def main():
    """主函數"""
    
    # 創建場景配置
    scene_cfg = SurgeryRoomSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    
    # 創建仿真上下文
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    
    # 設置主相機視角
    sim.set_camera_view(eye=(3.0, 3.0, 2.5), target=(0.0, 0.0, 0.5))
    
    # 創建場景
    scene = InteractiveScene(scene_cfg)
    
    print(f"\n✅ 場景創建成功!")
    print(f"   環境數量: {scene.num_envs}")
    print(f"   環境間距: {scene_cfg.env_spacing}m")
    
    # 分析場景
    analyze_scene(scene)
    
    # 執行幾個仿真步驟
    print("\n▶ 執行仿真步驟...")
    sim.reset()
    
    for i in range(10):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.cfg.dt)
    
    print("\n✅ 仿真測試完成!")
    print("\n💡 提示: 查看上面的場景結構,確認您想要使用的物件路徑")
    print("   然後我們可以配置機械臂和其他物件的交互")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠ 使用者中斷")
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
