"""
修復版本：Dynamic Raycast Detection
- 修復 debug draw 消失問題
- 修復 raycast 不檢測新物體問題
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fixed Dynamic Raycast Test")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time
import numpy as np
import math
import omni.usd
import omni.kit.raycast.query
from pxr import UsdGeom, UsdPhysics, Gf
from omni.kit.raycast.query import Ray

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaacsim.util.debug_draw import _debug_draw


def main():
    print("\n" + "="*70)
    print("🔬 Fixed Dynamic Raycast Detection Test")
    print("="*70)
    
    # 初始化
    sim_cfg = sim_utils.SimulationCfg(device="cuda:0", dt=1/60.0)
    sim = SimulationContext(sim_cfg)
    stage = omni.usd.get_context().get_stage()
    
    # 獲取 raycast 和 debug draw 接口
    raycast_interface = omni.kit.raycast.query.acquire_raycast_query_interface()
    debug_draw = _debug_draw.acquire_debug_draw_interface()
    print("✅ Raycast and debug draw acquired\n")
    
    # ========== 創建場景 ==========
    print("📦 Creating scene...")
    
    # 地面
    ground = stage.DefinePrim("/World/Ground", "Cube")
    UsdGeom.Cube(ground).GetSizeAttr().Set(20.0)
    xform = UsdGeom.Xformable(ground)
    xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.5))
    UsdPhysics.CollisionAPI.Apply(ground)
    print("   ✅ Ground")
    
    # 固定球體 1
    sphere1 = stage.DefinePrim("/World/Sphere1", "Sphere")
    UsdGeom.Sphere(sphere1).GetRadiusAttr().Set(0.3)
    xform = UsdGeom.Xformable(sphere1)
    xform.AddTranslateOp().Set(Gf.Vec3d(2, 0, 1))
    UsdPhysics.CollisionAPI.Apply(sphere1)
    print("   ✅ Sphere1 at (2, 0, 1)")
    
    # 射線起點立方體
    cube = stage.DefinePrim("/World/Cube", "Cube")
    UsdGeom.Cube(cube).GetSizeAttr().Set(0.5)
    xform = UsdGeom.Xformable(cube)
    xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 1))
    cube_color = UsdGeom.Cube(cube)
    cube_color.GetDisplayColorAttr().Set([(0.5, 0.5, 1.0)])
    UsdPhysics.CollisionAPI.Apply(cube)
    print("   ✅ Cube at (0, 0, 1)")
    
    sim.reset()
    print("\n⏳ Initializing physics (30 frames)...")
    for _ in range(30):
        sim.step()
    print("✅ Physics initialized\n")
    
    # ========== 主測試：10 秒觀察期 ==========
    print("📊 OBSERVATION PERIOD: 10 seconds")
    print("-" * 70)
    print("🟢 Green = HIT (raycast detected collision)")
    print("🔴 Red = MISS (no collision detected)")
    print("🔵 Blocking cube will be created at 3 seconds\n")
    
    blocker_created = False
    
    for frame in range(600):  # 10 seconds @ 60 fps
        # 在第 180 幀時創建阻擋立方體 (3 秒)
        if frame == 180 and not blocker_created:
            print("\n🚧 Creating blocking cube at (1, 0, 1)...")
            blocker = stage.DefinePrim("/World/Blocker", "Cube")
            UsdGeom.Cube(blocker).GetSizeAttr().Set(0.25)
            xform = UsdGeom.Xformable(blocker)
            xform.AddTranslateOp().Set(Gf.Vec3d(1, 0, 1))
            UsdPhysics.RigidBodyAPI.Apply(blocker)
            UsdPhysics.MassAPI.Apply(blocker).CreateMassAttr(1.0)
            UsdPhysics.CollisionAPI.Apply(blocker)
            
            blocker_color = UsdGeom.Cube(blocker)
            blocker_color.GetDisplayColorAttr().Set([(1.0, 0.0, 0.0)])
            
            print("✅ Blocker created")
            blocker_created = True
            
            # 讓物理引擎更新
            print("⏳ Physics update (10 frames)...")
            for _ in range(10):
                sim.step()
            print("✅ Ready to test\n")
            continue
        
        # 清除前一幀
        debug_draw.clear_lines()
        debug_draw.clear_points()
        
        # ========== 關鍵：重新執行 raycast 查詢 ==========
        # 建立回調存儲結果
        results = {'hit': False}
        
        def raycast_callback(ray, hit_info):
            """回調函數 - raycast 結果回傳到此"""
            nonlocal results
            if hit_info and hit_info.valid:
                results['hit'] = True
        
        # 射線：從 (-3, 0, 1) 到 (3, 0, 1)
        # 這條射線經過阻擋立方體 (1, 0, 1) 到達球體 (2, 0, 1)
        ray = Ray((-3, 0, 1), (3, 0, 1))
        raycast_interface.submit_raycast_query(ray, raycast_callback)
        
        # ⚠️ 關鍵：必須在此調用 sim.step() 讓 raycast 處理
        sim.step()
        
        # ========== 根據結果繪製 ==========
        color = (0.0, 1.0, 0.0, 1.0) if results['hit'] else (1.0, 0.0, 0.0, 1.0)
        status = "🟢 HIT" if results['hit'] else "🔴 MISS"
        
        debug_draw.draw_lines(
            [(-3, 0, 1)],
            [(3, 0, 1)],
            [color],
            [3.0]
        )
        
        # 繪製原點
        debug_draw.draw_points(
            [(-3, 0, 1)],
            [(0.0, 0.0, 1.0, 1.0)],
            [8.0]
        )
        
        # 定期打印狀態
        if frame % 60 == 0:
            elapsed = frame / 60.0
            remaining = 10.0 - elapsed
            
            if blocker_created and frame >= 180:
                print(f"⏱️  {elapsed:.1f}s (Blocker present) → {status}")
            else:
                print(f"⏱️  {elapsed:.1f}s (No blocker yet) → {status}")
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETE")
    print("="*70)
    
    print("\n📋 ANALYSIS:")
    print("   If ray stayed 🟢 GREEN after blocker created at 3s:")
    print("   ❌ Raycast did NOT detect the blocking cube")
    print("   ❌ Problem: Raycast caches scene, doesn't update dynamically")
    print("\n   If ray turned 🔴 RED after blocker created at 3s:")
    print("   ✅ Raycast works dynamically!")
    print("   ✅ Blocker was detected and blocked the ray")
    
    print("\n💡 SOLUTION IF PROBLEM 2 EXISTS:")
    print("   Use PhysX API to force scene update after adding geometry")
    print("   Or recreate raycast interface after scene changes")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🔚 Closing...")
        simulation_app.close()
