"""
診斷腳本：測試 raycast 是否能檢測動態添加的物體

核心問題：
1. Debug draw 是否正常顯示？
2. Raycast 是否能檢測到在 sim.reset() 後添加的物體？

預期行為：
- 如果 raycast 能檢測動態物體：射線顏色應從綠色變紅色
- 如果 raycast 無法檢測：射線始終保持綠色（即使阻擋物在路徑上）
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test Dynamic Raycast Detection")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time
import numpy as np
import omni.usd
from pxr import UsdGeom, UsdPhysics, Gf
from omni.kit.raycast.query import Ray

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaacsim.util.debug_draw import _debug_draw


class SimpleTester:
    def __init__(self):
        import omni.kit.raycast.query
        self.raycast_interface = omni.kit.raycast.query.acquire_raycast_query_interface()
        self.debug_draw_interface = _debug_draw.acquire_debug_draw_interface()
        print("✅ Interfaces acquired")
    
    def draw_ray(self, start, end, color):
        if self.debug_draw_interface:
            self.debug_draw_interface.draw_lines([start], [end], [color], [3.0])
    
    def clear_drawings(self):
        if self.debug_draw_interface:
            self.debug_draw_interface.clear_lines()
            self.debug_draw_interface.clear_points()


def main():
    print("\n" + "="*70)
    print("🔬 DIAGNOSTIC: Testing Dynamic Raycast Detection")
    print("="*70)
    
    # 簡單場景
    sim_cfg = sim_utils.SimulationCfg(device="cuda:0", dt=1/60.0)
    sim = SimulationContext(sim_cfg)
    
    stage = omni.usd.get_context().get_stage()
    
    # 地面
    ground = stage.DefinePrim("/World/Ground", "Cube")
    UsdGeom.Cube(ground).GetSizeAttr().Set(10.0)
    xform = UsdGeom.Xformable(ground)
    xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.5))
    UsdPhysics.CollisionAPI.Apply(ground)
    
    # 射線路徑上的固定目標
    target = stage.DefinePrim("/World/Target", "Sphere")
    UsdGeom.Sphere(target).GetRadiusAttr().Set(0.2)
    xform = UsdGeom.Xformable(target)
    xform.AddTranslateOp().Set(Gf.Vec3d(2, 0, 1))
    UsdPhysics.CollisionAPI.Apply(target)
    
    sim.reset()
    for _ in range(30):
        sim.step()
    
    tester = SimpleTester()
    
    print("\n📊 TEST PHASE 1: Before Adding Blocker")
    print("-" * 70)
    
    # 測試 1：空場景
    hit1 = {'result': False}
    def on_hit1(ray, hit_info):
        if hit_info and hit_info.valid:
            hit1['result'] = True
    
    ray_start = (-3, 0, 1)
    ray_end = (3, 0, 1)
    ray = Ray(ray_start, ray_end)
    tester.raycast_interface.submit_raycast_query(ray, on_hit1)
    sim.step()
    
    print(f"Ray from {ray_start} to {ray_end}")
    print(f"Target at (2, 0, 1)")
    print(f"Result: {'🟢 HIT' if hit1['result'] else '🔴 MISS'}")
    
    # 保持結果用於比較
    baseline_hit = hit1['result']
    
    print("\n⏳ PHASE 2: Creating Blocking Cube on Ray Path")
    print("-" * 70)
    print("Creating blocker at (1, 0, 1) - directly on ray path")
    print("This should block the ray from reaching target\n")
    
    # 創建阻擋立方體
    blocker = stage.DefinePrim("/World/Blocker", "Cube")
    UsdGeom.Cube(blocker).GetSizeAttr().Set(0.3)
    xform = UsdGeom.Xformable(blocker)
    xform.AddTranslateOp().Set(Gf.Vec3d(1, 0, 1))
    
    # 添加 Physics
    UsdPhysics.RigidBodyAPI.Apply(blocker)
    UsdPhysics.MassAPI.Apply(blocker).CreateMassAttr(1.0)
    UsdPhysics.CollisionAPI.Apply(blocker)
    
    print("✅ Blocker created with CollisionAPI")
    
    # 等待物理更新
    print("⏳ Waiting for physics update...")
    for i in range(30):
        sim.step()
        if i % 10 == 0:
            print(f"   Frame {i}")
    
    print("\n📊 TEST PHASE 2: After Adding Blocker")
    print("-" * 70)
    
    # 測試 2：有阻擋物的場景
    hit2 = {'result': False}
    def on_hit2(ray, hit_info):
        if hit_info and hit_info.valid:
            hit2['result'] = True
    
    ray = Ray(ray_start, ray_end)
    tester.raycast_interface.submit_raycast_query(ray, on_hit2)
    sim.step()
    
    print(f"Ray from {ray_start} to {ray_end}")
    print(f"Blocker at (1, 0, 1) - SHOULD BLOCK")
    print(f"Target at (2, 0, 1)")
    print(f"Result: {'🟢 HIT' if hit2['result'] else '🔴 MISS'}")
    
    # 診斷結果
    print("\n" + "="*70)
    print("🔍 DIAGNOSTIC RESULTS")
    print("="*70)
    print(f"\nBefore blocker: {'🟢 HIT' if baseline_hit else '🔴 MISS'}")
    print(f"After blocker:  {'🟢 HIT' if hit2['result'] else '🔴 MISS'}")
    
    if baseline_hit and hit2['result']:
        print("\n❌ PROBLEM IDENTIFIED:")
        print("   Raycast did NOT detect the blocking cube!")
        print("   The ray hit count remained unchanged after adding blocker")
        print("\n   ROOT CAUSE: Raycast likely caches scene geometry at sim.reset()")
        print("   and does not dynamically update for newly added objects")
        print("\n   POSSIBLE SOLUTIONS:")
        print("   1. Reset physics context/simulation after adding new objects")
        print("   2. Use PhysX scene-level API to notify of geometry changes")
        print("   3. Add geometry BEFORE sim.reset()")
        
    elif not baseline_hit and not hit2['result']:
        print("\n⚠️ UNUSUAL: Neither test hit anything")
        print("   This suggests the raycast target setup may be incorrect")
        
    else:
        print("\n✅ SUCCESS:")
        print("   Raycast correctly detected state change!")
        print("   Blocker blocked the ray as expected")
    
    # 視覺調試
    print("\n📺 Starting 5-second visual debug (watch ray colors)")
    print("-" * 70)
    
    for i in range(300):
        tester.clear_drawings()
        
        # 重新執行查詢
        hit_test = {'result': False}
        def on_hit(ray, hit_info):
            if hit_info and hit_info.valid:
                hit_test['result'] = True
        
        ray = Ray(ray_start, ray_end)
        tester.raycast_interface.submit_raycast_query(ray, on_hit)
        sim.step()
        
        # 根據結果繪製
        color = (0.0, 1.0, 0.0, 1.0) if hit_test['result'] else (1.0, 0.0, 0.0, 1.0)
        tester.draw_ray(ray_start, ray_end, color)
        
        if i % 60 == 0:
            status = "🟢 HIT" if hit_test['result'] else "🔴 MISS"
            remaining = 5 - (i // 60)
            print(f"   {remaining}s remaining... Current: {status}")
        
        time.sleep(1/60)
    
    print("\n✅ Diagnostic complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
