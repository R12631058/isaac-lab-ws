"""
動態射線檢測測試 - 測試新加入的物體是否能被 raycast 檢測到

測試目標：
1. 驗證 raycast 在動態添加物體後是否能檢測到
2. 檢查 debug draw 是否正常工作
3. 驗證基於 raycast 結果的實時色彩更新

執行方式:
    isaaclab.bat -p scripts/isaaclab_ws/experiments/test_dynamic_raycast.py
"""

import argparse
from isaaclab.app import AppLauncher

# 解析命令行參數
parser = argparse.ArgumentParser(description="Test Raycast in Isaac Lab")
parser.add_argument("--num_rays", type=int, default=5, help="Number of test rays")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 啟動模擬器
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===== 以下是模擬器啟動後的代碼 =====

import time
import numpy as np
import omni.usd
from pxr import UsdGeom, UsdPhysics, Gf, Sdf

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

# Debug Draw for visualization
from isaacsim.util.debug_draw import _debug_draw


class RaycastTester:
    """Raycast 測試類 - 使用與 bestviewpoint_autocost.py 相同的方式"""
    
    def __init__(self):
        self.raycast_interface = None
        self.debug_draw_interface = None
        self.test_results = []
        self.pending_results = []
        
    def initialize(self):
        """初始化 raycast 接口 - 與原始代碼相同"""
        try:
            import omni.kit.raycast.query
            self.raycast_interface = omni.kit.raycast.query.acquire_raycast_query_interface()
            print("✅ Raycast interface acquired successfully")
            print(f"   Interface type: {type(self.raycast_interface)}")
            
            # 初始化 debug draw 接口 (與 bestviewpoint_autocost.py 相同)
            self.debug_draw_interface = _debug_draw.acquire_debug_draw_interface()
            print("✅ Debug draw interface acquired successfully")
            
            return True
        except Exception as e:
            print(f"❌ Failed to acquire interfaces: {e}")
            return False
    
    def draw_ray(self, start, end, color=(1.0, 0.0, 0.0, 1.0), thickness=3.0):
        """
        繪製射線 (視覺化)
        color: RGBA 格式 (紅, 綠, 藍, 透明度)
        """
        if self.debug_draw_interface:
            # 使用 draw_lines (複數形式) - 接受列表
            self.debug_draw_interface.draw_lines(
                [start],  # 起點列表
                [end],    # 終點列表
                [color],  # 顏色列表
                [thickness]  # 粗細列表
            )
    
    def draw_point(self, position, color=(0.0, 1.0, 0.0, 1.0), size=10.0):
        """繪製點 (碰撞點視覺化)"""
        if self.debug_draw_interface:
            # 使用 draw_points (複數形式)
            self.debug_draw_interface.draw_points(
                [position],
                [color],
                [size]
            )
    
    def clear_drawings(self):
        """清除所有繪製"""
        if self.debug_draw_interface:
            self.debug_draw_interface.clear_lines()
            self.debug_draw_interface.clear_points()
    
    def test_single_ray_sync(self, origin: tuple, direction: tuple, max_distance: float = 100.0, draw_ray: bool = True):
        """
        測試單條射線 (同步方式)
        模擬 bestviewpoint_autocost.py 中的 _perform_raycast 方法
        """
        import omni.kit.raycast.query
        
        result_data = {
            'origin': origin,
            'direction': direction,
            'hit': False,
            'hit_position': None,
            'hit_normal': None,
            'hit_path': None,
            'distance': None,
            'draw_pending': draw_ray  # 標記需要繪製
        }
        
        # 保存引用以便在回調中繪製
        tester_ref = self
        
        def on_hit(ray, hit_result):
            """射線碰撞回調 - 與原始代碼中的 _on_raycast_hit 類似"""
            if hit_result.valid:
                result_data['hit'] = True
                result_data['hit_path'] = str(hit_result.get_target_usd_path())
                # 計算碰撞點 (注意: Isaac Lab 使用屬性而非方法)
                try:
                    hit_pos = hit_result.hit_position  # 使用屬性
                    if hit_pos:
                        result_data['hit_position'] = (hit_pos[0], hit_pos[1], hit_pos[2])
                        # 計算距離
                        dist = np.sqrt(sum((hit_pos[i] - origin[i])**2 for i in range(3)))
                        result_data['distance'] = dist
                        
                        # 在回調中繪製 (此時已確定命中)
                        if result_data['draw_pending']:
                            # 命中: 綠色射線
                            tester_ref.draw_ray(origin, result_data['hit_position'], color=(0.0, 1.0, 0.0, 1.0))
                            # 紅色碰撞點
                            tester_ref.draw_point(result_data['hit_position'], color=(1.0, 0.0, 0.0, 1.0), size=15.0)
                            result_data['draw_pending'] = False
                            
                except AttributeError:
                    pass  # 如果屬性不存在，跳過
        
        # 創建射線並執行查詢
        ray = omni.kit.raycast.query.Ray(origin, direction)
        self.raycast_interface.submit_raycast_query(ray, on_hit)
        
        self.test_results.append(result_data)
        
        # 延遲繪製未命中的射線 (在主循環中處理)
        # 未命中的射線會在後面統一繪製
        
        return result_data
    
    def draw_missed_rays(self):
        """繪製所有未命中的射線 (紅色)"""
        for result in self.test_results:
            if result.get('draw_pending', False) and not result['hit']:
                origin = result['origin']
                direction = result['direction']
                # 未命中射線延伸 10 單位
                end_point = tuple(origin[i] + direction[i] * 10.0 for i in range(3))
                self.draw_ray(origin, end_point, color=(1.0, 0.0, 0.0, 1.0))
                result['draw_pending'] = False
    
    def test_line_segment(self, start_pos: tuple, end_pos: tuple):
        """
        測試線段射線 (從起點到終點)
        這是 bestviewpoint_autocost.py 中最常用的方式
        """
        # 計算方向向量
        direction = [end_pos[i] - start_pos[i] for i in range(3)]
        length = np.sqrt(sum(d*d for d in direction))
        
        if length < 0.001:
            print("⚠️ Line segment too short")
            return None
        
        # 正規化方向
        normalized_dir = tuple(d / length for d in direction)
        
        # 稍微偏移起點以避免自碰撞 (與原始代碼相同)
        offset = 0.01
        adjusted_start = tuple(start_pos[i] + normalized_dir[i] * offset for i in range(3))
        
        return self.test_single_ray_sync(adjusted_start, normalized_dir, length)
    
    def print_results(self):
        """打印測試結果"""
        print("\n" + "="*60)
        print("Raycast Test Results")
        print("="*60)
        
        for i, result in enumerate(self.test_results):
            print(f"\n🔹 Ray {i+1}:")
            print(f"   Origin: {result['origin']}")
            print(f"   Direction: {result['direction']}")
            if result['hit']:
                print(f"   ✅ HIT!")
                print(f"   Hit Path: {result['hit_path']}")
                if result['hit_position']:
                    print(f"   Hit Position: {result['hit_position']}")
                if result['distance']:
                    print(f"   Distance: {result['distance']:.4f}")
            else:
                print(f"   ❌ No hit")
        
        print("\n" + "-"*60)
        hit_count = sum(1 for r in self.test_results if r['hit'])
        print(f"Summary: {hit_count}/{len(self.test_results)} rays hit targets")


def create_blocking_cube(prim_path: str, position: tuple):
    """在運行時創建一個可阻擋射線的立方體"""
    stage = omni.usd.get_context().get_stage()
    
    print(f"\n🔧 Creating blocking cube at {prim_path}")
    print(f"   Position: {position}")
    
    # 創建立方體
    cube_prim = stage.DefinePrim(prim_path, "Cube")
    cube_geom = UsdGeom.Cube(cube_prim)
    cube_geom.GetSizeAttr().Set(0.3)
    
    # 設置位置
    xform = UsdGeom.Xformable(cube_prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    
    # 設置顏色為紅色
    cube_geom.GetDisplayColorAttr().Set([(1.0, 0.0, 0.0)])
    
    # 添加 Physics API (重要！)
    UsdPhysics.RigidBodyAPI.Apply(cube_prim)
    mass_api = UsdPhysics.MassAPI.Apply(cube_prim)
    mass_api.CreateMassAttr(1.0)
    
    # 添加碰撞 (最重要！)
    UsdPhysics.CollisionAPI.Apply(cube_prim)
    
    print(f"   ✅ Blocking cube created with CollisionAPI")
    return prim_path


def create_test_scene():
    """創建測試場景"""
    print("\n📦 Creating test scene...")
    stage = omni.usd.get_context().get_stage()
    
    # 0. 添加燈光 (重要！否則場景會全黑)
    light_path = "/World/DomeLight"
    light_prim = stage.DefinePrim(light_path, "DomeLight")
    light_prim.GetAttribute("inputs:intensity").Set(1000.0)
    print(f"   ✅ Dome light created at {light_path}")
    
    # 添加方向光
    dist_light_path = "/World/DistantLight"
    dist_light_prim = stage.DefinePrim(dist_light_path, "DistantLight")
    dist_light_prim.GetAttribute("inputs:intensity").Set(500.0)
    xform = UsdGeom.Xformable(dist_light_prim)
    xform.AddRotateXYZOp().Set(Gf.Vec3f(-45, 45, 0))
    print(f"   ✅ Distant light created at {dist_light_path}")
    
    # 1. 創建地面平面
    ground_path = "/World/Ground"
    ground_prim = stage.DefinePrim(ground_path, "Cube")
    ground_geom = UsdGeom.Cube(ground_prim)
    ground_geom.GetSizeAttr().Set(20.0)
    
    # 設置地面位置 (平鋪在 z=0)
    xform = UsdGeom.Xformable(ground_prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.5))  # 向下移動半個高度
    xform.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 0.05))  # 壓扁成平面
    
    # 添加碰撞
    UsdPhysics.CollisionAPI.Apply(ground_prim)
    print(f"   ✅ Ground plane created at {ground_path}")
    
    # 2. 創建測試立方體
    cube_path = "/World/TestCube"
    cube_prim = stage.DefinePrim(cube_path, "Cube")
    cube_geom = UsdGeom.Cube(cube_prim)
    cube_geom.GetSizeAttr().Set(1.0)
    
    # 設置立方體位置
    xform = UsdGeom.Xformable(cube_prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 1.0))  # 放在地面上方
    
    # 添加碰撞
    UsdPhysics.CollisionAPI.Apply(cube_prim)
    print(f"   ✅ Test cube created at {cube_path}")
    
    # 3. 創建測試球體
    sphere_path = "/World/TestSphere"
    sphere_prim = stage.DefinePrim(sphere_path, "Sphere")
    sphere_geom = UsdGeom.Sphere(sphere_prim)
    sphere_geom.GetRadiusAttr().Set(0.3)
    
    # 設置球體位置
    xform = UsdGeom.Xformable(sphere_prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(2, 0, 0.5))
    
    # 添加碰撞
    UsdPhysics.CollisionAPI.Apply(sphere_prim)
    print(f"   ✅ Test sphere created at {sphere_path}")
    
    return {
        'ground': ground_path,
        'cube': cube_path,
        'sphere': sphere_path
    }


def main():
    """主測試流程"""
    print("\n" + "="*60)
    print("🔬 Isaac Lab Raycast Functionality Test")
    print("="*60)
    print("This test validates that raycast works in Isaac Lab")
    print("using the same API as bestviewpoint_autocost.py")
    print("="*60)
    
    # 設置模擬上下文
    sim_cfg = sim_utils.SimulationCfg(
        device="cuda:0",
        dt=1/60.0,
    )
    sim = SimulationContext(sim_cfg)
    
    # 設置相機視角
    sim.set_camera_view(eye=(8, 8, 5), target=(0, 0, 1))
    
    # 創建測試場景
    scene_paths = create_test_scene()
    
    # 重置模擬
    sim.reset()
    print("\n✅ Simulation reset complete")
    
    # 讓場景穩定幾個步驟
    print("⏳ Stabilizing scene...")
    for _ in range(30):
        sim.step()
        time.sleep(0.02)
    
    # 初始化 Raycast 測試器
    print("\n🔧 Initializing raycast tester...")
    tester = RaycastTester()
    if not tester.initialize():
        print("❌ Test aborted: Failed to initialize raycast")
        simulation_app.close()
        return
    
    # ===== 執行測試 =====
    print("\n" + "="*60)
    print("🎯 Running Raycast Tests")
    print("="*60)
    
    # 測試 1: 向下射線（應該擊中地面）
    print("\n--- Test 1: Downward ray (should hit ground) ---")
    result1 = tester.test_single_ray_sync(
        origin=(0, 0, 5),
        direction=(0, 0, -1)
    )
    
    # 讓 raycast 回調有時間處理
    for _ in range(10):
        sim.step()
        time.sleep(0.02)
    
    # 測試 2: 向立方體射線
    print("\n--- Test 2: Ray towards cube ---")
    result2 = tester.test_single_ray_sync(
        origin=(-5, 0, 1),
        direction=(1, 0, 0)
    )
    
    for _ in range(10):
        sim.step()
        time.sleep(0.02)
    
    # 測試 3: 向球體射線
    print("\n--- Test 3: Ray towards sphere ---")
    result3 = tester.test_single_ray_sync(
        origin=(5, 0, 0.5),
        direction=(-1, 0, 0)
    )
    
    for _ in range(10):
        sim.step()
        time.sleep(0.02)
    
    # 測試 4: 空射線（不應擊中任何物體）
    print("\n--- Test 4: Ray into empty space ---")
    result4 = tester.test_single_ray_sync(
        origin=(0, 0, 5),
        direction=(0, 0, 1)  # 向上射
    )
    
    for _ in range(10):
        sim.step()
        time.sleep(0.02)
    
    # 測試 5: 線段射線（模擬 NDI 檢測方式）
    print("\n--- Test 5: Line segment ray (NDI-style) ---")
    result5 = tester.test_line_segment(
        start_pos=(0, 5, 1),
        end_pos=(0, 0, 1)  # 射向立方體
    )
    
    # 等待所有 raycast 回調完成
    print("\n⏳ Waiting for raycast callbacks to complete...")
    for _ in range(30):
        sim.step()
        time.sleep(0.05)
    
    # 繪製未命中的射線
    tester.draw_missed_rays()
    
    # 打印結果
    tester.print_results()
    
    # 總結
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    hit_count = sum(1 for r in tester.test_results if r['hit'])
    total_tests = len(tester.test_results)
    
    print(f"Total rays tested: {total_tests}")
    print(f"Hits detected: {hit_count}")
    
    # 預期結果分析
    expected_hits = 4  # Tests 1-3 and 5 should hit
    if hit_count >= expected_hits:
        print(f"\n✅ SUCCESS: Raycast is working in Isaac Lab!")
        print(f"   Expected at least {expected_hits} hits, got {hit_count}")
    else:
        print(f"\n⚠️ WARNING: Expected {expected_hits} hits, but got {hit_count}")
        print("   Some raycasts may not be working correctly")
    
    print("\n💡 Next steps:")
    print("   1. If successful, proceed to test_trigger_volume_isaaclab.py")
    print("   2. You can now port NDI raycast detection to Isaac Lab")
    print("="*60)
    
    # 保持場景打開一會兒以便觀察 (非 headless 模式)
    if not args_cli.headless:
        print("\n⏳ Keeping scene open for 15 seconds to test dynamic raycast detection...")
        print("   🟢 Green lines = rays that HIT (real-time)")
        print("   🔴 Red lines = rays that MISSED (real-time)")
        print("   Blocking cube will be created at 3 seconds\n")
        
        blocking_cube_created = False
        hit_count_ray1 = 0
        hit_count_ray2 = 0
        
        for i in range(900):  # 15 seconds at 60 fps
            # 在第 3 秒時創建阻擋立方體 (180 frames at 60 fps)
            if i == 180 and not blocking_cube_created:
                blocking_cube_pos = (1.0, 0.0, 0.5)  # 在射線 1 的路徑上
                create_blocking_cube("/World/BlockingCube", blocking_cube_pos)
                blocking_cube_created = True
                # 等待物理引擎更新
                print("   ⏳ Waiting for physics engine to update...")
                for _ in range(5):
                    sim.step()
            
            # 清除上一幀的繪製
            tester.clear_drawings()
            
            # ========== 重新執行 raycast 查詢 ==========
            # 射線 1: 指向球體 (經過潛在的阻擋立方體)
            hit_info1 = {'hit': False}
            def on_hit1(ray, hit_info):
                if hit_info and hit_info.valid:
                    hit_info1['hit'] = True
            
            from omni.kit.raycast.query import Ray
            ray1 = Ray((-5, 0, 1), (5, 0, 1))  # 從左到右水平射線
            tester.raycast_interface.submit_raycast_query(ray1, on_hit1)
            
            # 射線 2: 向下射線 (擊中地面)
            hit_info2 = {'hit': False}
            def on_hit2(ray, hit_info):
                if hit_info and hit_info.valid:
                    hit_info2['hit'] = True
            
            ray2 = Ray((0, 0, 5), (0, 0, -5))  # 從上到下垂直射線
            tester.raycast_interface.submit_raycast_query(ray2, on_hit2)
            
            # 執行 sim.step() 讓 raycast 處理
            sim.step()
            
            # ========== 根據新的查詢結果繪製 ==========
            # 射線 1
            color1 = (0.0, 1.0, 0.0, 1.0) if hit_info1['hit'] else (1.0, 0.0, 0.0, 1.0)
            tester.draw_ray((-5, 0, 1), (5, 0, 1), color=color1, thickness=3.0)
            if hit_info1['hit']:
                hit_count_ray1 += 1
            
            # 射線 2
            color2 = (0.0, 1.0, 0.0, 1.0) if hit_info2['hit'] else (1.0, 0.0, 0.0, 1.0)
            tester.draw_ray((0, 0, 5), (0, 0, -5), color=color2, thickness=3.0)
            if hit_info2['hit']:
                hit_count_ray2 += 1
            
            time.sleep(1/60)
            
            # 每3秒提醒一次剩餘時間
            if i % 180 == 0 and i > 0:
                remaining = 15 - (i // 60)
                print(f"   ⏱️ {remaining} seconds remaining...")
                if blocking_cube_created:
                    ray1_status = "🟢 HITTING" if hit_count_ray1 > (i-180)//3 else "🔴 MISSING"
                    ray2_status = "🟢 HITTING" if hit_count_ray2 > (i-180)//3 else "🔴 MISSING"
                    print(f"      Ray 1 (horizontal): {ray1_status}")
                    print(f"      Ray 2 (vertical): {ray2_status}")
        
        print("\n✅ Scene observation complete")
        print(f"   Ray 1 total hits: {hit_count_ray1}")
        print(f"   Ray 2 total hits: {hit_count_ray2}")
        if blocking_cube_created:
            print("   ✅ Blocking cube was created")
            print("   📌 Check if Ray 1 color changed from green to red after 3 seconds")
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 關閉模擬器
        print("\n🔚 Closing simulation...")
        simulation_app.close()
