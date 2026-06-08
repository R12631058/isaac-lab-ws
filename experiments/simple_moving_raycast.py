"""
Isaac Lab 簡單射線測試：固定方塊 + 移動球體 + 動態 Raycast

執行方式:
    isaaclab.bat -p scripts/isaaclab_ws/experiments/simple_moving_raycast.py
"""

import argparse
from isaaclab.app import AppLauncher

# 解析命令行參數
parser = argparse.ArgumentParser(description="Simple Moving Raycast Test")
parser.add_argument("--test_duration", type=float, default=10.0, help="Test duration in seconds")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 啟動模擬器
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===== Isaac Lab/Sim 導入 =====

import math
import numpy as np
import omni.usd
from pxr import UsdGeom, UsdPhysics, Gf

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

# Raycast 接口
from omni.kit.raycast.query import acquire_raycast_query_interface, Ray

# Debug Draw
from isaacsim.util.debug_draw import _debug_draw


class MovingRaycastTest:
    """從方塊中心射出的射線測試，球體綠色、其他紅色"""
    
    def __init__(self):
        self.raycast_interface = None
        self.debug_draw = None
        self.time = 0.0
        self.hit_count = 0
        self.total_rays = 0
        self.cube_center = np.array([0.0, 0.0, 1.0])
        
    def initialize(self):
        print("\n🔧 Initializing raycast interface...")
        
        # 獲取 raycast 介面
        self.raycast_interface = acquire_raycast_query_interface()
        if self.raycast_interface:
            print("✅ Raycast interface acquired")
        else:
            print("❌ Failed to acquire raycast interface")
            return False
        
        # 獲取 debug draw 介面
        self.debug_draw = _debug_draw.acquire_debug_draw_interface()
        if self.debug_draw:
            print("✅ Debug draw interface acquired")
        else:
            print("❌ Failed to acquire debug draw interface")
            return False
            
        return True
    
    def get_sphere_position(self, dt):
        """獲取球體的動態位置（繞圈運動）"""
        self.time += dt
        
        # 繞圈半徑和中心
        radius = 2.0
        center_x, center_y, center_z = 0.0, 0.0, 1.0
        
        # 繞 Z 軸旋轉，同時上下振動
        x = center_x + radius * math.cos(self.time)
        y = center_y + radius * math.sin(self.time)
        z = center_z + 0.5 * math.sin(self.time * 2)
        
        return np.array([x, y, z])
    
    def get_cube_position(self):
        """獲取立方體位置（固定）"""
        return np.array([0.0, 0.0, 1.0])
    
    def is_sphere(self, usd_path: str) -> bool:
        """檢查 USD 路徑是否指向球體"""
        try:
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(usd_path)
            if prim.IsValid():
                prim_type = prim.GetPrimTypeInfo().GetTypeName()
                return "Sphere" in str(prim_type)
            return False
        except:
            return False
    
    def perform_raycast(self, origin, direction):
        """執行射線追蹤 - 從立方體中心沿指定方向（只提交查詢，不檢查結果）"""
        hit_result = {"hit": False, "position": None, "is_sphere": False, "hit_path": None, "callback_called": False}
        
        def on_hit(ray, hit_info):
            hit_result["callback_called"] = True  # 記錄回調是否被調用
            
            if hit_info is None:
                return
            
            if not hit_info.valid:
                return
            
            hit_result["hit"] = True
            hit_result["hit_path"] = str(hit_info.get_target_usd_path())
            try:
                hit_pos = hit_info.hit_position
                if hit_pos:
                    hit_result["position"] = np.array([hit_pos[0], hit_pos[1], hit_pos[2]])
                    hit_result["is_sphere"] = True
            except Exception as e:
                pass
        
        # Ray 需要起點和終點，而不是方向向量
        # 射線長度設為 10.0 以確保能從立方體延伸到球體
        ray_end = origin + direction * 10.0
        ray = Ray(tuple(origin), tuple(ray_end))
        
        self.raycast_interface.submit_raycast_query(ray, on_hit)
        
        # 不在這裡睡眠或檢查結果
        # 結果會在 sim.step() 之後被處理
        
        return hit_result
    
    def draw_rays_batch(self, rays_list):
        """一次繪製多條射線（不會互相清除）
        rays_list: [(start, end, is_sphere, thickness), ...]
        """
        if not self.debug_draw or not rays_list:
            return
        
        # 只清除一次
        self.debug_draw.clear_lines()
        self.debug_draw.clear_points()
        
        # 收集所有射線的數據
        line_starts = []
        line_ends = []
        line_colors = []
        line_thicknesses = []
        point_positions = []
        point_colors = []
        point_sizes = []
        
        for start, end, is_sphere, thickness in rays_list:
            # 決定顏色
            if is_sphere:
                color = (0.0, 1.0, 0.0, 1.0)  # 綠色 - 球體
            else:
                color = (1.0, 0.0, 0.0, 1.0)  # 紅色 - 其他物體
            
            # 添加射線
            line_starts.append(tuple(start))
            line_ends.append(tuple(end))
            line_colors.append(color)
            line_thicknesses.append(thickness)
            
            # 添加起點（藍色）
            point_positions.append(tuple(start))
            point_colors.append((0, 0, 1, 1))
            point_sizes.append(8.0)
            
            # 添加終點（黃色）
            point_positions.append(tuple(end))
            point_colors.append((1, 1, 0, 1))
            point_sizes.append(10.0)
        
        # 一次性繪製所有線
        if line_starts:
            self.debug_draw.draw_lines(line_starts, line_ends, line_colors, line_thicknesses)
        
        # 一次性繪製所有點
        if point_positions:
            self.debug_draw.draw_points(point_positions, point_colors, point_sizes)


def verify_scene():
    """驗證場景中的球體是否真的存在"""
    stage = omni.usd.get_context().get_stage()
    
    moving_sphere = stage.GetPrimAtPath("/World/MovingSphere")
    test_sphere = stage.GetPrimAtPath("/World/TestSphere")
    
    print("\n🔍 場景驗證:")
    print(f"  移動球體 /World/MovingSphere: {'✅ 存在' if moving_sphere.IsValid() else '❌ 不存在'}")
    print(f"  測試球體 /World/TestSphere: {'✅ 存在' if test_sphere.IsValid() else '❌ 不存在'}")
    
    if moving_sphere.IsValid():
        xform = UsdGeom.Xformable(moving_sphere)
        local_transform = xform.ComputeLocalToWorldTransform(0)
        print(f"    位置: {local_transform.ExtractTranslation()}")
    
    if test_sphere.IsValid():
        xform = UsdGeom.Xformable(test_sphere)
        local_transform = xform.ComputeLocalToWorldTransform(0)
        print(f"    位置: {local_transform.ExtractTranslation()}")
    print()


def create_scene():
    """創建場景 - 使用與 test_raycast_isaaclab.py 相同的方式"""
    print("\n📦 Creating scene...")
    stage = omni.usd.get_context().get_stage()
    
    # 添加光源
    light_path = "/World/DomeLight"
    light_prim = stage.DefinePrim(light_path, "DomeLight")
    light_prim.GetAttribute("inputs:intensity").Set(1000.0)
    print(f"   ✅ DomeLight")
    
    dist_light_path = "/World/DistantLight"
    dist_light_prim = stage.DefinePrim(dist_light_path, "DistantLight")
    dist_light_prim.GetAttribute("inputs:intensity").Set(500.0)
    xform = UsdGeom.Xformable(dist_light_prim)
    xform.AddRotateXYZOp().Set(Gf.Vec3f(-45, 45, 0))
    print(f"   ✅ DistantLight")
    
    # 地面
    ground_path = "/World/Ground"
    ground_prim = stage.DefinePrim(ground_path, "Cube")
    ground_geom = UsdGeom.Cube(ground_prim)
    ground_geom.GetSizeAttr().Set(20.0)
    xform = UsdGeom.Xformable(ground_prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.5))
    xform.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 0.05))
    UsdPhysics.CollisionAPI.Apply(ground_prim)
    print(f"   ✅ Ground plane")
    
    # 射線起點的立方體（固定）
    cube_path = "/World/RayOrigin"
    cube_prim = stage.DefinePrim(cube_path, "Cube")
    cube_geom = UsdGeom.Cube(cube_prim)
    cube_geom.GetSizeAttr().Set(0.2)
    xform = UsdGeom.Xformable(cube_prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 1.0))
    cube_geom.GetDisplayColorAttr().Set([(0.5, 0.5, 1.0)])  # 淡藍色
    UsdPhysics.CollisionAPI.Apply(cube_prim)
    print(f"   ✅ Ray origin cube")
    
    # 固定的測試球體 - 綠色
    test_sphere_path = "/World/TestSphere"
    test_sphere_prim = stage.DefinePrim(test_sphere_path, "Sphere")
    test_sphere_geom = UsdGeom.Sphere(test_sphere_prim)
    test_sphere_geom.GetRadiusAttr().Set(0.3)
    xform = UsdGeom.Xformable(test_sphere_prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(3, 0, 1))
    test_sphere_geom.GetDisplayColorAttr().Set([(0.0, 1.0, 0.0)])  # 綠色
    UsdPhysics.CollisionAPI.Apply(test_sphere_prim)
    print(f"   ✅ Static test sphere")
    
    # 移動的球體 - 紅色
    sphere_path = "/World/MovingSphere"
    sphere_prim = stage.DefinePrim(sphere_path, "Sphere")
    sphere_geom = UsdGeom.Sphere(sphere_prim)
    sphere_geom.GetRadiusAttr().Set(0.2)
    xform = UsdGeom.Xformable(sphere_prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(2, 0, 1.5))
    sphere_geom.GetDisplayColorAttr().Set([(1.0, 0.0, 0.0)])  # 紅色
    UsdPhysics.CollisionAPI.Apply(sphere_prim)
    print(f"   ✅ Moving sphere with CollisionAPI")
    
    return True


def update_sphere_position(sphere_path, position):
    """更新球體位置 - 直接修改 translate 屬性"""
    stage = omni.usd.get_context().get_stage()
    sphere_prim = stage.GetPrimAtPath(sphere_path)
    
    if sphere_prim.IsValid():
        # 直接設置 xformOp:translate 屬性（不清除 xform order）
        xform = UsdGeom.Xformable(sphere_prim)
        
        # 獲取或創建 translate op
        translate_op = None
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        
        if translate_op:
            # 直接更新現有的 translate op
            translate_op.Set(Gf.Vec3d(*position))
        else:
            # 如果沒有 translate op，就添加一個（第一次調用時）
            xform.AddTranslateOp().Set(Gf.Vec3d(*position))


def main():
    print("\n" + "="*70)
    print("🎯 Isaac Lab Simple Moving Raycast Test")
    print("="*70)
    print("🔵 藍色方塊 = 固定目標")
    print("🔴 紅色球體 = 移動射線來源")
    print("🟢 綠色線 = 射線命中")
    print("🔴 紅色線 = 射線未命中")
    print("="*70 + "\n")
    
    # 設置模擬
    sim_cfg = sim_utils.SimulationCfg(
        device="cuda:0",
        dt=1/60.0,
        gravity=(0.0, 0.0, -9.81)
    )
    sim = SimulationContext(sim_cfg)
    
    # 設置相機 - 調整以看到所有物體（包括遠處的測試球體）
    sim.set_camera_view(eye=(8, 10, 5), target=(2, 3, 1.5))
    
    # 創建場景
    create_scene()
    
    # 驗證場景
    verify_scene()
    
    # 創建測試器
    tester = MovingRaycastTest()
    if not tester.initialize():
        return
    
    # 重置模擬
    sim.reset()
    
    # 等待物理引擎初始化碰撞體（重要！）
    print("⏳ Initializing physics collisions...")
    for i in range(30):  # 運行30幀讓物理引擎準備好
        sim.step()
    
    print("\n✅ Simulation initialized\n")
    
    # 主模擬循環
    test_duration = args_cli.test_duration
    dt = 1/60.0
    elapsed_time = 0.0
    step_count = 0
    
    print("🎬 Starting simulation...\n")
    print("🟦 立方體中心射出射線")
    print("🟢 綠色線 = 命中球體")
    print("🔴 紅色線 = 命中其他物體\n")
    
    # 固定球體位置（與場景中設置的位置一致）
    test_sphere_pos = np.array([3.0, 0.0, 1.0])
    
    # 統計：固定球體和移動球體的分別命中數
    hit_count_fixed = 0
    hit_count_moving = 0
    total_rays_per_target = 0
    
    while elapsed_time < test_duration:
        # 獲取移動球體的動態位置
        sphere_pos = tester.get_sphere_position(dt)
        
        # 獲取射線起點的實際位置（從 USD 讀取）
        stage = omni.usd.get_context().get_stage()
        ray_origin_prim = stage.GetPrimAtPath("/World/RayOrigin")
        if ray_origin_prim.IsValid():
            xform = UsdGeom.Xformable(ray_origin_prim)
            local_transform = xform.ComputeLocalToWorldTransform(0)
            cube_center = np.array(local_transform.ExtractTranslation())
        else:
            cube_center = np.array([0.0, 0.0, 1.0])
        
        # 更新移動球體位置
        update_sphere_position("/World/MovingSphere", sphere_pos)
        
        # ========== 射線1：射向固定測試球體 ==========
        direction_to_fixed = test_sphere_pos - cube_center
        direction_norm = np.linalg.norm(direction_to_fixed)
        if direction_norm > 0:
            direction_to_fixed = direction_to_fixed / direction_norm
        
        # 提交查詢但不檢查結果
        hit_result_fixed = tester.perform_raycast(cube_center, direction_to_fixed)
        
        # ========== 射線2：射向移動球體 ==========
        direction_to_moving = sphere_pos - cube_center
        direction_norm = np.linalg.norm(direction_to_moving)
        if direction_norm > 0:
            direction_to_moving = direction_to_moving / direction_norm
        
        # 提交查詢但不檢查結果
        hit_result_moving = tester.perform_raycast(cube_center, direction_to_moving)
        
        # ========== 步進模擬 - 這樣 raycast 查詢才會被處理 ==========
        sim.step()
        
        # ========== 現在再檢查結果 ==========
        # 繪製射線
        rays_to_draw = []
        
        # 射線1：固定球體
        if hit_result_fixed["hit"] and hit_result_fixed["position"] is not None:
            ray_end_fixed = hit_result_fixed["position"]
            is_hit_fixed = (hit_result_fixed["hit_path"] == "/World/TestSphere")
            if is_hit_fixed:
                hit_count_fixed += 1
        else:
            ray_end_fixed = test_sphere_pos
            is_hit_fixed = False
        
        rays_to_draw.append((cube_center, ray_end_fixed, is_hit_fixed, 3.0))
        
        # 射線2：移動球體
        if hit_result_moving["hit"] and hit_result_moving["position"] is not None:
            ray_end_moving = hit_result_moving["position"]
            is_hit_moving = (hit_result_moving["hit_path"] == "/World/MovingSphere")
            if is_hit_moving:
                hit_count_moving += 1
        else:
            ray_end_moving = sphere_pos
            is_hit_moving = False
        
        rays_to_draw.append((cube_center, ray_end_moving, is_hit_moving, 3.0))
        
        # 一次性繪製所有射線
        tester.draw_rays_batch(rays_to_draw)
        
        total_rays_per_target += 1
        
        # 步進模擬
        sim.step()
        
        elapsed_time += dt
        step_count += 1
        
        # 每秒打印一次狀態
        if step_count % 60 == 0:
            hit_rate_fixed = (hit_count_fixed / total_rays_per_target * 100) if total_rays_per_target > 0 else 0
            hit_rate_moving = (hit_count_moving / total_rays_per_target * 100) if total_rays_per_target > 0 else 0
            
            print(f"\n⏱️  {elapsed_time:.1f}s")
            print(f"  📍 固定球體 (/World/TestSphere @ [5.0, 8.2, 3.2])")
            print(f"     命中: {hit_count_fixed}/{total_rays_per_target} ({hit_rate_fixed:.1f}%)")
            print(f"     路徑: {hit_result_fixed.get('hit_path')}")
            print(f"  📍 移動球體 (/World/MovingSphere @ [{sphere_pos[0]:.2f}, {sphere_pos[1]:.2f}, {sphere_pos[2]:.2f}])")
            print(f"     命中: {hit_count_moving}/{total_rays_per_target} ({hit_rate_moving:.1f}%)")
            print(f"     路徑: {hit_result_moving.get('hit_path')}")
    
    # 最終統計
    print("\n" + "="*70)
    print("📊 測試完成")
    print("="*70)
    print(f"📍 固定球體 (/World/TestSphere):")
    print(f"   總射線: {total_rays_per_target}")
    print(f"   命中: {hit_count_fixed}")
    print(f"   命中率: {hit_count_fixed/total_rays_per_target*100:.1f}%")
    print(f"\n📍 移動球體 (/World/MovingSphere):")
    print(f"   總射線: {total_rays_per_target}")
    print(f"   命中: {hit_count_moving}")
    print(f"   命中率: {hit_count_moving/total_rays_per_target*100:.1f}%")
    
    if hit_count_fixed > 0 and hit_count_moving == 0:
        print("\n⚠️  發現問題：固定球體能被擊中，但移動球體不能")
        print("    原因：移動球體的位置更新可能沒有同步到物理引擎")
    elif hit_count_fixed > 0 and hit_count_moving > 0:
        print("\n✅ 成功：兩個球體都能被擊中")
    elif hit_count_fixed == 0:
        print("\n❌ 問題：連固定球體都沒有被擊中")
        print("    原因可能是：")
        print("    1. CollisionAPI 沒有正確應用到球體")
        print("    2. Raycast 介面配置有問題")
        print("    3. 等待時間不足")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🔚 Closing simulation...\n")
        simulation_app.close()
