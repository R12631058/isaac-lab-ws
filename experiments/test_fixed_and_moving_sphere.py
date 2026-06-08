"""
測試固定和移動球體的 Raycast
基於 test_raycast_isaaclab.py 的工作方式
conda activate env_isaaclab
.\isaaclab.bat -p scripts\isaaclab_ws\test_ndi_joint_motion_v2.py --hold_frames 600
"""

import argparse
from isaaclab.app import AppLauncher

# 解析命令行參數
parser = argparse.ArgumentParser(description="Test Fixed and Moving Sphere Raycast")
parser.add_argument("--test_duration", type=float, default=10.0, help="Test duration in seconds")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 啟動模擬器
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 在這之後才能導入 omni 模組
import math
import numpy as np
import omni.usd
from pxr import UsdGeom, UsdPhysics, UsdShade, Gf, Sdf

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from omni.kit.raycast.query import Ray
from isaacsim.util.debug_draw import _debug_draw


class RaycastTester:
    """Raycast 測試類 - 使用與 test_raycast_isaaclab.py 相同的方式"""
    
    def __init__(self):
        self.raycast_interface = None
        self.debug_draw_interface = None
        self.test_results = []
        
    def initialize(self):
        """初始化 raycast 接口"""
        try:
            import omni.kit.raycast.query
            self.raycast_interface = omni.kit.raycast.query.acquire_raycast_query_interface()
            self.debug_draw_interface = _debug_draw.acquire_debug_draw_interface()
            print("✅ Raycast and debug draw interfaces acquired")
            return True
        except Exception as e:
            print(f"❌ Failed to acquire interfaces: {e}")
            return False
    
    def draw_ray(self, start, end, color=(1.0, 0.0, 0.0, 1.0), thickness=3.0):
        """繪製射線"""
        if self.debug_draw_interface:
            self.debug_draw_interface.draw_lines([start], [end], [color], [thickness])
    
    def draw_point(self, position, color=(0.0, 1.0, 0.0, 1.0), size=10.0):
        """繪製點"""
        if self.debug_draw_interface:
            self.debug_draw_interface.draw_points([position], [color], [size])
    
    def clear_drawings(self):
        """清除所有繪製"""
        if self.debug_draw_interface:
            self.debug_draw_interface.clear_lines()
            self.debug_draw_interface.clear_points()





def create_scene():
    """創建場景 - 與 test_raycast_isaaclab.py 相同"""
    print("\n📦 Creating scene...")
    stage = omni.usd.get_context().get_stage()
    
    # 燈光
    light_path = "/World/DomeLight"
    light_prim = stage.DefinePrim(light_path, "DomeLight")
    light_prim.GetAttribute("inputs:intensity").Set(1000.0)
    
    dist_light_path = "/World/DistantLight"
    dist_light_prim = stage.DefinePrim(dist_light_path, "DistantLight")
    dist_light_prim.GetAttribute("inputs:intensity").Set(500.0)
    xform = UsdGeom.Xformable(dist_light_prim)
    xform.AddRotateXYZOp().Set(Gf.Vec3f(-45, 45, 0))
    
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
    
    # 射線起點立方體
    cube_path = "/World/TestCube"
    cube_prim = stage.DefinePrim(cube_path, "Cube")
    cube_geom = UsdGeom.Cube(cube_prim)
    cube_geom.GetSizeAttr().Set(1.0)
    xform = UsdGeom.Xformable(cube_prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 1.0))
    cube_geom.GetDisplayColorAttr().Set([(0.5, 0.5, 1.0)])
    UsdPhysics.CollisionAPI.Apply(cube_prim)
    print(f"   ✅ Test cube (ray origin)")
    
    # 固定測試球體
    test_sphere_path = "/World/TestSphere"
    test_sphere_prim = stage.DefinePrim(test_sphere_path, "Sphere")
    test_sphere_geom = UsdGeom.Sphere(test_sphere_prim)
    test_sphere_geom.GetRadiusAttr().Set(0.3)
    xform = UsdGeom.Xformable(test_sphere_prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(2, 0, 0.5))
    test_sphere_geom.GetDisplayColorAttr().Set([(0.0, 1.0, 0.0)])  # 綠色
    UsdPhysics.CollisionAPI.Apply(test_sphere_prim)
    print(f"   ✅ Static test sphere")
    
    # 移動的球體
    moving_sphere_path = "/World/MovingSphere"
    moving_sphere_prim = stage.DefinePrim(moving_sphere_path, "Sphere")
    moving_sphere_geom = UsdGeom.Sphere(moving_sphere_prim)
    moving_sphere_geom.GetRadiusAttr().Set(0.2)
    xform = UsdGeom.Xformable(moving_sphere_prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(0, 2, 1.0))
    moving_sphere_geom.GetDisplayColorAttr().Set([(1.0, 0.0, 0.0)])  # 紅色
    UsdPhysics.CollisionAPI.Apply(moving_sphere_prim)
    print(f"   ✅ 移動球體")
    
    # 預建多個阻擋立方體在不同位置
    # 射線從 (0,0,1.0) 到 (2,0,0.5)，方向向量 (2,0,-0.5)
    # 在 x=1.0 時 z = 1.0 + (1.0/2.0)*(-0.5) = 0.75
    # 在 x=0.5 時 z = 1.0 + (0.5/2.0)*(-0.5) = 0.875
    # 在 x=1.5 時 z = 1.0 + (1.5/2.0)*(-0.5) = 0.625
    blocker_positions = {
        'blocker_center': Gf.Vec3d(1.0, 0.0, 0.75),
        'blocker_left': Gf.Vec3d(0.5, 0.0, 0.875),
        'blocker_right': Gf.Vec3d(1.5, 0.0, 0.625),
    }
    
    blocker_paths = {}
    for blocker_name, position in blocker_positions.items():
        blocker_path = f"/World/{blocker_name}"
        blocker_prim = stage.DefinePrim(blocker_path, "Cube")
        blocker_geom = UsdGeom.Cube(blocker_prim)
        blocker_geom.GetSizeAttr().Set(0.5)  # 加大尺寸確保能擋到射線
        xform = UsdGeom.Xformable(blocker_prim)
        xform.AddTranslateOp().Set(position)
        blocker_geom.GetDisplayColorAttr().Set([(1.0, 0.0, 0.0)])  # 紅色
        UsdPhysics.CollisionAPI.Apply(blocker_prim)
        blocker_paths[blocker_name] = blocker_path
        print(f"   ✅ 預建阻擋立方體: {blocker_name} @ ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.3f})")
    
    return {
        'cube': cube_path,
        'test_sphere': test_sphere_path,
        'moving_sphere': moving_sphere_path,
        'blockers': blocker_paths
    }


def create_blocking_cube(prim_path: str, position: np.ndarray):
    """Create a blocking cube with collision enabled at runtime"""
    stage = omni.usd.get_context().get_stage()
    
    # 確保不存在重複的 prim
    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(prim_path)
    
    cube_prim = UsdGeom.Cube.Define(stage, prim_path)
    cube_prim.CreateSizeAttr(0.2)
    
    # 設定位置 - 使用 XformOp 而不是 XformCommonAPI
    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    
    # 添加 CollisionAPI (不添加 RigidBodyAPI 以避免 GPU API 衝突)
    UsdPhysics.CollisionAPI.Apply(prim)
    
    # 設定顏色為紅色
    geom = UsdGeom.Cube(prim)
    geom.GetDisplayColorAttr().Set([(1.0, 0.0, 0.0)])
    
    print(f"✅ Created blocking cube at {prim_path}: {position}")
    return prim_path


def update_moving_sphere_position(sphere_path, time_elapsed):
    """更新移動球體的位置"""
    stage = omni.usd.get_context().get_stage()
    sphere_prim = stage.GetPrimAtPath(sphere_path)
    
    if sphere_prim.IsValid():
        # 繞圈運動
        radius = 1.5
        x = radius * math.cos(time_elapsed)
        y = radius * math.sin(time_elapsed)
        z = 1.0 + 0.3 * math.sin(time_elapsed * 2)
        
        xform = UsdGeom.Xformable(sphere_prim)
        # 找到 translate op 並更新（避免觸發 GPU API）
        translate_op = None
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        
        if translate_op is not None:
            # 只更新 USD 屬性，不觸發 PhysX 直接 API
            translate_op.Set(Gf.Vec3d(x, y, z))


def get_sphere_position(sphere_path):
    """獲取球體的實際位置"""
    stage = omni.usd.get_context().get_stage()
    sphere_prim = stage.GetPrimAtPath(sphere_path)
    
    if sphere_prim.IsValid():
        xform = UsdGeom.Xformable(sphere_prim)
        local_transform = xform.ComputeLocalToWorldTransform(0)
        pos = local_transform.ExtractTranslation()
        return np.array([pos[0], pos[1], pos[2]])
    return np.array([0, 0, 0])


def main():
    print("\n" + "="*70)
    print("🎯 Test Fixed and Moving Sphere Raycast")
    print("="*70 + "\n")
    
    # 建立模擬
    sim_cfg = sim_utils.SimulationCfg(
        device="cuda:0",
        dt=1/60.0,
        gravity=(0.0, 0.0, -9.81)
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(5, 5, 3), target=(0, 0, 1))
    
    # 創建場景
    prim_paths = create_scene()
    
    # 初始化 raycast 和 debug draw
    tester = RaycastTester()
    if not tester.initialize():
        return
    
    # 重置並等待初始化
    sim.reset()
    print("⏳ Initializing physics...")
    for _ in range(30):
        sim.step()
    print("✅ Physics initialized\n")
    
    # 測試統計
    hit_count_fixed = 0
    hit_count_moving = 0
    total_rays = 0
    
    # 主循環
    elapsed_time = 0.0
    dt = 1/60.0
    test_duration = args_cli.test_duration
    step_count = 0
    
    print("🎬 Starting raycast test...\n")
    
    while elapsed_time < test_duration:
        # 清除之前的繪製
        tester.clear_drawings()
        
        # 獲取位置
        ray_origin = get_sphere_position(prim_paths['cube'])
        fixed_sphere_pos = get_sphere_position(prim_paths['test_sphere'])
        
        # 更新移動球體
        update_moving_sphere_position(prim_paths['moving_sphere'], elapsed_time)
        moving_sphere_pos = get_sphere_position(prim_paths['moving_sphere'])
        
        # 執行射線測試
        rays_data = []
        
        # 射線1：指向固定球體
        direction1 = fixed_sphere_pos - ray_origin
        dist1 = np.linalg.norm(direction1)
        if dist1 > 0:
            direction1 = direction1 / dist1
        
        hit_info1 = {'hit': False, 'path': None}
        def on_hit1(ray, hit_info):
            if hit_info and hit_info.valid:
                hit_info1['hit'] = True
                hit_info1['path'] = str(hit_info.get_target_usd_path())
        
        ray1 = Ray(tuple(ray_origin), tuple(ray_origin + direction1 * 5.0))
        tester.raycast_interface.submit_raycast_query(ray1, on_hit1)
        rays_data.append((ray_origin, fixed_sphere_pos, hit_info1, 'fixed'))
        
        # 射線2：指向移動球體
        direction2 = moving_sphere_pos - ray_origin
        dist2 = np.linalg.norm(direction2)
        if dist2 > 0:
            direction2 = direction2 / dist2
        
        hit_info2 = {'hit': False, 'path': None}
        def on_hit2(ray, hit_info):
            if hit_info and hit_info.valid:
                hit_info2['hit'] = True
                hit_info2['path'] = str(hit_info.get_target_usd_path())
        
        ray2 = Ray(tuple(ray_origin), tuple(ray_origin + direction2 * 5.0))
        tester.raycast_interface.submit_raycast_query(ray2, on_hit2)
        rays_data.append((ray_origin, moving_sphere_pos, hit_info2, 'moving'))
        
        # 步進模擬（這樣 raycast 才會被處理）
        sim.step()
        
        # 繪製射線和點
        for start, end, hit_info, label in rays_data:
            if hit_info.get('hit'):
                color = (0.0, 1.0, 0.0, 1.0)  # 綠色 - 命中
                if 'fixed' in label:
                    hit_count_fixed += 1
                else:
                    hit_count_moving += 1
            else:
                color = (1.0, 0.0, 0.0, 1.0)  # 紅色 - 未命中
            
            # 繪製射線
            tester.draw_ray(tuple(start), tuple(end), color=color, thickness=3.0)
            # 繪製起點（藍色）
            tester.draw_point(tuple(start), color=(0.0, 0.0, 1.0, 1.0), size=8.0)
            # 繪製終點（黃色）
            tester.draw_point(tuple(end), color=(1.0, 1.0, 0.0, 1.0), size=10.0)
        
        # 步進時間
        elapsed_time += dt
        step_count += 1
        
        # 每秒打印一次狀態
        if step_count % 60 == 0:
            rate_fixed = (hit_count_fixed / (step_count // 60)) * 100
            rate_moving = (hit_count_moving / (step_count // 60)) * 100
            print(f"⏱️  {elapsed_time:.1f}s")
            print(f"   固定球體: {hit_count_fixed} hits ({rate_fixed:.1f}%)")
            print(f"   移動球體: {hit_count_moving} hits ({rate_moving:.1f}%)")
            print(f"   移動球體位置: [{moving_sphere_pos[0]:.2f}, {moving_sphere_pos[1]:.2f}, {moving_sphere_pos[2]:.2f}]")
    
    # 最終統計
    print("\n" + "="*70)
    print("📊 Test Complete")
    print("="*70)
    print(f"固定球體命中: {hit_count_fixed}/{step_count} ({hit_count_fixed/step_count*100:.1f}%)")
    print(f"移動球體命中: {hit_count_moving}/{step_count} ({hit_count_moving/step_count*100:.1f}%)")
    
    if hit_count_fixed > 0:
        print("\n✅ 固定球體能被擊中 - raycast 工作正常")
    else:
        print("\n❌ 固定球體也不能被擊中 - raycast 或碰撞體有問題")
    
    print("="*70 + "\n")
    
    # ========== 方案 1: 預建阻擋立方體 - 通過啟用/禁用來控制 ==========
    print("\n" + "="*70)
    print("🧪 方案 1: 預建阻擋立方體 - 使用可見性切換")
    print("="*70)
    
    # 獲取 stage
    stage = omni.usd.get_context().get_stage()
    
    # 首先確保所有阻擋器都是可見的
    print("初始化阻擋器... 所有都設為可見")
    for blocker_name, blocker_path in prim_paths['blockers'].items():
        blocker_prim = stage.GetPrimAtPath(blocker_path)
        if blocker_prim.IsValid():
            blocker_prim.GetAttribute("visibility").Set("inherited")
    
    # 物理更新
    for _ in range(30):
        sim.step()
    
    print("✅ 所有阻擋器已初始化為可見\n")
    
    # 簡單的兩個配置測試
    test_configs = [
        {'name': '所有阻擋器可見', 'visible': True},
        {'name': '所有阻擋器隱藏', 'visible': False},
    ]
    
    results = []
    
    for config in test_configs:
        print(f"🔄 測試配置: {config['name']}")
        print("-" * 50)
        
        # 設定所有阻擋器的可見性
        for blocker_name, blocker_path in prim_paths['blockers'].items():
            blocker_prim = stage.GetPrimAtPath(blocker_path)
            if blocker_prim.IsValid():
                visibility = "inherited" if config['visible'] else "invisible"
                blocker_prim.GetAttribute("visibility").Set(visibility)
        
        # 物理更新確保可見性變化生效
        for _ in range(15):
            sim.step()
        
        # 執行 180 幀測試（3 秒）
        hit_sphere = 0      # 射線打到目標球體
        hit_blocker = 0      # 射線打到阻擋器
        hit_other = 0        # 射線打到其他物件
        miss = 0             # 射線未命中
        
        for frame in range(180):
            tester.clear_drawings()
            
            # 獲取當前位置
            ray_origin = get_sphere_position(prim_paths['cube'])
            fixed_pos = get_sphere_position(prim_paths['test_sphere'])
            
            # 計算方向
            direction = fixed_pos - ray_origin
            dist = np.linalg.norm(direction)
            
            if dist > 0.5:  # 只在有效距離時查詢
                direction = direction / dist
                
                # 建立射線
                ray = Ray(tuple(ray_origin), tuple(ray_origin + direction * 5.0))
                
                # 提交查詢 - 記錄打到什麼
                hit_result = {'valid': False, 'path': ''}
                def on_hit(ray_hit, hit_info):
                    if hit_info and hit_info.valid:
                        hit_result['valid'] = True
                        hit_result['path'] = str(hit_info.get_target_usd_path())
                
                tester.raycast_interface.submit_raycast_query(ray, on_hit)
                
                # 步進讓 raycast 處理
                sim.step()
                
                # 分類結果
                if hit_result['valid']:
                    path = hit_result['path']
                    if 'TestSphere' in path:
                        hit_sphere += 1
                    elif 'blocker' in path:
                        hit_blocker += 1
                    else:
                        hit_other += 1
                else:
                    miss += 1
                
                # 繪製
                if hit_result['valid']:
                    if 'blocker' in hit_result['path']:
                        color = (1.0, 0.5, 0.0, 1.0)  # 橘色 - 被阻擋
                    else:
                        color = (0.0, 1.0, 0.0, 1.0)  # 綠色 - 打到目標
                else:
                    color = (1.0, 0.0, 0.0, 1.0)  # 紅色 - 未命中
                tester.draw_ray(tuple(ray_origin), tuple(fixed_pos), color=color, thickness=3.0)
        
        # 計算統計
        total = hit_sphere + hit_blocker + hit_other + miss
        
        result = {
            'config': config['name'],
            'visible': config['visible'],
            'hit_sphere': hit_sphere,
            'hit_blocker': hit_blocker,
            'hit_other': hit_other,
            'miss': miss,
            'total': total,
        }
        results.append(result)
        
        print(f"   🎯 打到目標球體: {hit_sphere}/{total} ({hit_sphere/total*100:.1f}%)")
        print(f"   🚧 打到阻擋器:   {hit_blocker}/{total} ({hit_blocker/total*100:.1f}%)")
        print(f"   🔘 打到其他物件: {hit_other}/{total} ({hit_other/total*100:.1f}%)")
        print(f"   ❌ 未命中:       {miss}/{total} ({miss/total*100:.1f}%)")
        print(f"   可見性: {'可見' if config['visible'] else '隱藏'}\n")
    
    # 顯示結果
    print("="*70)
    print("📊 方案 1 測試結果")
    print("="*70)
    print(f"{'配置':<20} {'打到球體':<12} {'打到阻擋':<12} {'其他':<8} {'未命中':<8}")
    print("-" * 70)
    
    for r in results:
        t = r['total']
        print(f"{r['config']:<20} {r['hit_sphere']}/{t} ({r['hit_sphere']/t*100:4.1f}%)  "
              f"{r['hit_blocker']}/{t} ({r['hit_blocker']/t*100:4.1f}%)  "
              f"{r['hit_other']:<6}  {r['miss']:<6}")
    
    # 分析
    print("\n" + "="*70)
    print("🔍 分析結果:")
    print("="*70)
    
    if len(results) >= 2:
        visible_r = results[0]
        hidden_r = results[1]
        
        print(f"\n阻擋器可見時:")
        print(f"   🎯 打到目標球體: {visible_r['hit_sphere']}/{visible_r['total']}")
        print(f"   🚧 被阻擋器攔截: {visible_r['hit_blocker']}/{visible_r['total']}")
        
        print(f"\n阻擋器隱藏時:")
        print(f"   🎯 打到目標球體: {hidden_r['hit_sphere']}/{hidden_r['total']}")
        print(f"   🚧 被阻擋器攔截: {hidden_r['hit_blocker']}/{hidden_r['total']}")
        
        # 關鍵比較
        blocked_visible = visible_r['hit_blocker']
        blocked_hidden = hidden_r['hit_blocker']
        sphere_visible = visible_r['hit_sphere']
        sphere_hidden = hidden_r['hit_sphere']
        
        print(f"\n📊 關鍵比較:")
        print(f"   打到目標球體差異: {sphere_hidden} - {sphere_visible} = {sphere_hidden - sphere_visible}")
        print(f"   被阻擋次數差異:   {blocked_visible} - {blocked_hidden} = {blocked_visible - blocked_hidden}")
        
        if blocked_visible > blocked_hidden + 5:
            print(f"\n✅ 結論: 方案 1 有效！")
            print(f"   阻擋器可見時成功攔截了 {blocked_visible} 次射線")
            print(f"   隱藏後只攔截了 {blocked_hidden} 次")
            print(f"   可見性切換可以有效控制射線阻擋！")
        elif blocked_visible > 0:
            print(f"\n⚠️  結論: 阻擋器有部分效果")
            print(f"   可見時攔截 {blocked_visible} 次，隱藏時 {blocked_hidden} 次")
            print(f"   可見性切換的影響需要進一步測試")
        else:
            print(f"\n❌ 結論: 預建方案無效")
            print(f"   阻擋器從未攔截過任何射線")
            print(f"   raycast 可能在 sim.reset() 時已固化加速結構")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
