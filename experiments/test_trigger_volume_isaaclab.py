"""
實驗 2: 驗證 Isaac Lab 中的 Trigger Volume 功能
測試目標：確認觸發器體積檢測在 Isaac Lab 環境中正常運作

本腳本會：
1. 創建觸發器體積 (使用 PhysxSchema.PhysxTriggerAPI)
2. 創建測試物體並讓其進入觸發器
3. 使用 PhysxTriggerStateAPI 獲取碰撞物體列表
4. 驗證觸發器是否正確檢測物體進入/離開

這是移植 bestviewpoint_autocost.py 中 _initialize_trigger_volume 和 
get_current_trigger_objects 方法的關鍵驗證

執行方式:
    isaaclab.bat -p scripts/isaaclab_ws/experiments/test_trigger_volume_isaaclab.py
    isaaclab.bat -p scripts/isaaclab_ws/experiments/test_trigger_volume_isaaclab.py --headless
"""

import argparse
from isaaclab.app import AppLauncher

# 解析命令行參數
parser = argparse.ArgumentParser(description="Test Trigger Volume in Isaac Lab")
parser.add_argument("--test_duration", type=float, default=5.0, help="Test duration in seconds")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 啟動模擬器
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===== 以下是模擬器啟動後的代碼 =====

import time
import numpy as np
import omni.usd
from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext


class TriggerVolumeTester:
    """
    Trigger Volume 測試類
    基於 bestviewpoint_autocost.py 中的 _initialize_trigger_volume 和
    get_current_trigger_objects 方法
    """
    
    def __init__(self):
        self.trigger_path = "/World/TriggerVolume"
        self.test_ball_path = "/World/TestBall"
        self.trigger_detector = None
        self.collision_events = []
        
    def create_trigger_volume(self, position=(0, 0, 1.5), size=(2, 2, 2)):
        """
        創建觸發器體積 - 完全按照 bestviewpoint_autocost.py 的方式
        """
        print(f"\n🎯 Creating trigger volume at {self.trigger_path}...")
        
        stage = omni.usd.get_context().get_stage()
        
        # 創建立方體作為觸發器
        trigger_prim = stage.DefinePrim(self.trigger_path, "Cube")
        trigger_geom = UsdGeom.Cube(trigger_prim)
        trigger_geom.GetSizeAttr().Set(1.0)
        
        # 設置變換
        xform = UsdGeom.Xformable(trigger_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        xform.AddScaleOp().Set(Gf.Vec3d(*size))
        
        # 設置為半透明以便觀察
        # (在實際應用中可以設為不可見)
        try:
            from pxr import UsdShade
            # 設置顯示顏色為半透明綠色
            trigger_geom.GetDisplayColorAttr().Set([(0.0, 1.0, 0.0)])
            trigger_geom.GetDisplayOpacityAttr().Set([0.3])
        except:
            pass
        
        print(f"   ✅ Cube geometry created")
        
        # ===== 核心：應用 Physics 和 Trigger API =====
        # 這是與 bestviewpoint_autocost.py 相同的方式
        
        # 1. 應用 Collision API
        if not trigger_prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(trigger_prim)
            print(f"   ✅ UsdPhysics.CollisionAPI applied")
        
        # 2. 應用 PhysxTriggerAPI (將碰撞體變成觸發器)
        if not trigger_prim.HasAPI(PhysxSchema.PhysxTriggerAPI):
            PhysxSchema.PhysxTriggerAPI.Apply(trigger_prim)
            print(f"   ✅ PhysxSchema.PhysxTriggerAPI applied")
        
        # 3. 應用 PhysxTriggerStateAPI (用於查詢碰撞)
        if not trigger_prim.HasAPI(PhysxSchema.PhysxTriggerStateAPI):
            trigger_state_api = PhysxSchema.PhysxTriggerStateAPI.Apply(trigger_prim)
        else:
            trigger_state_api = PhysxSchema.PhysxTriggerStateAPI(trigger_prim)
        print(f"   ✅ PhysxSchema.PhysxTriggerStateAPI applied")
        
        # 設置觸發器檢測器 (與原始代碼相同的結構)
        self.trigger_detector = {
            "path": self.trigger_path,
            "prim": trigger_prim,
            "state_api": trigger_state_api,
            "objects": {},
            "previous_collisions": []
        }
        
        print(f"   Position: {position}")
        print(f"   Size: {size}")
        print(f"✅ Trigger volume created successfully")
        
        return True
    
    def create_test_ball(self, position=(0, 0, 5)):
        """創建測試球體（會掉入觸發器）"""
        print(f"\n⚽ Creating test ball at {self.test_ball_path}...")
        
        stage = omni.usd.get_context().get_stage()
        
        # 創建球體
        ball_prim = stage.DefinePrim(self.test_ball_path, "Sphere")
        ball_geom = UsdGeom.Sphere(ball_prim)
        ball_geom.GetRadiusAttr().Set(0.3)
        
        # 設置顏色
        ball_geom.GetDisplayColorAttr().Set([(1.0, 0.0, 0.0)])  # 紅色
        
        # 設置位置
        xform = UsdGeom.Xformable(ball_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        
        # ===== 添加剛體物理 =====
        UsdPhysics.RigidBodyAPI.Apply(ball_prim)
        print(f"   ✅ UsdPhysics.RigidBodyAPI applied")
        
        UsdPhysics.CollisionAPI.Apply(ball_prim)
        print(f"   ✅ UsdPhysics.CollisionAPI applied")
        
        # 設置質量
        mass_api = UsdPhysics.MassAPI.Apply(ball_prim)
        mass_api.GetMassAttr().Set(1.0)
        print(f"   ✅ Mass set to 1.0 kg")
        
        print(f"   Position: {position}")
        print(f"✅ Test ball created successfully")
        
        return True
    
    def get_current_trigger_objects(self):
        """
        獲取當前在觸發器內的物體
        這是 bestviewpoint_autocost.py 中 get_current_trigger_objects 的完全複製
        """
        if not self.trigger_detector:
            return set()
        
        try:
            state_api = self.trigger_detector["state_api"]
            current_collisions = state_api.GetTriggeredCollisionsRel().GetTargets()
            
            # 轉換為字符串集合
            objects_in_trigger = set()
            for collision_path in current_collisions:
                objects_in_trigger.add(str(collision_path))
            
            return objects_in_trigger
            
        except Exception as e:
            print(f"❌ Error getting trigger objects: {e}")
            return set()
    
    def check_and_record_status(self, step_count, elapsed_time):
        """檢查並記錄觸發器狀態"""
        objects = self.get_current_trigger_objects()
        
        event = {
            'step': step_count,
            'time': elapsed_time,
            'objects_in_trigger': list(objects),
            'count': len(objects)
        }
        self.collision_events.append(event)
        
        return event
    
    def print_current_status(self, event):
        """打印當前狀態"""
        print(f"\n📋 [Step {event['step']}, Time: {event['time']:.2f}s]")
        print(f"   Objects in trigger: {event['count']}")
        for obj_path in event['objects_in_trigger']:
            print(f"   🔹 {obj_path}")


def create_ground():
    """創建地面和燈光"""
    print("\n🏗️ Creating ground plane and lighting...")
    stage = omni.usd.get_context().get_stage()
    
    # 添加燈光 (重要！否則場景會全黑)
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
    
    ground_path = "/World/Ground"
    ground_prim = stage.DefinePrim(ground_path, "Cube")
    ground_geom = UsdGeom.Cube(ground_prim)
    ground_geom.GetSizeAttr().Set(20.0)
    
    # 設置為平面
    xform = UsdGeom.Xformable(ground_prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.5))
    xform.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 0.05))
    
    # 設置顏色
    ground_geom.GetDisplayColorAttr().Set([(0.5, 0.5, 0.5)])
    
    # 添加碰撞
    UsdPhysics.CollisionAPI.Apply(ground_prim)
    
    print(f"   ✅ Ground plane created at {ground_path}")
    return ground_path


def main():
    """主測試流程"""
    print("\n" + "="*60)
    print("🔬 Isaac Lab Trigger Volume Functionality Test")
    print("="*60)
    print("This test validates that trigger volumes work in Isaac Lab")
    print("using the same API as bestviewpoint_autocost.py")
    print("="*60)
    
    # 設置模擬上下文
    # 注意：PhysxTriggerStateAPI 需要 CPU 模式才能正確查詢碰撞
    # GPU Direct API 模式下 GetTriggeredCollisionsRel() 可能不工作
    sim_cfg = sim_utils.SimulationCfg(
        device="cpu",  # 使用 CPU 以支援 TriggerStateAPI
        dt=1/60.0,
        gravity=(0.0, 0.0, -9.81)
    )
    sim = SimulationContext(sim_cfg)
    print("📍 Using CPU physics mode for TriggerStateAPI compatibility")
    
    # 設置相機視角
    sim.set_camera_view(eye=(8, 8, 5), target=(0, 0, 1.5))
    
    # 創建場景元素
    create_ground()
    
    # 創建測試器
    tester = TriggerVolumeTester()
    
    # 創建觸發器體積（在地面上方）
    tester.create_trigger_volume(position=(0, 0, 1.5), size=(2, 2, 2))
    
    # 創建測試球體（在觸發器上方，會掉落）
    tester.create_test_ball(position=(0, 0, 5))
    
    # 重置模擬
    sim.reset()
    print("\n✅ Simulation reset complete")
    
    # 運行模擬並監測觸發器
    print("\n" + "="*60)
    print("🎬 Starting Simulation")
    print("="*60)
    print(f"Monitoring trigger volume for {args_cli.test_duration} seconds...")
    print("The ball should fall into the trigger volume...")
    
    start_time = time.time()
    step_count = 0
    check_interval = 30  # 每 30 步檢查一次
    
    ball_entered_trigger = False
    first_detection_time = None
    
    while simulation_app.is_running():
        elapsed = time.time() - start_time
        
        if elapsed > args_cli.test_duration:
            break
        
        # 步進模擬
        sim.step()
        step_count += 1
        
        # 定期檢查觸發器狀態
        if step_count % check_interval == 0:
            event = tester.check_and_record_status(step_count, elapsed)
            
            # 檢測到球體進入觸發器
            if event['count'] > 0 and not ball_entered_trigger:
                ball_entered_trigger = True
                first_detection_time = elapsed
                print("\n🎉 DETECTION! Ball entered trigger volume!")
                tester.print_current_status(event)
            elif step_count % (check_interval * 5) == 0:  # 每 150 步打印一次狀態
                tester.print_current_status(event)
    
    # ===== 分析結果 =====
    print("\n" + "="*60)
    print("📊 Test Results Analysis")
    print("="*60)
    
    print(f"\nSimulation Statistics:")
    print(f"   Total steps: {step_count}")
    print(f"   Total events recorded: {len(tester.collision_events)}")
    
    # 分析碰撞事件
    triggered_events = [e for e in tester.collision_events if e['count'] > 0]
    
    print(f"\nTrigger Detection:")
    print(f"   Events with objects: {len(triggered_events)}")
    
    if triggered_events:
        first_event = triggered_events[0]
        print(f"   First detection at: Step {first_event['step']}, Time {first_event['time']:.2f}s")
        print(f"   Objects detected: {first_event['objects_in_trigger']}")
    
    # 最終判定
    print("\n" + "="*60)
    print("📋 Final Verdict")
    print("="*60)
    
    if ball_entered_trigger:
        print("✅ SUCCESS: Trigger volume is working in Isaac Lab!")
        print(f"   Ball detected at time: {first_detection_time:.2f}s")
        print("\n💡 You can now port the NDI trigger detection to Isaac Lab")
        print("   The following APIs are confirmed working:")
        print("   - PhysxSchema.PhysxTriggerAPI")
        print("   - PhysxSchema.PhysxTriggerStateAPI")
        print("   - GetTriggeredCollisionsRel().GetTargets()")
    else:
        print("⚠️ WARNING: No trigger events detected")
        print("\nPossible reasons:")
        print("   1. Ball may not have fallen far enough")
        print("   2. Physics simulation may need more time")
        print("   3. Trigger APIs may not be working as expected")
        print("\nTry running with longer duration:")
        print("   --test_duration 10")
    
    print("\n" + "="*60)
    
    # 保持場景打開 (非 headless 模式)
    if not args_cli.headless:
        print("\n⏳ Keeping scene open for observation...")
        additional_time = 3.0
        obs_start = time.time()
        while time.time() - obs_start < additional_time:
            sim.step()
            time.sleep(1/60)


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
