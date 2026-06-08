# Isaac Lab 重新定位測試腳本
# 測試場景：兩個方塊（固定A方塊 + 變動B方塊）的相對位置恢復

"""Launch Isaac Sim Simulator first."""

import argparse
import time
import numpy as np
from typing import Tuple

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Repositioning Test with Two Cubes")
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
import omni.usd
from pxr import Gf, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sim.spawners.from_files import spawn_from_usd


@configclass
class RepositioningSceneCfg(InteractiveSceneCfg):
    """重新定位測試場景配置"""
    
    # 地面（使用 AssetBaseCfg 而不是 RigidObjectCfg）
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    
    # 燈光（使用 AssetBaseCfg）
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    
    # 固定方塊 A（參考點）
    cube_a = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CubeA",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # 紅色
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),  # 固定在原點上方
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    
    # 變動方塊 B（目標物件）
    cube_b = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CubeB",
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 0.15, 0.15),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),  # 藍色
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.3, 0.4),  # 初始相對位置
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


class RepositioningController:
    """重新定位控制器"""
    
    def __init__(self, scene: InteractiveScene):
        self.scene = scene
        self.stage = omni.usd.get_context().get_stage()
        
        # 獲取方塊的引用
        self.cube_a = scene["cube_a"]
        self.cube_b = scene["cube_b"]
        
        # 儲存 B 方塊的原始位置和旋轉
        self.original_b_position = None
        self.original_b_rotation = None
        self.step_counter = 0
        
        print("🤖 重新定位控制器初始化完成")
    
    def print_positions(self):
        """顯示當前兩個方塊的位置"""
        try:
            # 獲取 A 方塊位置
            pos_a = self.cube_a.data.root_pos_w[0].cpu().numpy()
            rot_a = self.cube_a.data.root_quat_w[0].cpu().numpy()
            
            # 獲取 B 方塊位置
            pos_b = self.cube_b.data.root_pos_w[0].cpu().numpy()
            rot_b = self.cube_b.data.root_quat_w[0].cpu().numpy()
            
            print(f"📍 方塊 A 位置: {pos_a}, 旋轉: {rot_a}")
            print(f"📍 方塊 B 位置: {pos_b}, 旋轉: {rot_b}")
            
            # 計算距離
            distance = np.linalg.norm(pos_b - pos_a)
            print(f"📏 A 到 B 的距離: {distance:.3f}")
            
        except Exception as e:
            print(f"❌ 顯示位置失敗: {e}")
    
    def step_1_record_initial_position(self):
        """步驟一：記錄 B 方塊的原始位置"""
        print("\n🔍 步驟一：記錄 B 方塊的原始位置")
        
        # 等待幾步讓物理系統穩定
        time.sleep(1.0)
        
        try:
            # 直接記錄 B 方塊的當前位置和旋轉
            self.original_b_position = self.cube_b.data.root_pos_w[0].clone()
            self.original_b_rotation = self.cube_b.data.root_quat_w[0].clone()
            
            print("✅ 成功記錄 B 方塊的原始位置")
            print("📝 初始狀態:")
            self.print_positions()
            return True
            
        except Exception as e:
            print(f"❌ 記錄位置失敗: {e}")
            return False
    
    def step_2_move_cube_b(self):
        """步驟二：移動 B 方塊到新位置"""
        print("\n🔄 步驟二：移動 B 方塊到新位置")
        
        try:
            # 新位置和旋轉
            new_position = torch.tensor([1.0, -0.5, 0.8], device=self.cube_b.device, dtype=torch.float32)
            
            # 創建四元數旋轉（90度繞X軸）：[w, x, y, z]
            import math
            angle = math.pi / 2  # 90 度
            new_rotation = torch.tensor([math.cos(angle/2), math.sin(angle/2), 0.0, 0.0], 
                                      device=self.cube_b.device, dtype=torch.float32)
            
            # 組合位置和旋轉 [x, y, z, w, x, y, z]
            full_pose = torch.cat([new_position, new_rotation]).unsqueeze(0)
            
            # 使用 Isaac Lab RigidObject 的正確方法
            self.cube_b.write_root_pose_to_sim(full_pose)
            
            # 重置速度以確保物件靜止
            zero_velocity = torch.zeros((1, 6), device=self.cube_b.device, dtype=torch.float32)
            self.cube_b.write_root_velocity_to_sim(zero_velocity)
            
            print("✅ B 方塊已移動到新位置")
            print("📝 移動後狀態:")
            # 等待一下讓變更生效
            time.sleep(0.5)
            self.print_positions()
            return True
            
        except Exception as e:
            print(f"❌ 移動 B 方塊失敗: {e}")
            return False
    
    def step_3_restore_position(self):
        """步驟三：恢復 B 方塊到原始位置"""
        print("\n🎯 步驟三：恢復 B 方塊到原始位置")
        
        if self.original_b_position is None or self.original_b_rotation is None:
            print("❌ 沒有記錄的原始位置")
            return False
        
        try:
            # 組合原始位置和旋轉 [x, y, z, w, x, y, z]
            full_pose = torch.cat([self.original_b_position, self.original_b_rotation]).unsqueeze(0)
            
            # 使用 Isaac Lab 的方法恢復位置
            self.cube_b.write_root_pose_to_sim(full_pose)
            
            # 重置速度
            zero_velocity = torch.zeros((1, 6), device=self.cube_b.device, dtype=torch.float32)
            self.cube_b.write_root_velocity_to_sim(zero_velocity)
            
            print("✅ B 方塊已恢復到原始位置")
            print("� 恢復後狀態:")
            # 等待一下讓變更生效
            time.sleep(0.5)
            self.print_positions()
            return True
            
        except Exception as e:
            print(f"❌ 恢復位置失敗: {e}")
            return False
    
def run_repositioning_test():
    """運行重新定位測試"""
    print("🚀 開始重新定位測試")
    
    # 創建模擬環境
    try:
        sim_cfg = sim_utils.SimulationCfg(dt=1/60, device="cuda:0")
        sim = sim_utils.SimulationContext(sim_cfg)
        
        # 創建場景
        scene_cfg = RepositioningSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
        scene = InteractiveScene(scene_cfg)
        
        print("✅ 場景創建成功")
        
    except Exception as e:
        print(f"❌ 場景創建失敗: {e}")
        return
    
    # 初始化控制器
    try:
        controller = RepositioningController(scene)
        print("✅ 控制器初始化成功")
        
    except Exception as e:
        print(f"❌ 控制器初始化失敗: {e}")
        return
    
    # 運行測試步驟
    try:
        # 重置並運行幾步讓場景穩定
        sim.reset()
        for _ in range(30):
            sim.step()
            scene.update(sim.get_physics_dt())
        
        # 步驟一：記錄初始位置
        if not controller.step_1_record_initial_position():
            print("❌ 步驟一失敗")
            return
        
        # 運行幾步
        for _ in range(30):
            sim.step()
            scene.update(sim.get_physics_dt())
        
        # 步驟二：移動 B 方塊
        if not controller.step_2_move_cube_b():
            print("❌ 步驟二失敗")
            return
        
        # 運行幾步讓移動生效
        for _ in range(60):
            sim.step()
            scene.update(sim.get_physics_dt())
        
        # 步驟三：恢復位置
        if not controller.step_3_restore_position():
            print("❌ 步驟三失敗")
            return
        
        # 運行更多步讓恢復生效
        for _ in range(60):
            sim.step()
            scene.update(sim.get_physics_dt())
        
        print("\n🎉 重新定位測試完成！")
        
        # 保持場景運行一段時間供觀察
        print("🔄 保持場景運行供觀察...")
        for i in range(300):
            sim.step()
            scene.update(sim.get_physics_dt())
            
            if i % 60 == 0:
                print(f"   運行步驟: {i}/300")
        
    except Exception as e:
        print(f"❌ 測試過程失敗: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函數"""
    print("🎯 Isaac Lab 重新定位測試腳本")
    print("📋 測試內容：")
    print("   1. 創建兩個方塊（固定紅色A + 變動藍色B）")
    print("   2. 記錄 A 到 B 的變換矩陣")
    print("   3. 移動 B 方塊到新位置")
    print("   4. 使用變換矩陣恢復 B 的原始相對位置")
    print()
    
    run_repositioning_test()


if __name__ == "__main__":
    # 運行主函數
    main()
    # 關閉模擬
    simulation_app.close()
