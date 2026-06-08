"""
計算目標位置相對於機器人基座的座標
用於配置 Isaac Lab 的 UniformPoseCommandCfg
"""

import argparse
from isaaclab.app import AppLauncher

# 創建參數解析器
parser = argparse.ArgumentParser(description="計算相對座標")
parser.add_argument("--num_envs", type=int, default=1, help="環境數量")
args_cli = parser.parse_args()

# 啟動 Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 導入必要的模組
from isaaclab.sim import SimulationContext
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
import torch

@configclass
class CalculateSceneCfg(InteractiveSceneCfg):
    """計算用的場景配置"""
    
    surgery_room = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SurgeryRoom",
        spawn=sim_utils.UsdFileCfg(
            usd_path="C:/Nick/surgery_team/surgery_team/USD/isaaclab/surgeryroom_isaac_lab.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )


def main():
    """主函數"""
    
    # 創建仿真上下文
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    
    # 創建場景
    scene_cfg = CalculateSceneCfg(num_envs=1, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)
    
    # 執行一步仿真以初始化所有物件
    sim.reset()
    
    print("\n" + "="*80)
    print("🔍 座標計算工具")
    print("="*80)
    
    # 獲取 prim 路徑
    import omni.isaac.core.utils.stage as stage_utils
    
    # 1. 獲取機器人基座位置
    robot_base_path = "/World/envs/env_0/SurgeryRoom/robotarm_base"
    robot_base_prim = stage_utils.get_prim_at_path(robot_base_path)
    
    if robot_base_prim.IsValid():
        robot_base_translate = robot_base_prim.GetAttribute("xformOp:translate").Get()
        if robot_base_translate is None:
            # 嘗試獲取累積的世界變換
            from pxr import UsdGeom
            xformable = UsdGeom.Xformable(robot_base_prim)
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            robot_base_pos = world_transform.ExtractTranslation()
        else:
            robot_base_pos = robot_base_translate
            
        print(f"\n📍 機器人基座位置 (世界座標):")
        print(f"   Path: {robot_base_path}")
        print(f"   Position: ({robot_base_pos[0]:.4f}, {robot_base_pos[1]:.4f}, {robot_base_pos[2]:.4f})")
    else:
        print(f"\n❌ 找不到機器人基座: {robot_base_path}")
        robot_base_pos = [0, 0, 0]
    
    # 2. 檢查是否有 target_pose 標記
    target_pose_path = "/World/envs/env_0/SurgeryRoom/target_pose"
    target_pose_prim = stage_utils.get_prim_at_path(target_pose_path)
    
    if target_pose_prim.IsValid():
        target_translate = target_pose_prim.GetAttribute("xformOp:translate").Get()
        if target_translate is None:
            from pxr import UsdGeom
            xformable = UsdGeom.Xformable(target_pose_prim)
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            target_pos = world_transform.ExtractTranslation()
        else:
            target_pos = target_translate
            
        print(f"\n🎯 Target Pose 標記位置 (世界座標):")
        print(f"   Path: {target_pose_path}")
        print(f"   Position: ({target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f})")
        
        # 計算相對位置
        relative_x = target_pos[0] - robot_base_pos[0]
        relative_y = target_pos[1] - robot_base_pos[1]
        relative_z = target_pos[2] - robot_base_pos[2]
        
        print(f"\n✅ Target 相對於機器人基座:")
        print(f"   Relative: ({relative_x:.4f}, {relative_y:.4f}, {relative_z:.4f})")
        
    else:
        print(f"\n⚠️  場景中沒有找到 target_pose 標記")
        print(f"   檢查路徑: {target_pose_path}")
        relative_x, relative_y, relative_z = None, None, None
    
    # 3. 使用用戶提供的世界座標計算相對位置
    print("\n" + "="*80)
    print("📊 基於您提供的座標計算")
    print("="*80)
    
    user_world_x = 1.13
    user_world_y = -0.65
    user_world_z = 1.29
    
    print(f"\n您原先確認可達的世界座標:")
    print(f"   World: ({user_world_x:.4f}, {user_world_y:.4f}, {user_world_z:.4f})")
    
    user_relative_x = user_world_x - robot_base_pos[0]
    user_relative_y = user_world_y - robot_base_pos[1]
    user_relative_z = user_world_z - robot_base_pos[2]
    
    print(f"\n✅ 轉換為相對座標:")
    print(f"   Relative: ({user_relative_x:.4f}, {user_relative_y:.4f}, {user_relative_z:.4f})")
    
    # 4. 生成配置代碼
    print("\n" + "="*80)
    print("🔧 建議的配置代碼")
    print("="*80)
    
    radius = 0.2
    
    print(f"""
# 在 tm5_reach_surgery_room_cfg.py 中使用這些值:

center_x = {user_relative_x:.2f}  # 相對於機器人基座
center_y = {user_relative_y:.2f}  # 相對於機器人基座
center_z = {user_relative_z:.2f}  # 相對於機器人基座
radius = {radius}    # 半徑 20cm

self.commands.ee_pose = mdp.UniformPoseCommandCfg(
    asset_name="robot",
    body_name="flange",
    resampling_time_range=(8.0, 8.0),
    debug_vis=True,
    ranges=mdp.UniformPoseCommandCfg.Ranges(
        pos_x=({user_relative_x - radius:.2f}, {user_relative_x + radius:.2f}),  # {user_world_x - radius:.2f} ~ {user_world_x + radius:.2f} (世界座標)
        pos_y=({user_relative_y - radius:.2f}, {user_relative_y + radius:.2f}),  # {user_world_y - radius:.2f} ~ {user_world_y + radius:.2f} (世界座標)
        pos_z=({user_relative_z - radius:.2f}, {user_relative_z + radius:.2f}),  # {user_world_z - radius:.2f} ~ {user_world_z + radius:.2f} (世界座標)
        roll=(0.0, 0.0),
        pitch=(math.pi, math.pi),
        yaw=(-math.pi, math.pi),
    ),
)
""")
    
    print("\n" + "="*80)
    print("📝 說明")
    print("="*80)
    print(f"""
1. 機器人基座在世界座標: ({robot_base_pos[0]:.2f}, {robot_base_pos[1]:.2f}, {robot_base_pos[2]:.2f})
2. 您的目標在世界座標: ({user_world_x:.2f}, {user_world_y:.2f}, {user_world_z:.2f})
3. 相對座標 = 世界座標 - 基座座標
4. Isaac Lab 的 UniformPoseCommandCfg 使用相對座標系統
5. 這樣所有環境的目標都會相對於各自的機器人生成
""")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
