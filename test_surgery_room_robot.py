#!/usr/bin/env python3
"""測試手術室場景中的 TM5-700 控制"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import omni.usd
from pxr import Usd, UsdPhysics, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg


@configclass
class SurgeryRoomTestSceneCfg(InteractiveSceneCfg):
    """測試手術室場景配置"""

    # 載入完整手術室
    surgery_room = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SurgeryRoom",
        spawn=sim_utils.UsdFileCfg(
            usd_path="C:/Nick/surgery_team/surgery_team/USD/isaaclab/surgeryroom_isaac_lab.usd",
        ),
    )

    # 嘗試將 USD 中的機械臂配置為 Articulation
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/SurgeryRoom/robotarm_base/tm5_700",
        spawn=None,  # 不生成新的,使用 USD 中已存在的
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "joint_1": 0.0,
                "joint_2": -0.5,
                "joint_3": 0.5,
                "joint_4": 0.0,
                "joint_5": 0.5,
                "joint_6": 0.0,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint_[1-6]"],
                effort_limit=200.0,
                velocity_limit=2.0,
                stiffness=200.0,
                damping=20.0,
            ),
        },
    )

    # 光源
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )


def inspect_articulation(stage, env_path, robot_path):
    """檢查 USD 中的 Articulation 結構"""
    print("\n" + "="*80)
    print(f"🔍 檢查 Articulation: {env_path}{robot_path}")
    print("="*80)
    
    full_path = env_path + robot_path
    robot_prim = stage.GetPrimAtPath(full_path)
    
    if not robot_prim.IsValid():
        print(f"❌ 找不到路徑: {full_path}")
        return
    
    print(f"✅ 找到機械臂 Prim")
    print(f"   類型: {robot_prim.GetTypeName()}")
    
    # 尋找所有關節
    print("\n🔗 關節列表:")
    joints = []
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if full_path in prim_path and prim.IsA(UsdPhysics.Joint):
            joint_name = prim.GetName()
            joint_type = prim.GetTypeName()
            joints.append((joint_name, joint_type, prim_path))
    
    if joints:
        for name, jtype, path in joints:
            print(f"  - {name:20s} [{jtype:25s}] {path}")
    else:
        print("  ❌ 未找到關節")
    
    # 尋找所有 Link
    print("\n🔗 Link 列表:")
    links = []
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if full_path in prim_path and ("link" in prim.GetName().lower() or prim.GetName() == "flange"):
            links.append((prim.GetName(), prim_path))
    
    if links:
        for name, path in links:
            print(f"  - {name:20s} {path}")
    else:
        print("  ❌ 未找到 Links")
    
    # 檢查 target_pose
    print("\n🎯 目標姿態標記:")
    target_path = env_path + "/SurgeryRoom/target_pose/path_1"
    target_prim = stage.GetPrimAtPath(target_path)
    if target_prim.IsValid():
        print(f"  ✅ 找到目標: {target_path}")
        # 嘗試獲取位置
        xform = UsdGeom.Xformable(target_prim)
        if xform:
            local_transform = xform.GetLocalTransformation()
            translation = local_transform.ExtractTranslation()
            print(f"  📍 位置: ({translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f})")
    else:
        print(f"  ❌ 找不到目標: {target_path}")


def main():
    """主函數"""
    
    # 創建場景配置
    scene_cfg = SurgeryRoomTestSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    
    # 創建仿真
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(3.0, 3.0, 2.5), target=(0.0, 0.0, 0.5))
    
    print(f"\n✅ 正在創建場景 ({args_cli.num_envs} 個環境)...")
    
    try:
        # 創建場景
        scene = InteractiveScene(scene_cfg)
        
        print(f"\n✅ 場景創建成功!")
        print(f"   場景物件: {list(scene.keys())}")
        
        # 獲取 stage
        stage = omni.usd.get_context().get_stage()
        
        # 檢查環境 0 的結構
        inspect_articulation(stage, "/World/envs/env_0", "/SurgeryRoom/robotarm_base/tm5_700")
        
        # 重置仿真以初始化 Articulation
        print("\n▶ 初始化仿真...")
        sim.reset()
        
        # 檢查是否成功創建了 robot articulation
        try:
            robot: Articulation = scene["robot"]
            print("\n" + "="*80)
            print("🤖 機械臂 Articulation 資訊")
            print("="*80)
            print(f"✅ Robot articulation 已建立")
            print(f"   Prim 路徑: {robot.cfg.prim_path}")
            print(f"   關節數量: {robot.num_joints}")
            print(f"   關節名稱: {robot.joint_names}")
            print(f"   Body 數量: {robot.num_bodies}")
            print(f"   Body 名稱: {robot.body_names}")
            
            # 測試控制
            print("\n▶ 測試關節控制...")
            sim.reset()
            
            for step in range(100):
                # 簡單的正弦波控制
                target_pos = torch.sin(torch.tensor(step * 0.05)) * 0.3
                joint_targets = torch.zeros((scene.num_envs, robot.num_joints), device=sim.device)
                joint_targets[:, 1] = target_pos  # 控制 joint_2
                
                robot.set_joint_position_target(joint_targets)
                
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim.cfg.dt)
                
                if step % 20 == 0:
                    current_pos = robot.data.joint_pos[0, 1].item()
                    print(f"  Step {step:3d}: joint_2 目標={target_pos:.3f}, 實際={current_pos:.3f}")
            
            print("\n✅ 控制測試完成!")
            
        except KeyError:
            print("\n❌ Robot articulation 建立失敗")
            print("   可能原因:")
            print("   1. USD 中的機械臂路徑不正確")
            print("   2. 機械臂不是標準的 Articulation")
            print("   3. 需要特殊的配置方式")
        
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n💡 可能的解決方案:")
        print("   1. 檢查 USD 檔案是否正確")
        print("   2. 確認機械臂路徑是否正確")
        print("   3. 可能需要手動配置 Articulation")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠ 使用者中斷")
    except Exception as e:
        print(f"\n❌ 致命錯誤: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
