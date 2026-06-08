#!/usr/bin/env python3
"""
測量 extension_link_out.usd 中 needle_tip body 的位置

目的：確認 needle_tip rigid body 是 articulation 的一部分，並測量其位置
"""

import argparse
from isaaclab.app import AppLauncher

# 解析命令行參數
parser = argparse.ArgumentParser(description="Measure needle_tip body position")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

# 啟動應用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 導入必要模組
import torch
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

# 場景配置
@configclass
class MeasureSceneCfg(InteractiveSceneCfg):
    """測量場景配置"""
    
    surgery_room = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Root",
        spawn=sim_utils.UsdFileCfg(
            usd_path="C:\\Nick\\surgery_team\\surgery_team\\USD\\isaaclab\\extension_link_out.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Root/robotarm_base/robotarm_base/tm5_700",
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "joint_1": -1.656318,
                "joint_2": 0.757474,
                "joint_3": 1.003564,
                "joint_4": 1.368337,
                "joint_5": -0.062832,
                "joint_6": -1.570797,
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


def main():
    # 設置仿真
    sim_cfg = sim_utils.SimulationCfg(dt=1/60)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.0, 3.0, 3.0], target=[0.0, 0.0, 0.0])
    
    # 創建場景
    scene_cfg = MeasureSceneCfg(num_envs=1, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)
    
    # 重置並運行仿真
    sim.reset()
    scene.reset()
    
    # 等待幾幀讓場景穩定
    for _ in range(10):
        sim.step()
        scene.update(sim.get_physics_dt())
    
    # 獲取機器人
    robot = scene["robot"]
    
    print("\n" + "=" * 60)
    print("📊 Robot Articulation Body 列表")
    print("=" * 60)
    
    # 列出所有 body names
    body_names = robot.data.body_names
    print(f"\n總共 {len(body_names)} 個 bodies:\n")
    
    for i, name in enumerate(body_names):
        # 獲取該 body 的世界座標
        body_pos = robot.data.body_state_w[0, i, :3].cpu().numpy()
        print(f"  [{i:2d}] {name:30s} - 世界座標: ({body_pos[0]:.4f}, {body_pos[1]:.4f}, {body_pos[2]:.4f})")
    
    print("\n" + "=" * 60)
    print("🔍 尋找 needle 相關 bodies")
    print("=" * 60)
    
    # 尋找包含 "needle" 的 body
    needle_bodies = [(i, name) for i, name in enumerate(body_names) if "needle" in name.lower()]
    
    if needle_bodies:
        print(f"\n找到 {len(needle_bodies)} 個 needle 相關 bodies:\n")
        for idx, name in needle_bodies:
            body_pos = robot.data.body_state_w[0, idx, :3].cpu().numpy()
            body_quat = robot.data.body_state_w[0, idx, 3:7].cpu().numpy()
            print(f"  [{idx}] {name}")
            print(f"      位置: ({body_pos[0]:.6f}, {body_pos[1]:.6f}, {body_pos[2]:.6f})")
            print(f"      四元數: ({body_quat[0]:.6f}, {body_quat[1]:.6f}, {body_quat[2]:.6f}, {body_quat[3]:.6f})")
    else:
        print("\n⚠️  未找到包含 'needle' 的 body！")
    
    # 獲取 robot base 位置
    robot_base_pos = robot.data.root_pos_w[0].cpu().numpy()
    print(f"\n\n📍 Robot Base 世界座標: ({robot_base_pos[0]:.6f}, {robot_base_pos[1]:.6f}, {robot_base_pos[2]:.6f})")
    
    # 計算相對位置
    if needle_bodies:
        print("\n" + "=" * 60)
        print("📐 相對於 Robot Base 的位置")
        print("=" * 60)
        
        for idx, name in needle_bodies:
            body_pos = robot.data.body_state_w[0, idx, :3].cpu().numpy()
            rel_pos = body_pos - robot_base_pos
            print(f"\n  {name}:")
            print(f"    相對位置: ({rel_pos[0]:.6f}, {rel_pos[1]:.6f}, {rel_pos[2]:.6f})")
    
    # 檢查是否有 needle_tip
    needle_tip_found = any("needle_tip" in name.lower() for _, name in needle_bodies)
    
    print("\n" + "=" * 60)
    print("✅ 結論")
    print("=" * 60)
    
    if needle_tip_found:
        print("\n✅ needle_tip body 存在於 articulation 中！")
        print("   可以直接使用 body_names=['needle_tip'] 而不需要 offset")
    else:
        print("\n❌ needle_tip body 不在 articulation 中")
        print("   需要確認 USD 結構或使用其他方法")
    
    print("\n")
    
    # 關閉仿真
    simulation_app.close()


if __name__ == "__main__":
    main()
