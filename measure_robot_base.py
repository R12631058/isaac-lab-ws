"""
測量 extended_needle_holder.usd 中 robot base 的世界座標
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, Gf

# 載入 USD
usd_path = r"C:\Nick\surgery_team\surgery_team\USD\isaaclab\extended_needle_holder.usd"
stage = Usd.Stage.Open(usd_path)

print("=" * 60)
print("測量 Robot Base 的世界座標")
print("=" * 60)

# 檢查 robot base 路徑
robot_paths = [
    "/Root/robotarm_base",
    "/Root/robotarm_base/robotarm_base",
    "/Root/robotarm_base/robotarm_base/tm5_700",
    "/Root/robotarm_base/robotarm_base/tm5_700/base",
]

for path in robot_paths:
    prim = stage.GetPrimAtPath(path)
    if prim:
        print(f"\n路徑: {path}")
        
        # 取得 world transform
        xformable = UsdGeom.Xformable(prim)
        if xformable:
            world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            translation = world_transform.ExtractTranslation()
            print(f"  世界座標: X={translation[0]:.4f}, Y={translation[1]:.4f}, Z={translation[2]:.4f}")
    else:
        print(f"\n路徑不存在: {path}")

# 也測量 phantom 和 frame_prim 的世界座標
print("\n" + "=" * 60)
print("Phantom 和 frame_prim 的世界座標（供對比）")
print("=" * 60)

other_paths = [
    "/Root/phantom",
    "/Root/frame_prim",
]

for path in other_paths:
    prim = stage.GetPrimAtPath(path)
    if prim:
        print(f"\n路徑: {path}")
        xformable = UsdGeom.Xformable(prim)
        if xformable:
            world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            translation = world_transform.ExtractTranslation()
            print(f"  世界座標: X={translation[0]:.4f}, Y={translation[1]:.4f}, Z={translation[2]:.4f}")

# 計算相對座標
print("\n" + "=" * 60)
print("計算 frame_prim 相對於 robot base 的座標")
print("=" * 60)

robot_base_path = "/Root/robotarm_base/robotarm_base/tm5_700"
frame_prim_path = "/Root/frame_prim"

robot_prim = stage.GetPrimAtPath(robot_base_path)
frame_prim = stage.GetPrimAtPath(frame_prim_path)

if robot_prim and frame_prim:
    robot_xform = UsdGeom.Xformable(robot_prim)
    frame_xform = UsdGeom.Xformable(frame_prim)
    
    robot_world = robot_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    frame_world = frame_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    
    robot_pos = robot_world.ExtractTranslation()
    frame_pos = frame_world.ExtractTranslation()
    
    # 相對座標 = frame_prim 世界座標 - robot base 世界座標
    rel_x = frame_pos[0] - robot_pos[0]
    rel_y = frame_pos[1] - robot_pos[1]
    rel_z = frame_pos[2] - robot_pos[2]
    
    print(f"\nRobot Base 世界座標: ({robot_pos[0]:.4f}, {robot_pos[1]:.4f}, {robot_pos[2]:.4f})")
    print(f"frame_prim 世界座標: ({frame_pos[0]:.4f}, {frame_pos[1]:.4f}, {frame_pos[2]:.4f})")
    print(f"\nframe_prim 相對於 Robot Base 的座標:")
    print(f"  rel_X = {rel_x:.4f}")
    print(f"  rel_Y = {rel_y:.4f}")
    print(f"  rel_Z = {rel_z:.4f}")
    
    # Bounding box 範圍轉換
    print("\n" + "=" * 60)
    print("frame_prim bounding box 相對於 Robot Base 的範圍")
    print("=" * 60)
    
    # 原始世界座標 bounding box
    world_x_min, world_x_max = -0.5308, -0.3558
    world_y_min, world_y_max = -0.8260, -0.6510
    world_z_min, world_z_max = 1.0649, 1.2399
    
    # 轉換到 robot base 座標系
    rel_x_min = world_x_min - robot_pos[0]
    rel_x_max = world_x_max - robot_pos[0]
    rel_y_min = world_y_min - robot_pos[1]
    rel_y_max = world_y_max - robot_pos[1]
    rel_z_min = world_z_min - robot_pos[2]
    rel_z_max = world_z_max - robot_pos[2]
    
    print(f"\n相對於 Robot Base 的 pos_x 範圍: ({rel_x_min:.4f}, {rel_x_max:.4f})")
    print(f"相對於 Robot Base 的 pos_y 範圍: ({rel_y_min:.4f}, {rel_y_max:.4f})")
    print(f"相對於 Robot Base 的 pos_z 範圍: ({rel_z_min:.4f}, {rel_z_max:.4f})")
    
    # Phantom 表面 Z = 1.081 的相對座標
    phantom_z_top = 1.081
    rel_phantom_z = phantom_z_top - robot_pos[2]
    print(f"\nPhantom 表面 (Z=1.081) 相對於 Robot Base 的 Z = {rel_phantom_z:.4f}")

print("\n" + "=" * 60)
print("完成測量")
print("=" * 60)
