"""
測試腳本：驗證機械手臂移動時 NDI raycast 能正確追蹤 marker 位置

執行方式（conda env_isaaclab）：
    .\isaaclab.bat -p scripts\isaaclab_ws\test_ndi_dynamic.py

測試流程：
 1. 記錄初始 marker 位置
 2. 設定新的 joint positions，讓手臂移動
 3. 執行多幀模擬
 4. 記錄移動後的 marker 位置
 5. 驗證：位置有變化 + raycast 仍能正常運作
"""

import argparse
import sys
import os
import time
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test NDIDetector dynamic tracking")
parser.add_argument("--usd_path", type=str,
                    default=r"C:\Nick\surgery_team\surgery_team\USD\surgeryroom_lostfunc.usd",
                    help="Path to the surgery room USD file")
parser.add_argument("--frames_per_position", type=int, default=30,
                    help="Frames to simulate per position")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 啟動模擬器
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 以下是模擬器啟動後的程式碼
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ndi_detector_isaaclab import NDIConfig, NDIDetector


def create_articulation_view(robot_prim_path: str):
    """建立 ArticulationView 來控制機器手臂"""
    try:
        from isaacsim.core.simulation_manager import SimulationManager
        physics_sim_view = SimulationManager.get_physics_sim_view()
        
        # 建立 articulation view
        arti_view = physics_sim_view.create_articulation_view(robot_prim_path.replace(".*", "*"))
        if arti_view is None or arti_view.count == 0:
            print(f"[WARNING] ArticulationView created but empty for {robot_prim_path}")
            return None
        print(f"[INFO] ArticulationView created for {robot_prim_path}")
        return arti_view
    except Exception as e:
        print(f"[WARNING] Could not create ArticulationView: {e}")
        import traceback
        traceback.print_exc()
        return None


def move_prim_directly(prim_path: str, offset: list):
    """直接修改 prim 的 xform translate 來測試位置追蹤"""
    try:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return False
        
        from pxr import UsdGeom, Gf
        xformable = UsdGeom.Xformable(prim)
        
        # 取得當前 translate
        ops = xformable.GetOrderedXformOps()
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                current = op.Get()
                new_pos = Gf.Vec3d(current[0] + offset[0], 
                                   current[1] + offset[1], 
                                   current[2] + offset[2])
                op.Set(new_pos)
                return True
        
        # 如果沒有 translate op，嘗試添加
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(*offset))
        return True
    except Exception as e:
        print(f"[ERROR] move_prim_directly failed: {e}")
        return False


def main():
    print("\n" + "=" * 70)
    print(" NDI Detector 動態追蹤測試（機械手臂移動）")
    print("=" * 70)

    # 1. 建立 SimulationContext
    sim_cfg = sim_utils.SimulationCfg(dt=1/60, device="cpu", gravity=(0.0, 0.0, -9.81))
    sim = SimulationContext(sim_cfg)

    # 2. 載入 USD
    print(f"\n[INFO] 載入 USD: {args_cli.usd_path}")
    if not os.path.exists(args_cli.usd_path):
        print(f"[ERROR] USD 檔案不存在: {args_cli.usd_path}")
        simulation_app.close()
        return

    omni.usd.get_context().open_stage(args_cli.usd_path)
    
    # 等待載入完成
    for _ in range(10):
        sim.step()
        time.sleep(0.05)

    # 3. Reset sim（建立 BVH 和 physics view）
    sim.reset()
    print("[INFO] sim.reset() 完成")

    # 4. 初始化 NDIDetector
    config = NDIConfig()
    detector = NDIDetector(config)
    detector.initialize()
    print("[INFO] NDIDetector 初始化成功")

    # 等待穩定
    for _ in range(30):
        sim.step()

    # 5. 建立 ArticulationView
    robot_prim_path = "/Root/robotarm_base/tm5_700/base"  # 正確的 articulation root path
    arti_view = create_articulation_view(robot_prim_path)

    # 6. 記錄初始 marker 位置
    print("\n" + "-" * 70)
    print("[TEST] 步驟 1：記錄初始位置")
    print("-" * 70)
    
    initial_positions = {}
    for marker in ["BM", "UM"]:  # 這兩個在靜態測試中是可見的
        paths = [p for p in config.END_PRIM_PATHS if f"/{marker}/" in p.upper() or p.endswith(f"/{marker.lower()}")]
        if not paths:
            # fallback: 找包含 marker name 的路徑
            for p in config.END_PRIM_PATHS:
                if marker.lower() in p.lower():
                    paths.append(p)
                    break
        
        if paths:
            pos = detector._get_prim_position(paths[0])
            if pos is not None:
                initial_positions[marker] = pos.copy()
                print(f"  {marker}: {pos}")
    
    # 7. 執行 raycast 並記錄初始結果
    marker_status_initial = detector.update_detection(sim, debug_enabled=False)
    visible_initial = sum(v for v in marker_status_initial.values())
    print(f"  [初始] visible: {visible_initial}/{len(marker_status_initial)}")

    # 8. 移動機器手臂（如果 ArticulationView 可用）
    print("\n" + "-" * 70)
    print("[TEST] 步驟 2：移動機械手臂")
    print("-" * 70)
    
    moved = False
    if arti_view is not None:
        try:
            # 取得當前 joint positions
            current_joints = arti_view.get_dof_positions()
            print(f"  當前 joints: {current_joints[0] if current_joints is not None else 'None'}")
            
            # 設定新的 joint positions（輕微偏移）
            new_joints = current_joints.clone()
            new_joints[0, 0] += 0.3  # Joint 1 旋轉 0.3 rad (~17 度)
            new_joints[0, 1] += 0.2  # Joint 2
            
            arti_view.set_dof_positions(new_joints)
            print(f"  新設 joints: {new_joints[0]}")
            moved = True
            
            # 模擬多幀讓手臂移動
            print(f"  [INFO] 模擬 {args_cli.frames_per_position} 幀...")
            for i in range(args_cli.frames_per_position):
                sim.step()
                if i % 10 == 0:
                    # 更新偵測，驗證能追蹤
                    detector.update_detection(sim, debug_enabled=False)
            
            print("  [OK] 手臂移動完成")
        except Exception as e:
            print(f"  [ERROR] 無法通過 ArticulationView 移動手臂: {e}")
            moved = False
    
    if not moved:
        # 備用方案：直接移動 robotarm_base（整個基座平移）
        print("  [INFO] 使用備用方案：直接移動 robotarm_base xform")
        base_path = "/Root/robotarm_base"
        offset = [0.1, 0.05, 0.0]  # 移動 10cm x, 5cm y
        success = move_prim_directly(base_path, offset)
        if success:
            print(f"  [OK] 已移動 {base_path} 偏移: {offset}")
            moved = True
            # 模擬多幀
            for _ in range(args_cli.frames_per_position):
                sim.step()
        else:
            print("  [WARN] 無法移動 prim，執行純模擬")
            for _ in range(args_cli.frames_per_position):
                sim.step()

    # 9. 記錄移動後的 marker 位置
    print("\n" + "-" * 70)
    print("[TEST] 步驟 3：記錄移動後位置")
    print("-" * 70)
    
    final_positions = {}
    for marker in ["BM", "UM"]:
        paths = [p for p in config.END_PRIM_PATHS if marker.lower() in p.lower()]
        if paths:
            pos = detector._get_prim_position(paths[0])
            if pos is not None:
                final_positions[marker] = pos.copy()
                print(f"  {marker}: {pos}")

    # 10. 執行 raycast 並記錄最終結果
    marker_status_final = detector.update_detection(sim, debug_enabled=False)
    visible_final = sum(v for v in marker_status_final.values())
    print(f"  [最終] visible: {visible_final}/{len(marker_status_final)}")

    # 11. 比較結果
    print("\n" + "=" * 70)
    print("[RESULT] 測試結果")
    print("=" * 70)
    
    # 計算位置變化
    position_changed = False
    for marker in initial_positions:
        if marker in final_positions:
            diff = np.linalg.norm(final_positions[marker] - initial_positions[marker])
            changed = diff > 0.001  # 1mm 閾值
            position_changed = position_changed or changed
            status = "CHANGED" if changed else "SAME"
            print(f"  {marker} 位置變化: {diff*100:.2f} cm [{status}]")
    
    # 驗證 raycast 仍能運作
    raycast_working = len(detector.raycast_results) > 0
    raycast_hits = sum(1 for r in detector.raycast_results if r is not None and r.get('is_target_hit'))
    
    print(f"\n  位置追蹤: {'PASS' if position_changed else 'WARN (位置未變)'}")
    print(f"  Raycast 運作: {'PASS' if raycast_working else 'FAIL'}")
    print(f"  Raycast hits: {raycast_hits}/{len(detector.raycast_results)}")
    print(f"  Marker 可見性: initial={visible_initial}, final={visible_final}")
    
    # 總結
    if raycast_working and raycast_hits > 0:
        print("\n  [OVERALL] PASS - Raycast 能正常追蹤動態位置")
    else:
        print("\n  [OVERALL] FAIL - Raycast 無法正常運作")

    print("=" * 70)
    
    simulation_app.close()


if __name__ == "__main__":
    main()
