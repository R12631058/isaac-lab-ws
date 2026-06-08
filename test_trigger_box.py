"""
測試腳本：載入手術室 USD 場景，初始化 trigger volume 並每 30 frame 印出觸發器內的物件

執行方式：
    .\isaaclab.bat -p scripts\isaaclab_ws\test_trigger_box.py
    .\isaaclab.bat -p scripts\isaaclab_ws\test_trigger_box.py --hold_frames 600
    .\isaaclab.bat -p scripts\isaaclab_ws\test_trigger_box.py --trigger_path "/World/Marker/surgery_room/NDI/NDI_mesh/VegaST_XT_cam/node_/volume"
"""

import argparse
import sys
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test Trigger Box in Surgery Room USD")
parser.add_argument("--usd_path", type=str,
                    default=r"C:\Nick\surgery_team\surgery_team\USD\surgeryroom_lostfunc.usd",
                    help="Path to the surgery room USD file")
parser.add_argument("--trigger_path", type=str,
                    default="/Root/NDI_02/mesh_",
                    help="USD path to the trigger volume prim")
parser.add_argument("--hold_frames", type=int, default=600,
                    help="Total frames to run (default: 600 = 10 seconds at 60fps)")
parser.add_argument("--check_interval", type=int, default=30,
                    help="Check trigger every N frames (default: 30)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 啟動模擬器
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 模擬器啟動後 ---
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import omni.usd
from pxr import UsdPhysics, PhysxSchema
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext


def initialize_trigger_volume(trigger_path: str):
    """初始化觸發體積，回傳 state_api 或 None"""
    stage = omni.usd.get_context().get_stage()
    trigger_prim = stage.GetPrimAtPath(trigger_path)

    if not trigger_prim.IsValid():
        print(f"[ERROR] Trigger volume not found at: {trigger_path}")
        return None

    print(f"[OK] Found trigger volume at: {trigger_path}")

    # 套用必要的 PhysX API
    if not trigger_prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(trigger_prim)
        print("  Applied UsdPhysics.CollisionAPI")

    if not trigger_prim.HasAPI(PhysxSchema.PhysxTriggerAPI):
        PhysxSchema.PhysxTriggerAPI.Apply(trigger_prim)
        print("  Applied PhysxSchema.PhysxTriggerAPI")

    if not trigger_prim.HasAPI(PhysxSchema.PhysxTriggerStateAPI):
        state_api = PhysxSchema.PhysxTriggerStateAPI.Apply(trigger_prim)
        print("  Applied PhysxSchema.PhysxTriggerStateAPI")
    else:
        state_api = PhysxSchema.PhysxTriggerStateAPI(trigger_prim)

    print("[OK] Trigger volume initialized")
    return state_api


def get_trigger_objects(state_api):
    """回傳目前在觸發器內的物件路徑集合"""
    if state_api is None:
        return set()
    try:
        targets = state_api.GetTriggeredCollisionsRel().GetTargets()
        return {str(p) for p in targets}
    except Exception as e:
        print(f"[ERROR] get_trigger_objects: {e}")
        return set()


def main():
    print("\n" + "=" * 60)
    print(" Trigger Box Test - Surgery Room USD")
    print("=" * 60)

    # 1. SimulationContext
    sim_cfg = sim_utils.SimulationCfg(dt=1/60, device="cpu", gravity=(0.0, 0.0, -9.81))
    sim = SimulationContext(sim_cfg)

    # 2. 載入 USD
    print(f"\nLoading USD: {args_cli.usd_path}")
    if not os.path.exists(args_cli.usd_path):
        print("[ERROR] USD file not found")
        simulation_app.close()
        return

    omni.usd.get_context().open_stage(args_cli.usd_path)

    # 等待場景載入
    for _ in range(10):
        sim.step()
        time.sleep(0.05)

    sim.reset()
    for _ in range(30):
        sim.step()

    # 3. 初始化 trigger volume
    state_api = initialize_trigger_volume(args_cli.trigger_path)
    if state_api is None:
        print("\n[WARN] Trigger volume unavailable. Will still run simulation.\n")

    # 等待穩定
    for _ in range(30):
        sim.step()

    # 4. 主迴圈
    print(f"\nRunning for {args_cli.hold_frames} frames ({args_cli.hold_frames/60:.1f}s)")
    print(f"Checking trigger every {args_cli.check_interval} frames")
    print("-" * 60)

    frame = 0
    try:
        while frame < args_cli.hold_frames and simulation_app.is_running():
            sim.step()
            frame += 1

            if frame % args_cli.check_interval == 0:
                objects = get_trigger_objects(state_api)
                t = frame / 60.0
                print(f"\n[Frame {frame:4d} | {t:.1f}s] Objects in trigger: {len(objects)}")
                if objects:
                    for obj_path in sorted(objects):
                        # 顯示簡短路徑
                        short = obj_path.split("/")[-1] if "/" in obj_path else obj_path
                        print(f"  -> {short}  ({obj_path})")
                else:
                    print("  (empty)")

    except KeyboardInterrupt:
        print("\n[Interrupted]")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)
    simulation_app.close()


if __name__ == "__main__":
    main()
