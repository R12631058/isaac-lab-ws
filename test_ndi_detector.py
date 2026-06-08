"""
測試腳本：驗證 ndi_detector_isaaclab.py 的 NDIDetector 模組

執行方式（conda env_isaaclab）：
    .\isaaclab.bat -p scripts\isaaclab_ws\test_ndi_detector.py
    .\isaaclab.bat -p scripts\isaaclab_ws\test_ndi_detector.py --debug
    .\isaaclab.bat -p scripts\isaaclab_ws\test_ndi_detector.py --cost_frames 200

重點驗證項目：
 1. raycast 能回傳非零結果（hit_count > 0）
 2. 至少一個 marker 為 True（FM / HM 最靠近鏡頭，通常最容易看到）
 3. calculate_frame_cost() 回傳合理數值（不全為 0 或 NaN）
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test NDIDetector in Isaac Lab")
parser.add_argument("--usd_path", type=str,
                    default=r"C:\Nick\surgery_team\surgery_team\USD\surgeryroom_lostfunc.usd",
                    help="Path to the surgery room USD file")
parser.add_argument("--test_frames", type=int, default=100,
                    help="Total number of simulation frames to run")
parser.add_argument("--cost_frames", type=int, default=50,
                    help="Number of frames to collect cost data")
parser.add_argument("--debug", action="store_true",
                    help="Enable debug line drawing")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ── 啟動模擬器 ───────────────────────────────────────────
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 以下是模擬器啟動後的程式碼 ───────────────────────────

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import time
import numpy as np

import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

# ndi_detector_isaaclab.py 位於同一目錄
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ndi_detector_isaaclab import NDIConfig, NDIDetector


def main():
    print("\n" + "=" * 60)
    print(" NDIDetector Isaac Lab 整合測試")
    print("=" * 60)

    # ── 1. 建立 SimulationContext ──────────────────────────
    sim_cfg = sim_utils.SimulationCfg(dt=1/60, device="cpu",
                                      gravity=(0.0, 0.0, -9.81))
    sim = SimulationContext(sim_cfg)

    # ── 2. 載入 USD（open_stage 保留 /Root/... 路徑結構）──
    print(f"\n[INFO] 載入 USD: {args_cli.usd_path}")
    if not os.path.exists(args_cli.usd_path):
        print(f"[ERROR] USD 檔案不存在: {args_cli.usd_path}")
        simulation_app.close()
        return

    omni.usd.get_context().open_stage(args_cli.usd_path)
    stage = omni.usd.get_context().get_stage()

    # 等待 USD load 完成
    for _ in range(10):
        sim.step()
        time.sleep(0.05)
    print("[INFO] USD open_stage 完成")
    print(f"[INFO] Root layer: {stage.GetRootLayer().realPath if stage else 'None'}")

    # ── 3. 重置 sim（BVH 在這裡建立）──────────────────────
    sim.reset()
    print("[INFO] sim.reset() 完成，BVH 已建立")

    # ── 4. 初始化 NDIDetector ──────────────────────────────
    config = NDIConfig()
    config.print_paths()

    detector = NDIDetector(config)
    ok = detector.initialize()
    if not ok:
        print("[ERROR] NDIDetector.initialize() 失敗，中止測試")
        simulation_app.close()
        return

    print("[INFO] NDIDetector 初始化成功")

    # ── 5. 等待 USD 穩定（60 幀）─────────────────
    print("[INFO] 等待 USD 初始化（60 幀）...")
    for _ in range(60):
        sim.step()
    print("[INFO] USD 初始化完成")

    # 驗證關鍵 prim 是否存在
    print("\n[INFO] 驗證 prim 位置...")
    for path in config.START_PRIM_PATHS + config.END_PRIM_PATHS[:4]:
        pos = detector._get_prim_position(path)
        ok = "OK" if pos is not None else "MISSING"
        pos_str = f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})" if pos is not None else ""
        print(f"  [{ok}] {path} {pos_str}")

    # ── 6. 啟動成本計算 ────────────────────────────────────
    detector.start_cost_calculation()

    # ── 7. 主迴圈 ─────────────────────────────────────────
    hit_frame_count = 0
    any_visible_count = 0
    TOTAL_FRAMES = args_cli.test_frames
    COST_FRAMES = args_cli.cost_frames

    print(f"\n[INFO] 開始主迴圈（{TOTAL_FRAMES} 幀）...")
    print("-" * 60)

    for frame_i in range(TOTAL_FRAMES):
        # 停止成本計算
        if detector.cost_calculation['calculation_active'] and \
           detector.cost_calculation['frame_count'] >= COST_FRAMES:
            results = detector.stop_cost_calculation()

        # 更新 NDI 偵測
        marker_status = detector.update_detection(sim, debug_enabled=args_cli.debug)

        # 統計有效幀
        total_rays = len(detector.raycast_results)
        valid_hits = sum(1 for r in detector.raycast_results
                         if r is not None and r.get('is_target_hit'))
        any_visible = any(marker_status.values())

        if valid_hits > 0:
            hit_frame_count += 1
        if any_visible:
            any_visible_count += 1

        # 每 20 幀詳細報告
        if frame_i % 20 == 0:
            print(f"[Frame {frame_i:4d}] rays={total_rays}, hits={valid_hits}, "
                  f"visible={sum(marker_status.values())}/{len(marker_status)}")
            data = detector.get_all_marker_data()
            for m, d in data.items():
                sa_str = f"{d['solid_angle']:.4f}" if d['solid_angle'] is not None else "None"
                vol_str = f"{d['volume']:.2f}" if d['volume'] is not None else "None"
                vis_str = "V" if d['visible'] else "-"
                print(f"  {m}: [{vis_str}] solid_angle={sa_str}  volume={vol_str}")

        sim.step()

    # ── 8. 最終報告 ────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" 測試結果摘要")
    print("=" * 60)
    rate_hit = hit_frame_count / TOTAL_FRAMES * 100
    rate_vis = any_visible_count / TOTAL_FRAMES * 100
    print(f"  有效 raycast 幀數: {hit_frame_count}/{TOTAL_FRAMES}  ({rate_hit:.1f}%)")
    print(f"  任意 marker 可見幀: {any_visible_count}/{TOTAL_FRAMES}  ({rate_vis:.1f}%)")

    if rate_hit > 0:
        print("  [PASS] Raycast 有命中結果!")
    else:
        print("  [FAIL] Raycast 0 命中 → 請確認 USD 路徑 / BVH / prim 位置")

    if rate_vis > 0:
        print("  [PASS] 至少一個 marker 被偵測到!")
    else:
        print("  [WARN] 無任何 marker 可見 → 可能被遮擋或路徑錯誤")

    print("=" * 60)

    # 最終成本（若尚未停止）
    if detector.cost_calculation['calculation_active']:
        detector.stop_cost_calculation()

    # 清除 debug draw
    detector.clear_debug_draw()

    print("\n[INFO] 測試完成，關閉模擬器")
    simulation_app.close()


if __name__ == "__main__":
    main()
