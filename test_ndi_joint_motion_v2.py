"""
測試腳本：驗證機械手臂關節運動後 NDI raycast 能正確追蹤 marker 位置

功能：
1. 手臂維持在目標位置並持續 raycast
2. 開啟 debug draw 視覺化射線
3. 簡化輸出：每 frame 顯示每個 marker 狀態，沒打到 marker 則顯示打到的 prim path
4. Trigger volume 偵測：印出觸發器內的物件

執行方式：
    .\isaaclab.bat -p scripts\isaaclab_ws\test_ndi_joint_motion_v2.py
    .\isaaclab.bat -p scripts\isaaclab_ws\test_ndi_joint_motion_v2.py --hold_frames 300
"""

import argparse
import sys
import os
import time
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test NDIDetector with joint motion (v2)")
parser.add_argument("--usd_path", type=str,
                    default=r"C:\Nick\surgery_team\surgery_team\USD\surgeryroom_lostfunc.usd",
                    help="Path to the surgery room USD file")
parser.add_argument("--hold_frames", type=int, default=300,
                    help="Frames to hold at target position (default: 300 = 5 seconds)")
parser.add_argument("--raycast_interval", type=int, default=30,
                    help="Execute raycast every N frames (default: 30)")
parser.add_argument("--trigger_path", type=str,
                    default="/Root/NDI_02/mesh_",
                    help="USD path to the trigger volume prim")
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
import omni.ui as ui
from pxr import UsdPhysics, PhysxSchema
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaacsim.core.simulation_manager import SimulationManager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ndi_detector_isaaclab import NDIConfig, NDIDetector


# ---------------------------------------------------------------------------
# UI Panel
# ---------------------------------------------------------------------------
class NDIStatusPanel:
    """omni.ui 面板，即時顯示 5 個 marker 的 GREEN / RED 狀態"""

    MARKERS = ["FM", "HM", "BM", "EM", "UM"]
    GREEN = 0xFF00CC00   # ARGB
    RED   = 0xFFCC0000
    GREY  = 0xFF555555

    def __init__(self):
        self._labels = {}      # marker name -> ui.Label
        self._indicators = {}  # marker name -> ui.Rectangle
        self._reasons = {}     # marker name -> ui.Label
        self._summary_label = None
        self._frame_label = None
        self._build_window()

    # ---- build ----
    def _build_window(self):
        self._window = ui.Window(
            "NDI Marker Status", width=320, height=280,
            dockPreference=ui.DockPreference.RIGHT_TOP,
        )
        with self._window.frame:
            with ui.VStack(spacing=4, style={"margin": 6}):
                ui.Label("NDI Marker Status", height=28,
                         alignment=ui.Alignment.CENTER,
                         style={"font_size": 16, "color": 0xFFFFFFFF})
                ui.Spacer(height=2)

                # 每個 marker 一列
                for m in self.MARKERS:
                    with ui.HStack(height=26, spacing=6):
                        lbl = ui.Label(m, width=32,
                                       style={"font_size": 14, "color": 0xFFFFFFFF})
                        rect = ui.Rectangle(width=22, height=22,
                                            style={"background_color": self.GREY,
                                                   "border_radius": 4})
                        reason = ui.Label("waiting...", width=200,
                                          style={"font_size": 12, "color": 0xFFAAAAAA})
                        self._labels[m] = lbl
                        self._indicators[m] = rect
                        self._reasons[m] = reason

                ui.Spacer(height=6)
                self._summary_label = ui.Label("GREEN: -/5", height=24,
                                               alignment=ui.Alignment.CENTER,
                                               style={"font_size": 15, "color": 0xFFFFFFFF})
                self._frame_label = ui.Label("Frame: 0", height=20,
                                             alignment=ui.Alignment.CENTER,
                                             style={"font_size": 12, "color": 0xFFAAAAAA})

    # ---- update ----
    def update(self, results: dict, frame: int, time_sec: float):
        """results = {marker: (is_green:bool, reason:str)}"""
        green_count = 0
        for m in self.MARKERS:
            is_green, reason = results.get(m, (False, "no data"))
            color = self.GREEN if is_green else self.RED
            self._indicators[m].set_style({"background_color": color,
                                           "border_radius": 4})
            self._reasons[m].text = reason
            if is_green:
                green_count += 1
        self._summary_label.text = f"GREEN: {green_count}/5"
        self._frame_label.text = f"Frame {frame} | {time_sec:.1f}s"

    def destroy(self):
        if self._window:
            self._window.destroy()
            self._window = None


def deg_to_rad(degrees):
    return np.array(degrees) * np.pi / 180.0


def initialize_trigger_volume(trigger_path: str):
    """初始化觸發體積，回傳 state_api 或 None"""
    stage = omni.usd.get_context().get_stage()
    trigger_prim = stage.GetPrimAtPath(trigger_path)

    if not trigger_prim.IsValid():
        print(f"[ERROR] Trigger volume not found at: {trigger_path}")
        return None

    print(f"[OK] Found trigger volume at: {trigger_path}")

    if not trigger_prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(trigger_prim)

    if not trigger_prim.HasAPI(PhysxSchema.PhysxTriggerAPI):
        PhysxSchema.PhysxTriggerAPI.Apply(trigger_prim)

    if not trigger_prim.HasAPI(PhysxSchema.PhysxTriggerStateAPI):
        state_api = PhysxSchema.PhysxTriggerStateAPI.Apply(trigger_prim)
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


# 建立 marker 名稱 → end prim 路徑的對應表（用於 trigger 判定）
MARKER_PATH_KEYWORDS = {
    "FM": "/FM/",
    "HM": "/HM/",
    "BM": "/BM/",
    "EM": "/EM/",
    "UM": "/UM/",
}


def build_marker_end_paths(config):
    """從 NDIConfig 建立 {marker: [end_prim_paths]} 對應表"""
    mapping = {m: [] for m in config.SUPPORTED_MARKERS}
    for path in config.END_PRIM_PATHS:
        for marker, keyword in MARKER_PATH_KEYWORDS.items():
            if keyword in path:
                mapping[marker].append(path)
                break
    return mapping


def check_marker_in_trigger(marker, marker_end_paths, trigger_objs):
    """檢查某 marker 的任意 end prim 是否在 trigger volume 內"""
    for path in marker_end_paths.get(marker, []):
        for obj in trigger_objs:
            if path in obj or obj in path:
                return True
    return False


def main():
    print("\n" + "=" * 70)
    print(" NDI Detector 關節運動測試 v2")
    print(" - 手臂維持在目標位置")
    print(" - Debug draw 視覺化")
    print(" - 簡化輸出")
    print("=" * 70)

    # 目標關節位置（度數）
    target_joints_deg = [-107.26, 63.4, 60.3, -125.0, 98.4, 90.1]
    target_joints_rad = deg_to_rad(target_joints_deg)
    
    print(f"\n目標關節角度: {target_joints_deg}")

    # 1. 建立 SimulationContext
    sim_cfg = sim_utils.SimulationCfg(dt=1/60, device="cpu", gravity=(0.0, 0.0, -9.81))
    sim = SimulationContext(sim_cfg)

    # 2. 載入 USD
    print(f"\n載入 USD: {args_cli.usd_path}")
    if not os.path.exists(args_cli.usd_path):
        print(f"[ERROR] USD 檔案不存在")
        simulation_app.close()
        return

    omni.usd.get_context().open_stage(args_cli.usd_path)
    
    for _ in range(10):
        sim.step()
        time.sleep(0.05)

    # 3. Reset sim
    sim.reset()
    for _ in range(30):
        sim.step()

    # 4. 搜尋 ArticulationRootAPI
    stage = omni.usd.get_context().get_stage()
    from pxr import UsdPhysics
    
    articulation_paths = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_paths.append(str(prim.GetPath()))
    
    if not articulation_paths:
        print("[ERROR] 沒有找到 ArticulationRootAPI")
        simulation_app.close()
        return
    
    robot_prim_path = articulation_paths[0]
    print(f"使用 articulation: {robot_prim_path}")
    
    # 5. 建立 ArticulationView
    physics_sim_view = SimulationManager.get_physics_sim_view()
    arti_view = physics_sim_view.create_articulation_view(robot_prim_path)
    
    if arti_view is None:
        print("[ERROR] ArticulationView 建立失敗")
        simulation_app.close()
        return
    
    print(f"DOF: {arti_view.max_dofs}, Bodies: {arti_view.max_links}")

    # 6. 初始化 NDIDetector（開啟 debug draw）
    config = NDIConfig()
    detector = NDIDetector(config)
    detector.initialize()
    print("NDIDetector 初始化完成")

    # 6b. 初始化 Trigger Volume
    trigger_state_api = initialize_trigger_volume(args_cli.trigger_path)

    # 建立 marker → end paths 對應表
    marker_end_paths = build_marker_end_paths(config)

    # 7. 建立 UI 面板
    status_panel = NDIStatusPanel()

    # 等待穩定
    for _ in range(30):
        sim.step()

    # 7. 設定並維持目標關節位置
    num_dof = arti_view.max_dofs
    target_array = np.zeros((1, num_dof), dtype=np.float32)
    zero_vel = np.zeros((1, num_dof), dtype=np.float32)
    
    num_to_set = min(len(target_joints_rad), num_dof)
    for i in range(num_to_set):
        target_array[0, i] = target_joints_rad[i]
    
    print(f"\n從全零位置慢慢移動到目標並維持 {args_cli.hold_frames} 幀 ({args_cli.hold_frames/60:.1f} 秒)...")
    print(f"每 {args_cli.raycast_interval} 幀執行一次 raycast")
    print("-" * 70)
    
    frame_count = 0
    raycast_count = 0
    
    # 設定移動過程的總幀數 (例如前 300 幀 = 5秒 內從 0 變到目標位置)
    move_duration_frames = 300

    # 強制初始位置及速度全為 0
    zero_positions = np.zeros((1, num_dof), dtype=np.float32)
    arti_view.set_dof_velocities(zero_positions, indices=np.array([0]))
    arti_view.set_dof_positions(zero_positions, indices=np.array([0]))
    
    try:
        while frame_count < args_cli.hold_frames:
            # 計算當前進度 (0.0 到 1.0)
            progress = min(1.0, frame_count / move_duration_frames)
            
            # 使用線性插值計算當前的 target_position
            current_target = progress * target_array
            
            # 透過 set_dof_position_targets 給定 PD 控制器的目標值，慢慢移動到達目標
            arti_view.set_dof_position_targets(current_target, indices=np.array([0]))
            
            sim.step()
            frame_count += 1
            
            # 每幀都重繪 debug lines（跟隨機器人動態位置）
            if detector.debug_draw_interface and detector.raycast_results:
                detector._draw_debug_lines()
            
            # 定期執行 raycast
            if frame_count % args_cli.raycast_interval == 0:
                raycast_count += 1
                
                # 執行 raycast（開啟 debug draw）
                marker_status = detector.update_detection(sim, debug_enabled=True)
                
                # Trigger volume 偵測
                trigger_objs = get_trigger_objects(trigger_state_api)
                
                # 綜合判定：raycast visible + in trigger = GREEN，否則 RED
                time_sec = frame_count / 60.0
                green_count = 0
                lines = []
                ui_results = {}
                for marker in ['FM', 'HM', 'BM', 'EM', 'UM']:
                    ray_ok = marker_status.get(marker, False)
                    in_trigger = check_marker_in_trigger(marker, marker_end_paths, trigger_objs)
                    if ray_ok and in_trigger:
                        green_count += 1
                        reason = "visible + in trigger"
                        lines.append(f"  {marker}: [GREEN] {reason}")
                        ui_results[marker] = (True, reason)
                    elif ray_ok and not in_trigger:
                        reason = "visible but NOT in trigger"
                        lines.append(f"  {marker}: [RED]   {reason}")
                        ui_results[marker] = (False, reason)
                    elif not ray_ok and in_trigger:
                        reason = "in trigger but blocked"
                        lines.append(f"  {marker}: [RED]   {reason}")
                        ui_results[marker] = (False, reason)
                    else:
                        reason = "not visible, not in trigger"
                        lines.append(f"  {marker}: [RED]   {reason}")
                        ui_results[marker] = (False, reason)
                
                # 更新 UI 面板
                status_panel.update(ui_results, frame_count, time_sec)
                
                print(f"\n[Frame {frame_count:4d} | {time_sec:.1f}s] GREEN: {green_count}/5")
                for line in lines:
                    print(line)
        
        print("\n" + "-" * 70)
        print(f"完成！執行了 {raycast_count} 次 raycast")
        
    except KeyboardInterrupt:
        print("\n[中斷] 使用者停止")
    
    print("=" * 70)
    status_panel.destroy()
    simulation_app.close()


if __name__ == "__main__":
    main()
