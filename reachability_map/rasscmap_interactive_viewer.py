"""
rasscmap_interactive_viewer.py

互動式 RASSCMAP 視覺化工具 (Isaac Sim + omni.ui)

功能:
  - 開啟手術場景 USD
  - 從 backup_live_records CSV 預載所有環境的 RI 資料
  - 從 optimization_results CSV 讀取各環境的 base_x/base_y 和評分
  - 在 Isaac Sim 視窗右側顯示控制面板:
      * 每個環境一個按鈕 (顯示座標 + fg_score + success%)
      * 點擊後: 移動機器人底座 + 更新所有 waypoint 顏色球體
      * 底部顯示當前環境的詳細資訊
  - 若傳入 --tumor_pos，在場景中顯示腫瘤位置橙色球體
  - "Highlight Nearest WP" 按鈕：高亮最靠近腫瘤的 waypoint 並顯示可達性資訊

用法 (在 Isaac Sim 中):
    isaaclab.bat -p scripts/isaaclab_ws/reachability_map/rasscmap_interactive_viewer.py ^
        --backup_csv scripts/isaaclab_ws/reachability_map/heatmap/backup_live_records_*.csv ^
        --opt_csv    scripts/isaaclab_ws/reachability_map/heatmap/optimization_results_*.csv ^
        --tumor_pos  -0.30 -0.75 1.05
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import os
import sys

from ndi_verified_cache import (
    DEFAULT_CACHE_PATH,
    append_record,
    build_key,
    default_entry_pos,
)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Interactive RASSCMAP Viewer")
parser.add_argument(
    "--backup_csv", type=str, required=True,
    help="Path to backup_live_records_*.csv"
)
parser.add_argument(
    "--opt_csv", type=str, required=True,
    help="Path to optimization_results_*.csv"
)
parser.add_argument(
    "--usd_path", type=str,
    default=r"C:\Nick\surgery_team\surgery_team\USD\animation\isaaclab_multi_env.usd",
    help="USD scene file (default: isaaclab_multi_env.usd)"
)
parser.add_argument(
    "--num_orientations", type=int, default=5,
    help="Total orientations tested per waypoint (default: 5)"
)
parser.add_argument(
    "--sphere_scale", type=float, default=0.015,
    help="Radius of overlay spheres in meters (default: 0.015)"
)
parser.add_argument(
    "--tumor_pos", type=float, nargs=3, default=None,
    metavar=("TX", "TY", "TZ"),
    help="Tumor position (X Y Z) to display as an orange marker sphere in the scene"
)
parser.add_argument(
    "--entry_pos", type=float, nargs=3, default=None,
    metavar=("EX", "EY", "EZ"),
    help="Entry position (X Y Z) to display as a green marker sphere in the scene"
)
parser.add_argument(
    "--tumor_angle", type=float, default=0.0,
    help="Needle insertion angle in degrees for tumor target (default: 0.0)"
)
parser.add_argument(
    "--block_id", type=str, default=None,
    help="Matched CMAP block id from the query UI. Used only for verified-cache records."
)
parser.add_argument(
    "--verified_cache", type=str, default=DEFAULT_CACHE_PATH,
    help="Path to append exact 3D NDI verification records."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force GUI mode
args_cli.headless = False

import sys
sys.argv.extend([
    "--disable-ext", "omni.kit.viewport.menubar.core",
    "--disable-ext", "omni.kit.scripting",
    "--/app/vulkan=false",
    "--/rtx/ecoMode/enabled=true",
    "--/rtx/directLighting/sampledLighting/enabled=false",
    "--/rtx/ambientOcclusion/enabled=false",
    "--/rtx/reflections/enabled=false",
    "--/rtx/translucency/enabled=false",
    "--/rtx/sceneDb/maxInstances=2500000",
    "--/rtx/scenedb/maxInstances=2500000",
    "--/rtx-transient/scenedb/forceMaxTLASInstancesLimit=2500000",
    "--/rtx-transient/scenedb/maxInstancesLimit=2500000",
    "--/rtx-transient/scenedb/instanceBudget=2500000",
    "--/app/profilerBackend=",
    "--/carb/profiler/enabled=false",
])

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports ──────────────────────────────────────────────────────
import csv
import sys
import numpy as np
import pandas as pd
import matplotlib.cm as mpl_cm

# Path to NDI detector
sys.path.append(r"C:\Users\RMML\IsaacLab\scripts\isaaclab_ws\final")
try:
    from ndi_detector_isaaclab_experimental import NDIConfig, NDIDetector
except Exception as e:
    print(f"[NDI][WARN] Failed to import NDIDetector: {e}")
    NDIDetector = None
    NDIConfig = None

import omni.usd
import omni.ui as ui
from pxr import Usd, UsdGeom, Gf, Vt

# ─── Globals ────────────────────────────────────────────────────────────────
CURRENT_ENV_ID      = None
CURRENT_LACP        = None
_PENDING_ENV_ID     = None    # set by UI thread; consumed by main thread
_PENDING_HIGHLIGHT  = False   # request to highlight nearest WP to tumor
_PENDING_FOLLOW     = False   # request to run Follow Target
_PENDING_REACH_ENTRY = False # request to reach entry point
_PENDING_TOGGLE_CMAP = False  # request to toggle CMAP visibility
_PENDING_NDI_LOOK_AT = False  # request to look at center
NDI_DETECTOR        = None
UI_NDI_STATUS_LABEL = None
UI_TUMOR_LABEL      = None
NDI_MANUAL_ANGLE_OVERRIDE = None
LAST_OPTIMIZED_ANGLE = 0.0
UI_ANGLE_SLIDER = None
_IN_UI_UPDATE = False
ALL_ENV_DATA        = {}    # env_id → {base_x, base_y, fg_score, success_rate, sr_min, wp_data}
STAGE               = None
OPTIMAL_NDI_ANGLES  = {}
SPHERE_PRIMS        = {}    # wp_name → sphere prim
ORIGINAL_COLORS     = {}    # wp_name → (r, g, b)  — saved before highlight
CMAP_VISIBLE        = False  # current visibility of the CMAP overlay spheres

UI_WINDOW           = None
UI_STATUS_LABEL     = None
UI_NEAREST_LABEL    = None
UI_FOLLOW_LABEL     = None
UI_ENV_LABELS       = {}    # env_id → omni.ui.Button (for highlighting)
UI_TOGGLE_BTN       = None  # reference to toggle button for label update
_RB_VIEWS           = None  # PhysX RigidBodyView dict for external rigid bodies (lazy init)
_ROOT_BASE_OFFSET   = None  # articulation root - robotarm_base offset in world coordinates

ROBOTARM_BASE_DEFAULT_X = 0.2
ROBOTARM_BASE_DEFAULT_Y = 0.0
OVERLAY_ROOT      = "/World/RASSCMAP_Overlay"
TUMOR_ROOT        = "/World/RASSCMAP_Tumor"
TUMOR_SPHERE_PATH = "/World/RASSCMAP_Tumor/tumor_marker"


# ─── Colour mapping ─────────────────────────────────────────────────────────
def ri_to_rgb(ri: float) -> tuple:
    cmap = mpl_cm.get_cmap("jet_r")
    r, g, b, _ = cmap(float(np.clip(ri, 0.0, 1.0)))
    return (r, g, b)


# ─── CSV loading ─────────────────────────────────────────────────────────────
def load_all_env_data(backup_csv_path: str, opt_csv_path: str, num_orientations: int) -> dict:
    """
    Returns dict keyed by env_id:
      {
        base_x, base_y, fg_score, success_rate, sr_min,
        wp_data: { wp_name: {ri, ami, pos} }
      }
    """
    print("[CSV] Loading backup records …")
    raw = {}   # env_id → { wp_name → {success, total, mi_sum, pos} }
    with open(backup_csv_path, newline='') as f:
        for row in csv.DictReader(f):
            eid = int(row["Env_ID"])
            wp  = row["Waypoint"]
            mi  = float(row["Manipulability_Index"])
            pos = (float(row["Pos_X"]), float(row["Pos_Y"]), float(row["Pos_Z"]))
            if eid not in raw:
                raw[eid] = {}
            if wp not in raw[eid]:
                raw[eid][wp] = {"success": 0, "total": 0, "mi_sum": 0.0, "pos": pos}
            raw[eid][wp]["total"] += 1
            if mi > 0.0:
                raw[eid][wp]["success"] += 1
                raw[eid][wp]["mi_sum"] += mi

    print("[CSV] Loading optimization results …")
    opt_df = pd.read_csv(opt_csv_path)

    result = {}
    for eid, wps in raw.items():
        wp_data = {}
        for wp, data in wps.items():
            total = data["total"] if data["total"] > 0 else num_orientations
            ri  = data["success"] / total
            ami = data["mi_sum"] / data["success"] if data["success"] > 0 else 0.0
            wp_data[wp] = {"ri": ri, "ami": ami, "pos": data["pos"]}

        row = opt_df[opt_df["env_id"] == eid]
        if not row.empty:
            r = row.iloc[0]
            base_x      = float(r["base_x"])
            base_y      = float(r["base_y"])
            fg_score    = float(r["fg_score"])    if "fg_score"    in r else 0.0
            succ_rate   = float(r["success_rate"]) if "success_rate" in r else 0.0
            sr_min      = int(r["sr_min"])         if "sr_min"      in r else 0
            avg_manip   = float(r["avg_manip"])    if "avg_manip"   in r else 0.0
        else:
            base_x = base_y = fg_score = succ_rate = avg_manip = 0.0
            sr_min = 0

        result[eid] = {
            "base_x":       base_x,
            "base_y":       base_y,
            "fg_score":     fg_score,
            "success_rate": succ_rate,
            "sr_min":       sr_min,
            "avg_manip":    avg_manip,
            "wp_data":      wp_data,
        }

    print(f"[CSV] Loaded {len(result)} environments, "
          f"{sum(len(d['wp_data']) for d in result.values())} total waypoint records.")
    return result


# ─── Best env selection ──────────────────────────────────────────────────────
def get_best_env_id(env_data: dict) -> int:
    """
    Pick the best environment using (success_rate, sr_min, fg_score, avg_manip) as sort key.
    Prioritize success_rate (cs) first, then sr_min, then fg_score.
    """
    return max(
        env_data,
        key=lambda e: (
            env_data[e]["success_rate"],
            env_data[e]["sr_min"],
            env_data[e]["fg_score"],
            env_data[e].get("avg_manip", 0.0)
        )
    )


# ─── USD stage helpers ───────────────────────────────────────────────────────
def get_ndi_test_target(entry_pos, tumor_pos, z_test=0.95):
    if entry_pos is None or tumor_pos is None:
        ref = tumor_pos if tumor_pos is not None else np.array([0.0, -0.75, 1.05])
        return np.array([ref[0], ref[1], z_test])
    ez = entry_pos[2]
    tz = tumor_pos[2]
    if abs(tz - ez) < 1e-4:
        return np.array([tumor_pos[0], tumor_pos[1], z_test])
    factor = (z_test - ez) / (tz - ez)
    x_test = entry_pos[0] + factor * (tumor_pos[0] - entry_pos[0])
    y_test = entry_pos[1] + factor * (tumor_pos[1] - entry_pos[1])
    return np.array([x_test, y_test, z_test])


def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-12 else np.zeros_like(v)


def rot_matrix_to_quat_wxyz(m):
    trace = np.trace(m)
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)


def look_at_quaternion_world(ndi_pos, target_pos):
    forward = target_pos - ndi_pos
    forward_len = np.linalg.norm(forward)
    if forward_len < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    forward = forward / forward_len

    up_hint = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, up_hint)) > 0.999:
        up_hint = np.array([0.0, 1.0, 0.0])

    right = np.cross(forward, up_hint)
    right = right / np.linalg.norm(right)

    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    new_right = up
    new_up = -right

    R = np.column_stack([new_right, new_up, -forward])
    return rot_matrix_to_quat_wxyz(R)


def get_prim_world_position(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    xform = UsdGeom.Xformable(prim)
    transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = transform.ExtractTranslation()
    return np.array([t[0], t[1], t[2]], dtype=float)


def get_env_origin(stage, env_id):
    if env_id is None:
        return np.zeros(3, dtype=float)
    env_path = f"/World/envs/env_{env_id}"
    prim = stage.GetPrimAtPath(env_path)
    if not prim.IsValid():
        return np.zeros(3, dtype=float)
    xform = UsdGeom.Xformable(prim)
    transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = transform.ExtractTranslation()
    return np.array([t[0], t[1], t[2]], dtype=float)


def get_current_usd_base_pos_3d(stage: Usd.Stage):
    candidates = ["/Root/robotarm_base", "/World/robotarm_base"]
    for p in candidates:
        prim = stage.GetPrimAtPath(p)
        if prim.IsValid():
            xformable = UsdGeom.Xformable(prim)
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    val = op.Get()
                    if val:
                        return np.array([val[0], val[1], val[2]], dtype=float)
    return np.array([0.2, 0.0, 0.9493139386177063], dtype=float)


def check_marker_in_frustum(camera_pos, quat_wxyz, target_pos):
    # NDI Polaris Vega pyramid frustum check (in meters)
    NEAR_D, NEAR_X, NEAR_Y = 0.950, 0.224, 0.240
    MID_D,  MID_X,  MID_Y  = 1.532, 0.398, 0.572
    FAR_D,  FAR_X,  FAR_Y  = 2.400, 0.656, 0.783

    # Inverse of unit quat [w, x, y, z] is [w, -x, -y, -z]
    q_inv = np.array([quat_wxyz[0], -quat_wxyz[1], -quat_wxyz[2], -quat_wxyz[3]])
    vec = target_pos - camera_pos
    
    # Vector rotation via quaternion: v' = q * v * q^-1
    qw, qx, qy, qz = q_inv
    t = 2 * np.cross([qx, qy, qz], vec)
    local_vec = vec + qw * t + np.cross([qx, qy, qz], t)
    
    X, Y, Z = local_vec[0], local_vec[1], local_vec[2]
    depth = -Z
    if depth < NEAR_D or depth > FAR_D:
        return False
        
    if depth <= MID_D:
        t_val = (depth - NEAR_D) / (MID_D - NEAR_D)
        max_x = NEAR_X + t_val * (MID_X - NEAR_X)
        max_y = NEAR_Y + t_val * (MID_Y - NEAR_Y)
    else:
        t_val = (depth - MID_D) / (FAR_D - MID_D)
        max_x = MID_X + t_val * (FAR_X - MID_X)
        max_y = MID_Y + t_val * (FAR_Y - MID_Y)
        
    if abs(X) > max_x or abs(Y) > max_y:
        return False
        
    return True


def optimize_ndi_camera_angle(stage, lacp, env_id):
    global NDI_MANUAL_ANGLE_OVERRIDE, LAST_OPTIMIZED_ANGLE, OPTIMAL_NDI_ANGLES
    if NDI_MANUAL_ANGLE_OVERRIDE is not None:
        print(f"[NDI_MANUAL] Using manual camera angle override = {NDI_MANUAL_ANGLE_OVERRIDE}°")
        return NDI_MANUAL_ANGLE_OVERRIDE
    if OPTIMAL_NDI_ANGLES and env_id in OPTIMAL_NDI_ANGLES:
        opt_angle = OPTIMAL_NDI_ANGLES[env_id]
        LAST_OPTIMIZED_ANGLE = opt_angle
        print(f"[NDI_OPTIMIZE] Env {env_id}: Using pre-calculated optimal angle = {opt_angle}°")
        return opt_angle
        
    groups = {
        "BM": [
            "/robotarm_base/robotarm_base/BM/BM_meter/bm_a",
            "/robotarm_base/robotarm_base/BM/BM_meter/bm_b",
            "/robotarm_base/robotarm_base/BM/BM_meter/bm_c",
            "/robotarm_base/robotarm_base/BM/BM_meter/bm_d",
        ],
        "EM": [
            "/robotarm_base/robotarm_base/EM/EM/em_a",
            "/robotarm_base/robotarm_base/EM/EM/em_b",
            "/robotarm_base/robotarm_base/EM/EM/em_c",
            "/robotarm_base/robotarm_base/EM/EM/em_d",
        ],
        "FM": [
            "/FM/FM/FM/FM/fm_a",
            "/FM/FM/FM/FM/fm_b",
            "/FM/FM/FM/FM/fm_c",
            "/FM/FM/FM/FM/fm_d",
        ],
        "UM": [
            "/robotarm_base/robotarm_base/UM/UM/UM/um_a",
            "/robotarm_base/robotarm_base/UM/UM/UM/um_b",
            "/robotarm_base/robotarm_base/UM/UM/UM/um_c",
            "/robotarm_base/robotarm_base/UM/UM/UM/um_d",
        ]
    }
    
    prefix = "/Root"
    if env_id is not None:
        test_prefix = f"/World/envs/env_{env_id}/Root"
        if stage.GetPrimAtPath(test_prefix).IsValid():
            prefix = test_prefix
            
    markers_pos = []
    for name, paths in groups.items():
        for rel in paths:
            full_path = f"{prefix}{rel}"
            pos = get_prim_world_position(stage, full_path)
            if pos is not None:
                markers_pos.append(pos)
                
    default_theta = (env_id % 10) * 5.0 if env_id is not None else 0.0
    if not markers_pos:
        LAST_OPTIMIZED_ANGLE = default_theta
        return default_theta
        
    R_h = 1.52
    H = 0.978
    
    best_theta = default_theta
    best_hits = -1
    
    for theta_deg in range(91):
        theta_rad = np.deg2rad(theta_deg)
        ndi_world_x = lacp[0] + R_h * np.sin(theta_rad)
        ndi_world_y = lacp[1] - R_h * np.cos(theta_rad)
        ndi_world_z = lacp[2] + H
        ndi_world_pos = np.array([ndi_world_x, ndi_world_y, ndi_world_z])
        
        quat_world = look_at_quaternion_world(ndi_world_pos, lacp)
        hits = sum(1 for m_pos in markers_pos if check_marker_in_frustum(ndi_world_pos, quat_world, m_pos))
        
        if hits > best_hits:
            best_hits = hits
            best_theta = theta_deg
        elif hits == best_hits:
            if abs(theta_deg - default_theta) < abs(best_theta - default_theta):
                best_theta = theta_deg
                
    LAST_OPTIMIZED_ANGLE = float(best_theta)
    print(f"[NDI_OPTIMIZE] Env {env_id}: Optimized camera angle theta = {best_theta}° (Frustum hits: {best_hits}/{len(markers_pos)})")
    return float(best_theta)


def get_all_markers_bbox_center(stage):
    groups = {
        "BM": [
            "/robotarm_base/robotarm_base/BM/BM_meter/bm_a",
            "/robotarm_base/robotarm_base/BM/BM_meter/bm_b",
            "/robotarm_base/robotarm_base/BM/BM_meter/bm_c",
            "/robotarm_base/robotarm_base/BM/BM_meter/bm_d",
        ],
        "EM": [
            "/robotarm_base/robotarm_base/EM/EM/em_a",
            "/robotarm_base/robotarm_base/EM/EM/em_b",
            "/robotarm_base/robotarm_base/EM/EM/em_c",
            "/robotarm_base/robotarm_base/EM/EM/em_d",
        ],
        "FM": [
            "/FM/FM/FM/FM/fm_a",
            "/FM/FM/FM/FM/fm_b",
            "/FM/FM/FM/FM/fm_c",
            "/FM/FM/FM/FM/fm_d",
        ],
        "UM": [
            "/robotarm_base/robotarm_base/UM/UM/UM/um_a",
            "/robotarm_base/robotarm_base/UM/UM/UM/um_b",
            "/robotarm_base/robotarm_base/UM/UM/UM/um_c",
            "/robotarm_base/robotarm_base/UM/UM/UM/um_d",
        ]
    }
    
    prefix = "/Root"
    global CURRENT_ENV_ID
    if CURRENT_ENV_ID is not None:
        test_prefix = f"/World/envs/env_{CURRENT_ENV_ID}/Root"
        if stage.GetPrimAtPath(test_prefix).IsValid():
            prefix = test_prefix
            
    pts = []
    for name, paths in groups.items():
        for rel in paths:
            full_path = f"{prefix}{rel}"
            pos = get_prim_world_position(stage, full_path)
            if pos is not None:
                pts.append(pos)
                
    if not pts:
        return None
    pts = np.array(pts)
    min_coords = np.min(pts, axis=0)
    max_coords = np.max(pts, axis=0)
    return (min_coords + max_coords) / 2.0


def align_ndi_to_default_lacp(env_id):
    global STAGE, ALL_ENV_DATA, CURRENT_LACP
    if env_id not in ALL_ENV_DATA:
        return
    data = ALL_ENV_DATA[env_id]
    wp_data = data["wp_data"]
    if not wp_data:
        return
        
    # 1. Compute bounding box center containing waypoints and um_pos
    wps_pos = [np.array(wp["pos"]) for wp in wp_data.values()]
    base_pos = get_current_usd_base_pos_3d(STAGE)
    um_pos = base_pos + np.array([0.031517988675667796, -0.50597902213614, 0.9193813560801563])
    
    all_pts = wps_pos + [um_pos]
    min_coords = np.min(all_pts, axis=0)
    max_coords = np.max(all_pts, axis=0)
    default_lacp = (min_coords + max_coords) / 2.0
    CURRENT_LACP = default_lacp
    
    # 4. Look NDI at default_lacp
    ndi_path = "/Root/NDI"
    if env_id is not None:
        test_path = f"/World/envs/env_{env_id}/Root/NDI"
        if STAGE.GetPrimAtPath(test_path).IsValid():
            ndi_path = test_path
            
    ndi_prim = STAGE.GetPrimAtPath(ndi_path)
    if not ndi_prim.IsValid():
        return
        
    xform = UsdGeom.Xformable(ndi_prim)
    
    # Optimize NDI position on horizontal circular arc relative to default_lacp
    R_h = 1.52
    H = 0.978
    theta_deg = optimize_ndi_camera_angle(STAGE, default_lacp, env_id)
    theta_rad = np.deg2rad(theta_deg)
    
    ndi_world_x = default_lacp[0] + R_h * np.sin(theta_rad)
    ndi_world_y = default_lacp[1] - R_h * np.cos(theta_rad)
    ndi_world_z = default_lacp[2] + H
    ndi_world_pos = np.array([ndi_world_x, ndi_world_y, ndi_world_z])
    
    # Get local position relative to parent
    parent_path = ndi_path.rsplit("/", 1)[0]
    parent_prim = STAGE.GetPrimAtPath(parent_path)
    parent_matrix = Gf.Matrix4d(1.0)
    if parent_prim.IsValid():
        parent_xform = UsdGeom.Xformable(parent_prim)
        parent_matrix = parent_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        
    local_pos = np.array(parent_matrix.GetInverse().Transform(Gf.Vec3d(*ndi_world_pos)))
        
    # Calculate look-at quaternion (world frame) from the new position
    quat_world = look_at_quaternion_world(ndi_world_pos, default_lacp)
    
    # Factor out parent rotation
    q_parent = parent_matrix.ExtractRotation().GetQuaternion()
    gf_quat_world = Gf.Quaternion(float(quat_world[0]), Gf.Vec3d(float(quat_world[1]), float(quat_world[2]), float(quat_world[3])))
    q_local = q_parent.GetInverse() * gf_quat_world
    
    # Apply local translation and orientation
    set_prim_transform_orient(STAGE, ndi_path, local_pos, q_local)
    print(f"[NDI] Aligned NDI to default LACP (startup/switch): {default_lacp.round(3)} (theta: {theta_deg}°, um_pos: {um_pos.round(3)})")
    sync_slider_to_optimized()


def update_ndi_detector_for_env(env_id):
    global NDI_DETECTOR, STAGE
    if NDIDetector is None or STAGE is None:
        return
    try:
        env_prim_path = f"/World/envs/env_{env_id}"
        if not STAGE.GetPrimAtPath(env_prim_path).IsValid():
            env_prim_path = "/Root"
        
        # Search structure
        detected_map = {}
        env_prim = STAGE.GetPrimAtPath(env_prim_path)
        if env_prim.IsValid():
            def search(prim, depth=0):
                if depth > 10: return
                name = prim.GetName()
                path = str(prim.GetPath())
                if name == "link_6" and "link_6_path" not in detected_map:
                    detected_map["link_6_path"] = path
                if name == "needle_tip" and "needle_tip_path" not in detected_map:
                    detected_map["needle_tip_path"] = path
                if name == "NDI_emitter" and "emitter_path" not in detected_map:
                    detected_map["emitter_path"] = path
                for child in prim.GetChildren():
                    search(child, depth + 1)
                    if "link_6_path" in detected_map and "emitter_path" in detected_map and "needle_tip_path" in detected_map: return
            search(env_prim)

        # Build custom config
        class EnvNDIConfig(NDIConfig):
            def __init__(self, prefix_replacement, d_map):
                super().__init__()
                if d_map and "emitter_path" in d_map:
                    p = d_map["emitter_path"]
                    if "/NDI/NDI_emitter" in p:
                        prefix_replacement = p.split("/NDI/NDI_emitter")[0]
                self.START_PRIM_PATHS = [path.replace("/Root", prefix_replacement) for path in self.START_PRIM_PATHS]
                self.START_HIGHLIGHT_PATHS = [path.replace("/Root", prefix_replacement) for path in self.START_HIGHLIGHT_PATHS]
                self.END_PRIM_PATHS = [path.replace("/Root", prefix_replacement) for path in self.END_PRIM_PATHS]
                self.TRIGGER_VOLUME_PATH = self.TRIGGER_VOLUME_PATH.replace("/Root", prefix_replacement)
                if d_map and "link_6_path" in d_map:
                    self.LINK_6_PATH = d_map["link_6_path"]
                else:
                    self.LINK_6_PATH = f"{prefix_replacement}/robotarm_base/robotarm_base/tm5_700/link_6"
                if d_map and "needle_tip_path" in d_map:
                    self.NEEDLE_TIP_PATH = d_map["needle_tip_path"]
                else:
                    self.NEEDLE_TIP_PATH = f"{prefix_replacement}/robotarm_base/robotarm_base/needle/needle_tip"
                    
        config = EnvNDIConfig(env_prim_path, detected_map)
        NDI_DETECTOR = NDIDetector(config)
        NDI_DETECTOR._link_paths["link_6"] = config.LINK_6_PATH
        NDI_DETECTOR._link_paths["needle_tip"] = config.NEEDLE_TIP_PATH
        NDI_DETECTOR._link_paths["link_base"] = config.LINK_6_PATH.replace("link_6", "link_0")
        NDI_DETECTOR.initialize()
        print(f"[NDI] Detector re-initialized for Env {env_id} at {env_prim_path}")
    except Exception as e:
        print(f"[NDI][WARN] Failed to re-initialize detector for Env {env_id}: {e}")


def set_prim_transform_orient(stage, prim_path, translation, q_local):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    xform = UsdGeom.Xformable(prim)
    
    # Clear old ops order to prevent orientation conflicts
    xform.ClearXformOpOrder()
    
    translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(*translation))
    
    orient_op = xform.AddOrientOp()
    attr = orient_op.GetAttr()
    typeName = str(attr.GetTypeName())
    
    if "quatf" in typeName.lower():
        gf_quat = Gf.Quatf(float(q_local.GetReal()), Gf.Vec3f(float(q_local.GetImaginary()[0]), float(q_local.GetImaginary()[1]), float(q_local.GetImaginary()[2])))
    else:
        gf_quat = Gf.Quatd(float(q_local.GetReal()), Gf.Vec3d(float(q_local.GetImaginary()[0]), float(q_local.GetImaginary()[1]), float(q_local.GetImaginary()[2])))
    orient_op.Set(gf_quat)


def do_ndi_look_at(use_stage_centroid=True):
    global STAGE, NDI_DETECTOR, CURRENT_LACP, CURRENT_ENV_ID
    
    centroid = None
    if use_stage_centroid:
        centroid = get_all_markers_bbox_center(STAGE)
        
    if centroid is None and NDI_DETECTOR is not None:
        try:
            NDI_DETECTOR.update_detection(sim=None, debug_enabled=True, error_only_log=True)
        except Exception as e:
            print(f"[NDI] Detector update before look-at failed: {e}")
        valid_positions = [pos for pos in NDI_DETECTOR.line_ends if pos is not None]
        if valid_positions:
            valid_positions = np.array(valid_positions)
            min_coords = np.min(valid_positions, axis=0)
            max_coords = np.max(valid_positions, axis=0)
            centroid = (min_coords + max_coords) / 2.0
            
    if centroid is None:
        print("[NDI] No markers available for look-at target.")
        return
        
    CURRENT_LACP = centroid
        
    # Get NDI camera current world position
    ndi_path = "/Root/NDI"
    if CURRENT_ENV_ID is not None:
        test_path = f"/World/envs/env_{CURRENT_ENV_ID}/Root/NDI"
        if STAGE.GetPrimAtPath(test_path).IsValid():
            ndi_path = test_path
            
    ndi_prim = STAGE.GetPrimAtPath(ndi_path)
    if not ndi_prim.IsValid():
        print(f"[NDI] {ndi_path} prim not found on stage.")
        return
        
    xform = UsdGeom.Xformable(ndi_prim)
    
    # Optimize NDI position on horizontal circular arc relative to updated centroid (LACP)
    R_h = 1.52
    H = 0.978
    theta_deg = optimize_ndi_camera_angle(STAGE, centroid, CURRENT_ENV_ID)
    theta_rad = np.deg2rad(theta_deg)
    
    ndi_world_x = centroid[0] + R_h * np.sin(theta_rad)
    ndi_world_y = centroid[1] - R_h * np.cos(theta_rad)
    ndi_world_z = centroid[2] + H
    ndi_world_pos = np.array([ndi_world_x, ndi_world_y, ndi_world_z])
    
    # Get local position relative to parent
    parent_path = ndi_path.rsplit("/", 1)[0]
    parent_prim = STAGE.GetPrimAtPath(parent_path)
    parent_matrix = Gf.Matrix4d(1.0)
    if parent_prim.IsValid():
        parent_xform = UsdGeom.Xformable(parent_prim)
        parent_matrix = parent_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        
    local_pos = np.array(parent_matrix.GetInverse().Transform(Gf.Vec3d(*ndi_world_pos)))
    
    # Calculate look-at quaternion (world frame) from the new position
    quat_world = look_at_quaternion_world(ndi_world_pos, centroid)
    
    # Convert quat_world [w, x, y, z] to Gf.Quaternion
    gf_quat_world = Gf.Quaternion(float(quat_world[0]), Gf.Vec3d(float(quat_world[1]), float(quat_world[2]), float(quat_world[3])))
    q_parent = parent_matrix.ExtractRotation().GetQuaternion()
    q_local = q_parent.GetInverse() * gf_quat_world

    # Apply local translation and orientation
    set_prim_transform_orient(STAGE, ndi_path, local_pos, q_local)
    
    if NDI_DETECTOR is not None:
        try:
            NDI_DETECTOR.update_detection(sim=None, debug_enabled=True, error_only_log=True)
        except Exception as e:
            pass
            
    print(f"[NDI] Look At Center completed. Centroid: {centroid.round(3)}")
    sync_slider_to_optimized()


def request_ndi_look_at():
    global _PENDING_NDI_LOOK_AT
    _PENDING_NDI_LOOK_AT = True


import math

def load_usd_scene(usd_path: str):
    if not os.path.exists(usd_path):
        print(f"[WARN] USD not found: {usd_path}  (continuing with empty stage)")
        return
    print(f"[INFO] Opening: {os.path.basename(usd_path)}")
    omni.usd.get_context().open_stage(usd_path)
    for _ in range(60):
        simulation_app.update()
    print("[INFO] Scene loaded.")


def get_stage() -> Usd.Stage:
    return omni.usd.get_context().get_stage()


def move_robot_base(stage: Usd.Stage, base_x: float, base_y: float):
    """Move robotarm_base Xform prim to (base_x, base_y, z_preserved)."""
    candidates = ["/Root/robotarm_base", "/World/robotarm_base"]
    prim = None
    for p in candidates:
        c = stage.GetPrimAtPath(p)
        if c.IsValid():
            prim = c
            break
    if prim is None:
        print(f"[WARN] robotarm_base prim not found in {candidates}")
        return

    xformable = UsdGeom.Xformable(prim)
    translate_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break

    current_z = 0.0
    if translate_op is not None:
        cur = translate_op.Get()
        current_z = cur[2] if cur else 0.0
    else:
        xformable.ClearXformOpOrder()
        translate_op = xformable.AddTranslateOp()

    translate_op.Set(Gf.Vec3d(base_x, base_y, current_z))
    print(f"[BASE] robotarm_base → ({base_x:.3f}, {base_y:.3f}, {current_z:.3f})")


def get_current_usd_base_pos(stage: Usd.Stage):
    """Retrieve the current USD translation of the robotarm_base Xform prim."""
    candidates = ["/Root/robotarm_base", "/World/robotarm_base"]
    for p in candidates:
        prim = stage.GetPrimAtPath(p)
        if prim.IsValid():
            xformable = UsdGeom.Xformable(prim)
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    val = op.Get()
                    if val:
                        return val[0], val[1]
    return 0.2, 0.0


def _get_root_base_offset(stage: Usd.Stage, robot):
    """Return the fixed articulation-root minus robotarm_base offset."""
    global _ROOT_BASE_OFFSET
    if _ROOT_BASE_OFFSET is None:
        base_x, base_y = get_current_usd_base_pos(stage)
        root_pos = robot.data.root_pos_w[0].detach().clone()
        _ROOT_BASE_OFFSET = root_pos.clone()
        _ROOT_BASE_OFFSET[0] -= float(base_x)
        _ROOT_BASE_OFFSET[1] -= float(base_y)
        print(
            "[BASE] root-base offset = "
            f"({_ROOT_BASE_OFFSET[0].item():.4f}, "
            f"{_ROOT_BASE_OFFSET[1].item():.4f}, "
            f"{_ROOT_BASE_OFFSET[2].item():.4f})"
        )
    return _ROOT_BASE_OFFSET


def move_markers_base(stage: Usd.Stage, base_x: float, base_y: float):
    """
    Teleports the UM, EM, and BM rigid body prims relative to the robotarm_base USD parent.
    If simulation is active, the parent's USD transform is locked (to prevent PhysX crashes),
    so we adjust the local transforms of markers relative to the locked parent.
    """
    init_base_x, init_base_y = get_current_usd_base_pos(stage)
    dx = base_x - init_base_x
    dy = base_y - init_base_y

    markers_config = {
        "UM": {
            "paths": ["/Root/robotarm_base/robotarm_base/UM", "/World/robotarm_base/robotarm_base/UM"],
            "default_x": 0.031517988675667796,
            "default_y": -0.50597902213614,
            "default_z": 0.9193813560801563
        },
        "EM": {
            "paths": ["/Root/robotarm_base/robotarm_base/EM", "/World/robotarm_base/robotarm_base/EM"],
            "default_x": 0.09775413331911276,
            "default_y": -0.8538055212722788,
            "default_z": 1.795369673705961
        },
        "BM": {
            "paths": ["/Root/robotarm_base/robotarm_base/BM", "/World/robotarm_base/robotarm_base/BM"],
            "default_x": 0.004339944904531157,
            "default_y": -0.9445060681122224,
            "default_z": 1.7960204582484482
        },
        "needle_holder": {
            "paths": ["/Root/robotarm_base/robotarm_base/needle_holder", "/World/robotarm_base/robotarm_base/needle_holder"],
            "default_x": 0.059785796752960474,
            "default_y": -0.7598351588384606,
            "default_z": 1.8363891647097481
        },
        "needle": {
            "paths": ["/Root/robotarm_base/robotarm_base/needle/needle", "/World/robotarm_base/robotarm_base/needle/needle"],
            "default_x": 0.061558348475197755,
            "default_y": -1.0098351417071814,
            "default_z": 1.8374446562365232
        },
        "needle_tip": {
            "paths": ["/Root/robotarm_base/robotarm_base/needle/needle_tip", "/World/robotarm_base/robotarm_base/needle/needle_tip"],
            "default_x": -0.038474891759153515,
            "default_y": -1.0095436424780713,
            "default_z": 2.3374354817316756
        }
    }

    for name, cfg in markers_config.items():
        prim = None
        for p in cfg["paths"]:
            c = stage.GetPrimAtPath(p)
            if c.IsValid():
                prim = c
                break

        if prim is not None:
            xformable = UsdGeom.Xformable(prim)
            translate_op = None
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                    break

            current_z = cfg["default_z"]
            if translate_op is not None:
                cur = translate_op.Get()
                current_z = cur[2] if cur else current_z
            else:
                xformable.ClearXformOpOrder()
                translate_op = xformable.AddTranslateOp()

            new_local_x = cfg["default_x"] + dx
            new_local_y = cfg["default_y"] + dy

            translate_op.Set(Gf.Vec3d(new_local_x, new_local_y, current_z))
            print(f"[{name}] {name} local translation set to ({new_local_x:.3f}, {new_local_y:.3f}, {current_z:.3f})")
        else:
            print(f"[WARN] {name} prim not found on stage.")


def _init_rigid_body_views():
    """Lazily create PhysX RigidBodyView objects for all external (non-articulation) rigid bodies.
    Must be called AFTER _SIM.reset() has been invoked (i.e., after _init_ik_system)."""
    global _RB_VIEWS
    if _RB_VIEWS is not None:
        return
    _RB_VIEWS = {}
    if _SIM is None:
        return
    try:
        from isaacsim.core.simulation_manager import SimulationManager
        sim_view = SimulationManager.get_physics_sim_view()
        paths = {
            "car":           "/Root/robotarm_base/robotarm_base/car",
            "UM":            "/Root/robotarm_base/robotarm_base/UM",
            "EM":            "/Root/robotarm_base/robotarm_base/EM",
            "BM":            "/Root/robotarm_base/robotarm_base/BM",
            "needle_holder": "/Root/robotarm_base/robotarm_base/needle_holder",
            "needle":        "/Root/robotarm_base/robotarm_base/needle/needle",
            "needle_tip":    "/Root/robotarm_base/robotarm_base/needle/needle_tip",
        }
        for name, path in paths.items():
            try:
                view = sim_view.create_rigid_body_view(path)
                if view is not None and view._backend is not None:
                    _RB_VIEWS[name] = view
                    print(f"[RB] Created PhysX view for {name}")
                else:
                    print(f"[RB] WARN: view for {name} has no backend (not a physics body?)")
            except Exception as e:
                print(f"[RB] WARN: Cannot create view for {name}: {e}")
        print(f"[RB] Initialized {len(_RB_VIEWS)} rigid body views.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[RB] WARN: Cannot access physics sim view: {e}")


def _teleport_rigid_bodies(dx: float, dy: float):
    """Shift all external rigid bodies by (dx, dy) in world space via PhysX tensor API.
    This is the ONLY reliable way to move non-articulation rigid bodies during GPU simulation."""
    global _RB_VIEWS
    if _RB_VIEWS is None:
        _init_rigid_body_views()
    if not _RB_VIEWS:
        return
    import torch
    for name, view in _RB_VIEWS.items():
        try:
            transforms = view.get_transforms()  # (N, 7): x, y, z, qx, qy, qz, qw
            transforms[:, 0] += dx
            transforms[:, 1] += dy
            view.set_transforms(transforms)
            p = transforms[0].cpu().numpy()
            print(f"[RB] {name} teleported to ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")
        except Exception as e:
            print(f"[RB] WARN: Cannot teleport {name}: {e}")



def create_spheres(stage: Usd.Stage, wp_data: dict, radius: float) -> dict:
    """Create (or re-create) overlay spheres. Returns {wp_name: sphere_prim}."""
    old = stage.GetPrimAtPath(OVERLAY_ROOT)
    if old.IsValid():
        stage.RemovePrim(old.GetPath())

    UsdGeom.Xform.Define(stage, OVERLAY_ROOT)
    prims = {}
    for wp_name, data in wp_data.items():
        rgb = ri_to_rgb(data["ri"])
        pos = data["pos"]
        path = f"{OVERLAY_ROOT}/{wp_name}_sphere"
        sphere = UsdGeom.Sphere.Define(stage, path)
        sphere.GetRadiusAttr().Set(radius)
        x = UsdGeom.Xformable(sphere.GetPrim())
        x.ClearXformOpOrder()
        x.AddTranslateOp().Set(Gf.Vec3d(pos[0], pos[1], pos[2]))
        sphere.GetDisplayColorAttr().Set(
            Vt.Vec3fArray(1, (Gf.Vec3f(rgb[0], rgb[1], rgb[2]),))
        )
        sphere.GetDisplayOpacityAttr().Set([1.0])
        prims[wp_name] = sphere.GetPrim()
    return prims


def update_sphere_colors(sphere_prims: dict, wp_data: dict):
    """Update displayColor on existing spheres (faster than recreating)."""
    global ORIGINAL_COLORS
    ORIGINAL_COLORS = {}
    for wp_name, prim in sphere_prims.items():
        if wp_name not in wp_data:
            continue
        rgb = ri_to_rgb(wp_data[wp_name]["ri"])
        ORIGINAL_COLORS[wp_name] = rgb
        gprim = UsdGeom.Gprim(prim)
        if gprim:
            gprim.GetDisplayColorAttr().Set(
                Vt.Vec3fArray(1, (Gf.Vec3f(rgb[0], rgb[1], rgb[2]),))
            )


# ─── Toggle CMAP visibility ─────────────────────────────────────────────────
def update_cmap_visibility_to_current(stage: Usd.Stage):
    """Update visibility in USD scene and button text according to CMAP_VISIBLE."""
    global CMAP_VISIBLE, UI_TOGGLE_BTN
    overlay = stage.GetPrimAtPath(OVERLAY_ROOT)
    if overlay.IsValid():
        imageable = UsdGeom.Imageable(overlay)
        if CMAP_VISIBLE:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()
    label = "  Hide CMAP Spheres" if CMAP_VISIBLE else "  Show CMAP Spheres"
    if UI_TOGGLE_BTN is not None:
        UI_TOGGLE_BTN.text = label

def toggle_cmap_visibility(stage: Usd.Stage):
    """Toggle the visibility of all CMAP overlay spheres."""
    global CMAP_VISIBLE
    CMAP_VISIBLE = not CMAP_VISIBLE
    update_cmap_visibility_to_current(stage)
    print(f"[CMAP] Overlay {'shown' if CMAP_VISIBLE else 'hidden'}")



# ─── IK constants (same as parallel_base_optimizer_linear.py) ───────────────
_DEFAULT_JOINT_POS = np.array([-1.48178, 0.75747, 1.00356, 1.36834, 0.0, -1.60570])
_JOINT_CLAMPS = np.array([
    [-3.14, -0.1], [-0.3,  2.5], [-0.3, 2.5],
    [-0.5,  3.14], [-1.57, 1.57], [-3.14, 0.5]
], dtype=np.float64)
_ARM_VIEW     = None  # ArticulationView, initialized once
_NEEDLE_PATHS = [
    "/Root/robotarm_base/robotarm_base/needle/needle_tip",
    "/Root/robotarm_base/robotarm_base/tm5_700/flange",
]


def _get_ee_world_pos(stage) -> np.ndarray:
    """Return the needle_tip world-space position from the USD stage."""
    xform_cache = UsdGeom.XformCache()
    for path in _NEEDLE_PATHS:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            mat = xform_cache.GetLocalToWorldTransform(prim)
            return np.array([mat[3][0], mat[3][1], mat[3][2]])
    return None


_SIM = None
_SCENE = None
_IK_CONTROLLER = None
_ROBOT_ENTITY_CFG = None

def _init_ik_system(stage) -> bool:
    global _SIM, _SCENE, _IK_CONTROLLER, _ROBOT_ENTITY_CFG
    if _SCENE is not None:
        return True

    try:
        import torch
        from isaaclab.sim import SimulationContext, SimulationCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.assets import ArticulationCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.utils import configclass
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
        from isaaclab.managers import SceneEntityCfg
        from pxr import UsdPhysics

        # 1. Check or create SimulationContext
        _SIM = SimulationContext.instance()
        if _SIM is None:
            print("[IK] SimulationContext not found, initializing standard SimulationContext...")
            sim_cfg = SimulationCfg(dt=1/60, device="cuda:0")
            _SIM = SimulationContext(sim_cfg)

        # 2. Find robot articulation path on stage
        robot_path = None
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                robot_path = str(prim.GetPath())
                break

        if robot_path is None:
            print("[IK] ERROR: No ArticulationRootAPI found on stage.")
            return False

        print(f"[IK] Found robot articulation path: {robot_path}")

        # 3. Create Scene config
        @configclass
        class MinimalSceneCfg(InteractiveSceneCfg):
            num_envs = 1
            env_spacing = 5.0
            robot = ArticulationCfg(
                prim_path=robot_path,
                spawn=None,
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

        print("[IK] Constructing InteractiveScene...")
        _SCENE = InteractiveScene(MinimalSceneCfg())
        print("[IK] InteractiveScene created successfully.")
        
        # Reset Simulation Context to sync PhysX views
        _SIM.reset()
        _SCENE.update(dt=1/60)

        robot = _SCENE["robot"]
        _get_root_base_offset(stage, robot)
        _init_rigid_body_views()
        
        # Configure scene entity for target body and joint ids
        _ROBOT_ENTITY_CFG = SceneEntityCfg("robot", joint_names=["joint_[1-6]"], body_names=["needle_tip"])
        _ROBOT_ENTITY_CFG.resolve(_SCENE)

        # Configure DifferentialIKController
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.01}
        )
        _IK_CONTROLLER = DifferentialIKController(diff_ik_cfg, num_envs=1, device=_SIM.device)
        print("[IK] DifferentialIKController created successfully.")
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False


def follow_target_ik(stage, tumor_pos: list):
    """
    Solve IK so the needle_tip reaches tumor_pos, using Isaac Lab's 
    DifferentialIKController and updates joint states to simulation.
    """
    global _SIM, _SCENE, _IK_CONTROLLER, _ROBOT_ENTITY_CFG, UI_FOLLOW_LABEL

    def _ui(msg):
        if UI_FOLLOW_LABEL is not None:
            UI_FOLLOW_LABEL.text = msg
        print(f"[IK] {msg.strip()}")

    _ui("  [IK] Initialising arm view and IK controller...")
    if not _init_ik_system(stage):
        _ui("  [IK] ERROR: Failed to initialize Isaac Lab Articulation / IK controller.")
        return

    import torch
    from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms, quat_mul

    robot = _SCENE["robot"]
    ik_joint_ids = _ROBOT_ENTITY_CFG.joint_ids
    ee_idx = _ROBOT_ENTITY_CFG.body_ids[0]
    _get_root_base_offset(stage, robot)
    _init_rigid_body_views()

    # Save current joint states
    current_joint_pos = robot.data.joint_pos.clone()
    current_joint_vel = robot.data.joint_vel.clone()

    # Reset arm to known default joint positions temporarily to get default EE orientation and marker offsets
    default_joint_pos = torch.tensor([_DEFAULT_JOINT_POS], dtype=torch.float32, device=_SIM.device)
    robot.write_joint_state_to_sim(default_joint_pos, torch.zeros_like(default_joint_pos))
    _SCENE.write_data_to_sim()
    _SCENE.update(dt=1/60)

    # === Compute initial marker offsets from EE ===
    init_ee_pos_w = robot.data.body_pos_w[:, ee_idx].clone()
    init_ee_quat_w = robot.data.body_quat_w[:, ee_idx].clone()

    # Restore arm to current joint states so we start the motion from the current posture
    robot.write_joint_state_to_sim(current_joint_pos, current_joint_vel)
    robot.set_joint_position_target(current_joint_pos, joint_ids=ik_joint_ids)
    _SCENE.write_data_to_sim()

    # Step simulation 5 times to let physics and stage sync up the restored joints
    for _ in range(5):
        _SIM.step()
        _SCENE.update(dt=1/60)

    
    marker_offsets = {}
    if _RB_VIEWS:
        for name, view in _RB_VIEWS.items():
            if name == "car": continue
            try:
                tf = view.get_transforms().clone()
                pos = tf[:, 0:3]
                quat_wxyz = torch.cat([tf[:, 6:7], tf[:, 3:6]], dim=1)
                loc_pos, loc_quat = subtract_frame_transforms(init_ee_pos_w, init_ee_quat_w, pos, quat_wxyz)
                marker_offsets[name] = (loc_pos, loc_quat)
            except Exception as e:
                print(f"[IK] Warning computing offset for {name}: {e}")
    else:
        print("[IK] WARN: No marker rigid body views available; UM/EM/BM cannot follow the arm.")

    def _sync_markers():
        if _RB_VIEWS and marker_offsets:
            curr_ee_pos = robot.data.body_pos_w[:, ee_idx]
            curr_ee_quat = robot.data.body_quat_w[:, ee_idx]
            for name, (loc_pos, loc_quat) in marker_offsets.items():
                try:
                    world_pos, world_quat = combine_frame_transforms(curr_ee_pos, curr_ee_quat, loc_pos, loc_quat)
                    world_q_xyzw = torch.cat([world_quat[:, 1:4], world_quat[:, 0:1]], dim=1)
                    view = _RB_VIEWS[name]
                    tf = view.get_transforms().clone()
                    tf[:, 0:3] = world_pos
                    tf[:, 3:7] = world_q_xyzw
                    view.set_transforms(tf)
                except Exception as e:
                    pass

    # Save target pose and default EE orientation
    target_pos_w = torch.tensor([tumor_pos], dtype=torch.float32, device=_SIM.device)
    default_ee_quat_w = robot.data.body_quat_w[:, ee_idx].clone()
    
    # Apply custom tumor angle (rotation around X-axis in local EE frame relative to default orientation)
    angle_deg = getattr(args_cli, "tumor_angle", 0.0)
    half_rad = math.radians(angle_deg) / 2.0
    q_rot_x = torch.tensor([[math.cos(half_rad), math.sin(half_rad), 0.0, 0.0]], device=_SIM.device)
    target_quat_w = quat_mul(q_rot_x, default_ee_quat_w)

    _ui("  [IK] Solving (DifferentialIKController, DLS)...")

    MAX_ITER = 100
    TOL = 0.001  # 1mm convergence
    dist = float("inf")

    for it in range(MAX_ITER):
        _SCENE.update(dt=1/60)

        curr_ee_pos_w = robot.data.body_pos_w[:, ee_idx]
        curr_ee_quat_w = robot.data.body_quat_w[:, ee_idx]

        dist = torch.norm(target_pos_w - curr_ee_pos_w).item()
        if dist < TOL:
            break

        root_pos_w = robot.data.root_pos_w
        root_quat_w = robot.data.root_quat_w

        ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, curr_ee_pos_w, curr_ee_quat_w)
        target_pos_b, target_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, target_pos_w, target_quat_w)

        # Compute Jacobian
        ik_ee_jacobi_idx = ee_idx - 1 if robot.is_fixed_base else ee_idx
        jacobian_col_ids = ik_joint_ids if robot.is_fixed_base else [j + 6 for j in ik_joint_ids]

        jacobian = robot.root_physx_view.get_jacobians()[:, ik_ee_jacobi_idx, :, jacobian_col_ids]

        # Get joint positions
        joint_pos = robot.data.joint_pos[:, ik_joint_ids]
        ik_command = torch.cat([target_pos_b, target_quat_b], dim=-1)
        _IK_CONTROLLER.set_command(ik_command, ee_quat=ee_quat_b)
        joint_pos_des = _IK_CONTROLLER.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # Write to simulator and set target
        robot.write_joint_state_to_sim(joint_pos_des, torch.zeros_like(joint_pos_des))
        robot.set_joint_position_target(joint_pos_des, joint_ids=ik_joint_ids)
        _SCENE.write_data_to_sim()

        _SIM.step()
        _sync_markers()

        if (it + 1) % 10 == 0:
            _ui("  [IK] iter " + str(it+1) + "/" + str(MAX_ITER) + "  residual=" + f"{dist*100:.1f}" + " cm")

    # ── Final status ──
    _SCENE.update(dt=1/60)
    ee_final = robot.data.body_pos_w[:, ee_idx][0].cpu().numpy()
    dist_cm = dist * 100
    icon = " [OK]" if dist_cm < 1.0 else " [WARN]"
    
    # Ensure final state is written to simulation drives
    robot.set_joint_position_target(joint_pos_des, joint_ids=ik_joint_ids)
    _SCENE.write_data_to_sim()
    _SIM.step()
    _sync_markers()
    
    msg = (
        "  [IK] Done (" + str(it+1) + " iters)\n"
        "  Needle: (" + f"{ee_final[0]:.3f}" + ", " + f"{ee_final[1]:.3f}" + ", " + f"{ee_final[2]:.3f}" + ")\n"
        "  Tumor:  (" + f"{tumor_pos[0]:.3f}" + ", " + f"{tumor_pos[1]:.3f}" + ", " + f"{tumor_pos[2]:.3f}" + ")\n"
        "  Angle: " + f"{getattr(args_cli, 'tumor_angle', 0.0):+.1f}°\n"
        "  Residual: " + f"{dist_cm:.2f}" + " cm" + icon
    )
    _ui(msg)



def create_tumor_sphere(stage: Usd.Stage, pos: list, radius: float = 0.030):
    """Create a bright orange sphere at the tumor position."""
    # TUMOR_ROOT must exist
    tumor_root = stage.GetPrimAtPath(TUMOR_ROOT)
    if not tumor_root.IsValid():
        UsdGeom.Xform.Define(stage, TUMOR_ROOT)

    # Remove old tumor marker if exists
    old = stage.GetPrimAtPath(TUMOR_SPHERE_PATH)
    if old.IsValid():
        stage.RemovePrim(old.GetPath())

    sphere = UsdGeom.Sphere.Define(stage, TUMOR_SPHERE_PATH)
    sphere.GetRadiusAttr().Set(radius)
    x = UsdGeom.Xformable(sphere.GetPrim())
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    # Bright orange (1.0, 0.5, 0.0)
    sphere.GetDisplayColorAttr().Set(
        Vt.Vec3fArray(1, (Gf.Vec3f(1.0, 0.5, 0.0),))
    )
    sphere.GetDisplayOpacityAttr().Set([0.95])
    print(f"[TUMOR] Marker sphere placed at X={pos[0]:.3f} Y={pos[1]:.3f} Z={pos[2]:.3f}")
    return sphere.GetPrim()


ENTRY_ROOT = "/World/RASSCMAP_Entry"
ENTRY_SPHERE_PATH = "/World/RASSCMAP_Entry/entry_marker"

def create_entry_sphere(stage: Usd.Stage, pos: list, radius: float = 0.030):
    """Create a green sphere at the entry point position."""
    if not stage.GetPrimAtPath(ENTRY_ROOT).IsValid():
        UsdGeom.Xform.Define(stage, ENTRY_ROOT)

    old = stage.GetPrimAtPath(ENTRY_SPHERE_PATH)
    if old.IsValid():
        stage.RemovePrim(old.GetPath())

    sphere = UsdGeom.Sphere.Define(stage, ENTRY_SPHERE_PATH)
    sphere.GetRadiusAttr().Set(radius)
    x = UsdGeom.Xformable(sphere.GetPrim())
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    # Green (0.0, 0.8, 0.2)
    sphere.GetDisplayColorAttr().Set(
        Vt.Vec3fArray(1, (Gf.Vec3f(0.0, 0.8, 0.2),))
    )
    sphere.GetDisplayOpacityAttr().Set([0.95])
    print(f"[ENTRY] Marker sphere placed at X={pos[0]:.3f} Y={pos[1]:.3f} Z={pos[2]:.3f}")
    return sphere.GetPrim()


def find_nearest_waypoint(wp_data: dict, tumor_pos: list):
    """Return (wp_name, dist, ri, pos) of the nearest waypoint to tumor_pos."""
    best_wp   = None
    best_dist = float("inf")
    best_ri   = 0.0
    best_pos  = None
    tp = np.array(tumor_pos)
    for wp_name, data in wp_data.items():
        p = np.array(data["pos"])
        d = float(np.linalg.norm(p - tp))
        if d < best_dist:
            best_dist = d
            best_wp   = wp_name
            best_ri   = data["ri"]
            best_pos  = data["pos"]
    return best_wp, best_dist, best_ri, best_pos


def highlight_nearest_wp(tumor_pos: list):
    """
    Find the waypoint sphere nearest to tumor_pos and highlight it
    in bright yellow. Restores original color to all others.
    Updates UI_NEAREST_LABEL with the result.
    """
    global SPHERE_PRIMS, ORIGINAL_COLORS, UI_NEAREST_LABEL

    if CURRENT_ENV_ID is None or not SPHERE_PRIMS:
        return

    wp_data = ALL_ENV_DATA[CURRENT_ENV_ID]["wp_data"]
    wp_name, dist, ri, pos = find_nearest_waypoint(wp_data, tumor_pos)

    if wp_name is None:
        return

    # Restore all sphere colors to original first
    for wn, prim in SPHERE_PRIMS.items():
        orig_rgb = ORIGINAL_COLORS.get(wn, ri_to_rgb(wp_data.get(wn, {}).get("ri", 0.0)))
        gprim = UsdGeom.Gprim(prim)
        if gprim:
            gprim.GetDisplayColorAttr().Set(
                Vt.Vec3fArray(1, (Gf.Vec3f(orig_rgb[0], orig_rgb[1], orig_rgb[2]),))
            )

    # Highlight the nearest waypoint in bright yellow
    if wp_name in SPHERE_PRIMS:
        gprim = UsdGeom.Gprim(SPHERE_PRIMS[wp_name])
        if gprim:
            gprim.GetDisplayColorAttr().Set(
                Vt.Vec3fArray(1, (Gf.Vec3f(1.0, 1.0, 0.0),))  # Yellow
            )

    reach_str = f"{ri * 100:.1f}%" if ri > 0 else "UNREACHABLE (0%)"
    label_text = (
        f"  Nearest WP to Tumor: {wp_name}\n"
        f"  Distance: {dist * 100:.1f} cm\n"
        f"  Reachability: {reach_str}\n"
        f"  WP Pos: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
    )
    print(f"[HIGHLIGHT] {wp_name}  dist={dist*100:.1f}cm  ri={ri:.2f}")

    if UI_NEAREST_LABEL is not None:
        UI_NEAREST_LABEL.text = label_text


# ─── Environment switching ───────────────────────────────────────────────────
def switch_env(env_id: int):
    """
    Called from the MAIN THREAD only (never from UI callbacks).
    Moves the robot base and updates sphere colours.
    """
    global CURRENT_ENV_ID, SPHERE_PRIMS, STAGE, ORIGINAL_COLORS, NDI_MANUAL_ANGLE_OVERRIDE
    NDI_MANUAL_ANGLE_OVERRIDE = None

    if env_id not in ALL_ENV_DATA:
        print(f"[WARN] env_id={env_id} not in data.")
        return

    data    = ALL_ENV_DATA[env_id]
    wp_data = data["wp_data"]
    STAGE   = get_stage()

    # 1. Move robot base
    if _SCENE is None:
        # Before IK is initialized, we can safely modify USD
        move_robot_base(STAGE, data["base_x"], data["base_y"])
        move_markers_base(STAGE, data["base_x"], data["base_y"])
    else:
        # After IK is initialized and GPU physics is running, USD edits to robot base trigger
        # setGlobalPose on articulation links, which is illegal. Move via tensor API instead.
        import torch
        robot = _SCENE["robot"]
        root_base_offset = _get_root_base_offset(STAGE, robot)
        target_root_x = data["base_x"] + root_base_offset[0].item()
        target_root_y = data["base_y"] + root_base_offset[1].item()
        target_root_z = root_base_offset[2].item()
        
        # 1a. Record the OLD robotarm_base position inferred from the articulation root.
        old_base_x = robot.data.root_pos_w[0, 0].item() - root_base_offset[0].item()
        old_base_y = robot.data.root_pos_w[0, 1].item() - root_base_offset[1].item()
        dx = data["base_x"] - old_base_x
        dy = data["base_y"] - old_base_y
        print(f"[SWITCH] Teleporting base from ({old_base_x:.3f}, {old_base_y:.3f}) "
              f"to ({data['base_x']:.3f}, {data['base_y']:.3f}), delta=({dx:.3f}, {dy:.3f})")
        
        # 1b. Teleport articulation root via tensor API. The articulation root is not
        #     robotarm_base itself; keep the fixed root-base offset measured at reset.
        root_state = robot.data.root_state_w.clone()
        root_state[:, 0] = target_root_x
        root_state[:, 1] = target_root_y
        root_state[:, 2] = target_root_z
        root_state[:, 7:] = 0.0  # Zero velocities
        robot.write_root_state_to_sim(root_state)
        
        # Update default root state so that resets use the new position
        robot.data.default_root_state[:, 0] = target_root_x
        robot.data.default_root_state[:, 1] = target_root_y
        robot.data.default_root_state[:, 2] = target_root_z
        robot.data.default_root_state[:, 7:] = 0.0

        # 1c. Teleport ALL external rigid bodies by the same (dx, dy) via PhysX tensor API
        #     This is critical: USD xform changes are IGNORED by PhysX during GPU simulation.
        #     Only the tensor API can move rigid bodies in the physics engine.
        _teleport_rigid_bodies(dx, dy)

        # 1d. Also update USD xforms for visual consistency when simulation is paused
        move_markers_base(STAGE, data["base_x"], data["base_y"])

        # 4b. Reset joints to default if IK scene is active
        ik_joint_ids = _ROBOT_ENTITY_CFG.joint_ids
        default_joint_pos = torch.tensor([_DEFAULT_JOINT_POS], dtype=torch.float32, device=_SIM.device)
        robot.write_joint_state_to_sim(default_joint_pos, torch.zeros_like(default_joint_pos))
        robot.set_joint_position_target(default_joint_pos, joint_ids=ik_joint_ids)
        _SCENE.write_data_to_sim()
        
        # Step simulation to sync root state, joints, and constraint resolution
        for _ in range(30):
            _SIM.step()
            _SCENE.update(dt=1/60)

    # 2. Update sphere colours (or create on first call)
    if not SPHERE_PRIMS:
        SPHERE_PRIMS = create_spheres(STAGE, wp_data, args_cli.sphere_scale)
        # Save original colors
        ORIGINAL_COLORS = {wn: ri_to_rgb(d["ri"]) for wn, d in wp_data.items()}
    else:
        update_sphere_colors(SPHERE_PRIMS, wp_data)

    update_cmap_visibility_to_current(STAGE)


    # 3. Re-create tumor sphere if needed (spheres were wiped by create_spheres)
    if args_cli.tumor_pos is not None:
        create_tumor_sphere(STAGE, args_cli.tumor_pos)
    if args_cli.entry_pos is not None:
        create_entry_sphere(STAGE, args_cli.entry_pos)

    # 4. Update UI labels
    CURRENT_ENV_ID = env_id
    _refresh_ui_status(env_id, data)
    _refresh_ui_env_buttons(env_id)

    # 5. Reset nearest WP label
    if UI_NEAREST_LABEL is not None:
        UI_NEAREST_LABEL.text = "  [Click 'Highlight Nearest WP to Tumor' to find nearest waypoint]"

    print(f"[UI] Switched to env_id={env_id}  "
          f"base=({data['base_x']:.2f},{data['base_y']:.2f})  "
          f"fg={data['fg_score']:.3e}  succ={data['success_rate']:.1%}")

    # Update NDI Detector for this environment
    update_ndi_detector_for_env(env_id)

    # Align NDI to default LACP
    try:
        align_ndi_to_default_lacp(env_id)
    except Exception as e:
        print(f"[NDI][WARN] Default LACP alignment failed: {e}")

    # NOTE: do NOT call simulation_app.update() here —
    # the main loop will do it on the next frame.


def sync_slider_to_optimized():
    global UI_ANGLE_SLIDER, LAST_OPTIMIZED_ANGLE, _IN_UI_UPDATE, NDI_MANUAL_ANGLE_OVERRIDE
    if NDI_MANUAL_ANGLE_OVERRIDE is None and UI_ANGLE_SLIDER is not None and LAST_OPTIMIZED_ANGLE is not None:
        _IN_UI_UPDATE = True
        UI_ANGLE_SLIDER.model.set_value(float(LAST_OPTIMIZED_ANGLE))
        _IN_UI_UPDATE = False


def _on_angle_slider_changed(value):
    global NDI_MANUAL_ANGLE_OVERRIDE, _IN_UI_UPDATE
    if _IN_UI_UPDATE:
        return
    NDI_MANUAL_ANGLE_OVERRIDE = float(value)
    request_ndi_look_at()


def _on_reset_angle_clicked():
    global NDI_MANUAL_ANGLE_OVERRIDE
    NDI_MANUAL_ANGLE_OVERRIDE = None
    request_ndi_look_at()
    sync_slider_to_optimized()


# ─── UI construction ─────────────────────────────────────────────────────────
_STYLE_NORMAL   = {"Button":{"background_color": 0xFF3A3A3A, "color": 0xFFDDDDDD}}
_STYLE_SELECTED = {"Button":{"background_color": 0xFF007ACC, "color": 0xFFFFFFFF}}
_STYLE_BEST     = {"Button":{"background_color": 0xFF006633, "color": 0xFFFFFFFF}}
_STYLE_ACTION   = {"Button":{"background_color": 0xFF8B4500, "color": 0xFFFFFFFF}}


def build_ui(env_data: dict):
    global UI_WINDOW, UI_STATUS_LABEL, UI_ENV_LABELS, UI_NEAREST_LABEL, UI_NDI_STATUS_LABEL

    # Sort envs; mark the best using (fg_score, success_rate, avg_manip) tie-breaker
    sorted_envs = sorted(env_data.items(), key=lambda x: x[0])
    best_eid = get_best_env_id(env_data)

    UI_WINDOW = ui.Window(
        "RASSCMAP Environment Selector",
        width=360, height=min(80 + 72 * len(env_data) + 320, 1000),
        flags=ui.WINDOW_FLAGS_NO_CLOSE
    )

    with UI_WINDOW.frame:
        with ui.VStack(spacing=4):

            # ── Header ──
            ui.Spacer(height=6)
            ui.Label("  RASSCMAP Interactive Viewer",
                     style={"color": 0xFF00AAFF, "font_size": 16})
            ui.Label(f"  jet_r  |  0=Red (unreachable)  →  1=Blue (full reach)",
                     style={"color": 0xFF888888, "font_size": 12})
            if args_cli.tumor_pos is not None:
                tp = args_cli.tumor_pos
                ep = args_cli.entry_pos if args_cli.entry_pos is not None else [0.0, 0.0, 0.0]
                t_angle = getattr(args_cli, "tumor_angle", 0.0)
                global UI_TUMOR_LABEL
                UI_TUMOR_LABEL = ui.Label(
                    f"  Entry Point: X={ep[0]:.3f} Y={ep[1]:.3f} Z={ep[2]:.3f}\n"
                    f"  Tumor: X={tp[0]:.3f} Y={tp[1]:.3f} Z={tp[2]:.3f}\n"
                    f"  Angle: {t_angle:+.1f}°",
                    style={"color": 0xFF00AAFF, "font_size": 12}
                )

            ui.Separator()
            ui.Spacer(height=4)

            # ── Colormap legend bar (5 steps) ──
            with ui.HStack(height=20):
                steps  = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                labels = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
                cmap   = mpl_cm.get_cmap("jet_r")
                for s, lbl in zip(steps, labels):
                    r, g, b, _ = cmap(s)
                    col = int(b*255)<<16 | int(g*255)<<8 | int(r*255) | 0xFF000000  # omni.ui = 0xAABBGGRR
                    ui.Rectangle(width=40, style={"background_color": col})
                    ui.Label(lbl, width=20, style={"color": 0xFFCCCCCC, "font_size": 10})
            ui.Spacer(height=6)
            ui.Separator()
            ui.Spacer(height=6)

            # ── Env buttons ──
            ui.Label("  Select Environment:", style={"font_size": 13, "color": 0xFFCCCCCC})
            ui.Spacer(height=4)

            for eid, edata in sorted_envs:
                if edata['success_rate'] <= 0.0:
                    continue
                is_best = (eid == best_eid)
                label_str = (
                    f"  Env {eid:02d}  "
                    f"X={edata['base_x']:.2f} Y={edata['base_y']:.2f}  |  "
                    f"cs={edata['success_rate']:.1%}  "
                    f"fg={edata['fg_score']:.2e}"
                    + ("  [BEST]" if is_best else "")
                )
                style = _STYLE_BEST if is_best else _STYLE_NORMAL
                btn   = ui.Button(label_str, height=52, style=style)
                # ⚠ Only set a flag here — actual switch runs on main thread
                btn.set_clicked_fn(lambda e=eid: _request_env_switch(e))
                UI_ENV_LABELS[eid] = btn

            # ── Tumor / Robot Arm Pose Select ──
            if args_cli.tumor_pos is not None:
                ui.Spacer(height=8)
                ui.Separator()
                ui.Spacer(height=4)

                ui.Label("  Robot Arm Pose Select:", style={"font_size": 13, "color": 0xFFCCCCCC})
                with ui.HStack(height=38):
                    entry_pose_btn = ui.Button("  Reach Skin Entry Point (IK)", style=_STYLE_NORMAL)
                    entry_pose_btn.set_clicked_fn(lambda: _request_reach_entry())
                    btn_follow = ui.Button(
                        "  Reach Tumor Target (IK)",
                        style={"Button": {"background_color": 0xFF6B0080, "color": 0xFFFFFFFF}}
                    )
                    btn_follow.set_clicked_fn(lambda: _request_follow())

                ui.Spacer(height=4)

            # ── Toggle CMAP ──
            ui.Spacer(height=8)
            ui.Separator()
            ui.Spacer(height=4)
            UI_TOGGLE_BTN = ui.Button(
                "  Hide CMAP Spheres" if CMAP_VISIBLE else "  Show CMAP Spheres",
                height=40,
                style={"Button": {"background_color": 0xFF444444, "color": 0xFFCCCCCC}}
            )
            UI_TOGGLE_BTN.set_clicked_fn(lambda: _request_toggle_cmap())


            # ── NDI Camera Control ──
            ui.Spacer(height=8)
            ui.Separator()
            ui.Spacer(height=4)
            ui.Label("  NDI Camera Control:", style={"font_size": 13, "color": 0xFFCCCCCC})
            
            with ui.HStack(height=24):
                ui.Label("  Angle Override (°):", width=120)
                global UI_ANGLE_SLIDER
                UI_ANGLE_SLIDER = ui.FloatSlider(min=0.0, max=90.0, step=1.0)
                UI_ANGLE_SLIDER.model.add_value_changed_fn(lambda m: _on_angle_slider_changed(m.as_float))
            ui.Spacer(height=4)
            
            reset_angle_btn = ui.Button(
                "  Reset to Auto-Optimize Angle",
                height=32,
                style={"Button": {"background_color": 0xFF555555, "color": 0xFFDDDDDD}}
            )
            reset_angle_btn.set_clicked_fn(lambda: _on_reset_angle_clicked())
            ui.Spacer(height=4)
            
            UI_NDI_STATUS_LABEL = ui.Label(
                "  NDI Live Status:\n  Initializing...",
                style={"color": 0xFF00FF00, "font_size": 11},
                word_wrap=True
            )

            # ── Status ──
            ui.Spacer(height=8)
            ui.Separator()
            ui.Spacer(height=6)
            UI_STATUS_LABEL = ui.Label(
                "  [Click a button to switch environment]",
                style={"color": 0xFFAAAAAA, "font_size": 12},
                word_wrap=True
            )
            ui.Spacer(height=4)


def _refresh_ui_status(env_id: int, data: dict):
    if UI_STATUS_LABEL is None:
        return
    reach_total = sum(1 for d in data["wp_data"].values() if d["ri"] > 0)
    avg_ri = np.mean([d["ri"] for d in data["wp_data"].values()])
    UI_STATUS_LABEL.text = (
        f"  Current: Env {env_id:02d}\n"
        f"  Base X={data['base_x']:.3f} m  Y={data['base_y']:.3f} m\n"
        f"  Success rate (cs) = {data['success_rate']:.1%}\n"
        f"  Sr_min (all WP reachable?) = {'YES' if data['sr_min'] else 'NO'}\n"
        f"  fg_score = {data['fg_score']:.4e}\n"
        f"  Waypoints reachable = {reach_total}/{len(data['wp_data'])}\n"
        f"  Average RI = {avg_ri:.3f}"
    )


def _refresh_ui_env_buttons(selected_eid: int):
    best_eid = get_best_env_id(ALL_ENV_DATA)
    for eid, btn in UI_ENV_LABELS.items():
        if eid == selected_eid:
            btn.style = _STYLE_SELECTED
        elif eid == best_eid:
            btn.style = _STYLE_BEST
        else:
            btn.style = _STYLE_NORMAL


def _request_env_switch(env_id: int):
    """Called from UI callback thread — only writes a flag, never touches GPU."""
    global _PENDING_ENV_ID
    _PENDING_ENV_ID = env_id
    print(f"[UI] Env {env_id} requested (will switch on next main-loop frame)")


def _request_follow():
    """Called from UI callback thread — only writes a flag."""
    global _PENDING_FOLLOW
    _PENDING_FOLLOW = True
    print("[UI] Follow Target requested")


def _request_reach_entry():
    """Called from UI callback thread — only writes a flag."""
    global _PENDING_REACH_ENTRY
    _PENDING_REACH_ENTRY = True
    print("[UI] Reach Entry Point requested")



def _request_reset_joints():
    """Called from UI callback thread — only writes a flag."""
    global _PENDING_RESET_JOINTS
    _PENDING_RESET_JOINTS = True
    print("[UI] Reset joints to default requested")


def _request_toggle_cmap():
    """Called from UI callback thread — only writes a flag."""
    global _PENDING_TOGGLE_CMAP
    _PENDING_TOGGLE_CMAP = True
    print("[UI] Toggle CMAP requested")


def _request_highlight():
    """Called from UI callback thread — only writes a flag."""
    global _PENDING_HIGHLIGHT
    _PENDING_HIGHLIGHT = True
    print("[UI] Highlight nearest WP requested")


def run_automated_ndi_search():
    global STAGE, NDI_DETECTOR, OPTIMAL_NDI_ANGLES, _SIM, _SCENE, _IK_CONTROLLER, _ROBOT_ENTITY_CFG, _RB_VIEWS
    if args_cli.tumor_pos is None:
        return

    print("[NDI Search] Starting automated NDI optimal angle search...")
    if not _init_ik_system(STAGE):
        print("[NDI Search] [ERROR] Failed to initialize IK system for search.")
        return

    if NDI_DETECTOR is None:
        print("[NDI Search] [ERROR] NDI Detector is not initialized.")
        return

    import torch
    import math
    from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms, quat_mul

    robot = _SCENE["robot"]
    ik_joint_ids = _ROBOT_ENTITY_CFG.joint_ids
    ee_idx = _ROBOT_ENTITY_CFG.body_ids[0]
    root_base_offset = _get_root_base_offset(STAGE, robot)
    
    # Save original base root state and joints to restore later
    orig_root_state = robot.data.root_state_w.clone()
    orig_joints = robot.data.joint_pos.clone()
    orig_joints_vel = robot.data.joint_vel.clone()

    R_h = 1.52
    H = 0.978
    angles_deg = np.linspace(0, 45, 10)
    SUPPORTED_MARKERS = ["BM", "EM", "FM", "UM"]

    for env_id, data in sorted(ALL_ENV_DATA.items()):
        if data['success_rate'] <= 0.0:
            continue

        # Teleport robot base via tensor API
        target_root_x = data["base_x"] + root_base_offset[0].item()
        target_root_y = data["base_y"] + root_base_offset[1].item()
        target_root_z = root_base_offset[2].item()

        root_state = robot.data.root_state_w.clone()
        root_state[:, 0] = target_root_x
        root_state[:, 1] = target_root_y
        root_state[:, 2] = target_root_z
        root_state[:, 7:] = 0.0
        robot.write_root_state_to_sim(root_state)
        
        # Reset joints to default first
        default_joint_pos = torch.tensor([_DEFAULT_JOINT_POS], dtype=torch.float32, device=_SIM.device)
        robot.write_joint_state_to_sim(default_joint_pos, torch.zeros_like(default_joint_pos))
        robot.set_joint_position_target(default_joint_pos, joint_ids=ik_joint_ids)
        _SCENE.write_data_to_sim()
        
        for _ in range(5):
            _SIM.step()
            _SCENE.update(dt=1/60)

        # Setup initial ee pos/quat for marker offset tracking
        init_ee_pos_w = robot.data.body_pos_w[:, ee_idx].clone()
        init_ee_quat_w = robot.data.body_quat_w[:, ee_idx].clone()

        marker_offsets = {}
        if _RB_VIEWS:
            for name, view in _RB_VIEWS.items():
                if name == "car": continue
                try:
                    tf = view.get_transforms().clone()
                    pos = tf[:, 0:3]
                    quat_wxyz = torch.cat([tf[:, 6:7], tf[:, 3:6]], dim=1)
                    loc_pos, loc_quat = subtract_frame_transforms(init_ee_pos_w, init_ee_quat_w, pos, quat_wxyz)
                    marker_offsets[name] = (loc_pos, loc_quat)
                except Exception as e:
                    pass

        def _sync_markers():
            if _RB_VIEWS and marker_offsets:
                curr_ee_pos = robot.data.body_pos_w[:, ee_idx]
                curr_ee_quat = robot.data.body_quat_w[:, ee_idx]
                for name, (loc_pos, loc_quat) in marker_offsets.items():
                    try:
                        world_pos, world_quat = combine_frame_transforms(curr_ee_pos, curr_ee_quat, loc_pos, loc_quat)
                        world_q_xyzw = torch.cat([world_quat[:, 1:4], world_quat[:, 0:1]], dim=1)
                        view = _RB_VIEWS[name]
                        tf = view.get_transforms().clone()
                        tf[:, 0:3] = world_pos
                        tf[:, 3:7] = world_q_xyzw
                        view.set_transforms(tf)
                    except Exception as e:
                        pass

        # Solve IK for target Z=0.95 along trajectory
        target_pos_w = torch.tensor([get_ndi_test_target(args_cli.entry_pos, args_cli.tumor_pos, z_test=0.95)], dtype=torch.float32, device=_SIM.device)
        default_ee_quat_w = robot.data.body_quat_w[:, ee_idx].clone()
        angle_deg = getattr(args_cli, "tumor_angle", 0.0)
        half_rad = math.radians(angle_deg) / 2.0
        q_rot_x = torch.tensor([[math.cos(half_rad), math.sin(half_rad), 0.0, 0.0]], device=_SIM.device)
        target_quat_w = quat_mul(q_rot_x, default_ee_quat_w)

        MAX_ITER = 50
        TOL = 0.001
        success = False

        for it in range(MAX_ITER):
            _SCENE.update(dt=1/60)
            curr_ee_pos_w = robot.data.body_pos_w[:, ee_idx]
            curr_ee_quat_w = robot.data.body_quat_w[:, ee_idx]
            dist = torch.norm(target_pos_w - curr_ee_pos_w).item()
            if dist < TOL:
                success = True
                break

            root_pos_w = robot.data.root_pos_w
            root_quat_w = robot.data.root_quat_w
            ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, curr_ee_pos_w, curr_ee_quat_w)
            target_pos_b, target_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, target_pos_w, target_quat_w)

            ik_ee_jacobi_idx = ee_idx - 1 if robot.is_fixed_base else ee_idx
            jacobian_col_ids = ik_joint_ids if robot.is_fixed_base else [j + 6 for j in ik_joint_ids]
            jacobian = robot.root_physx_view.get_jacobians()[:, ik_ee_jacobi_idx, :, jacobian_col_ids]

            joint_pos = robot.data.joint_pos[:, ik_joint_ids]
            ik_command = torch.cat([target_pos_b, target_quat_b], dim=-1)
            _IK_CONTROLLER.set_command(ik_command, ee_quat=ee_quat_b)
            joint_pos_des = _IK_CONTROLLER.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

            robot.write_joint_state_to_sim(joint_pos_des, torch.zeros_like(joint_pos_des))
            robot.set_joint_position_target(joint_pos_des, joint_ids=ik_joint_ids)
            _SCENE.write_data_to_sim()
            _SIM.step()
            _sync_markers()

        if not success:
            print(f"[NDI Search] Env {env_id}: IK failed to converge to Z=0.95. Skipping NDI search.")
            continue

        # Sync visual markers to reached arm state
        update_ndi_detector_for_env(env_id)
        
        # Update lines to locate markers
        NDI_DETECTOR._update_lines()
        unique_markers = {}
        for info in NDI_DETECTOR.ray_info:
            path = info['end_path']
            pos = info['end_pos']
            if pos is not None:
                unique_markers[path] = pos

        if unique_markers:
            pts_m = np.array(list(unique_markers.values()))
            min_coords = np.min(pts_m, axis=0)
            max_coords = np.max(pts_m, axis=0)
            centroid = (min_coords + max_coords) / 2.0
        else:
            print(f"[NDI Search] Env {env_id}: No visible markers found. Skipping.")
            continue

        # Now query all 10 angles
        angle_passes = []
        ndi_path = "/Root/NDI"
        test_ndi_path = f"/World/envs/env_{env_id}/Root/NDI"
        if STAGE.GetPrimAtPath(test_ndi_path).IsValid():
            ndi_path = test_ndi_path

        for theta in angles_deg:
            theta_rad = np.deg2rad(theta)
            ndi_world_x = centroid[0] + R_h * np.sin(theta_rad)
            ndi_world_y = centroid[1] - R_h * np.cos(theta_rad)
            ndi_world_z = centroid[2] + H
            ndi_world_pos = np.array([ndi_world_x, ndi_world_y, ndi_world_z])

            quat_world = look_at_quaternion_world(ndi_world_pos, centroid)

            # Convert to local relative to NDI parent
            parent_path = ndi_path.rsplit("/", 1)[0]
            parent_prim = STAGE.GetPrimAtPath(parent_path)
            parent_matrix = Gf.Matrix4d(1.0)
            if parent_prim.IsValid():
                parent_xform = UsdGeom.Xformable(parent_prim)
                parent_matrix = parent_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

            q_parent = parent_matrix.ExtractRotation().GetQuaternion()
            gf_quat_world = Gf.Quaternion(float(quat_world[0]), Gf.Vec3d(float(quat_world[1]), float(quat_world[2]), float(quat_world[3])))
            q_local = q_parent.GetInverse() * gf_quat_world
            ndi_local_pos = np.array(parent_matrix.GetInverse().Transform(Gf.Vec3d(*ndi_world_pos)))

            set_prim_transform_orient(STAGE, ndi_path, ndi_local_pos, q_local)

            # Run detector raycast
            NDI_DETECTOR.update_detection(sim=None, debug_enabled=False, error_only_log=True)

            group_visible_balls = {m: 0 for m in SUPPORTED_MARKERS}
            group_total_balls = {m: 0 for m in SUPPORTED_MARKERS}
            for r_info in NDI_DETECTOR.raycast_results:
                group = r_info['end_path'].split("/")[-1].split("_")[0].upper()
                if group not in SUPPORTED_MARKERS: continue
                group_total_balls[group] += 1
                if r_info.get('is_target_hit', False):
                    group_visible_balls[group] += 1

            angle_passed = True
            for group in SUPPORTED_MARKERS:
                vis_count = group_visible_balls[group]
                total = group_total_balls[group]
                if total == 0 or (vis_count < total):
                    angle_passed = False
                    break
            angle_passes.append(angle_passed)

        # Segment logic
        longest_start = -1
        longest_len = 0
        current_start = -1
        current_len = 0

        for i, passed in enumerate(angle_passes):
            if passed:
                if current_start == -1:
                    current_start = i
                current_len += 1
            else:
                if current_len > longest_len:
                    longest_len = current_len
                    longest_start = current_start
                current_start = -1
                current_len = 0
        if current_len > longest_len:
            longest_len = current_len
            longest_start = current_start

        if longest_len > 0:
            mid_idx = longest_start + (longest_len // 2)
            optimal_angle = float(mid_idx * 5.0)
        else:
            optimal_angle = 22.5

        OPTIMAL_NDI_ANGLES[env_id] = optimal_angle
        if args_cli.block_id is not None and args_cli.tumor_pos is not None:
            try:
                entry_pos = args_cli.entry_pos or default_entry_pos(args_cli.tumor_pos, args_cli.tumor_angle)
                lacp_local = centroid - get_env_origin(STAGE, env_id)
                result = {
                    "lacp": [float(v) for v in lacp_local],
                    "hit_rate": int(sum(1 for passed in angle_passes if passed)),
                    "angle_passes": [1 if passed else 0 for passed in angle_passes],
                    "optimal_angle": float(optimal_angle),
                    "env_id": int(env_id),
                    "base_x": float(data["base_x"]),
                    "base_y": float(data["base_y"]),
                }
                record = {
                    "schema_version": 1,
                    "key": build_key(
                        args_cli.tumor_pos,
                        entry_pos,
                        args_cli.tumor_angle,
                        args_cli.block_id,
                        data["base_x"],
                    ),
                    "result": result,
                    "context": {
                        "source": "rasscmap_interactive_viewer",
                        "backup_csv": os.path.abspath(args_cli.backup_csv),
                        "opt_csv": os.path.abspath(args_cli.opt_csv),
                    },
                }
                append_record(args_cli.verified_cache, record)
                print(f"[NDI Cache] Appended verified result for Block {args_cli.block_id}, base_x={data['base_x']:.3f}")
            except Exception as e:
                print(f"[NDI Cache][WARN] Failed to append verified result for env {env_id}: {e}")
        print(f"[NDI Search] Env {env_id} passes: {[1 if p else 0 for p in angle_passes]} -> Optimal Angle chosen: {optimal_angle}°")

    # Restore original base state and joints
    robot.write_root_state_to_sim(orig_root_state)
    robot.write_joint_state_to_sim(orig_joints, orig_joints_vel)
    robot.set_joint_position_target(orig_joints, joint_ids=ik_joint_ids)
    _SCENE.write_data_to_sim()
    for _ in range(10):
        _SIM.step()
        _SCENE.update(dt=1/60)
    print("[NDI Search] Automated NDI optimal angle search completed.")


def main():
    global ALL_ENV_DATA, SPHERE_PRIMS, STAGE, NDI_DETECTOR, _PENDING_NDI_LOOK_AT, _PENDING_RESET_JOINTS, _PENDING_FOLLOW
    backup_csv = args_cli.backup_csv
    opt_csv    = args_cli.opt_csv
    usd_path   = args_cli.usd_path

    # Validate inputs
    for path, name in [(backup_csv, "backup_csv"), (opt_csv, "opt_csv")]:
        if not os.path.exists(path):
            print(f"[ERROR] {name} not found: {path}")
            simulation_app.close()
            return

    # 1. Load all environment data
    ALL_ENV_DATA = load_all_env_data(backup_csv, opt_csv, args_cli.num_orientations)
    if not ALL_ENV_DATA:
        print("[ERROR] No environment data loaded.")
        simulation_app.close()
        return

    # 2. Load USD scene
    load_usd_scene(usd_path)
    STAGE = get_stage()

    # 2b. SimulationContext will be dynamically initialized when Follow Target (IK) is requested

    if args_cli.tumor_pos is None:
        args_cli.tumor_pos = [-0.20, -0.75, 1.05]
        
    create_tumor_sphere(STAGE, args_cli.tumor_pos)

    if args_cli.entry_pos is None:
        theta_rad = math.radians(args_cli.tumor_angle)
        args_cli.entry_pos = [
            args_cli.tumor_pos[0],
            args_cli.tumor_pos[1] - 0.10 * math.tan(theta_rad),
            args_cli.tumor_pos[2] + 0.10
        ]
        
    create_entry_sphere(STAGE, args_cli.entry_pos)
    args_cli.tumor_angle = math.degrees(math.atan2(args_cli.tumor_pos[1] - args_cli.entry_pos[1], args_cli.entry_pos[2] - args_cli.tumor_pos[2]))


    # Initialize NDIDetector
    if NDIDetector is not None:
        try:
            ndi_cfg = NDIConfig()
            NDI_DETECTOR = NDIDetector(ndi_cfg)
            NDI_DETECTOR.initialize()
            print("[NDI] Detector initialized.")
        except Exception as e:
            print(f"[NDI][ERROR] Failed to initialize NDIDetector: {e}")

    # 4. Build UI panel
    build_ui(ALL_ENV_DATA)

    if args_cli.tumor_pos is not None:
        try:
            run_automated_ndi_search()
        except Exception as e:
            print(f"[NDI Search][ERROR] Automated search failed: {e}")
            import traceback
            traceback.print_exc()

    # 5. Auto-select the best environment on startup using correct tie-breaker
    best_eid = get_best_env_id(ALL_ENV_DATA)
    print(f"\n[INFO] Auto-selecting best env: env_id={best_eid}  "
          f"(fg={ALL_ENV_DATA[best_eid]['fg_score']:.3e}, "
          f"cs={ALL_ENV_DATA[best_eid]['success_rate']:.1%})")
    switch_env(best_eid)

    # 6. Main loop — process pending env switches on the main thread
    print("\n[INFO] Interactive viewer ready.")
    print("[INFO] Use the panel on the right to switch environments.")
    print("[INFO] Close the Isaac Sim window to exit.\n")

    try:
        while simulation_app.is_running():
            # Dynamic tumor and entry point tracking: check if either sphere has been dragged in 3D
            if STAGE is not None:
                updated = False
                if args_cli.tumor_pos is not None:
                    current_tumor_pos = get_prim_world_position(STAGE, TUMOR_SPHERE_PATH)
                    if current_tumor_pos is not None:
                        if np.linalg.norm(current_tumor_pos - np.array(args_cli.tumor_pos)) > 0.001:
                            args_cli.tumor_pos = current_tumor_pos.tolist()
                            updated = True
                            print(f"[TUMOR] Tumor dragged in 3D to: {args_cli.tumor_pos}")
                if args_cli.entry_pos is not None:
                    current_entry_pos = get_prim_world_position(STAGE, ENTRY_SPHERE_PATH)
                    if current_entry_pos is not None:
                        if np.linalg.norm(current_entry_pos - np.array(args_cli.entry_pos)) > 0.001:
                            args_cli.entry_pos = current_entry_pos.tolist()
                            updated = True
                            print(f"[ENTRY] Entry point dragged in 3D to: {args_cli.entry_pos}")
                if updated:
                    if args_cli.tumor_pos is not None and args_cli.entry_pos is not None:
                        args_cli.tumor_angle = math.degrees(math.atan2(args_cli.tumor_pos[1] - args_cli.entry_pos[1], args_cli.entry_pos[2] - args_cli.tumor_pos[2]))
                    if UI_TUMOR_LABEL is not None:
                        t_angle = getattr(args_cli, "tumor_angle", 0.0)
                        UI_TUMOR_LABEL.text = (
                            f"  Entry Point: X={args_cli.entry_pos[0]:.3f} Y={args_cli.entry_pos[1]:.3f} Z={args_cli.entry_pos[2]:.3f}\n"
                            f"  Tumor: X={args_cli.tumor_pos[0]:.3f} Y={args_cli.tumor_pos[1]:.3f} Z={args_cli.tumor_pos[2]:.3f}\n"
                            f"  Angle: {t_angle:+.1f}°"
                        )


            global _PENDING_ENV_ID, _PENDING_HIGHLIGHT, _PENDING_FOLLOW, _PENDING_REACH_ENTRY, _PENDING_RESET_JOINTS, _PENDING_TOGGLE_CMAP, _PENDING_NDI_LOOK_AT
            if _PENDING_ENV_ID is not None:
                eid = _PENDING_ENV_ID
                _PENDING_ENV_ID = None
                switch_env(eid)
            if _PENDING_HIGHLIGHT:
                _PENDING_HIGHLIGHT = False
                if args_cli.tumor_pos is not None:
                    highlight_nearest_wp(args_cli.tumor_pos)
            if _PENDING_REACH_ENTRY:
                _PENDING_REACH_ENTRY = False
                if args_cli.entry_pos is not None:
                    if UI_FOLLOW_LABEL is not None:
                        UI_FOLLOW_LABEL.text = "  [IK] Solving with DifferentialIKController for Entry..."
                    follow_target_ik(STAGE, args_cli.entry_pos)
                    # Automatically update LACP and look-at after follow target finishes
                    try:
                        do_ndi_look_at(use_stage_centroid=True)
                    except Exception as e:
                        print(f"[NDI][WARN] Auto look-at post-IK failed: {e}")
                elif UI_FOLLOW_LABEL is not None:
                     UI_FOLLOW_LABEL.text = "  [IK] No entry position was provided."
            if _PENDING_FOLLOW:
                _PENDING_FOLLOW = False
                if args_cli.tumor_pos is not None:
                    follow_target_ik(STAGE, args_cli.tumor_pos)
                    # Automatically update LACP and look-at after follow target finishes
                    try:
                        do_ndi_look_at(use_stage_centroid=True)
                    except Exception as e:
                        print(f"[NDI][WARN] Auto look-at post-IK failed: {e}")

            if _PENDING_TOGGLE_CMAP:
                _PENDING_TOGGLE_CMAP = False
                toggle_cmap_visibility(STAGE)
            if _PENDING_NDI_LOOK_AT:
                _PENDING_NDI_LOOK_AT = False
                do_ndi_look_at()
                
            simulation_app.update()
            
            # NDI Live updates
            if NDI_DETECTOR is not None:
                try:
                    NDI_DETECTOR.update_detection(sim=None, debug_enabled=True, error_only_log=True)
                    NDI_DETECTOR.draw_live(clear_first=True)
                    
                    # Draw blue cross at LACP
                    global CURRENT_LACP
                    if CURRENT_LACP is not None and NDI_DETECTOR.debug_draw_interface is not None:
                        r = 0.05
                        cx, cy, cz = CURRENT_LACP
                        draw_s = [
                            (cx - r, cy, cz),
                            (cx, cy - r, cz),
                            (cx, cy, cz - r)
                        ]
                        draw_e = [
                            (cx + r, cy, cz),
                            (cx, cy + r, cz),
                            (cx, cy, cz + r)
                        ]
                        colors = [
                            [0.0, 0.0, 1.0, 1.0],  # Blue
                            [0.0, 0.0, 1.0, 1.0],
                            [0.0, 0.0, 1.0, 1.0]
                        ]
                        sizes = [3, 3, 3]
                        try:
                            NDI_DETECTOR.debug_draw_interface.draw_lines(draw_s, draw_e, colors, sizes)
                        except Exception:
                            pass
                    
                    status_str = "  NDI Live Marker Status:\n"
                    for m in NDI_DETECTOR.config.SUPPORTED_MARKERS:
                        visible = NDI_DETECTOR.marker_status.get(m, False)
                        vol = NDI_DETECTOR.marker_volumes.get(m)
                        vol_str = f"{vol:.1f} cm³" if vol is not None else "N/A"
                        status_char = "✓" if visible else "✗"
                        status_str += f"    {m:2s}: {status_char} (Vol: {vol_str})\n"
                        
                        # Display individual spheres (a, b, c, d)
                        spheres = [f"{m.lower()}_a", f"{m.lower()}_b", f"{m.lower()}_c", f"{m.lower()}_d"]
                        sphere_chars = []
                        for s in spheres:
                            s_vis = getattr(NDI_DETECTOR, "sphere_status", {}).get(s, False)
                            s_char = "✓" if s_vis else "✗"
                            sphere_chars.append(f"{s[-1]}:{s_char}")
                        status_str += f"        ({' | '.join(sphere_chars)})\n"
                    if UI_NDI_STATUS_LABEL is not None:
                        UI_NDI_STATUS_LABEL.text = status_str

                except Exception as e:
                    print(f"[NDI][LOOP_ERR] {e}")
                    import traceback
                    traceback.print_exc()
                    sys.stdout.flush()
    except KeyboardInterrupt:
        pass

    simulation_app.close()


if __name__ == "__main__":
    main()
