# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use("Agg")

import argparse
import csv
import math
import os
import time

import numpy as np

from ndi_verified_cache import (
    DEFAULT_CACHE_PATH,
    append_record,
    build_key,
    default_entry_pos,
)

from isaacsim import SimulationApp




parser = argparse.ArgumentParser(description="Isaac Sim RASSCMAP replay viewer")
parser.add_argument("--backup_csv", required=True, help="Lab backup_live_records_*.csv")
parser.add_argument("--opt_csv", required=True, help="Lab optimization_results_*.csv")
parser.add_argument(
    "--usd_path",
    default=r"C:\Nick\surgery_team\surgery_team\USD\animation\isaaclab_multi_env.usd",
    help="USD scene to open",
)
parser.add_argument("--num_orientations", type=int, default=5)
parser.add_argument("--sphere_scale", type=float, default=0.015)
parser.add_argument("--tumor_pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument("--entry_pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument("--tumor_angle", type=float, default=0.0, help="Needle insertion angle in degrees")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--robot_base_prim_path", default="/Root/robotarm_base")
parser.add_argument("--robot_prim_path", default="/Root/robotarm_base/robotarm_base/tm5_700")
parser.add_argument("--ee_frame", default="flange")
parser.add_argument("--needle_tip_prim_path", default="/Root/robotarm_base/robotarm_base/needle/needle_tip")
parser.add_argument("--urdf_path", default=r"C:\Nick\surgery_team\tmr_ros2-humble\tmr_ros2-humble\tm_description\urdf\tm5-700-nominal.urdf")
parser.add_argument("--robot_description_path", default=r"C:\Nick\surgery_team\surgery_team\robot_tm5700_skrew.yaml")
parser.add_argument("--ik_max_iters", type=int, default=200)
parser.add_argument("--ik_pos_tol", type=float, default=0.005)
parser.add_argument("--ik_ori_tol", type=float, default=0.05)
parser.add_argument("--no_ik_position_fallback", action="store_true")
parser.add_argument("--allow_positive_joint1", action="store_true")
parser.add_argument("--block_id", type=str, default=None)
parser.add_argument("--verified_cache", type=str, default=DEFAULT_CACHE_PATH)
args = parser.parse_args()

simulation_app = SimulationApp({
    "headless": bool(args.headless),
    "anti_aliasing": 0,  # Disable AA to reduce GPU memory pressure
    "extra_args": [
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
    ]
})

import omni.usd
import omni.ui as ui
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.motion_generation import LulaKinematicsSolver
from pxr import Gf, Usd, UsdGeom, Vt

# Path to NDI detector
import sys
sys.path.append(r"C:\Users\RMML\IsaacLab\scripts\isaaclab_ws\final")
try:
    from ndi_detector_isaaclab_experimental import NDIConfig, NDIDetector
except Exception as e:
    print(f"[NDI][WARN] Failed to import NDIDetector: {e}")
    NDIDetector = None
    NDIConfig = None


CURRENT_ENV_ID = None
CURRENT_LACP = None
PENDING_ENV_ID = None
PENDING_HIGHLIGHT = False
PENDING_TOGGLE_CMAP = False
PENDING_FOLLOW = False
PENDING_REACH_ENTRY = False
PENDING_NDI_LOOK_AT = False
LAST_DRAG_TIME = None
PENDING_NDI_RECOMPUTE = False
NDI_DETECTOR = None
UI_NDI_STATUS_LABEL = None
UI_TUMOR_LABEL = None
ALL_ENV_DATA = {}
NDI_MANUAL_ANGLE_OVERRIDE = None
LAST_OPTIMIZED_ANGLE = 0.0
UI_ANGLE_SLIDER = None
_IN_UI_UPDATE = False
SPHERE_PRIMS = {}
ORIGINAL_COLORS = {}
CMAP_VISIBLE = True
STAGE = None
OPTIMAL_NDI_ANGLES = {}
FULLY_OCCLUDED_ENVS = {}

BASE_TRANSITION_ACTIVE = False
BASE_TRANSITION_START_POS = None
BASE_TRANSITION_TARGET_POS = None
BASE_TRANSITION_STEP = 0
BASE_TRANSITION_TOTAL_STEPS = 90  # 增加此數值可放慢底座移動速度 (例如 90 步 = 1.5 秒，120 步 = 2 秒)

ARM_FOLLOW_TOTAL_STEPS = 90  # 增加此數值可放慢手臂移動速度 (例如 90 步 = 1.5 秒，120 步 = 2 秒，180 步 = 3 秒)

def get_robot_base_translation():
    for path in BASE_CANDIDATES:
        prim = STAGE.GetPrimAtPath(path)
        if prim.IsValid():
            op = get_translate_op(prim)
            current = op.Get() or Gf.Vec3d(0.0, 0.0, 0.0)
            return np.array([current[0], current[1], current[2]], dtype=float)
    return np.array([0.0, 0.0, 0.0])

def move_robot_base_to_pos(stage, base_x, base_y):
    for path in BASE_CANDIDATES:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            op = get_translate_op(prim)
            current = op.Get() or Gf.Vec3d(0.0, 0.0, 0.0)
            op.Set(Gf.Vec3d(float(base_x), float(base_y), float(current[2])))
            return True
    return False

UI_WINDOW = None
UI_STATUS_LABEL = None
UI_NEAREST_LABEL = None
UI_FOLLOW_LABEL = None
UI_ENV_BUTTONS = {}
UI_TOGGLE_BTN = None
IK_CONTROLLER = None

OVERLAY_ROOT = "/World/RASSCMAP_Overlay"
TUMOR_ROOT = "/World/RASSCMAP_Tumor"
TUMOR_SPHERE_PATH = "/World/RASSCMAP_Tumor/tumor_marker"
BASE_CANDIDATES = [args.robot_base_prim_path, "/Root/robotarm_base", "/World/robotarm_base"]

DEFAULT_JOINT_POS = np.deg2rad([-76.4, 21.7, 57.5, 78.4, 9.8, -112.4])

STYLE_NORMAL = {"Button": {"background_color": 0xFF3A3A3A, "color": 0xFFDDDDDD}}
STYLE_SELECTED = {"Button": {"background_color": 0xFF007ACC, "color": 0xFFFFFFFF}}
STYLE_BEST = {"Button": {"background_color": 0xFF006633, "color": 0xFFFFFFFF}}
STYLE_ACTION = {"Button": {"background_color": 0xFF8B4500, "color": 0xFFFFFFFF}}


def jet_r_rgb(value):
    """Small jet_r approximation without matplotlib dependency."""
    v = max(0.0, min(1.0, float(value)))
    jet_v = 1.0 - v
    r = max(0.0, min(1.0, 1.5 - abs(4.0 * jet_v - 3.0)))
    g = max(0.0, min(1.0, 1.5 - abs(4.0 * jet_v - 2.0)))
    b = max(0.0, min(1.0, 1.5 - abs(4.0 * jet_v - 1.0)))
    return r, g, b


def rgb_to_abgr(rgb):
    r, g, b = rgb
    return (int(b * 255) << 16) | (int(g * 255) << 8) | int(r * 255) | 0xFF000000


def quat_wxyz_from_euler_xyz_deg(rx, ry, rz):
    """Return a wxyz quaternion for intrinsic xyz Euler angles in degrees."""
    hx = math.radians(rx) * 0.5
    hy = math.radians(ry) * 0.5
    hz = math.radians(rz) * 0.5
    cx, sx = math.cos(hx), math.sin(hx)
    cy, sy = math.cos(hy), math.sin(hy)
    cz, sz = math.cos(hz), math.sin(hz)
    w = cx * cy * cz - sx * sy * sz
    x = sx * cy * cz + cx * sy * sz
    y = cx * sy * cz - sx * cy * sz
    z = cx * cy * sz + sx * sy * cz
    return np.array([w, x, y, z], dtype=float)


def quat_wxyz_to_rot_matrix(quat):
    """Convert a wxyz quaternion to a 3x3 rotation matrix."""
    q = np.array(quat, dtype=float)
    norm = np.linalg.norm(q)
    if norm == 0.0:
        return np.eye(3)
    w, x, y, z = q / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=float)


def rot_matrix_to_quat_wxyz(rot):
    """Convert a 3x3 rotation matrix to a wxyz quaternion."""
    m = np.array(rot, dtype=float)
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


def gf_matrix_rotation_np(mat):
    rot = np.array([
        [mat[0][0], mat[1][0], mat[2][0]],
        [mat[0][1], mat[1][1], mat[2][1]],
        [mat[0][2], mat[1][2], mat[2][2]],
    ], dtype=float)
    # Normalize columns to eliminate USD scaling factors
    norms = np.linalg.norm(rot, axis=0)
    norms = np.where(norms == 0, 1.0, norms)
    return rot / norms


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(path)


def get_ndi_test_target(entry_pos, tumor_pos, z_test=0.95):
    if entry_pos is None or tumor_pos is None:
        ref = tumor_pos if tumor_pos is not None else np.array([0.0, -0.75, 0.95])
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
    base_pos = get_robot_base_translation()
    base_x = data["base_x"]
    base_y = data["base_y"]
    base_z = base_pos[2]
    um_pos = np.array([base_x, base_y, base_z]) + np.array([0.031517988675667796, -0.50597902213614, 0.9193813560801563])
    
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
        except Exception:
            pass
            
    print(f"[NDI] Look At Center completed. Centroid: {centroid.round(3)}")
    sync_slider_to_optimized()


def request_ndi_look_at():
    global PENDING_NDI_LOOK_AT
    PENDING_NDI_LOOK_AT = True


def load_all_env_data(backup_csv_path, opt_csv_path, num_orientations):
    raw = {}
    with open(backup_csv_path, newline="") as f:
        for row in csv.DictReader(f):
            eid = int(row["Env_ID"])
            wp = row["Waypoint"]
            mi = float(row["Manipulability_Index"])
            pos = (float(row["Pos_X"]), float(row["Pos_Y"]), float(row["Pos_Z"]))
            raw.setdefault(eid, {})
            raw[eid].setdefault(wp, {"success": 0, "total": 0, "mi_sum": 0.0, "pos": pos})
            raw[eid][wp]["total"] += 1
            if mi > 0.0:
                raw[eid][wp]["success"] += 1
                raw[eid][wp]["mi_sum"] += mi

    opt_rows = {}
    with open(opt_csv_path, newline="") as f:
        for row in csv.DictReader(f):
            opt_rows[int(row["env_id"])] = row

    result = {}
    for eid, wps in raw.items():
        wp_data = {}
        for wp, data in wps.items():
            total = data["total"] or num_orientations
            success = data["success"]
            ri = success / total
            ami = data["mi_sum"] / success if success else 0.0
            wp_data[wp] = {"ri": ri, "ami": ami, "pos": data["pos"]}

        opt = opt_rows.get(eid, {})
        result[eid] = {
            "base_x": float(opt.get("base_x", 0.0)),
            "base_y": float(opt.get("base_y", 0.0)),
            "fg_score": float(opt.get("fg_score", 0.0)),
            "success_rate": float(opt.get("success_rate", 0.0)),
            "sr_min": int(float(opt.get("sr_min", 0))),
            "avg_manip": float(opt.get("avg_manip", 0.0)),
            "wp_data": wp_data,
        }
    print(f"[CSV] Loaded {len(result)} envs from Lab outputs.")
    return result


def get_best_env_id(env_data):
    return max(
        env_data,
        key=lambda eid: (
            env_data[eid]["success_rate"],
            env_data[eid]["sr_min"],
            env_data[eid]["fg_score"],
            env_data[eid]["avg_manip"],
        ),
    )


def load_usd_scene(path):
    if not os.path.exists(path):
        print(f"[WARN] USD not found: {path}")
        return
    print(f"[USD] Opening {path}")
    omni.usd.get_context().open_stage(path)
    for _ in range(90):
        simulation_app.update()


def get_stage():
    return omni.usd.get_context().get_stage()


def get_prim_world_transform_matrix(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)


def get_prim_world_position(stage, prim_path):
    mat = get_prim_world_transform_matrix(stage, prim_path)
    if mat is None:
        return None
    t = mat.ExtractTranslation()
    return np.array([t[0], t[1], t[2]], dtype=float)


def get_env_origin(stage, env_id):
    if env_id is None:
        return np.zeros(3, dtype=float)
    mat = get_prim_world_transform_matrix(stage, f"/World/envs/env_{env_id}")
    if mat is None:
        return np.zeros(3, dtype=float)
    return matrix_translation_np(mat)


def matrix_translation_np(mat):
    t = mat.ExtractTranslation()
    return np.array([t[0], t[1], t[2]], dtype=float)


def get_translate_op(prim):
    xformable = UsdGeom.Xformable(prim)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return op
    xformable.ClearXformOpOrder()
    return xformable.AddTranslateOp()


def move_robot_base(stage, base_x, base_y):
    for path in BASE_CANDIDATES:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            op = get_translate_op(prim)
            current = op.Get() or Gf.Vec3d(0.0, 0.0, 0.0)
            op.Set(Gf.Vec3d(float(base_x), float(base_y), float(current[2])))
            print(f"[BASE] {path} -> ({base_x:.3f}, {base_y:.3f}, {float(current[2]):.3f})")
            return True
    print(f"[WARN] robotarm_base not found in {BASE_CANDIDATES}")
    return False


def create_spheres(stage, wp_data, radius):
    old = stage.GetPrimAtPath(OVERLAY_ROOT)
    if old.IsValid():
        stage.RemovePrim(old.GetPath())
    UsdGeom.Xform.Define(stage, OVERLAY_ROOT)

    prims = {}
    for wp_name, data in wp_data.items():
        path = f"{OVERLAY_ROOT}/{wp_name}_sphere"
        sphere = UsdGeom.Sphere.Define(stage, path)
        sphere.GetRadiusAttr().Set(radius)
        xform = UsdGeom.Xformable(sphere.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*data["pos"]))
        rgb = jet_r_rgb(data["ri"])
        sphere.GetDisplayColorAttr().Set(Vt.Vec3fArray(1, (Gf.Vec3f(*rgb),)))
        sphere.GetDisplayOpacityAttr().Set([1.0])
        prims[wp_name] = sphere.GetPrim()
    return prims


def update_sphere_colors(sphere_prims, wp_data):
    global ORIGINAL_COLORS
    ORIGINAL_COLORS = {}
    for wp_name, prim in sphere_prims.items():
        if wp_name not in wp_data:
            continue
        rgb = jet_r_rgb(wp_data[wp_name]["ri"])
        ORIGINAL_COLORS[wp_name] = rgb
        UsdGeom.Gprim(prim).GetDisplayColorAttr().Set(Vt.Vec3fArray(1, (Gf.Vec3f(*rgb),)))


ENTRY_ROOT = "/World/RASSCMAP_Entry"
ENTRY_SPHERE_PATH = "/World/RASSCMAP_Entry/entry_marker"

def create_entry_sphere(stage, pos, radius=0.03):
    if not stage.GetPrimAtPath(ENTRY_ROOT).IsValid():
        UsdGeom.Xform.Define(stage, ENTRY_ROOT)
    old = stage.GetPrimAtPath(ENTRY_SPHERE_PATH)
    if old.IsValid():
        stage.RemovePrim(old.GetPath())
    sphere = UsdGeom.Sphere.Define(stage, ENTRY_SPHERE_PATH)
    sphere.GetRadiusAttr().Set(radius)
    xform = UsdGeom.Xformable(sphere.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    sphere.GetDisplayColorAttr().Set(Vt.Vec3fArray(1, (Gf.Vec3f(0.0, 0.8, 0.2),)))
    sphere.GetDisplayOpacityAttr().Set([1.0])


def create_tumor_sphere(stage, pos, radius=0.03):
    if not stage.GetPrimAtPath(TUMOR_ROOT).IsValid():
        UsdGeom.Xform.Define(stage, TUMOR_ROOT)
    old = stage.GetPrimAtPath(TUMOR_SPHERE_PATH)
    if old.IsValid():
        stage.RemovePrim(old.GetPath())
    sphere = UsdGeom.Sphere.Define(stage, TUMOR_SPHERE_PATH)
    sphere.GetRadiusAttr().Set(radius)
    xform = UsdGeom.Xformable(sphere.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    sphere.GetDisplayColorAttr().Set(Vt.Vec3fArray(1, (Gf.Vec3f(1.0, 0.45, 0.05),)))
    sphere.GetDisplayOpacityAttr().Set([1.0])


def switch_env(env_id):
    global CURRENT_ENV_ID, SPHERE_PRIMS, BASE_TRANSITION_ACTIVE, BASE_TRANSITION_START_POS, BASE_TRANSITION_TARGET_POS, BASE_TRANSITION_STEP, NDI_MANUAL_ANGLE_OVERRIDE
    NDI_MANUAL_ANGLE_OVERRIDE = None
    data = ALL_ENV_DATA[env_id]
    
    current_pos = get_robot_base_translation()
    target_pos = np.array([data["base_x"], data["base_y"], current_pos[2]], dtype=float)
    
    # Trigger smooth base transition animation
    BASE_TRANSITION_START_POS = current_pos
    BASE_TRANSITION_TARGET_POS = target_pos
    BASE_TRANSITION_STEP = 0
    BASE_TRANSITION_ACTIVE = True
    
    if IK_CONTROLLER is not None:
        IK_CONTROLLER.on_base_switched()
    if not SPHERE_PRIMS:
        SPHERE_PRIMS = create_spheres(STAGE, data["wp_data"], args.sphere_scale)
    else:
        update_sphere_colors(SPHERE_PRIMS, data["wp_data"])
    if args.tumor_pos is not None:
        create_tumor_sphere(STAGE, args.tumor_pos)
    if args.entry_pos is not None:
        create_entry_sphere(STAGE, args.entry_pos)
    CURRENT_ENV_ID = env_id
    refresh_ui_status(env_id)
    refresh_env_buttons(env_id)
    
    # Update NDI Detector for this environment
    update_ndi_detector_for_env(env_id)
    
    try:
        align_ndi_to_default_lacp(env_id)
    except Exception as e:
        print(f"[NDI][WARN] Default LACP alignment failed: {e}")


def highlight_nearest_wp(tumor_pos):
    if CURRENT_ENV_ID is None or not SPHERE_PRIMS:
        return
    data = ALL_ENV_DATA[CURRENT_ENV_ID]["wp_data"]
    best_name = None
    best_dist = float("inf")
    for wp_name, wp in data.items():
        pos = wp["pos"]
        dist = math.sqrt(sum((float(tumor_pos[i]) - pos[i]) ** 2 for i in range(3)))
        if dist < best_dist:
            best_name = wp_name
            best_dist = dist

    update_sphere_colors(SPHERE_PRIMS, data)
    prim = SPHERE_PRIMS[best_name]
    UsdGeom.Gprim(prim).GetDisplayColorAttr().Set(Vt.Vec3fArray(1, (Gf.Vec3f(1.0, 1.0, 0.0),)))
    wp = data[best_name]
    if UI_NEAREST_LABEL:
        UI_NEAREST_LABEL.text = (
            f"  Nearest WP: {best_name}\n"
            f"  Distance: {best_dist * 100.0:.1f} cm\n"
            f"  RI: {wp['ri']:.2f}\n"
            f"  Pos: ({wp['pos'][0]:.3f}, {wp['pos'][1]:.3f}, {wp['pos'][2]:.3f})"
        )


def toggle_cmap_visibility():
    global CMAP_VISIBLE
    CMAP_VISIBLE = not CMAP_VISIBLE
    overlay = STAGE.GetPrimAtPath(OVERLAY_ROOT)
    if overlay.IsValid():
        imageable = UsdGeom.Imageable(overlay)
        imageable.MakeVisible() if CMAP_VISIBLE else imageable.MakeInvisible()
    if UI_TOGGLE_BTN:
        UI_TOGGLE_BTN.text = "  Hide CMAP Spheres" if CMAP_VISIBLE else "  Show CMAP Spheres"


class NativeLulaFollowTarget:
    """Isaac Sim native Follow Target controller using LulaKinematicsSolver."""

    def __init__(self):
        self.world = None
        self.robot = None
        self.solver = None
        self.articulation_controller = None
        self.fixed_local_rot = quat_wxyz_from_euler_xyz_deg(90.0, -120.0, -90.0)
        self.previous_action = None
        self.flange_to_tip_pos = None
        self.flange_to_tip_rot = None
        self.target_flange_rot = None
        self.joint_1_index = 0
        self.last_candidate_j1 = []
        self.initialized = False
        self.last_reachable = False
        self.last_residual = float("inf")

    def initialize(self):
        if self.initialized:
            return True
        missing = [path for path in [args.urdf_path, args.robot_description_path] if not os.path.exists(path)]
        if missing:
            print("[IK] Missing Lula config files:")
            for path in missing:
                print(f"  {path}")
            return False

        robot_prim = STAGE.GetPrimAtPath(args.robot_prim_path)
        if not robot_prim.IsValid():
            print(f"[IK] Robot prim not found: {args.robot_prim_path}")
            return False

        print("[IK] Initializing Isaac Sim World + LulaKinematicsSolver...")
        self.world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
        self.robot = Articulation(args.robot_prim_path, name="rasscmap_tm5_robot")
        self.world.scene.add(self.robot)
        self.world.reset()
        self.world.play()

        self.solver = LulaKinematicsSolver(
            robot_description_path=args.robot_description_path,
            urdf_path=args.urdf_path,
        )
        self.articulation_controller = self.robot.get_articulation_controller()
        self._resolve_joint_1_index()
        robot_mat = get_prim_world_transform_matrix(STAGE, args.robot_prim_path)
        if robot_mat is not None:
            r_base_w = gf_matrix_rotation_np(robot_mat)
            robot_base_pos = matrix_translation_np(robot_mat)
            robot_base_ori = rot_matrix_to_quat_wxyz(r_base_w)
            self.robot.set_world_pose(position=robot_base_pos, orientation=robot_base_ori)
        self._set_default_joints()
        if not self._measure_flange_to_tip_offset():
            return False
        self.initialized = True
        print("[IK] Native Lula IK ready.")
        return True

    def _resolve_joint_1_index(self):
        names = []
        for attr_name in ("dof_names", "joint_names"):
            if hasattr(self.robot, attr_name):
                try:
                    names = list(getattr(self.robot, attr_name))
                    break
                except Exception:
                    pass
        if names:
            for idx, name in enumerate(names):
                if str(name) == "joint_1" or str(name).endswith("/joint_1"):
                    self.joint_1_index = idx
                    break
            print(f"[IK] articulation joints: {names}")
            print(f"[IK] joint_1 index for branch lock: {self.joint_1_index}")
        else:
            print("[IK] Could not read articulation joint names; assuming joint_1 index 0.")

    def _default_joints(self):
        joints = DEFAULT_JOINT_POS.copy()
        if self.robot is not None and self.robot.num_dof > len(joints):
            joints = np.pad(joints, (0, self.robot.num_dof - len(joints)))
        return joints

    def _seed_candidates(self):
        base = self._default_joints()
        candidates = [base]
        for j1 in [-0.25, -0.55, -0.9, -1.25, -1.65, -2.05, -2.45, -2.85]:
            seed = base.copy()
            if self.joint_1_index < len(seed):
                seed[self.joint_1_index] = j1
            candidates.append(seed)
        current = self.robot.get_joint_positions()
        if current is not None:
            candidates.append(self._normalize_joint_positions(current))
        return candidates

    def _set_default_joints(self):
        joints = self._default_joints()
        self.robot.set_joint_positions(joints)
        self.previous_action = None
        for _ in range(5):
            self.world.step(render=True)

    def _measure_flange_to_tip_offset(self):
        robot_mat = get_prim_world_transform_matrix(STAGE, args.robot_prim_path)
        flange_mat = get_prim_world_transform_matrix(STAGE, args.robot_prim_path + "/" + args.ee_frame)
        tip_mat = get_prim_world_transform_matrix(STAGE, args.needle_tip_prim_path)
        if robot_mat is None:
            print(f"[IK] Robot prim transform not found: {args.robot_prim_path}")
            return False
        if flange_mat is None:
            print(f"[IK] Flange frame prim not found: {args.robot_prim_path}/{args.ee_frame}")
            return False
        if tip_mat is None:
            print(f"[IK] Needle tip prim not found: {args.needle_tip_prim_path}")
            return False
        tip_in_flange = flange_mat.GetInverse().Transform(tip_mat.ExtractTranslation())
        self.flange_to_tip_pos = np.array([tip_in_flange[0], tip_in_flange[1], tip_in_flange[2]], dtype=float)

        robot_rot = gf_matrix_rotation_np(robot_mat)
        flange_rot = gf_matrix_rotation_np(flange_mat)
        tip_rot = gf_matrix_rotation_np(tip_mat)
        flange_rot_local = robot_rot.T @ flange_rot
        tip_rot_local = robot_rot.T @ tip_rot
        self.flange_to_tip_rot = flange_rot_local.T @ tip_rot_local

        # Optimizer angle 0 uses the default needle_tip orientation. Lula solves
        # flange, so convert that default needle_tip pose into a flange target.
        self.target_flange_rot = tip_rot_local @ self.flange_to_tip_rot.T
        self.fixed_local_rot = rot_matrix_to_quat_wxyz(self.target_flange_rot)
        print(
            "[IK] flange -> needle_tip offset = "
            f"({self.flange_to_tip_pos[0]:.4f}, {self.flange_to_tip_pos[1]:.4f}, {self.flange_to_tip_pos[2]:.4f})"
        )
        print(f"[IK] 0deg flange target quat(wxyz) = {self.fixed_local_rot}")
        return True

    def on_base_switched(self):
        if self.initialized:
            robot_mat = get_prim_world_transform_matrix(STAGE, args.robot_prim_path)
            if robot_mat is not None:
                r_base_w = gf_matrix_rotation_np(robot_mat)
                robot_base_pos = matrix_translation_np(robot_mat)
                robot_base_ori = rot_matrix_to_quat_wxyz(r_base_w)
                self.robot.set_world_pose(position=robot_base_pos, orientation=robot_base_ori)
            self._set_default_joints()

    def _extract_joint_positions(self, action):
        if hasattr(action, "joint_positions"):
            return np.array(action.joint_positions, dtype=float)
        return np.array(action, dtype=float)

    def _normalize_joint_positions(self, joint_positions):
        joints = np.array(joint_positions, dtype=float).copy()
        revolute_count = min(6, len(joints))
        joints[:revolute_count] = (joints[:revolute_count] + math.pi) % (2.0 * math.pi) - math.pi
        return joints

    def _joint_1_value(self, joint_positions):
        if len(joint_positions) <= self.joint_1_index:
            return float("nan")
        return float(joint_positions[self.joint_1_index])

    def _find_branch_locked_solution(self, target_pos_w, use_orientation):
        self.last_candidate_j1 = []
        best_positive = None
        for seed in self._seed_candidates():
            action, success = self._compute_ik(target_pos_w, seed, use_orientation=use_orientation)
            if not success:
                continue
            joint_positions = self._normalize_joint_positions(self._extract_joint_positions(action))
            j1 = self._joint_1_value(joint_positions)
            self.last_candidate_j1.append(j1)
            if args.allow_positive_joint1 or j1 <= -0.1:
                return ArticulationAction(joint_positions=joint_positions), joint_positions, True
            if best_positive is None:
                best_positive = joint_positions
        if best_positive is not None and args.allow_positive_joint1:
            return ArticulationAction(joint_positions=best_positive), best_positive, True
        return None, None, False

    def _to_articulation_action(self, action):
        if isinstance(action, ArticulationAction):
            return action
        return ArticulationAction(joint_positions=np.array(action, dtype=float))

    def _compute_ik(self, target_tip_pos_w, warm_start, use_orientation=True):
        robot_mat = get_prim_world_transform_matrix(STAGE, args.robot_prim_path)
        if robot_mat is None:
            return None, False

        r_base_w = gf_matrix_rotation_np(robot_mat)
        
        # Apply custom tumor angle (rotation around X-axis in base frame)
        angle_deg = getattr(args, "tumor_angle", 0.0)
        rad = math.radians(angle_deg)
        c = math.cos(rad)
        s = math.sin(rad)
        r_rot_x = np.array([
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c]
        ], dtype=float)
        
        # Rotated flange target in base frame
        r_flange_target_b = r_rot_x @ self.target_flange_rot
        
        # Target flange rotation in world frame
        r_flange_target_w = r_base_w @ r_flange_target_b
        q_flange_target_w = rot_matrix_to_quat_wxyz(r_flange_target_w)
        
        # Target flange position in world frame
        offset = r_flange_target_w @ self.flange_to_tip_pos
        target_flange_pos_w = target_tip_pos_w - offset

        kwargs = {
            "frame_name": args.ee_frame,
            "target_position": target_flange_pos_w,
            "position_tolerance": args.ik_pos_tol,
            "warm_start": warm_start,
        }
        if use_orientation:
            kwargs["target_orientation"] = q_flange_target_w
            kwargs["orientation_tolerance"] = args.ik_ori_tol
        return self.solver.compute_inverse_kinematics(**kwargs)

    def follow(self, target_pos_w):
        if not self.initialize():
            self.last_reachable = False
            return False, "  [IK] ERROR: Native Lula IK initialization failed."

        if CURRENT_ENV_ID is not None:
            data = ALL_ENV_DATA[CURRENT_ENV_ID]
            move_robot_base(STAGE, data["base_x"], data["base_y"])

        # Synchronize base pose between USD, PhysX articulation, and Lula Kinematics Solver
        robot_mat = get_prim_world_transform_matrix(STAGE, args.robot_prim_path)
        if robot_mat is not None:
            r_base_w = gf_matrix_rotation_np(robot_mat)
            robot_base_pos = matrix_translation_np(robot_mat)
            robot_base_ori = rot_matrix_to_quat_wxyz(r_base_w)
            
            # Explicitly set the articulation world pose in PhysX
            self.robot.set_world_pose(position=robot_base_pos, orientation=robot_base_ori)
            for _ in range(5):
                self.world.step(render=False)
            
            # Update Lula kinematics solver base pose
            phys_pos, phys_ori = self.robot.get_world_pose()
            self.solver.set_robot_base_pose(phys_pos, phys_ori)

        self._set_default_joints()
        target_pos_w = np.array(target_pos_w, dtype=float)

        # 1. Solve IK ONCE at the beginning of the action
        solve_mode = "pose"
        success = False
        action, joint_positions, success = self._find_branch_locked_solution(target_pos_w, use_orientation=True)
        
        if not success and not args.no_ik_position_fallback:
            solve_mode = "position-only fallback"
            action, joint_positions, success = self._find_branch_locked_solution(target_pos_w, use_orientation=False)
            if success:
                print("[IK] Pose IK failed; using position-only fallback for needle_tip reach.")

        if not success:
            if self.last_candidate_j1:
                vals = ", ".join(f"{v:.3f}" for v in self.last_candidate_j1[:8])
                failure_reason = f"joint_1 positive branch rejected; candidates [{vals}] rad"
            else:
                failure_reason = f"{solve_mode} IK did not converge"
            print(f"[IK] {failure_reason}")
            self.last_reachable = False
            return False, f"  [IK] Failed: {failure_reason}"

        # We have a valid target joint configuration!
        q_target = np.array(joint_positions, dtype=float)
        q_target = self._normalize_joint_positions(q_target)
        
        # Get starting joint configuration from robot posture
        q_start = self.robot.get_joint_positions()
        if q_start is not None:
            q_current = self._normalize_joint_positions(q_start)
        else:
            q_current = self._default_joints()

        residual = float("inf")

        # 2. Smoothly sweep/interpolate the robot joint configuration in a loop
        # We use a Sine Ease-In-Out linear interpolation and set joint positions directly
        # to bypass PD controller lag, preventing sudden catch-up accelerations.
        q_start_posture = q_current.copy()
        for step in range(ARM_FOLLOW_TOTAL_STEPS):
            t = (step + 1) / ARM_FOLLOW_TOTAL_STEPS
            # Sine ease-in-out interpolation for perfectly smooth transition
            t_smooth = 0.5 * (1.0 - math.cos(t * math.pi))
            
            q_current = q_start_posture + (q_target - q_start_posture) * t_smooth
            
            # Teleport joints directly to prevent physics lag and catch-up jerks
            self.robot.set_joint_positions(q_current)
            self.world.step(render=True)

            # Check convergence to ensure accurate tracking
            tip_pos = get_prim_world_position(STAGE, args.needle_tip_prim_path)
            if tip_pos is not None:
                residual = float(np.linalg.norm(target_pos_w - tip_pos))

        # Retrieve final residual
        tip_pos = get_prim_world_position(STAGE, args.needle_tip_prim_path)
        if tip_pos is not None:
            residual = float(np.linalg.norm(target_pos_w - tip_pos))

        self.last_residual = residual
        reached = bool(math.isfinite(residual) and residual <= max(args.ik_pos_tol, 0.01))
        self.last_reachable = reached
        icon = "[OK]" if reached else "[WARN]"
        residual_text = "unknown" if not math.isfinite(residual) else f"{residual * 100.0:.2f} cm"
        msg = (
            f"  [IK] Done {icon}\n"
            f"  Solver: LulaKinematicsSolver\n"
            f"  Mode: {solve_mode}\n"
            f"  IK frame: {args.ee_frame}\n"
            f"  Reach body: needle_tip\n"
            f"  Tip target: ({target_pos_w[0]:.3f}, {target_pos_w[1]:.3f}, {target_pos_w[2]:.3f})\n"
            f"  Angle: {getattr(args, 'tumor_angle', 0.0):+.1f}°\n"
            f"  Residual: {residual_text}"
        )
        return reached, msg


def request_env_switch(env_id):
    global PENDING_ENV_ID
    PENDING_ENV_ID = env_id


def request_highlight():
    global PENDING_HIGHLIGHT
    PENDING_HIGHLIGHT = True


def request_toggle():
    global PENDING_TOGGLE_CMAP
    PENDING_TOGGLE_CMAP = True


def request_follow():
    global PENDING_FOLLOW
    PENDING_FOLLOW = True


def request_reach_entry():
    global PENDING_REACH_ENTRY
    PENDING_REACH_ENTRY = True


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


def refresh_ui_status(env_id):
    if UI_STATUS_LABEL is None:
        return
    data = ALL_ENV_DATA[env_id]
    reach_total = sum(1 for wp in data["wp_data"].values() if wp["ri"] > 0.0)
    avg_ri = sum(wp["ri"] for wp in data["wp_data"].values()) / max(1, len(data["wp_data"]))
    
    status_text = (
        f"  Current: Env {env_id:02d}\n"
        f"  Base X={data['base_x']:.3f} m  Y={data['base_y']:.3f} m\n"
        f"  Success rate = {data['success_rate']:.1%}\n"
        f"  Sr_min = {'YES' if data['sr_min'] else 'NO'}\n"
        f"  fg_score = {data['fg_score']:.4e}\n"
        f"  Waypoints reachable = {reach_total}/{len(data['wp_data'])}\n"
        f"  Average RI = {avg_ri:.3f}"
    )
    
    if FULLY_OCCLUDED_ENVS.get(env_id, False):
        status_text += "\n\n  *** WARNING: NDI FULLY OCCLUDED! ***\n  DO NOT USE THIS ROBOT BASE!"
        UI_STATUS_LABEL.style = {"color": 0xFF0000FF, "font_size": 12}
    else:
        UI_STATUS_LABEL.style = {"color": 0xFFCCCCCC, "font_size": 11}
        
    UI_STATUS_LABEL.text = status_text


def refresh_env_buttons(selected_eid):
    best_eid = get_best_env_id(ALL_ENV_DATA)
    for eid, btn in UI_ENV_BUTTONS.items():
        btn.style = STYLE_SELECTED if eid == selected_eid else STYLE_BEST if eid == best_eid else STYLE_NORMAL


def build_ui():
    global UI_WINDOW, UI_STATUS_LABEL, UI_NEAREST_LABEL, UI_FOLLOW_LABEL, UI_TOGGLE_BTN, UI_NDI_STATUS_LABEL
    best_eid = get_best_env_id(ALL_ENV_DATA)
    UI_WINDOW = ui.Window("RASSCMAP Sim Replay", width=380, height=980, flags=ui.WINDOW_FLAGS_NO_CLOSE)
    with UI_WINDOW.frame:
        with ui.VStack(spacing=4):
            ui.Spacer(height=6)
            ui.Label("  RASSCMAP Isaac Sim Replay", style={"color": 0xFF00AAFF, "font_size": 16})
            ui.Label("  Lab CSV/JSON -> Sim USD verification", style={"color": 0xFFAAAAAA, "font_size": 12})
            ui.Separator()
            with ui.HStack(height=20):
                for s in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                    ui.Rectangle(width=42, style={"background_color": rgb_to_abgr(jet_r_rgb(s))})
                    ui.Label(f"{s:.1f}", width=24, style={"color": 0xFFCCCCCC, "font_size": 10})
            ui.Separator()
            ui.Label("  Select Environment:", style={"font_size": 13, "color": 0xFFCCCCCC})
            for eid, edata in sorted(ALL_ENV_DATA.items()):
                if edata['success_rate'] <= 0.0:
                    continue
                label = (
                    f"  Env {eid:02d}  X={edata['base_x']:.2f} Y={edata['base_y']:.2f} | "
                    f"cs={edata['success_rate']:.1%} fg={edata['fg_score']:.2e}"
                    + ("  [BEST]" if eid == best_eid else "")
                )
                btn = ui.Button(label, height=48, style=STYLE_BEST if eid == best_eid else STYLE_NORMAL)
                btn.set_clicked_fn(lambda selected=eid: request_env_switch(selected))
                UI_ENV_BUTTONS[eid] = btn

            ui.Separator()
            if args.tumor_pos is not None:
                ep = args.entry_pos if args.entry_pos is not None else [0.0, 0.0, 0.0]
                t_angle = getattr(args, "tumor_angle", 0.0)
                global UI_TUMOR_LABEL
                UI_TUMOR_LABEL = ui.Label(
                    f"  Entry Point: X={ep[0]:.3f} Y={ep[1]:.3f} Z={ep[2]:.3f}\n"
                    f"  Tumor: X={args.tumor_pos[0]:.3f} Y={args.tumor_pos[1]:.3f} Z={args.tumor_pos[2]:.3f}\n"
                    f"  Angle: {t_angle:+.1f}°",
                    style={"color": 0xFF00AAFF, "font_size": 12},
                )
                nearest_btn = ui.Button("  Highlight Nearest WP", height=38, style=STYLE_ACTION)
                nearest_btn.set_clicked_fn(request_highlight)
                with ui.HStack(height=38):
                    entry_pose_btn = ui.Button("  Reach Skin Entry Point (IK)", style=STYLE_NORMAL)
                    entry_pose_btn.set_clicked_fn(request_reach_entry)
                    follow_btn = ui.Button(
                        "  Reach Deep Tumor Target (IK)",
                        style={"Button": {"background_color": 0xFF6B0080, "color": 0xFFFFFFFF}},
                    )
                    follow_btn.set_clicked_fn(request_follow)
                UI_NEAREST_LABEL = None
                UI_FOLLOW_LABEL = None

            UI_TOGGLE_BTN = ui.Button("  Hide CMAP Spheres", height=38)
            UI_TOGGLE_BTN.set_clicked_fn(request_toggle)
            ui.Separator()
            
            # NDI controls
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
            
            UI_NDI_STATUS_LABEL = ui.Label("  NDI Live Status:\n  Initializing...", style={"color": 0xFF00FF00, "font_size": 11}, word_wrap=True)
            ui.Separator()
            
            UI_STATUS_LABEL = ui.Label("  [No env selected]", word_wrap=True)


def run_automated_ndi_search():
    global IK_CONTROLLER, NDI_DETECTOR, OPTIMAL_NDI_ANGLES, STAGE
    if args.tumor_pos is None:
        return
    
    print("[NDI Search] Starting automated NDI optimal angle search...")
    if IK_CONTROLLER is None:
        IK_CONTROLLER = NativeLulaFollowTarget()
    if not IK_CONTROLLER.initialized:
        if not IK_CONTROLLER.initialize():
            print("[NDI Search] [ERROR] Failed to initialize IK Controller for search.")
            return
    
    if NDI_DETECTOR is None:
        print("[NDI Search] [ERROR] NDI Detector is not initialized.")
        return

    # Save original robot joint positions and base translation to restore later
    orig_base_pos = get_robot_base_translation()
    orig_joints = IK_CONTROLLER.robot.get_joint_positions()

    # Target at height 0.95 (Z=0.95) along trajectory
    target_pos_w = get_ndi_test_target(args.entry_pos, args.tumor_pos, z_test=0.95)
    
    R_h = 1.52
    H = 0.978
    angles_deg = np.linspace(0, 45, 10) # 0.0, 5.0, ..., 45.0
    SUPPORTED_MARKERS = ["BM", "EM", "FM", "UM"]

    for env_id, data in sorted(ALL_ENV_DATA.items()):
        if data['success_rate'] <= 0.0:
            continue
            
        # Move robot base temporarily
        move_robot_base_to_pos(STAGE, data["base_x"], data["base_y"])
        
        # Sync base pose in solver & PhysX
        robot_mat = get_prim_world_transform_matrix(STAGE, args.robot_prim_path)
        if robot_mat is not None:
            r_base_w = gf_matrix_rotation_np(robot_mat)
            robot_base_pos = matrix_translation_np(robot_mat)
            robot_base_ori = rot_matrix_to_quat_wxyz(r_base_w)
            IK_CONTROLLER.robot.set_world_pose(position=robot_base_pos, orientation=robot_base_ori)
            IK_CONTROLLER.solver.set_robot_base_pose(robot_base_pos, robot_base_ori)
        
        # Set default joints first
        IK_CONTROLLER._set_default_joints()
        
        # Solve IK to reach target_pos_w
        action, joint_positions, success = IK_CONTROLLER._find_branch_locked_solution(target_pos_w, use_orientation=True)
        if not success and not args.no_ik_position_fallback:
            action, joint_positions, success = IK_CONTROLLER._find_branch_locked_solution(target_pos_w, use_orientation=False)
            
        if not success:
            print(f"[NDI Search] Env {env_id}: IK failed to reach Z=0.95 target. Skipping NDI search.")
            continue
            
        # Apply solved joint positions
        IK_CONTROLLER.robot.set_joint_positions(joint_positions)
        for _ in range(5):
            IK_CONTROLLER.world.step(render=False)
            
        # Update NDI detector for this env to get the correct marker paths
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
        
        # Find optimal angle from angle_passes
        # We want to find the longest contiguous segment of True
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
            FULLY_OCCLUDED_ENVS[env_id] = False
        else:
            optimal_angle = 22.5
            FULLY_OCCLUDED_ENVS[env_id] = True
            
        OPTIMAL_NDI_ANGLES[env_id] = optimal_angle
        if args.block_id is not None and args.tumor_pos is not None:
            try:
                entry_pos = args.entry_pos or default_entry_pos(args.tumor_pos, args.tumor_angle)
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
                        args.tumor_pos,
                        entry_pos,
                        args.tumor_angle,
                        args.block_id,
                        data["base_x"],
                    ),
                    "result": result,
                    "context": {
                        "source": "sim_rasscmap_interactive_viewer",
                        "backup_csv": os.path.abspath(args.backup_csv),
                        "opt_csv": os.path.abspath(args.opt_csv),
                    },
                }
                append_record(args.verified_cache, record)
                print(f"[NDI Cache] Appended verified result for Block {args.block_id}, base_x={data['base_x']:.3f}")
            except Exception as e:
                print(f"[NDI Cache][WARN] Failed to append verified result for env {env_id}: {e}")
        print(f"[NDI Search] Env {env_id} passes: {[1 if p else 0 for p in angle_passes]} -> Optimal Angle chosen: {optimal_angle}°")
    
    # Restore original robot state
    move_robot_base_to_pos(STAGE, orig_base_pos[0], orig_base_pos[1])
    if orig_joints is not None:
        IK_CONTROLLER.robot.set_joint_positions(orig_joints)
    for _ in range(5):
        IK_CONTROLLER.world.step(render=False)
        
    # Update button texts to flag occluded bases
    for eid, btn in UI_ENV_BUTTONS.items():
        if FULLY_OCCLUDED_ENVS.get(eid, False):
            if "OCCLUDED" not in btn.text:
                btn.text += " [OCCLUDED]"
                
    print("[NDI Search] Automated NDI optimal angle search completed.")


def main():
    global ALL_ENV_DATA, STAGE, PENDING_ENV_ID, PENDING_HIGHLIGHT, PENDING_TOGGLE_CMAP, PENDING_FOLLOW, PENDING_REACH_ENTRY, IK_CONTROLLER, BASE_TRANSITION_STEP, BASE_TRANSITION_ACTIVE, NDI_DETECTOR, PENDING_NDI_LOOK_AT, LAST_DRAG_TIME, PENDING_NDI_RECOMPUTE
    backup_csv = resolve_path(args.backup_csv)
    opt_csv = resolve_path(args.opt_csv)
    usd_path = resolve_path(args.usd_path)
    for path in [backup_csv, opt_csv]:
        if not os.path.exists(path):
            print(f"[ERROR] Missing input: {path}")
            simulation_app.close()
            return

    ALL_ENV_DATA = load_all_env_data(backup_csv, opt_csv, args.num_orientations)
    load_usd_scene(usd_path)
    STAGE = get_stage()
    if args.tumor_pos is None:
        args.tumor_pos = [-0.20, -0.75, 0.95]
    create_tumor_sphere(STAGE, args.tumor_pos)
    if args.entry_pos is None:
        theta_rad = math.radians(args.tumor_angle)
        args.entry_pos = [
            args.tumor_pos[0],
            args.tumor_pos[1] - 0.10 * math.tan(theta_rad),
            args.tumor_pos[2] + 0.10
        ]
    create_entry_sphere(STAGE, args.entry_pos)
    args.tumor_angle = math.degrees(math.atan2(args.tumor_pos[1] - args.entry_pos[1], args.entry_pos[2] - args.tumor_pos[2]))
        
    # Initialize NDIDetector
    if NDIDetector is not None:
        try:
            ndi_cfg = NDIConfig()
            NDI_DETECTOR = NDIDetector(ndi_cfg)
            NDI_DETECTOR.initialize()
            print("[NDI] Detector initialized.")
            sys.stdout.flush()
        except Exception as e:
            print(f"[NDI][ERROR] Failed to initialize NDIDetector: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            
    build_ui()
    if args.tumor_pos is not None:
        try:
            run_automated_ndi_search()
        except Exception as e:
            print(f"[NDI Search][ERROR] Automated search failed: {e}")
            import traceback
            traceback.print_exc()

    switch_env(get_best_env_id(ALL_ENV_DATA))
    print("[READY] Isaac Sim RASSCMAP replay viewer is running.")
    sys.stdout.flush()

    try:
        while simulation_app.is_running():
            # Dynamic tumor and entry tracking
            if STAGE is not None:
                updated = False
                if args.tumor_pos is not None:
                    current_tumor_pos = get_prim_world_position(STAGE, TUMOR_SPHERE_PATH)
                    if current_tumor_pos is not None:
                        if np.linalg.norm(current_tumor_pos - np.array(args.tumor_pos)) > 0.001:
                            args.tumor_pos = current_tumor_pos.tolist()
                            updated = True
                            print(f"[TUMOR] Tumor dragged in 3D to: {args.tumor_pos}")
                if args.entry_pos is not None:
                    current_entry_pos = get_prim_world_position(STAGE, ENTRY_SPHERE_PATH)
                    if current_entry_pos is not None:
                        if np.linalg.norm(current_entry_pos - np.array(args.entry_pos)) > 0.001:
                            args.entry_pos = current_entry_pos.tolist()
                            updated = True
                            print(f"[ENTRY] Entry point dragged in 3D to: {args.entry_pos}")
                if updated:
                    LAST_DRAG_TIME = time.time()
                    PENDING_NDI_RECOMPUTE = True
                    if args.tumor_pos is not None and args.entry_pos is not None:
                        args.tumor_angle = math.degrees(math.atan2(args.tumor_pos[1] - args.entry_pos[1], args.entry_pos[2] - args.tumor_pos[2]))
                    if UI_TUMOR_LABEL is not None:
                        ep = args.entry_pos if args.entry_pos is not None else [0.0, 0.0, 0.0]
                        t_angle = getattr(args, "tumor_angle", 0.0)
                        UI_TUMOR_LABEL.text = (
                            f"  Entry Point: X={ep[0]:.3f} Y={ep[1]:.3f} Z={ep[2]:.3f}\n"
                            f"  Tumor: X={args.tumor_pos[0]:.3f} Y={args.tumor_pos[1]:.3f} Z={args.tumor_pos[2]:.3f}\n"
                            f"  Angle: {t_angle:+.1f}°"
                        )

            # Debounced automated NDI recomputation after dragging ends
            if PENDING_NDI_RECOMPUTE and LAST_DRAG_TIME is not None and (time.time() - LAST_DRAG_TIME > 0.5):
                PENDING_NDI_RECOMPUTE = False
                print("[NDI Search] Dragging stopped. Re-running automated NDI search...")
                try:
                    run_automated_ndi_search()
                    if CURRENT_ENV_ID is not None:
                        align_ndi_to_default_lacp(CURRENT_ENV_ID)
                except Exception as e:
                    print(f"[NDI Search][ERROR] Automated NDI search re-run failed: {e}")
                    import traceback
                    traceback.print_exc()

            if PENDING_ENV_ID is not None:
                selected = PENDING_ENV_ID
                PENDING_ENV_ID = None
                switch_env(selected)
            if PENDING_HIGHLIGHT:
                PENDING_HIGHLIGHT = False
                if args.tumor_pos is not None:
                    highlight_nearest_wp(args.tumor_pos)
            if PENDING_TOGGLE_CMAP:
                PENDING_TOGGLE_CMAP = False
                toggle_cmap_visibility()
            if PENDING_REACH_ENTRY:
                PENDING_REACH_ENTRY = False
                if args.entry_pos is not None:
                    if UI_FOLLOW_LABEL:
                        UI_FOLLOW_LABEL.text = "  [IK] Solving with LulaKinematicsSolver..."
                    if IK_CONTROLLER is None:
                        IK_CONTROLLER = NativeLulaFollowTarget()
                    _, msg = IK_CONTROLLER.follow(args.entry_pos)
                    if UI_FOLLOW_LABEL:
                        UI_FOLLOW_LABEL.text = msg
                    try:
                        do_ndi_look_at(use_stage_centroid=True)
                    except Exception as e:
                        print(f"[NDI][WARN] Auto look-at post-IK failed: {e}")
                elif UI_FOLLOW_LABEL:
                     UI_FOLLOW_LABEL.text = "  [IK] No entry position was provided."
            if PENDING_FOLLOW:
                PENDING_FOLLOW = False
                if args.tumor_pos is not None:
                    if UI_FOLLOW_LABEL:
                        UI_FOLLOW_LABEL.text = "  [IK] Solving with LulaKinematicsSolver..."
                    if IK_CONTROLLER is None:
                        IK_CONTROLLER = NativeLulaFollowTarget()
                    deep_target = get_ndi_test_target(args.entry_pos, args.tumor_pos, z_test=0.95)
                    _, msg = IK_CONTROLLER.follow(deep_target)
                    if UI_FOLLOW_LABEL:
                        UI_FOLLOW_LABEL.text = msg
                    # Automatically update LACP and look-at after follow target finishes
                    try:
                        do_ndi_look_at(use_stage_centroid=True)
                    except Exception as e:
                        print(f"[NDI][WARN] Auto look-at post-IK failed: {e}")
                elif UI_FOLLOW_LABEL:
                     UI_FOLLOW_LABEL.text = "  [IK] No --tumor_pos was provided."
            if PENDING_NDI_LOOK_AT:
                PENDING_NDI_LOOK_AT = False
                do_ndi_look_at()
            
            # --- Smooth base transition update ---
            if BASE_TRANSITION_ACTIVE:
                BASE_TRANSITION_STEP += 1
                alpha = BASE_TRANSITION_STEP / BASE_TRANSITION_TOTAL_STEPS
                # Ease in-out sine interpolation
                alpha = 0.5 * (1.0 - math.cos(alpha * math.pi))
                interp_pos = BASE_TRANSITION_START_POS + (BASE_TRANSITION_TARGET_POS - BASE_TRANSITION_START_POS) * alpha
                move_robot_base_to_pos(STAGE, interp_pos[0], interp_pos[1])
                
                if IK_CONTROLLER is not None and IK_CONTROLLER.initialized:
                    robot_mat = get_prim_world_transform_matrix(STAGE, args.robot_prim_path)
                    if robot_mat is not None:
                        r_base_w = gf_matrix_rotation_np(robot_mat)
                        robot_base_pos = matrix_translation_np(robot_mat)
                        robot_base_ori = rot_matrix_to_quat_wxyz(r_base_w)
                        IK_CONTROLLER.robot.set_world_pose(position=robot_base_pos, orientation=robot_base_ori)
                        IK_CONTROLLER.solver.set_robot_base_pose(robot_base_pos, robot_base_ori)
                
                if BASE_TRANSITION_STEP >= BASE_TRANSITION_TOTAL_STEPS:
                    BASE_TRANSITION_ACTIVE = False

            if IK_CONTROLLER is not None and IK_CONTROLLER.initialized:
                IK_CONTROLLER.world.step(render=True)
            else:
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
                    if UI_NDI_STATUS_LABEL is not None:
                        UI_NDI_STATUS_LABEL.text = status_str
                except Exception as e:
                    print(f"[NDI][LOOP_ERR] {e}")
                    import traceback
                    traceback.print_exc()
                    sys.stdout.flush()
    except BaseException as e:
        print(f"[CRITICAL_ERROR] Loop exited with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
