# -*- coding: utf-8 -*-
"""ndi_diffpose.py — NDI 不同位姿偵測測試（含 Look-At 朝向 + Frustum 診斷）

使用 extension_link_out.usd 場景，在多環境下將 NDI 放在不同的 X 位置
（Y、Z 固定），並自動計算 look-at 朝向使 Frustum 最大程度涵蓋 marker。

輸出：
  - 每個 env 的 NDI pose (pos + quat)
  - 每個 env 的 Robot arm base pose
  - 每顆 marker 的 FRUSTUM_DIAG（depth、side 是否通過）
  - 彙整表格 + JSON

用法:
    .\\isaaclab.bat -p scripts\\isaaclab_ws\\final\\ndi_diffpose.py --num_envs 9
"""

import argparse
import sys
import os
import math
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

from isaaclab.app import AppLauncher

# ── CLI 參數 ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="NDI Diffpose - 不同 NDI 位置 + Look-At 朝向測試")
parser.add_argument("--num_envs", type=int, default=9,
                    help="平行環境數（= NDI X 位置取樣數）")
parser.add_argument("--episodes", type=int, default=1,
                    help="評估回合數")
parser.add_argument("--checkpoint", type=str,
                    default=r"logs\rsl_rl\tm5_reach_stable_v2\2026-02-04_00-28-35\model_9999.pt",
                    help="RL policy checkpoint 路徑")
parser.add_argument("--x_start", type=float, default=0.6,
                    help="NDI X 起始座標（env local frame）")
parser.add_argument("--x_end", type=float, default=2.2,
                    help="NDI X 結束座標（env local frame）")
# --lookat_alpha 已移除，改用所有 marker 的無加權中心
parser.add_argument("--debug_verbose", action="store_true",
                    help="啟用詳細輸出")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ── 啟動 Isaac Sim ────────────────────────────────────────────────────────
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Isaac Sim 啟動後才可 import ───────────────────────────────────────────
from pxr import UsdGeom, Gf, Usd
import omni.usd

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg
from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_extension_linkout_cone_orientation_cfg import (
    TM5ExtensionLinkOutFanOrientationEnvCfg,
)
from rsl_rl.modules import ActorCritic

# ── 常數 ──────────────────────────────────────────────────────────────────
# NDI 預設 local 座標（在 extension_link_out.usd 裡的原始位置）
NDI_DEFAULT_LOCAL = np.array([1.0687789916999098, -1.9981836889836426, 2.0277802554202555])

# Robot base local 座標
ROBOTARM_BASE_LOCAL_X = 0.3
ROBOTARM_BASE_LOCAL_Y = -0.1741
ROBOTARM_BASE_LOCAL_Z = 0.9493

# NDI Polaris Vega Frustum 規格（三截面）
# 每項: (depth_m, half_width_X, half_height_Y)
FRUSTUM_PLANES = [
    (0.950, 0.224, 0.240),   # 近端面: 448mm(X) × 480mm(Y)
    (1.532, 0.398, 0.572),   # 中間面: 796mm(X) × 1144mm(Y)
    (2.400, 0.656, 0.783),   # 遠端面: 1312mm(X) × 1566mm(Y)
]
DEPTH_MIN = FRUSTUM_PLANES[0][0]   # 0.950 m
DEPTH_MAX = FRUSTUM_PLANES[-1][0]  # 2.400 m

# Marker 路徑（relative to /Root，16 顆）
MARKER_PATHS_REL = [
    "/robotarm_base/robotarm_base/BM/BM_meter/bm_a",
    "/robotarm_base/robotarm_base/BM/BM_meter/bm_b",
    "/robotarm_base/robotarm_base/BM/BM_meter/bm_c",
    "/robotarm_base/robotarm_base/BM/BM_meter/bm_d",
    "/robotarm_base/robotarm_base/EM/EM/em_a",
    "/robotarm_base/robotarm_base/EM/EM/em_b",
    "/robotarm_base/robotarm_base/EM/EM/em_c",
    "/robotarm_base/robotarm_base/EM/EM/em_d",
    "/FM/FM/FM/FM/fm_a",
    "/FM/FM/FM/FM/fm_b",
    "/FM/FM/FM/FM/fm_c",
    "/FM/FM/FM/FM/fm_d",
    "/robotarm_base/robotarm_base/UM/UM/UM/um_a",
    "/robotarm_base/robotarm_base/UM/UM/UM/um_b",
    "/robotarm_base/robotarm_base/UM/UM/UM/um_c",
    "/robotarm_base/robotarm_base/UM/UM/UM/um_d",
]

# 輸出資料夾
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "occlusion_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# 數學工具
# ══════════════════════════════════════════════════════════════════════════

def rotate_by_quat(vec: np.ndarray, q: np.ndarray) -> np.ndarray:
    """用四元數旋轉向量。q = [qx, qy, qz, qw]"""
    t = 2.0 * np.cross(q[:3], vec)
    return vec + q[3] * t + np.cross(q[:3], t)


def rotate_by_quat_inverse(vec: np.ndarray, q: np.ndarray) -> np.ndarray:
    """用四元數的逆旋轉向量（等效於 q_conj * vec * q）。q = [qx, qy, qz, qw]"""
    q_inv = np.array([-q[0], -q[1], -q[2], q[3]])
    return rotate_by_quat(vec, q_inv)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """四元數乘法 q1 * q2。格式: [qx, qy, qz, qw]"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """四元數的逆（共軛，假設 unit quaternion）。格式: [qx, qy, qz, qw]"""
    return np.array([-q[0], -q[1], -q[2], q[3]])


def look_at_quaternion_world(ndi_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    """計算讓 NDI 的 -Z 軸指向 target 的世界座標四元數，並約束 Roll。

    NDI 相機座標系:
      -Z = 前方（Frustum 射出方向）
       Y = 上方（Frustum 較寬的軸，需對齊世界垂直方向）
       X = 右方

    Up hint = 世界 Z 軸 [0,0,1]，確保 frustum 的 Y 軸（寬邊）對齊上下方向。
    回傳格式: [qx, qy, qz, qw]
    """
    forward = target_pos - ndi_pos
    forward_len = np.linalg.norm(forward)
    if forward_len < 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0])
    forward = forward / forward_len  # 這是 NDI 的 -Z 方向

    # Up hint：世界 Z 軸（天花板方向）
    up_hint = np.array([0.0, 0.0, 1.0])

    # 如果 forward 幾乎平行於 up_hint，改用 world Y 作為 fallback
    if abs(np.dot(forward, up_hint)) > 0.999:
        up_hint = np.array([0.0, 1.0, 0.0])

    # 構建右手座標系
    # Right (X) = cross(forward, up_hint)  → 水平方向
    right = np.cross(forward, up_hint)
    right = right / np.linalg.norm(right)

    # True Up (Y) = cross(right, forward)  → 垂直方向
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # ── 繞 forward 軸轉 90°：讓 frustum 寬邊（Y）橫擺 ──
    # 交換 right ↔ up（加符號維持右手座標系）
    new_right = up
    new_up = -right

    # 旋轉矩陣: 列向量分別是 X(right), Y(up), Z(-forward) 在世界座標系中的表示
    R = np.column_stack([new_right, new_up, -forward])

    # 旋轉矩陣 → 四元數
    return _rotation_matrix_to_quat(R)


def _rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 旋轉矩陣 → 四元數 [qx, qy, qz, qw]"""
    tr = R[0, 0] + R[1, 1] + R[2, 2]

    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qx, qy, qz, qw])
    return q / np.linalg.norm(q)  # 正規化


# ══════════════════════════════════════════════════════════════════════════
# Frustum 檢查（純數學，與 NDIDetector 脫鉤）
# ══════════════════════════════════════════════════════════════════════════

def frustum_check(marker_world: np.ndarray,
                  ndi_pos: np.ndarray,
                  ndi_quat: np.ndarray) -> dict:
    """檢查一顆 marker 是否在 NDI Frustum 內。

    回傳 dict:
        in_frustum: bool
        depth: float         # 沿 NDI -Z 軸的深度（正值 = 在前方）
        local_x, local_y: float  # 市局部 X/Y 偏移
        max_x, max_y: float      # 該深度的容許邊界
        reason: str          # "OK", "FAILED_DEPTH", "FAILED_SIDE"
    """
    # 1. 轉換到 NDI local frame
    diff = marker_world - ndi_pos
    local_vec = rotate_by_quat_inverse(diff, ndi_quat)
    X, Y, Z = local_vec

    # NDI 的前方是 -Z
    depth = -Z

    # 2. 深度檢查
    if depth < DEPTH_MIN or depth > DEPTH_MAX:
        return {
            "in_frustum": False, "depth": depth,
            "local_x": X, "local_y": Y,
            "max_x": 0.0, "max_y": 0.0,
            "reason": "FAILED_DEPTH",
        }

    # 3. 分段線性插值取得容許邊界
    max_x, max_y = _interpolate_frustum_bounds(depth)

    # 4. 側向檢查
    if abs(X) > max_x or abs(Y) > max_y:
        return {
            "in_frustum": False, "depth": depth,
            "local_x": X, "local_y": Y,
            "max_x": max_x, "max_y": max_y,
            "reason": "FAILED_SIDE",
        }

    return {
        "in_frustum": True, "depth": depth,
        "local_x": X, "local_y": Y,
        "max_x": max_x, "max_y": max_y,
        "reason": "OK",
    }


def _interpolate_frustum_bounds(depth: float) -> tuple:
    """根據深度在三截面間線性插值，回傳 (max_x, max_y)"""
    for i in range(len(FRUSTUM_PLANES) - 1):
        d0, hw0, hh0 = FRUSTUM_PLANES[i]
        d1, hw1, hh1 = FRUSTUM_PLANES[i + 1]
        if d0 <= depth <= d1:
            t = (depth - d0) / (d1 - d0)
            return (hw0 + t * (hw1 - hw0),
                    hh0 + t * (hh1 - hh0))
    # 超出範圍時回傳最近截面
    if depth <= FRUSTUM_PLANES[0][0]:
        return FRUSTUM_PLANES[0][1], FRUSTUM_PLANES[0][2]
    return FRUSTUM_PLANES[-1][1], FRUSTUM_PLANES[-1][2]


# ══════════════════════════════════════════════════════════════════════════
# USD 操作
# ══════════════════════════════════════════════════════════════════════════

def quat_to_euler_xyz_deg(q: np.ndarray) -> np.ndarray:
    """四元數 [qx,qy,qz,qw] → Euler XYZ (degrees)"""
    qx, qy, qz, qw = q
    # Roll (X)
    sinr = 2.0 * (qw * qx + qy * qz)
    cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr, cosr)
    # Pitch (Y)
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    # Yaw (Z)
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny, cosy)
    return np.degrees(np.array([roll, pitch, yaw]))


def set_ndi_pose(stage, env_idx: int, new_local_pos: np.ndarray, quat_xyzw: np.ndarray):
    """設定 env_{env_idx} 的 NDI 位置和朝向。

    quat_xyzw: [qx, qy, qz, qw]
    寫入方式: xformOp:translate + xformOp:rotateXYZ (degrees)
    """
    ndi_path = f"/World/envs/env_{env_idx}/Root/NDI"
    prim = stage.GetPrimAtPath(ndi_path)
    if not prim.IsValid():
        print(f"[WARN] NDI prim not found: {ndi_path}")
        return False

    xformable = UsdGeom.Xformable(prim)
    ops = xformable.GetOrderedXformOps()

    # ── 尋找現有 ops ──
    translate_op = None
    rotate_op = None
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            rotate_op = op

    # ── translate ──
    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(float(new_local_pos[0]),
                               float(new_local_pos[1]),
                               float(new_local_pos[2])))

    # ── rotateXYZ (degrees) ──
    euler_deg = quat_to_euler_xyz_deg(quat_xyzw)
    if rotate_op is None:
        rotate_op = xformable.AddRotateXYZOp()
    rotate_op.Set(Gf.Vec3f(float(euler_deg[0]),
                            float(euler_deg[1]),
                            float(euler_deg[2])))
    return True


def get_prim_world_pos(stage, prim_path: str) -> np.ndarray:
    """取得 prim 的 world 座標"""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    wt = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    t = wt.ExtractTranslation()
    return np.array([t[0], t[1], t[2]], dtype=np.float64)


def get_prim_world_transform(stage, prim_path: str):
    """取得 prim 的 world pos + quat [qx,qy,qz,qw]"""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None, None
    wt = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    t = wt.ExtractTranslation()
    pos = np.array([t[0], t[1], t[2]], dtype=np.float64)
    rot = wt.ExtractRotation().GetQuaternion()
    im = rot.GetImaginary()
    quat = np.array([im[0], im[1], im[2], rot.GetReal()], dtype=np.float64)
    return pos, quat


def get_fm_center(stage, env_idx: int) -> np.ndarray:
    """取得 FM 四顆 marker 的中心世界座標"""
    fm_paths = [f"/World/envs/env_{env_idx}/Root/FM/FM/FM/FM/fm_{c}" for c in "abcd"]
    pts = []
    for p in fm_paths:
        pos = get_prim_world_pos(stage, p)
        if pos is not None:
            pts.append(pos)
    if len(pts) == 0:
        return None
    return np.mean(pts, axis=0)


def get_all_markers_centroid(stage, env_idx: int) -> np.ndarray:
    """取得所有 16 顆 marker 的無加權中心世界座標"""
    pts = []
    for rel_path in MARKER_PATHS_REL:
        full_path = f"/World/envs/env_{env_idx}/Root{rel_path}"
        pos = get_prim_world_pos(stage, full_path)
        if pos is not None:
            pts.append(pos)
    if len(pts) == 0:
        return None
    return np.mean(pts, axis=0)


def get_parent_world_quat(stage, env_idx: int) -> np.ndarray:
    """取得 NDI 的 parent prim (/Root) 的世界旋轉四元數 [qx,qy,qz,qw]"""
    root_path = f"/World/envs/env_{env_idx}/Root"
    _, quat = get_prim_world_transform(stage, root_path)
    if quat is None:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return quat


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    num_envs = args_cli.num_envs
    x_positions = np.linspace(args_cli.x_start, args_cli.x_end, num_envs)

    print("=" * 70)
    print("NDI Diffpose — 不同 NDI X 位置 + Look-At 朝向 + Frustum 診斷")
    print("=" * 70)
    print(f"  環境數: {num_envs}")
    print(f"  NDI X 範圍: [{args_cli.x_start:.2f}, {args_cli.x_end:.2f}]")
    print(f"  NDI X 取樣: {[f'{x:.3f}' for x in x_positions]}")
    print(f"  Y 固定: {NDI_DEFAULT_LOCAL[1]:.4f}")
    print(f"  Z 固定: {NDI_DEFAULT_LOCAL[2]:.4f}")
    print(f"  Look-At: 全部 marker 無加權中心")
    print(f"  Episodes: {args_cli.episodes}")
    print("=" * 70)

    # ── 1. 建立環境 ──────────────────────────────────────────────────────
    env_cfg = TM5ExtensionLinkOutFanOrientationEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.env_spacing = 5.0
    env_cfg.scene.replicate_physics = False

    if hasattr(env_cfg.commands, "ee_pose"):
        env_cfg.commands.ee_pose.resampling_time_range = (1e10, 1e10)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    stage = env.unwrapped.scene.stage
    print(f"\n[OK] 環境已建立 (device: {device}, num_envs: {num_envs})")

    # ── 2. 計算每個 env 的 look-at 並設定 NDI pose ──────────────────────
    print("\n[INFO] 計算 Look-At 朝向並設定 NDI 位姿...")
    env_ndi_info = []  # 記錄每個 env 的設定

    for i, x in enumerate(x_positions):
        ndi_local_pos = np.array([x, NDI_DEFAULT_LOCAL[1], NDI_DEFAULT_LOCAL[2]])

        # 確認 NDI prim 存在
        ndi_prim_check = get_prim_world_pos(stage, f"/World/envs/env_{i}/Root/NDI")
        if ndi_prim_check is None:
            print(f"  env_{i}: [SKIP] NDI prim not found")
            env_ndi_info.append(None)
            continue

        # 取得 env_origin
        env_root_pos = get_prim_world_pos(stage, f"/World/envs/env_{i}")
        if env_root_pos is None:
            env_root_pos = np.zeros(3)

        # NDI 的 world 座標 ≈ env_origin + ndi_local_pos
        ndi_world_pos = env_root_pos + ndi_local_pos

        # 取得 robot arm base 的 world 座標
        robot_base_path = f"/World/envs/env_{i}/Root/robotarm_base/robotarm_base/tm5_700"
        robot_base_world = get_prim_world_pos(stage, robot_base_path)
        if robot_base_world is None:
            robot_base_world = env_root_pos + np.array([ROBOTARM_BASE_LOCAL_X,
                                                         ROBOTARM_BASE_LOCAL_Y,
                                                         ROBOTARM_BASE_LOCAL_Z])

        # 取得所有 marker 的無加權中心
        marker_centroid = get_all_markers_centroid(stage, i)
        if marker_centroid is None:
            marker_centroid = robot_base_world  # fallback

        # ── 計算 look-at quaternion（世界座標系） ──
        ndi_quat_world = look_at_quaternion_world(ndi_world_pos, marker_centroid)

        # ── 轉換為 parent-local quaternion ──
        # orient op 是相對於 parent (/Root) 的，需要扣除 parent 的旋轉
        # q_local = q_parent_inv * q_world
        parent_quat = get_parent_world_quat(stage, i)
        ndi_quat_local = quat_multiply(quat_inverse(parent_quat), ndi_quat_world)

        # 正規化
        ndi_quat_local = ndi_quat_local / np.linalg.norm(ndi_quat_local)

        # 寫入 NDI pose（local translate + local orient）
        ok = set_ndi_pose(stage, i, ndi_local_pos, ndi_quat_local)

        # 重新讀取確認（world transform）
        ndi_world_actual, ndi_quat_actual = get_prim_world_transform(
            stage, f"/World/envs/env_{i}/Root/NDI")

        info = {
            "env_idx": i,
            "ndi_local_x": float(x),
            "ndi_world_pos": ndi_world_pos.tolist(),
            "ndi_quat_world_xyzw": ndi_quat_world.tolist(),
            "ndi_quat_local_xyzw": ndi_quat_local.tolist(),
            "ndi_world_actual": ndi_world_actual.round(4).tolist() if ndi_world_actual is not None else None,
            "ndi_quat_actual": ndi_quat_actual.round(4).tolist() if ndi_quat_actual is not None else None,
            "robot_base_world": robot_base_world.tolist(),
            "marker_centroid_world": marker_centroid.tolist(),
        }
        env_ndi_info.append(info)

        status = "OK" if ok else "FAIL"
        print(f"  env_{i}: NDI_x={x:.3f}  "
              f"ndi_w={ndi_world_pos.round(3)}  "
              f"centroid={marker_centroid.round(3)}  "
              f"q_local={ndi_quat_local.round(3)}  [{status}]")

    # ── 3. 載入 RL Policy ────────────────────────────────────────────────
    obs_dim = env.unwrapped.observation_manager.group_obs_dim["policy"][0]
    action_dim = env.unwrapped.action_manager.total_action_dim

    policy = ActorCritic(
        num_actor_obs=obs_dim,
        num_critic_obs=obs_dim,
        num_actions=action_dim,
        actor_hidden_dims=[768, 512, 512, 256],
        critic_hidden_dims=[768, 512, 512, 256],
        activation='elu',
        init_noise_std=1.0,
    ).to(device)

    ckpt_path = Path(args_cli.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        policy.load_state_dict(ckpt["model_state_dict"])
    else:
        policy.load_state_dict(ckpt)
    policy.eval()
    print(f"[OK] Policy 已載入: {ckpt_path.name}")

    # ── 4. 執行模擬 + Frustum 診斷（sim.render 同步 + 自有 frustum_check） ──────
    all_results = []

    for ep_idx in range(args_cli.episodes):
        obs, _ = env.reset()

        # 讓手臂運動到 target pose
        max_steps = 300
        step = 0
        dones = torch.zeros(num_envs, dtype=torch.bool, device=device)
        while not dones.all() and step < max_steps:
            with torch.no_grad():
                obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
                actions = policy.act(obs_tensor, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(actions)
            dones = (dones | terminated.squeeze(-1) | truncated.squeeze(-1)
                     if terminated.dim() > 1 else dones | terminated | truncated)
            step += 1

        # ── 強制 physics → USD 同步，讓 get_prim_world_pos 取得手臂運動後的即時位置 ──
        env.sim.render()

        print(f"\n{'='*70}")
        print(f"Episode {ep_idx + 1}/{args_cli.episodes}  (steps={step})")
        print(f"{'='*70}")

        for i in range(num_envs):
            if env_ndi_info[i] is None:
                continue

            # 讀取 NDI 的 world transform
            ndi_pos, ndi_quat = get_prim_world_transform(
                stage, f"/World/envs/env_{i}/Root/NDI")
            if ndi_pos is None:
                continue

            # 讀取 robot arm base world（physics 即時值）
            robot = env.scene["robot"]
            robot_base_phys = robot.data.root_pos_w[i].cpu().numpy()

            print(f"\n── env_{i} ──")
            print(f"  NDI   pos_w = {ndi_pos.round(4)}")
            print(f"  NDI   quat  = {ndi_quat.round(4)} (xyzw)")
            print(f"  Robot pos_w = {robot_base_phys.round(4)}")

            env_marker_results = []

            for rel_path in MARKER_PATHS_REL:
                marker_path = f"/World/envs/env_{i}/Root{rel_path}"
                marker_name = rel_path.split("/")[-1]
                marker_group = marker_name.split("_")[0].upper()

                # ── sim.render() 後，get_prim_world_pos 回傳 physics 同步後的位置 ──
                marker_pos = get_prim_world_pos(stage, marker_path)
                if marker_pos is None:
                    print(f"  [FRUSTUM_DIAG] {marker_name:6s} ({marker_group}) : NOT FOUND")
                    continue

                result = frustum_check(marker_pos, ndi_pos, ndi_quat)

                tag = "  OK  " if result["in_frustum"] else result["reason"]
                print(f"  [FRUSTUM_DIAG] {marker_name:6s} ({marker_group}) : [{tag}]  "
                      f"depth={result['depth']:+.3f}  "
                      f"X={result['local_x']:+.3f}/{result['max_x']:.3f}  "
                      f"Y={result['local_y']:+.3f}/{result['max_y']:.3f}  "
                      f"marker_w={marker_pos.round(3)}")

                env_marker_results.append({
                    "marker": marker_name,
                    "group": marker_group,
                    "in_frustum": result["in_frustum"],
                    "depth": round(result["depth"], 4),
                    "local_x": round(result["local_x"], 4),
                    "local_y": round(result["local_y"], 4),
                    "max_x": round(result["max_x"], 4),
                    "max_y": round(result["max_y"], 4),
                    "reason": result["reason"],
                    "marker_world": marker_pos.round(4).tolist(),
                })

            # 彙整該 env
            in_count = sum(1 for m in env_marker_results if m["in_frustum"])
            total = len(env_marker_results)
            groups = {}
            for m in env_marker_results:
                g = m["group"]
                if g not in groups:
                    groups[g] = {"in": 0, "total": 0}
                groups[g]["total"] += 1
                if m["in_frustum"]:
                    groups[g]["in"] += 1

            group_str = "  ".join(f"{g}:{v['in']}/{v['total']}" for g, v in sorted(groups.items()))
            print(f"  ── 小計: {in_count}/{total} in frustum  [{group_str}]")

            all_results.append({
                "episode": ep_idx,
                "env_idx": i,
                "ndi_local_x": float(x_positions[i]),
                "ndi_world_pos": ndi_pos.round(4).tolist(),
                "ndi_quat_xyzw": ndi_quat.round(4).tolist(),
                "robot_base_world": robot_base_phys.round(4).tolist(),
                "markers_in_frustum": in_count,
                "markers_total": total,
                "group_breakdown": groups,
                "markers": env_marker_results,
            })

    # ── 5. 彙整報告 ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("彙整結果")
    print(f"{'='*70}")
    print(f"{'Env':>4} | {'NDI X':>7} | {'In':>3}/{' Total':>5} | {'Rate':>6} | {'BM':>5} | {'EM':>5} | {'FM':>5} | {'UM':>5}")
    print("-" * 70)
    for r in all_results:
        groups = r["group_breakdown"]
        def gs(g):
            return f"{groups.get(g,{}).get('in',0)}/{groups.get(g,{}).get('total',0)}"
        rate = r["markers_in_frustum"] / r["markers_total"] * 100 if r["markers_total"] > 0 else 0
        print(f"{r['env_idx']:>4} | {r['ndi_local_x']:>7.3f} | "
              f"{r['markers_in_frustum']:>3}/{r['markers_total']:>5} | {rate:>5.1f}% | "
              f"{gs('BM'):>5} | {gs('EM'):>5} | {gs('FM'):>5} | {gs('UM'):>5}")

    # ── 6. 儲存 JSON ────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(OUTPUT_DIR, f"ndi_diffpose_{ts}.json")
    import json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "x_range": [args_cli.x_start, args_cli.x_end],
                "y_fixed": NDI_DEFAULT_LOCAL[1],
                "z_fixed": NDI_DEFAULT_LOCAL[2],
                "lookat": "all_markers_centroid",
                "num_envs": num_envs,
                "episodes": args_cli.episodes,
            },
            "env_setup": [info for info in env_ndi_info if info is not None],
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 結果已儲存: {json_path}")

    env.close()
    print("[DONE] NDI Diffpose 測試完畢")


if __name__ == "__main__":
    main()
