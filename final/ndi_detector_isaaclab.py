"""
NDI 檢測模組 - Isaac Lab 版本
移植自 Bestpose_v7_NDI_enable.py 的 NDIDetector + MarkerVolumeCalculator

主要差異（相對於原版）：
1. Ray 建構子：Ray(start, end_point) 而非 Ray(start, direction)
2. Callback 閉包修正：使用 factory function 避免 idx 被覆蓋
3. Sim 步進：接受 sim 物件，呼叫 sim.step() 而非 world.step()
4. 移除 PhysxTriggerAPI：Isaac Lab 4.5 中觸發器與 eENABLE_DIRECT_GPU_API 衝突
5. 可見性判定：僅依賴 raycast，不require in-trigger 條件
"""

import sys
import os
import time
import json
import itertools
import traceback as tb
import numpy as np
from datetime import datetime
from typing import Optional

# pxr / omni imports（模擬器啟動後才可使用）
import omni.usd
import omni.kit.raycast.query
from pxr import UsdGeom
from isaacsim.util.debug_draw import _debug_draw

# Isaac Sim 的 SimulationManager（用於物理狀態同步）
try:
    from isaacsim.core.simulation_manager import SimulationManager
    _HAS_SIM_MANAGER = True
except ImportError:
    _HAS_SIM_MANAGER = False

try:
    from scipy.spatial import ConvexHull
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False


# ─────────────────────────────────────────────
# 1. 設定類
# ─────────────────────────────────────────────

class NDIConfig:
    """
    NDI 偵測所需的設定 - 精簡版（不含 robot / traj / UI 設定）
    """

    def __init__(self):
        # HM 在 extension_link_out.usd 場景中不存在，故移除
        self.SUPPORTED_MARKERS = ["FM", "BM", "EM", "UM"]
        self.TRIGGER_VOLUME_PATH = "/Root/NDI/mesh_"

        # ── NDI 路徑設定（與 Bestpose_v7_NDI_enable.py 完全一致） ──
        start_prims = (
            "(/Root/NDI/NDI_emitter),"
            "(/Root/NDI/NDI_left_sensor),"
            "(/Root/NDI/NDI_right_sensor)"
        )
        end_prims = (
            "(/Root/robotarm_base/robotarm_base/BM/BM_meter/bm_a),"
            "(/Root/robotarm_base/robotarm_base/BM/BM_meter/bm_b),"
            "(/Root/robotarm_base/robotarm_base/BM/BM_meter/bm_c),"
            "(/Root/robotarm_base/robotarm_base/BM/BM_meter/bm_d),"
            "(/Root/robotarm_base/robotarm_base/EM/EM/em_a),"
            "(/Root/robotarm_base/robotarm_base/EM/EM/em_b),"
            "(/Root/robotarm_base/robotarm_base/EM/EM/em_c),"
            "(/Root/robotarm_base/robotarm_base/EM/EM/em_d),"
            "(/Root/FM/FM/FM/FM/fm_a),"
            "(/Root/FM/FM/FM/FM/fm_b),"
            "(/Root/FM/FM/FM/FM/fm_c),"
            "(/Root/FM/FM/FM/FM/fm_d),"
            "(/Root/robotarm_base/robotarm_base/UM/UM/UM/um_a),"
            "(/Root/robotarm_base/robotarm_base/UM/UM/UM/um_b),"
            "(/Root/robotarm_base/robotarm_base/UM/UM/UM/um_c),"
            "(/Root/robotarm_base/robotarm_base/UM/UM/UM/um_d)"
        )

        self.START_PRIM_PATHS = []
        self.START_HIGHLIGHT_PATHS = []
        for path in start_prims.split(','):
            path = path.strip()
            if path.startswith('(') and path.endswith(')'):
                clean = path[1:-1].strip()
                self.START_PRIM_PATHS.append(clean)
                self.START_HIGHLIGHT_PATHS.append(clean)
            else:
                self.START_PRIM_PATHS.append(path)

        self.END_PRIM_PATHS = []
        self.END_HIGHLIGHT_PATHS = []
        for path in end_prims.split(','):
            path = path.strip()
            if path.startswith('(') and path.endswith(')'):
                clean = path[1:-1].strip()
                self.END_PRIM_PATHS.append(clean)
                self.END_HIGHLIGHT_PATHS.append(clean)
            else:
                self.END_PRIM_PATHS.append(path)

    def print_paths(self):
        print(f"[NDIConfig] 起點: {len(self.START_PRIM_PATHS)} (highlight: {len(self.START_HIGHLIGHT_PATHS)})")
        print(f"[NDIConfig] 終點: {len(self.END_PRIM_PATHS)} (highlight: {len(self.END_HIGHLIGHT_PATHS)})")


# ─────────────────────────────────────────────
# 2. 標記體積 / 固態角計算器
# ─────────────────────────────────────────────

class MarkerVolumeCalculator:
    """計算標記體積和固態角 - 針對單一 NDI（/Root/NDI）"""

    def __init__(self, emitter_path: str = "/Root/NDI"):
        self.emitter_path = emitter_path
        self.emitter_position: Optional[np.ndarray] = None

    def get_emitter_position(self, get_prim_position_func) -> Optional[np.ndarray]:
        pos = get_prim_position_func(self.emitter_path)
        if pos is not None:
            self.emitter_position = np.asarray(pos)
        return self.emitter_position

    def calculate_tetrahedron_volume(self, p0, p1, p2, p3) -> float:
        p0, p1, p2, p3 = np.asarray(p0), np.asarray(p1), np.asarray(p2), np.asarray(p3)
        v1, v2, v3 = p1 - p0, p2 - p0, p3 - p0
        return abs(np.dot(v1, np.cross(v2, v3))) / 6.0

    def calculate_marker_volume(self, visible_positions) -> Optional[float]:
        if self.emitter_position is None or len(visible_positions) < 3:
            return None
        try:
            if _HAS_SCIPY:
                all_pts = np.vstack([self.emitter_position, visible_positions])
                hull = ConvexHull(all_pts)
                return hull.volume * 1_000_000  # m³ → cm³
            else:
                return self._calculate_using_tetrahedra(visible_positions)
        except Exception as e:
            return None

    def _calculate_using_tetrahedra(self, visible_positions) -> float:
        total = 0.0
        vp = visible_positions
        if len(vp) == 3:
            total = self.calculate_tetrahedron_volume(
                self.emitter_position, vp[0], vp[1], vp[2])
        else:
            for i in range(len(vp) - 2):
                total += self.calculate_tetrahedron_volume(
                    self.emitter_position, vp[i], vp[i + 1], vp[i + 2])
        return total * 1_000_000

    def calculate_solid_angle(self, visible_positions) -> Optional[float]:
        if self.emitter_position is None or len(visible_positions) < 3:
            return None
        total = 0.0
        combos = list(itertools.combinations(visible_positions, 3))
        for a, b, c in combos:
            sa = self._calculate_triangle_solid_angle(a, b, c)
            if sa is not None:
                total += sa
        return (total / len(combos)) if combos else None

    def _calculate_triangle_solid_angle(self, a, b, c) -> Optional[float]:
        try:
            v = self.emitter_position
            A = np.asarray(a) - v
            B = np.asarray(b) - v
            C = np.asarray(c) - v
            la, lb, lc = np.linalg.norm(A), np.linalg.norm(B), np.linalg.norm(C)
            if la == 0 or lb == 0 or lc == 0:
                return 0.0
            Au, Bu, Cu = A / la, B / lb, C / lc
            scalar_triple = np.dot(Au, np.cross(Bu, Cu))
            denom = 1 + np.dot(Au, Bu) + np.dot(Bu, Cu) + np.dot(Cu, Au)
            if denom <= 0:
                return 0.0
            return 2 * np.arctan2(abs(scalar_triple), denom)
        except Exception:
            return 0.0


# ─────────────────────────────────────────────
# 3. NDI 檢測器（Isaac Lab 版）
# ─────────────────────────────────────────────

class NDIDetector:
    """
    NDI raycast 偵測 - Isaac Lab 版本

    使用方式：
        config = NDIConfig()
        detector = NDIDetector(config)
        detector.initialize()

        # 每個模擬幀呼叫（需傳入 sim 物件）：
        marker_status = detector.update_detection(sim)
    """

    def __init__(self, config: NDIConfig):
        self.config = config

        # ── 狀態 ──
        self.marker_status = {m: False for m in config.SUPPORTED_MARKERS}
        self.previous_marker_status = {m: False for m in config.SUPPORTED_MARKERS}
        self.frame_count = 0
        self.env_index = -1
        self.error_only_log = False

        # ── Raycast 相關 ──
        self.raycast_interface = None
        self.raycast_results: list = []      # 每條射線的回調結果
        self.ray_info: list = []             # 每條射線的 meta 資訊
        self.line_starts: list = []
        self.line_ends: list = []

        # ── Debug draw ──
        self.debug_draw_interface = None

        # ── Physics Sim View（用於同步 kinematic 狀態）──
        self._physics_sim_view = None

        # ── RigidBodyView 快取（直接從 PhysX 讀取位置）──
        self._rigid_body_views = {}  # path -> RigidBodyView
        self._rb_view_attempted = set()  # 已嘗試建立的路徑

        # ── Marker 動態追蹤（使用 link transform + local offset）──
        # 由於 marker emitters 不是獨立的 rigid body，需要透過父級 link 追蹤
        self._marker_link_map = {}  # marker path -> link path
        self._marker_local_offsets = {}  # marker path -> (relative_pos, local_transform)
        self._marker_offsets_initialized = False
        
        # 定義 marker 所屬的 articulation link
        self._link_paths = {
            "link_6": "/Root/robotarm_base/robotarm_base/tm5_700/link_6",  # 機械臂最後一個 link
            "link_base": "/Root/robotarm_base/robotarm_base/tm5_700/link_0",  # 機械臂根連桿（底座）
        }

        # ── 路徑設定 ──
        self.start_prim_paths = config.START_PRIM_PATHS
        self.start_highlight_paths = config.START_HIGHLIGHT_PATHS
        self.end_prim_paths = config.END_PRIM_PATHS
        self.end_highlight_paths = config.END_HIGHLIGHT_PATHS

        # ── 體積 / 固態角計算器 ──
        self.volume_calculator = MarkerVolumeCalculator(emitter_path="/Root/NDI_02")
        self.marker_volumes = {m: None for m in config.SUPPORTED_MARKERS}
        self.marker_solid_angles = {m: None for m in config.SUPPORTED_MARKERS}

        # ── 成本計算 ──
        self.cost_calculation = {
            'total_cost': 0.0,
            'frame_count': 0,
            'cost_samples': [],
            'start_time': None,
            'calculation_active': False,
        }
        self.cost_params = {
            'temperature': 0.1,
            'occlusion_penalty_base': 50.0,
            'occlusion_penalty_increment': 10.0,
            'penalty_mode': 'dynamic',
        }
        self.occlusion_history = {
            m: {'consecutive_frames': 0,
                'total_occluded_frames': 0,
                'last_status': False}
            for m in config.SUPPORTED_MARKERS
        }

    # ─────────────── 初始化 ───────────────

    def initialize(self) -> bool:
        try:
            self.raycast_interface = omni.kit.raycast.query.acquire_raycast_query_interface()
            self.debug_draw_interface = _debug_draw.acquire_debug_draw_interface()
            
            # physics_sim_view 會在第一次 _update_lines() 時延遲初始化
            # （必須在 sim.reset() 之後才能取得）
            self._physics_sim_view = None
            self._physics_sim_view_attempted = False
            
            # 清空 RigidBodyView 快取（防止重複初始化時留有舊快取）
            self._rigid_body_views = {}
            self._rb_view_attempted = set()
            
            print("[NDIDetector] Initialized (trigger detector disabled in Isaac Lab mode)")
            return True
        except Exception as e:
            print(f"[NDIDetector] initialize failed: {e}")
            return False

    # ─────────────── prim 位置取得 ───────────────

    def _get_physx_position(self, prim_path: str) -> Optional[np.ndarray]:
        """直接從 PhysX RigidBodyView 讀取 prim 的世界座標
        
        這是關鍵修正：當機器人關節運動時，USD xform 可能不會即時同步，
        但 PhysX 中的 RigidBody 位置是正確的。
        """
        if self._physics_sim_view is None:
            return None
        
        # 檢查快取
        if prim_path in self._rigid_body_views:
            rb_view = self._rigid_body_views[prim_path]
            if rb_view is not None:
                try:
                    poses = rb_view.get_transforms()
                    if poses is not None and len(poses) > 0:
                        # poses shape: (N, 7) - [x, y, z, qw, qx, qy, qz]
                        pos = poses[0, :3]
                        return np.array(pos, dtype=float)
                except Exception:
                    pass
            return None
        
        # 尚未嘗試建立此 path 的 RigidBodyView
        if prim_path not in self._rb_view_attempted:
            self._rb_view_attempted.add(prim_path)
            
            # 【抑制警告的關鍵修正】
            # 先確認 USD prim 是 RigidBody 再去建立 view，避免 PhysX 底層印出一堆 Error
            stage = omni.usd.get_context().get_stage()
            if stage:
                from pxr import UsdPhysics
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    self._rigid_body_views[prim_path] = None
                    return None

            try:
                rb_view = self._physics_sim_view.create_rigid_body_view(prim_path)
                if rb_view is not None and rb_view.count > 0:
                    self._rigid_body_views[prim_path] = rb_view
                    # 嘗試立即讀取位置
                    poses = rb_view.get_transforms()
                    if poses is not None and len(poses) > 0:
                        pos = poses[0, :3]
                        return np.array(pos, dtype=float)
            except Exception as e:
                self._rigid_body_views[prim_path] = None  # 標記為不可用
        
        return None

    def _get_prim_position(self, prim_path: str) -> Optional[np.ndarray]:
        """回傳 prim 的世界座標（np.ndarray）或 None
        
        優先使用 PhysX RigidBodyView（動態追蹤），失敗時回退到 USD xform。
        對於 marker（非 rigid body），使用 link transform + local offset。
        """
        # 1. 先嘗試直接從 PhysX 讀取（如果是 rigid body）
        physx_pos = self._get_physx_position(prim_path)
        if physx_pos is not None:
            return physx_pos
        
        # 2. 檢查是否是 marker，使用 link transform + offset 方法
        marker_pos = self._get_marker_position_via_link(prim_path)
        if marker_pos is not None:
            return marker_pos
        
        # 3. 回退到 USD xform（靜態物件或初始化階段）
        try:
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return None
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                return None
            xformable = UsdGeom.Xformable(prim)
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            t = world_transform.ExtractTranslation()
            return np.array([t[0], t[1], t[2]], dtype=float)
        except Exception:
            return None

    def _init_marker_link_tracking(self):
        """初始化 marker 的 link 追蹤系統
        
        對於每個 marker 路徑，找到其父級 link 並計算 local offset。
        【關鍵修正】使用純 USD 座標計算 offset，避免 USD/PhysX 不同步問題。
        """
        if self._marker_offsets_initialized:
            return
        
        if self._physics_sim_view is None:
            return
        
        self._marker_offsets_initialized = True
        
        # 為每個 marker 找到其父級 link 並計算 offset
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        
        # ── 各 link 的 RigidBodyView 及 USD 初始 transform ──
        # marker 追蹤規則：
        #   BM / EM → link_6（隨機械臂末端移動）
        #   UM      → link_base / link_0（隨底座平移，但不跟 link_6 旋轉）
        #   FM      → 不動（世界空間固定），直接用 USD xform fallback
        MARKER_LINK_RULES = {
            "BM": "link_6",
            "EM": "link_6",
            "UM": "link_base",
        }

        # 建立每個需要的 link 的 RigidBodyView 並記錄 USD 初始 transform
        link_usd_transforms = {}   # link_key -> (usd_pos, usd_quat)
        from pxr import Gf
        for link_key, link_path in self._link_paths.items():
            if link_key not in ["link_6", "link_base"]:
                continue
            try:
                rb_view = self._physics_sim_view.create_rigid_body_view(link_path)
                if rb_view is not None and rb_view.count > 0:
                    self._rigid_body_views[link_path] = rb_view
                    print(f"[PhysX] Link tracking enabled ({link_key}): {link_path}")
            except Exception as e:
                print(f"[PhysX] Failed to create link view for {link_key}: {e}")
                continue

            # 從 USD 讀取初始 world transform
            try:
                prim = stage.GetPrimAtPath(link_path)
                if not prim.IsValid():
                    continue
                xformable = UsdGeom.Xformable(prim)
                world = xformable.ComputeLocalToWorldTransform(0)
                t = world.ExtractTranslation()
                usd_pos = np.array([t[0], t[1], t[2]], dtype=float)
                rot = world.ExtractRotation()
                q = rot.GetQuaternion()
                im = q.GetImaginary()
                usd_quat = np.array([im[0], im[1], im[2], q.GetReal()], dtype=float)
                link_usd_transforms[link_key] = (usd_pos, usd_quat)
            except Exception as e:
                print(f"[PhysX] Failed to get USD transform for {link_key}: {e}")

        if not link_usd_transforms:
            print("[PhysX] No link transforms available, marker tracking disabled")
            return

        # 計算每個 marker 相對於對應 link 的 local offset
        for marker_path in self.end_prim_paths:
            path_upper = marker_path.upper()

            # 決定此 marker 應追蹤哪個 link
            assigned_link_key = None
            for keyword, link_key in MARKER_LINK_RULES.items():
                if keyword in path_upper:
                    assigned_link_key = link_key
                    break

            if assigned_link_key is None:
                # FM 等完全靜態的 marker — 由 USD xform fallback 處理
                continue

            link_path = self._link_paths.get(assigned_link_key)
            if link_path is None or assigned_link_key not in link_usd_transforms:
                continue

            link_usd_pos, link_usd_quat = link_usd_transforms[assigned_link_key]

            try:
                prim = stage.GetPrimAtPath(marker_path)
                if not prim.IsValid():
                    continue
                xformable = UsdGeom.Xformable(prim)
                world_transform = xformable.ComputeLocalToWorldTransform(0)
                t = world_transform.ExtractTranslation()
                marker_pos = np.array([t[0], t[1], t[2]], dtype=float)

                world_offset = marker_pos - link_usd_pos
                local_offset = self._rotate_vector_by_quat_inverse(world_offset, link_usd_quat)

                self._marker_local_offsets[marker_path] = local_offset
                self._marker_link_map[marker_path] = link_path

            except Exception:
                pass
        
        if self._marker_local_offsets:
            print(f"[PhysX] Initialized {len(self._marker_local_offsets)} marker offsets")

    def _get_link_transform(self, link_path: str):
        """獲取 link 的 world transform (位置, 四元數)"""
        rb_view = self._rigid_body_views.get(link_path)
        if rb_view is None:
            return None
        try:
            poses = rb_view.get_transforms()  # shape: (N, 7) - [x, y, z, qx, qy, qz, qw]
            if poses is None or len(poses) == 0:
                return None
            pose = poses[0]
            if hasattr(pose, 'cpu'):
                pose = pose.cpu().numpy()
            pos = np.array([pose[0], pose[1], pose[2]], dtype=float)
            # 四元數格式: [qx, qy, qz, qw]
            quat = np.array([pose[3], pose[4], pose[5], pose[6]], dtype=float)
            return pos, quat
        except Exception:
            return None

    def _rotate_vector_by_quat(self, vec: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """用四元數旋轉向量 (quat: [qx, qy, qz, qw])"""
        # 將 vec 轉為純四元數 [vx, vy, vz, 0]
        qx, qy, qz, qw = quat
        vx, vy, vz = vec
        
        # q * v * q^-1 的計算（簡化版）
        # 這裡使用 Rodrigues rotation via quaternion
        t = 2 * np.cross(quat[:3], vec)
        return vec + qw * t + np.cross(quat[:3], t)

    def _rotate_vector_by_quat_inverse(self, vec: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """用四元數的逆旋轉向量 (quat: [qx, qy, qz, qw])"""
        # 四元數的逆 = [−qx, −qy, −qz, qw] (對於 unit quaternion)
        inv_quat = np.array([-quat[0], -quat[1], -quat[2], quat[3]])
        return self._rotate_vector_by_quat(vec, inv_quat)

    def _get_marker_position_via_link(self, prim_path: str) -> Optional[np.ndarray]:
        """使用 link transform + offset 計算 marker 的世界座標
        
        這是針對非 rigid body marker 的動態追蹤方法。
        正確處理 link 的旋轉變換。
        """
        if prim_path not in self._marker_local_offsets:
            return None
        
        link_path = self._marker_link_map.get(prim_path)
        if link_path is None:
            return None
        
        # 獲取 link 的當前 transform
        link_transform = self._get_link_transform(link_path)
        if link_transform is None:
            return None
        
        link_pos, link_quat = link_transform
        
        # 將 local offset 用當前旋轉變換到 world frame
        local_offset = self._marker_local_offsets[prim_path]
        world_offset = self._rotate_vector_by_quat(local_offset, link_quat)
        
        # 計算 marker 的世界座標
        return link_pos + world_offset

    # ─────────────── 線段更新 ───────────────

    def _update_lines(self) -> bool:
        """重新計算所有起點→終點的線段，回傳是否有有效線段
        
        重要：在取得 prim 位置前，先同步 PhysX kinematic 狀態，
        確保機器人移動後 USD xform 能正確反映當前物理位置。
        """
        # 【延遲初始化】physics_sim_view 必須在 sim.reset() 後取得
        if self._physics_sim_view is None and not self._physics_sim_view_attempted:
            self._physics_sim_view_attempted = True
            print(f"[NDIDetector] Attempting to get physics_sim_view (_HAS_SIM_MANAGER={_HAS_SIM_MANAGER})")
            if _HAS_SIM_MANAGER:
                try:
                    self._physics_sim_view = SimulationManager.get_physics_sim_view()
                    print("[NDIDetector] Physics sim view acquired (kinematic sync enabled)")
                except Exception as e:
                    print(f"[NDIDetector] Could not acquire physics sim view: {e}")
        
        # 【延遲初始化】初始化 marker 的 link 追蹤
        if self._physics_sim_view is not None and not self._marker_offsets_initialized:
            self._init_marker_link_tracking()
        
        # 【關鍵修正】同步 kinematic 狀態，確保 xform 反映最新物理位置
        if self._physics_sim_view is not None:
            try:
                self._physics_sim_view.update_articulations_kinematic()
            except Exception as e:
                # 首次同步可能失敗（尚未有 articulation），忽略
                pass
        
        self.line_starts = []
        self.line_ends = []
        self.ray_info = []

        start_valid = [(p, self._get_prim_position(p)) for p in self.start_prim_paths]
        start_valid = [(p, pos) for p, pos in start_valid if pos is not None]

        end_valid = [(p, self._get_prim_position(p)) for p in self.end_prim_paths]
        end_valid = [(p, pos) for p, pos in end_valid if pos is not None]

        for sp, spos in start_valid:
            for ep, epos in end_valid:
                self.line_starts.append(spos)
                self.line_ends.append(epos)
                self.ray_info.append({
                    'start_path': sp,
                    'end_path': ep,
                    'start_pos': spos,
                    'end_pos': epos,
                })

        return len(self.line_starts) > 0

    # ─────────────── Raycast 執行 ───────────────

    def _make_raycast_callback(self, idx: int):
        """
        Factory function：為每條射線建立獨立的 callback，
        正確捕捉 idx（避免閉包共用同一個變數的 bug）。
        """
        def callback(ray, result):
            self._on_raycast_hit(ray, result, idx)
        return callback

    def _perform_raycast(self):
        """
        提交所有射線查詢。

        關鍵修正：
        - Ray(start_tuple, end_point_tuple)   ← Isaac Lab API
          原版是 Ray(start, normalized_direction)，為舊版 Isaac Sim API
        """
        if not self.line_starts:
            return

        self.raycast_results = [None] * len(self.line_starts)

        for idx, (start, end, info) in enumerate(
                zip(self.line_starts, self.line_ends, self.ray_info)):

            direction = end - start
            length = np.linalg.norm(direction)
            if length < 0.001:
                self.raycast_results[idx] = {
                    'ray_index': idx,
                    'start_path': info['start_path'],
                    'end_path': info['end_path'],
                    'hit_path': 'None',
                    'is_target_hit': False,
                    'miss_reason': 'degenerate_ray',
                }
                continue

            normalized = direction / length
            adjusted_start = start + normalized * 0.01          # 偏移 1cm 避免自碰撞

            ray = omni.kit.raycast.query.Ray(
                tuple(adjusted_start.tolist()),
                tuple(normalized.tolist()),
                # length * 2.0  # 可選的最大距離，為了避免太多誤判可以考慮加上
            )
            self.raycast_interface.submit_raycast_query(ray, self._make_raycast_callback(idx))

    def _on_raycast_hit(self, ray, result, idx: int):
        """Raycast 回調，透過 idx 直接寫入對應位置"""
        self._callback_count = getattr(self, '_callback_count', 0) + 1
        
        if idx >= len(self.raycast_results):
            return
        if idx >= len(self.ray_info):
            return

        info = self.ray_info[idx]
        start_path = info['start_path']
        end_path = info['end_path']

        hit_position = None
        if result is not None and result.valid:
            hit_path = str(result.get_target_usd_path())
            hit_position = np.array([result.hit_position[0], result.hit_position[1], result.hit_position[2]])
            
            # 首先檢查是否直接命中目標
            is_direct_hit = (end_path in hit_path) or (hit_path in end_path)
            
            # 最終判定：必須命中目標 (不使用忽略清單過濾)
            is_target_hit = is_direct_hit
            miss_reason = None if is_target_hit else 'hit_other_geometry'
            
            # DEBUG: 印出前 5 個有效結果
            if getattr(self, 'debug_enabled_flag', False) and self._callback_count <= 5:
                print(f"[DEBUG] callback #{self._callback_count}: idx={idx}, valid=True")
                print(f"        hit_path={hit_path}")
                print(f"        end_path={end_path}")
                print(f"        is_direct_hit={is_direct_hit} -> is_target_hit={is_target_hit}")
        else:
            hit_path = 'None'
            is_target_hit = False
            miss_reason = 'not_in_volume'
            # DEBUG: 印出 invalid 結果（只印前 3 個）
            if getattr(self, 'debug_enabled_flag', False) and self._callback_count <= 3:
                print(f"[DEBUG] callback #{self._callback_count}: idx={idx}, result={result}, valid={result.valid if result else 'None'}")

        self.raycast_results[idx] = {
            'ray_index': idx,
            'start_path': start_path,
            'end_path': end_path,
            'hit_path': hit_path,
            'is_target_hit': is_target_hit,
            'hit_position': hit_position,
            'miss_reason': miss_reason,
        }

    # ─────────────── 結果處理 ───────────────

    def _process_detection_results(self) -> dict:
        """
        依 raycast 結果判定每個 marker 的可見性。
        （Isaac Lab 版：移除 PhysxTrigger 條件，僅依賴 raycast）
        """
        if not self.raycast_results:
            return self.marker_status

        # 統計每個終點路徑的命中情況
        end_hit_map = {p: {'hit_count': 0, 'total_count': 0}
                       for p in self.end_prim_paths}

        for res in self.raycast_results:
            if res is None:
                continue
            ep = res['end_path']
            if ep in end_hit_map:
                end_hit_map[ep]['total_count'] += 1
                if res['is_target_hit']:
                    end_hit_map[ep]['hit_count'] += 1

        # 判定每個 marker（75% 閾值）
        VISIBILITY_THRESHOLD = 0.75
        marker_status = {m: False for m in self.config.SUPPORTED_MARKERS}

        for marker in self.config.SUPPORTED_MARKERS:
            marker_paths = [p for p in self.end_prim_paths
                            if marker.upper() in p.upper()]
            if not marker_paths:
                continue

            visible_balls = 0
            total_balls = len(marker_paths)

            for path in marker_paths:
                if path not in end_hit_map:
                    continue
                stats = end_hit_map[path]
                if stats['total_count'] == 0:
                    continue
                # Isaac Lab 版：只需 raycast 全部命中（不要求 in-trigger）
                if stats['hit_count'] == stats['total_count']:
                    visible_balls += 1

            if total_balls > 0:
                marker_status[marker] = (visible_balls / total_balls) >= VISIBILITY_THRESHOLD

        # error-only 模式：只在有 miss 時輸出精簡摘要
        if self.error_only_log:
            miss_counts = {}
            blocked_by = {}   # hit_path -> count
            total_rays = 0
            for res in self.raycast_results:
                if res is None:
                    continue
                total_rays += 1
                if res.get('is_target_hit', False):
                    continue
                reason = res.get('miss_reason') or 'unknown'
                miss_counts[reason] = miss_counts.get(reason, 0) + 1
                if reason == 'hit_other_geometry':
                    raw_path = res.get('hit_path') or 'unknown'
                    # 只取最後兩層路徑，避免過長
                    parts = [p for p in raw_path.split('/') if p]
                    short_path = '/'.join(parts[-2:]) if len(parts) >= 2 else raw_path
                    blocked_by[short_path] = blocked_by.get(short_path, 0) + 1
            if miss_counts:
                miss_total = sum(miss_counts.values())
                hit_total = max(total_rays - miss_total, 0)
                reason_text = ",".join(f"{k}:{v}" for k, v in sorted(miss_counts.items()))
                # 依次數排序，只顯示前 3 種遮蔽 geometry
                top_blockers = sorted(blocked_by.items(), key=lambda x: -x[1])[:3]
                blocker_text = " blocked_by=" + ",".join(f"{p}:{c}" for p, c in top_blockers) if top_blockers else ""
                print(
                    f"[NDI_MISS] env={self.env_index} frame={self.frame_count} "
                    f"hit={hit_total}/{total_rays} miss={miss_total}/{total_rays} reason={reason_text}"
                    f"{blocker_text}"
                )
        else:
            # 每 50 幀印一次摘要
            if self.frame_count % 50 == 0:
                print(f"[NDIDetector] Frame {self.frame_count} visibility:", end=" ")
                for m in self.config.SUPPORTED_MARKERS:
                    sym = "V" if marker_status[m] else "X"
                    print(f"{m}:{sym}", end=" ")
                print()

        # 計算體積和固態角
        self._calculate_marker_volumes_and_angles(marker_status, end_hit_map)

        return marker_status

    def _calculate_marker_volumes_and_angles(self, marker_status: dict, end_hit_map: dict):
        """計算每個 marker 可見球體的體積和固態角"""
        self.volume_calculator.get_emitter_position(self._get_prim_position)
        if self.volume_calculator.emitter_position is None:
            return

        for marker in self.config.SUPPORTED_MARKERS:
            marker_paths = [p for p in self.end_prim_paths
                            if marker.upper() in p.upper()]
            visible_positions = []
            for path in marker_paths:
                if path not in end_hit_map:
                    continue
                stats = end_hit_map[path]
                if stats['total_count'] > 0 and stats['hit_count'] == stats['total_count']:
                    pos = self._get_prim_position(path)
                    if pos is not None:
                        visible_positions.append(pos)

            if len(visible_positions) >= 3:
                self.marker_volumes[marker] = self.volume_calculator.calculate_marker_volume(visible_positions)
                self.marker_solid_angles[marker] = self.volume_calculator.calculate_solid_angle(visible_positions)
            else:
                self.marker_volumes[marker] = None
                self.marker_solid_angles[marker] = None

    # ─────────────── Debug draw ───────────────

    def _draw_debug_lines(self, clear_first: bool = True):
        """繪製 raycast 的 debug 線段（命中畫綠色，被擋畫紅色）
        
        Args:
            clear_first: 若 True（預設）則先清除全部線段再繪製；
                         設為 False 可讓多個 detector 在同一幀累積繪製而不互相蓋掉。
        """
        try:
            if not self.debug_draw_interface:
                return
            
            draw_s, draw_e = [], []
            colors, sizes = [], []
            
            # 遍歷所有的 raycast result
            for result in self.raycast_results:
                if result is None:
                    continue
                
                start_path = result['start_path']
                end_path = result['end_path']
                
                # 若只要畫 highlight 可以打開以下過濾：
                # if start_path not in self.start_highlight_paths or end_path not in self.end_highlight_paths:
                #     continue
                
                start_pos = self._get_prim_position(start_path)
                
                # 直接畫到目標點，以確保視線連線可以動態跟隨著 end prim 移動
                end_pos = self._get_prim_position(end_path)
                
                if start_pos is None or end_pos is None:
                    continue
                    
                draw_s.append(tuple(start_pos.tolist()))
                draw_e.append(tuple(end_pos.tolist()))
                
                if result['is_target_hit']:
                    colors.append([0.0, 1.0, 0.0, 0.8])  # 綠色：成功看到
                else:
                    colors.append([1.0, 0.0, 0.0, 0.8])  # 紅色：被擋住
                sizes.append(2)

            if not draw_s:
                return
            if clear_first:
                self.debug_draw_interface.clear_lines()
            self.debug_draw_interface.draw_lines(draw_s, draw_e, colors, sizes)
        except Exception as e:
            print(f"[NDIDetector] _draw_debug_lines error: {e}")

    def clear_debug_draw(self):
        if self.debug_draw_interface:
            try:
                self.debug_draw_interface.clear_lines()
            except Exception:
                pass

    def draw_live(self, clear_first: bool = True):
        """Update line positions and redraw debug lines WITHOUT raycasting or sim.step().

        Call this inside the RL while-loop every step for live viewport visualization.
        Line colors (green/red) show the result from the LAST full update_detection() call.
        Line endpoints are updated to the current physics state every call.
        This does NOT disturb the RL physics loop.

        Args:
            clear_first: 若 False，不呼叫 clear_lines()，讓多個 detector 的線段
                         在同一幀累積顯示而不互相蓋掉。第一個 detector 應傳 True。
        """
        if not self.debug_draw_interface:
            return
        try:
            # Refresh kinematic state so _get_prim_position returns current positions
            if self._physics_sim_view is not None:
                try:
                    self._physics_sim_view.update_articulations_kinematic()
                except Exception:
                    pass
            # Redraw using cached raycast hit/miss results but live positions
            self._draw_debug_lines(clear_first=clear_first)
        except Exception:
            pass

    # ─────────────── 主更新入口 ───────────────

    def update_detection(self, sim, debug_enabled: bool = False, error_only_log: bool = False) -> dict:
        """
        每幀呼叫，回傳 {marker: bool} 字典。

        Args:
            sim: Isaac Lab 的 SimulationContext 物件（用於 sim.step()）
            debug_enabled: 是否繪製 debug 線段
            error_only_log: 若為 True，只輸出 miss 的精簡錯誤資訊
        """
        self.debug_enabled_flag = debug_enabled  # Store flag for callbacks
        self.error_only_log = error_only_log

        if not self.raycast_interface:
            return self.marker_status

        try:
            has_lines = self._update_lines()
            if not has_lines:
# print("[NDIDetector] No valid raycast lines")
                return self.marker_status

            self._callback_count = 0  # reset before raycast
            self._perform_raycast()

            # 等待 callback 處理（Isaac Lab 用 sim.step()）
            for _ in range(5):
                sim.step()

            # DEBUG: 統計 callback 呼叫次數
            callback_count = getattr(self, '_callback_count', 0)
            non_none_results = sum(1 for r in self.raycast_results if r is not None)
            total_rays = len(self.raycast_results)
            if getattr(self, 'debug_enabled_flag', False) and self.frame_count == 0:
                print(f"[DEBUG] Frame {self.frame_count}: callbacks={callback_count}, results={non_none_results}/{total_rays}")

            # 確認有回調寫入結果
            valid_results = [r for r in self.raycast_results if r is not None]
            if valid_results:
                self.marker_status = self._process_detection_results()

            # 成本計算
            self.check_and_calculate_cost_if_needed(self.marker_status)

            # Debug draw
            if debug_enabled and self.debug_draw_interface:
                self._draw_debug_lines()
            elif self.debug_draw_interface:
                self.debug_draw_interface.clear_lines()

            self.frame_count += 1

        except Exception as e:
            print(f"[NDIDetector] update_detection error: {e}")
            tb.print_exc()

        return self.marker_status

    # ─────────────── 狀態查詢 ───────────────

    def get_status_changes(self) -> dict:
        changes = {}
        for m in self.config.SUPPORTED_MARKERS:
            if self.marker_status[m] != self.previous_marker_status[m]:
                changes[m] = self.marker_status[m]
                self.previous_marker_status[m] = self.marker_status[m]
        return changes

    def get_marker_volume(self, marker: str) -> Optional[float]:
        return self.marker_volumes.get(marker)

    def get_marker_solid_angle(self, marker: str) -> Optional[float]:
        return self.marker_solid_angles.get(marker)

    def get_all_marker_data(self) -> dict:
        return {
            m: {
                'visible': self.marker_status[m],
                'volume': self.marker_volumes[m],
                'solid_angle': self.marker_solid_angles[m],
            }
            for m in self.config.SUPPORTED_MARKERS
        }

    def get_occlusion_statistics(self) -> dict:
        return {
            m: {
                'total_occluded_frames': self.occlusion_history[m]['total_occluded_frames'],
                'current_consecutive': self.occlusion_history[m]['consecutive_frames'],
                'currently_visible': self.occlusion_history[m]['last_status'],
            }
            for m in self.config.SUPPORTED_MARKERS
        }

    # ─────────────── 成本計算 ───────────────

    def start_cost_calculation(self):
        print("[NDIDetector] 開始成本計算（固態角 + 遮蔽懲罰）")
        self.cost_calculation.update({
            'total_cost': 0.0,
            'frame_count': 0,
            'cost_samples': [],
            'start_time': time.time(),
            'calculation_active': True,
        })
        for m in self.config.SUPPORTED_MARKERS:
            self.occlusion_history[m] = {
                'consecutive_frames': 0,
                'total_occluded_frames': 0,
                'last_status': False,
            }

    def stop_cost_calculation(self) -> Optional[dict]:
        if not self.cost_calculation['calculation_active']:
            return None

        self.cost_calculation['calculation_active'] = False
        fc = self.cost_calculation['frame_count']
        total = self.cost_calculation['total_cost']
        elapsed = time.time() - self.cost_calculation['start_time']
        avg = total / fc if fc > 0 else 0.0
        occ_stats = self.get_occlusion_statistics()

        results = {
            'total_cost': total,
            'frame_count': fc,
            'average_cost_per_frame': avg,
            'elapsed_time': elapsed,
            'cost_samples': self.cost_calculation['cost_samples'],
            'occlusion_statistics': occ_stats,
            'cost_params': self.cost_params.copy(),
        }

        print(f"\n[NDIDetector] ===== 成本計算結果 =====")
        print(f"  總幀數: {fc}")
        print(f"  總成本: {total:.6f}")
        print(f"  平均每幀: {avg:.6f}")
        print(f"  執行時間: {elapsed:.2f}s")
        for m, s in occ_stats.items():
            ratio = s['total_occluded_frames'] / fc * 100 if fc > 0 else 0
            print(f"  {m} 遮蔽: {s['total_occluded_frames']} 幀 ({ratio:.1f}%)")
        print(f"=====================================\n")

        self._save_cost_results(results)
        return results

    def calculate_frame_cost(self, marker_status: dict) -> float:
        frame_cost = 0.0
        for m in self.config.SUPPORTED_MARKERS:
            is_visible = marker_status.get(m, False)
            if is_visible:
                self.occlusion_history[m]['consecutive_frames'] = 0
                sa = self.marker_solid_angles.get(m)
                if sa is not None and sa > 0:
                    T = self.cost_params['temperature']
                    frame_cost += -np.log(sa + 1e-10) / T
            else:
                self.occlusion_history[m]['consecutive_frames'] += 1
                self.occlusion_history[m]['total_occluded_frames'] += 1
                base = self.cost_params['occlusion_penalty_base']
                inc = self.cost_params['occlusion_penalty_increment']
                consec = self.occlusion_history[m]['consecutive_frames']
                penalty = min(base + (consec - 1) * inc, base * 5)
                frame_cost += penalty
            self.occlusion_history[m]['last_status'] = is_visible
        return frame_cost

    def check_and_calculate_cost_if_needed(self, marker_status: dict):
        if not self.cost_calculation['calculation_active']:
            return
        self.volume_calculator.get_emitter_position(self._get_prim_position)
        cost = self.calculate_frame_cost(marker_status)
        self.cost_calculation['total_cost'] += cost
        self.cost_calculation['frame_count'] += 1
        self.cost_calculation['cost_samples'].append(cost)

        if self.cost_calculation['frame_count'] % 100 == 0:
            fc = self.cost_calculation['frame_count']
            total = self.cost_calculation['total_cost']
            print(f"[NDIDetector] Cost frame {fc}: total={total:.4f}, avg={total/fc:.6f}")

    # ─────────────── 結果儲存 ───────────────

    def _save_cost_results(self, results: dict):
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 使用 Isaac Lab outputs 目錄
            workspace_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            out_dir = os.path.join(workspace_root, "outputs", "ndi_cost")
            os.makedirs(out_dir, exist_ok=True)

            # JSON
            json_path = os.path.join(out_dir, f"ndi_cost_{ts}.json")
            # cost_samples 可能很長，序列化前先轉 list
            save_data = {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                         for k, v in results.items()}
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=4)
            print(f"[NDIDetector] JSON 已儲存: {json_path}")

            # CSV + 圖表
            samples = results.get('cost_samples', [])
            if samples:
                import csv
                csv_path = os.path.join(out_dir, f"cost_vs_frame_{ts}.csv")
                with open(csv_path, 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['Frame', 'Cost'])
                    w.writerows(enumerate(samples, 1))
                print(f"[NDIDetector] CSV 已儲存: {csv_path}")
                self._generate_cost_plot(samples, ts, out_dir)

        except Exception as e:
            print(f"[NDIDetector] _save_cost_results error: {e}")
            tb.print_exc()

    def _generate_cost_plot(self, samples: list, ts: str, out_dir: str):
        if not _HAS_MATPLOTLIB:
            return
        try:
            plt.figure(figsize=(12, 6))
            frames = range(1, len(samples) + 1)
            plt.plot(frames, samples, linewidth=1.5, color='#2E86AB', alpha=0.8, label='Cost')
            avg = np.mean(samples)
            plt.axhline(y=avg, color='#F18F01', linestyle=':', linewidth=2,
                        label=f'Average = {avg:.4f}')
            plt.xlabel('Frame Number', fontsize=12, fontweight='bold')
            plt.ylabel('Cost', fontsize=12, fontweight='bold')
            plt.title('NDI Cost Function vs Frame Number', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.legend(fontsize=10)
            stats_text = (
                f'Frames: {len(samples)}\n'
                f'Total: {sum(samples):.2f}\n'
                f'Min: {min(samples):.4f}\n'
                f'Max: {max(samples):.4f}\n'
                f'Std: {np.std(samples):.4f}'
            )
            plt.text(0.98, 0.02, stats_text, transform=plt.gca().transAxes,
                     fontsize=9, va='bottom', ha='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            plot_path = os.path.join(out_dir, f"cost_vs_frame_{ts}.png")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[NDIDetector] 圖表已儲存: {plot_path}")
        except Exception as e:
            print(f"[NDIDetector] _generate_cost_plot error: {e}")
            print(f"[NDIDetector] 圖表已儲存: {plot_path}")
        except Exception as e:
            print(f"[NDIDetector] _generate_cost_plot error: {e}")
