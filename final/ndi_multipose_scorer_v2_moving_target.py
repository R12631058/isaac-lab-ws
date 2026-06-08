"""
ndi_multipose_scorer_v2_moving_target.py

Variant of ndi_multipose_scorer_v2 with a BREATHING MOTION target.
The /Root/tumor target oscillates to simulate respiratory motion,
and the robot arm dynamically tracks it via per-step command updates.

Breathing waveform: 0.7 * triangle + 0.3 * cosine, period ~12.6s
  X oscillation: base_x +/- 0.03 m
  Z oscillation: base_z +/- 0.01 m
"""
import argparse
import sys
import os
import time
import json
import math
import itertools
from pathlib import Path

# ISAAC SIM

from isaaclab.app import AppLauncher

# CLI
parser = argparse.ArgumentParser(description="NDI Multi-Pose Scorer")
parser.add_argument("--num_envs", type=int, default=10,
                    help="Number of environments (NDI positions)")
parser.add_argument("--episodes", type=int, default=1,
                    help="Episodes to run")
parser.add_argument("--x_start", type=float, default=0.6,
                    help="NDI X Start")
parser.add_argument("--x_end", type=float, default=2.8,
                    help="NDI X End")
parser.add_argument("--debug_verbose", action="store_true",
                    help="Enable detailed output")
parser.add_argument("--robot_x_list", type=float, nargs='+', default=None,
                    help="List of robot arm base X positions (env-local) to sweep.")
parser.add_argument("--showcase_only", action="store_true",
                    help="Showcase mode: continuous animation without scoring interruptions.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch simulation
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports post-simulation App
import torch
import numpy as np
import scipy.optimize
from pxr import Usd, UsdGeom, Gf

from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_extension_linkout_cone_orientation_cfg import TM5ExtensionLinkOutFanOrientationEnvCfg

# ── Initial joint config (deg → rad) ──
# True initial: -84.9°, 43.4°, 57.5°, 78.4°, 0.0°, -92.0°
INIT_JOINT_POS_RAD = [-1.48178, 0.75747, 1.00356, 1.36834, 0.0, -1.60570]

# ── Per-joint clamping to lock arm configuration ──
JOINT_CLAMPS = {
    0: (-3.14, -0.1),    # joint_1: init=-84.9° → must stay negative
    1: (-0.3,   2.5),    # joint_2: init=+43.4° → must stay positive-ish
    2: (-0.3,   2.5),    # joint_3: init=+57.5° → must stay positive-ish
    3: (-0.5,   3.14),   # joint_4: init=+78.4° → must stay positive-ish
    4: (-1.57,  1.57),   # joint_5: init= 0.0°  → near zero, allow both
    5: (-3.14,  0.5),    # joint_6: init=-92.0° → must stay negative
}

SUPPORTED_MARKERS = ["BM", "EM", "FM", "UM"]

# ── NDIDetector Integration ──
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ndi_detector_isaaclab_experimental import NDIConfig, NDIDetector

class EnvNDIConfig(NDIConfig):
    """Custom NDI Config for individual environments"""
    def __init__(self, env_prim_path, detected_map=None):
        super().__init__()
        prefix_replacement = env_prim_path
        if detected_map and "emitter_path" in detected_map:
            p = detected_map["emitter_path"]
            if "/NDI/NDI_emitter" in p:
                prefix_replacement = p.split("/NDI/NDI_emitter")[0]
        self.START_PRIM_PATHS = [p.replace("/Root", prefix_replacement) for p in self.START_PRIM_PATHS]
        self.START_HIGHLIGHT_PATHS = [p.replace("/Root", prefix_replacement) for p in self.START_HIGHLIGHT_PATHS]
        self.END_PRIM_PATHS = [p.replace("/Root", prefix_replacement) for p in self.END_PRIM_PATHS]
        self.TRIGGER_VOLUME_PATH = self.TRIGGER_VOLUME_PATH.replace("/Root", prefix_replacement)
        if detected_map and "link_6_path" in detected_map:
            self.LINK_6_PATH = detected_map["link_6_path"]
        else:
            self.LINK_6_PATH = f"{prefix_replacement}/robotarm_base/robotarm_base/tm5_700/link_6"
            
        if detected_map and "needle_tip_path" in detected_map:
            self.NEEDLE_TIP_PATH = detected_map["needle_tip_path"]
        else:
            self.NEEDLE_TIP_PATH = f"{prefix_replacement}/robotarm_base/robotarm_base/needle/needle_tip"

def find_env_structure(stage, env_path):
    result = {}
    env_prim = stage.GetPrimAtPath(env_path)
    if not env_prim.IsValid(): return result
    def search(prim, depth=0):
        if depth > 10: return
        name = prim.GetName()
        path = str(prim.GetPath())
        if name == "link_6" and "link_6_path" not in result:
            result["link_6_path"] = path
        if name == "needle_tip" and "needle_tip_path" not in result:
            result["needle_tip_path"] = path
        if name == "NDI_emitter" and "emitter_path" not in result:
            result["emitter_path"] = path
        for child in prim.GetChildren():
            search(child, depth + 1)
            if "link_6_path" in result and "emitter_path" in result and "needle_tip_path" in result: return
    search(env_prim)
    return result

# ── Math Utilities ──
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-12 else np.zeros_like(v)

def cross(v1, v2):
    return np.cross(v1, v2)

def matrix_to_quat(m):
    # m is 3x3 rotation matrix
    trace = np.trace(m)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2,1] - m[1,2]) / s
        qy = (m[0,2] - m[2,0]) / s
        qz = (m[1,0] - m[0,1]) / s
    elif (m[0,0] > m[1,1]) and (m[0,0] > m[2,2]):
        s = np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2.0
        qw = (m[2,1] - m[1,2]) / s
        qx = 0.25 * s
        qy = (m[0,1] + m[1,0]) / s
        qz = (m[0,2] + m[2,0]) / s
    elif m[1,1] > m[2,2]:
        s = np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2.0
        qw = (m[0,2] - m[2,0]) / s
        qx = (m[0,1] + m[1,0]) / s
        qy = 0.25 * s
        qz = (m[1,2] + m[2,1]) / s
    else:
        s = np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2.0
        qw = (m[1,0] - m[0,1]) / s
        qx = (m[0,2] + m[2,0]) / s
        qy = (m[1,2] + m[2,1]) / s
        qz = 0.25 * s
    return np.array([qw, qx, qy, qz])

def euler_angles_to_quat(roll, pitch, yaw):
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.array([qw, qx, qy, qz])

def look_at_quaternion_world(ndi_pos, target_pos):
    """
    Returns (qw, qx, qy, qz) representing a rotation where the camera points
    its -Z axis toward target_pos, with a fixed 90-degree roll (frustum wide side horizontal).
    """
    forward = normalize(target_pos - ndi_pos) # This is where we want -Z to point
    # We want +Z to point opposite to forward
    z_axis = -forward 
    
    # World up hint
    up_hint = np.array([0, 0, 1.0])
    
    if abs(np.dot(z_axis, up_hint)) > 0.999: # Edge case: target directly above/below
        up_hint = np.array([1.0, 0, 0])
        
    x_axis = normalize(cross(up_hint, z_axis))
    y_axis = cross(z_axis, x_axis)
    
    # Normally [x_axis | y_axis | z_axis] is standard look-at
    # We want a 90 deg roll so we swap axes.
    # Base Rotation R0: +X=right, +Y=up, -Z=forward
    # Want to apply Roll=90: +X -> -Y, +Y -> +X
    rolled_x = -y_axis
    rolled_y = x_axis
    rolled_z = z_axis
    
    R = np.column_stack((rolled_x, rolled_y, rolled_z))
    return matrix_to_quat(R)

def get_prim_world_transform(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None, None
    xform = UsdGeom.Xformable(prim)
    time_code = Usd.TimeCode.Default()
    transform = xform.ComputeLocalToWorldTransform(time_code)
    translation = transform.ExtractTranslation()
    rotation = transform.ExtractRotationQuat()
    # Gf.Quatd -> real, imag...
    quat = np.array([rotation.GetReal(), rotation.GetImaginary()[0], rotation.GetImaginary()[1], rotation.GetImaginary()[2]])
    return np.array(translation), quat

def set_prim_transform_xformop(stage, prim_path, translation, rotation_xyz_deg=None):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    xform = UsdGeom.Xformable(prim)
    
    # Translation
    translate_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if not translate_op:
        translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(*translation))

    # Rotation
    if rotation_xyz_deg is not None:
        rotate_op = None
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                rotate_op = op
                break
        if not rotate_op:
            rotate_op = xform.AddRotateXYZOp()
        rotate_op.Set(Gf.Vec3d(*rotation_xyz_deg))


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


def get_usd_target_position(env: ManagerBasedRLEnv):
    """Get /Root/tumor position from Env 0 (World Frame)"""
    stage = env.unwrapped.scene.stage
    prim_path = "/World/envs/env_0/Root/tumor"

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        prim_path = "/World/envs/env_0/Root/target"
        prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        print(f"[WARN] Could not find target in USD at {prim_path}")
        return None

    xformable = UsdGeom.Xformable(prim)
    world_transform = xformable.ComputeLocalToWorldTransform(0)
    translation = world_transform.ExtractTranslation()

    target_world_env0 = torch.tensor([translation[0], translation[1], translation[2]], device=env.device)

    if env.scene.env_origins is not None:
        env0_origin = env.scene.env_origins[0]
        return target_world_env0 - env0_origin
    else:
        return target_world_env0

# ── Polaris Vega Frustum Math ──
F_NEAR = 1.4
F_FAR  = 2.4
H_NEAR_H = 1.0  # Horizontal
H_NEAR_V = 1.0  # Vertical
H_FAR_H  = 1.6
H_FAR_V  = 1.2

SLOPE_H = ((H_FAR_H - H_NEAR_H) / 2.0) / (F_FAR - F_NEAR)
SLOPE_V = ((H_FAR_V - H_NEAR_V) / 2.0) / (F_FAR - F_NEAR)

def frustum_check(marker_world, ndi_world_pos, ndi_quat):
    import scipy.spatial.transform as sst
    R = sst.Rotation.from_quat([ndi_quat[1], ndi_quat[2], ndi_quat[3], ndi_quat[0]]).as_matrix()
    
    vec = marker_world - ndi_world_pos
    local_pos = R.T @ vec
    
    x, y, z = local_pos[0], local_pos[1], local_pos[2]
    # NDI negative Z is depth
    depth = -z 
    
    if depth < F_NEAR or depth > F_FAR:
        return {"in_frustum": False, "depth": depth, "local_x": x, "local_y": y, "max_x": 0, "max_y": 0, "reason": "OUT_OF_DEPTH"}
        
    delta_z = depth - F_NEAR
    max_x = (H_NEAR_H / 2.0) + delta_z * SLOPE_H
    max_y = (H_NEAR_V / 2.0) + delta_z * SLOPE_V
    
    in_frustum = (abs(x) <= max_x) and (abs(y) <= max_y)
    return {
        "in_frustum": in_frustum,
        "depth": depth,
        "local_x": x,
        "local_y": y,
        "max_x": max_x,
        "max_y": max_y,
        "reason": "OK" if in_frustum else "OUT_OF_FOV"
    }

# ── Solid Angle & Cost Functions from v7 ──
def calculate_triangle_solid_angle(a, b, c, observer_pos):
    try:
        v = np.array(observer_pos)
        A = np.array(a) - v
        B = np.array(b) - v
        C = np.array(c) - v
        
        a_len = np.linalg.norm(A)
        b_len = np.linalg.norm(B)
        c_len = np.linalg.norm(C)
        
        if a_len == 0 or b_len == 0 or c_len == 0:
            return 0.0
            
        A_unit = A / a_len
        B_unit = B / b_len
        C_unit = C / c_len
        
        scalar_triple = np.dot(A_unit, np.cross(B_unit, C_unit))
        denominator = 1 + np.dot(A_unit, B_unit) + np.dot(B_unit, C_unit) + np.dot(C_unit, A_unit)
        
        if denominator <= 0:
            return 0.0
            
        return 2 * np.arctan2(abs(scalar_triple), denominator)
    except Exception:
        return 0.0

def calculate_group_solid_angle(visible_positions, observer_pos):
    if len(visible_positions) < 3:
        return 0.0
    
    total_solid_angle = 0.0
    combinations = list(itertools.combinations(visible_positions, 3))
    for triangle in combinations:
        sa = calculate_triangle_solid_angle(triangle[0], triangle[1], triangle[2], observer_pos)
        total_solid_angle += sa
        
    return total_solid_angle / len(combinations) if combinations else 0.0

class MultiPoseScorer:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        # Tracking states for occlusion across sampled frames
        # Env_idx -> Marker_group -> consecutive_frames
        self.occlusion_history = {
            i: {m: 0 for m in SUPPORTED_MARKERS} for i in range(num_envs)
        }
        
        self.base_penalty = 50.0
        self.penalty_increment = 10.0
        self.temperature = 1.0
        
        # Results storage
        self.frame_results = []
        
    def score_frame(self, frame_idx, env_visibility, n_positions_w, env_marker_groups_positions):
        """
        env_visibility: dict -> env_idx -> group -> True/False
        n_positions_w: dict -> env_idx -> observer world position
        env_marker_groups_positions: dict -> env_idx -> group -> list of visible world pos
        """
        frame_cost_data = []
        
        for env_idx in range(self.num_envs):
            env_cost = 0.0
            env_group_costs = {}
            
            for group in SUPPORTED_MARKERS:
                is_visible = env_visibility[env_idx][group]
                group_cost = 0.0
                
                if is_visible:
                    # Reset occlusion tracking
                    self.occlusion_history[env_idx][group] = 0
                    
                    # Compute solid angle and cost
                    vis_pos_list = env_marker_groups_positions[env_idx][group]
                    sa = calculate_group_solid_angle(vis_pos_list, n_positions_w[env_idx])
                    
                    if sa > 0:
                        group_cost = -np.log(sa + 1e-10) / self.temperature
                    else:
                        group_cost = self.base_penalty
                else:
                    # Occluded penalty
                    self.occlusion_history[env_idx][group] += 1
                    consecutive = self.occlusion_history[env_idx][group]
                    
                    penalty = self.base_penalty + (consecutive - 1) * self.penalty_increment
                    penalty = min(penalty, self.base_penalty * 5.0)  # Max 250
                    group_cost = penalty
                    
                env_group_costs[group] = group_cost
                env_cost += group_cost
                
            frame_cost_data.append({
                "env_idx": env_idx,
                "total_cost": env_cost,
                "group_costs": env_group_costs
            })
            
        self.frame_results.append({
            "frame": frame_idx,
            "envs": frame_cost_data
        })
        return frame_cost_data


class BreathingTarget:
    """Simulates respiratory motion on /Root/tumor for each environment.
    
    A smooth quadratic Bezier curve is used to simulate typical human breathing,
    passing from P0 to P2 through P1.
    """
    def __init__(self, num_envs, stage, dt=0.0167):
        self.num_envs = num_envs
        self.stage = stage
        self.dt = dt
        self.current_t = 0.0
        
        # ── Smooth Curved Breathing Parameters ──
        self.P0 = np.array([-0.4, -0.8, 1.07])
        self.P1 = np.array([-0.33, -0.8, 1.05])
        self.P2 = np.array([-0.3, -0.8, 1.05])
        
        # Control point for quadratic bezier to pass through P1 at s=0.5
        self.PC = 2.0 * self.P1 - 0.5 * self.P0 - 0.5 * self.P2
        self.PERIOD = 12.6   # seconds
        
        # Prevent PhysX from overwriting our manual transform updates
        from pxr import UsdPhysics
        for i in range(self.num_envs):
            tumor_path = f"/World/envs/env_{i}/Root/tumor"
            prim = self.stage.GetPrimAtPath(tumor_path)
            if prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
    
    def reset(self):
        self.current_t = 0.0
    
    def step(self):
        """Advance time and calculate new local position.
        Also updates the /Root/tumor prim in every environment.
        Returns the current local position [x, y, z] as a numpy array.
        """
        self.current_t += self.dt
        t = self.current_t
        
        # Smoothly oscillate between 0.0 and 1.0
        phase = 2 * math.pi * (t / self.PERIOD)
        s = (1.0 - math.cos(phase)) / 2.0
        
        # Quadratic Bezier formula
        local_pos = ((1.0 - s)**2) * self.P0 + (2.0 * (1.0 - s) * s) * self.PC + (s**2) * self.P2
        
        # Update the Tumor prim in every environment
        for i in range(self.num_envs):
            tumor_path = f"/World/envs/env_{i}/Root/tumor"
            set_prim_transform_xformop(self.stage, tumor_path, local_pos)
        
        return local_pos
    
    def get_current_local_pos(self):
        """Return the current breathing position without advancing time."""
        t = self.current_t
        cos_val = math.cos(t * self.OMEGA)
        triangle_val = (2.0 / math.pi) * math.asin(max(-1.0, min(1.0, cos_val)))
        wave_val = 0.7 * triangle_val + 0.3 * cos_val
        return np.array([self.BASE_X + self.AMP_X * wave_val,
                         self.BASE_Y,
                         self.BASE_Z + self.AMP_Z * wave_val])

# ── Main Script ──
def main():
    num_envs = args_cli.num_envs
    ndi_x_positions = np.linspace(args_cli.x_start, args_cli.x_end, num_envs)
    
    print(f"\n[INIT] 準備啟動 {num_envs} 個環境, NDI X 位置從 {args_cli.x_start} 到 {args_cli.x_end}")
    
    env_cfg = TM5ExtensionLinkOutFanOrientationEnvCfg()
    env_cfg.scene.num_envs = num_envs
    
    # Disable command resampling - keep target fixed
    if hasattr(env_cfg.commands, "ee_pose"):
        env_cfg.commands.ee_pose.resampling_time_range = (1e10, 1e10)
        env_cfg.commands.ee_pose.debug_vis = False  # Remove coordinate axes from viewport
        
    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.device
    
    # ── UI Setup ──
    import omni.ui as ui
    class UIDebugManager:
        def __init__(self):
            self.show_raycast = True
            self.is_started = False
            self.window = ui.Window("NDI Simulation Control", width=300, height=140)
            with self.window.frame:
                with ui.VStack(height=0, spacing=8):
                    ui.Label("Click Start when ready to begin.", word_wrap=True)
                    self.start_btn = ui.Button("Start Simulation", height=30)
                    self.start_btn.set_clicked_fn(self._on_start)
                    ui.Spacer(height=4)
                    with ui.HStack(height=0):
                        ui.Label("Draw Raycast:", width=100)
                        self.checkbox_model = ui.SimpleBoolModel(self.show_raycast)
                        ui.CheckBox(self.checkbox_model)
        def _on_start(self):
            self.is_started = True
            self.start_btn.text = "Running..."
            self.start_btn.enabled = False
        @property
        def is_enabled(self):
            return self.checkbox_model.as_bool
    
    ui_manager = UIDebugManager()
    # ───────────────
    
    # 取得 Target
    def _local_get_usd_target_position(env_obj: ManagerBasedRLEnv):
        st = env_obj.unwrapped.scene.stage
        pr = st.GetPrimAtPath("/World/envs/env_0/Root/tumor")
        if not pr.IsValid(): pr = st.GetPrimAtPath("/World/envs/env_0/SurgeryRoom/target_pose")
        if not pr.IsValid(): pr = st.GetPrimAtPath("/World/envs/env_0/Root/target")
        if not pr.IsValid(): return None
        from pxr import UsdGeom
        tr = UsdGeom.Xformable(pr).ComputeLocalToWorldTransform(0).ExtractTranslation()
        t_w = torch.tensor([tr[0], tr[1], tr[2]], device=env_obj.device)
        return t_w - env_obj.scene.env_origins[0] if env_obj.scene.env_origins is not None else t_w

    fixed_target_local = _local_get_usd_target_position(env)
    if fixed_target_local is not None:
        print(f"[INFO] Using Fixed USD Target (Local): {fixed_target_local.cpu().numpy()}")
    
    # Check marker paths
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    
    # Initialize NDIDetectors
    print("\n[NDI] Initializing full NDIDetectors for accurate physical tracking...")
    detectors = []
    for i in range(num_envs):
        env_prim_path = f"/World/envs/env_{i}"
        try:
            detected_map = find_env_structure(stage, env_prim_path)
            ndi_config = EnvNDIConfig(env_prim_path, detected_map)
            detector = NDIDetector(ndi_config)
            detector.env_index = i
            detector.error_only_log = True
            detector._link_paths["link_6"] = ndi_config.LINK_6_PATH
            detector._link_paths["needle_tip"] = ndi_config.NEEDLE_TIP_PATH
            detector._link_paths["link_base"] = ndi_config.LINK_6_PATH.replace("link_6", "link_0")
            detector.initialize()
            detectors.append(detector)
        except Exception as e:
            print(f"[ERROR] Failed to init NDI for env {i}: {e}")
            detectors.append(None)
    print(f"[NDI] Initialized {len([d for d in detectors if d])} detectors.\n")
    
    # ── Differential IK Controller (replaces RL policy) ──
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.01},
    )
    diff_ik_controller = DifferentialIKController(
        diff_ik_cfg, num_envs=num_envs, device=device
    )

    # Resolve robot entity for IK
    robot_entity_cfg = SceneEntityCfg(
        "robot", joint_names=["joint_[1-6]"], body_names=["needle_tip"]
    )
    robot_entity_cfg.resolve(env.scene)
    ik_joint_ids = robot_entity_cfg.joint_ids
    ik_body_idx = robot_entity_cfg.body_ids[0]
    robot = env.scene["robot"]
    ik_ee_jacobi_idx = ik_body_idx - 1 if robot.is_fixed_base else ik_body_idx

    # Jacobian column indices (offset by 6 for non-fixed-base floating DOFs)
    if robot.is_fixed_base:
        jacobian_col_ids = ik_joint_ids
    else:
        jacobian_col_ids = [j + 6 for j in ik_joint_ids]

    print(f"[IK] Controller: DifferentialIK (DLS, position-only)")
    print(f"[IK] Tracked body: needle_tip (idx={ik_body_idx}), Jacobian idx: {ik_ee_jacobi_idx}")
    print(f"[IK] Joint IDs: {ik_joint_ids}, Jacobian col IDs: {jacobian_col_ids}")
    print(f"[IK] Joint clamps: {JOINT_CLAMPS}")
    
    # ── Simulation and Mult-Pose Scoring ──
    scorer = MultiPoseScorer(num_envs)
    target_frames = [10, 30, 50, 70, 90]
    
    global_summary_rows = []
    global_json_batches = []
    
    robot_x_list = args_cli.robot_x_list if getattr(args_cli, 'robot_x_list', None) else [None]
    n_robot = len(robot_x_list)

    for batch_idx, robot_x_val in enumerate(robot_x_list):
        rx_label = f"{robot_x_val:.2f}" if robot_x_val is not None else "default"
        print(f"\n{'#'*60}")
        print(f"# BATCH {batch_idx+1}/{n_robot}  |  Robot X = {rx_label}")
        print(f"{'#'*60}")
        
        # Reset Scorer state explicitly for each batch
        scorer.occlusion_history = {i: {m: 0 for m in SUPPORTED_MARKERS} for i in range(num_envs)}
        scorer.frame_results = []
        
        for ep_idx in range(args_cli.episodes):
            obs, _ = env.reset()
            robot = env.scene["robot"]
            
            # Apply Robot Base Position
            if robot_x_val is not None:
                root_pos_w = robot.data.root_pos_w.clone()
                env_origins_clone = env.scene.env_origins.clone()
                root_pos_w[:, 0] = env_origins_clone[:, 0] + robot_x_val
                robot.write_root_pose_to_sim(torch.cat([root_pos_w, robot.data.root_quat_w], dim=-1))
                env.sim.step()
            
            # ── Force initial joint posture for all envs ──
            init_joint_pos = torch.tensor(
                [INIT_JOINT_POS_RAD], device=device
            ).repeat(num_envs, 1)
            init_joint_vel = torch.zeros_like(init_joint_pos)
            robot.write_joint_state_to_sim(init_joint_pos, init_joint_vel)
            robot.set_joint_position_target(init_joint_pos, joint_ids=ik_joint_ids)
            
            # Let PD controller settle into surgical posture (200 steps)
            sim_dt = env.sim.get_physics_dt()
            for _settle in range(200):
                robot.set_joint_position_target(init_joint_pos, joint_ids=ik_joint_ids)
                env.scene.write_data_to_sim()
                env.sim.step()
                env.scene.update(sim_dt)
            print(f"[OK] All {num_envs} envs initialized to surgical posture")
            
            # Reset IK controller
            diff_ik_controller.reset()
            
            # Capture initial EE orientation (surgical posture downwards) to maintain it
            initial_ee_quat_w = robot.data.body_quat_w[:, ik_body_idx].clone()
            
            # --- BREATHING TARGET SETUP ---
            breathing = BreathingTarget(num_envs, stage, dt=sim_dt)
            breathing.reset()
            print(f"[BREATHING] Initialized: path {breathing.P0} -> {breathing.P1} -> {breathing.P2}, "
                  f"period={breathing.PERIOD:.1f}s, dt={sim_dt:.4f}")
            has_reached_target = False
            # -----------------------------------------------
            
            # Per-frame visibility log: env_idx -> frame -> {group: bool}
            env_frame_vis_log = {i: {} for i in range(num_envs)}
            # Last sampled frame NDI quaternion per env
            last_sampled_ndi_quat = {i: None for i in range(num_envs)}
            
            # Set NDI positions (approximate initial positions)
            for i in range(num_envs):
                ndi_path = f"/World/envs/env_{i}/Root/NDI"
                local_pos = np.array([ndi_x_positions[i], -1.998, 2.027])
                set_prim_transform_xformop(stage, ndi_path, local_pos)
                
            step = 0
            max_step = max(target_frames) if not args_cli.showcase_only else 999999
            
            while step <= max_step:
                # Wait for user to press Start
                if not ui_manager.is_started:
                    robot.set_joint_position_target(init_joint_pos, joint_ids=ik_joint_ids)
                    env.scene.write_data_to_sim()
                    env.sim.step()
                    env.scene.update(sim_dt)
                    continue
                
                # If not yet reached, keep target stationary at t=0
                if not has_reached_target:
                    ee_pos_env0 = robot.data.body_pos_w[0, ik_body_idx].cpu().numpy()
                    static_target = breathing.P0
                    # Convert P0 from env-local to world frame for correct distance check
                    env0_origin = env.scene.env_origins[0].cpu().numpy()
                    static_target_world = env0_origin + static_target
                    dist = np.linalg.norm(ee_pos_env0 - static_target_world)
                    if dist < 0.01:
                        has_reached_target = True
                        print("\n[Target Reached] Needle is in position. Starting tumor breathing animation...\n")
                    # Update tumor to static P0 position
                    for i in range(num_envs):
                        set_prim_transform_xformop(stage, f"/World/envs/env_{i}/Root/tumor", static_target)
                    breathing_local_pos = static_target
                else:
                    # ── 1. Advance breathing target ──
                    breathing_local_pos = breathing.step()  # updates all Tumor prims
                
                # ── 2. IK-based tracking (replaces RL policy) ──
                root_pos_w = robot.data.root_pos_w
                root_quat_w = robot.data.root_quat_w
                env_origins = env.scene.env_origins
                
                # Target in world frame
                target_local_t = torch.tensor(breathing_local_pos, dtype=torch.float32, device=device)
                target_local_expanded = target_local_t.unsqueeze(0).repeat(num_envs, 1)
                target_pos_w = env_origins + target_local_expanded
                target_quat_w = initial_ee_quat_w
                
                # Target in robot base frame
                target_pos_b, target_quat_b = subtract_frame_transforms(
                    root_pos_w, root_quat_w,
                    target_pos_w, target_quat_w
                )
                
                # Current EE state in base frame
                ee_pos_w = robot.data.body_pos_w[:, ik_body_idx]
                ee_quat_w = robot.data.body_quat_w[:, ik_body_idx]
                ee_pos_b, ee_quat_b = subtract_frame_transforms(
                    root_pos_w, root_quat_w,
                    ee_pos_w, ee_quat_w
                )
                
                # Jacobian and joint positions
                jacobian = robot.root_physx_view.get_jacobians()[:, ik_ee_jacobi_idx, :, jacobian_col_ids]
                joint_pos = robot.data.joint_pos[:, ik_joint_ids]
                
                # IK compute (Pose mode: [pos, quat])
                ik_commands = torch.cat([target_pos_b, target_quat_b], dim=-1)
                diff_ik_controller.set_command(ik_commands, ee_quat=ee_quat_b)
                joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
                
                # Joint clamping to prevent self-collision
                for j_idx, (j_min, j_max) in JOINT_CLAMPS.items():
                    joint_pos_des[:, j_idx] = torch.clamp(joint_pos_des[:, j_idx], min=j_min, max=j_max)
                
                # ── 3. Apply and step ──
                robot.set_joint_position_target(joint_pos_des, joint_ids=ik_joint_ids)
                env.scene.write_data_to_sim()
                env.sim.step()
                env.scene.update(sim_dt)
                
                # ── 4. NDI Raycasting & Visualization (Runs continuously) ──
                # Run every 5 steps (e.g. 12Hz) to keep performance smooth but visually continuous
                if step % 5 == 0:
                    for i in range(num_envs):
                        detector = detectors[i]
                        if detector is None: continue
                        
                        detector._update_lines()
                        valid_positions = [pos for pos in detector.line_ends if pos is not None]
                        centroid = np.mean(valid_positions, axis=0) if valid_positions else None
                        ndi_path = f"/World/envs/env_{i}/Root/NDI"
                        ndi_pos, _ = get_prim_world_transform(stage, ndi_path)
                        
                        if centroid is not None and ndi_pos is not None:
                            quat_world = look_at_quaternion_world(ndi_pos, centroid)
                            
                            # Factor out parent rotation to convert to local orientation
                            parent_path = ndi_path.rsplit("/", 1)[0]
                            parent_prim = stage.GetPrimAtPath(parent_path)
                            parent_matrix = Gf.Matrix4d(1.0)
                            if parent_prim.IsValid():
                                parent_xform = UsdGeom.Xformable(parent_prim)
                                parent_matrix = parent_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                            
                            q_parent = parent_matrix.ExtractRotation().GetQuaternion()
                            gf_quat_world = Gf.Quaternion(float(quat_world[0]), Gf.Vec3d(float(quat_world[1]), float(quat_world[2]), float(quat_world[3])))
                            q_local = q_parent.GetInverse() * gf_quat_world
                            
                            local_pos = np.array([ndi_x_positions[i], -1.998, 2.027])
                            set_prim_transform_orient(stage, ndi_path, local_pos, q_local)
                            
                    show_raycast = ui_manager.is_enabled
                    
                    if not show_raycast:
                        if detectors[0] is not None:
                            detectors[0].clear_debug_draw()
                            
                    for i in range(num_envs):
                        if detectors[i] is None: continue
                        detectors[i].update_detection(env.sim, debug_enabled=show_raycast, error_only_log=True)
                        if show_raycast:
                            detectors[i].draw_live(clear_first=(i == 0))
                
                # ── 5. Scoring & Logging (Only on target frames in normal mode) ──
                if step in target_frames and not args_cli.showcase_only:
                    print(f"\n=========================================")
                    print(f"🎯 抽出 Frame {step} 進行評分分析")
                    print(f"=========================================")
                    
                    env_visibility = {}
                    n_positions_w = {}
                    env_marker_groups_positions = {}
                    
                    for i in range(num_envs):
                        detector = detectors[i]
                        if detector is None: continue
                        
                        group_visible_balls = {m: [] for m in SUPPORTED_MARKERS}
                        group_total_balls = {m: 0 for m in SUPPORTED_MARKERS}
                        
                        for r_info in detector.raycast_results:
                            group = r_info['end_path'].split("/")[-1].split("_")[0].upper()
                            if group not in SUPPORTED_MARKERS: continue
                            
                            group_total_balls[group] += 1
                            if r_info.get('is_target_hit', False):
                                idx = r_info['ray_index']
                                group_visible_balls[group].append(detector.line_ends[idx])
                                
                        env_visibility[i] = {}
                        env_marker_groups_positions[i] = group_visible_balls
                        ndi_path = f"/World/envs/env_{i}/Root/NDI"
                        n_pos, n_quat = get_prim_world_transform(stage, ndi_path)
                        n_positions_w[i] = n_pos
                        
                        for group in SUPPORTED_MARKERS:
                            vis_count = len(group_visible_balls[group])
                            total = group_total_balls[group]
                            env_visibility[i][group] = True if total > 0 and (vis_count / total >= 0.75) else False
                        
                        env_frame_vis_log[i][step] = dict(env_visibility[i])
                        last_sampled_ndi_quat[i] = n_quat
                            
                    frame_costs = scorer.score_frame(step, env_visibility, n_positions_w, env_marker_groups_positions)
                    
                    for env_idx, fe_data in enumerate(frame_costs):
                        vis_str = []
                        for g in SUPPORTED_MARKERS:
                            status = "✓" if env_visibility[env_idx][g] else "x"
                            cost = fe_data["group_costs"][g]
                            vis_str.append(f"{g}:{status}({cost:.2f})")
                        print(f"  Env {env_idx:2d} (x={ndi_x_positions[env_idx]:.2f}) | Total Cost: {fe_data['total_cost']:6.2f} | {'  '.join(vis_str)}")
                        
                step += 1
                
        # End of episode, summarize the matrix for THIS batch
        print("\n" + "="*50)
        print(" NDI Cost Matrix [BATCH]")
        print("="*50)
        
        print(f"Env (X Pos) | " + " | ".join([f"Frame {tf:2d}" for tf in target_frames]) + " | Mean Cost")
        print("-" * 75)
        
        env_mean_costs = []
        for i in range(num_envs):
            x_pos = ndi_x_positions[i]
            env_costs = []
            for res in scorer.frame_results:
                e_data = [e for e in res["envs"] if e["env_idx"] == i][0]
                env_costs.append(e_data["total_cost"])
            
            mean_cost = np.mean(env_costs)
            env_mean_costs.append(mean_cost)
            
            cost_str = " | ".join([f"{c:8.2f}" for c in env_costs])
            print(f"Env {i} ({x_pos:.1f}) | {cost_str} | {mean_cost:8.2f}")
            
        best_env = np.argmin(env_mean_costs)
        print(f"\n🏆 最佳 NDI 位置: Env {best_env} (X={ndi_x_positions[best_env]:.2f}) 平均成本最低 {env_mean_costs[best_env]:.2f}")
        
        import scipy.spatial.transform as _sst2
        robot_ref2 = env.scene["robot"]
        env_origins_np = env.scene.env_origins.cpu().numpy()  # (num_envs, 3)
        
        batch_env_meta = {}
        target_export = fixed_target_local.cpu().numpy().tolist() if fixed_target_local is not None else None

        for i in range(num_envs):
            env_orig = env_origins_np[i]  # world offset for env i
            # Robot base: world -> env-local
            rbp_w = robot_ref2.data.root_pos_w[i].cpu().numpy()
            rbp_local = rbp_w - env_orig
            # NDI: world -> env-local
            ndi_path = f"/World/envs/env_{i}/Root/NDI"
            np_w, _ = get_prim_world_transform(stage, ndi_path)
            np_local = (np_w - env_orig) if np_w is not None else None
            q2 = last_sampled_ndi_quat[i]
            if q2 is not None:
                eu = _sst2.Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]]).as_euler('xyz', degrees=True)
                rot_s = f"({eu[0]:.1f}, {eu[1]:.1f}, {eu[2]:.1f}) deg"
            else:
                rot_s = "N/A"
            np_s = f"{np_local[0]:.2f},{np_local[1]:.2f},{np_local[2]:.2f}" if np_local is not None else "N/A"
            rb_s = f"{rbp_local[0]:.2f},{rbp_local[1]:.2f},{rbp_local[2]:.2f}"
            occ_cols = []
            for g in SUPPORTED_MARKERS:
                n_occ = sum(
                    1 for f in target_frames
                    if i in env_frame_vis_log and f in env_frame_vis_log[i]
                    and not env_frame_vis_log[i][f].get(g, True)
                )
                if n_occ == len(target_frames):
                    occ_cols.append("ALWAYS")
                elif n_occ == 0:
                    occ_cols.append("OK")
                else:
                    occ_cols.append(f"P{n_occ}/{len(target_frames)}")
                    
            # 5. Check Reachability directly from PyTorch physics state instead of USD
            robot = env.scene["robot"]
            
            # Find index of needle_tip body
            try:
                needle_idx = robot.find_bodies(".*needle_tip")[0][0]
                ee_pos_w = robot.data.body_pos_w[i, needle_idx].cpu().numpy()
            except Exception as e:
                # Fallback or generic index if 'needle_tip' not found
                try:
                    ee_pos_w = robot.data.body_pos_w[i, -1].cpu().numpy() # fallback to last body
                except:
                    ee_pos_w = None

            # For moving target: use the LAST breathing position as the reference
            target_pos_local = breathing.get_current_local_pos()

            if ee_pos_w is not None and target_pos_local is not None:
                ee_pos_local = ee_pos_w - env_orig
                dist = np.linalg.norm(ee_pos_local - target_pos_local)
                is_reachable = dist <= 0.01
                print(f"DEBUG env_{i} (X={robot_x_val}): ee={ee_pos_local}, tgt={target_pos_local}, dist={dist:.4f}")
                reach_label = "Yes" if is_reachable else "No"
            else:
                print(f"DEBUG Fail env_{i}: ee_pos_w={ee_pos_w}, target_pos_local={target_pos_local}")
                reach_label = "N/A"

            # Append to global_summary_rows
            # Append None for rank since we will assign it later, along with batch id
            global_summary_rows.append([None, i, ndi_x_positions[i], env_mean_costs[i],
                                  rb_s, np_s, rot_s, occ_cols, rx_label, reach_label])
            
            # Record for JSON
            batch_env_meta[f"env_{i}"] = {
                "robot_base_position": rbp_w.tolist(),
                "ndi_position": np_w.tolist() if np_w is not None else [],
                "ndi_orientation_quat_wxyz": q2.tolist() if q2 is not None else [],
                "matrix_cost": env_mean_costs[i],
                "occlusion_summary": occ_cols
            }
        
        batch_export_data = {
            "robot_x_val": robot_x_val,
            "environments": batch_env_meta,
            "results": scorer.frame_results,
            "frame_visibility": {
                str(f): { str(env_i): env_frame_vis_log[env_i].get(f, {}) for env_i in range(num_envs) }
                for f in target_frames
            }
        }
        global_json_batches.append(batch_export_data)

    # ==============================================================
    # END ALL BATCHES >> GENERATE GLOBAL SUMMARY
    # ==============================================================
    # Sort ALL combinations by mean cost
    global_summary_rows.sort(key=lambda r: float(r[3]))
    
    # Re-assign absolute rank across all batches
    for rk_idx, r in enumerate(global_summary_rows):
        r[0] = rk_idx + 1
        
    print("\n" + "="*80)
    print("📋 跨批次全域詳細摘要 (GLOBAL NDI Placement Summary)")
    target_str_g = (f"[{target_export[0]:.3f}, {target_export[1]:.3f}, {target_export[2]:.3f}]" if target_export else "N/A")
    print(f"   Target Pose Position (Local): {target_str_g}")
    print("="*80)
    
    # ── ASCII compact table ──
    def _row_line(rk, ei, xp, mc, rb_s, np_s, rot_s, occ_cols, rx_label, reach_label):
        occ_part = " | ".join([f"{c:<5}" for c in occ_cols])
        return f" #{rk:<2} | E{ei:<1} | {xp:>5.2f} | {mc:>8.2f} | {rx_label:>7} | {reach_label:>5} | {rb_s:<20} | {np_s:<20} | {rot_s:<13} | {occ_part}"

    marker_hdr = " | ".join([f"{g:<5}" for g in SUPPORTED_MARKERS])
    hdr_line = f" {'Rk':<3} | {'E':<2} | {'X(m)':>5} | {'MeanCost':>8} | {'RobtX':>7} | {'Reach':>5} | {'RobotBase_local(xyz)':<20} | {'NDI_Pos_local(xyz)':<20} | {'NDIRot(xyz)':<13} | {marker_hdr}"
    sep = "-" * len(hdr_line)
    
    print(hdr_line)
    print(sep)
    for row in global_summary_rows:
        print(_row_line(*row))
    print(sep)

    # ── matplotlib colour table PNG ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        def _cell_color(val: str):
            if val == "OK":      return "#c8f7c5"
            if val == "ALWAYS":  return "#f7c5c5"
            if val == "Yes":     return "#a8e6cf"  # Distinct green for reachability
            if val == "No":      return "#ffb3b3"  # Distinct red for reachability
            if val == "N/A":     return "#e2e2e2"  # Gray for N/A
            return                      "#fff3c5"

        col_labels = [
            "Rank", "Env", "NDI X (m)", "Mean Cost", "RobotX", "Reach (1cm)",
            "Robot Base local\n(x,y,z)", "NDI Pos local\n(x,y,z)", "NDI Rot\n(x,y,z)"
        ] + SUPPORTED_MARKERS

        cell_data, cell_colors = [], []
        for r in global_summary_rows:
            rk, ei, xp, mc, rb_s, np_s, rot_s, occ_cols, rx_label, reach_label = r
            cell_data.append([f"#{rk}", str(ei), f"{xp:.2f}", f"{mc:.1f}", rx_label, reach_label, rb_s, np_s, rot_s] + occ_cols)
            
            row_colors = ["#dce9f7"] * 9
            row_colors[5] = _cell_color(reach_label)  # Apply specific color for Reach column
            cell_colors.append(row_colors + [_cell_color(c) for c in occ_cols])

        COL_WIDTHS = {0: 0.04, 1: 0.04, 2: 0.06, 3: 0.08, 4: 0.06, 5: 0.06, 6: 0.16, 7: 0.16, 8: 0.15, 9: 0.06, 10: 0.06, 11: 0.06, 12: 0.06}

        n_cols = len(col_labels)
        total_w = sum(COL_WIDTHS.get(c, 0.08) for c in range(n_cols))
        fig_w = max(16, total_w * 72 / 5.5)
        fig_h = max(3, len(cell_data) * 0.4 + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis('off')
        fig.suptitle(f"GLOBAL NDI Placement Summary\nTarget Pose (local): {target_str_g}", fontsize=12, fontweight='bold', y=0.98)
        
        tbl = ax.table(cellText=cell_data, colLabels=col_labels, cellColours=cell_colors,
            colColours=["#2c6fad"] * n_cols, loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.7)
        for c_idx in range(n_cols):
            w = COL_WIDTHS.get(c_idx, 0.08)
            for r_idx in range(len(cell_data) + 1):
                tbl[(r_idx, c_idx)].set_width(w)
        for j in range(n_cols):
            tbl[(0, j)].set_text_props(color='white', fontweight='bold')
            
        patches = [
            mpatches.Patch(facecolor='#c8f7c5', edgecolor='gray', label='OK  - No Occlusion'),
            mpatches.Patch(facecolor='#fff3c5', edgecolor='gray', label='P   - Partial Occlusion'),
            mpatches.Patch(facecolor='#f7c5c5', edgecolor='gray', label='ALWAYS - Always Occluded'),
            mpatches.Patch(facecolor='#a8e6cf', edgecolor='gray', label='REACH - Yes'),
            mpatches.Patch(facecolor='#ffb3b3', edgecolor='gray', label='REACH - No'),
        ]
        ax.legend(handles=patches, loc='lower right', fontsize=8, framealpha=0.9, bbox_to_anchor=(1.0, -0.02))
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        
        output_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "occlusion_output"
        output_dir.mkdir(exist_ok=True)
        from datetime import datetime as _dt
        _ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        table_img_path = output_dir / f"ndi_moving_target_table_{_ts}.png"
        plt.savefig(table_img_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"\n[OK] Global Placement table PNG saved: {table_img_path}")
    except Exception as _e:
        print(f"[WARN] matplotlib table failed: {_e}")

    # Save JSON
    final_json_data = {
        "global_info": {
            "target_pose_position_local": target_export,
            "robot_x_list": robot_x_list,
            "num_envs": num_envs,
            "ndi_x_positions": ndi_x_positions.tolist(),
            "target_frames": target_frames
        },
        "batches": global_json_batches,
        "global_ranking": [
            {
                "global_rank": r[0],
                "env_idx": r[1],
                "ndi_x_pos": r[2],
                "mean_cost": r[3],
                "robot_x_val": r[8],
                "occlusion_cols": r[7]
            } for r in global_summary_rows
        ]
    }
    
    export_path = output_dir / f"ndi_moving_target_score_{_ts}.json"
    with open(export_path, "w") as f:
        import json
        json.dump(final_json_data, f, indent=4)
    print(f"\n✅ Moving-target report saved: {export_path}")

    # Keep app open if not headless so user can observe the scenes
    if not args_cli.headless:
        print("\n[INFO] 正在非 Headless 模式... 保持視窗開啟供使用者觀察。請在終端機按 Ctrl+C 結束。")
        try:
            while simulation_app.is_running():
                simulation_app.update()
        except KeyboardInterrupt:
            pass
            
    env.close()

if __name__ == "__main__":
    main()
