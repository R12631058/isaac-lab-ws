"""
ndi_multipose_scorer.py

Evaluates 10 NDI positions across 10 sampled robot poses (frames 10~100) using a solid angle 
and continuous occlusion penalty cost function derived from Bestpose_v7.
"""
import argparse
import sys
import os
import time
import json
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
parser.add_argument("--checkpoint", type=str,
                    default=r"logs\rsl_rl\tm5_reach_stable_v2\2026-02-04_00-28-35\model_9999.pt",
                    help="RL policy checkpoint")
parser.add_argument("--x_start", type=float, default=0.6,
                    help="NDI X Start")
parser.add_argument("--x_end", type=float, default=2.8,
                    help="NDI X End")
parser.add_argument("--debug_verbose", action="store_true",
                    help="Enable detailed output")
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
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_extension_linkout_cone_orientation_cfg import TM5ExtensionLinkOutFanOrientationEnvCfg
from rsl_rl.modules import ActorCritic

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
        if name == "NDI_emitter" and "emitter_path" not in result:
            result["emitter_path"] = path
        for child in prim.GetChildren():
            search(child, depth + 1)
            if "link_6_path" in result and "emitter_path" in result: return
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

def get_usd_target_position(env: ManagerBasedRLEnv):
    """Get /Root/Cube position from Env 0 (World Frame)"""
    stage = env.unwrapped.scene.stage
    prim_path = "/World/envs/env_0/Root/Cube"

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

# ── Main Script ──
def main():
    num_envs = args_cli.num_envs
    x_positions = np.linspace(args_cli.x_start, args_cli.x_end, num_envs)
    
    print(f"\n[INIT] 準備啟動 {num_envs} 個環境, NDI X 位置從 {args_cli.x_start} 到 {args_cli.x_end}")
    
    env_cfg = TM5ExtensionLinkOutFanOrientationEnvCfg()
    env_cfg.scene.num_envs = num_envs
    
    # Disable command resampling - keep target fixed
    if hasattr(env_cfg.commands, "ee_pose"):
        env_cfg.commands.ee_pose.resampling_time_range = (1e10, 1e10)
        
    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.device
    
    # 取得 Target
    def _local_get_usd_target_position(env_obj: ManagerBasedRLEnv):
        st = env_obj.unwrapped.scene.stage
        pr = st.GetPrimAtPath("/World/envs/env_0/Root/Cube")
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
            detector._link_paths["link_base"] = ndi_config.LINK_6_PATH.replace("link_6", "link_0")
            detector.initialize()
            detectors.append(detector)
        except Exception as e:
            print(f"[ERROR] Failed to init NDI for env {i}: {e}")
            detectors.append(None)
    print(f"[NDI] Initialized {len([d for d in detectors if d])} detectors.\n")
    
    # Policy
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
    
    # ── Simulation and Mult-Pose Scoring ──
    scorer = MultiPoseScorer(num_envs)
    target_frames = [10, 30, 50, 70, 90]
    
    for ep_idx in range(args_cli.episodes):
        obs, _ = env.reset()
        
        # --- COMMAND OVERRIDE FOR FIXED WORLD TARGET ---
        if fixed_target_local is not None:
            with torch.no_grad():
                robot = env.scene["robot"]
                robot_root_pos_w = robot.data.root_pos_w
                robot_root_quat_w = robot.data.root_quat_w

                env_origins = env.scene.env_origins
                target_local_expanded = fixed_target_local.unsqueeze(0).repeat(num_envs, 1)
                target_pos_w = env_origins + target_local_expanded

                target_quat_w = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device).unsqueeze(0).repeat(num_envs, 1)

                cmd_pos_b, cmd_quat_b = subtract_frame_transforms(
                    robot_root_pos_w, robot_root_quat_w,
                    target_pos_w, target_quat_w
                )

                term_name = "ee_pose"
                try:
                    cmd_term = env.command_manager.get_term(term_name)
                    if cmd_term.command.shape[1] == 3:
                        cmd_term.command[:] = cmd_pos_b
                    elif cmd_term.command.shape[1] == 7:
                        cmd_term.command[:] = torch.cat([cmd_pos_b, cmd_quat_b], dim=1)

                    obs = env.observation_manager.compute()
                except Exception as e:
                    print(f"[WARN] Failed to override command: {e}")
        # -----------------------------------------------
        
        # Reset Scorer state 
        scorer.occlusion_history = {i: {m: 0 for m in SUPPORTED_MARKERS} for i in range(num_envs)}
        scorer.frame_results = []
        
        # Per-frame visibility log: env_idx -> frame -> {group: bool}
        env_frame_vis_log = {i: {} for i in range(num_envs)}
        # Last sampled frame NDI quaternion per env
        last_sampled_ndi_quat = {i: None for i in range(num_envs)}
        
        # Set NDI positions (approximate initial positions)
        for i in range(num_envs):
            ndi_path = f"/World/envs/env_{i}/Root/NDI"
            local_pos = np.array([x_positions[i], -1.998, 2.027])
            set_prim_transform_xformop(stage, ndi_path, local_pos)
            
        step = 0
        dones = torch.zeros(num_envs, dtype=torch.bool, device=device)
        
        while not dones.all() and step <= max(target_frames):
            with torch.no_grad():
                obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
                actions = policy.act(obs_tensor, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(actions)
            dones = (dones | terminated.squeeze(-1) | truncated.squeeze(-1)
                     if terminated.dim() > 1 else dones | terminated | truncated)
            
            # Target Frame Hit
            if step in target_frames:
                env.sim.render() # force physics -> USD
                
                print(f"\n=========================================")
                print(f"🎯 抽出 Frame {step} 進行評分分析")
                print(f"=========================================")
                
                env_visibility = {}
                n_positions_w = {}
                env_marker_groups_positions = {}
                
                for i in range(num_envs):
                    detector = detectors[i]
                    if detector is None:
                        continue
                        
                    # 1. Look-at target (Centroid)
                    detector._update_lines() # Kinematic sync happens here
                    
                    valid_positions = [pos for pos in detector.line_ends if pos is not None]
                    centroid = np.mean(valid_positions, axis=0) if valid_positions else None
                    
                    ndi_path = f"/World/envs/env_{i}/Root/NDI"
                    ndi_pos, _ = get_prim_world_transform(stage, ndi_path)
                    
                    if centroid is not None and ndi_pos is not None:
                        quat_world = look_at_quaternion_world(ndi_pos, centroid)
                        import scipy.spatial.transform as sst
                        rot = sst.Rotation.from_quat([quat_world[1], quat_world[2], quat_world[3], quat_world[0]])
                        euler_xyz = rot.as_euler('xyz', degrees=True)
                        
                        local_pos = np.array([x_positions[i], -1.998, 2.027])
                        set_prim_transform_xformop(stage, ndi_path, local_pos, euler_xyz)
                        env.sim.render() # 確保 NDI 轉動反映到 USD
                        
                    # 2. RUN FULL DETECTION 
                    detector.update_detection(env.sim, debug_enabled=True, error_only_log=True)
                    detector.draw_live(clear_first=(i == 0))
                    
                    # 3. 解析結果
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
                    n_pos, n_quat = get_prim_world_transform(stage, ndi_path)
                    n_positions_w[i] = n_pos
                    
                    for group in SUPPORTED_MARKERS:
                        vis_count = len(group_visible_balls[group])
                        total = group_total_balls[group]
                        env_visibility[i][group] = True if total > 0 and (vis_count / total >= 0.75) else False
                    
                    # 紀錄各幀可見性 & 最後一幀的 NDI 四元數
                    env_frame_vis_log[i][step] = dict(env_visibility[i])
                    last_sampled_ndi_quat[i] = n_quat  # 每次都覆蓋，最終剩 frame 90
                        
                # 4. Score calculating
                frame_costs = scorer.score_frame(step, env_visibility, n_positions_w, env_marker_groups_positions)
                
                # Output stats
                for env_idx, fe_data in enumerate(frame_costs):
                    vis_str = []
                    for g in SUPPORTED_MARKERS:
                        status = "✓" if env_visibility[env_idx][g] else "x"
                        cost = fe_data["group_costs"][g]
                        vis_str.append(f"{g}:{status}({cost:.2f})")
                        
                    print(f"  Env {env_idx:2d} (x={x_positions[env_idx]:.2f}) | Total Cost: {fe_data['total_cost']:6.2f} | {'  '.join(vis_str)}")
                    
            step += 1
            
        # End of episode, summarize the matrix
        print("\n" + "="*50)
        print("📊 NDI 位置 × 採樣幀 總成本矩陣 (Cost Matrix)")
        print("="*50)
        
        print("Env (X Pos) | " + " | ".join([f"Frame {tf:2d}" for tf in target_frames]) + " | Mean Cost")
        print("-" * 75)
        
        env_mean_costs = []
        for i in range(num_envs):
            x_pos = x_positions[i]
            env_costs = []
            for res in scorer.frame_results:
                e_data = [e for e in res["envs"] if e["env_idx"] == i][0]
                env_costs.append(e_data["total_cost"])
            
            mean_cost = np.mean(env_costs)
            env_mean_costs.append(mean_cost)
            
            cost_str = " | ".join([f"{c:8.2f}" for c in env_costs])
            print(f"Env {i} ({x_pos:.1f}) | {cost_str} | {mean_cost:8.2f}")
            
        best_env = np.argmin(env_mean_costs)
        print(f"\n🏆 最佳 NDI 位置: Env {best_env} (X={x_positions[best_env]:.2f}) 平均成本最低 {env_mean_costs[best_env]:.2f}")
        
        # ── Second List: 環境詳細摘要 ──
        target_str = (
            f"[{fixed_target_local[0]:.3f}, {fixed_target_local[1]:.3f}, {fixed_target_local[2]:.3f}]"
            if fixed_target_local is not None else "N/A"
        )
        
        # 排名 (依 meancost 升序)
        rank_order = np.argsort(env_mean_costs)  # index 0 = best
        env_rank = {int(env_i): int(rank+1) for rank, env_i in enumerate(rank_order)}
        
        print("\n" + "="*80)
        print("📋 環境詳細摘要 (NDI Placement Summary)")
        print(f"   Target Pose Position (Local): {target_str}")
        print("="*80)
        
        robot_ref = env.scene["robot"]
        import scipy.spatial.transform as _sst
        
        for i in range(num_envs):
            robot_pos_w = robot_ref.data.root_pos_w[i].cpu().numpy()
            ndi_path = f"/World/envs/env_{i}/Root/NDI"
            ndi_pos_w, _ = get_prim_world_transform(stage, ndi_path)
            
            # NDI Euler (xyz, degrees) from last sampled frame quaternion
            q = last_sampled_ndi_quat[i]  # [w, x, y, z]
            if q is not None:
                r = _sst.Rotation.from_quat([q[1], q[2], q[3], q[0]])  # scipy: [x,y,z,w]
                euler_deg = r.as_euler('xyz', degrees=True)
                ndi_rot_str = f"({euler_deg[0]:.1f}°, {euler_deg[1]:.1f}°, {euler_deg[2]:.1f}°)"
            else:
                ndi_rot_str = "N/A"
            
            ndi_pos_str = (
                f"({ndi_pos_w[0]:.3f}, {ndi_pos_w[1]:.3f}, {ndi_pos_w[2]:.3f})"
                if ndi_pos_w is not None else "N/A"
            )
            
            # 組合遮蔽狀況
            group_occ_status = {}
            for g in SUPPORTED_MARKERS:
                n_frames_occluded = sum(
                    1 for f in target_frames
                    if i in env_frame_vis_log and f in env_frame_vis_log[i]
                    and not env_frame_vis_log[i][f].get(g, True)  # False = occluded
                )
                if n_frames_occluded == len(target_frames):
                    group_occ_status[g] = "總是遮蔽"
                elif n_frames_occluded == 0:
                    group_occ_status[g] = "無遮蔽"
                else:
                    group_occ_status[g] = f"有時遮蔽({n_frames_occluded}/{len(target_frames)}幀)"
            
            # (per-env text suppressed – see ASCII table and PNG below)
            pass

        # ── Collect all env data for table ──
        import scipy.spatial.transform as _sst2
        robot_ref2 = env.scene["robot"]
        env_origins_np = env.scene.env_origins.cpu().numpy()  # (num_envs, 3)
        summary_rows = []
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
            summary_rows.append((env_rank[i], i, x_positions[i], env_mean_costs[i],
                                  rb_s, np_s, rot_s, occ_cols))
        summary_rows.sort(key=lambda r: r[0])

        # ── ASCII compact table (env-local coords) ──
        def _row_line(rk, ei, xp, mc, rb_s, np_s, rot_s, occ_cols):
            occ_part = " | ".join([f"{c:<5}" for c in occ_cols])
            return f" #{rk:<2} | E{ei:<1} | {xp:>5.2f} | {mc:>8.2f} | {rb_s:<20} | {np_s:<20} | {rot_s:<13} | {occ_part}"

        marker_hdr = " | ".join([f"{g:<5}" for g in SUPPORTED_MARKERS])
        hdr_line = f" {'Rk':<3} | {'E':<2} | {'X(m)':>5} | {'MeanCost':>8} | {'RobotBase_local(xyz)':<20} | {'NDI_Pos_local(xyz)':<20} | {'NDIRot(xyz)':<13} | {marker_hdr}"
        sep = "-" * len(hdr_line)
        target_str2 = (
            f"[{fixed_target_local[0]:.3f}, {fixed_target_local[1]:.3f}, {fixed_target_local[2]:.3f}]"
            if fixed_target_local is not None else "N/A"
        )
        print(f"\n{'='*len(sep)}")
        print(f" NDI Placement Summary (env-local coords)  |  Target Pose (local): {target_str2}")
        print(f"{'='*len(sep)}")
        print(hdr_line)
        print(sep)
        for row in summary_rows:
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
                return                      "#fff3c5"

            # ── Column labels – full names ──
            col_labels = [
                "Rank",                         # col 0
                "Env",                          # col 1
                "NDI X (m)",                    # col 2
                "Mean Cost",                    # col 3
                "Robot Base local\n(x,y,z)",   # col 4  ← wide
                "NDI Pos local\n(x,y,z)",      # col 5  ← wide
                "NDI Rot\n(x,y,z)",            # col 6  ← wide
            ] + SUPPORTED_MARKERS               # col 7,8,9,10
            cell_data, cell_colors = [], []
            for rk, ei, xp, mc, rb_s, np_s, rot_s, occ_cols in summary_rows:
                cell_data.append([f"#{rk}", str(ei), f"{xp:.2f}", f"{mc:.1f}", rb_s, np_s, rot_s] + occ_cols)
                cell_colors.append(["#dce9f7"]*7 + [_cell_color(c) for c in occ_cols])

            # ── Column widths (inches) – edit here to tune layout ──
            # Wide columns (4,5,6) are for xyz coordinate strings.
            # Narrow columns: Rank / Env / NDI X / Mean Cost / Marker flags.
            # Increase a value to widen, decrease to narrow.
            COL_WIDTHS = {
                0: 0.05,   # Rank
                1: 0.05,   # Env
                2: 0.07,   # NDI X (m)
                3: 0.09,   # Mean Cost
                4: 0.17,   # Robot Base local (x,y,z)   ← keep wide
                5: 0.17,   # NDI Pos local (x,y,z)      ← keep wide
                6: 0.17,   # NDI Rot (x,y,z)            ← keep wide
                7: 0.07,   # BM occlusion flag
                8: 0.07,   # EM occlusion flag
                9: 0.07,   # FM occlusion flag
                10: 0.07,  # UM occlusion flag
            }

            n_cols = len(col_labels)
            total_w = sum(COL_WIDTHS.get(c, 0.08) for c in range(n_cols))
            fig_w = max(14, total_w * 72 / 6)   # rough pt→inch scaling
            fig_h = max(3, len(cell_data) * 0.6 + 2.0)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            ax.axis('off')
            fig.suptitle(
                f"NDI Placement Summary (env-local coords)\nTarget Pose (local): {target_str2}",
                fontsize=12, fontweight='bold', y=0.98
            )
            tbl = ax.table(
                cellText=cell_data, colLabels=col_labels,
                cellColours=cell_colors,
                colColours=["#2c6fad"] * n_cols,
                loc='center', cellLoc='center',
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1, 1.7)
            # Apply per-column widths from COL_WIDTHS
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
            ]
            ax.legend(handles=patches, loc='lower right', fontsize=8,
                      framealpha=0.9, bbox_to_anchor=(1.0, -0.04))
            plt.tight_layout(rect=[0, 0, 1, 0.92])
            output_dir2 = Path(os.path.dirname(os.path.abspath(__file__))) / "occlusion_output"
            output_dir2.mkdir(exist_ok=True)
            from datetime import datetime as _dt
            _ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            table_img_path = output_dir2 / f"ndi_placement_table_{_ts}.png"
            plt.savefig(table_img_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"\nPlacement table PNG saved: {table_img_path}")
        except Exception as _e:
            print(f"[WARN] matplotlib table failed: {_e}")


        # Save JSON
        from datetime import datetime
        output_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "occlusion_output"
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = output_dir / f"ndi_multipose_score_{timestamp}.json"
        
        # Collect environment meta info
        env_meta = {}
        target_export = fixed_target_local.cpu().numpy().tolist() if fixed_target_local is not None else None
        
        robot = env.scene["robot"]
        import omni.usd
        from pxr import UsdGeom
        stage = omni.usd.get_context().get_stage()
        
        for i in range(num_envs):
            robot_pos = robot.data.root_pos_w[i].cpu().numpy().tolist()
            ndi_path = f"/World/envs/env_{i}/Root/NDI"
            try:
                wt = UsdGeom.Xformable(stage.GetPrimAtPath(ndi_path)).ComputeLocalToWorldTransform(0)
                t = wt.ExtractTranslation()
                q = wt.ExtractRotationQuat()
                ndi_pos = [t[0], t[1], t[2]]
                ndi_quat = [q.GetReal(), q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2]] # w, x, y, z
            except Exception:
                n_pw = n_positions_w.get(i, np.zeros(3))
                ndi_pos = n_pw.tolist() if isinstance(n_pw, np.ndarray) else list(n_pw)
                ndi_quat = [1.0, 0.0, 0.0, 0.0]
                
            env_meta[f"env_{i}"] = {
                "robot_base_position": robot_pos,
                "ndi_position": ndi_pos,
                "ndi_orientation_quat_wxyz": ndi_quat
            }
            
        export_data = {
            "global_info": {
                "target_pose_position_local": target_export
            },
            "num_envs": num_envs,
            "x_positions": x_positions.tolist(),
            "target_frames": target_frames,
            "environments": env_meta,
            "results": scorer.frame_results,
            "matrix_summary": {
                f"env_{i}": {
                    "mean_cost": env_mean_costs[i],
                    "x_pos": x_positions[i]
                } for i in range(num_envs)
            },
            "placement_summary": {
                f"env_{i}": {
                    "rank": env_rank[i],
                    "mean_cost": env_mean_costs[i],
                    "ndi_x_pos": float(x_positions[i]),
                    "occlusion_per_group": {
                        g: (
                            "always_occluded" if sum(
                                1 for f in target_frames
                                if i in env_frame_vis_log and f in env_frame_vis_log[i]
                                and not env_frame_vis_log[i][f].get(g, True)
                            ) == len(target_frames)
                            else "never_occluded" if sum(
                                1 for f in target_frames
                                if i in env_frame_vis_log and f in env_frame_vis_log[i]
                                and not env_frame_vis_log[i][f].get(g, True)
                            ) == 0
                            else f"sometimes_occluded_{sum(1 for f in target_frames if i in env_frame_vis_log and f in env_frame_vis_log[i] and not env_frame_vis_log[i][f].get(g, True))}_of_{len(target_frames)}"
                        )
                        for g in SUPPORTED_MARKERS
                    },
                    "frame_visibility": {
                        str(f): env_frame_vis_log[i].get(f, {})
                        for f in target_frames
                    }
                } for i in range(num_envs)
            }
        }
        with open(export_path, "w") as f:
            json.dump(export_data, f, indent=4)
        print(f"\n✅ 詳細報告已輸出至: {export_path}")

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
