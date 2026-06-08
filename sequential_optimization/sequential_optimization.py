# -*- coding: utf-8 -*-
"""sequential_optimization.py — Sequential Robot Base & NDI Camera Optimization

Finds the optimal robot base position (guaranteeing high reachability and manipulability) 
and NDI camera placement angle (guaranteeing high visibility and occlusion tolerance) 
by evaluating candidates sequentially.
"""
import argparse
import sys
import os
import csv
import json
import numpy as np
import torch
from pathlib import Path

from isaaclab.app import AppLauncher

# ── CLI Setup ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Sequential Robot Base & NDI Camera Optimization")
parser.add_argument("--tumor_pos", type=float, nargs=3, default=[-0.20, -0.75, 1.05],
                    help="Tumor target position in world coordinates (X Y Z)")
parser.add_argument("--num_candidates", type=int, default=5,
                    help="Number of top base candidates to retrieve and evaluate")
parser.add_argument("--num_envs", type=int, default=10,
                    help="Number of parallel environments (corresponds to NDI angles 0 to 45 degrees)")
parser.add_argument("--checkpoint", type=str,
                    default=r"logs\rsl_rl\tm5_reach_stable_v2\2026-02-04_00-28-35\model_9999.pt",
                    help="RL policy checkpoint path")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Directory to save the CSV summary and JSON results")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-Launch Imports ───────────────────────────────────────────────────
from pxr import Usd, UsdGeom, Gf
import omni.usd
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_extension_linkout_cone_orientation_cfg import (
    TM5ExtensionLinkOutFanOrientationEnvCfg,
)
from rsl_rl.modules import ActorCritic

# Path setup for NDI detector
FINAL_WS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "final")
sys.path.append(FINAL_WS_DIR)
from ndi_detector_isaaclab_experimental import NDIConfig, NDIDetector

SUPPORTED_MARKERS = ["BM", "EM", "FM", "UM"]

# ── Config and Structure Helpers ──────────────────────────────────────────
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
    if not env_prim.IsValid(): 
        return result
    def search(prim, depth=0):
        if depth > 10: 
            return
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
            if "link_6_path" in result and "emitter_path" in result and "needle_tip_path" in result: 
                return
    search(env_prim)
    return result

# ── Math Utilities ────────────────────────────────────────────────────────
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-12 else np.zeros_like(v)

def cross(v1, v2):
    return np.cross(v1, v2)

def matrix_to_quat(m):
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

def look_at_quaternion_world(ndi_pos, target_pos):
    """
    Returns (qw, qx, qy, qz) representing a rotation where the camera points
    its -Z axis toward target_pos, with a fixed 90-degree roll (frustum wide side horizontal).
    """
    forward = normalize(target_pos - ndi_pos) # This is where we want -Z to point
    z_axis = -forward 
    
    # World up hint
    up_hint = np.array([0, 0, 1.0])
    if abs(np.dot(z_axis, up_hint)) > 0.999: # Edge case
        up_hint = np.array([1.0, 0, 0])
        
    x_axis = normalize(cross(up_hint, z_axis))
    y_axis = cross(z_axis, x_axis)
    
    # Apply Roll=90: map camera X to y_axis, and camera Y to -x_axis
    new_right = y_axis
    new_up = -x_axis
    new_z = z_axis
    
    R = np.column_stack((new_right, new_up, new_z))
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
    quat = np.array([rotation.GetReal(), rotation.GetImaginary()[0], rotation.GetImaginary()[1], rotation.GetImaginary()[2]])
    return np.array(translation), quat

def set_prim_transform_orient(stage, prim_path, translation, q_local):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    xform = UsdGeom.Xformable(prim)
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

# ── Bounding Box Distance Calculation ─────────────────────────────────────
def bbox_distance(tx, ty, tz, bounds):
    dx = max(bounds["x_min"] - tx, 0, tx - bounds["x_max"])
    dy = max(bounds["y_min"] - ty, 0, ty - bounds["y_max"])
    dz = max(bounds["z_min"] - tz, 0, tz - bounds["z_max"])
    return (dx**2 + dy**2 + dz**2) ** 0.5

# ── Main Stage Implementation ─────────────────────────────────────────────
def main():
    tumor_pos = np.array(args_cli.tumor_pos)
    num_envs = args_cli.num_envs
    num_candidates = args_cli.num_candidates
    
    print("\n" + "="*80)
    print("      SEQUENTIAL BASE & NDI CAMERA OPTIMIZATION SYSTEM")
    print("="*80)
    print(f"Target Tumor Position: X={tumor_pos[0]:.3f}, Y={tumor_pos[1]:.3f}, Z={tumor_pos[2]:.3f}")
    print(f"Parallel Environments (NDI Angles): {num_envs} (Angles: 0 to 45 deg)")
    
    output_dir = args_cli.output_dir if args_cli.output_dir else os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    # ── STAGE 1: Retrieve Robot Base Candidates ───────────────────────────
    print("\n[Stage 1] Querying cmap_database.json for matched workspace block...")
    db_path = os.path.abspath(os.path.join("scripts", "isaaclab_ws", "reachability_map", "cmap_database.json"))
    if not os.path.exists(db_path):
        print(f"[ERROR] Database file not found at: {db_path}")
        sys.exit(1)
        
    with open(db_path, "r") as f:
        db = json.load(f)
        
    tx, ty, tz = tumor_pos
    matching_blocks = []
    
    # Strictly check bounds
    for bid, bdata in db["blocks"].items():
        bounds = bdata.get("wp_bounds", {})
        if bounds:
            in_x = bounds["x_min"] <= tx <= bounds["x_max"]
            in_y = bounds["y_min"] <= ty <= bounds["y_max"]
            in_z = bounds["z_min"] <= tz <= bounds["z_max"]
            if in_x and in_y and in_z:
                matching_blocks.append((bid, bdata))
                
    best_block_id = None
    block_data = None
    if matching_blocks:
        def block_sort_key(item):
            opt = item[1].get("optimal_base", {})
            return (opt.get("success_rate", 0.0), opt.get("sr_min", 0), opt.get("fg_score", 0.0), opt.get("avg_manip", 0.0))
        best_block_id, block_data = max(matching_blocks, key=block_sort_key)
        print(f"-> Strict Match: Found tumor inside Block {best_block_id}'s bounds.")
    else:
        # Fallback to closest bounding box
        min_dist = float("inf")
        for bid, bdata in db["blocks"].items():
            bounds = bdata.get("wp_bounds", {})
            if bounds:
                dist = bbox_distance(tx, ty, tz, bounds)
                if dist < min_dist:
                    min_dist = dist
                    best_block_id = bid
                    block_data = bdata
        print(f"-> BBox Fallback: Closest block is Block {best_block_id} (distance: {min_dist:.4f}m)")
        
    # Read optimization CSV for candidates
    opt_csv_rel = block_data.get("opt_csv")
    opt_csv_path = os.path.abspath(opt_csv_rel)
    print(f"Loading candidate base positions from CSV: {opt_csv_path}")
    
    candidates = []
    if os.path.exists(opt_csv_path):
        with open(opt_csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # row: env_id,base_x,base_y,success_count,success_rate,avg_manip,min_manip,sr_min,prod_c,fg_score
                candidates.append({
                    "base_x": float(row["base_x"]),
                    "base_y": float(row["base_y"]),
                    "success_rate": float(row["success_rate"]),
                    "sr_min": int(row["sr_min"]),
                    "fg_score": float(row["fg_score"]),
                    "avg_manip": float(row["avg_manip"]),
                    "min_manip": float(row["min_manip"])
                })
    else:
        print(f"[ERROR] Candidates CSV file not found: {opt_csv_path}")
        sys.exit(1)
        
    # Sort candidates by: success_rate DESC, sr_min DESC, fg_score DESC, avg_manip DESC
    candidates.sort(key=lambda c: (c["success_rate"], c["sr_min"], c["fg_score"], c["avg_manip"]), reverse=True)
    
    # Select top candidates
    selected_candidates = candidates[:num_candidates]
    print(f"\nTop {len(selected_candidates)} Robot Base Candidates retrieved:")
    for idx, c in enumerate(selected_candidates):
        print(f"  [{idx}] X={c['base_x']:.2f}, Y={c['base_y']:.2f} | Success Rate: {c['success_rate']*100:.1f}%, Avg Manip: {c['avg_manip']:.4f}")
        
    # ── STAGE 2: Setup Environment and Sweep Candidates ───────────────────
    print("\n[Stage 2] Starting simulation environment for occlusion sweeps...")
    env_cfg = TM5ExtensionLinkOutFanOrientationEnvCfg()
    env_cfg.scene.num_envs = num_envs
    
    # Disable command resampling
    if hasattr(env_cfg.commands, "ee_pose"):
        env_cfg.commands.ee_pose.resampling_time_range = (1e10, 1e10)
        
    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.device
    stage = omni.usd.get_context().get_stage()
    
    # We compute default block center and default UM position to derive default LACP
    # base_pos_w for env_0 is env_origins[0] + ROBOTARM_BASE_LOCAL
    env_origins_np = env.scene.env_origins.cpu().numpy()
    default_base_pos_w = env_origins_np[0] + np.array([0.3, -0.1741, 0.9493])
    default_um_pos = default_base_pos_w + np.array([0.031517988675667796, -0.50597902213614, 0.9193813560801563])
    block_center = np.array(block_data["block_center"])
    block_center_w = env_origins_np[0] + block_center
    default_lacp = (block_center_w + default_um_pos) / 2.0
    
    # Constant NDI Arc parameters based on default camera distance to LACP
    R_h = 1.52
    H = 0.978
    print(f"Using NDI Arc parameters: Height Offset H={H:.4f}m, Horizontal Radius R_h={R_h:.4f}m")
    
    # Initialize NDIDetectors for all environments
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
            print(f"[ERROR] Failed to initialize detector for env {i}: {e}")
            detectors.append(None)
            
    # Load RL policy
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
    print(f"[OK] Policy loaded successfully from: {ckpt_path.name}")
    
    sweep_results = []
    
    # Loop over robot base candidates
    for c_idx, candidate in enumerate(selected_candidates):
        c_x = candidate["base_x"]
        c_y = candidate["base_y"]
        print(f"\nEvaluating Base Candidate [{c_idx}]: X={c_x:.2f}, Y={c_y:.2f}...")
        
        # Reset and position robot bases in simulation
        obs, _ = env.reset()
        robot = env.scene["robot"]
        root_pos_w = robot.data.root_pos_w.clone()
        
        # Shift X and Y locally per environment origin
        root_pos_w[:, 0] = env.scene.env_origins[:, 0] + c_x
        root_pos_w[:, 1] = env.scene.env_origins[:, 1] + c_y
        robot.write_root_pose_to_sim(torch.cat([root_pos_w, robot.data.root_quat_w], dim=-1))
        env.sim.step()
        obs = env.observation_manager.compute()
        
        # Override target pose command to fixed tumor target (expressed locally relative to each env origin)
        sim_dtype = robot.data.root_pos_w.dtype
        fixed_target_local = torch.tensor([tx, ty, tz], device=device, dtype=sim_dtype)
        target_local_expanded = fixed_target_local.unsqueeze(0).repeat(num_envs, 1)
        target_pos_w = env.scene.env_origins + target_local_expanded
        target_quat_w = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device, dtype=sim_dtype).unsqueeze(0).repeat(num_envs, 1)
        
        cmd_pos_b, cmd_quat_b = subtract_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w,
            target_pos_w, target_quat_w
        )
        
        try:
            cmd_term = env.command_manager.get_term("ee_pose")
            if cmd_term.command.shape[1] == 3:
                cmd_term.command[:] = cmd_pos_b
            elif cmd_term.command.shape[1] == 7:
                cmd_term.command[:] = torch.cat([cmd_pos_b, cmd_quat_b], dim=1)
            obs = env.observation_manager.compute()
        except Exception as e:
            print(f"[WARN] Command override failed: {e}")
            
        # Drive robot arm to target waypoint using policy
        dones = torch.zeros(num_envs, dtype=torch.bool, device=device)
        step = 0
        max_steps = 100
        
        while not dones.all() and step <= max_steps:
            with torch.no_grad():
                obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
                actions = policy.act(obs_tensor, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(actions)
            dones = (dones | terminated.squeeze(-1) | truncated.squeeze(-1)
                     if terminated.dim() > 1 else dones | terminated | truncated)
            step += 1
            
        # Settle render context
        env.sim.render()
        
        # NDI Sweep Angles from 0° (front) to 45° (slanted) in 5° increments
        angles_deg = np.linspace(0, 45, num_envs) # 10 angles
        env_visibility = {i: {} for i in range(num_envs)}
        env_passed = np.zeros(num_envs, dtype=bool)
        
        # Calculate LACP and position cameras
        final_lacp_9 = default_lacp
        for i in range(num_envs):
            detector = detectors[i]
            if detector is None:
                continue
                
            # Update kinematics to read current joint positions and marker poses
            detector._update_lines()
            valid_positions = [pos for pos in detector.line_ends if pos is not None]
            
            # Compute look-at center point (LACP) of active markers
            centroid = np.mean(valid_positions, axis=0) if valid_positions else None
            if i == 9 and centroid is not None:
                final_lacp_9 = centroid
            
            # Circular arc mathematics
            theta_rad = np.deg2rad(angles_deg[i])
            ndi_path = f"/World/envs/env_{i}/Root/NDI"
            
            if centroid is not None:
                # Target-dependent LACP
                LACP = centroid
                
                # Position coordinates on circular horizontal arc around LACP
                # X = LACP_x + R_h * sin(theta)
                # Y = LACP_y - R_h * cos(theta)
                # Z = LACP_z + H
                ndi_world_x = LACP[0] + R_h * np.sin(theta_rad)
                ndi_world_y = LACP[1] - R_h * np.cos(theta_rad)
                ndi_world_z = LACP[2] + H
                ndi_world_pos = np.array([ndi_world_x, ndi_world_y, ndi_world_z])
                
                # Look NDI camera directly at LACP with type-safe alignment
                quat_world = look_at_quaternion_world(ndi_world_pos, LACP)
                
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
                
                # Convert world position to parent-local coordinates
                ndi_local_pos = np.array(parent_matrix.GetInverse().Transform(Gf.Vec3d(*ndi_world_pos)))
                
                # Apply pose in simulation
                set_prim_transform_orient(stage, ndi_path, ndi_local_pos, q_local)
                
        # Sync changes to simulation engine
        env.sim.render()
        
        # Run detection checks
        for i in range(num_envs):
            detector = detectors[i]
            if detector is None:
                continue
                
            detector.update_detection(env.sim, debug_enabled=True, error_only_log=True)
            detector.draw_live(clear_first=(i == 0))
            
            # Aggregate visibility results
            group_visible_balls = {m: 0 for m in SUPPORTED_MARKERS}
            group_total_balls = {m: 0 for m in SUPPORTED_MARKERS}
            
            for r_info in detector.raycast_results:
                group = r_info['end_path'].split("/")[-1].split("_")[0].upper()
                if group not in SUPPORTED_MARKERS: 
                    continue
                group_total_balls[group] += 1
                if r_info.get('is_target_hit', False):
                    group_visible_balls[group] += 1
                    
            env_visibility[i] = {}
            for group in SUPPORTED_MARKERS:
                vis_count = group_visible_balls[group]
                total = group_total_balls[group]
                env_visibility[i][group] = True if total > 0 and (vis_count == total) else False
                
            # Total pass requires all groups visible
            env_passed[i] = all(env_visibility[i].values())
            
        hit_rate = int(np.sum(env_passed))
        print(f"-> Base X={c_x:.2f} results: Hit Rate = {hit_rate}/10.")
        
        # Record results
        sweep_results.append({
            "candidate_id": c_idx,
            "base_x": c_x,
            "base_y": c_y,
            "success_rate": candidate["success_rate"],
            "avg_manip": candidate["avg_manip"],
            "hit_rate": hit_rate,
            "angle_passes": env_passed.tolist(),
            "lacp_world_9": final_lacp_9.tolist()
        })
        
    # ── STAGE 3: Optimal Selection and Output ─────────────────────────────
    print("\n[Stage 3] Selecting optimal base and generating outputs...")
    
    # Filter candidates with hit_rate >= 8
    qualified = [r for r in sweep_results if r["hit_rate"] >= 8]
    
    is_fallback = False
    if qualified:
        # Pick highest average manipulability
        optimal = max(qualified, key=lambda r: r["avg_manip"])
    else:
        # Fallback: pick highest hit rate, break ties with manipulability
        optimal = max(sweep_results, key=lambda r: (r["hit_rate"], r["avg_manip"]))
        is_fallback = True
        print("[WARNING] No base candidate satisfied hit_rate >= 8. Falling back to highest hit rate candidate.")
        
    print("\n" + "="*80)
    print("                      OPTIMIZATION SELECTION RESULTS")
    print("="*80)
    if is_fallback:
        print(f"*** OPTIMAL BASE (FALLBACK): X={optimal['base_x']:.2f}m, Y={optimal['base_y']:.2f}m ***")
    else:
        print(f"*** OPTIMAL BASE SELECTED: X={optimal['base_x']:.2f}m, Y={optimal['base_y']:.2f}m ***")
    print(f"  Reachability Success Rate: {optimal['success_rate']*100:.1f}%")
    print(f"  Average Manipulability Index: {optimal['avg_manip']:.6f}")
    print(f"  NDI Camera Hit Rate: {optimal['hit_rate']}/10 (Qualified for >=8: {'Yes' if not is_fallback else 'No'})")
    
    # Compute and output the slanted 45° NDI position (angle index 9)
    # Re-calculate LACP for optimal base in simulation
    opt_c_idx = optimal["candidate_id"]
    # Rerun or get simulated positions
    # We can retrieve the final LACP of env 9 for the optimal base
    # (Since we ran env 9 in the loop, we can store it or calculate it)
    print(f"Calculated 45° slanted NDI position:")
    
    # Angle index 9 corresponds to 45° slant
    theta_opt = np.deg2rad(45.0)
    
    # Find the LACP computed during env_9 for the optimal base runs
    final_opt_lacp = np.array(optimal["lacp_world_9"])
    
    # Apply arc math for 45° position centered at final_opt_lacp
    ndi_45_x = final_opt_lacp[0] + R_h * np.sin(theta_opt)
    ndi_45_y = final_opt_lacp[1] - R_h * np.cos(theta_opt)
    ndi_45_z = final_opt_lacp[2] + H
    ndi_45_pos = np.array([ndi_45_x, ndi_45_y, ndi_45_z])
    
    # Look NDI at final_opt_lacp
    quat_45 = look_at_quaternion_world(ndi_45_pos, final_opt_lacp)
    print(f"  NDI 45° Position (World): X={ndi_45_pos[0]:.4f}, Y={ndi_45_pos[1]:.4f}, Z={ndi_45_pos[2]:.4f}")
    print(f"  NDI 45° Orientation (Quat WXYZ): W={quat_45[0]:.4f}, X={quat_45[1]:.4f}, Y={quat_45[2]:.4f}, Z={quat_45[3]:.4f}")
    print(f"  Look At Center Point (LACP) (World): X={final_opt_lacp[0]:.4f}, Y={final_opt_lacp[1]:.4f}, Z={final_opt_lacp[2]:.4f}")
    
    # Save outputs to CSV summary
    csv_filename = os.path.join(output_dir, "optimization_summary.csv")
    with open(csv_filename, "w", newline="") as csvfile:
        fieldnames = ["candidate_id", "base_x", "base_y", "success_rate", "avg_manip", "hit_rate", "is_optimal"]
        # Add angle columns
        for a in range(0, 46, 5):
            fieldnames.append(f"angle_{a}_pass")
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in sweep_results:
            row_data = {
                "candidate_id": r["candidate_id"],
                "base_x": r["base_x"],
                "base_y": r["base_y"],
                "success_rate": r["success_rate"],
                "avg_manip": r["avg_manip"],
                "hit_rate": r["hit_rate"],
                "is_optimal": (r["candidate_id"] == optimal["candidate_id"])
            }
            for idx, a in enumerate(range(0, 46, 5)):
                row_data[f"angle_{a}_pass"] = 1 if r["angle_passes"][idx] else 0
            writer.writerow(row_data)
            
    print(f"\n[OK] Optimization summary saved to: {csv_filename}")
    
    # Save optimal placement details to JSON
    json_filename = os.path.join(output_dir, "optimal_placement.json")
    json_data = {
        "tumor_pos": tumor_pos.tolist(),
        "matched_block_id": int(best_block_id),
        "robot_base_pos": [optimal["base_x"], optimal["base_y"], 0.9493],
        "ndi_45_pos": ndi_45_pos.tolist(),
        "ndi_45_orientation_quat_wxyz": quat_45.tolist(),
        "lacp_world": final_opt_lacp.tolist(),
        "success_rate": optimal["success_rate"],
        "avg_manip": optimal["avg_manip"],
        "hit_rate": optimal["hit_rate"],
        "is_fallback": is_fallback
    }
    with open(json_filename, "w") as jsonfile:
        json.dump(json_data, jsonfile, indent=4)
    print(f"[OK] Optimal placement configuration saved to: {json_filename}")
    print("="*80 + "\n")
    
    # Clean up and exit
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
