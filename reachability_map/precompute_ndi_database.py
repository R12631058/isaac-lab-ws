# -*- coding: utf-8 -*-
"""precompute_ndi_database.py — Precomputes NDI occlusion results for all blocks and candidates.

Generates the unified offline database `ndi_precomputed_database.json` 
to enable instant offline visual verification in the Tkinter GUI.
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

# CLI Setup
parser = argparse.ArgumentParser(description="Precompute NDI Occlusion Database")
parser.add_argument("--num_envs", type=int, default=10,
                    help="Number of environments (NDI angles)")
parser.add_argument("--checkpoint", type=str,
                    default=r"logs\rsl_rl\tm5_reach_stable_v2\2026-02-04_00-28-35\model_9999.pt",
                    help="RL policy checkpoint path")
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

class EnvNDIConfig(NDIConfig):
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

# Math Utilities
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-12 else np.zeros_like(v)

def cross(v1, v2): return np.cross(v1, v2)

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
    forward = normalize(target_pos - ndi_pos)
    z_axis = -forward 
    up_hint = np.array([0, 0, 1.0])
    if abs(np.dot(z_axis, up_hint)) > 0.999:
        up_hint = np.array([1.0, 0, 0])
    x_axis = normalize(cross(up_hint, z_axis))
    y_axis = cross(z_axis, x_axis)
    new_right = y_axis
    new_up = -x_axis
    new_z = z_axis
    R = np.column_stack((new_right, new_up, new_z))
    return matrix_to_quat(R)

def get_prim_world_transform(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid(): return None, None
    xform = UsdGeom.Xformable(prim)
    time_code = Usd.TimeCode.Default()
    transform = xform.ComputeLocalToWorldTransform(time_code)
    translation = transform.ExtractTranslation()
    rotation = transform.ExtractRotationQuat()
    quat = np.array([rotation.GetReal(), rotation.GetImaginary()[0], rotation.GetImaginary()[1], rotation.GetImaginary()[2]])
    return np.array(translation), quat

def set_prim_transform_orient(stage, prim_path, translation, q_local):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid(): return
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


def main():
    num_envs = args_cli.num_envs
    print("\n" + "="*80)
    print("      OFFLINE NDI OCCLUSION DATABASE PRECOMPUTATION")
    print("="*80)
    
    # Load database
    db_path = os.path.abspath(os.path.join("scripts", "isaaclab_ws", "reachability_map", "cmap_database.json"))
    if not os.path.exists(db_path):
        print(f"[ERROR] Database file not found at: {db_path}")
        sys.exit(1)
        
    with open(db_path, "r") as f:
        db = json.load(f)
        
    # Start simulator
    env_cfg = TM5ExtensionLinkOutFanOrientationEnvCfg()
    env_cfg.scene.num_envs = num_envs
    if hasattr(env_cfg.commands, "ee_pose"):
        env_cfg.commands.ee_pose.resampling_time_range = (1e10, 1e10)
        
    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.device
    stage = omni.usd.get_context().get_stage()
    
    # Initialize the simulation state to read the default root pose
    env.reset()
    robot = env.scene["robot"]
    env_origins = env.scene.env_origins
    
    # Calculate base-to-root offset T (same as parallel_base_optimizer_linear.py)
    ROBOTARM_BASE_DEFAULT = np.array([0.2, 0.0, 0.0])
    tm5_700_default_local = (robot.data.root_pos_w[0] - env_origins[0]).cpu().numpy()
    T = tm5_700_default_local - ROBOTARM_BASE_DEFAULT
    T_torch = torch.tensor(T, device=device, dtype=torch.float32)
    print(f"[INFO] T (base-to-root offset): ({T[0]:.4f}, {T[1]:.4f}, {T[2]:.4f})")
    
    # Constant parameters for NDI Arc
    R_h = 1.52
    H = 0.978
    
    env_origins_np = env_origins.cpu().numpy()
    
    # Compute default LACP (World Frame)
    default_base_pos_w = env_origins_np[0] + np.array([0.3, -0.1741, 0.9493])
    default_um_pos = default_base_pos_w + np.array([0.031517988675667796, -0.50597902213614, 0.9193813560801563])
    
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
            print(f"[ERROR] Detector init failed: {e}")
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
    
    # Target database output dictionary
    precomputed_db = {"blocks": {}}
    
    # Sweep through all 10 blocks (0 to 9)
    for block_id in sorted(db["blocks"].keys(), key=int):
        block_data = db["blocks"][block_id]
        print(f"\nEvaluating Block {block_id}...")
        
        # Bounding box center as representative target
        bounds = block_data["wp_bounds"]
        tx = (bounds["x_min"] + bounds["x_max"]) / 2.0
        ty = (bounds["y_min"] + bounds["y_max"]) / 2.0
        tz = 0.95
        
        # Determine default block center in world frame for fallback LACP (bounding box containing waypoints and default_um_pos)
        block_min_w = env_origins_np[0] + np.array([bounds["x_min"], bounds["y_min"], bounds["z_min"]])
        block_max_w = env_origins_np[0] + np.array([bounds["x_max"], bounds["y_max"], bounds["z_max"]])
        min_coords = np.minimum(block_min_w, default_um_pos)
        max_coords = np.maximum(block_max_w, default_um_pos)
        default_lacp = (min_coords + max_coords) / 2.0
        
        # Load block CSV candidates
        opt_csv_rel = block_data.get("opt_csv")
        opt_csv_path = os.path.abspath(opt_csv_rel)
        candidates = []
        if os.path.exists(opt_csv_path):
            with open(opt_csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sr = float(row["success_rate"])
                    # ONLY evaluate bases where mechanical success_rate is > 0
                    if sr > 0.0:
                        candidates.append({
                            "base_x": float(row["base_x"]),
                            "base_y": float(row["base_y"])
                        })
        else:
            print(f"[WARN] No CSV for block {block_id}, skipping.")
            continue
            
        precomputed_db["blocks"][block_id] = {
            "representative_target": [tx, ty, tz],
            "candidates": {}
        }
        
        # Evaluate each base candidate
        for c in candidates:
            c_x = c["base_x"]
            c_y = c["base_y"]
            print(f"  Base X={c_x:.2f}, Y={c_y:.2f}...")
            
            # Reset & Move base in simulation
            obs, _ = env.reset()
            robot = env.scene["robot"]
            root_pos_w = robot.data.root_pos_w.clone()
            root_pos_w[:, 0] = env.scene.env_origins[:, 0] + c_x + T_torch[0]
            root_pos_w[:, 1] = env.scene.env_origins[:, 1] + c_y + T_torch[1]
            root_pos_w[:, 2] = env.scene.env_origins[:, 2] + T_torch[2]
            robot.write_root_pose_to_sim(torch.cat([root_pos_w, robot.data.root_quat_w], dim=-1))
            env.sim.step()
            obs = env.observation_manager.compute()
            
            # Command Override to target
            fixed_target_local = torch.tensor([tx, ty, tz], device=device, dtype=robot.data.root_pos_w.dtype)
            target_local_expanded = fixed_target_local.unsqueeze(0).repeat(num_envs, 1)
            target_pos_w = env.scene.env_origins + target_local_expanded
            target_quat_w = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device, dtype=robot.data.root_pos_w.dtype).unsqueeze(0).repeat(num_envs, 1)
            
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
                pass
                
            # Drive arm (100 steps)
            dones = torch.zeros(num_envs, dtype=torch.bool, device=device)
            step = 0
            while not dones.all() and step <= 100:
                with torch.no_grad():
                    obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
                    actions = policy.act(obs_tensor, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(actions)
                dones = (dones | terminated.squeeze(-1) | truncated.squeeze(-1)
                         if terminated.dim() > 1 else dones | terminated | truncated)
                step += 1
                
            env.sim.render()
            
            # Position camera array on arc
            angles_deg = np.linspace(0, 45, num_envs)
            env_passed = np.zeros(num_envs, dtype=bool)
            final_lacp_9 = default_lacp
            
            for i in range(num_envs):
                detector = detectors[i]
                if detector is None: continue
                detector._update_lines()
                # Calculate bounding box center of active markers
                unique_markers = {}
                for info in detector.ray_info:
                    path = info['end_path']
                    pos = info['end_pos']
                    if pos is not None:
                        unique_markers[path] = pos
                
                if unique_markers:
                    pts_m = list(unique_markers.values())
                    pts_m = np.array(pts_m)
                    min_coords = np.min(pts_m, axis=0)
                    max_coords = np.max(pts_m, axis=0)
                    centroid = (min_coords + max_coords) / 2.0
                else:
                    centroid = None
                
                theta_rad = np.deg2rad(angles_deg[i])
                ndi_path = f"/World/envs/env_{i}/Root/NDI"
                
                if centroid is not None:
                    LACP = centroid
                    if i == 9:
                        final_lacp_9 = centroid
                        
                    # Arc math
                    ndi_world_x = LACP[0] + R_h * np.sin(theta_rad)
                    ndi_world_y = LACP[1] - R_h * np.cos(theta_rad)
                    ndi_world_z = LACP[2] + H
                    ndi_world_pos = np.array([ndi_world_x, ndi_world_y, ndi_world_z])
                    
                    quat_world = look_at_quaternion_world(ndi_world_pos, LACP)
                    
                    # Convert to local
                    parent_path = ndi_path.rsplit("/", 1)[0]
                    parent_prim = stage.GetPrimAtPath(parent_path)
                    parent_matrix = Gf.Matrix4d(1.0)
                    if parent_prim.IsValid():
                        parent_xform = UsdGeom.Xformable(parent_prim)
                        parent_matrix = parent_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                        
                    q_parent = parent_matrix.ExtractRotation().GetQuaternion()
                    gf_quat_world = Gf.Quaternion(float(quat_world[0]), Gf.Vec3d(float(quat_world[1]), float(quat_world[2]), float(quat_world[3])))
                    q_local = q_parent.GetInverse() * gf_quat_world
                    ndi_local_pos = np.array(parent_matrix.GetInverse().Transform(Gf.Vec3d(*ndi_world_pos)))
                    
                    set_prim_transform_orient(stage, ndi_path, ndi_local_pos, q_local)
                    
            env.sim.render()
            
            # Raycast Check
            for i in range(num_envs):
                detector = detectors[i]
                if detector is None: continue
                detector.update_detection(env.sim, debug_enabled=True, error_only_log=True)
                
                group_visible_balls = {m: 0 for m in SUPPORTED_MARKERS}
                group_total_balls = {m: 0 for m in SUPPORTED_MARKERS}
                
                for r_info in detector.raycast_results:
                    group = r_info['end_path'].split("/")[-1].split("_")[0].upper()
                    if group not in SUPPORTED_MARKERS: continue
                    group_total_balls[group] += 1
                    if r_info.get('is_target_hit', False):
                        group_visible_balls[group] += 1
                        
                env_visibility_groups = {}
                for group in SUPPORTED_MARKERS:
                    vis_count = group_visible_balls[group]
                    total = group_total_balls[group]
                    env_visibility_groups[group] = True if total > 0 and (vis_count == total) else False
                    
                env_passed[i] = all(env_visibility_groups.values())
                
            hit_rate = int(np.sum(env_passed))
            
            # Save precomputed NDI metrics for this candidate in local frame coordinates
            is_fallback = np.array_equal(final_lacp_9, default_lacp)
            lacp_offset = env_origins_np[0] if is_fallback else env_origins_np[9]
            final_lacp_local = final_lacp_9 - lacp_offset
            
            precomputed_db["blocks"][block_id]["candidates"][f"{c_x:.1f}"] = {
                "lacp": final_lacp_local.tolist(),
                "hit_rate": hit_rate,
                "angle_passes": [1 if p else 0 for p in env_passed]
            }
            
    # Save offline database
    output_json_path = os.path.abspath(os.path.join("scripts", "isaaclab_ws", "reachability_map", "ndi_precomputed_database.json"))
    with open(output_json_path, "w") as f:
        json.dump(precomputed_db, f, indent=4)
        
    print(f"\n[OK] Precomputed NDI Database successfully saved to:\n     {output_json_path}")
    
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
