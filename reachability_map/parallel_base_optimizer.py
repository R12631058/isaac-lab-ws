"""
parallel_base_optimizer.py

Multi-environment parallel evaluation of TM5 robot base positions.
It runs a 2D grid search over X and Y base offsets and evaluates the success rate 
and manipulability index for 70 waypoints concurrently across all environments.
"""
import argparse
import math
import time
import csv
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Parallel Base Optimization")
parser.add_argument("--max_waypoints", type=int, default=70)
parser.add_argument("--pos_threshold", type=float, default=0.0015,
                    help="Position error threshold (meters)")
parser.add_argument("--time_per_waypoint", type=float, default=4.0,
                    help="Max time allowed to reach each waypoint (seconds)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Default to headless mode for optimization (override with --no-headless)
if not hasattr(args_cli, 'headless') or args_cli.headless is None:
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.usd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_extension_linkout_cone_orientation_cfg import TM5ExtensionLinkOutFanOrientationEnvCfg

EXTENSION_LINK_OUT_JOINT_POS = {
    "joint_1": -1.48178,
    "joint_2":  0.75747,
    "joint_3":  1.00356,
    "joint_4":  1.36834,
    "joint_5":  0.0,
    "joint_6": -1.60570,
}

JOINT_CLAMPS = {
    0: (-3.14, -0.1),
    1: (-0.3,   2.5),
    2: (-0.3,   2.5),
    3: (-0.5,   3.14),
    4: (-1.57,  1.57),
    5: (-3.14,  0.5),
}

def get_waypoint_positions(stage, max_count):
    from pxr import Usd, UsdGeom
    waypoints = []
    usd_path = r"C:\Nick\surgery_team\surgery_team\USD\animation\isaaclab_multi_env.usd"
    source_stage = Usd.Stage.Open(usd_path)
    if not source_stage:
        return waypoints
    for i in range(1, max_count + 1):
        name = f"waypoint_{i:02d}"
        path = f"/World/waypoint/{name}"
        prim = source_stage.GetPrimAtPath(path)
        if not prim.IsValid():
            prim = source_stage.GetPrimAtPath(f"/Root/waypoint/{name}")
        if prim.IsValid():
            xform = UsdGeom.Xformable(prim)
            world_transform = xform.ComputeLocalToWorldTransform(0.0)
            translation = world_transform.ExtractTranslation()
            waypoints.append({
                "name": name,
                "pos": np.array(translation)
            })
    return waypoints

def main():
    t_start = time.time()
    
    # ── Grid Search Definition ──
    x_vals = np.linspace(0.1, 0.3, 3)  # 3 points
    y_vals = np.linspace(0, 0.1, 2)  # 2 points
    X, Y = np.meshgrid(x_vals, y_vals)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    num_envs = len(X_flat)  # 28
    
    print(f"\n[INIT] Starting Parallel Simulation with {num_envs} environments.")
    print(f"[INIT] Grid Search: X={x_vals[0]:.2f}~{x_vals[-1]:.2f}, Y={y_vals[0]:.2f}~{y_vals[-1]:.2f}\n")

    # ── Build environment ──
    env_cfg = TM5ExtensionLinkOutFanOrientationEnvCfg()
    env_cfg.scene.num_envs = num_envs
    
    if hasattr(env_cfg.commands, "ee_pose"):
        env_cfg.commands.ee_pose.resampling_time_range = (1e10, 1e10)
        env_cfg.commands.ee_pose.debug_vis = False
        
    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.device
    
    robot = env.scene["robot"]
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["joint_[1-6]"], body_names=["needle_tip"])
    robot_entity_cfg.resolve(env.scene)
    
    ik_joint_ids = robot_entity_cfg.joint_ids
    ik_body_idx = robot_entity_cfg.body_ids[0]
    ik_ee_jacobi_idx = ik_body_idx - 1 if robot.is_fixed_base else ik_body_idx
    jacobian_col_ids = ik_joint_ids if robot.is_fixed_base else [j + 6 for j in ik_joint_ids]
        
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": 0.01})
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=env.num_envs, device=env.device)
    
    env.reset()
    sim_dt = env.sim.get_physics_dt()
    
    # ── Apply Base Grid Positions ──
    # X_flat[i], Y_flat[i] = desired position of `robotarm_base` in Isaac Sim coordinates.
    # But write_root_pose_to_sim() controls `tm5_700` (the articulation root), NOT
    # `robotarm_base`. There is a FIXED offset T between them:
    #
    #   tm5_700_local = robotarm_base_pos + T
    #
    # We must account for T when setting the physics position.
    
    env_origins = env.scene.env_origins.clone()
    default_root_quat_w = robot.data.root_quat_w.clone()
    
    # robotarm_base default position in Isaac Sim (user-confirmed)
    ROBOTARM_BASE_DEFAULT = np.array([0.2, 0.0, 0.0])
    
    # tm5_700 default local position (read from physics after env.reset())
    tm5_700_default_local = (robot.data.root_pos_w[0] - env_origins[0]).cpu().numpy()
    
    # T = fixed offset from robotarm_base to tm5_700 (from USD hierarchy)
    T = tm5_700_default_local - ROBOTARM_BASE_DEFAULT
    print(f"[INFO] robotarm_base default: ({ROBOTARM_BASE_DEFAULT[0]:.4f}, {ROBOTARM_BASE_DEFAULT[1]:.4f}, {ROBOTARM_BASE_DEFAULT[2]:.4f})")
    print(f"[INFO] tm5_700 default local:  ({tm5_700_default_local[0]:.4f}, {tm5_700_default_local[1]:.4f}, {tm5_700_default_local[2]:.4f})")
    print(f"[INFO] T (base→root offset):   ({T[0]:.4f}, {T[1]:.4f}, {T[2]:.4f})")
    
    # 1) Compute correct tm5_700 world positions for each env
    new_root_pos_w = robot.data.root_pos_w.clone()
    for i in range(num_envs):
        # desired robotarm_base = (X_flat[i], Y_flat[i], 0)
        # → tm5_700 should be at desired_base + T
        new_root_pos_w[i, 0] = env_origins[i, 0] + X_flat[i] + T[0]
        new_root_pos_w[i, 1] = env_origins[i, 1] + Y_flat[i] + T[1]
        new_root_pos_w[i, 2] = env_origins[i, 2] + T[2]
    
    # 2) Update physics (articulation root) — single call to avoid GPU API errors
    robot.write_root_pose_to_sim(torch.cat([new_root_pos_w, default_root_quat_w], dim=-1))
    
    # Note: USD Xform modification of robotarm_base is SKIPPED because it triggers
    # "PxArticulationLink::setGlobalPose() illegal with eENABLE_DIRECT_GPU_API" errors.
    # The physics positions are correct via write_root_pose_to_sim. Visual mismatch
    # only matters in non-headless mode.
    
    # ── Initial Joint Positions + Settle ──
    init_joint_pos = torch.tensor([[EXTENSION_LINK_OUT_JOINT_POS[f"joint_{j}"] for j in range(1, 7)]], device=device).repeat(num_envs, 1)
    robot.write_joint_state_to_sim(init_joint_pos, torch.zeros_like(init_joint_pos))
    robot.set_joint_position_target(init_joint_pos, joint_ids=ik_joint_ids)
    env.scene.write_data_to_sim()
    env.sim.step()
    env.scene.update(sim_dt)
    
    # Settle: only set joint targets (root pose is already correct)
    for _ in range(50):
        robot.set_joint_position_target(init_joint_pos, joint_ids=ik_joint_ids)
        env.scene.write_data_to_sim()
        env.sim.step()
        env.scene.update(sim_dt)
    
    # Verify: compute actual robotarm_base position = tm5_700_pos - T
    actual_tm5 = robot.data.root_pos_w - env_origins
    print(f"\n[VERIFY] Positions after settle:")
    for i in range(num_envs):
        tm5 = actual_tm5[i].cpu().numpy()
        base_actual = tm5 - T  # infer robotarm_base from tm5_700
        err = ((base_actual[0] - X_flat[i])**2 + (base_actual[1] - Y_flat[i])**2)**0.5
        status = "✓" if err < 0.01 else "✗"
        print(f"  env_{i:02d}: robotarm_base=({base_actual[0]:.4f}, {base_actual[1]:.4f}) "
              f"| target=({X_flat[i]:.4f}, {Y_flat[i]:.4f}) | err={err*1000:.1f}mm {status}")
        
    initial_ee_quat_w = robot.data.body_quat_w[0, ik_body_idx].clone().unsqueeze(0)
    
    # ── Find Waypoints ──
    stage = omni.usd.get_context().get_stage()
    waypoints = get_waypoint_positions(stage, args_cli.max_waypoints)
    
    if not waypoints:
        print("[ERROR] No waypoints found.")
        env.close()
        simulation_app.close()
        return

    # Serpentine sweep: alternate direction per waypoint to minimize angular travel
    # Even WPs: -30 → -15 → 0 → +15 → +30  (forward sweep)
    # Odd  WPs: +30 → +15 → 0 → -15 → -30  (reverse sweep)
    # Angular travel: ~60°/WP vs ~180°/WP for the old 0,-15,+15,-30,+30 order
    ANGLES_FWD = [-30.0, -15.0,  0.0, 15.0, 30.0]
    ANGLES_REV = [ 30.0,  15.0,  0.0, -15.0, -30.0]

    from isaaclab.utils.math import quat_mul
    tasks = []
    for idx, wp in enumerate(waypoints):
        angle_list = ANGLES_FWD if idx % 2 == 0 else ANGLES_REV
        for angle_deg in angle_list:
            half_rad = math.radians(angle_deg) / 2.0
            q_rot_x = torch.tensor([[math.cos(half_rad), math.sin(half_rad), 0.0, 0.0]], device=device)
            target_q = quat_mul(q_rot_x, initial_ee_quat_w)
            tasks.append({
                "wp": wp,
                "angle": angle_deg,
                "target_quat": target_q
            })

    # ── Track & Evaluate ──
    print(f"{'='*60}\n  Starting Parallel Waypoint Tracking (Sync Mode)\n  Testing {len(tasks)} target configurations ({len(waypoints)} waypoints * 5 angles)\n{'='*60}")
    
    success_counts = np.zeros(num_envs)
    manip_sum = np.zeros(num_envs)
    manip_min = np.full(num_envs, 1e9)
    env_wp_records = [[] for _ in range(num_envs)]  # Track wp details for each env
    
    max_steps = int(args_cli.time_per_waypoint / sim_dt)
    
    # 建立即時備份的 CSV 防止當機遺失
    import os
    from datetime import datetime
    output_dir = os.path.join("scripts", "isaaclab_ws", "reachability_map", "heatmap")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"X{x_vals[0]:.2f}-{x_vals[-1]:.2f}_Y{y_vals[0]:.2f}-{y_vals[-1]:.2f}_{timestamp}"
    
    backup_csv = os.path.join(output_dir, f"backup_live_records_{run_tag}.csv")
    with open(backup_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Env_ID", "Waypoint", "Angle_X", "Pos_X", "Pos_Y", "Pos_Z", "Error_mm", "Manipulability_Index"])
    
    for task_idx, task in enumerate(tasks):
        wp = task["wp"]
        angle_deg = task["angle"]
        target_pos_local = wp["pos"]
        
        # Waypoint positions from the USD are in env_0's local frame.
        # For each environment, shift by its env_origin.
        target_pos_w = torch.tensor(target_pos_local, dtype=torch.float32, device=device).unsqueeze(0).repeat(num_envs, 1)
        
        for i in range(num_envs):
            target_pos_w[i, 0] += env_origins[i, 0]
            target_pos_w[i, 1] += env_origins[i, 1]
            
        target_quat_w = task["target_quat"].repeat(num_envs, 1)
        cartesian_integral = torch.zeros(num_envs, 3, device=device)
        
        print(f"  [{task_idx+1:03d}/{len(tasks):03d}] {wp['name']} @ {angle_deg}°... ", end="", flush=True)
        
        for step in range(max_steps):
            if not simulation_app.is_running(): break
                
            ee_pos_w = robot.data.body_pos_w[:, ik_body_idx]
            ee_quat_w = robot.data.body_quat_w[:, ik_body_idx]
            dist = torch.norm(ee_pos_w - target_pos_w, dim=-1)
            
            close_mask = dist < 0.03
            cartesian_integral[close_mask] += (target_pos_w[close_mask] - ee_pos_w[close_mask]) * 0.02
            cartesian_integral = torch.clamp(cartesian_integral, -0.05, 0.05)
            cartesian_integral[~close_mask] = 0.0
            
            jacobian = robot.root_physx_view.get_jacobians()[:, ik_ee_jacobi_idx, :, jacobian_col_ids]
            joint_pos = robot.data.joint_pos[:, ik_joint_ids]
            
            boosted_target_pos = target_pos_w + cartesian_integral
            ik_command = torch.cat([boosted_target_pos, target_quat_w], dim=-1)
            diff_ik_controller.set_command(ik_command, ee_quat=ee_quat_w)
            joint_pos_des = diff_ik_controller.compute(ee_pos_w, ee_quat_w, jacobian, joint_pos)
            
            for j_idx, (j_min, j_max) in JOINT_CLAMPS.items():
                joint_pos_des[:, j_idx] = torch.clamp(joint_pos_des[:, j_idx], min=j_min, max=j_max)
                
            robot.set_joint_position_target(joint_pos_des, joint_ids=ik_joint_ids)
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(sim_dt)
            
            if torch.all(dist < args_cli.pos_threshold):
                break
                
        # End of tracking time for this task
        ee_pos_w = robot.data.body_pos_w[:, ik_body_idx]
        dist = torch.norm(ee_pos_w - target_pos_w, dim=-1)
        success_mask = dist <= args_cli.pos_threshold
        
        jacobian = robot.root_physx_view.get_jacobians()[:, ik_ee_jacobi_idx, :, jacobian_col_ids]
        manip = torch.abs(torch.linalg.det(jacobian))
        
        s_count = 0
        for i in range(num_envs):
            is_success = success_mask[i].item()
            m_val = manip[i].item()
            if is_success:
                success_counts[i] += 1
                manip_sum[i] += m_val
                if m_val < manip_min[i]:
                    manip_min[i] = m_val
                s_count += 1
                
            env_wp_records[i].append({
                "Waypoint": wp['name'],
                "Angle_X": angle_deg,
                "Pos_X": target_pos_local[0],
                "Pos_Y": target_pos_local[1],
                "Pos_Z": target_pos_local[2],
                "Error_mm": dist[i].item() * 1000.0,
                "Manipulability_Index": m_val if is_success else 0.0
            })
            
            # 即時寫入每一步結果，避免當機遺失
            with open(backup_csv, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    i, wp['name'], angle_deg, 
                    target_pos_local[0], target_pos_local[1], target_pos_local[2], 
                    dist[i].item() * 1000.0, m_val if is_success else 0.0
                ])
                
        print(f"({s_count}/{num_envs} OK)")

    # ── Compile & Save Results ──
    print(f"\n{'='*60}")
    print("  Results Summary")
    print(f"{'='*60}")
    results = []
    
    for i in range(num_envs):
        avg_manip = manip_sum[i] / success_counts[i] if success_counts[i] > 0 else 0.0
        min_manip = manip_min[i] if success_counts[i] > 0 else 0.0
        
        # ─── Distinguish RI vs MI vs AMI ───
        # RI  (Reachability Index): % of successful orientations per waypoint
        # MI  (Manipulability Index): det(J) at a single joint config
        # AMI (Average Manipulability Index): avg(MI) of successful orientations per waypoint
        # c(t,k): The dexterity cost = AMI per waypoint
        
        wp_success = {}          # Count of successful orientations
        wp_count = {}            # Total orientations attempted
        wp_manip_sum = {}        # Sum of MI values for successful orientations
        
        for rec in env_wp_records[i]:
            wp = rec["Waypoint"]
            is_succ = 1.0 if rec["Manipulability_Index"] > 0.0 else 0.0
            wp_success[wp] = wp_success.get(wp, 0.0) + is_succ
            wp_count[wp] = wp_count.get(wp, 0) + 1
            if is_succ > 0.0:
                wp_manip_sum[wp] = wp_manip_sum.get(wp, 0.0) + rec["Manipulability_Index"]
        
        # Calculate RI and c(t,k) (AMI) per waypoint
        ri_list = []
        c_values = []  # c(t,k) = AMI per waypoint
        for wp, count in wp_count.items():
            ri_k = wp_success[wp] / count  # RI for this waypoint
            ri_list.append(ri_k)
            
            # c(t,k) = AMI per waypoint
            successful_count = wp_success[wp]
            if successful_count > 0.0:
                ami_k = wp_manip_sum[wp] / successful_count
            else:
                ami_k = 0.0
            c_values.append(ami_k)
            
        cs = success_counts[i] / len(tasks) if len(tasks) > 0 else 0.0
        sr_min = 1.0 if (len(ri_list) > 0 and min(ri_list) > 0.0) else 0.0
        prod_c = float(np.prod(c_values)) if len(c_values) > 0 else 0.0
        fg_score = sr_min * prod_c * cs
        
        results.append({
            "env_id": i,
            "base_x": round(X_flat[i], 3),
            "base_y": round(Y_flat[i], 3),
            "success_count": int(success_counts[i]),
            "success_rate": round(cs, 3),
            "avg_manip": float(avg_manip),
            "min_manip": float(min_manip),
            "sr_min": int(sr_min),
            "prod_c": float(prod_c),  # ∏ c(t,k) = ∏ AMI
            "fg_score": float(fg_score)
        })

    # Find the best env index based on fg_score, then avg_manip
    best_env_idx = -1
    best_value = (-1.0, -1.0)
    for res in results:
        val = (res["fg_score"], res["avg_manip"])
        if val > best_value:
            best_value = val
            best_env_idx = res["env_id"]
            
    for res in results:
        i = res["env_id"]
        mark = " 👑" if i == best_env_idx else ""
        # Line 1: identification + reachability
        print(f"  env_{i:02d} X=({res['base_x']:.2f},{res['base_y']:.2f}) "
              f"Succ:{res['success_count']:3d}/{len(tasks)} "
              f"Sr_min:{res['sr_min']} cs:{res['success_rate']:.3f}")
        # Line 2: manipulability metrics (∏AMI always visible)
        print(f"         "
              f"∏AMI_k={res['prod_c']:.4e}  "
              f"Avg_w={res['avg_manip']:.4e}  "
              f"fg={res['fg_score']:.4e}{mark}")
        
    import os
    from datetime import datetime
    output_dir = os.path.join("scripts", "isaaclab_ws", "reachability_map", "heatmap")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"X{x_vals[0]:.2f}-{x_vals[-1]:.2f}_Y{y_vals[0]:.2f}-{y_vals[-1]:.2f}_{timestamp}"
    
    csv_file = os.path.join(output_dir, f"optimization_results_{run_tag}.csv")
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[INFO] Saved metrics to {csv_file}")
    
    # ── Generate Heatmap ──
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        
        pivot_success = df.pivot(index="base_y", columns="base_x", values="success_count")
        pivot_manip = df.pivot(index="base_y", columns="base_x", values="avg_manip")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Success Heatmap
        im0 = axes[0].imshow(pivot_success.values, cmap="YlGnBu", origin='lower', aspect='auto')
        axes[0].set_xticks(np.arange(len(pivot_success.columns)))
        axes[0].set_yticks(np.arange(len(pivot_success.index)))
        axes[0].set_xticklabels([f"{v:.2f}" for v in pivot_success.columns])
        axes[0].set_yticklabels([f"{v:.2f}" for v in pivot_success.index])
        axes[0].set_xlabel("Base X (m)")
        axes[0].set_ylabel("Base Y (m)")
        axes[0].set_title(f"Success Count (out of {len(tasks)})")
        fig.colorbar(im0, ax=axes[0])
        for i in range(len(pivot_success.index)):
            for j in range(len(pivot_success.columns)):
                axes[0].text(j, i, f"{pivot_success.values[i, j]:.0f}", ha="center", va="center", color="black", fontsize=9)
        
        # Manipulability Heatmap
        im1 = axes[1].imshow(pivot_manip.values, cmap="viridis", origin='lower', aspect='auto')
        axes[1].set_xticks(np.arange(len(pivot_manip.columns)))
        axes[1].set_yticks(np.arange(len(pivot_manip.index)))
        axes[1].set_xticklabels([f"{v:.2f}" for v in pivot_manip.columns])
        axes[1].set_yticklabels([f"{v:.2f}" for v in pivot_manip.index])
        axes[1].set_xlabel("Base X (m)")
        axes[1].set_ylabel("Base Y (m)")
        axes[1].set_title("Average Manipulability Index")
        fig.colorbar(im1, ax=axes[1])
        for i in range(len(pivot_manip.index)):
            for j in range(len(pivot_manip.columns)):
                val = pivot_manip.values[i, j]
                axes[1].text(j, i, f"{val:.4f}", ha="center", va="center", color="white", fontsize=8)
                
        plt.suptitle("TM5 Robot Base Position Optimization", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plot_file = os.path.join(output_dir, f"manipulability_heatmap_{run_tag}.png")
        plt.savefig(plot_file, dpi=150)
        print(f"[INFO] Saved heatmap visualization to {plot_file}")
        
    except Exception as e:
        print(f"[ERROR] Failed to generate heatmap plot: {e}")

    # ── Generate RASSCMAP for All Environments ──
    try:
        from mpl_toolkits.mplot3d import Axes3D
        print(f"\n[INFO] Generating 3D RASSCMAP for all {num_envs} Base Positions...")
        
        base_rasscmap_dir = r"C:\Users\RMML\IsaacLab\scripts\isaaclab_ws\reachability_map\rasscmaps"
        # 建立一個以本次實驗專屬標籤 run_tag 為名的子資料夾，將大量圖片分類裝好
        rasscmap_dir = os.path.join(base_rasscmap_dir, run_tag)
        os.makedirs(rasscmap_dir, exist_ok=True)

        for res in results:
            env_id = res["env_id"]
            base_x = res["base_x"]
            base_y = res["base_y"]
            
            env_df = pd.DataFrame(env_wp_records[env_id])
            if not env_df.empty:
                # 重新計算 Reachability Index (RI): 成功到達的方向數 / 測試的總方向數
                # 我們的腳本中，若成功 'Manipulability_Index' 會大於 0.0
                env_df['Is_Success'] = (env_df['Manipulability_Index'] > 0.0).astype(float)
                
                df_gp = env_df.groupby('Waypoint').agg({
                    'Pos_X': 'mean',
                    'Pos_Y': 'mean',
                    'Pos_Z': 'mean',
                    'Is_Success': 'mean' # 這即是 RI: 範圍 0.0 ~ 1.0
                }).reset_index()

                df_gp.rename(columns={'Is_Success': 'RI'}, inplace=True)
                df_gp = df_gp.sort_values('Waypoint')

                fig2 = plt.figure(figsize=(10, 8))
                ax3d = fig2.add_subplot(111, projection='3d')
                
                ax3d.plot(df_gp['Pos_X'], df_gp['Pos_Y'], df_gp['Pos_Z'], 
                        color='gray', linestyle='dashed', linewidth=2, label='Insertion Path')
                
                import matplotlib.cm as cm
                # 使用 jet_r HSV 漸層 (0為紅色 -> 橘 -> 黃 -> 綠 -> 天藍 -> 1為深藍)
                cmap_rasscmap = cm.jet_r
                
                # 強制映射標準介於 0.0 到 1.0 (Fixed Scale)
                sc = ax3d.scatter(df_gp['Pos_X'], df_gp['Pos_Y'], df_gp['Pos_Z'], 
                                c=df_gp['RI'], cmap=cmap_rasscmap, vmin=0.0, vmax=1.0, s=100, label='Waypoints')
                plt.colorbar(sc, label='Reachability Index (RI)')
                ax3d.set_title(f"3D RASSCMAP (Env {env_id} Base: X={base_x:.2f}, Y={base_y:.2f})")
                ax3d.set_xlabel('X (m)')
                ax3d.set_ylabel('Y (m)')
                ax3d.set_zlabel('Z (m)')
                ax3d.legend()
                
                # 調整 3D 視角 (elev: 上下仰角, azim: 左右旋轉角)
                # 預設大約為 elev=30, azim=-60。這裡設為一個新的角度作為範例
                ax3d.view_init(elev=20, azim= 45)
                
                vis_path = os.path.join(rasscmap_dir, f"rasscmap_3d_Env{env_id}_X{base_x:.2f}_Y{base_y:.2f}_{run_tag}.png")
                plt.savefig(vis_path)
                plt.close(fig2)  # Close the figure to free memory!
                
        print(f"[INFO] Saved all {num_envs} RASSCMAP visualizations to: {os.path.abspath(rasscmap_dir)}")

    except Exception as e:
        print(f"[ERROR] Failed to generate RASSCMAP visualization: {e}")

    elapsed = time.time() - t_start
    minutes, seconds = divmod(elapsed, 60)
    print(f"\n[DONE] Total elapsed time: {int(minutes)}m {seconds:.1f}s")
    
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
