"""
parallel_base_optimizer_linear.py

Phase 1a: Single-block optimizer with 1D linear robot base grid search.

Changes vs parallel_base_optimizer.py:
  - 1D linear X grid: X = linspace(0.1, 1.5, 15), Y = 0.0 fixed
  - --block_center X Y Z : offset applied to all waypoint positions
  - --block_id N         : for output file naming
  - Outputs:
      heatmap/blocks/optimal_block_{N}.json   <- only optimal env
      heatmap/blocks/backup_live_block{N}_*.csv
      heatmap/blocks/optimization_results_block{N}_*.csv
      heatmap/blocks/rasscmap_optimal_block{N}_*.png  <- only optimal env

Usage:
    isaaclab.bat -p scripts\\isaaclab_ws\\reachability_map\\parallel_base_optimizer_linear.py ^
        --block_center 0.0 0.0 0.0 --block_id 0
"""
import argparse
import math
import time
import csv
import json
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Linear 1D Base Optimizer (per block)")
parser.add_argument("--max_waypoints",    type=int,   default=70)
parser.add_argument("--pos_threshold",    type=float, default=0.0015)
parser.add_argument("--time_per_waypoint",type=float, default=4.0)
parser.add_argument("--block_center",     type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    metavar=("BCX", "BCY", "BCZ"),
                    help="Translate offset for /Root/waypoint (m). "
                         "Block 0=(0,0,0), Block 1=(0.1,0,0), etc.")
parser.add_argument("--block_id",         type=int,   default=0,
                    help="Block index, used for output file naming.")
parser.add_argument("--x_min",            type=float, default=0.1,
                    help="Start of robot base X search (m). Default=0.1")
parser.add_argument("--x_max",            type=float, default=1.5,
                    help="End of robot base X search (m). Default=1.5")
parser.add_argument("--x_steps",          type=int,   default=15,
                    help="Number of X grid points. Default=15")
parser.add_argument("--y_fixed",          type=float, default=0.0,
                    help="Fixed robot base Y (m). Default=0.0")
parser.add_argument("--prune_threshold",  type=int,   default=10,
                    help="Prune env after this many fully-failed waypoints. "
                         "A waypoint 'fully fails' if all 5 angles fail. Default=10")
parser.add_argument("--no-headless", action="store_true", 
                    help="Force GUI mode (disable default headless)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Default to headless unless --no-headless is passed
if args_cli.no_headless:
    args_cli.headless = False
elif getattr(args_cli, "headless", False) is False:
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

USD_PATH = r"C:\Nick\surgery_team\surgery_team\USD\animation\isaaclab_multi_env.usd"


def get_waypoint_positions(max_count: int, block_offset: np.ndarray):
    """
    Read waypoints from source USD, then apply block_offset to all positions.
    block_offset = np.array([bcx, bcy, bcz]) ??the /Root/waypoint translate for this block.
    """
    from pxr import Usd, UsdGeom
    waypoints = []
    source_stage = Usd.Stage.Open(USD_PATH)
    if not source_stage:
        print(f"[ERROR] Cannot open USD: {USD_PATH}")
        return waypoints

    for i in range(1, max_count + 1):
        name = f"waypoint_{i:02d}"
        prim = source_stage.GetPrimAtPath(f"/World/waypoint/{name}")
        if not prim.IsValid():
            prim = source_stage.GetPrimAtPath(f"/Root/waypoint/{name}")
        if prim.IsValid():
            xform = UsdGeom.Xformable(prim)
            t = xform.ComputeLocalToWorldTransform(0.0).ExtractTranslation()
            # Apply block offset: shift entire waypoint cloud
            pos = np.array(t) + block_offset
            waypoints.append({"name": name, "pos": pos})

    print(f"[BLOCK {args_cli.block_id}] Loaded {len(waypoints)} waypoints "
          f"(offset: {block_offset})")
    return waypoints


def compute_waypoint_bounds(waypoints: list) -> dict:
    """Return axis-aligned bounding box of all waypoint positions."""
    pts = np.array([wp["pos"] for wp in waypoints])
    return {
        "x_min": float(pts[:, 0].min()), "x_max": float(pts[:, 0].max()),
        "y_min": float(pts[:, 1].min()), "y_max": float(pts[:, 1].max()),
        "z_min": float(pts[:, 2].min()), "z_max": float(pts[:, 2].max()),
    }


def main():
    t_start = time.time()

    block_id     = args_cli.block_id
    block_center = np.array(args_cli.block_center, dtype=np.float64)

    # ?? 1D linear grid ??
    x_vals   = np.linspace(args_cli.x_min, args_cli.x_max, args_cli.x_steps)
    y_fixed  = args_cli.y_fixed
    X_flat   = x_vals
    Y_flat   = np.full_like(x_vals, y_fixed)
    num_envs = len(X_flat)

    print(f"\n{'='*65}")
    print(f"  Parallel Base Optimizer ??Linear 1D Grid")
    print(f"  Block ID     : {block_id}")
    print(f"  Block center : {block_center}")
    print(f"  Robot base X : {x_vals[0]:.2f} ~ {x_vals[-1]:.2f}  ({num_envs} steps)")
    print(f"  Robot base Y : {y_fixed:.2f} (fixed)")
    print(f"{'='*65}\n")

    # ?? Build environment ??
    env_cfg = TM5ExtensionLinkOutFanOrientationEnvCfg()
    env_cfg.scene.num_envs = num_envs
    if hasattr(env_cfg.commands, "ee_pose"):
        env_cfg.commands.ee_pose.resampling_time_range = (1e10, 1e10)
        env_cfg.commands.ee_pose.debug_vis = False

    env    = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.device

    robot = env.scene["robot"]
    robot_entity_cfg = SceneEntityCfg(
        "robot", joint_names=["joint_[1-6]"], body_names=["needle_tip"]
    )
    robot_entity_cfg.resolve(env.scene)
    ik_joint_ids      = robot_entity_cfg.joint_ids
    ik_body_idx       = robot_entity_cfg.body_ids[0]
    ik_ee_jacobi_idx  = ik_body_idx - 1 if robot.is_fixed_base else ik_body_idx
    jacobian_col_ids  = (ik_joint_ids if robot.is_fixed_base
                         else [j + 6 for j in ik_joint_ids])

    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False,
        ik_method="dls", ik_params={"lambda_val": 0.01}
    )
    diff_ik_controller = DifferentialIKController(
        diff_ik_cfg, num_envs=env.num_envs, device=device
    )

    env.reset()
    sim_dt = env.sim.get_physics_dt()

    # ?? Apply base grid positions ??
    env_origins = env.scene.env_origins.clone()
    default_root_quat_w = robot.data.root_quat_w.clone()

    ROBOTARM_BASE_DEFAULT = np.array([0.2, 0.0, 0.0])
    tm5_700_default_local = (robot.data.root_pos_w[0] - env_origins[0]).cpu().numpy()
    T = tm5_700_default_local - ROBOTARM_BASE_DEFAULT

    print(f"[INFO] T (base?root offset): ({T[0]:.4f}, {T[1]:.4f}, {T[2]:.4f})")

    new_root_pos_w = robot.data.root_pos_w.clone()
    for i in range(num_envs):
        new_root_pos_w[i, 0] = env_origins[i, 0] + X_flat[i] + T[0]
        new_root_pos_w[i, 1] = env_origins[i, 1] + Y_flat[i] + T[1]
        new_root_pos_w[i, 2] = env_origins[i, 2] + T[2]

    robot.write_root_pose_to_sim(
        torch.cat([new_root_pos_w, default_root_quat_w], dim=-1)
    )

    # ?? Settle ??
    init_joint_pos = torch.tensor(
        [[EXTENSION_LINK_OUT_JOINT_POS[f"joint_{j}"] for j in range(1, 7)]],
        device=device
    ).repeat(num_envs, 1)
    robot.write_joint_state_to_sim(init_joint_pos, torch.zeros_like(init_joint_pos))
    robot.set_joint_position_target(init_joint_pos, joint_ids=ik_joint_ids)
    env.scene.write_data_to_sim(); env.sim.step(); env.scene.update(sim_dt)
    for _ in range(50):
        robot.set_joint_position_target(init_joint_pos, joint_ids=ik_joint_ids)
        env.scene.write_data_to_sim(); env.sim.step(); env.scene.update(sim_dt)

    initial_ee_quat_w = robot.data.body_quat_w[0, ik_body_idx].clone().unsqueeze(0)

    # ?? Waypoints (with block offset) ??
    waypoints = get_waypoint_positions(args_cli.max_waypoints, block_center)
    if not waypoints:
        print("[ERROR] No waypoints found.")
        env.close(); simulation_app.close(); return

    wp_bounds = compute_waypoint_bounds(waypoints)
    print(f"[BLOCK {block_id}] Waypoint bounds: "
          f"X[{wp_bounds['x_min']:.3f}, {wp_bounds['x_max']:.3f}] "
          f"Y[{wp_bounds['y_min']:.3f}, {wp_bounds['y_max']:.3f}] "
          f"Z[{wp_bounds['z_min']:.3f}, {wp_bounds['z_max']:.3f}]")

    # ?? Serpentine task expansion ??
    from isaaclab.utils.math import quat_mul
    ANGLES_FWD = [-30.0, -15.0, 0.0, 15.0, 30.0]
    ANGLES_REV = [ 30.0,  15.0, 0.0, -15.0, -30.0]
    tasks = []
    for idx, wp in enumerate(waypoints):
        for angle_deg in (ANGLES_FWD if idx % 2 == 0 else ANGLES_REV):
            half_rad = math.radians(angle_deg) / 2.0
            q_rot_x  = torch.tensor(
                [[math.cos(half_rad), math.sin(half_rad), 0.0, 0.0]], device=device
            )
            tasks.append({
                "wp":          wp,
                "angle":       angle_deg,
                "target_quat": quat_mul(q_rot_x, initial_ee_quat_w),
            })

    print(f"\n{'='*65}")
    print(f"  Tracking {len(tasks)} tasks  ({len(waypoints)} WPs ? 5 angles, serpentine)")
    print(f"{'='*65}")

    # ?? Output directories ??
    import os
    from datetime import datetime
    output_dir = os.path.join(
        "scripts", "isaaclab_ws", "reachability_map", "heatmap", "blocks"
    )
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag   = f"block{block_id}_{timestamp}"

    backup_csv_path = os.path.join(output_dir, f"backup_live_{run_tag}.csv")
    with open(backup_csv_path, mode='w', newline='') as f:
        csv.writer(f).writerow(
            ["Env_ID", "Waypoint", "Angle_X",
             "Pos_X", "Pos_Y", "Pos_Z", "Error_mm", "Manipulability_Index"]
        )

    # ?? Tracking loop ??
    success_counts  = np.zeros(num_envs)
    manip_sum       = np.zeros(num_envs)
    manip_min       = np.full(num_envs, 1e9)
    env_wp_records  = [[] for _ in range(num_envs)]
    max_steps       = int(args_cli.time_per_waypoint / sim_dt)

    # ?? Early pruning state ??
    pruned          = np.zeros(num_envs, dtype=bool)  # True = skip this env
    wp_fail_count   = np.zeros(num_envs, dtype=int)   # fully-failed WP count
    # Track per-waypoint success within current WP group (resets every 5 tasks)
    cur_wp_hits     = np.zeros(num_envs, dtype=int)   # successes in current WP
    prune_threshold = args_cli.prune_threshold
    num_active      = num_envs
    active_mask_t   = torch.ones(num_envs, dtype=torch.bool, device=device)

    for task_idx, task in enumerate(tasks):
        wp             = task["wp"]
        angle_deg      = task["angle"]
        target_pos_local = wp["pos"]

        # Reset per-waypoint hit tracker at the start of each new waypoint
        # (tasks are grouped: 5 consecutive angles per waypoint)
        if task_idx % 5 == 0:
            cur_wp_hits[:] = 0

        target_pos_w = torch.tensor(
            target_pos_local, dtype=torch.float32, device=device
        ).unsqueeze(0).repeat(num_envs, 1)
        for i in range(num_envs):
            target_pos_w[i, 0] += env_origins[i, 0]
            target_pos_w[i, 1] += env_origins[i, 1]

        target_quat_w      = task["target_quat"].repeat(num_envs, 1)
        cartesian_integral = torch.zeros(num_envs, 3, device=device)

        active_str = f"{num_active}/{num_envs}" if num_active < num_envs else f"{num_envs}"
        print(f"  [{task_idx+1:03d}/{len(tasks):03d}] {wp['name']} @ {angle_deg:+.0f}° "
              f"[active: {active_str}]... ", end="", flush=True)

        # If all envs are pruned, skip physics entirely
        if num_active == 0:
            print("(all pruned, skip)")
            # Still record zeros for consistency
            with open(backup_csv_path, mode='a', newline='') as f:
                w = csv.writer(f)
                for i in range(num_envs):
                    env_wp_records[i].append({
                        "Waypoint": wp["name"], "Angle_X": angle_deg,
                        "Pos_X": target_pos_local[0], "Pos_Y": target_pos_local[1],
                        "Pos_Z": target_pos_local[2],
                        "Error_mm": 9999.0, "Manipulability_Index": 0.0,
                    })
                    w.writerow([i, wp["name"], angle_deg,
                                target_pos_local[0], target_pos_local[1],
                                target_pos_local[2], 9999.0, 0.0])
            # Check waypoint boundary for pruning update
            if (task_idx + 1) % 5 == 0:
                cur_wp_hits[:] = 0
            continue

        for step in range(max_steps):
            if not simulation_app.is_running(): break

            ee_pos_w  = robot.data.body_pos_w[:, ik_body_idx]
            ee_quat_w = robot.data.body_quat_w[:, ik_body_idx]
            dist      = torch.norm(ee_pos_w - target_pos_w, dim=-1)

            close_mask = dist < 0.03
            cartesian_integral[close_mask]  += (
                target_pos_w[close_mask] - ee_pos_w[close_mask]
            ) * 0.02
            cartesian_integral  = torch.clamp(cartesian_integral, -0.05, 0.05)
            cartesian_integral[~close_mask] = 0.0

            jacobian      = robot.root_physx_view.get_jacobians()[
                :, ik_ee_jacobi_idx, :, jacobian_col_ids
            ]
            joint_pos     = robot.data.joint_pos[:, ik_joint_ids]
            ik_command    = torch.cat(
                [target_pos_w + cartesian_integral, target_quat_w], dim=-1
            )
            diff_ik_controller.set_command(ik_command, ee_quat=ee_quat_w)
            joint_pos_des = diff_ik_controller.compute(
                ee_pos_w, ee_quat_w, jacobian, joint_pos
            )
            for j_idx, (j_min, j_max) in JOINT_CLAMPS.items():
                joint_pos_des[:, j_idx] = torch.clamp(
                    joint_pos_des[:, j_idx], min=j_min, max=j_max
                )
            robot.set_joint_position_target(joint_pos_des, joint_ids=ik_joint_ids)
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(sim_dt)

            # Early break: only check ACTIVE (non-pruned) envs
            if torch.all(dist[active_mask_t] < args_cli.pos_threshold):
                break

        # ?? Record results ??
        ee_pos_w     = robot.data.body_pos_w[:, ik_body_idx]
        dist         = torch.norm(ee_pos_w - target_pos_w, dim=-1)
        success_mask = dist <= args_cli.pos_threshold
        jacobian     = robot.root_physx_view.get_jacobians()[
            :, ik_ee_jacobi_idx, :, jacobian_col_ids
        ]
        manip = torch.abs(torch.linalg.det(jacobian))

        s_count = 0
        with open(backup_csv_path, mode='a', newline='') as f:
            w = csv.writer(f)
            for i in range(num_envs):
                is_ok  = success_mask[i].item()
                m_val  = manip[i].item()
                if is_ok:
                    success_counts[i] += 1
                    manip_sum[i]      += m_val
                    if m_val < manip_min[i]:
                        manip_min[i]  = m_val
                    s_count += 1
                    if not pruned[i]:
                        cur_wp_hits[i] += 1
                env_wp_records[i].append({
                    "Waypoint":           wp["name"],
                    "Angle_X":            angle_deg,
                    "Pos_X":              target_pos_local[0],
                    "Pos_Y":              target_pos_local[1],
                    "Pos_Z":              target_pos_local[2],
                    "Error_mm":           dist[i].item() * 1000.0,
                    "Manipulability_Index": m_val if is_ok else 0.0,
                })
                w.writerow([
                    i, wp["name"], angle_deg,
                    target_pos_local[0], target_pos_local[1], target_pos_local[2],
                    dist[i].item() * 1000.0, m_val if is_ok else 0.0,
                ])

        print(f"({s_count}/{num_active} OK)")

        # ?? Pruning check: at waypoint boundary (every 5 tasks) ??
        if (task_idx + 1) % 5 == 0:
            wp_name = wp["name"]
            wp_idx  = (task_idx + 1) // 5  # which waypoint just finished (1-indexed)
            remaining_wps = len(waypoints) - wp_idx  # waypoints not yet tested

            # Check if any env currently has sr_min=1 (all waypoints reached so far)
            # Require the winner to have passed at least half of all waypoints to
            # avoid premature pruning of high-success-rate envs early in the run.
            MIN_WP_BEFORE_PRUNE = max(1, len(waypoints) // 2)
            any_sr_min_winner = any(
                wp_fail_count[i] == 0 and not pruned[i] and wp_idx >= MIN_WP_BEFORE_PRUNE
                for i in range(num_envs)
            )

            newly_pruned = []
            for i in range(num_envs):
                if pruned[i]:
                    continue
                if cur_wp_hits[i] == 0:
                    # This waypoint fully failed for env i
                    wp_fail_count[i] += 1

                    # Only prune if:
                    # 1. There EXISTS an env that has passed >= half WPs with 0 failures
                    # 2. AND this env has failed >= prune_threshold waypoints
                    # 3. AND even if this env succeeds every remaining WP, it can't beat sr_min=1
                    if any_sr_min_winner and wp_fail_count[i] >= prune_threshold:
                        pruned[i] = True
                        active_mask_t[i] = False
                        newly_pruned.append(i)

            if newly_pruned:
                num_active = int((~pruned).sum())
                print(f"    [PRUNED] env(s) {newly_pruned} after {wp_name} "
                      f"(fail_count >= {prune_threshold}, sr_min winner stable at wp {wp_idx}/{len(waypoints)}). "
                      f"Active: {num_active}/{num_envs}")
            elif not any_sr_min_winner:
                # No winner yet — report fail counts but don't prune
                worst = max(wp_fail_count[i] for i in range(num_envs) if not pruned[i])
                if worst > 0 and worst % 5 == 0:  # print every 5 failures
                    print(f"    [INFO] No sr_min=1 env yet (or winner not stable). "
                          f"Max fully-failed WPs: {worst}. Pruning suspended.")
            cur_wp_hits[:] = 0


    # ?? Compile results ??
    print(f"\n{'='*65}")
    print(f"  Block {block_id} ??Results Summary")
    print(f"{'='*65}")

    results = []
    for i in range(num_envs):
        avg_manip = manip_sum[i] / success_counts[i] if success_counts[i] > 0 else 0.0
        min_manip = manip_min[i] if success_counts[i] > 0 else 0.0

        wp_success, wp_count, wp_manip_sum = {}, {}, {}
        for rec in env_wp_records[i]:
            wp   = rec["Waypoint"]
            is_s = 1.0 if rec["Manipulability_Index"] > 0.0 else 0.0
            wp_success[wp]   = wp_success.get(wp, 0.0) + is_s
            wp_count[wp]     = wp_count.get(wp, 0) + 1
            if is_s:
                wp_manip_sum[wp] = wp_manip_sum.get(wp, 0.0) + rec["Manipulability_Index"]

        ri_list, c_values = [], []
        for wp, count in wp_count.items():
            ri_k = wp_success[wp] / count
            ri_list.append(ri_k)
            c_values.append(
                wp_manip_sum[wp] / wp_success[wp] if wp_success[wp] > 0 else 0.0
            )

        cs     = success_counts[i] / len(tasks) if tasks else 0.0
        sr_min = 1.0 if ri_list and min(ri_list) > 0.0 else 0.0
        prod_c = float(np.prod(c_values)) if c_values else 0.0
        fg     = sr_min * prod_c * cs

        results.append({
            "env_id":        i,
            "base_x":        round(X_flat[i], 4),
            "base_y":        round(Y_flat[i], 4),
            "success_count": int(success_counts[i]),
            "success_rate":  round(cs, 4),
            "avg_manip":     float(avg_manip),
            "min_manip":     float(min_manip),
            "sr_min":        int(sr_min),
            "prod_c":        float(prod_c),
            "fg_score":      float(fg),
        })

    # Best env: primary success_rate, secondary sr_min, tertiary fg_score
    best = max(results, key=lambda r: (r["success_rate"], r["sr_min"], r["fg_score"], r["avg_manip"]))
    best_idx = best["env_id"]

    for res in results:
        mark = " <<BEST>>" if res["env_id"] == best_idx else ""
        print(f"  env_{res['env_id']:02d}  X={res['base_x']:.3f}  "
              f"cs={res['success_rate']:.3f}  sr_min={res['sr_min']}  "
              f"fg={res['fg_score']:.4e}{mark}")

    # ?? Save optimization CSV ??
    opt_csv_path = os.path.join(output_dir, f"optimization_results_{run_tag}.csv")
    with open(opt_csv_path, mode='w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader(); w.writerows(results)
    print(f"\n[INFO] Optimization CSV ??{opt_csv_path}")

    # ?? Save optimal block JSON ??
    block_json = {
        "block_id":       block_id,
        "block_center":   block_center.tolist(),
        "wp_bounds":      wp_bounds,
        "optimal_env_id": best_idx,
        "optimal_base": {
            "base_x":       best["base_x"],
            "base_y":       best["base_y"],
            "fg_score":     best["fg_score"],
            "success_rate": best["success_rate"],
            "sr_min":       best["sr_min"],
            "avg_manip":    best["avg_manip"],
        },
        "backup_csv":   backup_csv_path,
        "opt_csv":      opt_csv_path,
        "timestamp":    timestamp,
    }
    json_path = os.path.join(output_dir, f"optimal_block_{block_id}.json")
    with open(json_path, "w") as f:
        json.dump(block_json, f, indent=2)
    print(f"[INFO] Optimal block JSON ??{json_path}")
    print(f"\n[RESULT] Block {block_id} optimal: "
          f"base_x={best['base_x']:.3f}  fg={best['fg_score']:.4e}  "
          f"cs={best['success_rate']:.1%}  sr_min={best['sr_min']}")

    # ?? 1D bar chart (success rate + fg_score) ??
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        x_labels = [f"{r['base_x']:.2f}" for r in results]
        x_pos    = np.arange(len(results))

        bars1 = ax1.bar(x_pos, [r["success_rate"] for r in results],
                        color="steelblue", alpha=0.8)
        ax1.axvline(best_idx, color="gold", linewidth=2, linestyle="--", label="Best")
        ax1.set_ylabel("Success Rate (cs)"); ax1.set_ylim(0, 1.05)
        ax1.set_title(f"Block {block_id} ??Robot Base X Search  "
                      f"(center: {block_center})"); ax1.legend()

        fg_vals = [r["fg_score"] for r in results]
        bars2 = ax2.bar(x_pos, fg_vals,
                        color=["gold" if r["env_id"] == best_idx else "coral"
                               for r in results], alpha=0.9)
        ax2.set_ylabel("fg_score"); ax2.set_xlabel("Robot base X (m)")
        ax2.set_xticks(x_pos); ax2.set_xticklabels(x_labels, rotation=45)
        ax2.set_title("fg_score = sr_min ? ?AMI ? cs")

        plt.tight_layout()
        chart_path = os.path.join(output_dir, f"bar_chart_{run_tag}.png")
        plt.savefig(chart_path, dpi=150); plt.close()
        print(f"[INFO] Bar chart ??{chart_path}")
    except Exception as e:
        print(f"[WARN] Chart failed: {e}")

    # ?? RASSCMAP for optimal env only ??
    try:
        import pandas as pd
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.cm as cm

        opt_records = env_wp_records[best_idx]
        df = pd.DataFrame(opt_records)
        if not df.empty:
            df["Is_Success"] = (df["Manipulability_Index"] > 0.0).astype(float)
            df_gp = (df.groupby("Waypoint")
                       .agg(Pos_X=("Pos_X","mean"), Pos_Y=("Pos_Y","mean"),
                            Pos_Z=("Pos_Z","mean"), RI=("Is_Success","mean"))
                       .reset_index().sort_values("Waypoint"))

            fig3 = plt.figure(figsize=(10, 8))
            ax3d = fig3.add_subplot(111, projection="3d")
            ax3d.plot(df_gp["Pos_X"], df_gp["Pos_Y"], df_gp["Pos_Z"],
                      color="gray", linestyle="dashed", linewidth=1.5)
            sc = ax3d.scatter(df_gp["Pos_X"], df_gp["Pos_Y"], df_gp["Pos_Z"],
                              c=df_gp["RI"], cmap=cm.jet_r,
                              vmin=0, vmax=1, s=100)
            plt.colorbar(sc, label="RI")
            ax3d.set_title(
                f"RASSCMAP ??Block {block_id}  "
                f"Optimal base X={best['base_x']:.3f} Y={best['base_y']:.3f}"
            )
            ax3d.set_xlabel("X (m)"); ax3d.set_ylabel("Y (m)"); ax3d.set_zlabel("Z (m)")
            ax3d.view_init(elev=20, azim=25)
            rasscmap_path = os.path.join(
                output_dir, f"rasscmap_optimal_{run_tag}.png"
            )
            plt.savefig(rasscmap_path, dpi=150); plt.close(fig3)
            print(f"[INFO] RASSCMAP ??{rasscmap_path}")
    except Exception as e:
        print(f"[WARN] RASSCMAP plot failed: {e}")

    elapsed = time.time() - t_start
    m, s = divmod(elapsed, 60)
    print(f"\n[DONE] Block {block_id} finished in {int(m)}m {s:.1f}s")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
