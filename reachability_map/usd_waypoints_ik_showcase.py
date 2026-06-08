"""
usd_waypoints_ik_showcase.py

Single-environment SHOWCASE using Differential IK to sequentially track manually placed 
waypoints in the USD file (e.g. /World/waypoint/waypoint_01 to waypoint_70).

The robot uses DifferentialIKController with needle_tip as the tracked body,
maintaining the initial downward posture and only updating target positions.

Usage:
    .\\isaaclab.bat -p scripts\isaaclab_ws\reachability_map\\usd_waypoints_ik_showcase.py
"""
import argparse
import math
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="USD Waypoints IK Showcase")
parser.add_argument("--max_waypoints", type=int, default=70,
                    help="Maximum number of waypoints to look for (waypoint_01 to waypoint_XX)")
parser.add_argument("--pos_threshold", type=float, default=0.001,
                    help="Position error threshold (meters) to consider a waypoint reached")
parser.add_argument("--wait_time", type=float, default=1.0,
                    help="Seconds to wait at each waypoint before moving to the next")
parser.add_argument("--visualize", action="store_true",
                    help="Run visualization (enabled by default unless --headless is used)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.usd
import torch
import numpy as np
from pxr import UsdGeom, Gf

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms, matrix_from_quat, quat_inv
from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_extension_linkout_cone_orientation_cfg import TM5ExtensionLinkOutFanOrientationEnvCfg

# True initial: -84.9°, 43.4°, 57.5°, 78.4°, 0.0°, -92.0°
EXTENSION_LINK_OUT_JOINT_POS = {
    "joint_1": -1.48178,   # -84.9°
    "joint_2":  0.75747,   #  43.4°
    "joint_3":  1.00356,   #  57.5°
    "joint_4":  1.36834,   #  78.4°
    "joint_5":  0.0,       #   0.0°
    "joint_6": -1.60570,   # -92.0°
}

JOINT_CLAMPS = {
    0: (-3.14, -0.1),    # joint_1: init=-84.9° ??must stay negative
    1: (-0.3,   2.5),    # joint_2: init=+43.4° ??must stay positive-ish
    2: (-0.3,   2.5),    # joint_3: init=+57.5° ??must stay positive-ish
    3: (-0.5,   3.14),   # joint_4: init=+78.4° ??must stay positive-ish
    4: (-1.57,  1.57),   # joint_5: init= 0.0°  ??near zero, allow both
    5: (-3.14,  0.5),    # joint_6: init=-92.0° ??must stay negative
}

def get_waypoint_positions(stage, max_count):
    """Scan for waypoints in the USD file and return their positions."""
    from pxr import Usd, UsdGeom
    waypoints = []
    
    print("\n[INFO] Scanning for waypoints in USD...")
    
    # User placed waypoints under /World/waypoint in the source USD.
    # Because defaultPrim is /Root, IsaacLab's payload does not load /World.
    # Therefore, we directly open the source USD to read their positions.
    usd_path = r"C:\Nick\surgery_team\surgery_team\USD\animation\isaaclab_multi_env.usd"
    source_stage = Usd.Stage.Open(usd_path)
    
    if not source_stage:
        print(f"[ERROR] Could not open source USD at {usd_path}")
        return waypoints

    for i in range(1, max_count + 1):
        name = f"waypoint_{i:02d}"
        
        # Check standard path requested by user
        path = f"/World/waypoint/{name}"
        prim = source_stage.GetPrimAtPath(path)
        
        # Fallback to check if they put it under Root just in case
        if not prim.IsValid():
            prim = source_stage.GetPrimAtPath(f"/Root/waypoint/{name}")
            
        if prim.IsValid():
            # Get position
            xform = UsdGeom.Xformable(prim)
            world_transform = xform.ComputeLocalToWorldTransform(0.0)
            translation = world_transform.ExtractTranslation()
            pos = np.array(translation)
            waypoints.append({
                "name": name,
                "prim_path": prim.GetPath().pathString,
                "pos": pos
            })
            print(f"  [Found] {name} at {prim.GetPath().pathString} -> Pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    print(f"[INFO] Total waypoints found: {len(waypoints)}\n")
    return waypoints

def main():
    env_cfg = TM5ExtensionLinkOutFanOrientationEnvCfg()
    env_cfg.scene.num_envs = 1
    
    # Disable command resampling
    if hasattr(env_cfg.commands, "ee_pose"):
        env_cfg.commands.ee_pose.resampling_time_range = (1e10, 1e10)
        env_cfg.commands.ee_pose.debug_vis = False
        
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    robot = env.scene["robot"]
    robot_entity_cfg = SceneEntityCfg(
        "robot", joint_names=["joint_[1-6]"], body_names=["needle_tip"]
    )
    robot_entity_cfg.resolve(env.scene)
    
    ik_joint_ids = robot_entity_cfg.joint_ids
    ik_body_idx = robot_entity_cfg.body_ids[0]
    ik_ee_jacobi_idx = ik_body_idx - 1 if robot.is_fixed_base else ik_body_idx

    if robot.is_fixed_base:
        jacobian_col_ids = ik_joint_ids
    else:
        jacobian_col_ids = [j + 6 for j in ik_joint_ids]
        
    # Setup Differential IK
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.01},
    )
    diff_ik_controller = DifferentialIKController(
        diff_ik_cfg, num_envs=env.num_envs, device=env.device
    )
    
    env.reset()
    sim_dt = env.sim.get_physics_dt()
    
    # Force initial joint positions
    device = env.device
    init_joint_pos = torch.tensor([[EXTENSION_LINK_OUT_JOINT_POS[f"joint_{i}"] for i in range(1, 7)]], device=device)
    robot.set_joint_position_target(init_joint_pos, joint_ids=ik_joint_ids)
    
    # Let it settle (uses PD control to move to the initial pose naturally, avoiding PhysX GPU write errors)
    for _ in range(50):
        robot.set_joint_position_target(init_joint_pos, joint_ids=ik_joint_ids)
        env.scene.write_data_to_sim()
        env.sim.step()
        env.scene.update(sim_dt)
        
    # Record the initial downward orientation
    initial_ee_quat_w = robot.data.body_quat_w[0, ik_body_idx].clone().unsqueeze(0)
    
    stage = omni.usd.get_context().get_stage()
    waypoints = get_waypoint_positions(stage, args_cli.max_waypoints)
    
    if not waypoints:
        print("[ERROR] No waypoints found in the USD. Exiting.")
        env.close()
        simulation_app.close()
        return

    # Serpentine sweep: alternate direction per waypoint to minimize angular travel
    # Even WPs: -30 → -15 → 0 → +15 → +30  (forward sweep)
    # Odd  WPs: +30 → +15 → 0 → -15 → -30  (reverse sweep)
    # Total angular travel: ~60°/WP  vs  ~180°/WP for the old 0,-15,+15,-30,+30 order
    ANGLES_FWD = [-30.0, -15.0, 0.0, 15.0, 30.0]
    ANGLES_REV = [ 30.0,  15.0, 0.0, -15.0, -30.0]

    from isaaclab.utils.math import quat_mul
    tasks = []
    for idx, wp in enumerate(waypoints):
        angle_list = ANGLES_FWD if idx % 2 == 0 else ANGLES_REV
        for angle_deg in angle_list:
            # Rotate around X axis (left/right tilt)
            half_rad = math.radians(angle_deg) / 2.0
            q_rot_x = torch.tensor([[math.cos(half_rad), math.sin(half_rad), 0.0, 0.0]], device=device)
            # multiply global rotation by initial orientation
            target_q = quat_mul(q_rot_x, initial_ee_quat_w)
            
            tasks.append({
                "wp": wp,
                "angle": angle_deg,
                "target_quat": target_q
            })

    print(f"[INFO] Expanded waypoints to {len(tasks)} tracking tasks")
    print(f"       5 angles/WP, serpentine sweep (-30→+30 / +30→-30 alternating)")
    print(f"       Estimated angular travel: ~60°/WP (vs 180° for non-serpentine)")

    # ── VIZ: Draw Insertion Paths ──
    if not args_cli.headless:
        try:
            from isaacsim.util.debug_draw import _debug_draw
            draw = _debug_draw.acquire_debug_draw_interface()
        except ImportError:
            try:
                from omni.isaac.debug_draw import _debug_draw
                draw = _debug_draw.acquire_debug_draw_interface()
            except ImportError:
                draw = None
                
        if draw:
            point_list, color_list, size_list = [], [], []
            st_list, end_list, lc_list, ls_list = [], [], [], []
            
            for task in tasks:
                wp_pos = task["wp"]["pos"]
                point_list.append(wp_pos.tolist())
                color_list.append((0.1, 0.8, 1.0, 0.8))
                size_list.append(5)
                
                # Forward direction from quaternion (+Z axis)
                R = matrix_from_quat(task["target_quat"]) # (1, 3, 3)
                z_vec = R[0, :, 2].cpu().numpy()
                
                # The line segment backwards from the waypoint along -Z (insertion backward path)
                start_pt = wp_pos.tolist()
                end_pt = (wp_pos - 0.10 * z_vec).tolist()
                
                st_list.append(start_pt)
                end_list.append(end_pt)
                
                # Different colors for different angles
                ang = task["angle"]
                if ang == 0.0:
                    lc_list.append((0.0, 1.0, 0.2, 0.8))   # Green  — 0°
                elif abs(ang) == 15.0:
                    if ang > 0:
                        lc_list.append((1.0, 0.5, 0.0, 0.7))  # Orange — +15°
                    else:
                        lc_list.append((1.0, 0.2, 0.8, 0.7))  # Pink   — -15°
                else:  # ±30°
                    if ang > 0:
                        lc_list.append((1.0, 0.0, 0.0, 0.9))  # Red    — +30°
                    else:
                        lc_list.append((0.2, 0.4, 1.0, 0.9))  # Blue   — -30°
                ls_list.append(2)
                
            draw.draw_points(point_list, color_list, size_list)
            draw.draw_lines(st_list, end_list, lc_list, ls_list)
            print(f"[VIZ] Drew {len(tasks)} target points and insertion paths via Debug Draw")

    current_task_idx = 0
    state = "MOVING"  # "MOVING" or "WAITING"
    wait_start_time = 0.0
    moving_start_time = 0.0
    step = 0
    cartesian_integral = torch.zeros(1, 3, device=env.device)
    
    import csv
    import os
    csv_filename = "manipulability_results.csv"
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Waypoint", "Angle_X", "Pos_X", "Pos_Y", "Pos_Z", "Error_mm", "Manipulability_Index"])
    
    # Check if we are running in headless mode. omni.ui is only available with GUI.
    if not args_cli.headless:
        import omni.ui as ui
        class UIManager:
            def __init__(self):
                self.is_started = False
                self.window = ui.Window("Waypoint Tracking Control", width=300, height=100)
                with self.window.frame:
                    with ui.VStack(height=0, spacing=8):
                        ui.Label("Click Start when ready to begin tracking.", word_wrap=True)
                        self.start_btn = ui.Button("Start Tracking", height=30)
                        self.start_btn.set_clicked_fn(self._on_start)
            def _on_start(self):
                self.is_started = True
                self.start_btn.text = "Running..."
                self.start_btn.enabled = False
                
        ui_manager = UIManager()
    else:
        # Dummy UI manager for headless mode
        class DummyUIManager:
            def __init__(self):
                self.is_started = True
        ui_manager = DummyUIManager()
    
    print("=" * 60)
    print("  Starting Waypoint Tracking Sequence")
    print("=" * 60)
    
    try:
        print("[DEBUG] Entering loop...", flush=True)
        while simulation_app.is_running():
            # Wait for user to press Start
            if not ui_manager.is_started:
                robot.set_joint_position_target(init_joint_pos, joint_ids=ik_joint_ids)
                env.scene.write_data_to_sim()
                env.sim.step()
                env.scene.update(sim_dt)
                continue
                
            if step == 0:
                print("[DEBUG] Loop started successfully.", flush=True)
            
            # Get current task target
            if current_task_idx < len(tasks):
                task = tasks[current_task_idx]
                wp = task["wp"]
                target_pos_local = wp["pos"]
                
                # Transform target to robot base frame
                root_pos_w = robot.data.root_pos_w     # (1, 3)
                root_quat_w = robot.data.root_quat_w   # (1, 4)
                
                # Target pos in world frame
                target_pos_w = torch.tensor(target_pos_local, dtype=torch.float32, device=env.device).unsqueeze(0)
                target_quat_w = task["target_quat"]
            else:
                # Finished all waypoints
                if state != "FINISHED":
                    print("\n[SUCCESS] All waypoints visited!")
                    if not args_cli.headless:
                        print("Maintaining final position. Close window to exit.")
                    state = "FINISHED"
                    
                target_pos_w = None
                
                # If headless, we can exit immediately to generate the plot
                if args_cli.headless:
                    break
                
            # Current EE pose in world frame
            ee_pos_w = robot.data.body_pos_w[:, ik_body_idx]
            ee_quat_w = robot.data.body_quat_w[:, ik_body_idx]
            
            # State Machine Logic
            if target_pos_w is not None:
                # Calculate distance
                dist = torch.norm(ee_pos_w - target_pos_w, dim=-1).item()
                
                if state == "MOVING":
                    # Initialize moving start time if needed
                    if moving_start_time == 0.0:
                        moving_start_time = time.time()
                        
                    # Cartesian Integral to close steady state error
                    if dist < 0.03:
                        cartesian_integral += (target_pos_w[:, :3] - ee_pos_w[:, :3]) * 0.02
                        cartesian_integral = torch.clamp(cartesian_integral, -0.05, 0.05)
                    else:
                        cartesian_integral.zero_()
                        
                    if step % 50 == 0:
                        print(f"  [MOVING] Target: {wp['name']} (Angle: {task['angle']}°) | Dist: {dist*1000:.1f} mm | EE_Z: {ee_pos_w[0,2].item():.3f} | Target_Z: {target_pos_w[0,2].item():.3f}")
                        print(f"           Joints: {[f'{x:.3f}' for x in robot.data.joint_pos[0, ik_joint_ids].tolist()]}", flush=True)
                        
                    if dist < args_cli.pos_threshold:
                        # Reached target
                        state = "WAITING"
                        wait_start_time = time.time()
                        moving_start_time = 0.0
                        
                        # Calculate Manipulability Index: w = sqrt(det(J * J^T)). For 6x6 square J, w = abs(det(J))
                        J = robot.root_physx_view.get_jacobians()[0, ik_ee_jacobi_idx, :, jacobian_col_ids]
                        manipulability = torch.abs(torch.linalg.det(J)).item()
                        
                        print(f"  [{current_task_idx+1}/{len(tasks)}] Reached {wp['name']} @ {task['angle']}°! (Error: {dist*1000:.1f} mm | Manipulability: {manipulability:.4e})")
                        
                        # Save to CSV
                        with open(csv_filename, mode='a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([wp['name'], task['angle'], wp['pos'][0], wp['pos'][1], wp['pos'][2], f"{dist*1000:.2f}", f"{manipulability:.6e}"])
                            
                    elif time.time() - moving_start_time > 8.0:
                        # Timeout logic (e.g. singularity or unreachable)
                        print(f"  [TIMEOUT] Failed to reach {wp['name']} @ {task['angle']}° within 8 seconds. (Final Error: {dist*1000:.1f} mm). Skipping to next...")
                        current_task_idx += 1
                        moving_start_time = 0.0
                        cartesian_integral.zero_()
                        if current_task_idx < len(tasks):
                            print(f"\nMoving to {tasks[current_task_idx]['wp']['name']} ({tasks[current_task_idx]['angle']}°)...")
                            
                elif state == "WAITING":
                    cartesian_integral.zero_()
                    # Wait for stability/visualization
                    if time.time() - wait_start_time > args_cli.wait_time:
                        current_task_idx += 1
                        state = "MOVING"
                        moving_start_time = 0.0
                        if current_task_idx < len(tasks):
                            print(f"\nMoving to {tasks[current_task_idx]['wp']['name']} ({tasks[current_task_idx]['angle']}°)...")
            
            # Compute IK
            jacobian = robot.root_physx_view.get_jacobians()[:, ik_ee_jacobi_idx, :, jacobian_col_ids]
            joint_pos = robot.data.joint_pos[:, ik_joint_ids]
            
            # If tracking, set command
            if target_pos_w is not None:
                boosted_target_pos = target_pos_w + cartesian_integral
                ik_command = torch.cat([boosted_target_pos, target_quat_w], dim=-1)
                diff_ik_controller.set_command(ik_command, ee_quat=ee_quat_w)
            else:
                # No target, keep current position to freeze
                ik_command = torch.cat([ee_pos_w, ee_quat_w], dim=-1)
                diff_ik_controller.set_command(ik_command, ee_quat=ee_quat_w)
                
            # Compute joint positions
            joint_pos_des = diff_ik_controller.compute(ee_pos_w, ee_quat_w, jacobian, joint_pos)
            
            # Joint clamping to prevent self-collision
            for j_idx, (j_min, j_max) in JOINT_CLAMPS.items():
                joint_pos_des[:, j_idx] = torch.clamp(joint_pos_des[:, j_idx], min=j_min, max=j_max)
            
            # Apply to robot
            robot.set_joint_position_target(joint_pos_des, joint_ids=ik_joint_ids)
            try:
                env.scene.write_data_to_sim()
                env.sim.step()
                env.scene.update(sim_dt)
            except Exception as e:
                print(f"[ERROR] Simulation step failed: {e}", flush=True)
            
            step += 1

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        # Visualize RASSCMAP
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import os
            
            df = pd.read_csv(csv_filename)
            if not df.empty:
                df_gp = df.groupby('Waypoint').agg({
                    'Pos_X': 'mean',
                    'Pos_Y': 'mean',
                    'Pos_Z': 'mean',
                    'Manipulability_Index': 'mean'
                }).reset_index()

                # 確保照著 waypoint 順序排序，這樣連線才會是正確的下針路徑
                df_gp = df_gp.sort_values('Waypoint')

                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # 畫出下針路徑的連線 (灰色虛線)
                ax.plot(df_gp['Pos_X'], df_gp['Pos_Y'], df_gp['Pos_Z'], 
                        color='gray', linestyle='dashed', linewidth=2, label='Insertion Path')
                
                sc = ax.scatter(df_gp['Pos_X'], df_gp['Pos_Y'], df_gp['Pos_Z'], 
                                c=df_gp['Manipulability_Index'], cmap='viridis', s=100, label='Waypoints')
                plt.colorbar(sc, label='Average Manipulability')
                ax.set_title("3D Reachability & Manipulability Map (RASSCMAP)")
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_zlabel('Z (m)')
                ax.legend()
                
                vis_dir = r"C:\Users\RMML\IsaacLab\scripts\isaaclab_ws\reachability_map\rasscmaps"
                os.makedirs(vis_dir, exist_ok=True)
                
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                vis_path = os.path.join(vis_dir, f"rasscmap_3d_{timestamp}.png")
                
                plt.savefig(vis_path)
                abs_vis_path = os.path.abspath(vis_path)
                print(f"\n[INFO] Saved 3D RASSCMAP visualization to: {abs_vis_path}")
                
                # Show the interactive 3D plot window
                plt.show()
        except Exception as e:
            print(f"\n[ERROR] Failed to generate RASSCMAP visualization: {e}")
            
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()
