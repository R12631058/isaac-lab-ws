#已經棄用，改由手動在 isaac sim 擺放
"""
generate_task_waypoints.py

    .\\isaaclab.bat -p scripts\\isaaclab_ws\\reachability_map\\generate_task_waypoints.py
    .\\isaaclab.bat -p scripts\\isaaclab_ws\\reachability_map\\generate_task_waypoints.py --num_envs 1
    .\\isaaclab.bat -p scripts\\isaaclab_ws\\reachability_map\\generate_task_waypoints.py --visualize
    .\\isaaclab.bat -p scripts\\isaaclab_ws\\reachability_map\\generate_task_waypoints.py --tumor_x -0.4 --tumor_y -0.8 --tumor_z 1.07

scripts/isaaclab_ws/reachability_map/task_waypoints_50.json
"""

import argparse
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Stage 1: Generate 10x5 needle insertion task waypoints around tumor"
)
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of environments (only 1 needed for waypoint generation)")
parser.add_argument("--visualize", action="store_true",
                    help="Keep the simulation running to visually inspect waypoints")

# Tumor position (env-local frame)
# Default: from BreathingTarget P0 in ndi_multipose_scorer_v2_moving_target.py
parser.add_argument("--tumor_x", type=float, default=-0.4,
                    help="Tumor X position (env-local)")
parser.add_argument("--tumor_y", type=float, default=-0.8,
                    help="Tumor Y position (env-local)")
parser.add_argument("--tumor_z", type=float, default=1.07,
                    help="Tumor Z position (env-local)")

# Hemisphere sampling parameters
parser.add_argument("--n_azimuth", type=int, default=10,
                    help="Number of azimuthal samples (around the hemisphere)")
parser.add_argument("--n_elevation", type=int, default=5,
                    help="Number of elevation samples (from shallow to steep)")
parser.add_argument("--approach_distance", type=float, default=0.10,
                    help="Distance from tumor surface to position needle tip (m)")
parser.add_argument("--elev_min_deg", type=float, default=15.0,
                    help="Minimum elevation angle from horizontal (deg)")
parser.add_argument("--elev_max_deg", type=float, default=75.0,
                    help="Maximum elevation angle from horizontal (deg)")

parser.add_argument("--output", type=str,
                    default="scripts/isaaclab_ws/reachability_map/task_waypoints_50.json",
                    help="Output JSON path")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch simulation
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
from copy import deepcopy
from datetime import datetime

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_extension_linkout_cone_orientation_cfg import (
    TM5ExtensionLinkOutFanOrientationEnvCfg,
)


# Initial joint config (deg ??rad) ??surgical posture
# from breathing_target_ik_showcase.py / ndi_multipose_scorer_v2_moving_target.py
INIT_JOINT_POS_RAD = [-1.48178, 0.75747, 1.00356, 1.36834, 0.0, -1.60570]

# Needle tip body is directly in the articulation tree ??no offset needed
# EXTENSION_LINK_OUT_TIP_OFFSET = (0.0, 0.0, 0.0)

# Preferred initial orientation of needle_tip (quat w,x,y,z)
# Needle pointing downward: Roll=-180°, Pitch=0°, Yaw=180° ??quat(0, 0, 1, 0)
NEEDLE_DOWN_QUAT = np.array([0.0, 0.0, 1.0, 0.0])  # w, x, y, z



def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros_like(v)


def rotation_matrix_from_z_axis(z_axis: np.ndarray) -> np.ndarray:
    """Build rotation matrix where local +Z points along z_axis.
    
    For needle insertion, the needle tip's local +Z axis is the
    insertion direction (pointing INTO the tumor).

    The TM5-700 needle_tip body convention:
      - Local +Z (blue) is the needle shaft direction (toward tumor)
      - We construct a right-handed frame around this direction.
    """
    z = normalize(z_axis)

    # Choose an arbitrary 'up' hint to build the frame
    up_hint = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(z, up_hint)) > 0.999:
        up_hint = np.array([1.0, 0.0, 0.0])

    x = normalize(np.cross(up_hint, z))
    y = np.cross(z, x)

    # Rotation matrix [x | y | z] maps local to world
    return np.column_stack((x, y, z))


def rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    trace = np.trace(R)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def generate_hemisphere_waypoints(
    tumor_pos: np.ndarray,
    n_azimuth: int = 10,
    n_elevation: int = 5,
    approach_distance: float = 0.10,
    elev_min_deg: float = 15.0,
    elev_max_deg: float = 75.0,
) -> list[dict]:
    """Generate 6D waypoints on an upper hemisphere centered at the tumor.

    The hemisphere is above the tumor (Z+ direction in world frame).
    Each waypoint has:
      - position: on the hemisphere surface, at `approach_distance` from tumor center
      - orientation: needle tip +Z axis pointing TOWARD tumor center (insertion direction)

    Coordinate convention (env-local / world-like):
      - Z = up
      - Elevation is measured from the horizontal plane (0° = horizontal, 90° = straight down)
      - Azimuth sweeps 360° in the XY plane

    Args:
        tumor_pos: (3,) tumor center position in env-local frame
        n_azimuth: number of azimuthal samples
        n_elevation: number of elevation samples
        approach_distance: distance from tumor center to needle tip position (m)
        elev_min_deg: minimum elevation angle (deg, from horizontal)
        elev_max_deg: maximum elevation angle (deg, from horizontal)

    Returns:
        List of dicts, each containing:
            - 'position': [x, y, z] needle_tip target position (env-local)
            - 'orientation': [qw, qx, qy, qz] target quaternion
            - 'azimuth_deg': azimuthal angle
            - 'elevation_deg': elevation angle
            - 'insertion_direction': [dx, dy, dz] unit vector toward tumor
            - 'index': sequential index
    """
    waypoints = []
    elev_min_rad = math.radians(elev_min_deg)
    elev_max_rad = math.radians(elev_max_deg)

    idx = 0
    for i_elev in range(n_elevation):
        # Elevation: angle above horizontal plane
        if n_elevation > 1:
            elev = elev_min_rad + (elev_max_rad - elev_min_rad) * i_elev / (n_elevation - 1)
        else:
            elev = (elev_min_rad + elev_max_rad) / 2.0

        for i_az in range(n_azimuth):
            # Azimuth: angle in XY plane (0 ~ 2?)
            az = 2.0 * math.pi * i_az / n_azimuth

            # Direction FROM tumor TO waypoint (outward on hemisphere)
            # In spherical coords: elev from horizontal, az in XY
            dx = math.cos(elev) * math.cos(az)
            dy = math.cos(elev) * math.sin(az)
            dz = math.sin(elev)  # positive = above tumor

            outward_dir = np.array([dx, dy, dz])

            # Waypoint position: tumor + outward * distance
            wp_pos = tumor_pos + outward_dir * approach_distance

            # Insertion direction: from waypoint TOWARD tumor (= -outward)
            insertion_dir = -outward_dir

            # Build orientation: needle_tip +Z axis = insertion direction
            R = rotation_matrix_from_z_axis(insertion_dir)
            quat = rotation_matrix_to_quat(R)

            waypoints.append({
                "index": idx,
                "position": wp_pos.tolist(),
                "orientation": quat.tolist(),
                "azimuth_deg": round(math.degrees(az), 2),
                "elevation_deg": round(math.degrees(elev), 2),
                "insertion_direction": insertion_dir.tolist(),
            })
            idx += 1

    return waypoints



def main():
    tumor_pos = np.array([args_cli.tumor_x, args_cli.tumor_y, args_cli.tumor_z])

    print("\n" + "=" * 64)
    print("  Stage 1: Generate Task Waypoints (10?5 Needle Insertion Poses)")
    print("=" * 64)
    print(f"  Tumor position (env-local): {tumor_pos}")
    print(f"  Azimuth samples:   {args_cli.n_azimuth}")
    print(f"  Elevation samples: {args_cli.n_elevation}")
    print(f"  Total waypoints:   {args_cli.n_azimuth * args_cli.n_elevation}")
    print(f"  Approach distance: {args_cli.approach_distance} m")
    print(f"  Elevation range:   [{args_cli.elev_min_deg}°, {args_cli.elev_max_deg}°]")
    print("=" * 64 + "\n")

    waypoints = generate_hemisphere_waypoints(
        tumor_pos=tumor_pos,
        n_azimuth=args_cli.n_azimuth,
        n_elevation=args_cli.n_elevation,
        approach_distance=args_cli.approach_distance,
        elev_min_deg=args_cli.elev_min_deg,
        elev_max_deg=args_cli.elev_max_deg,
    )

    print(f"[OK] Generated {len(waypoints)} waypoints\n")

    # Print summary table
    print(f"{'Idx':>4} {'Azimuth':>8} {'Elev':>6}  {'Position (x, y, z)':>30}  {'Quat (w,x,y,z)':>36}")
    print("-" * 96)
    for wp in waypoints:
        pos = wp["position"]
        quat = wp["orientation"]
        print(
            f"{wp['index']:4d} {wp['azimuth_deg']:7.1f}° {wp['elevation_deg']:5.1f}° "
            f" ({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}) "
            f" ({quat[0]:+.4f}, {quat[1]:+.4f}, {quat[2]:+.4f}, {quat[3]:+.4f})"
        )

    output_data = {
        "metadata": {
            "description": "Stage 1: 10x5 needle insertion task waypoints for robot base optimization",
            "reference": "Sundaram et al. (2022) RASSCMAP",
            "timestamp": datetime.now().isoformat(),
            "tumor_position": tumor_pos.tolist(),
            "approach_distance_m": args_cli.approach_distance,
            "n_azimuth": args_cli.n_azimuth,
            "n_elevation": args_cli.n_elevation,
            "elevation_range_deg": [args_cli.elev_min_deg, args_cli.elev_max_deg],
            "needle_tip_body": "needle_tip",
            "tip_offset": [0.0, 0.0, 0.0],
            "notes": (
                "Each waypoint is a 6D target pose for the needle_tip body. "
                "The needle local +Z axis points toward the tumor center (insertion direction). "
                "Positions are in env-local frame. "
                "Orientation is quaternion (w, x, y, z)."
            ),
        },
        "waypoints": waypoints,
    }

    # Resolve output path relative to IsaacLab root
    out_path = args_cli.output
    if not os.path.isabs(out_path):
        # If running via isaaclab.bat, CWD is usually the IsaacLab root
        out_path = os.path.join(os.getcwd(), out_path)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Saved to {out_path}")

    if args_cli.visualize:
        print("\n[VIZ] Launching visualization...")
        _visualize_waypoints(waypoints, tumor_pos)
    else:
        print("\n[INFO] Add --visualize to inspect waypoints in Isaac Lab viewport")
        simulation_app.close()


def _visualize_waypoints(waypoints: list[dict], tumor_pos: np.ndarray):
    """Launch Isaac Lab env and draw waypoint markers in the viewport."""

    env_cfg = TM5ExtensionLinkOutFanOrientationEnvCfg()
    env_cfg.scene.num_envs = 1

    if hasattr(env_cfg.commands, "ee_pose"):
        env_cfg.commands.ee_pose.resampling_time_range = (1e10, 1e10)
        env_cfg.commands.ee_pose.debug_vis = False

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()
    robot = env.scene["robot"]

    # Force initial surgical posture
    device = env.device
    init_joint_pos = torch.tensor([INIT_JOINT_POS_RAD], device=device)
    init_joint_vel = torch.zeros_like(init_joint_pos)
    robot.write_joint_state_to_sim(init_joint_pos, init_joint_vel)
    robot.set_joint_position_target(init_joint_pos)

    sim_dt = env.sim.get_physics_dt()
    for _ in range(200):
        robot.set_joint_position_target(init_joint_pos)
        env.scene.write_data_to_sim()
        env.sim.step()
        env.scene.update(sim_dt)

    print("[OK] Robot initialized to surgical posture")

    try:
        from isaacsim.util.debug_draw import _debug_draw
        draw = _debug_draw.acquire_debug_draw_interface()
    except Exception:
        try:
            from omni.isaac.debug_draw import _debug_draw
            draw = _debug_draw.acquire_debug_draw_interface()
        except Exception:
            print("[WARN] Debug draw not available, running sim loop only")
            draw = None

    if draw:
        # Draw tumor center
        draw.draw_points(
            [tumor_pos.tolist()],
            [(1.0, 0.0, 0.0, 1.0)],  # red
            [15],
        )

        # Draw each waypoint as a small sphere and a line toward tumor
        for wp in waypoints:
            pos = wp["position"]
            ins_dir = wp["insertion_direction"]

            # Waypoint position (blue sphere)
            draw.draw_points(
                [pos],
                [(0.2, 0.5, 1.0, 0.9)],  # blue
                [8],
            )

            # Insertion line (waypoint ??tumor, green)
            draw.draw_lines(
                [pos],
                [tumor_pos.tolist()],
                [(0.0, 1.0, 0.3, 0.6)],
                [1],
            )

        print(f"[VIZ] Drew {len(waypoints)} waypoints + insertion lines")

    print("[VIZ] Running... (close window or Ctrl+C to exit)")
    try:
        while simulation_app.is_running():
            robot.set_joint_position_target(init_joint_pos)
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(sim_dt)
    except KeyboardInterrupt:
        pass

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
