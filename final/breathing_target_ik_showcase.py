"""
breathing_target_ik_showcase.py

Single-environment SHOWCASE using Differential IK (not RL policy) to smoothly
track a breathing /Root/tumor target in real-time.

The robot uses DifferentialIKController with needle_tip as the tracked body,
providing stable, jitter-free motion compared to the RL policy approach.

Usage:
    .\isaaclab.bat -p scripts\isaaclab_ws\final\breathing_target_ik_showcase.py
    .\isaaclab.bat -p scripts\isaaclab_ws\final\breathing_target_ik_showcase.py --robot_x 0.46
"""
import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Breathing Target IK Showcase")
parser.add_argument("--robot_x", type=float, default=None,
                    help="Robot arm base X position (env-local). If omitted, uses USD default.")
parser.add_argument("--max_steps", type=int, default=0,
                    help="Max simulation steps (0 = run forever until Ctrl+C)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
from copy import deepcopy

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_extension_linkout_cone_orientation_cfg import TM5ExtensionLinkOutFanOrientationEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from pxr import UsdGeom, Gf


# ── Smooth Curved Breathing Parameters ──
P0 = np.array([-0.4, -0.8, 1.07])
P1 = np.array([-0.33, -0.8, 1.05])
P2 = np.array([-0.3, -0.8, 1.05])

# Control point for quadratic bezier to pass through P1 at s=0.5
PC = 2.0 * P1 - 0.5 * P0 - 0.5 * P2
PERIOD = 12.6   # seconds

def breathing_pos(t):
    """Compute smooth curved tumor position at time t (env-local)."""
    phase = 2 * math.pi * (t / PERIOD)
    s = (1.0 - math.cos(phase)) / 2.0  # Smoothly oscillates between 0.0 and 1.0
    pos = ((1.0 - s)**2) * P0 + (2.0 * (1.0 - s) * s) * PC + (s**2) * P2
    return pos


# ── Extension Link Out initial joint positions (deg → rad) ──
# True initial: -84.9°, 43.4°, 57.5°, 78.4°, 0.0°, -92.0°
EXTENSION_LINK_OUT_JOINT_POS = {
    "joint_1": -1.48178,   # -84.9°
    "joint_2":  0.75747,   #  43.4°
    "joint_3":  1.00356,   #  57.5°
    "joint_4":  1.36834,   #  78.4°
    "joint_5":  0.0,       #   0.0°
    "joint_6": -1.60570,   # -92.0°
}


def set_prim_translate(stage, prim_path, translation):
    """Set translation XformOp on a prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    xform = UsdGeom.Xformable(prim)
    translate_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if not translate_op:
        translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(*[float(v) for v in translation]))


def main():
    # ── Build identical environment as scorer ──
    env_cfg = TM5ExtensionLinkOutFanOrientationEnvCfg()
    env_cfg.scene.num_envs = 1
    
    # Disable command resampling
    if hasattr(env_cfg.commands, "ee_pose"):
        env_cfg.commands.ee_pose.resampling_time_range = (1e10, 1e10)
        env_cfg.commands.ee_pose.debug_vis = False
        
    env = ManagerBasedRLEnv(cfg=env_cfg)
    sim = env.sim
    scene = env.scene

    sim.set_camera_view((-1.5, -1.5, 1.5), (-0.5, -0.5, 0.5))
    env.reset()
    robot = scene["robot"]
    
    # Apply Robot Base Position if specified
    if args_cli.robot_x is not None:
        root_pos_w = robot.data.root_pos_w.clone()
        env_origins_clone = env.scene.env_origins.clone()
        root_pos_w[:, 0] = env_origins_clone[:, 0] + args_cli.robot_x
        robot.write_root_pose_to_sim(torch.cat([root_pos_w, robot.data.root_quat_w], dim=-1))
        sim.step()
        print(f"  [INIT] Robot base moved to X = {args_cli.robot_x}")

    # ── Force initial joint posture (avoid starting from zero-pose) ──
    init_joint_pos = torch.tensor(
        [list(EXTENSION_LINK_OUT_JOINT_POS.values())],
        device=sim.device
    )
    init_joint_vel = torch.zeros_like(init_joint_pos)

    # Write state directly to physics AND set PD target to hold this pose
    robot.write_joint_state_to_sim(init_joint_pos, init_joint_vel)
    robot.set_joint_position_target(init_joint_pos)

    # Let PD controller settle into the initial posture (200 steps = 2s at dt=0.01)
    sim_dt_init = sim.get_physics_dt()
    for _ in range(200):
        robot.set_joint_position_target(init_joint_pos)  # keep PD target every step
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt_init)

    # Verify actual joint positions after settling
    actual_joints = robot.data.joint_pos[0].cpu().numpy()
    actual_deg = [f"{v*180/3.14159:.1f}" for v in actual_joints]
    target_deg = [f"{v*180/3.14159:.1f}" for v in init_joint_pos[0].tolist()]
    print(f"[OK] Robot initialized to surgical posture")
    print(f"  Target joints (deg): {target_deg}")
    print(f"  Actual joints (deg): {actual_deg}")

    print(f"\n[Robot Info]")
    print(f"  Body names: {robot.body_names}")
    print(f"  Joint names: {robot.joint_names}")
    print(f"  Is fixed base: {robot.is_fixed_base}")

    # ── Setup Differential IK Controller ──
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.01},
    )
    diff_ik_controller = DifferentialIKController(
        diff_ik_cfg, num_envs=1, device=sim.device
    )

    # Track needle_tip body
    robot_entity_cfg = SceneEntityCfg(
        "robot", joint_names=["joint_[1-6]"], body_names=["needle_tip"]
    )
    robot_entity_cfg.resolve(scene)

    joint_ids = robot_entity_cfg.joint_ids
    body_idx = robot_entity_cfg.body_ids[0]
    ee_jacobi_idx = body_idx - 1 if robot.is_fixed_base else body_idx

    # For non-fixed-base: Jacobian from PhysX has 6 extra floating-base DOF columns
    # joint_ids indexes into the joint array, but Jacobian columns include floating DOFs first
    if robot.is_fixed_base:
        jacobian_col_ids = joint_ids
    else:
        # Offset by 6 to skip floating-base DOFs in the Jacobian matrix
        jacobian_col_ids = [j + 6 for j in joint_ids]

    print(f"  Tracking body: needle_tip (idx={body_idx})")
    print(f"  Jacobian idx: {ee_jacobi_idx}")
    print(f"  Joint IDs: {joint_ids}")
    print(f"  Jacobian col IDs: {jacobian_col_ids}")
    print(f"  is_fixed_base: {robot.is_fixed_base}")

    # ── Per-joint clamping to lock arm configuration ──
    # Based on initial joint config (the known collision-free posture)
    # Each joint is clamped to stay on the same "side" as the initial value
    # with a generous range to allow IK freedom but prevent config flipping
    INIT_JOINTS = EXTENSION_LINK_OUT_JOINT_POS
    JOINT_CLAMPS = {
        # idx: (min, max) -- derived from initial config side
        0: (-3.14, -0.1),    # joint_1: init=-84.9°  → must stay negative
        1: (-0.3,   2.5),    # joint_2: init=+43.4°  → must stay positive-ish
        2: (-0.3,   2.5),    # joint_3: init=+57.5°  → must stay positive-ish
        3: (-0.5,   3.14),   # joint_4: init=+78.4°  → must stay positive-ish
        4: (-1.57,  1.57),   # joint_5: init= 0.0°   → near zero, allow both
        5: (-3.14,  0.5),    # joint_6: init=-92.0°  → must stay negative
    }
    print(f"  Joint clamps: {JOINT_CLAMPS}")


    # ── Get USD stage for Tumor manipulation ──
    import omni.usd
    stage = omni.usd.get_context().get_stage()

    # ── Optionally move robot base ──
    if args_cli.robot_x is not None:
        robotarm_base_path = "/World/envs/env_0/Root/robotarm_base"
        prim = stage.GetPrimAtPath(robotarm_base_path)
        if prim.IsValid():
            xform = UsdGeom.Xformable(prim)
            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    current = op.Get()
                    op.Set(Gf.Vec3d(args_cli.robot_x, float(current[1]), float(current[2])))
                    print(f"[ROBOT] Moved robotarm_base X to {args_cli.robot_x:.3f}")
                    break

    # ── Sim loop ──
    sim_dt = sim.get_physics_dt()
    current_t = 0.0
    step = 0

    print(f"\n{'='*60}")
    print(f"  BREATHING TARGET IK SHOWCASE")
    print(f"  Controller: DifferentialIK (DLS, pose)")
    print(f"  Tracked body: needle_tip")
    print(f"  Tumor path: {P0} -> {P1} -> {P2}")
    print(f"  Period: ~{PERIOD:.1f}s  |  dt={sim_dt:.4f}s")
    if args_cli.max_steps > 0:
        print(f"  Max steps: {args_cli.max_steps}")
    else:
        print(f"  Running indefinitely (Ctrl+C to stop)")
    print(f"{'='*60}\n")

    # Initial IK reset
    diff_ik_controller.reset()
    
    # Capture initial EE orientation (surgical posture downwards) to maintain it
    initial_ee_quat_w = robot.data.body_quat_w[:, body_idx].clone()

    # ── Simulation Control UI ──
    import omni.ui as ui
    class SimControlUI:
        def __init__(self):
            self.is_started = False
            self.window = ui.Window("Simulation Control", width=300, height=100)
            with self.window.frame:
                with ui.VStack(height=0, spacing=10):
                    ui.Label("Click Start when ready to observe tracking.", word_wrap=True)
                    self.start_btn = ui.Button("Start Simulation", height=30)
                    self.start_btn.set_clicked_fn(self.on_start)
                    
        def on_start(self):
            self.is_started = True
            self.start_btn.text = "Running..."
            self.start_btn.enabled = False

    control_ui = SimControlUI()
    has_reached_target = False

    try:
        while simulation_app.is_running():
            if not control_ui.is_started:
                # Just hold the initial surgical posture and render
                robot.set_joint_position_target(init_joint_pos, joint_ids=joint_ids)
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim_dt)
                continue

            if args_cli.max_steps > 0 and step >= args_cli.max_steps:
                break

            # If not yet reached, keep target stationary at t=0
            if not has_reached_target:
                ee_local = robot.data.body_pos_w[0, body_idx].cpu().numpy()
                static_target = np.array(breathing_pos(0.0))
                dist = np.linalg.norm(ee_local - static_target)
                if dist < 0.01:
                    has_reached_target = True
                    print("\n[Target Reached] Needle is in position. Starting target breathing animation...\n")
            else:
                current_t += sim_dt

            # 1. Update tumor position (breathing)
            tumor_local = breathing_pos(current_t)
            set_prim_translate(stage, "/World/envs/env_0/Root/tumor", tumor_local)

            # 2. Compute IK target in robot base frame
            #    tumor_local is in env-local = world frame (single env at origin)
            root_pos_w = robot.data.root_pos_w     # (1, 3)
            root_quat_w = robot.data.root_quat_w   # (1, 4)

            target_pos_w = torch.tensor(tumor_local, dtype=torch.float32, device=sim.device).unsqueeze(0)
            target_quat_w = initial_ee_quat_w

            target_pos_b, target_quat_b = subtract_frame_transforms(
                root_pos_w, root_quat_w,
                target_pos_w, target_quat_w
            )

            # 3. Get current EE state and Jacobian
            ee_pos_w = robot.data.body_pos_w[:, body_idx]
            ee_quat_w = robot.data.body_quat_w[:, body_idx]

            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pos_w, root_quat_w,
                ee_pos_w, ee_quat_w
            )

            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, jacobian_col_ids]
            joint_pos = robot.data.joint_pos[:, joint_ids]

            # 4. Set IK command (Pose mode: [pos, quat]) and compute joint targets
            ik_commands = torch.cat([target_pos_b, target_quat_b], dim=-1)
            diff_ik_controller.set_command(ik_commands)
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

            # 4b. Per-joint clamping to lock arm configuration & prevent self-collision
            for j_idx, (j_min, j_max) in JOINT_CLAMPS.items():
                joint_pos_des[:, j_idx] = torch.clamp(joint_pos_des[:, j_idx], min=j_min, max=j_max)

            # 5. Apply joint targets
            robot.set_joint_position_target(joint_pos_des, joint_ids=joint_ids)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)


            # 7. Periodic logging (every 100 steps ~ 1s at dt=0.01)
            if step % 100 == 0:
                ee_local = robot.data.body_pos_w[0, body_idx].cpu().numpy()
                dist = np.linalg.norm(ee_local - tumor_local)
                print(f"  step {step:5d} | t={current_t:6.2f}s | "
                      f"tumor=({tumor_local[0]:+.4f}, {tumor_local[2]:+.4f}) | "
                      f"needle=({ee_local[0]:+.4f}, {ee_local[2]:+.4f}) | "
                      f"dist={dist:.4f}m {'✓' if dist < 0.05 else '✗'}")

            step += 1

    except KeyboardInterrupt:
        pass

    print(f"\n[DONE] Showcase ended after {step} steps ({current_t:.1f}s)")
    simulation_app.close()


if __name__ == "__main__":
    main()
