"""Probe IK convergence loop to target position."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd_path", type=str, default=r"C:\Nick\surgery_team\surgery_team\USD\animation\isaaclab_multi_env.usd")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import omni.usd
from pxr import UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

print(f"--- Loading USD: {args_cli.usd_path} ---")
omni.usd.get_context().open_stage(args_cli.usd_path)

print("\n--- Initializing SimulationContext ---")
sim_cfg = sim_utils.SimulationCfg(dt=1/60, device="cuda:0", gravity=(0.0, 0.0, -9.81))
sim = SimulationContext(sim_cfg)

# Find articulation path
stage = omni.usd.get_context().get_stage()
robot_path = None
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        robot_path = str(prim.GetPath())
        break

@configclass
class MinimalSceneCfg(InteractiveSceneCfg):
    num_envs = 1
    env_spacing = 5.0
    robot = ArticulationCfg(
        prim_path=robot_path,
        spawn=None,
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint_[1-6]"],
                effort_limit=200.0,
                velocity_limit=2.0,
                stiffness=200.0,
                damping=20.0,
            ),
        },
    )

try:
    scene = InteractiveScene(MinimalSceneCfg())
    sim.reset()
    scene.update(dt=1/60)
    robot = scene["robot"]
    
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["joint_[1-6]"], body_names=["needle_tip"])
    robot_entity_cfg.resolve(scene)
    
    ik_joint_ids = robot_entity_cfg.joint_ids
    ik_body_idx = robot_entity_cfg.body_ids[0]
    ik_ee_jacobi_idx = ik_body_idx - 1 if robot.is_fixed_base else ik_body_idx
    jacobian_col_ids = ik_joint_ids if robot.is_fixed_base else [j + 6 for j in ik_joint_ids]
    
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", 
        use_relative_mode=False, 
        ik_method="dls", 
        ik_params={"lambda_val": 0.01}
    )
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=1, device=sim.device)
    
    # 🎯 Let's define a target position (e.g. 10cm offset in X, 10cm in Y, 5cm in Z)
    ee_pos_w = robot.data.body_pos_w[:, ik_body_idx].clone()
    ee_quat_w = robot.data.body_quat_w[:, ik_body_idx].clone()
    
    target_pos_w = ee_pos_w + torch.tensor([[-0.05, 0.1, -0.05]], device=sim.device)
    target_quat_w = ee_quat_w
    
    print("\nInitial EE pos:", ee_pos_w[0].tolist())
    print("Target EE pos:", target_pos_w[0].tolist())
    
    # Converge loop
    for i in range(100):
        # Update scene to refresh robot states
        scene.update(dt=1/60)
        
        # Current poses
        curr_ee_pos_w = robot.data.body_pos_w[:, ik_body_idx]
        curr_ee_quat_w = robot.data.body_quat_w[:, ik_body_idx]
        
        # Calculate distance
        dist = torch.norm(target_pos_w - curr_ee_pos_w).item()
        if dist < 0.001:
            print(f"Converged at iteration {i}! Final EE pos: {curr_ee_pos_w[0].tolist()} (dist: {dist:.6f} m)")
            break
            
        root_pos_w = robot.data.root_pos_w
        root_quat_w = robot.data.root_quat_w
        
        ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, curr_ee_pos_w, curr_ee_quat_w)
        target_pos_b, target_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, target_pos_w, target_quat_w)
        
        jacobian = robot.root_physx_view.get_jacobians()[:, ik_ee_jacobi_idx, :, jacobian_col_ids]
        
        joint_pos = robot.data.joint_pos[:, ik_joint_ids]
        ik_command = torch.cat([target_pos_b, target_quat_b], dim=-1)
        diff_ik_controller.set_command(ik_command, ee_quat=ee_quat_b)
        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        
        # Write to physics state and target
        robot.write_joint_state_to_sim(joint_pos_des, torch.zeros_like(joint_pos_des))
        robot.set_joint_position_target(joint_pos_des, joint_ids=ik_joint_ids)
        
        # Step physics to apply the state and propagate kinematics
        sim.step()
        
    else:
        print(f"Failed to converge in 100 iterations. Final distance: {dist:.6f} m")
        
except Exception as e:
    import traceback
    traceback.print_exc()

simulation_app.close()
