import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import omni.usd
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.math import subtract_frame_transforms, matrix_from_quat, quat_inv
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg

class MinimalEnvCfg(DirectMARLEnvCfg):
    decimation = 2
    episode_length_s = 5.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    sim: SimulationCfg = SimulationCfg(dt=1/120, render_interval=2)

def main():
    from isaaclab_ws.final.ndi_multipose_scorer_v2_moving_target import NDIRobotTaskEnvCfg
    
    env_cfg = NDIRobotTaskEnvCfg()
    env_cfg.scene.num_envs = 1
    
    env = DirectMARLEnv(cfg=env_cfg)
    env.reset()
    
    robot = env.scene["robot"]
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["joint_[1-6]"], body_names=["needle_tip"])
    robot_entity_cfg.resolve(env.scene)
    
    ik_joint_ids = robot_entity_cfg.joint_ids
    ik_body_idx = robot_entity_cfg.body_ids[0]
    ik_ee_jacobi_idx = ik_body_idx - 1 if robot.is_fixed_base else ik_body_idx
    if robot.is_fixed_base:
        jacobian_col_ids = ik_joint_ids
    else:
        jacobian_col_ids = [j + 6 for j in ik_joint_ids]
        
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": 0.01})
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=1, device=env.device)
    
    init_joint_pos = torch.tensor([[-1.48178, 0.75747, 1.00356, 1.36834, 0.0, -1.60570]], device=env.device)
    robot.set_joint_position_target(init_joint_pos, joint_ids=ik_joint_ids)
    
    for _ in range(50):
        env.sim.step()
        env.scene.update(1/60)
        
    ee_pos_w = robot.data.body_pos_w[:, ik_body_idx]
    ee_quat_w = robot.data.body_quat_w[:, ik_body_idx]
    
    target_pos_w = ee_pos_w + torch.tensor([[0.0, 0.1, 0.0]], device=env.device)
    target_quat_w = ee_quat_w
    
    print("Initial EE Pos (World):", ee_pos_w)
    print("Target EE Pos (World):", target_pos_w)
    
    # 1. Transform to base
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
    target_pos_b, target_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, target_pos_w, target_quat_w)
    
    # 2. Get Jacobian
    jacobian = robot.root_physx_view.get_jacobians()[:, ik_ee_jacobi_idx, :, jacobian_col_ids]
    
    print("Jacobian Shape:", jacobian.shape)
    print("Jacobian Norm:", torch.norm(jacobian).item())
    
    # 3. Compute
    joint_pos = robot.data.joint_pos[:, ik_joint_ids]
    ik_command = torch.cat([target_pos_b, target_quat_b], dim=-1)
    diff_ik_controller.set_command(ik_command, ee_quat=ee_quat_b)
    
    joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
    
    print("Current Joint Pos:", joint_pos)
    print("Desired Joint Pos:", joint_pos_des)
    print("Delta Joint Pos:", joint_pos_des - joint_pos)
    
    simulation_app.close()

if __name__ == "__main__":
    main()
