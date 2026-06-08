# 這個script是從C:\Users\RMML\IsaacLab\scripts\tutorials\05_controllers\run_diff_ik.py 複製而來
# 孰悉 isaaclab 操作以及測試"相同的任務"指定給自定義的手臂會發生甚麼事

"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator.

The differential IK controller can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/05_controllers/run_diff_ik.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="tm5_700", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=5, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip
from isaaclab_assets import TM5_700_CFG  # 添加 TM5_700_CFG


@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
    )

    # articulation
    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "tm5_700":
        robot = TM5_700_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")    
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")


def generate_random_ee_goals(num_goals, device, workspace_bounds=None):
    """
    生成隨機的 end-effector 目標
    
    Args:
        num_goals: 目標數量
        device: torch device
        workspace_bounds: 工作空間邊界 {'x': [min, max], 'y': [min, max], 'z': [min, max]}
    
    Returns:
        torch.Tensor: [num_goals, 7] 的目標矩陣 (x, y, z, qx, qy, qz, qw)
    """
    if workspace_bounds is None:
        # 預設工作空間邊界
        workspace_bounds = {
            'x': [0.2, 0.8],    # X 軸範圍
            'y': [-0.6, 0.6],   # Y 軸範圍  
            'z': [0.3, 0.9]     # Z 軸範圍
        }
    
    goals = torch.zeros(num_goals, 7, device=device)
    
    # 隨機位置
    goals[:, 0] = torch.rand(num_goals, device=device) * (workspace_bounds['x'][1] - workspace_bounds['x'][0]) + workspace_bounds['x'][0]  # X
    goals[:, 1] = torch.rand(num_goals, device=device) * (workspace_bounds['y'][1] - workspace_bounds['y'][0]) + workspace_bounds['y'][0]  # Y
    goals[:, 2] = torch.rand(num_goals, device=device) * (workspace_bounds['z'][1] - workspace_bounds['z'][0]) + workspace_bounds['z'][0]  # Z
    
    # 隨機方向（四元數）
    # 方法 1: 使用預定義的方向
    orientations = [
        [0.707, 0, 0.707, 0],      # 向下看
        [0.707, 0.707, 0.0, 0.0],  # 側向
        [0.0, 1.0, 0.0, 0.0],      # 翻轉
        [1.0, 0.0, 0.0, 0.0],      # 其他方向
        [0.5, 0.5, 0.5, 0.5],      # 45度角
    ]
    
    for i in range(num_goals):
        # 隨機選擇一個方向
        orient_idx = torch.randint(0, len(orientations), (1,)).item()
        goals[i, 3:7] = torch.tensor(orientations[orient_idx], device=device)
    
    return goals

def generate_uniform_random_quaternions(num_goals, device):
    """
    生成均勻分佈的隨機四元數
    """
    # Marsaglia 方法生成均勻分佈的四元數
    u1 = torch.rand(num_goals, device=device)
    u2 = torch.rand(num_goals, device=device) * 2 * torch.pi
    u3 = torch.rand(num_goals, device=device) * 2 * torch.pi
    
    sqrt_1_u1 = torch.sqrt(1 - u1)
    sqrt_u1 = torch.sqrt(u1)
    
    quat = torch.zeros(num_goals, 4, device=device)
    quat[:, 0] = sqrt_1_u1 * torch.sin(u2)  # qx
    quat[:, 1] = sqrt_1_u1 * torch.cos(u2)  # qy
    quat[:, 2] = sqrt_u1 * torch.sin(u3)    # qz
    quat[:, 3] = sqrt_u1 * torch.cos(u3)    # qw
    
    return quat

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    robot = scene["robot"]

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # === 隨機目標生成 ===
    # 方法 1: 每次重置生成新的隨機目標
    num_random_goals = 5
    
    # 方法 2: 一次生成多個隨機目標並循環使用
    ee_goals = generate_random_ee_goals(
        num_random_goals, 
        device=sim.device,
        workspace_bounds={
            'x': [0.3, 0.7],    # 根據機器人調整
            'y': [-0.5, 0.5], 
            'z': [0.4, 0.8]
        }
    )
    
    # 方法 3: 完全隨機方向
    # ee_goals[:, 3:7] = generate_uniform_random_quaternions(num_random_goals, sim.device)
    
    print(f"Generated {num_random_goals} random ee_goals:")
    for i, goal in enumerate(ee_goals):
        print(f"  Goal {i}: pos={goal[:3]}, quat={goal[3:]}")

    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals[current_goal_idx]


    # Specify robot-specific parameters
    if args_cli.robot == "franka_panda":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    elif args_cli.robot == "ur10":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_link"])
    elif args_cli.robot == "tm5_700":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["joint_.*"], body_names=["flange"])    
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")
    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 150 == 0:
            # reset time
            count = 0
            # 選項 B: 每次重置時生成新的隨機目標
            if count > 0:  # 跳過第一次
                new_goal = generate_random_ee_goals(1, device=sim.device)
                ee_goals[current_goal_idx] = new_goal[0]            
            # reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
            
            # 設置 link_1 (通常是 joint_1) 為負角度
            # 假設 joint_1 對應 link_1
            joint_1_idx = 1  # 根據您的機器人調整索引
            joint_pos[:, joint_1_idx] = -1.0  # 設置為負角度 (約 -30度)
            
            # 或者設置多個關節的初始位置
            custom_joint_positions = torch.tensor([
                0.0,   # joint_0
                -100.0,  # joint_1 (link_1) - 設為負角度
                0.0,   # joint_2
                0.0,  # joint_3
                0.0,   # joint_4
                0.0,   # joint_5
            ], device=robot.device)
            
            # 應用到所有環境
            if len(custom_joint_positions) <= joint_pos.shape[1]:
                joint_pos[:, :len(custom_joint_positions)] = custom_joint_positions
                            
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            # reset actions
            ik_commands[:] = ee_goals[current_goal_idx]
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            # reset controller
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)
            # change goal
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
        else:
            # obtain quantities from simulation
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = robot.data.root_state_w[:, 0:7]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # apply actions
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        # obtain quantities from simulation
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
