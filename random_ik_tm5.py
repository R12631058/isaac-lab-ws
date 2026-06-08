import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
from copy import deepcopy

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
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
from isaaclab_assets import TM5_700_CFG, TM5_700_HIGH_PD_CFG  # 添加 TM5 配置


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

    # articulation - 根據 robot 參數選擇不同的配置
    if args_cli.robot == "franka_panda":
        robot_cfg = deepcopy(FRANKA_PANDA_HIGH_PD_CFG)
        robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
        robot = robot_cfg
    elif args_cli.robot == "ur10":
        robot_cfg = deepcopy(UR10_CFG)
        robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
        robot = robot_cfg
    elif args_cli.robot == "tm5_700":
        # 使用新的高剛性 TM5 配置，類似 FRANKA_PANDA_HIGH_PD_CFG
        print("[INFO]: Using TM5_700_HIGH_PD_CFG with high stiffness (400.0) and damping (80.0)")
        robot_cfg = deepcopy(TM5_700_HIGH_PD_CFG)
        robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
        robot = robot_cfg
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10, tm5_700")


# Function to create scene config with selected robot - 不再需要
def create_scene_cfg():
    """Create scene configuration with selected robot."""
    return TableTopSceneCfg()


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers - 使用 deepcopy 創建標記配置
    ee_marker_cfg = deepcopy(FRAME_MARKER_CFG)
    ee_marker_cfg.prim_path = "/Visuals/ee_current"
    ee_marker = VisualizationMarkers(ee_marker_cfg)
    
    goal_marker_cfg = deepcopy(FRAME_MARKER_CFG)
    goal_marker_cfg.prim_path = "/Visuals/ee_goal"
    goal_marker = VisualizationMarkers(goal_marker_cfg)

    # 根據機器人類型定義不同的目標點
    if args_cli.robot == "tm5_700":
        # TM5-700 適合的工作空間目標 (調整到更近的可及範圍)
        ee_goals = [
            # [x, y, z, qx, qy, qz, qw] - 更保守的目標位置
            [0.3, 0.2, 0.4, 0.0, 0.707, 0.0, 0.707],    # 目標1：近距離右前方，向下
            [0.35, -0.15, 0.35, 0.0, 0.707, 0.0, 0.707], # 目標2：近距離右後方，向下
            [0.25, 0.0, 0.45, 0.0, 0.707, 0.0, 0.707],   # 目標3：正前方，向下
            [0.3, 0.1, 0.3, 0.707, 0.0, 0.707, 0.0],     # 目標4：側向姿態
            [0.28, -0.1, 0.4, 0.0, 0.5, 0.0, 0.866],     # 目標5：傾斜姿態 (60度)
        ]
    elif args_cli.robot == "franka_panda":
        # 原始的 Franka Panda 目標
        ee_goals = [
            [0.5, 0.5, 0.7, 0.707, 0, 0.707, 0],
            [0.5, -0.4, 0.6, 0.707, 0.707, 0.0, 0.0],
            [0.5, 0, 0.5, 0.0, 1.0, 0.0, 0.0],
        ]
    elif args_cli.robot == "ur10":
        # UR10 適合的目標
        ee_goals = [
            [0.6, 0.4, 0.8, 0.0, 0.707, 0.0, 0.707],
            [0.6, -0.3, 0.6, 0.707, 0.0, 0.707, 0.0],
            [0.4, 0.0, 0.9, 0.0, 1.0, 0.0, 0.0],
        ]
    else:
        # 預設目標
        ee_goals = [
            [0.5, 0.5, 0.7, 0.707, 0, 0.707, 0],
            [0.5, -0.4, 0.6, 0.707, 0.707, 0.0, 0.0],
            [0.5, 0, 0.5, 0.0, 1.0, 0.0, 0.0],
        ]
    
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    print(f"[INFO]: Using {len(ee_goals)} goals for {args_cli.robot}")
    for i, goal in enumerate(ee_goals):
        print(f"  Goal {i}: pos=[{goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f}], quat=[{goal[3]:.3f}, {goal[4]:.3f}, {goal[5]:.3f}, {goal[6]:.3f}]")
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
        if count % 300 == 0:
            # reset time
            count = 0
            # reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
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
    sim.set_camera_view((2.5, 2.5, 2.5), (0.0, 0.0, 0.0))
    # Design scene
    scene_cfg = create_scene_cfg()
    scene_cfg.num_envs = args_cli.num_envs
    scene_cfg.env_spacing = 2.0
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
