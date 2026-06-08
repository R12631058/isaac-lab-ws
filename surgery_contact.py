# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to load surgery room environments with existing TM5-700 robot control.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/isaaclab_ws/surgery_contact.py --num_envs 4

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Load multiple surgery room USD files with robot control.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import numpy as np
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, Articulation, ArticulationCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms
from isaacsim.util.debug_draw import _debug_draw

##
# Pre-defined configs
##
from isaaclab_assets import TM5_700_CFG


@configclass
class SurgeryRoomSceneCfg(InteractiveSceneCfg):
    """Configuration for the surgery room scene with existing TM5-700 robot and needle."""

    # 從手術室 USD 中讀取既有的 TM5-700 機器人（包含 needle）
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/SurgeryRoom/robotarm_base/tm5_700",
        spawn=None,  # 不生成新的，使用既有的
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),  # 保持 USD 中的原始位置
            rot=(1.0, 0.0, 0.0, 0.0),
            # 明確設定關節初始位置 - 覆寫 USD 中的值
            joint_pos={
                "joint_1": -1.57,  # -90度，讓手臂傾向負方向
                "joint_2": 0.0,
                "joint_3": 0.0,
                "joint_4": 0.0,
                "joint_5": 0.0,
                "joint_6": 0.0,
            },
            # 設定初始關節速度為零
            joint_vel={
                "joint_1": 0.0,
                "joint_2": 0.0,
                "joint_3": 0.0,
                "joint_4": 0.0,
                "joint_5": 0.0,
                "joint_6": 0.0,
            },
        ),
        actuators=TM5_700_CFG.actuators,  # 使用 TM5-700 的致動器配置
        # 注意：如果 needle 是連接到 flange 的，Isaac Lab 應該自動包含它
    )


def design_scene(num_envs: int, env_spacing: float = 5.0):
    """手動設計場景，載入多個手術室 USD 檔案"""
    
    # 創建地面平台
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    
    # 創建照明
    light_cfg = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    light_cfg.func("/World/lightDistant", light_cfg)
    
    # 計算環境排列（網格佈局）
    envs_per_row = int(torch.sqrt(torch.tensor(num_envs, dtype=torch.float32)).ceil().int())
    
    for env_idx in range(num_envs):
        # 計算環境位置
        row = env_idx // envs_per_row
        col = env_idx % envs_per_row
        
        # 計算偏移量
        x_offset = (col - (envs_per_row - 1) / 2) * env_spacing
        y_offset = (row - (envs_per_row - 1) / 2) * env_spacing
        
        # 載入手術室 USD 檔案
        surgery_room_cfg = sim_utils.UsdFileCfg(
            usd_path="C:/Nick/surgery_team/surgery_team/USD/surgeryroom_isaaclab.usd"
        )
        
        # 使用 func 方法載入到特定位置
        surgery_room_cfg.func(
            f"/World/envs/env_{env_idx}/SurgeryRoom",
            surgery_room_cfg,
            translation=(x_offset, y_offset, 0.0)
        )
        
        print(f"[INFO]: Loaded surgery room {env_idx} at position ({x_offset:.1f}, {y_offset:.1f}, 0.0)")
        print(f"[INFO]: Expected robot path: /World/envs/env_{env_idx}/SurgeryRoom/robotarm_base/tm5_700")


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
        # TM5-700 工作空間邊界（相對於機器人基座）
        workspace_bounds = {
            'x': [0.3, 0.7],    # X 軸範圍
            'y': [-0.5, 0.5],   # Y 軸範圍  
            'z': [0.2, 0.8]     # Z 軸範圍
        }
    
    goals = torch.zeros(num_goals, 7, device=device)
    
    # 隨機位置
    goals[:, 0] = torch.rand(num_goals, device=device) * (workspace_bounds['x'][1] - workspace_bounds['x'][0]) + workspace_bounds['x'][0]  # X
    goals[:, 1] = torch.rand(num_goals, device=device) * (workspace_bounds['y'][1] - workspace_bounds['y'][0]) + workspace_bounds['y'][0]  # Y
    goals[:, 2] = torch.rand(num_goals, device=device) * (workspace_bounds['z'][1] - workspace_bounds['z'][0]) + workspace_bounds['z'][0]  # Z
    
    # 隨機方向（四元數）- 適合手術室操作的方向
    orientations = [
        #[0.0, 0.707, 0.0, 0.707],      # 向下看（手術姿態）
        [0.707, 0.0, 0.707, 0.0],      # 側向
        #[0.0, 1.0, 0.0, 0.0],          # 翻轉
        #[0.5, 0.5, 0.5, 0.5],          # 45度角
        #[0.0, 0.0, 0.0, 1.0],          # 無旋轉
    ]
    
    for i in range(num_goals):
        # 隨機選擇一個方向
        orient_idx = torch.randint(0, len(orientations), (1,)).item()
        goals[i, 3:7] = torch.tensor(orientations[orient_idx], device=device)
    
    return goals


def draw_workspace_bounds(draw_interface, workspace_bounds, robot_base_pos=(0, 0, 0), color=(0, 1, 0, 1)):
    """
    使用 DebugDraw 繪製工作空間邊界框
    
    Args:
        draw_interface: DebugDrawerInterface 實例
        workspace_bounds: 工作空間邊界字典 {'x': [min, max], 'y': [min, max], 'z': [min, max]}
        robot_base_pos: 機器人基座位置 (x, y, z)
        color: 線條顏色 (r, g, b, a)
    """
    # 計算8個角點 (相對於機器人基座)
    x_min, x_max = workspace_bounds['x']
    y_min, y_max = workspace_bounds['y']
    z_min, z_max = workspace_bounds['z']
    
    # 轉換到世界座標
    base_x, base_y, base_z = robot_base_pos
    
    corners = [
        (base_x + x_min, base_y + y_min, base_z + z_min),  # 0: 左下前
        (base_x + x_max, base_y + y_min, base_z + z_min),  # 1: 右下前
        (base_x + x_max, base_y + y_max, base_z + z_min),  # 2: 右下後
        (base_x + x_min, base_y + y_max, base_z + z_min),  # 3: 左下後
        (base_x + x_min, base_y + y_min, base_z + z_max),  # 4: 左上前
        (base_x + x_max, base_y + y_min, base_z + z_max),  # 5: 右上前
        (base_x + x_max, base_y + y_max, base_z + z_max),  # 6: 右上後
        (base_x + x_min, base_y + y_max, base_z + z_max),  # 7: 左上後
    ]
    
    # 繪製底面 (z_min)
    draw_interface.draw_lines([corners[0]], [corners[1]], [color], [1])
    draw_interface.draw_lines([corners[1]], [corners[2]], [color], [1])
    draw_interface.draw_lines([corners[2]], [corners[3]], [color], [1])
    draw_interface.draw_lines([corners[3]], [corners[0]], [color], [1])
    
    # 繪製頂面 (z_max)
    draw_interface.draw_lines([corners[4]], [corners[5]], [color], [1])
    draw_interface.draw_lines([corners[5]], [corners[6]], [color], [1])
    draw_interface.draw_lines([corners[6]], [corners[7]], [color], [1])
    draw_interface.draw_lines([corners[7]], [corners[4]], [color], [1])
    
    # 繪製垂直邊
    draw_interface.draw_lines([corners[0]], [corners[4]], [color], [1])
    draw_interface.draw_lines([corners[1]], [corners[5]], [color], [1])
    draw_interface.draw_lines([corners[2]], [corners[6]], [color], [1])
    draw_interface.draw_lines([corners[3]], [corners[7]], [color], [1])
    
    print(f"[INFO]: Drew workspace bounds: X[{x_min:.2f}, {x_max:.2f}], Y[{y_min:.2f}, {y_max:.2f}], Z[{z_min:.2f}, {z_max:.2f}]")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """運行模擬器的主迴圈，包含 TM5-700 控制"""
    
    # 初始化 Debug Draw Interface
    draw_interface = _debug_draw.acquire_debug_draw_interface()
    
    # Extract scene entities - 嘗試獲取機器人
    try:
        robot = scene["robot"]
        robot_available = True
        print(f"[INFO]: Found TM5-700 robot in surgery room with {robot.num_instances} instances")
        print(f"[INFO]: Robot prim path: {robot.cfg.prim_path}")
        print(f"[INFO]: Available body names: {robot.body_names}")
    except KeyError:
        print("[WARNING]: TM5-700 robot not found in scene. Running without robot control.")
        print("[INFO]: Available scene entities:", list(scene.keys()))
        robot_available = False
        robot = None
    
    # 如果有機器人，設置控制器
    if robot_available:
        # === 設定 IK 控制器與關節限制 ===
        print("[INFO]: Initializing Differential IK Controller with joint constraints...")
        
        # === 設定固定 IK 解模式 (類似 CGA [1,1,1] 配置) ===
        # 由於機械臂有掛件,需要固定特定的關節配置模式以避免碰撞
        
        # 設定偏好的關節配置 - 這是"安全"的配置,不會導致掛件碰撞
        preferred_joint_config = torch.tensor([
            -1.57,  # joint_1: -90度 (固定模式 1)
            0.0,    # joint_2: 0度   (固定模式 1)  
            1.57,   # joint_3: 90度  (固定模式 1)
            0.0,    # joint_4: 0度
            0.0,    # joint_5: 0度
            3.14,   # joint_6: 180度 (讓最後關節往另一側,避免掛件撞到自身)
        ], device=sim.device)
        
        # 初始化 Differential IK Controller with 強阻尼
        # 使用較大的 damping 讓 IK 更傾向維持當前配置
        # 減小 k_val 讓運動更平滑,接近直線運動
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", 
            use_relative_mode=False, 
            ik_method="dls",  # Damped Least Squares
            ik_params={
                "k_val": 0.5,  # IK 增益降低,讓運動更平滑更接近直線
                "lambda_val": 0.15,  # DLS damping - 更高的阻尼維持當前配置
            }
        )
        diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
        
        print("[INFO]: IK solver configured for fixed joint pattern (CGA-like mode [1,1,1]):")
        print(f"  Preferred config (deg): [{torch.rad2deg(preferred_joint_config[0]):.1f}, {torch.rad2deg(preferred_joint_config[1]):.1f}, {torch.rad2deg(preferred_joint_config[2]):.1f}, {torch.rad2deg(preferred_joint_config[3]):.1f}, {torch.rad2deg(preferred_joint_config[4]):.1f}, {torch.rad2deg(preferred_joint_config[5]):.1f}]")
        print(f"  This configuration avoids collisions with robot attachments")

        # Markers for needle visualization
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)  # 更小的標記適合針
        needle_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/needle_current"))
        goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/needle_goal"))

        # === 隨機目標生成 ===
        num_random_goals = 5
        workspace_bounds = {
            'x': [0.5, 1.0],    # 根據手術室調整
            'y': [-0.75, -0.5], 
            'z': [1.1, 1.5]
        }
        ee_goals = generate_random_ee_goals(
            num_random_goals, 
            device=sim.device,
            workspace_bounds=workspace_bounds
        )
        
        print(f"Generated {num_random_goals} random needle goals for surgery room:")
        for i, goal in enumerate(ee_goals):
            print(f"  Goal {i}: pos={goal[:3]}, quat={goal[3:]}")
        
        # 繪製工作空間邊界
        # 假設機器人基座在環境原點
        robot_base_pos = scene.env_origins[0].cpu().numpy()
        draw_workspace_bounds(
            draw_interface, 
            workspace_bounds, 
            robot_base_pos=tuple(robot_base_pos),
            color=(0, 1, 0, 1)  # 綠色
        )

        # Track the given command
        current_goal_idx = 0
        ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
        ik_commands[:] = ee_goals[current_goal_idx]

        # TM5-700 with needle specific parameters - 修正順序
        try:
            # 首先嘗試使用 End_needle 作為目標身體
            robot_entity_cfg = SceneEntityCfg("robot", joint_names=["joint_.*"], body_names=["End_needle"])
            robot_entity_cfg.resolve(scene)
            print(f"[INFO]: Found End_needle body - Robot joint IDs: {robot_entity_cfg.joint_ids}")
            print(f"[INFO]: Found End_needle body - Robot body IDs: {robot_entity_cfg.body_ids}")
            print(f"[INFO]: Using End_needle as target end-effector")
            target_body_name = "End_needle"
        except Exception as e:
            print(f"[WARNING]: Failed to find End_needle body: {e}")
            
            # 備用方案：嘗試使用 flange
            try:
                robot_entity_cfg = SceneEntityCfg("robot", joint_names=["joint_.*"], body_names=["flange"])
                robot_entity_cfg.resolve(scene)
                print(f"[INFO]: Fallback to flange - Robot joint IDs: {robot_entity_cfg.joint_ids}")
                print(f"[INFO]: Fallback to flange - Robot body IDs: {robot_entity_cfg.body_ids}")
                print(f"[WARNING]: Using flange instead of End_needle")
                target_body_name = "flange"
            except Exception as e2:
                print(f"[ERROR]: Failed to resolve any body: {e2}")
                # 使用所有關節和第一個身體作為最後備用方案
                robot_entity_cfg = SceneEntityCfg("robot", joint_names=robot.joint_names, body_names=[robot.body_names[0]])
                robot_entity_cfg.resolve(scene)
                print(f"[WARNING]: Using first available body: {robot.body_names[0]}")
                target_body_name = robot.body_names[0]
        
        # Obtain the frame index of the end-effector (針)
        if robot.is_fixed_base:
            needle_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
        else:
            needle_jacobi_idx = robot_entity_cfg.body_ids[0]
        
        print(f"[INFO]: Using body '{target_body_name}' for control with jacobian index {needle_jacobi_idx}")
    
    # 模擬參數
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    
    print(f"[INFO]: Scene ready with {scene.num_envs} surgery room environments")
    if robot_available:
        print(f"[INFO]: TM5-700 robot with {target_body_name} control enabled")
    
    # 模擬迴圈
    while simulation_app.is_running():
        # 機器人控制邏輯
        if robot_available:
            # reset every 150 frames
            if count % 150 == 0:
                # reset time
                count = 0
                # 生成新的隨機目標
                if count > 0:  # 跳過第一次
                    new_goal = generate_random_ee_goals(1, device=sim.device)
                    ee_goals[current_goal_idx] = new_goal[0]
                
                # === 重置到偏好的安全配置 (類似 CGA 固定解) ===
                # 將機器人重置到偏好配置而不是預設配置
                joint_pos = robot.data.default_joint_pos.clone()
                # 覆寫為偏好配置
                joint_pos[:, robot_entity_cfg.joint_ids] = preferred_joint_config.unsqueeze(0).expand(joint_pos.shape[0], -1)
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
                print(f"[INFO]: Switching {target_body_name} to goal {current_goal_idx}: {ee_goals[current_goal_idx][:3]}")
            else:
                # obtain quantities from simulation - 針相關
                jacobian = robot.root_physx_view.get_jacobians()[:, needle_jacobi_idx, :, robot_entity_cfg.joint_ids]
                needle_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
                root_pose_w = robot.data.root_state_w[:, 0:7]
                joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
                
                # compute frame in root frame - 針位置
                needle_pos_b, needle_quat_b = subtract_frame_transforms(
                    root_pose_w[:, 0:3], root_pose_w[:, 3:7], needle_pose_w[:, 0:3], needle_pose_w[:, 3:7]
                )
                
                # === 計算 IK 解 ===
                joint_pos_des = diff_ik_controller.compute(needle_pos_b, needle_quat_b, jacobian, joint_pos)
                
                # === 方案 C: 直接鎖定特定關節 (模擬 CGA [1,1,1] 固定解) ===
                # 類似 Lula 的 cga_config_params = [1, 1, 1]
                # 由於初始姿態已設計好,只需直線運動 + 微調即可抵達
                # 因此使用極強的鎖定來避免 IK 計算出複雜路徑
                
                # 強力鎖定的關節 (CGA [1,1,1] 模式)
                cga_locked_joints = [0, 1, 2]  # joint_1, joint_2, joint_3
                cga_blend_factor = 0.995  # 99.5% 鎖定,幾乎固定
                
                # 將指定關節強力鎖定到偏好配置
                for joint_idx in cga_locked_joints:
                    joint_pos_des[:, joint_idx] = (
                        cga_blend_factor * preferred_joint_config[joint_idx] + 
                        (1 - cga_blend_factor) * joint_pos_des[:, joint_idx]
                    )
                
                # 引導 joint_5 往安全方向 (180度,避免掛件碰撞)
                # 增加混合因子讓 joint_5 也更傾向保持 180 度
                joint_5_blend_factor = 0.85  # 85% 偏好, 15% IK
                joint_pos_des[:, 5] = (
                    joint_5_blend_factor * preferred_joint_config[5] + 
                    (1 - joint_5_blend_factor) * joint_pos_des[:, 5]
                )

            # apply actions
            robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        
        # 執行模擬步驟
        scene.write_data_to_sim()
        sim.step()
        
        # 更新時間
        sim_time += sim_dt
        count += 1
        
        # 更新場景
        scene.update(sim_dt)
        
        # 更新可視化標記 - 針位置
        if robot_available:
            needle_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            needle_marker.visualize(needle_pose_w[:, 0:3], needle_pose_w[:, 3:7])
            goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])
        
        # 每 50 幀輸出一次資訊 (更頻繁以便調試)
        if count % 50 == 0:
            if robot_available:
                current_needle_pos = robot.data.body_state_w[0, robot_entity_cfg.body_ids[0], 0:3]
                current_goal = ik_commands[0, 0:3] + scene.env_origins[0]
                distance = torch.norm(current_needle_pos - current_goal)
                current_joints = robot.data.joint_pos[0, robot_entity_cfg.joint_ids]
                print(f"[INFO][Frame {count}] Distance: {distance:.4f}m | Needle: [{current_needle_pos[0]:.3f}, {current_needle_pos[1]:.3f}, {current_needle_pos[2]:.3f}]")
                print(f"[INFO][Frame {count}] Goal: [{current_goal[0]:.3f}, {current_goal[1]:.3f}, {current_goal[2]:.3f}] | Joints (deg): [{torch.rad2deg(current_joints[0]):.1f}, {torch.rad2deg(current_joints[1]):.1f}, {torch.rad2deg(current_joints[2]):.1f}, {torch.rad2deg(current_joints[3]):.1f}, {torch.rad2deg(current_joints[4]):.1f}, {torch.rad2deg(current_joints[5]):.1f}]")

def main():
    """Main function."""
    
    # 解析環境數量
    num_envs = args_cli.num_envs
    env_spacing = 8.0  # 環境間距
    
    # 建立模擬上下文
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # 手動設計場景並載入多個手術室
    design_scene(num_envs, env_spacing)
    
    # 建立場景配置（包含機器人）
    scene_cfg = SurgeryRoomSceneCfg(num_envs=num_envs, env_spacing=env_spacing)
    
    # 建立互動場景
    scene = InteractiveScene(scene_cfg)
    
    # 設置攝影機位置 - 調整以觀察多個環境
    if num_envs == 1:
        sim.set_camera_view(eye=[8.0, 8.0, 6.0], target=[0.0, 0.0, 1.0])
    elif num_envs <= 4:
        sim.set_camera_view(eye=[15.0, 15.0, 10.0], target=[0.0, 0.0, 1.0])
    else:
        sim.set_camera_view(eye=[25.0, 25.0, 15.0], target=[0.0, 0.0, 1.0])

    # 重置模擬
    sim.reset()
    
    print(f"[INFO]: Setup complete with {num_envs} surgery room environments!")
    print(f"[INFO]: Environment spacing: {env_spacing}m")
    
    # 運行模擬
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()