import math
import torch
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_assets import TM5_700_CFG


@configclass
class SurgeryRLSceneCfg(InteractiveSceneCfg):
    """手術室強化學習場景配置"""

    # 地板
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # 燈光
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9)),
    )

    # 機器人配置 - 使用標準 TM5-700
    robot = TM5_700_CFG.copy()
    robot.prim_path = "{ENV_REGEX_NS}/Robot"
    robot.init_state.joint_pos = {
        "joint_1": -1.57,
        "joint_2": -0.5,
        "joint_3": 0.0,
        "joint_4": 0.0,
        "joint_5": 0.0,
        "joint_6": 0.0,
    }


@configclass 
class SurgeryRLEnvCfg(ManagerBasedRLEnvCfg):
    """手術室強化學習環境配置"""

    # 場景設定
    scene: SurgeryRLSceneCfg = SurgeryRLSceneCfg(num_envs=1024, env_spacing=5.0)
    
    # 基本設定
    episode_length_s = 10.0  # 每個 episode 10 秒
    decimation = 2  # 控制頻率減半
    
    # 觀察空間配置
    observations = ObsGroup(
        policy=ObsGroup(
            # 機器人狀態
            joint_pos=ObsTerm(func=get_joint_positions),
            joint_vel=ObsTerm(func=get_joint_velocities, noise=Unoise(n_min=-0.01, n_max=0.01)),
            
            # 末端執行器狀態
            ee_pos=ObsTerm(func=get_ee_position),
            ee_quat=ObsTerm(func=get_ee_orientation),
            
            # 目標位置
            target_pos=ObsTerm(func=get_target_position),
            target_quat=ObsTerm(func=get_target_orientation),
            
            # 相對位置和距離
            ee_to_target=ObsTerm(func=get_ee_to_target_distance),
            
            # 避障信息（如果需要）
            # collision_info=ObsTerm(func=get_collision_info),
        )
    )
    
    # 動作空間（關節位置控制）
    actions = {
        "robot": {
            "joint_pos": SceneEntityCfg("robot", joint_names=["joint_.*"]),
        }
    }
    
    # 獎勵函數配置
    rewards = {
        # 主要任務獎勵
        "reach_target": RewTerm(func=reward_reach_target, weight=100.0),
        "ee_position_tracking": RewTerm(func=reward_ee_position_tracking, weight=50.0),
        "ee_orientation_tracking": RewTerm(func=reward_ee_orientation_tracking, weight=25.0),
        
        # 運動平滑性獎勵
        "joint_vel_l2": RewTerm(func=reward_joint_vel_l2, weight=-0.01),
        "action_smoothness": RewTerm(func=reward_action_smoothness, weight=-0.1),
        
        # 避障獎勵
        "collision_penalty": RewTerm(func=penalty_collision, weight=-10.0),
        "workspace_boundary": RewTerm(func=penalty_workspace_boundary, weight=-5.0),
        
        # 任務完成獎勵
        "task_completion": RewTerm(func=reward_task_completion, weight=200.0),
    }
    
    # 終止條件
    terminations = {
        "time_out": DoneTerm(func=time_out, time_out=True),
        "collision": DoneTerm(func=collision_termination),
        "workspace_exceeded": DoneTerm(func=workspace_boundary_termination),
        "joint_limits": DoneTerm(func=joint_limits_termination),
    }
    
    # 事件（重置、隨機化等）
    events = {
        "reset_robot_joints": EventTerm(
            func=reset_joints_by_scale,
            mode="reset",
            params={
                "position_range": (0.8, 1.2),
                "velocity_range": (0.0, 0.0),
            },
        ),
        "reset_target": EventTerm(
            func=reset_target_position,
            mode="reset",
            params={
                "workspace_bounds": {
                    'x': [0.3, 0.8],
                    'y': [-0.4, 0.4], 
                    'z': [0.2, 1.0]
                }
            },
        ),
    }


# 觀察函數
def get_joint_positions(env: ManagerBasedRLEnv) -> torch.Tensor:
    """獲取關節位置"""
    return env.scene["robot"].data.joint_pos

def get_joint_velocities(env: ManagerBasedRLEnv) -> torch.Tensor:
    """獲取關節速度"""
    return env.scene["robot"].data.joint_vel

def get_ee_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """獲取末端執行器位置"""
    robot = env.scene["robot"]
    # 假設使用 End_needle 或 flange 作為末端執行器
    ee_body_idx = robot.body_names.index("End_needle") if "End_needle" in robot.body_names else robot.body_names.index("flange")
    return robot.data.body_state_w[:, ee_body_idx, :3]

def get_ee_orientation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """獲取末端執行器姿態"""
    robot = env.scene["robot"]
    ee_body_idx = robot.body_names.index("End_needle") if "End_needle" in robot.body_names else robot.body_names.index("flange")
    return robot.data.body_state_w[:, ee_body_idx, 3:7]

def get_target_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """獲取目標位置"""
    return env.target_pos

def get_target_orientation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """獲取目標姿態"""
    return env.target_quat

def get_ee_to_target_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """計算末端執行器到目標的距離向量"""
    ee_pos = get_ee_position(env)
    target_pos = get_target_position(env)
    return target_pos - ee_pos


# 獎勵函數
def reward_reach_target(env: ManagerBasedRLEnv) -> torch.Tensor:
    """到達目標的獎勵"""
    ee_pos = get_ee_position(env)
    target_pos = get_target_position(env)
    distance = torch.norm(target_pos - ee_pos, dim=-1)
    
    # 距離越小獎勵越大
    reward = 1.0 / (1.0 + distance * 10.0)
    
    # 如果距離小於閾值，給予額外獎勵
    close_threshold = 0.05  # 5cm
    bonus = torch.where(distance < close_threshold, 2.0, 0.0)
    
    return reward + bonus

def reward_ee_position_tracking(env: ManagerBasedRLEnv) -> torch.Tensor:
    """末端執行器位置追蹤獎勵"""
    ee_pos = get_ee_position(env)
    target_pos = get_target_position(env)
    position_error = torch.norm(target_pos - ee_pos, dim=-1)
    
    # 指數衰減獎勵
    return torch.exp(-position_error * 5.0)

def reward_ee_orientation_tracking(env: ManagerBasedRLEnv) -> torch.Tensor:
    """末端執行器姿態追蹤獎勵"""
    ee_quat = get_ee_orientation(env)
    target_quat = get_target_orientation(env)
    
    # 計算四元數差異
    quat_error = 1.0 - torch.abs(torch.sum(ee_quat * target_quat, dim=-1))
    return torch.exp(-quat_error * 2.0)

def reward_joint_vel_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """關節速度 L2 懲罰（鼓勵平滑運動）"""
    joint_vel = get_joint_velocities(env)
    return -torch.sum(joint_vel**2, dim=-1)

def reward_action_smoothness(env: ManagerBasedRLEnv) -> torch.Tensor:
    """動作平滑性獎勵"""
    if hasattr(env, 'previous_actions'):
        current_actions = env.actions["robot"]["joint_pos"]
        action_diff = torch.norm(current_actions - env.previous_actions, dim=-1)
        env.previous_actions = current_actions.clone()
        return -action_diff
    else:
        env.previous_actions = env.actions["robot"]["joint_pos"].clone()
        return torch.zeros(env.num_envs, device=env.device)

def penalty_collision(env: ManagerBasedRLEnv) -> torch.Tensor:
    """碰撞懲罰"""
    # 這裡需要實現碰撞檢測邏輯
    # 可以使用 Isaac Lab 的碰撞檢測功能
    robot = env.scene["robot"]
    
    # 簡單的自碰撞檢測（關節限制）
    joint_pos = robot.data.joint_pos
    collision_penalty = torch.zeros(env.num_envs, device=env.device)
    
    # 檢查是否超出關節限制
    for i, joint_name in enumerate(robot.joint_names):
        if joint_name in ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]:
            # TM5-700 關節限制（根據實際規格調整）
            if "joint_2" in joint_name:
                # joint_2 限制在 -90° 到 +90°
                out_of_bounds = torch.logical_or(joint_pos[:, i] < -1.57, joint_pos[:, i] > 1.57)
                collision_penalty += torch.where(out_of_bounds, 1.0, 0.0)
    
    return collision_penalty

def penalty_workspace_boundary(env: ManagerBasedRLEnv) -> torch.Tensor:
    """工作空間邊界懲罰"""
    ee_pos = get_ee_position(env)
    
    # 定義工作空間邊界
    workspace_bounds = {
        'x': [0.2, 0.9],
        'y': [-0.5, 0.5],
        'z': [0.1, 1.2]
    }
    
    penalty = torch.zeros(env.num_envs, device=env.device)
    
    # 檢查 X 軸邊界
    penalty += torch.where(ee_pos[:, 0] < workspace_bounds['x'][0], 1.0, 0.0)
    penalty += torch.where(ee_pos[:, 0] > workspace_bounds['x'][1], 1.0, 0.0)
    
    # 檢查 Y 軸邊界
    penalty += torch.where(ee_pos[:, 1] < workspace_bounds['y'][0], 1.0, 0.0)
    penalty += torch.where(ee_pos[:, 1] > workspace_bounds['y'][1], 1.0, 0.0)
    
    # 檢查 Z 軸邊界
    penalty += torch.where(ee_pos[:, 2] < workspace_bounds['z'][0], 1.0, 0.0)
    penalty += torch.where(ee_pos[:, 2] > workspace_bounds['z'][1], 1.0, 0.0)
    
    return penalty

def reward_task_completion(env: ManagerBasedRLEnv) -> torch.Tensor:
    """任務完成獎勵"""
    ee_pos = get_ee_position(env)
    target_pos = get_target_position(env)
    distance = torch.norm(target_pos - ee_pos, dim=-1)
    
    # 如果距離小於閾值，認為任務完成
    completion_threshold = 0.02  # 2cm
    completed = distance < completion_threshold
    
    return torch.where(completed, 1.0, 0.0)


# 終止條件函數
def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """時間超時"""
    return env.episode_length_buf >= env.max_episode_length

def collision_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """碰撞終止"""
    collision_penalty = penalty_collision(env)
    return collision_penalty > 0.5

def workspace_boundary_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """工作空間邊界終止"""
    boundary_penalty = penalty_workspace_boundary(env)
    return boundary_penalty > 0.5

def joint_limits_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """關節限制終止"""
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos
    
    # 檢查關節是否超出安全範圍
    joint_limits_exceeded = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    
    for i, joint_name in enumerate(robot.joint_names):
        if "joint" in joint_name:
            # 通用關節限制檢查
            if torch.any(torch.abs(joint_pos[:, i]) > 3.14):  # 超出 ±180°
                joint_limits_exceeded = torch.logical_or(joint_limits_exceeded, torch.abs(joint_pos[:, i]) > 3.14)
    
    return joint_limits_exceeded


# 事件函數
def reset_joints_by_scale(env: ManagerBasedRLEnv, env_ids: torch.Tensor, position_range: tuple, velocity_range: tuple):
    """重置關節位置"""
    robot = env.scene["robot"]
    
    # 隨機化關節位置
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_pos += torch.rand_like(joint_pos) * 0.1 - 0.05  # ±0.05 弧度的隨機化
    
    # 確保 joint_1 偏向負角度
    joint_1_idx = robot.joint_names.index("joint_1")
    joint_pos[:, joint_1_idx] = -1.57 + torch.rand(len(env_ids), device=env.device) * 0.5 - 0.25
    
    joint_vel = torch.zeros_like(joint_pos)
    
    # 設置關節狀態
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids)

def reset_target_position(env: ManagerBasedRLEnv, env_ids: torch.Tensor, workspace_bounds: dict):
    """重置目標位置"""
    num_resets = len(env_ids)
    
    # 生成隨機目標位置
    target_pos = torch.zeros(num_resets, 3, device=env.device)
    target_pos[:, 0] = torch.rand(num_resets, device=env.device) * (workspace_bounds['x'][1] - workspace_bounds['x'][0]) + workspace_bounds['x'][0]
    target_pos[:, 1] = torch.rand(num_resets, device=env.device) * (workspace_bounds['y'][1] - workspace_bounds['y'][0]) + workspace_bounds['y'][0]
    target_pos[:, 2] = torch.rand(num_resets, device=env.device) * (workspace_bounds['z'][1] - workspace_bounds['z'][0]) + workspace_bounds['z'][0]
    
    # 生成隨機目標姿態（主要是向下看的姿態）
    target_quat = torch.zeros(num_resets, 4, device=env.device)
    target_quat[:, :] = torch.tensor([0.0, 0.707, 0.0, 0.707], device=env.device)  # 向下看
    
    # 保存到環境
    if not hasattr(env, 'target_pos'):
        env.target_pos = torch.zeros(env.num_envs, 3, device=env.device)
        env.target_quat = torch.zeros(env.num_envs, 4, device=env.device)
    
    env.target_pos[env_ids] = target_pos
    env.target_quat[env_ids] = target_quat