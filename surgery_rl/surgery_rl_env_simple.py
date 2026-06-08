# 簡化版手術 RL 環境 - 參考 Isaac Lab reach 任務結構
# 執行方式: C:\Users\RMML\anaconda3\envs\env_isaaclab\python.exe scripts\isaaclab_ws\surgery_rl\train_surgery_simple.py --num_envs 4

import torch
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab_assets import TM5_700_CFG

##
# 場景配置
##

@configclass
class SurgeryRLSceneCfg(InteractiveSceneCfg):
    """手術室 RL 場景"""

    # 地板
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # 燈光
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    # 機器人 (使用 TM5-700)
    robot = TM5_700_CFG.copy()
    robot.prim_path = "{ENV_REGEX_NS}/Robot"


##
# 環境配置
##

@configclass
class SurgeryRLEnvCfg(ManagerBasedRLEnvCfg):
    """手術室 RL 環境配置"""

    # 場景
    scene: SurgeryRLSceneCfg = SurgeryRLSceneCfg(num_envs=512, env_spacing=2.0)

    # 基本設定
    episode_length_s = 10.0
    decimation = 2

    # 觀察空間
    observations = ObsGroup(
        policy=ObsGroup(
            joint_pos=ObsTerm(func=lambda env: env.scene["robot"].data.joint_pos),
            joint_vel=ObsTerm(func=lambda env: env.scene["robot"].data.joint_vel),
        )
    )

    # 動作空間
    actions = {
        "robot_joint_pos": ActionTerm(
            func=lambda env, action: env.scene["robot"].set_joint_position_target(action),
            params={},
        ),
    }

    # 獎勵
    rewards = {
        "alive": RewTerm(func=lambda env: torch.ones(env.num_envs, device=env.device), weight=1.0),
    }

    # 終止條件
    terminations = {
        "time_out": DoneTerm(func=lambda env: env.episode_length_buf >= env.max_episode_length, time_out=True),
    }

    # 事件
    events = {}
