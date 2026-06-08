# 最簡化的手術 RL 訓練腳本 - 僅測試環境建立
# 執行方式: C:\Users\RMML\anaconda3\envs\env_isaaclab\python.exe scripts\isaaclab_ws\surgery_rl\train_surgery_test.py --num_envs 1

import argparse
import torch

from isaaclab.app import AppLauncher

# 添加參數
parser = argparse.ArgumentParser(description="Test Surgery RL Environment")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 啟動 app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 導入 Isaac Lab 模組
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets import TM5_700_CFG


##
# 場景配置
##

@configclass
class SimpleSurgerySceneCfg(InteractiveSceneCfg):
    """最簡單的手術場景"""

    # 地板
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # 機器人
    robot: ArticulationCfg = TM5_700_CFG
    robot.prim_path = "{ENV_REGEX_NS}/Robot"


##
# 環境配置
##

@configclass
class SimpleSurgeryEnvCfg(ManagerBasedEnvCfg):
    """最簡單的環境配置"""

    # 場景
    scene: SimpleSurgerySceneCfg = SimpleSurgerySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)

    # 基本設定
    episode_length_s = 10.0
    decimation = 2


def main():
    """測試環境建立"""

    # 創建環境配置
    env_cfg = SimpleSurgeryEnvCfg()

    # 創建環境
    print("Creating environment...")
    env = ManagerBasedEnv(cfg=env_cfg)

    print(f"✅ Environment created successfully!")
    print(f"   Number of environments: {env.num_envs}")
    print(f"   Scene entities: {list(env.scene.keys())}")

    # 重置環境
    print("\nResetting environment...")
    obs, _ = env.reset()
    print("✅ Environment reset successful!")

    # 執行幾步
    print("\nRunning simulation for 100 steps...")
    for i in range(100):
        # 保持當前關節位置
        actions = torch.zeros((env.num_envs, env.scene["robot"].num_joints), device=env.device)
        obs, _ = env.step(actions)

        if i % 20 == 0:
            print(f"   Step {i}/100")

    print("\n✅ Test completed successfully!")

    # 關閉環境
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
