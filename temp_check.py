import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_extension_linkout_cone_orientation_cfg import TM5ExtensionLinkOutFanOrientationEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

env_cfg = TM5ExtensionLinkOutFanOrientationEnvCfg()
env_cfg.scene.num_envs = 1
env = ManagerBasedRLEnv(cfg=env_cfg)
robot = env.scene['robot']
print('Robot body names:', robot.body_names)

simulation_app.close()
