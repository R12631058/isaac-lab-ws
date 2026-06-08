import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_extension_linkout_cone_orientation_cfg import TM5ExtensionLinkOutFanOrientationEnvCfg

env_cfg = TM5ExtensionLinkOutFanOrientationEnvCfg()
env_cfg.scene.num_envs = 2
env = ManagerBasedRLEnv(cfg=env_cfg)
env.reset()

robot = env.scene["robot"]
print("env_origins:\n", env.scene.env_origins)
print("root_pos_w:\n", robot.data.root_pos_w)
print("difference:\n", robot.data.root_pos_w - env.scene.env_origins)

env.close()
simulation_app.close()
