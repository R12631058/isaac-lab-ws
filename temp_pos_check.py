import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
sim_app = app_launcher.app

from pxr import Usd, UsdGeom
print("\n=== USD File Positions (isaaclab_multi_env.usd) ===")
usd_path = r"C:\Nick\surgery_team\surgery_team\USD\animation\isaaclab_multi_env.usd"
try:
    stage = Usd.Stage.Open(usd_path)
    if stage:
        for prim in stage.Traverse():
            if "robotarm_base" in prim.GetName():
                xform = UsdGeom.Xformable(prim)
                if xform:
                    tf = xform.ComputeLocalToWorldTransform(0.0)
                    trans = tf.ExtractTranslation()
                    print(f"Path: {prim.GetPath()} | World Pos: [{trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f}]")
except Exception as e:
    print("Error:", e)

print("\n=== Env Setup Positions ===")
from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_extension_linkout_cone_orientation_cfg import TM5ExtensionLinkOutFanOrientationEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
env_cfg = TM5ExtensionLinkOutFanOrientationEnvCfg()
env_cfg.scene.num_envs = 2 # Check just 2
env_cfg.commands.ee_pose.resampling_time_range = (1e10, 1e10)
env = ManagerBasedRLEnv(cfg=env_cfg)
print("Env Origins:", env.scene.env_origins)
robot = env.scene['robot']
print("Robot Initial Root Pos (local to world):", robot.data.default_root_state[:, :3])
sim_app.close()
