import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom
import numpy as np

# 1. Get USD base position
print("\n" + "="*50)
print("=== USD Base Position ===")
usd_path = r"C:\Nick\surgery_team\surgery_team\USD\animation\isaaclab_multi_env.usd"
stage = Usd.Stage.Open(usd_path)
if stage:
    prim = stage.GetPrimAtPath("/Root/robotarm_base")
    if prim.IsValid():
        xform = UsdGeom.Xformable(prim)
        world_transform = xform.ComputeLocalToWorldTransform(0.0)
        translation = world_transform.ExtractTranslation()
        print(f"/Root/robotarm_base absolute translation: {translation}")
    else:
        print("Prim /Root/robotarm_base not found!")
else:
    print("Failed to open USD")

# 2. Get env origins + offsets
print("\n=== Environment Origins and Grid Offsets ===")
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_extension_linkout_cone_orientation_cfg import TM5ExtensionLinkOutFanOrientationEnvCfg

x_vals = np.linspace(0.1, 0.7, 7)
y_vals = np.linspace(0.0, 0.3, 4)
X, Y = np.meshgrid(x_vals, y_vals)
X_flat = X.flatten()
Y_flat = Y.flatten()
num_envs = len(X_flat)

env_cfg = TM5ExtensionLinkOutFanOrientationEnvCfg()
env_cfg.scene.num_envs = num_envs
env = ManagerBasedRLEnv(cfg=env_cfg)

origins = env.scene.env_origins.cpu().numpy()

print(f"{'Env ID':<6} | {'Env Origin (X, Y, Z)':<25} | {'Offset (X, Y)':<15} | {'Absolute Base Pos (X, Y, Z)':<25}")
print("-" * 80)
for i in range(num_envs):
    ox, oy, oz = origins[i]
    dx, dy = X_flat[i], Y_flat[i]
    ax, ay, az = ox + dx, oy + dy, oz
    print(f"{i:<6} | {ox:>7.3f}, {oy:>7.3f}, {oz:>7.3f} | {dx:>7.3f}, {dy:>7.3f} | {ax:>7.3f}, {ay:>7.3f}, {az:>7.3f}")
print("="*50 + "\n")

simulation_app.close()
