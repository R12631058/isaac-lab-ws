import argparse
import torch
import numpy as np
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)

import omni.usd
from pxr import UsdGeom
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_extension_linkout_cone_orientation_cfg import TM5ExtensionLinkOutFanOrientationEnvCfg

env_cfg = TM5ExtensionLinkOutFanOrientationEnvCfg()
env_cfg.scene.num_envs = 2
env = ManagerBasedRLEnv(cfg=env_cfg)
env.reset()

robot = env.scene["robot"]
waypoints = []
stage = omni.usd.get_context().get_stage()

for i in range(1, 10):
    name = f"Object_{i:03d}"
    prim = stage.GetPrimAtPath(f"/Root/waypoint/{name}")
    if prim.IsValid():
        xform = UsdGeom.Xformable(prim)
        world_transform = xform.ComputeLocalToWorldTransform(0.0)
        waypoints.append((name, np.array(world_transform.ExtractTranslation())))

print("\n" + "="*80)
print("             🌍 COORDINATE SYSTEM ANALYSIS 🌍")
print("="*80)

print("\n1. Environment Clone Origins (World Coordinates):")
for i, orig in enumerate(env.scene.env_origins):
    print(f"   Env {i} Origin  : X = {orig[0]:.4f}, Y = {orig[1]:.4f}, Z = {orig[2]:.4f}")

print("\n2. Default Robot Root Positions (World Coordinates):")
for i, pos in enumerate(robot.data.root_pos_w):
    print(f"   Env {i} Robot   : X = {pos[0]:.4f}, Y = {pos[1]:.4f}, Z = {pos[2]:.4f}")

print("\n3. Default Robot Local Offsets (World - Env Origin):")
print("   * This is exactly the transform you see in the Isaac Sim Inspector for the root ")
for i in range(2):
    local_pos = robot.data.root_pos_w[i] - env.scene.env_origins[i]
    print(f"   Env {i} Local   : X = {local_pos[0]:.4f}, Y = {local_pos[1]:.4f}, Z = {local_pos[2]:.4f}")

print("\n4. Sample Waypoints extracted from '/Root/waypoint/' (World Coordinates):")
for name, pos in waypoints[:3]:
    print(f"   {name:10} : X = {pos[0]:.4f}, Y = {pos[1]:.4f}, Z = {pos[2]:.4f}")

print("\n5. Effect of current parallel_base_optimizer.py logic:")
print("   Suppose user provides X_flat = 0.33, Y_flat = 0.05")
test_X = 0.33
test_Y = 0.05
for i in range(2):
    test_wX = env.scene.env_origins[i, 0] + test_X
    test_wY = env.scene.env_origins[i, 1] + test_Y
    print(f"   If Env {i} replaced base at local (X={test_X}, Y={test_Y}):")
    print(f"     => World Position would become: X = {test_wX:.4f}, Y = {test_wY:.4f}")

print("="*80 + "\n")

env.close()
app_launcher.app.close()
