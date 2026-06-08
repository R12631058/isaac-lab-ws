"""Verify target range generation after config update"""

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args([])
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_reach_surgery_room_cfg import TM5ExtensionLinkOutReachEnvCfg

env_cfg = TM5ExtensionLinkOutReachEnvCfg()
env_cfg.scene.num_envs = 1
env = ManagerBasedRLEnv(cfg=env_cfg)

print("\n" + "="*60)
print("Target Range Verification")
print("="*60)

# Run a few resets to see target distribution
for i in range(5):
    env.reset()
    
    robot = env.scene['robot']
    robot_pos = robot.data.root_pos_w[0].cpu().numpy()
    
    cmd = env.command_manager.get_term('ee_pose')
    target_cmd = cmd.command[0][:3].cpu().numpy()
    target_world = cmd.pose_command_w[0][:3].cpu().numpy()
    
    if i == 0:
        print(f"\nRobot Root (World): {robot_pos}")
        print(f"Expected Phantom relative: (-0.889, -0.643, 0.037)")
        print("-"*60)
    
    print(f"Reset {i+1}: Target(cmd)={target_cmd}, Target(world)={target_world}")

print("\n" + "="*60)
print("Expected target range (Robot Frame):")
print("  X: [-0.94, -0.84]")
print("  Y: [-0.69, -0.59]")
print("  Z: [0.04, 0.10]")
print("="*60 + "\n")

env.close()
simulation_app.close()
