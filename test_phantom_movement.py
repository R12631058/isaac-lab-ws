"""Test moving phantom in multiple environments to verify USD modification works"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test phantom movement in multiple environments")
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
from pxr import Gf, UsdGeom
from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_reach_surgery_room_cfg import (
    TM5ExtensionLinkOutReachEnvCfg
)
from isaaclab.envs import ManagerBasedRLEnv


def move_phantom_multi_env(env, num_envs, offset_x_start, offset_x_step):
    """在多個環境中移動 phantom 到不同位置"""
    stage = env.scene.stage
    
    for env_idx in range(num_envs):
        prim_path = f"/World/envs/env_{env_idx}/Root/Phantom"
        phantom_prim = stage.GetPrimAtPath(prim_path)
        
        if not phantom_prim.IsValid():
            print(f"[ERROR] Cannot find {prim_path}")
            continue
        
        xformable = UsdGeom.Xformable(phantom_prim)
        
        # 清除並設置新位置
        xformable.ClearXformOpOrder()
        if phantom_prim.HasAttribute("xformOp:transform"):
            phantom_prim.RemoveProperty("xformOp:transform")
        
        # 每個環境有不同的 X 偏移
        x_offset = offset_x_start + env_idx * offset_x_step
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(x_offset, 0.0, 0.0))
        
        # 驗證
        computed_transform = xformable.ComputeLocalToWorldTransform(0)
        translation = computed_transform.ExtractTranslation()
        print(f"  Env {env_idx}: Phantom USD X offset = {x_offset:.2f}, Computed pos = ({translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f})")


def main():
    print("="*60)
    print(f"Testing Phantom Movement in {args_cli.num_envs} Environments")
    print("="*60)
    
    # Create environment with multiple envs
    env_cfg = TM5ExtensionLinkOutReachEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 5.0
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    print(f"[OK] Environment created with {args_cli.num_envs} environments\n")
    
    # Test sequence
    test_offsets = [
        (0.0, 0.2),    # 每個環境相差 0.2m
        (-0.5, 0.2),   # 从 -0.5 开始，每個環境相差 0.2m
        (0.5, -0.1),   # 从 0.5 开始，每個環境相差 -0.1m
    ]
    
    for test_idx, (start_offset, step) in enumerate(test_offsets):
        print(f"\n{'='*60}")
        print(f"Test {test_idx+1}: X offset range from {start_offset:.2f} with step {step:.2f}")
        print(f"{'='*60}")
        
        # Move phantoms
        print(f"[INFO] Setting phantom positions:")
        move_phantom_multi_env(env, args_cli.num_envs, start_offset, step)
        
        # Reset to apply
        env.reset()
        print(f"\n[INFO] Positions applied. Reset environment.")
        
        # Simulate for a few steps
        for step_idx in range(50):
            obs, rewards, dones, truncated, info = env.step(
                torch.zeros(env.action_space.shape, device=env.device)
            )
        
        print(">> Check the viewer - do phantoms have different X positions?")
        print(f">> Expected X range: [{start_offset:.2f}, {start_offset + (args_cli.num_envs-1)*step:.2f}]")
        try:
            input("Press Enter to continue to next test...")
        except EOFError:
            # Handle non-interactive mode
            print("(Skipping input in non-interactive mode)")
            for _ in range(200):
                obs, rewards, dones, truncated, info = env.step(
                    torch.zeros(env.action_space.shape, device=env.device)
                )
    
    print("\n[OK] Test complete")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
