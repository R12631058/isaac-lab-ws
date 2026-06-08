"""測試不同 robot arm base 位置下的環境初始化

這個腳本用於測試在多個環境中設定不同的 robotarm_base X 位置。

參數說明：
- 初始位置：X = -0.6596245218672151
- 目標範圍：X 從 -0.6596 到 1.0
- 步進：每個環境 X 增加 0.01
- 測試目的：驗證不同環境可以有不同的 robotarm_base 位置
"""

import argparse
import torch
from isaaclab.app import AppLauncher

# 解析命令行參數
parser = argparse.ArgumentParser(description="測試變動 robot arm base 位置")
parser.add_argument("--num_envs", type=int, default=10, help="環境數量")
parser.add_argument("--headless", action="store_true", help="Headless 模式")
parser.add_argument("--cpu", action="store_true", help="使用 CPU")
args_cli = parser.parse_args()

# 啟動 Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================
# 以下代碼在 Isaac Sim 啟動後執行
# ============================================================

import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.manager_based.manipulation.reach import agents

def set_robotarm_base_positions(env, x_positions):
    """設定每個環境的 robotarm_base X 位置
    
    Args:
        env: Isaac Lab 環境
        x_positions: 每個環境的 X 位置列表 (tensor)
    """
    import omni
    from pxr import Gf, UsdGeom
    
    print(f"\n{'='*60}")
    print(f"設定 robotarm_base 位置")
    print(f"{'='*60}")
    
    num_envs = len(x_positions)
    base_y = 0.2693808034227855  # 固定 Y
    base_z = -0.026487045595559477  # 固定 Z
    
    # 取得 USD stage
    stage = omni.usd.get_context().get_stage()
    
    for env_idx in range(num_envs):
        # 構造 robotarm_base 的路徑
        base_prim_path = f"/World/envs/env_{env_idx}/Root/robotarm_base"
        
        # 取得 prim
        base_prim = stage.GetPrimAtPath(base_prim_path)
        
        if not base_prim.IsValid():
            print(f"⚠️  找不到 prim: {base_prim_path}")
            continue
        
        # 設定位置
        x_pos = x_positions[env_idx].item()
        xformable = UsdGeom.Xformable(base_prim)
        
        # 獲取或創建平移操作
        translate_ops = [op for op in xformable.GetOrderedXformOps() 
                        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
        
        if translate_ops:
            # 如果已經有 translate op，直接修改
            translate_op = translate_ops[0]
        else:
            # 否則創建新的
            translate_op = xformable.AddTranslateOp()
        
        translate_op.Set(Gf.Vec3d(x_pos, base_y, base_z))
        
        print(f"Env {env_idx:3d}: X = {x_pos:8.4f}")
    
    print(f"{'='*60}\n")


def main():
    """主函數"""
    
    # ============================================
    # 1. 計算每個環境的 X 位置
    # ============================================
    start_x = -0.6596245218672151
    x_step = 0.01
    
    # 生成 X 位置序列
    x_positions = torch.tensor([start_x + i * x_step for i in range(args_cli.num_envs)])
    
    print(f"\n{'='*60}")
    print(f"測試參數")
    print(f"{'='*60}")
    print(f"環境數量: {args_cli.num_envs}")
    print(f"起始 X: {start_x:.4f}")
    print(f"步進: {x_step}")
    print(f"X 範圍: [{x_positions[0]:.4f}, {x_positions[-1]:.4f}]")
    print(f"{'='*60}\n")
    
    # ============================================
    # 2. 創建環境配置
    # ============================================
    from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_reach_surgery_room_cfg import (
        TM5ExtensionLinkOutReachEnvCfg_PLAY,
    )
    from isaaclab.envs import ManagerBasedRLEnv
    
    # 修改配置中的環境數量
    env_cfg = TM5ExtensionLinkOutReachEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # 創建環境
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    print(f"✅ 環境創建成功")
    
    # ============================================
    # 3. Reset 環境（初始化物理）
    # ============================================
    print(f"\n正在執行環境 reset...")
    obs, info = env.reset()
    print(f"✅ 環境 reset 完成")
    
    # ============================================
    # 4. 設定不同的 robotarm_base 位置
    # ============================================
    set_robotarm_base_positions(env, x_positions)
    
    # ============================================
    # 5. 再次 reset 確認位置生效
    # ============================================
    print(f"\n再次 reset 確認位置...")
    obs, info = env.reset()
    print(f"✅ 第二次 reset 完成")
    
    # ============================================
    # 6. 執行幾步模擬，驗證位置保持
    # ============================================
    print(f"\n執行 10 步模擬驗證...")
    
    # 從環境的 action manager 獲取正確的 action dimension
    action_dim = env.unwrapped.action_manager.total_action_dim
    print(f"Action dimension: {action_dim}")
    
    for step in range(10):
        # 使用零動作
        actions = torch.zeros((args_cli.num_envs, action_dim), device=env.unwrapped.device)
        obs, rewards, dones, truncated, info = env.step(actions)
        print(f"  Step {step+1}/10 完成")
    
    print(f"✅ 模擬驗證完成")
    
    # ============================================
    # 7. 驗證最終位置
    # ============================================
    print(f"\n{'='*60}")
    print(f"驗證最終 robotarm_base 位置")
    print(f"{'='*60}")
    
    import omni
    from pxr import UsdGeom
    
    stage = omni.usd.get_context().get_stage()
    
    for env_idx in range(args_cli.num_envs):
        base_prim_path = f"/World/envs/env_{env_idx}/Root/robotarm_base"
        base_prim = stage.GetPrimAtPath(base_prim_path)
        
        if base_prim.IsValid():
            xformable = UsdGeom.Xformable(base_prim)
            translate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
            
            if translate_ops:
                current_pos = translate_ops[0].Get()
                expected_x = x_positions[env_idx].item()
                
                # 檢查 X 位置是否正確
                x_diff = abs(current_pos[0] - expected_x)
                status = "✅" if x_diff < 0.001 else "❌"
                
                print(f"Env {env_idx:3d}: {status} X = {current_pos[0]:8.4f} (期望: {expected_x:8.4f}, 誤差: {x_diff:.6f})")
    
    print(f"{'='*60}\n")
    
    # ============================================
    # 8. 清理
    # ============================================
    env.close()
    print(f"\n✅ 測試完成！")


if __name__ == "__main__":
    main()
    simulation_app.close()
