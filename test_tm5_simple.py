#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
簡單的 TM5 配置測試腳本
"""

import argparse
from isaaclab.app import AppLauncher

# 添加命令行參數
parser = argparse.ArgumentParser(description="TM5 配置測試")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 啟動應用程式
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""其餘代碼"""

def test_tm5_import():
    """測試 TM5 配置導入"""
    try:
        print("[INFO] 測試 TM5 配置導入...")
        from isaaclab_assets.robots.tm5_700 import TM5_700_CFG
        print("[SUCCESS] TM5_700_CFG 導入成功!")
        print(f"[INFO] 配置類型: {type(TM5_700_CFG)}")
        return True
    except Exception as e:
        print(f"[ERROR] TM5 配置導入失敗: {e}")
        return False

def test_environment_import():
    """測試環境導入"""
    try:
        print("[INFO] 測試環境導入...")
        import isaaclab_tasks  # noqa: F401
        print("[SUCCESS] isaaclab_tasks 導入成功!")
        
        # 嘗試創建環境來測試註冊
        import gymnasium as gym
        try:
            # 只檢查是否可以找到環境，不實際創建
            env_spec = gym.spec("Isaac-Reach-TM5-v0")
            print(f"[SUCCESS] 找到 TM5 環境: Isaac-Reach-TM5-v0")
            print(f"[INFO] 環境入口點: {env_spec.entry_point}")
        except Exception as env_error:
            print(f"[WARNING] 環境註冊可能有問題: {env_error}")
        
        return True
    except Exception as e:
        print(f"[ERROR] 環境導入失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("=== TM5 配置測試 ===")
    
    # 測試 TM5 導入
    if not test_tm5_import():
        return
    
    # 測試環境導入  
    if not test_environment_import():
        return
        
    print("\n[SUCCESS] 所有基本測試通過!")

if __name__ == "__main__":
    main()
    simulation_app.close()
