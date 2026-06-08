"""測試可視化框是否正確創建"""

import argparse
from isaaclab.app import AppLauncher

# 解析參數
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 啟動模擬器
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg

# 載入環境配置
env_cfg = parse_env_cfg(
    "Isaac-Reach-TM5-NeedleTip-v0",
    device="cuda:0",
    num_envs=args_cli.num_envs,
)

# 創建環境
env = ManagerBasedRLEnv(cfg=env_cfg)

print("\n=== 場景 Prim 路徑檢查 ===")
import omni
from pxr import Usd

# 檢查場景中的 prims
stage = omni.usd.get_context().get_stage()
for prim in stage.Traverse():
    prim_path = str(prim.GetPath())
    if "TargetRangeBox" in prim_path or ("robotarm_base" in prim_path and "tm5_700" not in prim_path):
        print(f"Found: {prim_path}")

print("\n=== 等待顯示場景... ===")
print("按 Ctrl+C 結束")

# 運行模擬
while simulation_app.is_running():
    env.step(env.action_manager.action)
    
simulation_app.close()
