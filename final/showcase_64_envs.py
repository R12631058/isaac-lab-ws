"""
showcase_64_envs.py

專門用來展示多環境排開 (例如 64 個) 壯觀畫面的純推論腳本。
加入 NDI Raycast Debug Draw 視覺化與自動攝影機視角移動功能。
"""
import argparse
import sys
import os
import torch
import numpy as np
from pathlib import Path

from isaaclab.app import AppLauncher

# CLI
parser = argparse.ArgumentParser(description="Showcase 64 Environments with NDI Raycast")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to spawn")
parser.add_argument("--checkpoint", type=str,
                    default=r"logs\rsl_rl\tm5_reach_stable_v2\2026-02-04_00-28-35\model_9999.pt",
                    help="RL policy checkpoint")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch simulation
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_extension_linkout_cone_orientation_cfg import TM5ExtensionLinkOutFanOrientationEnvCfg
from rsl_rl.modules import ActorCritic

# ── NDI 整合 ──
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ndi_detector_isaaclab_experimental import NDIConfig, NDIDetector
import omni.usd

class EnvNDIConfig(NDIConfig):
    def __init__(self, env_prim_path, detected_map=None):
        super().__init__()
        prefix_replacement = env_prim_path
        if detected_map and "emitter_path" in detected_map:
            p = detected_map["emitter_path"]
            if "/NDI/NDI_emitter" in p:
                prefix_replacement = p.split("/NDI/NDI_emitter")[0]
        self.START_PRIM_PATHS = [p.replace("/Root", prefix_replacement) for p in self.START_PRIM_PATHS]
        self.START_HIGHLIGHT_PATHS = [p.replace("/Root", prefix_replacement) for p in self.START_HIGHLIGHT_PATHS]
        self.END_PRIM_PATHS = [p.replace("/Root", prefix_replacement) for p in self.END_PRIM_PATHS]
        self.TRIGGER_VOLUME_PATH = self.TRIGGER_VOLUME_PATH.replace("/Root", prefix_replacement)
        if detected_map and "link_6_path" in detected_map:
            self.LINK_6_PATH = detected_map["link_6_path"]
        else:
            self.LINK_6_PATH = f"{prefix_replacement}/robotarm_base/robotarm_base/tm5_700/link_6"
            
        if detected_map and "needle_tip_path" in detected_map:
            self.NEEDLE_TIP_PATH = detected_map["needle_tip_path"]
        else:
            self.NEEDLE_TIP_PATH = f"{prefix_replacement}/robotarm_base/robotarm_base/needle/needle_tip"

def find_env_structure(stage, env_path):
    result = {}
    env_prim = stage.GetPrimAtPath(env_path)
    if not env_prim.IsValid(): return result
    def search(prim, depth=0):
        if depth > 10: return
        name = prim.GetName()
        path = str(prim.GetPath())
        if name == "link_6" and "link_6_path" not in result:
            result["link_6_path"] = path
        if name == "needle_tip" and "needle_tip_path" not in result:
            result["needle_tip_path"] = path
        if name == "NDI_emitter" and "emitter_path" not in result:
            result["emitter_path"] = path
        for child in prim.GetChildren():
            search(child, depth + 1)
            if "link_6_path" in result and "emitter_path" in result and "needle_tip_path" in result: return
    search(env_prim)
    return result

def main():
    print(f"\n[INIT] 🚀 準備生成 {args_cli.num_envs} 個環境的壯觀畫面 (包含 NDI 射線)！")
    
    # 建立環境設定
    env_cfg = TM5ExtensionLinkOutFanOrientationEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    if hasattr(env_cfg.commands, "ee_pose"):
        env_cfg.commands.ee_pose.resampling_time_range = (1e10, 1e10)
    
    # 創建環境
    print("[INIT] 載入場景與實體 (這可能需要一點時間載入高數量 USD Instance)...")
    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.device
    
    # 攝影機視角設定 (上帝視角)
    try:
        from isaacsim.core.utils.viewports import set_camera_view
        # 當 env_spacing=8.0，64個環境約是 8x8，跨度 X:0~56, Y:0~56。中心點大約是 (28, 28)
        # 攝影機從南邊(Y負)往北邊(Y正)看，由上往下俯視
        set_camera_view(eye=[28.0, -10.0, 45.0], target=[28.0, 20.0, 0.0])
        print("[VIEWPORT] 攝影機已自動調整至上帝全景視角。")
    except Exception as e:
        print(f"[VIEWPORT] 自動調整攝影機失敗 (不影響運行): {e}")

    # NDI 偵測器初始化
    stage = omni.usd.get_context().get_stage()
    print(f"\n[NDI] 初始化 {args_cli.num_envs} 個 NDI 射線追蹤偵測器...")
    detectors = []
    
    # 若數量過多，為了效能與不讓記憶體爆炸，這裡我們允許載入，但後續畫線降頻
    for i in range(args_cli.num_envs):
        env_prim_path = f"/World/envs/env_{i}"
        try:
            detected_map = find_env_structure(stage, env_prim_path)
            ndi_config = EnvNDIConfig(env_prim_path, detected_map)
            detector = NDIDetector(ndi_config)
            detector.env_index = i
            detector.error_only_log = True  # 關閉除錯文字，避免 Terminal 洗版與卡頓
            detector._link_paths["link_6"] = ndi_config.LINK_6_PATH
            detector._link_paths["needle_tip"] = ndi_config.NEEDLE_TIP_PATH
            detector._link_paths["link_base"] = ndi_config.LINK_6_PATH.replace("link_6", "link_0")
            detector.initialize()
            detectors.append(detector)
        except Exception as e:
            print(f"[ERROR] Failed to init NDI for env {i}: {e}")
            detectors.append(None)
    
    # 載入神經網路模型 (Policy)
    print("\n[INIT] 載入神經網路行動策略...")
    obs_dim = env.unwrapped.observation_manager.group_obs_dim["policy"][0]
    action_dim = env.unwrapped.action_manager.total_action_dim
    policy = ActorCritic(
        num_actor_obs=obs_dim,
        num_critic_obs=obs_dim,
        num_actions=action_dim,
        actor_hidden_dims=[768, 512, 512, 256],
        critic_hidden_dims=[768, 512, 512, 256],
        activation='elu',
        init_noise_std=1.0,
    ).to(device)

    ckpt_path = Path(args_cli.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        policy.load_state_dict(ckpt["model_state_dict"])
    else:
        policy.load_state_dict(ckpt)
    policy.eval()
    print(f"[OK] Policy 已載入: {ckpt_path.name}")
    
    print("\n=======================================================")
    print("🎥 模擬開始運作！展示模式已開啟。")
    print("已結合 NDI Raycast 及自動鏡頭移動。")
    print("如果畫面太卡，可以嘗試減少 --num_envs 的數量 (例如 16)。")
    print("=======================================================\n")

    obs, _ = env.reset()
    step_count = 0
    # 每 3 步算一次 NDI 射線，這能在展示高數量環境時，大幅減輕 GPU PhysX 及 CPU 劃線的負擔，維持 FPS
    RAYCAST_INTERVAL = 3 

    try:
        while simulation_app.is_running():
            with torch.no_grad():
                # 從 policy 產生動作
                obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
                actions = policy.act(obs_tensor, deterministic=True)
            
            # 將動作輸入至模擬環境，推進物理時間
            obs, _, _, _, _ = env.step(actions)
            
            # --- 執行 NDI Raycast Display ---
            if step_count % RAYCAST_INTERVAL == 0:
                for i, detector in enumerate(detectors):
                    if detector is None: continue
                    # 更新骨架點
                    detector._update_lines()
                    # 執行物理引擎 Raycast Check
                    detector.update_detection(env.sim, debug_enabled=True, error_only_log=True)
                    # 執行畫面畫線 debug_draw
                    detector.draw_live(clear_first=(i == 0))
            
            step_count += 1
            
    except KeyboardInterrupt:
        print("\n使用者中斷程式。")
        
    finally:
        env.close()

if __name__ == "__main__":
    main()