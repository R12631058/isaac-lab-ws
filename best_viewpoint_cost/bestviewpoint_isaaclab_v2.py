# Isaac Lab 版本的 best viewpoint cost 腳本
# 分段測試 - 第二階段：USD 載入和機器人配置

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import time
import numpy as np
import pandas as pd
import json

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Best Viewpoint Cost Analysis with Isaac Lab - Stage 2")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args (device is already included in AppLauncher args)
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
from copy import deepcopy

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg

# Isaac Lab 替代模組
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

# USD 相關
from pxr import Gf, Usd, UsdGeom
import omni.usd

# TM5 相關 - 使用 Isaac Lab 內建的配置
# from isaaclab_tasks.manager_based.manipulation.tm5_reach import agents
# from isaaclab_tasks.manager_based.manipulation.tm5_reach.tm5_reach_env_cfg import TM5ReachEnvCfg


class SimulationConfig:
    """統一管理所有配置參數 - Isaac Lab 版本"""
    def __init__(self):
        # 基本路徑配置
        self.USD_FILE_PATH = "C:/Nick/surgery_team/surgery_team/USD/surgery_room_demo.usd"
        self.URDF_PATH = "C:/Nick/surgery_team/tmr_ros2-humble/tmr_ros2-humble/tm_description/urdf/tm5-700-nominal.urdf"
        self.ROBOT_DESCRIPTION_PATH = "C:/Nick/surgery_team/surgery_team/robot_tm5700_skrew.yaml"
        
        # CSV 路徑選項
        self.CSV_PATH = r"C:\Nick\surgery_team\Huan\TM5-700 Data_1800.csv"
        
        # 物理時間步長配置
        self.PHYSICS_DT = 1/125  # 8ms
        self.ROBOT_PRIM_PATH = "/World/Marker/robotarm_base/tm5_700"
        self.END_EFFECTOR_LINK = "link_6"
        self.NEEDLETIP_PATH = "/World/Marker/robotarm_base/End_needle/End_needle/needle_tip"

        # UI相關配置
        self.SUPPORTED_MARKERS = ["FM", "HM", "BM", "EM", "UM"]
        self.GREEN_IMG_PATH = "C:/Nick/surgery_team/surgery_team/Green.png"
        self.RED_IMG_PATH = "C:/Nick/surgery_team/surgery_team/Red.png"
        
        # NDI 檢測路徑配置
        start_prims = "(/World/Marker/surgery_room/NDI/NDI_mesh/VegaST_XT_cam/NDI_emitter)"
        end_prims = """(/World/Marker/robotarm_base/End_needle/BM/bm_rball_0/node_/mesh_),
        /World/Marker/robotarm_base/End_needle/BM/bm_rball_1/node_/mesh_,
        /World/Marker/robotarm_base/End_needle/BM/bm_rball_2/node_/mesh_,
        /World/Marker/robotarm_base/End_needle/BM/bm_rball_3/node_/mesh_"""

        # 解析起點路徑
        self.START_PRIM_PATHS = []
        self.START_HIGHLIGHT_PATHS = []
        for path in start_prims.split(','):
            path = path.strip()
            if path.startswith('(') and path.endswith(')'):
                clean_path = path[1:-1].strip()
                self.START_PRIM_PATHS.append(clean_path)
                self.START_HIGHLIGHT_PATHS.append(clean_path)
            else:
                self.START_PRIM_PATHS.append(path)
        
        # 解析終點路徑
        self.END_PRIM_PATHS = []
        self.END_HIGHLIGHT_PATHS = []
        for path in end_prims.split(','):
            path = path.strip()
            if path.startswith('(') and path.endswith(')'):
                clean_path = path[1:-1].strip()
                self.END_PRIM_PATHS.append(clean_path)
                self.END_HIGHLIGHT_PATHS.append(clean_path)
            else:
                self.END_PRIM_PATHS.append(path)
        
        # 觸發體積路徑
        self.TRIGGER_VOLUME_PATH = "/World/Marker/surgery_room/NDI/NDI_mesh/VegaST_XT_cam/node_/volume"

    def validate_configuration(self):
        """驗證配置檔案是否存在且有效"""
        print("🔍 驗證配置檔案:")
        
        # 檢查 USD 檔案
        if os.path.exists(self.USD_FILE_PATH):
            print(f"✅ USD 檔案存在: {self.USD_FILE_PATH}")
        else:
            print(f"❌ USD 檔案不存在: {self.USD_FILE_PATH}")
            
        # 檢查 URDF 檔案
        if os.path.exists(self.URDF_PATH):
            print(f"✅ URDF 檔案存在: {self.URDF_PATH}")
        else:
            print(f"❌ URDF 檔案不存在: {self.URDF_PATH}")
            
        # 檢查機器人描述檔案
        if os.path.exists(self.ROBOT_DESCRIPTION_PATH):
            print(f"✅ 機器人描述檔案存在: {self.ROBOT_DESCRIPTION_PATH}")
        else:
            print(f"❌ 機器人描述檔案不存在: {self.ROBOT_DESCRIPTION_PATH}")
            
        # 檢查 CSV 檔案
        if os.path.exists(self.CSV_PATH):
            print(f"✅ CSV 檔案存在: {self.CSV_PATH}")
            file_size = os.path.getsize(self.CSV_PATH)
            print(f"   檔案大小: {file_size} bytes")
        else:
            print(f"❌ CSV 檔案不存在: {self.CSV_PATH}")
            
        print(f"📝 物理時間步長: {self.PHYSICS_DT} 秒")
        print(f"📝 機器人路徑: {self.ROBOT_PRIM_PATH}")
        print(f"📝 末端執行器: {self.END_EFFECTOR_LINK}")


@configclass
class SceneCfg(InteractiveSceneCfg):
    """場景配置 - Isaac Lab 版本 (包含機器人)"""
    
    # 地面
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    
    # 燈光
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    
    # TM5 機器人 - 使用 URDF 轉換配置
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path="C:/Nick/surgery_team/tmr_ros2-humble/tmr_ros2-humble/tm_description/urdf/tm5-700-nominal.urdf",
            activate_contact_sensors=False,  # 暫時關閉接觸感測器
            fix_base=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=800.0, damping=40.0)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.5, 0.0, 0.0),
            joint_pos={
                "joint_1": 0.0,
                "joint_2": 0.0,
                "joint_3": 0.0,
                "joint_4": 0.0,
                "joint_5": 0.0,
                "joint_6": 0.0,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint_.*"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=800.0,
                damping=40.0,
            ),
        },
    )


class USDLoader:
    """USD 檔案載入器 - Isaac Lab 版本"""
    
    def __init__(self, config):
        self.config = config
        self.stage = None
        self.loaded_assets = {}
        
    def load_usd_file(self, usd_path: str, target_prim_path: str = "/World/Scene"):
        """載入 USD 檔案到場景中"""
        try:
            print(f"🔄 載入 USD 檔案: {usd_path}")
            
            if not os.path.exists(usd_path):
                print(f"❌ USD 檔案不存在: {usd_path}")
                return False
                
            # 獲取當前 stage
            self.stage = omni.usd.get_context().get_stage()
            
            if self.stage is None:
                print("❌ 無法獲取 USD stage")
                return False
            
            # 在 Isaac Lab 中使用 Reference 載入 USD
            from pxr import Sdf
            
            # 創建參考
            prim = self.stage.DefinePrim(target_prim_path)
            if prim:
                # 添加參考
                prim.GetReferences().AddReference(usd_path)
                print(f"✅ USD 檔案載入成功: {target_prim_path}")
                self.loaded_assets[target_prim_path] = usd_path
                return True
            else:
                print(f"❌ 無法創建 prim: {target_prim_path}")
                return False
                
        except Exception as e:
            print(f"❌ USD 載入過程中發生錯誤: {e}")
            return False
    
    def get_prim_paths(self):
        """獲取所有載入的 prim 路徑"""
        try:
            if self.stage is None:
                return []
                
            all_prims = []
            for prim in self.stage.Traverse():
                all_prims.append(prim.GetPath().pathString)
                
            return all_prims
        except Exception as e:
            print(f"Error getting prim paths: {e}")
            return []


class RobotControllerIsaacLab:
    """Isaac Lab 版本的機器人控制器"""
    
    def __init__(self, config):
        self.config = config
        self.robot_articulation = None
        self.usd_loader = USDLoader(config)
        self._initialized = False
        
    def initialize(self, scene):
        """初始化機器人系統 - Isaac Lab 版本"""
        try:
            print("🤖 初始化 Isaac Lab 機器人控制器...")
            
            # 載入 USD 場景檔案
            usd_success = self.usd_loader.load_usd_file(
                self.config.USD_FILE_PATH,
                "/World/Scene"
            )
            
            if usd_success:
                print("✅ USD 場景載入成功")
            else:
                print("⚠️ USD 場景載入失敗，繼續使用基本場景")
            
            # 檢查機器人是否存在於場景中
            if hasattr(scene, 'robot'):
                self.robot_articulation = scene["robot"]
                print("✅ 機器人 articulation 已連接")
            else:
                print("⚠️ 場景中沒有機器人，使用基本配置")
            
            print("✅ 機器人控制器初始化完成")
            self._initialized = True
            return True
            
        except Exception as e:
            print(f"❌ 機器人初始化失敗: {e}")
            self._initialized = False
            return False
    
    def is_initialized(self):
        """檢查是否已初始化"""
        return self._initialized
    
    def get_robot_info(self):
        """獲取機器人狀態信息"""
        if not self.is_initialized():
            return None
            
        try:
            info = {
                "status": "initialized",
                "framework": "Isaac Lab",
                "usd_loaded": len(self.usd_loader.loaded_assets) > 0,
                "robot_available": self.robot_articulation is not None,
            }
            
            if self.robot_articulation is not None:
                info["robot_dof"] = self.robot_articulation.num_dof
                info["robot_joint_names"] = self.robot_articulation.joint_names
                
            return info
        except Exception as e:
            print(f"Error getting robot info: {e}")
            return None
    
    def get_loaded_prims(self):
        """獲取載入的 prim 路徑"""
        return self.usd_loader.get_prim_paths()


def test_basic_setup():
    """測試基本設置"""
    print("🧪 測試第一階段：基本設置")
    
    # 測試配置
    config = SimulationConfig()
    config.validate_configuration()
    
    # 測試機器人控制器
    robot_controller = RobotControllerIsaacLab(config)
    
    print("✅ 基本設置測試完成")
    return config, robot_controller


def test_scene_creation():
    """測試場景創建"""
    print("🧪 測試第二階段：場景創建 (含機器人)")
    
    try:
        # 創建模擬設置
        sim_cfg = sim_utils.SimulationCfg(dt=1/125, device="cuda:0")
        sim = sim_utils.SimulationContext(sim_cfg)
        
        # 創建場景配置 (包含機器人)
        scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
        
        # 創建場景
        scene = InteractiveScene(scene_cfg)
        
        print("✅ 場景創建測試完成 (含機器人)")
        return scene, sim
        
    except Exception as e:
        print(f"❌ 場景創建失敗: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_usd_loading(robot_controller):
    """測試 USD 檔案載入"""
    print("🧪 測試第三階段：USD 檔案載入")
    
    try:
        # 測試載入 USD 檔案
        success = robot_controller.usd_loader.load_usd_file(
            robot_controller.config.USD_FILE_PATH,
            "/World/SurgeryRoom"
        )
        
        if success:
            print("✅ USD 檔案載入測試完成")
            
            # 列出一些載入的 prims
            prims = robot_controller.get_loaded_prims()
            if prims:
                print(f"📝 載入了 {len(prims)} 個 prims")
                print("   前 10 個 prim 路徑:")
                for i, prim_path in enumerate(prims[:10]):
                    print(f"     {i+1}. {prim_path}")
            
            return True
        else:
            print("❌ USD 檔案載入測試失敗")
            return False
            
    except Exception as e:
        print(f"❌ USD 載入測試失敗: {e}")
        return False


def main():
    """主函數 - 分段測試"""
    print("🚀 開始 Isaac Lab 移植測試 - 第二階段")
    
    # 第一階段：基本設置測試
    try:
        config, robot_controller = test_basic_setup()
        print("✅ 第一階段完成：基本設置")
    except Exception as e:
        print(f"❌ 第一階段失敗: {e}")
        return
    
    # 第二階段：場景創建測試 (含機器人)
    try:
        scene, sim = test_scene_creation()
        if scene is None or sim is None:
            print("❌ 第二階段失敗：場景創建")
            return
        print("✅ 第二階段完成：場景創建 (含機器人)")
    except Exception as e:
        print(f"❌ 第二階段失敗: {e}")
        return
    
    # 第三階段：機器人初始化測試
    try:
        success = robot_controller.initialize(scene)
        if success:
            print("✅ 第三階段完成：機器人初始化")
            
            # 顯示機器人信息
            robot_info = robot_controller.get_robot_info()
            if robot_info:
                print("📝 機器人狀態:")
                for key, value in robot_info.items():
                    print(f"   {key}: {value}")
        else:
            print("❌ 第三階段失敗：機器人初始化")
    except Exception as e:
        print(f"❌ 第三階段失敗: {e}")
    
    # 第四階段：USD 載入測試
    try:
        usd_success = test_usd_loading(robot_controller)
        if usd_success:
            print("✅ 第四階段完成：USD 載入")
        else:
            print("⚠️ 第四階段部分完成：USD 載入有問題")
    except Exception as e:
        print(f"❌ 第四階段失敗: {e}")
    
    # 運行基本模擬循環
    try:
        print("🔄 開始基本模擬循環...")
        sim.reset()
        
        for step in range(100):
            sim.step()
            scene.update(sim.get_physics_dt())
            
            if step % 25 == 0:
                print(f"   步驟 {step}/100")
        
        print("✅ 基本模擬循環完成")
        
    except Exception as e:
        print(f"❌ 模擬循環失敗: {e}")
    
    print("🎉 Isaac Lab 移植測試完成 - 第二階段")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
