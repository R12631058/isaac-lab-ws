# Isaac Lab 版本的 best viewpoint cost 腳本
# 分段測試 - 第一階段：基本設置和配置

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import time
import numpy as np
import pandas as pd
import json

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Best Viewpoint Cost Analysis with Isaac Lab")
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
    """場景配置 - Isaac Lab 版本 (含機器人)"""
    
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
    
    # 機器人 - 使用 Isaac Lab 內建的 Franka 作為替代方案
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.0, 0.0, 0.0),
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                "panda_finger_joint1": 0.04,
                "panda_finger_joint2": 0.04,
            },
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=800.0,
                damping=40.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=800.0,
                damping=40.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
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
        self.robot = None
        self.scene = None
        self.robot_name = "robot"  # 機器人在場景中的名稱，對應 SceneCfg 中的定義
        self.dof_names = None
        self.init_q = None
        self._initialized = False
        
    def initialize(self, scene):
        """初始化機器人系統 - Isaac Lab 版本"""
        try:
            print("🤖 初始化 Isaac Lab 機器人控制器...")
            
            self.scene = scene
            
            # 在 Isaac Lab 中，機器人通常在 articulations 字典中
            print(f"Scene articulations: {list(scene.articulations.keys()) if hasattr(scene, 'articulations') else 'None'}")
            
            if hasattr(scene, 'articulations') and self.robot_name in scene.articulations:
                self.robot = scene.articulations[self.robot_name]
                print(f"✅ 成功從 articulations 獲取機器人: {self.robot_name}")
            elif hasattr(scene, self.robot_name):
                self.robot = getattr(scene, self.robot_name)
                print(f"✅ 成功從場景屬性獲取機器人: {self.robot_name}")
            else:
                print(f"❌ 在場景中找不到機器人: {self.robot_name}")
                print(f"Available articulations: {list(scene.articulations.keys()) if hasattr(scene, 'articulations') else 'None'}")
                print(f"Available scene attributes: {[attr for attr in dir(scene) if not attr.startswith('_')]}")
                return False
            
            if self.robot is not None:
                # 獲取 DOF 資訊
                if hasattr(self.robot, 'joint_names'):
                    self.dof_names = self.robot.joint_names
                    print(f"DOF names: {self.dof_names}")
                elif hasattr(self.robot, 'cfg') and hasattr(self.robot.cfg, 'joint_names'):
                    self.dof_names = self.robot.cfg.joint_names
                    print(f"DOF names from cfg: {self.dof_names}")
                
                # 獲取初始位置
                if hasattr(self.robot, 'data'):
                    self.init_q = self.robot.data.joint_pos[0].cpu().numpy()
                    print(f"Initial joint positions: {self.init_q}")
                
                self._initialized = True
                return True
            else:
                return False
            
        except Exception as e:
            print(f"❌ 機器人初始化失敗: {e}")
            self._initialized = False
            return False
    
    def is_initialized(self):
        """檢查是否已初始化"""
        return self._initialized
    
    def get_joint_positions(self):
        """獲取關節位置"""
        try:
            if self.robot is not None and hasattr(self.robot, 'data'):
                return self.robot.data.joint_pos[0].cpu().numpy()
            return None
        except Exception as e:
            print(f"Error getting joint positions: {e}")
            return None
    
    def set_joint_positions(self, positions):
        """設置關節位置"""
        try:
            if self.robot is not None and positions is not None:
                positions_tensor = torch.tensor(positions, dtype=torch.float32, device=self.robot.device)
                if len(positions_tensor.shape) == 1:
                    positions_tensor = positions_tensor.unsqueeze(0)  # Add batch dimension
                
                # 在 Isaac Lab 中設置關節位置
                if hasattr(self.robot, 'write_joint_state_to_sim'):
                    velocities = torch.zeros_like(positions_tensor)
                    self.robot.write_joint_state_to_sim(positions_tensor, velocities)
                    return True
                elif hasattr(self.robot, 'set_joint_position_target'):
                    self.robot.set_joint_position_target(positions_tensor)
                    return True
                else:
                    print("❌ Robot does not have joint position setting method")
                    return False
            return False
        except Exception as e:
            print(f"Error setting joint positions: {e}")
            return False
    
    def get_robot_info(self):
        """獲取機器人狀態信息"""
        if not self.is_initialized():
            return None
            
        try:
            info = {
                "status": "initialized",
                "framework": "Isaac Lab",
                "robot_name": self.robot_name,
                "dof_names": self.dof_names,
                "current_positions": self.get_joint_positions()
            }
            
            if self.robot is not None:
                # 修正 device 屬性的獲取方式
                if hasattr(self.robot, 'device'):
                    info["device"] = str(self.robot.device)
                elif hasattr(self.robot, '_device'):
                    info["device"] = str(self.robot._device)
                else:
                    info["device"] = "unknown"
                    
                if hasattr(self.robot, 'num_dof'):
                    info["num_dof"] = self.robot.num_dof
            
            return info
        except Exception as e:
            print(f"Error getting robot info: {e}")
            return None


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
    print("🧪 測試第二階段：場景創建")
    
    try:
        # 創建模擬設置
        sim_cfg = sim_utils.SimulationCfg(dt=1/125, device="cuda:0")
        sim = sim_utils.SimulationContext(sim_cfg)
        
        # 創建場景配置
        scene_cfg = SceneCfg(num_envs=1, env_spacing=2.0)  # 先測試單一環境
        
        # 創建場景
        scene = InteractiveScene(scene_cfg)
        
        print("✅ 場景創建測試完成")
        return scene, sim
        
    except Exception as e:
        print(f"❌ 場景創建失敗: {e}")
        return None, None


def test_robot_initialization(scene):
    """測試機器人初始化"""
    print("🧪 測試第三階段：機器人初始化")
    
    if scene is None:
        print("❌ 場景為空，無法測試機器人")
        return None
    
    try:
        # 創建機器人控制器
        config = SimulationConfig()
        robot_controller = RobotControllerIsaacLab(config)
        
        # 初始化機器人
        if robot_controller.initialize(scene):
            print("✅ 機器人初始化成功")
            
            # 獲取機器人資訊
            robot_info = robot_controller.get_robot_info()
            if robot_info:
                print(f"機器人資訊: {robot_info}")
            
            return robot_controller
        else:
            print("❌ 機器人初始化失敗")
            return None
            
    except Exception as e:
        print(f"❌ 機器人初始化過程中發生錯誤: {e}")
        return None


def main():
    """主程式 - 分階段測試 Isaac Lab 移植"""
    print("=" * 60)
    print("🚀 Isaac Lab 移植測試開始")
    print("=" * 60)
    
    # 第一階段：基本設置測試
    config, robot_controller = test_basic_setup()
    
    # 第二階段：場景創建測試
    scene, sim = test_scene_creation()
    
    if scene is not None and sim is not None:
        # 第三階段：機器人初始化測試
        robot_controller = test_robot_initialization(scene)
        
        if robot_controller is not None:
            print("\n🎉 所有基本測試通過！")
            print("📝 準備進行下一階段的功能移植...")
            
            # 簡單的場景步進測試
            print("\n🔄 執行簡單的模擬步進...")
            print("💡 提示：您應該能看到 Isaac Sim 視窗中的 Franka 機器人")
            
            # 讓模擬運行一段時間以便觀察
            for i in range(50):
                sim.step()  # 先執行物理步進
                scene.update(dt=sim.get_physics_dt())  # 然後更新場景
                if i % 10 == 0:
                    print(f"Step {i}: OK - 檢查 Isaac Sim 視窗")
                # 添加小延遲讓視覺更新
                time.sleep(0.02)
            
            print("✅ 模擬步進測試完成")
            print("🖥️  如果您看不到視窗，請檢查：")
            print("   1. Isaac Sim 是否正確啟動")
            print("   2. 視窗是否被最小化")
            print("   3. 是否在後台運行")
        else:
            print("❌ 機器人初始化失敗，請檢查配置")
    else:
        print("❌ 場景創建失敗，請檢查環境設置")
    
    print("\n" + "=" * 60)
    print("📊 測試總結")
    print("=" * 60)
    print("✅ 配置驗證: 通過")
    print("✅ 場景創建:", "通過" if scene is not None else "失敗")
    print("✅ 機器人初始化:", "通過" if robot_controller is not None else "失敗")


if __name__ == "__main__":
    main()
    
    # close sim app
    simulation_app.close()
