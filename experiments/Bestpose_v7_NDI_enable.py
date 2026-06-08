# v5 newlayout + cost function
# v5 + v6 都沒有完整的 NDI checking
# 首先保證能計算單個 NDI 的 cost
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import time
import numpy as np
import pandas as pd
import os
import omni
import omni.usd 
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver
from omni.isaac.motion_generation import LulaCSpaceTrajectoryGenerator, ArticulationTrajectory
from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.stage import add_reference_to_stage
from isaacsim.util.debug_draw import _debug_draw
import omni.ui as ui
import omni.kit.raycast.query
from pxr import Gf, Usd, UsdGeom
import importlib
traceback = importlib.import_module("traceback")
from scipy.spatial import ConvexHull
import itertools
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # 使用非交互式後端
import matplotlib.pyplot as plt

# 1. 核心配置管理
class SimulationConfig:
    """統一管理所有配置參數"""
    def __init__(self):
        self.USD_FILE_PATH = "C:\\Nick\\surgery_team\\surgery_team\\USD\\surgeryroom_lostfunc.usd"
        self.URDF_PATH = "C:/Nick/surgery_team/tmr_ros2-humble/tmr_ros2-humble/tm_description/urdf/tm5-700-nominal.urdf"
        self.ROBOT_DESCRIPTION_PATH = "C:/Nick/surgery_team/surgery_team/robot_tm5700_skrew.yaml"
        
        # CSV 路徑選項
        self.CSV_PATH = r"C:\Nick\surgery_team\Huan\TM5-700 Data_1800.csv"  # 原始真實軌跡
        #self.CSV_PATH = r"C:\Users\RMML\AppData\Local\ov\pkg\4.5.0\standalone_ws\robot_arm_linear_trajectory.csv"  # 優化線性軌跡

        
        # 物理時間步長配置
        # 快速測試: 1/125 = 8ms (執行較快，精度較低)
        # 標準配置: 1/250 = 4ms (平衡速度和精度)  
        # 高精度: 1/500 = 2ms (執行較慢，精度較高)
        self.PHYSICS_DT = 1/125 # 物理時間步長 - 改為更快的設定
        self.ROBOT_PRIM_PATH = "/Root/robotarm_base/tm5_700"
        self.END_EFFECTOR_LINK = "link_6"
        self.NEEDLETIP_PATH ="/Root/robotarm_base/End_needle/End_needle/needle_tip"

        # UI相關配置
        self.SUPPORTED_MARKERS = ["FM", "HM", "BM", "EM", "UM"]
        self.GREEN_IMG_PATH = "C:/Nick/surgery_team/surgery_team/Green.png"
        self.RED_IMG_PATH = "C:/Nick/surgery_team/surgery_team/Red.png"
        
        # NDI 檢測路徑配置 - 完全按照v2的方式
        start_prims = "(/Root/NDI_02/NDI_emitter),/Root/NDI_02/NDI_left_sensor,/Root/NDI_02/NDI_right_sensor"
        end_prims = """(/Root/robotarm_base/End_needle/BM/BM_meter/bm_a),
        /Root/robotarm_base/End_needle/BM/BM_meter/bm_b,
        /Root/robotarm_base/End_needle/BM/BM_meter/bm_c,
        /Root/robotarm_base/End_needle/BM/BM_meter/bm_d,
        (/Root/robotarm_base/End_needle/EM/EM/em_a),
        /Root/robotarm_base/End_needle/EM/EM/em_b,
        /Root/robotarm_base/End_needle/EM/EM/em_c,
        /Root/robotarm_base/End_needle/EM/EM/em_d,
        (/Root/FM/FM/FM/FM/fm_a),
        (/Root/FM/FM/FM/FM/fm_b),
        (/Root/FM/FM/FM/FM/fm_c),
        (/Root/FM/FM/FM/FM/fm_d),
        (/Root/robotarm_base/End_needle/HM/HM_frame/hm_rball_0/node_/mesh_),
        /Root/robotarm_base/End_needle/HM/HM_frame/hm_rball_1/node_/mesh_,
        /Root/robotarm_base/End_needle/HM/HM_frame/hm_rball_2/node_/mesh_,
        /Root/robotarm_base/End_needle/HM/HM_frame/hm_rball_3/node_/mesh_,
        (/Root/robotarm_base/UM/UM/UM/um_a),
        (/Root/robotarm_base/UM/UM/UM/um_b),
        (/Root/robotarm_base/UM/UM/UM/um_c),
        (/Root/robotarm_base/UM/UM/UM/um_d)"""

        # 解析起點路徑
        self.START_PRIM_PATHS = []
        self.START_HIGHLIGHT_PATHS = []
        for path in start_prims.split(','):
            path = path.strip()
            # 檢查是否有括號
            if path.startswith('(') and path.endswith(')'):
                # 移除括號並加入高亮列表
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
        self.TRIGGER_VOLUME_PATH = "/Root/NDI_02/mesh_"

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
            # 檢查檔案大小
            file_size = os.path.getsize(self.CSV_PATH)
            print(f"   檔案大小: {file_size} bytes")
        else:
            print(f"❌ CSV 檔案不存在: {self.CSV_PATH}")
            
        print(f"📝 物理時間步長: {self.PHYSICS_DT} 秒")
        print(f"📝 機器人路徑: {self.ROBOT_PRIM_PATH}")
        print(f"📝 末端執行器: {self.END_EFFECTOR_LINK}")

# 2. 機器人控制器模組
class RobotController:
    """專門處理機器人控制邏輯"""
    def __init__(self, config):
        self.config = config
        self.robot_articulation = None
        self.ik_solver = None
        self.ik_solver_interface = None
        self.c_traj_gen = None
        self._initialized = False
        
    def initialize(self, world):
        """初始化機器人系統"""
        try:
            print("Initializing robot controller...")
            
            # 確保場景已經穩定
            print("Waiting for scene to stabilize...")
            for _ in range(10):  # 增加等待時間
                world.step(render=True)
                time.sleep(0.1)
            
            # 初始化機器人
            print(f"Creating robot articulation at: {self.config.ROBOT_PRIM_PATH}")
            self.robot_articulation = Articulation(self.config.ROBOT_PRIM_PATH)
            world.scene.add(self.robot_articulation)
            
            # 強制重置世界以確保所有對象都被正確載入
            print("Resetting world to ensure proper loading...")
            world.reset()
            
            # 等待重置完成
            for _ in range(20):
                world.step(render=True)
                time.sleep(0.1)
            
            # 多次嘗試初始化直到成功
            max_attempts = 10  # 增加嘗試次數
            for attempt in range(max_attempts):
                try:
                    print(f"Robot initialization attempt {attempt + 1}/{max_attempts}")
                    
                    # 初始化機器人
                    self.robot_articulation.initialize()
                    
                    # 等待更多時間步讓機器人完全初始化
                    print("Waiting for robot to fully initialize...")
                    for step in range(30):  # 增加等待步數
                        world.step(render=True)
                        time.sleep(0.05)
                        
                        # 每10步檢查一次關節狀態
                        if step % 10 == 0:
                            try:
                                joint_positions = self.robot_articulation.get_joint_positions()
                                if joint_positions is not None and len(joint_positions) > 0:
                                    print(f"Step {step}: Found {len(joint_positions)} joints")
                            except:
                                print(f"Step {step}: Joints not ready yet")
                    
                    # 最終檢查機器人是否真的可以獲取關節位置
                    joint_positions = self.robot_articulation.get_joint_positions()
                    if joint_positions is not None and len(joint_positions) > 0:
                        print(f"Robot joints successfully initialized: {len(joint_positions)} joints")
                        print(f"Joint positions: {joint_positions}")
                        
                        # 額外驗證：嘗試獲取其他機器人狀態
                        try:
                            joint_velocities = self.robot_articulation.get_joint_velocities()
                            world_pose = self.robot_articulation.get_world_pose()
                            print(f"Additional verification successful - velocities: {len(joint_velocities) if joint_velocities is not None else 0}")
                            break
                        except Exception as e:
                            print(f"Additional verification failed: {e}")
                            if attempt < max_attempts - 1:
                                time.sleep(2.0)
                                continue
                            else:
                                raise Exception("Robot state verification failed")
                    else:
                        print(f"Attempt {attempt + 1}: Robot joints still not ready")
                        if attempt < max_attempts - 1:
                            time.sleep(2.0)
                            continue
                        else:
                            raise Exception("Robot joints never became available after all attempts")
                            
                except Exception as e:
                    print(f"Robot initialization attempt {attempt + 1} failed: {e}")
                    if attempt == max_attempts - 1:
                        raise Exception(f"All {max_attempts} initialization attempts failed. Last error: {e}")
                    time.sleep(2.0)
            
            # 再次等待確保穩定
            print("Final stabilization wait...")
            for _ in range(20):
                world.step(render=True)
                time.sleep(0.05)
            
            # 初始化IK求解器
            print("Initializing IK solver...")
            try:
                self.ik_solver = LulaKinematicsSolver(
                    robot_description_path=self.config.ROBOT_DESCRIPTION_PATH,
                    urdf_path=self.config.URDF_PATH
                )
                
                self.ik_solver_interface = ArticulationKinematicsSolver(
                    self.robot_articulation, 
                    self.ik_solver, 
                    self.config.END_EFFECTOR_LINK
                )
            except Exception as e:
                print(f"IK solver initialization failed: {e}")
                raise
            
            # 測試IK求解器 - 使用更保守的方法
            print("Testing IK solver...")
            test_attempts = 5
            for test_attempt in range(test_attempts):
                try:
                    # 額外等待時間
                    for _ in range(10):
                        world.step(render=True)
                        time.sleep(0.1)
                    
                    current_position, current_rotation = self.ik_solver_interface.compute_end_effector_pose()
                    print(f"IK solver test successful. Current EE position: {current_position}")
                    break
                    
                except Exception as e:
                    print(f"IK solver test attempt {test_attempt + 1} failed: {e}")
                    if test_attempt < test_attempts - 1:
                        time.sleep(1.0)
                        continue
                    else:
                        raise Exception(f"IK solver test failed after {test_attempts} attempts: {e}")
            
            # 初始化軌跡生成器
            print("Initializing trajectory generator...")
            try:
                # 驗證配置檔案
                print("🔍 驗證軌跡生成器配置:")
                print(f"   機器人描述檔案: {self.config.ROBOT_DESCRIPTION_PATH}")
                print(f"   URDF 檔案: {self.config.URDF_PATH}")
                
                # 檢查檔案是否存在
                if not os.path.exists(self.config.ROBOT_DESCRIPTION_PATH):
                    raise Exception(f"機器人描述檔案不存在: {self.config.ROBOT_DESCRIPTION_PATH}")
                
                if not os.path.exists(self.config.URDF_PATH):
                    raise Exception(f"URDF 檔案不存在: {self.config.URDF_PATH}")
                
                print("✅ 配置檔案驗證通過")
                
                # 創建軌跡生成器
                print("🎬 創建 Lula 軌跡生成器...")
                self.c_traj_gen = LulaCSpaceTrajectoryGenerator(
                    robot_description_path=self.config.ROBOT_DESCRIPTION_PATH,
                    urdf_path=self.config.URDF_PATH
                )
                
                print(f"✅ 軌跡生成器創建成功: {type(self.c_traj_gen)}")
                
                # 測試軌跡生成器功能
                print("🧪 測試軌跡生成器基本功能...")
                try:
                    # 嘗試創建一個簡單的測試軌跡
                    current_joints = self.robot_articulation.get_joint_positions()
                    if current_joints is not None and len(current_joints) > 0:
                        # 創建一個小的關節變化作為測試
                        test_waypoints = np.array([
                            current_joints,
                            current_joints + 0.01  # 很小的變化
                        ])
                        
                        print(f"   測試軌跡點形狀: {test_waypoints.shape}")
                        print(f"   測試關節數量: {len(current_joints)}")
                        
                        # 嘗試生成測試軌跡
                        test_traj = self.c_traj_gen.compute_c_space_trajectory(test_waypoints)
                        
                        if test_traj is not None:
                            print("✅ 軌跡生成器測試成功")
                        else:
                            print("⚠️ 軌跡生成器測試返回 None，但初始化可能仍然成功")
                    else:
                        print("⚠️ 無法獲取當前關節位置進行測試")
                        
                except Exception as test_e:
                    print(f"⚠️ 軌跡生成器測試失敗: {test_e}")
                    print("   這可能不會影響正常使用")
                
            except Exception as e:
                print(f"❌ 軌跡生成器初始化失敗: {e}")
                print(f"   錯誤類型: {type(e).__name__}")
                print(f"   可能原因:")
                print(f"   • 機器人描述檔案格式錯誤")
                print(f"   • URDF 檔案格式錯誤")
                print(f"   • Lula 依賴項未正確安裝")
                print(f"   • 檔案路徑包含特殊字符")
                raise
            
            self._initialized = True
            print("Robot controller initialized successfully")
            return True
            
        except Exception as e:
            print(f"Robot initialization failed: {e}")
            self._initialized = False
            return False       
    def reset(self):
        """重置機器人狀態"""
        try:
            if self.robot_articulation:
                # 重置關節位置到初始狀態
                self.robot_articulation.initialize()
                
                # 等待幾個時間步
                world = World.instance()
                if world:
                    for _ in range(10):
                        world.step(render=True)
                        time.sleep(0.05)
                
                print("Robot reset completed")
                return True
        except Exception as e:
            print(f"Robot reset failed: {e}")
        return False

    def is_initialized(self):
        """檢查是否已初始化"""
        return self._initialized and self.robot_articulation is not None

    def get_robot_info(self):
        """獲取機器人狀態信息"""
        if not self.is_initialized():
            return None
            
        try:
            current_position, current_rotation = self.ik_solver_interface.compute_end_effector_pose(position_only=False)
            joint_positions = self.robot_articulation.get_joint_positions()
            
            return {
                "position": current_position,
                "rotation": current_rotation,
                "joint_positions": joint_positions,
                "joint_count": len(joint_positions) if joint_positions is not None else 0
            }
        except Exception as e:
            print(f"Error getting robot info: {e}")
            return None

# 3. 軌跡執行器模組
class TrajectoryExecutor:
    """專門處理軌跡執行邏輯"""
    def __init__(self, robot_controller, config, ndi_detector=None):
        self.robot_controller = robot_controller
        self.config = config
        self.ndi_detector = ndi_detector  # 添加 NDI 檢測器引用
        self.execution_active = False
        self.stop_requested = False
        self.action_queue = []
        self.action_index = 0
        self.execution_mode = False
        self.paused = False
        self.progress_callback = None  # 添加進度回調存儲



    def execute_trajectory_from_csv(self, progress_callback=None):
        """從CSV準備軌跡執行 - 修正reset後的問題"""
        if self.execution_active and not self.paused:
            return False, "Already executing"
            
        if not self.robot_controller.is_initialized():
            return False, "Robot not initialized"
        
        # 如果是暫停狀態，恢復執行
        if self.paused:
            self.paused = False
            self.execution_active = True
            self.execution_mode = True
            self.stop_requested = False  # 確保重置停止標誌
            self.progress_callback = progress_callback
            if progress_callback:
                progress_callback("Resuming execution...", 100, 100)
            print(f"Resuming trajectory execution from action {self.action_index}/{len(self.action_queue)}")
            return True, f"Resumed execution from action {self.action_index}"
        
        try:
            # 重要：在開始新執行前重置所有狀態標誌
            print("Resetting execution flags for new trajectory...")
            self.stop_requested = False  # 確保不會立即停止
            self.execution_active = False  # 將在最後設為True
            self.execution_mode = False
            self.paused = False
            self.action_index = 0
            self.action_queue = []
            
            # 保存進度回調
            self.progress_callback = progress_callback
            
            # 檢查CSV文件是否存在
            if not os.path.exists(self.config.CSV_PATH):
                return False, f"CSV file not found: {self.config.CSV_PATH}"
            
            # 讀取CSV
            if progress_callback:
                progress_callback("Reading CSV...", 0, 100)
                
            print(f"Reading CSV from: {self.config.CSV_PATH}")
            
            # 先嘗試讀取原始數據
            try:
                df_raw = pd.read_csv(self.config.CSV_PATH)
                print(f"Raw CSV shape: {df_raw.shape}")
                print(f"Raw CSV columns: {df_raw.columns.tolist()}")
                print(f"First few rows:\n{df_raw.head()}")
            except Exception as e:
                return False, f"Failed to read CSV: {e}"
            
            # 轉換為數值數據
            try:
                # 嘗試轉換所有列為數值
                df = df_raw.apply(pd.to_numeric, errors='coerce')
                
                # 移除包含NaN的行
                df_clean = df.dropna()
                
                if df_clean.empty:
                    return False, "No valid numeric data found in CSV"
                
                # 檢查是否有足夠的列
                if df_clean.shape[1] < 7:
                    return False, f"Insufficient columns in CSV: {df_clean.shape[1]} < 7 (need: time, x, y, z, rx, ry, rz)"
                
                print(f"Cleaned CSV shape: {df_clean.shape}")
                
            except Exception as e:
                return False, f"Data conversion failed: {e}"
            
            # 確保停止標誌在IK計算前是False
            print(f"Before IK computation - stop_requested: {self.stop_requested}")
            
            # 計算IK解
            waypoints = self._compute_ik_solutions_sync(df_clean, progress_callback)
            if waypoints is None:
                return False, "IK computation failed"
            
            # 生成軌跡
            if progress_callback:
                progress_callback("Generating trajectory...", 75, 100)
                
            # 添加詳細的軌跡生成調試信息
            print(f"\n🎬 軌跡生成調試信息:")
            print(f"   Waypoints 數量: {len(waypoints)}")
            print(f"   Waypoints 形狀: {waypoints.shape}")
            print(f"   每個 waypoint 的關節數: {waypoints.shape[1] if len(waypoints.shape) > 1 else 'N/A'}")
            
            # 檢查 waypoints 的數據範圍
            if len(waypoints) > 0:
                waypoint_min = np.min(waypoints, axis=0)
                waypoint_max = np.max(waypoints, axis=0)
                waypoint_range = waypoint_max - waypoint_min
                print(f"   關節角度範圍 (rad): {waypoint_min} 到 {waypoint_max}")
                print(f"   最大關節變化量: {waypoint_range}")
                print(f"   最大單關節變化: {np.max(waypoint_range):.3f} rad ({np.max(waypoint_range) * 180 / np.pi:.1f} deg)")
                
                # 檢查相鄰點之間的距離
                if len(waypoints) > 1:
                    max_step_size = 0
                    for i in range(1, len(waypoints)):
                        step_diff = np.max(np.abs(waypoints[i] - waypoints[i-1]))
                        max_step_size = max(max_step_size, step_diff)
                    print(f"   最大步長: {max_step_size:.3f} rad ({max_step_size * 180 / np.pi:.1f} deg)")
                    
                    # 如果步長太大，嘗試插值
                    if max_step_size > 0.5:  # 超過0.5弧度約28.6度
                        print(f"⚠️  檢測到大步長，嘗試插值...")
                        original_count = len(waypoints)
                        waypoints = self._interpolate_waypoints(waypoints, max_step=0.3)
                        print(f"   插值後: {original_count} -> {len(waypoints)} waypoints")
            
            try:
                print(f"🎬 調用 Lula 軌跡生成器...")
                print(f"   生成器類型: {type(self.robot_controller.c_traj_gen)}")
                
                # 嘗試生成軌跡
                traj = self.robot_controller.c_traj_gen.compute_c_space_trajectory(waypoints)
                
                if traj is None:
                    error_msg = "Lula 軌跡生成返回 None。可能原因:\n"
                    error_msg += f"   • Waypoints 數量太少 ({len(waypoints)})\n"
                    error_msg += f"   • 軌跡點之間距離太大\n"
                    error_msg += f"   • 關節限制衝突\n"
                    error_msg += f"   • 機器人描述檔案問題"
                    print(f"❌ {error_msg}")
                    return False, error_msg
                    
                print(f"✅ 軌跡生成成功!")
                print(f"   軌跡類型: {type(traj)}")
                
                # 嘗試獲取軌跡信息
                try:
                    if hasattr(traj, 'get_num_time_samples'):
                        num_samples = traj.get_num_time_samples()
                        print(f"   時間採樣點數: {num_samples}")
                    if hasattr(traj, 'get_duration'):
                        duration = traj.get_duration()
                        print(f"   軌跡持續時間: {duration:.3f} 秒")
                except:
                    print(f"   無法獲取軌跡詳細信息")
                    
            except Exception as e:
                print(f"❌ 軌跡生成異常:")
                print(f"   錯誤類型: {type(e).__name__}")
                print(f"   錯誤訊息: {str(e)}")
                print(f"   Traceback:")
                traceback.print_exc()
                
                # 提供解決建議
                error_suggestions = f"\n💡 可能的解決方案:\n"
                error_suggestions += f"   1. 檢查機器人描述檔案: {self.robot_controller.config.ROBOT_DESCRIPTION_PATH}\n"
                error_suggestions += f"   2. 檢查 URDF 檔案: {self.robot_controller.config.URDF_PATH}\n"
                error_suggestions += f"   3. 減少軌跡點數量或增加插值\n"
                error_suggestions += f"   4. 檢查關節限制設定"
                print(error_suggestions)
                
                return False, f"軌跡生成錯誤: {e}"
                
            try:
                artic_traj = ArticulationTrajectory(
                    self.robot_controller.robot_articulation, 
                    traj, 
                    self.config.PHYSICS_DT
                )
                self.action_queue = artic_traj.get_action_sequence()
            except Exception as e:
                return False, f"Articulation trajectory creation failed: {e}"
            
            # 最後設置執行狀態
            self.execution_active = True
            self.execution_mode = True
            self.action_index = 0
            self.stop_requested = False  # 再次確認
            self.paused = False
            
            # 💰 啟動成本計算
            if self.ndi_detector:
                self.ndi_detector.start_cost_calculation()
            
            if progress_callback:
                progress_callback("Trajectory ready, starting execution...", 100, 100)
                
            print(f"Trajectory prepared with {len(self.action_queue)} actions")
            print(f"Final state - stop_requested: {self.stop_requested}, execution_active: {self.execution_active}")
            return True, f"Trajectory prepared with {len(self.action_queue)} actions"
            
        except Exception as e:
            print(f"Trajectory preparation error: {e}")
            return False, f"Preparation failed: {e}"   
    
    # 在 TrajectoryExecutor 類別中修改 _compute_ik_solutions_sync 方法的開始部分
    def _compute_ik_solutions_sync(self, df, progress_callback):
        """同步計算IK解 - 加入時間分析和速度控制"""
        try:
            total_steps = len(df)
            if total_steps == 0:
                raise Exception("DataFrame is empty")
                
            waypoints = []
            
            print(f"Processing {total_steps} trajectory points")
            print(f"Initial stop_requested state: {self.stop_requested}")
            
            # 🕒 時間分析 - 新增
            self._analyze_trajectory_timing(df)
            
            # 驗證機器人狀態
            if not self.robot_controller.is_initialized():
                raise Exception("Robot controller not initialized")
            
            # 額外驗證：確保機器人關節真的可用
            print("Verifying robot joint availability...")
            max_verification_attempts = 5
            for verify_attempt in range(max_verification_attempts):
                try:
                    joint_positions = self.robot_controller.robot_articulation.get_joint_positions()
                    if joint_positions is not None and len(joint_positions) > 0:
                        print(f"Robot verification successful: {len(joint_positions)} joints available")
                        break
                    else:
                        print(f"Verification attempt {verify_attempt + 1}: Joints not available")
                        if verify_attempt < max_verification_attempts - 1:
                            time.sleep(1.0)
                            # 額外的world步進
                            world = World.instance()
                            if world:
                                for _ in range(10):
                                    world.step(render=True)
                                    time.sleep(0.1)
                            continue
                        else:
                            raise Exception("Robot joints never became available for IK computation")
                except Exception as e:
                    print(f"Joint verification attempt {verify_attempt + 1} failed: {e}")
                    if verify_attempt == max_verification_attempts - 1:
                        raise Exception(f"Joint verification failed: {e}")
                    time.sleep(1.0)
            
            # 驗證IK求解器狀態
            print("Verifying IK solver state...")
            try:
                current_position, current_rotation = self.robot_controller.ik_solver_interface.compute_end_effector_pose()
                print(f"IK solver verification successful. EE position: {current_position}")
            except Exception as e:
                raise Exception(f"IK solver verification failed: {e}")
            
            # 小心提取位置和旋轉數據
            try:
                # 假設CSV格式為: [time, x, y, z, rx, ry, rz, ...]
                pos_mm = df.iloc[:, 1:4].to_numpy()  # 列1-3: x, y, z (毫米)
                pos_m = pos_mm / 1000.0  # 轉換為米
                rotation_rad = df.iloc[:, 4:7].to_numpy()  # 列4-6: rx, ry, rz (弧度)
                
                print(f"Position data shape: {pos_m.shape}")
                print(f"Rotation data shape: {rotation_rad.shape}")
                
            except Exception as e:
                raise Exception(f"Data extraction failed: {e}")
            
            # 其餘代碼保持不變...
            print(f"Position range (m): {np.min(pos_m, axis=0)} to {np.max(pos_m, axis=0)}")
            print(f"Rotation range (rad): {np.min(rotation_rad, axis=0)} to {np.max(rotation_rad, axis=0)}")
            
            # 檢查數據合理性
            if np.any(np.abs(pos_m) > 10):  # 位置超過10米可能不合理
                print("Warning: Position values seem very large")
            
            if np.any(np.abs(rotation_rad) > 2*np.pi):  # 旋轉角度超過2π可能需要檢查
                print("Warning: Rotation values seem large, assuming radians")
            
            # 轉換旋轉角度為四元數
            try:
                rotation_quat = euler_angles_to_quats(rotation_rad)
                print(f"Quaternion data shape: {rotation_quat.shape}")
            except Exception as e:
                raise Exception(f"Quaternion conversion failed: {e}")
            
            # 測試第一個IK計算 - 更保守的方法
            print("Testing first IK calculation...")
            test_ik_attempts = 3
            for ik_test_attempt in range(test_ik_attempts):
                try:
                    joint_angles, success = self.robot_controller.ik_solver_interface.compute_inverse_kinematics(
                        pos_m[0], rotation_quat[0]
                    )
                    
                    if not success:
                        print(f"IK test attempt {ik_test_attempt + 1}: IK computation failed for position: {pos_m[0]}, quaternion: {rotation_quat[0]}")
                        if ik_test_attempt < test_ik_attempts - 1:
                            time.sleep(0.5)
                            continue
                        else:
                            raise Exception("Initial IK test failed - target may be unreachable")
                    
                    print(f"IK test successful on attempt {ik_test_attempt + 1}, proceeding with full calculation...")
                    break
                    
                except Exception as e:
                    print(f"IK test attempt {ik_test_attempt + 1} failed with error: {e}")
                    if ik_test_attempt == test_ik_attempts - 1:
                        raise Exception(f"IK test failed after {test_ik_attempts} attempts: {e}")
                    time.sleep(0.5)
                        
            # 處理所有waypoints
            failed_points = []
            for i in range(total_steps):
                # 添加調試信息
                if i == 0:
                    print(f"Starting IK computation - stop_requested: {self.stop_requested}")
                
                if self.stop_requested:
                    print(f"IK computation stopped by user at step {i} - stop_requested: {self.stop_requested}")
                    break
                    
                try:
                    joint_angles, success = self.robot_controller.ik_solver_interface.compute_inverse_kinematics(
                        pos_m[i], rotation_quat[i]
                    )
                    
                    if not success:
                        failed_points.append(i)
                        print(f"IK failed at step {i+1}/{total_steps} - Position: {pos_m[i]}")
                        
                        # 如果失敗太多，就停止
                        if len(failed_points) > total_steps * 0.1:  # 超過10%失敗
                            raise Exception(f"Too many IK failures: {len(failed_points)}/{i+1}")
                        
                        # 跳過失敗的點，或使用前一個成功的點
                        if waypoints:
                            waypoints.append(waypoints[-1])  # 重複上一個成功的點
                        else:
                            continue  # 如果是第一個點就跳過
                    else:
                        # IK成功
                        if hasattr(joint_angles, 'joint_positions'):
                            joint_positions = joint_angles.joint_positions
                        else:
                            joint_positions = joint_angles
                            
                        waypoints.append(joint_positions)
                    
                    # 進度回調
                    if progress_callback and i % 50 == 0:
                        progress = int(25 + (i / total_steps) * 50)  # 25-75%
                        progress_callback(f"Computing IK: {i+1}/{total_steps}", progress, 100)
                        
                except Exception as e:
                    print(f"IK computation failed at step {i}: {e}")
                    raise
            
            if not waypoints:
                print(f"No valid waypoints generated - processed {len([i for i in range(total_steps) if not self.stop_requested or i == 0])} steps")
                raise Exception("No valid waypoints generated")
            
            if failed_points:
                print(f"Warning: {len(failed_points)} IK calculations failed out of {total_steps}")
            
            print(f"IK computation completed. Generated {len(waypoints)} waypoints")
            return np.array(waypoints)
            
        except Exception as e:
            print(f"IK computation error: {e}")
            return None
        
    def _analyze_trajectory_timing(self, df):
        """分析軌跡時間特性並提供執行速度建議"""
        try:
            # 提取時間數據
            time_data = df.iloc[:, 0].to_numpy()
            time_start = time_data[0]
            time_end = time_data[-1]
            time_diff = time_end - time_start
            data_points = len(df)
            
            print(f"\n⏱️ 軌跡時間分析:")
            print(f"   起始時間: {time_start}")
            print(f"   結束時間: {time_end}")
            print(f"   時間差: {time_diff}")
            print(f"   數據點數: {data_points}")
            
            # 判斷時間單位
            if time_diff > 100000:  # 可能是微秒
                time_unit = "微秒"
                duration_seconds = time_diff / 1000000
                time_per_point = time_diff / (data_points - 1) / 1000  # 毫秒
            elif time_diff > 1000:  # 可能是毫秒
                time_unit = "毫秒"
                duration_seconds = time_diff / 1000
                time_per_point = time_diff / (data_points - 1)  # 毫秒
            else:  # 已經是秒
                time_unit = "秒"
                duration_seconds = time_diff
                time_per_point = time_diff / (data_points - 1) * 1000  # 轉為毫秒
            
            print(f"   檢測到時間單位: {time_unit}")
            print(f"   軌跡總時間: {duration_seconds:.3f} 秒")
            print(f"   平均時間間隔: {time_per_point:.2f} 毫秒/點")
            
            # 與物理時間步長比較
            physics_dt_ms = self.config.PHYSICS_DT * 1000
            print(f"   當前物理步長: {physics_dt_ms:.1f} 毫秒")
            
            # 執行時間預估
            estimated_actions = data_points * 8  # 估算軌跡生成後的動作數量
            estimated_time = estimated_actions * self.config.PHYSICS_DT
            
            print(f"\n🎯 執行預估:")
            print(f"   預估動作數量: {estimated_actions}")
            print(f"   預估執行時間: {estimated_time:.2f} 秒")
            
            # 速度建議
            if estimated_time > 30:
                print(f"⚠️  預估執行時間較長，建議:")
                print(f"   • 增加 PHYSICS_DT 到 1/125 (8ms)")
                print(f"   • 或減少數據點數")
            elif estimated_time < 2:
                print(f"💡 執行時間較短，如需更精確可:")
                print(f"   • 減少 PHYSICS_DT 到 1/500 (2ms)")
                print(f"   • 或增加數據點數")
            else:
                print(f"✅ 執行時間適中")
                
        except Exception as e:
            print(f"⚠️ 時間分析失敗: {e}")
    
    def _interpolate_waypoints(self, waypoints, max_step=0.5):
        """插值軌跡點以避免大幅跳躍"""
        if len(waypoints) < 2:
            return waypoints
            
        interpolated = [waypoints[0]]
        
        for i in range(1, len(waypoints)):
            current = waypoints[i-1]
            next_point = waypoints[i]
            
            # 計算關節移動距離
            diff = np.array(next_point) - np.array(current)
            max_diff = np.max(np.abs(diff))
            
            if max_diff > max_step:
                # 需要插值
                num_segments = int(np.ceil(max_diff / max_step))
                for j in range(1, num_segments + 1):
                    t = j / num_segments
                    interpolated_point = current + t * diff
                    interpolated.append(interpolated_point)
            else:
                interpolated.append(next_point)
                
        return np.array(interpolated)
    
    def update_execution(self):
        """在主循環中調用此方法來更新執行狀態"""
        if not self.execution_mode or not self.execution_active or self.paused:
            return True  # 繼續運行
            
        if self.stop_requested or self.action_index >= len(self.action_queue):
            self.execution_active = False
            self.execution_mode = False
            self.paused = False
            
            # 💰 停止成本計算並獲取結果
            if self.ndi_detector:
                cost_results = self.ndi_detector.stop_cost_calculation()
                if cost_results:
                    print(f"\n🎉 軌跡執行完成！最終成本結果:")
                    print(f"   總成本: {cost_results['total_cost']:.6f}")
                    print(f"   總幀數: {cost_results['frame_count']}")
                    print(f"   平均成本: {cost_results['average_cost_per_frame']:.6f}\n")
            
            # 執行完成時通知UI
            if self.progress_callback:
                self.progress_callback("Execution completed", 100, 100)
            print("Trajectory execution completed")
            return True
            
        try:
            # 在主線程中安全地應用動作
            action = self.action_queue[self.action_index]
            self.robot_controller.robot_articulation.apply_action(action)
            self.action_index += 1
            
            # 實時更新執行進度
            if len(self.action_queue) > 0 and self.progress_callback:
                progress = int((self.action_index / len(self.action_queue)) * 100)
                
                # 每10步更新一次UI
                if self.action_index % 10 == 0:
                    self.progress_callback(
                        f"Executing: {self.action_index}/{len(self.action_queue)}", 
                        self.action_index, 
                        len(self.action_queue)
                    )
            
            # 每50步打印一次進度
            if self.action_index % 50 == 0:
                progress = int((self.action_index / len(self.action_queue)) * 100)
                print(f"Execution progress: {progress}% ({self.action_index}/{len(self.action_queue)})")
            
        except Exception as e:
            print(f"Action execution error: {e}")
            self.execution_active = False
            self.execution_mode = False
            self.paused = False
            if self.progress_callback:
                self.progress_callback("Execution failed", 0, 100)
            
        return True
    
    def stop_execution(self):
        """暫停執行（改為暫停而非完全停止）"""
        if self.execution_active:
            self.paused = True
            self.execution_active = False
            print(f"Trajectory execution paused at action {self.action_index}/{len(self.action_queue)}")
        else:
            print("No active execution to pause")
    
    
    def move_to_initial_position_from_csv(self, progress_callback=None):
        """從CSV讀取第一個位置並移動到該位置 - 修復版本"""
        if self.execution_active or getattr(self, 'moving_to_initial', False):
            return False, "Cannot move to initial position while executing or already moving"
            
        if not self.robot_controller.is_initialized():
            return False, "Robot not initialized"
        
        try:
            # 檢查CSV文件是否存在
            if not os.path.exists(self.config.CSV_PATH):
                return False, f"CSV file not found: {self.config.CSV_PATH}"
            
            if progress_callback:
                progress_callback("Reading CSV for initial position...", 0, 100)
                
            print(f"📖 Reading CSV to get initial position: {self.config.CSV_PATH}")
            
            # 讀取CSV
            try:
                df_raw = pd.read_csv(self.config.CSV_PATH)
                print(f"Raw CSV shape: {df_raw.shape}")
                print(f"CSV columns: {df_raw.columns.tolist()}")
                print(f"First row data:\n{df_raw.iloc[0]}")
            except Exception as e:
                return False, f"Failed to read CSV: {e}"
            
            # 轉換數據
            try:
                df = df_raw.apply(pd.to_numeric, errors='coerce')
                df_clean = df.dropna()
                
                if df_clean.empty:
                    return False, "No valid numeric data found in CSV"
                
                if df_clean.shape[1] < 7:
                    return False, f"Insufficient columns in CSV: {df_clean.shape[1]} < 7"
                
                print(f"Cleaned CSV shape: {df_clean.shape}")
                
            except Exception as e:
                return False, f"Data conversion failed: {e}"
            
            if progress_callback:
                progress_callback("Computing initial position IK...", 25, 100)
            
            # 只計算第一個點的IK
            try:
                # 提取第一行數據 - 更詳細的調試
                first_row = df_clean.iloc[0]
                print(f"First row values: {first_row.values}")
                
                # 假設格式：[time, x, y, z, rx, ry, rz, ...]
                pos_mm = first_row.iloc[1:4].to_numpy()  # x, y, z (毫米)
                pos_m = pos_mm / 1000.0  # 轉換為米
                rotation_rad = first_row.iloc[4:7].to_numpy()  # rx, ry, rz (弧度)
                
                print(f"📍 Target position (mm): {pos_mm}")
                print(f"📍 Target position (m): {pos_m}")
                print(f"🔄 Target rotation (rad): {rotation_rad}")
                print(f"🔄 Target rotation (deg): {rotation_rad * 180 / np.pi}")
                
                # 驗證數據合理性
                if np.any(np.abs(pos_m) > 5.0):  # 位置超過5米可能不合理
                    print(f"⚠️ Warning: Position values seem large: {pos_m}")
                
                # 轉換旋轉角度為四元數
                rotation_quat = euler_angles_to_quats(rotation_rad.reshape(1, -1))[0]
                print(f"🔄 Target quaternion: {rotation_quat}")
                
                # 檢查當前機器人狀態
                try:
                    current_joint_positions = self.robot_controller.robot_articulation.get_joint_positions()
                    current_ee_pos, current_ee_rot = self.robot_controller.ik_solver_interface.compute_end_effector_pose()
                    print(f"🤖 Current joint positions: {current_joint_positions}")
                    print(f"🤖 Current EE position: {current_ee_pos}")
                    print(f"🤖 Current EE rotation: {current_ee_rot}")
                except Exception as e:
                    print(f"⚠️ Could not get current robot state: {e}")
                
                # 計算IK - 多次嘗試
                ik_success = False
                target_joint_positions = None
                
                for ik_attempt in range(3):
                    try:
                        print(f"🔧 IK attempt {ik_attempt + 1}/3...")
                        joint_angles, success = self.robot_controller.ik_solver_interface.compute_inverse_kinematics(
                            pos_m, rotation_quat
                        )
                        
                        if success:
                            # 獲取關節位置
                            if hasattr(joint_angles, 'joint_positions'):
                                target_joint_positions = joint_angles.joint_positions
                            else:
                                target_joint_positions = joint_angles
                                
                            print(f"✅ IK successful! Target joint positions: {target_joint_positions}")
                            ik_success = True
                            break
                        else:
                            print(f"❌ IK attempt {ik_attempt + 1} failed")
                            if ik_attempt < 2:
                                time.sleep(0.5)
                                
                    except Exception as e:
                        print(f"❌ IK attempt {ik_attempt + 1} error: {e}")
                        if ik_attempt < 2:
                            time.sleep(0.5)
                
                if not ik_success:
                    return False, f"All IK attempts failed for position: {pos_m}, rotation: {rotation_rad}"
                    
            except Exception as e:
                print(f"❌ Initial position IK computation failed: {e}")
                traceback.print_exc()
                return False, f"Initial position IK computation failed: {e}"
            
            if progress_callback:
                progress_callback("Generating trajectory to initial position...", 50, 100)
            
            # 生成移動軌跡
            try:
                # 獲取當前關節位置
                current_joint_positions = self.robot_controller.robot_articulation.get_joint_positions()
                print(f"🤖 Current joint positions: {current_joint_positions}")
                print(f"🎯 Target joint positions: {target_joint_positions}")
                
                # 計算關節移動距離
                joint_diff = np.array(target_joint_positions) - np.array(current_joint_positions)
                max_joint_diff = np.max(np.abs(joint_diff))
                print(f"📏 Maximum joint movement: {max_joint_diff:.3f} rad ({max_joint_diff * 180 / np.pi:.1f} deg)")
                
                # 檢查是否已經在目標位置附近
                if max_joint_diff < 0.01:  # 小於0.01弧度(約0.6度)
                    print("✅ Already at target position!")
                    return True, "Already at initial position"
                
                # 創建兩點軌跡
                waypoints = np.array([current_joint_positions, target_joint_positions])
                print(f"📋 Waypoints shape: {waypoints.shape}")
                
                # 生成軌跡
                print("🎬 Generating initial position trajectory...")
                print(f"📊 Current joints: {current_joint_positions}")
                print(f"📊 Target joints: {target_joint_positions}")
                
                # 檢查關節移動距離
                joint_diff = np.array(target_joint_positions) - np.array(current_joint_positions)
                max_joint_diff = np.max(np.abs(joint_diff))
                print(f"📊 Maximum joint movement: {max_joint_diff:.3f} rad ({max_joint_diff * 180 / np.pi:.1f} deg)")
                
                # 創建兩點軌跡
                waypoints = np.array([current_joint_positions, target_joint_positions])
                print(f"📋 Waypoints shape: {waypoints.shape}")
                
                # 如果移動距離太大，嘗試插值
                if max_joint_diff > 1.0:  # 超過1弧度
                    print(f"⚠️ Large joint movement detected, attempting interpolation...")
                    waypoints = self._interpolate_waypoints(waypoints, max_step=0.3)
                    print(f"📊 After interpolation: {len(waypoints)} waypoints")
                
                print("🎬 Calling Lula trajectory generator for initial move...")
                traj = self.robot_controller.c_traj_gen.compute_c_space_trajectory(waypoints)
                
                if traj is None:
                    # 初始位置軌跡生成失敗的回退策略
                    print("❌ Initial trajectory generation failed, trying simpler approach...")
                    
                    # 嘗試更保守的插值
                    if max_joint_diff > 0.1:  # 更小的步長
                        simple_waypoints = self._interpolate_waypoints(
                            np.array([current_joint_positions, target_joint_positions]), 
                            max_step=0.1
                        )
                        print(f"🔧 Trying conservative interpolation with {len(simple_waypoints)} points")
                        traj = self.robot_controller.c_traj_gen.compute_c_space_trajectory(simple_waypoints)
                    
                    if traj is None:
                        error_msg = f"Initial position trajectory generation failed. "
                        error_msg += f"Joint movement: {max_joint_diff:.3f} rad, "
                        error_msg += f"Distance might be too large for single trajectory."
                        return False, error_msg
                    
                print("🎬 Creating articulation trajectory...")
                artic_traj = ArticulationTrajectory(
                    self.robot_controller.robot_articulation, 
                    traj, 
                    self.config.PHYSICS_DT
                )
                
                # 設置為初始化軌跡（不是主執行軌跡）
                self.initial_move_queue = artic_traj.get_action_sequence()
                self.initial_move_index = 0
                self.moving_to_initial = True
                self.initial_move_active = True
                
                if progress_callback:
                    progress_callback("Moving to initial position...", 75, 100)
                
                print(f"✅ Initial position trajectory prepared with {len(self.initial_move_queue)} actions")
                print(f"🚀 Starting movement to initial position...")
                return True, f"Moving to initial position with {len(self.initial_move_queue)} actions"
                
            except Exception as e:
                print(f"❌ Initial position trajectory creation failed: {e}")
                import traceback
                traceback.print_exc()
                return False, f"Initial position trajectory creation failed: {e}"
                
        except Exception as e:
            print(f"❌ Move to initial position error: {e}")
            import traceback
            traceback.print_exc()
            return False, f"Move to initial failed: {e}"
    
    def update_initial_move(self):
        """更新移動到初始位置的執行 - 加強調試"""
        if not getattr(self, 'moving_to_initial', False) or not getattr(self, 'initial_move_active', False):
            return True
            
        try:
            if self.initial_move_index >= len(self.initial_move_queue):
                # 移動完成
                self.moving_to_initial = False
                self.initial_move_active = False
                final_joint_positions = self.robot_controller.robot_articulation.get_joint_positions()
                print(f"✅ Successfully moved to initial position!")
                print(f"📍 Final joint positions: {final_joint_positions}")
                
                # 驗證到達位置
                try:
                    final_ee_pos, final_ee_rot = self.robot_controller.ik_solver_interface.compute_end_effector_pose()
                    print(f"📍 Final EE position: {final_ee_pos}")
                    print(f"🔄 Final EE rotation: {final_ee_rot}")
                except Exception as e:
                    print(f"⚠️ Could not verify final position: {e}")
                
                self.initial_move_index = 0
                self.initial_move_queue = []
                
                if hasattr(self, 'progress_callback') and self.progress_callback:
                    self.progress_callback("Reached initial position", 100, 100)
                
                return True
            
            # 執行移動動作
            action = self.initial_move_queue[self.initial_move_index]
            self.robot_controller.robot_articulation.apply_action(action)
            self.initial_move_index += 1
            
            # 更新進度和調試信息
            if self.initial_move_index % 50 == 0:  # 每50步打印一次
                current_joints = self.robot_controller.robot_articulation.get_joint_positions()
                progress = int((self.initial_move_index / len(self.initial_move_queue)) * 100)
                print(f"🚀 Initial move progress: {progress}% ({self.initial_move_index}/{len(self.initial_move_queue)})")
                print(f"🤖 Current joints: {current_joints}")
            
            if hasattr(self, 'progress_callback') and self.progress_callback and self.initial_move_index % 10 == 0:
                progress = int((self.initial_move_index / len(self.initial_move_queue)) * 100)
                self.progress_callback(
                    f"Moving to initial: {self.initial_move_index}/{len(self.initial_move_queue)}", 
                    self.initial_move_index, 
                    len(self.initial_move_queue)
                )
            
            return True
            
        except Exception as e:
            print(f"❌ Initial move execution error: {e}")
            import traceback
            traceback.print_exc()
            self.moving_to_initial = False
            self.initial_move_active = False
            if hasattr(self, 'progress_callback') and self.progress_callback:
                self.progress_callback("Initial move failed", 0, 100)
            return 
        
    def is_moving_to_initial(self):
        """檢查是否正在移動到初始位置"""
        return getattr(self, 'moving_to_initial', False)

    def stop_initial_move(self):
        """停止移動到初始位置"""
        if self.moving_to_initial:
            self.moving_to_initial = False
            self.initial_move_active = False
            print("Initial move stopped")

    def reset_execution(self):
        """重置執行狀態 - 包含初始移動狀態"""
        try:
            print("Resetting trajectory execution...")
            
            # 重置主執行狀態
            self.stop_requested = True
            self.execution_active = False
            self.execution_mode = False
            self.paused = False
            
            # 重置初始移動狀態
            self.moving_to_initial = False
            self.initial_move_active = False
            self.initial_move_index = 0
            if hasattr(self, 'initial_move_queue'):
                self.initial_move_queue = []
            
            # 清除軌跡數據
            self.action_queue = []
            self.action_index = 0
            self.progress_callback = None
            
            print("Trajectory execution reset completed")
            
        except Exception as e:
            print(f"Trajectory execution reset error: {e}")
            
    def is_executing(self):
        """檢查是否正在執行"""
        return self.execution_active
    
    def is_paused(self):
        """檢查是否暫停"""
        return self.paused
    
    def get_execution_status(self):
        """獲取執行狀態"""
        if self.paused:
            return f"Paused at {self.action_index}/{len(self.action_queue)}"
        elif self.execution_active:
            return f"Executing {self.action_index}/{len(self.action_queue)}"
        elif len(self.action_queue) > 0:
            return "Ready to execute"
        else:
            return "No trajectory loaded"
    
# 3.5. 標記體積和固態角計算器
class MarkerVolumeCalculator:
    """專門計算標記體積和固態角的類別 - 針對單一NDI"""
    
    def __init__(self, emitter_path="/Root/NDI_02"):
        """初始化體積計算器
        
        Args:
            emitter_path: NDI發射器路徑（針對單一NDI，預設為 /Root/NDI_02）
        """
        self.emitter_path = emitter_path
        self.emitter_position = None
    
    def get_emitter_position(self, get_prim_position_func):
        """獲取NDI發射器的當前位置
        
        Args:
            get_prim_position_func: 獲取prim位置的函數
            
        Returns:
            發射器位置或None
        """
        position = get_prim_position_func(self.emitter_path)
        if position:
            self.emitter_position = position
        return self.emitter_position
    
    def calculate_tetrahedron_volume(self, p0, p1, p2, p3):
        """計算四面體體積
        
        Args:
            p0, p1, p2, p3: 四面體頂點的3D座標
            
        Returns:
            四面體體積
        """
        # 轉換為numpy數組以便計算
        p0 = np.array(p0)
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        
        # 計算來自p0的向量
        v1 = p1 - p0
        v2 = p2 - p0
        v3 = p3 - p0
        
        # 使用標量三重乘積計算體積
        volume = abs(np.dot(v1, np.cross(v2, v3))) / 6.0
        return volume
    
    def calculate_marker_volume(self, visible_positions):
        """基於可見標記點使用凸包計算體積
        
        Args:
            visible_positions: 可見標記點位置列表
            
        Returns:
            總體積或None（如果點不足或沒有發射器）
        """
        if not self.emitter_position or len(visible_positions) < 3:
            return None
        
        try:
            # 結合發射器位置和可見位置
            all_points = np.vstack([self.emitter_position, visible_positions])
            
            # 計算凸包
            hull = ConvexHull(all_points)
            
            # 轉換為立方厘米
            return hull.volume * 1000000  # m³ to cm³
            
        except ImportError:
            # 如果scipy不可用，使用四面體方法
            return self._calculate_using_tetrahedra(visible_positions)
        except Exception as e:
            print(f"Volume calculation error: {e}")
            return None
            
    def _calculate_using_tetrahedra(self, visible_positions):
        """使用四面體方法計算體積（備用方法）
        
        Args:
            visible_positions: 可見標記點位置列表
            
        Returns:
            立方厘米體積
        """
        total_volume = 0
        
        if len(visible_positions) == 3:
            # 三個點：形成一個四面體
            total_volume = self.calculate_tetrahedron_volume(
                self.emitter_position, 
                visible_positions[0], 
                visible_positions[1], 
                visible_positions[2]
            )
        else:
            # 多個點：將它們組合成多個四面體
            for i in range(len(visible_positions) - 2):
                for j in range(i + 1, len(visible_positions) - 1):
                    for k in range(j + 1, len(visible_positions)):
                        volume = self.calculate_tetrahedron_volume(
                            self.emitter_position,
                            visible_positions[i],
                            visible_positions[j], 
                            visible_positions[k]
                        )
                        total_volume += volume
        
        return total_volume * 1000000

    def calculate_solid_angle(self, visible_positions):
        """計算從發射器觀察標記點的固態角
        
        Args:
            visible_positions: 可見標記點位置列表
            
        Returns:
            總固態角（球面度）或None
        """
        if not self.emitter_position or len(visible_positions) < 3:
            return None
        
        # 計算所有可能三點組合的固態角
        total_solid_angle = 0
        
        # 獲取所有可能的三點組合
        combinations = list(itertools.combinations(visible_positions, 3))
        
        # 使用固態角公式計算
        for triangle in combinations:
            a, b, c = triangle
            solid_angle = self._calculate_triangle_solid_angle(a, b, c)
            if solid_angle is not None:
                total_solid_angle += solid_angle
        
        # 返回平均固態角
        if len(combinations) > 0:
            return total_solid_angle / len(combinations)
        
        return None
    
    def _calculate_triangle_solid_angle(self, a, b, c):
        """計算三角形對觀察者的固態角
        
        Args:
            a, b, c: 三角形頂點座標
            
        Returns:
            固態角（球面度）
        """
        try:
            # 觀察者位置
            v = np.array(self.emitter_position)
            
            # 計算從觀察者到三角形頂點的向量
            A = np.array(a) - v
            B = np.array(b) - v
            C = np.array(c) - v
            
            # 計算向量長度
            a_len = np.linalg.norm(A)
            b_len = np.linalg.norm(B)
            c_len = np.linalg.norm(C)
            
            if a_len == 0 or b_len == 0 or c_len == 0:
                return 0
            
            # 單位向量
            A_unit = A / a_len
            B_unit = B / b_len
            C_unit = C / c_len
            
            # 計算標量三重乘積
            scalar_triple = np.dot(A_unit, np.cross(B_unit, C_unit))
            
            # 計算分母
            denominator = 1 + np.dot(A_unit, B_unit) + np.dot(B_unit, C_unit) + np.dot(C_unit, A_unit)
            
            if denominator <= 0:
                return 0
            
            # 計算固態角
            solid_angle = 2 * np.arctan2(abs(scalar_triple), denominator)
            
            return solid_angle
            
        except Exception as e:
            print(f"Solid angle calculation error: {e}")
            return 0

# 4. NDI檢測模組 (完全按照v2的邏輯實現)
class NDIDetector:
    """專門處理NDI檢測邏輯"""
    def __init__(self, config):
        self.config = config
        self.marker_status = {marker: False for marker in config.SUPPORTED_MARKERS}
        self.previous_marker_status = {marker: False for marker in config.SUPPORTED_MARKERS}
        self.raycast_results = []
        self.line_starts = []
        self.line_ends = []
        self.raycast_interface = None
        self.debug_draw_interface = None
        self.frame_count = 0
        self.ray_info = []
        self.trigger_detector = None
        self.trigger_volume_path = config.TRIGGER_VOLUME_PATH
        
        # 體積和固態角計算器 (針對單一NDI: /Root/NDI_02)
        self.volume_calculator = MarkerVolumeCalculator(emitter_path="/Root/NDI_02")
        self.marker_volumes = {marker: None for marker in config.SUPPORTED_MARKERS}
        self.marker_solid_angles = {marker: None for marker in config.SUPPORTED_MARKERS}  
          
        self.start_prim_paths = config.START_PRIM_PATHS           # 所有起點參與檢測
        self.start_highlight_paths = config.START_HIGHLIGHT_PATHS # 只有括號起點畫線
        self.end_prim_paths = config.END_PRIM_PATHS               # 所有終點參與檢測
        self.end_highlight_paths = config.END_HIGHLIGHT_PATHS     # 只有括號終點畫線

        '''
        # 調試輸出路徑配置
        print("📍 NDI路徑配置:")
        print(f"  起點總數: {len(self.start_prim_paths)}")
        print(f"  起點畫線: {len(self.start_highlight_paths)}")
        print(f"  終點總數: {len(self.end_prim_paths)}")
        print(f"  終點畫線: {len(self.end_highlight_paths)}")
        '''
        
        for i, path in enumerate(self.start_prim_paths):
            highlight = "(畫線)" if path in self.start_highlight_paths else ""
            print(f"    起點{i+1}: {path} {highlight}")
        
        for i, path in enumerate(self.end_prim_paths[:10]):  # 只顯示前10個
            highlight = "(畫線)" if path in self.end_highlight_paths else ""
            # 檢查屬於哪個標記
            marker_type = "Unknown"
            for marker in config.SUPPORTED_MARKERS:
                if marker.upper() in path.upper():
                    marker_type = marker
                    break
            print(f"    終點{i+1}[{marker_type}]: {path} {highlight}")
        
        if len(self.end_prim_paths) > 10:
            print(f"    ... 還有 {len(self.end_prim_paths) - 10} 個終點")

        # 成本計算相關變數 - 每幀計算固態角成本
        self.cost_calculation = {
            'total_cost': 0.0,
            'frame_count': 0,
            'cost_samples': [],
            'start_time': None,
            'calculation_active': False,
        }
        
        # 成本計算參數
        self.cost_params = {
            'temperature': 0.1,  # 固態角成本函數的溫度參數
            'occlusion_penalty_base': 50.0,  # 基礎遮蔽懲罰
            'occlusion_penalty_increment': 10.0,  # 連續遮蔽每幀增加的懲罰
            'penalty_mode': 'dynamic'  # 動態懲罰模式
        }
        
        # 追蹤每個 marker 的遮蔽歷史
        self.occlusion_history = {
            marker: {
                'consecutive_frames': 0,  # 連續遮蔽幀數
                'total_occluded_frames': 0,  # 總遮蔽幀數
                'last_status': False  # 上一幀狀態
            } for marker in config.SUPPORTED_MARKERS
        }

        # 暫時跳過觸發器初始化
        #print("⚠️ 跳過觸發器初始化，使用簡化檢測")
        self.trigger_detector = None
        
    def initialize(self):
        """初始化檢測接口"""
        try:
            self.raycast_interface = omni.kit.raycast.query.acquire_raycast_query_interface()
            self.debug_draw_interface = _debug_draw.acquire_debug_draw_interface()
            
            # 🔧 重新啟用觸發器初始化
            self._initialize_trigger_volume()
            
            print("NDI detector interfaces initialized")
            return True
        except Exception as e:
            print(f"NDI detector initialization failed: {e}")
            return False

    def _initialize_trigger_volume(self):
        """初始化觸發體積 - 基於測試成功的代碼"""
        try:
            stage = omni.usd.get_context().get_stage()
            trigger_prim = stage.GetPrimAtPath(self.trigger_volume_path)
            
            if not trigger_prim.IsValid():
                print(f"⚠️ Trigger volume not found at: {self.trigger_volume_path}")
                self.trigger_detector = None
                return False
            
            print(f"✅ Found trigger volume at: {self.trigger_volume_path}")
            
            # 應用必要的 PhysX API
            from pxr import UsdPhysics, PhysxSchema
            
            if not trigger_prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(trigger_prim)
            
            if not trigger_prim.HasAPI(PhysxSchema.PhysxTriggerAPI):
                PhysxSchema.PhysxTriggerAPI.Apply(trigger_prim)
            
            if not trigger_prim.HasAPI(PhysxSchema.PhysxTriggerStateAPI):
                trigger_state_api = PhysxSchema.PhysxTriggerStateAPI.Apply(trigger_prim)
            else:
                trigger_state_api = PhysxSchema.PhysxTriggerStateAPI(trigger_prim)
            
            # 設置觸發器檢測器
            self.trigger_detector = {
                "path": self.trigger_volume_path,
                "prim": trigger_prim,
                "state_api": trigger_state_api,
                "objects": {},
                "previous_collisions": []
            }
            
            print("🎯 Trigger volume initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Trigger volume initialization failed: {e}")
            self.trigger_detector = None
            return False
        

    def get_current_trigger_objects(self):
        """獲取當前在觸發器內的物體 - 修復版本"""
        if not self.trigger_detector:
            return set()  # 返回空集合而不是空列表
        
        try:
            state_api = self.trigger_detector["state_api"]
            current_collisions = state_api.GetTriggeredCollisionsRel().GetTargets()
            
            # 轉換為字符串集合，方便後續查詢
            objects_in_trigger = set()
            for collision_path in current_collisions:
                objects_in_trigger.add(str(collision_path))
            
            return objects_in_trigger
            
        except Exception as e:
            print(f"❌ Error getting trigger objects: {e}")
            return set()

    def print_trigger_status(self):
        """暫時跳過觸發器狀態打印"""
        print("📋 觸發器檢查已跳過")

    def _get_prim_position(self, prim_path):
        """獲取prim的世界位置"""
        try:
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(prim_path)
            
            if not prim.IsValid():
                return None
                
            from pxr import UsdGeom
            xformable = UsdGeom.Xformable(prim)
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            translation = world_transform.ExtractTranslation()
            
            return [translation[0], translation[1], translation[2]]
            
        except Exception as e:
            return None

    def _update_lines(self):
        """更新檢測線段 - 完全按照v2邏輯"""
        try:
            self.line_starts = []
            self.line_ends = []
            self.ray_info = []
            
            # 獲取所有起點位置（用於檢測）
            start_positions = []
            start_path_valid = []
            for start_path in self.start_prim_paths:
                pos = self._get_prim_position(start_path)
                if pos is not None:
                    start_positions.append(pos)
                    start_path_valid.append(start_path)
            
            # 獲取所有終點位置（用於檢測）
            end_positions = []
            end_path_valid = []
            for end_path in self.end_prim_paths:
                pos = self._get_prim_position(end_path)
                if pos is not None:
                    end_positions.append(pos)
                    end_path_valid.append(end_path)
            
            # 創建所有起點到所有終點的檢測線段
            for start_idx, start_pos in enumerate(start_positions):
                for end_idx, end_pos in enumerate(end_positions):
                    self.line_starts.append(start_pos)
                    self.line_ends.append(end_pos)
                    
                    # 記錄射線信息
                    self.ray_info.append({
                        'start_path': start_path_valid[start_idx],
                        'end_path': end_path_valid[end_idx],
                        'start_pos': start_pos,
                        'end_pos': end_pos
                    })
            
            # 統計各標記的檢測線段數
            marker_line_count = {}
            for marker in self.config.SUPPORTED_MARKERS:
                count = 0
                for ray in self.ray_info:
                    if marker.upper() in ray['end_path'].upper():
                        count += 1
                marker_line_count[marker] = count
            '''
            print(f"🔍 檢測線段統計:")
            print(f"  有效起點: {len(start_positions)}/{len(self.start_prim_paths)}")
            print(f"  有效終點: {len(end_positions)}/{len(self.end_prim_paths)}")
            print(f"  總檢測線段: {len(self.line_starts)}")
            for marker, count in marker_line_count.items():
                print(f"  {marker}標記線段: {count}")
            '''
            return len(self.line_starts) > 0
            
        except Exception as e:
            print(f"Line update error: {e}")
            return False

    def _perform_raycast(self):
        """執行射線檢測 - v2方式"""
        if not self.line_starts or not self.line_ends:
            return
        
        self.raycast_results = []
        
        for i, (start, end) in enumerate(zip(self.line_starts, self.line_ends)):
            if i >= len(self.ray_info):
                continue
                
            # 計算方向向量
            direction = [end[j] - start[j] for j in range(3)]
            length = np.sqrt(sum([d*d for d in direction]))
            
            if length < 0.001:
                continue
                
            normalized_dir = [d / length for d in direction]
            offset = 0.01
            adjusted_start = [start[j] + normalized_dir[j] * offset for j in range(3)]
            
            ray = omni.kit.raycast.query.Ray(tuple(adjusted_start), tuple(normalized_dir))
            self.raycast_interface.submit_raycast_query(ray, self._on_raycast_hit)

    def _on_raycast_hit(self, ray, result):
        """處理射線碰撞回調 - v2方式"""
        ray_index = len(self.raycast_results)
        
        if ray_index >= len(self.ray_info):
            return
        
        info = self.ray_info[ray_index]
        start_path = info['start_path']
        end_path = info['end_path']
        
        if result.valid:
            hit_path = str(result.get_target_usd_path())
            is_target_hit = end_path in hit_path or hit_path in end_path
            
            self.raycast_results.append({
                'ray_index': ray_index,
                'start_path': start_path, 
                'end_path': end_path,
                'hit_path': hit_path,
                'is_target_hit': is_target_hit
            })
        else:
            self.raycast_results.append({
                'ray_index': ray_index,
                'start_path': start_path,
                'end_path': end_path,
                'hit_path': "None",
                'is_target_hit': False
            })

    def _process_detection_results(self):
        """處理檢測結果 - 恢復完整的觸發器邏輯"""
        if not self.raycast_results:
            return self.marker_status
        
        # 🎯 獲取觸發器中的物體
        objects_in_trigger = self.get_current_trigger_objects()
        '''
        # 每100幀打印一次觸發器狀態
        if self.frame_count % 100 == 0 and objects_in_trigger:
            print(f"🔍 Trigger status: {len(objects_in_trigger)} objects in trigger")
            for i, obj_path in enumerate(list(objects_in_trigger)[:5]):  # 只顯示前5個
                print(f"   {i+1}. {obj_path}")
            if len(objects_in_trigger) > 5:
                print(f"   ... and {len(objects_in_trigger) - 5} more")
        '''
        # 為每個終點路徑統計命中率
        end_hit_map = {}
        for end_path in self.end_prim_paths:
            end_hit_map[end_path] = {
                'hit_count': 0,
                'total_count': 0
            }
        
        # 統計射線命中
        for result in self.raycast_results:
            end_path = result['end_path']
            hit_path = result['hit_path']
            is_target_hit = result['is_target_hit']
            
            if end_path in end_hit_map:
                end_hit_map[end_path]['total_count'] += 1
                if is_target_hit:
                    end_hit_map[end_path]['hit_count'] += 1
        
        # 標記可見性判定 - 恢復75%閾值 + 觸發器檢測
        marker_status = {marker: False for marker in self.config.SUPPORTED_MARKERS}
        VISIBILITY_THRESHOLD = 0.75  # 75% 的球體需要可見
        
        for marker in self.config.SUPPORTED_MARKERS:
            marker_paths = [path for path in self.end_prim_paths if marker.upper() in path.upper()]
            
            if not marker_paths:
                continue
            
            visible_balls = 0
            total_balls = len(marker_paths)
            
            for path in marker_paths:
                if path in end_hit_map:
                    stats = end_hit_map[path]
                    hit_count = stats['hit_count']
                    total_count = stats['total_count']
                    
                    if total_count == 0:
                        continue
                    
                    # 🎯 恢復真正的觸發器檢測
                    is_in_trigger = self._check_path_in_trigger(path, objects_in_trigger)
                    
                    # 球體可見性：射線全部命中 AND 在觸發器內
                    is_ball_visible = (hit_count == total_count) and is_in_trigger
                    
                    if is_ball_visible:
                        visible_balls += 1
            
            # 75%閾值判定
            if total_balls > 0:
                visibility_ratio = visible_balls / total_balls
                marker_status[marker] = visibility_ratio >= VISIBILITY_THRESHOLD
        
        # 每50幀顯示檢測統計
        if self.frame_count % 50 == 0:
            #print(f"🎯 檢測統計 (Frame {self.frame_count}) - 完整檢測:")
            for marker in self.config.SUPPORTED_MARKERS:
                marker_paths = [path for path in self.end_prim_paths if marker.upper() in path.upper()]
                visible_count = 0
                in_trigger_count = 0
                total_balls = len(marker_paths)
                
                for path in marker_paths:
                    if path in end_hit_map:
                        stats = end_hit_map[path]
                        is_raycast_hit = stats['total_count'] > 0 and stats['hit_count'] == stats['total_count']
                        is_in_trigger = self._check_path_in_trigger(path, objects_in_trigger)
                        
                        if is_in_trigger:
                            in_trigger_count += 1
                        if is_raycast_hit and is_in_trigger:
                            visible_count += 1
                
                visibility_ratio = visible_count / total_balls if total_balls > 0 else 0
                status = "✅" if marker_status[marker] else "❌"
                #print(f"  {marker}: {status} ({visible_count}/{total_balls} visible, {in_trigger_count}/{total_balls} in trigger = {visibility_ratio:.1%})")
        
        # 計算可見標記的體積和固態角
        self._calculate_marker_volumes_and_angles(marker_status, end_hit_map, objects_in_trigger)
        
        return marker_status
    
    def _calculate_marker_volumes_and_angles(self, marker_status, end_hit_map, objects_in_trigger):
        """計算每個標記的體積和固態角"""
        # 更新NDI位置
        ndi_position = self.volume_calculator.get_emitter_position(self._get_prim_position)
        
        if ndi_position is None:
            return
        
        # 為每個標記計算
        for marker in self.config.SUPPORTED_MARKERS:
            # 獲取此標記的所有球體路徑
            marker_paths = [path for path in self.end_prim_paths if marker.upper() in path.upper()]
            
            if not marker_paths:
                continue
            
            # 收集可見且在觸發器內的球體位置
            visible_positions = []
            for path in marker_paths:
                if path in end_hit_map:
                    stats = end_hit_map[path]
                    is_raycast_hit = stats['total_count'] > 0 and stats['hit_count'] == stats['total_count']
                    is_in_trigger = self._check_path_in_trigger(path, objects_in_trigger)
                    
                    # 只計算可見且在觸發器內的球體
                    if is_raycast_hit and is_in_trigger:
                        pos = self._get_prim_position(path)
                        if pos is not None:
                            visible_positions.append(pos)
            
            # 計算體積和固態角
            if len(visible_positions) >= 3:
                volume = self.volume_calculator.calculate_marker_volume(visible_positions)
                solid_angle = self.volume_calculator.calculate_solid_angle(visible_positions)
                
                self.marker_volumes[marker] = volume
                self.marker_solid_angles[marker] = solid_angle
            else:
                self.marker_volumes[marker] = None
                self.marker_solid_angles[marker] = None

    def _check_path_in_trigger(self, target_path, objects_in_trigger):
        """檢查指定路徑是否在觸發器內"""
        if not objects_in_trigger:
            return False
        
        # 直接路徑匹配
        if target_path in objects_in_trigger:
            return True
        
        # 部分路徑匹配 (處理路徑層次結構)
        for trigger_path in objects_in_trigger:
            if target_path in trigger_path or trigger_path in target_path:
                return True
        
        return False
       
    def _draw_debug_lines(self):
        """繪製調試線段 - 只畫highlight路徑"""
        try:
            if not self.debug_draw_interface:
                print("Debug draw interface not available")
                return
            
            # 只為highlight路徑創建畫線
            highlight_starts = []
            highlight_ends = []
            
            # 獲取highlight起點位置
            for start_path in self.start_highlight_paths:
                pos = self._get_prim_position(start_path)
                if pos is not None:
                    highlight_starts.append(pos)
            
            # 獲取highlight終點位置  
            for end_path in self.end_highlight_paths:
                pos = self._get_prim_position(end_path)
                if pos is not None:
                    highlight_ends.append(pos)
            
            # 創建highlight線段
            draw_starts = []
            draw_ends = []
            for start_pos in highlight_starts:
                for end_pos in highlight_ends:
                    draw_starts.append(start_pos)
                    draw_ends.append(end_pos)
            
            if not draw_starts:
                print("No highlight lines to draw")
                return
            
            #print(f"🎨 Drawing {len(draw_starts)} highlight lines")
            
            # 清除並繪製
            self.debug_draw_interface.clear_lines()
            colors = [[1.0, 0.5, 0.0, 0.8]] * len(draw_starts)  # 橙色
            sizes = [2] * len(draw_starts)
            
            self.debug_draw_interface.draw_lines(
                draw_starts,
                draw_ends, 
                colors,
                sizes
            )
            
        except Exception as e:
            print(f"Debug draw error: {e}")
            traceback.print_exc()

    def clear_debug_draw(self):
        """清除所有調試繪製"""
        try:
            if self.debug_draw_interface:
                self.debug_draw_interface.clear_lines()
                print("Debug draw cleared")
        except Exception as e:
            print(f"Clear debug draw error: {e}")

    def update_detection(self, debug_enabled=False):
        """更新檢測狀態 - v2方式"""
        if not self.raycast_interface:
            return self.marker_status
            
        try:
            # 更新線段
            has_lines = self._update_lines()
            
            if has_lines:
                # 執行射線檢測
                self._perform_raycast()
                
                # 等待回調完成
                world = World.instance()
                if world:
                    for _ in range(5):
                        world.step(render=True)
                        time.sleep(0.01)
                
                # 處理檢測結果
                if len(self.raycast_results) > 0:
                    self.marker_status = self._process_detection_results()
                
                # 💰 檢查並計算成本（如果成本計算已啟動）
                self.check_and_calculate_cost_if_needed(self.marker_status)
                
                # 繪製調試線段
                if debug_enabled and self.debug_draw_interface:
                    self._draw_debug_lines()
                elif self.debug_draw_interface:
                    # 如果debug關閉，清除線段
                    self.debug_draw_interface.clear_lines()
            else:
                print("No lines available for detection")
                
            self.frame_count += 1
                
        except Exception as e:
            print(f"Detection update error: {e}")
            traceback.print_exc()
        
        return self.marker_status

    def get_status_changes(self):
        """獲取狀態變化"""
        changes = {}
        for marker in self.config.SUPPORTED_MARKERS:
            if self.marker_status[marker] != self.previous_marker_status[marker]:
                changes[marker] = self.marker_status[marker]
                self.previous_marker_status[marker] = self.marker_status[marker]
        return changes
    
    def get_marker_volume(self, marker):
        """獲取標記體積
        
        Args:
            marker: 標記名稱
            
        Returns:
            體積（cm³）或None
        """
        return self.marker_volumes.get(marker)
    
    def get_marker_solid_angle(self, marker):
        """獲取標記固態角
        
        Args:
            marker: 標記名稱
            
        Returns:
            固態角（sr）或None
        """
        return self.marker_solid_angles.get(marker)
    
    def get_all_marker_data(self):
        """獲取所有標記的完整數據
        
        Returns:
            字典，包含每個標記的狀態、體積和固態角
        """
        data = {}
        for marker in self.config.SUPPORTED_MARKERS:
            data[marker] = {
                'visible': self.marker_status[marker],
                'volume': self.marker_volumes[marker],
                'solid_angle': self.marker_solid_angles[marker]
            }
        return data
    
    def get_occlusion_statistics(self):
        """獲取遮蔽統計資訊"""
        stats = {}
        for marker in self.config.SUPPORTED_MARKERS:
            history = self.occlusion_history[marker]
            stats[marker] = {
                'total_occluded_frames': history['total_occluded_frames'],
                'current_consecutive': history['consecutive_frames'],
                'currently_visible': history['last_status']
            }
        return stats
    
    def start_cost_calculation(self):
        """開始成本計算 - 每幀計算固態角成本（含遮蔽懲罰）"""
        print("💰 開始每幀固態角成本計算（含遮蔽懲罰）...")
        print(f"📍 將計算單一 NDI (/Root/NDI_02) 的固態角成本")
        print(f"⚙️  懲罰模式: {self.cost_params['penalty_mode']}")
        print(f"   基礎遮蔽懲罰: {self.cost_params['occlusion_penalty_base']}")
        print(f"   連續遮蔽增量: {self.cost_params['occlusion_penalty_increment']}")
        
        self.cost_calculation['total_cost'] = 0.0
        self.cost_calculation['frame_count'] = 0
        self.cost_calculation['cost_samples'] = []
        self.cost_calculation['start_time'] = time.time()
        self.cost_calculation['calculation_active'] = True
        
        # 重置遮蔽歷史
        for marker in self.config.SUPPORTED_MARKERS:
            self.occlusion_history[marker] = {
                'consecutive_frames': 0,
                'total_occluded_frames': 0,
                'last_status': False
            }
        
        print("✅ 成本計算已啟動")
        
    def stop_cost_calculation(self):
        """停止成本計算並返回結果 - 添加遮蔽統計"""
        if not self.cost_calculation['calculation_active']:
            return None
        
        self.cost_calculation['calculation_active'] = False
        
        # 計算統計數據
        frame_count = self.cost_calculation['frame_count']
        total_cost = self.cost_calculation['total_cost']
        elapsed_time = time.time() - self.cost_calculation['start_time']
        
        if frame_count > 0:
            avg_cost_per_frame = total_cost / frame_count
        else:
            avg_cost_per_frame = 0.0
        
        # 獲取遮蔽統計
        occlusion_stats = self.get_occlusion_statistics()
        
        results = {
            'total_cost': total_cost,
            'frame_count': frame_count,
            'average_cost_per_frame': avg_cost_per_frame,
            'elapsed_time': elapsed_time,
            'cost_samples': self.cost_calculation['cost_samples'],
            'occlusion_statistics': occlusion_stats,  # 新增
            'cost_params': self.cost_params.copy()  # 新增
        }
        
        print(f"\n💰 ========== 成本計算結果 ==========")
        print(f"📊 總幀數: {frame_count}")
        print(f"💵 總成本: {total_cost:.6f}")
        print(f"📈 平均每幀成本: {avg_cost_per_frame:.6f}")
        print(f"⏱️  執行時間: {elapsed_time:.2f} 秒")
        print(f"\n🚫 遮蔽統計:")
        for marker, stats in occlusion_stats.items():
            occluded_ratio = (stats['total_occluded_frames'] / frame_count * 100) if frame_count > 0 else 0
            print(f"   {marker}: {stats['total_occluded_frames']} 幀被遮蔽 ({occluded_ratio:.1f}%)")
        print(f"=====================================\n")
        
        # 保存結果到文件（包含圖表和CSV）
        self._save_cost_results(results)
        
        return results
    
    def _save_cost_results(self, results):
        """保存成本計算結果到文件（JSON、CSV、圖表）"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 創建輸出目錄
            output_dir = r"C:\Users\RMML\AppData\Local\ov\pkg\4.5.0\standalone_ws\solidangle\cost_vs_frame"
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. 保存 JSON 結果
            json_filename = f"ndi_cost_results_{timestamp}.json"
            json_filepath = os.path.join(output_dir, json_filename)
            
            with open(json_filepath, 'w') as f:
                json.dump(results, f, indent=4)
            
            print(f"✅ JSON 結果已保存到: {json_filepath}")
            
            # 2. 保存 CSV 文件
            cost_samples = results.get('cost_samples', [])
            if cost_samples:
                csv_filename = f"cost_vs_frame_{timestamp}.csv"
                csv_filepath = os.path.join(output_dir, csv_filename)
                
                # 創建 DataFrame
                df = pd.DataFrame({
                    'Frame': range(1, len(cost_samples) + 1),
                    'Cost': cost_samples
                })
                
                df.to_csv(csv_filepath, index=False)
                print(f"✅ CSV 文件已保存到: {csv_filepath}")
                
                # 3. 生成圖表
                self._generate_cost_plot(cost_samples, timestamp, output_dir)
            else:
                print("⚠️ 沒有成本數據可保存")
            
        except Exception as e:
            print(f"❌ 保存成本結果失敗: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_cost_plot(self, cost_samples, timestamp, output_dir):
        """生成 Cost vs Frame 圖表"""
        try:
            # 創建圖表
            plt.figure(figsize=(12, 6))
            
            frames = range(1, len(cost_samples) + 1)
            
            # 繪製成本曲線
            plt.plot(frames, cost_samples, linewidth=1.5, color='#2E86AB', alpha=0.8, label='Cost')
            
            # 添加平均線
            avg_cost = np.mean(cost_samples)
            plt.axhline(y=avg_cost, color='#F18F01', linestyle=':', linewidth=2, 
                       label=f'Average = {avg_cost:.4f}')
            
            # 圖表設置
            plt.xlabel('Frame Number', fontsize=12, fontweight='bold')
            plt.ylabel('Cost', fontsize=12, fontweight='bold')
            plt.title('NDI Cost Function vs Frame Number', fontsize=14, fontweight='bold', pad=20)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.legend(loc='upper left', fontsize=10)
            
            # 添加統計信息文本框 - 放在右下角避免遮擋圖表
            stats_text = f'Total Frames: {len(cost_samples)}\n'
            stats_text += f'Total Cost: {sum(cost_samples):.2f}\n'
            stats_text += f'Min Cost: {min(cost_samples):.4f}\n'
            stats_text += f'Max Cost: {max(cost_samples):.4f}\n'
            stats_text += f'Std Dev: {np.std(cost_samples):.4f}'
            
            plt.text(0.98, 0.02, stats_text, transform=plt.gca().transAxes,
                    fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor='gray'))
            
            # 保存圖表
            plot_filename = f"cost_vs_frame_{timestamp}.png"
            plot_filepath = os.path.join(output_dir, plot_filename)
            
            plt.tight_layout()
            plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ 圖表已保存到: {plot_filepath}")
            
        except Exception as e:
            print(f"❌ 生成圖表失敗: {e}")
            import traceback
            traceback.print_exc()
    
    def calculate_frame_cost(self, marker_status):
        """計算當前幀的成本 - 結合固態角和遮蔽懲罰（動態模式）"""
        if not self.volume_calculator.emitter_position:
            return 0.0
        
        frame_cost = 0.0
        penalty_mode = self.cost_params['penalty_mode']
        
        # 遍歷每個標記
        for marker in self.config.SUPPORTED_MARKERS:
            is_visible = marker_status.get(marker, False)
            
            if is_visible:
                # 標記可見 - 重置連續遮蔽計數
                self.occlusion_history[marker]['consecutive_frames'] = 0
                
                # 獲取固態角並計算成本
                solid_angle = self.marker_solid_angles.get(marker)
                if solid_angle is not None and solid_angle > 0:
                    # 使用固態角成本函數: cost = -log(solid_angle) / T
                    T = self.cost_params['temperature']
                    marker_cost = -np.log(solid_angle + 1e-10) / T
                    frame_cost += marker_cost
            else:
                # 標記不可見 - 增加遮蔽計數
                self.occlusion_history[marker]['consecutive_frames'] += 1
                self.occlusion_history[marker]['total_occluded_frames'] += 1
                
                # 動態懲罰：連續遮蔽懲罰遞增
                base_penalty = self.cost_params['occlusion_penalty_base']
                increment = self.cost_params['occlusion_penalty_increment']
                consecutive = self.occlusion_history[marker]['consecutive_frames']
                
                # 連續遮蔽懲罰：基礎懲罰 + 連續幀數 * 增量
                occlusion_penalty = base_penalty + (consecutive - 1) * increment
                
                # 限制最大懲罰（避免過度懲罰）
                max_penalty = base_penalty * 5
                occlusion_penalty = min(occlusion_penalty, max_penalty)
                
                frame_cost += occlusion_penalty
            
            # 更新上一幀狀態
            self.occlusion_history[marker]['last_status'] = is_visible
        
        return frame_cost
    
    def check_and_calculate_cost_if_needed(self, marker_status):
        """檢查是否需要計算成本，如果需要則計算"""
        if not self.cost_calculation['calculation_active']:
            return
        
        # 更新 NDI 位置
        self.volume_calculator.get_emitter_position(self._get_prim_position)
        
        # 計算當前幀成本
        frame_cost = self.calculate_frame_cost(marker_status)
        
        # 更新總成本
        self.cost_calculation['total_cost'] += frame_cost
        self.cost_calculation['frame_count'] += 1
        self.cost_calculation['cost_samples'].append(frame_cost)
        
        # 每100幀顯示一次進度
        if self.cost_calculation['frame_count'] % 100 == 0:
            avg_cost = self.cost_calculation['total_cost'] / self.cost_calculation['frame_count']
            print(f"💰 成本計算進度: Frame {self.cost_calculation['frame_count']}, "
                  f"總成本: {self.cost_calculation['total_cost']:.4f}, "
                  f"平均: {avg_cost:.6f}")

# 5. UI管理器
class UIManager:
    """專門處理UI邏輯"""
    def __init__(self, config):
        self.config = config
        self.window = None
        self.ui_elements = {}
        self.callbacks = {}
        
    def setup_ui(self):
        """設置UI介面"""
        try:
            self.window = ui.Window("NDI Control Panel", width=300, height=420, 
                                  dockPreference=ui.DockPreference.RIGHT_TOP)
            with self.window.frame:
                with ui.VStack(spacing=5):
                    # 標題
                    ui.Label("NDI Status Panel", height=40, alignment=ui.Alignment.CENTER,
                            style={"font_size": 18, "color": 0xFFFFFFFF})
                    
                    ui.Spacer(height=5)
                    
                    # 狀態指示器
                    self._create_status_indicators()
                    
                    ui.Spacer(height=10)
                    
                    # 控制按鈕
                    self._create_control_buttons()
                    
                    ui.Spacer(height=10)
                    
                    # Debug控制
                    self._create_debug_controls()
                    
                    ui.Spacer(height=10)
                    
                    # 信息顯示
                    self._create_info_display()
            
            print("UI setup completed")
            return True
            
        except Exception as e:
            print(f"UI setup failed: {e}")
            return False

    def _create_control_buttons(self):
        """創建控制按鈕 - 新增Initial按鍵"""
        with ui.HStack(height=40):
            # 新增Initial按鍵
            initial_btn = ui.Button("Initial", width=60, height=30)
            initial_btn.set_clicked_fn(lambda: self._trigger_callback("initial"))
            
            ui.Spacer(width=5)
            
            execute_btn = ui.Button("Execute", width=60, height=30)
            execute_btn.set_clicked_fn(lambda: self._trigger_callback("execute"))
            
            ui.Spacer(width=5)
            
            stop_btn = ui.Button("Stop", width=60, height=30)
            stop_btn.set_clicked_fn(lambda: self._trigger_callback("stop"))
            
            ui.Spacer(width=5)
            
            reset_btn = ui.Button("Reset", width=60, height=30)
            reset_btn.set_clicked_fn(lambda: self._trigger_callback("reset"))
            
            ui.Spacer(width=5)
            
            info_btn = ui.Button("Info", width=60, height=30)
            info_btn.set_clicked_fn(lambda: self._trigger_callback("info"))
            
    def _create_status_indicators(self):
        """創建狀態指示器 - 包含體積和固態角顯示"""
        for marker in self.config.SUPPORTED_MARKERS:
            with ui.VStack(spacing=2):
                # 第一行：標記名稱和狀態圖示
                with ui.HStack(height=30):
                    ui.Label(f"{marker}", width=40, alignment=ui.Alignment.LEFT)
                    with ui.Frame(width=24, height=24) as frame:
                        self.ui_elements[f"{marker}_img_frame"] = frame
                        img = ui.Image(self.config.RED_IMG_PATH)
                        self.ui_elements[f"{marker}_img"] = img
                    ui.Spacer()
                
                # 第二行：體積和固態角
                with ui.HStack(height=20):
                    ui.Spacer(width=10)
                    volume_label = ui.Label("V: --", width=100, 
                                          style={"font_size": 10, "color": 0xAAAAAAAA})
                    self.ui_elements[f"{marker}_volume"] = volume_label
                    
                    angle_label = ui.Label("Ω: --", width=100,
                                         style={"font_size": 10, "color": 0xAAAAAAAA})
                    self.ui_elements[f"{marker}_angle"] = angle_label
                    ui.Spacer()
            
            ui.Spacer(height=5)
            
    def _create_debug_controls(self):
        """創建調試控制"""
        with ui.HStack(height=30):
            ui.Label("Debug Draw:", width=80)
            debug_checkbox = ui.CheckBox(width=20)
            debug_checkbox.model.set_value(False)
            debug_checkbox.model.add_value_changed_fn(lambda model: self._trigger_callback("debug_toggle"))
            self.ui_elements["debug_checkbox"] = debug_checkbox
            ui.Label("Show Lines", width=80)
    
    def _create_info_display(self):
        """創建信息顯示區域"""
        self.ui_elements["status_label"] = ui.Label("Ready", height=30, 
                                                   style={"color": 0xAAFFAAFF})
        self.ui_elements["info_label"] = ui.Label("", height=60, word_wrap=True,
                                                 style={"color": 0xFFFFFFFF})
        
        with ui.Frame(height=100):
            self.ui_elements["log_area"] = ui.Label("System initialized", 
                                                   word_wrap=True,
                                                   style={"font_size": 12, "color": 0xDDDDDDFF})
    
    def update_marker_status(self, marker_status):
        """更新標記狀態"""
        try:
            for marker, is_visible in marker_status.items():
                img_key = f"{marker}_img"
                if img_key in self.ui_elements:
                    frame = self.ui_elements[f"{marker}_img_frame"]
                    frame.clear()
                    with frame:
                        img_path = self.config.GREEN_IMG_PATH if is_visible else self.config.RED_IMG_PATH
                        self.ui_elements[img_key] = ui.Image(img_path)
        except Exception as e:
            print(f"UI marker update error: {e}")
    
    def update_marker_volume_and_angle(self, marker, volume, solid_angle):
        """更新標記的體積和固態角顯示
        
        Args:
            marker: 標記名稱
            volume: 體積（cm³）
            solid_angle: 固態角（sr）
        """
        try:
            volume_key = f"{marker}_volume"
            angle_key = f"{marker}_angle"
            
            if volume_key in self.ui_elements:
                if volume is not None:
                    self.ui_elements[volume_key].text = f"V: {volume:.2f} cm³"
                else:
                    self.ui_elements[volume_key].text = "V: --"
            
            if angle_key in self.ui_elements:
                if solid_angle is not None:
                    self.ui_elements[angle_key].text = f"Ω: {solid_angle:.4f} sr"
                else:
                    self.ui_elements[angle_key].text = "Ω: --"
                    
        except Exception as e:
            print(f"UI volume/angle update error for {marker}: {e}")
    
    def set_callback(self, event_name, callback):
        """設置回調函數"""
        self.callbacks[event_name] = callback
    
    def _trigger_callback(self, event_name):
        """觸發回調"""
        if event_name in self.callbacks:
            try:
                self.callbacks[event_name]()
            except Exception as e:
                print(f"Callback error for {event_name}: {e}")
    
    def update_status(self, message):
        """更新狀態信息"""
        if "status_label" in self.ui_elements:
            self.ui_elements["status_label"].text = message
    
    def update_info(self, message):
        """更新信息顯示"""
        if "info_label" in self.ui_elements:
            self.ui_elements["info_label"].text = str(message)
    
    def update_log(self, message):
        """更新日誌區域"""
        if "log_area" in self.ui_elements:
            self.ui_elements["log_area"].text = str(message)
    
    def get_debug_enabled(self):
        """獲取debug狀態"""
        if "debug_checkbox" in self.ui_elements:
            return self.ui_elements["debug_checkbox"].model.get_value_as_bool()
        return False

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
from datetime import datetime

# 6. 軌跡記錄和可視化模組
class TrajectoryRecorder:
    """專門處理標記軌跡記錄和可視化"""
    def __init__(self, config):
       
        self.config = config
        self.recording = False
        self.trajectory_data = {}
        self.start_time = None
        self.output_dir = "C:/Nick/surgery_team/trajectory_recordings"
        
        # 初始化軌跡數據結構 - 添加needle tip
        self.supported_objects = config.SUPPORTED_MARKERS + ['NEEDLE_TIP']
        
        for obj in self.supported_objects:
            self.trajectory_data[obj] = {
                'timestamps': [],
                'positions': [],
                'is_visible': []
            }
        
        # 確保輸出目錄存在
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 定義每個標記的四個球體路徑模式
        self.marker_ball_patterns = {
            'BM': [
                '/Root/robotarm_base/End_needle/BM/bm_rball_0/node_/mesh_',
                '/Root/robotarm_base/End_needle/BM/bm_rball_1/node_/mesh_',
                '/Root/robotarm_base/End_needle/BM/bm_rball_2/node_/mesh_',
                '/Root/robotarm_base/End_needle/BM/bm_rball_3/node_/mesh_'
            ],
            'EM': [
                '/Root/robotarm_base/End_needle/EM/em_rball_0',
                '/Root/robotarm_base/End_needle/EM/em_rball_1',
                '/Root/robotarm_base/End_needle/EM/em_rball_2',
                '/Root/robotarm_base/End_needle/EM/em_rball_3'
            ],
            'FM': [
                '/Root/FM/FM/FM/FM/fm_a',
                '/Root/FM/FM/FM/FM/fm_b',
                '/Root/FM/FM/FM/FM/fm_c',
                '/Root/FM/FM/FM/FM/fm_d'
            ],
            'HM': [
                '/Root/robotarm_base/End_needle/HM/HM_frame/hm_rball_0/node_/mesh_',
                '/Root/robotarm_base/End_needle/HM/HM_frame/hm_rball_1/node_/mesh_',
                '/Root/robotarm_base/End_needle/HM/HM_frame/hm_rball_2/node_/mesh_',
                '/Root/robotarm_base/End_needle/HM/HM_frame/hm_rball_3/node_/mesh_'
            ],
            'UM': [
                '/Root/robotarm_base/UM/UM/UM/um_a',
                '/Root/robotarm_base/UM/UM/UM/um_b',
                '/Root/robotarm_base/UM/UM/UM/um_c',
                '/Root/robotarm_base/UM/UM/UM/um_d'
            ]
        }
        
        # 添加needle tip路徑
        self.needle_tip_path = config.NEEDLETIP_PATH
        print(f"📍 Needle tip path configured: {self.needle_tip_path}")
    
    def _get_prim_position(self, prim_path):
        """獲取prim的世界位置"""
        try:
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(prim_path)
            
            if not prim.IsValid():
                return None
                
            from pxr import UsdGeom
            xformable = UsdGeom.Xformable(prim)
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            translation = world_transform.ExtractTranslation()
            
            return [translation[0], translation[1], translation[2]]
            
        except Exception as e:
            return None
    
    def _calculate_marker_center(self, marker):
        """計算標記中心位置（四個球體的平均值）"""
        if marker not in self.marker_ball_patterns:
            return None
        
        ball_paths = self.marker_ball_patterns[marker]
        valid_positions = []
        
        for ball_path in ball_paths:
            pos = self._get_prim_position(ball_path)
            if pos is not None:
                valid_positions.append(pos)
        
        if len(valid_positions) == 0:
            return None
        
        # 計算平均位置
        center_pos = [
            sum(pos[0] for pos in valid_positions) / len(valid_positions),
            sum(pos[1] for pos in valid_positions) / len(valid_positions),
            sum(pos[2] for pos in valid_positions) / len(valid_positions)
        ]
        
        return center_pos
    
    def _get_needle_tip_position(self):
        """獲取needle tip位置"""
        return self._get_prim_position(self.needle_tip_path)
    
    def start_recording(self):
        """開始記錄軌跡"""
        self.recording = True
        self.start_time = time.time()
        
        # 清除之前的數據 - 包括needle tip
        for obj in self.supported_objects:
            self.trajectory_data[obj] = {
                'timestamps': [],
                'positions': [],
                'is_visible': []
            }
        
        print("📈 Trajectory recording started (including needle tip)")
    
    def stop_recording(self):
        """停止記錄並保存數據"""
        if not self.recording:
            return None, None
        
        self.recording = False
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # 保存CSV文件
            csv_filename = self._save_csv(timestamp)
            
            # 生成3D可視化
            png_filename = self._generate_3d_plot(timestamp)
            
            print(f"📊 Trajectory data saved:")
            print(f"  CSV: {csv_filename}")
            print(f"  PNG: {png_filename}")
            
            return csv_filename, png_filename
            
        except Exception as e:
            print(f"❌ Error saving trajectory data: {e}")
            return None, None
    
    def update_recording(self, marker_status):
        """更新記錄數據（在主循環中調用）- 添加needle tip記錄"""
        if not self.recording:
            return
        
        current_time = time.time() - self.start_time
        
        # 記錄所有markers
        for marker in self.config.SUPPORTED_MARKERS:
            # 計算標記中心位置
            center_pos = self._calculate_marker_center(marker)
            is_visible = marker_status.get(marker, False)
            
            # 記錄數據
            self.trajectory_data[marker]['timestamps'].append(current_time)
            self.trajectory_data[marker]['positions'].append(center_pos)
            self.trajectory_data[marker]['is_visible'].append(is_visible)
        
        # 記錄needle tip - needle tip沒有可見性檢測，總是記錄位置
        needle_tip_pos = self._get_needle_tip_position()
        self.trajectory_data['NEEDLE_TIP']['timestamps'].append(current_time)
        self.trajectory_data['NEEDLE_TIP']['positions'].append(needle_tip_pos)
        self.trajectory_data['NEEDLE_TIP']['is_visible'].append(needle_tip_pos is not None)
    
    def _save_csv(self, timestamp):
        """保存軌跡數據到CSV文件 - 添加needle tip列"""
        csv_filename = f"{self.output_dir}/trajectory_{timestamp}.csv"
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # 寫入標題 - 添加needle tip
            header = ['timestamp']
            for obj in self.supported_objects:
                if obj == 'NEEDLE_TIP':
                    header.extend(['NEEDLE_TIP_x', 'NEEDLE_TIP_y', 'NEEDLE_TIP_z', 'NEEDLE_TIP_valid'])
                else:
                    header.extend([f'{obj}_x', f'{obj}_y', f'{obj}_z', f'{obj}_visible'])
            writer.writerow(header)
            
            # 找到最長的時間序列
            max_length = max(len(self.trajectory_data[obj]['timestamps']) 
                           for obj in self.supported_objects)
            
            # 寫入數據
            for i in range(max_length):
                row = []
                
                # 使用第一個對象的時間戳
                if i < len(self.trajectory_data[self.supported_objects[0]]['timestamps']):
                    row.append(self.trajectory_data[self.supported_objects[0]]['timestamps'][i])
                else:
                    row.append('')
                
                # 寫入每個對象的數據
                for obj in self.supported_objects:
                    if i < len(self.trajectory_data[obj]['positions']):
                        pos = self.trajectory_data[obj]['positions'][i]
                        is_visible_or_valid = self.trajectory_data[obj]['is_visible'][i]
                        
                        if pos is not None:
                            row.extend([pos[0], pos[1], pos[2], int(is_visible_or_valid)])
                        else:
                            row.extend(['', '', '', int(is_visible_or_valid)])
                    else:
                        row.extend(['', '', '', ''])
                
                writer.writerow(row)
        
        return csv_filename
    
    def _generate_3d_plot(self, timestamp):
        """生成3D軌跡可視化圖 - 三合一視角版本"""
        try:
            plt.style.use('default')  # 確保使用默認樣式
            
            # 創建包含三個子圖的圖形
            fig = plt.figure(figsize=(18, 6))  # 調整為橫向排列的大尺寸
            
            # 定義每個標記的顏色和樣式 - 添加needle tip
            object_colors = {
                'BM': '#FF0000',      # 紅色
                'EM': '#00FF00',      # 綠色
                'FM': '#0000FF',      # 藍色
                'HM': '#FFD700',      # 金色
                'UM': '#FF00FF',      # 洋紅色
                'NEEDLE_TIP': '#000000'  # 黑色
            }
            
            # 收集所有有效數據
            all_positions = []
            plot_data = {}
            
            for obj in self.supported_objects:
                positions = self.trajectory_data[obj]['positions']
                is_visible_or_valid = self.trajectory_data[obj]['is_visible']
                
                if not positions:
                    continue
                
                # 分離有效位置和可見性/有效性
                valid_positions = []
                visible_positions = []
                invisible_positions = []
                
                for i, (pos, visible) in enumerate(zip(positions, is_visible_or_valid)):
                    if pos is not None:
                        valid_positions.append(pos)
                        all_positions.append(pos)
                        if visible:
                            visible_positions.append(pos)
                        else:
                            invisible_positions.append(pos)
                
                if valid_positions:
                    plot_data[obj] = {
                        'valid': np.array(valid_positions),
                        'visible': np.array(visible_positions) if visible_positions else None,
                        'invisible': np.array(invisible_positions) if invisible_positions else None
                    }
            
            if not all_positions:
                # 如果沒有數據，創建一個簡單的文字圖
                fig.text(0.5, 0.5, 'No trajectory data recorded', 
                        fontsize=16, ha='center', va='center')
                png_filename = f"{self.output_dir}/trajectory_3d_{timestamp}.png"
                plt.savefig(png_filename, dpi=300, bbox_inches='tight')
                plt.close()
                return png_filename
            
            # 計算數據邊界用於設置軸範圍
            all_positions = np.array(all_positions)
            margin = 0.1  # 10% 邊距
            
            axis_ranges = []
            for i in range(3):
                min_val = np.min(all_positions[:, i])
                max_val = np.max(all_positions[:, i])
                range_val = max_val - min_val
                
                if range_val > 0:
                    ax_min = min_val - range_val * margin
                    ax_max = max_val + range_val * margin
                else:
                    ax_min = min_val - 0.1
                    ax_max = max_val + 0.1
                
                axis_ranges.append((ax_min, ax_max))
            
            # 定義三個視角參數 (仰角, 方位角, 標題)
            views = [
                (90, 0, "Top View"),      # 正上方往下看
                (10, 0, "Side View"),     # 從側面看
                (30, 50, "Isometric View")  # 等角視圖
            ]
            
            # 創建三個子圖
            for i, (elev, azim, title) in enumerate(views):
                ax = fig.add_subplot(1, 3, i+1, projection='3d')
                
                # 繪製每個對象的數據
                for obj, data in plot_data.items():
                    valid_positions = data['valid']
                    visible_positions = data['visible']
                    invisible_positions = data['invisible']
                    
                    # Needle tip特殊處理
                    if obj == 'NEEDLE_TIP':
                        # 繪製needle tip軌跡線
                        ax.plot(valid_positions[:, 0], valid_positions[:, 1], valid_positions[:, 2], 
                               color=object_colors[obj], alpha=0.8, linewidth=2, 
                               linestyle='--')
                        
                        # 繪製needle tip點（較大的黑點）
                        if visible_positions is not None:
                            ax.scatter(visible_positions[:, 0], visible_positions[:, 1], visible_positions[:, 2],
                                      c=object_colors[obj], s=50, alpha=0.9, marker='*')
                        
                        # 標記needle tip起始點和結束點
                        ax.scatter(valid_positions[0, 0], valid_positions[0, 1], valid_positions[0, 2],
                                  c=object_colors[obj], s=150, marker='^', 
                                  edgecolors='white', linewidth=2)
                        ax.scatter(valid_positions[-1, 0], valid_positions[-1, 1], valid_positions[-1, 2],
                                  c=object_colors[obj], s=150, marker='v',
                                  edgecolors='white', linewidth=2)
                    else:
                        # 原有的marker處理邏輯
                        # 繪製整個軌跡線（淡色）
                        ax.plot(valid_positions[:, 0], valid_positions[:, 1], valid_positions[:, 2], 
                               color=object_colors[obj], alpha=0.3, linewidth=1, 
                               linestyle='--')
                        
                        # 繪製可見點（實心）
                        if visible_positions is not None:
                            ax.scatter(visible_positions[:, 0], visible_positions[:, 1], visible_positions[:, 2],
                                      c=object_colors[obj], s=30, alpha=0.8, marker='o')
                        
                        # 繪製不可見點（空心）
                        if invisible_positions is not None:
                            ax.scatter(invisible_positions[:, 0], invisible_positions[:, 1], invisible_positions[:, 2],
                                      c='none', edgecolors=object_colors[obj], s=20, alpha=0.5, marker='o')
                        
                        # 標記起始點和結束點
                        ax.scatter(valid_positions[0, 0], valid_positions[0, 1], valid_positions[0, 2],
                                  c=object_colors[obj], s=100, marker='^')
                        ax.scatter(valid_positions[-1, 0], valid_positions[-1, 1], valid_positions[-1, 2],
                                  c=object_colors[obj], s=100, marker='v')
                
                # 設置視角
                ax.view_init(elev=elev, azim=azim)
                
                # 設置軸範圍
                ax.set_xlim(axis_ranges[0])
                ax.set_ylim(axis_ranges[1])
                ax.set_zlim(axis_ranges[2])
                
                # 設置標籤和標題
                ax.set_xlabel('X Position (m)', fontsize=10)
                ax.set_ylabel('Y Position (m)', fontsize=10)
                ax.set_zlabel('Z Position (m)', fontsize=10)
                ax.set_title(title, fontsize=12, fontweight='bold')
                
                # 調整刻度標籤大小
                ax.tick_params(axis='both', which='major', labelsize=8)
            
            # 創建統一的圖例
            legend_elements = []
            for obj in plot_data.keys():
                if obj == 'NEEDLE_TIP':
                    legend_elements.append(plt.Line2D([0], [0], color=object_colors[obj], 
                                                    linestyle='--', linewidth=2, 
                                                    label='Needle Tip'))
                else:
                    legend_elements.append(plt.Line2D([0], [0], color=object_colors[obj], 
                                                    marker='o', linestyle='--', 
                                                    markersize=6, alpha=0.8, 
                                                    label=f'{obj} Marker'))
            
            # 在圖形底部添加圖例
            fig.legend(handles=legend_elements, loc='lower center', 
                      bbox_to_anchor=(0.5, -0.05), ncol=len(legend_elements), 
                      fontsize=10)
            
            # 設置整體標題
            fig.suptitle(f'Marker & Needle Tip Trajectories - {timestamp}', 
                        fontsize=16, fontweight='bold', y=0.95)
            
            # 調整佈局
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15, top=0.85)  # 為圖例和標題留出空間
            
            # 保存圖片
            png_filename = f"{self.output_dir}/trajectory_3d_{timestamp}.png"
            plt.savefig(png_filename, dpi=300, bbox_inches='tight')
            plt.close()  # 關閉圖形以釋放內存
            
            return png_filename
            
        except Exception as e:
            print(f"❌ 3D plot generation error: {e}")
            import traceback
            traceback.print_exc()
            return None   
    
    def is_recording(self):
        """檢查是否正在記錄"""
        return self.recording
    
    def get_recording_status(self):
        """獲取記錄狀態信息 - 包括needle tip"""
        if not self.recording:
            return "Not recording"
        
        duration = time.time() - self.start_time
        data_points = len(self.trajectory_data[self.supported_objects[0]]['timestamps'])
        return f"Recording: {duration:.1f}s, {data_points} points (markers + needle tip)"

# 7. 主控制器
class SimulationManager:
    """主控制器，協調各個模組"""
    def __init__(self):
        self.config = SimulationConfig()
        self.world = World()
        
        # 初始化各個模組
        self.robot_controller = RobotController(self.config)
        self.ndi_detector = NDIDetector(self.config)
        self.trajectory_executor = TrajectoryExecutor(self.robot_controller, self.config, self.ndi_detector)
        self.ui_manager = UIManager(self.config)
        self.trajectory_recorder = TrajectoryRecorder(self.config)  # 新增
        
        # 設置回調
        self._setup_callbacks()
        
        # 狀態變量
        self.frame_count = 0
        self.last_update_time = time.time()
        self.initialized = False
        
    def _update_detection(self):
        """更新檢測狀態"""
        try:
            debug_enabled = self.ui_manager.get_debug_enabled()
            marker_status = self.ndi_detector.update_detection(debug_enabled)
            
            # 更新軌跡記錄
            self.trajectory_recorder.update_recording(marker_status)
            
            # 檢查狀態變化
            changes = self.ndi_detector.get_status_changes()
            if changes:
                self.ui_manager.update_marker_status(marker_status)
                
                # 添加記錄狀態信息
                recording_status = self.trajectory_recorder.get_recording_status()
                self.ui_manager.update_log(f"Markers: {list(changes.keys())} | {recording_status}")
            
            # 更新體積和固態角顯示（每幀都更新）
            for marker in self.config.SUPPORTED_MARKERS:
                volume = self.ndi_detector.get_marker_volume(marker)
                solid_angle = self.ndi_detector.get_marker_solid_angle(marker)
                self.ui_manager.update_marker_volume_and_angle(marker, volume, solid_angle)
                
        except Exception as e:
            print(f"Detection update error: {e}")
                    
    def _on_execute(self):
        """執行軌跡回調 - 同時開始記錄"""
        def progress_callback(message, current, total):
            self.ui_manager.update_log(f"{message}")
            if total > 0:
                percentage = int((current / total) * 100)
                if any(keyword in message for keyword in ["Reading", "Computing", "Generating"]):
                    self.ui_manager.update_status(f"Preparing: {percentage}%")
                elif "Executing:" in message or "Execution" in message:
                    self.ui_manager.update_status(f"Executing: {percentage}%")
                else:
                    self.ui_manager.update_status(message)
        
        # 檢查當前狀態
        if self.trajectory_executor.is_paused():
            # 如果是暫停狀態，則恢復執行和記錄
            success, message = self.trajectory_executor.execute_trajectory_from_csv(progress_callback)
            if success:
                if not self.trajectory_recorder.is_recording():
                    self.trajectory_recorder.start_recording()
                self.ui_manager.update_info("Execution and recording resumed")
            else:
                self.ui_manager.update_info(f"Resume failed: {message}")
        else:
            # 開始新的執行和記錄
            self.ui_manager.update_info("Starting trajectory execution and recording...")
            
            # 開始記錄軌跡
            self.trajectory_recorder.start_recording()
            
            success, message = self.trajectory_executor.execute_trajectory_from_csv(progress_callback)
            
            if not success:
                # 如果執行失敗，停止記錄
                self.trajectory_recorder.stop_recording()
                self.ui_manager.update_info(f"Execution failed: {message}")
                self.ui_manager.update_status("Ready")
            else:
                self.ui_manager.update_info("Trajectory execution and recording started")


    def _setup_callbacks(self):
        """設置UI回調"""
        self.ui_manager.set_callback("execute", self._on_execute)
        self.ui_manager.set_callback("initial", self._on_initial)
        self.ui_manager.set_callback("stop", self._on_stop)
        self.ui_manager.set_callback("reset", self._on_reset)
        self.ui_manager.set_callback("info", self._on_info)
        self.ui_manager.set_callback("debug_toggle", self._on_debug_toggle)

    def _on_initial(self):
        """移動到初始位置回調"""
        def progress_callback(message, current, total):
            self.ui_manager.update_log(f"{message}")
            if total > 0:
                percentage = int((current / total) * 100)
                if "Reading" in message or "Computing" in message or "Generating" in message:
                    self.ui_manager.update_status(f"Preparing: {percentage}%")
                elif "Moving to initial:" in message:
                    self.ui_manager.update_status(f"Moving: {percentage}%")
                else:
                    self.ui_manager.update_status(message)
        
        # 檢查是否已在執行中
        if self.trajectory_executor.is_executing() or self.trajectory_executor.is_moving_to_initial():
            self.ui_manager.update_info("Cannot move to initial: already executing or moving")
            return
        
        self.ui_manager.update_info("Moving to initial position...")
        
        success, message = self.trajectory_executor.move_to_initial_position_from_csv(progress_callback)
        
        if not success:
            self.ui_manager.update_info(f"Move to initial failed: {message}")
            self.ui_manager.update_status("Ready")
        else:
            self.ui_manager.update_info("Moving to initial position from CSV")

    def _on_stop(self):
        """停止執行回調 - 也停止初始移動"""
        # 停止主執行
        self.trajectory_executor.stop_execution()
        
        # 停止初始移動
        self.trajectory_executor.stop_initial_move()
        
        # 停止記錄並保存數據
        if self.trajectory_recorder.is_recording():
            csv_file, png_file = self.trajectory_recorder.stop_recording()
            if csv_file and png_file:
                self.ui_manager.update_info(f"Recording saved:\nCSV: {csv_file}\nPNG: {png_file}")
            else:
                self.ui_manager.update_info("Execution stopped, recording save failed")
        else:
            self.ui_manager.update_info("Stopped")
            
        self.ui_manager.update_status("Ready")
    
    def initialize(self):
        """初始化系統"""
        try:
            # 設置UI
            if not self.ui_manager.setup_ui():
                return False
                
            self.ui_manager.update_status("Initializing...")
            self.ui_manager.update_log("Setting up scene...")
            
            # 設置場景
            self._setup_scene()
            
            # 初始化NDI檢測器
            if not self.ndi_detector.initialize():
                self.ui_manager.update_status("NDI detector initialization failed")
                return False
            
            # 初始化機器人
            self.ui_manager.update_log("Initializing robot...")
            if self.robot_controller.initialize(self.world):
                self.ui_manager.update_status("System ready")
                self.ui_manager.update_log("All systems initialized successfully")
                self.initialized = True
                return True
            else:
                self.ui_manager.update_status("Robot initialization failed")
                return False
                
        except Exception as e:
            error_msg = f"Initialization error: {e}"
            self.ui_manager.update_status(error_msg)
            self.ui_manager.update_log(error_msg)
            print(error_msg)
            return False
    
    def _setup_scene(self):
        """設置場景"""
        try:
            print("Setting up scene...")
            
            # 重置世界
            self.world.reset()
            
            # 等待物理引擎初始化
            print("Waiting for physics initialization...")
            for i in range(20):  # 增加等待時間
                self.world.step(render=True)
                time.sleep(0.1)
                if i % 5 == 0:
                    print(f"Physics init step {i+1}/20")
            
            # 導入USD模型
            print("Loading USD scene...")
            add_reference_to_stage(usd_path=self.config.USD_FILE_PATH, prim_path="/Root")
            
            # 等待場景完全載入
            print("Waiting for scene to load...")
            for i in range(50):  # 大幅增加等待時間
                self.world.step(render=True)
                time.sleep(0.1)
                if i % 10 == 0:
                    print(f"Scene loading step {i+1}/50")
            
            # 檢查機器人路徑是否存在
            stage = omni.usd.get_context().get_stage()
            robot_prim = stage.GetPrimAtPath(self.config.ROBOT_PRIM_PATH)
            
            if not robot_prim.IsValid():
                available_prims = []
                for prim in stage.Traverse():
                    if prim.GetName() and "tm" in prim.GetName().lower():
                        available_prims.append(str(prim.GetPath()))
                
                error_msg = f"Robot prim not found at {self.config.ROBOT_PRIM_PATH}"
                if available_prims:
                    error_msg += f"\nAvailable TM robot prims: {available_prims[:5]}"
                raise Exception(error_msg)
            
            print("Scene setup completed successfully")
            
        except Exception as e:
            print(f"Scene setup error: {e}")
            raise
        
    def run(self):
        """主運行循環 - 添加初始移動更新"""
        if not self.initialize():
            print("Initialization failed, exiting...")
            return
            
        print("Starting simulation loop...")
        
        try:
            while simulation_app.is_running():
                # 檢查重置請求
                if hasattr(self, 'reset_requested') and self.reset_requested:
                    print("Processing reset request...")
                    self._perform_simple_reset()
                    self.reset_requested = False
                    continue
                
                # 主模擬步進
                self.world.step(render=True)
                
                # 初始移動更新
                if self.trajectory_executor.is_moving_to_initial():
                    try:
                        completed = self.trajectory_executor.update_initial_move()
                        if completed and not self.trajectory_executor.is_moving_to_initial():
                            self.ui_manager.update_info("Reached initial position - ready to execute")
                            self.ui_manager.update_status("Ready")
                    except Exception as e:
                        print(f"Initial move error: {e}")
                        self.trajectory_executor.stop_initial_move()
                        self.ui_manager.update_status("Initial move failed")
                
                # 軌跡執行更新
                if self.trajectory_executor.is_executing():
                    try:
                        self.trajectory_executor.update_execution()
                    except Exception as e:
                        print(f"Trajectory execution error: {e}")
                        self.trajectory_executor.stop_execution()
                        if self.trajectory_recorder.is_recording():
                            self.trajectory_recorder.stop_recording()
                        self.ui_manager.update_status("Execution failed")
                
                # 檢查執行完成
                if (self.trajectory_recorder.is_recording() and 
                    not self.trajectory_executor.is_executing() and 
                    not self.trajectory_executor.is_paused()):
                    
                    csv_file, png_file = self.trajectory_recorder.stop_recording()
                    if csv_file and png_file:
                        self.ui_manager.update_info(f"Execution completed!\nTrajectory saved:\nCSV: {csv_file}\nPNG: {png_file}")
                        print(f"📊 Trajectory execution completed, files saved:")
                        print(f"  CSV: {csv_file}")
                        print(f"  PNG: {png_file}")
                    else:
                        self.ui_manager.update_info("Execution completed, but recording save failed")
                
                # NDI 檢測更新
                current_time = time.time()
                if current_time - self.last_update_time >= 0.02:  # 50Hz
                    try:
                        self._update_detection()
                        self.last_update_time = current_time
                    except Exception as e:
                        print(f"Detection update error: {e}")
                
                self.frame_count += 1
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("🛑 Simulation stopped by user (Ctrl+C)")
            if self.trajectory_recorder.is_recording():
                csv_file, png_file = self.trajectory_recorder.stop_recording()
                print(f"📊 Final recording saved: CSV: {csv_file}, PNG: {png_file}")
        except Exception as e:
            print(f"💥 Critical simulation error: {e}")
            traceback.print_exc()
        finally:
            print("🔚 Closing simulation...")
            simulation_app.close()
        
    def _perform_simple_reset(self):
        """執行簡單重置 - 完全模仿 NDI_Lula_v4_ee.py"""
        try:
            print("🔄 Performing simple reset...")
            
            # 第1步：停止軌跡執行（但不等待）
            self.trajectory_executor.reset_execution()
            
            # 第2步：清除debug draw（但忽略錯誤）
            try:
                self.ndi_detector.clear_debug_draw()
            except:
                pass
            
            # 第3步：重置世界（關鍵！模仿 NDI_Lula_v4_ee.py）
            self.world.reset()
            
            # 第4步：重新初始化機器人（模仿 NDI_Lula_v4_ee.py）
            if hasattr(self, 'robot_controller') and self.robot_controller.robot_articulation:
                self.robot_controller.robot_articulation.initialize()
                
                # 重新初始化 IK 求解器（模仿 NDI_Lula_v4_ee.py）
                try:
                    if hasattr(self.robot_controller, 'ik_solver_interface'):
                        # 簡單驗證IK求解器
                        current_position, current_rotation = self.robot_controller.ik_solver_interface.compute_end_effector_pose()
                        print("IK solver verification successful after reset")
                except Exception as ik_error:
                    print(f"IK solver reset warning: {ik_error}")
            
            # 更新UI
            self.ui_manager.update_info("Reset successful")
            self.ui_manager.update_status("Ready")
            print("✅ Simple reset completed successfully")
            
        except Exception as e:
            error_msg = f"Reset failed: {e}"
            print(f"❌ {error_msg}")
            self.ui_manager.update_info(error_msg)
            self.ui_manager.update_status("Reset failed")   
                    
    def _on_reset(self):
        """重置回調 - 停止記錄"""
        try:
            # 停止記錄（如果正在記錄）
            if self.trajectory_recorder.is_recording():
                csv_file, png_file = self.trajectory_recorder.stop_recording()
                if csv_file and png_file:
                    print(f"📊 Recording saved before reset: CSV: {csv_file}, PNG: {png_file}")
            
            # 更新UI顯示重置進行中
            self.ui_manager.update_info("Reset requested...")
            self.ui_manager.update_status("Reset in progress...")
            
            # 設置重置標誌 - 在主循環中執行
            self.reset_requested = True
            
            print("Reset requested - will execute during next simulation step")
        except Exception as e:
            self.ui_manager.update_info(f"Reset request error: {str(e)}")
            print(f"Error during reset request: {e}")
    
    def _on_info(self):
        """信息回調"""
        info = self.robot_controller.get_robot_info()
        if info:
            info_text = f"Position: {info['position']}\nJoints: {info['joint_positions']}"
            self.ui_manager.update_info(info_text)
        else:
            self.ui_manager.update_info("Robot info unavailable")
    
    def _on_debug_toggle(self):
        """調試切換回調"""
        debug_enabled = self.ui_manager.get_debug_enabled()
        status = "enabled" if debug_enabled else "disabled"
        self.ui_manager.update_log(f"Debug draw {status}")

def main():
    """主函數 - 增強錯誤處理和診斷"""
    print("🚀 啟動 Lula C-Space 軌跡 v5...")
    
    try:
        # 在開始前驗證配置
        print("📋 驗證系統配置...")
        config = SimulationConfig()
        config.validate_configuration()
        
        print("🎮 創建模擬管理器...")
        sim_manager = SimulationManager()
        
        print("▶️ 開始運行模擬...")
        sim_manager.run()
        
    except KeyboardInterrupt:
        print("\n🛑 用戶中斷 (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ 主函數錯誤:")
        print(f"   錯誤類型: {type(e).__name__}")
        print(f"   錯誤訊息: {str(e)}")
        print(f"\n🔧 故障排除建議:")
        print(f"   1. 檢查所有配置檔案路徑是否正確")
        print(f"   2. 確認 Isaac Sim 已完全啟動")
        print(f"   3. 檢查機器人是否已載入場景")
        print(f"   4. 確認 URDF 和機器人描述檔案格式正確")
        print(f"\n📊 Traceback:")
        traceback.print_exc()
    finally:
        print("🏁 應用程式結束")
        try:
            simulation_app.close()
        except:
            pass

if __name__ == "__main__":
    main()