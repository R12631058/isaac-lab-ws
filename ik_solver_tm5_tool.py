"""
TM5-700 + Extension Link Out 工具偏移 IK 求解器
==============================================

解決問題：
  - 手臂末端有掛件（持針裝置），目標不在 EE (flange) 而在工具尖端 (needle_tip)
  - IK 需要考慮工具偏移（Jacobian 修正）
  - 需要判定目標是否可達，若可達則回傳關節解

使用方式：
  .\isaaclab.bat -p scripts\isaaclab_ws\ik_solver_tm5_tool.py
  .\isaaclab.bat -p scripts\isaaclab_ws\ik_solver_tm5_tool.py --target -0.89 -0.64 0.07 --target_quat -0.8007 0.0423 0.5955 0.0497
  .\isaaclab.bat -p scripts\isaaclab_ws\ik_solver_tm5_tool.py --interactive
  .\isaaclab.bat -p scripts\isaaclab_ws\ik_solver_tm5_tool.py --batch_test

核心概念：
  Differential IK 本質上追蹤的是 body_name 指定的 body frame。
  當 EE 上有工具掛件時，我們有兩個策略：

  策略 A（推薦）: 直接追蹤 needle_tip body
    - body_name = "needle_tip"
    - Jacobian 從 PhysX 取得，index 對應 needle_tip
    - 不需要額外偏移計算
    - 前提：USD 中 needle_tip 必須在 articulation tree 中

  策略 B: 追蹤 flange + 手動工具偏移
    - body_name = "flange"
    - 在 Jacobian 和位姿上套用偏移修正
    - 公式：J_tool = [J_v + J_w × r_offset; R_offset @ J_w]
    - 適用於 needle_tip 不在 articulation tree，或使用獨立 TM5 USD + 外部工具

  本腳本同時示範兩種策略，並支援多迭代收斂 + 可達性判定。
"""

import argparse
import torch
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="IK Solver for TM5 with tool offset (needle holder)")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument(
    "--target", type=float, nargs=3, default=None,
    help="Target position [x, y, z] in robot base frame"
)
parser.add_argument(
    "--target_quat", type=float, nargs=4, default=None,
    help="Target orientation [w, x, y, z] in robot base frame"
)
parser.add_argument("--interactive", action="store_true", help="Interactive mode: prompt for target poses")
parser.add_argument("--batch_test", action="store_true", help="Test a grid of target poses for reachability")
parser.add_argument("--max_iterations", type=int, default=300, help="Max IK iterations per target")
parser.add_argument("--pos_threshold", type=float, default=0.005, help="Position convergence threshold (m)")
parser.add_argument("--rot_threshold", type=float, default=0.05, help="Orientation convergence threshold (rad)")
parser.add_argument(
    "--strategy", type=str, default="needle_tip", choices=["needle_tip", "flange_offset"],
    help="IK strategy: 'needle_tip' = track needle_tip body directly, 'flange_offset' = track flange + offset"
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
from copy import deepcopy

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    subtract_frame_transforms,
    combine_frame_transforms,
    skew_symmetric_matrix,
    matrix_from_quat,
    quat_inv,
    compute_pose_error,
)


# ============================================================
# 場景配置：使用完整的 extension_link_out USD
# ============================================================

# Extension Link Out 初始關節角度
EXTENSION_LINK_OUT_JOINT_POS = {
    "joint_1": -1.656318,
    "joint_2": 0.757474,
    "joint_3": 1.003564,
    "joint_4": 1.368337,
    "joint_5": -0.062832,
    "joint_6": -1.570797,
}

# Preferred orientation (w, x, y, z)
PREFERRED_QUAT = (-0.8007, 0.0423, 0.5955, 0.0497)

# Target 範圍（Phantom 表面附近，robot base frame）
TARGET_X_RANGE = (-0.94, -0.84)
TARGET_Y_RANGE = (-0.69, -0.59)
TARGET_Z_RANGE = (0.04, 0.10)


@configclass
class IKSolverSceneCfg(InteractiveSceneCfg):
    """IK Solver 場景 - 使用 extension_link_out.usd（包含完整手術工具延伸結構）"""

    # 載入 extension_link_out USD
    surgery_room = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Root",
        spawn=sim_utils.UsdFileCfg(
            usd_path=r"C:\Nick\surgery_team\surgery_team\USD\isaaclab\extension_link_out.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # Robot - 使用 USD 中既有的 TM5-700
    # 高 PD 增益確保 IK position target 追蹤穩定
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Root/robotarm_base/robotarm_base/tm5_700",
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos=EXTENSION_LINK_OUT_JOINT_POS,
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint_[1-6]"],
                effort_limit=200.0,
                velocity_limit=2.0,
                stiffness=400.0,   # 高剛性
                damping=80.0,      # 高阻尼
            ),
        },
    )

    # 光源
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


# ============================================================
# IK Solver 核心類別
# ============================================================

class ToolOffsetIKSolver:
    """
    帶工具偏移的 IK 求解器

    支援兩種策略：
    1. needle_tip: 直接追蹤 needle_tip body (推薦，無需偏移計算)
    2. flange_offset: 追蹤 flange + Jacobian 偏移修正
    """

    def __init__(
        self,
        sim: sim_utils.SimulationContext,
        scene: InteractiveScene,
        strategy: str = "needle_tip",
        ik_method: str = "dls",
        max_iterations: int = 300,
        pos_threshold: float = 0.005,
        rot_threshold: float = 0.05,
    ):
        self.sim = sim
        self.scene = scene
        self.robot = scene["robot"]
        self.strategy = strategy
        self.max_iterations = max_iterations
        self.pos_threshold = pos_threshold
        self.rot_threshold = rot_threshold
        self.device = sim.device

        # 建立 Differential IK Controller (DLS method)
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method=ik_method,
            ik_params={"lambda_val": 0.01},
        )
        self.controller = DifferentialIKController(
            diff_ik_cfg, num_envs=scene.num_envs, device=self.device
        )

        # 根據策略選擇追蹤 body
        if self.strategy == "needle_tip":
            # 直接追蹤 needle_tip body
            self.robot_entity_cfg = SceneEntityCfg(
                "robot", joint_names=["joint_[1-6]"], body_names=["needle_tip"]
            )
            self._tool_offset_pos = None  # 不需要偏移
            self._tool_offset_rot = None
            print("[IK Solver] 策略: 直接追蹤 needle_tip body")
        else:
            # 追蹤 flange + 偏移
            self.robot_entity_cfg = SceneEntityCfg(
                "robot", joint_names=["joint_[1-6]"], body_names=["flange"]
            )
            # 工具偏移：從 flange 到 needle_tip 的局部變換
            # 注意：這個偏移值需要從 USD 測量或已知的機械設計中取得
            # extension_link_out 的 needle_tip 相對於 flange 的偏移
            # 此處為示意值，你需要根據實際 USD 結構測量
            self._tool_offset_pos = torch.tensor([[0.0, 0.0, 0.5]], device=self.device).repeat(scene.num_envs, 1)
            self._tool_offset_rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).repeat(scene.num_envs, 1)
            print("[IK Solver] 策略: flange + 工具偏移")
            print(f"  偏移位置: {self._tool_offset_pos[0].tolist()}")
            print(f"  偏移姿態: {self._tool_offset_rot[0].tolist()}")

        # 解析場景實體
        self.robot_entity_cfg.resolve(scene)
        self.joint_ids = self.robot_entity_cfg.joint_ids
        self.body_idx = self.robot_entity_cfg.body_ids[0]

        # Jacobian index（固定基座需 -1）
        if self.robot.is_fixed_base:
            self.ee_jacobi_idx = self.body_idx - 1
        else:
            self.ee_jacobi_idx = self.body_idx

        print(f"  追蹤 body index: {self.body_idx}")
        print(f"  Jacobian index: {self.ee_jacobi_idx}")
        print(f"  Joint IDs: {self.joint_ids}")

        # 記錄初始關節角度用於重置
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.default_joint_vel = self.robot.data.default_joint_vel.clone()

        # 設置 Visualization Markers
        self._setup_markers()

    def _setup_markers(self):
        """設置視覺化標記"""
        ee_cfg = deepcopy(FRAME_MARKER_CFG)
        ee_cfg.prim_path = "/Visuals/ee_current"
        ee_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        self.ee_marker = VisualizationMarkers(ee_cfg)

        goal_cfg = deepcopy(FRAME_MARKER_CFG)
        goal_cfg.prim_path = "/Visuals/goal"
        goal_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
        self.goal_marker = VisualizationMarkers(goal_cfg)

    def _get_ee_pose_in_base(self) -> tuple[torch.Tensor, torch.Tensor]:
        """取得追蹤 body 在 robot base frame 的位姿"""
        ee_pos_w = self.robot.data.body_pos_w[:, self.body_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, self.body_idx]
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w

        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )

        # 若使用 flange_offset 策略，加上偏移
        if self.strategy == "flange_offset" and self._tool_offset_pos is not None:
            ee_pos_b, ee_quat_b = combine_frame_transforms(
                ee_pos_b, ee_quat_b, self._tool_offset_pos, self._tool_offset_rot
            )

        return ee_pos_b, ee_quat_b

    def _get_jacobian(self) -> torch.Tensor:
        """取得 Jacobian 矩陣（含工具偏移修正）"""
        # 從 PhysX 取得原始 Jacobian（世界座標）
        jacobian_w = self.robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.joint_ids]

        # 轉換到 base frame
        base_rot = self.robot.data.root_quat_w
        base_rot_matrix = matrix_from_quat(quat_inv(base_rot))
        jacobian = jacobian_w.clone()
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian_w[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian_w[:, 3:, :])

        # 若使用 flange_offset 策略，修正 Jacobian
        if self.strategy == "flange_offset" and self._tool_offset_pos is not None:
            # 平移部分: v_tool = v_flange + w_flange × r_tool = (J_v - [r]× @ J_w) * q
            jacobian[:, 0:3, :] += torch.bmm(
                -skew_symmetric_matrix(self._tool_offset_pos), jacobian[:, 3:, :]
            )
            # 旋轉部分: w_tool = R_tool_flange @ w_flange
            jacobian[:, 3:, :] = torch.bmm(
                matrix_from_quat(self._tool_offset_rot), jacobian[:, 3:, :]
            )

        return jacobian

    def reset_robot(self):
        """將機器人重置到預設姿態"""
        self.robot.write_joint_state_to_sim(self.default_joint_pos, self.default_joint_vel)
        self.robot.reset()
        # 跑幾步讓模擬穩定
        sim_dt = self.sim.get_physics_dt()
        for _ in range(10):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(sim_dt)

    def solve(
        self,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        verbose: bool = True,
    ) -> dict:
        """
        求解 IK：給定目標位姿，回傳關節解或報告不可達

        Args:
            target_pos: 目標位置 (N, 3) 在 robot base frame
            target_quat: 目標四元數 (N, 4) [w, x, y, z] 在 robot base frame
            verbose: 是否印出詳細資訊

        Returns:
            dict: {
                "success": bool,        # 是否收斂
                "joint_pos": Tensor,     # 解出的關節角度 (N, 6)
                "pos_error": float,      # 最終位置誤差 (m)
                "rot_error": float,      # 最終旋轉誤差 (rad)
                "iterations": int,       # 使用的迭代次數
                "ee_pos_achieved": Tensor,   # 實際達到的位置
                "ee_quat_achieved": Tensor,  # 實際達到的姿態
            }
        """
        self.reset_robot()

        # 設定目標命令 [x, y, z, qw, qx, qy, qz]
        ik_commands = torch.zeros(self.scene.num_envs, 7, device=self.device)
        ik_commands[:, 0:3] = target_pos
        ik_commands[:, 3:7] = target_quat

        # 重置控制器
        self.controller.reset()
        self.controller.set_command(ik_commands)

        sim_dt = self.sim.get_physics_dt()
        best_pos_error = float('inf')
        best_rot_error = float('inf')
        best_joint_pos = None
        converged = False
        stagnation_count = 0
        prev_pos_error = float('inf')

        for i in range(self.max_iterations):
            # 取得當前狀態
            ee_pos_b, ee_quat_b = self._get_ee_pose_in_base()
            jacobian = self._get_jacobian()
            joint_pos = self.robot.data.joint_pos[:, self.joint_ids]

            # 計算誤差
            pos_error_vec, rot_error_vec = compute_pose_error(
                ee_pos_b, ee_quat_b, target_pos, target_quat, rot_error_type="axis_angle"
            )
            pos_error = pos_error_vec.norm(dim=-1).mean().item()
            rot_error = rot_error_vec.norm(dim=-1).mean().item()

            # 記錄最佳解
            if pos_error < best_pos_error:
                best_pos_error = pos_error
                best_rot_error = rot_error
                best_joint_pos = joint_pos.clone()

            # 檢查收斂
            if pos_error < self.pos_threshold and rot_error < self.rot_threshold:
                converged = True
                if verbose:
                    print(f"  [✓] 收斂於第 {i+1} 步: 位置誤差={pos_error*1000:.2f}mm, 旋轉誤差={math.degrees(rot_error):.2f}°")
                break

            # 檢查停滯（誤差不再改善）
            if abs(prev_pos_error - pos_error) < 1e-6:
                stagnation_count += 1
                if stagnation_count > 50:
                    if verbose:
                        print(f"  [!] 停滯於第 {i+1} 步: 位置誤差={pos_error*1000:.2f}mm, 旋轉誤差={math.degrees(rot_error):.2f}°")
                    break
            else:
                stagnation_count = 0
            prev_pos_error = pos_error

            # 計算 IK
            joint_pos_des = self.controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

            # 關節限制裁剪 (TM5-700 的關節限制大約 ±270°)
            joint_limits_low = torch.tensor([-4.712, -1.571, -2.618, -3.142, -4.712, -4.712], device=self.device)
            joint_limits_high = torch.tensor([4.712, 2.618, 2.618, 3.142, 4.712, 4.712], device=self.device)
            joint_pos_des = torch.clamp(joint_pos_des, joint_limits_low, joint_limits_high)

            # 套用關節位置目標
            self.robot.set_joint_position_target(joint_pos_des, joint_ids=self.joint_ids)
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(sim_dt)

            # 更新視覺化
            ee_pos_w = self.robot.data.body_pos_w[:, self.body_idx]
            ee_quat_w = self.robot.data.body_quat_w[:, self.body_idx]
            self.ee_marker.visualize(ee_pos_w[:, 0:3], ee_quat_w)
            self.goal_marker.visualize(
                target_pos + self.robot.data.root_pos_w,
                target_quat,
            )

            # 定期印出進度
            if verbose and (i + 1) % 50 == 0:
                print(f"  [{i+1}/{self.max_iterations}] 位置誤差={pos_error*1000:.2f}mm, 旋轉誤差={math.degrees(rot_error):.2f}°")

        # 最終結果
        ee_pos_b, ee_quat_b = self._get_ee_pose_in_base()
        joint_pos_final = self.robot.data.joint_pos[:, self.joint_ids]

        result = {
            "success": converged,
            "joint_pos": best_joint_pos if best_joint_pos is not None else joint_pos_final,
            "pos_error": best_pos_error,
            "rot_error": best_rot_error,
            "iterations": i + 1,
            "ee_pos_achieved": ee_pos_b,
            "ee_quat_achieved": ee_quat_b,
        }

        return result

    def solve_position_only(
        self,
        target_pos: torch.Tensor,
        verbose: bool = True,
    ) -> dict:
        """
        只求解位置（不管姿態），更容易收斂

        Args:
            target_pos: 目標位置 (N, 3) 在 robot base frame

        Returns:
            dict: 與 solve() 相同格式
        """
        self.reset_robot()

        # 使用 position-only 控制器
        pos_ik_cfg = DifferentialIKControllerCfg(
            command_type="position",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.01},
        )
        pos_controller = DifferentialIKController(
            pos_ik_cfg, num_envs=self.scene.num_envs, device=self.device
        )

        ik_commands = target_pos.clone()
        ee_pos_b, ee_quat_b = self._get_ee_pose_in_base()
        pos_controller.reset()
        pos_controller.set_command(ik_commands, ee_quat=ee_quat_b)

        sim_dt = self.sim.get_physics_dt()
        best_pos_error = float('inf')
        best_joint_pos = None
        converged = False

        for i in range(self.max_iterations):
            ee_pos_b, ee_quat_b = self._get_ee_pose_in_base()
            jacobian = self._get_jacobian()
            joint_pos = self.robot.data.joint_pos[:, self.joint_ids]

            pos_error = (target_pos - ee_pos_b).norm(dim=-1).mean().item()

            if pos_error < best_pos_error:
                best_pos_error = pos_error
                best_joint_pos = joint_pos.clone()

            if pos_error < self.pos_threshold:
                converged = True
                if verbose:
                    print(f"  [✓] 位置收斂於第 {i+1} 步: 誤差={pos_error*1000:.2f}mm")
                break

            # 只用位置部分的 Jacobian
            jacobian_pos = jacobian[:, 0:3, :]
            pos_error_vec = target_pos - ee_pos_b
            delta_joint_pos = self._dls_solve(pos_error_vec, jacobian_pos)
            joint_pos_des = joint_pos + delta_joint_pos

            self.robot.set_joint_position_target(joint_pos_des, joint_ids=self.joint_ids)
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(sim_dt)

            if verbose and (i + 1) % 50 == 0:
                print(f"  [{i+1}/{self.max_iterations}] 位置誤差={pos_error*1000:.2f}mm")

        return {
            "success": converged,
            "joint_pos": best_joint_pos,
            "pos_error": best_pos_error,
            "rot_error": 0.0,
            "iterations": i + 1,
            "ee_pos_achieved": ee_pos_b,
            "ee_quat_achieved": ee_quat_b,
        }

    def _dls_solve(self, delta_pose: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
        """Damped Least Squares 求解"""
        lambda_val = 0.01
        jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
        lambda_matrix = (lambda_val ** 2) * torch.eye(n=jacobian.shape[1], device=self.device)
        delta_joint_pos = (
            jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
        )
        return delta_joint_pos.squeeze(-1)


# ============================================================
# 主要功能函式
# ============================================================

def print_result(result: dict, target_pos, target_quat=None):
    """格式化印出 IK 結果"""
    print("\n" + "=" * 60)
    if result["success"]:
        print("  IK 求解成功 ✓")
    else:
        print("  IK 求解失敗 ✗ (目標可能不可達)")
    print("=" * 60)
    print(f"  目標位置:  [{target_pos[0, 0]:.4f}, {target_pos[0, 1]:.4f}, {target_pos[0, 2]:.4f}]")
    if target_quat is not None:
        print(f"  目標姿態:  [{target_quat[0, 0]:.4f}, {target_quat[0, 1]:.4f}, {target_quat[0, 2]:.4f}, {target_quat[0, 3]:.4f}]")
    print(f"  位置誤差:  {result['pos_error']*1000:.2f} mm")
    print(f"  旋轉誤差:  {math.degrees(result['rot_error']):.2f}°")
    print(f"  迭代次數:  {result['iterations']}")
    print(f"  達到位置:  [{result['ee_pos_achieved'][0, 0]:.4f}, {result['ee_pos_achieved'][0, 1]:.4f}, {result['ee_pos_achieved'][0, 2]:.4f}]")

    if result["success"]:
        joint_pos = result["joint_pos"][0]
        print(f"\n  關節解 (rad):")
        joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        for name, val in zip(joint_names, joint_pos):
            print(f"    {name}: {val:.6f} rad ({math.degrees(val):.2f}°)")
    print("=" * 60)


def run_single_target(solver, args_cli):
    """求解單一目標"""
    if args_cli.target is not None:
        target_pos = torch.tensor([args_cli.target], device=solver.device)
    else:
        # 預設目標：Phantom 中心附近
        target_pos = torch.tensor([[-0.89, -0.64, 0.07]], device=solver.device)

    if args_cli.target_quat is not None:
        target_quat = torch.tensor([args_cli.target_quat], device=solver.device)
    else:
        target_quat = torch.tensor([list(PREFERRED_QUAT)], device=solver.device)

    print(f"\n[IK Solver] 求解目標:")
    print(f"  位置: {target_pos[0].tolist()}")
    print(f"  姿態: {target_quat[0].tolist()} (w,x,y,z)")

    result = solver.solve(target_pos, target_quat, verbose=True)
    print_result(result, target_pos, target_quat)

    # 停留一段時間讓使用者觀察
    print("\n[INFO] 維持姿態中... (Ctrl+C 結束)")
    sim_dt = solver.sim.get_physics_dt()
    while simulation_app.is_running():
        solver.scene.write_data_to_sim()
        solver.sim.step()
        solver.scene.update(sim_dt)

        # 更新標記
        ee_pos_w = solver.robot.data.body_pos_w[:, solver.body_idx]
        ee_quat_w = solver.robot.data.body_quat_w[:, solver.body_idx]
        solver.ee_marker.visualize(ee_pos_w, ee_quat_w)
        solver.goal_marker.visualize(
            target_pos + solver.robot.data.root_pos_w,
            target_quat,
        )


def run_interactive(solver):
    """互動模式：反覆輸入目標求解"""
    print("\n[互動模式] 輸入目標位姿進行 IK 求解")
    print("  格式: x y z [qw qx qy qz]")
    print("  只輸入位置 (3 個值) 會使用預設姿態")
    print("  輸入 'quit' 結束\n")

    while simulation_app.is_running():
        try:
            user_input = input("目標 > ").strip()
        except EOFError:
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        values = [float(v) for v in user_input.split()]
        if len(values) == 3:
            target_pos = torch.tensor([values], device=solver.device)
            target_quat = torch.tensor([list(PREFERRED_QUAT)], device=solver.device)
        elif len(values) == 7:
            target_pos = torch.tensor([values[:3]], device=solver.device)
            target_quat = torch.tensor([values[3:]], device=solver.device)
        else:
            print("  錯誤：請輸入 3 或 7 個數值")
            continue

        result = solver.solve(target_pos, target_quat, verbose=True)
        print_result(result, target_pos, target_quat)


def run_batch_test(solver):
    """批次測試：在 target 範圍內測試一網格的目標"""
    print("\n[批次測試模式] 在 Phantom 表面上測試可達性")

    # 建立測試網格
    n_x, n_y, n_z = 5, 5, 3
    xs = torch.linspace(TARGET_X_RANGE[0], TARGET_X_RANGE[1], n_x)
    ys = torch.linspace(TARGET_Y_RANGE[0], TARGET_Y_RANGE[1], n_y)
    zs = torch.linspace(TARGET_Z_RANGE[0], TARGET_Z_RANGE[1], n_z)

    target_quat = torch.tensor([list(PREFERRED_QUAT)], device=solver.device)

    results_grid = []
    total = n_x * n_y * n_z
    success_count = 0
    tested = 0

    print(f"  測試 {total} 個目標點 (X: {n_x}, Y: {n_y}, Z: {n_z})\n")

    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            for iz, z in enumerate(zs):
                tested += 1
                target_pos = torch.tensor([[x.item(), y.item(), z.item()]], device=solver.device)

                result = solver.solve(target_pos, target_quat, verbose=False)
                status = "✓" if result["success"] else "✗"
                if result["success"]:
                    success_count += 1

                print(
                    f"  [{tested}/{total}] ({x:.3f}, {y:.3f}, {z:.3f}) "
                    f"{status} 誤差={result['pos_error']*1000:.1f}mm / {math.degrees(result['rot_error']):.1f}°"
                )
                results_grid.append({
                    "pos": (x.item(), y.item(), z.item()),
                    "success": result["success"],
                    "pos_error": result["pos_error"],
                    "rot_error": result["rot_error"],
                })

    print(f"\n{'='*60}")
    print(f"  批次測試結果: {success_count}/{total} 可達 ({100*success_count/total:.1f}%)")
    print(f"{'='*60}")

    # 印出失敗的目標
    failed = [r for r in results_grid if not r["success"]]
    if failed:
        print(f"\n  不可達的目標 ({len(failed)} 個):")
        for r in failed:
            print(f"    ({r['pos'][0]:.3f}, {r['pos'][1]:.3f}, {r['pos'][2]:.3f}) "
                  f"誤差={r['pos_error']*1000:.1f}mm / {math.degrees(r['rot_error']):.1f}°")


def main():
    """主函式"""
    # 初始化模擬
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view((-1.5, -1.5, 1.5), (-0.5, -0.5, 0.5))

    # 建立場景
    scene_cfg = IKSolverSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # 啟動模擬
    sim.reset()
    print("[INFO] 模擬啟動完成")

    # 顯示機器人 body 資訊
    robot = scene["robot"]
    print(f"\n[Robot Info]")
    print(f"  Body names: {robot.body_names}")
    print(f"  Joint names: {robot.joint_names}")
    print(f"  Num bodies: {robot.num_bodies}")
    print(f"  Num joints: {robot.num_joints}")
    print(f"  Is fixed base: {robot.is_fixed_base}")

    # 建立 IK Solver
    solver = ToolOffsetIKSolver(
        sim=sim,
        scene=scene,
        strategy=args_cli.strategy,
        max_iterations=args_cli.max_iterations,
        pos_threshold=args_cli.pos_threshold,
        rot_threshold=args_cli.rot_threshold,
    )

    # 根據模式執行
    if args_cli.interactive:
        run_interactive(solver)
    elif args_cli.batch_test:
        run_batch_test(solver)
    else:
        run_single_target(solver, args_cli)


if __name__ == "__main__":
    main()
    simulation_app.close()
