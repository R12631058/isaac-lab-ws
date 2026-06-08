"""
測量新 extended_needle_holder.usd 場景的關鍵參數

輸出:
1. needle_tip body 的位置和四元數
2. phantom 的世界座標位置
3. 所有 body names 和 indices
4. needle_tip 相對於 End_needle 的偏移量
"""

import argparse
from isaaclab.app import AppLauncher

# 創建參數解析器
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args([])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 導入必要模組
import math
import torch
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_rotate, quat_inv, quat_mul

# 新的初始關節角度（從用戶提供的 USD 狀態）
NEW_JOINT_POSITIONS = {
    "joint_1": math.radians(-94.90002),   # -1.6564 rad
    "joint_2": math.radians(43.40004),    # 0.7576 rad
    "joint_3": math.radians(57.49996),    # 1.0036 rad
    "joint_4": math.radians(78.39996),    # 1.3683 rad
    "joint_5": math.radians(-3.600002),   # -0.0628 rad
    "joint_6": math.radians(-90.00003),   # -1.5708 rad
}


@configclass
class ExtendedNeedleMeasureSceneCfg(InteractiveSceneCfg):
    """測量用場景配置"""

    # 載入新的 extended_needle_holder USD
    surgery_room = AssetBaseCfg(
        prim_path="/World/Root",
        spawn=sim_utils.UsdFileCfg(
            usd_path="C:\\Nick\\surgery_team\\surgery_team\\USD\\isaaclab\\extended_needle_holder.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # Robot - 使用 USD 中現有的 TM5-700
    robot = ArticulationCfg(
        prim_path="/World/Root/robotarm_base/robotarm_base/tm5_700",
        spawn=None,  # 不生成新的，使用 USD 中的
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos=NEW_JOINT_POSITIONS,
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint_[1-6]"],
                effort_limit=200.0,
                velocity_limit=2.0,
                stiffness=200.0,
                damping=20.0,
            ),
        },
    )


def main():
    # 創建仿真上下文
    sim_cfg = sim_utils.SimulationCfg(device="cuda:0")
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # 創建場景
    scene_cfg = ExtendedNeedleMeasureSceneCfg(num_envs=1, env_spacing=8.0)
    scene = InteractiveScene(scene_cfg)

    # 播放仿真以初始化
    sim.reset()
    
    # 更新場景
    scene.update(sim.cfg.dt)

    print("\n" + "="*80)
    print("Extended Needle Holder 場景測量結果")
    print("="*80)

    # 獲取機器人
    robot = scene["robot"]
    
    # ============================================
    # 1. 列出所有 body names 和 indices
    # ============================================
    print("\n【1. Body Names 和 Indices】")
    body_names = robot.data.body_names
    for idx, name in enumerate(body_names):
        print(f"  Index {idx:2d}: {name}")
    
    # ============================================
    # 2. 查找 needle 相關 body 的索引
    # ============================================
    print("\n【2. 關鍵 Body 索引】")
    
    needle_idx = None  # 新 USD 使用 "needle" 作為針尖 body
    needle_holder_idx = None
    flange_idx = None
    
    for idx, name in enumerate(body_names):
        if name == "needle":  # 精確匹配 "needle"
            needle_idx = idx
            print(f"  needle 索引: {idx} (名稱: {name})")
        if "needle_holder" in name.lower():
            needle_holder_idx = idx
            print(f"  needle_holder 索引: {idx} (名稱: {name})")
        if "flange" in name.lower():
            flange_idx = idx
            print(f"  flange 索引: {idx} (名稱: {name})")
    
    # ============================================
    # 3. 測量 needle body 的位置和四元數
    # ============================================
    print("\n【3. needle Body 狀態】")
    
    if needle_idx is not None:
        body_state_w = robot.data.body_state_w
        
        # needle 位置和姿態（世界座標）
        tip_pos_w = body_state_w[0, needle_idx, :3]
        tip_quat_w = body_state_w[0, needle_idx, 3:7]
        
        print(f"  世界座標位置 (m):")
        print(f"    X: {tip_pos_w[0].item():.6f}")
        print(f"    Y: {tip_pos_w[1].item():.6f}")
        print(f"    Z: {tip_pos_w[2].item():.6f}")
        print(f"  世界座標四元數 (w, x, y, z):")
        print(f"    ({tip_quat_w[0].item():.6f}, {tip_quat_w[1].item():.6f}, {tip_quat_w[2].item():.6f}, {tip_quat_w[3].item():.6f})")
        
        # 計算相對於機器人基座的位置
        base_pos_w = robot.data.root_pos_w[0]
        base_quat_w = robot.data.root_quat_w[0]
        
        tip_pos_rel = tip_pos_w - base_pos_w
        print(f"\n  相對於機器人基座的位置 (m):")
        print(f"    X: {tip_pos_rel[0].item():.6f}")
        print(f"    Y: {tip_pos_rel[1].item():.6f}")
        print(f"    Z: {tip_pos_rel[2].item():.6f}")
        
        # 輸出用於配置的 preferred_quat
        print(f"\n  📋 preferred_quat 配置值:")
        print(f"    preferred_quat=({tip_quat_w[0].item():.4f}, {tip_quat_w[1].item():.4f}, {tip_quat_w[2].item():.4f}, {tip_quat_w[3].item():.4f})")
    else:
        print("  ⚠️ 未找到 needle body！")
        print("  可用的 body names:")
        for idx, name in enumerate(body_names):
            print(f"    Index {idx}: {name}")
    
    # ============================================
    # 4. 計算 needle 相對於 flange 的偏移
    # ============================================
    print("\n【4. needle 相對於 flange 的偏移】")
    
    if needle_idx is not None and flange_idx is not None:
        body_state_w = robot.data.body_state_w
        
        # flange 位置和姿態
        flange_pos_w = body_state_w[0, flange_idx, :3]
        flange_quat_w = body_state_w[0, flange_idx, 3:7]
        
        # needle 位置
        needle_pos_w = body_state_w[0, needle_idx, :3]
        
        # 計算世界座標中的偏移
        offset_world = needle_pos_w - flange_pos_w
        print(f"  世界座標偏移 (m):")
        print(f"    X: {offset_world[0].item():.6f}")
        print(f"    Y: {offset_world[1].item():.6f}")
        print(f"    Z: {offset_world[2].item():.6f}")
        
        # 將偏移轉換到 flange 的本地座標系
        flange_quat_inv = quat_inv(flange_quat_w.unsqueeze(0))
        offset_local = quat_rotate(flange_quat_inv, offset_world.unsqueeze(0))[0]
        
        print(f"\n  flange 本地座標偏移 (m):")
        print(f"    X: {offset_local[0].item():.6f}")
        print(f"    Y: {offset_local[1].item():.6f}")
        print(f"    Z: {offset_local[2].item():.6f}")
        
        print(f"\n  📋 tip_offset 配置值 (相對於 flange):")
        print(f"    tip_offset=({offset_local[0].item():.6f}, {offset_local[1].item():.6f}, {offset_local[2].item():.6f})")
        
        # 同時計算相對於 needle_holder 的偏移
        if needle_holder_idx is not None:
            holder_pos_w = body_state_w[0, needle_holder_idx, :3]
            holder_quat_w = body_state_w[0, needle_holder_idx, 3:7]
            
            offset_from_holder_world = needle_pos_w - holder_pos_w
            holder_quat_inv = quat_inv(holder_quat_w.unsqueeze(0))
            offset_from_holder_local = quat_rotate(holder_quat_inv, offset_from_holder_world.unsqueeze(0))[0]
            
            print(f"\n  📋 tip_offset 配置值 (相對於 needle_holder):")
            print(f"    tip_offset=({offset_from_holder_local[0].item():.6f}, {offset_from_holder_local[1].item():.6f}, {offset_from_holder_local[2].item():.6f})")
    else:
        print("  ⚠️ 缺少必要的 body 資訊")
    
    # ============================================
    # 5. 測量 Phantom 和 all_components_assemb 位置
    # ============================================
    print("\n【5. 場景物件位置測量】")
    
    # 嘗試從 USD 讀取位置
    import omni.usd
    from pxr import UsdGeom, Gf, Usd
    
    stage = omni.usd.get_context().get_stage()
    
    # 測量所有關鍵物件
    key_objects = [
        "/World/Root/phantom",
        "/World/Root/Phantom",
        "/World/Root/phantom_head",
        "/World/Root/all_components_assemb",
        "/World/Root/all_components_assem",
        "/World/Root/frame_prim",  # 這是放在 phantom 表面的參考框架
    ]
    
    def get_prim_bounds(prim):
        """獲取 prim 的 bounding box"""
        try:
            imageable = UsdGeom.Imageable(prim)
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            bbox = bbox_cache.ComputeWorldBound(prim)
            bbox_range = bbox.ComputeAlignedRange()
            min_pt = bbox_range.GetMin()
            max_pt = bbox_range.GetMax()
            return min_pt, max_pt
        except:
            return None, None
    
    phantom_info = None
    donut_info = None
    
    # 先列出 /World/Root 下所有子物件
    print("\n  /World/Root 下的所有物件:")
    root_prim = stage.GetPrimAtPath("/World/Root")
    if root_prim.IsValid():
        for child in root_prim.GetChildren():
            child_path = str(child.GetPath())
            xformable = UsdGeom.Xformable(child)
            if xformable:
                try:
                    world_transform = xformable.ComputeLocalToWorldTransform(0)
                    translation = world_transform.ExtractTranslation()
                    min_pt, max_pt = get_prim_bounds(child)
                    
                    print(f"\n    {child_path}")
                    print(f"      Position: X={translation[0]:.4f}, Y={translation[1]:.4f}, Z={translation[2]:.4f}")
                    if min_pt and max_pt:
                        print(f"      BBox Min: X={min_pt[0]:.4f}, Y={min_pt[1]:.4f}, Z={min_pt[2]:.4f}")
                        print(f"      BBox Max: X={max_pt[0]:.4f}, Y={max_pt[1]:.4f}, Z={max_pt[2]:.4f}")
                    
                    # 識別 phantom
                    if "phantom" in child_path.lower():
                        phantom_info = {
                            "path": child_path,
                            "pos": translation,
                            "min": min_pt,
                            "max": max_pt
                        }
                    
                    # 識別 frame_prim (放在 phantom 表面的參考框架)
                    if "frame_prim" in child_path.lower():
                        donut_info = {
                            "path": child_path,
                            "pos": translation,
                            "min": min_pt,
                            "max": max_pt
                        }
                except Exception as e:
                    print(f"    {child_path}: 無法獲取變換 ({e})")
    
    # 輸出工作區域建議
    print("\n" + "="*80)
    print("【工作區域建議 (使用 frame_prim 作為參考)】")
    print("="*80)
    
    if phantom_info and donut_info:
        phantom_z_top = phantom_info["max"][2] if phantom_info["max"] else phantom_info["pos"][2]
        frame_z_min = donut_info["min"][2] if donut_info["min"] else donut_info["pos"][2]
        frame_z_max = donut_info["max"][2] if donut_info["max"] else donut_info["pos"][2]
        
        # frame_prim 中心位置
        frame_center_x = donut_info["pos"][0]
        frame_center_y = donut_info["pos"][1]
        frame_center_z = donut_info["pos"][2]
        
        print(f"\n  Phantom 頂部 Z: {phantom_z_top:.4f}")
        print(f"\n  frame_prim (Target 參考區域):")
        print(f"    位置 (中心): X={frame_center_x:.4f}, Y={frame_center_y:.4f}, Z={frame_center_z:.4f}")
        print(f"    BBox Min Z: {frame_z_min:.4f}")
        print(f"    BBox Max Z: {frame_z_max:.4f}")
        
        # frame_prim 的 X/Y 範圍
        if donut_info["min"] and donut_info["max"]:
            frame_x_min = donut_info["min"][0]
            frame_x_max = donut_info["max"][0]
            frame_y_min = donut_info["min"][1]
            frame_y_max = donut_info["max"][1]
            
            print(f"    BBox X 範圍: {frame_x_min:.4f} ~ {frame_x_max:.4f}")
            print(f"    BBox Y 範圍: {frame_y_min:.4f} ~ {frame_y_max:.4f}")
            
            # 計算工作區域尺寸
            x_size = frame_x_max - frame_x_min
            y_size = frame_y_max - frame_y_min
            z_size = frame_z_max - frame_z_min
            
            print(f"\n  frame_prim 尺寸:")
            print(f"    X: {x_size:.4f} m")
            print(f"    Y: {y_size:.4f} m")
            print(f"    Z: {z_size:.4f} m")
            
            print(f"\n  📋 建議的 Target 生成範圍配置值:")
            print(f"    # frame_prim 位置 (相對於世界座標)")
            print(f"    EXTENDED_NEEDLE_FRAME_X = {frame_center_x:.4f}")
            print(f"    EXTENDED_NEEDLE_FRAME_Y = {frame_center_y:.4f}")
            print(f"    EXTENDED_NEEDLE_FRAME_Z = {frame_center_z:.4f}")
            print(f"")
            print(f"    # Target 生成區域 (以 frame_prim 為中心)")
            print(f"    # X 範圍")
            print(f"    pos_x = ({frame_x_min:.4f}, {frame_x_max:.4f})")
            print(f"    # 或使用半徑: pos_x = ({frame_center_x:.4f} - {x_size/2:.4f}, {frame_center_x:.4f} + {x_size/2:.4f})")
            print(f"")
            print(f"    # Y 範圍")
            print(f"    pos_y = ({frame_y_min:.4f}, {frame_y_max:.4f})")
            print(f"    # 或使用半徑: pos_y = ({frame_center_y:.4f} - {y_size/2:.4f}, {frame_center_y:.4f} + {y_size/2:.4f})")
            print(f"")
            print(f"    # Z 範圍 (在 frame_prim 範圍內)")
            print(f"    pos_z = ({frame_z_min:.4f}, {frame_z_max:.4f})")
            print(f"")
            print(f"    # 也可以使用更小的範圍來聚焦在 frame 中心")
            print(f"    # pos_z = ({frame_center_z - 0.01:.4f}, {frame_center_z + 0.01:.4f})")
    
    elif phantom_info:
        print(f"\n  只找到 Phantom，未找到 frame_prim")
        print(f"  Phantom 位置: X={phantom_info['pos'][0]:.4f}, Y={phantom_info['pos'][1]:.4f}, Z={phantom_info['pos'][2]:.4f}")
    
    elif donut_info:
        print(f"\n  只找到 frame_prim，未找到 Phantom")
        print(f"  frame_prim 位置: X={donut_info['pos'][0]:.4f}, Y={donut_info['pos'][1]:.4f}, Z={donut_info['pos'][2]:.4f}")
    
    # ============================================
    # 6. 輸出完整配置摘要
    # ============================================
    print("\n" + "="*80)
    print("配置摘要 - 複製到 tm5_reach_surgery_room_cfg.py")
    print("="*80)
    
    print(f"""
# 新的初始關節角度（弧度）
joint_pos={{
    "joint_1": {NEW_JOINT_POSITIONS["joint_1"]:.6f},  # {math.degrees(NEW_JOINT_POSITIONS["joint_1"]):.2f}°
    "joint_2": {NEW_JOINT_POSITIONS["joint_2"]:.6f},  # {math.degrees(NEW_JOINT_POSITIONS["joint_2"]):.2f}°
    "joint_3": {NEW_JOINT_POSITIONS["joint_3"]:.6f},  # {math.degrees(NEW_JOINT_POSITIONS["joint_3"]):.2f}°
    "joint_4": {NEW_JOINT_POSITIONS["joint_4"]:.6f},  # {math.degrees(NEW_JOINT_POSITIONS["joint_4"]):.2f}°
    "joint_5": {NEW_JOINT_POSITIONS["joint_5"]:.6f},  # {math.degrees(NEW_JOINT_POSITIONS["joint_5"]):.2f}°
    "joint_6": {NEW_JOINT_POSITIONS["joint_6"]:.6f},  # {math.degrees(NEW_JOINT_POSITIONS["joint_6"]):.2f}°
}}
""")

    if needle_idx is not None:
        tip_quat_w = robot.data.body_state_w[0, needle_idx, 3:7]
        print(f"# preferred_quat (needle 初始四元數)")
        print(f"preferred_quat = ({tip_quat_w[0].item():.4f}, {tip_quat_w[1].item():.4f}, {tip_quat_w[2].item():.4f}, {tip_quat_w[3].item():.4f})")
    
    if needle_idx is not None and flange_idx is not None:
        body_state_w = robot.data.body_state_w
        flange_pos_w = body_state_w[0, flange_idx, :3]
        flange_quat_w = body_state_w[0, flange_idx, 3:7]
        needle_pos_w = body_state_w[0, needle_idx, :3]
        offset_world = needle_pos_w - flange_pos_w
        flange_quat_inv = quat_inv(flange_quat_w.unsqueeze(0))
        offset_local = quat_rotate(flange_quat_inv, offset_world.unsqueeze(0))[0]
        print(f"\n# tip_offset (needle 相對於 flange 的本地偏移)")
        print(f"tip_offset = ({offset_local[0].item():.6f}, {offset_local[1].item():.6f}, {offset_local[2].item():.6f})")

    print("\n" + "="*80)
    
    # 清理
    sim.stop()


if __name__ == "__main__":
    main()
    simulation_app.close()
