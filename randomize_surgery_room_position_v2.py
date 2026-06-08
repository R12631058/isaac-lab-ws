"""Randomize surgery room position - Version 2

使用 surgery_room.init_state.pos 來移動整個場景（包括 phantom）
測試多環境下每個環境能否有不同的位置
"""

import torch
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv


@configclass
class RandomizeSurgeryRoomPositionCfg:
    """Configuration for randomizing surgery room position."""

    x_range: tuple[float, float] = (-0.3, 0.3)  # X 偏移範圍
    y_range: tuple[float, float] = (0.0, 0.0)    # Y 固定
    z_range: tuple[float, float] = (0.0, 0.0)    # Z 固定


def randomize_surgery_room_position(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
):
    """
    隨機化 surgery_room 的位置（移動整個場景）
    
    這會影響場景中的所有物件（robot, phantom, 等等）
    robotarm_base 在 USD 中的本地位置是 (0.3596, -0.1741, 0.9493)
    """
    from pxr import Gf, UsdGeom
    
    # Startup mode 時 env_ids 可能是 None
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    
    num_envs_to_reset = len(env_ids)
    
    # 生成線性分布的 X 偏移（方便驗證）
    x_offsets = torch.linspace(x_range[0], x_range[1], num_envs_to_reset, device=env.device)
    y_offsets = torch.zeros(num_envs_to_reset, device=env.device) + y_range[0]
    z_offsets = torch.zeros(num_envs_to_reset, device=env.device) + z_range[0]
    
    # 獲取 USD stage
    stage = env.scene.stage
    
    print(f"[Randomizer] Setting surgery_room positions for {num_envs_to_reset} environments:")
    
    # 為每個環境設置位置
    for i, env_id in enumerate(env_ids):
        env_id_int = env_id.item()
        prim_path = f"/World/envs/env_{env_id_int}/Root"
        
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"[WARN] Cannot find {prim_path}")
            continue
        
        xformable = UsdGeom.Xformable(prim)
        
        # 清除現有的 xform 操作
        xformable.ClearXformOpOrder()
        
        # 移除 pivot (如果存在)
        if prim.HasAttribute("xformOp:transform"):
            prim.RemoveProperty("xformOp:transform")
        
        # 添加新的 translate
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(
            float(x_offsets[i].item()),
            float(y_offsets[i].item()),
            float(z_offsets[i].item())
        ))
        
        if i < 5:  # 只顯示前5個
            print(f"  Env {env_id_int}: X offset = {x_offsets[i].item():.3f}m")
    
    if num_envs_to_reset > 5:
        print(f"  ... and {num_envs_to_reset - 5} more environments")

