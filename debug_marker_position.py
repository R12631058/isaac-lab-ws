"""
診斷 marker 位置追蹤問題
比較 USD xform vs PhysX link transform + offset
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug marker position tracking")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import omni.usd
from pxr import UsdGeom
from isaaclab.sim import SimulationContext
import isaaclab.sim as sim_utils
from isaacsim.core.simulation_manager import SimulationManager


def get_usd_position(prim_path: str):
    """從 USD xform 取得位置"""
    try:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return None
        xformable = UsdGeom.Xformable(prim)
        world_transform = xformable.ComputeLocalToWorldTransform(0)
        t = world_transform.ExtractTranslation()
        return np.array([t[0], t[1], t[2]], dtype=float)
    except Exception as e:
        return None


def get_physx_position(physics_sim_view, prim_path: str):
    """從 PhysX RigidBodyView 取得位置"""
    try:
        rb_view = physics_sim_view.create_rigid_body_view(prim_path)
        if rb_view is None or rb_view.count == 0:
            return None
        transforms = rb_view.get_transforms()
        pos = transforms[0, :3]
        if hasattr(pos, 'cpu'):
            pos = pos.cpu().numpy()
        return np.array(pos, dtype=float)
    except Exception as e:
        return None


def main():
    print("\n" + "="*70)
    print("診斷：USD xform vs PhysX 位置")
    print("="*70)

    # 設定模擬
    sim_cfg = sim_utils.SimulationCfg(
        device="cuda:0",
        dt=1/60.0,
    )
    sim = SimulationContext(sim_cfg)
    
    # 載入場景
    sim_cfg_loader = sim_utils.UsdFileCfg(
        usd_path="C:/Users/RMML/OneDrive - National Yang Ming Chiao Tung University/surgery_room_v10.usd"
    )
    sim_cfg_loader.func("/Root", sim_cfg_loader)
    
    sim.reset()
    for _ in range(30):
        sim.step()
    
    # 取得 physics_sim_view
    physics_sim_view = SimulationManager.get_physics_sim_view()
    
    # 測試路徑
    test_paths = {
        "link_6": "/Root/robotarm_base/tm5_700/link_6",
        "End_needle": "/Root/robotarm_base/End_needle",
        "BM marker": "/Root/robotarm_base/End_needle/BM/BM_meter/bm_a",
        "NDI_emitter": "/Root/NDI_02/NDI_emitter",
    }
    
    print("\n初始位置比較：")
    print("-"*70)
    initial_usd = {}
    initial_physx = {}
    
    for name, path in test_paths.items():
        usd_pos = get_usd_position(path)
        physx_pos = get_physx_position(physics_sim_view, path)
        
        initial_usd[name] = usd_pos
        initial_physx[name] = physx_pos
        
        print(f"\n{name}:")
        if usd_pos is not None:
            print(f"  USD:   ({usd_pos[0]:.4f}, {usd_pos[1]:.4f}, {usd_pos[2]:.4f})")
        else:
            print(f"  USD:   [無效]")
        if physx_pos is not None:
            print(f"  PhysX: ({physx_pos[0]:.4f}, {physx_pos[1]:.4f}, {physx_pos[2]:.4f})")
        else:
            print(f"  PhysX: [不是 RigidBody]")
    
    # 設定關節位置
    print("\n" + "="*70)
    print("移動關節到目標位置...")
    print("="*70)
    
    arti_path = "/Root/robotarm_base/tm5_700/base"
    arti_view = physics_sim_view.create_articulation_view(arti_path)
    
    target_joints_deg = np.array([-107.26, 63.4, 60.3, -125, 98.4, 90.1])
    target_joints_rad = np.radians(target_joints_deg)
    target_array = target_joints_rad.reshape(1, -1).astype(np.float32)
    zero_vel = np.zeros((1, 6), dtype=np.float32)
    
    # 設定位置並維持
    for i in range(60):
        arti_view.set_dof_positions(target_array, indices=np.array([0]))
        arti_view.set_dof_velocities(zero_vel, indices=np.array([0]))
        sim.step()
    
    # 嘗試同步
    physics_sim_view.update_articulations_kinematic()
    sim.step()
    
    print("\n移動後位置比較：")
    print("-"*70)
    
    for name, path in test_paths.items():
        usd_pos = get_usd_position(path)
        physx_pos = get_physx_position(physics_sim_view, path)
        
        print(f"\n{name}:")
        if usd_pos is not None:
            print(f"  USD:   ({usd_pos[0]:.4f}, {usd_pos[1]:.4f}, {usd_pos[2]:.4f})")
            if initial_usd.get(name) is not None:
                diff = np.linalg.norm(usd_pos - initial_usd[name])
                print(f"         USD 移動: {diff*100:.2f} cm")
        else:
            print(f"  USD:   [無效]")
            
        if physx_pos is not None:
            print(f"  PhysX: ({physx_pos[0]:.4f}, {physx_pos[1]:.4f}, {physx_pos[2]:.4f})")
            if initial_physx.get(name) is not None:
                diff = np.linalg.norm(physx_pos - initial_physx[name])
                print(f"         PhysX 移動: {diff*100:.2f} cm")
        else:
            print(f"  PhysX: [不是 RigidBody]")
    
    print("\n" + "="*70)
    print("結論")
    print("="*70)
    print("如果 USD 位置沒變但 PhysX link_6 位置變了，")
    print("就證實 update_articulations_kinematic() 無法同步 USD xform")
    print("="*70)
    
    simulation_app.close()


if __name__ == "__main__":
    main()
