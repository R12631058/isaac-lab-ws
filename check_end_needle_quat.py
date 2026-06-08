"""
查看 End_needle 在初始姿態下的世界坐標四元數
這個腳本用於驗證 TM5NeedleTipStableReachEnvCfg 中使用的 preferred_quat 是否正確
"""
import argparse
from isaaclab.app import AppLauncher

# 啟動 app
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args([])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 導入必要模組
from pxr import Usd, UsdGeom, Gf
import omni.usd

def get_world_transform_with_rotation(stage, prim_path):
    """獲取 prim 的世界座標變換（包含旋轉）"""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"Prim {prim_path} not found!")
        return None, None
    
    xformable = UsdGeom.Xformable(prim)
    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    
    # 提取位置
    translation = world_transform.ExtractTranslation()
    
    # 提取旋轉四元數
    rotation_quat = world_transform.ExtractRotationQuat()
    
    return translation, rotation_quat

def main():
    # 開啟場景
    usd_path = "C:/Nick/surgery_team/surgery_team/USD/isaaclab/phantom.usd"
    omni.usd.get_context().open_stage(usd_path)
    stage = omni.usd.get_context().get_stage()
    
    print("=" * 60)
    print("End_needle Quaternion Information (for TM5NeedleTipStableReachEnvCfg)")
    print("=" * 60)
    
    # 查看關鍵物件的四元數
    paths_to_check = [
        "/Root/robotarm_base/robotarm_base/tm5_700/End_needle",
        "/Root/robotarm_base/robotarm_base/End_needle",
        "/Root/robotarm_base/robotarm_base/End_needle/End_needle",
        "/Root/robotarm_base/robotarm_base/End_needle/End_needle/needle_tip",
    ]
    
    for path in paths_to_check:
        pos, quat = get_world_transform_with_rotation(stage, path)
        if pos is not None:
            print(f"\n{path}:")
            print(f"  World Position: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
            # 四元數：Gf.Quatd 使用 (w, x, y, z) 格式
            w = quat.GetReal()
            imaginary = quat.GetImaginary()
            x, y, z = imaginary[0], imaginary[1], imaginary[2]
            print(f"  World Quaternion (w, x, y, z): ({w:.4f}, {x:.4f}, {y:.4f}, {z:.4f})")
            
            # 也轉換為 Euler 角度方便理解
            rotation_matrix = Gf.Matrix3d(quat)
            euler = Gf.Rotation(rotation_matrix).Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))
            print(f"  Euler Angles (XYZ deg): ({euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f})")
    
    print("\n" + "=" * 60)
    print("使用方式：")
    print("將上面 End_needle 的 World Quaternion 複製到")
    print("TM5NeedleTipStableReachEnvCfg 的 preferred_quat 參數中")
    print("=" * 60)

if __name__ == "__main__":
    main()
    simulation_app.close()
