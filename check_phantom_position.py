"""
查看 phantom.usd 中各物件的位置資訊
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
from pxr import Usd, UsdGeom
import omni.usd

def get_world_transform(stage, prim_path):
    """獲取 prim 的世界座標變換"""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"Prim {prim_path} not found!")
        return None
    
    xformable = UsdGeom.Xformable(prim)
    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    
    # 提取位置
    translation = world_transform.ExtractTranslation()
    return translation

def main():
    # 開啟場景
    usd_path = "C:/Nick/surgery_team/surgery_team/USD/isaaclab/phantom.usd"
    omni.usd.get_context().open_stage(usd_path)
    stage = omni.usd.get_context().get_stage()
    
    print("=" * 60)
    print("Phantom Scene Position Information")
    print("=" * 60)
    
    # 查看關鍵物件位置
    paths_to_check = [
        "/Root",
        "/Root/phantom",
        "/Root/robotarm_base",
        "/Root/robotarm_base/robotarm_base",
        "/Root/robotarm_base/robotarm_base/tm5_700",
        "/Root/robotarm_base/robotarm_base/tm5_700/base",
    ]
    
    for path in paths_to_check:
        pos = get_world_transform(stage, path)
        if pos:
            print(f"{path}:")
            print(f"  World Position: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
    
    # 查看 phantom 的 bounding box
    phantom_prim = stage.GetPrimAtPath("/Root/phantom")
    if phantom_prim.IsValid():
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeWorldBound(phantom_prim)
        bbox_range = bbox.GetRange()
        print("\n" + "=" * 60)
        print("Phantom Bounding Box (World Coordinates):")
        print(f"  Min: ({bbox_range.GetMin()[0]:.4f}, {bbox_range.GetMin()[1]:.4f}, {bbox_range.GetMin()[2]:.4f})")
        print(f"  Max: ({bbox_range.GetMax()[0]:.4f}, {bbox_range.GetMax()[1]:.4f}, {bbox_range.GetMax()[2]:.4f})")
        center = bbox_range.GetMidpoint()
        print(f"  Center: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
        print(f"  Size: ({bbox_range.GetSize()[0]:.4f}, {bbox_range.GetSize()[1]:.4f}, {bbox_range.GetSize()[2]:.4f})")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
    simulation_app.close()
