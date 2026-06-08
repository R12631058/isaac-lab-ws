#!/usr/bin/env python3
"""檢查手術室 USD 檔案結構"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass


@configclass
class SurgeryRoomSceneCfg(InteractiveSceneCfg):
    surgery_room = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SurgeryRoom",
        spawn=sim_utils.UsdFileCfg(
            usd_path="C:/Nick/surgery_team/surgery_team/USD/isaaclab/surgeryroom_isaac_lab.usd",
        ),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )


def main():
    scene_cfg = SurgeryRoomSceneCfg(num_envs=4, env_spacing=3.0)
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(3.0, 3.0, 2.5), target=(0.0, 0.0, 0.5))
    
    scene = InteractiveScene(scene_cfg)
    
    print("✅ 場景載入成功!")
    
    # 獲取 stage
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    
    # 檢查環境 0
    env_0_path = "/World/envs/env_0/SurgeryRoom"
    root_prim = stage.GetPrimAtPath(env_0_path)
    
    output_file = "surgery_room_structure.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("手術室 USD 場景結構分析\n")
        f.write("="*80 + "\n\n")
        
        if root_prim.IsValid():
            f.write(f"✅ 找到場景根: {env_0_path}\n\n")
            
            def print_tree(prim, indent=0, max_depth=10):
                if indent > max_depth:
                    return
                
                prim_type = prim.GetTypeName()
                prim_name = prim.GetName()
                prim_path = str(prim.GetPath())
                
                # 標記特殊類型
                marker = ""
                if "joint" in prim_name.lower():
                    marker = " [JOINT]"
                elif "link" in prim_name.lower():
                    marker = " [LINK]"
                elif prim_type == "Mesh":
                    marker = " [MESH]"
                elif prim_type == "Xform":
                    marker = " [XFORM]"
                
                indent_str = "  " * indent
                if prim_type:
                    f.write(f"{indent_str}├─ {prim_name} <{prim_type}>{marker}\n")
                else:
                    f.write(f"{indent_str}├─ {prim_name}{marker}\n")
                
                # 遞歸子節點
                for child in prim.GetChildren():
                    print_tree(child, indent + 1, max_depth)
            
            print_tree(root_prim)
            
        else:
            f.write(f"❌ 找不到路徑: {env_0_path}\n")
        
        # 搜尋關鍵物件
        f.write("\n" + "="*80 + "\n")
        f.write("關鍵物件搜尋\n")
        f.write("="*80 + "\n\n")
        
        keywords = ["tm5", "robot", "arm", "patient", "table", "workstation", "base"]
        for keyword in keywords:
            f.write(f"\n🔍 搜尋關鍵字: '{keyword}'\n")
            f.write("-" * 40 + "\n")
            
            found_count = 0
            for prim in stage.Traverse():
                prim_path = str(prim.GetPath()).lower()
                prim_name = prim.GetName().lower()
                
                if keyword in prim_path or keyword in prim_name:
                    f.write(f"  {prim.GetPath()} [{prim.GetTypeName()}]\n")
                    found_count += 1
                    if found_count >= 20:  # 限制數量
                        f.write(f"  ... (更多 {keyword} 相關物件未列出)\n")
                        break
            
            if found_count == 0:
                f.write(f"  (未找到)\n")
    
    print(f"✅ 結構分析完成!")
    print(f"📄 結果已保存到: {output_file}")
    print(f"\n請查看該檔案了解場景結構")
    
    # 執行幾步仿真
    sim.reset()
    for _ in range(5):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.cfg.dt)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
