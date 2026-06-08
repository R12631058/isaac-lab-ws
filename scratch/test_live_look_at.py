from isaacsim import SimulationApp
app = SimulationApp({'headless': True})

import sys
import os
from pxr import Usd, UsdGeom
import omni

with open("scratch/live_look_at_results.txt", "w") as f:
    usd_path = r"C:\Nick\surgery_team\surgery_team\USD\animation\isaaclab_multi_env.usd"
    f.write(f"USD path exists: {os.path.exists(usd_path)}\n")
    
    try:
        context = omni.usd.get_context()
        context.open_stage(usd_path)
        for _ in range(30):
            app.update()
        
        stage = context.get_stage()
        f.write("USD stage loaded.\n")
        
        # Traverse stage and find all prim paths containing NDI
        ndi_paths = []
        for prim in stage.Traverse():
            path_str = str(prim.GetPath())
            if "NDI" in path_str or "ndi" in path_str.lower():
                ndi_paths.append(path_str)
                
        f.write(f"Found {len(ndi_paths)} prim paths containing NDI:\n")
        for p in ndi_paths[:50]:
            f.write(f"  {p}\n")
            
    except Exception as e:
        f.write(f"Exception raised: {e}\n")

app.close()
