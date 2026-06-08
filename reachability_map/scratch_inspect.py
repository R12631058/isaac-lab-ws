import sys
import os

from isaacsim import SimulationApp
sim = SimulationApp({"headless": True})

import omni.usd
from pxr import Usd, UsdGeom

usd_path = r"C:\Nick\surgery_team\surgery_team\USD\animation\isaaclab_multi_env.usd"
print(f"Loading stage: {usd_path}")
sys.stdout.flush()

opened = omni.usd.get_context().open_stage(usd_path)
stage = omni.usd.get_context().get_stage()

prim = stage.GetPrimAtPath("/Root/NDI")
if prim.IsValid():
    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    print("Ordered XformOps:")
    for op in ops:
        print(f"  Op Name: {op.GetOpName()}, Op Type: {op.GetOpType()}, Value: {op.Get()}")
    sys.stdout.flush()
    
    attr = prim.GetAttribute("xformOpOrder")
    if attr.IsValid():
        print(f"xformOpOrder value: {attr.Get()}")
    else:
        print("xformOpOrder not valid")
    sys.stdout.flush()
else:
    print("/Root/NDI is invalid")
    sys.stdout.flush()

sim.close()
print("Done")
sys.stdout.flush()
