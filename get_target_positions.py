"""
Get target positions from USD file
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.app import AppLauncher

# Create app launcher
app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

# Import after Isaac Sim is loaded
import omni.usd
from pxr import Usd, UsdGeom

# Open the USD file
stage = omni.usd.get_context().open_stage("source/isaaclab_assets/assets/surgeryroom_ctgantry_isaaclab.usd")

print("\n=== Target Position Coordinates ===\n")

# Find all target_pose paths
for prim in stage.Traverse():
    prim_path = str(prim.GetPath())
    if "/target_pose/path_" in prim_path and prim.IsA(UsdGeom.Xformable):
        # Get the world transform
        xformable = UsdGeom.Xformable(prim)
        world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        
        # Extract translation
        translation = world_transform.ExtractTranslation()
        
        # Get the path name
        path_name = prim_path.split("/")[-1]
        
        print(f"{path_name}: x={translation[0]:.3f}, y={translation[1]:.3f}, z={translation[2]:.3f}")

print("\n")

# Cleanup
simulation_app.close()
