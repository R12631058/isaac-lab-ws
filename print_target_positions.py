"""
Print target positions by creating a minimal environment
"""

import gymnasium as gym
import torch
import omni.isaac.lab_tasks

# Create environment
env = gym.make("Isaac-Reach-TM5-NeedleTip-v0", num_envs=1)

# Get the scene
scene = env.unwrapped.scene

# Print available prims in scene
print("\n=== Scene Structure ===")
print(f"Scene prims: {list(scene.keys())}")

# Check if we have target_pose in the scene
if hasattr(env.unwrapped, 'scene'):
    stage = env.unwrapped.sim.stage
    
    print("\n=== Target Position Coordinates ===\n")
    
    # Find all target_pose paths
    from pxr import UsdGeom
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if "/target_pose/path_" in prim_path and prim.IsA(UsdGeom.Xformable):
            # Get the world transform
            xformable = UsdGeom.Xformable(prim)
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            
            # Extract translation
            translation = world_transform.ExtractTranslation()
            
            # Get the path name
            path_name = prim_path.split("/")[-1]
            
            print(f"{path_name}: x={translation[0]:.3f}, y={translation[1]:.3f}, z={translation[2]:.3f}")

print("\n")

# Clean up
env.close()
