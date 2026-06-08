#!/usr/bin/env python3
"""Extract robot base coordinates from USD and environment origins."""

import json
from pathlib import Path

# First, try to load USD data without IsaacLab initialization
try:
    from pxr import Usd, UsdGeom
    
    usd_path = r"C:\Nick\surgery_team\surgery_team\USD\animation\isaaclab_multi_env.usd"
    stage = Usd.Stage.Open(usd_path)
    
    usd_base = None
    if stage:
        prim = stage.GetPrimAtPath("/Root/robotarm_base")
        if prim.IsValid():
            xform = UsdGeom.Xformable(prim)
            world_transform = xform.ComputeLocalToWorldTransform(0.0)
            t = world_transform.ExtractTranslation()
            usd_base = [float(t[0]), float(t[1]), float(t[2])]
except Exception as e:
    usd_base = {"error": str(e)}

# Now load IsaacLab environment data
try:
    import numpy as np
    from isaaclab.app import AppLauncher
    import argparse
    
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_extension_linkout_cone_orientation_cfg import (
        TM5ExtensionLinkOutFanOrientationEnvCfg
    )
    
    x_vals = np.linspace(0.1, 0.7, 7)
    y_vals = np.linspace(0.0, 0.3, 4)
    X, Y = np.meshgrid(x_vals, y_vals)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    num_envs = len(X_flat)
    
    env_cfg = TM5ExtensionLinkOutFanOrientationEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    origins = env.scene.env_origins.cpu().numpy()
    
    envs_data = []
    for i in range(num_envs):
        ox, oy, oz = float(origins[i][0]), float(origins[i][1]), float(origins[i][2])
        dx, dy = float(X_flat[i]), float(Y_flat[i])
        ax, ay, az = ox + dx, oy + dy, oz
        envs_data.append({
            "id": i,
            "origin": [ox, oy, oz],
            "offset": [dx, dy],
            "absolute_base": [ax, ay, az]
        })
    
    simulation_app.close()
    
except Exception as e:
    envs_data = {"error": str(e)}

# Write results to file
output = {
    "usd_base_position": usd_base,
    "environments": envs_data
}

with open(Path(__file__).parent / "coords_output.json", "w") as f:
    json.dump(output, f, indent=2)

print("SUCCESS: coords_output.json written")
