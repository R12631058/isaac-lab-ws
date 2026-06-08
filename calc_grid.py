#!/usr/bin/env python3
"""Calculate grid coordinates without full Isaac Lab environment."""

import json
import numpy as np
from pathlib import Path

# Define grid as per parallel_base_optimizer.py
x_vals = np.linspace(0.1, 0.7, 7)  # 7 points
y_vals = np.linspace(0.0, 0.3, 4)  # 4 points
X, Y = np.meshgrid(x_vals, y_vals)
X_flat = X.flatten()
Y_flat = Y.flatten()

# Assuming environment origins start at (0, 0, height)
# and default USD robot base is at local coordinates
# Based on parallel_base_optimizer.py:
# root_pos_w[i, 0] = env_origins[i, 0] + X_flat[i]
# root_pos_w[i, 1] = default_root_pos_w[i, 1] + Y_flat[i]

# For now, let's assume env_origins are approximately centered
# We need actual data, so let's compute theoretically

results = []
for i in range(len(X_flat)):
    x_offset = X_flat[i]
    y_offset = Y_flat[i]
    # env_origins typically start at (0, 0, z_height) or similar
    # The actual absolute positions would be:
    # absolute_x = env_origin_x + x_offset
    # absolute_y = env_origin_y + y_offset
    
    results.append({
        "env_id": i,
        "row": i // 7,
        "col": i % 7,
        "x_offset": float(x_offset),
        "y_offset": float(y_offset),
        "grid_x": float(x_vals[i % 7]),
        "grid_y": float(y_vals[i // 7])
    })

output = {
    "grid_definition": {
        "x_range": [float(x_vals[0]), float(x_vals[-1])],
        "y_range": [float(y_vals[0]), float(y_vals[-1])],
        "x_points": len(x_vals),
        "y_points": len(y_vals),
        "total_envs": len(X_flat)
    },
    "grid_points": results
}

output_path = Path(__file__).parent / "grid_coordinates.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"✓ Grid coordinates saved to {output_path}")
print(f"Total environments: {len(X_flat)}")
print(f"X range: {x_vals[0]:.2f} to {x_vals[-1]:.2f}")
print(f"Y range: {y_vals[0]:.2f} to {y_vals[-1]:.2f}")
