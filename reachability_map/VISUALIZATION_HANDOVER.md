# Inverse Reachability Map 3D Visualization Handover

## Overview
我們已經透過 RL Evaluation Script (`multi_position_eval.py`) 掃描了機器人底座在不同位置針對「固定世界目標 (Target)」的抓取成功率。結果已儲存為 JSON。
現在目標是在 **Isaac Sim 4.5.0** 中讀取此 JSON，並在場景中生成 3D Marker (Spheres) 來視覺化這些數據（類似 Reuleaux 的 Inverse Reachability Map）。

## Input Data
- **File**: `multi_position_results.json`
- **Path Example**: `C:\Users\RMML\IsaacLab\scripts\reachability_map\multi_position_results.json`
- **Format**:
```json
{
  "target_pose": [-0.32, -0.80, 1.08],  // Target Center (World)
  "per_position_results": [
    {
      "x": 0.30,          // Robot Base X (World)
      "y": -0.1741,       // Robot Base Y (World)
      "success_rate": 100.0,
      "mean_return": 444.9
    },
    ...
  ]
}
```

## Visualization Requirements
在 Isaac Sim 中執行一個 Python Script (Extension 或 Standalone)，邏輯如下：

1.  **Read JSON**: 載入上述 JSON 檔案。
2.  **Draw Target**: 在 `target_pose` 位置生成一個 **Blue Cube** 或 Star。
3.  **Draw Reachability Map**: 遍歷 `per_position_results`：
    *   在 `(x, y, 0.0)` 位置生成一個 **Sphere** (半徑約 0.02m - 0.05m)。
    *   **Color Coding**:
        *   **Green (0, 1, 0)**: Success Rate >= 95%
        *   **Red (1, 0, 0)**: Success Rate <= 5%
        *   **Yellow (1, 1, 0)**: 其他 (Gradient)

## Sample Code (USD / Omni API)
以下是給 Isaac Sim Agent 的參考實作 snippet：

```python
import json
import numpy as np
from pxr import Usd, UsdGeom, Gf
import omni.usd

def create_marker(stage, path, position, color, size=0.03):
    """Crates a colored sphere marker at specific position."""
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.GetRadiusAttr().Set(size)
    
    # Set Translate
    xform = UsdGeom.Xformable(sphere)
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    
    # Set Color
    sphere.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    return sphere

def visualize_reachability(json_path):
    # 1. Load Data
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    ctx = omni.usd.get_context()
    stage = ctx.get_stage()
    
    # Create Root Group
    root_path = "/World/ReachabilityMap"
    UsdGeom.Xform.Define(stage, root_path)
    
    # 2. Draw Target
    if "target_pose" in data:
        t_pos = data["target_pose"]
        # Blue Cube for Target
        cube = UsdGeom.Cube.Define(stage, f"{root_path}/Target")
        cube.GetSizeAttr().Set(0.05)
        UsdGeom.Xformable(cube).AddTranslateOp().Set(Gf.Vec3d(*t_pos))
        cube.GetDisplayColorAttr().Set([Gf.Vec3f(0, 0, 1)]) # Blue
        
    # 3. Draw Inverse Reachability Spheres
    results = data.get("per_position_results", [])
    
    for i, res in enumerate(results):
        x, y = res["x"], res["y"]
        pos = (x, y, 0.05) # Lift slightly off ground
        rate = res["success_rate"]
        
        # Determine Color
        if rate >= 95.0:
            color = (0, 1, 0) # Green
        elif rate <= 5.0:
            color = (1, 0, 0) # Red
        else:
            color = (1, 1, 0) # Yellow (Intermediate)
            
        prim_path = f"{root_path}/Marker_{i}"
        create_marker(stage, prim_path, pos, color)
        
    print(f"[Map] Generated {len(results)} markers from {json_path}")

# Example Usage in Script Editor:
# visualize_reachability(r"C:\Users\RMML\IsaacLab\scripts\reachability_map\multi_position_results.json")
```

## Next Steps for Isaac Sim Agent
1.  Open Isaac Sim 4.5.0.
2.  Open `Script Editor` (Window -> Script Editor).
3.  Paste the **Sample Code** above.
4.  Update the `json_path` to the actual location.
5.  Run cleanly to see the 3D map.
