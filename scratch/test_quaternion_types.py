from isaacsim import SimulationApp
app = SimulationApp({'headless': True})

import sys
from pxr import Gf

with open("scratch/quaternion_type_results.txt", "w") as f:
    f.write("Testing quaternion types and multiplication mixtures...\n")
    
    # 1. Check parent_matrix.ExtractRotation().GetQuaternion()
    m = Gf.Matrix4d(1.0)
    rot = m.ExtractRotation()
    q_extracted = rot.GetQuaternion()
    f.write(f"Type of q_extracted: {type(q_extracted)}\n")
    
    # 2. Check if we can multiply Gf.Quatd and Gf.Quaternion
    q_gf = Gf.Quaternion(1, Gf.Vec3d(0))
    
    try:
        res = q_extracted * q_gf
        f.write(f"q_extracted * q_gf success: {res}\n")
    except Exception as e:
        f.write(f"q_extracted * q_gf failed: {e}\n")
        
    try:
        res = q_gf * q_extracted
        f.write(f"q_gf * q_extracted success: {res}\n")
    except Exception as e:
        f.write(f"q_gf * q_extracted failed: {e}\n")

    # 3. Check Gf.Rotation multiplication
    try:
        r1 = Gf.Rotation(Gf.Vec3d(1,0,0), 30)
        r2 = Gf.Rotation(Gf.Vec3d(0,1,0), 45)
        r_mul = r1 * r2
        f.write(f"Rotation multiplication success: {r_mul.GetQuaternion()}\n")
    except Exception as e:
        f.write(f"Rotation multiplication failed: {e}\n")

app.close()
