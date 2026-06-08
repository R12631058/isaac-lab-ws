from isaacsim import SimulationApp
app = SimulationApp({'headless': True})

import numpy as np
import math
from pxr import Gf, UsdGeom

def rot_matrix_to_quat_wxyz(rot):
    m = np.array(rot, dtype=float)
    trace = np.trace(m)
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)

def look_at_quaternion_world(ndi_pos, target_pos):
    forward = target_pos - ndi_pos
    forward_len = np.linalg.norm(forward)
    if forward_len < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    forward = forward / forward_len

    up_hint = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, up_hint)) > 0.999:
        up_hint = np.array([0.0, 1.0, 0.0])

    right = np.cross(forward, up_hint)
    right = right / np.linalg.norm(right)

    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    new_right = up
    new_up = -right

    R = np.column_stack([new_right, new_up, -forward])
    return rot_matrix_to_quat_wxyz(R)

with open("scratch/look_at_results.txt", "w") as f:
    f.write("Testing look_at math...\n")
    try:
        ndi_pos = np.array([0.5, 0.5, 1.5])
        target_pos = np.array([0.0, 0.0, 0.9])
        quat_world = look_at_quaternion_world(ndi_pos, target_pos)
        f.write(f"quat_world: {quat_world}\n")
        
        # parent transform factorization
        parent_matrix = Gf.Matrix4d(1.0)
        gf_quat_world = Gf.Quaternion(float(quat_world[0]), Gf.Vec3d(float(quat_world[1]), float(quat_world[2]), float(quat_world[3])))
        q_parent = parent_matrix.ExtractRotation().GetQuaternion()
        
        f.write(f"q_parent type: {type(q_parent)}\n")
        f.write(f"gf_quat_world type: {type(gf_quat_world)}\n")
        
        q_parent_inv = q_parent.GetInverse()
        f.write(f"q_parent_inv: {q_parent_inv}\n")
        
        q_local = q_parent_inv * gf_quat_world
        f.write(f"q_local: {q_local}\n")
        
        # Test constructing Gf.Quatd from q_local
        gf_quat = Gf.Quatd(float(q_local.GetReal()), Gf.Vec3d(q_local.GetImaginary()))
        f.write(f"gf_quat: {gf_quat}\n")
        
    except Exception as e:
        f.write(f"Error occurred: {e}\n")

app.close()
