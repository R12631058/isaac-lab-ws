from isaacsim import SimulationApp
app = SimulationApp({'headless': True})

import sys
import os
import numpy as np
import math
from pxr import Gf, Usd, UsdGeom
import omni

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

def set_prim_transform_orient(stage, prim_path, translation, q_local):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return False
    xform = UsdGeom.Xformable(prim)
    
    print("Clearing XformOpOrder...")
    xform.ClearXformOpOrder()
    
    print("Adding TranslateOp...")
    translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(*translation))
    
    print("Adding OrientOp...")
    orient_op = xform.AddOrientOp()
    
    # Check if the attribute type is float or double, and convert accordingly
    attr = orient_op.GetAttr()
    typeName = str(attr.GetTypeName())
    print(f"Attribute typeName: {typeName}")
    
    if "quatf" in typeName.lower():
        gf_quat = Gf.Quatf(float(q_local.GetReal()), Gf.Vec3f(float(q_local.GetImaginary()[0]), float(q_local.GetImaginary()[1]), float(q_local.GetImaginary()[2])))
    else:
        gf_quat = Gf.Quatd(float(q_local.GetReal()), Gf.Vec3d(float(q_local.GetImaginary()[0]), float(q_local.GetImaginary()[1]), float(q_local.GetImaginary()[2])))
    
    print(f"Setting OrientOp value: {gf_quat}...")
    orient_op.Set(gf_quat)
    print("Set completed.")
    return True

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
        
        ndi_path = "/Root/NDI"
        ndi_prim = stage.GetPrimAtPath(ndi_path)
        f.write(f"NDI prim path: {ndi_path}, valid: {ndi_prim.IsValid()}\n")
        
        if ndi_prim.IsValid():
            xform = UsdGeom.Xformable(ndi_prim)
            transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            ndi_pos = np.array(transform.ExtractTranslation())
            f.write(f"NDI world pos: {ndi_pos}\n")
            
            centroid = np.array([0.0, 0.0, 0.9])
            quat_world = look_at_quaternion_world(ndi_pos, centroid)
            f.write(f"quat_world: {quat_world}\n")
            
            parent_path = ndi_path.rsplit("/", 1)[0]
            parent_prim = stage.GetPrimAtPath(parent_path)
            parent_matrix = Gf.Matrix4d(1.0)
            if parent_prim.IsValid():
                parent_xform = UsdGeom.Xformable(parent_prim)
                parent_matrix = parent_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            f.write("Parent matrix computed.\n")
            
            local_pos = None
            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    val = op.Get()
                    if val is not None:
                        local_pos = np.array([val[0], val[1], val[2]])
                        break
            if local_pos is None:
                local_pos = np.array(parent_matrix.GetInverse().Transform(Gf.Vec3d(*ndi_pos)))
            f.write(f"local_pos: {local_pos}\n")
            
            gf_quat_world = Gf.Quaternion(float(quat_world[0]), Gf.Vec3d(float(quat_world[1]), float(quat_world[2]), float(quat_world[3])))
            q_parent = parent_matrix.ExtractRotation().GetQuaternion()
            q_local = q_parent.GetInverse() * gf_quat_world
            f.write(f"q_local: {q_local}\n")
            
            f.write("Applying transform...\n")
            success = set_prim_transform_orient(stage, ndi_path, local_pos, q_local)
            f.write(f"Transform application success: {success}\n")
            
    except Exception as e:
        f.write(f"Exception raised: {e}\n")

app.close()
