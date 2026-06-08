"""
檢查 End Effector 和 Needle Tip 的姿態
用於確定扇形角度生成的"前後"和"左右"方向
"""

from pxr import Usd, UsdGeom, Gf
import math
import numpy as np

def quat_to_euler(quat):
    """四元數轉歐拉角 (度)"""
    w, x, y, z = quat
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

def matrix_to_quat(matrix):
    """從 4x4 變換矩陣提取四元數"""
    # 提取旋轉矩陣 (3x3)
    rot = Gf.Matrix3d(
        matrix[0][0], matrix[0][1], matrix[0][2],
        matrix[1][0], matrix[1][1], matrix[1][2],
        matrix[2][0], matrix[2][1], matrix[2][2]
    )
    quat = rot.ExtractRotation().GetQuat()
    return (quat.GetReal(), quat.GetImaginary()[0], quat.GetImaginary()[1], quat.GetImaginary()[2])

def get_direction_vectors(matrix):
    """從變換矩陣提取方向向量（正規化）"""
    # X 軸 (通常是"右"方向)
    x_axis = np.array([matrix[0][0], matrix[1][0], matrix[2][0]])
    x_axis = x_axis / np.linalg.norm(x_axis) if np.linalg.norm(x_axis) > 0 else x_axis
    # Y 軸 (通常是"前"方向)  
    y_axis = np.array([matrix[0][1], matrix[1][1], matrix[2][1]])
    y_axis = y_axis / np.linalg.norm(y_axis) if np.linalg.norm(y_axis) > 0 else y_axis
    # Z 軸 (通常是 normal / "上"方向)
    z_axis = np.array([matrix[0][2], matrix[1][2], matrix[2][2]])
    z_axis = z_axis / np.linalg.norm(z_axis) if np.linalg.norm(z_axis) > 0 else z_axis
    
    return x_axis, y_axis, z_axis

# 載入 USD
usd_path = "C:/Nick/surgery_team/surgery_team/USD/isaaclab/extension_link_out.usd"
stage = Usd.Stage.Open(usd_path)

print("=" * 70)
print("Extension Link Out - End Effector 方向分析 (正規化版本)")
print("=" * 70)

# 要分析的 bodies
bodies_to_check = [
    'link_6',      # 機器人最後一個 link
    'flange',      # 法蘭 (標準 EE 參考點)
    'End_needle',  # 針具末端
    'needle_tip',  # 針尖
]

results = {}

for body_name in bodies_to_check:
    for prim in stage.Traverse():
        if prim.GetName() == body_name:
            xform = UsdGeom.Xformable(prim)
            if xform:
                try:
                    world_xform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    
                    # 提取位置
                    pos = (world_xform[3][0], world_xform[3][1], world_xform[3][2])
                    
                    # 提取四元數
                    quat = matrix_to_quat(world_xform)
                    
                    # 提取方向向量（正規化）
                    x_axis, y_axis, z_axis = get_direction_vectors(world_xform)
                    
                    # 歐拉角
                    roll, pitch, yaw = quat_to_euler(quat)
                    
                    results[body_name] = {
                        'pos': pos,
                        'quat': quat,
                        'euler': (roll, pitch, yaw),
                        'x_axis': x_axis,
                        'y_axis': y_axis,
                        'z_axis': z_axis,
                    }
                    
                    print(f"\n{'='*50}")
                    print(f"Body: {body_name}")
                    print(f"{'='*50}")
                    print(f"位置 (World): ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
                    print(f"四元數 (w,x,y,z): ({quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f})")
                    print(f"歐拉角 (Roll, Pitch, Yaw): ({roll:.2f}°, {pitch:.2f}°, {yaw:.2f}°)")
                    print(f"\n方向向量 (World Frame) - 正規化:")
                    print(f"  X軸 (Local Right):    [{x_axis[0]:7.4f}, {x_axis[1]:7.4f}, {x_axis[2]:7.4f}]")
                    print(f"  Y軸 (Local Forward):  [{y_axis[0]:7.4f}, {y_axis[1]:7.4f}, {y_axis[2]:7.4f}]")
                    print(f"  Z軸 (Local Normal):   [{z_axis[0]:7.4f}, {z_axis[1]:7.4f}, {z_axis[2]:7.4f}]")
                    
                except Exception as e:
                    print(f"Error processing {body_name}: {e}")
            break

# 詳細分析
print("\n" + "=" * 70)
print("方向關係分析")
print("=" * 70)

if 'flange' in results and 'needle_tip' in results:
    fl = results['flange']
    nt = results['needle_tip']
    
    print("\n【Flange (法蘭) 方向】")
    print(f"  Z軸 (Normal): [{fl['z_axis'][0]:.4f}, {fl['z_axis'][1]:.4f}, {fl['z_axis'][2]:.4f}]")
    print(f"  → 法蘭 Normal 指向 +Y 方向（世界座標），即朝向機器人「前方」")
    
    print("\n【Needle Tip (針尖) 方向】")
    print(f"  Z軸 (Normal): [{nt['z_axis'][0]:.4f}, {nt['z_axis'][1]:.4f}, {nt['z_axis'][2]:.4f}]")
    
    # 分析針尖指向
    nt_z = nt['z_axis']
    if abs(nt_z[0]) > 0.9:
        if nt_z[0] > 0:
            print(f"  → 針尖 Normal 指向 +X 方向（世界座標）")
        else:
            print(f"  → 針尖 Normal 指向 -X 方向（世界座標）")
    elif abs(nt_z[1]) > 0.9:
        if nt_z[1] > 0:
            print(f"  → 針尖 Normal 指向 +Y 方向（世界座標）")
        else:
            print(f"  → 針尖 Normal 指向 -Y 方向（世界座標）")
    elif abs(nt_z[2]) > 0.9:
        if nt_z[2] > 0:
            print(f"  → 針尖 Normal 指向 +Z 方向（世界座標，向上）")
        else:
            print(f"  → 針尖 Normal 指向 -Z 方向（世界座標，向下）")
    else:
        print(f"  → 針尖 Normal 指向混合方向")
    
    print("\n【Link 6 旋轉軸】")
    if 'link_6' in results:
        l6 = results['link_6']
        print(f"  Z軸 (旋轉軸): [{l6['z_axis'][0]:.4f}, {l6['z_axis'][1]:.4f}, {l6['z_axis'][2]:.4f}]")
        print(f"  → Link 6 繞 +Y 軸旋轉（世界座標）")
        print(f"  → 這表示 Joint 6 的旋轉會改變針尖在 XZ 平面的投影方向")

print("\n" + "=" * 70)
print("扇形角度建議")
print("=" * 70)
print("""
根據分析結果:

1. 【座標系統定義】
   - 世界座標: X=右, Y=前, Z=上
   - Flange Z軸指向 +Y (朝前)
   - Needle Tip Z軸指向 +X (大致)
   
2. 【Joint 6 旋轉效果】
   - Link 6 的旋轉軸是 +Y（世界座標）
   - 旋轉 Joint 6 會讓針尖在 XZ 平面內擺動
   - 這就是你說的「左右」傾斜！
   
3. 【機械限制】
   - 「左右」傾斜 = 繞 Y 軸旋轉 = Joint 6 可控制
   - 「前後」傾斜 = 繞 X 軸旋轉 = 受下針機構限制
   
4. 【扇形範圍建議】
   - 只允許繞 Flange 的 Z 軸（即世界 Y 軸）旋轉
   - 這對應於只改變 Yaw 角度
   - 範圍: Yaw ±30° (或你希望的角度)
   - Roll 和 Pitch 保持固定
""")

# 具體計算
print("\n" + "=" * 70)
print("扇形 Orientation 生成參數")
print("=" * 70)

if 'needle_tip' in results:
    nt = results['needle_tip']
    print(f"\n針尖初始四元數: ({nt['quat'][0]:.4f}, {nt['quat'][1]:.4f}, {nt['quat'][2]:.4f}, {nt['quat'][3]:.4f})")
    print(f"針尖初始歐拉角: Roll={nt['euler'][0]:.2f}°, Pitch={nt['euler'][1]:.2f}°, Yaw={nt['euler'][2]:.2f}°")
    
    print(f"""
【扇形生成方案】
基準姿態:
  - Roll = {nt['euler'][0]:.2f}° (固定)
  - Pitch = {nt['euler'][1]:.2f}° (固定)  
  - Yaw = {nt['euler'][2]:.2f}° (可變)
  
扇形範圍:
  - Roll: 固定在 {nt['euler'][0]:.2f}°
  - Pitch: 固定在 {nt['euler'][1]:.2f}°
  - Yaw: {nt['euler'][2]:.2f}° ± 30° = [{nt['euler'][2] - 30:.2f}°, {nt['euler'][2] + 30:.2f}°]
""")
