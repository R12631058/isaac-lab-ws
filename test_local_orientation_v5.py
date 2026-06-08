"""
測試 V5 Local 座標系方向對齊

這個腳本用於快速測試新的 orientation_command_error_local 函數是否正常工作。
"""

import torch
import math

# 模擬 quaternion 操作
def quat_inv(q):
    """計算四元數的逆"""
    # q = [w, x, y, z]
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

def quat_mul(q1, q2):
    """四元數乘法"""
    w1, x1, y1, z1 = q1[..., 0:1], q1[..., 1:2], q1[..., 2:3], q1[..., 3:4]
    w2, x2, y2, z2 = q2[..., 0:1], q2[..., 1:2], q2[..., 2:3], q2[..., 3:4]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return torch.cat([w, x, y, z], dim=-1)

def quat_error_magnitude(q1, q2):
    """計算兩個四元數之間的角度誤差 (弧度)"""
    # 計算相對旋轉: q_error = q1^-1 * q2
    q_error = quat_mul(quat_inv(q1), q2)
    # 角度 = 2 * arccos(|w|)
    w = torch.abs(q_error[..., 0])
    w = torch.clamp(w, -1.0, 1.0)
    angle = 2.0 * torch.acos(w)
    return angle

# 測試案例
print("=" * 80)
print("測試 Local vs Global 座標系方向對齊")
print("=" * 80)

# 假設數據 (batch_size = 2)
batch_size = 2

# 目標方向 (在機器人基座座標系中)
# pitch = 180° (朝下), roll = 0°, yaw = 0°
des_quat_b = torch.tensor([
    [0.0, 1.0, 0.0, 0.0],  # env 0: pitch=180°
    [0.0, 1.0, 0.0, 0.0],  # env 1: pitch=180°
])

# 機器人基座方向 (在世界座標系中)
# env 0: 基座無旋轉
# env 1: 基座繞 Z 軸旋轉 90°
root_quat_w = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],  # env 0: 無旋轉
    [0.707, 0.0, 0.0, 0.707],  # env 1: yaw=90°
])

# 當前 EE 方向 (在世界座標系中)
# 假設兩個環境的 EE 都完美對齊到 local 目標
# env 0: 朝下 (世界座標系)
# env 1: 朝下但繞 Z 軸旋轉 90° (因為基座旋轉了)
curr_quat_w_perfect = torch.tensor([
    [0.0, 1.0, 0.0, 0.0],  # env 0: pitch=180° (世界座標)
    [-0.5, 0.5, 0.5, 0.5],  # env 1: pitch=180° + yaw=90° (世界座標)
])

# 另一個測試: EE 方向錯誤 (偏離 45°)
curr_quat_w_wrong = torch.tensor([
    [0.924, 0.383, 0.0, 0.0],  # env 0: pitch=135° (偏離45°)
    [0.653, 0.271, 0.271, 0.653],  # env 1: pitch=135° + yaw=90°
])

print("\n測試 1: 完美對齊 (Local 座標系)")
print("-" * 80)

# Global 方法計算誤差
des_quat_w_global = quat_mul(root_quat_w, des_quat_b)
error_global = quat_error_magnitude(curr_quat_w_perfect, des_quat_w_global)
print(f"Global 方法誤差:")
print(f"  Env 0: {error_global[0]:.4f} rad ({math.degrees(error_global[0]):.2f}°)")
print(f"  Env 1: {error_global[1]:.4f} rad ({math.degrees(error_global[1]):.2f}°)")

# Local 方法計算誤差
root_quat_w_inv = quat_inv(root_quat_w)
curr_quat_b_local = quat_mul(root_quat_w_inv, curr_quat_w_perfect)
error_local = quat_error_magnitude(curr_quat_b_local, des_quat_b)
print(f"\nLocal 方法誤差:")
print(f"  Env 0: {error_local[0]:.4f} rad ({math.degrees(error_local[0]):.2f}°)")
print(f"  Env 1: {error_local[1]:.4f} rad ({math.degrees(error_local[1]):.2f}°)")

print("\n✅ 預期結果: Local 方法兩個環境都應該接近 0° (完美對齊)")
print("   Global 方法 Env 1 可能有誤差 (因為基座旋轉了)")

print("\n" + "=" * 80)
print("測試 2: 方向錯誤 (偏離 45°)")
print("-" * 80)

# Global 方法
error_global_wrong = quat_error_magnitude(curr_quat_w_wrong, des_quat_w_global)
print(f"Global 方法誤差:")
print(f"  Env 0: {error_global_wrong[0]:.4f} rad ({math.degrees(error_global_wrong[0]):.2f}°)")
print(f"  Env 1: {error_global_wrong[1]:.4f} rad ({math.degrees(error_global_wrong[1]):.2f}°)")

# Local 方法
curr_quat_b_wrong = quat_mul(root_quat_w_inv, curr_quat_w_wrong)
error_local_wrong = quat_error_magnitude(curr_quat_b_wrong, des_quat_b)
print(f"\nLocal 方法誤差:")
print(f"  Env 0: {error_local_wrong[0]:.4f} rad ({math.degrees(error_local_wrong[0]):.2f}°)")
print(f"  Env 1: {error_local_wrong[1]:.4f} rad ({math.degrees(error_local_wrong[1]):.2f}°)")

print("\n✅ 預期結果: 兩種方法都應該檢測到約 45° 的誤差")

print("\n" + "=" * 80)
print("結論")
print("=" * 80)
print("""
Local 座標系對齊的優勢:
1. 方向相對於機器人基座定義,更符合操作直覺
2. 即使機器人基座在世界中旋轉,目標方向仍保持一致
3. 更適合機械臂操作任務 (例如: 末端執行器相對於基座朝下)

Global 座標系對齊的問題:
1. 如果機器人基座旋轉,目標方向在世界座標系中也會改變
2. 可能導致不一致的行為 (同樣的基座相對姿態,不同的獎勵)

建議: 使用 orientation_command_error_local 進行 V5 訓練!
""")
