#!/usr/bin/env python3
"""
簡單的座標轉換工具
根據已知的機器人基座位置計算相對座標
"""

# 從 check_robot_position.py 的輸出得知:
# 機器人基座世界位置: (1.3885, -0.2414, 0.9492)
ROBOT_BASE_WORLD_POS = (1.3885, -0.2414, 0.9492)

# 您確認可達的世界座標
USER_WORLD_CENTER = (1.13, -0.65, 1.29)

# 半徑
RADIUS = 0.2

print("=" * 80)
print("🔍 相對座標計算工具")
print("=" * 80)
print()

print("📍 機器人基座世界位置 (從 check_robot_position.py):")
print(f"   X = {ROBOT_BASE_WORLD_POS[0]:.4f} 公尺")
print(f"   Y = {ROBOT_BASE_WORLD_POS[1]:.4f} 公尺")
print(f"   Z = {ROBOT_BASE_WORLD_POS[2]:.4f} 公尺")
print()

print("🎯 您確認可達的世界座標:")
print(f"   X = {USER_WORLD_CENTER[0]:.2f} 公尺")
print(f"   Y = {USER_WORLD_CENTER[1]:.2f} 公尺")
print(f"   Z = {USER_WORLD_CENTER[2]:.2f} 公尺")
print(f"   半徑 = {RADIUS:.2f} 公尺")
print()

# 計算相對座標
relative_x = USER_WORLD_CENTER[0] - ROBOT_BASE_WORLD_POS[0]
relative_y = USER_WORLD_CENTER[1] - ROBOT_BASE_WORLD_POS[1]
relative_z = USER_WORLD_CENTER[2] - ROBOT_BASE_WORLD_POS[2]

print("📐 計算公式:")
print(f"   相對X = 世界X - 基座X = {USER_WORLD_CENTER[0]} - {ROBOT_BASE_WORLD_POS[0]} = {relative_x:.4f}")
print(f"   相對Y = 世界Y - 基座Y = {USER_WORLD_CENTER[1]} - {ROBOT_BASE_WORLD_POS[1]} = {relative_y:.4f}")
print(f"   相對Z = 世界Z - 基座Z = {USER_WORLD_CENTER[2]} - {ROBOT_BASE_WORLD_POS[2]} = {relative_z:.4f}")
print()

print("✅ 結果 - 相對於機器人基座的座標:")
print(f"   相對X = {relative_x:.4f} 公尺 ({relative_x*100:.2f} 公分)")
print(f"   相對Y = {relative_y:.4f} 公尺 ({relative_y*100:.2f} 公分)")
print(f"   相對Z = {relative_z:.4f} 公尺 ({relative_z*100:.2f} 公分)")
print()

print("=" * 80)
print("💡 配置代碼建議")
print("=" * 80)
print()
print("請在 tm5_reach_surgery_room_cfg.py 中使用以下值:")
print()
print("```python")
print("# ========== 目標位置配置 ==========")
print("# 使用相對座標 (相對於機器人基座)")
print(f"center_x = {relative_x:.2f}  # 機器人{'後方' if relative_x < 0 else '前方'} {abs(relative_x*100):.0f} 公分")
print(f"center_y = {relative_y:.2f}  # 機器人{'右側' if relative_y < 0 else '左側'} {abs(relative_y*100):.0f} 公分")
print(f"center_z = {relative_z:.2f}   # 機器人{'下方' if relative_z < 0 else '上方'} {abs(relative_z*100):.0f} 公分")
print(f"radius = {RADIUS}")
print()
print("# 計算範圍")
print(f"min_x = center_x - radius  # {relative_x - RADIUS:.2f}")
print(f"max_x = center_x + radius  # {relative_x + RADIUS:.2f}")
print(f"min_y = center_y - radius  # {relative_y - RADIUS:.2f}")
print(f"max_y = center_y + radius  # {relative_y + RADIUS:.2f}")
print(f"min_z = center_z - radius  # {relative_z - RADIUS:.2f}")
print(f"max_z = center_z + radius  # {relative_z + RADIUS:.2f}")
print("```")
print()

print("=" * 80)
print("📝 說明")
print("=" * 80)
print()
print("✅ 為什麼要使用相對座標?")
print("   - Isaac Lab 的多環境訓練中,每個環境的機器人位於不同的世界位置")
print("   - env_0 的機器人在世界 (1.39, -0.24, 0.95)")
print("   - env_1 的機器人在世界 (3.89, -0.24, 0.95)  [env_spacing = 2.5m]")
print("   - env_2 的機器人在世界 (6.39, -0.24, 0.95)")
print("   - 等等...")
print()
print("❌ 使用世界座標會發生什麼?")
print("   - 如果目標設在世界座標 (1.13, -0.65, 1.29)")
print("   - env_0: 目標相對於機器人 = (1.13-1.39, ...) = (-0.26, ...) ✅ 正確")
print("   - env_1: 目標相對於機器人 = (1.13-3.89, ...) = (-2.76, ...) ❌ 2.5m 外!")
print("   - env_2: 目標相對於機器人 = (1.13-6.39, ...) = (-5.26, ...) ❌ 5m 外!")
print()
print("✅ 使用相對座標後:")
print(f"   - 目標設在相對座標 ({relative_x:.2f}, {relative_y:.2f}, {relative_z:.2f})")
print("   - env_0: 目標世界位置 = (1.39, -0.24, 0.95) + 相對 = (1.13, -0.65, 1.29) ✅")
print("   - env_1: 目標世界位置 = (3.89, -0.24, 0.95) + 相對 = (3.63, -0.65, 1.29) ✅")
print("   - env_2: 目標世界位置 = (6.39, -0.24, 0.95) + 相對 = (6.13, -0.65, 1.29) ✅")
print("   - 所有環境的目標都在機器人可達範圍內!")
print()

print("=" * 80)
print("🎯 下一步操作")
print("=" * 80)
print()
print("1. 複製上方的配置代碼")
print("2. 更新 source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/tm5/")
print("   tm5_reach_surgery_room_cfg.py 文件的第 95-114 行")
print("3. 重新測試多環境")
print()
