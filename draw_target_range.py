"""
繪製目標範圍的線框可視化
世界座標: 起點 (1.184, -1.195, 1.557)
         終點 (1.584, -1.495, 1.657)
尺寸: 0.4 x 0.3 x 0.1 m^3
"""

import argparse
from omni.isaac.lab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import carb

# 創建環境
env = gym.make("Isaac-Reach-TM5-NeedleTip-v0", num_envs=1)

# 定義長方體的 8 個頂點 (世界座標)
p1 = (1.184, -1.195, 1.557)  # 起點
p2 = (1.584, -1.195, 1.557)  # x+
p3 = (1.184, -1.495, 1.557)  # y-
p4 = (1.584, -1.495, 1.557)  # x+, y-
p5 = (1.184, -1.195, 1.657)  # z+
p6 = (1.584, -1.195, 1.657)  # x+, z+
p7 = (1.184, -1.495, 1.657)  # y-, z+
p8 = (1.584, -1.495, 1.657)  # 終點: x+, y-, z+

# 綠色線條
color = carb.Float4(0.0, 1.0, 0.0, 1.0)
size = 2.0

print("\n繪製目標範圍線框...")
print(f"起點: {p1}")
print(f"終點: {p8}")
print(f"尺寸: 0.4 x 0.3 x 0.1 m³")

# 獲取 debug draw 接口
from omni.isaac.debug_draw import _debug_draw
draw = _debug_draw.acquire_debug_draw_interface()

# 持續繪製線框
try:
    step_count = 0
    while True:
        # 每隔一段時間重繪線框 (debug draw 的線條會消失)
        if step_count % 10 == 0:
            draw.clear_lines()
            # 底面 (z=1.557)
            draw.draw_line(p1, p2, color, size)
            draw.draw_line(p1, p3, color, size)
            draw.draw_line(p2, p4, color, size)
            draw.draw_line(p3, p4, color, size)
            # 頂面 (z=1.657)
            draw.draw_line(p5, p6, color, size)
            draw.draw_line(p5, p7, color, size)
            draw.draw_line(p6, p8, color, size)
            draw.draw_line(p7, p8, color, size)
            # 垂直邊
            draw.draw_line(p1, p5, color, size)
            draw.draw_line(p2, p6, color, size)
            draw.draw_line(p3, p7, color, size)
            draw.draw_line(p4, p8, color, size)
        
        env.step(env.action_space.sample())
        step_count += 1
        
except KeyboardInterrupt:
    print("\n關閉環境...")
    env.close()
    simulation_app.close()
