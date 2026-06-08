# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates debug line drawing with floating cubes that move directionally.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/isaaclab_ws/simtolab.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on debug line drawing with floating cubes.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import numpy as np

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext

# Debug draw imports
from isaacsim.util.debug_draw import _debug_draw
import omni.usd


def design_scene():
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Create two cube objects - 使用普通的動態物體，但通過程式控制
    cube1_cfg = RigidObjectCfg(
        prim_path="/World/cube1",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,  # 保持重力，我們會用力來抵消
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.5, 1.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 3.0)),
    )
    cube1_object = RigidObject(cfg=cube1_cfg)

    cube2_cfg = RigidObjectCfg(
        prim_path="/World/cube2",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.5), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 3.0)),
    )
    cube2_object = RigidObject(cfg=cube2_cfg)

    # return the scene information
    scene_entities = {"cube1": cube1_object, "cube2": cube2_object}
    return scene_entities


def sample_directional_positions(base_positions, direction, distance_range, lateral_range, device):
    """沿著指定方向進行隨機採樣"""
    num_objects = base_positions.shape[0]
    
    # 正規化方向向量
    direction = direction / torch.norm(direction)
    
    # 計算垂直於主方向的兩個正交向量（修正 torch.cross 警告）
    if abs(direction[0]) < 0.9:
        temp = torch.tensor([1.0, 0.0, 0.0], device=device)
    else:
        temp = torch.tensor([0.0, 1.0, 0.0], device=device)
    
    lateral1 = torch.linalg.cross(direction, temp)
    lateral1 = lateral1 / torch.norm(lateral1)
    
    lateral2 = torch.linalg.cross(direction, lateral1)
    lateral2 = lateral2 / torch.norm(lateral2)
    
    # 生成隨機偏移
    main_distances = torch.rand(num_objects, device=device) * (distance_range[1] - distance_range[0]) + distance_range[0]
    lateral1_offsets = (torch.rand(num_objects, device=device) - 0.5) * 2 * (lateral_range[1] - lateral_range[0]) + lateral_range[0]
    lateral2_offsets = (torch.rand(num_objects, device=device) - 0.5) * 2 * (lateral_range[1] - lateral_range[0]) + lateral_range[0]
    
    offsets = (main_distances.unsqueeze(1) * direction.unsqueeze(0) + 
               lateral1_offsets.unsqueeze(1) * lateral1.unsqueeze(0) + 
               lateral2_offsets.unsqueeze(1) * lateral2.unsqueeze(0))
    
    return offsets


def get_scene_object_positions():
    """Get positions of all objects in the scene"""
    stage = omni.usd.get_context().get_stage()
    object_positions = []
    
    for prim in stage.Traverse():
        if prim.IsValid() and prim.GetTypeName() in ['Cube', 'Sphere', 'Cylinder', 'Mesh', 'Xform']:
            try:
                xform = omni.usd.get_world_transform_matrix(prim)
                position = xform.ExtractTranslation()
                object_positions.append({
                    'path': prim.GetPath(),
                    'position': [position[0], position[1], position[2]]
                })
            except Exception as e:
                print(f"Could not get position for {prim.GetPath()}: {e}")
    
    return object_positions

def update_lines(draw, cube1_object, cube2_object):
    """Update debug lines to all objects in the scene using Isaac Lab data."""
    # 直接從 Isaac Lab 物件獲取實時位置
    cube1_pos = cube1_object.data.root_state_w[0, :3].cpu().numpy()
    cube2_pos = cube2_object.data.root_state_w[0, :3].cpu().numpy()
    
    line_starts = []
    line_ends = []
    colors = []
    sizes = []
    
    start_point = [10, 0, 10]
    
    # 為每個方塊創建線條
    objects_data = [
        {'position': cube1_pos.tolist(), 'name': 'cube1'},
        {'position': cube2_pos.tolist(), 'name': 'cube2'}
    ]
    
    color_options = [
        [1, 0, 0, 1],    # Red for cube1
        [0, 1, 0, 1],    # Green for cube2
        [0, 0, 1, 1],    # Blue
        [1, 1, 0, 1],    # Yellow
        [1, 0, 1, 1],    # Magenta
        [0, 1, 1, 1]     # Cyan
    ]
    
    for i, obj in enumerate(objects_data):
        line_starts.append(start_point)
        line_ends.append(obj['position'])
        colors.append(color_options[i % len(color_options)])
        sizes.append(5)
        
        # 可選：印出位置資訊用於除錯
        if i == 0:  # 只印第一個物體避免太多輸出
            # print(f"Debug line to {obj['name']}: {start_point} -> {obj['position']}")
            pass
    
    # 清除之前的線條並繪製新的
    draw.clear_lines()
    
    if line_starts:
        draw.draw_lines(line_starts, line_ends, colors, sizes)


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject]):
    """Runs the simulation loop."""
    # Extract scene entities
    cube1_object = entities["cube1"]
    cube2_object = entities["cube2"]
    
    # Acquire the debug draw interface
    draw = _debug_draw.acquire_debug_draw_interface()
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    
    # 定義移動參數
    cube1_direction = torch.tensor([1.0, 0.0, 0.0], device=sim.device)  # cube1 沿 X 軸移動
    cube2_direction = torch.tensor([0.0, 1.0, 0.0], device=sim.device)  # cube2 沿 Y 軸移動
    movement_force = 10.0  # 移動力度
    gravity_compensation = 9.81  # 重力補償
    object_mass = 1.0  # 物體質量（從配置中已知）
    
    # 定義重置參數
    reset_direction = torch.tensor([1.0, 0.5, 0.0], device=sim.device)
    distance_range = (0.0, 3.0)
    lateral_range = (-1.0, 1.0)
    fixed_z = 3.0
    
    # Simulate physics
    while simulation_app.is_running():
        # reset every 500 frames (約10秒)
        if count % 500 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            
            # Reset cube1
            cube1_root_state = cube1_object.data.default_root_state.clone()
            cube1_base_pos = torch.tensor([[0.0, 0.0, fixed_z]], device=sim.device)
            cube1_offset = sample_directional_positions(
                cube1_base_pos, reset_direction, distance_range, lateral_range, sim.device
            )
            cube1_root_state[:, :3] = cube1_base_pos + cube1_offset
            cube1_root_state[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=sim.device)  # 重置四元數
            cube1_root_state[:, 7:] = 0.0  # 重置速度和角速度
            
            # Reset cube2
            cube2_root_state = cube2_object.data.default_root_state.clone()
            cube2_base_pos = torch.tensor([[2.0, 1.0, fixed_z]], device=sim.device)
            cube2_offset = sample_directional_positions(
                cube2_base_pos, reset_direction, distance_range, lateral_range, sim.device
            )
            cube2_root_state[:, :3] = cube2_base_pos + cube2_offset
            cube2_root_state[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=sim.device)  # 重置四元數
            cube2_root_state[:, 7:] = 0.0  # 重置速度和角速度
            
            # 安全地重置物體狀態
            cube1_object.write_root_state_to_sim(cube1_root_state)
            cube2_object.write_root_state_to_sim(cube2_root_state)
            
            print("----------------------------------------")
            print("[INFO]: Resetting floating cubes...")
            print(f"New cube1 position: {cube1_root_state[0, :3]}")
            print(f"New cube2 position: {cube2_root_state[0, :3]}")
        
        # 對 cube1 施加力：重力補償 + 沿 X 軸移動
        cube1_force = torch.zeros((cube1_object.num_instances, 3), device=sim.device)
        cube1_force[:, 2] = gravity_compensation * object_mass  # 重力補償
        cube1_force[:, 0] = movement_force  # X 軸移動力
        
        # 對 cube2 施加力：重力補償 + 沿 Y 軸移動  
        cube2_force = torch.zeros((cube2_object.num_instances, 3), device=sim.device)
        cube2_force[:, 2] = gravity_compensation * object_mass  # 重力補償
        cube2_force[:, 1] = movement_force  # Y 軸移動力
        
        # 施加外力
        cube1_object.set_external_force_and_torque(cube1_force, torch.zeros_like(cube1_force))
        cube2_object.set_external_force_and_torque(cube2_force, torch.zeros_like(cube2_force))
        
        # Update both objects
        cube1_object.write_data_to_sim()
        cube2_object.write_data_to_sim()
        
        # perform step
        sim.step()
        
        # update sim-time
        sim_time += sim_dt
        count += 1
        
        # update buffers
        cube1_object.update(sim_dt)
        cube2_object.update(sim_dt)
        
        # Update debug lines every frame - 傳入物件以獲取實時位置
        update_lines(draw, cube1_object, cube2_object)
        
        # print positions every 100 frames
        if count % 100 == 0:
            print(f"Frame {count}:")
            print(f"  Cube1 position: {cube1_object.data.root_state_w[0, :3]}")
            print(f"  Cube2 position: {cube2_object.data.root_state_w[0, :3]}")

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    
    # Set main camera
    sim.set_camera_view(eye=[15.0, 5.0, 15.0], target=[0.0, 0.0, 0.0])
    
    # Design scene
    scene_entities = design_scene()
    
    # Play the simulator
    sim.reset()
    
    # Now we are ready!
    print("[INFO]: Setup complete...")
    
    # Run the simulator
    run_simulator(sim, scene_entities)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()