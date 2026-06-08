"""
Visualize Inverse Reachability Map
Reads output from multi_position_eval.py and plots Success Rate vs Robot Base Position.
"""

import matplotlib.pyplot as plt
import argparse
import json
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser(description="Plot Reachability Map")
    parser.add_argument("--file", type=str, default="scripts/isaaclab_ws/reachability_map/multi_position_results.json", help="Path to JSON results")
    parser.add_argument("--save_plot", type=str, default="scripts/isaaclab_ws/reachability_map/inverse_reachability_map.png", help="Path to save plot image")
    parser.add_argument("--show", action="store_true", help="Show plot window")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"[ERROR] JSON file not found: {args.file}")
        return

    with open(args.file, 'r') as f:
        data = json.load(f)

    print(f"[INFO] Loaded results from {data['timestamp']}")
    print(f"[INFO] Checkpoint: {data['checkpoint']}")
    print(f"[INFO] Target Pose: {data.get('target_pose', 'Unknown')}")

    results = data['per_position_results']
    
    # Extract data
    x_positions = [r['x'] for r in results]
    y_positions = [r['y'] for r in results]
    success_rates = [r['success_rate'] for r in results]
    
    # 2D Map (Top-Down View)
    plt.figure(figsize=(10, 6))
    
    # Plot Robot Base Positions
    # Color mapping: Red (0%) -> Yellow (50%) -> Green (100%)
    sc = plt.scatter(x_positions, y_positions, c=success_rates, cmap='RdYlGn', vmin=0, vmax=100, s=100, edgecolors='black', label='Robot Base')
    cbar = plt.colorbar(sc)
    cbar.set_label('Success Rate (%)')
    
    # Plot Target
    target_pose = data.get('target_pose', None)
    if target_pose:
        plt.scatter([target_pose[0]], [target_pose[1]], c='blue', marker='*', s=200, label='Target (Cube)')
        plt.text(target_pose[0], target_pose[1]+0.02, 'Target', ha='center', color='blue')
    
    plt.title(f"Inverse Reachability Map\n(Target: {data.get('target_pose', 'N/A')})")
    plt.xlabel("Robot Base X (m)")
    plt.ylabel("Robot Base Y (m)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Annotate success rates
    for r in results:
        plt.text(r['x'], r['y']+0.01, f"{r['success_rate']:.0f}%", fontsize=8, ha='center')

    # Save
    plt.savefig(args.save_plot, dpi=300)
    print(f"[OK] Plot saved to {args.save_plot}")
    
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
