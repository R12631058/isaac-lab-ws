"""
run_blocks_batch.py
依序執行 parallel_base_optimizer_linear.py for Block 0, 3, 4, 5, 6, 7, 8, 9
每個 block 跑完後自動更新 optimal_block_N.json 並繼續下一個
"""

import subprocess
import sys
import os
import time

# Block centers (block_id → [cx, cy, cz])
# Block 0 center = [0.0, 0.0, 0.0]  (re-run with fixed pruning)
# Blocks 3~9: center X = block_id * 0.1
BLOCKS = [
    {"block_id": 0,  "block_center": [0.0, 0.0, 0.0]},
    {"block_id": 3,  "block_center": [0.3, 0.0, 0.0]},
    {"block_id": 4,  "block_center": [0.4, 0.0, 0.0]},
    {"block_id": 5,  "block_center": [0.5, 0.0, 0.0]},
    {"block_id": 6,  "block_center": [0.6, 0.0, 0.0]},
    {"block_id": 7,  "block_center": [0.7, 0.0, 0.0]},
    {"block_id": 8,  "block_center": [0.8, 0.0, 0.0]},
    {"block_id": 9,  "block_center": [0.9, 0.0, 0.0]},
]

ISAACLAB_ROOT = r"C:\Users\RMML\IsaacLab"
SCRIPT = r"scripts\isaaclab_ws\reachability_map\parallel_base_optimizer_linear.py"

def run_block(block_id: int, block_center: list):
    cx, cy, cz = block_center
    cmd = [
        "conda", "run", "-n", "env_isaaclab", "--no-capture-output",
        "python", os.path.join(ISAACLAB_ROOT, SCRIPT),
        "--block_id", str(block_id),
        "--block_center", str(cx), str(cy), str(cz),
        "--max_waypoints", "70",
    ]
    print(f"\n{'='*65}")
    print(f"  Starting Block {block_id}  center=({cx}, {cy}, {cz})")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*65}\n")

    start = time.time()
    result = subprocess.run(cmd, cwd=ISAACLAB_ROOT)
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n[OK] Block {block_id} finished in {elapsed/60:.1f} min")
    else:
        print(f"\n[ERROR] Block {block_id} failed (exit code {result.returncode})")
        print("  Stopping batch.")
        sys.exit(1)

if __name__ == "__main__":
    total_start = time.time()
    for block in BLOCKS:
        run_block(block["block_id"], block["block_center"])

    total_elapsed = time.time() - total_start
    print(f"\n{'='*65}")
    print(f"  All blocks completed in {total_elapsed/60:.1f} min total")
    print(f"{'='*65}")
    print("\nNext step: run build_cmap_database.py to rebuild the database.")
