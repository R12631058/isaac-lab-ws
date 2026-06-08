"""Script to update all optimal_block_X.json files using the new success_rate-prioritized sort key."""
import os
import json
import glob
import csv

blocks_dir = os.path.join("scripts", "isaaclab_ws", "reachability_map", "heatmap", "blocks")

def get_latest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

for block_id in range(10):
    json_path = os.path.join(blocks_dir, f"optimal_block_{block_id}.json")
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} does not exist.")
        continue
        
    with open(json_path, "r") as f:
        block_data = json.load(f)
        
    opt_csv_pattern = os.path.join(blocks_dir, f"optimization_results_block{block_id}_*.csv")
    latest_opt_csv = get_latest_file(opt_csv_pattern)
    if not latest_opt_csv:
        print(f"Error: No opt CSV found for Block {block_id}")
        continue
        
    print(f"Updating Block {block_id} using {os.path.basename(latest_opt_csv)}...")
    
    # Read optimization CSV
    results = []
    with open(latest_opt_csv, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "env_id": int(row["env_id"]),
                "base_x": float(row["base_x"]),
                "base_y": float(row["base_y"]),
                "success_rate": float(row["success_rate"]),
                "sr_min": int(row["sr_min"]),
                "fg_score": float(row["fg_score"]),
                "avg_manip": float(row["avg_manip"])
            })
            
    # Sort with success_rate first, then sr_min, then fg_score
    best = max(results, key=lambda r: (r["success_rate"], r["sr_min"], r["fg_score"], r["avg_manip"]))
    
    block_data["optimal_env_id"] = best["env_id"]
    block_data["optimal_base"] = {
        "base_x": best["base_x"],
        "base_y": best["base_y"],
        "fg_score": best["fg_score"],
        "success_rate": best["success_rate"],
        "sr_min": best["sr_min"],
        "avg_manip": best["avg_manip"]
    }
    
    # Save back
    with open(json_path, "w") as f:
        json.dump(block_data, f, indent=2)
    print(f"  Block {block_id} optimal base: Env {best['env_id']} (cs={best['success_rate']:.1%}, sr_min={best['sr_min']}, fg={best['fg_score']:.4e})")

print("\nAll JSON files updated. Rebuilding database...")
import subprocess
subprocess.run(["python", os.path.join("scripts", "isaaclab_ws", "reachability_map", "build_cmap_database.py")], check=True)
print("Database rebuilt successfully!")
