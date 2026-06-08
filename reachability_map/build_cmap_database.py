import os
import json
import glob

def main():
    blocks_dir = os.path.join("scripts", "isaaclab_ws", "reachability_map", "heatmap", "blocks")
    output_db_path = os.path.join("scripts", "isaaclab_ws", "reachability_map", "cmap_database.json")
    
    print(f"Scanning for JSON files in: {blocks_dir}")
    json_pattern = os.path.join(blocks_dir, "optimal_block_*.json")
    json_files = glob.glob(json_pattern)
    
    database = {
        "blocks": {}
    }
    
    for filepath in sorted(json_files):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            
            block_id = str(data.get("block_id"))
            database["blocks"][block_id] = {
                "block_center": data.get("block_center"),
                "wp_bounds": data.get("wp_bounds"),
                "optimal_base": data.get("optimal_base"),
                "backup_csv": data.get("backup_csv"),
                "opt_csv": data.get("opt_csv"),
                "timestamp": data.get("timestamp")
            }
            print(f"  Loaded Block {block_id} from {os.path.basename(filepath)}")
        except Exception as e:
            print(f"  [ERROR] Failed to read {filepath}: {e}")
            
    # Save the consolidated database
    with open(output_db_path, "w") as f:
        json.dump(database, f, indent=2)
        
    print(f"\nSuccessfully compiled database with {len(database['blocks'])} blocks!")
    print(f"Database saved to: {output_db_path}")

if __name__ == "__main__":
    main()
