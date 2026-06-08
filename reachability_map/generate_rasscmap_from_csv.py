import os
import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

def find_backup_csv(opt_csv_path):
    directory = os.path.dirname(opt_csv_path)
    filename = os.path.basename(opt_csv_path)
    
    # Extract the prefix e.g., "X0.10-0.30_Y0.00-0.10"
    parts = filename.split('_')
    if len(parts) >= 5:
        prefix = f"{parts[2]}_{parts[3]}"
    else:
        print("[ERROR] Cannot parse prefix from optimization_results filename.")
        return None
        
    search_pattern = os.path.join(directory, f"backup_live_records_{prefix}_*.csv")
    candidates = glob.glob(search_pattern)
    
    if not candidates:
        print(f"[ERROR] No matching backup_live_records found for prefix {prefix}")
        return None
        
    # Sort by modification time and pick the most recent one
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]
    
def main():
    parser = argparse.ArgumentParser(description="Generate 3D RASSCMAPs from a given optimization_results CSV.")
    parser.add_argument("csv_path", type=str, help="Path to the optimization_results_*.csv file")
    args = parser.parse_args()
    
    opt_csv = args.csv_path
    if not os.path.exists(opt_csv):
        print(f"[ERROR] File not found: {opt_csv}")
        return
        
    backup_csv = find_backup_csv(opt_csv)
    if not backup_csv:
        print("[ERROR] Could not locate the corresponding backup logs to generate 3D plots.")
        return
        
    print(f"[INFO] Using Optimization Results: {opt_csv}")
    print(f"[INFO] Using Backup Realtime Logs: {backup_csv}")
    
    opt_df = pd.read_csv(opt_csv)
    env_info = {}
    for _, row in opt_df.iterrows():
        env_info[int(row['env_id'])] = {'base_x': row['base_x'], 'base_y': row['base_y']}
        
    print("[INFO] Loading backup data...")
    backup_df = pd.read_csv(backup_csv)
    
    run_tag = os.path.basename(opt_csv).replace("optimization_results_", "").replace(".csv", "")
    
    base_rasscmap_dir = os.path.join(os.path.dirname(os.path.dirname(opt_csv)), "rasscmaps")
    rasscmap_dir = os.path.join(base_rasscmap_dir, run_tag)
    os.makedirs(rasscmap_dir, exist_ok=True)
    print(f"[INFO] Generated images will be saved in: {rasscmap_dir}")
    
    envs = backup_df['Env_ID'].unique()
    
    for env_id in envs:
        env_df = backup_df[backup_df['Env_ID'] == env_id].copy()
        
        info = env_info.get(env_id, {'base_x': 0.0, 'base_y': 0.0})
        base_x = info['base_x']
        base_y = info['base_y']
        
        env_df['Is_Success'] = (env_df['Manipulability_Index'] > 0.0).astype(float)
        
        df_gp = env_df.groupby('Waypoint').agg({
            'Pos_X': 'mean',
            'Pos_Y': 'mean',
            'Pos_Z': 'mean',
            'Is_Success': 'mean' 
        }).reset_index()

        df_gp.rename(columns={'Is_Success': 'RI'}, inplace=True)
        df_gp = df_gp.sort_values('Waypoint')

        fig = plt.figure(figsize=(10, 8))
        ax3d = fig.add_subplot(111, projection='3d')
        
        ax3d.plot(df_gp['Pos_X'], df_gp['Pos_Y'], df_gp['Pos_Z'], 
                color='gray', linestyle='dashed', linewidth=2, label='Insertion Path')
        
        cmap_rasscmap = cm.jet_r
        sc = ax3d.scatter(df_gp['Pos_X'], df_gp['Pos_Y'], df_gp['Pos_Z'], 
                        c=df_gp['RI'], cmap=cmap_rasscmap, vmin=0.0, vmax=1.0, s=100, label='Waypoints')
                        
        plt.colorbar(sc, label='Reachability Index (RI)')
        ax3d.set_title(f"3D RASSCMAP (Env {env_id} Base: X={base_x:.2f}, Y={base_y:.2f})")
        ax3d.set_xlabel('X (m)')
        ax3d.set_ylabel('Y (m)')
        ax3d.set_zlabel('Z (m)')
        ax3d.legend()
        
        ax3d.view_init(elev=20, azim=25)
        
        vis_path = os.path.join(rasscmap_dir, f"rasscmap_3d_Env{env_id}_X{base_x:.2f}_Y{base_y:.2f}_{run_tag}.png")
        plt.savefig(vis_path)
        plt.close(fig)
        
    print(f"\n[DONE] Saved {len(envs)} 3D RASSCMAP plots successfully!")

if __name__ == "__main__":
    main()