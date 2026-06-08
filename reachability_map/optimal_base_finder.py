import os
import json
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import subprocess
import threading

from ndi_verified_cache import (
    DEFAULT_CACHE_PATH,
    default_entry_pos,
    find_latest,
    load_records,
)

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

DATABASE_PATH = os.path.join("scripts", "isaaclab_ws", "reachability_map", "cmap_database.json")
NDI_DATABASE_PATH = os.path.join("scripts", "isaaclab_ws", "reachability_map", "ndi_precomputed_database.json")
VERIFIED_CACHE_PATH = DEFAULT_CACHE_PATH

class OptimalBaseFinderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Optimal Robot Base & NDI Verification")
        self.root.geometry("1000x720")
        
        # Load databases
        self.db = self.load_database()
        self.ndi_db = self.load_ndi_database()
        self.verified_records = self.load_verified_records()
        
        # Keep track of active query block
        self.current_block_id = None
        self.current_block_data = None
        self.is_inside_bounds = False
        
        # Set UI Style
        self.setup_styles()
        
        # Create Layout
        self.create_widgets()
        
        # Initialize Plot
        self.update_plot(None, None, None, None)

    def load_ndi_database(self):
        if not os.path.exists(NDI_DATABASE_PATH):
            return None
        try:
            with open(NDI_DATABASE_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load NDI database: {e}")
            return None

    def load_verified_records(self):
        try:
            return load_records(VERIFIED_CACHE_PATH)
        except Exception as e:
            print(f"[ERROR] Failed to load verified NDI cache: {e}")
            return []

    def current_entry_pos(self):
        if self.active_tumor_pos is None:
            return None
        return default_entry_pos(self.active_tumor_pos, getattr(self, "active_tumor_angle", 0.0))

    def lookup_ndi_candidate(self, block_id, base_x):
        if self.active_tumor_pos is not None:
            verified = find_latest(
                self.verified_records,
                self.active_tumor_pos,
                self.current_entry_pos(),
                getattr(self, "active_tumor_angle", 0.0),
                block_id,
                base_x,
            )
            if verified:
                result = dict(verified.get("result", {}))
                result["created_at"] = verified.get("created_at")
                return result, "verified"

        if self.ndi_db and "blocks" in self.ndi_db:
            block_ndi = self.ndi_db["blocks"].get(str(block_id))
            if block_ndi:
                cand_data = block_ndi.get("candidates", {}).get(f"{base_x:.1f}")
                if cand_data:
                    return cand_data, "precomputed"

        return None, None

    def load_database(self):
        if not os.path.exists(DATABASE_PATH):
            messagebox.showwarning("Database Missing", f"Could not find cmap_database.json at:\n{DATABASE_PATH}\n\nPlease run build_cmap_database.py first.")
            return {"blocks": {}}
        try:
            with open(DATABASE_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to parse database: {e}")
            return {"blocks": {}}

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(".", font=("Inter", 10))
        self.style.configure("TLabel", foreground="#333")
        self.style.configure("Header.TLabel", font=("Outfit", 14, "bold"), foreground="#1e3a8a")
        self.style.configure("TButton", font=("Inter", 10, "bold"), padding=6)
        self.style.configure("Action.TButton", background="#2563eb", foreground="white")
        self.style.map("Action.TButton", background=[("active", "#1d4ed8")])
        self.style.configure("Viewer.TButton", background="#059669", foreground="white")
        self.style.map("Viewer.TButton", background=[("active", "#047857")])

    def create_widgets(self):
        # Left Panel (Inputs and Outputs)
        left_panel = ttk.Frame(self.root, padding=15, width=320)
        left_panel.pack_propagate(False)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        
        # Right Panel (Visualization)
        right_panel = ttk.Frame(self.root, padding=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # ─── LEFT PANEL: INPUTS ───
        ttk.Label(left_panel, text="Tumor Parameters", style="Header.TLabel").grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 15))
        
        ttk.Label(left_panel, text="X (m):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.entry_x = ttk.Entry(left_panel, width=15)
        self.entry_x.insert(0, "-0.20")  # default to a valid coordinate inside Block 1
        self.entry_x.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(left_panel, text="Y (m):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.entry_y = ttk.Entry(left_panel, width=15)
        self.entry_y.insert(0, "-0.75")  # default
        self.entry_y.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(left_panel, text="Z (m):").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.entry_z = ttk.Entry(left_panel, width=15)
        self.entry_z.insert(0, "0.95")  # default
        self.entry_z.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(left_panel, text="Insertion Angle (°):").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.combo_angle = ttk.Combobox(left_panel, width=13, values=["-30.0", "-15.0", "0.0", "15.0", "30.0"])
        self.combo_angle.insert(0, "0.0")
        self.combo_angle.grid(row=4, column=1, sticky=tk.W, pady=5)
        
        self.btn_query = ttk.Button(left_panel, text="Query Optimal Base", style="Action.TButton", command=self.query_base)
        self.btn_query.grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=15)
        
        # Divider line
        ttk.Separator(left_panel, orient="horizontal").grid(row=6, column=0, columnspan=2, sticky=tk.EW, pady=10)
        
        # ─── LEFT PANEL: OUTPUTS ───
        ttk.Label(left_panel, text="Optimal Base Results", style="Header.TLabel").grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=(10, 15))
        
        self.lbl_block = ttk.Label(left_panel, text="Matched Block: -")
        self.lbl_block.grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=3)
        
        # Candidate selection dropdown
        self.lbl_candidate_title = ttk.Label(left_panel, text="Select Base Candidate:")
        self.lbl_candidate_title.grid(row=9, column=0, sticky=tk.W, pady=5)
        self.combo_base_candidate = ttk.Combobox(left_panel, width=15, state="disabled")
        self.combo_base_candidate.grid(row=9, column=1, sticky=tk.W, pady=5)
        self.combo_base_candidate.bind("<<ComboboxSelected>>", self.on_candidate_selected)
        
        self.lbl_base_pos = ttk.Label(left_panel, text="Base Position: -")
        self.lbl_base_pos.grid(row=10, column=0, columnspan=2, sticky=tk.W, pady=3)
        
        self.lbl_score = ttk.Label(left_panel, text="Fine-grained Score: -")
        self.lbl_score.grid(row=11, column=0, columnspan=2, sticky=tk.W, pady=3)
        
        self.lbl_sr = ttk.Label(left_panel, text="Success Rate: -")
        self.lbl_sr.grid(row=12, column=0, columnspan=2, sticky=tk.W, pady=3)
        
        self.lbl_manip = ttk.Label(left_panel, text="Avg Manipulability: -")
        self.lbl_manip.grid(row=13, column=0, columnspan=2, sticky=tk.W, pady=3)
        
        # NDI Hit Rate label
        self.lbl_ndi_hit = ttk.Label(left_panel, text="NDI Hit Rate: -")
        self.lbl_ndi_hit.grid(row=14, column=0, columnspan=2, sticky=tk.W, pady=3)
        
        # Warning field for no valid base
        self.lbl_warning = ttk.Label(left_panel, text="", foreground="#dc2626", font=("Inter", 9, "bold"), wraplength=280)
        self.lbl_warning.grid(row=15, column=0, columnspan=2, sticky=tk.W, pady=10)
        
        # 3D Viewer Launch Button
        self.btn_viewer = ttk.Button(left_panel, text="Open 3D CMAP Viewer", style="Viewer.TButton", state=tk.DISABLED, command=self.open_viewer)
        self.btn_viewer.grid(row=16, column=0, columnspan=2, sticky=tk.EW, pady=15)
        
        # Save active run info
        self.active_backup = None
        self.active_opt = None
        self.active_tumor_pos = None  # (tx, ty, tz) of last query
        
        # ─── RIGHT PANEL: VISUALIZATION ───
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def query_base(self):
        # Refresh database in case user ran compilation recently
        self.db = self.load_database()
        self.ndi_db = self.load_ndi_database()
        self.verified_records = self.load_verified_records()
        if not self.db or "blocks" not in self.db or not self.db["blocks"]:
            messagebox.showerror("Query Error", "Database is empty or invalid.")
            return
            
        try:
            tx = float(self.entry_x.get())
            ty = float(self.entry_y.get())
            tz = float(self.entry_z.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric coordinates.")
            return

        try:
            t_angle = float(self.combo_angle.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid numeric angle.")
            return

        # Find matched block
        best_block_id = None
        block_data = None
        is_inside_bounds = False
        
        # Helper: compute clamped distance from point to a bounding box
        def bbox_distance(tx, ty, tz, bounds):
            dx = max(bounds["x_min"] - tx, 0, tx - bounds["x_max"])
            dy = max(bounds["y_min"] - ty, 0, ty - bounds["y_max"])
            dz = max(bounds["z_min"] - tz, 0, tz - bounds["z_max"])
            return (dx**2 + dy**2 + dz**2) ** 0.5

        # 1. First pass: collect ALL blocks containing point
        matching_blocks = []
        for bid, bdata in self.db["blocks"].items():
            bounds = bdata.get("wp_bounds", {})
            if bounds:
                in_x = bounds["x_min"] <= tx <= bounds["x_max"]
                in_y = bounds["y_min"] <= ty <= bounds["y_max"]
                in_z = bounds["z_min"] <= tz <= bounds["z_max"]
                if in_x and in_y and in_z:
                    matching_blocks.append((bid, bdata))
                    
        if matching_blocks:
            def block_sort_key(item):
                opt = item[1].get("optimal_base", {})
                return (opt.get("success_rate", 0.0), opt.get("sr_min", 0), opt.get("fg_score", 0.0), opt.get("avg_manip", 0.0))
            best_block_id, block_data = max(matching_blocks, key=block_sort_key)
            is_inside_bounds = True
        else:
            min_dist = float("inf")
            for bid, bdata in self.db["blocks"].items():
                bounds = bdata.get("wp_bounds", {})
                if bounds:
                    dist = bbox_distance(tx, ty, tz, bounds)
                    if dist < min_dist:
                        min_dist = dist
                        best_block_id = bid
                        block_data = bdata
        
        if best_block_id is None:
            messagebox.showerror("Error", "No blocks found in the database.")
            return
            
        self.current_block_id = best_block_id
        self.current_block_data = block_data
        self.active_tumor_pos = (tx, ty, tz)
        self.active_tumor_angle = t_angle
        self.active_backup = block_data.get("backup_csv")
        self.active_opt = block_data.get("opt_csv")
        self.is_inside_bounds = is_inside_bounds
        
        if self.active_backup and self.active_opt:
            self.btn_viewer.config(state=tk.NORMAL)
        else:
            self.btn_viewer.config(state=tk.DISABLED)
            
        # Display block info
        self.lbl_block.config(text=f"Matched Block: Block {best_block_id} (Center: {block_data['block_center']})")

        # Load candidate base positions from opt_csv
        opt_csv_rel = block_data.get("opt_csv")
        opt_csv_path = os.path.abspath(opt_csv_rel)
        candidates = []
        if os.path.exists(opt_csv_path):
            import csv
            with open(opt_csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if float(row["success_rate"]) > 0.0:
                        candidates.append(float(row["base_x"]))
        candidates = sorted(list(set(candidates)))
        
        opt_base = block_data.get("optimal_base", {})
        base_x = opt_base.get("base_x", 0.0)
        
        if candidates:
            self.combo_base_candidate.config(state="readonly")
            combo_values = []
            for x in candidates:
                is_occluded = False
                cand_data, source = self.lookup_ndi_candidate(best_block_id, x)
                if cand_data and cand_data.get("hit_rate", 0) == 0:
                    is_occluded = True
                if is_occluded:
                    source_tag = "VERIFIED" if source == "verified" else "PRECOMP"
                    combo_values.append(f"{x:.1f} [{source_tag} OCCLUDED]")
                else:
                    combo_values.append(f"{x:.1f}")
            self.combo_base_candidate["values"] = combo_values
            
            opt_base_x_str = f"{base_x:.1f}"
            matched_val = None
            for val in combo_values:
                if val.startswith(opt_base_x_str):
                    matched_val = val
                    break
            if matched_val:
                self.combo_base_candidate.set(matched_val)
            else:
                self.combo_base_candidate.current(0)
        else:
            self.combo_base_candidate.config(state="disabled")
            self.combo_base_candidate.set("")
            
        # Trigger visual/label updates with chosen candidate
        self.on_candidate_selected(None)

    def on_candidate_selected(self, event):
        val = self.combo_base_candidate.get()
        best_block_id = self.current_block_id
        block_data = self.current_block_data
        
        if not val or best_block_id is None or block_data is None:
            self.lbl_base_pos.config(text="Base Position: -")
            self.lbl_score.config(text="Fine-grained Score: -")
            self.lbl_sr.config(text="Success Rate: -")
            self.lbl_manip.config(text="Avg Manipulability: -")
            self.lbl_ndi_hit.config(text="NDI Hit Rate: -")
            self.lbl_warning.config(text="[WARNING] No valid base candidate with success rate > 0 found.")
            self.update_plot(None, None, None, None)
            return

        base_x = float(val.split()[0])
        
        # Load block CSV candidate metrics
        opt_base = {}
        opt_csv_rel = block_data.get("opt_csv")
        opt_csv_path = os.path.abspath(opt_csv_rel)
        if os.path.exists(opt_csv_path):
            import csv
            with open(opt_csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if abs(float(row["base_x"]) - base_x) < 1e-3:
                        opt_base = {
                            "base_x": float(row["base_x"]),
                            "base_y": float(row["base_y"]),
                            "fg_score": float(row["fg_score"]),
                            "success_rate": float(row["success_rate"]),
                            "avg_manip": float(row["avg_manip"])
                        }
                        break
        
        # Display outputs
        fg_score = opt_base.get("fg_score", 0.0)
        success_rate = opt_base.get("success_rate", 0.0)
        avg_manip = opt_base.get("avg_manip", 0.0)
        
        self.lbl_base_pos.config(text=f"Base Position: X = {base_x:.3f} m, Y = 0.000 m")
        self.lbl_score.config(text=f"Fine-grained Score: {fg_score:.4e}")
        self.lbl_sr.config(text=f"Success Rate: {success_rate * 100:.1f}%")
        self.lbl_manip.config(text=f"Avg Manipulability: {avg_manip:.4e}")
        
        # NDI verification database query
        ndi_hit_str = "NDI Hit Rate: N/A (Offline Database Missing)"
        is_fully_occluded = False
        cand_data, source = self.lookup_ndi_candidate(best_block_id, base_x)
        if cand_data:
            hit_rate = cand_data.get("hit_rate", 0)
            source_label = "Verified 3D run" if source == "verified" else "Precomputed estimate"
            if hit_rate == 0:
                ndi_hit_str = f"NDI Hit Rate: {hit_rate}/10 angles passed (Fully Occluded, {source_label})"
                is_fully_occluded = True
            else:
                ndi_hit_str = f"NDI Hit Rate: {hit_rate}/10 angles passed ({source_label})"
        elif self.ndi_db is None:
            ndi_hit_str = "NDI Hit Rate: N/A (Offline Database Missing)"
        else:
            ndi_hit_str = f"NDI Hit Rate: N/A (Candidate {base_x:.1f} not in NDI DB/cache)"
        self.lbl_ndi_hit.config(text=ndi_hit_str)

        # Warnings logic
        warning_text = ""
        if not getattr(self, "is_inside_bounds", True):
            bounds = block_data.get("wp_bounds", {})
            warning_text = f"[WARNING] Entered coordinate is OUTSIDE Block {best_block_id}'s searchable range:\n" \
                           f"X: {bounds.get('x_min', 0.0):.3f} ~ {bounds.get('x_max', 0.0):.3f}\n" \
                           f"Y: {bounds.get('y_min', 0.0):.3f} ~ {bounds.get('y_max', 0.0):.3f}\n" \
                           f"Z: {bounds.get('z_min', 0.0):.3f} ~ {bounds.get('z_max', 0.0):.3f}"
        elif fg_score == 0.0:
            warning_text = f"[WARNING] Within robot base range 0.1~1.5 m, no base position was found with 100% reachability for Block {best_block_id}.\nShowing best available base position."

        if is_fully_occluded:
            source_label = "verified 3D run" if source == "verified" else "precomputed estimate"
            ndi_warn = f"[WARNING] Candidate base position X={base_x:.1f} has 0/10 occlusion-free NDI angles ({source_label}).\nFully occluded, NOT recommended!"
            if warning_text:
                warning_text += "\n\n" + ndi_warn
            else:
                warning_text = ndi_warn

        self.lbl_warning.config(text=warning_text)

        # Update visual plot
        target_pt = np.array(self.active_tumor_pos) if self.active_tumor_pos else None
        self.update_plot(target_pt, best_block_id, base_x, block_data.get("wp_bounds"))

    def update_plot(self, target_pt, block_id, base_x, bounds):
        self.ax.clear()
        self.ax.set_title("Robot Base & Tumor Workspace (Top View / X-Y Plane)", fontsize=11, fontweight="bold", pad=10)
        self.ax.set_xlabel("X coordinate (m)", fontsize=9)
        self.ax.set_ylabel("Y coordinate (m)", fontsize=9)
        
        # Base translation offset (removed to align 2D plot with local USD coordinates)
        T_x = 0.0
        T_y = 0.0
        
        # 1. Plot linear base track (in local USD frame)
        track_x = np.linspace(0.1, 1.5, 100)
        track_y = np.full_like(track_x, T_y)
        self.ax.plot(track_x + T_x, track_y, color="#6b7280", linestyle="--", linewidth=1.5, label=f"Linear Base Track (Y={T_y:.2f})")
        
        # 2. Draw bounds bounding box if exists
        if bounds:
            bx_min, bx_max = bounds["x_min"], bounds["x_max"]
            by_min, by_max = bounds["y_min"], bounds["y_max"]
            # Rectangle
            rect = matplotlib.patches.Rectangle(
                (bx_min, by_min), bx_max - bx_min, by_max - by_min,
                linewidth=1, edgecolor="#2563eb", facecolor="#93c5fd", alpha=0.3, label=f"Block {block_id} Workspace"
            )
            self.ax.add_patch(rect)
            
        # 3. Plot Tumor/Target point
        if target_pt is not None:
            self.ax.scatter(target_pt[0], target_pt[1], color="#dc2626", s=100, zorder=5, label="Tumor Coordinate")
            
        # 4. Plot Selected Robot Base Position (in physical USD frame)
        if base_x is not None:
            phys_base_x = base_x + T_x
            phys_base_y = T_y
            self.ax.scatter(phys_base_x, phys_base_y, color="#059669", marker="X", s=150, zorder=6, label=f"Selected Base (X={base_x:.1f})")
            # Draw line between base and tumor
            if target_pt is not None:
                self.ax.plot([phys_base_x, target_pt[0]], [phys_base_y, target_pt[1]], color="#10b981", linestyle=":", alpha=0.8)

        # 5. Plot NDI cameras if database contains this candidate
        if base_x is not None:
            cand_data, source = self.lookup_ndi_candidate(block_id, base_x)
            if cand_data:
                    lacp = cand_data.get("lacp") # [lx, ly, lz]
                    angle_passes = cand_data.get("angle_passes") # [1, 0, 1, ...]
                    
                    if lacp and angle_passes:
                        # Draw LACP as blue cross (+)
                        source_label = "Verified LACP" if source == "verified" else "Precomputed LACP"
                        self.ax.scatter(lacp[0], lacp[1], color="#2563eb", marker="+", s=120, linewidths=2.0, zorder=7, label=source_label)
                        
                        # Plot NDI camera positions
                        angles_deg = np.linspace(0, 45, len(angle_passes))
                        R_h = 1.52
                        
                        first_pass_legend = True
                        first_fail_legend = True
                        
                        for i, passed in enumerate(angle_passes):
                            theta_rad = np.deg2rad(angles_deg[i])
                            # Arc math in local coords
                            ndi_x = lacp[0] + R_h * np.sin(theta_rad)
                            ndi_y = lacp[1] - R_h * np.cos(theta_rad)
                            
                            color = "#10b981" if passed == 1 else "#ef4444"
                            label_val = None
                            if passed == 1 and first_pass_legend:
                                label_val = "NDI Pass Angle"
                                first_pass_legend = False
                            elif passed == 0 and first_fail_legend:
                                label_val = "NDI Occluded Angle"
                                first_fail_legend = False
                                
                            # Highlight the slanted 45 degree camera (last one)
                            if i == len(angle_passes) - 1:
                                self.ax.scatter(ndi_x, ndi_y, facecolors=color, edgecolors="black", linewidths=1.5, s=90, zorder=8, label="45° NDI Camera")
                                self.ax.text(ndi_x + 0.05, ndi_y, "45° NDI", fontsize=8, fontweight="bold", color="#1e293b")
                            else:
                                self.ax.scatter(ndi_x, ndi_y, color=color, s=50, zorder=8, label=label_val)
                                
                            # Draw dotted line to LACP
                            self.ax.plot([ndi_x, lacp[0]], [ndi_y, lacp[1]], color=color, linestyle=":", alpha=0.4)

        # Set limits to cover the entire arc and track
        self.ax.set_xlim(-0.8, 1.8)
        self.ax.set_ylim(-2.6, 0.6)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True, linestyle=":", alpha=0.6)
        self.ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
        self.canvas.draw()

    def open_viewer(self):
        if not self.active_backup or not self.active_opt:
            messagebox.showerror("Error", "No active environment data loaded. Please query a tumor position first.")
            return
            
        backup_path = os.path.abspath(self.active_backup)
        opt_path = os.path.abspath(self.active_opt)
        
        # Check database files
        if not os.path.exists(backup_path):
            messagebox.showerror("File Missing", f"Required backup CSV file does not exist:\n{backup_path}\n\nPlease check if database compilation succeeded.")
            return
        if not os.path.exists(opt_path):
            messagebox.showerror("File Missing", f"Required optimization results CSV file does not exist:\n{opt_path}\n\nPlease check if database compilation succeeded.")
            return
            
        # Check launcher paths
        python_bat = r"C:\Users\RMML\AppData\Local\ov\pkg\4.5.0\python.bat"
        viewer_script = os.path.abspath(os.path.join(
            "scripts", "isaaclab_ws", "reachability_map", "sim_rasscmap_interactive_viewer.py"
        ))
        
        if not os.path.exists(python_bat):
            messagebox.showerror("Launcher Missing", f"Could not find Isaac Sim python.bat at:\n{python_bat}")
            return
        if not os.path.exists(viewer_script):
            messagebox.showerror("Script Missing", f"Could not find viewer script at:\n{viewer_script}")
            return
            
        # Run viewer in a separate thread so it doesn't block the UI
        def launch():
            cmd = [
                python_bat,
                viewer_script,
                "--backup_csv", backup_path,
                "--opt_csv", opt_path,
            ]
            # Pass tumor coordinates so the viewer can display the marker sphere
            if self.active_tumor_pos is not None:
                tx, ty, tz = self.active_tumor_pos
                cmd += ["--tumor_pos", f"{tx}", f"{ty}", f"{tz}"]
            if self.active_tumor_angle is not None:
                cmd += ["--tumor_angle", f"{self.active_tumor_angle}"]
            if self.current_block_id is not None:
                cmd += ["--block_id", str(self.current_block_id)]
            cmd += ["--verified_cache", os.path.abspath(VERIFIED_CACHE_PATH)]
                
            print(f"[UI Launcher] Running command: {' '.join(cmd)}")
            
            try:
                # Launch directly using python.bat without shell=True/conda environment activation
                # since python.bat runs standalone inside Isaac Sim's kit packages.
                subprocess.Popen(cmd)
            except Exception as e:
                print(f"[UI Launcher] [ERROR] Failed to launch viewer: {e}")
                
        threading.Thread(target=launch, daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = OptimalBaseFinderApp(root)
    root.mainloop()
