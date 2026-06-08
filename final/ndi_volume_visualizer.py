"""
NDI Volume 幾何視覺化器 (方案 B)

這個腳本在 Isaac Sim 環境啟動後，讀取 NDI Volume 的實際 USD Mesh 頂點，
計算 Convex Hull，並繪製包含以下元素的 3D 圖：
  - NDI Emitter 的位置（紅色大點）
  - NDI Volume 的 Convex Hull Wireframe（藍色框線）
  - Marker 球體的位置（若有提供，可選）

執行方式:
    .\\isaaclab.bat -p scripts\\isaaclab_ws\\final\\ndi_volume_visualizer.py --num_envs 1
"""

import argparse
import sys
import os

# ── Isaac Lab AppLauncher ──
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="NDI Volume Visualizer")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 以下需要在 simulation_app 啟動後才能 import ──
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非互動後端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.spatial import ConvexHull

import omni.usd
from pxr import Usd, UsdGeom, Gf

# Import IsaacLab 模組
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv

# Import 我們的環境設定
from isaaclab_tasks.manager_based.manipulation.reach.config.tm5.tm5_extension_linkout_cone_orientation_cfg import (
    TM5ExtensionLinkOutFanOrientationEnvCfg
)


def get_mesh_world_vertices(prim_path: str, stage) -> np.ndarray | None:
    """從 USD Mesh prim 讀取所有頂點並轉換到世界座標系"""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"[ERROR] Prim not found: {prim_path}")
        return None

    # 取得 Mesh 幾何
    mesh = UsdGeom.Mesh(prim)
    if not mesh:
        # 也許是 BasisCurves 或其他 Imageable
        print(f"[WARN] Not a UsdGeom.Mesh at {prim_path}, trying Xformable children...")
        return None

    points_attr = mesh.GetPointsAttr()
    if not points_attr:
        print(f"[ERROR] No points attribute at {prim_path}")
        return None

    local_points = points_attr.Get()
    if local_points is None or len(local_points) == 0:
        print(f"[ERROR] Empty points at {prim_path}")
        return None

    # 計算世界座標轉換矩陣
    xformable = UsdGeom.Xformable(prim)
    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    # 轉換所有頂點到世界座標
    world_points = []
    for p in local_points:
        world_pt = world_transform.Transform(Gf.Vec3d(p[0], p[1], p[2]))
        world_points.append([world_pt[0], world_pt[1], world_pt[2]])

    return np.array(world_points)


def find_ndi_volume_prims(stage, trigger_volume_prefix: str):
    """搜尋所有符合 NDI Volume 前綴的 Mesh Prim 路徑"""
    found_paths = []
    for prim in stage.Traverse():
        path_str = str(prim.GetPath())
        if trigger_volume_prefix in path_str:
            found_paths.append(path_str)
    return found_paths


def get_emitter_world_transform(prim_path: str, stage):
    """取得 NDI Emitter 的世界座標和旋轉四元數 (pos, quat [qx,qy,qz,qw])"""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"[WARN] Emitter not found: {prim_path}")
        return None, None
    xformable = UsdGeom.Xformable(prim)
    wt = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = wt.ExtractTranslation()
    pos = np.array([t[0], t[1], t[2]], dtype=float)
    rot = wt.ExtractRotation()
    q = rot.GetQuaternion()
    im = q.GetImaginary()
    quat = np.array([im[0], im[1], im[2], q.GetReal()], dtype=float)  # [qx,qy,qz,qw]
    return pos, quat


def rotate_by_quat(vec: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Rodrigues quaternion rotation: vec rotated by quat [qx,qy,qz,qw]"""
    t = 2 * np.cross(quat[:3], vec)
    return vec + quat[3] * t + np.cross(quat[:3], t)


def generate_math_frustum_world_vertices(emitter_pos: np.ndarray, emitter_quat: np.ndarray) -> np.ndarray:
    """從 NDI Polaris Vega 官方規格產生 12 個 Frustum 頂點，轉換到世界座標系

    Pyramid Volume 實際是「雙段梯形」，在 1532mm 處有中間截面注孔:
      近端面: -Z=0.950m, 480x448mm  (半: 0.240 x 0.224)
      中間面: -Z=1.532m, 1144x796mm (半: 0.572 x 0.398)
      遠端面: -Z=2.400m, 1566x1312mm(半: 0.783 x 0.656)
    """
    # [depth, half_w(X), half_h(Y)]  — 長寬對調（寬=Y軸, 高=X軸）
    planes = [
        (-0.950, 0.224, 0.240),  # 近端: 448mm(X) x 480mm(Y)
        (-1.532, 0.398, 0.572),  # 中間: 796mm(X) x 1144mm(Y)
        (-2.400, 0.656, 0.783),  # 遠端: 1312mm(X) x 1566mm(Y)
    ]
    local_pts = []
    for z, hw, hh in planes:
        local_pts += [
            [-hw, -hh, z], [ hw, -hh, z],
            [ hw,  hh, z], [-hw,  hh, z],
        ]
    local_pts = np.array(local_pts, dtype=float)
    world_pts = np.array([rotate_by_quat(v, emitter_quat) + emitter_pos for v in local_pts])
    return world_pts  # shape (12, 3)



def draw_frustum_12_edges(ax, verts12: np.ndarray, color="orangered", alpha=0.7, lw=2.0, label="Math Frustum"):
    """給定 12 個頂點（順序: 0-3 近, 4-7 中, 8-11 遠），畫出雙段梯形 Wireframe"""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    rings = [verts12[0:4], verts12[4:8], verts12[8:12]]

    # 每 ring 的周面
    for ring in rings:
        for a, b in [(0,1),(1,2),(2,3),(3,0)]:
            ax.plot(*zip(ring[a], ring[b]), color=color, lw=lw, alpha=alpha)

    # 中間構戶（近→中，中→遠）
    for seg_start in [0, 4]:
        for i in range(4):
            ax.plot(*zip(verts12[seg_start + i], verts12[seg_start + 4 + i]),
                    color=color, lw=lw, alpha=alpha)

    ax.scatter(verts12[:, 0], verts12[:, 1], verts12[:, 2], color=color, s=40, zorder=6)
    ax.plot([], [], color=color, lw=lw, label=label)

    # 半透明面（共 10 局面）
    r0, r1, r2 = rings
    faces = [
        list(r0),  # Near face
        list(r2),  # Far face
        # Segment 1 sides
        [r0[0], r0[1], r1[1], r1[0]], [r0[1], r0[2], r1[2], r1[1]],
        [r0[2], r0[3], r1[3], r1[2]], [r0[3], r0[0], r1[0], r1[3]],
        # Segment 2 sides
        [r1[0], r1[1], r2[1], r2[0]], [r1[1], r1[2], r2[2], r2[1]],
        [r1[2], r1[3], r2[3], r2[2]], [r1[3], r1[0], r2[0], r2[3]],
    ]
    poly = Poly3DCollection(faces, alpha=0.07, facecolor=color, edgecolor="none")
    ax.add_collection3d(poly)



def draw_emitter_axes(ax, emitter_pos: np.ndarray, emitter_quat: np.ndarray, length: float = 0.3):
    """在 3D 圖上畫出 Emitter 的局部座標軸
    
    紅色箭頭 = 局部 +X 軸
    綠色箭頭 = 局部 +Y 軸
    藍色箭頭 = 局部 +Z 軸
    """
    axes = {
        "+X": (np.array([1, 0, 0]), "red"),
        "+Y": (np.array([0, 1, 0]), "limegreen"),
        "+Z": (np.array([0, 0, 1]), "dodgerblue"),
    }
    for name, (local_dir, color) in axes.items():
        world_dir = rotate_by_quat(local_dir, emitter_quat)
        end = emitter_pos + world_dir * length
        ax.quiver(
            emitter_pos[0], emitter_pos[1], emitter_pos[2],
            world_dir[0] * length, world_dir[1] * length, world_dir[2] * length,
            color=color, linewidth=2, arrow_length_ratio=0.25,
        )
        ax.text(end[0], end[1], end[2], name, color=color, fontsize=8)

def draw_convex_hull_wireframe(ax, points: np.ndarray, color="blue", alpha=0.15, lw=1.5):
    """計算並繪製 ConvexHull Wireframe"""
    if len(points) < 4:
        print("[WARN] Not enough points for ConvexHull")

        return None

    hull = ConvexHull(points)
    hull_vertices = points[hull.vertices]

    # 繪製頂點
    ax.scatter(hull_vertices[:, 0], hull_vertices[:, 1], hull_vertices[:, 2],
               color="red", s=40, zorder=5, label=f"{len(hull.vertices)} 頂點")

    # 繪製全部 60 個原始頂點（淡色）
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               color=color, s=5, alpha=0.3, label=f"{len(points)} 原始頂點")

    # 繪製 ConvexHull 邊線
    edge_set = set()
    for simplex in hull.simplices:
        for i in range(len(simplex)):
            a, b = simplex[i], simplex[(i + 1) % len(simplex)]
            edge = tuple(sorted([a, b]))
            if edge not in edge_set:
                edge_set.add(edge)
                ax.plot([points[a, 0], points[b, 0]],
                        [points[a, 1], points[b, 1]],
                        [points[a, 2], points[b, 2]],
                        color=color, lw=lw, alpha=0.7)

    # 繪製面（半透明）
    verts = [points[simplex] for simplex in hull.simplices]
    poly = Poly3DCollection(verts, alpha=alpha, facecolor="cyan", edgecolor=color, lw=0.5)
    ax.add_collection3d(poly)
    return hull


ENV_COLORS = ["steelblue", "seagreen", "darkorchid", "saddlebrown", "crimson", "teal"]


def get_all_prim_positions_by_keyword(stage, keyword: str) -> list:
    """掃描 Stage 中所有路徑含 keyword 的 Mesh/Xform，回傳世界座標列表"""
    results = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if keyword in path:
            xf = UsdGeom.Xformable(prim)
            if xf:
                wt = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                t = wt.ExtractTranslation()
                results.append((path, np.array([t[0], t[1], t[2]], dtype=float)))
    return results


def main():
    num_envs = args_cli.num_envs
    env_cfg = TM5ExtensionLinkOutFanOrientationEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = "cuda:0" if hasattr(args_cli, "device") and args_cli.device else "cpu"

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[ERROR] Cannot get USD stage!")
        env.close()
        simulation_app.close()
        return

    # ── 收集每個 env 的資料 ──
    env_data = []  # list of dicts per env

    for env_idx in range(num_envs):
        env_prefix = f"/World/envs/env_{env_idx}"
        print(f"\n[ENV {env_idx}] Scanning {env_prefix} ...")

        # 1. NDI Volume Mesh 頂點
        vol_prefix = f"{env_prefix}/Root/NDI/mesh_"
        vol_paths = find_ndi_volume_prims(stage, vol_prefix)
        vol_verts_list = []
        for vp in vol_paths:
            v = get_mesh_world_vertices(vp, stage)
            if v is not None and len(v) > 0:
                vol_verts_list.append(v)
        all_pts = np.vstack(vol_verts_list) if vol_verts_list else None
        if all_pts is not None:
            print(f"  Volume: {len(all_pts)} vertices")

        # 2. NDI 父節點 Transform
        ndi_parent_path = f"{env_prefix}/Root/NDI"
        ndi_pos, ndi_quat = get_emitter_world_transform(ndi_parent_path, stage)
        if ndi_pos is not None:
            print(f"  NDI parent: pos={ndi_pos.round(3)}")

        # 3. 數學 Frustum 頂點
        math_verts = None
        if ndi_pos is not None and ndi_quat is not None:
            math_verts = generate_math_frustum_world_vertices(ndi_pos, ndi_quat)

        # 4. Marker 位置（掃描 bm_a, bm_b, bm_c, bm_d 等已知 Marker prim 名稱）
        marker_positions = []
        marker_keywords = ["bm_a", "bm_b", "bm_c", "bm_d"]
        for keyword in marker_keywords:
            hits = get_all_prim_positions_by_keyword(stage, f"{env_prefix}/Root")
            for path, pos in hits:
                if keyword in path.split("/")[-1]:
                    marker_positions.append((keyword, pos))
        # 去重
        seen = set()
        unique_markers = []
        for name, pos in marker_positions:
            key = tuple(pos.round(3))
            if key not in seen:
                seen.add(key)
                unique_markers.append((name, pos))
        if unique_markers:
            print(f"  Markers found: {len(unique_markers)}")

        env_data.append({
            "idx": env_idx,
            "vol_pts": all_pts,
            "ndi_pos": ndi_pos,
            "ndi_quat": ndi_quat,
            "math_verts": math_verts,
            "markers": unique_markers,
        })

    # ── 繪圖：4 個視角（左上=透視, 右上=俯視, 左下=正視, 右下=側視）──
    views = [
        (25,  -60, "Perspective"),
        (90,  -90, "Top-Down XY"),
        (0,   -90, "Front XZ"),
        (0,     0, "Side YZ"),
    ]

    fig = plt.figure(figsize=(24, 18))
    fig.suptitle(
        f"NDI Multi-Env Debug ({num_envs} envs): Volume (solid) | Math Frustum (wireframe) | Markers (X)",
        fontsize=12
    )

    for vi, (elev, azim, vlabel) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, vi, projection="3d")

        for d in env_data:
            idx = d["idx"]
            color = ENV_COLORS[idx % len(ENV_COLORS)]
            label_prefix = f"env{idx}"

            # USD Volume
            if d["vol_pts"] is not None and len(d["vol_pts"]) >= 4:
                draw_convex_hull_wireframe(ax, d["vol_pts"], color=color, alpha=0.10, lw=1.2)

            # Math Frustum（同色但用虛線框表示）
            if d["math_verts"] is not None:
                draw_frustum_12_edges(ax, d["math_verts"], color=color, alpha=0.5, lw=1.5,
                                      label=f"{label_prefix} Frustum")

            # NDI 位置
            if d["ndi_pos"] is not None:
                ax.scatter(*d["ndi_pos"], color=color, s=200, marker="*", zorder=10)
                draw_emitter_axes(ax, d["ndi_pos"], d["ndi_quat"], length=0.20)

            # Marker 位置
            for mname, mpos in d["markers"]:
                ax.scatter(*mpos, color=color, s=80, marker="X", zorder=9)
                if vi == 1:  # 只在透視圖加文字
                    ax.text(mpos[0], mpos[1], mpos[2], f"e{idx}", fontsize=6, color=color)

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("X (m)", fontsize=8)
        ax.set_ylabel("Y (m)", fontsize=8)
        ax.set_zlabel("Z (m)", fontsize=8)
        ax.set_title(vlabel, fontsize=10)
        if vi == 2:  # 俯視圖加圖例
            ax.legend(loc="upper left", fontsize=7)

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "occlusion_output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"ndi_multienv_debug_{num_envs}envs.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[OK] Multi-env debug plot saved: {out_path}")
    plt.close()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
