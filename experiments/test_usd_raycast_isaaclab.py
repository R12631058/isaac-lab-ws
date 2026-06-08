"""
測試：USD 檔案中的 Raycast 檢測
目標：驗證 surgeryroom_lostfunc.usd 的 NDI raycast 是否在 Isaac Lab 中正常工作

這個測試基於 Bestpose_v7_NDI_enable.py 中的 raycast 邏輯：
- 從 NDI 發射器射向各個標記球體 (FM, HM, BM, EM, UM)
- 驗證射線是否命中這些標記
- 記錄命中率作為驗證依據

執行方式:
    isaaclab.bat -p scripts/isaaclab_ws/experiments/test_usd_raycast_isaaclab.py    
    .\isaaclab.bat -p scripts\isaaclab_ws\test_ndi_detector.py
"""

import argparse
from isaaclab.app import AppLauncher

# 解析命令行參數
parser = argparse.ArgumentParser(description="Test USD Raycast in Isaac Lab")
parser.add_argument("--test_duration", type=float, default=5.0, help="Test duration in seconds")
parser.add_argument("--observation_time", type=float, default=30.0, help="Observation time after test (seconds)")
parser.add_argument("--ray_steps", type=int, default=1, help="Perform raycast every N simulation steps (1 = every step)")
parser.add_argument("--freeze_arm", action="store_true", help="Freeze robot arm after USD loads (prevents marker movement)")
parser.add_argument("--usd_path", type=str, 
                   default=r"C:\Nick\surgery_team\surgery_team\USD\surgeryroom_lostfunc.usd",
                   help="Path to USD file")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 啟動模擬器
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===== 以下是模擬器啟動後的代碼 =====

import sys
# 修正 Windows cp950 編碼問題 - 讓 emoji/unicode 不會炸掉
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import time
import numpy as np
import omni.usd
from pxr import Gf, UsdGeom, UsdPhysics
import omni.kit.raycast.query
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaacsim.util.debug_draw import _debug_draw


class USDRaycastTester:
    """
    USD Raycast 測試類
    基於 Bestpose_v7_NDI_enable.py 中的 raycast 邏輯
    """
    
    def __init__(self, config):
        self.config = config
        self.raycast_interface = None
        self.debug_draw = None
        self.frame_count = 0
        self.raycast_results = []
        self.ray_info = []
        
        # 阻擋器管理 (方案1: 預建 + 可見性切換)
        self.blocker_prims = {}  # {marker_name: {path, prim, position, size}}
        self.blocker_paths_set = set()  # 快速查詢 hit_path 是否為阻擋器
        
        # NDI 檢測路徑配置 - 直接複製 Bestpose_v7_NDI_enable.py
        start_prims = "(/Root/NDI_02/NDI_emitter),/Root/NDI_02/NDI_left_sensor,/Root/NDI_02/NDI_right_sensor"
        end_prims = """(/Root/robotarm_base/End_needle/BM/BM_meter/bm_a),
        /Root/robotarm_base/End_needle/BM/BM_meter/bm_b,
        /Root/robotarm_base/End_needle/BM/BM_meter/bm_c,
        /Root/robotarm_base/End_needle/BM/BM_meter/bm_d,
        (/Root/robotarm_base/End_needle/EM/EM/em_a),
        /Root/robotarm_base/End_needle/EM/EM/em_b,
        /Root/robotarm_base/End_needle/EM/EM/em_c,
        /Root/robotarm_base/End_needle/EM/EM/em_d,
        (/Root/FM/FM/FM/FM/fm_a),
        (/Root/FM/FM/FM/FM/fm_b),
        (/Root/FM/FM/FM/FM/fm_c),
        (/Root/FM/FM/FM/FM/fm_d),
        (/Root/robotarm_base/End_needle/HM/HM_frame/hm_rball_0/node_/mesh_),
        /Root/robotarm_base/End_needle/HM/HM_frame/hm_rball_1/node_/mesh_,
        /Root/robotarm_base/End_needle/HM/HM_frame/hm_rball_2/node_/mesh_,
        /Root/robotarm_base/End_needle/HM/HM_frame/hm_rball_3/node_/mesh_,
        (/Root/robotarm_base/UM/UM/UM/um_a),
        (/Root/robotarm_base/UM/UM/UM/um_b),
        (/Root/robotarm_base/UM/UM/UM/um_c),
        (/Root/robotarm_base/UM/UM/UM/um_d)"""
        
        # 解析起點路徑
        self.start_prim_paths = []
        self.start_highlight_paths = []
        for path in start_prims.split(','):
            path = path.strip()
            if path.startswith('(') and path.endswith(')'):
                clean_path = path[1:-1].strip()
                self.start_prim_paths.append(clean_path)
                self.start_highlight_paths.append(clean_path)
            else:
                self.start_prim_paths.append(path)
        
        # 解析終點路徑
        self.end_prim_paths = []
        self.end_highlight_paths = []
        for path in end_prims.split(','):
            path = path.strip()
            if path.startswith('(') and path.endswith(')'):
                clean_path = path[1:-1].strip()
                self.end_prim_paths.append(clean_path)
                self.end_highlight_paths.append(clean_path)
            else:
                self.end_prim_paths.append(path)
        
        print(f"\n📍 NDI 路徑配置:")
        print(f"  起點: {len(self.start_prim_paths)} 個")
        for i, path in enumerate(self.start_prim_paths):
            print(f"    {i+1}. {path}")
        print(f"  終點: {len(self.end_prim_paths)} 個")
        print(f"  (只顯示前 5 個終點)")
        for i, path in enumerate(self.end_prim_paths[:5]):
            print(f"    {i+1}. {path}")
        if len(self.end_prim_paths) > 5:
            print(f"    ... 還有 {len(self.end_prim_paths) - 5} 個")
    
    def initialize(self):
        """初始化 raycast 介面"""
        try:
            self.raycast_interface = omni.kit.raycast.query.acquire_raycast_query_interface()
            if self.raycast_interface:
                print("\n✅ Raycast interface acquired")
            else:
                print("\n❌ Failed to acquire raycast interface")
                return False
            
            # 初始化 debug draw
            self.debug_draw = _debug_draw.acquire_debug_draw_interface()
            if self.debug_draw:
                print("✅ Debug draw interface acquired")
            else:
                print("⚠️  Debug draw interface not available")
            
            return True
        except Exception as e:
            print(f"\n❌ Initialization failed: {e}")
            return False
    
    # ========== 阻擋器管理方法 (方案1: 預建 + 可見性切換) ==========
    
    def create_blockers_for_markers(self, stage):
        """在 sim.reset() 之前為每個標記群組預建阻擋立方體
        
        原理：Raycast BVH 在 sim.reset() 時建立，之後不再更新。
        因此必須在 reset 前建立所有阻擋器幾何體，再用 visibility 控制是否生效。
        
        Args:
            stage: USD Stage 物件
        Returns:
            bool: 是否成功建立阻擋器
        """
        # 獲取 NDI 發射器位置
        emitter_pos = None
        for start_path in self.start_prim_paths:
            pos = self._get_prim_position(start_path)
            if pos is not None:
                emitter_pos = pos
                break
        
        if emitter_pos is None:
            print("❌ 無法取得發射器位置，跳過阻擋器建立")
            return False
        
        print(f"\n🔧 預建 Raycast 阻擋器 (方案1)...")
        print(f"   NDI 發射器位置: ({emitter_pos[0]:.4f}, {emitter_pos[1]:.4f}, {emitter_pos[2]:.4f})")
        
        markers = ['FM', 'BM', 'EM', 'HM', 'UM']
        blocker_root = "/Root/RaycastBlockers"
        
        # 建立阻擋器根節點
        stage.DefinePrim(blocker_root, "Xform")
        
        for marker in markers:
            # 找出該標記的所有球體路徑
            marker_paths = [p for p in self.end_prim_paths if marker.upper() in p.upper()]
            if not marker_paths:
                print(f"   ⚠️ 找不到 {marker} 的終點路徑")
                continue
            
            # 計算標記球體的質心
            positions = []
            for path in marker_paths:
                pos = self._get_prim_position(path)
                if pos is not None:
                    positions.append(pos)
            
            if not positions:
                print(f"   ⚠️ {marker} 無有效位置")
                continue
            
            centroid = np.mean(positions, axis=0)
            
            # 計算球體分佈範圍，決定阻擋器大小
            if len(positions) > 1:
                max_spread = max(np.linalg.norm(np.array(p) - centroid) for p in positions)
                # 阻擋器大小 = 球體分佈範圍 * 安全係數，最小 0.15m
                blocker_size = max(0.15, max_spread * 4.0)
            else:
                blocker_size = 0.15
            
            # 在發射器到標記質心的 70% 處放置阻擋器
            # 放在靠近標記的位置比較不會影響其他射線
            direction = centroid - emitter_pos
            distance = np.linalg.norm(direction)
            if distance < 0.01:
                print(f"   ⚠️ {marker} 與發射器距離太近")
                continue
            
            placement_ratio = 0.7  # 70% 處
            blocker_pos = emitter_pos + direction * placement_ratio
            
            # 建立阻擋器立方體
            blocker_path = f"{blocker_root}/Blocker_{marker}"
            blocker_prim = stage.DefinePrim(blocker_path, "Cube")
            blocker_geom = UsdGeom.Cube(blocker_prim)
            blocker_geom.GetSizeAttr().Set(float(blocker_size))
            
            xform = UsdGeom.Xformable(blocker_prim)
            xform.AddTranslateOp().Set(Gf.Vec3d(
                float(blocker_pos[0]), float(blocker_pos[1]), float(blocker_pos[2])
            ))
            
            # 橙色顯示
            blocker_geom.GetDisplayColorAttr().Set([(1.0, 0.5, 0.0)])
            
            # 添加碰撞 API（不加 RigidBodyAPI 避免 GPU API 衝突）
            UsdPhysics.CollisionAPI.Apply(blocker_prim)
            
            # 初始設為不可見（不影響正常偵測）
            blocker_prim.GetAttribute("visibility").Set("invisible")
            
            self.blocker_prims[marker] = {
                'path': blocker_path,
                'prim': blocker_prim,
                'position': blocker_pos,
                'size': blocker_size
            }
            self.blocker_paths_set.add(blocker_path)
            
            print(f"   ✅ Blocker_{marker}: "
                  f"pos=({blocker_pos[0]:.4f}, {blocker_pos[1]:.4f}, {blocker_pos[2]:.4f}), "
                  f"size={blocker_size:.3f}m, dist={distance:.3f}m, 初始=invisible")
        
        print(f"\n   共建立 {len(self.blocker_prims)} 個阻擋器")
        return len(self.blocker_prims) > 0
    
    def set_blocker_visibility(self, marker, visible):
        """切換特定標記的阻擋器可見性
        
        Args:
            marker: 標記名稱 (FM, BM, EM, HM, UM)
            visible: True=可見(阻擋射線), False=不可見(射線穿過)
        """
        if marker not in self.blocker_prims:
            return False
        
        prim = self.blocker_prims[marker]['prim']
        visibility_value = "inherited" if visible else "invisible"
        prim.GetAttribute("visibility").Set(visibility_value)
        return True
    
    def set_all_blockers_visibility(self, visible):
        """切換所有阻擋器的可見性
        
        Args:
            visible: True=可見(阻擋射線), False=不可見(射線穿過)
        """
        visibility_value = "inherited" if visible else "invisible"
        for marker, info in self.blocker_prims.items():
            info['prim'].GetAttribute("visibility").Set(visibility_value)
        
        status = "可見 (阻擋射線)" if visible else "不可見 (射線穿過)"
        print(f"   🔄 所有阻擋器: {status}")
    
    def get_phase_statistics(self):
        """取得當前 raycast 結果的統計數據，包含阻擋器命中分類
        
        Returns:
            dict: {marker: {total, target_hits, blocker_hits, other_hits, miss}}
        """
        if not self.raycast_results:
            return {}
        
        marker_stats = {}
        for marker in ['FM', 'BM', 'EM', 'HM', 'UM']:
            marker_paths = [p for p in self.end_prim_paths if marker.upper() in p.upper()]
            if not marker_paths:
                continue
            
            total = 0
            target_hits = 0
            blocker_hits = 0
            other_hits = 0
            miss = 0
            
            for result in self.raycast_results:
                if result['end_path'] not in marker_paths:
                    continue
                total += 1
                
                if result['is_blocker_hit']:
                    blocker_hits += 1
                elif result['is_target_hit']:
                    target_hits += 1
                elif result['hit_path'] != "None":
                    other_hits += 1
                else:
                    miss += 1
            
            marker_stats[marker] = {
                'total': total,
                'target_hits': target_hits,
                'blocker_hits': blocker_hits,
                'other_hits': other_hits,
                'miss': miss,
                'target_rate': target_hits / total if total > 0 else 0,
                'blocker_rate': blocker_hits / total if total > 0 else 0
            }
        
        return marker_stats
    
    # ========== 以下為原有的輔助方法 ==========
    
    def _get_prim_position(self, prim_path, debug=False):
        """獲取 prim 的世界位置"""
        try:
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                if debug:
                    print(f"      [DEBUG] stage is None!")
                return None
            
            prim = stage.GetPrimAtPath(prim_path)
            
            if not prim.IsValid():
                if debug:
                    print(f"      [DEBUG] prim not valid: {prim_path}")
                return None
            
            from pxr import UsdGeom
            xformable = UsdGeom.Xformable(prim)
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            translation = world_transform.ExtractTranslation()
            
            return np.array([translation[0], translation[1], translation[2]])
            
        except Exception as e:
            if debug:
                print(f"      [DEBUG] _get_prim_position exception for {prim_path}: {type(e).__name__}: {e}")
            return None
    
    def _check_prim_exists(self, prim_path):
        """檢查 prim 是否存在"""
        try:
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(prim_path)
            return prim.IsValid()
        except:
            return False
    
    def _update_raycast_lines(self):
        """更新 raycast 線段 - 從所有起點到所有終點"""
        self.line_starts = []
        self.line_ends = []
        self.ray_info = []
        
        try:
            stage = omni.usd.get_context().get_stage()
            
            print("\n🔍 Checking prim paths...")
            
            # 檢查起點路徑
            print(f"\nStart paths to check: {len(self.start_prim_paths)}")
            for i, start_path in enumerate(self.start_prim_paths):
                exists = self._check_prim_exists(start_path)
                print(f"  {i+1}. {start_path}: {'✅' if exists else '❌'}")
            
            # 檢查終點路徑
            print(f"\nEnd paths to check: {len(self.end_prim_paths)}")
            for i, end_path in enumerate(self.end_prim_paths[:5]):
                exists = self._check_prim_exists(end_path)
                print(f"  {i+1}. {end_path}: {'✅' if exists else '❌'}")
            if len(self.end_prim_paths) > 5:
                print(f"  ... and {len(self.end_prim_paths) - 5} more")
            
            # 獲取所有起點位置
            start_positions = {}
            for start_path in self.start_prim_paths:
                pos = self._get_prim_position(start_path)
                if pos is not None:
                    start_positions[start_path] = pos
            
            print(f"\n✅ Valid start positions: {len(start_positions)}/{len(self.start_prim_paths)}")
            
            if not start_positions:
                print("❌ No valid start positions found")
                print("\n🔧 Debugging info:")
                print(f"  Stage valid: {stage.IsValid()}")
                print(f"  Stage root: {stage.GetRootLayer().realPath}")
                return False
            
            # 獲取所有終點位置
            end_positions = {}
            for end_path in self.end_prim_paths:
                pos = self._get_prim_position(end_path)
                if pos is not None:
                    end_positions[end_path] = pos
            
            print(f"✅ Valid end positions: {len(end_positions)}/{len(self.end_prim_paths)}")
            
            if not end_positions:
                print("❌ No valid end positions found")
                return False
            
            # 為每個起點-終點組合建立射線
            for start_path, start_pos in start_positions.items():
                for end_path, end_pos in end_positions.items():
                    self.line_starts.append(start_pos)
                    self.line_ends.append(end_pos)
                    self.ray_info.append({
                        'start_path': start_path,
                        'end_path': end_path,
                        'start_pos': start_pos,
                        'end_pos': end_pos
                    })
            
            print(f"\n📊 Raycast Lines Created:")
            print(f"  Total rays: {len(self.line_starts)}")
            print(f"  Combinations: {len(start_positions)} starts × {len(end_positions)} ends")
            
            return len(self.line_starts) > 0
            
        except Exception as e:
            print(f"❌ Line update error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _perform_raycast(self):
        """執行射線檢測（動態） - 每次使用當前 prim 位置發射射線"""
        # 這個方法會在每次檢查時重新抓取 prim 的世界位置，避免使用靜態預先計算的位置
        if not self.start_prim_paths or not self.end_prim_paths:
            return

        self.raycast_results = []
        self.ray_info = []

        # 診斷：第一次呼叫時顯示詳細資訊
        self.frame_count += 1
        is_first_call = (self.frame_count <= 1)
        if is_first_call:
            print(f"      [DEBUG] _perform_raycast 首次呼叫")
            print(f"      [DEBUG] start_prim_paths: {len(self.start_prim_paths)}, end_prim_paths: {len(self.end_prim_paths)}")

        idx = 0
        for start_path in self.start_prim_paths:
            start_pos = self._get_prim_position(start_path, debug=is_first_call)
            if start_pos is None:
                if is_first_call:
                    print(f"      [DEBUG] ❌ start_pos=None: {start_path}")
                continue

            if is_first_call and idx == 0:
                print(f"      [DEBUG] ✅ start: {start_path} -> ({start_pos[0]:.4f}, {start_pos[1]:.4f}, {start_pos[2]:.4f})")

            for end_path in self.end_prim_paths:
                end_pos = self._get_prim_position(end_path)
                if end_pos is None:
                    continue

                # 計算方向向量
                direction = end_pos - start_pos
                length = np.linalg.norm(direction)
                if length < 0.001:
                    continue

                normalized_dir = direction / length
                offset = 0.01
                adjusted_start = start_pos + normalized_dir * offset

                # 計算射線終點 (Ray 第二個參數是終點，不是方向!)
                # 使用 2 倍距離確保射線足夠長
                ray_end = adjusted_start + normalized_dir * (length * 2.0)

                # 記錄 ray info 供回調和繪製使用
                self.ray_info.append({
                    'start_path': start_path,
                    'end_path': end_path,
                    'start_pos': start_pos,
                    'end_pos': end_pos
                })

                ray = omni.kit.raycast.query.Ray(tuple(adjusted_start), tuple(ray_end))
                callback = self._make_raycast_callback(idx)
                self.raycast_interface.submit_raycast_query(ray, callback)
                idx += 1
        
        if is_first_call:
            print(f"      [DEBUG] 提交射線數: {idx}, ray_info 長度: {len(self.ray_info)}")
    
    def _make_raycast_callback(self, idx):
        """創建 raycast 回調函數 - 避免索引捕捉問題"""
        def callback(ray, result):
            self._on_raycast_hit(ray, result, idx)
        return callback
    
    def draw_raycast_debug_visualization(self, verbose=False):
        """畫出所有射線的視覺化 - 綠色為命中目標，紅色為未命中"""
        if not self.debug_draw or not self.raycast_results:
            return
        
        self.debug_draw.clear_lines()
        self.debug_draw.clear_points()
        
        if verbose:
            print(f"\n🎨 Drawing {len(self.raycast_results)} rays:")
        
        for i, result in enumerate(self.raycast_results):
            start_pos = result['start_pos']
            end_pos = result['end_pos']
            hit_pos = result['hit_pos']
            is_target_hit = result['is_target_hit']
            
            # 根據是否命中選擇顏色
            if is_target_hit:
                # 綠色線 - 命中目標
                line_color = (0.0, 1.0, 0.0, 1.0)
            else:
                # 紅色線 - 未命中或命中其他物體
                line_color = (1.0, 0.0, 0.0, 1.0)
            
            # 畫射線
            self.debug_draw.draw_lines(
                [tuple(start_pos)],
                [tuple(end_pos if hit_pos is None else hit_pos)],
                [line_color],
                [2.0]
            )
            
            # 起點（藍色）
            self.debug_draw.draw_points(
                [tuple(start_pos)],
                [(0.0, 0.0, 1.0, 1.0)],
                [6.0]
            )
            
            # 終點（紫色）
            self.debug_draw.draw_points(
                [tuple(end_pos)],
                [(1.0, 0.0, 1.0, 1.0)],
                [5.0]
            )
            
            # 命中點（黃色）
            if hit_pos is not None:
                self.debug_draw.draw_points(
                    [tuple(hit_pos)],
                    [(1.0, 1.0, 0.0, 1.0)],
                    [8.0]
                )
    
    def _on_raycast_hit(self, ray, result, idx):
        """處理射線碰撞回調 - 增加阻擋器命中分類"""
        if idx >= len(self.ray_info):
            return
        
        info = self.ray_info[idx]
        start_path = info['start_path']
        end_path = info['end_path']
        start_pos = info['start_pos']
        end_pos = info['end_pos']
        
        if result is not None and result.valid:
            hit_path = str(result.get_target_usd_path())
            hit_pos = result.hit_position
            # 檢查是否命中目標終點
            is_target_hit = end_path in hit_path or hit_path in end_path
            # 檢查是否命中阻擋器
            is_blocker_hit = any(bp in hit_path for bp in self.blocker_paths_set)
            
            self.raycast_results.append({
                'ray_index': idx,
                'start_path': start_path,
                'end_path': end_path,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'hit_path': hit_path,
                'hit_pos': np.array([hit_pos[0], hit_pos[1], hit_pos[2]]) if hit_pos else None,
                'is_target_hit': is_target_hit,
                'is_blocker_hit': is_blocker_hit
            })
        else:
            self.raycast_results.append({
                'ray_index': idx,
                'start_path': start_path,
                'end_path': end_path,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'hit_path': "None",
                'hit_pos': None,
                'is_target_hit': False,
                'is_blocker_hit': False
            })
    
    def analyze_results(self):
        """分析射線檢測結果"""
        if not self.raycast_results:
            print("\n⚠️  No raycast results")
            return {}
        
        # 詳細檢查每個命中
        print("\n🔍 Detailed Raycast Results (first 20):")
        for i, result in enumerate(self.raycast_results[:20]):
            print(f"  Ray {i}:")
            print(f"    End target: {result['end_path'].split('/')[-1]}")
            print(f"    Hit: {result['hit_path']}")
            print(f"    Match: {result['is_target_hit']}")
        
        # 統計每個終點的命中情況
        end_hit_stats = {}
        for end_path in self.end_prim_paths:
            end_hit_stats[end_path] = {
                'total': 0,
                'hit': 0,
                'miss': 0,
                'hit_details': []
            }
        
        for result in self.raycast_results:
            end_path = result['end_path']
            if end_path in end_hit_stats:
                end_hit_stats[end_path]['total'] += 1
                if result['is_target_hit']:
                    end_hit_stats[end_path]['hit'] += 1
                    end_hit_stats[end_path]['hit_details'].append(result['hit_path'])
                else:
                    end_hit_stats[end_path]['miss'] += 1
        
        # 按標記分類統計
        marker_stats = {}
        for marker in ['FM', 'BM', 'EM', 'HM', 'UM']:
            marker_paths = [p for p in self.end_prim_paths if marker.upper() in p.upper()]
            if marker_paths:
                total_rays = sum(end_hit_stats[p]['total'] for p in marker_paths if p in end_hit_stats)
                total_hits = sum(end_hit_stats[p]['hit'] for p in marker_paths if p in end_hit_stats)
                
                # 詳細信息
                print(f"\n📊 {marker} Details:")
                for path in marker_paths:
                    if path in end_hit_stats:
                        stats = end_hit_stats[path]
                        print(f"    {path.split('/')[-1]}: {stats['hit']}/{stats['total']} hits")
                        if stats['hit_details']:
                            print(f"      Hit: {stats['hit_details'][0] if stats['hit_details'] else 'None'}")
                
                hit_rate = total_hits / total_rays if total_rays > 0 else 0
                
                marker_stats[marker] = {
                    'total_rays': total_rays,
                    'total_hits': total_hits,
                    'hit_rate': hit_rate
                }
        
        return marker_stats


def freeze_robot_arm(arm_root_path="/Root/robotarm_base"):
    """暫停機械臂移動 - 簡單方式是讓場景穩定下來"""
    # 這個函數的實質作用是讓模擬步進足夠多次以讓運動停止
    # 在主迴圈中由 --freeze_arm 參數控制是否在開始 raycast 前先暫停
    pass


def main():
    """主測試流程"""
    print("\n" + "="*70)
    print("🔬 USD Raycast Detection Test in Isaac Lab")
    print("="*70)
    print(f"USD File: {args_cli.usd_path}")
    print("="*70)
    
    # 設置模擬上下文
    sim_cfg = sim_utils.SimulationCfg(
        device="cpu",
        dt=1/60.0,
        gravity=(0.0, 0.0, -9.81)
    )
    sim = SimulationContext(sim_cfg)
    
    # 載入 USD 檔案
    print(f"\n📁 Loading USD file: {args_cli.usd_path}")
    
    import os
    if not os.path.exists(args_cli.usd_path):
        print(f"❌ USD file not found: {args_cli.usd_path}")
        return
    
    try:
        stage = omni.usd.get_context().get_stage()
        from pxr import Sdf
        
        # 直接打開舞台
        omni.usd.get_context().open_stage(args_cli.usd_path)
        stage = omni.usd.get_context().get_stage()
        
        # 等待舞台載入
        for i in range(10):
            sim.step()
            time.sleep(0.1)
        
        print("✅ USD file opened successfully")
        print(f"   Root prim: {stage.GetRootLayer().realPath if stage else 'None'}")
        
    except Exception as e:
        print(f"❌ Failed to load USD file: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========== 預建阻擋器 (必須在 sim.reset() 之前) ==========
    # Raycast BVH 在 sim.reset() 時建立，之後不動態更新
    # 因此阻擋器幾何體必須在 reset 前存在於場景中
    
    tester = USDRaycastTester({"supported_markers": ['FM', 'BM', 'EM', 'HM', 'UM']})
    
    print("\n🔧 Pre-building raycast blockers (before sim.reset)...")
    blockers_created = tester.create_blockers_for_markers(stage)
    if blockers_created:
        print("✅ Blockers created successfully (initially invisible)")
    else:
        print("⚠️  No blockers created - will run normal detection only")
    
    # 重置模擬 - BVH 現在包含阻擋器幾何體
    sim.reset()
    print("✅ Simulation reset (BVH includes pre-built blockers)")
    
    # 多步模擬以確保 USD 完全初始化
    print("⏳ Waiting for USD initialization...")
    for i in range(60):
        sim.step()
    print("✅ USD initialization complete")
    
    # 可選：凍結機械臂防止 marker 移動
    if args_cli.freeze_arm:
        print("\n⏳ Waiting for planned trajectory to complete...")
        print("   (This allows the robot arm to finish its motion)")
        
        # 根據軌跡時間等待（通常手術軌跡需要 10-20 秒）
        wait_time = 15.0  # 15 秒
        wait_steps = int(wait_time * 60)  # 按 60fps 計算
        
        for i in range(wait_steps):
            sim.step()
            if i % 60 == 0:
                elapsed = i / 60.0
                print(f"   ... {wait_time - elapsed:.0f}s remaining")
        
        print("✅ Trajectory complete - marker positions are now stable")
    
    # 初始化 raycast interface (在 sim.reset 之後)
    if not tester.initialize():
        print("❌ Failed to initialize raycast interface")
        return
    
    # 更新射線（驗證 prim 路徑是否有效）
    print("\n🔎 Scanning USD structure...")
    
    if not tester._update_raycast_lines():
        print("❌ Failed to update raycast lines")
        return
    
    # 驗證：在 phase 測試前先做一次快速 raycast 確認回調是否運作
    print("\n🧪 Quick raycast validation test...")
    print(f"   raycast_interface: {tester.raycast_interface}")
    
    # 手動測試一條射線
    test_start = tester._get_prim_position(tester.start_prim_paths[0], debug=True)
    test_end = tester._get_prim_position(tester.end_prim_paths[0], debug=True)
    print(f"   Test start pos: {test_start}")
    print(f"   Test end pos: {test_end}")
    
    if test_start is not None and test_end is not None:
        direction = test_end - test_start
        length = np.linalg.norm(direction)
        print(f"   Distance: {length:.4f}m")
        
        if length > 0.001:
            normalized_dir = direction / length
            adjusted_start = test_start + normalized_dir * 0.01
            ray_end_pt = adjusted_start + normalized_dir * (length * 2.0)
            
            test_result = {'valid': False, 'path': '', 'called': False}
            def test_callback(ray, result):
                test_result['called'] = True
                if result is not None and result.valid:
                    test_result['valid'] = True
                    test_result['path'] = str(result.get_target_usd_path())
            
            from omni.kit.raycast.query import Ray as RayQuery
            test_ray = RayQuery(tuple(adjusted_start), tuple(ray_end_pt))
            tester.raycast_interface.submit_raycast_query(test_ray, test_callback)
            
            # 步進讓回調處理
            sim.step()
            time.sleep(0.05)
            
            print(f"   Callback called: {test_result['called']}")
            print(f"   Hit valid: {test_result['valid']}")
            print(f"   Hit path: {test_result['path']}")
    else:
        print("   ❌ Cannot get test positions!")
    
    # 確認 _perform_raycast 也能取得位置
    print(f"\n   Testing _perform_raycast positions...")
    tester.frame_count = 0  # 重設計數器讓診斷輸出
    tester._perform_raycast()
    sim.step()
    time.sleep(0.05)
    print(f"   After _perform_raycast: raycast_results={len(tester.raycast_results)}, ray_info={len(tester.ray_info)}")
    
    # ===================================================================
    # 三階段測試：驗證阻擋器可見性切換對 Raycast 的影響
    # ===================================================================
    
    def run_detection_phase(sim_ctx, tester_obj, phase_name, num_rounds=5, steps_per_round=10):
        """執行一個偵測階段，收集多輪 raycast 結果
        
        Args:
            sim_ctx: SimulationContext
            tester_obj: USDRaycastTester
            phase_name: 階段名稱
            num_rounds: 偵測輪數
            steps_per_round: 每輪模擬步數
        Returns:
            dict: 該階段的聚合統計
        """
        all_phase_results = []
        
        for round_idx in range(num_rounds):
            # 模擬步進
            for _ in range(steps_per_round):
                sim_ctx.step()
            
            # 執行動態 raycast
            tester_obj._perform_raycast()
            
            # 關鍵：submit_raycast_query 的回調需要經過一次更新周期才會被執行
            # 必須 step 一次讓回調被處理，否則 raycast_results 會是空的
            sim_ctx.step()
            time.sleep(0.01)  # 額外等待
            
            # 繪製視覺化
            tester_obj.draw_raycast_debug_visualization(verbose=False)
            
            # 收集本輪結果
            if round_idx == 0:
                print(f"      [Round {round_idx+1}] raycast_results count: {len(tester_obj.raycast_results)}")
            all_phase_results.extend(tester_obj.raycast_results)
        
        # 聚合統計
        stats = {}
        for marker in ['FM', 'BM', 'EM', 'HM', 'UM']:
            marker_paths = [p for p in tester_obj.end_prim_paths if marker.upper() in p.upper()]
            if not marker_paths:
                continue
            
            total = 0
            target_hits = 0
            blocker_hits = 0
            other_hits = 0
            miss = 0
            
            for result in all_phase_results:
                if result['end_path'] not in marker_paths:
                    continue
                total += 1
                
                if result.get('is_blocker_hit', False):
                    blocker_hits += 1
                elif result['is_target_hit']:
                    target_hits += 1
                elif result['hit_path'] != "None":
                    other_hits += 1
                else:
                    miss += 1
            
            stats[marker] = {
                'total': total,
                'target_hits': target_hits,
                'blocker_hits': blocker_hits,
                'other_hits': other_hits,
                'miss': miss,
                'target_rate': target_hits / total if total > 0 else 0,
                'blocker_rate': blocker_hits / total if total > 0 else 0
            }
        
        # 顯示統計
        print(f"\n   📊 {phase_name} 統計:")
        for marker, s in stats.items():
            if s['total'] > 0:
                print(f"      {marker}: 總射線={s['total']}, "
                      f"命中目標={s['target_hits']}({s['target_rate']*100:.1f}%), "
                      f"命中阻擋器={s['blocker_hits']}({s['blocker_rate']*100:.1f}%), "
                      f"命中其他={s['other_hits']}, 未命中={s['miss']}")
        
        return stats
    
    # ---------- Phase 1: 正常偵測 (阻擋器不可見) ----------
    print("\n" + "="*70)
    print("📍 Phase 1: 正常偵測 (阻擋器 invisible)")
    print("="*70)
    tester.set_all_blockers_visibility(False)
    
    # 先跑幾步讓可見性生效
    for _ in range(5):
        sim.step()
    
    phase1_stats = run_detection_phase(sim, tester, "Phase 1 正常偵測")
    
    # ---------- Phase 2: 阻擋測試 (阻擋器可見) ----------
    print("\n" + "="*70)
    print("🚧 Phase 2: 阻擋測試 (阻擋器 visible)")
    print("="*70)
    tester.set_all_blockers_visibility(True)
    
    for _ in range(5):
        sim.step()
    
    phase2_stats = run_detection_phase(sim, tester, "Phase 2 阻擋測試")
    
    # ---------- Phase 3: 恢復偵測 (阻擋器再次不可見) ----------
    print("\n" + "="*70)
    print("📍 Phase 3: 恢復偵測 (阻擋器 invisible)")
    print("="*70)
    tester.set_all_blockers_visibility(False)
    
    for _ in range(5):
        sim.step()
    
    phase3_stats = run_detection_phase(sim, tester, "Phase 3 恢復偵測")
    
    # ===================================================================
    # 最終比較報告
    # ===================================================================
    print("\n" + "="*70)
    print("📋 三階段比較報告")
    print("="*70)
    
    print(f"\n{'標記':<6} {'Phase1 目標命中率':<20} {'Phase2 阻擋命中率':<20} {'Phase3 目標命中率':<20} {'結論'}")
    print("-" * 86)
    
    all_blocking_works = True
    for marker in ['FM', 'BM', 'EM', 'HM', 'UM']:
        p1 = phase1_stats.get(marker, {})
        p2 = phase2_stats.get(marker, {})
        p3 = phase3_stats.get(marker, {})
        
        p1_rate = p1.get('target_rate', 0) * 100
        p2_blocker = p2.get('blocker_rate', 0) * 100
        p2_target = p2.get('target_rate', 0) * 100
        p3_rate = p3.get('target_rate', 0) * 100
        
        # 判定阻擋是否有效
        if p1.get('total', 0) == 0:
            verdict = "⚠️ 無資料"
        elif p2_blocker > 50 and p2_target < p1_rate * 0.5:
            verdict = "✅ 阻擋有效"
        elif p2_blocker > 0:
            verdict = "⚠️ 部分有效"
        else:
            verdict = "❌ 阻擋無效"
            all_blocking_works = False
        
        print(f"{marker:<6} {p1_rate:>6.1f}%              {p2_blocker:>6.1f}%              {p3_rate:>6.1f}%              {verdict}")
    
    print("\n" + "="*70)
    if all_blocking_works and any(phase1_stats.get(m, {}).get('total', 0) > 0 for m in ['FM', 'BM', 'EM', 'HM', 'UM']):
        print("🎉 方案1驗證成功！預建阻擋器 + 可見性切換方案在 NDI 場景中有效！")
        print("\n💡 使用方式:")
        print("   tester.set_blocker_visibility('BM', True)   # 阻擋 BM 標記的射線")
        print("   tester.set_blocker_visibility('BM', False)  # 恢復 BM 偵測")
        print("   tester.set_all_blockers_visibility(True)    # 阻擋所有標記")
    else:
        print("⚠️  測試結果需要檢查，部分標記的阻擋效果不如預期")
        print("\n可能原因:")
        print("   1. 阻擋器尺寸或位置需要調整")
        print("   2. 標記球體太分散")
        print("   3. 射線路徑上有其他遮擋物")
    print("="*70 + "\n")
    
    # 保持模擬運行供觀察 (阻擋器保持可見以便觀察)
    print("⏳ Keeping simulation open for observation...")
    print("   Blockers are now VISIBLE (orange cubes) for inspection")
    print("   Press Ctrl+C to close, or wait for timeout...")
    tester.set_all_blockers_visibility(True)
    
    observation_time = args_cli.observation_time
    obs_start = time.time()
    obs_step = 0
    
    while simulation_app.is_running() and time.time() - obs_start < observation_time:
        sim.step()
        obs_step += 1
        
        # 持續 raycast 和繪製
        if obs_step % 6 == 0:
            tester._perform_raycast()
            time.sleep(0.001)
            tester.draw_raycast_debug_visualization(verbose=False)
        
        time.sleep(1/60.0)
    
    print("\n✅ Observation period ended\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🔚 Closing simulation...")
        simulation_app.close()
