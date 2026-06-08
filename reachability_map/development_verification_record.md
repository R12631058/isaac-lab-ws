# 開發、修復與驗證紀錄 (Development, Bug Fixes & Verification Record)

本文件詳細記錄了 **RASSCMAP 協同模擬系統** 的完整開發軌跡、所有已開發與重構腳本的對照清單，以及各個開發時期中解決的關鍵物理、控制與數學 Bug。本紀錄旨在為系統的後續維護、調試與臨床整合提供完整的技術 Ledger。

---

## 1. 完整已開發檔案與執行指令清單 (Comprehensive File & Command Ledger)

系統涵蓋 **Isaac Lab 實驗端**與 **Isaac Sim Standalone Replay 運行端** 兩個主要工作空間。以下為所有已開發檔案的詳細清單，包含其主要功能、相依關係、執行指令與參數：

### 1.1 Isaac Lab 實驗工作空間 (Conda 環境: `env_isaaclab` | 路徑: `C:/Users/RMML/IsaacLab`)

此工作空間主要負責底座的批量平行優化搜尋、資料庫彙整計算以及前端 Tkinter GUI 的查詢。

| 檔案名稱與連結 | 主要功能說明 | 相依檔案與模組 | 執行環境與指令 | 關鍵參數與說明 | 狀態 (Active / Legacy) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [parallel_base_optimizer.py](file:///C:/Users/RMML/IsaacLab/scripts/isaaclab_ws/reachability_map/parallel_base_optimizer.py) | 2D 網格底座優化搜尋（離線計算）。計算各底座位置對 70 個路徑點的 RI、AMI 等指標。 | `isaaclab` 套件<br>USD 場景 | `.\isaaclab.bat -p scripts/isaaclab_ws/reachability_map/parallel_base_optimizer.py` | `--max_waypoints 70`<br>`--pos_threshold 0.0015` | **Active** (用於全面搜尋) |
| [parallel_base_optimizer_linear.py](file:///C:/Users/RMML/IsaacLab/scripts/isaaclab_ws/reachability_map/parallel_base_optimizer_linear.py) | 1D X軸網格底座優化搜尋（固定 Y=0.0，共 15 個並行 envs）。針對指定腫瘤區塊 Block 中心計算最優底座。 | `isaaclab` 套件<br>USD 檔案 | `.\isaaclab.bat -p scripts/isaaclab_ws/reachability_map/parallel_base_optimizer_linear.py` | `--block_center X Y Z`<br>`--block_id N` | **Active** (1D 快速優化核心) |
| [build_cmap_database.py](file:///C:/Users/RMML/IsaacLab/scripts/isaaclab_ws/reachability_map/build_cmap_database.py) | 遍歷並整合 `heatmap/blocks/` 下所有區塊的優化 JSON，計算 $T_{NDI\_UM}$ 矩陣，導出最終整合資料庫。 | `heatmap/blocks/*.json`<br>`numpy` | `python scripts/isaaclab_ws/reachability_map/build_cmap_database.py` | `--block_dir <path>` | **Active** (資料庫編譯器) |
| [optimal_base_finder.py](file:///C:/Users/RMML/IsaacLab/scripts/isaaclab_ws/reachability_map/optimal_base_finder.py) | 基於 Tkinter 製作的查詢 GUI。提供使用者輸入腫瘤座標，即時查詢最近區塊並顯示 NDI 定位資訊，並透過 "Open 3D CMAP Viewer" 按鈕自動背景執行 `python.bat` 拉起 Isaac Sim。 | `cmap_database.json`<br>`tkinter` | `python scripts/isaaclab_ws/reachability_map/optimal_base_finder.py` | (無，主要透過 GUI 輸入座標與按鈕操作) | **Active** (查詢與啟動前端) |
| [update_optimal_jsons.py](file:///C:/Users/RMML/IsaacLab/scripts/isaaclab_ws/reachability_map/update_optimal_jsons.py) | 批次修正腳本。重新解析原始 CSV 並調整排序邏輯，將成功率（cs）作為第一優先級，提升最優底座的實際可靠性。 | `heatmap/blocks/` | `python scripts/isaaclab_ws/reachability_map/update_optimal_jsons.py` | (無，自動遍歷 10 個 blocks) | **Active** (資料維護工具) |
| [generate_task_waypoints.py](file:///C:/Users/RMML/IsaacLab/scripts/isaaclab_ws/reachability_map/generate_task_waypoints.py) | 以數學公式在腫瘤周圍生成 10×7 個半球形導航點。 | `numpy`, `json` | `.\isaaclab.bat -p scripts/isaaclab_ws/reachability_map/generate_task_waypoints.py` | `--num_envs 1` | **Legacy** (目前已改由直接在 USD 中手動標註 70 個 Waypoints) |
| [generate_plots_from_csv.py](file:///C:/Users/RMML/IsaacLab/scripts/isaaclab_ws/reachability_map/generate_plots_from_csv.py) | 讀取優化結果 CSV，繪製可達性直方圖與散佈圖，便於分析各底座性能。 | `matplotlib`, `pandas` | `python scripts/isaaclab_ws/reachability_map/generate_plots_from_csv.py` | `--csv <csv_path>` | **Active** (數據分析工具) |
| [generate_rasscmap_from_csv.py](file:///C:/Users/RMML/IsaacLab/scripts/isaaclab_ws/reachability_map/generate_rasscmap_from_csv.py) | 讀取優化結果 CSV，生成 2D/3D 可達性熱圖（Rasscmap）圖片（PNG）。 | `matplotlib` | `python scripts/isaaclab_ws/reachability_map/generate_rasscmap_from_csv.py` | `--csv <csv_path>` | **Active** (熱圖生成器) |
| [usd_waypoints_ik_showcase.py](file:///C:/Users/RMML/IsaacLab/scripts/isaaclab_ws/reachability_map/usd_waypoints_ik_showcase.py) | 機械手臂循跡測試。使手臂在單一環境下循序訪問 USD 中所有的 70 個路徑點，用於快速驗證。 | `isaaclab` 套件 | `.\isaaclab.bat -p scripts/isaaclab_ws/reachability_map/usd_waypoints_ik_showcase.py` | `--max_waypoints 70`<br>`--pos_threshold 0.001` | **Active** (循跡功能演示) |
| [debug_ik.py](file:///C:/Users/RMML/IsaacLab/scripts/isaaclab_ws/reachability_map/debug_ik.py) | 極簡的馬可夫決策環境（Env）測試，用於調試 IK 矩陣、座標轉換與控制器臨界狀態。 | `isaaclab` 套件 | `.\isaaclab.bat -p scripts/isaaclab_ws/reachability_map/debug_ik.py` | (無，主要作為測試沙盒) | **Active** (除錯工具) |
| [test_coordinate_v2.py](file:///C:/Users/RMML/IsaacLab/scripts/isaaclab_ws/reachability_map/test_coordinate_v2.py) | 座標轉換與 $T_{NDI\_UM}$ 數學驗證腳本，測試 local-to-world 運算與 NDI 空間轉換。 | `pxr` USD API | `python scripts/isaaclab_ws/reachability_map/test_coordinate_v2.py` | (無，直接在腳本中寫死測試 Prim) | **Active** (數學驗證工具) |
| [run_blocks_batch.bat](file:///C:/Users/RMML/IsaacLab/scripts/isaaclab_ws/reachability_map/run_blocks_batch.bat) | 批次執行批次優化的 Windows 批次檔。自動循序執行 Block 0 ~ 9 的並行 1D 優化搜尋。 | `parallel_base_optimizer_linear.py` | `.\scripts\isaaclab_ws\reachability_map\run_blocks_batch.bat` | (自動依序代入 `--block_id` 和 `--block_center`) | **Active** (批量優化批次) |
| [run_blocks_batch.ps1](file:///C:/Users/RMML/IsaacLab/scripts/isaaclab_ws/reachability_map/run_blocks_batch.ps1) | 批次執行批次優化的 PowerShell 腳本。 | `parallel_base_optimizer_linear.py` | `.\scripts\isaaclab_ws\reachability_map\run_blocks_batch.ps1` | 同上 | **Active** (批量優化 PowerShell) |
| [VISUALIZATION_HANDOVER.md](file:///C:/Users/RMML/IsaacLab/scripts/isaaclab_ws/reachability_map/VISUALIZATION_HANDOVER.md) | 可視化管線的交接說明文檔。 | — | （文檔閱讀） | — | **Active** (交接文件) |
| [cmap_runs_summary.md](file:///C:/Users/RMML/IsaacLab/scripts/isaaclab_ws/reachability_map/cmap_runs_summary.md) | 歷史執行記錄摘要，記載每一次 Block 優化搜尋的時間戳記、最佳底座坐標、對應 CSV 檔案路徑與執行指令。 | — | （文檔閱讀） | — | **Active** (運作紀錄檔) |

### 1.2 Isaac Sim Standalone 工作空間 (內建環境: `python.bat` | 路徑: `C:/Users/RMML/AppData/Local/ov/pkg/4.5.0`)

此工作空間不相依於 `isaaclab` 套件，直接使用 Isaac Sim 內建的 python 核心，專注於 3D 視覺化 Replay、滑行變速動畫、以及 Lula IK 針尖精確循跡。

| 檔案名稱與連結 | 主要功能說明 | 相依檔案與模組 | 執行環境與指令 | 關鍵參數與說明 | 狀態 (Active / Legacy) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [sim_rasscmap_interactive_viewer.py](file:///C:/Users/RMML/AppData/Local/ov/pkg/4.5.0/standalone_ws/rasscmap/sim_rasscmap_interactive_viewer.py) | Standalone UI Replay 檢視器。包含 `omni.ui` 側邊面板、CMAP 覆蓋球體繪製、高亮最近點、底座 Sine 平滑滑行、Lula IK 針尖均速 Ease-In-Out 連續對焦。 | `LulaKinematicsSolver`<br>USD 檔案<br>Lab CSV 輸出 | `python.bat standalone_ws/rasscmap/sim_rasscmap_interactive_viewer.py` | `--backup_csv <path>`<br>`--opt_csv <path>`<br>`--tumor_pos X Y Z`<br>`--headless` | **Active** (核心 Replay 驗證程式) |
| [optimal_base_finder_sim.py](file:///C:/Users/RMML/AppData/Local/ov/pkg/4.5.0/standalone_ws/rasscmap/optimal_base_finder_sim.py) | Standalone 命令列查詢工具。在無 GUI 下檢索最優底座、計算 NDI 轉換並一鍵拉起 3D Viewer（主要用於測試）。 | `cmap_database.json` | `python.bat standalone_ws/rasscmap/optimal_base_finder_sim.py` | `--tumor_pos X Y Z`<br>`--launch` | **Active** (命令列查詢) |
| [MIGRATION_RECORD.md](file:///C:/Users/RMML/AppData/Local/ov/pkg/4.5.0/standalone_ws/rasscmap/MIGRATION_RECORD.md) | 移植說明文件。記錄了從 Isaac Lab 移植到 Isaac Sim Standalone 的邊界、架構重構原因以及驗證進度。 | — | （文檔閱讀） | — | **Active** (移植紀錄) |
| [README.md](file:///C:/Users/RMML/AppData/Local/ov/pkg/4.5.0/standalone_ws/rasscmap/README.md) | Standalone 執行說明書。提供 Standalone 腳本的基本說明與執行範例。 | — | （文檔閱讀） | — | **Active** (運作說明) |

---

## 2. 關鍵問題與 Bug 修復歷程 (Detailed Ledger of Resolved Issues)

在專案開發、優化與 3D 視覺化整合過程中，系統遇到了多項嚴重的物理、數學與控制學問題。以下是這些問題的詳細技術診斷與修復歷程：

### 2.1 針尖 (needle_tip) 追蹤 17.1 cm 巨大偏差與 USD 縮放累乘 Bug

#### 2.1.1 問題現象 (Symptom)
在 Standalone 檢視器中執行 `Follow Target` 時，機械手臂能順利求解並進行運動，但手臂末端（flange）卻直接貼到了腫瘤標記球體上，使得安裝在末端上的針頭（needle）直接穿過了腫瘤，針尖（needle_tip）與腫瘤實際位置產生了 **17.1 cm (171 毫米)** 的巨大空間偏差。

#### 2.1.2 技術診斷與數學原理 (Diagnosis & Math)
這是由於 **USD 縮放因子在旋轉矩陣中累乘** 所導致的：
1. 在 USD 場景中，針頭模型（needle prim）帶有 `0.02`（1/50）的縮放屬性（Scale）。
2. 當使用 `UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)` 獲取 4x4 矩陣時，該矩陣的 3x3 旋轉矩陣部分（Column 0, 1, 2）隱含了該縮放比例。
3. 我們需要計算從 flange 到 needle_tip 的局部變換，公式如下：
   $$T_{flange\_tip} = T_{World\_flange}^{-1} \cdot T_{World\_tip}$$
4. 當我們將未經歸一化的旋轉矩陣連續相乘時，`0.02` 的縮放因子被多次乘算：
   $$0.02 \times 0.02 = 0.0004$$
   這使得解析出的局部偏移向量（Offset Vector）大小從原本的 **28.5 公分** 被縮減為極微小的 **0.0004 倍**（即 0.1 毫米），在物理計算中近乎為零。Lula IK 求解器在缺乏偏移向量的情況下，便將 `flange` 直接當作針尖去碰觸腫瘤，進而導致針尖偏離目標 17.1 cm。

#### 2.1.3 程式碼修復對照 (Code Implementation Diff)
*   **Before (錯誤寫法 - 未進行旋轉矩陣列歸一化)**:
    ```python
    def gf_matrix_rotation_np(mat):
        # 直接提取 3x3 矩陣，保留了 USD 的 Scale 縮放係數
        return np.array([
            [mat[0][0], mat[1][0], mat[2][0]],
            [mat[0][1], mat[1][1], mat[2][1]],
            [mat[0][2], mat[1][2], mat[2][2]],
        ], dtype=float)
    ```
*   **After (修復寫法 - 對每一列向量進行歸一化以抽離縮放)**:
    ```python
    def gf_matrix_rotation_np(mat):
        rot = np.array([
            [mat[0][0], mat[1][0], mat[2][0]],
            [mat[0][1], mat[1][1], mat[2][1]],
            [mat[0][2], mat[1][2], mat[2][2]],
        ], dtype=float)
        # 對 3x3 旋轉矩陣的每一列（即軸向量）進行 L2 範數歸一化，強制使其成為標準正交基
        norms = np.linalg.norm(rot, axis=0)
        norms = np.where(norms == 0, 1.0, norms)
        return rot / norms
    ```

#### 2.1.4 驗證成果 (Verification)
修復後，flange -> needle_tip 的局部偏移量正確讀取為 `[0.0, 0.0, 0.285]` 米。再次運行 Lula IK，機械手臂能夠極為精準地在腫瘤表面貼合，針尖殘差誤差值從 17.1 cm 驟降至 **0.0064 cm (0.064 mm)**，達到高精度的臨床導航驗證要求。

---

### 2.2 切換環境底座後的手臂劇烈抖動與 Lula IK 求解失效 Bug

#### 2.2.1 問題現象 (Symptom)
在 Replay GUI 中切換不同的 Environment（例如從 Env 00 改選 Env 01）後，機器人底座在視覺上被傳送到新位置。但此時一旦按下 `Follow Target`，機械手臂會在原地劇烈抖動、四處亂竄，而 Lula 求解器拋出殘差無法收斂與限位超限的錯誤。

#### 2.2.2 技術診斷與物理引擎限制 (Diagnosis & Physics Engine)
這是由於 **USD 渲染座標與 PhysX 剛體約束 (Articulation Base) 產生衝突**：
1. 機械手臂的根節點 `/Root/robotarm_base/robotarm_base/tm5_700/link_0` 在 USD 中透過 Fixed Joint 被綁定在父級變換 `/Root/robotarm_base` 下。
2. 在模擬運行狀態（Simulation active）下，直接修改 USD 中的底座平移操作，僅能改變 USD 視圖的渲染位置。此時，**PhysX 物理引擎內部的 Articulation 剛體依然停留在舊環境位置**。
3. 此外，如果單獨使用 Tensor API 寫入機器人根剛體狀態，底座物理被傳送了，但 USD 的 Fixed Joint 關節約束物件卻依然留在舊位置，導致物理引擎在每個計算步中，試圖將機器人底座暴力拉回舊位置。
4. 這種巨大的物理拉扯力（物理引擎與 Tensor API 互寫衝突）引發底座嚴重偏離軌道、受力暴增而強烈抖動，雅可比矩陣數據徹底錯亂，導致 Lula 逆運動學求解器輸入無效而失效。

#### 2.2.3 程式碼修復對照 (Code Implementation Diff)
我們在 [sim_rasscmap_interactive_viewer.py](file:///C:/Users/RMML/AppData/Local/ov/pkg/4.5.0/standalone_ws/rasscmap/sim_rasscmap_interactive_viewer.py) 中的底座滑行、初始化以及 `follow()` 開始時，加入了完整的 PhysX Teleport 與 Lula Solver 座標對齊機制：
```python
# 1. 取得 USD 當前最新的 Base 世界矩陣
robot_mat = get_prim_world_transform_matrix(STAGE, args.robot_prim_path)
if robot_mat is not None:
    r_base_w = gf_matrix_rotation_np(robot_mat)
    robot_base_pos = matrix_translation_np(robot_mat)
    robot_base_ori = rot_matrix_to_quat_wxyz(r_base_w)
    
    # 2. 強制在 PhysX 中將 Articulation teleport 到與 USD 一致的 pose
    self.robot.set_world_pose(position=robot_base_pos, orientation=robot_base_ori)
    
    # 3. 讓物理引擎模擬步前進 5 步以穩定物理狀態，避免約束力突變
    for _ in range(5):
        self.world.step(render=False)
        
    # 4. 讀取 PhysX 穩定後的 pose，並將其寫入 LulaKinematicsSolver 作為參考座標原點
    phys_pos, phys_ori = self.robot.get_world_pose()
    self.solver.set_robot_base_pose(phys_pos, phys_ori)
```

#### 2.2.4 驗證成果 (Verification)
加入該機制後，無論底座是在滑行中、或是頻繁切換不同的環境底座位置，物理引擎與 Lula 求解器對機器人底座的世界座標認知均保持 100% 一致。切換底座後執行 `Follow Target`，機械手臂能以極為穩健的狀態抵達，完全消除了物理拉扯與手臂發狂般抖動的現象。

---

### 2.3 手臂運動「指數變速」與「突發急加速」Bug

#### 2.3.1 問題現象 (Symptom)
手臂在朝向腫瘤中心移動的動畫中，速度極不均勻且不連貫。在按下 follow 按鈕的最初幾影格移動極慢，但在中途會突然以不合常理的高速急衝（Catch-up acceleration）撞向目標。同時，如果手臂在行進過程中經過奇異點，哪怕只有一影格 Lula 解算失敗，整個運動流程就會直接中斷卡死。

#### 2.3.2 技術診斷與控制學原理 (Diagnosis & Control Theory)
這有兩個根本原因：
1. **反覆即時求解 IK**：舊系統在手臂運動的「每一影格」都即時去算一次 Lula IK。當手臂在插值過渡姿態時，由於 warm start 種子不好或局部奇異點，若有任一步求解失敗，程式會直接 break 中斷，使手臂卡在半路。
2. **關節 PD 控制器滯後追趕**：舊系統將插值後的關節角度，透過 `apply_action` 作為控制目標輸入給關節 PD 驅動器（Joint PD Drive）。由於機器人連桿具有物理質量與動態慣性，PD 驅動器在跟隨隨時間變化的目標時會有物理滯後（Lag）。當落後累積誤差過大時，PD 控制器的扭矩（Torque）呈比例暴增，引發手臂突然急加速追趕，導致視覺上的速度突變與抖動。

#### 2.3.3 重構與修復解決方案 (Refactoring & Solution)
為了解決此問題，我們對手臂追蹤運動架構進行了全面重構，捨棄了即時反覆求解與 PD 扭矩驅動，採用 **「單次求解 + 正弦 Ease-In-Out 均速插值 + 直寫關節角度 (set_joint_positions)」** 的架構：
*   **單次求解**：按下 Follow 時，僅在起點進行一次性的 Lula IK 求解，獲取精確的最終終點關節 `q_target`。
*   **正弦 Ease-In-Out 插值**：以 Sine Ease-In-Out 對起點關節 `q_start` 與 `q_target` 進行關節空間平滑插值：
    $$\theta(t) = q_{start} + (q_{target} - q_{start}) \cdot \left(0.5 \cdot (1 - \cos(t \cdot \pi))\right)$$
    其中 $t \in [0, 1]$。其速度導數在起點 $t=0$ 與終點 $t=1$ 皆為 0，中段均勻，能保證速度完美平滑。
*   **直寫關節 (Direct Joint Teleportation)**：在動畫迴圈中，利用 `robot.set_joint_positions()` 直接對機器人關節寫入插值角度，繞過 PD 驅動器的物理滯後與追趕現象。
*   **程式碼實作片段**:
    ```python
    # 獲取起始與目標關節
    q_start_posture = q_current.copy()
    q_target = np.array(joint_positions, dtype=float)
    
    # 在 ARM_FOLLOW_TOTAL_STEPS 步內進行正弦插值
    for step in range(ARM_FOLLOW_TOTAL_STEPS):
        t = (step + 1) / ARM_FOLLOW_TOTAL_STEPS
        # 正弦 Ease-In-Out 公式
        t_smooth = 0.5 * (1.0 - math.cos(t * math.pi))
        
        q_current = q_start_posture + (q_target - q_start_posture) * t_smooth
        # 繞過 PD 控制器，直接對物理關節寫入目標角度，消除追趕抖動
        self.robot.set_joint_positions(q_current)
        self.world.step(render=True)
    ```

#### 2.3.4 驗證成果 (Verification)
重構後，手臂能以**極度滑順、無任何加速抖動**的均勻速度行進。運動路徑在指定時間（預設 1.5 秒）內精準完成，且最終針尖貼合殘差誤差值僅 **0.0064 cm (0.064 毫米)**，視覺質感與工程精度均達到最優狀態。

---

### 2.4 最優底座（Optimal Base）排序優先權演算法漏洞

#### 2.4.1 問題現象 (Symptom)
優化器產出的 JSON 和整合資料庫在查詢某些腫瘤座標時，推薦的最優底座位置載入後，畫面上 70 個 Waypoints 中有幾個邊角點卻是「完全不可達」的（在 3D Viewer 中呈現純紅色球體）。然而，這個底座的 `fg_score` 分數卻很高，排在其他「70個球體皆完全可達」的底座之前。

#### 2.4.2 技術診斷與演算法漏洞 (Diagnosis & Algorithm)
底座優化評估指標的原始排序邏輯過於偏重 fine-grained 靈巧度得分 `fg_score`。
1. `fg_score` 是 70 個 Waypoints 靈巧度的乘積。如果底座在其中 68 個點的 manipulability 極高，乘積會非常大；但有 2 個點因為在邊角或奇異點完全不可達（實際成功率為 0，但在 CSV 的浮點運算中可能因為乘積指標被平滑掉，或者 `sr_min` 雖然為 0，但排序先比對了其他平均指標）。
2. 這導致系統錯誤地將「有紅色球體但其餘點靈巧度極高」的底座，排在「70個球體皆完全可達（全部藍色）但靈巧度一般」的底座之前。

#### 2.4.3 程式碼修復對照 (Code Implementation Diff)
我們寫了 [update_optimal_jsons.py](file:///C:/Users/RMML/IsaacLab/scripts/isaaclab_ws/reachability_map/update_optimal_jsons.py) 腳本，強制將 **「100% 全可達 (success_rate == 1.0 或 sr_min == 1)」** 作為最優先的一級排序鍵值，其次才是成功率與細粒度操作度分數 `fg_score`：
*   **Before (原始排序邏輯 - 單純比較 fine-grained 分數)**:
    ```python
    best_row = max(rows, key=lambda r: float(r["fg_score"]))
    ```
*   **After (修復排序邏輯 - 優先確保 sr_min == 1 即 100% 可達性，才進行靈巧度評分)**:
    ```python
    # 優先排序 sr_min == 1.0 (全可達)，再比對成功率與 fg_score
    best_row = max(rows, key=lambda r: (
        int(float(r["sr_min"]) >= 1.0),
        float(r["success_rate"]),
        float(r["fg_score"])
    ))
    ```

#### 2.4.4 驗證成果 (Verification)
使用修正後的排序邏輯重新處理 10 個 Block 的最優底座後，資料庫中推薦的最優底座在 3D Viewer 載入時，所有的 70 個 Waypoint 均展現了 **100% 全藍色（全可達）** 的完美覆蓋，徹底消除了臨床操作中的「死角」盲點。

---

### 2.5 Lula IK 求解手肘朝向 (Joint 1) 姿態分支衝突與天花板碰撞 Bug

#### 2.5.1 問題現象 (Symptom)
在 Follow Target 計算中，機械手臂在某些環境下會繞一大圈，並以「肘部高高揚起（Joint 1 旋轉為正角度）」的奇異姿態抵達腫瘤。這會導致機械手臂撞擊到 USD 場景中上方的蓋板，引發關節扭曲，甚至造成物理引擎碰撞穿透與 IK 崩潰。

#### 2.5.2 技術診斷與逆運動學分支 (Diagnosis & IK Branch)
TM5-700 機械手臂在抵達相同 6D 目標姿態時，數學上存在多組逆運動學解。Lula 求解器在預設狀態下若未加限制，會隨機收斂到任何一個解。
1. 在我們的 USD 場景中，當 `joint_1`（底座關節）旋轉到正角度時，手臂會朝向天花板方向拱起，引發天花板碰撞。
2. 為了確保安全性，底座關節必須限制在負向半球區域（即 $q_1 \le -0.1 \text{ 弧度}$），以實現「肘部朝下」的安全姿態。

#### 2.5.3 程式碼修復對照 (Code Implementation Diff)
我們在 `NativeLulaFollowTarget` 中套用了 **多種子負向分支篩選器 (Multi-seed Branch Lock)**。在 `_find_branch_locked_solution()` 中，我們生成了 8 個預設了 Joint 1 為負值的 Seed Candidates，依序求解，並在解出後檢查 Joint 1 角度。若不符合負值要求，則直接拒絕該解，確保只採用安全的肘部向下姿態：
```python
def _seed_candidates(self):
    base = self._default_joints()
    candidates = [base]
    # 生成多個強行設定 Joint 1 為不同負向角度的 seeds 候選配置
    for j1 in [-0.25, -0.55, -0.9, -1.25, -1.65, -2.05, -2.45, -2.85]:
        seed = base.copy()
        if self.joint_1_index < len(seed):
            seed[self.joint_1_index] = j1
        candidates.append(seed)
    
    # 同步將當前關節位置也加入 seeds 候選，實現 warm-start
    current = self.robot.get_joint_positions()
    if current is not None:
        candidates.append(self._normalize_joint_positions(current))
    return candidates
```

#### 2.5.4 驗證成果 (Verification)
導入多種子負向分支鎖定後，手臂 100% 穩定地以安全、肘部朝下的狀態前行。任何會導致 `joint_1 > -0.1` 的正向解均被自動排除，機械手臂在行進過程中與場景蓋板之間保持了充足的安全距離。

---

## 4. 自訂速度與動畫引導變數 (Aesthetic Speed Parameters)

為了便於您微調底座與手臂在視覺演示時的運動質感，以下兩個核心控制變數已置於獨立工作空間腳本 [sim_rasscmap_interactive_viewer.py](file:///C:/Users/RMML/AppData/Local/ov/pkg/4.5.0/standalone_ws/rasscmap/sim_rasscmap_interactive_viewer.py) 的最上方（第 69~71 行），可直接進行修改：

```python
# ==============================================================================
# 視覺動畫速度控制參數
# ==============================================================================
BASE_TRANSITION_TOTAL_STEPS = 90  # 增加此數值可放慢底座平移的速度。預設 90 步（於 60 FPS 下相當於 1.5 秒）
ARM_FOLLOW_TOTAL_STEPS      = 90  # 增加此數值可放慢機械手臂對焦/循跡的速度。預設 90 步（於 60 FPS 下相當於 1.5 秒）
# ==============================================================================
```

*   **底座平移速度**：若希望底座平移更平緩柔和，建議將 `BASE_TRANSITION_TOTAL_STEPS` 改為 `120`（2 秒）或 `180`（3 秒）。
*   **手臂循跡速度**：若希望手臂在跟隨腫瘤時更加平穩優雅，可將 `ARM_FOLLOW_TOTAL_STEPS` 調整為 `120` 或 `150`。

---

*本文件為系統開發修復之詳盡紀錄。關於系統的架構圖、座標變換數學公式與資料庫定義，請參考隨附的 [system_design_specs.md](file:///C:/Users/RMML/IsaacLab/scripts/isaaclab_ws/reachability_map/system_design_specs.md)。*
