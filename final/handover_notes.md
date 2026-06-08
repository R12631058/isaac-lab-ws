# 交接筆記: NDI 追蹤空間最佳化 (NDI Tracking Volume Optimization)

## 目前狀態 (Current Status)
我們已經完成了 `scripts/isaaclab_ws/final/ndi_multipose_scorer_v2.py` 的重構，支援**自動化多批次評估 (Automated Multi-batch Evaluation)**。這個版本能夠一次性針對多個機器人基座與多個 NDI 攝影機位置進行交叉比對與計分。

### 達成目標 (Completed Goals)
使用者現在可以透過清單提供多個機器人基座位置 (例如：`--robot_x_list 0.46 0.66 0.86`)，程式將會自動執行以下動作：
1. 輪詢每一個指定的機器人 X 軸位置 (批次處理)。
2. 針對該特定的機器人位置，平行評估所有 NDI 攝影機位置 (由 `num_envs` 以及 `x_start`、`x_end` 定義跨度)。
3. 在同一份報告中彙整所有結果 (M 個機器人位置 × N 個 NDI 位置)，計算平均遮擋成本並給出全域排名。
4. 加入了「任務可達性 (Reachability)」檢查，判定不同基座位置下的機器人是否能夠順利觸及目標點。

## 最近的變更 (Recent Changes)
- **多批次迴圈實作 (Multi-batch Pipeline)**: 將模擬與評分邏輯使用 `for batch_idx, robot_x_val in enumerate(robot_x_list):` 迴圈包裝。
- **Scorer 狀態重置機制**: 在每個批次運算開始前，明確重置 `scorer.occlusion_history` 和 `scorer.frame_results` 字典，避免上一批次的遮擋分數污染到下一批次。
- **可達性驗證 (Reachability Check)**: 透過 PyTorch 物理張量 `robot.data.body_pos_w` 取得 `needle_tip` 實際空間座標，並計算至 `target_pose` 的距離 (容差設定為 5cm)，標示為 "Yes" 或 "No" (同時在圖表上分別以綠色/紅色顯示)。
- **全域結果彙整 (Global Result Consolidation)**: 搜集所有批次數據至 `global_summary_rows`，依照 Mean Cost 自動排序，生成總表的 ASCII 終端機輸出，並匯出包含所有設定的 Matplotlib PNG 視覺化表格與 JSON 結構報表 (儲存於 `occlusion_output/` 下)。
- **Headless 修正 (於 Detector)**: NDI 偵測器核心 `ndi_detector_isaaclab_experimental.py` 加入了延遲載入 (lazy imports) 機制，避免在無介面 (`--headless`) 模式下引發繪圖模組錯誤 (`ModuleNotFoundError`)。

## 待辦事項 / 下一步 (Pending Tasks / Next Steps)
1. **結果驗證與實務分析**:
   - 檢查產出的 `occlusion_output/ndi_multipose_score_multibatch_*.json` 及對應的 PNG 總表，確認名次第一的組合 (Cost 最低且 Reach 為 Yes) 是否符合空間物理邏輯。
   - 確認這組「最佳 NDI 與 Robot X 位置」是否可直接應用於實際的手術室實驗架設。
2. **微調計分參數 (Optional)**:
   - 若對遮擋的懲罰不夠敏感，可進入腳本調整 `MultiPoseScorer` 內的 `base_penalty` (目前 50.0)、連續遮擋倍率或立體角(Solid Angle)的溫度參數 (`temperature`)。
   - 若需要更細緻的作動過程評估，可修改追蹤的分鏡 `target_frames = [10, 30, 50, 70, 90]`。

## 關鍵檔案 (Key Files)
- `scripts/isaaclab_ws/final/ndi_multipose_scorer_v2.py`: 主要多批次評估執行腳本 (已完成 V2 重構)。
- `scripts/isaaclab_ws/final/ndi_detector_isaaclab_experimental.py`: NDI 核心偵測與射線投射 (Raycast) 邏輯。
- `scripts/isaaclab_ws/final/Input_output.tex`: NDI 與機器人姿態最佳化的數學公式與定義文件。

## 場景與基礎設定檔 (Scene & Configurations)
這個評估腳本 (`ndi_multipose_scorer_v2.py`) 中初始化了 `TM5ExtensionLinkOutFanOrientationEnvCfg`，背後實際呼叫的設定與使用的 USD 關聯如下：
- **環境設定檔路徑**: `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/tm5/tm5_extension_linkout_cone_orientation_cfg.py` (負責定義扇形姿態、關節限制等環境參數)
- **場景設定檔路徑**: `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/tm5/tm5_reach_surgery_room_cfg.py` (包含 `ExtensionLinkOutSceneCfg` 場景配置)
- **目前使用的舊版 USD 檔案**: `C:\Nick\surgery_team\surgery_team\USD\isaaclab\extension_link_out.usd` (以此 USD 為基準抓取機器人、標記與目標點的位置)

### ⚠️ [進行中變更] 更換場景 USD 檔案 (變更時間：2026年4月28日)
預計將棄用上述的 `extension_link_out.usd`，並改用新的場景模型：
- **新版 USD 檔案目標路徑**: `C:\Nick\surgery_team\surgery_team\USD\animation\isaaclab_multi_env.usd`
- **更改原因**: 
  1. 先前的 USD 檔案已發生損壞。
  2. 新版的 `isaaclab_multi_env.usd` 檔案內涵蓋了**模擬病患 (Simulated Patient)**，能夠更真實地反映真實手術室的遮擋以及實體碰撞形況，提升評估分數的準確性。
- **後續需要進行的修改**: 接下來需要前去 `tm5_reach_surgery_room_cfg.py` 的 `ExtensionLinkOutSceneCfg` 中，將 `usd_path` 更新為這條新版路徑，並驗證新 USD 下的 `robotarm`、`target`、與 `marker` (標記) 階層結構是否發生了改變，以確保 Python 腳本依然能抓到正確的 Prim。
