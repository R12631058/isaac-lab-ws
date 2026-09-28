# IsaacLab 專案 NDI 檢測檔案架構整理 
---

## 1. 檔案與功能說明

### `scripts/isaaclab_ws/final/ndi_detector_isaaclab.py`
**目標**:
這 是專案當前「相對穩定」的 NDI 檢測模組。它移植自較舊的程式碼，並在裡面呼叫 `sim.step()`、提供異步的 kit raycast 以取得射線判定，藉此測量物體是否遮蔽光學標記 (markers: FM/BM/EM/UM)。
- **主要類別**: `NDIConfig` (各種 marker 及 trigger 範圍之路徑常數)、`NDIDetector` (處理 raycast 並繪製即時預覽線段)。

### `scripts/isaaclab_ws/final/ndi_detector_isaaclab_experimental.py`
**目標**:
實驗性 NDI 檢測模組，主要嘗試更換**同步物理射線追蹤** (`PhysX Scene Query`) 演算法來執行 raycast，用以解決在 RL (Reinforcement Learning) loop 內運作造成的延遲 (laggy) 與畫面上綠/紅線不同步的問題。
- **目前的狀態**: 被復原/修改過多次。為了避免匯入發生問題，先前使用轉發機制或部分修改進行存取。
- **處理重點**: 它試圖將所有射線穿透歷史找出，並過濾掉 `Trigger Volume` ("NDI/mesh_" 等) 作為非阻擋物的遮蔽判斷邏輯。

### `scripts/isaaclab_ws/final/eval_multi_position_ndi_experimental.py`
**目標**:
作為 RL 模型訓練後的**多重環境評估 (Evaluation) 程式**。此腳本支援並行環境，用於檢測模型執行「多點尋跡 / Reach」任務時的效能，並外掛了上方提到的 NDI Detector 腳本以計算遮蔽情形並存入日誌內。
- **關鍵改動**: 此程式中的執行循環 (RL `while` loop) 會調用 detector 並讀取狀態 (`_d.update_detection(...)`) 與即時繪製 (`_d.draw_live()`)。
- **指令**:
  執行評估任務並進行 NDI 檢定：
  ```pwsh
  .\isaaclab.bat -p scripts\isaaclab_ws\final\eval_multi_position_ndi_experimental.py --num_envs 4 --episodes 3 --check_ndi --use_usd_target --checkpoint logs\rsl_rl\tm5_reach_stable_v2\2026-02-04_00-28-35\model_9999.pt
  ```

---

## 2. 當前的問題與開發進度

**面臨的問題與待解項目**：
1. **Raycast 失準導致的所有線段變紅或綠**: 單一擊中（Single-hit）碰到假透明碰撞體(如 Trigger Volume)就會中斷。我們正在這方面將其實驗調整成**多擊中（Multi-hit / Penetrating raycast）**，但需要準確分離 `target_dist`（真實目標距離）、與過濾掉不需要計算碰撞的本體 `needle_holder` 與觸發空間。
2. **回歸與版本復原**: 在上一個指令中，部分程式因為被使用者手動撤原了 `eval_multi_position_ndi_experimental.py` 以及 `ndi_detector_isaaclab_experimental.py`。
3. **Keyword引數問題**: 若復原到特定版本，`update_detection` 可能無法接收 `clear_first`。目前的解法是將繪製指令抽出 `update_detection` 中而在迴圈內額外執行 `_d.draw_live(clear_first=clear)`。

## 3. 開發/接手建議指引

1. **確認實驗分支 `ndi_detector_isaaclab_experimental.py` 的程式碼結構**：
   - 確認 callback 內對 `PhysX SceneQuery` 取回結果陣列 (`hit.distance` / `hit.collision` 路徑字串)。
   - 確保它並**沒有**在 `Update_detection` 方法裡卡到型別錯誤或是呼叫 `clear_first` (這個參數要在 `_draw_debug_lines()` 或 `draw_live()` 裡)。
2. **運行除錯指令觀看 Log**： 依據使用指令執行 4 個 envs，確認是否印出 `[Raycast Debug] BLOCKED by ...` 等訊息，找出那些應該判為綠色卻判定紅色的情況是由哪一個 USD 體阻擋，把該名稱加到 Ignore List 內。

## 4. 關鍵突破：改用純數學計算取代 Trigger Volume 射線檢測

在最新的進展中，我們拿到了 **Polaris Vega Measurement Volume** 的精確尺寸與角度（三視圖）。這意味著我們**完全不需要**依賴在 Isaac Sim 裡面擺放一個透明的 `NDI/mesh_` Trigger Volume，也**不需要**為了區分「射線是打到障礙物還是只是碰到 Trigger」而大費周章寫複雜的過濾邏輯。

**這會重塑接下來的檢測邏輯方向**：
可以將「是否在可視範圍內 (In Volume)」拆分成兩個步驟：
1. **數學可視區內判定 (Point-in-Frustum)**:
   由於 NDI 偵測儀的位置與方向是固定的（或者可以直接取得），我們只需要把目標 marker 的世界座標轉換到 **NDI Sensor Local Frame**，然後簡單地做數學邊界檢查：
   *   **Depth ($d$)**: 只要在 $950\text{ mm} \le d \le 2400\text{ mm}$ 之間。
   *   **Vertical ($y$)**: 最大半高約為 $224 + (d - 950) \times \tan(16.58^\circ) \text{ mm}$，在此範圍內才算合格。
   *   **Horizontal ($x$)**: 
       * 前段 ($950 \le d \le 1532$): 半寬度約為 $240 + (d - 950) \times \tan(29.66^\circ) \text{ mm}$
       * 後段 ($1532 < d \le 2400$): 半寬度約為 $572 + (d - 1532) \times \tan(13.66^\circ) \text{ mm}$
   *(請注意要將上述毫米單位轉為 IsaacLab 的公尺單位計算)*

2. **射線遮蔽檢測 (Raycast for Occlusion)**:
   只有在前一步 **數學檢查通過** 的 Marker 才值得丟給射線去打！
   由於我們已經不再需要 Trigger Mesh 作為是否在框內的參考，我們可以直接對被認定的 Marker 從發射器發出射線：只要第一下打到的有**非本體**的實體就算作遮擋。這讓 Raycast 可以退回最簡單快速的形式。

**對 Antigravity 的修改建議**：請優先寫一個 `isInVolume(marker_pos, nsi_sensor_pose)` 的 Python 純數學矩陣檢定函數。先做數學篩選，篩選過的點再做單點 Raycast，一切將會變得無比穩健且迅速！
