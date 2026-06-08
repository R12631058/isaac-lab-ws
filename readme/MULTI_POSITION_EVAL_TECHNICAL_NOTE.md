# Multi-Position Evaluation with Fixed World Target (Technical Note)

本文件說明 `scripts/multi_position_eval.py` 的修改邏輯，以及為何在標準 Isaac Lab / RL 流程中無法直接達成「固定世界目標」的原因。

## 1. 核心目標 (Objective)
建立 **Inverse Reachability Map**。
即：**手術目標點（Target/Cube）在世界座標中是唯一的、不動的**。我們想測試當機器人底座（Robot Base）安裝在不同位置（X 軸變化）時，策略（Policy）的抓取成功率。

## 2. 與標準 RL 訓練流程的邏輯衝突 (The Logic Gap)

在標準的機械手臂 RL 訓練（如 `Reach` 任務）中，邏輯通常是這樣的：
1.  **以機器人為中心**：模擬器重置時，會將 Target 生成在機器人的「工作空間（Workspace）」內。
2.  **相對移動**：如果我們移動了機器人底座（由 `randomize_robot_root_pose`），標準的 `CommandTerm` 通常會相對於新的底座位置重新生成 Target。
    *   *結果*：機器人往左移，Target 也跟著往左移。相對距離不變。
    *   *問題*：這無法模擬「機器人移動，但病人不動」的情況。

**為什麼之前會失敗？**
如果您只移動 Robot Base，但沒有手動修正 Target Command：
*   神經網路的輸入（Observation）通常包含 `Target_Position_in_Robot_Frame`。
*   因為 Command Manager 認為 Target 是跟著機器人的，所以它傳給 Policy 的向量可能保持不變（或隨機變動），導致機器人去抓「相對於它自己」的位置，而不是「世界座標中的那個 Cube」。

## 3. 修改內容與技術解法 (Modifications)

為了達成需求，我們對 `multi_position_eval.py` 做了以下關鍵修改：

### A. 平行化環境配置 (Parallel Environment Configuration)
我們不再是隨機撒點，而是均勻分布：
```python
# 產生線性分布的 X 偏移量
x_offsets = torch.linspace(x_min, x_max, num_envs)
# 每個環境由 Env Origins + Default Local + Shift 計算出各自的 World Pose
```
這確保了由 `Env 0` 到 `Env N`，機器人是整齊排列在我們想測試的路徑上。

### B. 固定世界目標與座標轉換 (Fixed World Target & Frame Transformation)
這是最關鍵的部分。我們使用了 `isaaclab.utils.math.subtract_frame_transforms` API。

**邏輯流程：**
1.  **讀取真值**：從 USD 中讀取 `Env 0` 的 Cube 位置，視為「標準手術點」（相對於 Env 0 原點）。
2.  **廣播目標**：將這個相對位置應用到所有環境（Env 0...N），算出每個環境中 Cube 的**世界座標** (`Target_W`)。
3.  **逆向計算**：
    對於每一個環境 $i$，機器人底座位置 $Base_W^{(i)}$ 都不一樣。
    Policy 需要的是相對向量 $Command^{(i)}$：
    $$ Command^{(i)}_{robot} = Target^{(i)}_{world} - Base^{(i)}_{world} $$
    *(注意：實際運算是包含四元數的座標變換)*
4.  **強制覆寫 (Override)**：
    我們繞過了原本隨機生成的 `CommandManager`，直接將計算好的正確向量寫入緩衝區：
    ```python
    cmd_term.command[:] = cmd_pos_b # 寫入計算後的相對位置
    ```

### C. 觀測值更新 (Observation Re-computation)
因為我們手動修改了 Command，必須立即呼叫：
```python
obs = env.observation_manager.compute()
```
這確保在 Policy 執行 `act()` 之前，它看到的是修正後、指向真實世界目標的向量。

## 4. API 調用差異總結

| 功能 | 標準訓練流程 (Standard RL) | 修改後的評估流程 (Our Eval) |
| :--- | :--- | :--- |
| **機器人位置** | 固定 或 在小範圍隨機微調 | **大幅度線性分布** (Grid Search) |
| **目標生成** | 相對於機器人 (Robot Frame) 隨機生成 | **固定世界座標** (World Frame) 鎖定 USD 物件 |
| **Command** | `CommandManager` 自動 Resample | **手動計算** `subtract_frame_transforms` 並覆寫 |
| **Observation** | 自動計算 | 修改 Command 後需**手動觸發** `compute()` |

## 5. 如何使用

**使用 USD 場景中的 Cube 作為目標：**
```powershell
python scripts/multi_position_eval.py --use_usd_target --num_envs 16 --x_min 0.3 --x_max 1.0
```

**手動指定目標點（相對於環境原點）：**
```powershell
# 例如目標在每個環境原點前方 0.5m 處
python scripts/multi_position_eval.py --target_local_x 0.5 --num_envs 16 --x_min 0.3 --x_max 1.0
```
