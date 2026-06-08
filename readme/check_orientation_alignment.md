# TM5-700 End-Effector 方向對齊機制說明

## 🎯 您的問題解答

### 1. EE 的角度是否有對上?

**答案**: 目前**可能沒有完全對上**,這正是訓練的核心目標之一。

根據之前 14-59-17 模型的測試結果:
- ✅ **位置追蹤**: 成功 (誤差 0.12m)
- ✅ **穩定性**: 成功 (無震盪)
- ❌ **方向對齊**: 失敗 (誤差 1.66 rad ≈ 95°)

**V4 模型改進策略**:
- 將 `orientation_tracking` 權重從 `-0.2` 提升到 `-1.0` (5倍提升)
- 目標: 將方向誤差從 1.66 rad 降低到 ~1.0 rad 以下

---

### 2. 對齊的是 Local 還是 Global 座標?

**答案**: **Global (世界座標系)**

## 📐 技術細節說明

### 方向對齊的完整流程

#### Step 1: 目標方向生成 (在機器人 Base Frame)
```python
# 在 tm5_reach_surgery_room_cfg.py 中定義
self.commands.ee_pose = mdp.UniformPoseCommandCfg(
    ranges=mdp.UniformPoseCommandCfg.Ranges(
        roll=(0.0, 0.0),          # ← 固定為 0°
        pitch=(math.pi, math.pi),  # ← 固定為 180° (朝下)
        yaw=(-math.pi, math.pi),   # ← 隨機 -180° ~ +180°
    ),
)
```

**目標方向特徵**:
- **Roll = 0°**: 末端執行器不側傾
- **Pitch = 180°**: 末端執行器朝下 (重力方向)
- **Yaw = -180° ~ +180°**: 繞 Z 軸隨機旋轉 (允許任意方向接近目標)

這些是在**機器人基座座標系 (Robot Base Frame)** 中定義的!

---

#### Step 2: 座標轉換到世界座標系
```python
# 在 pose_command.py 的 _update_metrics() 中
self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
    self.robot.data.root_pos_w,      # 機器人基座在世界座標系中的位置
    self.robot.data.root_quat_w,     # 機器人基座在世界座標系中的方向
    self.pose_command_b[:, :3],      # 目標位置 (機器人基座座標系)
    self.pose_command_b[:, 3:],      # 目標方向 (機器人基座座標系)
)
```

**轉換結果**:
- `pose_command_b`: 目標在機器人基座座標系中的表示
- `pose_command_w`: 目標在世界座標系中的表示 ← **這是對齊的參考座標系!**

---

#### Step 3: 方向誤差計算 (在世界座標系)
```python
# 在 rewards.py 的 orientation_command_error() 中
def orientation_command_error(env, command_name, asset_cfg):
    # 1. 獲取目標方向 (機器人基座座標系)
    des_quat_b = command[:, 3:7]
    
    # 2. 轉換到世界座標系
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    
    # 3. 獲取當前 EE 方向 (世界座標系) ← 直接從物理引擎讀取
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]
    
    # 4. 計算方向誤差 (四元數最短路徑)
    return quat_error_magnitude(curr_quat_w, des_quat_w)
```

**關鍵點**:
- `des_quat_w`: 目標方向 (在世界座標系中)
- `curr_quat_w`: 當前 EE 方向 (在世界座標系中) ← **直接從 Isaac Sim 物理引擎讀取**
- 誤差計算在世界座標系中進行,使用四元數表示的最短旋轉路徑

---

## 🔍 如何檢查方向是否對齊?

### 方法 1: 觀察可視化標記 (推薦)
在 play.py 運行時:
```python
self.commands.ee_pose = mdp.UniformPoseCommandCfg(
    debug_vis=True,  # ← 啟用可視化
)
```

**您會看到**:
- 🟢 **綠色坐標軸**: 目標方向 (goal pose)
- 🔵 **藍色坐標軸**: 當前 EE 方向 (current pose)

**判斷標準**:
- ✅ **對齊成功**: 兩組坐標軸完全重疊或接近平行
- ❌ **對齊失敗**: 兩組坐標軸有明顯角度差異

---

### 方法 2: 檢查終端輸出的 Metrics
在 play.py 運行時,終端會輸出:
```
Metrics:
  position_error: 0.12 m       ← 位置誤差
  orientation_error: 1.66 rad  ← 方向誤差 (目前 V4 要改善這個!)
```

**判斷標準**:
- ✅ **成功**: `orientation_error < 0.3 rad` (約 17°)
- ⚠️ **可接受**: `0.3 < orientation_error < 0.5 rad` (約 17° ~ 30°)
- ❌ **失敗**: `orientation_error > 1.0 rad` (約 57°)

14-59-17 模型的 1.66 rad (95°) 表示 EE 方向與目標方向幾乎垂直!

---

### 方法 3: 在 Isaac Sim 中手動檢查
1. **暫停仿真** (Space 鍵)
2. **選取 EE**: 點選 `SurgeryRoom/robotarm_base/tm5_700/flange`
3. **查看屬性面板**:
   - `xformOp:orient`: 當前 EE 的四元數 (Quaternion)
   - 與目標標記的方向比較

---

## 📊 當前 V4 配置的方向要求

```python
# 目標方向 (在機器人基座座標系)
roll = 0°          # 不側傾
pitch = 180°       # 朝下
yaw = -180° ~ +180°  # 任意旋轉

# 獎勵權重
orientation_tracking = -1.0  # V4 提升 5 倍 (從 -0.2)
```

**含義**:
- 機器人必須讓 EE **朝下** (pitch = 180°)
- 但允許繞垂直軸旋轉 (yaw 任意)
- 不允許側傾 (roll = 0°)

**實際應用場景**: 模擬手術器械從上方垂直接近手術部位

---

## 🎮 如何觀察 V4 模型的方向對齊改善?

### 在當前運行的 play.py 視窗中觀察:

1. **注意綠色和藍色坐標軸**:
   - 如果 V4 改善成功,兩者的夾角會比 14-59-17 小很多
   - 理想情況: Z 軸 (向下) 應該平行

2. **觀察 EE 的實際動作**:
   - ✅ **改善成功**: EE 接近目標時會調整方向,保持朝下
   - ❌ **改善失敗**: EE 到達位置後,方向與目標仍有大角度偏差

3. **檢查終端輸出**:
   - 每個 episode 結束時會印出 `orientation_error`
   - 目標: `< 1.0 rad` (比 14-59-17 的 1.66 rad 改善)

---

## 🔧 如果方向仍然不對齊,下一步調整建議

### 情況 1: orientation_error 仍 > 1.0 rad
**原因**: 權重增加不夠
**解決方案**: 進一步提升權重
```python
self.rewards.end_effector_orientation_tracking.weight = -1.5  # 或 -2.0
```

### 情況 2: 機器人變得不穩定或無法 reach
**原因**: 權重過高,與 position_tracking 衝突
**解決方案**: 回退到較低的值
```python
self.rewards.end_effector_orientation_tracking.weight = -0.5
```

### 情況 3: orientation_error 在 0.5 ~ 1.0 rad 之間
**原因**: 需要更多訓練時間
**解決方案**: 繼續訓練到 5000 iterations

---

## 📝 總結

| 問題 | 答案 |
|------|------|
| **EE 角度是否對上?** | 目前 14-59-17 沒有 (1.66 rad 誤差),V4 正在嘗試改善 |
| **對齊參考座標系?** | **Global (世界座標系)** - Isaac Sim 物理引擎直接提供 |
| **目標方向定義?** | Pitch=180° (朝下), Roll=0°, Yaw=隨機 |
| **如何驗證對齊?** | 觀察可視化標記 + 檢查 `orientation_error` 數值 |
| **V4 改善策略?** | 保守提升 orientation_tracking 權重 (-0.2 → -1.0) |

---

## 🚀 立即行動

**當前正在運行的 play.py**:
- 觀察 4 個環境中的機器人行為
- 注意綠色 (目標) 和藍色 (當前) 坐標軸的夾角
- 如果夾角明顯小於 90° → V4 改善成功!
- 如果仍然接近 90° → 需要更激進的權重或更多訓練

**記得回報您的觀察結果!** 🎯
