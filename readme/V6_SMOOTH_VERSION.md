# V6 平滑版本 - 解決晃動問題

## 🔍 V5 觀察結果

**使用者反饋**: "確實是有,但是我理解在尋找正確旋轉時會有額外的晃動"

### V5 表現:
- ✅ **方向對齊有改善**: Local 座標系讓機器人能夠正確識別目標方向
- ✅ **能夠 reach 到目標**: 位置追蹤仍然正常
- ⚠️ **產生晃動**: 在調整方向時動作不夠平滑,有急促的旋轉動作

---

## 🎯 問題診斷

### V5 權重配置:
```python
orientation_tracking: -2.0    # 方向要求很嚴格
action_rate: -0.005           # 平滑度約束很弱
position_fine_grained: 18.0   # 位置要求很高
```

**權重不平衡問題**:
```
方向追蹤強度 / 平滑度約束 = -2.0 / -0.005 = 400 倍!
```

機器人的優先級:
1. 優先達到精確位置 (權重 18.0 最高)
2. 然後嘗試對齊方向 (權重 -2.0)
3. 但幾乎不在意動作平滑度 (權重 -0.005 太低)

**結果**: 為了快速滿足方向對齊,機器人會做出急促、晃動的調整動作。

---

## ✅ V6 解決方案

### 策略: 平衡方向追蹤與動作平滑

```python
# V6 配置調整
orientation_tracking: -1.0    # 從 -2.0 降到 -1.0 (降低 50%)
action_rate: -0.01            # 從 -0.005 提高到 -0.01 (提高 100%)
position_fine_grained: 18.0   # 維持不變
```

### 新的優先級平衡:
```
方向追蹤強度 / 平滑度約束 = -1.0 / -0.01 = 100 倍
```

**改善**:
- 方向追蹤仍然重要,但允許更漸進的調整
- 平滑度約束提高 2 倍,強制更柔和的動作
- 兩者比例從 400:1 降到 100:1,更平衡

---

## 📊 版本對比

| 版本 | 方向對齊 | orientation | action_rate | 位置 | 觀察結果 |
|------|----------|-------------|-------------|------|----------|
| **14-59-17** | Global | -0.2 | -0.005 | 15.0 | ✅ 穩定但方向差 (1.66 rad) |
| **V4** | Global | -1.0 | -0.005 | 18.0 | ❓ (未完整測試) |
| **V5** | **Local** | -2.0 | -0.005 | 18.0 | ⚠️ 方向改善但晃動 |
| **V6** | **Local** | **-1.0** | **-0.01** | 18.0 | 🎯 目標: 穩定+方向改善 |

---

## 🎯 V6 預期效果

### 相比 V5 的改善:
1. **減少晃動**: action_rate 提高 2 倍,強制更平滑的動作序列
2. **保持方向對齊**: orientation_tracking 仍有 -1.0,足以學習方向控制
3. **更自然的運動**: 方向調整會更漸進,像人類操作一樣

### 相比 14-59-17 的改善:
1. **正確的座標系**: Local 對齊比 Global 更合理
2. **更好的方向**: -1.0 比 -0.2 高 5 倍,應能顯著改善方向誤差
3. **維持穩定性**: action_rate=-0.01 與失敗的模型相同,但 Local 對齊更穩定

---

## 🚀 啟動 V6 訓練

### 快速測試 (推薦先試這個)
```bash
.\isaaclab_conda.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Reach-TM5-SurgeryRoom-v0 --num_envs 256 --max_iterations 2000 --seed 42 --headless
```

**為什麼用 256 envs?**
- 更多並行環境 → 更快收集經驗
- V6 的權重平衡更好 → 應該能快速收斂
- 2000 iterations 約 30-40 分鐘 (256 envs)

### 完整訓練 (如果快速測試成功)
```bash
.\isaaclab_conda.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Reach-TM5-SurgeryRoom-v0 --num_envs 256 --max_iterations 5000 --seed 42 --headless
```

---

## 📊 訓練監控重點

### 1. Reward Curves (TensorBoard 或終端)

**期待看到**:
```
end_effector_orientation_tracking:
- 從 -1.5 逐漸上升到 -0.5 左右
- 比 V5 收斂更快 (因為權重平衡更好)

action_rate:
- 保持在較高的負值 (約 -0.5 ~ -1.0)
- 證明動作平滑度受到重視

position_fine_grained:
- 維持 > 0.7 (穩定 reach)
```

### 2. Metrics

**目標**:
- `orientation_error`: < 0.8 rad (< 45°) 就算成功改善
- `position_error`: < 0.15 m (維持穩定)

### 3. 訓練穩定性

**好的信號**:
- 獎勵曲線平滑上升,無大幅震盪
- 無突然崩潰或性能退化
- loss 穩定下降

---

## 🎮 測試 V6 模型

訓練完成後:
```bash
.\isaaclab_conda.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task Isaac-Reach-TM5-SurgeryRoom-v0 --num_envs 4 --checkpoint logs\rsl_rl\tm5_reach_stable_v2\<timestamp>\model_XXXX.pt
```

### 觀察重點:

1. **動作平滑度** (相比 V5)
   - ✅ 成功: 方向調整更緩慢、更自然
   - ❌ 失敗: 仍有急促的旋轉或晃動

2. **方向對齊** (相比 14-59-17)
   - ✅ 成功: 坐標軸夾角明顯減小 (< 60°)
   - ❌ 失敗: 仍接近 90° 的偏差

3. **整體穩定性**
   - ✅ 成功: 能穩定 reach,動作流暢
   - ❌ 失敗: 無法 reach 或動作混亂

---

## 🔧 如果 V6 仍有問題

### 情況 1: 仍然晃動
**診斷**: action_rate 仍不夠高
**解決**:
```python
self.rewards.action_rate.weight = -0.015  # 或 -0.02
```
**風險**: 太高會導致無法 reach (像 23-17-35 一樣)

### 情況 2: 方向仍然很差
**診斷**: orientation_tracking 太低
**解決**:
```python
self.rewards.end_effector_orientation_tracking.weight = -1.5  # 介於 V5 和 V6 之間
```

### 情況 3: 無法 reach 或不穩定
**診斷**: action_rate 太高抑制探索
**解決**:
```python
self.rewards.action_rate.weight = -0.007  # 降低到 V5 和 14-59-17 之間
```

---

## 💡 進階調整策略

### 如果需要更精細的控制,可以考慮:

#### 選項 A: 分階段訓練 (Curriculum Learning)
```python
# 第 0-1000 iterations: 專注位置,忽略方向
orientation_tracking: -0.5
action_rate: -0.005

# 第 1000-3000 iterations: 逐步增加方向要求
orientation_tracking: -0.5 → -1.0 (線性增加)
action_rate: -0.005 → -0.01 (線性增加)

# 第 3000+ iterations: 完整要求
orientation_tracking: -1.0
action_rate: -0.01
```

#### 選項 B: 使用不同的平滑度函數
```python
# 當前: L2 norm (懲罰大動作)
action_rate = mdp.action_rate_l2

# 替代: 加速度懲罰 (懲罰急促變化)
# 可能需要自己實現
```

---

## 📝 V6 配置摘要

```yaml
# V6 關鍵配置
rewards:
  position_fine_grained:
    weight: 18.0          # 維持高精度位置追蹤
  
  orientation_tracking:
    function: orientation_command_error_local  # Local 座標系 ✨
    weight: -1.0          # 適中的方向要求
  
  action_rate:
    weight: -0.01         # 較高的平滑度要求
  
  joint_vel:
    weight: -0.01         # 配合 action_rate 減少晃動

training:
  num_envs: 256           # 加速訓練
  max_iterations: 2000    # 快速測試
  seed: 42                # 可重現
```

---

## ✅ 執行清單

### 已完成:
- [x] 診斷 V5 晃動問題
- [x] 設計 V6 平衡權重方案
- [x] 更新配置文件
- [x] 創建詳細文檔

### 待執行:
- [ ] 清理 pycache
- [ ] 啟動 V6 訓練 (2000 iterations)
- [ ] 監控訓練過程
- [ ] 測試 V6 模型
- [ ] 與 V5 和 14-59-17 對比
- [ ] 決定是否需要 V7 或可以部署

---

## 🎯 成功標準

**V6 達成目標**:
- ✅ 動作比 V5 更平滑 (無明顯晃動)
- ✅ 方向比 14-59-17 更好 (誤差 < 1.0 rad)
- ✅ 維持穩定 reach 能力
- ✅ 整體表現最佳,可用於部署

**如果達成**: 🎉 V6 就是最終版本!

**如果未達成**: 根據具體問題,微調權重或考慮進階策略。

---

**準備好開始 V6 訓練了嗎?** 這次應該能達到平滑+方向對齊的完美平衡! 🎯
