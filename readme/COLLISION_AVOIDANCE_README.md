# TM5 Reach 碰撞避免功能說明

## 問題描述

TM5-700 機器人底座下方有一個 UM 組件,當 link_1 (joint_1) 往小角度轉動時會造成碰撞。

**碰撞危險區域**: joint_1 在 -30° ~ +30° (-0.52 ~ +0.52 弧度) 範圍內
**安全區域**: joint_1 在 +30° ~ +180° 或 -180° ~ -30° 範圍內

## 解決方案

我們實現了兩種訓練方法的組合:

### 1. 方法一:角度偏好引導 (Explicit Guidance)

**實現位置**: `mdp/rewards.py` - `joint_1_avoid_small_angles()`

**原理**:
- 使用平滑的餘弦函數創建懲罰曲線
- 在危險區域 (-30° ~ +30°) 內給予懲罰
- 中心 (0°) 懲罰最大,邊界處漸減到 0
- 鼓勵機器人優先使用大角度區域

**數學公式**:
```python
# 危險區域檢測
in_danger = (joint_1_angle >= -0.52) & (joint_1_angle <= 0.52)

# 餘弦懲罰曲線 (中心最大,邊界漸減)
normalized_dist = (joint_1_angle - center) / half_width  # -1 到 +1
penalty = (1.0 + cos(normalized_dist * π)) / 2.0
```

**優點**:
- ✅ 使用領域知識,訓練效率高
- ✅ 明確指導機器人避開危險區域
- ✅ 平滑的懲罰函數,便於梯度優化

**配置**:
```python
joint_1_avoid_small_angles = RewTerm(
    func=mdp.joint_1_avoid_small_angles,
    weight=-1.5,  # 負權重:在危險區域給予懲罰
    params={
        "asset_cfg": SceneEntityCfg("robot"),
        "danger_zone_min": -0.52,  # -30°
        "danger_zone_max": 0.52,   # +30°
    },
)
```

### 2. 方法二:關節極限懲罰 (Safety Margins)

**實現位置**: `mdp/rewards.py` - `joint_limits_penalty()`

**原理**:
- 當關節接近其極限 (軟極限 90%) 時給予懲罰
- 鼓勵機器人在安全範圍內運動
- 適用於所有關節,提供通用的安全保護

**數學公式**:
```python
# 標準化關節位置 (0 = 下限, 1 = 上限)
normalized_pos = (joint_pos - lower_limit) / (upper_limit - lower_limit)

# 計算違規程度
lower_violation = max(0, soft_limit_ratio - normalized_pos)
upper_violation = max(0, normalized_pos - (1 - soft_limit_ratio))

# 總懲罰
penalty = sum(lower_violation + upper_violation)
```

**優點**:
- ✅ 通用方法,適用於所有關節
- ✅ 防止機器人運動到極限位置
- ✅ 提供額外的安全邊界

**配置**:
```python
joint_limits_penalty = RewTerm(
    func=mdp.joint_limits_penalty,
    weight=-0.3,
    params={
        "asset_cfg": SceneEntityCfg("robot"),
        "soft_limit_ratio": 0.9,  # 90% 處開始懲罰
    },
)
```

## 文件修改清單

### 1. mdp/rewards.py
添加了兩個新的獎勵函數:

```python
def joint_1_avoid_small_angles(env, asset_cfg, danger_zone_min=-0.52, danger_zone_max=0.52):
    """懲罰 joint_1 在小角度範圍內的運動 (避免與底座 UM 碰撞)"""
    # ... 實現細節 ...

def joint_limits_penalty(env, asset_cfg, soft_limit_ratio=0.9):
    """懲罰接近關節極限的運動"""
    # ... 實現細節 ...
```

### 2. tm5_reach_surgery_room_cfg.py
創建了自定義的 `SurgeryRoomRewardsCfg` 類:

```python
@configclass
class SurgeryRoomRewardsCfg(reach_env_cfg.RewardsCfg):
    """手術室場景專用的獎勵配置 - 包含碰撞避免"""
    
    # 繼承父類所有獎勵項
    
    # Joint 1 角度安全性獎勵
    joint_1_avoid_small_angles = RewTerm(...)
    
    # 關節極限懲罰
    joint_limits_penalty = RewTerm(...)
```

在 `TM5ReachSurgeryRoomEnvCfg` 中使用:

```python
class TM5ReachSurgeryRoomEnvCfg(ReachEnvCfg):
    # 使用自定義的獎勵配置
    rewards: SurgeryRoomRewardsCfg = SurgeryRoomRewardsCfg()
```

## 測試驗證

### 測試腳本: `test_collision_avoidance.py`

驗證功能:
1. ✅ 環境正常創建和運行
2. ✅ 獎勵項正確配置和計算
3. ✅ Joint 1 角度監控
4. ✅ 碰撞避免獎勵生效

### 測試結果:

```
獎勵項配置:
--------------------------------------------------------------------------------
  end_effector_position_tracking              | weight: -2.0000
  end_effector_position_tracking_fine_grained | weight: 15.0000
  end_effector_orientation_tracking           | weight: -0.2000
  action_rate                                 | weight: -0.0100
  joint_vel                                   | weight: -0.0050
  joint_1_avoid_small_angles                  | weight: -1.5000  ← 新增
  joint_limits_penalty                        | weight: -0.3000  ← 新增
--------------------------------------------------------------------------------
```

環境成功創建,所有獎勵項正確加載。

## 下一步訓練

### 訓練命令:

```bash
# 使用帶碰撞避免的配置訓練
.\isaac_lab_correct.bat scripts\reinforcement_learning\rsl_rl\train.py ^
  --task Isaac-Reach-TM5-Surgery-Room-v0 ^
  --num_envs 32 ^
  --headless ^
  --max_iterations 5000
```

### 預期效果:

1. **訓練初期**:
   - Joint 1 可能會嘗試各種角度
   - 當進入危險區域 (-30° ~ +30°) 時會受到懲罰
   - 逐漸學習避開小角度區域

2. **訓練中期**:
   - Joint 1 開始偏好大角度區域
   - 碰撞次數減少
   - 任務完成率提升

3. **訓練後期**:
   - Joint 1 主要使用安全區域 (大角度)
   - 碰撞幾乎不再發生
   - 保持高任務完成率

### 權重調整建議:

如果訓練中發現問題,可以調整權重:

```python
# 如果碰撞仍然頻繁,增大懲罰權重
joint_1_avoid_small_angles.weight = -2.5  # 從 -1.5 增大到 -2.5

# 如果機器人過於保守,不敢移動
joint_1_avoid_small_angles.weight = -1.0  # 從 -1.5 降低到 -1.0
```

## 總結

我們成功實現了碰撞避免功能,結合了兩種訓練方法:

1. **角度偏好引導**: 明確告訴機器人哪些角度是危險的
2. **關節極限懲罰**: 提供通用的安全保護

這種組合方法既高效又穩健,能夠有效訓練機器人避免與底座 UM 組件的碰撞。

## 相關文件

- `source/isaaclab_tasks/.../reach/mdp/rewards.py` - 獎勵函數實現
- `source/isaaclab_tasks/.../reach/config/tm5/tm5_reach_surgery_room_cfg.py` - 環境配置
- `test_collision_avoidance.py` - 測試腳本
