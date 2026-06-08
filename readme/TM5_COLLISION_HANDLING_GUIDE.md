# TM5-700 碰撞檢測與處理指南

## 概述

本指南說明如何在 TM5-700 機械臂的 Reach 任務中處理兩種類型的碰撞:
1. **自碰撞 (Self-Collision)**: 機械臂連桿之間的碰撞
2. **環境碰撞 (Environment Collision)**: 機械臂與外部物體(桌子、地面等)的碰撞

## 已實現的配置

### 1. `Isaac-Reach-TM5-Surgical-v2-Collision`

完整的碰撞處理配置,包含:

#### 碰撞傳感器
- 監測所有 TM5-700 連桿: base_link, link_1~6, flange
- 實時檢測接觸力 (每個模擬步更新)
- 保留 3 幀歷史數據用於穩定檢測

#### 碰撞懲罰機制

**1. 不期望的接觸 (Undesired Contacts)**
```python
self.rewards.undesired_robot_contacts = RewTerm(
    func=base_mdp.undesired_contacts,
    weight=-5.0,  # 每次違規懲罰 -5.0
    params={
        "sensor_cfg": SceneEntityCfg("contact_sensor", body_ids=[0, 1, 2, 3, 4, 5, 6, 7]),
        "threshold": 1.0,  # 1N 接觸力閾值
    },
)
```
- **用途**: 檢測輕微碰撞和自碰撞
- **工作原理**: 當任何連桿的接觸力 > 1N 時計為一次違規
- **懲罰**: 每次違規 -5.0 獎勵

**2. 過度接觸力 (Excessive Contact Forces)**
```python
self.rewards.excessive_contact_forces = RewTerm(
    func=base_mdp.contact_forces,
    weight=-2.0,  # 每牛頓超出懲罰 -2.0
    params={
        "sensor_cfg": SceneEntityCfg("contact_sensor", body_ids=[0, 1, 2, 3, 4, 5, 6, 7]),
        "threshold": 10.0,  # 10N 以上開始懲罰
    },
)
```
- **用途**: 檢測嚴重碰撞
- **工作原理**: 當接觸力 > 10N 時,懲罰超出的力量
- **懲罰**: 例如 15N 的接觸力會被懲罰 -2.0 × 5 = -10.0

## 使用方法

### 訓練帶碰撞檢測的模型

```powershell
# 基礎訓練
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py `
    --task Isaac-Reach-TM5-Surgical-v2-Collision `
    --num_envs 512 `
    --max_iterations 3000

# 如果從已有 checkpoint 繼續訓練
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py `
    --task Isaac-Reach-TM5-Surgical-v2-Collision `
    --num_envs 512 `
    --max_iterations 3000 `
    --resume `
    --checkpoint logs\rsl_rl\tm5_reach_surgical_v2\<timestamp>\model_XXXX.pt
```

### 測試訓練好的模型

```powershell
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py `
    --task Isaac-Reach-TM5-Surgical-v2-Collision-Play `
    --num_envs 16 `
    --checkpoint logs\rsl_rl\tm5_reach_surgical_v2\<timestamp>\model_2999.pt
```

## 配置對比

| 配置名稱 | 碰撞檢測 | 動作範圍 | PPO 噪聲 | 適用場景 |
|---------|---------|---------|---------|---------|
| `Isaac-Reach-TM5-Stable-v1` | ❌ | 0.5 (小) | 0.5 (高) | 原始配置,有抖動和可達性問題 |
| `Isaac-Reach-TM5-Surgical-Debug-v0` | ❌ | 1.5 (大) | 0.5 (高) | 激進測試,有抖動 |
| `Isaac-Reach-TM5-Surgical-v2` | ❌ | 0.8 (中) | 0.15 (低) | 平衡版本,無碰撞檢測 |
| `Isaac-Reach-TM5-Surgical-v2-Collision` | ✅ | 0.8 (中) | 0.15 (低) | **推薦使用** - 完整功能 |

## 碰撞檢測原理

### ContactSensor 工作流程

1. **傳感器配置**
   ```python
   contact_sensor = ContactSensorCfg(
       prim_path="{ENV_REGEX_NS}/Robot/.*",
       filter_prim_paths_expr=[
           "{ENV_REGEX_NS}/Robot/base_link",
           "{ENV_REGEX_NS}/Robot/link_1",
           # ... 其他連桿
       ],
   )
   ```

2. **接觸力檢測**
   - PhysX 引擎計算每個碰撞的接觸力向量
   - 傳感器收集 `net_forces_w` (世界坐標系下的淨力)
   - 歷史緩衝區保存最近 3 幀數據

3. **碰撞分類**
   - **自碰撞**: 當 `link_1` 與 `link_3` 接觸時
   - **環境碰撞**: 當任何連桿與桌子/地面接觸時
   - 兩者通過接觸力大小區分嚴重程度

### 懲罰函數詳解

#### `undesired_contacts()`
```python
# 源碼位置: isaaclab/envs/mdp/rewards.py
def undesired_contacts(env, threshold, sensor_cfg):
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)  # 返回違規次數
```

#### `contact_forces()`
```python
# 源碼位置: isaaclab/envs/mdp/rewards.py
def contact_forces(env, threshold, sensor_cfg):
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    return torch.sum(violation.clip(min=0.0), dim=1)  # 返回超出的力量總和
```

## 調優建議

### 如果訓練中出現過多碰撞

1. **增加碰撞懲罰**
   ```python
   self.rewards.undesired_robot_contacts.weight = -10.0  # 從 -5.0 增加到 -10.0
   ```

2. **降低接觸力閾值**
   ```python
   params={"threshold": 0.5}  # 從 1.0 降低到 0.5
   ```

3. **添加碰撞終止**
   - 需要自定義終止函數檢測嚴重碰撞
   - 當接觸力 > 50N 時立即終止 episode

### 如果模型過於保守(不敢移動)

1. **降低碰撞懲罰**
   ```python
   self.rewards.undesired_robot_contacts.weight = -2.0  # 從 -5.0 降低到 -2.0
   ```

2. **提高接觸力閾值**
   ```python
   params={"threshold": 2.0}  # 從 1.0 提高到 2.0
   ```

3. **調整動作範圍**
   ```python
   scale=1.0  # 從 0.8 提高到 1.0
   ```

## 監控碰撞數據

### 訓練時查看碰撞統計

```python
# 在 train.py 中添加日誌
if "undesired_robot_contacts" in env.reward_manager.active_terms:
    contact_penalty = env.reward_manager.compute_term("undesired_robot_contacts")
    print(f"Collision count: {contact_penalty.mean().item()}")
```

### TensorBoard 可視化

```powershell
# 啟動 TensorBoard
tensorboard --logdir logs/rsl_rl/tm5_reach_surgical_v2

# 查看指標:
# - Reward/undesired_robot_contacts (碰撞次數)
# - Reward/excessive_contact_forces (接觸力大小)
```

## 進階功能

### 自定義碰撞終止條件

如果需要在嚴重碰撞時立即終止 episode,可以添加自定義函數:

```python
# 在 mdp 模塊中添加
def severe_collision_termination(env, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """嚴重碰撞時終止"""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    max_force = torch.max(torch.norm(net_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    return max_force > threshold

# 在配置中使用
self.terminations.severe_collision = DoneTerm(
    func=mdp.severe_collision_termination,
    params={
        "sensor_cfg": SceneEntityCfg("contact_sensor", body_ids=[0,1,2,3,4,5,6,7]),
        "threshold": 50.0,
    },
)
```

### 區分自碰撞和環境碰撞

可以創建兩個不同的懲罰項:

```python
# 自碰撞 - 更嚴格
self.rewards.self_collision = RewTerm(
    func=base_mdp.undesired_contacts,
    weight=-10.0,  # 更重的懲罰
    params={
        "sensor_cfg": SceneEntityCfg("contact_sensor", body_ids=[1,2,3,4,5,6]),  # 排除 base
        "threshold": 0.5,  # 更低的閾值
    },
)

# 環境碰撞 - 較寬鬆
self.rewards.environment_collision = RewTerm(
    func=base_mdp.contact_forces,
    weight=-2.0,
    params={
        "sensor_cfg": SceneEntityCfg("contact_sensor", body_ids=[0,1,2,3,4,5,6,7]),
        "threshold": 5.0,
    },
)
```

## 常見問題

### Q1: 為什麼有 lint 錯誤?
**A**: 這些錯誤是 Python 類型檢查器的警告,因為我們動態添加獎勵項。運行時完全正常,可以忽略。

### Q2: 碰撞傳感器會影響訓練速度嗎?
**A**: 有輕微影響(約 5-10%),但可接受。接觸檢測在 GPU 上並行執行,效率很高。

### Q3: 如何可視化碰撞?
**A**: 設置 `contact_sensor.debug_vis = True`,會在模擬器中顯示接觸力箭頭。

### Q4: 碰撞檢測精度如何?
**A**: PhysX 引擎的接觸檢測非常精確,但需要注意:
- 高速碰撞可能穿透
- 需要適當的 `max_depenetration_velocity` 設置
- 建議 timestep 不要過大(當前 3× decimation 是合適的)

## 總結

使用 `Isaac-Reach-TM5-Surgical-v2-Collision` 配置可以:

✅ **自動檢測並懲罰自碰撞**
✅ **自動檢測並懲罰環境碰撞**
✅ **平衡的動作範圍避免無法到達目標**
✅ **低噪聲 PPO 避免抖動問題**
✅ **可自定義碰撞閾值和懲罰權重**

建議從這個配置開始訓練,根據實際情況調整參數!
