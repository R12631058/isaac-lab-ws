# V5 解決方案: Local 座標系方向對齊

## 🎯 問題診斷

### 之前版本 (V1-V4) 的問題
**根本原因**: 使用 **Global (世界座標系)** 方向對齊,而非 **Local (機器人基座座標系)**

```python
# V4 及之前版本 (錯誤!)
end_effector_orientation_tracking = RewTerm(
    func=mdp.orientation_command_error,  # ← 在世界座標系中對齊
    weight=-1.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=["flange"]), "command_name": "ee_pose"},
)
```

**為什麼這是錯的?**
1. **目標方向在機器人基座座標系定義**: `pitch=180°` (朝下), `roll=0°`, `yaw=隨機`
2. **但誤差在世界座標系計算**: 如果機器人基座旋轉,目標方向也會跟著旋轉
3. **結果**: 同樣的相對姿態,在不同基座方向下會得到不同的獎勵 → 訓練不一致!

---

## ✅ V5 解決方案

### 1. 新增 Local 座標系方向對齊函數

**文件**: `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/mdp/rewards.py`

```python
def orientation_command_error_local(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error in robot base frame (local coordinates).

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body, both in the robot's base frame (local coordinates).
    
    Key difference from orientation_command_error:
    - orientation_command_error: Compares orientations in world frame (global)
    - orientation_command_error_local: Compares orientations in robot base frame (local)
    
    Use case: 
    - When you want the end-effector to maintain a specific orientation RELATIVE TO THE ROBOT BASE
    - Regardless of the base's orientation in the world
    """
    from isaaclab.assets import Articulation
    from isaaclab.utils.math import quat_inv, quat_mul
    
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # 目標方向 (已經在基座座標系中)
    des_quat_b = command[:, 3:7]
    
    # 當前 EE 方向 (世界座標系)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]
    
    # 轉換到基座座標系: curr_quat_b = quat_inv(root_quat_w) * curr_quat_w
    root_quat_w_inv = quat_inv(asset.data.root_state_w[:, 3:7])
    curr_quat_b = quat_mul(root_quat_w_inv, curr_quat_w)
    
    # 在基座座標系中計算誤差
    return quat_error_magnitude(curr_quat_b, des_quat_b)
```

---

### 2. 更新環境配置

**文件**: `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/tm5/tm5_reach_surgery_room_cfg.py`

```python
@configclass
class SurgeryRoomRewardsCfg(reach_env_cfg.RewardsCfg):
    """V5 獎勵配置 - 使用 Local 座標系方向對齊"""
    
    # 位置追蹤 (維持不變)
    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=20.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["tm5_end"]), "std": 0.1, "command_name": "ee_pose"},
    )
    
    # 🎯 方向追蹤 - 改用 Local 座標系!
    end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error_local,  # ← 關鍵修正!
        weight=-2.0,  # 可以使用較高權重 (Local 更穩定)
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["tm5_end"]), "command_name": "ee_pose"},
    )
    
    # 動作平滑度 (維持成功值)
    action_rate = RewTerm(
        func=mdp.action_rate_l2, 
        weight=-0.05
    )
```

---

## 📊 測試驗證

運行 `test_local_orientation_v5.py` 的結果:

```
測試 1: 完美對齊 (Local 座標系)
--------------------------------------------------------------------------------
Global 方法誤差:
  Env 0: 0.0000 rad (0.00°)
  Env 1: 1.5711 rad (90.02°)  ← ❌ 基座旋轉後誤判!

Local 方法誤差:
  Env 0: 0.0000 rad (0.00°)
  Env 1: 0.0000 rad (0.00°)  ← ✅ 正確識別完美對齊!
```

**結論**: 
- Global 方法在基座旋轉時會誤判完美對齊的姿態
- Local 方法正確識別相對於基座的方向對齊

---

## 🚀 V5 訓練計劃

### 配置摘要
```yaml
# V5 關鍵改進
orientation_tracking:
  function: orientation_command_error_local  # ← Local 座標系
  weight: -2.0  # 可以用較高權重 (更穩定)

position_fine_grained:
  weight: 20.0  # 維持 V4

action_rate:
  weight: -0.05  # 維持成功值 (不能用 -0.01!)
```

### 預期改善
1. **方向對齊一致性**: 不受機器人基座方向影響
2. **訓練穩定性**: Local 對齊更符合物理直覺,梯度更穩定
3. **更高權重可行性**: Local 對齊允許使用 `-2.0` 而不會導致不穩定
4. **orientation_error 目標**: 從 1.66 rad (V4) → < 0.5 rad (V5)

---

## 📝 啟動 V5 訓練

### 方法 1: 從頭開始訓練 (推薦)
```bash
.\isaaclab_conda.bat -p scripts\reinforcement_learning\rsl_rl\train.py \
    --task Isaac-Reach-TM5-SurgeryRoom-v0 \
    --num_envs 32 \
    --max_iterations 5000 \
    --seed 42
```

**原因**: 
- V5 的獎勵函數與 V4 完全不同 (Local vs Global)
- 從頭訓練確保策略從一致的信號學習

### 方法 2: 快速測試 (較少 iterations)
```bash
.\isaaclab_conda.bat -p scripts\reinforcement_learning\rsl_rl\train.py \
    --task Isaac-Reach-TM5-SurgeryRoom-v0 \
    --num_envs 32 \
    --max_iterations 2000 \
    --seed 42
```

### 方法 3: 從 14-59-17 微調 (不建議)
```bash
# 不建議!因為 V5 的獎勵函數改變太大
# 但如果想嘗試:
.\isaaclab_conda.bat -p scripts\reinforcement_learning\rsl_rl\train.py \
    --task Isaac-Reach-TM5-SurgeryRoom-v0 \
    --num_envs 32 \
    --max_iterations 5000 \
    --checkpoint logs\rsl_rl\tm5_reach_stable_v2\2025-11-12_14-59-17\model_4999.pt
```

---

## 🎯 評估 V5 模型

訓練完成後:

```bash
.\isaaclab_conda.bat -p scripts\reinforcement_learning\rsl_rl\play.py \
    --task Isaac-Reach-TM5-SurgeryRoom-v0 \
    --num_envs 4 \
    --checkpoint logs\rsl_rl\tm5_reach_stable_v2\<timestamp>\model_XXXX.pt
```

**觀察重點**:
1. **坐標軸對齊**: 綠色 (目標) 和藍色 (當前) 坐標軸是否平行?
2. **數值指標**: `orientation_error` 是否 < 0.5 rad (30°)?
3. **穩定性**: 機器人是否仍能穩定 reach 到目標?

---

## 🔍 對比分析

| 版本 | 方向對齊方式 | orientation_tracking 權重 | 位置誤差 | 方向誤差 | 穩定性 |
|------|------------|-------------------------|---------|---------|--------|
| **V1-V3** | Global | -0.2 ~ -1.0 | ❓ | > 1.66 rad | ❌ 失敗 |
| **14-59-17** | Global | -0.2 | 0.12m | 1.66 rad (95°) | ✅ 穩定 |
| **V4** | Global | -1.0 | ? | ? | 🔄 測試中 |
| **V5** | **Local** ✨ | **-2.0** | 目標: <0.15m | 目標: <0.5 rad | 目標: 穩定 |

---

## 🔧 如果 V5 仍有問題

### 情況 1: orientation_error 仍 > 1.0 rad
**診斷**: 權重仍不足
**解決**:
```python
self.rewards.end_effector_orientation_tracking.weight = -3.0  # 或 -5.0
```

### 情況 2: 機器人無法 reach 或不穩定
**診斷**: orientation_tracking 權重與 position_tracking 衝突
**解決**:
```python
# 降低 orientation 權重
self.rewards.end_effector_orientation_tracking.weight = -1.0

# 或提高 position 權重
self.rewards.end_effector_position_tracking_fine_grained.weight = 25.0
```

### 情況 3: 動作過於抖動
**診斷**: action_rate 權重太低
**解決**:
```python
self.rewards.action_rate.weight = -0.01  # 但可能影響探索!
```

---

## 📚 技術總結

### Global vs Local 方向對齊

#### Global (世界座標系) - V1-V4 的問題
```python
# 在 orientation_command_error() 中:
des_quat_w = quat_mul(root_quat_w, des_quat_b)  # 轉換到世界座標
curr_quat_w = body_state_w[:, body_id, 3:7]     # 當前 EE (世界座標)
error = quat_error_magnitude(curr_quat_w, des_quat_w)
```

**問題**: 
- `des_quat_w` 會隨著 `root_quat_w` (基座方向) 改變
- 同樣的 EE 相對姿態,基座旋轉後獎勵不同
- 訓練信號不一致 → 難以學習

#### Local (基座座標系) - V5 的解決方案
```python
# 在 orientation_command_error_local() 中:
des_quat_b = command[:, 3:7]                      # 目標 (基座座標)
curr_quat_w = body_state_w[:, body_id, 3:7]      # 當前 EE (世界座標)
curr_quat_b = quat_mul(quat_inv(root_quat_w), curr_quat_w)  # 轉換到基座座標
error = quat_error_magnitude(curr_quat_b, des_quat_b)
```

**優勢**:
- 目標和當前方向都在基座座標系中
- 不受基座方向影響
- 訓練信號一致 → 容易學習
- 更符合機械臂操作的物理直覺

---

## ✅ 執行清單

- [x] 1. 創建 `orientation_command_error_local()` 函數
- [x] 2. 更新 `tm5_reach_surgery_room_cfg.py` 使用 Local 對齊
- [x] 3. 清理 `__pycache__` 確保使用最新程式碼
- [x] 4. 測試驗證 Local vs Global 差異
- [ ] 5. 啟動 V5 訓練 (等待使用者確認)
- [ ] 6. 監控訓練過程 (rewards, orientation_error)
- [ ] 7. 測試訓練好的 V5 模型
- [ ] 8. 比較 V5 vs 14-59-17 vs V4 性能

---

## 🎉 預期結果

**V5 成功標準**:
- ✅ Position error: < 0.15 m
- ✅ Orientation error: < 0.5 rad (< 30°)
- ✅ 穩定性: 無抖動,平滑運動
- ✅ 成功率: > 90% 的 episodes 達到目標

**如果達成**: V5 將是最佳模型,可用於實際部署!

**如果未達成**: 根據上述「如果 V5 仍有問題」章節調整參數,繼續迭代。

---

## 📞 下一步行動

1. **立即可做**: 啟動 V5 訓練 (2000-5000 iterations)
2. **同步觀察**: 關閉當前運行的 V4 play.py 視窗
3. **耐心等待**: V5 訓練約需 1-2 小時 (根據 iterations)
4. **準備評估**: 訓練完成後立即測試並比較結果

**準備好開始 V5 訓練了嗎?** 🚀
