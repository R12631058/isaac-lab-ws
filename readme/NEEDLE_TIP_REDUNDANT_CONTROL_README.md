# Needle Tip Redundant Control - 使用說明

## 概述

這是針尖追蹤 + EE 姿態保持的冗余控制策略實現。

### 核心特性

1. **針尖位置追蹤**
   - 使用 FrameTransformer 建立虛擬針尖追蹤點
   - 實測偏移量: (-0.313371, -0.721447, 3.687748) m
   - 獎勵函數直接追蹤針尖位置,而非 End_needle body 中心

2. **EE 姿態保持** (冗余控制)
   - 同時維持 End Effector 接近偏好姿態
   - 偏好四元數: (0.707, 0.707, 0.0, 0.0) - 向下指的姿態
   - 允許機械臂利用冗余自由度達成雙重目標

3. **觀察空間**
   - 關節位置和速度 (12維)
   - 針尖位置 (3維, 使用 FrameTransformer)
   - 位置命令 (3維, 僅位置無姿態)
   - 上一步動作 (6維)
   - **總計: ~24維** (比原本 21 維多了針尖位置)

4. **獎勵權重**
   - `needle_tip_position_tracking`: -2.0 (距離懲罰)
   - `needle_tip_position_tracking_fine_grained`: 25.0 (tanh 精細獎勵)
   - `ee_orientation_keep_preferred_tanh`: 5.0 (姿態保持獎勵)
   - `action_rate`: -0.005
   - `joint_vel`: -0.005
   - `joint_1_avoid_small_angles`: -1.0
   - `joint_limits_penalty`: -0.3

## 訓練

### 基本訓練 (3000 iterations)
```bash
cd C:\Users\RMML\IsaacLab
.\isaaclab_conda.bat -p scripts\reinforcement_learning\rsl_rl\train.py ^
    --task Isaac-Reach-TM5-NeedleTip-Redundant-v0 ^
    --num_envs 256 ^
    --max_iterations 3000
```

### 測試訓練 (快速驗證, 500 iterations)
```bash
.\isaaclab_conda.bat -p scripts\reinforcement_learning\rsl_rl\train.py ^
    --task Isaac-Reach-TM5-NeedleTip-Redundant-v0 ^
    --num_envs 128 ^
    --max_iterations 500
```

### 長期訓練 (10000 iterations)
```bash
.\isaaclab_conda.bat -p scripts\reinforcement_learning\rsl_rl\train.py ^
    --task Isaac-Reach-TM5-NeedleTip-Redundant-v0 ^
    --num_envs 512 ^
    --max_iterations 10000
```

## 播放 (測試)

### 播放訓練好的模型
```bash
.\isaaclab_conda.bat -p scripts\reinforcement_learning\rsl_rl\play.py ^
    --task Isaac-Reach-TM5-NeedleTip-Redundant-Play-v0 ^
    --checkpoint logs\rsl_rl\tm5_reach\<timestamp>\model_<iter>.pt ^
    --num_envs 16
```

## 預期行為

### 成功指標
1. **針尖到達**: 針尖 (needle_tip) 應該到達目標位置,而非 End_needle body
2. **姿態保持**: End Effector 維持接近向下指的偏好姿態
3. **平滑運動**: 利用冗余自由度,運動應該較平滑
4. **無 body 對齊**: End_needle body 中心不應該對齊目標位置

### 驗證方法
1. 開啟 `debug_vis=True` 觀察目標位置球體
2. 檢查針尖是否真的到達目標 (不是 EE body 中心)
3. 觀察 EE 姿態是否保持相對穩定
4. 檢查獎勵曲線:
   - `needle_tip_position_tracking_fine_grained` 應該接近 25.0
   - `ee_orientation_keep_preferred` 應該有合理值 (不是 0)

## 檔案結構

```
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/
├── config/tm5/
│   ├── tm5_reach_surgery_room_cfg.py  # 新增 TM5NeedleTipRedundantReachEnvCfg
│   └── __init__.py                     # 註冊新環境
├── mdp/
│   ├── needle_tip_observations.py     # 針尖觀察函數
│   ├── needle_tip_rewards.py          # 針尖追蹤 + EE 姿態保持獎勵
│   └── __init__.py                     # 導出新函數
```

## 配置詳細

### Scene: SurgeryRoomSceneWithNeedleTipCfg
- 繼承 `SurgeryRoomSceneCfg`
- 新增 `needle_tip_frame`: FrameTransformerCfg
  * 目標 body: "End_needle"
  * 偏移位置: (-0.313371, -0.721447, 3.687748)
  * 偏移旋轉: (0.0582, -0.1163, -0.9914, 0.0316)

### Observations: NeedleTipRedundantObservationsCfg
- `joint_pos`: 關節位置
- `joint_vel`: 關節速度
- `needle_tip_position`: 針尖位置 (NEW!)
- `position_command`: 3D 位置命令
- `actions`: 上一步動作

### Rewards: NeedleTipRedundantRewardsCfg
- `needle_tip_position_tracking`: L2 距離 (NEW!)
- `needle_tip_position_tracking_fine_grained`: Tanh 精細獎勵 (NEW!)
- `ee_orientation_keep_preferred`: 姿態保持獎勵 (NEW!)
- `action_rate`, `joint_vel`: 平滑獎勵
- `joint_1_avoid_small_angles`: 碰撞避免
- `joint_limits_penalty`: 關節極限

## 故障排除

### 問題: FrameTransformer 偏移值可能不正確
- **症狀**: 針尖沒有到達目標,距離很遠
- **原因**: 測量的 3.77m 偏移似乎過大
- **解決**: 
  1. 重新檢查 USD 結構中的 needle_tip 位置
  2. 嘗試調整偏移量為更合理的值 (例如 0.2m)
  3. 使用視覺化確認針尖位置

### 問題: EE 姿態沒有保持
- **症狀**: EE 姿態隨意變化
- **原因**: 姿態獎勵權重太小
- **解決**: 增加 `ee_orientation_keep_preferred` 權重至 10.0

### 問題: 針尖無法到達目標
- **症狀**: 針尖追蹤獎勵始終很低
- **原因**: 姿態約束太強,衝突
- **解決**: 降低 `ee_orientation_keep_preferred` 權重至 2.0

### 問題: 模型仍然對齊 EE body
- **症狀**: End_needle body 中心到達目標
- **原因**: 獎勵函數沒有正確使用 FrameTransformer
- **解決**: 檢查 `needle_tip_rewards.py` 中的 `curr_tip_pos_w` 是否正確提取

## 與舊版比較

### TM5NeedleTipReachEnvCfg (舊版)
- 觀察: 21維 (只有位置命令)
- 獎勵: 追蹤 End_needle body 位置
- 問題: 模型學習對齊 EE body,而非針尖

### TM5NeedleTipRedundantReachEnvCfg (新版)
- 觀察: ~24維 (加入針尖位置)
- 獎勵: 追蹤 needle_tip 位置 + 保持 EE 姿態
- 優勢: 
  * 真正追蹤針尖位置
  * 冗余控制允許雙重目標
  * EE 姿態更穩定

## 下一步

1. **測試訓練** (500 iterations):
   ```bash
   .\isaaclab_conda.bat -p scripts\reinforcement_learning\rsl_rl\train.py ^
       --task Isaac-Reach-TM5-NeedleTip-Redundant-v0 ^
       --num_envs 128 ^
       --max_iterations 500
   ```

2. **驗證行為**:
   - 檢查針尖是否真的到達目標
   - 確認 EE 姿態保持穩定
   - 觀察獎勵曲線

3. **調整權重** (如有必要):
   - 如果針尖不準: 增加針尖追蹤權重
   - 如果姿態亂: 增加姿態保持權重
   - 如果動作不平滑: 增加平滑獎勵

4. **完整訓練** (3000-10000 iterations)

5. **比較結果**:
   - 與 model_2999.pt (舊版 21 維) 比較
   - 確認新版真的只移動針尖,而非對齊 EE body
