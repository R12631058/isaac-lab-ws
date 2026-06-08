# Fixed Target Position Solution V1

**日期**: 2026-01-08  
**狀態**: ✅ 已驗證成功

## 問題概述

在評估訓練好的 RL policy 時，需要將 target position 固定在指定位置（例如 phantom 表面），但遇到以下問題：

1. **Target 隨機化問題**: UniformPoseCommand 會不斷重新生成隨機 target
2. **Command 被覆寫問題**: `env.step()` 會調用 `command_manager.compute()` 覆寫設定值
3. **座標系統混淆問題**: 誤將 world frame 當作 relative frame 處理，導致雙重偏移
4. **Observation 不同步問題**: Policy 讀取的 observation 包含舊的錯誤 target 信息

## 根本原因分析

### 1. 座標系統架構

```python
# UniformPoseCommand 內部結構
class UniformPoseCommand:
    pose_command_b: torch.Tensor  # Robot base frame (PRIMARY)
    pose_command_w: torch.Tensor  # World frame (for visualization)
    
    @property
    def command(self):
        return self.pose_command_b  # ⚠️ 返回 base frame!
```

**關鍵發現**:
- `pose_command_b` 是**主要數據源**，存儲 robot-relative coordinates
- `pose_command_w` 只用於**可視化**，從 `pose_command_b` 反推計算
- Observation function 通過 `command` property 讀取，獲得的是 `pose_command_b`
- Policy 訓練時使用的就是 robot-relative coordinates

### 2. Observation 數據流

```
generated_commands(env, "ee_pose")
  ↓
env.command_manager.get_command("ee_pose")
  ↓
ee_command.command (property)
  ↓
ee_command.pose_command_b  ← Policy 實際看到的數據
```

### 3. 座標轉換錯誤

**錯誤做法** (導致雙重偏移):
```python
# 錯誤: 將 world coords 當作 relative coords
target_rel_x = world_x - base_x  
pose_command_w[0, 0] = target_rel_x  # 設錯變量
# 結果: target 出現在 (world_x - base_x) - base_x 位置
```

**正確做法**:
```python
# World → Robot Base Frame 轉換
from isaaclab.utils.math import subtract_frame_transforms
target_base_pos, target_base_quat = subtract_frame_transforms(
    robot_pos_w, robot_quat_w, target_world_pos, target_world_quat
)
pose_command_b[0, :3] = target_base_pos  # 設置到正確變量
```

## 完整解決方案

### 核心修改點

#### 1. World → Base Frame 轉換

```python
# 獲取 robot 當前世界座標
robot = env.unwrapped.scene["robot"]
robot_pos = robot.data.root_pos_w[0, :3]
robot_quat = robot.data.root_quat_w[0]

# 目標世界座標
target_world_pos = torch.tensor([target_x, target_y, target_z], device=device)
target_world_quat = ee_command.pose_command_w[0, 3:]

# 轉換到 robot base frame
from isaaclab.utils.math import subtract_frame_transforms
target_base_pos, target_base_quat = subtract_frame_transforms(
    robot_pos, robot_quat, target_world_pos, target_world_quat
)

# 設置命令 (兩個都要設!)
ee_command.pose_command_b[0, :3] = target_base_pos  # Policy 看的
ee_command.pose_command_w[0, :3] = target_world_pos # 可視化用
```

#### 2. Observation 同步更新

```python
# 設置完 command 後，立即重新計算 observation
obs = env.unwrapped.observation_manager.compute()
```

#### 3. 每個 Step 後持續設置

```python
while not done:
    # 1. 重新設置 target (防止被覆寫)
    robot_pos = robot.data.root_pos_w[0, :3]
    robot_quat = robot.data.root_quat_w[0]
    
    target_base_pos, target_base_quat = subtract_frame_transforms(
        robot_pos, robot_quat, target_world_pos, target_world_quat
    )
    
    ee_command.pose_command_b[0, :3] = target_base_pos
    ee_command.pose_command_b[0, 3:] = target_base_quat
    
    # 2. 執行 action
    with torch.no_grad():
        actions = policy.act(obs["policy"], deterministic=True)
    
    # 3. Step environment
    obs, rewards, dones, truncated, info = env.step(actions)
    
    # 4. 重新設置 + 更新 observation
    # (重複步驟 1 的轉換和設置)
    obs = env.unwrapped.observation_manager.compute()
```

### 環境配置

```python
# 禁用 command resampling
env_cfg.commands.ee_pose.resampling_time_range = (1e10, 1e10)

# 單環境測試
env_cfg.scene.num_envs = 1
```

### Target 來源優先級

```python
# Priority 1: USD target object
if args_cli.use_usd_target:
    target_world_pos = get_target_position(env)

# Priority 2: Command line arguments
elif args_cli.target_world_x is not None:
    target_world_pos = {
        'x': args_cli.target_world_x,
        'y': args_cli.target_world_y,
        'z': args_cli.target_world_z
    }

# Priority 3: Phantom position (default)
else:
    target_world_pos = phantom_pos
```

## 驗證結果

✅ **座標系統修復**: Target 出現在正確的世界座標位置  
✅ **Policy 行為正確**: 手臂朝向正確的 target 移動  
✅ **Position 持續固定**: Target 不再隨機化  
✅ **Observation 同步**: Policy 看到正確的 target 信息  

## 關鍵學習點

1. **閱讀源碼**: 一定要檢查 `command` property 實際返回什麼
2. **座標系統**: 注意區分 world frame vs. robot base frame
3. **數據流追蹤**: 從 observation function → command manager → command term
4. **雙重設置**: 需要同時設置 `pose_command_b` (data) 和 `pose_command_w` (visualization)
5. **持續更新**: Robot 移動時 base frame 改變，需要重新計算 relative coordinates

## 使用方式

```powershell
# 使用 USD target object
.\isaac_lab_correct.bat scripts\record_current_position.py --episodes 5 --use_usd_target

# 使用 command line 指定世界座標
.\isaac_lab_correct.bat scripts\record_current_position.py --episodes 5 --target_world_x -0.32 --target_world_y -0.8 --target_world_z 1.08

# 使用 phantom position (default)
.\isaac_lab_correct.bat scripts\record_current_position.py --episodes 5

# 測試不同 robot base 位置
.\isaac_lab_correct.bat scripts\record_current_position.py --episodes 10 --pos_x -0.4 --pos_y -0.1 --use_usd_target
```

## 相關文件

- **主腳本**: `scripts/record_current_position.py`
- **環境配置**: `source/isaaclab_tasks/.../tm5_reach_stable_cfg.py`
- **Command 實現**: `source/isaaclab/isaaclab/envs/mdp/commands/pose_command.py`
- **Observation 實現**: `source/isaaclab/isaaclab/envs/mdp/observations.py`

## 下一步計劃

- [ ] 支持多個並行環境同時評估
- [ ] Batch 測試不同 robot base 位置
- [ ] 分析 generalization 性能
- [ ] 繪製性能 vs. 位置偏移曲線
