# NDI 動態追蹤管線 — 04/29 開發紀錄

## 目標

將 NDI 診斷管線中的機器人追蹤從 **RL Policy（抖動）** 遷移至 **Differential IK Controller（平滑）**，並整合呼吸運動模擬，確保手術追蹤穩定性。

---

## 修改的檔案

### 1. [NEW] `breathing_target_showcase.py`（RL Policy 版，已棄用）

> [!NOTE]
> 此為方案 2 的測試腳本，使用 RL Policy + EMA 平滑。測試後確認效果不佳，已被 IK 版取代。

- **路徑**: [breathing_target_showcase.py](file:///c:/Users/RMML/IsaacLab/scripts/isaaclab_ws/final/breathing_target_showcase.py)
- **功能**: 單環境 showcase，RL policy 驅動 needle_tip 追蹤呼吸 Cube
- **新增功能**: `--smooth_alpha` EMA 平滑參數
- **結論**: RL policy 天生抖動，EMA 只是治標，**不推薦使用**

---

### 2. [NEW] `breathing_target_ik_showcase.py`（IK 版，主要 showcase）

- **路徑**: [breathing_target_ik_showcase.py](file:///c:/Users/RMML/IsaacLab/scripts/isaaclab_ws/final/breathing_target_ik_showcase.py)
- **功能**: 單環境 showcase，**DifferentialIKController** 驅動 needle_tip 平滑追蹤呼吸 Cube

#### 執行指令
```powershell
# 基本執行（不需要 checkpoint）
.\isaaclab.bat -p scripts\isaaclab_ws\final\breathing_target_ik_showcase.py

# 指定 robot 位置
.\isaaclab.bat -p scripts\isaaclab_ws\final\breathing_target_ik_showcase.py --robot_x 0.46

# 限制步數
.\isaaclab.bat -p scripts\isaaclab_ws\final\breathing_target_ik_showcase.py --max_steps 500
```

#### 核心修改內容

| 修改項目 | 說明 |
|---------|------|
| IK Controller | `DifferentialIKControllerCfg(command_type="position", ik_method="dls")` |
| 場景載入 | `InteractiveScene` + `extension_link_out.usd` |
| 追蹤 body | `needle_tip` (非 flange) |
| PD 增益 | `stiffness=400.0, damping=80.0` |
| Jacobian 修正 | 非固定基座時 column offset +6 |
| 關節構型鎖定 | `JOINT_CLAMPS` 6 軸全約束 |
| 初始姿態 | `write_joint_state_to_sim` + `set_joint_position_target` × 200 步穩定 |

---

### 3. [MODIFY] `ndi_multipose_scorer_v2_moving_target.py`（批次評分主腳本）

- **路徑**: [ndi_multipose_scorer_v2_moving_target.py](file:///c:/Users/RMML/IsaacLab/scripts/isaaclab_ws/final/ndi_multipose_scorer_v2_moving_target.py)
- **功能**: 多環境 NDI 評分 + 呼吸運動目標 + **IK 追蹤**（從 showcase 移植回來）

#### 執行指令
```powershell
.\isaaclab.bat -p scripts\isaaclab_ws\final\ndi_multipose_scorer_v2_moving_target.py `
  --robot_x_list 0.66 0.46 0.26 `
  --num_envs 30 `
  --x_start 0.5 --x_end 2.5
```

#### 主要修改

```diff
# 移除
- from rsl_rl.modules import ActorCritic
- parser.add_argument("--checkpoint", ...)
- policy = ActorCritic(...)
- actions = policy.act(obs_tensor, deterministic=True)
- obs, _, terminated, truncated, _ = env.step(actions)

# 新增
+ from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
+ from isaaclab.managers import SceneEntityCfg
+ INIT_JOINT_POS_RAD = [-1.48178, 0.75747, 1.00356, 1.36834, 0.0, -1.60570]
+ JOINT_CLAMPS = {0: (-3.14, -0.1), 1: (-0.3, 2.5), ...}
+ diff_ik_controller = DifferentialIKController(...)
+ joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
+ robot.set_joint_position_target(joint_pos_des, joint_ids=ik_joint_ids)
```

---

## 遇到的問題與解法

### 問題 1: RL Policy 劇烈抖動
- **原因**: 訓練用的 `model_9999.pt` 只優化 reach，不含 smoothness penalty
- **嘗試**: EMA 平滑 (`--smooth_alpha 0.3`) → 效果有限
- **解決**: 改用 `DifferentialIKController`，天生平滑

### 問題 2: IK 解算穿模（self-collision）
- **原因**: IK 傾向使用 joint_1 > 0 的解，導致 link_2 碰撞
- **解決**: 根據初始手術姿態，對所有 6 軸設定 `JOINT_CLAMPS`

### 問題 3: Jacobian 索引錯誤（IK 完全不動）
- **原因**: `is_fixed_base = False` → PhysX Jacobian 前 6 欄是浮動基座 DOF
- **解決**: `jacobian_col_ids = [j + 6 for j in joint_ids]`

### 問題 4: 機器人從 0° 姿態開始
- **原因**: `robot.reset()` 會回到預設值（全零）
- **解決**: 移除 `robot.reset()`，改用 `write_joint_state_to_sim` + `set_joint_position_target` 持續 200 步讓 PD controller 穩定

---

## 初始關節姿態（手術構型）

| Joint   | 角度 (°)  | 弧度 (rad) | Clamp 範圍 | 邏輯 |
|---------|----------|-----------|-----------|------|
| joint_1 | **-84.9** | -1.482   | [-3.14, -0.1] | 必須負側 |
| joint_2 | **+43.4** | +0.757   | [-0.3, +2.5]  | 必須正側 |
| joint_3 | **+57.5** | +1.004   | [-0.3, +2.5]  | 必須正側 |
| joint_4 | **+78.4** | +1.368   | [-0.5, +3.14] | 必須正側 |
| joint_5 | **0.0**   | 0.0      | [-1.57, +1.57] | 雙側允許 |
| joint_6 | **-92.0** | -1.606   | [-3.14, +0.5] | 必須負側 |

---

## 呼吸運動參數

| 參數 | 值 |
|------|---|
| Cube 基準位置 (env-local) | `(-0.4, -0.8, 1.06)` |
| X 振幅 | ±0.03 m (±3 cm) |
| Z 振幅 | ±0.01 m (±1 cm) |
| 波形 | `0.7 × 三角波 + 0.3 × 餘弦波` |
| 角頻率 | `0.5 rad/s` (週期 ~12.6 秒) |

---

## 架構對比

| | v2 (靜態+RL) | v2_moving_target (動態+IK) |
|---|---|---|
| **目標** | 固定 Cube | 呼吸運動 Cube |
| **控制器** | RL Policy (ActorCritic) | DifferentialIK (DLS) |
| **checkpoint** | 需要 model_9999.pt | **不需要** |
| **初始姿態** | env 預設 | 強制手術構型 |
| **碰撞防護** | RL 訓練學的 | JOINT_CLAMPS 硬約束 |
| **sim 步進** | `env.step(actions)` | `robot.set_joint_position_target()` + `env.sim.step()` |
| **動作品質** | 抖動 | 平滑 |

---

## 輸出檔案

| 檔案 | 路徑 |
|------|------|
| 評分表格 PNG | `scripts/isaaclab_ws/final/occlusion_output/ndi_moving_target_table_YYYYMMDD_HHMMSS.png` |
| 詳細報告 JSON | `scripts/isaaclab_ws/final/occlusion_output/ndi_moving_target_score_YYYYMMDD_HHMMSS.json` |
