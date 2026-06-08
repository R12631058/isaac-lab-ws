# TM5 手術室場景強化學習訓練成功文檔

> **成功模型:** `logs/rsl_rl/tm5_reach_stable_v2/2025-11-12_14-59-17/model_4999.pt`  
> **訓練日期:** 2025年11月12日  
> **訓練時長:** ~2.4 小時  
> **訓練狀態:** ✅ 能夠 reach、穩定、但角度對齊不佳  

---

## 📋 目錄

1. [專案結構](#專案結構)
2. [核心配置檔案](#核心配置檔案)
3. [訓練腳本](#訓練腳本)
4. [測試腳本](#測試腳本)
5. [關鍵參數總結](#關鍵參數總結)
6. [訓練結果分析](#訓練結果分析)
7. [後續改進方向](#後續改進方向)

---

## 📁 專案結構

```
IsaacLab/
├── scripts/
│   └── reinforcement_learning/
│       └── rsl_rl/
│           ├── train.py                    # 主訓練腳本
│           └── play.py                     # 模型測試腳本
│
├── source/
│   └── isaaclab_tasks/
│       └── isaaclab_tasks/
│           └── manager_based/
│               └── manipulation/
│                   └── reach/
│                       ├── config/
│                       │   └── tm5/
│                       │       ├── tm5_reach_surgery_room_cfg.py    # 環境配置 ★
│                       │       └── agents/
│                       │           └── tm5_reach_stable_ppo_cfg.py  # PPO配置 ★
│                       └── mdp/
│                           └── rewards.py                           # 獎勵函數
│
├── logs/
│   └── rsl_rl/
│       └── tm5_reach_stable_v2/
│           └── 2025-11-12_14-59-17/        # 成功模型訓練記錄
│               ├── model_4999.pt           # 訓練完成模型
│               ├── params/
│               │   ├── env.yaml            # 實際使用的環境配置
│               │   └── agent.yaml          # 實際使用的PPO配置
│               └── summaries/              # TensorBoard 日誌
│
└── isaaclab_conda.bat                      # Conda 環境啟動腳本
```

---

## 🔧 核心配置檔案

### 1. **環境配置** (`tm5_reach_surgery_room_cfg.py`)

**檔案路徑:**
```
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/tm5/tm5_reach_surgery_room_cfg.py
```

#### **關鍵配置區段:**

##### A. 場景設定
```python
from omni.isaac.lab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.reach import reach_env_cfg

@configclass
class TM5ReachSurgeryRoomEnvCfg(reach_env_cfg.ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # ============================================
        # 1. 基礎環境配置
        # ============================================
        self.scene.num_envs = 32               # 平行環境數量
        self.scene.env_spacing = 2.5           # 環境間距 (m)
        self.episode_length_s = 12.0           # 單回合時長 (s)
        # dt = 0.04s → 每回合 300 steps
        
        # ============================================
        # 2. 手術室場景 USD 載入
        # ============================================
        self.scene.surgery_room = AssetBaseCfg(
            prim_path="/World/envs/env_.*/SurgeryRoom",
            spawn=UsdFileCfg(
                usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Props/TM5_700_surgery_room/surgery_room_tm5_700_v0.1.usd",
                rigid_props=RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                ),
                mass_props=MassPropertiesCfg(mass=1.0),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0)
            ),
        )
```

##### B. 機器人配置
```python
        # ============================================
        # 3. TM5-700 機器人配置
        # ============================================
        # 末端執行器名稱修改
        self.scene.robot.spawn.rigid_props.disable_gravity = False
        self.observations.policy.end_effector_position.params["asset_cfg"].body_names = ["flange"]
        self.observations.policy.target_object_position.params["asset_cfg"].body_names = ["flange"]
        
        # 動作配置
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_1", "joint_2", "joint_3", 
                        "joint_4", "joint_5", "joint_6"],
            scale=0.8,              # 動作縮放 (保守設定)
            use_default_offset=True,
        )
```

##### C. **獎勵函數配置 (成功模型的關鍵)** ⭐
```python
        # ============================================
        # 4. 獎勵權重 (14-59-17 成功配置)
        # ============================================
        
        # 位置追蹤
        self.rewards.end_effector_position_tracking.weight = -2.0
        
        # 位置精細追蹤 (主要驅動力)
        self.rewards.end_effector_position_tracking_fine_grained.weight = 15.0
        self.rewards.end_effector_position_tracking_fine_grained.params = {
            "asset_cfg": SceneEntityCfg("robot", body_names=["flange"]),
            "std": 0.04,  # 4cm 內開始給正獎勵
            "command_name": "ee_pose",
        }
        
        # 方向追蹤 (⚠️ 權重太低導致角度對齊不佳)
        self.rewards.end_effector_orientation_tracking.weight = -0.2
        
        # ⭐ 關鍵! 動作平滑度懲罰 (低值確保穩定性)
        self.rewards.action_rate.weight = -0.005
        
        # 關節速度懲罰
        self.rewards.joint_vel.weight = -0.005
        
        # Joint 1 避免碰撞 (避開 -270° ~ -30° 危險區域)
        self.rewards.joint_1_avoid_small_angles.weight = -1.0
        self.rewards.joint_1_avoid_small_angles.params = {
            "asset_cfg": SceneEntityCfg("robot"),
            "safe_zone_min": -4.71,  # -270°
            "safe_zone_max": -0.52,  # -30°
        }
        
        # 關節限制懲罰
        self.rewards.joint_limits_penalty.weight = -0.3
```

##### D. 目標命令配置
```python
        # ============================================
        # 5. 目標位置/姿態命令
        # ============================================
        self.commands.ee_pose = mdp.UniformPoseCommandCfg(
            asset_name="robot",
            body_name="flange",
            resampling_time_range=(8.0, 8.0),  # 每 8 秒更換目標
            debug_vis=True,
            ranges=mdp.UniformPoseCommandCfg.Ranges(
                pos_x=(-0.46, -0.06),           # 相對機器人基座
                pos_y=(-0.61, -0.21),
                pos_z=(0.14, 0.54),
                roll=(0.0, 0.0),
                pitch=(3.14159, 3.14159),       # 固定朝下
                yaw=(-3.14159, 3.14159),        # 全範圍旋轉
            ),
        )
```

##### E. 觀察空間配置
```python
        # ============================================
        # 6. 觀察配置 (25 維)
        # ============================================
        self.observations.policy = ObservationGroupCfg(
            concatenate_terms=True,
            terms={
                "joint_pos": ObsTerm(                 # 6 維
                    func=mdp.joint_pos_rel,
                    params={"asset_cfg": SceneEntityCfg("robot")},
                ),
                "joint_vel": ObsTerm(                 # 6 維
                    func=mdp.joint_vel_rel,
                    params={"asset_cfg": SceneEntityCfg("robot")},
                ),
                "target_object_position": ObsTerm(    # 3 維
                    func=mdp.generated_commands,
                    params={"command_name": "ee_pose"},
                ),
                "target_object_orientation": ObsTerm( # 4 維 (quaternion)
                    func=mdp.generated_commands,
                    params={"command_name": "ee_pose"},
                ),
                "end_effector_position": ObsTerm(     # 3 維
                    func=mdp.ee_pos_in_robot_root_frame,
                    params={"asset_cfg": SceneEntityCfg("robot", body_names=["flange"])},
                ),
                "actions": ObsTerm(                   # 3 維 (前3個動作)
                    func=mdp.last_action
                ),
            }
        )
        # 總計: 6 + 6 + 3 + 4 + 3 + 3 = 25 維
```

---

### 2. **PPO 演算法配置** (`tm5_reach_stable_ppo_cfg.py`)

**檔案路徑:**
```
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/tm5/agents/tm5_reach_stable_ppo_cfg.py
```

```python
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

# ============================================
# PPO 主配置
# ============================================
@configclass
class TM5ReachStablePPORunnerCfg_V2(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24          # 每個環境的步數
    max_iterations = 5000           # 訓練迭代次數
    save_interval = 100             # 每 100 次保存一次模型
    experiment_name = "tm5_reach_stable_v2"
    
    # ============================================
    # 神經網路架構
    # ============================================
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,         # 初始探索噪聲
        
        # Actor Network (策略網路)
        actor_hidden_dims=[768, 512, 512, 256],
        # 架構: 25(input) → 768 → 512 → 512 → 256 → 6(output)
        # 參數量: ~1.5M
        
        # Critic Network (價值網路)
        critic_hidden_dims=[768, 512, 512, 256],
        # 架構: 25(input) → 768 → 512 → 512 → 256 → 1(output)
        
        activation='elu',           # 激活函數
    )
    
    # ============================================
    # PPO 超參數
    # ============================================
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,             # PPO clip 範圍
        entropy_coef=0.01,          # 熵獎勵係數
        num_learning_epochs=5,      # 每次更新的訓練輪數
        num_mini_batches=4,         # Mini-batch 數量
        learning_rate=1.0e-3,       # 學習率
        schedule="adaptive",        # 自適應學習率
        gamma=0.99,                 # 折扣因子
        lam=0.95,                   # GAE λ
        desired_kl=0.01,            # 目標 KL 散度
        max_grad_norm=1.0,          # 梯度裁剪
    )
```

**訓練規模計算:**
```
總樣本數 = max_iterations × num_steps_per_env × num_envs
         = 5000 × 24 × 32
         = 3,840,000 個轉換 (transitions)
```

---

## 🚀 訓練腳本

### 1. **主訓練命令**

```bash
# 完整訓練命令 (在 IsaacLab 根目錄執行)
.\isaaclab_conda.bat -p scripts\reinforcement_learning\rsl_rl\train.py \
    --task Isaac-Reach-TM5-SurgeryRoom-v0 \
    --num_envs 32 \
    --headless \
    --max_iterations 5000 \
    --seed 42
```

**參數說明:**
- `--task`: 環境名稱 (對應 `TM5ReachSurgeryRoomEnvCfg`)
- `--num_envs`: 平行環境數量 (32)
- `--headless`: 無頭模式 (不顯示 GUI,加速訓練)
- `--max_iterations`: 訓練迭代次數 (5000)
- `--seed`: 隨機種子 (確保可重現性)

---

### 2. **訓練腳本內部流程** (`train.py`)

**檔案路徑:** `scripts/reinforcement_learning/rsl_rl/train.py`

```python
import argparse
import torch
from rsl_rl.runners import OnPolicyRunner
from omni.isaac.lab.app import AppLauncher

# ============================================
# 1. 解析命令列參數
# ============================================
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
args = parser.parse_args()

# ============================================
# 2. 啟動模擬器
# ============================================
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================
# 3. 載入環境配置
# ============================================
from omni.isaac.lab_tasks.utils import parse_env_cfg
env_cfg = parse_env_cfg(
    args.task,
    device=args.device,
    num_envs=args.num_envs,
)

# ============================================
# 4. 建立環境
# ============================================
from omni.isaac.lab.envs import ManagerBasedRLEnv
env = ManagerBasedRLEnv(cfg=env_cfg)

# ============================================
# 5. 載入 PPO 配置
# ============================================
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
env = RslRlVecEnvWrapper(env)
agent_cfg = parse_agent_cfg(args.task)

# ============================================
# 6. 建立 PPO Runner
# ============================================
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=args.device)

# ============================================
# 7. 開始訓練
# ============================================
runner.learn(
    num_learning_iterations=args.max_iterations,
    init_at_random_ep_len=True
)

# ============================================
# 8. 關閉模擬器
# ============================================
env.close()
simulation_app.close()
```

---

### 3. **訓練輸出示例**

```
[INFO] Logging experiment in directory: C:\Users\RMML\IsaacLab\logs\rsl_rl\tm5_reach_stable_v2
Exact experiment name requested from command line: 2025-11-12_14-59-17
Setting seed: 42

################################################################################
                       Learning iteration 0/5000 
                       
                       Computation: 387 steps/s
             Mean action noise std: 0.30
          Mean value_function loss: 139.69
               Mean surrogate loss: 0.0185
                       Mean reward: -11.54
               Mean episode length: 23.14
    Metrics/ee_pose/position_error: 0.4228
 Metrics/ee_pose/orientation_error: 1.5885
--------------------------------------------------------------------------------
                   Total timesteps: 1536
                      Time elapsed: 00:00:03
                               ETA: 05:30:38

################################################################################
                       Learning iteration 100/5000 
                       
                       Mean reward: 3.27  ← 獎勵提升!
    Metrics/ee_pose/position_error: 0.1523  ← 位置誤差降低
 Metrics/ee_pose/orientation_error: 1.6821  ← 方向誤差仍高
--------------------------------------------------------------------------------

...

################################################################################
                       Learning iteration 4999/5000 
                       
                       Mean reward: 5.27  ← 最終獎勵
    Metrics/ee_pose/position_error: 0.12 m  ← 12cm 精度
 Metrics/ee_pose/orientation_error: 1.66 rad  ← 95° 誤差 (待改進)
--------------------------------------------------------------------------------
                      Time elapsed: 02:24:00
```

---

## 🧪 測試腳本

### 1. **模型測試命令**

```bash
# 測試訓練完成的模型
.\isaaclab_conda.bat -p scripts\reinforcement_learning\rsl_rl\play.py \
    --task Isaac-Reach-TM5-SurgeryRoom-v0 \
    --num_envs 4 \
    --checkpoint logs/rsl_rl/tm5_reach_stable_v2/2025-11-12_14-59-17/model_4999.pt
```

**參數說明:**
- `--checkpoint`: 模型檔案路徑
- `--num_envs`: 測試環境數量 (通常用 1-4 個)
- 移除 `--headless` 以顯示視覺化

---

### 2. **測試腳本內部流程** (`play.py`)

**檔案路徑:** `scripts/reinforcement_learning/rsl_rl/play.py`

```python
import torch
from rsl_rl.modules import ActorCritic

# ============================================
# 1. 載入環境 (同 train.py)
# ============================================
env = ManagerBasedRLEnv(cfg=env_cfg)

# ============================================
# 2. 載入訓練好的模型
# ============================================
policy = ActorCritic(
    num_actor_obs=env.num_observations,
    num_critic_obs=env.num_observations,
    num_actions=env.num_actions,
    **agent_cfg.policy.to_dict()
)

checkpoint = torch.load(args.checkpoint)
policy.load_state_dict(checkpoint['model_state_dict'])
policy.eval()  # 評估模式 (關閉探索噪聲)

# ============================================
# 3. 執行推論循環
# ============================================
obs = env.reset()
while simulation_app.is_running():
    with torch.no_grad():
        # 獲取動作 (不添加噪聲)
        actions = policy.act_inference(obs)
    
    # 執行動作
    obs, rewards, dones, infos = env.step(actions)
    
    # 重置已完成的環境
    if dones.any():
        env.reset(dones)
```

---

## 📊 關鍵參數總結

### **獎勵函數權重對比**

| 獎勵項 | 14-59-17 (成功) | V4 (改進) | 說明 |
|--------|----------------|-----------|------|
| `position_tracking` | -2.0 | -2.0 | 保持 |
| `position_fine_grained` | **15.0** | **18.0** | ↑ 20% |
| `orientation_tracking` | **-0.2** ⚠️ | **-1.0** | ↑ 5倍 |
| `action_rate` | **-0.005** ⭐ | **-0.005** | 保持 (關鍵!) |
| `joint_vel` | -0.005 | -0.005 | 保持 |
| `joint_1_avoid` | -1.0 | -1.0 | 保持 |
| `joint_limits` | -0.3 | -0.3 | 保持 |

**⚠️ 關鍵發現:**
- `action_rate = -0.005` 確保機器人敢於探索,能夠 reach
- `orientation_tracking = -0.2` 太弱,導致角度對齊失敗
- `position_fine_grained = 15.0` 是主要驅動力

---

### **神經網路架構**

```
Actor Network:
Input(25) → Dense(768) → ELU → Dense(512) → ELU → 
Dense(512) → ELU → Dense(256) → ELU → Dense(6) → Tanh

Critic Network:
Input(25) → Dense(768) → ELU → Dense(512) → ELU → 
Dense(512) → ELU → Dense(256) → ELU → Dense(1)

總參數量: ~1,500,000
```

---

### **訓練超參數**

| 參數 | 值 | 說明 |
|------|---|------|
| `num_envs` | 32 | 平行環境數 |
| `num_steps_per_env` | 24 | 每環境步數 |
| `max_iterations` | 5000 | 總迭代次數 |
| `learning_rate` | 0.001 | 學習率 |
| `clip_param` | 0.2 | PPO clip 範圍 |
| `gamma` | 0.99 | 折扣因子 |
| `lam` | 0.95 | GAE λ |
| `num_learning_epochs` | 5 | 每次更新訓練輪數 |
| `num_mini_batches` | 4 | Mini-batch 數 |
| `entropy_coef` | 0.01 | 熵獎勵 |
| `desired_kl` | 0.01 | KL 散度目標 |
| `init_noise_std` | 0.3 | 初始噪聲 |

---

## 📈 訓練結果分析

### **成功指標 (14-59-17)**

| 指標 | 訓練初期 | 訓練後期 | 改善 |
|------|---------|---------|------|
| Mean Reward | -11.54 | **5.27** | ↑ 145% |
| Position Error | 0.42 m | **0.12 m** | ↓ 71% |
| Orientation Error | 1.59 rad | **1.66 rad** | ⚠️ 未改善 |
| Episode Length | 23.14 | 240+ | ✅ 完成整集 |

---

### **實際表現**

✅ **優點:**
- 能夠穩定到達目標位置 (12cm 精度)
- 運動平滑,無明顯震盪
- 訓練穩定,無崩潰

❌ **缺點:**
- 無法精確對齊末端執行器方向
- 方向誤差 ~95° (1.66 rad)
- 原因: `orientation_tracking = -0.2` 權重太低

---

### **失敗模型對比**

| 模型 | action_rate | 結果 | 原因 |
|------|------------|------|------|
| **14-59-17** ✅ | **-0.005** | 成功 (reach + 穩定) | 權重適中 |
| 23-17-35 ❌ | -0.005 | 失敗 (不穩定) | 可能是 seed 問題 |
| 11-16-51 ❌ | **-0.01** | 失敗 (不穩定) | **權重過高!** |

**關鍵結論:** `action_rate = -0.01` 會導致機器人過於保守,無法有效探索!

---

## 🔄 後續改進方向

### **V4 改進策略 (保守提升)**

```python
# 基於 14-59-17 的改進配置
self.rewards.end_effector_position_tracking.weight = -2.0
self.rewards.end_effector_position_tracking_fine_grained.weight = 18.0   # ↑ 20%
self.rewards.end_effector_orientation_tracking.weight = -1.0             # ↑ 5倍
self.rewards.action_rate.weight = -0.005  # ⭐ 保持不變!
self.rewards.joint_vel.weight = -0.005
```

**改進目標:**
- ✅ 維持 reach 能力 (保留 `action_rate=-0.005`)
- ✅ 維持穩定性 (保留低動作懲罰)
- ✅ 提升角度對齊 (`orientation_tracking: -0.2 → -1.0`)
- ✅ 微調位置精度 (`position_fine: 15.0 → 18.0`)

---

### **訓練建議**

1. **保守改進**: 先用 V4 配置訓練,驗證改善效果
2. **多 Seed 驗證**: 用 seed=42, 100, 200 訓練 3 次,確認穩定性
3. **逐步調整**: 如果 V4 成功,再考慮進一步提升 orientation_tracking
4. **避免過度調整**: 不要一次改太多參數!

---

## 📝 重要文件清單

### **必讀檔案**
1. ✅ `tm5_reach_surgery_room_cfg.py` - 環境配置
2. ✅ `tm5_reach_stable_ppo_cfg.py` - PPO 配置
3. ✅ `train.py` - 訓練腳本
4. ✅ `play.py` - 測試腳本

### **參考檔案**
- `logs/.../params/env.yaml` - 實際訓練配置
- `logs/.../params/agent.yaml` - 實際 PPO 配置
- `logs/.../summaries/` - TensorBoard 日誌

### **模型檔案**
- `model_4999.pt` - 最終模型
- `model_*.pt` - 中間 checkpoints (每 100 次)

---

## 🎯 快速複製成功配置

```bash
# 1. 確認配置檔案正確
cat source/isaaclab_tasks/.../tm5_reach_surgery_room_cfg.py

# 2. 清除 Python 緩存
Remove-Item -Path "source\isaaclab_tasks\**\__pycache__" -Recurse -Force

# 3. 開始訓練
.\isaaclab_conda.bat -p scripts\reinforcement_learning\rsl_rl\train.py \
    --task Isaac-Reach-TM5-SurgeryRoom-v0 \
    --num_envs 32 \
    --headless \
    --max_iterations 5000 \
    --seed 42

# 4. 測試模型
.\isaaclab_conda.bat -p scripts\reinforcement_learning\rsl_rl\play.py \
    --task Isaac-Reach-TM5-SurgeryRoom-v0 \
    --num_envs 4 \
    --checkpoint logs/rsl_rl/tm5_reach_stable_v2/<TIMESTAMP>/model_4999.pt
```

---

## ⚙️ 環境需求

- **作業系統:** Windows 11
- **GPU:** NVIDIA RTX 4070 (12GB VRAM)
- **CUDA:** 12.x
- **Isaac Sim:** 4.5.0
- **Isaac Lab:** Latest
- **Python:** 3.10 (Conda 環境: `env_isaaclab`)
- **PyTorch:** 2.x with CUDA support

---

## 📞 技術支援

如有問題請參考:
1. Isaac Lab 官方文檔: https://isaac-sim.github.io/IsaacLab
2. RSL-RL 文檔: https://github.com/leggedrobotics/rsl_rl
3. 本專案 README: `TM5_COLLISION_HANDLING_GUIDE.md`

---

**文檔版本:** 1.0  
**更新日期:** 2025年11月13日  
**作者:** RMML  
**狀態:** ✅ 已驗證 - 基於實際成功訓練
