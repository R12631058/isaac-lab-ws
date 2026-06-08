# 多環境並行評估使用說明

## V1 版本更新

✅ **完成**: Fixed Target Position Solution  
🆕 **新功能**: 支持多個並行環境同時評估

## 主要改進

1. **並行化**: 一次運行多個環境 (默認 50 個)
2. **效率提升**: 相同時間內可以收集 50 倍數據量
3. **統計更可靠**: 更多樣本，更準確的性能評估

## 使用方式

### 基本用法 (50 個並行環境)

```powershell
# 使用 USD target，5 個批次，每批次 50 個環境 = 250 個 episodes
.\isaac_lab_correct.bat scripts\record_current_position.py --episodes 5 --use_usd_target

# 使用指定世界座標
.\isaac_lab_correct.bat scripts\record_current_position.py --episodes 10 --target_world_x -0.32 --target_world_y -0.8 --target_world_z 1.08
```

### 自定義環境數量

```powershell
# 使用 100 個並行環境
.\isaac_lab_correct.bat scripts\record_current_position.py --episodes 10 --num_envs 100 --use_usd_target

# 使用 10 個並行環境 (適合測試/調試)
.\isaac_lab_correct.bat scripts\record_current_position.py --episodes 5 --num_envs 10 --use_usd_target
```

### 測試不同 robot base 位置

```powershell
# 位置 1: X = -0.28
.\isaac_lab_correct.bat scripts\record_current_position.py --episodes 10 --num_envs 50 --pos_x -0.28 --use_usd_target

# 位置 2: X = -0.36
.\isaac_lab_correct.bat scripts\record_current_position.py --episodes 10 --num_envs 50 --pos_x -0.36 --use_usd_target

# 位置 3: X = -0.44
.\isaac_lab_correct.bat scripts\record_current_position.py --episodes 10 --num_envs 50 --pos_x -0.44 --use_usd_target
```

## 參數說明

| 參數 | 默認值 | 說明 |
|------|--------|------|
| `--num_envs` | 50 | 並行環境數量 |
| `--episodes` | 10 | 批次數量 (每批次運行 num_envs 個 episodes) |
| `--use_usd_target` | False | 使用 `/Root/target` USD 物件 |
| `--target_world_x/y/z` | None | 指定 target 的世界座標 |
| `--pos_x/y/z` | None | 設置 robot base 位置 |
| `--checkpoint` | (default path) | 模型檢查點路徑 |

## 輸出信息

### 運行時輸出

```
======================================================================
Multi-Environment Evaluation - V1 (Parallel)
======================================================================
Configuration:
  Parallel Environments: 50
  Episodes per batch: 1
  Total batches: 10
  Total episodes: 500
  Checkpoint: logs/rsl_rl/tm5_reach_stable_v2/.../model_2999.pt
======================================================================

[INFO] Using /Root/target from USD:
  World: (-0.3200, -0.8000, 1.0800)

Running 10 episodes...
  Episode 1/10: Mean Return = 145.32, Mean Length = 89 (50 envs)
  Episode 2/10: Mean Return = 148.17, Mean Length = 87 (50 envs)
  ...

[SUMMARY] Total episodes: 500 (10 batches × 50 envs)
  Mean return: 146.85 ± 12.34
  Mean length: 88.3
  Success rate: 94.2%
```

### 結果文件 (JSON)

```json
{
  "checkpoint": "logs/rsl_rl/tm5_reach_stable_v2/.../model_2999.pt",
  "positions": [
    {
      "position_number": 1,
      "position": {"x": -0.28, "y": -0.1, "z": 0.0},
      "statistics": {
        "mean_return": 146.85,
        "std_return": 12.34,
        "mean_length": 88.3,
        "success_rate": 94.2,
        "num_envs": 50,
        "total_episodes": 500
      },
      "returns": [145.2, 148.3, ...]
    }
  ]
}
```

## 性能對比

| 配置 | Episodes | 時間估計 |
|------|----------|----------|
| 單環境 (V1-old) | 100 | ~10 分鐘 |
| 50 並行環境 (V1-new) | 500 | ~10 分鐘 |
| 100 並行環境 | 1000 | ~10 分鐘 |

**效率提升**: 50-100 倍 🚀

## 注意事項

1. **GPU 內存**: 更多環境需要更多 GPU 內存
   - 50 envs: ~6-8 GB
   - 100 envs: ~12-14 GB
   
2. **調試建議**: 使用 `--num_envs 1` 進行單環境調試

3. **穩定性**: 所有環境使用相同的 target position

4. **座標系統**: 已正確處理 world frame ↔ robot base frame 轉換

## 批次評估示例

測試不同位置的泛化能力：

```powershell
# 自動化批次測試腳本 (PowerShell)
$positions = -0.20, -0.28, -0.36, -0.44, -0.52
foreach ($x in $positions) {
    Write-Host "Testing position X = $x"
    .\isaac_lab_correct.bat scripts\record_current_position.py `
        --episodes 10 `
        --num_envs 50 `
        --pos_x $x `
        --use_usd_target
}
```

## 下一步

- [ ] 分析不同位置的性能曲線
- [ ] 繪製 mean return vs. X position 圖表
- [ ] 確定有效工作範圍
- [ ] 測試 Y, Z 方向的泛化能力
