# 🔍 Raycast Debug Issues - 修復報告

## 問題 1️⃣: Debug Draw 消失 ✅ 已修復

### 根本原因
在觀察循環中，每一幀都重新定義 `on_hit1` 和 `on_hit2` 回調函數，這導致了 **Python 閉包變數作用域問題**：

```python
# ❌ 舊代碼（有問題）
for i in range(600):
    hit_info1 = {'hit': False}
    def on_hit1(ray, hit_info):  # 每次都重新定義
        hit_info1['hit'] = True   # 閉包變數可能不同步
    
    # ... submit query ...
    sim.step()
    # 此時 hit_info1 可能已被垃圾回收或作用域混亂
```

### 修復方案 ✅ 已應用
使用 **工廠函數** 正確捕獲閉包變數：

```python
# ✅ 新代碼（修復）
ray_results = [{'hit': False}, {'hit': False}]

def make_callback(result_list, index):
    def callback(ray, hit_info):
        if hit_info and hit_info.valid:
            result_list[index]['hit'] = True
    return callback

# 提交查詢
ray1 = Ray(...)
tester.raycast_interface.submit_raycast_query(ray1, make_callback(ray_results, 0))

sim.step()  # 處理回調

# 結果立即可用
color1 = (0.0, 1.0, 0.0, 1.0) if ray_results[0]['hit'] else (1.0, 0.0, 0.0, 1.0)
tester.draw_ray(..., color=color1)  # 使用最新結果繪製
```

**修復位置**: `test_fixed_and_moving_sphere.py` 第 340-410 行

---

## 問題 2️⃣: Raycast 無視新物體 ❓ 待驗證

### 症狀
- 放置阻擋立方體後，射線命中率仍為 **99.5%**（應該變低）
- 表示 raycast 沒有檢測到新添加的碰撞體

### 可能原因

**最可能**: Raycast 在 `sim.reset()` 時快照場景，之後不會動態更新
```
sim.reset() → Raycast 拍攝場景快照 → 後續新加物體被忽視
```

### 待測試的解決方案

1. **檢查 PhysX 場景更新**
```python
# 創建物體後
create_blocking_cube(...)

# 可能需要通知 PhysX 場景已更新
# 方案 A: 重新獲取 raycast 接口
omni.kit.raycast.query.release_raycast_query_interface()
raycast_interface = omni.kit.raycast.query.acquire_raycast_query_interface()

# 方案 B: 檢查 physics context API
# sim.physics_context.mark_dirty()  # 或類似的
```

2. **在 sim.reset() 前添加所有物體**
```python
# 改成：先添加所有物體 → 再 sim.reset() → 再 raycast
create_scene()
create_blocking_cube(...)  # 都在 reset 前
sim.reset()
```

---

## 代碼修改清單

### 文件: `test_fixed_and_moving_sphere.py`

| 行數 | 修改內容 | 優先級 |
|------|--------|--------|
| 340-410 | 修復觀察循環的閉包問題 | ✅ 已完成 |
| 65-93 | 添加 `create_blocking_cube()` 函數 | ✅ 已完成 |

### 修改詳情

**修復 1: 閉包變數處理**
```python
# 之前
hit_info1 = {'hit': False}
def on_hit1(ray, hit_info):
    hit_info1['hit'] = True

# 之後
ray_results = [{'hit': False}, {'hit': False}]
def make_callback(result_list, index):
    def callback(ray, hit_info):
        if hit_info and hit_info.valid:
            result_list[index]['hit'] = True
    return callback

tester.raycast_interface.submit_raycast_query(ray1, make_callback(ray_results, 0))
```

**修復 2: Debug Draw 確保每幀更新**
```python
# 每一幀都要：
1. tester.clear_drawings()     # 清除舊繪製
2. submit_raycast_query()      # 提交新查詢
3. sim.step()                  # 處理回調 ⚠️ 關鍵！
4. draw_ray(color=...)         # 使用最新結果繪製
```

---

## 🧪 驗證方法

### 測試 Debug Draw 修復
```bash
.\isaaclab.bat -p scripts/isaaclab_ws/experiments/test_fixed_and_moving_sphere.py
```

**預期結果**:
- ✅ 10 秒觀察期應該看到綠色和紅色的線持續更新
- ✅ 不會消失或凍結

### 測試 Raycast 動態檢測
觀察 3 秒後（當阻擋立方體創建時）：
- 🟢 **線保持綠色** → ❌ 未解決（raycast 緩存問題）
- 🔴 **線變紅色** → ✅ 已解決（raycast 動態檢測正常）

---

## 📌 後續步驟

### 如果仍然觀察不到 Debug Draw
1. 檢查 `tester.debug_draw_interface` 是否為 `None`
2. 確認 `clear_drawings()` 是否被調用
3. 驗證 `draw_ray()` 和 `draw_point()` 的顏色參數格式

### 如果 Raycast 仍無視新物體
1. 嘗試在物體創建後重新獲取 raycast 接口
2. 检查 IsaacLab 物理上下文是否有更新場景的方法
3. 考慮在 `sim.reset()` 前創建所有物體

---

## 🔍 診斷代碼

如果上述修復仍未解決問題，運行診斷腳本：
```bash
.\isaaclab.bat -p scripts/isaaclab_ws/experiments/test_raycast_dynamic_detection.py
```

這個腳本會：
- 測試 debug draw 是否正常
- 將所有物體在 reset() 前創建
- 驗證 raycast 是否能檢測動態添加的物體
