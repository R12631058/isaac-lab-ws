# NVIDIA 顯示卡驅動程式更新後的系統問題與解決方案彙整 (NVIDIA Driver Update: Issues & Solutions Guide)

## 📌 背景說明
在更新 NVIDIA 顯示卡驅動程式（如 GeForce 610.47 版本及更高版本）後，由於 Isaac Sim / Omniverse 底層渲染引擎與新版驅動程式的相容性發生變化，導致系統在執行 3D 交互式檢視器及 NDI 模擬時遇到了一系列的崩潰（Access Violation）、可視化失效（Debug Draw 消失）以及 UI 卡死等重大問題。

本文件整理了驅動程式更新後發生的種種問題、根本原因，以及對應的解決方案，以便日後維護與查閱。

---

## 1. 🛑 3D 模擬場景記憶體崩潰 (Access Violation in rtx.scenedb)

### 💥 問題現象
啟動 3D 檢視器載入大型多環境 USD 場景時，模擬器瞬間崩潰並輸出：
```
Windows fatal exception: access violation
Thread 0x00005144 (most recent call first):
...
2026-06-03 12:37:28 [6,153ms] [Warning] [rtx.scenedb.plugin] SceneDbContext : TLAS limit buffer size 7512601600
2026-06-03 12:37:28 [6,153ms] [Warning] [rtx.scenedb.plugin] SceneDbContext : TLAS limit : valid true, within: false
```

### 🔍 根本原因
新版驅動程式的 Vulkan 渲染後端在處理 Isaac Sim 4.5.0 的場景加速結構（TLAS - Top Level Acceleration Structure）時，其記憶體管理與限制機制被觸發，導致顯示記憶體分配溢出或非法指針存取崩潰。

### 🛠️ 解決方式與配置
強制將渲染後端切換為 **DirectX 12 (DX12)**，同時在啟動參數中為 SceneDb 限制最大 Instance 實例預算，以減少 GPU 記憶體壓力：
1. **停用 Vulkan**：調用 `--/app/vulkan=false` 參數。
2. **限制實例預算**：設置 TLAS 實例預算上限（`maxInstances = 2500000`）。
3. **精簡渲染渲染特徵**：關閉全局環境光遮蔽（AO）、反射、半透明以及多光源取樣等，代碼如下：
   ```python
   sys.argv.extend([
       "--/app/vulkan=false",
       "--/rtx/ecoMode/enabled=true",
       "--/rtx/directLighting/sampledLighting/enabled=false",
       "--/rtx/ambientOcclusion/enabled=false",
       "--/rtx/reflections/enabled=false",
       "--/rtx/translucency/enabled=false",
       "--/rtx/sceneDb/maxInstances=2500000",
       "--/rtx/scenedb/maxInstances=2500000",
       "--/rtx-transient/scenedb/forceMaxTLASInstancesLimit=2500000",
       "--/rtx-transient/scenedb/maxInstancesLimit=2500000",
       "--/rtx-transient/scenedb/instanceBudget=2500000",
   ])
   ```

---

## 2. 🔇 視窗中所有 Debug Draw (射線/Waypoints/標記點) 消失

### 💥 問題現象
雖然更換為 DX12 後程式不再崩潰，但 Viewport 視窗內原有的藍色十字、綠色/紅色 raycast 追蹤線、腫瘤與 waypoints 彩色球體全部消失，只剩下機械臂和場景本體。

### 🔍 根本原因
在排查驅動程式崩潰期間，曾禁用了 `omni.kit.material.library`。因為 Omniverse 底層畫線渲染器（Debug Draw Interface）在視窗中渲染點與線條時，高度依賴此材質庫加載並編譯基本材質著色器，一旦禁用，任何 debug draw 的內容都將無法繪製。

### 🛠️ 解決方式
恢復啟用 `omni.kit.material.library`：
* **修復方式**：將啟動參數中的 `--disable-ext omni.kit.material.library` 移除。
* **備註**：在 DX12 後端下，首次編譯著色器（PSO）需要大約 **1.5 至 2 分鐘**，期間檢視器視窗會呈靜止狀態。請耐心等待其編譯完成，編譯完成後 debug draw 的射線和球體即可完整重現。

---

## 3. ⏳ NDI 狀態卡死在 "Initializing..." 且無可視化

### 💥 問題現象
3D 交互式 UI 左下角的 NDI Live Status 永遠卡在 `Initializing...`，且沒有任何 NDI 標記點狀態更新或線條更新。

### 🔍 根本原因
終端機輸出報錯：
`[NDI][WARN] Failed to import NDIDetector: module 'matplotlib._docstring' has no attribute 'dedent_interpd'`
這是因為驅動程式更新後，Conda 環境下的 `matplotlib.pyplot` 導入時發生了 docstring 相關的 `AttributeError`。原程式在導入時僅使用了 `except ImportError:`，因此 `AttributeError` 逸出導致整個 `NDIDetector` 模組導入失敗，直接為 `None`。

### 🛠️ 解決方式
將導入 Matplotlib 的異常捕獲範圍擴大為 `except Exception:`。由於 Matplotlib 在此專案中僅用於輸出離線的 Occlusion 圖表，這項修復使 NDI 模組在繪圖功能出錯時能自動跳過（Fallback），而不影響核心的即時追蹤和可視化渲染。

* **修改位置**：
  * [ndi_detector_isaaclab_experimental.py](file:///C:/Users/RMML/IsaacLab/scripts/isaaclab_ws/final/ndi_detector_isaaclab_experimental.py#L41-L47)
  * [ndi_detector_isaaclab.py (final)](file:///C:/Users/RMML/IsaacLab/scripts/isaaclab_ws/final/ndi_detector_isaaclab.py#L42-L48)
  * [ndi_detector_isaaclab.py (root)](file:///C:/Users/RMML/IsaacLab/scripts/isaaclab_ws/ndi_detector_isaaclab.py#L42-L48)

---

## 💡 總結檢查清單 (Deployment Checklist)
當更新顯示卡驅動後部署或啟動檢視器時，請檢查：
1. [x] **啟動後端參數**：已包含 `--/app/vulkan=false` 及 `--/rtx/sceneDb/maxInstances=2500000` 相關設定。
2. [x] **擴展啟用狀態**：確保 `omni.kit.material.library` 被載入，並等待 2 分鐘著色器編譯。
3. [x] **庫導入防禦**：確保 Matplotlib 的異常處理為 `except Exception:`，以防 NDI detector 卡死。

---
*最後更新時間：2026-06-04 15:30:00*
