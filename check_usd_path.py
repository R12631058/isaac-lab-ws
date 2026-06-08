"""檢查 USD 檔案中的機器人路徑結構"""

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from pxr import Usd

usd_path = "C:\\Nick\\surgery_team\\surgery_team\\USD\\isaaclab\\surgeryroom_ctgantry_isaaclab.usd"
stage = Usd.Stage.Open(usd_path)

print("\n=== 搜尋 TM5 相關路徑 ===")
tm5_prims = [str(p.GetPath()) for p in stage.Traverse() if 'tm5' in str(p.GetPath()).lower()]
for path in tm5_prims[:30]:
    print(path)

print("\n=== 搜尋 needle 相關路徑 ===")
needle_prims = [str(p.GetPath()) for p in stage.Traverse() if 'needle' in str(p.GetPath()).lower() or 'End_needle' in str(p.GetPath())]
for path in needle_prims[:50]:
    print(path)

print("\n=== 搜尋 target_pose 相關路徑 ===")
target_prims = [str(p.GetPath()) for p in stage.Traverse() if 'target_pose' in str(p.GetPath()).lower() or 'path_' in str(p.GetPath()).lower()]
for path in target_prims[:30]:
    print(path)

simulation_app.close()
