"""Test CGA IK import in IsaacLab environment."""
import sys
sys.path.insert(0, r"C:\Users\RMML\AppData\Local\ov\pkg\4.5.0\python_packages\TM_kine_py\TM_kine_py")

print(f"Python: {sys.executable}")
print(f"sys.path[0]: {sys.path[0]}")

try:
    import cga
    print(f"cga module: {cga.__file__}")
except Exception as e:
    print(f"cga import FAIL: {type(e).__name__}: {e}")

try:
    import tmr_utils
    print(f"tmr_utils module: {tmr_utils.__file__}")
except Exception as e:
    print(f"tmr_utils import FAIL: {type(e).__name__}: {e}")

try:
    import modern_robotics
    print(f"modern_robotics module: {modern_robotics.__file__}")
except Exception as e:
    print(f"modern_robotics import FAIL: {type(e).__name__}: {e}")

try:
    from cga_ik import CGAIK_CONFIG, CGAIK_ALL
    print("cga_ik import OK")
except Exception as e:
    print(f"cga_ik import FAIL: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

try:
    from cga_fk import CGAFK
    print("cga_fk import OK")
except Exception as e:
    print(f"cga_fk import FAIL: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
