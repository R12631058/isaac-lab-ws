import sys
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import inspect
from omni.isaac.motion_generation import LulaKinematicsSolver

print("=== LulaKinematicsSolver.compute_inverse_kinematics ===", flush=True)
try:
    source = inspect.getsource(LulaKinematicsSolver.compute_inverse_kinematics)
    print(source, flush=True)
except Exception as e:
    print(f"Error getting source: {e}", flush=True)
    # Print docstring
    print(LulaKinematicsSolver.compute_inverse_kinematics.__doc__, flush=True)

sys.stdout.flush()
simulation_app.close()
